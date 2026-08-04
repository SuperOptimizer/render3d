#include "vk/vkc5d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "brick.h"
#include "gpu/host_entropy.h"
#include "transform/dct_tables.h"
#include "vk/vkres.h"

#define BD 128u
#define NVOX (BD * BD * BD)
#define LVS (512u * 4096u) /* levels ints per brick */

/* push structs must match the kernels in ${R3D_C5D_DIR}/src/gpu/kernels */
typedef struct pc_ent {
  uint32_t chunks_per_sub, nchunk, nsub, nway, ctx2;
  uint32_t nbrick, brick0, lv_stride, tb_stride, sub_stride, st_stride;
} pc_ent;
typedef struct pc_dq {
  float q, hf_exp, dc_fine, dz_dq;
  uint32_t dim, bpa, levels_base, vol_base;
} pc_dq;
typedef struct pc_db {
  uint32_t dim, axis;
  int32_t c;
  uint32_t nface, vol_base;
} pc_db;
typedef struct pc_pack {
  uint32_t dim, bpa, vol_base, sx, sy, sz;
} pc_pack;

typedef struct pipe {
  VkDescriptorSetLayout dsl;
  VkPipelineLayout layout;
  VkPipeline pipe;
  VkDescriptorPool dpool;
  VkDescriptorSet dset;
} pipe;

struct r3d_vkc5d {
  r3d_vkctx *vk;
  uint32_t max_batch;
  pipe ent, dq, db, pk;
  r3d_vkbuf lv, vol, pay, freq, cum, slot, sub, stat, scan;
  VkCommandPool pool;
  VkCommandBuffer cmd;
  VkFence fence;
  double last_gpu_ms;
  bool hybrid;
};

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int pipe_create(r3d_vkctx *c, const char *spv_dir, const char *name, uint32_t nbuf,
                       bool image_binding, uint32_t push_size, pipe *p) {
  memset(p, 0, sizeof *p);
  VkDescriptorSetLayoutBinding binds[9];
  uint32_t nb = 0;
  for (; nb < nbuf; nb++)
    binds[nb] = (VkDescriptorSetLayoutBinding){.binding = nb,
                                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               .descriptorCount = 1,
                                               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  if (image_binding) {
    binds[nb] = (VkDescriptorSetLayoutBinding){.binding = nb,
                                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                               .descriptorCount = 1,
                                               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    nb++;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = nb,
      .pBindings = binds};
  if (vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, &p->dsl) != VK_SUCCESS) return -1;
  VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = push_size};
  VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1,
                                     .pSetLayouts = &p->dsl,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pcr};
  if (vkCreatePipelineLayout(c->dev, &plci, NULL, &p->layout) != VK_SUCCESS) return -1;

  char path[1024];
  snprintf(path, sizeof path, "%s/%s", spv_dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "vkc5d: missing shader %s\n", path);
    return -1;
  }
  fseek(f, 0, SEEK_END);
  long sn = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint32_t *code = malloc((size_t)sn);
  if (!code || fread(code, 1, (size_t)sn, f) != (size_t)sn) {
    fclose(f);
    free(code);
    return -1;
  }
  fclose(f);
  VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = (size_t)sn,
                                   .pCode = code};
  VkShaderModule mod;
  VkResult r = vkCreateShaderModule(c->dev, &smci, NULL, &mod);
  free(code);
  if (r != VK_SUCCESS) return -1;
  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main"},
      .layout = p->layout};
  r = vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p->pipe);
  vkDestroyShaderModule(c->dev, mod, NULL);
  if (r != VK_SUCCESS) return -1;

  VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nbuf ? nbuf : 1},
                                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
  VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 1,
                                     .poolSizeCount = image_binding ? 2u : 1u,
                                     .pPoolSizes = sizes};
  if (vkCreateDescriptorPool(c->dev, &dpci, NULL, &p->dpool) != VK_SUCCESS) return -1;
  VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = p->dpool,
                                      .descriptorSetCount = 1,
                                      .pSetLayouts = &p->dsl};
  return vkAllocateDescriptorSets(c->dev, &dsai, &p->dset) == VK_SUCCESS ? 0 : -1;
}

static void pipe_destroy(r3d_vkctx *c, pipe *p) {
  if (p->dpool) vkDestroyDescriptorPool(c->dev, p->dpool, NULL);
  if (p->pipe) vkDestroyPipeline(c->dev, p->pipe, NULL);
  if (p->layout) vkDestroyPipelineLayout(c->dev, p->layout, NULL);
  if (p->dsl) vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
}

static void bind_buffers(r3d_vkctx *c, pipe *p, const r3d_vkbuf *bufs, uint32_t nbuf) {
  VkDescriptorBufferInfo bi[9];
  VkWriteDescriptorSet w[9];
  for (uint32_t i = 0; i < nbuf; i++) {
    bi[i] = (VkDescriptorBufferInfo){.buffer = bufs[i].buf, .range = VK_WHOLE_SIZE};
    w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                  .dstSet = p->dset,
                                  .dstBinding = i,
                                  .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  .pBufferInfo = &bi[i]};
  }
  vkUpdateDescriptorSets(c->dev, nbuf, w, 0, NULL);
}

static void barrier_compute(VkCommandBuffer cmd) {
  VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
  };
  VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .memoryBarrierCount = 1,
                          .pMemoryBarriers = &mb};
  vkCmdPipelineBarrier2(cmd, &dep);
}

static void dispatch(VkCommandBuffer cmd, pipe *p, const void *push, uint32_t push_size,
                     uint32_t gx) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->dset, 0,
                          NULL);
  vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
  vkCmdDispatch(cmd, gx, 1, 1);
}

int r3d_vkc5d_create(r3d_vkc5d **out, r3d_vkctx *c, const char *spv_dir, uint32_t max_batch) {
  *out = NULL;
  r3d_vkc5d *d = calloc(1, sizeof *d);
  if (!d) return -1;
  d->vk = c;
  d->max_batch = max_batch ? max_batch : 8;
  d->hybrid = getenv("R3D_C5D_HYBRID") && *getenv("R3D_C5D_HYBRID") == '1';
  uint32_t K = d->max_batch;

  if (pipe_create(c, spv_dir, "c5d_entropy.spv", 8, false, sizeof(pc_ent), &d->ent) != 0 ||
      pipe_create(c, spv_dir, "c5d_dequant_idct.spv", 2, false, sizeof(pc_dq), &d->dq) != 0 ||
      pipe_create(c, spv_dir, "c5d_deblock.spv", 1, false, sizeof(pc_db), &d->db) != 0 ||
      pipe_create(c, spv_dir, "pack.spv", 1, true, sizeof(pc_pack), &d->pk) != 0)
    goto fail;

  VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (r3d_vkbuf_create_host(c, (VkDeviceSize)K * LVS * 4, u, &d->lv) != 0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * NVOX * 4, u, &d->vol) != 0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * (4u << 20), u, &d->pay) != 0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * HE_NMODELS * HE_NTOK * 4, u, &d->freq) != 0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * HE_NMODELS * (HE_NTOK + 1) * 4, u, &d->cum) !=
          0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * HE_NMODELS * 4096 * 4, u, &d->slot) != 0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * 32 * 6 * 4, u, &d->sub) != 0 ||
      r3d_vkbuf_create_host(c, (VkDeviceSize)K * 32 * 4, u, &d->stat) != 0 ||
      r3d_vkbuf_create_host(c, 4096 * 4, u, &d->scan) != 0)
    goto fail;
  for (uint32_t i = 0; i < 4096; i++) ((uint32_t *)d->scan.mapped)[i] = SCAN16_TAB[i];

  r3d_vkbuf ent_bufs[8] = {d->lv, d->pay, d->freq, d->cum, d->slot, d->sub, d->scan, d->stat};
  bind_buffers(c, &d->ent, ent_bufs, 8);
  r3d_vkbuf dq_bufs[2] = {d->lv, d->vol};
  bind_buffers(c, &d->dq, dq_bufs, 2);
  bind_buffers(c, &d->db, &d->vol, 1);
  bind_buffers(c, &d->pk, &d->vol, 1); /* image binding written per decode */

  VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                 .queueFamilyIndex = c->qfam};
  if (vkCreateCommandPool(c->dev, &cpi, NULL, &d->pool) != VK_SUCCESS) goto fail;
  VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                     .commandPool = d->pool,
                                     .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                     .commandBufferCount = 1};
  if (vkAllocateCommandBuffers(c->dev, &cai, &d->cmd) != VK_SUCCESS) goto fail;
  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(c->dev, &fci, NULL, &d->fence) != VK_SUCCESS) goto fail;

  *out = d;
  return 0;
fail:
  r3d_vkc5d_destroy(d);
  return -1;
}

void r3d_vkc5d_destroy(r3d_vkc5d *d) {
  if (!d) return;
  r3d_vkctx *c = d->vk;
  if (d->fence) vkDestroyFence(c->dev, d->fence, NULL);
  if (d->pool) vkDestroyCommandPool(c->dev, d->pool, NULL);
  r3d_vkbuf *bufs[] = {&d->lv, &d->vol, &d->pay, &d->freq, &d->cum,
                       &d->slot, &d->sub, &d->stat, &d->scan};
  for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) r3d_vkbuf_destroy(c, bufs[i]);
  pipe_destroy(c, &d->ent);
  pipe_destroy(c, &d->dq);
  pipe_destroy(c, &d->db);
  pipe_destroy(c, &d->pk);
  free(d);
}

static int deblock_strength(float q) {
  int c = (int)(0.8f * q + 1.0f);
  return c < 1 ? 1 : (c > 24 ? 24 : c);
}

int r3d_vkc5d_decode(r3d_vkc5d *d, const r3d_c5d_src *src, uint32_t n, VkImageView atlas_view,
                     uint8_t *out_max) {
  r3d_vkctx *c = d->vk;
  /* point the pack pipeline at the atlas image (GENERAL layout) */
  VkDescriptorImageInfo ii = {.imageView = atlas_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = d->pk.dset,
                            .dstBinding = 1,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            .pImageInfo = &ii};
  vkUpdateDescriptorSets(c->dev, 1, &w, 0, NULL);

  d->last_gpu_ms = 0.0;
  for (uint32_t i0 = 0; i0 < n; i0 += d->max_batch) {
    uint32_t nb = n - i0 < d->max_batch ? n - i0 : d->max_batch;
    he_gpu gpus[64];
    he_decoded hds[64];
    memset(gpus, 0, sizeof gpus);
    memset(hds, 0, sizeof hds);
    if (nb > 64) return -1;

    /* CPU front half + staging */
    size_t pay_off = 0;
    for (uint32_t i = 0; i < nb; i++) {
      const r3d_c5d_src *s = &src[i0 + i];
      if (d->hybrid) {
        if (he_decode(s->blob, s->n, BD, &hds[i]) != 0) goto brick_fail;
        memcpy((int32_t *)d->lv.mapped + (size_t)i * LVS, hds[i].levels,
               (size_t)hds[i].nchunk * 4096 * 4);
        /* fill params he_gpu-style from the hybrid struct */
        gpus[i].q = hds[i].q;
        gpus[i].hf_exp = hds[i].hf_exp;
        gpus[i].dc_fine = hds[i].dc_fine;
        gpus[i].dz_dq = hds[i].dz_dq;
        gpus[i].deblock = hds[i].deblock;
        gpus[i].dim = hds[i].dim;
        gpus[i].bpa = hds[i].bpa;
        gpus[i].nchunk = hds[i].nchunk;
        gpus[i].nsub = hds[i].nsub;
        gpus[i].chunks_per_sub = hds[i].chunks_per_sub;
      } else {
        if (he_gpu_setup(s->blob, s->n, BD, &gpus[i]) != 0) goto brick_fail;
        he_gpu *g = &gpus[i];
        if (pay_off + g->payload_n > (size_t)d->max_batch * (4u << 20)) goto brick_fail;
        memcpy((uint8_t *)d->pay.mapped + pay_off, g->payload, g->payload_n);
        uint32_t *si = (uint32_t *)d->sub.mapped + (size_t)i * 32 * 6;
        memcpy(si, g->subinfo, (size_t)g->nsub * 6 * 4);
        for (uint32_t ss = 0; ss < g->nsub; ss++) {
          si[ss * 6 + 0] += (uint32_t)pay_off;
          si[ss * 6 + 2] += (uint32_t)pay_off;
        }
        memcpy((uint32_t *)d->freq.mapped + (size_t)i * HE_NMODELS * HE_NTOK, g->freq,
               (size_t)HE_NMODELS * HE_NTOK * 4);
        memcpy((uint32_t *)d->cum.mapped + (size_t)i * HE_NMODELS * (HE_NTOK + 1), g->cum,
               (size_t)HE_NMODELS * (HE_NTOK + 1) * 4);
        memcpy((uint32_t *)d->slot.mapped + (size_t)i * HE_NMODELS * 4096, g->slot2sym,
               (size_t)HE_NMODELS * 4096 * 4);
        pay_off += (g->payload_n + 3) & ~(size_t)3;
      }
    }

    /* the flat entropy dispatch shares nway/ctx2/geometry across the batch */
    if (!d->hybrid)
      for (uint32_t i = 1; i < nb; i++)
        if (gpus[i].rans_nway != gpus[0].rans_nway || gpus[i].ctx2 != gpus[0].ctx2 ||
            gpus[i].nsub != gpus[0].nsub || gpus[i].chunks_per_sub != gpus[0].chunks_per_sub) {
          fprintf(stderr, "vkc5d: mixed stream flags in batch (brick %u)\n", i0 + i);
          goto brick_fail;
        }

    /* record */
    vkResetCommandBuffer(d->cmd, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(d->cmd, &bi);
    if (!d->hybrid) {
      vkCmdFillBuffer(d->cmd, d->lv.buf, 0, VK_WHOLE_SIZE, 0);
      barrier_compute(d->cmd);
      const he_gpu *g0 = &gpus[0];
      pc_ent pe = {.chunks_per_sub = g0->chunks_per_sub,
                   .nchunk = g0->nchunk,
                   .nsub = g0->nsub,
                   .nway = g0->rans_nway,
                   .ctx2 = g0->ctx2,
                   .nbrick = nb,
                   .brick0 = 0,
                   .lv_stride = LVS,
                   .tb_stride = HE_NMODELS,
                   .sub_stride = 32 * 6,
                   .st_stride = 32};
      dispatch(d->cmd, &d->ent, &pe, sizeof pe, (nb * g0->nsub + 63) / 64);
      barrier_compute(d->cmd);
    }
    for (uint32_t i = 0; i < nb; i++) {
      const he_gpu *g = &gpus[i];
      pc_dq pd = {.q = g->q, .hf_exp = g->hf_exp, .dc_fine = g->dc_fine, .dz_dq = g->dz_dq,
                  .dim = g->dim, .bpa = g->bpa, .levels_base = i * LVS, .vol_base = i * NVOX};
      dispatch(d->cmd, &d->dq, &pd, sizeof pd, g->nchunk);
    }
    barrier_compute(d->cmd);
    for (uint32_t axis = 0; axis < 3; axis++) {
      for (uint32_t i = 0; i < nb; i++) {
        const he_gpu *g = &gpus[i];
        if (!g->deblock) continue;
        uint32_t nface = g->dim / 16u - 1u;
        pc_db pb = {.dim = g->dim, .axis = axis, .c = deblock_strength(g->q), .nface = nface,
                    .vol_base = i * NVOX};
        dispatch(d->cmd, &d->db, &pb, sizeof pb, (nface * g->dim * g->dim + 63) / 64);
      }
      barrier_compute(d->cmd);
    }
    for (uint32_t i = 0; i < nb; i++) {
      const r3d_c5d_src *s = &src[i0 + i];
      const he_gpu *g = &gpus[i];
      pc_pack pp = {.dim = g->dim, .bpa = g->bpa, .vol_base = i * NVOX,
                    .sx = s->sx, .sy = s->sy, .sz = s->sz};
      dispatch(d->cmd, &d->pk, &pp, sizeof pp, g->nchunk);
    }
    /* pack writes -> sampled reads later (image stays GENERAL) */
    VkMemoryBarrier2 mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
    };
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .memoryBarrierCount = 1,
                            .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(d->cmd, &dep);
    vkEndCommandBuffer(d->cmd);

    uint64_t t0 = now_ns();
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1,
                       .pCommandBuffers = &d->cmd};
    if (vkQueueSubmit(c->queue, 1, &si, d->fence) != VK_SUCCESS) goto batch_fail;
    if (vkWaitForFences(c->dev, 1, &d->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) goto batch_fail;
    vkResetFences(c->dev, 1, &d->fence);
    d->last_gpu_ms += (double)(now_ns() - t0) / 1e6;

    /* status + per-brick max (host-visible, coherent) */
    if (!d->hybrid) {
      const uint32_t *st = d->stat.mapped;
      for (uint32_t i = 0; i < nb; i++)
        for (uint32_t ss = 0; ss < gpus[i].nsub; ss++)
          if (st[i * 32 + ss]) {
            fprintf(stderr, "vkc5d: truncated substream brick %u sub %u\n", i0 + i, ss);
            goto batch_fail;
          }
    }
    if (out_max) {
      const int32_t *vol = d->vol.mapped;
      for (uint32_t i = 0; i < nb; i++) {
        int32_t m = 0;
        const int32_t *v = vol + (size_t)i * NVOX;
        for (size_t k = 0; k < NVOX; k++)
          if (v[k] > m) m = v[k];
        out_max[i0 + i] = (uint8_t)(m < 0 ? 0 : (m > 255 ? 255 : m));
      }
    }
    for (uint32_t i = 0; i < nb; i++) {
      if (d->hybrid) he_free(&hds[i]);
      else he_gpu_free(&gpus[i]);
    }
    continue;
  brick_fail:
    for (uint32_t i = 0; i < nb; i++) {
      if (d->hybrid) he_free(&hds[i]);
      else he_gpu_free(&gpus[i]);
    }
    return -1;
  batch_fail:
    for (uint32_t i = 0; i < nb; i++) {
      if (d->hybrid) he_free(&hds[i]);
      else he_gpu_free(&gpus[i]);
    }
    return -1;
  }
  return 0;
}

double r3d_vkc5d_last_gpu_ms(const r3d_vkc5d *d) { return d->last_gpu_ms; }
