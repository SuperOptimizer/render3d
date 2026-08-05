/* Vulkan implementation of render.h. Frame graph (M1):
 *   raycast.comp (storage image, GENERAL) -> blit -> swapchain -> present
 * 2 frames in flight, timeline semaphore for CPU pacing, binary semaphores for
 * WSI, timestamp queries around the dispatch. */
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/clip.h"
#include "core/slab.h"
#include "render/render.h"
#include "shard.h" /* c5d .c5s reader */
#include "vk/vkc5d.h"
#include "vk/vkclip.h"
#include "vk/vkctx.h"
#include "vk/vkgui.h"
#include "vk/vkres.h"
#include "vk/vkswap.h"

#define FRAMES_IN_FLIGHT 2

struct r3d_renderer {
  SDL_Window *win;
  r3d_config cfg;
  r3d_vkctx vk;
  r3d_vkswap swap;

  r3d_vkimage offscreen; /* RGBA8 storage image at drawable size */
  r3d_vkimage volume;    /* R8 3D + full mip chain (dummy 2^3 until upload) */
  r3d_vkimage tf;        /* 256x1 RGBA8 transfer function */
  r3d_vkimage occ;       /* per-8^3-block max (dilated), for empty-space skip */
  VkSampler samp_vol;    /* trilinear + mip linear, clamp */
  VkSampler samp_tf;     /* linear, clamp */
  VkSampler samp_near;   /* nearest, clamp (occupancy) */
  VkSampler samp_slab;   /* trilinear, clamp UV + REPEAT W (ring z) */

  /* slab mode */
  bool slab_mode;
  r3d_slab_layout slab;
  r3d_vkimage tiles[16]; /* gy-major (element j*4+i); unused stay null */
  int64_t slab_z0;       /* current window start; -1 = nothing uploaded */
  uint8_t *slice_buf;    /* CPU assembly buffer, tile_w * tile_h bytes */
  r3d_vkimage overview;  /* whole composite at 1/4 res (anti-alias far field) */
  uint8_t *ov_buf;       /* downsampled slice, ov dims */

  r3d_vkclip *clipm;     /* clipmap mode (NULL unless r3d_clip_begin) */

  /* bricks mode (c5d GPU-decoded atlas) */
  r3d_vkc5d *c5d;
  r3d_vkimage brick_atlas;
  VkImageView brick_atlas_mip0; /* single-mip storage view for the pack kernel */
  r3d_vkimage brick_occ;        /* 8^3-block occupancy reduced from the atlas */
  r3d_vkbuf page_buf;
  uint32_t bricks_bpa, bricks_abpa;
  bool bricks_identity; /* atlas layout == world layout: direct sampling */

  /* bricks streaming (hot atlas smaller than the volume): two-tier GPU cache.
   * WARM = compressed blobs in a host-visible device buffer (LRU, first-fit
   * allocator); HOT = atlas slots (LRU, page-table indirected). The per-frame
   * pump (r3d_bricks_stream) turns frustum-prioritized requests into budgeted
   * warm->hot GPU decodes plus incremental per-slot mips and occupancy. */
  struct {
    bool active;
    c5d_shard_reader sr; /* stays open: streaming reads blobs on demand */
    bool sr_open;
    uint32_t nb, nslots, frame, last_inflight;
    uint32_t *slot_brick, *slot_use; /* per slot: brick idx / last-wanted frame */
    uint32_t *brick_slot;            /* per brick: slot or UINT32_MAX */
    int16_t *brick_maxk;             /* decoded max, -1 unknown (never re-request empties) */
    struct bcand { float d2; uint32_t b; } *cands;
    r3d_c5d_src *srcs;
    uint32_t *sel_b, *sel_slot;
    uint8_t *maxes;
    r3d_vkimage occraw;         /* world-indexed raw occupancy (pre-dilate) */
    r3d_vkcomp omax, odil;      /* region-form occupancy kernels */
    bool comp_ready;
    r3d_vkbuf warm;
    uint64_t warm_cap, warm_bytes;
    uint32_t warm_bricks;
    uint32_t *warm_off, *warm_len, *warm_use; /* per brick; off UINT32_MAX = absent */
    struct wfnode { uint32_t off, len; } *wfree; /* offset-sorted free list */
    uint32_t nwf, cwf;
  } bs;

  PFN_vkTransitionImageLayoutEXT fp_transition;
  PFN_vkCopyMemoryToImageEXT fp_copy_mem;

  VkDescriptorSetLayout dsl;
  VkPipelineLayout pipe_layout;
  VkPipeline raycast[4]; /* per-mode variants: cube, slab, clip, bricks */
  VkDescriptorPool dpool;
  VkDescriptorSet dset;

  uint32_t wg_x, wg_y; /* raycast workgroup size (R3D_WG=8x8|16x8|16x16) */

  VkCommandPool pool;
  VkCommandBuffer cmd[FRAMES_IN_FLIGHT];
  VkSemaphore acquire[FRAMES_IN_FLIGHT];
  VkSemaphore timeline;
  uint64_t timeline_value;
  uint64_t slot_value[FRAMES_IN_FLIGHT];
  uint64_t slot_gpu[FRAMES_IN_FLIGHT][4]; /* total, raycast, blit, gui (ns) */
  VkQueryPool query;
  bool slot_has_query[FRAMES_IN_FLIGHT];
  uint32_t slot;

  r3d_vkbuf readback; /* lazily sized for screenshots */

  bool gui_up;   /* cimgui initialized */
  bool gui_open; /* NewFrame issued, awaiting render/discard */
};

static int load_spv(const char *dir, const char *name, uint32_t **words, size_t *nbytes) {
  char path[1024];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "vk: cannot open shader %s\n", path);
    return -1;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0 || (n % 4) != 0) {
    fclose(f);
    return -1;
  }
  uint32_t *buf = malloc((size_t)n);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(buf);
    return -1;
  }
  fclose(f);
  *words = buf;
  *nbytes = (size_t)n;
  return 0;
}

static int create_offscreen(r3d_renderer *r) {
  VkExtent3D e = {r->swap.extent.width, r->swap.extent.height, 1};
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8G8B8A8_UNORM, e, 1,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         &r->offscreen) != 0)
    return -1;
  VkDescriptorImageInfo ii = {.imageView = r->offscreen.view,
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 2,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
  return 0;
}

static int create_pipeline(r3d_renderer *r) {
  VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, /* volume: array of 16 (slab tiles; cube uses [0] x16) */
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 16,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 3,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 4, /* bricks mode: c5d atlas */
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 5, /* bricks mode: page table */
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
  };
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 6,
      .pBindings = bindings,
  };
  if (vkCreateDescriptorSetLayout(r->vk.dev, &dslci, NULL, &r->dsl) != VK_SUCCESS) return -1;

  VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                             .size = sizeof(r3d_frame_params)};
  VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &r->dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
  };
  if (vkCreatePipelineLayout(r->vk.dev, &plci, NULL, &r->pipe_layout) != VK_SUCCESS) return -1;

  /* one pipeline per sampling mode; R3D_WG workgroup sweep applies to cube */
  const char *wg = getenv("R3D_WG");
  const char *names[4] = {"raycast_cube.spv", "raycast_slab.spv", "raycast_clip.spv",
                          "raycast_bricks.spv"};
  r->wg_x = 16;
  r->wg_y = 8;
  if (wg && strcmp(wg, "8x8") == 0) {
    names[0] = "raycast_8x8.spv";
    r->wg_x = r->wg_y = 8;
  } else if (wg && strcmp(wg, "16x16") == 0) {
    names[0] = "raycast_16x16.spv";
    r->wg_x = r->wg_y = 16;
  }
  for (uint32_t m = 0; m < 4; m++) {
    uint32_t *spv = NULL;
    size_t spv_n = 0;
    if (load_spv(r->cfg.spv_dir, names[m], &spv, &spv_n) != 0) return -1;
    VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                     .codeSize = spv_n,
                                     .pCode = spv};
    VkShaderModule mod;
    VkResult res = vkCreateShaderModule(r->vk.dev, &smci, NULL, &mod);
    free(spv);
    if (res != VK_SUCCESS) return -1;
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = mod,
                  .pName = "main"},
        .layout = r->pipe_layout,
    };
    res = vkCreateComputePipelines(r->vk.dev, VK_NULL_HANDLE, 1, &cpci, NULL, &r->raycast[m]);
    vkDestroyShaderModule(r->vk.dev, mod, NULL);
    if (res != VK_SUCCESS) return -1;
  }

  VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 19},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
  };
  VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 3,
      .pPoolSizes = sizes,
  };
  if (vkCreateDescriptorPool(r->vk.dev, &dpci, NULL, &r->dpool) != VK_SUCCESS) return -1;
  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = r->dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &r->dsl,
  };
  if (vkAllocateDescriptorSets(r->vk.dev, &dsai, &r->dset) != VK_SUCCESS) return -1;
  return 0;
}

/* --- occupancy build: per-8^3-block max, then 26-neighbor max dilation so
 * trilinear fine samples near block borders can never read past a "skipped"
 * verdict. Parallel over z-blocks. --- */
#define OCC_BLOCK 8u

typedef struct occ_job {
  const uint8_t *vox;
  uint8_t *out;
  uint32_t nx, ny, nz, ox, oy, oz, z0, z1;
} occ_job;

static void *occ_worker(void *arg) {
  occ_job *j = arg;
  for (uint32_t bz = j->z0; bz < j->z1; bz++) {
    uint32_t zlo = bz * OCC_BLOCK, zhi = zlo + OCC_BLOCK > j->nz ? j->nz : zlo + OCC_BLOCK;
    for (uint32_t by = 0; by < j->oy; by++) {
      uint32_t ylo = by * OCC_BLOCK, yhi = ylo + OCC_BLOCK > j->ny ? j->ny : ylo + OCC_BLOCK;
      for (uint32_t bx = 0; bx < j->ox; bx++) {
        uint32_t xlo = bx * OCC_BLOCK, xhi = xlo + OCC_BLOCK > j->nx ? j->nx : xlo + OCC_BLOCK;
        uint8_t m = 0;
        for (uint32_t z = zlo; z < zhi; z++)
          for (uint32_t y = ylo; y < yhi; y++) {
            const uint8_t *row = j->vox + ((size_t)z * j->ny + y) * j->nx;
            for (uint32_t x = xlo; x < xhi; x++)
              if (row[x] > m) m = row[x];
          }
        j->out[((size_t)bz * j->oy + by) * j->ox + bx] = m;
      }
    }
  }
  return NULL;
}

static uint8_t *build_occupancy(const uint8_t *vox, uint32_t nx, uint32_t ny, uint32_t nz,
                                uint32_t *ox_out, uint32_t *oy_out, uint32_t *oz_out) {
  uint32_t ox = (nx + OCC_BLOCK - 1) / OCC_BLOCK, oy = (ny + OCC_BLOCK - 1) / OCC_BLOCK,
           oz = (nz + OCC_BLOCK - 1) / OCC_BLOCK;
  size_t n = (size_t)ox * oy * oz;
  uint8_t *raw = malloc(n), *dil = malloc(n);
  if (!raw || !dil) {
    free(raw);
    free(dil);
    return NULL;
  }

  uint32_t nthreads = oz < 12 ? oz : 12;
  pthread_t tids[12];
  occ_job jobs[12];
  for (uint32_t i = 0; i < nthreads; i++) {
    jobs[i] = (occ_job){.vox = vox, .out = raw, .nx = nx, .ny = ny, .nz = nz,
                        .ox = ox, .oy = oy, .oz = oz,
                        .z0 = oz * i / nthreads, .z1 = oz * (i + 1) / nthreads};
    pthread_create(&tids[i], NULL, occ_worker, &jobs[i]);
  }
  for (uint32_t i = 0; i < nthreads; i++) pthread_join(tids[i], NULL);

  for (uint32_t z = 0; z < oz; z++)
    for (uint32_t y = 0; y < oy; y++)
      for (uint32_t x = 0; x < ox; x++) {
        uint8_t m = 0;
        for (int dz = -1; dz <= 1; dz++)
          for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
              int zz = (int)z + dz, yy = (int)y + dy, xx = (int)x + dx;
              if (zz < 0 || yy < 0 || xx < 0 || zz >= (int)oz || yy >= (int)oy || xx >= (int)ox)
                continue;
              uint8_t v = raw[((size_t)zz * oy + (size_t)yy) * ox + (size_t)xx];
              if (v > m) m = v;
            }
        dil[((size_t)z * oy + y) * ox + x] = m;
      }
  free(raw);
  *ox_out = ox;
  *oy_out = oy;
  *oz_out = oz;
  return dil;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Write binding 0 (the vol[16] array) — views may repeat (cube: same view x16). */
static void write_volume_dset(r3d_renderer *r, VkImageView views[16], VkSampler sampler) {
  VkDescriptorImageInfo ii[16];
  for (uint32_t i = 0; i < 16; i++)
    ii[i] = (VkDescriptorImageInfo){.sampler = sampler,
                                    .imageView = views[i],
                                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = 16,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
}

static void write_image_dset(r3d_renderer *r, uint32_t binding, VkDescriptorType type,
                             VkImageView view, VkSampler sampler, VkImageLayout layout) {
  VkDescriptorImageInfo ii = {.sampler = sampler, .imageView = view, .imageLayout = layout};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = binding,
      .descriptorCount = 1,
      .descriptorType = type,
      .pImageInfo = &ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
}

/* Staging upload for small images (TF, dummy volume): buffer -> mip0, then
 * transition the whole image to SHADER_READ_ONLY. */
static int upload_small_image(r3d_renderer *r, r3d_vkimage *im, const void *data,
                              VkDeviceSize nbytes) {
  r3d_vkbuf stage;
  if (r3d_vkbuf_create_host(&r->vk, nbytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stage) != 0)
    return -1;
  memcpy(stage.mapped, data, nbytes);
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  if (!cmd) return -1;
  r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0,
                       im->mips);
  VkBufferImageCopy region = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = im->extent,
  };
  vkCmdCopyBufferToImage(cmd, stage.buf, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &region);
  r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, im->mips);
  int rc = r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
  r3d_vkbuf_destroy(&r->vk, &stage);
  return rc;
}

/* mip0 is in TRANSFER_SRC; blit-minify the chain, then everything to
 * SHADER_READ_ONLY. Spec allows sloppy 3D linear blits — compute fallback
 * (mipdown.slang) is planned if test_gpu flags this driver (docs/measured.md). */
static int gen_mips(r3d_renderer *r, r3d_vkimage *im) {
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  if (!cmd) return -1;
  VkExtent3D e = im->extent;
  for (uint32_t m = 1; m < im->mips; m++) {
    VkExtent3D ne = {e.width > 1 ? e.width / 2 : 1, e.height > 1 ? e.height / 2 : 1,
                     e.depth > 1 ? e.depth / 2 : 1};
    r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, 0,
                         VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, m, 1);
    VkImageBlit2 blit = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 0, 1},
        .srcOffsets = {{0, 0, 0}, {(int32_t)e.width, (int32_t)e.height, (int32_t)e.depth}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1},
        .dstOffsets = {{0, 0, 0}, {(int32_t)ne.width, (int32_t)ne.height, (int32_t)ne.depth}},
    };
    VkBlitImageInfo2 bi = {
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage = im->img,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = im->img,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &blit,
        .filter = VK_FILTER_LINEAR,
    };
    vkCmdBlitImage2(cmd, &bi);
    r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, m, 1);
    e = ne;
  }
  r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, im->mips);
  return r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
}

int r3d_create(SDL_Window *win, const r3d_config *cfg, r3d_renderer **out) {
  *out = NULL;
  r3d_renderer *r = calloc(1, sizeof *r);
  if (!r) return -1;
  r->win = win;
  r->cfg = *cfg;

  uint32_t next = 0;
  const char *const *exts = SDL_Vulkan_GetInstanceExtensions(&next);
  if (r3d_vkctx_create(&r->vk, exts, next, cfg->validate) != 0) goto fail;
  if (r3d_vkswap_create(&r->vk, win, cfg->vsync, &r->swap) != 0) goto fail;
  if (create_pipeline(r) != 0) goto fail;
  if (create_offscreen(r) != 0) goto fail;
  r->fp_transition = (PFN_vkTransitionImageLayoutEXT)(void (*)(void))vkGetDeviceProcAddr(
      r->vk.dev, "vkTransitionImageLayoutEXT");
  r->fp_copy_mem = (PFN_vkCopyMemoryToImageEXT)(void (*)(void))vkGetDeviceProcAddr(
      r->vk.dev, "vkCopyMemoryToImageEXT");

  VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = r->vk.qfam,
  };
  if (vkCreateCommandPool(r->vk.dev, &cpi, NULL, &r->pool) != VK_SUCCESS) goto fail;
  VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = r->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = FRAMES_IN_FLIGHT,
  };
  if (vkAllocateCommandBuffers(r->vk.dev, &cai, r->cmd) != VK_SUCCESS) goto fail;

  for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (vkCreateSemaphore(r->vk.dev, &sci, NULL, &r->acquire[i]) != VK_SUCCESS) goto fail;
  }
  VkSemaphoreTypeCreateInfo tsi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
  };
  VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tsi};
  if (vkCreateSemaphore(r->vk.dev, &sci, NULL, &r->timeline) != VK_SUCCESS) goto fail;

  if (r->vk.caps.timestamps) {
    VkQueryPoolCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = FRAMES_IN_FLIGHT * 4,
    };
    if (vkCreateQueryPool(r->vk.dev, &qci, NULL, &r->query) != VK_SUCCESS) goto fail;
  }

  /* samplers */
  VkSamplerCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = VK_LOD_CLAMP_NONE,
  };
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_vol) != VK_SUCCESS) goto fail;
  smci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  smci.maxLod = 0.0f;
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_tf) != VK_SUCCESS) goto fail;
  smci.magFilter = VK_FILTER_NEAREST;
  smci.minFilter = VK_FILTER_NEAREST;
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_near) != VK_SUCCESS) goto fail;

  /* dummy 2^3 volume so the descriptor set is valid before the real upload
   * (2, not 1: vkres dimensionality heuristic needs depth>1 for a 3D image) */
  {
    uint8_t zeros[8] = {0};
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){2, 2, 2}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->volume) != 0)
      goto fail;
    if (upload_small_image(r, &r->volume, zeros, sizeof zeros) != 0) goto fail;
    VkImageView vv[16];
    for (uint32_t e = 0; e < 16; e++) vv[e] = r->volume.view;
    write_volume_dset(r, vv, r->samp_vol);
    /* dummy occupancy: fully occupied so nothing is skipped before upload */
    uint8_t full[8];
    memset(full, 0xff, sizeof full);
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){2, 2, 2}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->occ) != 0)
      goto fail;
    if (upload_small_image(r, &r->occ, full, sizeof full) != 0) goto fail;
    write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->occ.view, r->samp_near,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  /* default transfer function: grayscale ramp, linear alpha */
  {
    uint8_t ramp[256][4];
    for (uint32_t i = 0; i < 256; i++) {
      ramp[i][0] = ramp[i][1] = ramp[i][2] = (uint8_t)i;
      ramp[i][3] = (uint8_t)i;
    }
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8G8B8A8_UNORM, (VkExtent3D){256, 1, 1}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->tf) != 0)
      goto fail;
    if (upload_small_image(r, &r->tf, ramp, sizeof ramp) != 0) goto fail;
    write_image_dset(r, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->tf.view, r->samp_tf,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  /* bricks-mode dummies: atlas = dummy volume view; page = 4-byte buffer */
  {
    if (r3d_vkbuf_create_host(&r->vk, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &r->page_buf) != 0)
      goto fail;
    ((uint32_t *)r->page_buf.mapped)[0] = 0xFFFFFFFFu;
    write_image_dset(r, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->volume.view,
                     r->samp_vol, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkDescriptorBufferInfo pbi = {.buffer = r->page_buf.buf, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet pw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                               .dstSet = r->dset,
                               .dstBinding = 5,
                               .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               .pBufferInfo = &pbi};
    vkUpdateDescriptorSets(r->vk.dev, 1, &pw, 0, NULL);
  }

  if (r3d_vkgui_init(&r->vk, win, r->swap.format, r->swap.nimages) != 0) goto fail;
  r->gui_up = true;

  *out = r;
  return 0;
fail:
  r3d_destroy(r);
  return -1;
}

int r3d_gui_begin(r3d_renderer *r) {
  if (!r->gui_up) return -1;
  if (!r->gui_open) {
    r3d_vkgui_new_frame();
    r->gui_open = true;
  }
  return 0;
}

void r3d_gui_event(r3d_renderer *r, const SDL_Event *ev) {
  if (r->gui_up) r3d_vkgui_event(ev);
}

void r3d_destroy(r3d_renderer *r) {
  if (!r) return;
  if (r->vk.dev) vkDeviceWaitIdle(r->vk.dev);
  if (r->gui_up) {
    if (r->gui_open) r3d_vkgui_discard();
    r3d_vkgui_shutdown();
  }
  r3d_vkbuf_destroy(&r->vk, &r->readback);
  if (r->query) vkDestroyQueryPool(r->vk.dev, r->query, NULL);
  if (r->timeline) vkDestroySemaphore(r->vk.dev, r->timeline, NULL);
  for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    if (r->acquire[i]) vkDestroySemaphore(r->vk.dev, r->acquire[i], NULL);
  if (r->pool) vkDestroyCommandPool(r->vk.dev, r->pool, NULL);
  if (r->dpool) vkDestroyDescriptorPool(r->vk.dev, r->dpool, NULL);
  for (uint32_t m = 0; m < 4; m++)
    if (r->raycast[m]) vkDestroyPipeline(r->vk.dev, r->raycast[m], NULL);
  if (r->pipe_layout) vkDestroyPipelineLayout(r->vk.dev, r->pipe_layout, NULL);
  if (r->dsl) vkDestroyDescriptorSetLayout(r->vk.dev, r->dsl, NULL);
  r3d_vkclip_destroy(r->clipm);
  r3d_vkc5d_destroy(r->c5d);
  if (r->brick_atlas_mip0) vkDestroyImageView(r->vk.dev, r->brick_atlas_mip0, NULL);
  r3d_vkimage_destroy(&r->vk, &r->brick_atlas);
  r3d_vkimage_destroy(&r->vk, &r->brick_occ);
  r3d_vkbuf_destroy(&r->vk, &r->page_buf);
  if (r->bs.comp_ready) {
    r3d_vkcomp_destroy(&r->vk, &r->bs.omax);
    r3d_vkcomp_destroy(&r->vk, &r->bs.odil);
  }
  r3d_vkimage_destroy(&r->vk, &r->bs.occraw);
  r3d_vkbuf_destroy(&r->vk, &r->bs.warm);
  if (r->bs.sr_open) c5d_shard_close_reader(&r->bs.sr);
  free(r->bs.slot_brick);
  free(r->bs.slot_use);
  free(r->bs.brick_slot);
  free(r->bs.brick_maxk);
  free(r->bs.cands);
  free(r->bs.srcs);
  free(r->bs.sel_b);
  free(r->bs.sel_slot);
  free(r->bs.maxes);
  free(r->bs.warm_off);
  free(r->bs.warm_len);
  free(r->bs.warm_use);
  free(r->bs.wfree);
  if (r->samp_vol) vkDestroySampler(r->vk.dev, r->samp_vol, NULL);
  if (r->samp_tf) vkDestroySampler(r->vk.dev, r->samp_tf, NULL);
  if (r->samp_near) vkDestroySampler(r->vk.dev, r->samp_near, NULL);
  if (r->samp_slab) vkDestroySampler(r->vk.dev, r->samp_slab, NULL);
  for (uint32_t i = 0; i < 16; i++) r3d_vkimage_destroy(&r->vk, &r->tiles[i]);
  r3d_vkimage_destroy(&r->vk, &r->overview);
  free(r->slice_buf);
  free(r->ov_buf);
  r3d_vkimage_destroy(&r->vk, &r->volume);
  r3d_vkimage_destroy(&r->vk, &r->tf);
  r3d_vkimage_destroy(&r->vk, &r->occ);
  r3d_vkimage_destroy(&r->vk, &r->offscreen);
  r3d_vkswap_destroy(&r->vk, &r->swap);
  r3d_vkctx_destroy(&r->vk);
  free(r);
}

int r3d_resize(r3d_renderer *r) {
  int rc = r3d_vkswap_recreate(&r->vk, r->win, r->cfg.vsync, &r->swap);
  if (rc != 0) return rc; /* rc==1: minimized, keep old resources */
  r3d_vkimage_destroy(&r->vk, &r->offscreen);
  return create_offscreen(r);
}

int r3d_upload_volume(r3d_renderer *r, const r3d_volume_desc *d, const uint8_t *voxels) {
  uint32_t maxdim = d->nx > d->ny ? d->nx : d->ny;
  if (d->nz > maxdim) maxdim = d->nz;
  if (maxdim > r->vk.caps.max_dim_3d) {
    fprintf(stderr, "vk: volume %ux%ux%u exceeds maxImageDimension3D=%u (use bricks)\n", d->nx,
            d->ny, d->nz, r->vk.caps.max_dim_3d);
    return -1;
  }
  uint32_t mips = 1;
  while ((maxdim >> mips) >= 1 && mips < 16) mips++;

  bool use_hic = r->vk.caps.host_image_copy && r->fp_transition && r->fp_copy_mem &&
                 !(getenv("R3D_STAGING") && *getenv("R3D_STAGING") == '1');
  VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            (use_hic ? VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT : 0u);

  r3d_vkimage im;
  VkExtent3D extent = {d->nx, d->ny, d->nz};
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, extent, mips, usage, &im) != 0) return -1;

  size_t total = (size_t)d->nx * d->ny * d->nz;
  uint64_t t0 = now_ns();
  if (use_hic) {
    VkHostImageLayoutTransitionInfoEXT tr = {
        .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
        .image = im.img,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if (r->fp_transition(r->vk.dev, 1, &tr) != VK_SUCCESS) goto fail;
    VkMemoryToImageCopyEXT region = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
        .pHostPointer = voxels,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = extent,
    };
    VkCopyMemoryToImageInfoEXT ci = {
        .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
        .dstImage = im.img,
        .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .regionCount = 1,
        .pRegions = &region,
    };
    if (r->fp_copy_mem(r->vk.dev, &ci) != VK_SUCCESS) goto fail;
    /* host-written GENERAL -> TRANSFER_SRC for the mip chain */
    VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    if (!cmd) goto fail;
    r3d_vk_image_barrier(cmd, im.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, 0, 1);
    if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
  } else {
    /* reusable staging slab: whole slices, <=128 MiB per copy */
    size_t slice = (size_t)d->nx * d->ny;
    uint32_t zslab = (uint32_t)((128u << 20) / slice);
    if (zslab == 0) zslab = 1;
    if (zslab > d->nz) zslab = d->nz;
    r3d_vkbuf stage;
    if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)(slice * zslab),
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stage) != 0)
      goto fail;
    VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    r3d_vk_image_barrier(cmd, im.img, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
    if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
    for (uint32_t z = 0; z < d->nz; z += zslab) {
      uint32_t nz = z + zslab > d->nz ? d->nz - z : zslab;
      memcpy(stage.mapped, voxels + (size_t)z * slice, slice * nz);
      cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
      VkBufferImageCopy region = {
          .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .imageOffset = {0, 0, (int32_t)z},
          .imageExtent = {d->nx, d->ny, nz},
      };
      vkCmdCopyBufferToImage(cmd, stage.buf, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                             &region);
      if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
    }
    r3d_vkbuf_destroy(&r->vk, &stage);
    cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    r3d_vk_image_barrier(cmd, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, 0, 1);
    if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
  }
  double up_ms = (double)(now_ns() - t0) / 1e6;

  t0 = now_ns();
  if (gen_mips(r, &im) != 0) goto fail;
  double mip_ms = (double)(now_ns() - t0) / 1e6;
  printf("volume: %ux%ux%u (%zu MiB, %u mips) upload[%s] %.0f ms (%.2f GB/s), mips %.0f ms\n",
         d->nx, d->ny, d->nz, total >> 20, mips, use_hic ? "host-image-copy" : "staging", up_ms,
         (double)total / up_ms / 1e6, mip_ms);

  /* occupancy pyramid for empty-space skipping */
  t0 = now_ns();
  uint32_t ox, oy, oz;
  uint8_t *occ = build_occupancy(voxels, d->nx, d->ny, d->nz, &ox, &oy, &oz);
  if (!occ) goto fail;
  r3d_vkimage occ_im;
  if (oz < 2) { /* keep the image 3D (vkres heuristic): duplicate the layer */
    uint8_t *grown = realloc(occ, (size_t)ox * oy * 2);
    if (!grown) {
      free(occ);
      goto fail;
    }
    occ = grown;
    memcpy(occ + (size_t)ox * oy, occ, (size_t)ox * oy);
    oz = 2;
  }
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){ox, oy, oz}, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &occ_im) != 0) {
    free(occ);
    goto fail;
  }
  int occ_rc = upload_small_image(r, &occ_im, occ, (VkDeviceSize)ox * oy * oz);
  free(occ);
  if (occ_rc != 0) {
    r3d_vkimage_destroy(&r->vk, &occ_im);
    goto fail;
  }
  printf("volume: occupancy %ux%ux%u built+uploaded in %.0f ms\n", ox, oy, oz,
         (double)(now_ns() - t0) / 1e6);

  vkDeviceWaitIdle(r->vk.dev);
  r3d_vkimage_destroy(&r->vk, &r->volume);
  r3d_vkimage_destroy(&r->vk, &r->occ);
  r->volume = im;
  r->occ = occ_im;
  VkImageView vv[16];
    for (uint32_t e = 0; e < 16; e++) vv[e] = r->volume.view;
  write_volume_dset(r, vv, r->samp_vol);
  write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->occ.view, r->samp_near,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return 0;
fail:
  r3d_vkimage_destroy(&r->vk, &im);
  return -1;
}

int r3d_set_transfer(r3d_renderer *r, const uint8_t rgba[256][4]) {
  vkDeviceWaitIdle(r->vk.dev);
  return upload_small_image(r, &r->tf, rgba, 256 * 4);
}

int r3d_slab_init(r3d_renderer *r, const r3d_slab_desc *d) {
  if (r3d_slab_layout_init(&r->slab, d->nx, d->ny, d->nz, d->wz) != 0) {
    fprintf(stderr, "slab: unsupported layout %ux%ux%u wz=%u (max %u per tiled axis)\n", d->nx,
            d->ny, d->nz, d->wz, 2u * (R3D_SLAB_MAX_TILE - 2));
    return -1;
  }
  if (!r->vk.caps.host_image_copy || !r->fp_transition || !r->fp_copy_mem) {
    fprintf(stderr, "slab: needs VK_EXT_host_image_copy (staging scroll path not implemented)\n");
    return -1;
  }
  uint32_t tw = r3d_slab_tile_w(&r->slab), th = r3d_slab_tile_h(&r->slab);
  r->slice_buf = malloc((size_t)tw * th);
  if (!r->slice_buf) return -1;

  VkSamplerCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT, /* ring z wraps */
  };
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_slab) != VK_SUCCESS) return -1;

  for (uint32_t j = 0; j < r->slab.gy; j++)
    for (uint32_t i = 0; i < r->slab.gx; i++) {
      r3d_vkimage *t = &r->tiles[j * 4 + i];
      if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){tw, th, r->slab.wz}, 1,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
                             t) != 0)
        return -1;
      /* permanent GENERAL layout: host writes + shader reads without dances */
      VkHostImageLayoutTransitionInfoEXT tr = {
          .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
          .image = t->img,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      if (r->fp_transition(r->vk.dev, 1, &tr) != VK_SUCCESS) return -1;
    }

  VkImageView views[16];
  for (uint32_t e = 0; e < 16; e++)
    views[e] = r->tiles[e].view ? r->tiles[e].view : r->tiles[0].view;
  /* NOTE: descriptors must use GENERAL for these images */
  VkDescriptorImageInfo ii[16];
  for (uint32_t e = 0; e < 16; e++)
    ii[e] = (VkDescriptorImageInfo){
        .sampler = r->samp_slab, .imageView = views[e], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = 16,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);

  /* overview: the whole composite at 1/4 resolution in ONE texture (max
   * 8184/4 = 2046), same ring z — sampled by the shader once the ray-cone
   * footprint exceeds ~4 voxels, killing far-field aliasing without mips */
  uint32_t ox = (d->nx + 3) / 4, oy = (d->ny + 3) / 4;
  r->ov_buf = malloc((size_t)ox * oy);
  if (!r->ov_buf) return -1;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){ox, oy, r->slab.wz}, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
                         &r->overview) != 0)
    return -1;
  VkHostImageLayoutTransitionInfoEXT otr = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
      .image = r->overview.img,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  if (r->fp_transition(r->vk.dev, 1, &otr) != VK_SUCCESS) return -1;
  /* binding 3 (occupancy slot, unused in slab mode) becomes the overview */
  write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->overview.view,
                   r->samp_slab, VK_IMAGE_LAYOUT_GENERAL);

  r->slab_mode = true;
  r->slab_z0 = -1;
  return 0;
}

int r3d_slab_window(r3d_renderer *r, const r3d_volume *src, uint32_t z0) {
  if (!r->slab_mode) return -1;
  if (z0 > r3d_slab_z0_max(&r->slab)) z0 = r3d_slab_z0_max(&r->slab);
  if ((int64_t)z0 == r->slab_z0) return 0;

  uint32_t s0, s1;
  int full;
  r3d_slab_scroll_range(&r->slab, r->slab_z0, (int64_t)z0, &s0, &s1, &full);
  uint64_t t0 = now_ns();

  /* in-flight frames may sample layers we are about to overwrite */
  if (r->timeline_value) {
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &r->timeline,
        .pValues = &r->timeline_value,
    };
    if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return -1;
  }

  uint32_t tw = r3d_slab_tile_w(&r->slab), th = r3d_slab_tile_h(&r->slab);
  const r3d_slab_layout *l = &r->slab;
  for (uint32_t s = s0; s < s1; s++) {
    uint32_t layer = r3d_slab_ring_layer(l, s);
    for (uint32_t j = 0; j < l->gy; j++)
      for (uint32_t i = 0; i < l->gx; i++) {
        /* assemble slice: payload run memcpy + clamped apron columns */
        for (uint32_t dr = 0; dr < th; dr++) {
          const uint8_t *srow =
              src->voxels + ((size_t)s * src->ny + r3d_slab_src_row(l, j, dr)) * src->nx;
          uint8_t *drow = r->slice_buf + (size_t)dr * tw;
          int64_t w0 = (int64_t)i * l->px - 1; /* world col of dst col 0 */
          uint32_t lo = w0 < 0 ? (uint32_t)(-w0) : 0;
          int64_t hi64 = (int64_t)l->nx - 1 - w0; /* last dst col with src */
          uint32_t hi = hi64 >= (int64_t)tw - 1 ? tw - 1 : (uint32_t)hi64;
          memcpy(drow + lo, srow + (w0 + lo), hi - lo + 1);
          for (uint32_t c = 0; c < lo; c++) drow[c] = srow[0];
          for (uint32_t c = hi + 1; c < tw; c++) drow[c] = srow[l->nx - 1];
        }
        VkMemoryToImageCopyEXT region = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
            .pHostPointer = r->slice_buf,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, (int32_t)layer},
            .imageExtent = {tw, th, 1},
        };
        VkCopyMemoryToImageInfoEXT ci = {
            .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
            .dstImage = r->tiles[j * 4 + i].img,
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount = 1,
            .pRegions = &region,
        };
        if (r->fp_copy_mem(r->vk.dev, &ci) != VK_SUCCESS) return -1;
      }

    /* overview: 4x4 box-average of the source slice -> same ring layer */
    uint32_t ox = (l->nx + 3) / 4, oyd = (l->ny + 3) / 4;
    const uint8_t *sbase = src->voxels + (size_t)s * src->ny * src->nx;
    for (uint32_t oy2 = 0; oy2 < oyd; oy2++) {
      uint32_t y0 = oy2 * 4, y1 = y0 + 4 > l->ny ? l->ny : y0 + 4;
      uint8_t *orow = r->ov_buf + (size_t)oy2 * ox;
      for (uint32_t ox2 = 0; ox2 < ox; ox2++) {
        uint32_t x0 = ox2 * 4, x1 = x0 + 4 > l->nx ? l->nx : x0 + 4;
        uint32_t sum = 0;
        for (uint32_t y = y0; y < y1; y++) {
          const uint8_t *sr = sbase + (size_t)y * src->nx;
          for (uint32_t x = x0; x < x1; x++) sum += sr[x];
        }
        orow[ox2] = (uint8_t)(sum / ((y1 - y0) * (x1 - x0)));
      }
    }
    VkMemoryToImageCopyEXT oregion = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
        .pHostPointer = r->ov_buf,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, (int32_t)layer},
        .imageExtent = {ox, oyd, 1},
    };
    VkCopyMemoryToImageInfoEXT oci = {
        .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
        .dstImage = r->overview.img,
        .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .regionCount = 1,
        .pRegions = &oregion,
    };
    if (r->fp_copy_mem(r->vk.dev, &oci) != VK_SUCCESS) return -1;
  }
  double ms = (double)(now_ns() - t0) / 1e6;
  if (full || ms > 5.0)
    printf("slab: %s %u slice(s) -> z0=%u in %.1f ms\n", full ? "window" : "scrolled", s1 - s0,
           z0, ms);
  r->slab_z0 = (int64_t)z0;
  return 0;
}

void r3d_slab_params(const r3d_renderer *r, r3d_frame_params *p) {
  p->slab_grid = r->slab.gx | (r->slab.gy << 8);
  p->slab_wz = r->slab.wz;
  p->slab_z0 = (float)r->slab_z0;
  p->slab_nx = (float)r->slab.nx;
  p->slab_ny = (float)r->slab.ny;
  p->slab_px = (float)r->slab.px;
  p->slab_py = (float)r->slab.py;
  p->slab_depth = r->slab.wz - 2; /* max; caller may lower it per frame */
}

/* ---------- bricks: two-tier GPU cache (warm compressed / hot atlas) ------ */

#define BR_INVALID 0xFFFFFFFFu
#define BR_SLOT_DIM 128u
#define BR_AMIPS 4u
#define BR_NOISE_FLOOR 5 /* decoded-air codec noise ceiling (LSBs) */
#define BR_MAX_BATCH 32u

static int bcand_cmp(const void *a, const void *b) {
  float d = ((const struct bcand *)a)->d2 - ((const struct bcand *)b)->d2;
  return d < 0.0f ? -1 : (d > 0.0f ? 1 : 0);
}

/* warm-tier allocator: offset-sorted first-fit free list, 64-byte granules */
static uint32_t warm_align(uint32_t n) { return (n + 63u) & ~63u; }

static uint32_t warm_alloc(r3d_renderer *r, uint32_t len) {
  len = warm_align(len);
  for (uint32_t i = 0; i < r->bs.nwf; i++) {
    struct wfnode *f = &r->bs.wfree[i];
    if (f->len < len) continue;
    uint32_t off = f->off;
    f->off += len;
    f->len -= len;
    if (f->len == 0) memmove(f, f + 1, (size_t)(--r->bs.nwf - i) * sizeof *f);
    return off;
  }
  return BR_INVALID;
}

static void warm_release(r3d_renderer *r, uint32_t off, uint32_t rawlen) {
  uint32_t len = warm_align(rawlen);
  struct wfnode *fl = r->bs.wfree;
  uint32_t i = 0;
  while (i < r->bs.nwf && fl[i].off < off) i++;
  if (i > 0 && fl[i - 1].off + fl[i - 1].len == off) { /* merge into predecessor */
    fl[i - 1].len += len;
    if (i < r->bs.nwf && fl[i - 1].off + fl[i - 1].len == fl[i].off) {
      fl[i - 1].len += fl[i].len;
      memmove(&fl[i], &fl[i + 1], (size_t)(--r->bs.nwf - i) * sizeof *fl);
    }
    return;
  }
  if (i < r->bs.nwf && off + len == fl[i].off) { /* merge into successor */
    fl[i].off = off;
    fl[i].len += len;
    return;
  }
  if (r->bs.nwf == r->bs.cwf) {
    r->bs.cwf = r->bs.cwf ? r->bs.cwf * 2 : 64;
    r->bs.wfree = fl = realloc(fl, (size_t)r->bs.cwf * sizeof *fl);
  }
  memmove(&fl[i + 1], &fl[i], (size_t)(r->bs.nwf++ - i) * sizeof *fl);
  fl[i] = (struct wfnode){off, len};
}

static bool warm_evict_one(r3d_renderer *r) {
  uint32_t best = BR_INVALID, bu = UINT32_MAX;
  for (uint32_t b = 0; b < r->bs.nb; b++)
    if (r->bs.warm_off[b] != BR_INVALID && r->bs.warm_use[b] != r->bs.frame &&
        r->bs.warm_use[b] < bu) {
      bu = r->bs.warm_use[b];
      best = b;
    }
  if (best == BR_INVALID) return false;
  warm_release(r, r->bs.warm_off[best], r->bs.warm_len[best]);
  r->bs.warm_bytes -= r->bs.warm_len[best];
  r->bs.warm_off[best] = BR_INVALID;
  r->bs.warm_bricks--;
  return true;
}

/* compressed blob for brick b, resident in the warm tier when it fits (LRU
 * evictions as needed); falls back to the mmap'd shard when the tier thrashes.
 * Current-frame entries are never evicted, so batch blob pointers stay valid. */
static const uint8_t *warm_get(r3d_renderer *r, uint32_t b, size_t *n) {
  if (r->bs.warm_off[b] != BR_INVALID) {
    r->bs.warm_use[b] = r->bs.frame;
    *n = r->bs.warm_len[b];
    return (const uint8_t *)r->bs.warm.mapped + r->bs.warm_off[b];
  }
  size_t sz = 0;
  const uint8_t *blob = c5d_shard_brick(&r->bs.sr, b, &sz);
  if (!blob) return NULL;
  *n = sz;
  if ((uint64_t)warm_align((uint32_t)sz) > r->bs.warm_cap) return blob;
  uint32_t off;
  while ((off = warm_alloc(r, (uint32_t)sz)) == BR_INVALID)
    if (!warm_evict_one(r)) return blob;
  memcpy((uint8_t *)r->bs.warm.mapped + off, blob, sz);
  r->bs.warm_off[b] = off;
  r->bs.warm_len[b] = (uint32_t)sz;
  r->bs.warm_use[b] = r->bs.frame;
  r->bs.warm_bricks++;
  r->bs.warm_bytes += sz;
  return (const uint8_t *)r->bs.warm.mapped + off;
}

static uint32_t bricks_pick_slot(const r3d_renderer *r) {
  uint32_t best = BR_INVALID, bu = UINT32_MAX;
  for (uint32_t s = 0; s < r->bs.nslots; s++) {
    if (r->bs.slot_brick[s] == BR_INVALID) return s;
    if (r->bs.slot_use[s] != r->bs.frame && r->bs.slot_use[s] < bu) {
      bu = r->bs.slot_use[s];
      best = s;
    }
  }
  return best;
}

/* after a decode batch: per-slot mip blits + incremental world-indexed
 * occupancy (region max-reduce, then re-dilate each brick plus a 1-block halo
 * so neighbor borders self-heal as fill order interleaves). One submission;
 * atlas and occupancy images live in GENERAL for their whole lifetime. */
static int bricks_post_fill(r3d_renderer *r, const uint32_t *sel_slot, const uint32_t *sel_b,
                            uint32_t n) {
  uint32_t abpa = r->bricks_abpa, bpa = r->bricks_bpa, odim = bpa * 16u;
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  for (uint32_t m = 1; m < BR_AMIPS; m++) {
    for (uint32_t i = 0; i < n; i++) {
      uint32_t s = sel_slot[i];
      int32_t sx = (int32_t)(s % abpa), sy = (int32_t)((s / abpa) % abpa),
              sz = (int32_t)(s / (abpa * abpa));
      int32_t d0 = (int32_t)(BR_SLOT_DIM >> (m - 1)), d1 = d0 / 2;
      VkImageBlit2 blit = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
          .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 0, 1},
          .srcOffsets = {{sx * d0, sy * d0, sz * d0},
                         {(sx + 1) * d0, (sy + 1) * d0, (sz + 1) * d0}},
          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1},
          .dstOffsets = {{sx * d1, sy * d1, sz * d1},
                         {(sx + 1) * d1, (sy + 1) * d1, (sz + 1) * d1}},
      };
      VkBlitImageInfo2 bi2 = {.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                              .srcImage = r->brick_atlas.img,
                              .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                              .dstImage = r->brick_atlas.img,
                              .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                              .regionCount = 1,
                              .pRegions = &blit,
                              .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2(cmd, &bi2);
    }
    r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, m, 1);
  }
  for (uint32_t i = 0; i < n; i++) {
    uint32_t s = sel_slot[i], b = sel_b[i];
    uint32_t pc[6] = {(s % abpa) * BR_SLOT_DIM,          ((s / abpa) % abpa) * BR_SLOT_DIM,
                      (s / (abpa * abpa)) * BR_SLOT_DIM, (b % bpa) * 16u,
                      ((b / bpa) % bpa) * 16u,           (b / (bpa * bpa)) * 16u};
    r3d_vkcomp_dispatch(cmd, &r->bs.omax, pc, sizeof pc, 64, 1, 1);
  }
  r3d_vk_image_barrier(cmd, r->bs.occraw.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
  for (uint32_t i = 0; i < n; i++) {
    uint32_t b = sel_b[i];
    uint32_t bx = (b % bpa) * 16u, by = ((b / bpa) % bpa) * 16u, bz = (b / (bpa * bpa)) * 16u;
    uint32_t pc[5] = {bx ? bx - 1 : 0, by ? by - 1 : 0, bz ? bz - 1 : 0, 18u, odim};
    r3d_vkcomp_dispatch(cmd, &r->bs.odil, pc, sizeof pc, (18 * 18 * 18 + 63) / 64, 1, 1);
  }
  r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, BR_AMIPS);
  r3d_vk_image_barrier(cmd, r->brick_occ.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
  return r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
}

static int img_general_clear(r3d_renderer *r, r3d_vkimage *img) {
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
  VkClearColorValue z = {{0}};
  VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(cmd, img->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rng);
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0, 1);
  return r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
}

int r3d_bricks_begin(r3d_renderer *r, const char *c5s_path, uint32_t pool_bpa,
                     uint32_t warm_mb) {
  if (c5d_shard_open(c5s_path, &r->bs.sr) != 0) {
    fprintf(stderr, "bricks: cannot open %s\n", c5s_path);
    return -1;
  }
  r->bs.sr_open = true;
  if (r->bs.sr.foot.brick_dim != 128) {
    fprintf(stderr, "bricks: brick_dim %u unsupported\n", r->bs.sr.foot.brick_dim);
    return -1;
  }
  uint32_t bpa = r->bs.sr.foot.shard_dim / 128, nb = r->bs.sr.foot.nbricks;
  /* hot pool: identity (slot == brick, direct sampling) when the whole volume
   * fits; otherwise a smaller LRU atlas fed by the streaming pump */
  uint32_t abpa = pool_bpa ? pool_bpa : (bpa < 8 ? bpa : 8);
  if (abpa > bpa) abpa = bpa;
  if (abpa > 12) abpa = 12; /* maxImageDimension3D=2048 / ~4 GiB allocation caps */
  bool streaming = abpa < bpa;
  const uint32_t SLOT = BR_SLOT_DIM, APRON = 0;
  uint32_t adim = abpa * SLOT;
  const uint32_t amips = BR_AMIPS;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){adim, adim, adim}, amips,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->brick_atlas) != 0)
    return -1;
  VkImageViewCreateInfo m0v = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = r->brick_atlas.img,
      .viewType = VK_IMAGE_VIEW_TYPE_3D,
      .format = VK_FORMAT_R8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  if (vkCreateImageView(r->vk.dev, &m0v, NULL, &r->brick_atlas_mip0) != VK_SUCCESS) return -1;
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0, amips);
  if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) return -1;

  if (r3d_vkc5d_create(&r->c5d, &r->vk, r->cfg.spv_dir, 8) != 0) return -1;

  /* world-indexed occupancy images (cleared: absent bricks read as empty) +
   * the persistent region-form occupancy kernels */
  uint32_t odim = bpa * 16u;
  VkImageUsageFlags ou =
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){odim, odim, odim}, 1, ou,
                         &r->bs.occraw) != 0 ||
      r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){odim, odim, odim}, 1, ou,
                         &r->brick_occ) != 0)
    return -1;
  if (img_general_clear(r, &r->bs.occraw) != 0 || img_general_clear(r, &r->brick_occ) != 0)
    return -1;
  {
    VkDescriptorType tt[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    char sp[1024];
    snprintf(sp, sizeof sp, "%s/occmax.spv", r->cfg.spv_dir);
    if (r3d_vkcomp_create(&r->vk, sp, tt, 2, 24, &r->bs.omax) != 0) return -1;
    snprintf(sp, sizeof sp, "%s/occdilate.spv", r->cfg.spv_dir);
    if (r3d_vkcomp_create(&r->vk, sp, tt, 2, 20, &r->bs.odil) != 0) return -1;
    r->bs.comp_ready = true;
    r3d_vkcomp_bind_image(&r->vk, &r->bs.omax, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->brick_atlas.view, r->samp_near, VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&r->vk, &r->bs.omax, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                          r->bs.occraw.view, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&r->vk, &r->bs.odil, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->bs.occraw.view, r->samp_near, VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&r->vk, &r->bs.odil, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                          r->brick_occ.view, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
  }

  /* page table + CPU residency state */
  r3d_vkbuf_destroy(&r->vk, &r->page_buf);
  if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)nb * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            &r->page_buf) != 0)
    return -1;
  uint32_t *page = r->page_buf.mapped;
  for (uint32_t b = 0; b < nb; b++) page[b] = BR_INVALID;
  uint32_t nslots = abpa * abpa * abpa;
  r->bs.nb = nb;
  r->bs.nslots = nslots;
  r->bs.slot_brick = malloc((size_t)nslots * 4);
  r->bs.slot_use = calloc(nslots, 4);
  r->bs.brick_slot = malloc((size_t)nb * 4);
  r->bs.brick_maxk = malloc((size_t)nb * 2);
  r->bs.cands = malloc((size_t)nb * sizeof(struct bcand));
  uint32_t scap = streaming ? BR_MAX_BATCH : nb;
  r->bs.srcs = malloc((size_t)scap * sizeof(r3d_c5d_src));
  r->bs.sel_b = malloc((size_t)scap * 4);
  r->bs.sel_slot = malloc((size_t)scap * 4);
  r->bs.maxes = malloc(scap);
  r->bs.warm_off = malloc((size_t)nb * 4);
  r->bs.warm_len = calloc(nb, 4);
  r->bs.warm_use = calloc(nb, 4);
  if (!r->bs.slot_brick || !r->bs.slot_use || !r->bs.brick_slot || !r->bs.brick_maxk ||
      !r->bs.cands || !r->bs.srcs || !r->bs.sel_b || !r->bs.sel_slot || !r->bs.maxes ||
      !r->bs.warm_off || !r->bs.warm_len || !r->bs.warm_use)
    return -1;
  memset(r->bs.slot_brick, 0xFF, (size_t)nslots * 4);
  memset(r->bs.brick_slot, 0xFF, (size_t)nb * 4);
  memset(r->bs.brick_maxk, 0xFF, (size_t)nb * 2); /* -1 = unknown */
  memset(r->bs.warm_off, 0xFF, (size_t)nb * 4);
  r->bricks_bpa = bpa;
  r->bricks_abpa = abpa;

  if (!streaming) {
    /* identity residency: decode the whole shard up front, slot == brick */
    uint32_t np = 0;
    for (uint32_t b = 0; b < nb; b++) {
      size_t n = 0;
      const uint8_t *blob = c5d_shard_brick(&r->bs.sr, b, &n);
      if (!blob) { /* missing brick: page stays invalid, never requested */
        r->bs.brick_maxk[b] = 0;
        continue;
      }
      uint32_t bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
      r->bs.srcs[np] = (r3d_c5d_src){.blob = blob, .n = n,
                                     .sx = bx * SLOT + APRON, .sy = by * SLOT + APRON,
                                     .sz = bz * SLOT + APRON};
      r->bs.sel_b[np] = b;
      r->bs.sel_slot[np] = b;
      np++;
    }
    printf("bricks: decoding %u/%u bricks on GPU (%s)...\n", np, nb,
           getenv("R3D_C5D_HYBRID") ? "hybrid" : "full-GPU");
    uint64_t t0 = now_ns();
    int rc = r3d_vkc5d_decode(r->c5d, r->bs.srcs, np, r->brick_atlas_mip0, r->bs.maxes);
    double ms = (double)(now_ns() - t0) / 1e6;
    if (rc != 0) {
      fprintf(stderr, "bricks: GPU decode failed\n");
      return -1;
    }
    printf("bricks: decoded %u bricks in %.0f ms (%.0f bricks/s, %.2f GB/s raw)\n", np, ms,
           (double)np * 1000.0 / ms, (double)np * 2.097152 / ms);
    if (bricks_post_fill(r, r->bs.sel_slot, r->bs.sel_b, np) != 0) return -1;
    for (uint32_t k = 0; k < np; k++) {
      uint32_t b = r->bs.sel_b[k];
      page[b] = b | ((uint32_t)r->bs.maxes[k] << 24);
      r->bs.slot_brick[b] = b;
      r->bs.brick_slot[b] = b;
      r->bs.brick_maxk[b] = r->bs.maxes[k];
    }
    r->bricks_identity = np == nb;
  } else {
    r->bs.warm_cap = (uint64_t)(warm_mb ? warm_mb : 256) << 20;
    if (r->bs.warm_cap > (3ull << 30)) r->bs.warm_cap = 3ull << 30; /* u32 offsets */
    if (r3d_vkbuf_create_host(&r->vk, r->bs.warm_cap,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &r->bs.warm) != 0)
      return -1;
    warm_release(r, 0, (uint32_t)r->bs.warm_cap); /* one node spanning the tier */
    r->bs.active = true;
    r->bricks_identity = false;
    printf("bricks: streaming %u^3 bricks through a %u^3-slot hot atlas (%llu MB warm tier)\n",
           bpa, abpa, (unsigned long long)(r->bs.warm_cap >> 20));
  }

  if (getenv("R3D_DUMP_MIP1")) { /* debug: write one z-slice of mip1 as PGM */
    uint32_t d1 = adim / 2;
    r3d_vkbuf rb;
    if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)d1 * d1, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &rb) == 0) {
      VkCommandBuffer dc = r3d_vk_oneshot_begin(&r->vk, r->pool);
      r3d_vk_image_barrier(dc, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, 1, 1);
      VkBufferImageCopy reg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1},
                               .imageOffset = {0, 0, (int32_t)(d1 / 2)},
                               .imageExtent = {d1, d1, 1}};
      vkCmdCopyImageToBuffer(dc, r->brick_atlas.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb.buf, 1, &reg);
      r3d_vk_image_barrier(dc, r->brick_atlas.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, 1, 1);
      r3d_vk_oneshot_end(&r->vk, r->pool, dc);
      FILE *f = fopen("mip1_slice.pgm", "wb");
      if (f) {
        fprintf(f, "P5\n%u %u\n255\n", d1, d1);
        fwrite(rb.mapped, 1, (size_t)d1 * d1, f);
        fclose(f);
        printf("bricks: dumped mip1_slice.pgm (%ux%u)\n", d1, d1);
      }
      r3d_vkbuf_destroy(&r->vk, &rb);
    }
  }

  vkDeviceWaitIdle(r->vk.dev);
  /* binding 3 (cube-mode occupancy slot) now serves bricks mode too */
  write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->brick_occ.view,
                   r->samp_near, VK_IMAGE_LAYOUT_GENERAL);
  write_image_dset(r, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->brick_atlas.view,
                   r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  VkDescriptorBufferInfo pbi = {.buffer = r->page_buf.buf, .range = VK_WHOLE_SIZE};
  VkWriteDescriptorSet pw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = r->dset,
                             .dstBinding = 5,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &pbi};
  vkUpdateDescriptorSets(r->vk.dev, 1, &pw, 0, NULL);
  return 0;
}

void r3d_bricks_params(const r3d_renderer *r, r3d_frame_params *p) {
  p->brick_mode = r->bricks_bpa | (r->bricks_abpa << 8) |
                  (r->bricks_identity ? 0x10000u : 0u);
}

void r3d_bricks_stream(r3d_renderer *r, const float eye[3], const float fwd[3], float half_tan,
                       float gate, uint32_t budget) {
  r->bs.last_inflight = 0;
  if (!r->bs.active || budget == 0) return;
  if (budget > BR_MAX_BATCH) budget = BR_MAX_BATCH;
  r->bs.frame++;
  uint32_t bpa = r->bricks_bpa, abpa = r->bricks_abpa, nb = r->bs.nb;
  float inv = 1.0f / (float)bpa;
  float brad = 0.8660254f * inv; /* half brick diagonal, normalized volume units */
  float tanw = half_tan * 1.15f + 1e-3f;
  int g8 = (int)(gate * 255.0f + 0.5f);
  if (g8 < BR_NOISE_FLOOR) g8 = BR_NOISE_FLOOR;

  /* desired set: bricks inside the (slightly widened) view cone or hugging the
   * camera. Resident ones get their LRU stamps; the rest become requests. */
  uint32_t ncand = 0;
  for (uint32_t b = 0; b < nb; b++) {
    if (r->bs.brick_maxk[b] >= 0 && r->bs.brick_maxk[b] < g8) continue; /* known empty */
    float cx = ((float)(b % bpa) + 0.5f) * inv - eye[0];
    float cy = ((float)((b / bpa) % bpa) + 0.5f) * inv - eye[1];
    float cz = ((float)(b / (bpa * bpa)) + 0.5f) * inv - eye[2];
    float d2 = cx * cx + cy * cy + cz * cz;
    float along = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
    bool vis = d2 < (inv + brad) * (inv + brad); /* hugging the camera */
    if (!vis && along > 0.0f) {
      float perp2 = d2 - along * along;
      float rad = along * tanw + brad;
      vis = perp2 < rad * rad;
    }
    if (!vis) continue;
    uint32_t slot = r->bs.brick_slot[b];
    if (slot != BR_INVALID) {
      r->bs.slot_use[slot] = r->bs.frame;
      if (r->bs.warm_off[b] != BR_INVALID) r->bs.warm_use[b] = r->bs.frame;
      continue;
    }
    r->bs.cands[ncand++] = (struct bcand){d2, b};
  }
  if (!ncand) return;
  qsort(r->bs.cands, ncand, sizeof(struct bcand), bcand_cmp);

  /* nearest-first: warm-tier blob + hot slot per request, up to the budget */
  uint32_t n = 0, nevict = 0;
  uint32_t evict[BR_MAX_BATCH];
  for (uint32_t k = 0; k < ncand && n < budget; k++) {
    uint32_t b = r->bs.cands[k].b;
    size_t bn = 0;
    const uint8_t *blob = warm_get(r, b, &bn);
    if (!blob) { /* absent in the shard: never request again */
      r->bs.brick_maxk[b] = 0;
      continue;
    }
    uint32_t s = bricks_pick_slot(r);
    if (s == BR_INVALID) break; /* every slot wanted this frame: don't thrash */
    uint32_t old = r->bs.slot_brick[s];
    if (old != BR_INVALID) { /* LRU eviction; page invalidated after the drain */
      r->bs.brick_slot[old] = BR_INVALID;
      evict[nevict++] = old;
    }
    r->bs.slot_brick[s] = b;
    r->bs.slot_use[s] = r->bs.frame;
    r->bs.brick_slot[b] = s;
    r->bs.srcs[n] = (r3d_c5d_src){.blob = blob,
                                  .n = bn,
                                  .sx = (s % abpa) * BR_SLOT_DIM,
                                  .sy = ((s / abpa) % abpa) * BR_SLOT_DIM,
                                  .sz = (s / (abpa * abpa)) * BR_SLOT_DIM};
    r->bs.sel_b[n] = b;
    r->bs.sel_slot[n] = s;
    n++;
  }
  if (!n) return;

  /* the decode overwrites atlas slots and the page table that in-flight frames
   * may still be sampling: drain the queue first (pipelined handoff via the
   * timeline semaphore is the known follow-up) */
  vkQueueWaitIdle(r->vk.queue);
  uint32_t *page = r->page_buf.mapped;
  for (uint32_t i = 0; i < nevict; i++) page[evict[i]] = BR_INVALID;
  if (r3d_vkc5d_decode(r->c5d, r->bs.srcs, n, r->brick_atlas_mip0, r->bs.maxes) != 0 ||
      bricks_post_fill(r, r->bs.sel_slot, r->bs.sel_b, n) != 0) {
    fprintf(stderr, "bricks: stream decode failed (batch of %u)\n", n);
    for (uint32_t i = 0; i < n; i++) { /* roll back; the bricks re-request */
      r->bs.slot_brick[r->bs.sel_slot[i]] = BR_INVALID;
      r->bs.brick_slot[r->bs.sel_b[i]] = BR_INVALID;
    }
    return;
  }
  for (uint32_t i = 0; i < n; i++) {
    uint32_t b = r->bs.sel_b[i], s = r->bs.sel_slot[i];
    uint8_t m = r->bs.maxes[i];
    r->bs.brick_maxk[b] = m;
    if (m < BR_NOISE_FLOOR) { /* decoded empty: free the slot, page stays invalid */
      r->bs.slot_brick[s] = BR_INVALID;
      r->bs.brick_slot[b] = BR_INVALID;
      continue;
    }
    page[b] = s | ((uint32_t)m << 24);
  }
  r->bs.last_inflight = n;
}

void r3d_bricks_get_stats(const r3d_renderer *r, r3d_bricks_stats *st) {
  memset(st, 0, sizeof *st);
  st->nb = r->bs.nb;
  st->hot_cap = r->bs.nslots;
  for (uint32_t s = 0; s < r->bs.nslots; s++)
    if (r->bs.slot_brick[s] != BR_INVALID) st->hot++;
  st->warm_bricks = r->bs.warm_bricks;
  st->warm_bytes = r->bs.warm_bytes;
  st->warm_cap = r->bs.warm_cap;
  st->inflight = r->bs.last_inflight;
}

int r3d_clip_begin(r3d_renderer *r, const char *band_dir, const char *pyramid_dir,
                   uint32_t band_z, uint32_t depth_max) {
  if (!r->vk.caps.host_image_copy || !r->fp_transition || !r->fp_copy_mem) {
    fprintf(stderr, "clip: needs VK_EXT_host_image_copy\n");
    return -1;
  }
  if (r->vk.caps.max_push_bytes < sizeof(r3d_frame_params)) {
    fprintf(stderr, "clip: push constants %zu > device max %u\n", sizeof(r3d_frame_params),
            r->vk.caps.max_push_bytes);
    return -1;
  }
  if (!r->samp_slab) { /* REPEAT-W ring sampler (shared with slab mode) */
    VkSamplerCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
    if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_slab) != VK_SUCCESS) return -1;
  }
  uint64_t z0 = (uint64_t)band_z * 1024 + 512 - depth_max / 2;
  if (r3d_vkclip_create(&r->clipm, &r->vk, r->fp_transition, r->fp_copy_mem, band_dir,
                        pyramid_dir, band_z, depth_max, 21504, 21504, z0) != 0)
    return -1;

  VkDescriptorImageInfo ii[16];
  for (uint32_t e = 0; e < 16; e++) {
    uint32_t l = e < R3D_CLIP_LEVELS ? e : R3D_CLIP_LEVELS - 1;
    ii[e] = (VkDescriptorImageInfo){.sampler = r->samp_slab,
                                    .imageView = r3d_vkclip_view(r->clipm, l),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  }
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = 16,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
  return 0;
}

int r3d_clip_frame(r3d_renderer *r, double fx, double fy, uint64_t z0, r3d_frame_params *p) {
  if (!r->clipm) return -1;
  r3d_vkclip_update(r->clipm, (int64_t)fx, (int64_t)fy, z0);
  if (r3d_vkclip_pump(r->clipm, r->timeline, r->timeline_value) != 0) return -1;
  r3d_vkclip_params(r->clipm, p);
  return 0;
}

int r3d_frame(r3d_renderer *r, const r3d_frame_params *p, r3d_frame_stats *st) {
  if (st) memset(st, 0, sizeof *st);
  uint32_t slot = r->slot;
  uint64_t tp = now_ns();

  /* pace: wait for this slot's previous submission, then read its timestamps */
  if (r->slot_value[slot]) {
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &r->timeline,
        .pValues = &r->slot_value[slot],
    };
    if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return -1;
    if (r->slot_has_query[slot]) {
      uint64_t ts[4] = {0};
      if (vkGetQueryPoolResults(r->vk.dev, r->query, slot * 4, 4, sizeof ts, ts, sizeof ts[0],
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        double period = r->vk.caps.ts_period_ns;
        r->slot_gpu[slot][0] = (uint64_t)((double)(ts[3] - ts[0]) * period);
        r->slot_gpu[slot][1] = (uint64_t)((double)(ts[1] - ts[0]) * period);
        r->slot_gpu[slot][2] = (uint64_t)((double)(ts[2] - ts[1]) * period);
        r->slot_gpu[slot][3] = (uint64_t)((double)(ts[3] - ts[2]) * period);
      }
    }
  }
  if (st) {
    st->gpu_ns = r->slot_gpu[slot][0];
    st->gpu_raycast_ns = r->slot_gpu[slot][1];
    st->gpu_blit_ns = r->slot_gpu[slot][2];
    st->gpu_gui_ns = r->slot_gpu[slot][3];
    st->cpu_wait_ns = now_ns() - tp;
  }

  tp = now_ns();
  uint32_t img = 0;
  VkResult ar = vkAcquireNextImageKHR(r->vk.dev, r->swap.swapchain, UINT64_MAX,
                                      r->acquire[slot], VK_NULL_HANDLE, &img);
  if (st) st->cpu_acquire_ns = now_ns() - tp;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    if (r->gui_open) {
      r3d_vkgui_discard();
      r->gui_open = false;
    }
    int rc = r3d_resize(r);
    return rc == 0 ? 1 : rc; /* 1 = frame skipped, retry next loop */
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) return -1;

  tp = now_ns();
  VkCommandBuffer cmd = r->cmd[slot];
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cmd, &bi);

  if (r->query) vkCmdResetQueryPool(cmd, r->query, slot * 4, 4);

  /* offscreen: whatever -> GENERAL for compute write (contents fully overwritten) */
  r3d_vk_image_barrier(cmd, r->offscreen.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       0, 1);

  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, r->query, slot * 4);
  uint32_t rmode = p->clip_valid ? 2u : (p->brick_mode ? 3u : (p->slab_grid ? 1u : 0u));
  uint32_t wgx = rmode == 0 ? r->wg_x : 16u, wgy = rmode == 0 ? r->wg_y : 8u;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->raycast[rmode]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe_layout, 0, 1, &r->dset, 0,
                          NULL);
  /* p->viewport may be smaller than the drawable (adaptive resolution while
   * the camera moves): render into the top-left region, blit upscales */
  uint32_t rw = p->viewport[0] ? p->viewport[0] : r->swap.extent.width;
  uint32_t rh = p->viewport[1] ? p->viewport[1] : r->swap.extent.height;
  if (rw > r->offscreen.extent.width) rw = r->offscreen.extent.width;
  if (rh > r->offscreen.extent.height) rh = r->offscreen.extent.height;
  vkCmdPushConstants(cmd, r->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(r3d_frame_params), p);
  vkCmdDispatch(cmd, (rw + wgx - 1) / wgx, (rh + wgy - 1) / wgy, 1);
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, r->query, slot * 4 + 1);
  r->slot_has_query[slot] = r->query != VK_NULL_HANDLE;

  /* offscreen -> blit src; swapchain image -> blit dst */
  r3d_vk_image_barrier(cmd, r->offscreen.img, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT, 0, 1);
  r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);

  VkImageBlit2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {(int32_t)rw, (int32_t)rh, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0},
                     {(int32_t)r->swap.extent.width, (int32_t)r->swap.extent.height, 1}},
  };
  VkBlitImageInfo2 blit = {
      .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage = r->offscreen.img,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage = r->swap.images[img],
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &region,
      /* LINEAR: identity when rendering at full size, smooth when upscaling */
      .filter = VK_FILTER_LINEAR,
  };
  vkCmdBlitImage2(cmd, &blit);
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, r->query, slot * 4 + 2);

  if (r->gui_open) {
    r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         0, 1);
    r3d_vkgui_render(cmd, r->swap.views[img], r->swap.extent);
    r->gui_open = false;
    r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, 0, 1);
  } else {
    r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                         0, 1);
  }
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r->query, slot * 4 + 3);
  vkEndCommandBuffer(cmd);
  if (st) st->cpu_record_ns = now_ns() - tp;

  tp = now_ns();
  uint64_t signal_value = ++r->timeline_value;
  VkSemaphoreSubmitInfo waits[] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = r->acquire[slot],
       .stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT},
  };
  VkSemaphoreSubmitInfo signals[] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = r->swap.render_done[img],
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = r->timeline,
       .value = signal_value,
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
  };
  VkCommandBufferSubmitInfo csi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                   .commandBuffer = cmd};
  VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = waits,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &csi,
      .signalSemaphoreInfoCount = 2,
      .pSignalSemaphoreInfos = signals,
  };
  if (vkQueueSubmit2(r->vk.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
  r->slot_value[slot] = signal_value;

  VkPresentInfoKHR pi = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &r->swap.render_done[img],
      .swapchainCount = 1,
      .pSwapchains = &r->swap.swapchain,
      .pImageIndices = &img,
  };
  VkResult pr = vkQueuePresentKHR(r->vk.queue, &pi);
  if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
    if (r3d_resize(r) != 0) return -1;
  } else if (pr != VK_SUCCESS) {
    return -1;
  }

  if (st) st->cpu_submit_ns = now_ns() - tp;
  r->slot = (slot + 1) % FRAMES_IN_FLIGHT;
  return 0;
}

int r3d_read_frame(r3d_renderer *r, uint8_t *rgba, uint32_t *w, uint32_t *h) {
  *w = r->offscreen.extent.width;
  *h = r->offscreen.extent.height;
  if (!rgba) return 0;

  VkDeviceSize need = (VkDeviceSize)*w * *h * 4;
  if (r->readback.size < need) {
    r3d_vkbuf_destroy(&r->vk, &r->readback);
    if (r3d_vkbuf_create_host(&r->vk, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &r->readback) != 0)
      return -1;
  }
  vkDeviceWaitIdle(r->vk.dev); /* screenshot path; simplicity over speed */

  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  if (!cmd) return -1;
  /* offscreen was left in TRANSFER_SRC by the last frame */
  VkBufferImageCopy2 region = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {*w, *h, 1},
  };
  VkCopyImageToBufferInfo2 ci = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
      .srcImage = r->offscreen.img,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstBuffer = r->readback.buf,
      .regionCount = 1,
      .pRegions = &region,
  };
  vkCmdCopyImageToBuffer2(cmd, &ci);
  if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) return -1;
  memcpy(rgba, r->readback.mapped, need);
  return 0;
}
