/* Vulkan implementation of render.h. Frame graph (M1):
 *   raycast.comp (storage image, GENERAL) -> blit -> swapchain -> present
 * 2 frames in flight, timeline semaphore for CPU pacing, binary semaphores for
 * WSI, timestamp queries around the dispatch. */
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "render/render.h"
#include "vk/vkctx.h"
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
  VkSampler samp_vol;    /* trilinear + mip linear, clamp */
  VkSampler samp_tf;     /* linear, clamp */

  PFN_vkTransitionImageLayoutEXT fp_transition;
  PFN_vkCopyMemoryToImageEXT fp_copy_mem;

  VkDescriptorSetLayout dsl;
  VkPipelineLayout pipe_layout;
  VkPipeline raycast;
  VkDescriptorPool dpool;
  VkDescriptorSet dset;

  uint32_t wg_x, wg_y; /* raycast workgroup size (R3D_WG=8x8|16x8|16x16) */

  VkCommandPool pool;
  VkCommandBuffer cmd[FRAMES_IN_FLIGHT];
  VkSemaphore acquire[FRAMES_IN_FLIGHT];
  VkSemaphore timeline;
  uint64_t timeline_value;
  uint64_t slot_value[FRAMES_IN_FLIGHT];
  uint64_t slot_gpu_ns[FRAMES_IN_FLIGHT];
  VkQueryPool query;
  bool slot_has_query[FRAMES_IN_FLIGHT];
  uint32_t slot;

  r3d_vkbuf readback; /* lazily sized for screenshots */
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
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
  };
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
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

  const char *wg = getenv("R3D_WG");
  const char *spv_name = "raycast.spv";
  r->wg_x = 16;
  r->wg_y = 8;
  if (wg && strcmp(wg, "8x8") == 0) {
    spv_name = "raycast_8x8.spv";
    r->wg_x = r->wg_y = 8;
  } else if (wg && strcmp(wg, "16x16") == 0) {
    spv_name = "raycast_16x16.spv";
    r->wg_x = r->wg_y = 16;
  }
  uint32_t *spv = NULL;
  size_t spv_n = 0;
  if (load_spv(r->cfg.spv_dir, spv_name, &spv, &spv_n) != 0) return -1;
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
  res = vkCreateComputePipelines(r->vk.dev, VK_NULL_HANDLE, 1, &cpci, NULL, &r->raycast);
  vkDestroyShaderModule(r->vk.dev, mod, NULL);
  if (res != VK_SUCCESS) return -1;

  VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
  };
  VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
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

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
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
        .queryCount = FRAMES_IN_FLIGHT * 2,
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

  /* dummy 2^3 volume so the descriptor set is valid before the real upload
   * (2, not 1: vkres dimensionality heuristic needs depth>1 for a 3D image) */
  {
    uint8_t zeros[8] = {0};
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){2, 2, 2}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->volume) != 0)
      goto fail;
    if (upload_small_image(r, &r->volume, zeros, sizeof zeros) != 0) goto fail;
    write_image_dset(r, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->volume.view,
                     r->samp_vol, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

  *out = r;
  return 0;
fail:
  r3d_destroy(r);
  return -1;
}

void r3d_destroy(r3d_renderer *r) {
  if (!r) return;
  if (r->vk.dev) vkDeviceWaitIdle(r->vk.dev);
  r3d_vkbuf_destroy(&r->vk, &r->readback);
  if (r->query) vkDestroyQueryPool(r->vk.dev, r->query, NULL);
  if (r->timeline) vkDestroySemaphore(r->vk.dev, r->timeline, NULL);
  for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    if (r->acquire[i]) vkDestroySemaphore(r->vk.dev, r->acquire[i], NULL);
  if (r->pool) vkDestroyCommandPool(r->vk.dev, r->pool, NULL);
  if (r->dpool) vkDestroyDescriptorPool(r->vk.dev, r->dpool, NULL);
  if (r->raycast) vkDestroyPipeline(r->vk.dev, r->raycast, NULL);
  if (r->pipe_layout) vkDestroyPipelineLayout(r->vk.dev, r->pipe_layout, NULL);
  if (r->dsl) vkDestroyDescriptorSetLayout(r->vk.dev, r->dsl, NULL);
  if (r->samp_vol) vkDestroySampler(r->vk.dev, r->samp_vol, NULL);
  if (r->samp_tf) vkDestroySampler(r->vk.dev, r->samp_tf, NULL);
  r3d_vkimage_destroy(&r->vk, &r->volume);
  r3d_vkimage_destroy(&r->vk, &r->tf);
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

  vkDeviceWaitIdle(r->vk.dev);
  r3d_vkimage_destroy(&r->vk, &r->volume);
  r->volume = im;
  write_image_dset(r, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->volume.view, r->samp_vol,
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

int r3d_frame(r3d_renderer *r, const r3d_frame_params *p, r3d_frame_stats *st) {
  if (st) st->gpu_ns = 0;
  uint32_t slot = r->slot;

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
      uint64_t ts[2] = {0, 0};
      if (vkGetQueryPoolResults(r->vk.dev, r->query, slot * 2, 2, sizeof ts, ts, sizeof ts[0],
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        r->slot_gpu_ns[slot] = (uint64_t)((double)(ts[1] - ts[0]) * r->vk.caps.ts_period_ns);
    }
  }
  if (st) st->gpu_ns = r->slot_gpu_ns[slot];

  uint32_t img = 0;
  VkResult ar = vkAcquireNextImageKHR(r->vk.dev, r->swap.swapchain, UINT64_MAX,
                                      r->acquire[slot], VK_NULL_HANDLE, &img);
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    int rc = r3d_resize(r);
    return rc == 0 ? 1 : rc; /* 1 = frame skipped, retry next loop */
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) return -1;

  VkCommandBuffer cmd = r->cmd[slot];
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cmd, &bi);

  if (r->query) vkCmdResetQueryPool(cmd, r->query, slot * 2, 2);

  /* offscreen: whatever -> GENERAL for compute write (contents fully overwritten) */
  r3d_vk_image_barrier(cmd, r->offscreen.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       0, 1);

  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, r->query, slot * 2);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->raycast);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe_layout, 0, 1, &r->dset, 0,
                          NULL);
  vkCmdPushConstants(cmd, r->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(r3d_frame_params), p);
  vkCmdDispatch(cmd, (r->swap.extent.width + r->wg_x - 1) / r->wg_x,
                (r->swap.extent.height + r->wg_y - 1) / r->wg_y, 1);
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, r->query, slot * 2 + 1);
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
      .srcOffsets = {{0, 0, 0},
                     {(int32_t)r->offscreen.extent.width, (int32_t)r->offscreen.extent.height, 1}},
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
      .filter = VK_FILTER_NEAREST,
  };
  vkCmdBlitImage2(cmd, &blit);

  r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, 0,
                       1);
  vkEndCommandBuffer(cmd);

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
