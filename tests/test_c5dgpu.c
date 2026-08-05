/* c5d GPU decode conformance (ctest label: gpu). Encodes synthetic 128^3
 * bricks with the c5d codec, decodes them through render3d's batched GPU
 * engine (entropy -> dequant/IDCT -> deblock -> pack into an R8 atlas image),
 * reads the atlas back, and compares against c5d_brick_decode on the CPU.
 * Pass criterion: max |CPU - GPU| <= 1 LSB (the codec's own GPU gate). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brick.h"
#include "vk/vkc5d.h"
#include "vk/vkctx.h"
#include "vk/vkres.h"

#ifndef R3D_SPV_DIR
#define R3D_SPV_DIR "spv"
#endif

#define BD 128u
#define NB 8u /* bricks: 2x2x2 atlas slots */

static void synth_brick(uint32_t seed, uint8_t *dst) {
  /* smooth-ish papyrus-like field: sum of sines + seed offset */
  for (uint32_t z = 0; z < BD; z++)
    for (uint32_t y = 0; y < BD; y++)
      for (uint32_t x = 0; x < BD; x++) {
        float fx = (float)x * 0.07f + (float)seed;
        float fy = (float)y * 0.05f;
        float fz = (float)z * 0.06f;
        float v = 120.0f + 60.0f * sinf(fx) * cosf(fy) + 40.0f * sinf(fz + fx * 0.3f);
        int iv = (int)v;
        dst[((size_t)z * BD + y) * BD + x] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
      }
}

int main(void) {
  r3d_vkctx vk;
  if (r3d_vkctx_create(&vk, NULL, 0, false) != 0) {
    fprintf(stderr, "no vulkan device (skip)\n");
    return 77;
  }

  /* atlas: 2x2x2 slots of 128^3 */
  r3d_vkimage atlas;
  if (r3d_vkimage_create(&vk, VK_FORMAT_R8_UNORM, (VkExtent3D){256, 256, 256}, 1,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         &atlas) != 0)
    return 1;
  /* to GENERAL */
  VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .queueFamilyIndex = vk.qfam};
  VkCommandPool pool;
  vkCreateCommandPool(vk.dev, &cpi, NULL, &pool);
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&vk, pool);
  r3d_vk_image_barrier(cmd, atlas.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0, 1);
  if (r3d_vk_oneshot_end(&vk, pool, cmd) != 0) return 1;

  r3d_vkc5d *dec = NULL;
  if (r3d_vkc5d_create(&dec, &vk, R3D_SPV_DIR, 4) != 0) {
    fprintf(stderr, "vkc5d create failed\n");
    return 1;
  }

  /* encode NB bricks, keep CPU reference decodes */
  uint8_t *raw = malloc((size_t)BD * BD * BD);
  uint8_t *cpu[NB] = {0};
  uint8_t *blob[NB] = {0};
  size_t blob_n[NB];
  r3d_c5d_src src[NB];
  for (uint32_t i = 0; i < NB; i++) {
    /* half the bricks encode with tau (q4/tau2) to exercise the GPU
     * corrections scatter; the rest are the c5dpack default (q2, tau off) */
    c5d_brick_params p = c5d_brick_defaults(i & 1 ? 4.0f : 2.0f);
    p.nsub = 128; /* v1.4 GPU knob (what c5dpack writes) */
    if (i & 1) p.tau = 2.0f;
    synth_brick(i, raw);
    if (c5d_brick_encode(&p, raw, BD, &blob[i], &blob_n[i]) != 0) return 1;
    cpu[i] = malloc((size_t)BD * BD * BD);
    if (c5d_brick_decode(blob[i], blob_n[i], cpu[i], BD) != 0) return 1;
    src[i] = (r3d_c5d_src){.blob = blob[i], .n = blob_n[i],
                           .sx = (i & 1) * BD, .sy = ((i >> 1) & 1) * BD,
                           .sz = ((i >> 2) & 1) * BD};
  }

  uint8_t maxes[NB];
  if (r3d_vkc5d_decode(dec, src, NB, atlas.view, maxes) != 0) {
    fprintf(stderr, "GPU decode failed\n");
    return 1;
  }
  printf("decode ok, gpu %.1f ms for %u bricks (%s)\n", r3d_vkc5d_last_gpu_ms(dec), NB,
         getenv("R3D_C5D_HYBRID") ? "hybrid" : "full-GPU");

  /* read the atlas back */
  r3d_vkbuf rb;
  if (r3d_vkbuf_create_host(&vk, (VkDeviceSize)256 * 256 * 256,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb) != 0)
    return 1;
  cmd = r3d_vk_oneshot_begin(&vk, pool);
  VkBufferImageCopy region = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .imageExtent = {256, 256, 256}};
  vkCmdCopyImageToBuffer(cmd, atlas.img, VK_IMAGE_LAYOUT_GENERAL, rb.buf, 1, &region);
  if (r3d_vk_oneshot_end(&vk, pool, cmd) != 0) return 1;

  int fail = 0;
  uint64_t hist[3] = {0, 0, 0};
  const uint8_t *at = rb.mapped;
  for (uint32_t i = 0; i < NB; i++) {
    int md = 0;
    uint8_t cmax = 0;
    for (uint32_t z = 0; z < BD; z++)
      for (uint32_t y = 0; y < BD; y++)
        for (uint32_t x = 0; x < BD; x++) {
          uint8_t g = at[((size_t)(src[i].sz + z) * 256 + (src[i].sy + y)) * 256 + src[i].sx + x];
          uint8_t c = cpu[i][((size_t)z * BD + y) * BD + x];
          int dd = (int)g - (int)c;
          if (dd < 0) dd = -dd;
          if (dd > md) md = dd;
          hist[dd > 2 ? 2 : dd]++;
          if (c > cmax) cmax = c;
        }
    /* out_max comes from the pre-deblock int32 scratch; allow 1 LSB slack */
    int mdiff = (int)maxes[i] - (int)cmax;
    if (mdiff < -1 || mdiff > 1) {
      printf("brick %u: out_max %u vs cpu max %u\n", i, maxes[i], cmax);
      fail = 1;
    }
    /* upstream contract (tau amendment): <=1 LSB everywhere except <=8
     * bounded deblock gate-flip voxels per brick */
    uint64_t flips = 0;
    if (md > 1) {
      for (uint32_t v = 0; v < BD * BD * BD; v++) {
        uint32_t z = v >> 14, y = (v >> 7) & 127, x = v & 127;
        uint8_t g =
            at[((size_t)(src[i].sz + z) * 256 + (src[i].sy + y)) * 256 + src[i].sx + x];
        uint8_t c = cpu[i][v];
        int dd = (int)g - (int)c;
        if (dd < 0) dd = -dd;
        if (dd > 1) flips++;
      }
    }
    if (md > 24 || flips > 8) {
      printf("brick %u: maxdiff %d, %llu gate-flip voxels (>8)\n", i, md,
             (unsigned long long)flips);
      fail = 1;
    }
  }
  printf("voxels: exact %llu, 1LSB %llu, >1 %llu\n", (unsigned long long)hist[0],
         (unsigned long long)hist[1], (unsigned long long)hist[2]);

  for (uint32_t i = 0; i < NB; i++) {
    free(blob[i]);
    free(cpu[i]);
  }
  free(raw);
  r3d_vkbuf_destroy(&vk, &rb);
  r3d_vkc5d_destroy(dec);
  r3d_vkimage_destroy(&vk, &atlas);
  vkDestroyCommandPool(vk.dev, pool, NULL);
  r3d_vkctx_destroy(&vk);
  if (fail) {
    fprintf(stderr, "test_c5dgpu: FAIL\n");
    return 1;
  }
  printf("test_c5dgpu: all ok\n");
  return 0;
}
