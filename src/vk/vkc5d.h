/* c5d GPU decode engine: compressed brick blobs -> (entropy -> dequant/IDCT ->
 * deblock) on the GPU -> pack (R8) into the brick-atlas 3D image. Batched:
 * up to `max_batch` bricks per submit through shared mega-buffers, mirroring
 * the batched design in ~/compressor/src/gpu (kernels are compiled from that
 * working tree; push structs here must match them — test_c5dgpu is the gate).
 * CPU side per brick is only he_gpu_setup (header parse + rANS tables), or
 * he_decode when R3D_C5D_HYBRID=1 (entropy on CPU, rest on GPU). */
#ifndef R3D_VKC5D_H
#define R3D_VKC5D_H

#include <stdint.h>

#include "vk/vkctx.h"

typedef struct r3d_vkc5d r3d_vkc5d;

typedef struct r3d_c5d_src {
  const uint8_t *blob; /* compressed brick (from a .c5s shard) */
  size_t n;
  uint32_t sx, sy, sz; /* atlas texel origin of the destination slot */
} r3d_c5d_src;

int r3d_vkc5d_create(r3d_vkc5d **out, r3d_vkctx *c, const char *spv_dir, uint32_t max_batch);
void r3d_vkc5d_destroy(r3d_vkc5d *d);

/* Decode `n` bricks (dim must be 128) into `atlas` (R8_UNORM 3D storage image
 * in GENERAL layout). Synchronous: returns when the atlas is written and
 * visible to compute sampling. Returns 0, or -1 (a brick may be rejected:
 * lossless/tau flags, corrupt header). out_max[i] (optional) receives each
 * brick's decoded max voxel value (for empty-skip page-table entries). */
int r3d_vkc5d_decode(r3d_vkc5d *d, const r3d_c5d_src *src, uint32_t n, VkImageView atlas_view,
                     uint8_t *out_max);

double r3d_vkc5d_last_gpu_ms(const r3d_vkc5d *d);

#endif /* R3D_VKC5D_H */
