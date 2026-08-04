/* Clipmap backend: 6 octave ring textures filled asynchronously from the
 * shard band (L0/L1 live dct3d decode) and pyramid files (L2-L5 mmap).
 * A single worker thread produces texel slices into staging; the main thread
 * pumps completed slices into the textures with host_image_copy (gated on the
 * frame timeline) and derives the per-level valid mask. Internal header. */
#ifndef R3D_VKCLIP_H
#define R3D_VKCLIP_H

#include <stdint.h>

#include "render/render_types.h"
#include "vk/vkctx.h"

typedef struct r3d_vkclip r3d_vkclip;

/* Creates textures + worker. band_z = shard row (world z band start = band_z*1024).
 * fx/fy = initial focus (world voxels); z0 = initial window start (world). */
int r3d_vkclip_create(r3d_vkclip **out, r3d_vkctx *c,
                      PFN_vkTransitionImageLayoutEXT fp_transition,
                      PFN_vkCopyMemoryToImageEXT fp_copy_mem, const char *band_dir,
                      const char *pyramid_dir, uint32_t band_z, uint32_t depth_max, int64_t fx,
                      int64_t fy, uint64_t z0);
void r3d_vkclip_destroy(r3d_vkclip *cl);

/* Per frame (main thread): recenters/scroll-requests, then uploads whatever
 * the worker finished (waits `timeline >= tv` once before host copies). */
void r3d_vkclip_update(r3d_vkclip *cl, int64_t fx, int64_t fy, uint64_t z0);
int r3d_vkclip_pump(r3d_vkclip *cl, VkSemaphore timeline, uint64_t tv);

/* Fill clip_valid/clip_orig + slab_nx/ny/z0/wz fields. */
void r3d_vkclip_params(const r3d_vkclip *cl, r3d_frame_params *p);

VkImageView r3d_vkclip_view(const r3d_vkclip *cl, uint32_t level);
uint32_t r3d_vkclip_band_z(const r3d_vkclip *cl); /* band start, world slices */

#endif /* R3D_VKCLIP_H */
