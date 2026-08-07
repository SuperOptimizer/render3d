/* Projected-footprint LOD policy shared by the CPU streamer and tests. */
#ifndef R3D_LOD_H
#define R3D_LOD_H

#include <math.h>
#include <stdint.h>

static inline uint32_t r3d_lod_pick(float distance, float pixel_cone, float base_voxels,
                                    uint32_t nlevels) {
  if (!nlevels) return 0;
  float footprint = fmaxf(distance * pixel_cone * base_voxels, 1.0f);
  int level = (int)floorf(log2f(footprint));
  if (level < 0) level = 0;
  if (level >= (int)nlevels) level = (int)nlevels - 1;
  return (uint32_t)level;
}

/* Sampling fallback order: desired, progressively coarser, then progressively
 * finer. Returns UINT32_MAX after all levels have been visited. */
static inline uint32_t r3d_lod_fallback(uint32_t desired, uint32_t ordinal,
                                        uint32_t nlevels) {
  if (!nlevels || desired >= nlevels || ordinal >= nlevels) return UINT32_MAX;
  uint32_t coarse_count = nlevels - desired;
  if (ordinal < coarse_count) return desired + ordinal;
  return desired - 1u - (ordinal - coarse_count);
}

#endif /* R3D_LOD_H */
