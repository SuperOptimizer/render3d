/* Tiled-slab layout math (pure, header-only — unit-tested in test_quick).
 *
 * A wide, thin window of the source volume is rendered from a grid of up to
 * 2x2 tile textures. Each tile stores `payload+2` texels per XY axis: its
 * payload plus a 1-voxel apron duplicated from the neighboring tile (or edge-
 * clamped at volume borders) so hardware trilinear filtering never shows a
 * seam. Z is a ring buffer of `wz` layers: world slice s lives in layer
 * s % wz, and a REPEAT sampler on W makes interpolation wrap correctly; only
 * the pair (newest, oldest) is world-discontiguous, excluded by marching
 * z in (z0+0.5, z0+wz-0.5) voxel units — wz-1 usable slices. */
#ifndef R3D_SLAB_H
#define R3D_SLAB_H

#include <stdint.h>

#define R3D_SLAB_MAX_GRID 8u    /* up to 8x8 tiles = 16368^2 composite */
#define R3D_SLAB_MAX_TILE 2048u /* maxImageDimension3D on target hardware */
#define R3D_SLAB_TILES (R3D_SLAB_MAX_GRID * R3D_SLAB_MAX_GRID)

typedef struct r3d_slab_layout {
  uint32_t nx, ny, nz; /* source volume dims (voxels) */
  uint32_t gx, gy;     /* tile grid, 1..2 per axis */
  uint32_t px, py;     /* uniform tile payload (voxels); last tile may cover less */
  uint32_t wz;         /* ring depth (z window, slices) */
  uint32_t ovs;        /* overview downscale (4 while <=8184, 8 to 16368...):
                          the whole-composite overview is ONE <=2048 texture */
} r3d_slab_layout;

/* Tile texture xy dims are payload + 2 apron texels. */
static inline uint32_t r3d_slab_tile_w(const r3d_slab_layout *l) { return l->px + 2; }
static inline uint32_t r3d_slab_tile_h(const r3d_slab_layout *l) { return l->py + 2; }

/* Computes grid + payload. Returns 0, or -1 if the volume needs >2 tiles per
 * axis or the ring depth is unusable. */
static inline int r3d_slab_layout_init(r3d_slab_layout *l, uint32_t nx, uint32_t ny, uint32_t nz,
                                       uint32_t wz) {
  if (nx == 0 || ny == 0 || nz == 0 || wz < 4 || wz > nz) return -1;
  l->nx = nx;
  l->ny = ny;
  l->nz = nz;
  l->wz = wz;
  uint32_t maxp = R3D_SLAB_MAX_TILE - 2;
  l->gx = (nx + maxp - 1) / maxp;
  l->gy = (ny + maxp - 1) / maxp;
  if (l->gx > R3D_SLAB_MAX_GRID || l->gy > R3D_SLAB_MAX_GRID) return -1;
  l->px = (nx + l->gx - 1) / l->gx;
  l->py = (ny + l->gy - 1) / l->gy;
  l->ovs = 4;
  while ((nx + l->ovs - 1) / l->ovs > maxp || (ny + l->ovs - 1) / l->ovs > maxp) l->ovs <<= 1;
  return 0;
}

/* Ring layer holding world slice s (s need not be < nz here). */
static inline uint32_t r3d_slab_ring_layer(const r3d_slab_layout *l, uint32_t s) {
  return s % l->wz;
}

/* Source column (world voxel x) for destination texel column `dst` of tile
 * `ti` — apron columns duplicate the neighbor, clamped at volume edges. */
static inline uint32_t r3d_slab_src_col(const r3d_slab_layout *l, uint32_t ti, uint32_t dst) {
  int64_t w = (int64_t)ti * l->px - 1 + dst;
  if (w < 0) w = 0;
  if (w >= (int64_t)l->nx) w = (int64_t)l->nx - 1;
  return (uint32_t)w;
}
static inline uint32_t r3d_slab_src_row(const r3d_slab_layout *l, uint32_t tj, uint32_t dst) {
  int64_t w = (int64_t)tj * l->py - 1 + dst;
  if (w < 0) w = 0;
  if (w >= (int64_t)l->ny) w = (int64_t)l->ny - 1;
  return (uint32_t)w;
}

/* Scroll from window start z0_old to z0_new: which world slices need
 * uploading? Sets [s0, s1) and full=1 when the whole window must be redone
 * (jump >= wz or first upload). Ranges are inclusive of z0_new's window. */
static inline void r3d_slab_scroll_range(const r3d_slab_layout *l, int64_t z0_old,
                                         int64_t z0_new, uint32_t *s0, uint32_t *s1, int *full) {
  int64_t d = z0_new - z0_old;
  if (z0_old < 0 || d >= (int64_t)l->wz || d <= -(int64_t)l->wz) {
    *s0 = (uint32_t)z0_new;
    *s1 = (uint32_t)z0_new + l->wz;
    *full = 1;
    return;
  }
  *full = 0;
  if (d > 0) { /* window moved deeper: new slices at the far end */
    *s0 = (uint32_t)(z0_old + (int64_t)l->wz);
    *s1 = (uint32_t)(z0_new + (int64_t)l->wz);
  } else if (d < 0) { /* window moved back: new slices at the near end */
    *s0 = (uint32_t)z0_new;
    *s1 = (uint32_t)z0_old;
  } else {
    *s0 = *s1 = 0;
  }
}

/* Highest legal window start for a source of nz slices. */
static inline uint32_t r3d_slab_z0_max(const r3d_slab_layout *l) { return l->nz - l->wz; }

#endif /* R3D_SLAB_H */
