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

#define R3D_SLAB_MAX_GRID 22u   /* up to 22x22 tiles = 45012^2 composite (whole 43k plane) */
#define R3D_SLAB_MAX_TILE 2048u /* default base-tile cap when no device cap is supplied */
/* Overview levels are capped separately and must stay at 2048: the shader's
 * pyramid math hardcodes the 2046 payload (raycast.slang, "ceil(... / 2046.0)").
 * Raising the BASE cap needs no shader change — the base grid reads slab_px/py
 * and slab_grid from push constants — so the two caps are deliberately split. */
#define R3D_SLAB_OV_TILE 2048u
#define R3D_SLAB_TILES 1024u /* descriptor slots: base grid + overview pyramid */
#define R3D_SLAB_OV_MAX 6u

typedef struct r3d_slab_layout {
  uint32_t nx, ny, nz; /* source volume dims (voxels) */
  uint32_t gx, gy;     /* tile grid, 1..2 per axis */
  uint32_t px, py;     /* uniform tile payload (voxels); last tile may cover less */
  uint32_t wz;         /* ring depth (z window, slices) */
  uint32_t ovs;        /* overview downscale (4 to 8184, 8 to 16368, 16 up):
                          the whole-composite overview is ONE <=2048 texture */
} r3d_slab_layout;

/* Tile texture xy dims are payload + 2 apron texels. */
static inline uint32_t r3d_slab_tile_w(const r3d_slab_layout *l) { return l->px + 2; }
static inline uint32_t r3d_slab_tile_h(const r3d_slab_layout *l) { return l->py + 2; }

/* Computes grid + payload, with an explicit base-tile edge cap (texels incl.
 * apron; pass the device maxImageDimension3D bounded by an allocation budget).
 * Returns 0, or -1 if the volume needs more than R3D_SLAB_MAX_GRID tiles per
 * axis or the ring depth is unusable.
 *
 * Total payload is nx*ny*wz however it is tiled, so a larger cap does not cost
 * memory — it duplicates fewer aprons. What it does change is allocation
 * shape: many small tile images become few large ones, which is why callers
 * bound the edge by a per-allocation budget rather than by the raw limit. */
static inline int r3d_slab_layout_init_cap(r3d_slab_layout *l, uint32_t nx, uint32_t ny,
                                           uint32_t nz, uint32_t wz, uint32_t max_tile) {
  if (nx == 0 || ny == 0 || nz == 0 || wz < 4 || wz > nz) return -1;
  if (max_tile < 6) return -1; /* need at least the 2 apron texels + payload */
  l->nx = nx;
  l->ny = ny;
  l->nz = nz;
  l->wz = wz;
  uint32_t maxp = max_tile - 2;
  l->gx = (nx + maxp - 1) / maxp;
  l->gy = (ny + maxp - 1) / maxp;
  if (l->gx > R3D_SLAB_MAX_GRID || l->gy > R3D_SLAB_MAX_GRID) return -1;
  l->px = (nx + l->gx - 1) / l->gx;
  l->py = (ny + l->gy - 1) / l->gy;
  /* overview scale is bounded by the OVERVIEW cap, not the base one: those
   * levels must keep matching the shader's hardcoded 2046 payload */
  uint32_t ovp = R3D_SLAB_OV_TILE - 2;
  l->ovs = 4;
  while ((nx + l->ovs - 1) / l->ovs > ovp || (ny + l->ovs - 1) / l->ovs > ovp) l->ovs <<= 1;
  return 0;
}

/* Default cap (2048) — used by the overview levels and by callers with no
 * device information. */
static inline int r3d_slab_layout_init(r3d_slab_layout *l, uint32_t nx, uint32_t ny, uint32_t nz,
                                       uint32_t wz) {
  return r3d_slab_layout_init_cap(l, nx, ny, nz, wz, R3D_SLAB_MAX_TILE);
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

/* Overview pyramid: prefiltered whole-composite levels at scale s = 4<<lev
 * (coarsest = ovs), each a virtual slab layout over the downsampled plane
 * (same ring z, never downsampled). Kills the unfiltered mid-zoom band:
 * without it, footprints between 1 and ovs voxels sample full-res tiles
 * with zero texture-cache locality. */
static inline uint32_t r3d_slab_ov_levels(const r3d_slab_layout *l) {
  uint32_t n = 1, s = 4;
  while (s < l->ovs) {
    s <<= 1;
    n++;
  }
  return n;
}
static inline void r3d_slab_ov_layout(const r3d_slab_layout *l, uint32_t lev,
                                      r3d_slab_layout *out) {
  uint32_t s = 4u << lev;
  (void)r3d_slab_layout_init(out, (l->nx + s - 1) / s, (l->ny + s - 1) / s, l->nz, l->wz);
}

#endif /* R3D_SLAB_H */
