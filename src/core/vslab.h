/* Virtual-slab layout math (pure, header-only — unit-tested in test_quick).
 *
 * A W x H x D window positioned ANYWHERE in a sharded export,
 * rendered from world-anchored toroidal tile grids: each level's plane is cut
 * into fixed world-space cells of R3D_VS_PAY * scale voxels, and cell (cx,cy)
 * always lives in physical tile (cx mod gx, cy mod gy). Scrolling any axis
 * therefore only decodes ENTERING cells/strips — resident texels never move.
 *
 * Levels: base (scale 1) + overview pyramid at 4/8/16/32 (full-res z like the
 * slab pyramid). Payload 2016 = 63*32, so every pyramid texel's s x s source
 * box lies entirely inside ONE base cell: pyramid content is a pure by-product
 * of base-cell fills and inherits base validity (1-texel aprons excepted).
 * Z is the slab ring: world slice s lives in layer s mod wz. The visible D
 * slices require D+2 filtering layers; an optional symmetric z margin makes
 * nearby scroll positions GPU-resident without changing visible thickness. */
#ifndef R3D_VSLAB_H
#define R3D_VSLAB_H

#include <stdint.h>

#define R3D_VS_PAY 2016u /* MAX payload voxels per tile axis (divisible by 32) */
#define R3D_VS_LEVELS 5u

static const uint32_t r3d_vs_scale[R3D_VS_LEVELS] = {1, 4, 8, 16, 32};

typedef struct r3d_vs_level {
  uint32_t s;      /* voxel spacing */
  uint32_t gx, gy; /* physical tiles per axis (>= resident cells); 0 = level absent */
  uint32_t base;   /* first index in the shared tile pool */
} r3d_vs_level;

typedef struct r3d_vslab {
  uint64_t nx, ny, nz; /* full volume dims */
  uint32_t W, H, D;    /* window dims (world voxels) */
  uint32_t px;         /* tile payload per xy axis (runtime, multiple of 32) */
  uint32_t wz;         /* z ring depth = D + 2 + 2*z_margin */
  uint32_t z_margin;   /* resident slices before/after the visible window */
  int64_t x0, y0, z0;  /* window origin (world voxels) */
  r3d_vs_level lv[R3D_VS_LEVELS];
  uint32_t ntiles;     /* total physical tiles across levels */
} r3d_vslab;

/* Cell world size at level l. */
static inline int64_t r3d_vs_cell(const r3d_vslab *v, uint32_t l) {
  return (int64_t)v->px * v->lv[l].s;
}

/* Window shapes are arbitrary as long as they fit in RAM: the tile payload
 * adapts to the extents (~8 cells across the larger xy extent) so overhead
 * stays proportional, and the z ring is a single image depth (aprons + ring
 * slack included), bounded by maxImageDimension3D = 2048 -> D <= 2044. An
 * axis that spans the whole volume cannot move and drops its straddle +
 * prefetch margin entirely. Visible D plus filtering and z margins must fit
 * that same 2046-layer ring limit. */
static inline int r3d_vslab_init_margin(r3d_vslab *v, uint64_t nx, uint64_t ny, uint64_t nz,
                                        uint32_t W, uint32_t H, uint32_t D,
                                        uint32_t z_margin) {
  uint64_t ring = (uint64_t)D + 2 + 2ull * z_margin;
  if (W < 64 || H < 64 || W > nx || H > ny || D < 2 || D > 2044 || D > nz ||
      ring > 2046 || ring > nz)
    return -1;
  v->nx = nx;
  v->ny = ny;
  v->nz = nz;
  v->W = W;
  v->H = H;
  v->D = D;
  v->wz = (uint32_t)ring;
  v->z_margin = z_margin;
  v->x0 = v->y0 = v->z0 = 0;
  uint32_t m = W > H ? W : H;
  uint32_t px = (m / 8 + 31) & ~31u;
  if (px < 64) px = 64;
  if (px > R3D_VS_PAY) px = R3D_VS_PAY;
  /* per-image cap: (px+2)^2 * wz must stay well under the 4 GB allocation
   * limit (3 GB margin) — deep rings shrink the xy payload */
  while (px > 64 && (uint64_t)(px + 2) * (px + 2) * v->wz > (3ull << 30)) px -= 32;
  v->px = px;
  /* +1 straddle (a window of W voxels can cover ceil(W/cs)+1 cells) and, at
   * the base level, +1 more for the prefetch ring — per axis, only when the
   * axis can move at all */
  uint32_t ex = W < nx ? 2 : 0, ey = H < ny ? 2 : 0;
  uint32_t nt = 0;
  for (uint32_t l = 0; l < R3D_VS_LEVELS; l++) {
    uint32_t s = r3d_vs_scale[l];
    uint64_t cs = (uint64_t)px * s;
    v->lv[l].s = s;
    /* pyramid levels exist only while the window is large enough to need
     * them (below ~1024 texels the base level serves zoomed-out sampling) */
    if (l && m / s < 1024) {
      v->lv[l].gx = v->lv[l].gy = 0;
      v->lv[l].base = nt;
      continue;
    }
    v->lv[l].gx = (uint32_t)((W + cs - 1) / cs) + ex;
    v->lv[l].gy = (uint32_t)((H + cs - 1) / cs) + ey;
    v->lv[l].base = nt;
    nt += v->lv[l].gx * v->lv[l].gy;
  }
  v->ntiles = nt;
  return 0;
}

static inline int r3d_vslab_init(r3d_vslab *v, uint64_t nx, uint64_t ny, uint64_t nz,
                                 uint32_t W, uint32_t H, uint32_t D) {
  return r3d_vslab_init_margin(v, nx, ny, nz, W, H, D, 0);
}

/* GPU-resident z range [a,b) for a visible window at z0. Keep a full ring
 * near volume faces by shifting the margin to the available side. */
static inline void r3d_vs_zrange(const r3d_vslab *v, int64_t z0, int64_t *a, int64_t *b) {
  int64_t za = z0 - (int64_t)v->z_margin;
  if (za < 0) za = 0;
  int64_t zb = za + (int64_t)v->wz;
  if (zb > (int64_t)v->nz) {
    zb = (int64_t)v->nz;
    za = zb - (int64_t)v->wz;
    if (za < 0) za = 0;
  }
  *a = za;
  *b = zb;
}

/* Resident cell range [c0, c1] at level l for the current origin (x axis;
 * y symmetric). Clamped to the volume. */
static inline void r3d_vs_range(const r3d_vslab *v, uint32_t l, int64_t o, uint64_t n,
                                uint32_t span, int64_t *c0, int64_t *c1) {
  int64_t cs = r3d_vs_cell(v, l);
  int64_t a = o / cs, b = (o + (int64_t)span - 1) / cs;
  int64_t last = ((int64_t)n - 1) / cs;
  if (a < 0) a = 0;
  if (b > last) b = last;
  *c0 = a;
  *c1 = b;
}

/* Physical tile slot of cell c on a g-wide ring. */
static inline uint32_t r3d_vs_phys(int64_t c, uint32_t g) { return (uint32_t)(c % g); }

/* Validity key for cell (cx,cy): compared against the shader-visible table.
 * 43008/2016 = 21 cells per axis at base — 10 bits each is generous. */
#define R3D_VS_VALID 0x80000000u
static inline uint32_t r3d_vs_key(int64_t cx, int64_t cy) {
  return ((uint32_t)cx & 1023u) | (((uint32_t)cy & 1023u) << 10) | R3D_VS_VALID;
}

/* Ring layer of world slice zs. */
static inline uint32_t r3d_vs_layer(const r3d_vslab *v, int64_t zs) {
  return (uint32_t)(zs % (int64_t)v->wz);
}

/* Highest legal window origin per axis. */
static inline int64_t r3d_vs_max0(uint64_t n, uint32_t span) {
  int64_t m = (int64_t)n - (int64_t)span;
  return m < 0 ? 0 : m;
}

#endif /* R3D_VSLAB_H */
