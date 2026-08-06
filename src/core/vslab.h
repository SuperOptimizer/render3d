/* Virtual-slab layout math (pure, header-only — unit-tested in test_quick).
 *
 * A W x H x D window positioned ANYWHERE in the full export (43008^2 x 68608),
 * rendered from world-anchored toroidal tile grids: each level's plane is cut
 * into fixed world-space cells of R3D_VS_PAY * scale voxels, and cell (cx,cy)
 * always lives in physical tile (cx mod gx, cy mod gy). Scrolling any axis
 * therefore only decodes ENTERING cells/strips — resident texels never move.
 *
 * Levels: base (scale 1) + overview pyramid at 4/8/16/32 (full-res z like the
 * slab pyramid). Payload 2016 = 63*32, so every pyramid texel's s x s source
 * box lies entirely inside ONE base cell: pyramid content is a pure by-product
 * of base-cell fills and inherits base validity (1-texel aprons excepted).
 * Z is the slab ring: world slice s lives in layer s mod wz, wz = D + 2. */
#ifndef R3D_VSLAB_H
#define R3D_VSLAB_H

#include <stdint.h>

#define R3D_VS_PAY 2016u /* payload voxels per tile axis (divisible by 32) */
#define R3D_VS_TEX 2018u /* payload + 1-voxel apron each side */
#define R3D_VS_LEVELS 5u

static const uint32_t r3d_vs_scale[R3D_VS_LEVELS] = {1, 4, 8, 16, 32};

typedef struct r3d_vs_level {
  uint32_t s;      /* voxel spacing */
  uint32_t gx, gy; /* physical tiles per axis (>= resident cells) */
  uint32_t base;   /* first index in the shared tile pool */
} r3d_vs_level;

typedef struct r3d_vslab {
  uint64_t nx, ny, nz; /* full volume dims */
  uint32_t W, H, D;    /* window dims (world voxels) */
  uint32_t wz;         /* z ring depth = D + 2 */
  int64_t x0, y0, z0;  /* window origin (world voxels) */
  r3d_vs_level lv[R3D_VS_LEVELS];
  uint32_t ntiles;     /* total physical tiles across levels */
} r3d_vslab;

/* Cell world size at level l. */
static inline int64_t r3d_vs_cell(const r3d_vslab *v, uint32_t l) {
  return (int64_t)R3D_VS_PAY * v->lv[l].s;
}

static inline int r3d_vslab_init(r3d_vslab *v, uint64_t nx, uint64_t ny, uint64_t nz,
                                 uint32_t W, uint32_t H, uint32_t D) {
  if (W < R3D_VS_PAY || H < R3D_VS_PAY || D < 2 || D > 62) return -1;
  v->nx = nx;
  v->ny = ny;
  v->nz = nz;
  v->W = W;
  v->H = H;
  v->D = D;
  v->wz = D + 2;
  v->x0 = v->y0 = v->z0 = 0;
  uint32_t nt = 0;
  for (uint32_t l = 0; l < R3D_VS_LEVELS; l++) {
    uint32_t s = r3d_vs_scale[l];
    uint64_t cs = (uint64_t)R3D_VS_PAY * s;
    /* +1: a window of W voxels can straddle ceil(W/cs)+1 cells; the base
     * level adds one more so a prefetch ring one cell beyond the window can
     * be resident without evicting a wanted cell (pyramid levels never
     * prefetch — their content arrives with base fills) */
    v->lv[l].s = s;
    v->lv[l].gx = (uint32_t)((W + cs - 1) / cs) + 1 + (l == 0);
    v->lv[l].gy = (uint32_t)((H + cs - 1) / cs) + 1 + (l == 0);
    v->lv[l].base = nt;
    nt += v->lv[l].gx * v->lv[l].gy;
  }
  v->ntiles = nt;
  return 0;
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
