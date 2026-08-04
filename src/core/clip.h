/* XY-clipmap layout math (pure, header-only — unit-tested in test_quick).
 *
 * Up to 6 nested octaves centered near the camera focus. Level ℓ has voxel
 * spacing 2^ℓ and one 2048² texture: 2046 payload texels + 1-texel apron on
 * each side (so trilinear stays seamless at level switches within a level's
 * own data; cross-level transitions are hard switches). Coverage of level ℓ
 * is 2046·2^ℓ voxels (~65k at L5 — a whole 43k cross-section).
 *
 * Z is a per-level ring of `wzl` LEVEL slices (one level slice = 2^ℓ world
 * slices), same wrap-sampling scheme as the slab renderer (core/slab.h).
 * Level origins snap to 16·2^ℓ so fills stay chunk-aligned for the shard
 * decoder. Recenter when the focus drifts beyond coverage/4 from center. */
#ifndef R3D_CLIP_H
#define R3D_CLIP_H

#include <stdbool.h>
#include <stdint.h>

#define R3D_CLIP_LEVELS 6u
#define R3D_CLIP_PAYLOAD 2046u
#define R3D_CLIP_TEX 2048u
#define R3D_CLIP_SNAP 16u /* origin snap in level-voxels (chunk-aligned fills) */

typedef struct r3d_clip_level {
  uint32_t s;      /* voxel spacing = 2^level */
  int64_t ox, oy;  /* world voxel coord of payload texel (0,0); snapped */
  uint32_t wzl;    /* z ring depth in level slices (>= 4) */
  bool valid;      /* filled and safe to sample (renderer-managed) */
} r3d_clip_level;

typedef struct r3d_clip {
  uint64_t nx, ny, nz; /* full volume dims */
  uint32_t nlev;
  uint32_t depth_max;  /* max visible world depth (voxels) */
  r3d_clip_level lv[R3D_CLIP_LEVELS];
} r3d_clip;

static inline int64_t r3d_clip_snap(int64_t v, uint32_t sv) {
  int64_t q = sv;
  int64_t r = v >= 0 ? v / q * q : -((-v + q - 1) / q) * q;
  return r;
}

/* Coverage of level ℓ in world voxels. */
static inline uint64_t r3d_clip_coverage(const r3d_clip *c, uint32_t l) {
  return (uint64_t)R3D_CLIP_PAYLOAD * c->lv[l].s;
}

/* Center level ℓ's payload on (fx,fy) (world voxels), snapped and clamped so
 * the payload stays inside [0,n?]. */
static inline void r3d_clip_recenter(r3d_clip *c, uint32_t l, int64_t fx, int64_t fy) {
  r3d_clip_level *lv = &c->lv[l];
  int64_t cov = (int64_t)r3d_clip_coverage(c, l);
  int64_t snap = (int64_t)(R3D_CLIP_SNAP * lv->s);
  int64_t ox = r3d_clip_snap(fx - cov / 2, (uint32_t)snap);
  int64_t oy = r3d_clip_snap(fy - cov / 2, (uint32_t)snap);
  if (ox < 0) ox = 0;
  if (oy < 0) oy = 0;
  if (ox + cov > (int64_t)c->nx) ox = r3d_clip_snap((int64_t)c->nx - cov, (uint32_t)snap);
  if (oy + cov > (int64_t)c->ny) oy = r3d_clip_snap((int64_t)c->ny - cov, (uint32_t)snap);
  if (ox < 0) ox = 0; /* volume narrower than coverage */
  if (oy < 0) oy = 0;
  lv->ox = ox;
  lv->oy = oy;
  lv->valid = false;
}

static inline void r3d_clip_init(r3d_clip *c, uint64_t nx, uint64_t ny, uint64_t nz,
                                 uint32_t depth_max, int64_t fx, int64_t fy) {
  c->nx = nx;
  c->ny = ny;
  c->nz = nz;
  c->nlev = R3D_CLIP_LEVELS;
  c->depth_max = depth_max;
  for (uint32_t l = 0; l < c->nlev; l++) {
    c->lv[l].s = 1u << l;
    /* ring must hold the window (depth/s) + 2 apron slices, min 4 */
    uint32_t need = depth_max / c->lv[l].s + 3;
    c->lv[l].wzl = need < 4 ? 4 : need;
    c->lv[l].valid = false;
    r3d_clip_recenter(c, l, fx, fy);
  }
}

/* Does level ℓ's payload contain world xy (with `margin` texels to spare)? */
static inline bool r3d_clip_contains(const r3d_clip *c, uint32_t l, int64_t wx, int64_t wy,
                                     uint32_t margin) {
  const r3d_clip_level *lv = &c->lv[l];
  int64_t m = (int64_t)margin * lv->s;
  int64_t cov = (int64_t)r3d_clip_coverage(c, l);
  return wx >= lv->ox + m && wx < lv->ox + cov - m && wy >= lv->oy + m && wy < lv->oy + cov - m;
}

/* Recenter policy: level drifted if focus is beyond coverage/4 from center. */
static inline bool r3d_clip_need_recenter(const r3d_clip *c, uint32_t l, int64_t fx, int64_t fy) {
  const r3d_clip_level *lv = &c->lv[l];
  int64_t cov = (int64_t)r3d_clip_coverage(c, l);
  int64_t cx = lv->ox + cov / 2, cy = lv->oy + cov / 2;
  int64_t dx = fx > cx ? fx - cx : cx - fx;
  int64_t dy = fy > cy ? fy - cy : cy - fy;
  /* no point recentering if the payload cannot move (covers whole volume) */
  if (cov >= (int64_t)c->nx && cov >= (int64_t)c->ny) return false;
  return dx > cov / 4 || dy > cov / 4;
}

/* Ring layer for level slice ls (ls = world_z / s). */
static inline uint32_t r3d_clip_ring_layer(const r3d_clip *c, uint32_t l, uint64_t ls) {
  return (uint32_t)(ls % c->lv[l].wzl);
}

/* Fill rect for one level slice: world rect the texture row data comes from.
 * dst texel (tx,ty) in [0,2048) maps to world (ox + (tx-1)*s, oy + (ty-1)*s),
 * clamped to the volume (apron duplication at edges). Returns the world
 * origin of dst texel (0,0) BEFORE clamping; callers clamp per texel/row. */
static inline void r3d_clip_fill_origin(const r3d_clip *c, uint32_t l, int64_t *wx0,
                                        int64_t *wy0) {
  *wx0 = c->lv[l].ox - (int64_t)c->lv[l].s;
  *wy0 = c->lv[l].oy - (int64_t)c->lv[l].s;
}

#endif /* R3D_CLIP_H */
