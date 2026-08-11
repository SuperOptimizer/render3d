/* Segment store: a whole scroll's tifxyz surfaces as one directory of
 * c5d-compressed grids (<name>.tfx) plus a binary manifest (segments.r3ds)
 * that doubles as the spatial index.
 *
 * The index is two-level (inspired by vc3d's SurfacePatchIndex, flattened
 * for C): a world AABB per segment, then a u16-quantized AABB per 16x16
 * grid tile (quantized over the segment's own bbox, ~12 B/tile — a whole
 * scroll's corpus indexes in tens of MB and loads with one read). Queries
 * never touch the compressed grids, so "which segments cross this plane /
 * sit near this point" works over corpora far larger than RAM; only
 * survivors get decoded, optionally decimated by a power-of-two stride for
 * overview polylines. */
#ifndef R3D_SEGSTORE_H
#define R3D_SEGSTORE_H

#include <stdbool.h>
#include <stdint.h>

#include "core/tifxyz.h"

#define R3D_SEGSTORE_TILE 16u
#define R3D_SEGSTORE_NAME 64u

typedef struct r3d_segmeta {
  char name[R3D_SEGSTORE_NAME]; /* tifxyz dir basename == <name>.tfx */
  uint32_t w, h;                /* full-resolution grid size */
  uint32_t tw, th;              /* tile grid size */
  float sx, sy;                 /* tifxyz scale (grid cells per voxel) */
  float bbox[2][3];             /* world AABB of valid points */
  uint64_t nvalid;
  uint64_t tile_ofs; /* first tile record in the store's tile array */
} r3d_segmeta;

typedef struct r3d_segtile { /* AABB quantized over the segment bbox */
  uint16_t lo[3], hi[3];     /* empty tile: lo > hi (0xffff / 0) */
} r3d_segtile;

typedef struct r3d_segstore {
  r3d_segmeta *segs;
  uint32_t n;
  uint64_t ntiles;
  r3d_segtile *tiles;
  char dir[512];
} r3d_segstore;

/* Ingest tifxyz dirs into store_dir: encode <basename>.tfx (log2q < 0 =
 * lossless, else 2^-log2q voxel quantization; existing .tfx reused unless
 * force) and write the manifest. Returns the number of segments stored. */
int r3d_segstore_build(const char *store_dir, const char *const *dirs, uint32_t ndirs,
                       int log2q, bool force);

int r3d_segstore_open(r3d_segstore *st, const char *store_dir);
void r3d_segstore_close(r3d_segstore *st);

/* Decode segment i into a normal r3d_tifxyz. stride > 1 keeps every
 * stride-th grid point (scale shrinks to match) for cheap overview tracing. */
int r3d_segstore_load(const r3d_segstore *st, uint32_t i, uint32_t stride, r3d_tifxyz *out);

/* Segments with a tile whose AABB straddles the plane dot(p, bn) == slice
 * (within margin) and touches the world AABB [lo, hi] (either may be NULL).
 * Returns the hit count; out (may be NULL) gets up to cap indices. */
uint32_t r3d_segstore_plane_query(const r3d_segstore *st, const double bn[3], double slice,
                                  double margin, const double lo[3], const double hi[3],
                                  uint32_t *out, uint32_t cap);

/* Segments with a tile AABB within radius of world point p, nearest tile
 * distance ascending. Returns the hit count; out gets up to cap indices. */
uint32_t r3d_segstore_near_query(const r3d_segstore *st, const double p[3], double radius,
                                 uint32_t *out, uint32_t cap);

/* Fraction of a's occupied index tiles whose tol-dilated AABB touches any
 * of b's tiles (coarse 64^3-cell approximation over the shared bbox —
 * meant for duplicate/conflict QC, not exact geometry). */
double r3d_segstore_overlap(const r3d_segstore *st, uint32_t a, uint32_t b, double tol);

#endif /* R3D_SEGSTORE_H */
