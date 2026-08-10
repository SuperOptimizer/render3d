/* Segment x plane intersection tracing (vc3d-style overlays).
 *
 * Marching squares over a tifxyz grid on the scalar field
 *   f(i,j) = coord_axis(i,j) [+ zoff * normal_axis(i,j)] - slice
 * Each crossing segment is emitted twice-parameterized: in WORLD coordinates
 * of the two in-plane axes (for drawing the segment curve on an axis-aligned
 * plane view) and in GRID coordinates (for drawing the plane's trace line on
 * the flattened segment view). Cells touching any invalid grid point are
 * skipped. */
#ifndef R3D_SEGTRACE_H
#define R3D_SEGTRACE_H

#include <stdint.h>

#include "core/tifxyz.h"

typedef void (*r3d_segtrace_emit)(void *ud, float wu0, float wv0, float wu1, float wv1,
                                  float gi0, float gj0, float gi1, float gj1);

/* Per-row coordinate bounds over valid grid points, one min/max pair per axis
 * per row. Built once per segment, it lets r3d_segtrace skip whole rows whose
 * coordinate range cannot straddle the slice (with a |zoff| margin, since
 * normals are unit length) — a full-grid march on a 1820x2530 segment
 * otherwise costs tens of ms per slice change. */
#define R3D_SEGROWS_TILE 16u

typedef struct r3d_segrows {
  float *mn, *mx;   /* [3][h]: bounds of axis a in row j at mn[a*h + j] */
  float *tmn, *tmx; /* [3][th*tw]: per 16x16-cell tile (whole-row bounds are
                       loose for the axis the rows run along; tiles capture
                       2D locality on oscillating surfaces) */
  uint32_t h, w, tw, th;
} r3d_segrows;

int r3d_segrows_build(const r3d_tifxyz *s, r3d_segrows *out);
void r3d_segrows_free(r3d_segrows *r);

/* axis_n = world axis of the plane normal (0=x 1=y 2=z); axis_u/axis_v = the
 * plane view's screen axes. normals may be NULL (zoff ignored). rows may be
 * NULL (no row skipping). Returns the number of segments emitted. */
uint32_t r3d_segtrace(const r3d_tifxyz *s, const r3d_segrows *rows, const float *normals_rgba,
                      float zoff, int axis_n, int axis_u, int axis_v, double slice,
                      r3d_segtrace_emit emit, void *ud);

#endif /* R3D_SEGTRACE_H */
