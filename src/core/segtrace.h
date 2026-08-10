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

/* axis_n = world axis of the plane normal (0=x 1=y 2=z); axis_u/axis_v = the
 * plane view's screen axes. normals may be NULL (zoff ignored). Returns the
 * number of segments emitted. */
uint32_t r3d_segtrace(const r3d_tifxyz *s, const float *normals_rgba, float zoff, int axis_n,
                      int axis_u, int axis_v, double slice, r3d_segtrace_emit emit, void *ud);

#endif /* R3D_SEGTRACE_H */
