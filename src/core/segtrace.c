#include "core/segtrace.h"

#include <math.h>
#include <stddef.h>

/* linear crossing parameter on an edge whose field values are fa, fb */
static float xt(float fa, float fb) {
  float d = fa - fb;
  return fabsf(d) < 1e-12f ? 0.5f : fa / d;
}

uint32_t r3d_segtrace(const r3d_tifxyz *s, const float *normals_rgba, float zoff, int axis_n,
                      int axis_u, int axis_v, double slice, r3d_segtrace_emit emit, void *ud) {
  /* per marching-squares case: pairs of edge indices forming segments
   * (edge 0 = bottom (j), 1 = right, 2 = top (j+1), 3 = left); -1 ends */
  static const int8_t cases[16][4] = {
      {-1, -1, -1, -1}, {3, 0, -1, -1},  {0, 1, -1, -1},  {3, 1, -1, -1},
      {1, 2, -1, -1},   {3, 0, 1, 2},    {0, 2, -1, -1},  {3, 2, -1, -1},
      {2, 3, -1, -1},   {2, 0, -1, -1},  {0, 1, 2, 3},    {2, 1, -1, -1},
      {1, 3, -1, -1},   {1, 0, -1, -1},  {0, 3, -1, -1},  {-1, -1, -1, -1}};
  uint32_t emitted = 0;
  for (uint32_t j = 0; j + 1 < s->h; j++)
    for (uint32_t i = 0; i + 1 < s->w; i++) {
      const float *c[4] = {r3d_tifxyz_at(s, i, j), r3d_tifxyz_at(s, i + 1, j),
                           r3d_tifxyz_at(s, i + 1, j + 1), r3d_tifxyz_at(s, i, j + 1)};
      if (!r3d_tifxyz_valid(c[0]) || !r3d_tifxyz_valid(c[1]) || !r3d_tifxyz_valid(c[2]) ||
          !r3d_tifxyz_valid(c[3]))
        continue;
      /* corner order: 0=(i,j) 1=(i+1,j) 2=(i+1,j+1) 3=(i,j+1) */
      float f[4], wu[4], wv[4];
      const float gi[4] = {(float)i, (float)i + 1, (float)i + 1, (float)i};
      const float gj[4] = {(float)j, (float)j, (float)j + 1, (float)j + 1};
      bool skip = false;
      for (int k = 0; k < 4; k++) {
        float p[3] = {c[k][0], c[k][1], c[k][2]};
        if (normals_rgba && zoff != 0.0f) {
          uint64_t gk = ((uint64_t)(gj[k]) * s->w + (uint64_t)(gi[k])) * 4;
          p[0] += normals_rgba[gk + 0] * zoff;
          p[1] += normals_rgba[gk + 1] * zoff;
          p[2] += normals_rgba[gk + 2] * zoff;
        }
        f[k] = (float)((double)p[axis_n] - slice);
        wu[k] = p[axis_u];
        wv[k] = p[axis_v];
        if (!isfinite(f[k])) skip = true;
      }
      if (skip) continue;
      uint32_t ci = (uint32_t)((f[0] > 0.0f) | ((f[1] > 0.0f) << 1) |
                               ((f[2] > 0.0f) << 2) | ((f[3] > 0.0f) << 3));
      const int8_t *cs = cases[ci];
      if (cs[0] < 0) continue;
      /* edge e connects corners e and (e+1)&3 */
      float ewu[4], ewv[4], egi[4], egj[4];
      for (int e = 0; e < 4; e++) {
        int a = e, b = (e + 1) & 3;
        float t = xt(f[a], f[b]);
        ewu[e] = wu[a] + (wu[b] - wu[a]) * t;
        ewv[e] = wv[a] + (wv[b] - wv[a]) * t;
        egi[e] = gi[a] + (gi[b] - gi[a]) * t;
        egj[e] = gj[a] + (gj[b] - gj[a]) * t;
      }
      for (int k = 0; k < 4 && cs[k] >= 0; k += 2) {
        int a = cs[k], b = cs[k + 1];
        emit(ud, ewu[a], ewv[a], ewu[b], ewv[b], egi[a], egj[a], egi[b], egj[b]);
        emitted++;
      }
    }
  return emitted;
}
