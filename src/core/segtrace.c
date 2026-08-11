#include "core/segtrace.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* linear crossing parameter on an edge whose field values are fa, fb */
static float xt(float fa, float fb) {
  float d = fa - fb;
  return fabsf(d) < 1e-12f ? 0.5f : fa / d;
}

int r3d_segrows_build(const r3d_tifxyz *s, r3d_segrows *out) {
  const uint32_t T = R3D_SEGROWS_TILE;
  memset(out, 0, sizeof *out);
  out->h = s->h;
  out->w = s->w;
  out->tw = (s->w + T - 1) / T;
  out->th = (s->h + T - 1) / T;
  size_t nt = (size_t)out->tw * out->th;
  out->mn = malloc((size_t)s->h * 3 * sizeof *out->mn);
  out->mx = malloc((size_t)s->h * 3 * sizeof *out->mx);
  out->tmn = malloc(nt * 3 * sizeof *out->tmn);
  out->tmx = malloc(nt * 3 * sizeof *out->tmx);
  if (!out->mn || !out->mx || !out->tmn || !out->tmx) {
    r3d_segrows_free(out);
    return -1;
  }
  for (size_t k = 0; k < nt * 3; k++) {
    out->tmn[k] = INFINITY;
    out->tmx[k] = -INFINITY;
  }
  for (uint32_t j = 0; j < s->h; j++) {
    float mn[3] = {INFINITY, INFINITY, INFINITY};
    float mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    /* a point on a tile's low edge also bounds cells of the previous tile */
    uint32_t tj0 = (j ? j - 1 : 0) / T, tj1 = j / T;
    for (uint32_t i = 0; i < s->w; i++) {
      const float *p = r3d_tifxyz_at(s, i, j);
      if (!r3d_tifxyz_valid(p)) continue;
      uint32_t ti0 = (i ? i - 1 : 0) / T, ti1 = i / T;
      for (int a = 0; a < 3; a++) {
        if (p[a] < mn[a]) mn[a] = p[a];
        if (p[a] > mx[a]) mx[a] = p[a];
        for (uint32_t tj = tj0; tj <= tj1; tj++)
          for (uint32_t ti = ti0; ti <= ti1; ti++) {
            size_t t = (size_t)a * out->tw * out->th + (size_t)tj * out->tw + ti;
            if (p[a] < out->tmn[t]) out->tmn[t] = p[a];
            if (p[a] > out->tmx[t]) out->tmx[t] = p[a];
          }
      }
    }
    for (int a = 0; a < 3; a++) {
      out->mn[(size_t)a * s->h + j] = mn[a];
      out->mx[(size_t)a * s->h + j] = mx[a];
    }
  }
  return 0;
}

void r3d_segrows_free(r3d_segrows *r) {
  free(r->mn);
  free(r->mx);
  free(r->tmn);
  free(r->tmx);
  memset(r, 0, sizeof *r);
}

/* bounds of dot(p, n) over a per-axis min/max box stored as [3][stride]
 * arrays at index idx; zero components are skipped so empty boxes (+-inf
 * bounds) stay inf/-inf instead of producing 0*inf NaNs */
static inline void dot_range(const float *mn, const float *mx, size_t stride, size_t idx,
                             const double n[3], double *lo, double *hi) {
  double l = 0.0, h = 0.0;
  for (int a = 0; a < 3; a++) {
    if (n[a] == 0.0) continue;
    double vmn = (double)mn[(size_t)a * stride + idx];
    double vmx = (double)mx[(size_t)a * stride + idx];
    if (n[a] > 0.0) {
      l += n[a] * vmn;
      h += n[a] * vmx;
    } else {
      l += n[a] * vmx;
      h += n[a] * vmn;
    }
  }
  *lo = l;
  *hi = h;
}

uint32_t r3d_segtrace(const r3d_tifxyz *s, const r3d_segrows *rows, const float *normals_rgba,
                      float zoff, int axis_n, int axis_u, int axis_v, double slice,
                      r3d_segtrace_emit emit, void *ud) {
  double org[3] = {0.0, 0.0, 0.0};
  double bu[3] = {0.0, 0.0, 0.0}, bv[3] = {0.0, 0.0, 0.0}, bn[3] = {0.0, 0.0, 0.0};
  bu[axis_u] = 1.0;
  bv[axis_v] = 1.0;
  bn[axis_n] = 1.0;
  return r3d_segtrace_basis(s, rows, normals_rgba, zoff, org, bu, bv, bn, slice, emit, ud);
}

uint32_t r3d_segtrace_basis(const r3d_tifxyz *s, const r3d_segrows *rows,
                            const float *normals_rgba, float zoff, const double org[3],
                            const double bu[3], const double bv[3], const double bn[3],
                            double slice, r3d_segtrace_emit emit, void *ud) {
  /* per marching-squares case: pairs of edge indices forming segments
   * (edge 0 = bottom (j), 1 = right, 2 = top (j+1), 3 = left); -1 ends */
  static const int8_t cases[16][4] = {
      {-1, -1, -1, -1}, {3, 0, -1, -1},  {0, 1, -1, -1},  {3, 1, -1, -1},
      {1, 2, -1, -1},   {3, 0, 1, 2},    {0, 2, -1, -1},  {3, 2, -1, -1},
      {2, 3, -1, -1},   {2, 0, -1, -1},  {0, 1, 2, 3},    {2, 1, -1, -1},
      {1, 3, -1, -1},   {1, 0, -1, -1},  {0, 3, -1, -1},  {-1, -1, -1, -1}};
  uint32_t emitted = 0;
  const uint32_t T = R3D_SEGROWS_TILE;
  bool use_rows = rows && rows->h == s->h && rows->w == s->w;
  double margin = normals_rgba ? fabs((double)zoff) + 1e-3 : 1e-3;
  /* compare dot(p, bn) against the plane offset in absolute terms */
  double sl = slice + bn[0] * org[0] + bn[1] * org[1] + bn[2] * org[2];
  double ou = bu[0] * org[0] + bu[1] * org[1] + bu[2] * org[2];
  double ov = bv[0] * org[0] + bv[1] * org[1] + bv[2] * org[2];
  for (uint32_t j = 0; j + 1 < s->h; j++) {
    if (use_rows) { /* row band [j, j+1] can't straddle the slice: skip it */
      double l0, h0, l1, h1;
      dot_range(rows->mn, rows->mx, s->h, j, bn, &l0, &h0);
      dot_range(rows->mn, rows->mx, s->h, j + 1, bn, &l1, &h1);
      if (sl < (l0 < l1 ? l0 : l1) - margin || sl > (h0 > h1 ? h0 : h1) + margin) continue;
    }
    size_t trow = use_rows ? (size_t)(j / T) * rows->tw : 0;
    for (uint32_t i = 0; i + 1 < s->w; i++) {
      if (use_rows) { /* tile can't straddle the slice: hop to the next tile */
        uint32_t ti = i / T;
        double tl, th;
        dot_range(rows->tmn, rows->tmx, (size_t)rows->tw * rows->th, trow + ti, bn, &tl,
                  &th);
        if (sl < tl - margin || sl > th + margin) {
          i = (ti + 1) * T - 1; /* loop increment lands on the next tile */
          continue;
        }
      }
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
        double px = (double)p[0], py = (double)p[1], pz = (double)p[2];
        f[k] = (float)(px * bn[0] + py * bn[1] + pz * bn[2] - sl);
        wu[k] = (float)(px * bu[0] + py * bu[1] + pz * bu[2] - ou);
        wv[k] = (float)(px * bv[0] + py * bv[1] + pz * bv[2] - ov);
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
  }
  return emitted;
}
