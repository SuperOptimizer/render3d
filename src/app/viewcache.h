/* Small application caches. No renderer or ImGui dependency: coverage and
 * slice selection are independently testable. */
#ifndef R3D_VIEWCACHE_H
#define R3D_VIEWCACHE_H
#include "core/segtrace.h"
#include "core/tracer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct r3d_surface_bounds {
  float *normal_min, *normal_max; /* normalized interpolated-normal bounds */
  uint32_t tw, th;
  uint64_t revision;
} r3d_surface_bounds;
static inline void r3d_surface_bounds_free(r3d_surface_bounds *b) {
  free(b->normal_min);
  free(b->normal_max);
  memset(b, 0, sizeof *b);
}
static inline int r3d_surface_bounds_build(r3d_surface_bounds *b,
                                           const r3d_tifxyz *s,
                                           const r3d_segrows *rows,
                                           const float *normals,
                                           uint64_t revision) {
  size_t nt = (size_t)rows->tw * rows->th;
  r3d_surface_bounds n = {.tw = rows->tw, .th = rows->th, .revision = revision};
  n.normal_min = malloc(nt * 3 * sizeof(float));
  n.normal_max = malloc(nt * 3 * sizeof(float));
  if (!n.normal_min || !n.normal_max) {
    r3d_surface_bounds_free(&n);
    return -1;
  }
  for (uint32_t ty = 0; ty < rows->th; ty++)
    for (uint32_t tx = 0; tx < rows->tw; tx++) {
      float lo[3] = {INFINITY, INFINITY, INFINITY},
            hi[3] = {-INFINITY, -INFINITY, -INFINITY};
      uint32_t x0 = tx * R3D_SEGROWS_TILE, y0 = ty * R3D_SEGROWS_TILE;
      uint32_t x1 = x0 + R3D_SEGROWS_TILE, y1 = y0 + R3D_SEGROWS_TILE;
      if (x1 >= s->w)
        x1 = s->w - 1;
      if (y1 >= s->h)
        y1 = s->h - 1;
      for (uint32_t y = y0; y <= y1; y++)
        for (uint32_t x = x0; x <= x1; x++) {
          size_t k = (size_t)y * s->w + x;
          if (!r3d_tifxyz_valid(s->xyz + k * 3))
            continue;
          for (int a = 0; a < 3; a++) {
            float v = normals ? normals[k * 4 + (size_t)a] : 0.0f;
            lo[a] = fminf(lo[a], v);
            hi[a] = fmaxf(hi[a], v);
          }
        }
      float minlen2 = 0.0f, maxlen2 = 0.0f;
      for (int a = 0; a < 3; a++) {
        float mn = lo[a] > 0 ? lo[a] : (hi[a] < 0 ? -hi[a] : 0.0f);
        float mx = fmaxf(fabsf(lo[a]), fabsf(hi[a]));
        minlen2 += mn * mn;
        maxlen2 += mx * mx;
      }
      float minlen = sqrtf(minlen2), maxlen = sqrtf(maxlen2);
      for (int a = 0; a < 3; a++) {
        size_t k = (size_t)a * nt + (size_t)ty * rows->tw + tx;
        /* If interpolation may cancel, any unit direction is conservative.
         * Otherwise bound division by its positive interval of lengths. */
        n.normal_min[k] = -1.0f;
        n.normal_max[k] = 1.0f;
        if (normals && minlen > 1e-6f && isfinite(maxlen)) {
          n.normal_min[k] = fmaxf(-1.0f, lo[a] / (lo[a] < 0 ? minlen : maxlen));
          n.normal_max[k] = fminf(1.0f, hi[a] / (hi[a] > 0 ? minlen : maxlen));
        }
      }
    }
  r3d_surface_bounds_free(b);
  *b = n;
  return 0;
}
typedef void (*r3d_surface_box_fn)(void *, const float[3], const float[3]);
/* Grid bounds are half-open. Tile bounds include shared cell vertices.
 * Extruding their boxes through the normal interval covers all bilinear
 * positions, including normalized normals, at both slab endpoints. */
static inline void r3d_surface_cover(const r3d_segrows *rows,
                                     const r3d_surface_bounds *b, uint32_t x0,
                                     uint32_t y0, uint32_t x1, uint32_t y1,
                                     float z0, float z1, float halo,
                                     r3d_surface_box_fn emit, void *ud) {
  if (!rows->tmn || x0 >= x1 || y0 >= y1)
    return;
  size_t nt = (size_t)rows->tw * rows->th;
  for (uint32_t ty = y0 / R3D_SEGROWS_TILE;
       ty < rows->th && ty <= (y1 - 1) / R3D_SEGROWS_TILE; ty++)
    for (uint32_t tx = x0 / R3D_SEGROWS_TILE;
         tx < rows->tw && tx <= (x1 - 1) / R3D_SEGROWS_TILE; tx++) {
      float lo[3], hi[3];
      bool valid = true;
      for (int a = 0; a < 3; a++) {
        size_t k = (size_t)a * nt + (size_t)ty * rows->tw + tx;
        float nl = b && b->normal_min ? b->normal_min[k] : -1.0f;
        float nh = b && b->normal_max ? b->normal_max[k] : 1.0f;
        float p0 = z0 * nl, p1 = z0 * nh, p2 = z1 * nl, p3 = z1 * nh;
        lo[a] = rows->tmn[k] + fminf(fminf(p0, p1), fminf(p2, p3)) - halo;
        hi[a] = rows->tmx[k] + fmaxf(fmaxf(p0, p1), fmaxf(p2, p3)) + halo;
        if (!isfinite(lo[a]) || !isfinite(hi[a]) || lo[a] > hi[a])
          valid = false;
      }
      if (valid)
        emit(ud, lo, hi);
    }
}

typedef struct r3d_trace_point {
  double u, v;
  size_t index;
} r3d_trace_point;
typedef struct r3d_trace_slice {
  r3d_trace_point *points;
  size_t n, cap;
  uint64_t revision;
  double key[14]; /* origin, basis, slice, thickness */
  bool valid;
} r3d_trace_slice;
static inline void r3d_trace_slice_free(r3d_trace_slice *c) {
  free(c->points);
  memset(c, 0, sizeof *c);
}
static inline int
r3d_trace_slice_update(r3d_trace_slice *c, uint64_t revision, const double *pos,
                       const uint8_t *state, size_t n, const double org[3],
                       const double basis[3][3], double slice, double thick) {
  double key[14];
  memcpy(key, org, 3 * sizeof(double));
  memcpy(key + 3, basis, 9 * sizeof(double));
  key[12] = slice;
  key[13] = thick;
  if (c->valid && c->revision == revision &&
      memcmp(c->key, key, sizeof key) == 0)
    return 0;
  c->valid = false;
  c->n = 0;
  for (size_t k = 0; k < n; k++) {
    if (state[k] != R3D_TR_SET)
      continue;
    double d[3] = {pos[k * 3] - org[0], pos[k * 3 + 1] - org[1],
                   pos[k * 3 + 2] - org[2]};
    double s = d[0] * basis[2][0] + d[1] * basis[2][1] + d[2] * basis[2][2];
    if (s < slice - 1.0 || s > slice + thick + 1.0)
      continue;
    if (c->n == c->cap) {
      size_t cap = c->cap ? c->cap * 2 : 256;
      r3d_trace_point *p = realloc(c->points, cap * sizeof *p);
      if (!p)
        return -1;
      c->points = p;
      c->cap = cap;
    }
    c->points[c->n++] = (r3d_trace_point){
        d[0] * basis[0][0] + d[1] * basis[0][1] + d[2] * basis[0][2],
        d[0] * basis[1][0] + d[1] * basis[1][1] + d[2] * basis[1][2], k};
  }
  memcpy(c->key, key, sizeof key);
  c->revision = revision;
  c->valid = true;
  return 1;
}
#endif
