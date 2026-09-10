/* Application performance fixes: conservative fine-block coverage and
 * generation-cached slice selection, plus benchmark interval accounting. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "app/benchmark.h"
#include "app/viewcache.h"
#include "core/cellset.h"
#include "core/mview.h"
#include "render/viewbounds.h"
#include <assert.h>
#include <stdio.h>
#define GRID 80
static unsigned char covered[GRID * GRID * GRID];
static uint32_t boxes;
static void mark_box(void *ud, const float lo[3], const float hi[3]) {
  (void)ud;
  int b0[3], b1[3];
  for (int a = 0; a < 3; a++) {
    b0[a] = (int)floorf(lo[a] / 16.0f);
    b1[a] = (int)floorf(hi[a] / 16.0f);
    if (b0[a] < 0)
      b0[a] = 0;
    if (b1[a] >= GRID)
      b1[a] = GRID - 1;
  }
  for (int z = b0[2]; z <= b1[2]; z++)
    for (int y = b0[1]; y <= b1[1]; y++)
      for (int x = b0[0]; x <= b1[0]; x++)
        covered[(z * GRID + y) * GRID + x] = 1;
  boxes++;
}
static void coverage(void) {
  r3d_tifxyz s = {.w = 1025, .h = 1025, .sx = 1, .sy = 1};
  size_t n = (size_t)s.w * s.h;
  s.xyz = malloc(n * 3 * sizeof(float));
  float *normals = calloc(n * 4, sizeof(float));
  assert(s.xyz && normals);
  for (uint32_t y = 0; y < s.h; y++)
    for (uint32_t x = 0; x < s.w; x++) {
      size_t k = (size_t)y * s.w + x;
      s.xyz[k * 3] = (float)(128 + x);
      s.xyz[k * 3 + 1] = (float)(128 + y);
      s.xyz[k * 3 + 2] = 128.0f;
      normals[k * 4 + 2] = 1.0f;
    }
  r3d_segrows rows = {0};
  r3d_surface_bounds bounds = {0};
  assert(r3d_segrows_build(&s, &rows) == 0);
  assert(r3d_surface_bounds_build(&bounds, &s, &rows, normals, 1) == 0);
  const float offsets[] = {-48.0f, 0.0f, 48.0f};
  for (size_t trial = 0; trial < 3; trial++) {
    memset(covered, 0, sizeof covered);
    boxes = 0;
    r3d_surface_cover(&rows, &bounds, 0, 0, 1024, 1024, offsets[trial] - 32.0f,
                      offsets[trial] + 32.0f, 1.0f, mark_box, NULL);
    assert(boxes == 4096); /* old384 samples could never cover these tiles */
    for (uint32_t y = 8; y < 72; y++)
      for (uint32_t x = 8; x < 72; x++)
        for (int z = (int)(96.0f + offsets[trial]) / 16;
             z <= (int)(160.0f + offsets[trial]) / 16; z++)
          assert(covered[((size_t)z * GRID + y) * GRID + x]);
    /* A constant normal must not inflate the XY footprint by |offset|. */
    assert(!covered[((size_t)8 * GRID + 4) * GRID + 4]);
  }
  memset(covered, 0, sizeof covered);
  boxes = 0;
  r3d_surface_cover(&rows, &bounds, 0, 0, 0, 1024, 0, 0, 1, mark_box, NULL);
  assert(boxes == 0);
  /* Normal cancellation is handled conservatively instead of dividing by0. */
  for (size_t k = 0; k < n; k++)
    normals[k * 4 + 2] = (k & 1) ? -1.0f : 1.0f;
  assert(r3d_surface_bounds_build(&bounds, &s, &rows, normals, 2) == 0);
  assert(bounds.normal_min[0] == -1.0f && bounds.normal_max[0] == 1.0f);
  r3d_surface_bounds_free(&bounds);
  r3d_segrows_free(&rows);
  free(s.xyz);
  free(normals);
}
static void slices(void) {
  double pos[3000];
  uint8_t states[1000];
  for (size_t k = 0; k < 1000; k++) {
    pos[k * 3] = (double)(k % 10);
    pos[k * 3 + 1] = (double)(k / 10 % 10);
    pos[k * 3 + 2] = (double)(k / 100);
    states[k] = k % 7 ? R3D_TR_SET : R3D_TR_EMPTY;
  }
  const double org[3] = {2, 3, 1},
               basis[3][3] = {{1, 0, 0}, {0, 0.8, 0.6}, {0, -0.6, 0.8}};
  r3d_trace_slice cache = {0};
  assert(r3d_trace_slice_update(&cache, 1, pos, states, 1000, org, basis, 2,
                                3) == 1);
  size_t count = 0;
  for (size_t k = 0; k < 1000; k++) {
    double u, v, s;
    r3d_mv_w2b(basis, org, pos + k * 3, &u, &v, &s);
    if (states[k] != R3D_TR_SET || s < 1 || s > 6)
      continue;
    assert(count < cache.n && cache.points[count].index == k);
    assert(fabs(cache.points[count].u - u) < 1e-12 &&
           fabs(cache.points[count].v - v) < 1e-12);
    count++;
  }
  assert(count == cache.n && count > 0);
  assert(r3d_trace_slice_update(&cache, 1, pos, states, 1000, org, basis, 2,
                                3) == 0);
  assert(r3d_trace_slice_update(&cache, 2, pos, states, 1000, org, basis, 2,
                                3) == 1);
  assert(r3d_trace_slice_update(&cache, 2, pos, states, 1000, org, basis, 4,
                                3) == 1);
  r3d_trace_slice_free(&cache);
}
static void benchmark(void) {
  app_frame_sample distinct = {.phases = {.gpu_ns = 101,
                                          .gpu_raycast_ns = 202,
                                          .gpu_blit_ns = 303,
                                          .gpu_gui_ns = 404,
                                          .panes_drawn = 987654,
                                          .cpu_wait_ns = 505,
                                          .cpu_acquire_ns = 606,
                                          .cpu_record_ns = 707,
                                          .cpu_submit_ns = 808},
                               .cpu_ns = 909};
  for (size_t field = 0; field < 9; field++)
    assert(app_sample_timing(&distinct, field) == 101 * (field + 1));
  app_samples samples = {0};
  r3d_frame_stats st = {.gpu_ns = 99};
  assert(samples.cap == 0);
  for (uint64_t i = 0; i < 10000; i++)
    assert(app_samples_push(&samples, &st, i));
  assert(samples.n == 10000 && samples.cap < 20000);
  assert(samples.data[0].cpu_ns == 0 && samples.data[9999].cpu_ns == 9999);
  assert(samples.data[9999].phases.gpu_ns == 99);
  free(samples.data);
  r3d_bricks_stats a = {.decoded = 100,
                        .jobs = 10,
                        .stream_ns = 500,
                        .failures = 1,
                        .lod_requests = {70}},
                   b = {.decoded = 145,
                        .jobs = 13,
                        .stream_ns = 800,
                        .failures = 2,
                        .lod_requests = {95}};
  r3d_bricks_stats d = app_bricks_delta(&b, &a);
  assert(d.decoded == 45 && d.jobs == 3 && d.stream_ns == 300 &&
         d.failures == 1 && d.lod_requests[0] == 25);
}
static void viewbounds(void) {
  const float extent[3] = {1, 1, 1};
  float lo[3], hi[3];
  r3d_frame_params p = {.brick_mode = 0x20001,
                        .view_flags = R3D_VIEW_ORTHO | R3D_VIEW_OBLIQUE,
                        .cam_origin = {0.5f, 0.5f, 0.1f},
                        .cam_forward = {0, 0, 1},
                        .cam_right = {0.2f, 0, 0},
                        .cam_up = {0, 0.1f, 0},
                        .vol_r0 = {1, 0, 0},
                        .vol_r1 = {0, 1, 0},
                        .vol_r2 = {0, 0, 1},
                        .slab_z0 = 20,
                        .slab_depth = 40};
  assert(r3d_view_bounds(&p, extent, 1000, lo, hi));
  assert(fabsf(lo[0] - 0.3f) < 2e-5f && fabsf(hi[0] - 0.7f) < 2e-5f);
  assert(fabsf(lo[2] - 0.12f) < 2e-5f && fabsf(hi[2] - 0.16f) < 2e-5f);
  /* Shader model transform: rotation about volume center plus translation. */
  p.vol_r0[0] = 0;
  p.vol_r0[2] = 1;
  p.vol_r2[0] = -1;
  p.vol_r2[2] = 0;
  p.vol_tx = 0.15f;
  assert(r3d_view_bounds(&p, extent, 1000, lo, hi));
  for (int y = 0; y <= 10; y++)
    for (int x = 0; x <= 10; x++)
      for (int z = 0; z <= 10; z++) {
        double origin[3] = {0.5 + ((double)x / 5 - 1) * 0.2 - 0.15 - 0.5,
                            0.5 + ((double)y / 5 - 1) * 0.1 - 0.5,
                            0.1 + 0.02 + (double)z * 0.004 - 0.5};
        /* Independently apply R^T for this known ninety-degree rotation. */
        double v[3] = {-origin[2] + 0.5, origin[1] + 0.5, origin[0] + 0.5};
        for (int a = 0; a < 3; a++)
          assert(v[a] >= (double)lo[a] && v[a] <= (double)hi[a]);
      }
  p.view_flags = R3D_VIEW_CROP;
  p.slab_x0 = 200;
  p.slab_y0 = 300;
  p.slab_z0 = 400;
  p.slab_px = 100;
  p.slab_py = 200;
  p.slab_nx = 300;
  assert(r3d_view_bounds(&p, extent, 1000, lo, hi));
  assert(fabsf(lo[0] - 0.2f) < 2e-5f && fabsf(hi[2] - 0.7f) < 2e-5f);
  p.view_flags = R3D_VIEW_AXIS(1);
  p.slab_z0 = 200;
  p.slab_depth = 10;
  assert(r3d_view_bounds(&p, extent, 1000, lo, hi));
  assert(fabsf(lo[0] - 0.2f) < 2e-5f && fabsf(hi[0] - 0.21f) < 2e-5f &&
         hi[2] == 1);
  p.view_flags = 0;
  p.slab_depth = 0;
  assert(r3d_view_bounds(&p, extent, 1000, lo, hi) && lo[0] == 0 && hi[0] == 1);
  p.view_flags = R3D_VIEW_SURF;
  assert(!r3d_view_bounds(&p, extent, 1000, lo, hi));
}
static void cellset(void) {
  r3d_cellset set = {0};
  for (uint32_t repeat = 0; repeat < 4; repeat++)
    for (uint32_t i = 0; i < 10000; i++)
      assert(r3d_cellset_add(&set, i, UINT32_MAX - i, i * 97, 10000) ==
             (repeat ? 0 : 1));
  assert(set.n == 10000 && set.capacity < 20000);
  for (uint32_t i = 0; i < 10000; i++)
    assert(set.cells[i][0] == i && set.cells[i][1] == UINT32_MAX - i &&
           set.cells[i][2] == i * 97);
  assert(r3d_cellset_add(&set, 0, 0, 0, 10000) == -2);
  r3d_cellset_clear(&set);
  assert(r3d_cellset_add(&set, 0, 0, 0, 10000) == 1);
  assert(r3d_cellset_add(&set, 0, 0, 0, 10000) == 0);
  r3d_cellset_free(&set);
}
int main(void) {
  viewbounds();
  cellset();
  coverage();
  slices();
  benchmark();
  puts("app performance helpers: OK");
  return 0;
}
