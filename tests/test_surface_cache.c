#include "core/surface.h"
#include "surfcomp.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Keep assertions active in release test builds. */
#undef assert
#define assert(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)
static void *read_points(void *arg) {
  r3d_surface_reader *r = arg;
  for (unsigned i = 0; i < 120; i++) {
    unsigned x = (i * 31) % 130, y = (i * 17) % 129;
    float p[3];
    assert(!r3d_surface_point(r, x, y, p));
    if (x == 64 && y == 64)
      assert(p[0] == -1);
    else
      for (unsigned a = 0; a < 3; a++)
        assert(fabsf(p[a] - (float)(100 + a * 1000 + x + 2 * y)) <= 0.01f);
  }
  return NULL;
}
int main(void) {
  for (unsigned joint = 0; joint < 2; joint++) {
    char path[] = "/tmp/r3d-surface-cache-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    unlink(path);
    sfc_channel c[3] = {0};
    for (unsigned a = 0; a < 3; a++) {
      c[a].name[0] = "xyz"[a];
      c[a].width = 130;
      c[a].height = 129;
      c[a].dtype = SFC_F32;
      c[a].flags = SFC_COORDINATE | (joint ? SFC_XYZ : 0);
      c[a].components = joint ? 3 : 0;
      c[a].component = joint ? a : 0;
      c[a].tolerance = .01;
    }
    const char meta[] = "{\"scale\":[0.5,0.25],\"custom\":true}";
    sfc_writer *writer = NULL;
    assert(!sfc_create(path, c, 3, meta, strlen(meta), &writer));
    float block[4096];
    uint8_t mask[4096];
    if (joint) {
      float xyz[4096][3];
      for (unsigned by = 0; by < 3; by++)
        for (unsigned bx = 0; bx < 3; bx++) {
          for (unsigned i = 0; i < 4096; i++) {
            unsigned xx = bx * 64 + i % 64, yy = by * 64 + i / 64;
            mask[i] = xx < 130 && yy < 129 && !(xx == 64 && yy == 64);
            for (unsigned a = 0; a < 3; a++)
              xyz[i][a] = (float)(100 + a * 1000 + xx + 2 * yy);
          }
          assert(!sfc_write_xyz_block(writer, xyz, 12, 768, mask));
        }
    } else
      for (unsigned a = 0; a < 3; a++)
        for (unsigned by = 0; by < 3; by++)
          for (unsigned bx = 0; bx < 3; bx++) {
            for (unsigned y = 0; y < 64; y++)
              for (unsigned x = 0; x < 64; x++) {
                unsigned xx = bx * 64 + x, yy = by * 64 + y, k = y * 64 + x;
                mask[k] =
                    (uint8_t)(xx < 130 && yy < 129 && !(xx == 64 && yy == 64));
                block[k] = (float)(100 + a * 1000 + xx + 2 * yy);
              }
            assert(!sfc_write_block(writer, block, 4, 256, mask));
          }
    assert(!sfc_finish(writer));
    r3d_surface_reader *r = NULL;
    assert(!r3d_surface_open(path, 2 * 4096 * 12, &r));
    r3d_surface_info info;
    r3d_surface_get_info(r, &info);
    assert(info.width == 130 && info.height == 129 && info.sx == .5f &&
           info.sy == .25f);
    float roi[4 * 4 * 3];
    assert(!r3d_surface_read(r, 62, 62, 4, 4, roi, 12));
    assert(roi[(2 * 4 + 2) * 3] == -1);
    assert(fabsf(roi[0] - 286.f) <= .01f);
    r3d_surface_stats stats;
    r3d_surface_get_stats(r, &stats);
    assert(stats.misses == 4 && stats.bytes == 2 * 4096 * 12 &&
           stats.bytes <= stats.limit);
    float p[3];
    assert(!r3d_surface_point(r, 65, 65, p));
    r3d_surface_get_stats(r, &stats);
    assert(stats.hits == 1);
    float unchanged[3] = {7, 8, 9};
    assert(r3d_surface_point(r, 130, 0, unchanged));
    assert(unchanged[0] == 7 && unchanged[1] == 8 && unchanged[2] == 9);
    float lo[3], hi[3];
    assert(!r3d_surface_block_bounds(r, 0, 0, lo, hi));
    assert(lo[0] <= 100 && hi[0] >= 289);
    pthread_t threads[4];
    for (unsigned i = 0; i < 4; i++)
      assert(!pthread_create(threads + i, NULL, read_points, r));
    for (unsigned i = 0; i < 4; i++)
      assert(!pthread_join(threads[i], NULL));
    r3d_surface_get_stats(r, &stats);
    assert(stats.bytes <= stats.limit);
    r3d_tifxyz window = {0};
    assert(!r3d_surface_window(r, 62, 62, 4, 4, &window));
    assert(window.w == 4 && window.h == 4 && window.nvalid == 15);
    assert(window.sx == .5f && window.xyz[0] == roi[0]);
    r3d_tifxyz_free(&window);
    uint64_t origin = UINT64_C(9007199254740992), next = 0, jump = origin + 999;
    double camera = 0;
    assert(!r3d_surface_axis_window(origin + 8192, 1024, origin, 800.25, NULL,
                                    &next, &camera));
    assert(next == origin + 256 && camera == 544.25);
    assert(!r3d_surface_axis_window(origin + 8192, 1024, origin, 0, &jump,
                                    &next, &camera));
    assert(next == origin + 448 && camera == 551);
    r3d_surface_close(r);
    unlink(path);
  }
  puts("surface cache passed: scalar and joint XYZ");
  return 0;
}
