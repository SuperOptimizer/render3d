/* Bounded, thread-safe 64x64 XYZ cache over surface-compressor files. */
#ifndef R3D_SURFACE_H
#define R3D_SURFACE_H
#include "core/tifxyz.h"
#include <stddef.h>
#include <stdint.h>
typedef struct r3d_surface_reader r3d_surface_reader;
typedef struct {
  uint64_t width, height;
  float sx, sy;
} r3d_surface_info;
typedef struct {
  uint64_t hits, misses, bytes, limit;
} r3d_surface_stats;
int r3d_surface_open(const char *path, size_t cache_bytes,
                     r3d_surface_reader **out);
void r3d_surface_close(r3d_surface_reader *);
void r3d_surface_get_info(const r3d_surface_reader *, r3d_surface_info *);
/* Coordinates are global surface-grid indices; output is interleaved XYZ.
 * A failed read leaves destination unchanged. Memory is bounded by the ROI. */
int r3d_surface_read(r3d_surface_reader *, uint64_t x, uint64_t y, uint32_t w,
                     uint32_t h, float *xyz, size_t row_stride_floats);
int r3d_surface_point(r3d_surface_reader *, uint64_t x, uint64_t y,
                      float xyz[3]);
int r3d_surface_block_bounds(r3d_surface_reader *, uint64_t bx, uint64_t by,
                             float lo[3], float hi[3]);
void r3d_surface_get_stats(r3d_surface_reader *, r3d_surface_stats *);
/* Bounded geometry window. out must be empty; release with r3d_tifxyz_free.
 * Output indices are local; x/y retain the corresponding global origin. */
int r3d_surface_window(r3d_surface_reader *, uint64_t x, uint64_t y, uint32_t w,
                       uint32_t h, r3d_tifxyz *out);
/* Recenter a block-aligned window without converting global indices to
 * double (which loses individual grid points above 2^53). jump is optional. */
int r3d_surface_axis_window(uint64_t full, uint32_t extent, uint64_t origin,
                            double local, const uint64_t *jump, uint64_t *next,
                            double *camera);
#endif
