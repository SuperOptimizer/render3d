/* Lossless float surface storage, independent of the u8 volume codec. */
#ifndef R3D_CODEC_TIFXYZ_H
#define R3D_CODEC_TIFXYZ_H
#include <stddef.h>
#include <stdint.h>
typedef struct r3d_surface_data {
  uint32_t w, h;
  float *plane[3];
  uint8_t *meta;
  size_t meta_len;
} r3d_surface_data;
int r3d_surface_encode(const r3d_surface_data *t, int log2q, uint8_t **out,
                       size_t *n);
/* Views borrow buffers; stride is measured in floats. Decode destinations
 * must match info dimensions and metadata length; callers publish on success. */
int r3d_surface_encode_strided(const r3d_surface_data *t, size_t stride,
                               uint8_t **out, size_t *n);
int r3d_surface_info(const uint8_t *in, size_t n, uint32_t *w, uint32_t *h, size_t *meta);
int r3d_surface_decode_strided(const uint8_t *in, size_t n, r3d_surface_data *t,
                               size_t stride);
int r3d_surface_decode(const uint8_t *in, size_t n, r3d_surface_data *t);
int r3d_surface_load_dir(const char *dir, r3d_surface_data *t);
int r3d_surface_save_dir(const char *dir, const r3d_surface_data *t);
void r3d_surface_free(r3d_surface_data *t);
#endif
