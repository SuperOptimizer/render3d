/* Application allocation wrapper around volume-compressor's native 128^3
 * stream. */
#ifndef R3D_CODEC_BRICK_H
#define R3D_CODEC_BRICK_H
#include <stddef.h>
#include <stdint.h>
typedef struct volcomp_brick_params {
  float q;
} volcomp_brick_params;
volcomp_brick_params volcomp_brick_defaults(float q);
int volcomp_brick_encode(const volcomp_brick_params *p, const uint8_t *src,
                         uint32_t dim, uint8_t **out, size_t *out_n);
int volcomp_brick_decode(const uint8_t *in, size_t n, uint8_t *dst,
                         uint32_t dim);
int r3d_decode_block(const uint8_t *in, size_t n, uint32_t bz, uint32_t by,
                     uint32_t bx, uint8_t *dst);
#endif
