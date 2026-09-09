#include "brick.h"
#include <stdlib.h>
#include <volcomp.h>
volcomp_brick_params volcomp_brick_defaults(float q) {
  return (volcomp_brick_params){.q = q};
}
int volcomp_brick_encode(const volcomp_brick_params *p, const uint8_t *src,
                         uint32_t dim, uint8_t **out, size_t *out_n) {
  if (!out || !out_n)
    return -1;
  *out = NULL;
  *out_n = 0;
  if (!p || !src || dim != 128)
    return -1;
  uint8_t *buf = malloc(VOLCOMP_ENCODE_BOUND);
  if (!buf)
    return -1;
  size_t n = 0;
  if (volcomp_encode(src, p->q, buf, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK) {
    free(buf);
    return -1;
  }
  uint8_t *small = realloc(buf, n);
  *out = small ? small : buf;
  *out_n = n;
  return 0;
}
int volcomp_brick_decode(const uint8_t *in, size_t n, uint8_t *dst,
                         uint32_t dim) {
  if (!in || !dst || dim != 128)
    return -1;
  return volcomp_decode(in, n, dst, VOLCOMP_CHUNK_VOXELS) == VOLCOMP_OK ? 0
                                                                        : -1;
}
int r3d_decode_block(const uint8_t *in, size_t n, uint32_t bz, uint32_t by,
                     uint32_t bx, uint8_t *dst) {
  return volcomp_decode_block(in, n, bz, by, bx, dst, VOLCOMP_BLOCK_VOXELS) ==
                 VOLCOMP_OK
             ? 0
             : -1;
}
