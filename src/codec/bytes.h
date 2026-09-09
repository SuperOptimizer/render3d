#ifndef R3D_CODEC_BYTES_H
#define R3D_CODEC_BYTES_H
#include <stdint.h>
static inline uint32_t r3c_u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static inline uint64_t r3c_u64(const uint8_t *p) {
  return r3c_u32(p) | (uint64_t)r3c_u32(p + 4) << 32;
}
static inline void r3c_put32(uint8_t *p, uint32_t v) {
  for (unsigned i = 0; i < 4; i++)
    p[i] = (uint8_t)(v >> (8 * i));
}
static inline void r3c_put64(uint8_t *p, uint64_t v) {
  r3c_put32(p, (uint32_t)v);
  r3c_put32(p + 4, (uint32_t)(v >> 32));
}
#endif
