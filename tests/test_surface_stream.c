/* Independent legacy R3F1 construction validates the streaming codec, exact
 * float bits, allocation ownership, malformed input and transactional output. */
#include "render3d/headless.h"
#include "bytes.h"
#include <zlib.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct alloc_state { size_t live, calls, fail_at; } alloc_state;
static void *allocate(void *user, size_t n) {
  alloc_state *a = user;
  if (++a->calls == a->fail_at) return NULL;
  void *p = malloc(n);
  if (p) a->live++;
  return p;
}
static void release(void *user, void *p) {
  alloc_state *a = user;
  if (p) { assert(a->live); a->live--; free(p); }
}
int main(void) {
  const uint32_t w = 257, h = 193;
  size_t np = (size_t)w * h, raw_n = np * 12u + 70003u;
  uint8_t *raw = malloc(raw_n);
  float *xyz = malloc(np * 12u);
  assert(raw && xyz);
  for (size_t i = 0; i < np; i++)
    for (size_t a = 0; a < 3; a++) {
      /* Includes NaNs, signed zeros, infinities and arbitrary float payloads. */
      uint32_t bits = i < 4 ? (uint32_t[]){0,0x80000000u,0x7f800000u,0x7fc12345u}[i]
                           : (uint32_t)((i * 1640531527u) ^ (a * 2246822519u));
      memcpy(xyz + i * 3u + a, &bits, 4);
      r3c_put32(raw + (a * np + i) * 4u, bits);
    }
  for (size_t i = np * 12u; i < raw_n; i++) raw[i] = (uint8_t)(i * 71u);
  uLongf cap = compressBound((uLong)raw_n);
  uint8_t *legacy = malloc((size_t)cap + 25u);
  assert(legacy);
  memcpy(legacy, "R3F1", 4);
  r3c_put32(legacy + 4, w); r3c_put32(legacy + 8, h);
  r3c_put32(legacy + 12, 0); r3c_put64(legacy + 16, 70003u);
  assert(compress2(legacy + 24, &cap, raw, (uLong)raw_n, Z_BEST_SPEED) == Z_OK);
  size_t legacy_n = (size_t)cap + 24u;
  alloc_state state = {0};
  r3d_headless_allocator alloc = {&state, allocate, release};
  r3d_headless_surface surface = {0};
  assert(r3d_headless_surface_decode_v1(legacy, legacy_n, &alloc, NULL, &surface) == 0);
  assert(state.live == 2 && state.calls == 2);
  assert(memcmp(surface.xyz, xyz, np * 12u) == 0);
  assert(memcmp(surface.metadata, raw + np * 12u, 70003u) == 0);
  r3d_headless_surface_release_v1(&surface, &alloc);
  assert(state.live == 0);
  r3d_headless_bytes enc = {0};
  assert(r3d_headless_surface_encode_v1(w,h,xyz,raw + np * 12u,70003u,-1,NULL,NULL,&enc)==0);
  uint8_t *decoded = malloc(raw_n);
  uLongf len = (uLongf)raw_n;
  assert(uncompress(decoded, &len, enc.data + 24, (uLong)(enc.size - 24)) == Z_OK);
  assert(len == raw_n && memcmp(raw, decoded, raw_n) == 0);
  r3d_headless_bytes_release_v1(&enc, NULL);
  /* Caller outputs survive truncated input, trailing garbage, wrong size,
   * and failure of either custom allocation. */
  surface.width = 123;
  assert(r3d_headless_surface_decode_v1(legacy,legacy_n-1,&alloc,NULL,&surface)!=0);
  assert(surface.width==123 && state.live==0);
  legacy[legacy_n] = 0;
  assert(r3d_headless_surface_decode_v1(legacy,legacy_n+1,&alloc,NULL,&surface)!=0);
  r3c_put32(legacy + 4, w + 1);
  assert(r3d_headless_surface_decode_v1(legacy,legacy_n,&alloc,NULL,&surface)!=0);
  r3c_put32(legacy + 4, w);
  for (size_t fail = 1; fail <= 2; fail++) {
    state = (alloc_state){.fail_at=fail};
    assert(r3d_headless_surface_decode_v1(legacy,legacy_n,&alloc,NULL,&surface)==R3D_HEADLESS_E_OUT_OF_MEMORY);
    assert(surface.width==123 && state.live==0);
  }
  free(decoded); free(legacy); free(raw); free(xyz);
  puts("streaming surface codec: exact legacy compatibility and ownership OK");
  return 0;
}
