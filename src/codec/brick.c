#include "brick.h"
#include <stdlib.h>
#include <volcomp.h>
#include "block_decode.h"
int r3d_validate_brick(const uint8_t *in, size_t n) {
  vf_parsed parsed;
  return in && vf_parse(in, n, &parsed) == VOLCOMP_OK ? 0 : -1;
}
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
  /* The pinned codec's public block API rebuilds ~42 KiB of entropy tables
   * on every call. Retain one parsed header per thread, never decoded voxels
   * or caller-owned storage. Compare header bytes rather than addresses:
   * shard mappings and network buffers can be freed/reused between calls.
   * Four owned substream copies also retain block-boundary entropy state;
   * content comparison makes checkpoints safe across buffer reuse/mutation.
   * This adds ~263 KiB per decoder thread, no extra decoded voxels. Larger
   * substreams decode without checkpoints. These internals are pinned;
   * keep public-decoder parity tests when updating volume-compressor. */
  static _Thread_local struct {
    vf_parsed parsed;
    uint8_t header[VF_HDR_BYTES + VF_TABLES_MAX_BYTES + VF_DIR_BYTES];
    size_t header_n, encoded_n;
    r3d_restart restart[R3D_RESTART_SLOTS];
    uint32_t victim;
  } cache;
  if (!in || !dst || bz >= 8 || by >= 8 || bx >= 8)
    return -1;
  if (!cache.header_n || n != cache.encoded_n ||
      memcmp(in, cache.header, cache.header_n) != 0) {
    cache.header_n = 0;
    for(unsigned i=0;i<R3D_RESTART_SLOTS;i++)cache.restart[i].n=0;
    if (vf_parse(in, n, &cache.parsed) != VOLCOMP_OK)
      return -1;
    size_t hn = (size_t)(cache.parsed.payload - in);
    if (hn > sizeof cache.header)
      return -1;
    memcpy(cache.header, in, hn);
    cache.header_n = hn;
    cache.encoded_n = n;
  }
  cache.parsed.payload = in + cache.header_n;
  uint32_t bi = bz * 64u + by * 8u + bx;
  uint32_t sub=bi/VF_BLOCKS_PER_SUB;
  size_t bytes=cache.parsed.off[sub+1]-cache.parsed.off[sub];
  r3d_restart *restart=NULL;
  const uint8_t *src=cache.parsed.payload+cache.parsed.off[sub];
  if (bytes<=R3D_RESTART_BYTES) {
    for(unsigned i=0;i<R3D_RESTART_SLOTS;i++) {
      r3d_restart *r=&cache.restart[i];
      if(r->n==bytes && r->sub==sub && !memcmp(src,r->bytes,bytes)) { restart=r; break; }
    }
    if(!restart) {
      restart=&cache.restart[cache.victim++%R3D_RESTART_SLOTS];
      restart->n=bytes;restart->sub=sub;restart->valid=0;
      memcpy(restart->bytes,src,bytes);
    }
  }
  return r3d_decode_sub(&cache.parsed,sub,bi,dst,restart)==VOLCOMP_OK ? 0 : -1;
}
