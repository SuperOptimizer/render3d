/* Native block decode, sparse residency, chunk crossings, and malformed
 * storage. */
#include "core/cpuvol.h"
#include "core/regvol.h"
#include "synthtree.h"
#include <volcomp.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  uint8_t *raw = malloc(VOLCOMP_CHUNK_VOXELS),
          *full = malloc(VOLCOMP_CHUNK_VOXELS), *enc = NULL;
  assert(raw && full);
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++)
        raw[((size_t)z * 128 + y) * 128 + x] = st_pat(x, y, z);
  volcomp_brick_params p = volcomp_brick_defaults(2);
  size_t n = 0;
  assert(volcomp_brick_encode(&p, raw, 128, &enc, &n) == 0);
  /* The allocation wrapper writes upstream's actual stream, not a renamed
   * legacy codec. */
  assert(volcomp_decode(enc, n, full, VOLCOMP_CHUNK_VOXELS) == VOLCOMP_OK);
  uint8_t block[4096];
  for (uint32_t z = 0; z < 8; z++)
    for (uint32_t y = 0; y < 8; y++)
      for (uint32_t x = 0; x < 8; x++) {
        assert(r3d_decode_block(enc, n, z, y, x, block) == 0);
        for (uint32_t zz = 0; zz < 16; zz++)
          for (uint32_t yy = 0; yy < 16; yy++)
            assert(memcmp(block + (zz * 16 + yy) * 16,
                          full + ((z * 16 + zz) * 128 + y * 16 + yy) * 128 +
                              x * 16,
                          16) == 0);
      }
  assert(r3d_decode_block(enc, 4, 0, 0, 0, block) != 0);
  assert(r3d_decode_block(enc, n, 8, 0, 0, block) != 0);
  assert(r3d_decode_block((const uint8_t *)"C5D1", 4, 0, 0, 0, block) != 0);
  free(enc);
  free(raw);
  free(full);

  char root[] = "/tmp/r3d_blocks_XXXXXX";
  assert(mkdtemp(root));
  uint32_t dims[3] = {256, 128, 128};
  assert(st_make_tree(root, dims, 2, 0) == 0);
  r3d_cpuvol v;
  assert(r3d_cpuvol_open(&v, root, 8) == 0);
  r3d_cpuvol_cache_stats st;
  r3d_cpuvol_get_cache_stats(&v, &st);
  assert(st.resident_blocks == 0 && st.capacity_bytes == 8 * 4096);
  (void)r3d_cpuvol_at(&v, 0, 16, 16, 16);
  r3d_cpuvol_get_cache_stats(&v, &st);
  assert(st.resident_blocks == 1 && st.resident_bytes == 4096 &&
         st.decoded_blocks == 1);
  (void)r3d_cpuvol_at(&v, 0, 31, 31, 31);
  r3d_cpuvol_get_cache_stats(&v, &st);
  assert(st.decoded_blocks == 1);
  (void)r3d_cpuvol_at(&v, 0, 32, 16, 16);
  r3d_cpuvol_get_cache_stats(&v, &st);
  assert(st.decoded_blocks == 2 && st.resident_bytes == 8192);

  /* Crossing both a 16^3 boundary and the 128^3 storage boundary. */
  uint8_t roi[6 * 6 * 6];
  r3d_cpuvol_read_block(&v, 0, 125, 13, 13, 6, 6, 6, roi);
  volcomp_shard_reader sr;
  char path[256];
  snprintf(path, sizeof path, "%s/volcomp/L0/0_0_0.vcs", root);
  assert(volcomp_shard_open(path, &sr) == 0);
  uint8_t *chunks = malloc(2u * VOLCOMP_CHUNK_VOXELS);
  assert(chunks);
  for (uint32_t i = 0; i < 2; i++) {
    const uint8_t *b = volcomp_shard_brick(&sr, i, &n);
    assert(b);
    assert(volcomp_decode(b, n, chunks + (size_t)i * VOLCOMP_CHUNK_VOXELS,
                          VOLCOMP_CHUNK_VOXELS) == VOLCOMP_OK);
  }
  for (uint32_t z = 0; z < 6; z++)
    for (uint32_t y = 0; y < 6; y++)
      for (uint32_t x = 0; x < 6; x++) {
        uint32_t wx = 125 + x;
        assert(roi[(z * 6 + y) * 6 + x] ==
               chunks[(size_t)(wx / 128) * VOLCOMP_CHUNK_VOXELS +
                      ((z + 13) * 128 + y + 13) * 128 + wx % 128]);
      }
  r3d_cpuvol_get_cache_stats(&v, &st);
  assert(st.resident_bytes <= 8 * 4096);
  volcomp_shard_close_reader(&sr);
  free(chunks);
  r3d_cpuvol_close(&v);
  /* Renderer registration callbacks also materialize only one 16^3 block. */
  r3d_regvol rv;
  assert(r3d_regvol_open(&rv, root, dims) == 0);
  uint8_t guarded[4098];
  memset(guarded, 0xab, sizeof guarded);
  r3d_regvol_blockfetch(&rv, 0, 8, 1, 1, guarded + 1);
  assert(guarded[0] == 0xab && guarded[4097] == 0xab);
  assert(r3d_cpuvol_open(&v, root, 8) == 0);
  r3d_cpuvol_read_block(&v, 0, 128, 16, 16, 16, 16, 16, block);
  assert(memcmp(block, guarded + 1, 4096) == 0);
  r3d_cpuvol_close(&v);
  r3d_regvol_close(&rv);
  st_rm_tree(root, 2);
  puts("512 native blocks match full decode; sparse CPU cache uses 4 KiB per "
       "touched block");
  return 0;
}
