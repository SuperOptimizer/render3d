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
static uint32_t crc_reference(const uint8_t *p, size_t n) {
  uint32_t c = UINT32_MAX;
  while (n--) {
    c ^= *p++;
    for (int bit = 0; bit < 8; bit++) c = (c >> 1) ^ (0x82f63b78u & (0u - (c & 1u)));
  }
  return ~c;
}
static void test_wide_coordinates(void) {
  char root[] = "/tmp/r3d-wide-cpu-XXXXXX", path[512];
  assert(mkdtemp(root));
  snprintf(path, sizeof path, "%s/manifest.json", root);
  FILE *f = fopen(path, "w");
  assert(f);
  fputs("{\"format\": \"render3d.volcomp-lod.v1\",\"shape\":[128,128,16777344],\"shard_shape\":[1024,1024,1024],"
        "\"brick_shape\":[128,128,128],\"levels\":[{\"level\":0,\"scale\":1,"
        "\"shape\":[128,128,16777344],\"shards\":[1,1,16385]}]}", f);
  assert(fclose(f) == 0);
  r3d_cpuvol v;
  assert(r3d_cpuvol_open(&v, root, 1024) == 0);
  uint8_t *raw = malloc(VOLCOMP_CHUNK_VOXELS);
  assert(raw);
  memset(raw, 100, VOLCOMP_CHUNK_VOXELS);
  r3d_cpuvol_cache_put(&v, 0, 0, 0, 0, raw);
  memset(raw, 200, VOLCOMP_CHUNK_VOXELS);
  r3d_cpuvol_cache_put(&v, 0, 131072, 0, 0, raw);
  /* These coordinates collided when block axes occupied only 20 bits. */
  for (int i = 0; i < 3; i++) {
    assert(r3d_cpuvol_at(&v, 0, 16777216, 0, 0) == 200);
    assert(r3d_cpuvol_at(&v, 0, 0, 16, 0) == 100);
  }
  /* A cached edge read must not wrap ceil(UINT32_MAX/16) to zero. The
   * sparse cache does not need a giant reader allocation to exercise this. */
  v.lev[0].vx=UINT32_MAX; v.lev[0].gx=268435456u;
  r3d_cpuvol_cache_put(&v,0,33554431u,0,0,raw);
  uint8_t edge[8];
  assert(r3d_cpuvol_read_block_status(&v,0,4294967290ll,0,0,8,1,1,edge));
  for(unsigned i=0;i<8;i++)assert(edge[i]==(i<5?200:0));
  free(raw);
  r3d_cpuvol_close(&v);
  unlink(path);
  rmdir(root);
}
struct batch_test { r3d_cpuvol *v; const uint8_t *chunks; uint32_t seed; };
static void *test_batch_reads(void *arg) {
  struct batch_test *t=arg;
  uint8_t output[40][512]; r3d_block_region regions[40];
  for(uint32_t rep=0;rep<80;rep++) {
    for(uint32_t i=0;i<40;i++) {
      uint32_t k=i+rep*13u+t->seed;
      regions[i]=(r3d_block_region){.x=(k*17u)%250u,.y=(k*31u)%120u,.z=(k*11u)%120u,
        .nx=6,.ny=4,.nz=3,.out=output[i]};
    }
    r3d_cpuvol_read_regions(t->v,0,regions,40);
    for(uint32_t i=0;i<40;i++) {
      const r3d_block_region *q=&regions[i]; assert(q->available);
      for(uint32_t z=0;z<q->nz;z++)for(uint32_t y=0;y<q->ny;y++)for(uint32_t x=0;x<q->nx;x++) {
        uint32_t wx=q->x+x;
        assert(output[i][(z*q->ny+y)*q->nx+x]==t->chunks[(size_t)(wx/128u)*VOLCOMP_CHUNK_VOXELS+
          ((q->z+z)*128u+q->y+y)*128u+wx%128u]);
      }
    }
  }
  r3d_cpuvol_release_thread(t->v); return NULL;
}
int main(void) {
  test_wide_coordinates();
  assert(volcomp_crc32c("123456789", 9) == 0xe3069283u);
  uint8_t crc_data[1040];
  for (size_t i = 0; i < sizeof crc_data; i++) crc_data[i] = (uint8_t)(i * 71u);
  for (size_t offset = 0; offset < 8; offset++)
    for (size_t len = 0; len <= 1024; len++)
      assert(volcomp_crc32c(crc_data + offset, len) == crc_reference(crc_data + offset, len));
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
  /* Parsed-table reuse must follow bytes, not allocation identity, and must
   * preserve public-API validation after both header and payload changes. */
  uint8_t *copy = malloc(n);
  assert(copy);
  uint8_t expected[4096];
  for (uint32_t i = 0; i < 64; i++) {
    memcpy(copy, enc, n);
    if (i) copy[((size_t)i * 461) % n] ^= (uint8_t)(1u << (i % 8));
    uint32_t b = (i * 17) % 512;
    volcomp_status reference = volcomp_decode_block(copy, n, b / 64, b / 8 % 8, b % 8,
                                        expected, sizeof expected);
    int cached = r3d_decode_block(copy, n, b / 64, b / 8 % 8, b % 8, block);
    assert((cached == 0) == (reference == VOLCOMP_OK));
    if (cached == 0) assert(memcmp(block, expected, sizeof block) == 0);
    /* Revisit the original buffer after the mutated allocation. */
    assert(r3d_decode_block(enc, n, 0, 0, 0, block) == 0);
    assert(r3d_decode_block(enc, n - 1, 0, 0, 0, block) != 0);
  }
  free(copy);
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
  /* Batched hits and misses must survive concurrent eviction from just eight
   * slots, including reads crossing block/chunk boundaries and >32 regions. */
  pthread_t threads[4]; struct batch_test batch[4];
  for(uint32_t i=0;i<4;i++) {
    batch[i]=(struct batch_test){&v,chunks,i*37u};
    assert(!pthread_create(&threads[i],NULL,test_batch_reads,&batch[i]));
  }
  for(uint32_t i=0;i<4;i++)assert(!pthread_join(threads[i],NULL));
  r3d_cpuvol hot;
  assert(!r3d_cpuvol_open(&hot,root,1024));
  for(uint32_t i=0;i<2;i++)r3d_cpuvol_cache_put(&hot,0,i,0,0,chunks+(size_t)i*VOLCOMP_CHUNK_VOXELS);
  struct batch_test resident={&hot,chunks,0};test_batch_reads(&resident);
  r3d_cpuvol_get_cache_stats(&hot,&st);assert(st.decoded_blocks==0);
  r3d_cpuvol_close(&hot);
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
  /* Explicit VCSzero is complete; a missing native source chunk is not. */
  snprintf(path, sizeof path, "%s/volcomp/L0/0_0_0.vcs", root);
  volcomp_shard_writer *zero_source = volcomp_shard_create(path, 1024, 128, 0, 2);
  assert(zero_source && volcomp_shard_put_zero(zero_source, 0) == 0);
  assert(volcomp_shard_close(zero_source) == 0);
  assert(r3d_cpuvol_open(&v, root, 8) == 0);
  assert(r3d_cpuvol_read_block_status(&v, 0, 0, 0, 0, 16, 16, 16, block));
  assert(r3d_cpuvol_read_block_status(&v, 0, 0, 0, 0, 16, 16, 16, block));
  assert(!r3d_cpuvol_read_block_status(&v, 0, 128, 0, 0, 16, 16, 16, block));
  assert(!r3d_cpuvol_read_block_status(&v, 0, 128, 0, 0, 16, 16, 16, block));
  assert(r3d_cpuvol_read_block_status(&v, 0, 256, 0, 0, 16, 16, 16, block));
  r3d_block_region states[3]={
    {.x=0,.nx=16,.ny=16,.nz=16,},
    {.x=128,.nx=16,.ny=16,.nz=16},
    {.x=256,.nx=16,.ny=16,.nz=16}};
  uint8_t state_data[3][4096];
  for(unsigned i=0;i<3;i++)states[i].out=state_data[i];
  r3d_cpuvol_read_regions(&v,0,states,3);
  assert(states[0].available && !states[1].available && states[2].available);
  for(unsigned i=0;i<3;i++)for(unsigned j=0;j<4096;j++)assert(!state_data[i][j]);
  r3d_cpuvol_close(&v);
  st_rm_tree(root, 2);
  puts("512 native blocks match full decode; sparse CPU cache uses 4 KiB per "
       "touched block");
  return 0;
}
