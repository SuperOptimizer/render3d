/* Render3d VCS1 shard: native volume-compressor payloads followed by a
 * little-endian index (offset:u64, size:u32, crc32c:u32) and 32-byte footer.
 * Distinct from both c5d shards and upstream zarr sharding_indexed. */
#ifndef R3D_CODEC_SHARD_H
#define R3D_CODEC_SHARD_H
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#define VOLCOMP_SHARD_MAGIC 0x31534356u
#define VOLCOMP_SHARD_OFFSET_MISSING UINT64_MAX
#define VOLCOMP_SHARD_OFFSET_ZERO (UINT64_MAX - UINT64_C(1))
typedef struct volcomp_shard_footer {
  uint32_t magic, version, shard_dim, brick_dim, nbricks, lod_level;
  float q;
  uint32_t reserved;
} volcomp_shard_footer;
typedef struct volcomp_shard_writer volcomp_shard_writer;
typedef struct volcomp_shard_reader {
  const uint8_t *map;
  size_t map_n;
  volcomp_shard_footer foot;
  size_t index_off;
  _Atomic uint8_t
      *verified; /* immutable mapped payload CRCs, shared by block requests */
} volcomp_shard_reader;
volcomp_shard_writer *volcomp_shard_create(const char *path, uint32_t shard_dim,
                                           uint32_t brick_dim, uint32_t lod,
                                           float q);
int volcomp_shard_put(volcomp_shard_writer *w, uint32_t i, const uint8_t *data,
                      size_t n);
int volcomp_shard_put_zero(volcomp_shard_writer *w, uint32_t i);
int volcomp_shard_close(volcomp_shard_writer *w);
int volcomp_shard_open(const char *path, volcomp_shard_reader *r);
const uint8_t *volcomp_shard_brick(const volcomp_shard_reader *r, uint32_t i,
                                   size_t *n);
int volcomp_shard_brick_is_zero(const volcomp_shard_reader *r, uint32_t i);
void volcomp_shard_close_reader(volcomp_shard_reader *r);
uint32_t volcomp_crc32c(const void *data, size_t n);
#endif
