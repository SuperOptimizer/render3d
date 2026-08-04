/* Local store of zarr-v3 sharded, dct3d-compressed volumes (the user's
 * PHercParis3 export format): 1024^3 shards holding 16^3 inner chunks with a
 * footer index ((offset,nbytes) u64le pairs in z-major chunk order + crc32c;
 * missing = 0xFF..FF). Shards live as files <dir>/<Z>_<Y>_<X>.shard.
 * All decode paths are thread-parallel and read-only. */
#ifndef R3D_SHARDIO_H
#define R3D_SHARDIO_H

#include <stddef.h>
#include <stdint.h>

#define R3D_SHARD_DIM 1024u
#define R3D_SHARD_CHUNK 16u
#define R3D_SHARD_GRID (R3D_SHARD_DIM / R3D_SHARD_CHUNK)                    /* 64 */
#define R3D_SHARD_NCHUNKS (R3D_SHARD_GRID * R3D_SHARD_GRID * R3D_SHARD_GRID) /* 262144 */
#define R3D_SHARD_INDEX_BYTES (R3D_SHARD_NCHUNKS * 16u + 4u)

typedef struct r3d_shard_store {
  char dir[512];
  uint64_t nz, ny, nx; /* full volume dims in voxels */
} r3d_shard_store;

/* One mmap'd shard. */
typedef struct r3d_shard {
  const uint8_t *map;
  size_t n;
  const uint8_t *index; /* R3D_SHARD_INDEX_BYTES at tail */
} r3d_shard;

int r3d_shard_store_init(r3d_shard_store *s, const char *dir, uint64_t nz, uint64_t ny,
                         uint64_t nx);

/* Open/close one shard by shard coords; open fails (-1) if absent. */
int r3d_shard_open(const r3d_shard_store *s, uint32_t sz, uint32_t sy, uint32_t sx,
                   r3d_shard *sh);
void r3d_shard_close(r3d_shard *sh);

/* Blob location of inner chunk (cz,cy,cx) in shard-local 16^3-grid coords.
 * Returns NULL if the chunk is missing (masked region). */
const uint8_t *r3d_shard_chunk_blob(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                                    size_t *nbytes);

/* Decode one inner chunk into out[4096] (z-major 16^3). 0 on success. */
int r3d_shard_chunk_decode(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                           uint8_t out[R3D_SHARD_CHUNK * R3D_SHARD_CHUNK * R3D_SHARD_CHUNK]);

/* Decode an arbitrary voxel region [z0,z0+dz) x [y0,y0+dy) x [x0,x0+dx) into
 * dst (x-fastest, row stride = dx bytes, slice stride = dx*dy). Threaded over
 * chunks (nthreads<=0 -> online cpus). Missing chunks/shards and out-of-range
 * voxels are zero-filled. Returns 0 on success (missing data is not an error). */
int r3d_shard_decode_region(const r3d_shard_store *s, uint64_t z0, uint64_t y0, uint64_t x0,
                            uint32_t dz, uint32_t dy, uint32_t dx, uint8_t *dst, int nthreads);

#endif /* R3D_SHARDIO_H */
