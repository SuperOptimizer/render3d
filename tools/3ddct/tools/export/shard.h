// shard.h — build a Zarr v3 sharding_indexed shard of dct3d-encoded 16^3 chunks.
//
// A shard is one SHARD_VOX^3 region (default 1024^3) tiled into 16^3 inner
// chunks, each independently dct3d-encoded. The on-disk/on-wire layout is
// exactly what zarr's `sharding_indexed` codec (index_location=end) expects, so
// the uploaded volumes are readable by stock zarr + the dct3d-zarr plugin:
//
//   [ inner chunk 0 bytes ][ inner chunk 1 bytes ] ... [ index ][ crc32c(4) ]
//
// index = one (u64 offset, u64 nbytes) little-endian pair per inner chunk in
// row-major inner-chunk order; a missing (all-fill) chunk is encoded as the
// sentinel (0xffffffffffffffff, 0xffffffffffffffff). The 4-byte crc32c trailer
// covers the index bytes only (zarr's default index_codecs = [bytes, crc32c]).
//
// The inner chunk order is row-major over the inner grid: for a shard of
// G = SHARD_VOX/16 inner chunks per axis, chunk (iz,iy,ix) is at linear index
// (iz*G + iy)*G + ix.

#ifndef DCT3D_EXPORT_SHARD_H
#define DCT3D_EXPORT_SHARD_H

#include <stddef.h>
#include <stdint.h>

#include "../../dct3d.h"

#define SHARD_VOX 1024                     // shard edge in voxels
#define INNER 16                           // dct3d block edge (== DCT3D_N)
#define GRID (SHARD_VOX / INNER)           // inner chunks per axis (64)
#define NINNER ((size_t)GRID * GRID * GRID)  // 262144

// Encode a dense SHARD_VOX^3 u8 volume (`vol`, z-major: (z*SHARD_VOX+y)*SHARD_VOX+x)
// into a sharding_indexed shard. Writes the shard bytes to `*out` (malloc'd;
// caller frees) and its length to `*out_len`. `quality`/`tau` are the dct3d
// knobs for this level. An all-zero inner chunk is stored as the missing
// sentinel (no bytes, fill_value=0 on read) so air costs nothing.
//
// Returns 0 on success, -1 on allocation failure.
int shard_encode_u8(const uint8_t *vol, float quality, float tau,
                    uint8_t **out, size_t *out_len);

// crc32c (Castagnoli) of `len` bytes — exposed for tests.
uint32_t shard_crc32c(const uint8_t *data, size_t len);

#endif  // DCT3D_EXPORT_SHARD_H
