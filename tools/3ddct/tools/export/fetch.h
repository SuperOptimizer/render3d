// fetch.h — assemble a dense SHARD_VOX^3 u8 region from a source OME-zarr level.
//
// The source is a raw (uncompressed) OME-zarr v2 level: 128^3 u8 chunks at
// <base>/<L>/<cz>/<cy>/<cx> (dimension_separator "/"). We fetch the up-to
// 8^3=512 source chunks covering one shard region concurrently via libs3's
// batched ranged GET, and scatter them into a dense SHARD_VOX^3 buffer.
//
// Zero-fill semantics: a chunk that is 404 (omitted == air, zarr convention) or
// lies past the volume's real extent is left as zeros. A present-but-wrong-size
// chunk is a hard error (never silently demoted to air).

#ifndef DCT3D_EXPORT_FETCH_H
#define DCT3D_EXPORT_FETCH_H

#include <stddef.h>
#include <stdint.h>

#include "shard.h"
#include "vendor/libs3.h"

typedef struct {
    char base[1024];       // e.g. https://<bucket>.s3.amazonaws.com/<scroll>/volumes/<vol>.zarr
    int level;             // pyramid level
    int64_t shape[3];      // real level shape (ZYX), unpadded
    int src_chunk;         // source chunk edge (128)
} src_level;

struct oracle_region;  // fwd decl (oracle.h); NULL when no oracle applies

// Throttle telemetry (defined in fetch.c): bump on each throttled/short read.
void fetch_note_throttle(void);
long fetch_throttle_count(void);

// Fill `vol` (SHARD_VOX^3, z-major, caller-allocated, pre-zeroed not required)
// with the region whose origin is (oz,oy,ox) voxels into the level. Regions past
// `shape` are zero-padded. `client` is a shared thread-safe s3_client. When
// `oracle` is non-NULL and valid, source 128^3 chunks it marks as air are not
// fetched (left zero). Returns 0 on success, -1 on a hard fetch error.
int fetch_shard_region(s3_client *client, const src_level *lvl,
                       int64_t oz, int64_t oy, int64_t ox, uint8_t *vol,
                       const struct oracle_region *oracle);

#endif  // DCT3D_EXPORT_FETCH_H
