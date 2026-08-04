// oracle.h — coarse-level air oracle for fast skipping of empty regions.
//
// A level L voxel downsampled 16x lives in level L+4. Because the volume is
// masked (air==0) and downsampling is an average, a coarse voxel is 0 IFF every
// fine voxel under it is 0 — so "coarse voxel == 0" is a SAFE air signal (never
// a false skip; verified against real data). We use it two ways per shard:
//
//   - whole-shard: a shard's 1024^3 region maps to a 64^3 region of the oracle
//     level; if that is entirely zero the whole shard is air.
//   - per-source-chunk: a source 128^3 chunk maps to an 8^3 region of oracle
//     voxels; if that is zero we can skip fetching that chunk.
//
// The oracle only exists for L0 (oracle=L4) and L1 (oracle=L5); deeper levels
// have no 16x-coarser level in the 0..5 pyramid and are fetched directly.

#ifndef DCT3D_EXPORT_ORACLE_H
#define DCT3D_EXPORT_ORACLE_H

#include <stdint.h>

#include "fetch.h"
#include "shard.h"
#include "vendor/libs3.h"

#define ORACLE_SCALE 16              // level L voxel -> level L+4 voxel
#define ORACLE_EDGE (SHARD_VOX / ORACLE_SCALE)  // 64: oracle voxels per shard axis

// A materialized oracle region for one shard: dense ORACLE_EDGE^3 u8 (z-major),
// the coarse voxels covering this shard's region. `valid` is 0 when no oracle
// level applies (caller then fetches everything).
typedef struct oracle_region {
    uint8_t vox[(size_t)ORACLE_EDGE * ORACLE_EDGE * ORACLE_EDGE];
    int valid;
    int all_zero;   // whole shard is air
} oracle_region;

// Oracle level number for a given export level, or -1 if none applies.
static inline int oracle_level_for(int level) {
    return (level == 0 || level == 1) ? level + 4 : -1;
}

// A whole oracle level materialized to a local dense file and mmap'd read-only.
// For big 2.4um volumes the L4 level is 20-30 GB — too big for RAM, but it fits
// on disk and the OS page-caches the hot pages, so per-shard lookups are just
// mapped-memory reads with no S3 traffic and near-zero resident RAM.
typedef struct {
    const uint8_t *base;   // mmap'd dense ZYX u8, shape[0]*shape[1]*shape[2]
    int64_t shape[3];
    int level;
    size_t bytes;
    int fd;
    int valid;
} oracle_map;

// Download `oracle_level` of the volume at `vol_base` into `path` (dense ZYX u8;
// reused if already complete) and mmap it. `client` fetches the 128^3 source
// chunks in parallel; missing chunks (404) are zero. Returns 0 on success (sets
// out->valid), -1 on failure (out->valid=0; caller falls back to S3 per-shard).
int oracle_map_build(s3_client *client, const char *vol_base, int oracle_level,
                     const int64_t oracle_shape[3], const char *path,
                     oracle_map *out);

void oracle_map_close(oracle_map *m);

// Fill `out` (a 64^3 region) for the shard at export-level origin (oz,oy,ox)
// straight from the mmap'd oracle — no S3. Always succeeds (out->valid=1).
void oracle_region_from_map(const oracle_map *m, int64_t oz, int64_t oy, int64_t ox,
                            oracle_region *out);

// Fill `out` with the oracle voxels covering the shard at (oz,oy,ox) voxels of
// the EXPORT level. `oracle_base`/`oracle_shape` describe the oracle level.
// Returns 0 on success (out->valid set appropriately), -1 on hard fetch error.
int oracle_fetch(s3_client *client, const char *oracle_base, int oracle_level,
                 const int64_t oracle_shape[3], int64_t oz, int64_t oy, int64_t ox,
                 oracle_region *out);

// True if the source 128^3 chunk whose EXPORT-level origin is (cz*128,cy*128,
// cx*128) is fully air per the oracle. (lz,ly,lx) is that chunk's origin
// relative to the shard, in export-level voxels. Requires out->valid.
int oracle_chunk_is_air(const oracle_region *o, int64_t lz, int64_t ly, int64_t lx);

#endif  // DCT3D_EXPORT_ORACLE_H
