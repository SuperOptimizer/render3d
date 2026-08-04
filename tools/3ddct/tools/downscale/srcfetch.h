// srcfetch.h — build one output shard by fetching (as compressed blobs) the
// up-to 2x2x2=8 level-(N-1) shards it covers, then decoding+downsampling them
// in Z-SLABS instead of ever materializing a dense 2*SHARD_VOX^3 cube (see
// downsample.h for the full rationale: peak residency drops from ~8 GiB to
// ~1.5-2 GiB per in-flight output shard).
//
// A level-N shard at shard-coord (sz,sy,sx) covers voxels
// [sz*SHARD_VOX, (sz+1)*SHARD_VOX) x ... of level N, which is exactly
// [2*sz*SHARD_VOX, 2*(sz+1)*SHARD_VOX) x ... of level N-1 (each level halves
// resolution, same shard edge in voxels) — i.e. the 8 level-(N-1) shards at
// coords (2*sz+{0,1}, 2*sy+{0,1}, 2*sx+{0,1}).

#ifndef DCT3D_DOWNSCALE_SRCFETCH_H
#define DCT3D_DOWNSCALE_SRCFETCH_H

#include <stddef.h>
#include <stdint.h>

#include "../export/upload.h"  // sftp_target

typedef struct {
    sftp_target sftp;
    const char *remote_root, *scroll, *vol;
    const char *local_root;  // NULL, or local mirror of the exports tree
                             // (<local_root>/<scroll>/<vol>/<level>/...);
                             // sources are read local-first, SFTP fallback
    int src_level;          // level N-1
    int64_t src_nshard[3];  // padded shard-grid extent of level N-1 (for bounds)
} src_ctx;

// Fetch, decode-and-downsample, and write the fully assembled output shard
// (dense SHARD_VOX^3 u8, z-major) into `out` (caller-allocated, ~1 GiB,
// reused across shards). Internally: downloads up to 8 compressed
// level-(N-1) blobs (small — tens/low-hundreds of MiB each, not the dense
// decode), then walks output Z-slabs, decoding only the Z-rows (+1-voxel
// halo) of each relevant blob needed for that slab into small reusable
// scratch buffers, folding into a slab-sized accumulator, and finalizing
// straight into `out`. Missing (past-extent or air-sentinel) source shards
// contribute as an all-zero region (matching the old dense-cube behavior:
// absent == zero, not "no contribution").
// Returns 0 on success, -1 on a hard fetch/decode error (corrupt shard).
int srcfetch_region(const src_ctx *ctx, int64_t sz, int64_t sy, int64_t sx,
                    uint8_t *out);

#endif  // DCT3D_DOWNSCALE_SRCFETCH_H
