// srcfetch.c — see srcfetch.h.

#include "srcfetch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decode.h"
#include "downsample.h"
#include "localio.h"
#include "sftpget.h"
#include "../export/shard.h"
#include "../export/upload.h"  // sftp_size

// Every inner chunk stored as the missing/air sentinel: pure index+crc, no
// payload bytes. Any real (non-air) content makes the blob strictly bigger.
#define AIR_SHARD_BYTES ((size_t)NINNER * 16 + 4)

// One of the 8 parent (level N-1) shards. `blob` is the raw compressed shard
// bytes (kept resident for the whole output-shard build — cheap, typically
// tens of MiB, nowhere near the ~1 GiB a dense decode would cost) or NULL if
// this octant is entirely absent/air (contributes zero).
typedef struct {
    uint8_t *blob;
    size_t len;
    int present;  // 0 -> absent/air, decode calls are skipped, zero-fill assumed
} octant_blob;

static int fetch_octant_blob(const src_ctx *ctx, int64_t csz, int64_t csy,
                             int64_t csx, octant_blob *ob) {
    ob->blob = NULL; ob->len = 0; ob->present = 0;
    if (csz >= ctx->src_nshard[0] || csy >= ctx->src_nshard[1] ||
        csx >= ctx->src_nshard[2])
        return 0;  // past extent -> stays absent/zero

    // Local-first: a shard on the local mirror is authoritative (we wrote it
    // ourselves at level N-1) — same air-size short-circuit as remote.
    if (ctx->local_root) {
        char lpath[2048];
        snprintf(lpath, sizeof(lpath), "%s/%s/%s/%d/c/%lld/%lld/%lld",
                 ctx->local_root, ctx->scroll, ctx->vol, ctx->src_level,
                 (long long)csz, (long long)csy, (long long)csx);
        long lsz = local_size(lpath);
        if (lsz >= 0) {
            if ((size_t)lsz <= AIR_SHARD_BYTES) return 0;
            if (local_read(lpath, &ob->blob, &ob->len) != 0) {
                fprintf(stderr, "srcfetch: local read failed %s\n", lpath);
                return -1;
            }
            ob->present = 1;
            return 0;
        }
        // fall through to SFTP (e.g. L0 lives only on the server)
    }

    if (!ctx->sftp.host) return 0;  // no remote configured -> treat as air

    char remote[1536];
    snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/c/%lld/%lld/%lld",
             ctx->remote_root, ctx->scroll, ctx->vol, ctx->src_level,
             (long long)csz, (long long)csy, (long long)csx);

    // Stat first: a shard exactly AIR_SHARD_BYTES long is all-air (every
    // inner chunk is the missing sentinel) — skip the download+decode
    // entirely and leave this octant zero.
    long sz_bytes = sftp_stat_size(&ctx->sftp, remote);
    if (sz_bytes >= 0 && (size_t)sz_bytes <= AIR_SHARD_BYTES) return 0;

    if (sftp_download(&ctx->sftp, remote, &ob->blob, &ob->len) != 0) {
        // Shard file itself absent == fully air for this whole 1024^3 region
        // (the export tool always uploads a shard even for all-air content,
        // so a true 404 here means past the real extent — leave zero, not an
        // error).
        return 0;
    }
    ob->present = 1;
    return 0;
}

static void free_octant_blob(octant_blob *ob) {
    free(ob->blob);
    ob->blob = NULL;
}

// Decode Z-rows [iz0, iz1) (in units of INNER-voxel inner-chunk rows) of one
// octant's blob into `scratch` (dense (iz1-iz0)*INNER x SHARD_VOX x
// SHARD_VOX, z-major, zeroed first). A NULL/absent blob leaves `scratch`
// zeroed (air). Returns 0 on success, -1 on a corrupt blob.
static int decode_octant_zrows(const octant_blob *ob, int iz0, int iz1,
                               uint8_t *scratch) {
    size_t n = (size_t)(iz1 - iz0) * INNER * SHARD_VOX * SHARD_VOX;
    memset(scratch, 0, n);
    if (!ob->present) return 0;
    return shard_decode_u8_zrows(ob->blob, ob->len, iz0, iz1, scratch);
}

int srcfetch_region(const src_ctx *ctx, int64_t sz, int64_t sy, int64_t sx,
                    uint8_t *out) {
    const size_t S = SHARD_VOX;
    const size_t SLAB = DOWNSAMPLE_SLAB_OUT;  // output voxels per slab
    const size_t IROWS_PER_SLAB = (2 * SLAB) / INNER;  // inner-chunk Z-rows/slab per octant

    // Fetch all up-to-8 parent blobs once, up front — small (compressed), so
    // keeping them all resident for the whole shard build is cheap and lets
    // every Z-slab reuse them without re-downloading. Sequential within a
    // region is fine: cross-region overlap from the dl-thread pool (each
    // thread holding a persistent SFTP connection, see sftpget.c) is what
    // saturates the pipe.
    octant_blob obs[2][2][2];  // [dz][dy][dx]
    int rc = 0;
    for (int dz = 0; dz < 2 && rc == 0; ++dz)
        for (int dy = 0; dy < 2 && rc == 0; ++dy)
            for (int dx = 0; dx < 2 && rc == 0; ++dx) {
                if (fetch_octant_blob(ctx, 2 * sz + dz, 2 * sy + dy, 2 * sx + dx,
                                      &obs[dz][dy][dx]) != 0)
                    rc = -1;
            }
    if (rc != 0) goto cleanup;

    // Scratch: one reusable decode buffer per (dy,dx) octant of the CURRENT
    // dz half (4 buffers), each sized with up to 1 halo row on each Z side
    // for an exact central difference across slab boundaries within the same
    // octant. (IROWS_PER_SLAB + 2) inner-chunk rows worst case.
    const size_t max_rows = IROWS_PER_SLAB + 2;
    uint8_t *scratch[2][2];
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            scratch[dy][dx] = (uint8_t *)malloc(max_rows * INNER * S * S);
            if (!scratch[dy][dx]) { rc = -1; goto free_scratch; }
        }

    {
        downsample_acc acc;
        acc.out_edge = S;
        acc.slab_out = SLAB;
        acc.wsum = (float *)malloc(S * S * SLAB * sizeof(float));
        acc.vsum = (float *)malloc(S * S * SLAB * sizeof(float));
        if (!acc.wsum || !acc.vsum) {
            free(acc.wsum); free(acc.vsum);
            rc = -1;
            goto free_scratch;
        }

        // Each octant is its OWN full SHARD_VOX^3 (level N-1) shard, i.e. its
        // own full [0, GRID) inner-chunk-row range — dz only selected WHICH
        // of the 2 Z-adjacent shards we fetched, it is not an offset within
        // one shard. Walking one octant's full GRID rows downsamples to the
        // full S/2 = 512 output rows that make up that dz-half of the output
        // shard (placed at output offset dz*(S/2)).
        for (int dz = 0; dz < 2 && rc == 0; ++dz) {
            for (int slab = 0; slab < GRID / (int)IROWS_PER_SLAB && rc == 0; ++slab) {
                int iz0 = slab * (int)IROWS_PER_SLAB;
                int iz1 = iz0 + (int)IROWS_PER_SLAB;

                downsample_acc_reset(&acc);

                for (int dy = 0; dy < 2 && rc == 0; ++dy) {
                    for (int dx = 0; dx < 2 && rc == 0; ++dx) {
                        // Extend by 1 inner-chunk row on each side for a true
                        // Z halo where available (i.e. not at the very top of
                        // dz=0 or the very bottom of dz=1 within THIS octant
                        // — those are the real outer edge of the octant and
                        // fall back to clamping, matching old behavior).
                        int halo_lo = (iz0 > 0) ? 1 : 0;
                        int halo_hi = (iz1 < GRID) ? 1 : 0;
                        int diz0 = iz0 - halo_lo, diz1 = iz1 + halo_hi;

                        if (decode_octant_zrows(&obs[dz][dy][dx], diz0, diz1,
                                                scratch[dy][dx]) != 0) {
                            fprintf(stderr, "srcfetch: corrupt octant dz=%d dy=%d dx=%d\n",
                                    dz, dy, dx);
                            rc = -1;
                            break;
                        }

                        // oct_zdim is the decoded buffer's actual Z extent;
                        // z0 (== halo_lo*INNER) locates the slab's real row 0
                        // within it. When halo_lo/halo_hi is 0 (true outer
                        // edge of the level-(N-1) volume, no neighbor row to
                        // borrow), grad_mag's own clamp-to-[0,oct_zdim)
                        // behavior reproduces the old design's edge clamp —
                        // no special-casing needed here.
                        size_t oct_zdim = (size_t)(diz1 - diz0) * INNER;
                        size_t z0 = (size_t)halo_lo * INNER;

                        downsample_fold_octant_slab(&acc, scratch[dy][dx], oct_zdim,
                                                    z0, dy, dx);
                    }
                }
                if (rc != 0) break;

                size_t out_z0 = (size_t)dz * (S / 2) + (size_t)slab * SLAB;
                downsample_finalize_slab(&acc, out + out_z0 * S * S);
            }
        }

        free(acc.wsum);
        free(acc.vsum);
    }

free_scratch:
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
            free(scratch[dy][dx]);

cleanup:
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
                free_octant_blob(&obs[dz][dy][dx]);
    return rc;
}
