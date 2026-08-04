// fetch.c — see fetch.h.

#include "fetch.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "oracle.h"

// Fleet-wide S3 throttle signal. Every throttled/short chunk bumps this; a
// milestone log lets an operator see when a box is getting rate-limited (and
// the per-chunk backoff below is easing it off automatically).
static atomic_long g_throttle_events = 0;
void fetch_note_throttle(void) {
    long n = atomic_fetch_add(&g_throttle_events, 1) + 1;
    if (n == 1 || n % 100 == 0)
        fprintf(stderr, "THROTTLE: %ld throttled/short S3 reads so far "
                "(backing off per-chunk)\n", n);
}
long fetch_throttle_count(void) { return atomic_load(&g_throttle_events); }

#define SRC_CHUNK 128
#define SRC_CHUNK3 ((size_t)SRC_CHUNK * SRC_CHUNK * SRC_CHUNK)

// A response is "throttled" when S3 pushed back (429/500/503) or delivered a
// truncated body (a real chunk arriving < its full size — seen under heavy
// fleet-wide fan-out). Both are transient: retry the individual chunk with
// backoff instead of hard-failing the whole shard.
static int is_throttle_status(long s) { return s == 429 || s == 500 || s == 503; }

static void backoff_sleep(int attempt) {
    // 0.5s, 1s, 2s, 4s, ... capped at ~8s, so a throttled box eases off S3.
    long ms = 500L << (attempt < 4 ? attempt : 4);
    if (ms > 8000) ms = 8000;
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

// Scatter a fetched 128^3 source chunk into the dense SHARD_VOX^3 vol, clipped to
// the shard bounds (a source chunk straddling the shard edge contributes only
// its in-shard part). (lz,ly,lx) is the chunk origin relative to the shard.
static void scatter_chunk(uint8_t *vol, const uint8_t *chunk,
                          int64_t lz, int64_t ly, int64_t lx) {
    const int64_t S = SHARD_VOX;
    for (int z = 0; z < SRC_CHUNK; ++z) {
        int64_t vz = lz + z;
        if (vz < 0 || vz >= S) continue;
        for (int y = 0; y < SRC_CHUNK; ++y) {
            int64_t vy = ly + y;
            if (vy < 0 || vy >= S) continue;
            // x run: intersect [lx, lx+128) with [0, S)
            int64_t x0 = lx < 0 ? 0 : lx;
            int64_t x1 = (lx + SRC_CHUNK) > S ? S : (lx + SRC_CHUNK);
            if (x1 <= x0) continue;
            const uint8_t *src = chunk + ((size_t)z * SRC_CHUNK + y) * SRC_CHUNK + (x0 - lx);
            uint8_t *dst = vol + (((size_t)vz * S + vy) * S + x0);
            memcpy(dst, src, (size_t)(x1 - x0));
        }
    }
}

int fetch_shard_region(s3_client *client, const src_level *lvl,
                       int64_t oz, int64_t oy, int64_t ox, uint8_t *vol,
                       const struct oracle_region *oracle) {
    memset(vol, 0, (size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX);

    // Source chunk index range covering [o, o+SHARD_VOX) on each axis, clipped to
    // the real level extent (chunks past `shape` don't exist -> stay zero).
    const int64_t *shp = lvl->shape;
    int64_t cz0 = oz / SRC_CHUNK, cz1 = (oz + SHARD_VOX - 1) / SRC_CHUNK;
    int64_t cy0 = oy / SRC_CHUNK, cy1 = (oy + SHARD_VOX - 1) / SRC_CHUNK;
    int64_t cx0 = ox / SRC_CHUNK, cx1 = (ox + SHARD_VOX - 1) / SRC_CHUNK;
    int64_t ncz = (shp[0] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncy = (shp[1] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncx = (shp[2] + SRC_CHUNK - 1) / SRC_CHUNK;

    // Build the list of in-range chunk coords + URLs.
    size_t maxn = (size_t)(cz1 - cz0 + 1) * (cy1 - cy0 + 1) * (cx1 - cx0 + 1);
    typedef struct { int64_t cz, cy, cx; } coord;
    coord *coords = (coord *)malloc(maxn * sizeof(coord));
    char (*urls)[1280] = malloc(maxn * sizeof(*urls));
    uint8_t *bufs = (uint8_t *)malloc(maxn * SRC_CHUNK3);
    s3_range_req *reqs = (s3_range_req *)calloc(maxn, sizeof(s3_range_req));
    s3_response *resp = (s3_response *)calloc(maxn, sizeof(s3_response));
    if (!coords || !urls || !bufs || !reqs || !resp) {
        free(coords); free(urls); free(bufs); free(reqs); free(resp);
        return -1;
    }

    size_t n = 0;
    for (int64_t cz = cz0; cz <= cz1; ++cz) {
        if (cz < 0 || cz >= ncz) continue;
        for (int64_t cy = cy0; cy <= cy1; ++cy) {
            if (cy < 0 || cy >= ncy) continue;
            for (int64_t cx = cx0; cx <= cx1; ++cx) {
                if (cx < 0 || cx >= ncx) continue;
                // Oracle air-skip: if the coarse level says this 128^3 chunk is
                // all air, don't fetch it (vol stays zero there).
                if (oracle && oracle_chunk_is_air(
                        oracle, cz * SRC_CHUNK - oz, cy * SRC_CHUNK - oy,
                        cx * SRC_CHUNK - ox))
                    continue;
                coords[n].cz = cz; coords[n].cy = cy; coords[n].cx = cx;
                snprintf(urls[n], sizeof(urls[n]), "%s/%d/%lld/%lld/%lld",
                         lvl->base, lvl->level,
                         (long long)cz, (long long)cy, (long long)cx);
                reqs[n].url = urls[n];
                reqs[n].offset = 0;
                reqs[n].length = SRC_CHUNK3;      // whole raw chunk
                reqs[n].dst = bufs + n * SRC_CHUNK3;  // zero-copy into our buffer
                n++;
            }
        }
    }

    int rc = 0;
    if (n > 0) {
        // Batched concurrent GET. Transport-level failures -> S3_OK still, per-req
        // status inspected below.
        s3_status st = s3_get_batch(client, reqs, n, 0, resp);
        int batch_failed = (st != S3_OK && st != S3_ERR_HTTP);

        // A chunk needs an individual retry if the batch failed wholesale, or it
        // came back throttled (429/500/503) or short (truncated body). 404 = air.
        // Persistent NoSuchKey and other 4xx are terminal (not retried).
        for (size_t i = 0; i < n && rc == 0; ++i) {
            long status = batch_failed ? 0 : resp[i].status;
            int need_retry = batch_failed || is_throttle_status(status) ||
                             (status >= 200 && status < 300 && resp[i].body_len != SRC_CHUNK3);

            if (!need_retry) {
                if (status == 404) continue;  // air
                if (status < 200 || status >= 300) {
                    fprintf(stderr, "fetch: %s -> HTTP %ld (not 404) — hard error\n",
                            urls[i], status);
                    rc = -1;
                    break;
                }
                int64_t lz = coords[i].cz * SRC_CHUNK - oz;
                int64_t ly = coords[i].cy * SRC_CHUNK - oy;
                int64_t lx = coords[i].cx * SRC_CHUNK - ox;
                scatter_chunk(vol, bufs + i * SRC_CHUNK3, lz, ly, lx);
                continue;
            }

            // Individual retry with exponential backoff — eases off S3 when it's
            // pushing back, and recovers truncated batched reads.
            fetch_note_throttle();
            int ok = 0;
            for (int attempt = 0; attempt < 6; ++attempt) {
                backoff_sleep(attempt);
                s3_response r = {0};
                s3_status rs = s3_get_range_into(client, urls[i], 0, SRC_CHUNK3,
                                                 bufs + i * SRC_CHUNK3, &r);
                long rstat = r.status;
                size_t got = r.body_len;
                s3_response_free(&r);
                if (rstat == 404) { ok = 1; break; }              // air
                if ((rs == S3_OK || rs == S3_ERR_HTTP) && rstat >= 200 &&
                    rstat < 300 && got == SRC_CHUNK3) {
                    int64_t lz = coords[i].cz * SRC_CHUNK - oz;
                    int64_t ly = coords[i].cy * SRC_CHUNK - oy;
                    int64_t lx = coords[i].cx * SRC_CHUNK - ox;
                    scatter_chunk(vol, bufs + i * SRC_CHUNK3, lz, ly, lx);
                    ok = 1;
                    break;
                }
                if (!batch_failed && !is_throttle_status(rstat) &&
                    !(rstat >= 200 && rstat < 300)) {
                    // A genuine terminal error (e.g. 403/404-body) — stop retrying.
                    fprintf(stderr, "fetch: %s -> HTTP %ld (terminal) — hard error\n",
                            urls[i], rstat);
                    break;
                }
                // else throttled/short — loop and back off further.
            }
            if (!ok) {
                fprintf(stderr, "fetch: %s -> unrecovered after retries — hard error\n",
                        urls[i]);
                rc = -1;
                break;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) s3_response_free(&resp[i]);
    free(coords); free(urls); free(bufs); free(reqs); free(resp);
    return rc;
}
