// oracle.c — see oracle.h.

#include "oracle.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SRC_CHUNK 128
#define SRC_CHUNK3 ((size_t)SRC_CHUNK * SRC_CHUNK * SRC_CHUNK)

static int is_throttle_status(long s) { return s == 429 || s == 500 || s == 503; }
static void backoff_sleep(int attempt) {
    long ms = 500L << (attempt < 4 ? attempt : 4);
    if (ms > 8000) ms = 8000;
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
// Retry one chunk into `dst` (SRC_CHUNK3) with backoff. Returns 1 on a full
// read or 404-air (dst zeroed), 0 on terminal failure.
static int retry_chunk(s3_client *c, const char *url, uint8_t *dst) {
    for (int a = 0; a < 6; ++a) {
        backoff_sleep(a);
        s3_response r = {0};
        s3_status rs = s3_get_range_into(c, url, 0, SRC_CHUNK3, dst, &r);
        long st = r.status; size_t got = r.body_len;
        s3_response_free(&r);
        if (st == 404) { memset(dst, 0, SRC_CHUNK3); return 1; }
        if ((rs == S3_OK || rs == S3_ERR_HTTP) && st >= 200 && st < 300 &&
            got == SRC_CHUNK3)
            return 1;
        if (!is_throttle_status(st) && !(st >= 200 && st < 300)) return 0;  // terminal
    }
    return 0;
}

// The oracle region for a shard is ORACLE_EDGE=64 voxels/axis. At oracle level
// those 64 voxels span oracle-voxel range [ooz, ooz+64) where ooz = oz/16.
// They come from source 128^3 oracle chunks, so a 64^3 region touches at most
// 1-2 oracle chunks per axis. We fetch those and scatter into out->vox.

static void scatter(uint8_t *dst, const uint8_t *chunk, int64_t lz, int64_t ly, int64_t lx) {
    const int64_t E = ORACLE_EDGE;
    for (int z = 0; z < SRC_CHUNK; ++z) {
        int64_t vz = lz + z; if (vz < 0 || vz >= E) continue;
        for (int y = 0; y < SRC_CHUNK; ++y) {
            int64_t vy = ly + y; if (vy < 0 || vy >= E) continue;
            int64_t x0 = lx < 0 ? 0 : lx, x1 = (lx + SRC_CHUNK) > E ? E : (lx + SRC_CHUNK);
            if (x1 <= x0) continue;
            memcpy(dst + ((size_t)vz * E + vy) * E + x0,
                   chunk + ((size_t)z * SRC_CHUNK + y) * SRC_CHUNK + (x0 - lx),
                   (size_t)(x1 - x0));
        }
    }
}

int oracle_fetch(s3_client *client, const char *oracle_base, int oracle_level,
                 const int64_t oshape[3], int64_t oz, int64_t oy, int64_t ox,
                 oracle_region *out) {
    memset(out, 0, sizeof(*out));
    if (oracle_level < 0) { out->valid = 0; return 0; }

    // Oracle-voxel origin of this shard's region.
    int64_t ooz = oz / ORACLE_SCALE, ooy = oy / ORACLE_SCALE, oox = ox / ORACLE_SCALE;
    const int64_t E = ORACLE_EDGE;

    int64_t cz0 = ooz / SRC_CHUNK, cz1 = (ooz + E - 1) / SRC_CHUNK;
    int64_t cy0 = ooy / SRC_CHUNK, cy1 = (ooy + E - 1) / SRC_CHUNK;
    int64_t cx0 = oox / SRC_CHUNK, cx1 = (oox + E - 1) / SRC_CHUNK;
    int64_t ncz = (oshape[0] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncy = (oshape[1] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncx = (oshape[2] + SRC_CHUNK - 1) / SRC_CHUNK;

    size_t maxn = (size_t)(cz1 - cz0 + 1) * (cy1 - cy0 + 1) * (cx1 - cx0 + 1);
    typedef struct { int64_t cz, cy, cx; } coord;
    coord *coords = malloc(maxn * sizeof(coord));
    char (*urls)[1280] = malloc(maxn * sizeof(*urls));
    uint8_t *bufs = malloc(maxn * SRC_CHUNK3);
    s3_range_req *reqs = calloc(maxn, sizeof(s3_range_req));
    s3_response *resp = calloc(maxn, sizeof(s3_response));
    if (!coords || !urls || !bufs || !reqs || !resp) {
        free(coords); free(urls); free(bufs); free(reqs); free(resp); return -1;
    }

    size_t n = 0;
    for (int64_t cz = cz0; cz <= cz1; ++cz) {
        if (cz < 0 || cz >= ncz) continue;
        for (int64_t cy = cy0; cy <= cy1; ++cy) {
            if (cy < 0 || cy >= ncy) continue;
            for (int64_t cx = cx0; cx <= cx1; ++cx) {
                if (cx < 0 || cx >= ncx) continue;
                coords[n] = (coord){cz, cy, cx};
                snprintf(urls[n], sizeof(urls[n]), "%s/%d/%lld/%lld/%lld",
                         oracle_base, oracle_level,
                         (long long)cz, (long long)cy, (long long)cx);
                reqs[n] = (s3_range_req){urls[n], 0, SRC_CHUNK3, bufs + n * SRC_CHUNK3};
                n++;
            }
        }
    }

    int rc = 0;
    if (n > 0) {
        s3_status st = s3_get_batch(client, reqs, n, 0, resp);
        int batch_failed = (st != S3_OK && st != S3_ERR_HTTP);
        for (size_t i = 0; i < n && rc == 0; ++i) {
            long s = batch_failed ? 0 : resp[i].status;
            int short_or_throttled = batch_failed || is_throttle_status(s) ||
                                     (s >= 200 && s < 300 && resp[i].body_len != SRC_CHUNK3);
            if (short_or_throttled) {
                fetch_note_throttle();
                if (!retry_chunk(client, urls[i], bufs + i * SRC_CHUNK3)) {
                    fprintf(stderr, "oracle: %s -> unrecovered after retries\n", urls[i]);
                    rc = -1; break;
                }
            } else {
                if (s == 404) continue;  // air
                if (s < 200 || s >= 300) {
                    fprintf(stderr, "oracle: %s -> HTTP %ld (hard error)\n", urls[i], s);
                    rc = -1; break;
                }
            }
            scatter(out->vox, bufs + i * SRC_CHUNK3,
                    coords[i].cz * SRC_CHUNK - ooz,
                    coords[i].cy * SRC_CHUNK - ooy,
                    coords[i].cx * SRC_CHUNK - oox);
        }
    }
    for (size_t i = 0; i < n; ++i) s3_response_free(&resp[i]);
    free(coords); free(urls); free(bufs); free(reqs); free(resp);

    if (rc != 0) return -1;
    out->valid = 1;
    out->all_zero = 1;
    for (size_t i = 0; i < (size_t)E * E * E; ++i)
        if (out->vox[i]) { out->all_zero = 0; break; }
    return 0;
}

int oracle_chunk_is_air(const oracle_region *o, int64_t lz, int64_t ly, int64_t lx) {
    if (!o->valid) return 0;
    // A source 128^3 chunk at export-level offset (lz,ly,lx) into the shard maps
    // to oracle voxels [lz/16, lz/16 + 8) etc (128/16 = 8).
    const int64_t E = ORACLE_EDGE;
    int64_t vz0 = lz / ORACLE_SCALE, vy0 = ly / ORACLE_SCALE, vx0 = lx / ORACLE_SCALE;
    for (int64_t z = vz0; z < vz0 + SRC_CHUNK / ORACLE_SCALE; ++z) {
        if (z < 0 || z >= E) continue;
        for (int64_t y = vy0; y < vy0 + SRC_CHUNK / ORACLE_SCALE; ++y) {
            if (y < 0 || y >= E) continue;
            for (int64_t x = vx0; x < vx0 + SRC_CHUNK / ORACLE_SCALE; ++x) {
                if (x < 0 || x >= E) continue;
                if (o->vox[((size_t)z * E + y) * E + x]) return 0;
            }
        }
    }
    return 1;
}

// ---- mmap-backed whole-level oracle ----------------------------------------

// Scatter a fetched 128^3 chunk into the dense L4 file buffer at (cz,cy,cx).
static void scatter_dense(uint8_t *dense, const int64_t shp[3],
                          int64_t cz, int64_t cy, int64_t cx, const uint8_t *chunk) {
    for (int z = 0; z < SRC_CHUNK; ++z) {
        int64_t vz = cz * SRC_CHUNK + z; if (vz >= shp[0]) break;
        for (int y = 0; y < SRC_CHUNK; ++y) {
            int64_t vy = cy * SRC_CHUNK + y; if (vy >= shp[1]) break;
            int64_t vx0 = cx * SRC_CHUNK;
            int64_t w = SRC_CHUNK; if (vx0 + w > shp[2]) w = shp[2] - vx0;
            if (w <= 0) continue;
            memcpy(dense + ((size_t)vz * shp[1] + vy) * shp[2] + vx0,
                   chunk + ((size_t)z * SRC_CHUNK + y) * SRC_CHUNK, (size_t)w);
        }
    }
}

int oracle_map_build(s3_client *client, const char *vol_base, int oracle_level,
                     const int64_t shp[3], const char *path, oracle_map *out) {
    memset(out, 0, sizeof(*out));
    const size_t bytes = (size_t)shp[0] * shp[1] * shp[2];

    // Reuse an existing complete file (a `.done` marker guards partial downloads).
    char done[2048];
    snprintf(done, sizeof(done), "%s.done", path);
    struct stat st;
    int have = (stat(path, &st) == 0 && (size_t)st.st_size == bytes &&
                stat(done, &st) == 0);

    if (!have) {
        // Fresh download written straight into a disk-backed, zero-filled file
        // mmap'd WRITABLE — the page cache holds only the hot pages, so we never
        // hold the whole 20-30 GB in RAM. Write to a temp path, rename on success.
        char tmp[2048];
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        int wfd = open(tmp, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (wfd < 0) { fprintf(stderr, "oracle_map: open %s failed\n", tmp); return -1; }
        if (ftruncate(wfd, (off_t)bytes) != 0) {
            fprintf(stderr, "oracle_map: ftruncate %zu failed\n", bytes);
            close(wfd); return -1;
        }
        uint8_t *dense = (uint8_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                         MAP_SHARED, wfd, 0);
        if (dense == MAP_FAILED) {
            fprintf(stderr, "oracle_map: mmap(write) failed\n"); close(wfd); return -1;
        }
        int64_t ncz = (shp[0] + SRC_CHUNK - 1) / SRC_CHUNK;
        int64_t ncy = (shp[1] + SRC_CHUNK - 1) / SRC_CHUNK;
        int64_t ncx = (shp[2] + SRC_CHUNK - 1) / SRC_CHUNK;
        size_t total = (size_t)ncz * ncy * ncx, got = 0, air = 0;
        fprintf(stderr, "oracle_map: downloading L%d [%lld,%lld,%lld] = %.1f GB "
                "(%zu chunks) -> %s\n", oracle_level, (long long)shp[0],
                (long long)shp[1], (long long)shp[2], bytes / 1e9, total, path);

        // Fetch in batches to bound the URL/response arrays.
        const size_t BATCH = 256;
        char (*urls)[1280] = malloc(BATCH * sizeof(*urls));
        uint8_t *bufs = malloc(BATCH * SRC_CHUNK3);
        s3_range_req *reqs = calloc(BATCH, sizeof(s3_range_req));
        s3_response *resp = calloc(BATCH, sizeof(s3_response));
        int64_t coords[256][3];
        if (!urls || !bufs || !reqs || !resp) { free(dense); return -1; }

        int rc = 0;
        for (int64_t cz = 0; cz < ncz && rc == 0; ++cz)
            for (int64_t cy = 0; cy < ncy && rc == 0; ++cy)
                for (int64_t cx0 = 0; cx0 < ncx && rc == 0; cx0 += BATCH) {
                    size_t n = 0;
                    for (int64_t cx = cx0; cx < ncx && n < BATCH; ++cx, ++n) {
                        coords[n][0] = cz; coords[n][1] = cy; coords[n][2] = cx;
                        snprintf(urls[n], sizeof(urls[n]), "%s/%d/%lld/%lld/%lld",
                                 vol_base, oracle_level, (long long)cz, (long long)cy,
                                 (long long)cx);
                        reqs[n] = (s3_range_req){urls[n], 0, SRC_CHUNK3, bufs + n * SRC_CHUNK3};
                    }
                    s3_status s = s3_get_batch(client, reqs, n, 0, resp);
                    int batch_failed = (s != S3_OK && s != S3_ERR_HTTP);
                    for (size_t i = 0; i < n; ++i) {
                        long code = batch_failed ? 0 : resp[i].status;
                        int retry = batch_failed || is_throttle_status(code) ||
                                    (code >= 200 && code < 300 &&
                                     resp[i].body_len != SRC_CHUNK3);
                        if (code == 404 && !retry) { air++; }
                        else if (!retry && code >= 200 && code < 300) {
                            scatter_dense(dense, shp, coords[i][0], coords[i][1],
                                          coords[i][2], bufs + i * SRC_CHUNK3);
                        } else if (retry) {
                            // Throttled/short during a 20-30 GB L4 download: retry
                            // the chunk with backoff rather than aborting the map.
                            fetch_note_throttle();
                            uint8_t *tmp = malloc(SRC_CHUNK3);
                            if (tmp && retry_chunk(client, urls[i], tmp)) {
                                int allzero = 1;
                                for (size_t b = 0; b < SRC_CHUNK3; ++b)
                                    if (tmp[b]) { allzero = 0; break; }
                                if (allzero) air++;
                                else scatter_dense(dense, shp, coords[i][0],
                                                   coords[i][1], coords[i][2], tmp);
                            } else {
                                fprintf(stderr, "oracle_map: %s -> unrecovered\n", urls[i]);
                                rc = -1;
                            }
                            free(tmp);
                        } else {
                            fprintf(stderr, "oracle_map: %s -> HTTP %ld / %zu B (hard)\n",
                                    urls[i], code, resp[i].body_len);
                            rc = -1;
                        }
                        got++;
                    }
                    for (size_t i = 0; i < n; ++i) s3_response_free(&resp[i]);
                    if (got % 4096 < BATCH)
                        fprintf(stderr, "oracle_map: %zu/%zu chunks (%zu air)\n",
                                got, total, air);
                }
        free(urls); free(bufs); free(reqs); free(resp);
        if (rc != 0) { munmap(dense, bytes); close(wfd); unlink(tmp); return -1; }

        // Flush the mapped pages to disk, unmap, close, atomic rename + marker.
        msync(dense, bytes, MS_SYNC);
        munmap(dense, bytes);
        close(wfd);
        rename(tmp, path);
        FILE *df = fopen(done, "w"); if (df) fclose(df);
        fprintf(stderr, "oracle_map: L%d materialized (%zu air of %zu chunks)\n",
                oracle_level, air, total);
    } else {
        fprintf(stderr, "oracle_map: reusing existing %s (%.1f GB)\n", path, bytes / 1e9);
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "oracle_map: open %s failed\n", path); return -1; }
    void *m = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { fprintf(stderr, "oracle_map: mmap failed\n"); close(fd); return -1; }
    madvise(m, bytes, MADV_RANDOM);  // random access, don't read-ahead the whole file
    out->base = (const uint8_t *)m;
    out->shape[0] = shp[0]; out->shape[1] = shp[1]; out->shape[2] = shp[2];
    out->level = oracle_level; out->bytes = bytes; out->fd = fd; out->valid = 1;
    return 0;
}

void oracle_map_close(oracle_map *m) {
    if (m->base) munmap((void *)m->base, m->bytes);
    if (m->fd >= 0) close(m->fd);
    m->base = NULL; m->valid = 0;
}

void oracle_region_from_map(const oracle_map *m, int64_t oz, int64_t oy, int64_t ox,
                            oracle_region *out) {
    memset(out, 0, sizeof(*out));
    const int64_t E = ORACLE_EDGE;
    // Shard origin in oracle-voxel space.
    int64_t ooz = oz / ORACLE_SCALE, ooy = oy / ORACLE_SCALE, oox = ox / ORACLE_SCALE;
    const int64_t *shp = m->shape;
    int any = 0;
    for (int64_t z = 0; z < E; ++z) {
        int64_t vz = ooz + z; if (vz < 0 || vz >= shp[0]) continue;
        for (int64_t y = 0; y < E; ++y) {
            int64_t vy = ooy + y; if (vy < 0 || vy >= shp[1]) continue;
            for (int64_t x = 0; x < E; ++x) {
                int64_t vx = oox + x; if (vx < 0 || vx >= shp[2]) continue;
                uint8_t v = m->base[((size_t)vz * shp[1] + vy) * shp[2] + vx];
                out->vox[((size_t)z * E + y) * E + x] = v;
                if (v) any = 1;
            }
        }
    }
    out->valid = 1;
    out->all_zero = !any;
}
