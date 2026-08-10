/* zarr2c5d -- transcode a raw zarr v2 u8 volume (blosc chunks, 128^3 -- the
 * AWS vesuvius-challenge-open-data layout) into render3d's c5d LOD tree +
 * manifest.json, without any zarr output tree (the renderer only reads
 * c5d/L* + manifest.json).
 *
 * Source is a LOCAL mirror <mirror>/<L>/<cz>/<cy>/<cx> (+ <L>/.zarray per
 * level).  A zarr chunk is exactly one 128^3 c5d brick, so bricks map 1:1.
 * Missing-chunk semantics differ by level class:
 *   - levels >= --full-from were mirrored in full (aws s3 sync): an absent
 *     file IS an absent object = all-fill (zero) chunk;
 *   - levels <  --full-from are fetched selectively: a chunk must exist
 *     either as data or as a "<path>.missing" 404 marker (written by
 *     tools/fetch_chunks.sh); anything else is not-yet-downloaded and is
 *     reported via --list-missing.
 * Shard selection for the selective levels follows a tifxyz surface
 * (--surface, --pad): only 1024^3 shards whose AABB intersects a valid
 * surface point +/- pad are transcoded.  Writes are atomic and completed
 * shards are skipped, so long builds are resumable. */
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <blosc.h>
#include <curl/curl.h>

#include "brick.h"
#include "core/tifxyz.h"
#include "shard.h"

#define SHARD 1024u
#define BRICK 128u
#define SHARD_BPA (SHARD / BRICK)
#define NBRICKS (SHARD_BPA * SHARD_BPA * SHARD_BPA)
#define BRICK_BYTES ((size_t)BRICK * BRICK * BRICK)
#define MAX_LEVELS 8u

typedef struct dims3 {
  uint64_t z, y, x;
} dims3;

typedef struct level_info {
  dims3 shape;    /* voxels (z,y,x) */
  dims3 chunks;   /* chunk grid */
  dims3 shards;   /* 1024^3 shard grid */
  uint8_t *want;  /* shard selection bitmap (1 byte per shard), NULL = all */
  uint8_t *cwant; /* chunk selection bitmap; unwanted chunks transcode as zero */
  bool raw;       /* "compressor": null — chunks are plain u8 payloads */
  uint32_t chsz;  /* zarr chunk edge (128 or 256; a chunk is (chsz/128)^3 bricks) */
} level_info;

typedef struct blob {
  uint8_t *p;
  uint32_t n;
} blob;

static const char *g_mirror, *g_out;
static const char *g_url = NULL; /* streaming ingest: fetch chunks, keep no mirror */
static _Atomic uint64_t g_fetched_bytes = 0, g_fetched_n = 0, g_absent_n = 0;
static level_info g_lv[MAX_LEVELS];
static uint32_t g_nlev = 0, g_full_from = 3;
static float g_quality = 2.0f;
static FILE *g_missing_list = NULL;
static uint64_t g_missing_count = 0;
static _Atomic uint64_t g_psnr_bricks = 0;
static double g_sse_sum = 0; /* guarded by g_verify_mu */
static pthread_mutex_t g_verify_mu = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_verify = 0;

static int mkdirs(const char *path, bool includes_leaf) {
  char tmp[2048];
  size_t n = strlen(path);
  if (n == 0 || n >= sizeof tmp) return -1;
  memcpy(tmp, path, n + 1u);
  if (!includes_leaf) {
    char *slash = strrchr(tmp, '/');
    if (!slash) return 0;
    *slash = 0;
  }
  for (char *p = tmp + 1; *p; p++) {
    if (*p != '/') continue;
    *p = 0;
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    *p = '/';
  }
  return mkdir(tmp, 0755) == 0 || errno == EEXIST ? 0 : -1;
}

static bool file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && st.st_size > 0;
}

static bool path_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0;
}

static int write_atomic(const char *path, const void *data, size_t n) {
  if (mkdirs(path, false) != 0) return -1;
  char tmp[2112];
  int pn = snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
  if (pn < 0 || (size_t)pn >= sizeof tmp) return -1;
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  int rc = fwrite(data, 1, n, f) == n && fflush(f) == 0 && fsync(fileno(f)) == 0 ? 0 : -1;
  if (fclose(f) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

static bool all_zero(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (p[i]) return false;
  return true;
}

/* Minimal .zarray probe: shape triplet + chunks triplet + dtype "|u1". */
static int read_zarray(uint32_t level, dims3 *shape, dims3 *chunks, bool *raw) {
  char path[2048];
  snprintf(path, sizeof path, "%s/%u/.zarray", g_mirror, level);
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  char buf[4096] = {0};
  size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  (void)n;
  const char *s = strstr(buf, "\"shape\"");
  const char *c = strstr(buf, "\"chunks\"");
  if (!s || !c || !strstr(buf, "|u1")) return -1;
  unsigned long long v[3];
  if (!(s = strchr(s, '[')) || sscanf(s, "[ %llu , %llu , %llu", &v[0], &v[1], &v[2]) != 3)
    return -1;
  *shape = (dims3){v[0], v[1], v[2]};
  if (!(c = strchr(c, '[')) || sscanf(c, "[ %llu , %llu , %llu", &v[0], &v[1], &v[2]) != 3)
    return -1;
  *chunks = (dims3){v[0], v[1], v[2]};
  const char *cp = strstr(buf, "\"compressor\"");
  *raw = cp && strncmp(cp + 12, ": null", 6) == 0;
  return 0;
}

static void chunk_path(char path[2048], uint32_t level, uint64_t cz, uint64_t cy,
                       uint64_t cx) {
  snprintf(path, 2048, "%s/%u/%llu/%llu/%llu", g_mirror, level, (unsigned long long)cz,
           (unsigned long long)cy, (unsigned long long)cx);
}

/* Load one 128^3 chunk. Returns 1 = data, 0 = zero-fill, -1 = error,
 * -2 = not downloaded (recorded in the missing list when open). */
/* --- streaming ingest: fetch a chunk over HTTP straight into memory ------- */

typedef struct fetch_buf {
  uint8_t *p;
  size_t n, cap;
} fetch_buf;

static size_t fetch_write(const void *data, size_t sz, size_t nm, void *ud) {
  fetch_buf *b = ud;
  size_t n = sz * nm;
  if (b->n + n > b->cap) {
    size_t nc = b->cap ? b->cap * 2 : (4u << 20);
    while (nc < b->n + n) nc *= 2;
    uint8_t *np = realloc(b->p, nc);
    if (!np) return 0;
    b->p = np;
    b->cap = nc;
  }
  memcpy(b->p + b->n, data, n);
  b->n += n;
  return n;
}

/* one CURL handle per worker thread (connection/TLS session reuse) */
static _Thread_local CURL *t_curl = NULL;

/* 1 = fetched into buf, 0 = absent (404 -> fill), -1 = error after retries */
static int fetch_chunk(uint32_t level, uint64_t cz, uint64_t cy, uint64_t cx,
                       fetch_buf *buf) {
  if (!t_curl) {
    t_curl = curl_easy_init();
    if (!t_curl) return -1;
    curl_easy_setopt(t_curl, CURLOPT_WRITEFUNCTION, fetch_write);
    curl_easy_setopt(t_curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(t_curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(t_curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(t_curl, CURLOPT_LOW_SPEED_TIME, 60L);
  }
  char url[2048];
  snprintf(url, sizeof url, "%s/%u/%llu/%llu/%llu", g_url, level, (unsigned long long)cz,
           (unsigned long long)cy, (unsigned long long)cx);
  for (int attempt = 0; attempt < 4; attempt++) {
    buf->n = 0;
    curl_easy_setopt(t_curl, CURLOPT_URL, url);
    curl_easy_setopt(t_curl, CURLOPT_WRITEDATA, buf);
    CURLcode rc = curl_easy_perform(t_curl);
    long code = 0;
    curl_easy_getinfo(t_curl, CURLINFO_RESPONSE_CODE, &code);
    if (rc == CURLE_OK && code == 200) {
      atomic_fetch_add(&g_fetched_bytes, buf->n);
      atomic_fetch_add(&g_fetched_n, 1);
      return 1;
    }
    if (rc == CURLE_OK && code == 404) {
      atomic_fetch_add(&g_absent_n, 1);
      return 0;
    }
    if (attempt < 3) sleep((unsigned)(1 << attempt)); /* 1s, 2s, 4s backoff */
  }
  fprintf(stderr, "zarr2c5d: fetch failed after retries: %s\n", url);
  return -1;
}

static bool chunk_wanted(const level_info *lv, uint64_t cz, uint64_t cy, uint64_t cx) {
  return !lv->cwant ||
         lv->cwant[(cz * lv->chunks.y + cy) * lv->chunks.x + cx];
}

static int decode_chunk_mem(const level_info *lv, const uint8_t *comp, size_t n,
                            uint8_t *dst) {
  size_t want = (size_t)lv->chsz * lv->chsz * lv->chsz;
  if (lv->raw) {
    if (n != want) return -1;
    memcpy(dst, comp, want);
    return 1;
  }
  size_t nbytes = 0, cbytes = 0, blocksize = 0;
  blosc_cbuffer_sizes(comp, &nbytes, &cbytes, &blocksize);
  if (nbytes != want || cbytes > n ||
      blosc_decompress_ctx(comp, dst, want, 1) != (int)want)
    return -1;
  return 1;
}

static _Thread_local fetch_buf t_buf = {0};

static int load_chunk(uint32_t level, uint64_t cz, uint64_t cy, uint64_t cx, uint8_t *dst) {
  const level_info *lv = &g_lv[level];
  if (cz >= lv->chunks.z || cy >= lv->chunks.y || cx >= lv->chunks.x) return 0;
  if (!chunk_wanted(lv, cz, cy, cx)) return 0; /* far from surface: air */
  if (g_url) { /* streaming ingest: network -> memory -> brick, no mirror */
    int rc = fetch_chunk(level, cz, cy, cx, &t_buf);
    if (rc <= 0) return rc;
    rc = decode_chunk_mem(lv, t_buf.p, t_buf.n, dst);
    if (rc < 0)
      fprintf(stderr, "zarr2c5d: bad fetched chunk %u/%llu/%llu/%llu (%zu bytes)\n", level,
              (unsigned long long)cz, (unsigned long long)cy, (unsigned long long)cx,
              t_buf.n);
    return rc;
  }
  char path[2048];
  chunk_path(path, level, cz, cy, cx);
  if (!file_exists(path)) {
    if (level >= g_full_from) return 0; /* fully-mirrored level: absent == fill */
    char marker[2064];
    snprintf(marker, sizeof marker, "%s.missing", path);
    if (path_exists(marker)) return 0;
    return -2;
  }
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long fn = ftell(f);
  fseek(f, 0, SEEK_SET);
  size_t want = (size_t)lv->chsz * lv->chsz * lv->chsz;
  if (fn <= 0 || fn > (long)(want + (16u << 20))) {
    fclose(f);
    return -1;
  }
  if (lv->raw) { /* "compressor": null — the file IS the chunk payload */
    size_t got = fread(dst, 1, want, f);
    fclose(f);
    if (got != want) {
      fprintf(stderr, "zarr2c5d: raw chunk %s is %zu bytes\n", path, got);
      return -1;
    }
    return 1;
  }
  uint8_t *comp = malloc((size_t)fn);
  if (!comp || fread(comp, 1, (size_t)fn, f) != (size_t)fn) {
    free(comp);
    fclose(f);
    return -1;
  }
  fclose(f);
  size_t nbytes = 0, cbytes = 0, blocksize = 0;
  blosc_cbuffer_sizes(comp, &nbytes, &cbytes, &blocksize);
  if (nbytes != want || (long)cbytes > fn ||
      blosc_decompress_ctx(comp, dst, want, 1) != (int)want) {
    fprintf(stderr, "zarr2c5d: bad chunk %s (nbytes %zu)\n", path, nbytes);
    free(comp);
    return -1;
  }
  free(comp);
  return 1;
}

/* --- shard transcode ------------------------------------------------------ */

typedef struct shard_job {
  uint32_t level;
  uint64_t oz, oy, ox; /* shard coords */
  blob bricks[NBRICKS];
  uint8_t zero[NBRICKS];
  _Atomic uint32_t next;
  _Atomic int failed;
  _Atomic uint64_t missing; /* chunks not yet downloaded */
} shard_job;

/* chunk-major: decode each zarr chunk once and cut it into its (chsz/128)^3
 * bricks — with 256^3 chunks a brick-major loop would decompress the same
 * 16 MB chunk eight times */
static void *brick_worker(void *arg) {
  shard_job *j = arg;
  const level_info *lv = &g_lv[j->level];
  uint32_t chsz = lv->chsz, cpa = SHARD / chsz, bpc = chsz / BRICK;
  size_t chunk_bytes = (size_t)chsz * chsz * chsz;
  uint8_t *chunk = malloc(chunk_bytes);
  uint8_t *raw = malloc(BRICK_BYTES);
  uint8_t *rec = g_verify ? malloc(BRICK_BYTES) : NULL;
  if (!chunk || !raw || (g_verify && !rec)) {
    free(chunk);
    free(raw);
    free(rec);
    atomic_store(&j->failed, 1);
    return NULL;
  }
  uint32_t ncells = cpa * cpa * cpa;
  for (;;) {
    uint32_t cell = atomic_fetch_add(&j->next, 1);
    if (cell >= ncells || atomic_load(&j->failed)) break;
    uint32_t lz = cell / (cpa * cpa), ly = (cell / cpa) % cpa, lx = cell % cpa;
    int rc = load_chunk(j->level, j->oz * cpa + lz, j->oy * cpa + ly, j->ox * cpa + lx,
                        chunk);
    if (rc == -2) {
      atomic_fetch_add(&j->missing, 1);
      continue;
    }
    if (rc < 0) {
      atomic_store(&j->failed, 1);
      break;
    }
    for (uint32_t sz = 0; sz < bpc && !atomic_load(&j->failed); sz++)
      for (uint32_t sy = 0; sy < bpc; sy++)
        for (uint32_t sx = 0; sx < bpc; sx++) {
          uint32_t bz = lz * bpc + sz, by = ly * bpc + sy, bx = lx * bpc + sx;
          uint32_t b = (bz * SHARD_BPA + by) * SHARD_BPA + bx;
          if (rc == 0) {
            j->zero[b] = 1;
            continue;
          }
          for (uint32_t r = 0; r < BRICK; r++) /* gather the 128^3 sub-cube */
            for (uint32_t q_ = 0; q_ < BRICK; q_++)
              memcpy(raw + ((size_t)r * BRICK + q_) * BRICK,
                     chunk + (((size_t)(sz * BRICK + r) * chsz + sy * BRICK + q_) * chsz +
                              sx * BRICK),
                     BRICK);
          if (all_zero(raw, BRICK_BYTES)) {
            j->zero[b] = 1;
            continue;
          }
          float q = g_quality / (float)(1u << (j->level < 3u ? j->level : 3u));
          if (q < 0.25f) q = 0.25f;
          c5d_brick_params p = c5d_brick_defaults(1.0f);
          p.q = q;
          size_t n = 0;
          if (c5d_brick_encode(&p, raw, BRICK, &j->bricks[b].p, &n) != 0 ||
              n > UINT32_MAX) {
            atomic_store(&j->failed, 1);
            break;
          }
          j->bricks[b].n = (uint32_t)n;
          if (g_verify && atomic_fetch_add(&g_psnr_bricks, 1) < g_verify) {
            if (c5d_brick_decode(j->bricks[b].p, n, rec, BRICK) != 0) {
              atomic_store(&j->failed, 1);
              break;
            }
            double sse = 0;
            for (size_t i = 0; i < BRICK_BYTES; i++) {
              double d = (double)raw[i] - (double)rec[i];
              sse += d * d;
            }
            pthread_mutex_lock(&g_verify_mu);
            g_sse_sum += sse / (double)BRICK_BYTES;
            pthread_mutex_unlock(&g_verify_mu);
          } else if (g_verify) {
            atomic_fetch_sub(&g_psnr_bricks, 1);
          }
        }
  }
  free(chunk);
  free(raw);
  free(rec);
  return NULL;
}

static int process_shard(uint32_t level, uint64_t oz, uint64_t oy, uint64_t ox,
                         uint32_t threads, bool force) {
  char cp[2048];
  snprintf(cp, sizeof cp, "%s/c5d/L%u/%llu_%llu_%llu.c5s", g_out, level,
           (unsigned long long)oz, (unsigned long long)oy, (unsigned long long)ox);
  if (!force && file_exists(cp)) return 1;

  /* missing-list pass: enumerate instead of transcode */
  if (g_missing_list) {
    const level_info *lv = &g_lv[level];
    uint32_t cpa = SHARD / lv->chsz;
    for (uint32_t b = 0; b < cpa * cpa * cpa; b++) {
      uint64_t cz = oz * cpa + b / (cpa * cpa);
      uint64_t cy = oy * cpa + (b / cpa) % cpa;
      uint64_t cx = ox * cpa + b % cpa;
      if (cz >= lv->chunks.z || cy >= lv->chunks.y || cx >= lv->chunks.x) continue;
      if (level >= g_full_from || !chunk_wanted(lv, cz, cy, cx)) continue;
      char path[2048], marker[2064];
      chunk_path(path, level, cz, cy, cx);
      snprintf(marker, sizeof marker, "%s.missing", path);
      if (!file_exists(path) && !path_exists(marker)) {
        fprintf(g_missing_list, "%u/%llu/%llu/%llu\n", level, (unsigned long long)cz,
                (unsigned long long)cy, (unsigned long long)cx);
        g_missing_count++;
      }
    }
    return 0;
  }

  shard_job *j = calloc(1, sizeof *j);
  if (!j) return -1;
  j->level = level;
  j->oz = oz;
  j->oy = oy;
  j->ox = ox;
  pthread_t tids[32];
  uint32_t nt = threads > 32u ? 32u : threads, created = 0;
  for (; created < nt; created++)
    if (pthread_create(&tids[created], NULL, brick_worker, j) != 0) {
      atomic_store(&j->failed, 1);
      break;
    }
  for (uint32_t t = 0; t < created; t++) pthread_join(tids[t], NULL);

  int rc = atomic_load(&j->failed) ? -1 : 0;
  uint64_t miss = atomic_load(&j->missing);
  if (rc == 0 && miss) {
    fprintf(stderr,
            "zarr2c5d: L%u shard %llu/%llu/%llu: %llu chunks not downloaded "
            "(run --list-missing + tools/fetch_chunks.sh first)\n",
            level, (unsigned long long)oz, (unsigned long long)oy,
            (unsigned long long)ox, (unsigned long long)miss);
    rc = -1;
  }
  if (rc == 0) {
    if (mkdirs(cp, false) != 0) rc = -1;
    char tmp[2112];
    snprintf(tmp, sizeof tmp, "%s.tmp.%ld", cp, (long)getpid());
    c5d_shard_writer *w = rc == 0 ? c5d_shard_create(tmp, SHARD, BRICK, level, 0.0f) : NULL;
    if (!w) rc = -1;
    for (uint32_t b = 0; b < NBRICKS && rc == 0; b++)
      rc = j->zero[b] || !j->bricks[b].p ? c5d_shard_put_zero(w, b)
                                         : c5d_shard_put(w, b, j->bricks[b].p, j->bricks[b].n);
    if (w && c5d_shard_close(w) != 0) rc = -1;
    if (rc == 0 && rename(tmp, cp) != 0) rc = -1;
    if (rc != 0) unlink(tmp);
  }
  for (uint32_t b = 0; b < NBRICKS; b++) free(j->bricks[b].p);
  free(j);
  return rc;
}

/* --- surface-driven shard selection --------------------------------------- */

static int mark_surface(const char *surf_dir, uint32_t pad, uint32_t min_level,
                        const uint32_t rect[4]) {
  r3d_tifxyz s;
  if (r3d_tifxyz_load(&s, surf_dir) != 0) return -1;
  printf("zarr2c5d: surface %s: %ux%u grid, %llu valid points\n", surf_dir, s.w, s.h,
         (unsigned long long)s.nvalid);
  for (uint32_t l = min_level; l < g_full_from && l < g_nlev; l++) {
    level_info *lv = &g_lv[l];
    uint64_t ns = lv->shards.z * lv->shards.y * lv->shards.x;
    uint64_t nc = lv->chunks.z * lv->chunks.y * lv->chunks.x;
    lv->want = calloc(ns, 1);
    lv->cwant = calloc(nc, 1);
    if (!lv->want || !lv->cwant) {
      r3d_tifxyz_free(&s);
      return -1;
    }
    uint32_t sc = 1u << l;
    for (uint64_t k = 0; k < (uint64_t)s.w * s.h; k++) {
      /* --rect confines FULL-RES (level 0) coverage to a grid sub-rect; the
       * coarser selective levels still track the whole surface */
      if (l == 0 && rect) {
        uint32_t gi = (uint32_t)(k % s.w), gj = (uint32_t)(k / s.w);
        if (gi < rect[0] || gj < rect[1] || gi >= rect[2] || gj >= rect[3]) continue;
      }
      const float *p = s.xyz + k * 3;
      if (!r3d_tifxyz_valid(p)) continue;
      /* world voxels -> level voxels -> chunk range covering p +/- pad */
      int64_t c0[3], c1[3];
      const double c[3] = {(double)p[2], (double)p[1], (double)p[0]}; /* z,y,x */
      const uint64_t g[3] = {lv->chunks.z, lv->chunks.y, lv->chunks.x};
      for (int a = 0; a < 3; a++) {
        double lo = (c[a] - (double)pad) / (double)sc, hi = (c[a] + (double)pad) / (double)sc;
        c0[a] = (int64_t)(lo / lv->chsz);
        c1[a] = (int64_t)(hi / lv->chsz);
        if (c0[a] < 0) c0[a] = 0;
        if (c1[a] >= (int64_t)g[a]) c1[a] = (int64_t)g[a] - 1;
      }
      for (int64_t az = c0[0]; az <= c1[0]; az++)
        for (int64_t ay = c0[1]; ay <= c1[1]; ay++)
          for (int64_t ax = c0[2]; ax <= c1[2]; ax++) {
            uint64_t ci = ((uint64_t)az * lv->chunks.y + (uint64_t)ay) * lv->chunks.x +
                          (uint64_t)ax;
            if (lv->cwant[ci]) continue;
            lv->cwant[ci] = 1;
            lv->want[(((uint64_t)az * lv->chsz) / SHARD * lv->shards.y +
                      ((uint64_t)ay * lv->chsz) / SHARD) *
                         lv->shards.x +
                     ((uint64_t)ax * lv->chsz) / SHARD] = 1;
          }
    }
  }
  r3d_tifxyz_free(&s);
  return 0;
}

/* --- manifest (lodpack's exact emission format) --------------------------- */

static int write_manifest(void) {
  char *json = calloc(1, 16384);
  if (!json) return -1;
  dims3 base = g_lv[0].shape;
  size_t n = (size_t)snprintf(json, 16384,
                              "{\n  \"format\": \"render3d.c5d-lod.v1\",\n"
                              "  \"shape\": [%llu, %llu, %llu],\n"
                              "  \"shard_shape\": [1024, 1024, 1024],\n"
                              "  \"brick_shape\": [128, 128, 128],\n  \"levels\": [\n",
                              (unsigned long long)base.z, (unsigned long long)base.y,
                              (unsigned long long)base.x);
  for (uint32_t l = 0; l < g_nlev; l++) {
    const level_info *lv = &g_lv[l];
    float q = g_quality / (float)(1u << (l < 3u ? l : 3u));
    if (q < 0.25f) q = 0.25f;
    n += (size_t)snprintf(json + n, 16384 - n,
                          "    {\"level\": %u, \"scale\": %u, "
                          "\"shape\": [%llu, %llu, %llu], "
                          "\"shards\": [%llu, %llu, %llu], "
                          "\"zarr\": \"zarr/L%u\", "
                          "\"c5d\": \"c5d/L%u/{z}_{y}_{x}.c5s\", "
                          "\"c5d_quality\": %.6g}%s\n",
                          l, 1u << l, (unsigned long long)lv->shape.z,
                          (unsigned long long)lv->shape.y, (unsigned long long)lv->shape.x,
                          (unsigned long long)lv->shards.z, (unsigned long long)lv->shards.y,
                          (unsigned long long)lv->shards.x, l, l, (double)q,
                          l + 1u == g_nlev ? "" : ",");
  }
  n += (size_t)snprintf(json + n, 16384 - n, "  ]\n}\n");
  char mp[2048];
  snprintf(mp, sizeof mp, "%s/manifest.json", g_out);
  int rc = n < 16384 ? write_atomic(mp, json, n) : -1;
  free(json);
  return rc;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: zarr2c5d <zarr-mirror-dir> <output-dir> [--url BASE] [--surface DIR] "
            "[--pad N] [--full-from L] [--min-level L] [--rect i0 j0 i1 j1] [--threads N] "
            "[--c5d-quality Q] [--only-level L] "
            "[--list-missing FILE] [--dry-run] [--verify N] [--force]\n"
            "  --url: streaming ingest — chunks are fetched straight into memory and only "
            "transcoded c5d shards are written (the mirror dir holds .zarray metadata "
            "only); without it, chunks are read from a pre-fetched local mirror\n");
    return 2;
  }
  g_mirror = argv[1];
  g_out = argv[2];
  const char *surf_dir = NULL, *missing_path = NULL;
  uint32_t pad = 64, threads = 0, only_level = UINT32_MAX, min_level = 0;
  uint32_t rect[4] = {0};
  bool have_rect = false;
  bool dry = false, force = false;
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--surface") == 0 && i + 1 < argc)
      surf_dir = argv[++i];
    else if (strcmp(argv[i], "--pad") == 0 && i + 1 < argc)
      pad = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--full-from") == 0 && i + 1 < argc)
      g_full_from = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
      threads = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--c5d-quality") == 0 && i + 1 < argc)
      g_quality = strtof(argv[++i], NULL);
    else if (strcmp(argv[i], "--only-level") == 0 && i + 1 < argc)
      only_level = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--min-level") == 0 && i + 1 < argc)
      min_level = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--rect") == 0 && i + 4 < argc) {
      for (int k = 0; k < 4; k++) rect[k] = (uint32_t)strtoul(argv[++i], NULL, 10);
      have_rect = true;
    }
    else if (strcmp(argv[i], "--list-missing") == 0 && i + 1 < argc)
      missing_path = argv[++i];
    else if (strcmp(argv[i], "--url") == 0 && i + 1 < argc)
      g_url = argv[++i];
    else if (strcmp(argv[i], "--dry-run") == 0)
      dry = true;
    else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc)
      g_verify = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--force") == 0)
      force = true;
    else {
      fprintf(stderr, "zarr2c5d: unknown/incomplete option %s\n", argv[i]);
      return 2;
    }
  }
  if (!threads) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    threads = ncpu > 0 ? (uint32_t)ncpu : 1u;
  }
  if (g_url) { /* streaming ingest: bootstrap per-level .zarray metadata only */
    curl_global_init(CURL_GLOBAL_DEFAULT);
    for (uint32_t l = 0; l < MAX_LEVELS; l++) {
      char zp[2048];
      snprintf(zp, sizeof zp, "%s/%u/.zarray", g_mirror, l);
      if (file_exists(zp)) continue;
      fetch_buf fb = {0};
      char zu[2048];
      snprintf(zu, sizeof zu, "%s/%u/.zarray", g_url, l);
      CURL *c = curl_easy_init();
      if (!c) return 1;
      curl_easy_setopt(c, CURLOPT_URL, zu);
      curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, fetch_write);
      curl_easy_setopt(c, CURLOPT_WRITEDATA, &fb);
      curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
      long code = 0;
      CURLcode rc = curl_easy_perform(c);
      curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
      curl_easy_cleanup(c);
      if (rc == CURLE_OK && code == 200 && fb.n && write_atomic(zp, fb.p, fb.n) == 0) {
        free(fb.p);
        continue;
      }
      free(fb.p);
      break; /* first absent level ends the pyramid */
    }
  }

  /* probe levels */
  for (uint32_t l = 0; l < MAX_LEVELS; l++) {
    dims3 shape, chunks;
    bool raw = false;
    if (read_zarray(l, &shape, &chunks, &raw) != 0) break;
    g_lv[l].raw = raw;
    if (chunks.z != chunks.y || chunks.y != chunks.x ||
        (chunks.z != 128 && chunks.z != 256 && chunks.z != 512)) {
      fprintf(stderr, "zarr2c5d: L%u chunks must be cubic 128/256/512\n", l);
      return 1;
    }
    uint32_t ch = (uint32_t)chunks.z;
    g_lv[l].chsz = ch;
    g_lv[l].shape = shape;
    g_lv[l].chunks = (dims3){(shape.z + ch - 1) / ch, (shape.y + ch - 1) / ch,
                             (shape.x + ch - 1) / ch};
    g_lv[l].shards = (dims3){(shape.z + SHARD - 1) / SHARD, (shape.y + SHARD - 1) / SHARD,
                             (shape.x + SHARD - 1) / SHARD};
    g_nlev = l + 1;
  }
  if (!g_nlev) {
    fprintf(stderr, "zarr2c5d: no <mirror>/<L>/.zarray found under %s\n", g_mirror);
    return 1;
  }
  /* the renderer requires halving levels (scale == 1<<l) */
  for (uint32_t l = 1; l < g_nlev; l++) {
    dims3 want = {(g_lv[l - 1].shape.z + 1) / 2, (g_lv[l - 1].shape.y + 1) / 2,
                  (g_lv[l - 1].shape.x + 1) / 2};
    if (want.z != g_lv[l].shape.z || want.y != g_lv[l].shape.y ||
        want.x != g_lv[l].shape.x) {
      fprintf(stderr, "zarr2c5d: L%u shape is not the ceil-half of L%u\n", l, l - 1);
      return 1;
    }
  }
  if (g_full_from > g_nlev) g_full_from = g_nlev;
  if (g_full_from > 0 && !surf_dir && only_level == UINT32_MAX) {
    fprintf(stderr, "zarr2c5d: levels 0..%u need --surface (or --full-from 0)\n",
            g_full_from - 1);
    return 2;
  }
  if (surf_dir && mark_surface(surf_dir, pad, min_level, have_rect ? rect : NULL) != 0)
    return 1;

  uint64_t plan[MAX_LEVELS] = {0}, total = 0;
  for (uint32_t l = 0; l < g_nlev; l++) {
    const level_info *lv = &g_lv[l];
    uint64_t ns = lv->shards.z * lv->shards.y * lv->shards.x;
    if (lv->want) {
      for (uint64_t i = 0; i < ns; i++) plan[l] += lv->want[i];
    } else if (l >= g_full_from) {
      plan[l] = ns;
    }
    if (only_level != UINT32_MAX && l != only_level) plan[l] = 0;
    total += plan[l];
    uint64_t wc = 0, nc = lv->chunks.z * lv->chunks.y * lv->chunks.x;
    if (lv->cwant)
      for (uint64_t i = 0; i < nc; i++) wc += lv->cwant[i];
    printf("zarr2c5d: L%u %llux%llux%llu -> %llu/%llu shards%s", l,
           (unsigned long long)lv->shape.z, (unsigned long long)lv->shape.y,
           (unsigned long long)lv->shape.x, (unsigned long long)plan[l],
           (unsigned long long)ns, lv->want ? " (surface" : "");
    if (lv->cwant)
      printf(", %llu/%llu chunks ~%.1f GB)", (unsigned long long)wc,
             (unsigned long long)nc,
             (double)wc * (double)lv->chsz * lv->chsz * lv->chsz / 1073741824.0);
    printf("\n");
  }
  printf("zarr2c5d: %llu shards total, %u levels, threads %u\n",
         (unsigned long long)total, g_nlev, threads);
  if (dry) return 0;

  if (missing_path) {
    g_missing_list = fopen(missing_path, "w");
    if (!g_missing_list) {
      fprintf(stderr, "zarr2c5d: cannot write %s\n", missing_path);
      return 1;
    }
  } else if (mkdirs(g_out, true) != 0 || write_manifest() != 0) {
    fprintf(stderr, "zarr2c5d: cannot initialise output\n");
    return 1;
  }

  for (uint32_t l = 0; l < g_nlev; l++) {
    if (!plan[l]) continue;
    const level_info *lv = &g_lv[l];
    uint64_t done = 0, skipped = 0;
    double started = now_seconds();
    for (uint64_t sz = 0; sz < lv->shards.z; sz++)
      for (uint64_t sy = 0; sy < lv->shards.y; sy++)
        for (uint64_t sx = 0; sx < lv->shards.x; sx++) {
          if (lv->want &&
              !lv->want[(sz * lv->shards.y + sy) * lv->shards.x + sx])
            continue;
          if (!lv->want && l < g_full_from) continue;
          int rc = process_shard(l, sz, sy, sx, threads, force);
          if (rc < 0) {
            fprintf(stderr, "zarr2c5d: L%u shard %llu/%llu/%llu failed\n", l,
                    (unsigned long long)sz, (unsigned long long)sy,
                    (unsigned long long)sx);
            return 1;
          }
          skipped += rc > 0;
          done++;
          if (!g_missing_list) {
            printf("zarr2c5d: L%u %llu/%llu (%s; %.1fs/shard)\n", l,
                   (unsigned long long)done, (unsigned long long)plan[l],
                   rc > 0 ? "resume-skip" : "written",
                   (now_seconds() - started) / (double)done);
            fflush(stdout);
          }
        }
    if (!g_missing_list)
      printf("zarr2c5d: L%u complete (%llu written, %llu skipped)\n", l,
             (unsigned long long)(done - skipped), (unsigned long long)skipped);
  }

  if (g_missing_list) {
    fclose(g_missing_list);
    printf("zarr2c5d: %llu chunks to fetch listed in %s\n",
           (unsigned long long)g_missing_count, missing_path);
    return g_missing_count ? 3 : 0;
  }
  if (g_url)
    printf("zarr2c5d: fetched %llu chunks (%.1f GB), %llu absent (fill)\n",
           (unsigned long long)atomic_load(&g_fetched_n),
           (double)atomic_load(&g_fetched_bytes) / 1073741824.0,
           (unsigned long long)atomic_load(&g_absent_n));
  uint64_t vb = atomic_load(&g_psnr_bricks);
  if (g_verify && vb) {
    uint64_t counted = vb < g_verify ? vb : g_verify;
    double mse = g_sse_sum / (double)counted;
    printf("zarr2c5d: verify %llu bricks, PSNR %.2f dB\n", (unsigned long long)counted,
           10.0 * log10(255.0 * 255.0 / (mse > 1e-12 ? mse : 1e-12)));
  }
  return 0;
}
