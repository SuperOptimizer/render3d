/* lodpack -- build a standard Zarr-v3 multiscale pyramid and matching c5d
 * shards without materialising an uncompressed volume.
 *
 * Source is render3d's mirrored dct3d store:
 *   <source>/<z>_<y>_<x>.shard
 * Output is:
 *   <output>/zarr/L<level>/c/<z>/<y>/<x>       (Zarr v3)
 *   <output>/c5d/L<level>/<z>_<y>_<x>.c5s     (c5d)
 *   <output>/manifest.json                    (renderer LOD manifest)
 *
 * Each 16^3 output Zarr chunk is a rounded 2x box reduction of eight decoded
 * parent chunks.  The chunk is encoded to dct3d and decoded again before it is
 * assembled into its c5d brick: c5d is therefore a true transcode of the Zarr
 * LOD, not a parallel encode of a subtly different source.  Writes are atomic
 * and completed shard pairs are skipped, making long builds resumable. */
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

#include "brick.h"
#include "dct3d.h"
#include "shard.h"
#include "shard/shardio.h"

#define SHARD 1024u
#define BRICK 128u
#define INNER 16u
#define SHARD_BPA (SHARD / BRICK)
#define BRICK_CPA (BRICK / INNER)
#define NBRICKS (SHARD_BPA * SHARD_BPA * SHARD_BPA)
#define NCHUNKS ((size_t)(SHARD / INNER) * (SHARD / INNER) * (SHARD / INNER))
#define ZARR_INDEX_BYTES (NCHUNKS * 16u)
#define MAX_LEVELS 8u

typedef struct dims3 {
  uint64_t z, y, x;
} dims3;

typedef struct blob {
  uint8_t *p;
  uint32_t n;
} blob;

typedef struct parent_set {
  r3d_shard shard[2][2][2];
  bool open[2][2][2];
} parent_set;

typedef struct shard_job {
  parent_set *parents;
  blob *zchunks;
  blob *cbricks;
  uint8_t *czero;
  float zq, ztau, c5quality;
  _Atomic uint32_t next;
  _Atomic int failed;
  uint32_t level;
  bool downsample;
} shard_job;

static dims3 level_shape(dims3 base, uint32_t level) {
  while (level--) {
    base.z = (base.z + 1u) / 2u;
    base.y = (base.y + 1u) / 2u;
    base.x = (base.x + 1u) / 2u;
  }
  return base;
}

static dims3 shard_grid(dims3 shape) {
  return (dims3){(shape.z + SHARD - 1u) / SHARD, (shape.y + SHARD - 1u) / SHARD,
                 (shape.x + SHARD - 1u) / SHARD};
}

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

static void put_u64le(uint8_t *p, uint64_t v) {
  for (uint32_t i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static void put_u32le(uint8_t *p, uint32_t v) {
  for (uint32_t i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static bool all_zero(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (p[i]) return false;
  return true;
}

static void source_path(char path[2048], const char *source, const char *out, uint32_t level,
                        uint64_t z, uint64_t y, uint64_t x) {
  if (level == 0)
    snprintf(path, 2048, "%s/%llu_%llu_%llu.shard", source, (unsigned long long)z,
             (unsigned long long)y, (unsigned long long)x);
  else
    snprintf(path, 2048, "%s/zarr/L%u/c/%llu/%llu/%llu", out, level,
             (unsigned long long)z, (unsigned long long)y, (unsigned long long)x);
}

static void zarr_shard_path(char path[2048], const char *out, uint32_t level, uint64_t z,
                            uint64_t y, uint64_t x) {
  snprintf(path, 2048, "%s/zarr/L%u/c/%llu/%llu/%llu", out, level,
           (unsigned long long)z, (unsigned long long)y, (unsigned long long)x);
}

static void c5d_shard_path(char path[2048], const char *out, uint32_t level, uint64_t z,
                           uint64_t y, uint64_t x) {
  snprintf(path, 2048, "%s/c5d/L%u/%llu_%llu_%llu.c5s", out, level,
           (unsigned long long)z, (unsigned long long)y, (unsigned long long)x);
}

static int open_parents(parent_set *ps, const char *source, const char *out, uint32_t src_level,
                        dims3 src_grid, uint64_t oz, uint64_t oy, uint64_t ox,
                        bool downsample) {
  memset(ps, 0, sizeof *ps);
  uint32_t reach = downsample ? 2u : 1u;
  for (uint32_t dz = 0; dz < reach; dz++)
    for (uint32_t dy = 0; dy < reach; dy++)
      for (uint32_t dx = 0; dx < reach; dx++) {
        uint64_t z = (downsample ? oz * 2u : oz) + dz;
        uint64_t y = (downsample ? oy * 2u : oy) + dy;
        uint64_t x = (downsample ? ox * 2u : ox) + dx;
        if (z >= src_grid.z || y >= src_grid.y || x >= src_grid.x) continue;
        char path[2048];
        source_path(path, source, out, src_level, z, y, x);
        int rc = r3d_shard_open_path(path, &ps->shard[dz][dy][dx]);
        if (rc != R3D_SHARD_OK) {
          fprintf(stderr, "lodpack: cannot open parent %s (%d)\n", path, rc);
          return -1;
        }
        ps->open[dz][dy][dx] = true;
      }
  return 0;
}

static void close_parents(parent_set *ps) {
  for (uint32_t z = 0; z < 2; z++)
    for (uint32_t y = 0; y < 2; y++)
      for (uint32_t x = 0; x < 2; x++)
        if (ps->open[z][y][x]) r3d_shard_close(&ps->shard[z][y][x]);
}

static int decode_parent_chunk(const parent_set *ps, uint32_t pz, uint32_t py, uint32_t px,
                               uint32_t cz, uint32_t cy, uint32_t cx, uint8_t out[DCT3D_N3]) {
  if (!ps->open[pz][py][px]) {
    memset(out, 0, DCT3D_N3);
    return 0;
  }
  return r3d_shard_chunk_decode(&ps->shard[pz][py][px], cz, cy, cx, out);
}

static int make_chunk_l0(const parent_set *ps, uint32_t bz, uint32_t by, uint32_t bx,
                         uint32_t lz, uint32_t ly, uint32_t lx, uint8_t out[DCT3D_N3]) {
  return decode_parent_chunk(ps, 0, 0, 0, bz * BRICK_CPA + lz, by * BRICK_CPA + ly,
                             bx * BRICK_CPA + lx, out);
}

static int make_chunk_down2(const parent_set *ps, uint32_t bz, uint32_t by, uint32_t bx,
                            uint32_t lz, uint32_t ly, uint32_t lx, uint8_t out[DCT3D_N3]) {
  uint32_t pz = bz / 4u, py = by / 4u, px = bx / 4u;
  uint32_t cz0 = (bz % 4u) * 16u + lz * 2u;
  uint32_t cy0 = (by % 4u) * 16u + ly * 2u;
  uint32_t cx0 = (bx % 4u) * 16u + lx * 2u;
  uint8_t in[8][DCT3D_N3];
  for (uint32_t dz = 0; dz < 2; dz++)
    for (uint32_t dy = 0; dy < 2; dy++)
      for (uint32_t dx = 0; dx < 2; dx++)
        if (decode_parent_chunk(ps, pz, py, px, cz0 + dz, cy0 + dy, cx0 + dx,
                                in[(dz * 2u + dy) * 2u + dx]) != 0)
          return -1;
  for (uint32_t z = 0; z < INNER; z++)
    for (uint32_t y = 0; y < INNER; y++)
      for (uint32_t x = 0; x < INNER; x++) {
        uint32_t sum = 4u;
        for (uint32_t dz = 0; dz < 2; dz++)
          for (uint32_t dy = 0; dy < 2; dy++)
            for (uint32_t dx = 0; dx < 2; dx++) {
              uint32_t zz = z * 2u + dz, yy = y * 2u + dy, xx = x * 2u + dx;
              uint32_t si = (((zz >> 4u) * 2u + (yy >> 4u)) * 2u + (xx >> 4u));
              sum += in[si][(((zz & 15u) * INNER + (yy & 15u)) * INNER + (xx & 15u))];
            }
        out[(z * INNER + y) * INNER + x] = (uint8_t)(sum / 8u);
      }
  return 0;
}

static void scatter_chunk(uint8_t *brick, uint32_t lz, uint32_t ly, uint32_t lx,
                          const uint8_t chunk[DCT3D_N3]) {
  for (uint32_t z = 0; z < INNER; z++)
    for (uint32_t y = 0; y < INNER; y++)
      memcpy(brick + (((size_t)(lz * INNER + z) * BRICK + ly * INNER + y) * BRICK +
                      lx * INNER),
             chunk + (z * INNER + y) * INNER, INNER);
}

static void free_blob(blob *b) {
  free(b->p);
  *b = (blob){0};
}

static void *brick_worker(void *arg) {
  shard_job *j = arg;
  uint8_t *raw = malloc((size_t)BRICK * BRICK * BRICK);
  uint8_t chunk[DCT3D_N3], recon[DCT3D_N3], enc[DCT3D_MAX_BYTES];
  if (!raw) {
    atomic_store(&j->failed, 1);
    return NULL;
  }
  for (;;) {
    uint32_t b = atomic_fetch_add(&j->next, 1);
    if (b >= NBRICKS || atomic_load(&j->failed)) break;
    uint32_t bz = b / (SHARD_BPA * SHARD_BPA), by = (b / SHARD_BPA) % SHARD_BPA,
             bx = b % SHARD_BPA;
    memset(raw, 0, (size_t)BRICK * BRICK * BRICK);
    for (uint32_t lz = 0; lz < BRICK_CPA; lz++)
      for (uint32_t ly = 0; ly < BRICK_CPA; ly++)
        for (uint32_t lx = 0; lx < BRICK_CPA; lx++) {
          int rc = j->downsample ? make_chunk_down2(j->parents, bz, by, bx, lz, ly, lx, chunk)
                                 : make_chunk_l0(j->parents, bz, by, bx, lz, ly, lx, chunk);
          if (rc != 0) {
            atomic_store(&j->failed, 1);
            goto worker_done;
          }
          uint32_t gcz = bz * BRICK_CPA + lz, gcy = by * BRICK_CPA + ly,
                   gcx = bx * BRICK_CPA + lx;
          size_t ci = ((size_t)gcz * (SHARD / INNER) + gcy) * (SHARD / INNER) + gcx;
          const uint8_t *for_c5d = chunk;
          if (j->downsample) {
            if (!all_zero(chunk, DCT3D_N3)) {
              size_t n = dct3d_encode_u8(chunk, j->zq, 0.0f, j->ztau, enc);
              uint8_t *copy = malloc(n);
              if (!copy || !dct3d_decode_u8(enc, n, recon)) {
                free(copy);
                atomic_store(&j->failed, 1);
                goto worker_done;
              }
              memcpy(copy, enc, n);
              j->zchunks[ci] = (blob){copy, (uint32_t)n};
              for_c5d = recon; /* c5d is a transcode of decoded Zarr */
            } else {
              j->zchunks[ci] = (blob){0}; /* Zarr missing == fill zero */
            }
          }
          scatter_chunk(raw, lz, ly, lx, for_c5d);
        }
    if (all_zero(raw, (size_t)BRICK * BRICK * BRICK)) {
      j->czero[b] = 1;
      continue;
    }
    c5d_brick_params p = c5d_brick_defaults(1.0f);
    size_t encoded_n = 0;
    p.q = j->c5quality;
    if (c5d_brick_encode(&p, raw, BRICK, &j->cbricks[b].p, &encoded_n) != 0 ||
        encoded_n > UINT32_MAX) {
      atomic_store(&j->failed, 1);
      goto worker_done;
    }
    j->cbricks[b].n = (uint32_t)encoded_n;
  }
worker_done:
  free(raw);
  return NULL;
}

static int write_zarr_shard(const char *path, blob *chunks) {
  if (mkdirs(path, false) != 0) return -1;
  char tmp[2112];
  snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
  FILE *f = fopen(tmp, "wb");
  uint8_t *idx = malloc(ZARR_INDEX_BYTES + 4u);
  if (!f || !idx) {
    if (f) fclose(f);
    free(idx);
    unlink(tmp);
    return -1;
  }
  uint64_t off = 0;
  int rc = 0;
  for (size_t ci = 0; ci < NCHUNKS; ci++) {
    if (!chunks[ci].p) {
      memset(idx + ci * 16u, 0xff, 16u);
      continue;
    }
    if (fwrite(chunks[ci].p, 1, chunks[ci].n, f) != chunks[ci].n) {
      rc = -1;
      break;
    }
    put_u64le(idx + ci * 16u, off);
    put_u64le(idx + ci * 16u + 8u, chunks[ci].n);
    off += chunks[ci].n;
  }
  if (rc == 0) {
    put_u32le(idx + ZARR_INDEX_BYTES, c5d_crc32c(idx, ZARR_INDEX_BYTES));
    if (fwrite(idx, 1, ZARR_INDEX_BYTES + 4u, f) != ZARR_INDEX_BYTES + 4u || fflush(f) != 0 ||
        fsync(fileno(f)) != 0)
      rc = -1;
  }
  free(idx);
  if (fclose(f) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

static int write_c5d_shard(const char *path, uint32_t level, blob *bricks, uint8_t *zero) {
  if (mkdirs(path, false) != 0) return -1;
  char tmp[2112];
  snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
  c5d_shard_writer *w = c5d_shard_create(tmp, SHARD, BRICK, level, 0.0f);
  if (!w) return -1;
  int rc = 0;
  for (uint32_t b = 0; b < NBRICKS && rc == 0; b++)
    rc = zero[b] ? c5d_shard_put_zero(w, b)
                 : c5d_shard_put(w, b, bricks[b].p, bricks[b].n);
  if (c5d_shard_close(w) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

static int process_shard(const char *source, const char *out, dims3 base, uint32_t level,
                         uint64_t oz, uint64_t oy, uint64_t ox, uint32_t threads,
                         float c5r0, bool c5d_only, bool force) {
  bool downsample = level > 0 && !c5d_only;
  uint32_t src_level = downsample ? level - 1u : level;
  dims3 src_grid = shard_grid(level_shape(base, src_level));
  char zp[2048], cp[2048];
  zarr_shard_path(zp, out, level, oz, oy, ox);
  c5d_shard_path(cp, out, level, oz, oy, ox);
  if (!force && (!downsample || file_exists(zp)) && file_exists(cp)) return 1;

  parent_set parents;
  if (open_parents(&parents, source, out, src_level, src_grid, oz, oy, ox, downsample) != 0) {
    close_parents(&parents);
    return -1;
  }
  blob *zchunks = downsample ? calloc(NCHUNKS, sizeof *zchunks) : NULL;
  blob *cbricks = calloc(NBRICKS, sizeof *cbricks);
  uint8_t *czero = calloc(NBRICKS, 1);
  if ((downsample && !zchunks) || !cbricks || !czero) {
    close_parents(&parents);
    free(zchunks);
    free(cbricks);
    free(czero);
    return -1;
  }
  float zq = 8.0f / (float)(1u << (level < 4u ? level : 3u));
  if (zq < 1.0f) zq = 1.0f;
  float quality = c5r0 / (float)(1u << (level < 3u ? level : 3u));
  if (quality < 0.25f) quality = 0.25f;
  shard_job job = {.parents = &parents,
                   .zchunks = zchunks,
                   .cbricks = cbricks,
                   .czero = czero,
                   .zq = zq,
                   .ztau = zq * 2.0f,
                   .c5quality = quality,
                   .level = level,
                   .downsample = downsample};
  uint32_t nt = threads ? threads : 1u;
  if (nt > 32u) nt = 32u;
  pthread_t tids[32];
  uint32_t created = 0;
  for (; created < nt; created++)
    if (pthread_create(&tids[created], NULL, brick_worker, &job) != 0) {
      atomic_store(&job.failed, 1);
      break;
    }
  for (uint32_t t = 0; t < created; t++) pthread_join(tids[t], NULL);

  int rc = atomic_load(&job.failed) ? -1 : 0;
  if (rc == 0 && downsample) rc = write_zarr_shard(zp, zchunks);
  if (rc == 0) rc = write_c5d_shard(cp, level, cbricks, czero);
  for (size_t i = 0; i < NCHUNKS; i++)
    if (zchunks) free_blob(&zchunks[i]);
  for (uint32_t i = 0; i < NBRICKS; i++) free_blob(&cbricks[i]);
  free(zchunks);
  free(cbricks);
  free(czero);
  close_parents(&parents);
  return rc;
}

static int write_level_metadata(const char *out, dims3 shape, uint32_t level) {
  float q = 8.0f / (float)(1u << (level < 4u ? level : 3u));
  if (q < 1.0f) q = 1.0f;
  char json[2048], path[2048];
  int n = snprintf(
      json, sizeof json,
      "{\n  \"zarr_format\": 3,\n  \"node_type\": \"array\",\n"
      "  \"shape\": [%llu, %llu, %llu],\n  \"data_type\": \"uint8\",\n"
      "  \"chunk_grid\": {\"name\": \"regular\", \"configuration\": "
      "{\"chunk_shape\": [1024, 1024, 1024]}},\n"
      "  \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": "
      "{\"separator\": \"/\"}},\n  \"fill_value\": 0,\n"
      "  \"codecs\": [{\"name\": \"sharding_indexed\", \"configuration\": {\n"
      "    \"chunk_shape\": [16, 16, 16],\n"
      "    \"codecs\": [{\"name\": \"dct3d\", \"configuration\": "
      "{\"quality\": %.6g, \"max_error\": 0.0, \"tau\": %.6g}}],\n"
      "    \"index_codecs\": [{\"name\": \"bytes\", \"configuration\": "
      "{\"endian\": \"little\"}}, {\"name\": \"crc32c\"}],\n"
      "    \"index_location\": \"end\"\n  }}]\n}\n",
      (unsigned long long)shape.z, (unsigned long long)shape.y,
      (unsigned long long)shape.x, (double)q, (double)(q * 2.0f));
  if (n < 0 || (size_t)n >= sizeof json) return -1;
  snprintf(path, sizeof path, "%s/zarr/L%u/zarr.json", out, level);
  return write_atomic(path, json, (size_t)n);
}

static int link_l0(const char *source, const char *out, dims3 base) {
  dims3 grid = shard_grid(base);
  for (uint64_t z = 0; z < grid.z; z++)
    for (uint64_t y = 0; y < grid.y; y++)
      for (uint64_t x = 0; x < grid.x; x++) {
        char src[2048], dst[2048];
        source_path(src, source, out, 0, z, y, x);
        zarr_shard_path(dst, out, 0, z, y, x);
        if (file_exists(dst)) continue;
        if (mkdirs(dst, false) != 0 || link(src, dst) != 0) {
          fprintf(stderr, "lodpack: cannot hard-link %s -> %s: %s\n", src, dst,
                  strerror(errno));
          return -1;
        }
      }
  return write_level_metadata(out, base, 0);
}

static int write_manifests(const char *out, dims3 base, uint32_t levels, float c5r0) {
  char *json = calloc(1, 16384);
  char *group = calloc(1, 16384);
  if (!json || !group) {
    free(json);
    free(group);
    return -1;
  }
  size_t n = (size_t)snprintf(json, 16384,
                              "{\n  \"format\": \"render3d.c5d-lod.v1\",\n"
                              "  \"shape\": [%llu, %llu, %llu],\n"
                              "  \"shard_shape\": [1024, 1024, 1024],\n"
                              "  \"brick_shape\": [128, 128, 128],\n  \"levels\": [\n",
                              (unsigned long long)base.z, (unsigned long long)base.y,
                              (unsigned long long)base.x);
  size_t gn = (size_t)snprintf(
      group, 16384,
      "{\n  \"zarr_format\": 3,\n  \"node_type\": \"group\",\n  \"attributes\": {\n"
      "    \"multiscales\": [{\"version\": \"0.4\", \"name\": \"PHerc1218\",\n"
      "      \"axes\": [{\"name\": \"z\", \"type\": \"space\"}, "
      "{\"name\": \"y\", \"type\": \"space\"}, "
      "{\"name\": \"x\", \"type\": \"space\"}],\n      \"datasets\": [\n");
  for (uint32_t l = 0; l < levels; l++) {
    dims3 s = level_shape(base, l), g = shard_grid(s);
    float quality = c5r0 / (float)(1u << (l < 3u ? l : 3u));
    if (quality < 0.25f) quality = 0.25f;
    n += (size_t)snprintf(json + n, 16384 - n,
                          "    {\"level\": %u, \"scale\": %u, "
                          "\"shape\": [%llu, %llu, %llu], "
                          "\"shards\": [%llu, %llu, %llu], "
                          "\"zarr\": \"zarr/L%u\", "
                          "\"c5d\": \"c5d/L%u/{z}_{y}_{x}.c5s\", "
                          "\"c5d_quality\": %.6g}%s\n",
                          l, 1u << l, (unsigned long long)s.z, (unsigned long long)s.y,
                          (unsigned long long)s.x, (unsigned long long)g.z,
                          (unsigned long long)g.y, (unsigned long long)g.x, l, l,
                          (double)quality, l + 1u == levels ? "" : ",");
    gn += (size_t)snprintf(
        group + gn, 16384 - gn,
        "        {\"path\": \"L%u\", \"coordinateTransformations\": "
        "[{\"type\": \"scale\", \"scale\": [%u, %u, %u]}]}%s\n",
        l, 1u << l, 1u << l, 1u << l, l + 1u == levels ? "" : ",");
  }
  n += (size_t)snprintf(json + n, 16384 - n, "  ]\n}\n");
  gn += (size_t)snprintf(group + gn, 16384 - gn, "      ]\n    }]\n  }\n}\n");
  char mp[2048], gp[2048];
  snprintf(mp, sizeof mp, "%s/manifest.json", out);
  snprintf(gp, sizeof gp, "%s/zarr/zarr.json", out);
  int rc = n < 16384 && gn < 16384 && write_atomic(mp, json, n) == 0 &&
                   write_atomic(gp, group, gn) == 0
               ? 0
               : -1;
  free(json);
  free(group);
  return rc;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr,
            "usage: lodpack <source-flat-dir> <output-dir> <nz> <ny> <nx> <max-level> "
            "[--threads N] [--c5d-quality Q] [--only-level L] [--limit N] "
            "[--only-shard z,y,x] [--skip-c5d0] [--c5d-only] [--force]\n");
    return 2;
  }
  const char *source = argv[1], *out = argv[2];
  dims3 base = {strtoull(argv[3], NULL, 10), strtoull(argv[4], NULL, 10),
                strtoull(argv[5], NULL, 10)};
  uint32_t max_level = (uint32_t)strtoul(argv[6], NULL, 10);
  uint32_t threads = 0, only_level = UINT32_MAX;
  uint64_t limit = UINT64_MAX;
  dims3 only_shard = {0};
  bool have_only_shard = false;
  float c5r0 = 2.0f;
  bool skip_c5d0 = false;
  bool c5d_only = false, force = false;
  for (int i = 7; i < argc; i++) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
      threads = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--c5d-quality") == 0 && i + 1 < argc)
      c5r0 = strtof(argv[++i], NULL);
    else if (strcmp(argv[i], "--only-level") == 0 && i + 1 < argc)
      only_level = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc)
      limit = strtoull(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--only-shard") == 0 && i + 1 < argc) {
      unsigned long long z, y, x;
      if (sscanf(argv[++i], "%llu,%llu,%llu", &z, &y, &x) != 3) return 2;
      only_shard = (dims3){z, y, x};
      have_only_shard = true;
    }
    else if (strcmp(argv[i], "--skip-c5d0") == 0)
      skip_c5d0 = true;
    else if (strcmp(argv[i], "--c5d-only") == 0)
      c5d_only = true;
    else if (strcmp(argv[i], "--force") == 0)
      force = true;
    else {
      fprintf(stderr, "lodpack: unknown/incomplete option %s\n", argv[i]);
      return 2;
    }
  }
  if (!base.z || !base.y || !base.x || max_level >= MAX_LEVELS || c5r0 <= 0.0f) return 2;
  if (!threads) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    threads = ncpu > 0 ? (uint32_t)ncpu : 1u;
  }
  if (mkdirs(out, true) != 0 || link_l0(source, out, base) != 0 ||
      write_manifests(out, base, max_level + 1u, c5r0) != 0) {
    fprintf(stderr, "lodpack: cannot initialise output metadata/layout\n");
    return 1;
  }

  for (uint32_t level = 0; level <= max_level; level++) {
    if (level == 0 && skip_c5d0) continue;
    if (only_level != UINT32_MAX && level != only_level) continue;
    dims3 shape = level_shape(base, level), grid = shard_grid(shape);
    if (level > 0 && write_level_metadata(out, shape, level) != 0) return 1;
    if (have_only_shard &&
        (only_shard.z >= grid.z || only_shard.y >= grid.y || only_shard.x >= grid.x)) {
      fprintf(stderr, "lodpack: --only-shard is outside L%u's shard grid\n", level);
      return 2;
    }
    uint64_t total = have_only_shard ? 1u : grid.z * grid.y * grid.x;
    uint64_t upto = total < limit ? total : limit;
    printf("lodpack: L%u shape=%llux%llux%llu shards=%llux%llux%llu (%llu), threads=%u\n",
           level, (unsigned long long)shape.z, (unsigned long long)shape.y,
           (unsigned long long)shape.x, (unsigned long long)grid.z,
           (unsigned long long)grid.y, (unsigned long long)grid.x,
           (unsigned long long)upto, threads);
    fflush(stdout);
    uint64_t done = 0, skipped = 0;
    double started = now_seconds();
    for (uint64_t i = 0; i < upto; i++) {
      uint64_t z = have_only_shard ? only_shard.z : i / (grid.y * grid.x);
      uint64_t y = have_only_shard ? only_shard.y : (i / grid.x) % grid.y;
      uint64_t x = have_only_shard ? only_shard.x : i % grid.x;
      double one = now_seconds();
      int rc = process_shard(source, out, base, level, z, y, x, threads, c5r0, c5d_only,
                             force);
      if (rc < 0) {
        fprintf(stderr, "lodpack: L%u shard %llu/%llu/%llu failed\n", level,
                (unsigned long long)z, (unsigned long long)y, (unsigned long long)x);
        return 1;
      }
      skipped += rc > 0;
      done++;
      printf("lodpack: L%u %llu/%llu (%s, %.1fs; %.1fs/shard overall)\n", level,
             (unsigned long long)done, (unsigned long long)upto,
             rc > 0 ? "resume-skip" : "written", now_seconds() - one,
             (now_seconds() - started) / (double)done);
      fflush(stdout);
    }
    printf("lodpack: L%u complete (%llu written, %llu skipped)\n", level,
           (unsigned long long)(done - skipped), (unsigned long long)skipped);
  }
  return 0;
}
