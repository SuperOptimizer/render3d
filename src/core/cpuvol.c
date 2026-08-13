#include "core/cpuvol.h"

#include <blosc.h>
#include <curl/curl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <brick.h> /* c5d brick decode */
#include <shard.h> /* c5d .c5s reader */

#define CV_BRICK 128u
#define CV_SHARD_BPA 8u /* 1024^3 shards = 8^3 bricks */
#define CV_RAW ((size_t)CV_BRICK * CV_BRICK * CV_BRICK)

typedef struct cv_reader {
  c5d_shard_reader sr;
  bool open, failed;
} cv_reader;

static int cv_u64_triplet(const char *p, uint64_t out[3]) {
  const char *b = strchr(p, '[');
  if (!b) return -1;
  return sscanf(b + 1, " %llu , %llu , %llu", (unsigned long long *)&out[0],
                (unsigned long long *)&out[1], (unsigned long long *)&out[2]) == 3
             ? 0
             : -1;
}

int r3d_cpuvol_open(r3d_cpuvol *v, const char *root, uint32_t cache_bricks) {
  memset(v, 0, sizeof *v);
  snprintf(v->root, sizeof v->root, "%s", root);
  char mp[1200];
  snprintf(mp, sizeof mp, "%s/manifest.json", root);
  FILE *f = fopen(mp, "rb");
  if (!f) return -1;
  char json[65536] = {0};
  size_t jn = fread(json, 1, sizeof json - 1, f);
  fclose(f);
  (void)jn;
  if (!strstr(json, "\"format\": \"render3d.c5d-lod.v1\"")) return -1;
  const char *shape = strstr(json, "\"shape\"");
  uint64_t base[3];
  if (!shape || cv_u64_triplet(shape, base) != 0) return -1;
  v->nz = base[0];
  v->ny = base[1];
  v->nx = base[2]; /* manifest order z,y,x */
  const char *p = strstr(json, "\"levels\"");
  if (!p) return -1;
  uint32_t nread = 0;
  while (v->nlev < R3D_CPUVOL_LEVELS && (p = strstr(p, "\"level\""))) {
    const char *shp = strstr(p, "\"shape\"");
    const char *shd = strstr(p, "\"shards\"");
    uint64_t vd[3], sd[3];
    if (!shp || !shd || cv_u64_triplet(shp, vd) != 0 || cv_u64_triplet(shd, sd) != 0)
      return -1;
    r3d_cpuvol_level *l = &v->lev[v->nlev];
    l->scale = 1u << v->nlev;
    l->vz = (uint32_t)vd[0];
    l->vy = (uint32_t)vd[1];
    l->vx = (uint32_t)vd[2];
    l->bx = (l->vx + CV_BRICK - 1) / CV_BRICK;
    l->by = (l->vy + CV_BRICK - 1) / CV_BRICK;
    l->bz = (l->vz + CV_BRICK - 1) / CV_BRICK;
    l->sz = (uint32_t)sd[0];
    l->sy = (uint32_t)sd[1];
    l->sx = (uint32_t)sd[2];
    l->shard_off = nread;
    nread += l->sx * l->sy * l->sz;
    v->nlev++;
    p += 7;
  }
  if (!v->nlev) return -1;
  v->nreaders = nread;
  v->readers = calloc(nread ? nread : 1, sizeof(cv_reader));
  v->nslots = cache_bricks ? cache_bricks : 64;
  v->slabs = malloc((size_t)v->nslots * CV_RAW);
  v->keys = malloc((size_t)v->nslots * sizeof *v->keys);
  v->use = calloc(v->nslots, sizeof *v->use);
  if (!v->readers || !v->slabs || !v->keys || !v->use) {
    r3d_cpuvol_close(v);
    return -1;
  }
  for (uint32_t i = 0; i < v->nslots; i++) v->keys[i] = UINT64_MAX;
  /* optional net source: chunk fetch config, same file the renderer uses */
  snprintf(mp, sizeof mp, "%s/source.json", root);
  f = fopen(mp, "rb");
  if (f) {
    char sj[16384] = {0};
    size_t sn = fread(sj, 1, sizeof sj - 1, f);
    fclose(f);
    (void)sn;
    const char *up = strstr(sj, "\"url\": \"");
    const char *qp = strstr(sj, "\"quality\": ");
    if (up) {
      up += 8;
      const char *ue = strchr(up, '"');
      if (ue && (size_t)(ue - up) < sizeof v->url) {
        memcpy(v->url, up, (size_t)(ue - up));
        v->url[ue - up] = 0;
      }
      v->q0 = qp ? strtof(qp + 11, NULL) : 2.0f;
      const char *lp = sj;
      for (uint32_t l = 0; l < v->nlev && (lp = strstr(lp, "\"chunk\": ")); l++) {
        v->chsz[l] = (uint32_t)strtoul(lp + 9, NULL, 10);
        const char *rp = strstr(lp, "\"raw\": ");
        v->raw[l] = rp && strncmp(rp + 7, "true", 4) == 0;
        if (v->chsz[l] < 32 || v->chsz[l] > 1024) {
          v->url[0] = 0;
          break;
        }
        lp += 9;
      }
    }
  }
  return 0;
}

void r3d_cpuvol_close(r3d_cpuvol *v) {
  cv_reader *rd = v->readers;
  if (rd)
    for (uint32_t i = 0; i < v->nreaders; i++)
      if (rd[i].open) c5d_shard_close_reader(&rd[i].sr);
  free(v->readers);
  free(v->slabs);
  free(v->keys);
  free(v->use);
  if (v->curl) curl_easy_cleanup(v->curl);
  memset(v, 0, sizeof *v);
}

/* decode brick (li,bx,by,bz) into a cache slot; NULL when absent on disk */
/* --- demand fetch: zarr chunks -> .c5b cache (mirrors ni_worker) --------- */

static size_t cv_curl_write(const void *data, size_t sz, size_t nm, void *ud) {
  struct cvbuf { uint8_t *p; size_t n, cap; } *b = ud;
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

static uint32_t cv_cell_dim(uint32_t chsz) { /* lcm(chsz, brick) */
  uint32_t a = chsz, b = CV_BRICK;
  while (b) {
    uint32_t t = a % b;
    a = b;
    b = t;
  }
  return chsz / a * CV_BRICK;
}

static int cv_write_file(const char *path, const void *data, size_t n) {
  char tmp[1460];
  int pn = snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
  if (pn < 0 || (size_t)pn >= sizeof tmp) return -1;
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  int rc = n && fwrite(data, 1, n, f) != n ? -1 : 0;
  if (fclose(f) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

/* Fetch the cell owning brick (bx,by,bz) at level li and write every brick
 * of the cell to <root>/bricks/L<li> (.c5b; empty file = absent/air). */
static void cv_net_fetch(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by,
                         uint32_t bz) {
  if (!v->url[0] || li >= v->nlev || !v->chsz[li]) return;
  uint64_t now = (uint64_t)time(NULL);
  if (v->net_cool > now) return;
  if (!v->curl) {
    v->curl = curl_easy_init();
    if (!v->curl) return;
    curl_easy_setopt(v->curl, CURLOPT_WRITEFUNCTION, cv_curl_write);
    curl_easy_setopt(v->curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(v->curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(v->curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(v->curl, CURLOPT_LOW_SPEED_TIME, 60L);
  }
  uint32_t chsz = v->chsz[li], cell = cv_cell_dim(chsz), cb = cell / CV_BRICK,
           cc = cell / chsz;
  uint32_t cz = bz / cb, cy = by / cb, cx = bx / cb;
  size_t chunk_bytes = (size_t)chsz * chsz * chsz;
  size_t cell_bytes = (size_t)cell * cell * cell;
  uint8_t *cellbuf = calloc(1, cell_bytes);
  uint8_t *chunk = malloc(chunk_bytes);
  uint8_t *raw = malloc(CV_RAW);
  struct { uint8_t *p; size_t n, cap; } buf = {0};
  if (!cellbuf || !chunk || !raw) goto done;
  bool any = false;
  for (uint32_t icz = 0; icz < cc; icz++)
    for (uint32_t icy = 0; icy < cc; icy++)
      for (uint32_t icx = 0; icx < cc; icx++) {
        char url[1600];
        snprintf(url, sizeof url, "%s/%u/%u/%u/%u", v->url, li, cz * cc + icz,
                 cy * cc + icy, cx * cc + icx);
        long code = 0;
        CURLcode crc = CURLE_OK;
        for (int attempt = 0; attempt < 3; attempt++) {
          buf.n = 0;
          curl_easy_setopt(v->curl, CURLOPT_URL, url);
          curl_easy_setopt(v->curl, CURLOPT_WRITEDATA, &buf);
          crc = curl_easy_perform(v->curl);
          curl_easy_getinfo(v->curl, CURLINFO_RESPONSE_CODE, &code);
          if (crc == CURLE_OK && (code == 200 || code == 404)) break;
          sleep((unsigned)(1u << attempt));
        }
        if (!(crc == CURLE_OK && (code == 200 || code == 404))) {
          v->net_cool = now + 30; /* network trouble: back off, stay air */
          goto done;
        }
        if (code == 404) continue;
        bool ok;
        if (v->raw[li]) {
          ok = buf.n == chunk_bytes;
          if (ok) memcpy(chunk, buf.p, chunk_bytes);
        } else {
          size_t nb = 0, cby = 0, bs = 0;
          blosc_cbuffer_sizes(buf.p, &nb, &cby, &bs);
          ok = nb == chunk_bytes && cby <= buf.n &&
               blosc_decompress_ctx(buf.p, chunk, chunk_bytes, 1) == (int)chunk_bytes;
        }
        if (!ok) continue; /* bad payload reads as air */
        for (uint32_t zz = 0; zz < chsz; zz++)
          for (uint32_t yy = 0; yy < chsz; yy++)
            memcpy(cellbuf + (((size_t)icz * chsz + zz) * cell +
                              ((size_t)icy * chsz + yy)) *
                                 cell +
                       (size_t)icx * chsz,
                   chunk + ((size_t)zz * chsz + yy) * chsz, chsz);
        any = true;
      }
  {
    const r3d_cpuvol_level *l = &v->lev[li];
    char dir[1360];
    snprintf(dir, sizeof dir, "%s/bricks", v->root);
    mkdir(dir, 0755);
    snprintf(dir, sizeof dir, "%s/bricks/L%u", v->root, li);
    mkdir(dir, 0755);
    float q = v->q0 / (float)(1u << (li < 3u ? li : 3u));
    if (q < 0.25f) q = 0.25f;
    for (uint32_t sz2 = 0; sz2 < cb; sz2++)
      for (uint32_t sy2 = 0; sy2 < cb; sy2++)
        for (uint32_t sx2 = 0; sx2 < cb; sx2++) {
          uint32_t obz = cz * cb + sz2, oby = cy * cb + sy2, obx = cx * cb + sx2;
          if (obx >= l->bx || oby >= l->by || obz >= l->bz) continue;
          char path[1500];
          snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", v->root, li, obz,
                   oby, obx);
          struct stat st;
          if (stat(path, &st) == 0) continue; /* another fetcher won */
          bool zero = !any;
          if (any) {
            for (uint32_t rr = 0; rr < CV_BRICK; rr++)
              for (uint32_t qq = 0; qq < CV_BRICK; qq++)
                memcpy(raw + ((size_t)rr * CV_BRICK + qq) * CV_BRICK,
                       cellbuf + (((size_t)(sz2 * CV_BRICK + rr) * cell +
                                   sy2 * CV_BRICK + qq) *
                                      cell +
                                  sx2 * CV_BRICK),
                       CV_BRICK);
            zero = true;
            for (size_t k = 0; k < (size_t)CV_RAW; k++)
              if (raw[k]) {
                zero = false;
                break;
              }
          }
          if (zero) {
            cv_write_file(path, NULL, 0);
            continue;
          }
          c5d_brick_params bp = c5d_brick_defaults(1.0f);
          bp.q = q;
          uint8_t *enc = NULL;
          size_t en = 0;
          if (c5d_brick_encode(&bp, raw, CV_BRICK, &enc, &en) == 0) {
            cv_write_file(path, enc, en);
            free(enc);
          }
        }
  }
done:
  free(cellbuf);
  free(chunk);
  free(raw);
  free(buf.p);
}

static const uint8_t *cv_brick(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by,
                               uint32_t bz) {
  uint64_t key = ((uint64_t)li << 60) | ((uint64_t)bz << 40) | ((uint64_t)by << 20) | bx;
  static _Thread_local uint64_t memo_key = UINT64_MAX;
  static _Thread_local const uint8_t *memo_ptr = NULL;
  static _Thread_local const r3d_cpuvol *memo_vol = NULL;
  if (key == memo_key && v == memo_vol) return memo_ptr; /* hot path */
  uint32_t victim = 0;
  uint64_t oldest = UINT64_MAX;
  for (uint32_t s = 0; s < v->nslots; s++) {
    if (v->keys[s] == key) {
      v->use[s] = ++v->tick;
      memo_key = key;
      memo_ptr = v->slabs + (size_t)s * CV_RAW;
      memo_vol = v;
      return memo_ptr;
    }
    if (v->use[s] < oldest) {
      oldest = v->use[s];
      victim = s;
    }
  }
  const r3d_cpuvol_level *l = &v->lev[li];
  const uint8_t *blob = NULL;
  size_t bn = 0;
  uint8_t *owned = NULL;
  uint32_t sx = bx / CV_SHARD_BPA, sy = by / CV_SHARD_BPA, sz = bz / CV_SHARD_BPA;
  if (sx < l->sx && sy < l->sy && sz < l->sz) {
    cv_reader *rd =
        (cv_reader *)v->readers + l->shard_off + (sz * l->sy + sy) * l->sx + sx;
    if (!rd->open && !rd->failed) {
      char path[1400];
      snprintf(path, sizeof path, "%s/c5d/L%u/%u_%u_%u.c5s", v->root, li, sz, sy, sx);
      if (c5d_shard_open(path, &rd->sr) == 0 && rd->sr.foot.brick_dim == CV_BRICK &&
          rd->sr.foot.shard_dim == 1024u)
        rd->open = true;
      else
        rd->failed = true;
    }
    if (rd->open) {
      uint32_t bi = ((bz % CV_SHARD_BPA) * CV_SHARD_BPA + (by % CV_SHARD_BPA)) *
                        CV_SHARD_BPA +
                    (bx % CV_SHARD_BPA);
      blob = c5d_shard_brick(&rd->sr, bi, &bn);
    }
  }
  if (!blob) { /* net-ingest cache file (empty = absent/air) */
    char path[1400];
    snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", v->root, li, bz, by, bx);
    FILE *bf = fopen(path, "rb");
    if (!bf && v->url[0]) { /* never fetched: pull the owning cell now */
      cv_net_fetch(v, li, bx, by, bz);
      bf = fopen(path, "rb");
    }
    if (bf) {
      fseek(bf, 0, SEEK_END);
      long fn = ftell(bf);
      fseek(bf, 0, SEEK_SET);
      if (fn > 0) {
        owned = malloc((size_t)fn);
        if (owned && fread(owned, 1, (size_t)fn, bf) == (size_t)fn) {
          blob = owned;
          bn = (size_t)fn;
        }
      }
      fclose(bf);
    }
  }
  uint8_t *dst = v->slabs + (size_t)victim * CV_RAW;
  bool ok = blob && c5d_brick_decode(blob, bn, dst, CV_BRICK) == 0;
  free(owned);
  if (!ok) return NULL; /* leave the slot untouched: absent bricks re-probe
                         * cheaply (file stat), no point caching zeros */
  v->keys[victim] = key;
  v->use[victim] = ++v->tick;
  memo_key = key;
  memo_ptr = dst;
  memo_vol = v;
  return dst;
}

static double cv_vox(r3d_cpuvol *v, uint32_t li, int64_t lx, int64_t ly, int64_t lz) {
  const r3d_cpuvol_level *l = &v->lev[li];
  if (lx < 0 || ly < 0 || lz < 0 || lx >= (int64_t)l->vx || ly >= (int64_t)l->vy ||
      lz >= (int64_t)l->vz)
    return 0.0;
  const uint8_t *b =
      cv_brick(v, li, (uint32_t)lx / CV_BRICK, (uint32_t)ly / CV_BRICK, (uint32_t)lz / CV_BRICK);
  if (!b) return 0.0;
  uint32_t ox = (uint32_t)lx % CV_BRICK, oy = (uint32_t)ly % CV_BRICK,
           oz = (uint32_t)lz % CV_BRICK;
  return (double)b[((size_t)oz * CV_BRICK + oy) * CV_BRICK + ox];
}

double r3d_cpuvol_tri(r3d_cpuvol *v, uint32_t li, const double p[3], double grad[3]) {
  if (grad) grad[0] = grad[1] = grad[2] = 0.0;
  if (li >= v->nlev) return 0.0;
  const r3d_cpuvol_level *l = &v->lev[li];
  double lx = p[0] / l->scale, ly = p[1] / l->scale, lz = p[2] / l->scale;
  double fx = floor(lx), fy = floor(ly), fz = floor(lz);
  double tx = lx - fx, ty = ly - fy, tz = lz - fz;
  int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz;
  double c[2][2][2];
  for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
      for (int dx = 0; dx < 2; dx++)
        c[dz][dy][dx] = cv_vox(v, li, ix + dx, iy + dy, iz + dz);
  double c00 = c[0][0][0] * (1 - tx) + c[0][0][1] * tx;
  double c01 = c[0][1][0] * (1 - tx) + c[0][1][1] * tx;
  double c10 = c[1][0][0] * (1 - tx) + c[1][0][1] * tx;
  double c11 = c[1][1][0] * (1 - tx) + c[1][1][1] * tx;
  double c0 = c00 * (1 - ty) + c01 * ty;
  double c1 = c10 * (1 - ty) + c11 * ty;
  if (grad) {
    double gx0 = (c[0][0][1] - c[0][0][0]) * (1 - ty) + (c[0][1][1] - c[0][1][0]) * ty;
    double gx1 = (c[1][0][1] - c[1][0][0]) * (1 - ty) + (c[1][1][1] - c[1][1][0]) * ty;
    grad[0] = (gx0 * (1 - tz) + gx1 * tz) / l->scale;
    grad[1] = ((c01 - c00) * (1 - tz) + (c11 - c10) * tz) / l->scale;
    grad[2] = (c1 - c0) / l->scale;
  }
  return c0 * (1 - tz) + c1 * tz;
}

uint8_t r3d_cpuvol_at(r3d_cpuvol *v, uint32_t li, double x, double y, double z) {
  if (li >= v->nlev || x < 0.0 || y < 0.0 || z < 0.0) return 0;
  const r3d_cpuvol_level *l = &v->lev[li];
  uint32_t lx = (uint32_t)(x / l->scale), ly = (uint32_t)(y / l->scale),
           lz = (uint32_t)(z / l->scale);
  if (lx >= l->vx || ly >= l->vy || lz >= l->vz) return 0;
  const uint8_t *b = cv_brick(v, li, lx / CV_BRICK, ly / CV_BRICK, lz / CV_BRICK);
  if (!b) return 0;
  uint32_t ox = lx % CV_BRICK, oy = ly % CV_BRICK, oz = lz % CV_BRICK;
  return b[((size_t)oz * CV_BRICK + oy) * CV_BRICK + ox];
}
