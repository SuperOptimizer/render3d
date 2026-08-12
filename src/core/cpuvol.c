#include "core/cpuvol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  memset(v, 0, sizeof *v);
}

/* decode brick (li,bx,by,bz) into a cache slot; NULL when absent on disk */
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
