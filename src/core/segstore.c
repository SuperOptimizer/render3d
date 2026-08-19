#include "core/segstore.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <tifxyz.h> /* c5d codec (angle include: render3d's core/tifxyz.h differs) */

/* magic/version bumped together: a checksum and explicit version field were
 * added to the header, so a store written by the previous format is
 * rejected outright (r3d_segstore_open fails) rather than misread; the
 * caller rebuilds from scratch, which is safe because rebuild is itself
 * fail-closed (see r3d_segstore_build). */
#define SGS_MAGIC "r3dsegs2"
#define SGS_VERSION 2u

/* sane ceilings for externally-supplied header counts, checked before any
 * size arithmetic or allocation derived from them: comfortably above any
 * real corpus (the format doc's own "tens of MB" whole-scroll estimate)
 * while staying far below the magnitudes at which count*sizeof(record) can
 * wrap a 64-bit (or narrower) size_t. */
#define SGS_MAX_SEGS ((uint32_t)1 << 20)
#define SGS_MAX_TILES ((uint64_t)1 << 30)
#define SGS_MAX_DIM ((uint32_t)1 << 20)

struct sgs_hdr {
  char magic[8];
  uint32_t version;
  uint32_t count;
  uint32_t tile;
  uint32_t reserved; /* explicit padding, always written 0 */
  uint64_t ntiles;
  uint64_t checksum; /* FNV-1a64 over the segs+tiles bytes following the header */
};

/* checked a*b -> out; false on overflow (out left unmodified) */
static bool sgs_mul_ov(uint64_t a, uint64_t b, uint64_t *out) {
  if (a != 0 && b > UINT64_MAX / a) return false;
  *out = a * b;
  return true;
}

/* checked a+b -> out; false on overflow (out left unmodified) */
static bool sgs_add_ov(uint64_t a, uint64_t b, uint64_t *out) {
  if (a > UINT64_MAX - b) return false;
  *out = a + b;
  return true;
}

static uint64_t sgs_fnv1a(const void *p, size_t n) {
  const unsigned char *b = p;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

/* name identity is the tifxyz dir's basename; truncation would silently
 * alias two different sources onto the same store entry, so reject rather
 * than truncate (false = doesn't fit). */
static bool sgs_basename(const char *dir, char *out, size_t outsz) {
  size_t len = strlen(dir);
  while (len > 1 && dir[len - 1] == '/') len--; /* tolerate trailing slash */
  size_t end = len;
  while (len > 0 && dir[len - 1] != '/') len--;
  size_t n = end - len;
  if (n == 0 || n >= outsz) return false;
  memcpy(out, dir + len, n);
  out[n] = 0;
  return true;
}

/* open-addressed set of names seen so far this build: catches two source
 * dirs (or a source dir and a leftover .tfx) resolving to the same
 * identity, which would otherwise silently alias one .tfx file/manifest
 * slot onto two different segments. */
typedef struct sgs_nameent {
  uint64_t hash;
  char name[R3D_SEGSTORE_NAME];
  bool used;
} sgs_nameent;

typedef struct sgs_nameset {
  sgs_nameent *e;
  uint32_t cap, n;
} sgs_nameset;

enum { SGS_NAME_OK = 0, SGS_NAME_DUP = 1, SGS_NAME_NOMEM = 2 };

static bool sgs_nameset_init(sgs_nameset *ns, uint32_t hint) {
  uint32_t cap = 16;
  while (cap < hint * 2u + 16u) cap <<= 1;
  ns->e = calloc(cap, sizeof *ns->e);
  ns->cap = ns->e ? cap : 0;
  ns->n = 0;
  return ns->e != NULL;
}

static bool sgs_nameset_grow(sgs_nameset *ns) {
  uint32_t ncap = ns->cap * 2;
  sgs_nameent *ne = calloc(ncap, sizeof *ne);
  if (!ne) return false;
  for (uint32_t i = 0; i < ns->cap; i++)
    if (ns->e[i].used) {
      uint32_t k = (uint32_t)(ns->e[i].hash & (ncap - 1));
      while (ne[k].used) k = (k + 1) & (ncap - 1);
      ne[k] = ns->e[i];
    }
  free(ns->e);
  ns->e = ne;
  ns->cap = ncap;
  return true;
}

/* SGS_NAME_DUP if name is already present; SGS_NAME_NOMEM if growth failed
 * (treated by callers as a build-aborting allocation failure); else inserts
 * and returns SGS_NAME_OK. */
static int sgs_nameset_check_insert(sgs_nameset *ns, const char *name) {
  if (ns->n * 2u >= ns->cap && !sgs_nameset_grow(ns)) return SGS_NAME_NOMEM;
  uint64_t h = sgs_fnv1a(name, strlen(name));
  uint32_t k = (uint32_t)(h & (ns->cap - 1));
  while (ns->e[k].used) {
    if (ns->e[k].hash == h && strcmp(ns->e[k].name, name) == 0) return SGS_NAME_DUP;
    k = (k + 1) & (ns->cap - 1);
  }
  ns->e[k].used = true;
  ns->e[k].hash = h;
  snprintf(ns->e[k].name, sizeof ns->e[k].name, "%s", name);
  ns->n++;
  return SGS_NAME_OK;
}

static void sgs_nameset_free(sgs_nameset *ns) {
  free(ns->e);
  memset(ns, 0, sizeof *ns);
}

/* frees every rebuild-scoped resource and returns -1; call at any point
 * where the rebuild must abort so the previous manifest (never touched
 * here) stays the published corpus. */
static int sgs_build_abort(r3d_segmeta *segs, r3d_segtile *tiles, sgs_nameset *names, DIR *dp,
                           r3d_segstore *old, bool have_old) {
  if (dp) closedir(dp);
  if (have_old) r3d_segstore_close(old);
  sgs_nameset_free(names);
  free(tiles);
  free(segs);
  return -1;
}

/* newest mtime among a tifxyz dir's component files, or 0 if none are
 * stat-able (treated by callers as "no freshness signal": an existing .tfx
 * is then only trusted if its own content checks (size) pass). */
static time_t sgs_source_mtime(const char *dir) {
  static const char *comp[4] = {"x.tif", "y.tif", "z.tif", "meta.json"};
  time_t mx = 0;
  for (int i = 0; i < 4; i++) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s", dir, comp[i]);
    struct stat st;
    if (stat(p, &st) == 0 && st.st_mtime > mx) mx = st.st_mtime;
  }
  return mx;
}

/* true if path is a regular file with nonzero size and (when src_mtime > 0)
 * is not older than the source it was packed from. */
static bool sgs_file_fresh(const char *path, time_t src_mtime) {
  struct stat st;
  if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) return false;
  return src_mtime <= 0 || st.st_mtime >= src_mtime;
}

static int sgs_write_file(const char *path, const void *data, size_t n) {
  char tmp[600];
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  int rc = fwrite(data, 1, n, f) == n ? 0 : -1;
  if (fclose(f) != 0) rc = -1;
  if (rc == 0) rc = rename(tmp, path);
  if (rc != 0) remove(tmp);
  return rc;
}

static uint8_t *sgs_read_file(const char *path, size_t *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = n >= 0 ? malloc(n ? (size_t)n : 1) : NULL;
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *n_out = (size_t)n;
  return buf;
}

/* Tile AABBs over valid grid points, segrows-style: a point on a tile's low
 * edge also bounds the previous tile's cells, so any cell's corners are
 * covered by the tiles it spans. Boxes quantize to u16 over the segment
 * bbox (floor for lo, ceil for hi: dequantized boxes only ever grow). */
static void sgs_tiles_build(const r3d_tifxyz *s, const float bbox[2][3], r3d_segtile *tiles,
                            uint32_t tw, uint32_t th) {
  const uint32_t T = R3D_SEGSTORE_TILE;
  float qs[3];
  for (int a = 0; a < 3; a++) {
    qs[a] = (bbox[1][a] - bbox[0][a]) / 65535.0f;
    if (qs[a] <= 0.0f) qs[a] = 1.0f;
  }
  for (uint64_t t = 0; t < (uint64_t)tw * th; t++)
    tiles[t] = (r3d_segtile){.lo = {0xffff, 0xffff, 0xffff}, .hi = {0, 0, 0}};
  for (uint32_t j = 0; j < s->h; j++) {
    uint32_t tj0 = (j ? j - 1 : 0) / T, tj1 = j / T;
    for (uint32_t i = 0; i < s->w; i++) {
      const float *p = r3d_tifxyz_at(s, i, j);
      if (!r3d_tifxyz_valid(p)) continue;
      uint32_t ti0 = (i ? i - 1 : 0) / T, ti1 = i / T;
      uint16_t q0[3], q1[3];
      for (int a = 0; a < 3; a++) {
        float v = (p[a] - bbox[0][a]) / qs[a];
        float lo = floorf(v), hi = ceilf(v);
        q0[a] = (uint16_t)(lo < 0.0f ? 0.0f : (lo > 65535.0f ? 65535.0f : lo));
        q1[a] = (uint16_t)(hi < 0.0f ? 0.0f : (hi > 65535.0f ? 65535.0f : hi));
      }
      for (uint32_t tj = tj0; tj <= tj1; tj++)
        for (uint32_t ti = ti0; ti <= ti1; ti++) {
          r3d_segtile *tl = &tiles[(uint64_t)tj * tw + ti];
          for (int a = 0; a < 3; a++) {
            if (q0[a] < tl->lo[a]) tl->lo[a] = q0[a];
            if (q1[a] > tl->hi[a]) tl->hi[a] = q1[a];
          }
        }
    }
  }
}

/* encode s decimated by stride into path (skipped if it already exists,
 * is nonempty, and is not older than src_mtime, unless force); the
 * decimated grid keeps exact source points so quantization error never
 * compounds across tiers. src_mtime <= 0 means "no source to compare
 * against" (existence + nonzero size only). */
static int sgs_encode_grid(const r3d_tifxyz *s, uint32_t stride, const uint8_t *meta,
                           size_t meta_len, int log2q, const char *path, bool force,
                           time_t src_mtime) {
  if (!force && sgs_file_fresh(path, src_mtime)) return 0;
  uint32_t w = (s->w + stride - 1) / stride, h = (s->h + stride - 1) / stride;
  uint64_t np = (uint64_t)w * h;
  float *planes = malloc(np * 3 * sizeof *planes);
  if (!planes) return -1;
  c5d_tifxyz ct = {.w = w, .h = h, .meta = (uint8_t *)meta, .meta_len = meta_len};
  for (uint64_t a = 0; a < 3; a++) ct.plane[a] = planes + a * np;
  for (uint32_t j = 0; j < h; j++)
    for (uint32_t i = 0; i < w; i++) {
      const float *p = s->xyz + ((uint64_t)(j * stride) * s->w + (uint64_t)i * stride) * 3;
      uint64_t k = (uint64_t)j * w + i;
      ct.plane[0][k] = p[0];
      ct.plane[1][k] = p[1];
      ct.plane[2][k] = p[2];
    }
  uint8_t *enc = NULL;
  size_t enc_n = 0;
  int rc = c5d_tifxyz_encode(&ct, log2q, &enc, &enc_n);
  if (rc == 0) rc = sgs_write_file(path, enc, enc_n);
  free(enc);
  free(planes);
  return rc;
}

/* decode a .tfx into an owned r3d_tifxyz; sx/sy parsed from the carried
 * meta.json ("scale": [sx, sy]) */
static int sgs_decode_tfx(const char *path, r3d_tifxyz *out) {
  size_t enc_n = 0;
  uint8_t *enc = sgs_read_file(path, &enc_n);
  if (!enc) return -1;
  c5d_tifxyz ct;
  int rc = c5d_tifxyz_decode(enc, enc_n, &ct);
  free(enc);
  if (rc != 0) return -1;
  memset(out, 0, sizeof *out);
  out->w = ct.w;
  out->h = ct.h;
  out->sx = out->sy = 0.05f;
  if (ct.meta && ct.meta_len) {
    char *mz = malloc(ct.meta_len + 1);
    if (mz) {
      memcpy(mz, ct.meta, ct.meta_len);
      mz[ct.meta_len] = 0;
      const char *sc = strstr(mz, "\"scale\"");
      double a = 0.0, b = 0.0;
      if (sc && sscanf(sc, "\"scale\"%*[^0-9.-]%lf%*[^0-9.-]%lf", &a, &b) == 2 && a > 0.0 &&
          b > 0.0) {
        out->sx = (float)a;
        out->sy = (float)b;
      }
      free(mz);
    }
  }
  uint64_t np = (uint64_t)ct.w * ct.h;
  out->xyz = malloc(np * 3 * sizeof *out->xyz);
  if (!out->xyz) {
    c5d_tifxyz_free(&ct);
    return -1;
  }
  float bb[2][3] = {{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
  uint64_t nvalid = 0;
  for (uint64_t k = 0; k < np; k++) {
    float *q = out->xyz + k * 3;
    q[0] = ct.plane[0][k];
    q[1] = ct.plane[1][k];
    q[2] = ct.plane[2][k];
    if (q[0] != -1.0f) {
      nvalid++;
      for (int a = 0; a < 3; a++) {
        if (q[a] < bb[0][a]) bb[0][a] = q[a];
        if (q[a] > bb[1][a]) bb[1][a] = q[a];
      }
    }
  }
  memcpy(out->bbox, bb, sizeof bb);
  out->nvalid = nvalid;
  c5d_tifxyz_free(&ct);
  return 0;
}

int r3d_segstore_build(const char *store_dir, const char *const *dirs, uint32_t ndirs,
                       int log2q, bool force) {
  uint32_t segs_cap = ndirs ? ndirs : 4;
  r3d_segmeta *segs = calloc(segs_cap, sizeof *segs);
  r3d_segtile *tiles = NULL;
  uint64_t ntiles = 0, tiles_cap = 0;
  uint32_t n = 0;
  sgs_nameset names;
  r3d_segstore old = {0};
  bool have_old = false;
  if (!segs) return -1;
  if (!sgs_nameset_init(&names, ndirs)) {
    free(segs);
    return -1;
  }
  for (uint32_t d = 0; d < ndirs; d++) {
    r3d_tifxyz s;
    if (r3d_tifxyz_load(&s, dirs[d]) != 0) {
      fprintf(stderr, "segstore: skipping %s (load failed)\n", dirs[d]);
      continue;
    }
    if (s.nvalid == 0) {
      fprintf(stderr, "segstore: skipping %s (no valid points)\n", dirs[d]);
      r3d_tifxyz_free(&s);
      continue;
    }
    char name[R3D_SEGSTORE_NAME];
    if (!sgs_basename(dirs[d], name, sizeof name)) {
      fprintf(stderr, "segstore: skipping %s (basename doesn't fit the %u-byte identity)\n",
              dirs[d], R3D_SEGSTORE_NAME - 1);
      r3d_tifxyz_free(&s);
      continue;
    }
    int nr = sgs_nameset_check_insert(&names, name);
    if (nr == SGS_NAME_NOMEM) {
      r3d_tifxyz_free(&s);
      return sgs_build_abort(segs, tiles, &names, NULL, &old, have_old);
    }
    if (nr == SGS_NAME_DUP) {
      fprintf(stderr, "segstore: skipping %s (name collides with another source this run)\n",
              dirs[d]);
      r3d_tifxyz_free(&s);
      continue;
    }
    r3d_segmeta *m = &segs[n];
    snprintf(m->name, sizeof m->name, "%s", name);
    m->w = s.w;
    m->h = s.h;
    m->sx = s.sx;
    m->sy = s.sy;
    memcpy(m->bbox, s.bbox, sizeof m->bbox);
    m->nvalid = s.nvalid;
    m->tw = (s.w + R3D_SEGSTORE_TILE - 1) / R3D_SEGSTORE_TILE;
    m->th = (s.h + R3D_SEGSTORE_TILE - 1) / R3D_SEGSTORE_TILE;
    m->tile_ofs = ntiles;
    uint64_t nt = (uint64_t)m->tw * m->th;
    if (ntiles + nt > tiles_cap) {
      tiles_cap = (ntiles + nt) * 2;
      r3d_segtile *nt2 = realloc(tiles, tiles_cap * sizeof *nt2);
      if (!nt2) {
        r3d_tifxyz_free(&s);
        return sgs_build_abort(segs, tiles, &names, NULL, &old, have_old);
      }
      tiles = nt2;
    }
    sgs_tiles_build(&s, m->bbox, tiles + ntiles, m->tw, m->th);
    ntiles += nt;

    char path[600], p4[600], mpath[600];
    snprintf(path, sizeof path, "%s/%s.tfx", store_dir, m->name);
    snprintf(p4, sizeof p4, "%s/%s.tfx4", store_dir, m->name);
    snprintf(mpath, sizeof mpath, "%s/meta.json", dirs[d]);
    size_t meta_len = 0;
    uint8_t *meta = sgs_read_file(mpath, &meta_len);
    time_t src_mtime = sgs_source_mtime(dirs[d]);
    /* full-res + stride-4 tier (fast overview decodes); the loader already
     * normalized invalids to exact (-1,-1,-1) incl. the z<=0 rule */
    int rc = sgs_encode_grid(&s, 1, meta, meta_len, log2q, path, force, src_mtime);
    if (rc == 0) rc = sgs_encode_grid(&s, 4, meta, meta_len, log2q, p4, force, src_mtime);
    free(meta);
    r3d_tifxyz_free(&s);
    if (rc != 0) {
      fprintf(stderr, "segstore: encode failed for %s\n", dirs[d]);
      return sgs_build_abort(segs, tiles, &names, NULL, &old, have_old);
    }
    n++;
  }
  /* store segments whose source dirs weren't given this run: keep them,
   * reusing the previous manifest entry when possible (so packed sources
   * can be deleted), else rebuild the entry from the .tfx itself. Any
   * allocation/encode failure from here on aborts the whole rebuild
   * (sgs_build_abort) rather than publishing a manifest missing whatever
   * hadn't been reached yet. */
  have_old = r3d_segstore_open(&old, store_dir) == 0;
  DIR *dp = opendir(store_dir);
  struct dirent *de;
  while (dp && (de = readdir(dp)) != NULL) {
    size_t nl = strlen(de->d_name);
    if (nl < 5 || nl - 4 >= R3D_SEGSTORE_NAME || strcmp(de->d_name + nl - 4, ".tfx") != 0)
      continue;
    char name[R3D_SEGSTORE_NAME];
    memcpy(name, de->d_name, nl - 4);
    name[nl - 4] = 0;
    int nr = sgs_nameset_check_insert(&names, name);
    if (nr == SGS_NAME_NOMEM) return sgs_build_abort(segs, tiles, &names, dp, &old, have_old);
    if (nr == SGS_NAME_DUP) continue; /* already added this run */
    if (n == segs_cap) {
      segs_cap *= 2;
      r3d_segmeta *ns = realloc(segs, (size_t)segs_cap * sizeof *ns);
      if (!ns) return sgs_build_abort(segs, tiles, &names, dp, &old, have_old);
      segs = ns;
    }
    r3d_segmeta *m = &segs[n];
    uint32_t oi = UINT32_MAX;
    if (have_old)
      for (uint32_t i = 0; i < old.n && oi == UINT32_MAX; i++)
        if (strcmp(old.segs[i].name, name) == 0) oi = i;
    char path[600], p4[600];
    snprintf(path, sizeof path, "%s/%s", store_dir, de->d_name);
    snprintf(p4, sizeof p4, "%s/%s.tfx4", store_dir, name);
    /* no source dir this run to compare mtimes against: reuse is gated on
     * existence + nonzero size only */
    bool primary_ok = sgs_file_fresh(path, 0);
    bool have4_ok = sgs_file_fresh(p4, 0);
    r3d_tifxyz s = {0};
    bool decoded = false;
    if (!(oi != UINT32_MAX && have4_ok && primary_ok)) { /* need the grid:
                                       * entry rebuild or tier backfill */
      bool ok = primary_ok && sgs_decode_tfx(path, &s) == 0 && s.nvalid != 0;
      if (!ok) {
        r3d_tifxyz_free(&s);
        if (oi == UINT32_MAX) {
          fprintf(stderr, "segstore: skipping stale %s (unreadable, no previous entry)\n",
                  de->d_name);
          continue;
        }
        /* a transient read/decode failure here must not silently drop a
         * segment that was already in the published corpus */
        fprintf(stderr, "segstore: keeping %s from previous manifest (packed grid unreadable this run)\n",
                name);
      } else {
        decoded = true;
        if (!have4_ok) {
          int rc4 = sgs_encode_grid(&s, 4, NULL, 0, log2q, p4, false, 0);
          if (rc4 != 0) {
            r3d_tifxyz_free(&s);
            return sgs_build_abort(segs, tiles, &names, dp, &old, have_old);
          }
        }
      }
    }
    if (oi == UINT32_MAX) { /* only reachable with a successful fresh decode */
      memset(m, 0, sizeof *m);
      snprintf(m->name, sizeof m->name, "%s", name);
      m->w = s.w;
      m->h = s.h;
      m->sx = s.sx;
      m->sy = s.sy;
      memcpy(m->bbox, s.bbox, sizeof m->bbox);
      m->nvalid = s.nvalid;
      m->tw = (s.w + R3D_SEGSTORE_TILE - 1) / R3D_SEGSTORE_TILE;
      m->th = (s.h + R3D_SEGSTORE_TILE - 1) / R3D_SEGSTORE_TILE;
    } else {
      *m = old.segs[oi];
    }
    uint64_t nt = (uint64_t)m->tw * m->th;
    if (ntiles + nt > tiles_cap) {
      tiles_cap = (ntiles + nt) * 2;
      r3d_segtile *nt2 = realloc(tiles, tiles_cap * sizeof *nt2);
      if (!nt2) {
        r3d_tifxyz_free(&s);
        return sgs_build_abort(segs, tiles, &names, dp, &old, have_old);
      }
      tiles = nt2;
    }
    if (oi == UINT32_MAX) sgs_tiles_build(&s, m->bbox, tiles + ntiles, m->tw, m->th);
    else memcpy(tiles + ntiles, old.tiles + old.segs[oi].tile_ofs, nt * sizeof *tiles);
    if (decoded) r3d_tifxyz_free(&s);
    m->tile_ofs = ntiles;
    ntiles += nt;
    n++;
  }
  if (dp) closedir(dp);
  if (have_old) r3d_segstore_close(&old);
  sgs_nameset_free(&names);
  /* manifest: header + metas + tile array, one atomic write. Sizes are
   * checked-overflow throughout: a pathological n/ntiles aborts rather
   * than wrapping into a too-small allocation or a corrupt file. */
  uint64_t segbytes, tilebytes, payload, total64;
  if (!sgs_mul_ov(n, sizeof *segs, &segbytes) || !sgs_mul_ov(ntiles, sizeof *tiles, &tilebytes) ||
      !sgs_add_ov(segbytes, tilebytes, &payload) || !sgs_add_ov(sizeof(struct sgs_hdr), payload,
                                                                 &total64) ||
      total64 > SIZE_MAX) {
    free(tiles);
    free(segs);
    return -1;
  }
  size_t total = (size_t)total64;
  uint8_t *blob = malloc(total ? total : 1);
  int rc = blob ? 0 : -1;
  if (blob) {
    memcpy(blob + sizeof(struct sgs_hdr), segs, (size_t)segbytes);
    memcpy(blob + sizeof(struct sgs_hdr) + (size_t)segbytes, tiles, (size_t)tilebytes);
    struct sgs_hdr hdr = {.version = SGS_VERSION,
                          .count = n,
                          .tile = R3D_SEGSTORE_TILE,
                          .ntiles = ntiles,
                          .checksum = sgs_fnv1a(blob + sizeof(struct sgs_hdr), (size_t)payload)};
    memcpy(hdr.magic, SGS_MAGIC, 8);
    memcpy(blob, &hdr, sizeof hdr);
    char path[600];
    snprintf(path, sizeof path, "%s/segments.r3ds", store_dir);
    rc = sgs_write_file(path, blob, total);
  }
  free(blob);
  free(tiles);
  free(segs);
  return rc == 0 ? (int)n : -1;
}

int r3d_segstore_open(r3d_segstore *st, const char *store_dir) {
  memset(st, 0, sizeof *st);
  char path[600];
  snprintf(path, sizeof path, "%s/segments.r3ds", store_dir);
  size_t n = 0;
  uint8_t *blob = sgs_read_file(path, &n);
  if (!blob) return -1;
  struct sgs_hdr hdr;
  if (n < sizeof hdr) goto fail;
  memcpy(&hdr, blob, sizeof hdr);
  if (memcmp(hdr.magic, SGS_MAGIC, 8) != 0 || hdr.version != SGS_VERSION ||
      hdr.tile != R3D_SEGSTORE_TILE)
    goto fail;
  /* reject before any arithmetic derived from these counts: past this
   * point count*sizeof(record) and ntiles*sizeof(record) cannot wrap a
   * (>=32-bit) size_t */
  if (hdr.count > SGS_MAX_SEGS || hdr.ntiles > SGS_MAX_TILES) goto fail;
  uint64_t segbytes, tilebytes, payload, want64;
  if (!sgs_mul_ov(hdr.count, sizeof(r3d_segmeta), &segbytes) ||
      !sgs_mul_ov(hdr.ntiles, sizeof(r3d_segtile), &tilebytes) ||
      !sgs_add_ov(segbytes, tilebytes, &payload) ||
      !sgs_add_ov(sizeof hdr, payload, &want64) || want64 > SIZE_MAX)
    goto fail;
  size_t want = (size_t)want64;
  if (n != want) goto fail;
  if (sgs_fnv1a(blob + sizeof hdr, (size_t)payload) != hdr.checksum) goto fail;
  st->segs = malloc(segbytes ? (size_t)segbytes : 1);
  st->tiles = malloc(tilebytes ? (size_t)tilebytes : 1);
  if (!st->segs || !st->tiles) goto fail;
  memcpy(st->segs, blob + sizeof hdr, (size_t)segbytes);
  memcpy(st->tiles, blob + sizeof hdr + (size_t)segbytes, (size_t)tilebytes);
  st->n = hdr.count;
  st->ntiles = hdr.ntiles;
  for (uint32_t i = 0; i < st->n; i++) { /* validate before trusting offsets */
    r3d_segmeta *m = &st->segs[i];
    m->name[R3D_SEGSTORE_NAME - 1] = 0;
    uint64_t nt, tile_end;
    if (m->w > SGS_MAX_DIM || m->h > SGS_MAX_DIM ||
        m->tw != (m->w + R3D_SEGSTORE_TILE - 1) / R3D_SEGSTORE_TILE ||
        m->th != (m->h + R3D_SEGSTORE_TILE - 1) / R3D_SEGSTORE_TILE ||
        !sgs_mul_ov(m->tw, m->th, &nt) || !sgs_add_ov(m->tile_ofs, nt, &tile_end) ||
        tile_end > st->ntiles)
      goto fail;
  }
  snprintf(st->dir, sizeof st->dir, "%s", store_dir);
  free(blob);
  return 0;
fail:
  free(blob);
  r3d_segstore_close(st);
  return -1;
}

void r3d_segstore_close(r3d_segstore *st) {
  free(st->segs);
  free(st->tiles);
  memset(st, 0, sizeof *st);
}

int r3d_segstore_load(const r3d_segstore *st, uint32_t i, uint32_t stride, r3d_tifxyz *out) {
  if (i >= st->n || stride == 0) return -1;
  const r3d_segmeta *m = &st->segs[i];
  /* stride-4 tier: strides divisible by 4 decode the 16x-smaller .tfx4
   * (same source points, so results match the full decode decimated) */
  uint32_t eff = stride, sw = m->w, sh = m->h;
  char path[600];
  bool tier4 = false;
  if (stride % 4 == 0) {
    snprintf(path, sizeof path, "%s/%s.tfx4", st->dir, m->name);
    FILE *probe = fopen(path, "rb");
    if (probe) {
      fclose(probe);
      tier4 = true;
      eff = stride / 4;
      sw = (m->w + 3) / 4;
      sh = (m->h + 3) / 4;
    }
  }
  if (!tier4) snprintf(path, sizeof path, "%s/%s.tfx", st->dir, m->name);
  size_t enc_n = 0;
  uint8_t *enc = sgs_read_file(path, &enc_n);
  if (!enc) return -1;
  c5d_tifxyz ct;
  int rc = c5d_tifxyz_decode(enc, enc_n, &ct);
  free(enc);
  if (rc != 0) return -1;
  if (ct.w != sw || ct.h != sh) {
    c5d_tifxyz_free(&ct);
    return -1;
  }
  memset(out, 0, sizeof *out);
  out->w = (m->w + stride - 1) / stride;
  out->h = (m->h + stride - 1) / stride;
  out->sx = m->sx / (float)stride; /* one decimated cell spans stride cells */
  out->sy = m->sy / (float)stride;
  out->xyz = malloc((uint64_t)out->w * out->h * 3 * sizeof *out->xyz);
  if (!out->xyz) {
    c5d_tifxyz_free(&ct);
    return -1;
  }
  float bb[2][3] = {{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
  uint64_t nvalid = 0;
  for (uint32_t j = 0; j < out->h; j++)
    for (uint32_t ii = 0; ii < out->w; ii++) {
      uint64_t src = (uint64_t)(j * eff) * sw + (uint64_t)ii * eff;
      float *q = out->xyz + ((uint64_t)j * out->w + ii) * 3;
      q[0] = ct.plane[0][src];
      q[1] = ct.plane[1][src];
      q[2] = ct.plane[2][src];
      if (q[0] != -1.0f) {
        nvalid++;
        for (int a = 0; a < 3; a++) {
          if (q[a] < bb[0][a]) bb[0][a] = q[a];
          if (q[a] > bb[1][a]) bb[1][a] = q[a];
        }
      }
    }
  memcpy(out->bbox, bb, sizeof bb);
  out->nvalid = nvalid;
  c5d_tifxyz_free(&ct);
  return 0;
}

static void sgs_tile_world(const r3d_segmeta *m, const r3d_segtile *t, double lo[3],
                           double hi[3]);

#define OV_GRID 64

double r3d_segstore_overlap(const r3d_segstore *st, uint32_t a, uint32_t b, double tol) {
  if (a >= st->n || b >= st->n || a == b) return 0.0;
  const r3d_segmeta *ma = &st->segs[a], *mb = &st->segs[b];
  /* shared bbox (tol-dilated); no intersection = no overlap */
  double lo[3], hi[3];
  for (int c = 0; c < 3; c++) {
    lo[c] = fmax((double)ma->bbox[0][c], (double)mb->bbox[0][c]) - tol;
    hi[c] = fmin((double)ma->bbox[1][c], (double)mb->bbox[1][c]) + tol;
    if (lo[c] >= hi[c]) return 0.0;
  }
  double cell[3];
  for (int c = 0; c < 3; c++) cell[c] = (hi[c] - lo[c]) / OV_GRID;
  static _Thread_local uint8_t occ[OV_GRID * OV_GRID * OV_GRID / 8];
  memset(occ, 0, sizeof occ);
  const r3d_segtile *tb = st->tiles + mb->tile_ofs;
  for (uint64_t t = 0; t < (uint64_t)mb->tw * mb->th; t++) {
    if (tb[t].lo[0] > tb[t].hi[0]) continue;
    double tlo[3], thi[3];
    sgs_tile_world(mb, &tb[t], tlo, thi);
    int c0[3], c1[3];
    bool out = false;
    for (int c = 0; c < 3; c++) {
      c0[c] = (int)floor((tlo[c] - tol - lo[c]) / cell[c]);
      c1[c] = (int)floor((thi[c] + tol - lo[c]) / cell[c]);
      if (c0[c] < 0) c0[c] = 0;
      if (c1[c] >= OV_GRID) c1[c] = OV_GRID - 1;
      if (c0[c] > c1[c]) out = true;
    }
    if (out) continue;
    for (int z = c0[2]; z <= c1[2]; z++)
      for (int y = c0[1]; y <= c1[1]; y++)
        for (int x = c0[0]; x <= c1[0]; x++) {
          uint32_t k = (uint32_t)((z * OV_GRID + y) * OV_GRID + x);
          occ[k >> 3] |= (uint8_t)(1u << (k & 7u));
        }
  }
  const r3d_segtile *ta = st->tiles + ma->tile_ofs;
  uint64_t total = 0, hit = 0;
  for (uint64_t t = 0; t < (uint64_t)ma->tw * ma->th; t++) {
    if (ta[t].lo[0] > ta[t].hi[0]) continue;
    total++;
    double tlo[3], thi[3];
    sgs_tile_world(ma, &ta[t], tlo, thi);
    double cx = (0.5 * (tlo[0] + thi[0]) - lo[0]) / cell[0];
    double cy = (0.5 * (tlo[1] + thi[1]) - lo[1]) / cell[1];
    double cz = (0.5 * (tlo[2] + thi[2]) - lo[2]) / cell[2];
    if (cx < 0.0 || cy < 0.0 || cz < 0.0 || cx >= OV_GRID || cy >= OV_GRID ||
        cz >= OV_GRID)
      continue;
    uint32_t k = (uint32_t)(((int)cz * OV_GRID + (int)cy) * OV_GRID + (int)cx);
    if (occ[k >> 3] & (1u << (k & 7u))) hit++;
  }
  return total ? (double)hit / (double)total : 0.0;
}

/* bounds of dot(p, n) over box [lo, hi] */
static void sgs_box_dot(const double lo[3], const double hi[3], const double bn[3],
                        double *dlo, double *dhi) {
  double l = 0.0, h = 0.0;
  for (int a = 0; a < 3; a++) {
    if (bn[a] >= 0.0) {
      l += bn[a] * lo[a];
      h += bn[a] * hi[a];
    } else {
      l += bn[a] * hi[a];
      h += bn[a] * lo[a];
    }
  }
  *dlo = l;
  *dhi = h;
}

static void sgs_tile_world(const r3d_segmeta *m, const r3d_segtile *t, double lo[3],
                           double hi[3]) {
  for (int a = 0; a < 3; a++) {
    double qs = ((double)m->bbox[1][a] - (double)m->bbox[0][a]) / 65535.0;
    if (qs <= 0.0) qs = 1.0;
    lo[a] = (double)m->bbox[0][a] + qs * t->lo[a];
    hi[a] = (double)m->bbox[0][a] + qs * t->hi[a];
  }
}

static bool sgs_box_overlap(const double alo[3], const double ahi[3], const double blo[3],
                            const double bhi[3]) {
  for (int a = 0; a < 3; a++)
    if (ahi[a] < blo[a] || alo[a] > bhi[a]) return false;
  return true;
}

uint32_t r3d_segstore_plane_query(const r3d_segstore *st, const double bn[3], double slice,
                                  double margin, const double lo[3], const double hi[3],
                                  uint32_t *out, uint32_t cap) {
  uint32_t hits = 0;
  for (uint32_t i = 0; i < st->n; i++) {
    const r3d_segmeta *m = &st->segs[i];
    double blo[3] = {(double)m->bbox[0][0], (double)m->bbox[0][1], (double)m->bbox[0][2]};
    double bhi[3] = {(double)m->bbox[1][0], (double)m->bbox[1][1], (double)m->bbox[1][2]};
    double dlo, dhi;
    sgs_box_dot(blo, bhi, bn, &dlo, &dhi);
    if (slice < dlo - margin || slice > dhi + margin) continue;
    if (lo && hi && !sgs_box_overlap(blo, bhi, lo, hi)) continue;
    const r3d_segtile *tiles = st->tiles + m->tile_ofs;
    bool hit = false;
    for (uint64_t t = 0; t < (uint64_t)m->tw * m->th && !hit; t++) {
      if (tiles[t].lo[0] > tiles[t].hi[0]) continue; /* empty tile */
      double tlo[3], thi[3];
      sgs_tile_world(m, &tiles[t], tlo, thi);
      sgs_box_dot(tlo, thi, bn, &dlo, &dhi);
      if (slice < dlo - margin || slice > dhi + margin) continue;
      if (lo && hi && !sgs_box_overlap(tlo, thi, lo, hi)) continue;
      hit = true;
    }
    if (!hit) continue;
    if (out && hits < cap) out[hits] = i;
    hits++;
  }
  return hits;
}

uint32_t r3d_segstore_near_query(const r3d_segstore *st, const double p[3], double radius,
                                 uint32_t *out, uint32_t cap) {
  double *dist = malloc((st->n ? st->n : 1) * sizeof *dist);
  uint32_t hits = 0;
  if (!dist) return 0;
  for (uint32_t i = 0; i < st->n; i++) {
    const r3d_segmeta *m = &st->segs[i];
    double best = 1e30;
    /* segment bbox distance first: skip the tile scan when hopeless */
    double d2 = 0.0;
    for (int a = 0; a < 3; a++) {
      double d = p[a] < (double)m->bbox[0][a]   ? (double)m->bbox[0][a] - p[a]
                 : p[a] > (double)m->bbox[1][a] ? p[a] - (double)m->bbox[1][a]
                                                : 0.0;
      d2 += d * d;
    }
    if (d2 > radius * radius) continue;
    const r3d_segtile *tiles = st->tiles + m->tile_ofs;
    for (uint64_t t = 0; t < (uint64_t)m->tw * m->th; t++) {
      if (tiles[t].lo[0] > tiles[t].hi[0]) continue;
      double tlo[3], thi[3];
      sgs_tile_world(m, &tiles[t], tlo, thi);
      double td2 = 0.0;
      for (int a = 0; a < 3; a++) {
        double d = p[a] < tlo[a] ? tlo[a] - p[a] : p[a] > thi[a] ? p[a] - thi[a] : 0.0;
        td2 += d * d;
      }
      if (td2 < best) best = td2;
    }
    if (best > radius * radius) continue;
    /* insertion sort by distance into the parallel arrays */
    uint32_t k = hits < cap ? hits : cap;
    while (k > 0 && dist[k - 1] > best) {
      if (k < cap) {
        dist[k] = dist[k - 1];
        if (out) out[k] = out[k - 1];
      }
      k--;
    }
    if (k < cap) {
      dist[k] = best;
      if (out) out[k] = i;
    }
    hits++;
  }
  free(dist);
  return hits;
}
