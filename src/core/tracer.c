#include "core/tracer.h"

#include <curl/curl.h>
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <unistd.h>

#include "core/cpuvol.h"
#include "core/tifxyz.h"

/* ============================= constants =============================
 * Weights and schedule mirror vc3d GrowPatch.cpp defaults. */
#define TR_W_DIST 1.0
#define TR_W_STRAIGHT 0.2
#define TR_W_SDIR 1.0
#define TR_W_SPACE 0.1   /* space-line data term (vc3d ships it optional;
                          * it is our data term in lieu of normal grids) */
#define TR_SPACE_STEPS 8 /* samples per edge, vc3d space_line_steps */
#define TR_KINK_COS 0.86602540378443864676 /* cos(30 deg) */
#define TR_SDIR_EPS_ABS 1e-8
#define TR_SDIR_EPS_REL 1e-2
#define TR_DT_TH 128.0 /* prediction "on sheet" threshold for the DT */
#define TR_CONF_R 12.0 /* conf = 1 - min(dt,R)/R */

/* ================= distance-transform chunk cache ==================
 * vc3d lineLossDistance: per 64^3 chunk (16-voxel border), exact
 * Euclidean distance to the nearest above-threshold prediction voxel,
 * clamped to u8. Chunks are in LEVEL voxel units; sampling converts. */
#define TD_CORE 64
#define TD_BORD 16
#define TD_EXT (TD_CORE + 2 * TD_BORD)
#define TD_SLOTS 512

typedef struct td_slot {
  uint64_t key; /* 0 = empty */
  uint64_t use;
  uint8_t *d; /* TD_CORE^3 */
} td_slot;

typedef struct td_cache {
  r3d_cpuvol *vol;
  uint32_t level;
  td_slot s[TD_SLOTS];
  uint64_t tick;
  float *sq;         /* TD_EXT^3 scratch */
  uint64_t memo_key; /* last-chunk memo */
  uint8_t *memo_d;
} td_cache;

static td_cache *td_open(r3d_cpuvol *vol, uint32_t level) {
  td_cache *c = calloc(1, sizeof *c);
  if (!c) return NULL;
  c->vol = vol;
  c->level = level;
  c->sq = malloc((size_t)TD_EXT * TD_EXT * TD_EXT * sizeof *c->sq);
  if (!c->sq) {
    free(c);
    return NULL;
  }
  return c;
}

static void td_close(td_cache *c) {
  if (!c) return;
  for (int i = 0; i < TD_SLOTS; i++) free(c->s[i].d);
  free(c->sq);
  free(c);
}

/* 1D squared EDT (Felzenszwalb), n = TD_EXT */
static void td_edt1d(float *f, int n) {
  static _Thread_local float d[TD_EXT], z[TD_EXT + 1];
  static _Thread_local int v[TD_EXT];
  int k = 0;
  v[0] = 0;
  z[0] = -1e30f;
  z[1] = 1e30f;
  for (int q = 1; q < n; q++) {
    float s;
    for (;;) {
      int p = v[k];
      s = ((f[q] + (float)(q * q)) - (f[p] + (float)(p * p))) / (float)(2 * (q - p));
      if (s > z[k]) break;
      k--;
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e30f;
  }
  k = 0;
  for (int q = 0; q < n; q++) {
    while (z[k + 1] < (float)q) k++;
    float dq = (float)q - (float)v[k];
    d[q] = dq * dq + f[v[k]];
  }
  memcpy(f, d, (size_t)n * sizeof *f);
}

static const uint8_t *td_chunk(td_cache *c, int64_t cx, int64_t cy, int64_t cz) {
  if (cx < 0 || cy < 0 || cz < 0 || cx >= (1 << 20) || cy >= (1 << 20) || cz >= (1 << 20))
    return NULL;
  uint64_t key = 1u + (((uint64_t)cz << 40) | ((uint64_t)cy << 20) | (uint64_t)cx);
  if (key == c->memo_key) return c->memo_d;
  uint64_t h = key * 0x9E3779B97F4A7C15ull;
  uint32_t base = (uint32_t)(h >> 32) % TD_SLOTS;
  td_slot *lru = NULL;
  for (uint32_t p = 0; p < 8; p++) {
    td_slot *s = &c->s[(base + p) % TD_SLOTS];
    if (s->key == key) {
      s->use = ++c->tick;
      c->memo_key = key;
      c->memo_d = s->d;
      return s->d;
    }
    if (s->key == 0) { /* empty slot wins outright */
      if (!lru || lru->key != 0) lru = s;
    } else if (!lru || (lru->key != 0 && s->use < lru->use)) {
      lru = s;
    }
  }
  if (!lru->d) {
    lru->d = malloc((size_t)TD_CORE * TD_CORE * TD_CORE);
    if (!lru->d) return NULL;
  }
  /* occupancy of the extended block, squared-EDT seeds */
  const double sc = (double)c->vol->lev[c->level].scale;
  float *sq = c->sq;
  for (int64_t lz = 0; lz < TD_EXT; lz++)
    for (int64_t ly = 0; ly < TD_EXT; ly++)
      for (int64_t lx = 0; lx < TD_EXT; lx++) {
        double bx = ((double)(cx * TD_CORE - TD_BORD + lx) + 0.5) * sc;
        double by = ((double)(cy * TD_CORE - TD_BORD + ly) + 0.5) * sc;
        double bz = ((double)(cz * TD_CORE - TD_BORD + lz) + 0.5) * sc;
        uint8_t v = r3d_cpuvol_at(c->vol, c->level, bx, by, bz);
        sq[((size_t)lz * TD_EXT + (size_t)ly) * TD_EXT + (size_t)lx] =
            (double)v >= TR_DT_TH ? 0.0f : 1e30f;
      }
  static _Thread_local float line[TD_EXT];
  for (int z2 = 0; z2 < TD_EXT; z2++) /* x pass */
    for (int y2 = 0; y2 < TD_EXT; y2++) {
      float *row = sq + ((size_t)z2 * TD_EXT + (size_t)y2) * TD_EXT;
      td_edt1d(row, TD_EXT);
    }
  for (int z2 = 0; z2 < TD_EXT; z2++) /* y pass */
    for (int x2 = 0; x2 < TD_EXT; x2++) {
      for (int y2 = 0; y2 < TD_EXT; y2++)
        line[y2] = sq[((size_t)z2 * TD_EXT + (size_t)y2) * TD_EXT + (size_t)x2];
      td_edt1d(line, TD_EXT);
      for (int y2 = 0; y2 < TD_EXT; y2++)
        sq[((size_t)z2 * TD_EXT + (size_t)y2) * TD_EXT + (size_t)x2] = line[y2];
    }
  for (int y2 = 0; y2 < TD_EXT; y2++) /* z pass */
    for (int x2 = 0; x2 < TD_EXT; x2++) {
      for (int z2 = 0; z2 < TD_EXT; z2++)
        line[z2] = sq[((size_t)z2 * TD_EXT + (size_t)y2) * TD_EXT + (size_t)x2];
      td_edt1d(line, TD_EXT);
      for (int z2 = 0; z2 < TD_EXT; z2++)
        sq[((size_t)z2 * TD_EXT + (size_t)y2) * TD_EXT + (size_t)x2] = line[z2];
    }
  for (int z2 = 0; z2 < TD_CORE; z2++)
    for (int y2 = 0; y2 < TD_CORE; y2++)
      for (int x2 = 0; x2 < TD_CORE; x2++) {
        float dd = sqrtf(sq[(((size_t)z2 + TD_BORD) * TD_EXT + ((size_t)y2 + TD_BORD)) * TD_EXT +
                            (size_t)x2 + TD_BORD]);
        lru->d[((size_t)z2 * TD_CORE + (size_t)y2) * TD_CORE + (size_t)x2] =
            dd >= 255.0f ? 255 : (uint8_t)(dd + 0.5f);
      }
  lru->key = key;
  lru->use = ++c->tick;
  c->memo_key = key;
  c->memo_d = lru->d;
  return lru->d;
}

static double td_vox(td_cache *c, int64_t vx, int64_t vy, int64_t vz) {
  if (vx < 0 || vy < 0 || vz < 0) return 255.0;
  const uint8_t *d = td_chunk(c, vx / TD_CORE, vy / TD_CORE, vz / TD_CORE);
  if (!d) return 255.0;
  return (double)d[(((size_t)(vz % TD_CORE)) * TD_CORE + (size_t)(vy % TD_CORE)) * TD_CORE +
                   (size_t)(vx % TD_CORE)];
}

/* trilinear DT value in BASE voxel units + gradient wrt base coords */
static double td_tri(td_cache *c, const double p[3], double grad[3]) {
  const double sc = (double)c->vol->lev[c->level].scale;
  double lx = p[0] / sc, ly = p[1] / sc, lz = p[2] / sc;
  double fx = floor(lx), fy = floor(ly), fz = floor(lz);
  double tx = lx - fx, ty = ly - fy, tz = lz - fz;
  int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz;
  double cv[2][2][2];
  for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
      for (int dx = 0; dx < 2; dx++)
        cv[dz][dy][dx] = td_vox(c, ix + dx, iy + dy, iz + dz);
  double c00 = cv[0][0][0] * (1 - tx) + cv[0][0][1] * tx;
  double c01 = cv[0][1][0] * (1 - tx) + cv[0][1][1] * tx;
  double c10 = cv[1][0][0] * (1 - tx) + cv[1][0][1] * tx;
  double c11 = cv[1][1][0] * (1 - tx) + cv[1][1][1] * tx;
  double c0 = c00 * (1 - ty) + c01 * ty;
  double c1 = c10 * (1 - ty) + c11 * ty;
  if (grad) {
    double gx0 = (cv[0][0][1] - cv[0][0][0]) * (1 - ty) + (cv[0][1][1] - cv[0][1][0]) * ty;
    double gx1 = (cv[1][0][1] - cv[1][0][0]) * (1 - ty) + (cv[1][1][1] - cv[1][1][0]) * ty;
    /* value is dt*scale (base units); d(dt*sc)/dbase = ddt/dlevel */
    grad[0] = gx0 * (1 - tz) + gx1 * tz;
    grad[1] = (c01 - c00) * (1 - tz) + (c11 - c10) * tz;
    grad[2] = c1 - c0;
  }
  return (c0 * (1 - tz) + c1 * tz) * sc;
}

/* ================== normal grids (vc3d GridStore v3) ==================
 * Per-slice polyline stores of sheet cross-sections in three plane
 * families (xy/xz/yz), published next to the surface predictions
 * (<pred>.normal-grids). This is the vc3d tracer's REAL data term: the
 * NormalConstraintPlane below aligns quad edges with the local polylines
 * and snaps them onto ONE polyline — sheet identity lives in polyline
 * connectivity, which is what stops wrap-jumping. Files are fetched on
 * demand ("%s/<plane>/%06d.grid"), cached on disk, decoded whole. */

static uint32_t ng_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

#define NG_BUDGET ((size_t)3u << 29) /* decoded-slice RAM budget (1.5 GB) */

typedef struct ng_grid {
  size_t bytes;         /* footprint (budget accounting) */
  int32_t bx, by;       /* bounds origin (always 0 in published grids) */
  uint32_t w, h;        /* slice extent, px */
  uint32_t cell, gw, gh; /* bucket grid */
  uint32_t npaths;
  uint8_t *blob;        /* paths region verbatim: {u32be sx, sy, noff,
                         * int8 deltas[noff]} records — 4x smaller than
                         * decoded float pairs; decoded per hood build */
  size_t blob_n;
  uint32_t *precoff;    /* [npaths] byte offset of each record (sorted) */
  uint32_t *bidx;       /* [gw*gh+1] CSR into boff */
  uint32_t *boff;       /* record byte offsets per bucket */
  bool empty;
} ng_grid;

static void ng_grid_free(ng_grid *g) {
  if (!g) return;
  free(g->blob);
  free(g->precoff);
  free(g->bidx);
  free(g->boff);
  free(g);
}

/* decode path pi into pts (caller cap >= npts); returns point count */
static uint32_t ng_path_decode(const ng_grid *g, uint32_t pi, float *pts,
                               uint32_t cap) {
  const uint8_t *r = g->blob + g->precoff[pi];
  double cx = ng_be32(r), cy = ng_be32(r + 4);
  uint32_t noff = ng_be32(r + 8), n = 0;
  const int8_t *dl = (const int8_t *)(r + 12);
  if (n < cap) {
    pts[n * 2] = (float)cx;
    pts[n * 2 + 1] = (float)cy;
  }
  n++;
  for (uint32_t k = 0; k < noff; k += 2) {
    cx += dl[k];
    cy += dl[k + 1];
    if (n < cap) {
      pts[n * 2] = (float)cx;
      pts[n * 2 + 1] = (float)cy;
    }
    n++;
  }
  return n < cap ? n : cap;
}

static int32_t ng_path_of(const ng_grid *g, uint32_t rec);

/* decode a whole .grid file (v3, big-endian); empty file = empty slice */
static ng_grid *ng_grid_load(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long fn = ftell(f);
  fseek(f, 0, SEEK_SET);
  ng_grid *g = calloc(1, sizeof *g);
  if (!g) {
    fclose(f);
    return NULL;
  }
  if (fn <= 0) { /* 0-byte marker: processed, nothing here */
    fclose(f);
    g->empty = true;
    return g;
  }
  uint8_t *d = malloc((size_t)fn);
  if (!d || fread(d, 1, (size_t)fn, f) != (size_t)fn) {
    free(d);
    fclose(f);
    free(g);
    return NULL;
  }
  fclose(f);
  size_t n = (size_t)fn;
  if (n < 52 || ng_be32(d) != 0x56434753u || ng_be32(d + 4) != 3u) goto bad;
  g->bx = (int32_t)ng_be32(d + 8);
  g->by = (int32_t)ng_be32(d + 12);
  g->w = ng_be32(d + 16);
  g->h = ng_be32(d + 20);
  g->cell = ng_be32(d + 24);
  uint32_t nbuck = ng_be32(d + 28);
  uint32_t bio = ng_be32(d + 36), po = ng_be32(d + 40), jmo = ng_be32(d + 44);
  if (!g->cell || g->w > (1u << 24) || g->h > (1u << 24)) goto bad;
  g->gw = (g->w + g->cell - 1) / g->cell;
  g->gh = (g->h + g->cell - 1) / g->cell;
  if (nbuck != g->gw * g->gh) goto bad;
  if ((size_t)bio + 4u * (nbuck + 1) > n || po > n) goto bad;
  size_t pend = jmo && jmo <= n ? jmo : n;
  /* walk path records sequentially: {u32 sx, u32 sy, u32 noff, int8[]} */
  uint32_t cap = 256, np = 0;
  g->precoff = malloc(cap * sizeof *g->precoff);
  if (!g->precoff) goto bad;
  size_t off = po;
  while (off + 12 <= pend) {
    uint32_t noff = ng_be32(d + off + 8);
    if (off + 12 + noff > pend || (noff & 1u)) break;
    if (np + 1 > cap) {
      cap *= 2;
      uint32_t *nr = realloc(g->precoff, cap * sizeof *nr);
      if (!nr) goto bad;
      g->precoff = nr;
    }
    g->precoff[np++] = (uint32_t)(off - po);
    off += 12 + noff;
  }
  g->npaths = np;
  g->blob_n = off - po;
  g->blob = malloc(g->blob_n ? g->blob_n : 1);
  if (!g->blob) goto bad;
  memcpy(g->blob, d + po, g->blob_n);
  /* bucket CSR: resolve record offsets to path indices once */
  uint32_t nboff = ng_be32(d + bio + 4u * nbuck);
  if ((size_t)po - bio - 4u * (nbuck + 1) < 4u * (size_t)nboff) goto bad;
  g->bidx = malloc((nbuck + 1) * sizeof *g->bidx);
  g->boff = malloc((nboff ? nboff : 1) * sizeof *g->boff);
  if (!g->bidx || !g->boff) goto bad;
  for (uint32_t i = 0; i <= nbuck; i++) g->bidx[i] = ng_be32(d + bio + 4u * i);
  for (uint32_t i = 0; i < nboff; i++)
    g->boff[i] = ng_be32(d + bio + 4u * (nbuck + 1) + 4u * i);
  free(d);
  g->bytes = sizeof *g + g->blob_n + (size_t)np * sizeof(uint32_t) +
             (size_t)(nbuck + 1 + nboff) * sizeof(uint32_t);
  return g;
bad:
  free(d);
  ng_grid_free(g);
  return NULL;
}

/* record byte offset -> path index (precoff is sorted by construction) */
static int32_t ng_path_of(const ng_grid *g, uint32_t rec) {
  uint32_t lo = 0, hi = g->npaths;
  while (lo < hi) {
    uint32_t mid = (lo + hi) / 2;
    if (g->precoff[mid] < rec) lo = mid + 1;
    else hi = mid;
  }
  return lo < g->npaths && g->precoff[lo] == rec ? (int32_t)lo : -1;
}

/* vc3d get(center, radius): truncating rect, inclusive bucket range;
 * appends unique path indices to out[] (caller-sized), returns count */
static uint32_t ng_query(const ng_grid *g, double cx, double cy, double radius,
                         uint32_t *out, uint32_t max) {
  if (!g || g->empty || !g->npaths) return 0;
  int x0 = (int)(cx - radius) - g->bx, y0 = (int)(cy - radius) - g->by;
  int sz = (int)(radius * 2.0);
  int x1 = x0 + sz, y1 = y0 + sz;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > (int)g->w) x1 = (int)g->w;
  if (y1 > (int)g->h) y1 = (int)g->h;
  if (x0 >= x1 || y0 >= y1) return 0;
  uint32_t bx0 = (uint32_t)x0 / g->cell, by0 = (uint32_t)y0 / g->cell;
  uint32_t bx1 = (uint32_t)x1 / g->cell, by1 = (uint32_t)y1 / g->cell;
  if (bx1 >= g->gw) bx1 = g->gw - 1;
  if (by1 >= g->gh) by1 = g->gh - 1;
  uint32_t nout = 0;
  for (uint32_t by2 = by0; by2 <= by1; by2++)
    for (uint32_t bx2 = bx0; bx2 <= bx1; bx2++) {
      uint32_t b = by2 * g->gw + bx2;
      for (uint32_t k = g->bidx[b]; k < g->bidx[b + 1]; k++) {
        int32_t pi = ng_path_of(g, g->boff[k]); /* resolved here: queries
                                * are amortized by hoods, loads are not */
        if (pi < 0) continue;
        bool dup = false;
        for (uint32_t q = 0; q < nout && !dup; q++) dup = out[q] == (uint32_t)pi;
        if (!dup && nout < max) out[nout++] = (uint32_t)pi;
      }
    }
  return nout;
}

/* --- slice cache + demand fetch ------------------------------------- */
#define NG_SLOTS 2048
#define NG_QMAX 64    /* paths per neighborhood query */
#define NG_NHOOD 512  /* cached segment neighborhoods */
#define NG_HOODSEG 384 /* segments per neighborhood */

/* phase timers, atomic ns: accumulated from every worker thread */
static _Atomic uint64_t tr_tm_ns[6]; /* 0 place 1 lopt 2 conf 3 ngfetch
                                      * 4 ngeval 5 hoodbuild */
static double tr_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void tr_tm_add(int k, double t0) {
  atomic_fetch_add(&tr_tm_ns[k], (uint64_t)((tr_now() - t0) * 1e9));
}
#define TR_TM(k) ((double)atomic_load(&tr_tm_ns[k]) * 1e-9)

/* frozen segment neighborhood around a query point (vc3d caches nearby
 * paths per cost and refreshes only after 16 px of movement; here keyed
 * by 16-px quantized position — the expensive full-path scan happens
 * once, residual evals then touch only ROI segments) */
typedef struct ng_hood {
  int plane, slice, qx, qy; /* plane<0 = free */
  const ng_grid *g;
  uint32_t nseg;
  float sg[NG_HOODSEG][13]; /* ax ay bx by nx ny prevx prevy nextx nexty
                             * dx dy invL2 (pt-seg without divides) */
  uint8_t nb[NG_HOODSEG];   /* bit0: prev exists, bit1: next exists */
  uint64_t use;
} ng_hood;

typedef struct ng_vol {
  char root[1200];  /* local cache dir (<pred_root>/ngrids) */
  char url[1400];   /* remote base; empty = local only */
  double spiral_step;
  bool active;
  void *curl;
  uint64_t net_cool;
  struct {
    int plane, slice; /* plane -1 = free */
    ng_grid *g;       /* NULL = known missing */
    uint64_t use;
    uint32_t ref;     /* pinned by in-flight residual evals */
  } s[NG_SLOTS];
  uint16_t idx[8192]; /* open-addressed (plane,slice) -> slot+1 */
  _Atomic size_t bytes; /* decoded total across slots (incl. lazy paths) */
  pthread_mutex_t gmu; /* guards the slot table */
  uint64_t tick;
  /* prefetch pool: the frontier crosses xz/yz slices sequentially, so a
   * sync miss enqueues its neighbors — fetch overlaps the solve (vc3d
   * runs 4 prefetch workers with radius 4) */
  pthread_t fth[8];
  uint32_t nfth;
  pthread_mutex_t fmu;
  pthread_cond_t fcv;
  struct { int plane, slice; } fq[4096];
  uint32_t fqn;
  uint64_t fseen[8192]; /* open-addressed recently-enqueued set */
  bool fquit;
} ng_vol;

static const char *ng_plane_dir[3] = {"xy", "xz", "yz"};

static size_t ng_curl_write(const void *data, size_t sz, size_t nm, void *ud) {
  FILE *f = ud;
  return fwrite(data, sz, nm, f);
}

static void *ng_prefetch_worker(void *ud);

/* open from the prediction tree: <root>/source.json url with ".zarr"
 * swapped for ".normal-grids"; metadata.json fetched once for the step */
static void ng_open(ng_vol *v, const char *pred_root) {
  memset(v, 0, sizeof *v);
  for (int i = 0; i < NG_SLOTS; i++) v->s[i].plane = -1;
  pthread_mutex_init(&v->gmu, NULL);
  snprintf(v->root, sizeof v->root, "%s/ngrids", pred_root);
  char sp[1300];
  snprintf(sp, sizeof sp, "%s/source.json", pred_root);
  FILE *f = fopen(sp, "rb");
  if (f) {
    char sj[8192] = {0};
    size_t sn = fread(sj, 1, sizeof sj - 1, f);
    fclose(f);
    (void)sn;
    const char *up = strstr(sj, "\"url\": \"");
    if (up) {
      up += 8;
      const char *ue = strchr(up, '"');
      size_t ul = ue ? (size_t)(ue - up) : 0;
      if (ul && ul < sizeof v->url - 16) {
        memcpy(v->url, up, ul);
        v->url[ul] = 0;
        char *z = strstr(v->url, ".zarr");
        if (z && z[5] == 0) strcpy(z, ".normal-grids");
        else v->url[0] = 0;
      }
    }
  }
  mkdir(v->root, 0755);
  char mp[1400];
  snprintf(mp, sizeof mp, "%s/metadata.json", v->root);
  if (!(f = fopen(mp, "rb")) && v->url[0]) { /* fetch once */
    CURL *c = curl_easy_init();
    if (c) {
      char tmp[1500];
      snprintf(tmp, sizeof tmp, "%s.tmp", mp);
      FILE *tf = fopen(tmp, "wb");
      if (tf) {
        char url[1600];
        snprintf(url, sizeof url, "%s/metadata.json", v->url);
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, ng_curl_write);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, tf);
        curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 20L);
        CURLcode rc = curl_easy_perform(c);
        long code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        fclose(tf);
        if (rc == CURLE_OK && code == 200) rename(tmp, mp);
        else unlink(tmp);
      }
      curl_easy_cleanup(c);
    }
    f = fopen(mp, "rb");
  }
  if (f) {
    char mj[4096] = {0};
    size_t mn = fread(mj, 1, sizeof mj - 1, f);
    fclose(f);
    (void)mn;
    const char *ss = strstr(mj, "\"spiral-step\"");
    v->spiral_step = ss ? strtod(ss + 14, NULL) : 20.0;
    v->active = true;
    printf("tracer: normal grids active (%s, spiral-step %.1f)\n",
           v->url[0] ? v->url : v->root, v->spiral_step);
    if (v->url[0]) {
      pthread_mutex_init(&v->fmu, NULL);
      pthread_cond_init(&v->fcv, NULL);
      for (uint32_t i = 0; i < 8; i++)
        if (pthread_create(&v->fth[v->nfth], NULL, ng_prefetch_worker, v) == 0)
          v->nfth++;
    }
  }
}

static uint32_t ng_ih(int plane, int slice) {
  uint64_t key = (((uint64_t)(unsigned)plane << 32) | (unsigned)slice);
  return (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 43) & 8191u;
}

/* find slot for (plane,slice) via the index; -1 = absent. gmu held. */
static int ng_islot(ng_vol *v, int plane, int slice) {
  for (uint32_t h = ng_ih(plane, slice), n = 0; n < 8192;
       h = (h + 1) & 8191u, n++) {
    uint16_t e = v->idx[h];
    if (!e) return -1;
    uint32_t sl = (uint32_t)e - 1;
    if (v->s[sl].plane == plane && v->s[sl].slice == slice) return (int)sl;
  }
  return -1;
}

static void ng_idx_put(ng_vol *v, int plane, int slice, uint32_t slot) {
  for (uint32_t h = ng_ih(plane, slice), n = 0; n < 8192;
       h = (h + 1) & 8191u, n++)
    if (!v->idx[h]) {
      v->idx[h] = (uint16_t)(slot + 1);
      return;
    }
}

static void ng_idx_del(ng_vol *v, int plane, int slice) {
  for (uint32_t h = ng_ih(plane, slice), n = 0; n < 8192;
       h = (h + 1) & 8191u, n++) {
    uint16_t e = v->idx[h];
    if (!e) return;
    uint32_t sl = (uint32_t)e - 1;
    if (v->s[sl].plane == plane && v->s[sl].slice == slice) {
      v->idx[h] = 0;
      /* re-insert the probe run after the hole (open addressing) */
      for (uint32_t h2 = (h + 1) & 8191u; v->idx[h2]; h2 = (h2 + 1) & 8191u) {
        uint16_t e2 = v->idx[h2];
        v->idx[h2] = 0;
        uint32_t s2 = (uint32_t)e2 - 1;
        ng_idx_put(v, v->s[s2].plane, v->s[s2].slice, s2);
      }
      return;
    }
  }
}

static void ng_close(ng_vol *v) {
  if (v->nfth) {
    pthread_mutex_lock(&v->fmu);
    v->fquit = true;
    pthread_cond_broadcast(&v->fcv);
    pthread_mutex_unlock(&v->fmu);
    for (uint32_t i = 0; i < v->nfth; i++) pthread_join(v->fth[i], NULL);
    pthread_mutex_destroy(&v->fmu);
    pthread_cond_destroy(&v->fcv);
  }
  for (int i = 0; i < NG_SLOTS; i++)
    if (v->s[i].plane >= 0) ng_grid_free(v->s[i].g);
  pthread_mutex_destroy(&v->gmu);
  if (v->curl) curl_easy_cleanup(v->curl);
  memset(v, 0, sizeof *v);
}

/* fetch one slice file to disk with the given handle; thread-safe
 * (unique tmp names, atomic rename). Returns 0 when the file (or a
 * .missing sentinel) now exists. */
static int ng_fetch_file(ng_vol *v, CURL *c, int plane, int slice, bool sync) {
  char dir[1300], path[1500], miss[1520];
  snprintf(dir, sizeof dir, "%s/%s", v->root, ng_plane_dir[plane]);
  snprintf(path, sizeof path, "%s/%06d.grid", dir, slice);
  snprintf(miss, sizeof miss, "%s.missing", path);
  struct stat st;
  if (stat(path, &st) == 0 || stat(miss, &st) == 0) return 0;
  if (!v->url[0] || !c) return -1;
  if (!sync && v->net_cool > (uint64_t)time(NULL)) return -1; /* growth must
      * not go data-blind: the solve path always retries, only the
      * background prefetchers back off */
  mkdir(dir, 0755);
  char tmp[1600], url[1600];
  snprintf(tmp, sizeof tmp, "%s.tmp.%ld.%lx", path, (long)getpid(),
           (unsigned long)(uintptr_t)c);
  snprintf(url, sizeof url, "%s/%s/%06d.grid", v->url, ng_plane_dir[plane], slice);
  FILE *tf = fopen(tmp, "wb");
  if (!tf) return -1;
  curl_easy_setopt(c, CURLOPT_URL, url);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, tf);
  CURLcode rc = curl_easy_perform(c);
  long code = 0;
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
  fclose(tf);
  if (rc == CURLE_OK && code == 200) {
    if (rename(tmp, path) == 0) return 0;
  } else {
    unlink(tmp);
    if (rc == CURLE_OK && code == 404) {
      FILE *mf = fopen(miss, "wb"); /* definitively absent */
      if (mf) {
        fclose(mf);
        return 0;
      }
    } else {
      v->net_cool = (uint64_t)time(NULL) + 30;
    }
  }
  return -1;
}

static CURL *ng_mkcurl(void) {
  CURL *c = curl_easy_init();
  if (!c) return NULL;
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, ng_curl_write);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 20L);
  curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);
  curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
  return c;
}

static void *ng_prefetch_worker(void *ud) {
  ng_vol *v = ud;
  CURL *c = ng_mkcurl();
  for (;;) {
    pthread_mutex_lock(&v->fmu);
    while (!v->fquit && v->fqn == 0) pthread_cond_wait(&v->fcv, &v->fmu);
    if (v->fquit) {
      pthread_mutex_unlock(&v->fmu);
      break;
    }
    int plane = v->fq[0].plane, slice = v->fq[0].slice;
    memmove(v->fq, v->fq + 1, --v->fqn * sizeof v->fq[0]);
    pthread_mutex_unlock(&v->fmu);
    if (c) ng_fetch_file(v, c, plane, slice, false);
  }
  if (c) curl_easy_cleanup(c);
  return NULL;
}

static void ng_prefetch_reset(ng_vol *v) { /* let a new generation re-ask */
  if (!v->nfth) return;
  pthread_mutex_lock(&v->fmu);
  memset(v->fseen, 0, sizeof v->fseen);
  pthread_mutex_unlock(&v->fmu);
}

static void ng_prefetch(ng_vol *v, int plane, int slice) {
  if (!v->nfth || plane < 0 || slice < 0) return;
  uint64_t key = 1u + (((uint64_t)(unsigned)plane << 32) | (unsigned)slice);
  uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 40) & 8191u;
  pthread_mutex_lock(&v->fmu);
  if (v->fseen[h] == key) { /* recently enqueued: skip */
    pthread_mutex_unlock(&v->fmu);
    return;
  }
  if (v->fqn < 4096) {
    v->fseen[h] = key;
    v->fq[v->fqn].plane = plane;
    v->fq[v->fqn].slice = slice;
    v->fqn++;
    pthread_cond_signal(&v->fcv);
  }
  pthread_mutex_unlock(&v->fmu);
}

/* Acquire the slice grid, pinned against eviction; balance every non-NULL
 * return with ng_put. Slow path (disk load / sync HTTP fetch) runs outside
 * the table lock; a racing loader of the same slice loses and frees. */
static ng_grid *ng_get(ng_vol *v, int plane, int slice, int *slot_out) {
  *slot_out = -1;
  if (!v->active || plane < 0 || plane > 2 || slice < 0) return NULL;
  pthread_mutex_lock(&v->gmu);
  int sl = ng_islot(v, plane, slice);
  if (sl >= 0) {
    v->s[sl].use = ++v->tick;
    ng_grid *g = v->s[sl].g;
    if (g) {
      v->s[sl].ref++;
      *slot_out = sl;
    }
    pthread_mutex_unlock(&v->gmu);
    return g;
  }
  pthread_mutex_unlock(&v->gmu);
  char path[1500];
  snprintf(path, sizeof path, "%s/%s/%06d.grid", v->root, ng_plane_dir[plane], slice);
  double tt0 = tr_now();
  ng_grid *g = ng_grid_load(path);
  if (!g) { /* sync fetch the one we need; prefetch its neighborhood */
    for (int d = 1; d <= 6; d++) {
      ng_prefetch(v, plane, slice + d);
      ng_prefetch(v, plane, slice - d);
    }
    static _Thread_local CURL *tcurl = NULL;
    if (!tcurl) tcurl = ng_mkcurl();
    if (ng_fetch_file(v, tcurl, plane, slice, true) == 0) g = ng_grid_load(path);
  }
  tr_tm_add(3, tt0);
  pthread_mutex_lock(&v->gmu);
  sl = ng_islot(v, plane, slice);
  if (sl >= 0) { /* lost a race */
    v->s[sl].use = ++v->tick;
    ng_grid *w = v->s[sl].g;
    if (w) {
      v->s[sl].ref++;
      *slot_out = sl;
    }
    pthread_mutex_unlock(&v->gmu);
    ng_grid_free(g);
    return w;
  }
  uint32_t victim = UINT32_MAX;
  uint64_t oldest = UINT64_MAX;
  for (uint32_t i = 0; i < NG_SLOTS; i++) {
    if (v->s[i].ref) continue; /* pinned */
    if (v->s[i].plane < 0) {
      victim = i;
      break;
    }
    if (v->s[i].use < oldest) {
      oldest = v->s[i].use;
      victim = i;
    }
  }
  if (victim == UINT32_MAX) { /* everything pinned: return uncached */
    pthread_mutex_unlock(&v->gmu);
    return g; /* ng_put frees it (not found in table) */
  }
  if (v->s[victim].plane >= 0) {
    ng_idx_del(v, v->s[victim].plane, v->s[victim].slice);
    if (v->s[victim].g) atomic_fetch_sub(&v->bytes, v->s[victim].g->bytes);
    ng_grid_free(v->s[victim].g);
  }
  v->s[victim].plane = plane;
  v->s[victim].slice = slice;
  v->s[victim].g = g; /* NULL cached too: known missing */
  v->s[victim].use = ++v->tick;
  v->s[victim].ref = g ? 1u : 0u;
  ng_idx_put(v, plane, slice, victim);
  if (g) {
    *slot_out = (int)victim;
    atomic_fetch_add(&v->bytes, g->bytes);
  }
  while (atomic_load(&v->bytes) > NG_BUDGET) { /* evict LRU unpinned */
    uint32_t ev = UINT32_MAX;
    uint64_t old2 = UINT64_MAX;
    for (uint32_t i = 0; i < NG_SLOTS; i++) {
      if (v->s[i].plane < 0 || v->s[i].ref || !v->s[i].g) continue;
      if (v->s[i].use < old2) {
        old2 = v->s[i].use;
        ev = i;
      }
    }
    if (ev == UINT32_MAX) break;
    atomic_fetch_sub(&v->bytes, v->s[ev].g->bytes);
    ng_idx_del(v, v->s[ev].plane, v->s[ev].slice);
    ng_grid_free(v->s[ev].g);
    v->s[ev].plane = -1;
    v->s[ev].g = NULL;
  }
  pthread_mutex_unlock(&v->gmu);
  return g;
}

static void ng_put(ng_vol *v, int slot, ng_grid *g) {
  if (!g) return;
  if (slot < 0) { /* was returned uncached */
    ng_grid_free(g);
    return;
  }
  pthread_mutex_lock(&v->gmu);
  if (v->s[slot].g == g && v->s[slot].ref) v->s[slot].ref--;
  pthread_mutex_unlock(&v->gmu);
}

typedef struct tr_env { /* one per solving thread */
  td_cache *dt;   /* coordinator-only (conf/DT fallback) — NULL in workers */
  ng_vol *ngv;
  ng_hood *hood;  /* [NG_NHOOD] private neighborhood cache */
  uint16_t hidx[2048]; /* open-addressed hood key -> slot+1 (the linear
                        * 512-entry scan was 40%% of trace CPU) */
  uint32_t hclock;     /* clock-hand eviction cursor */
  uint64_t htick;
  bool in_pool;   /* set inside pool workers: never nest pools */
  struct tr_pool *pl; /* coordinator: the persistent worker pool */
  unsigned rng;
  struct { int plane, slice, slot; ng_grid *g; } gc[12]; /* held grid refs:
      * one mutexed acquire per (slice, cell-solve) instead of per residual */
  uint32_t gcn;
} tr_env;

/* segment neighborhood lookup/build; center quantized to 16 px so the
 * full-path scan amortizes across residual evals and LM iterations */
static uint32_t ng_hood_hash(int plane, int slice, int qx, int qy) {
  uint64_t key = ((uint64_t)(unsigned)plane << 60) ^ ((uint64_t)(unsigned)slice << 32) ^
                 ((uint64_t)(unsigned)qx << 16) ^ (uint64_t)(unsigned)qy;
  return (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 45) & 2047u;
}

static void ng_hood_idx_del(tr_env *e, const ng_hood *h) {
  for (uint32_t p = ng_hood_hash(h->plane, h->slice, h->qx, h->qy), n = 0; n < 2048;
       p = (p + 1) & 2047u, n++) {
    uint16_t ent = e->hidx[p];
    if (!ent) return;
    if (&e->hood[ent - 1] == h) {
      e->hidx[p] = 0;
      for (uint32_t p2 = (p + 1) & 2047u; e->hidx[p2]; p2 = (p2 + 1) & 2047u) {
        uint16_t ent2 = e->hidx[p2];
        e->hidx[p2] = 0;
        const ng_hood *h2 = &e->hood[ent2 - 1];
        for (uint32_t p3 = ng_hood_hash(h2->plane, h2->slice, h2->qx, h2->qy);;
             p3 = (p3 + 1) & 2047u)
          if (!e->hidx[p3]) {
            e->hidx[p3] = ent2;
            break;
          }
      }
      return;
    }
  }
}

static ng_hood *ng_hood_get(tr_env *e, ng_grid *g, int plane, int slice, double mx,
                            double my) {
  if (!e->hood) return NULL;
  int qx = (int)floor(mx / 16.0), qy = (int)floor(my / 16.0);
  for (uint32_t p = ng_hood_hash(plane, slice, qx, qy), n = 0; n < 2048;
       p = (p + 1) & 2047u, n++) {
    uint16_t ent = e->hidx[p];
    if (!ent) break;
    ng_hood *h = &e->hood[ent - 1];
    if (h->plane == plane && h->slice == slice && h->qx == qx && h->qy == qy &&
        h->g == g) {
      h->use = ++e->htick;
      return h;
    }
  }
  uint32_t victim = e->hclock; /* clock hand: skip recently-used entries
                                * (full LRU scans were the next hotspot) */
  for (uint32_t nprobe = 0; nprobe < NG_NHOOD; nprobe++) {
    ng_hood *h = &e->hood[e->hclock];
    uint32_t cur = e->hclock;
    e->hclock = (e->hclock + 1) & (NG_NHOOD - 1);
    if (h->plane < 0 || h->use + 64 < e->htick) {
      victim = cur;
      break;
    }
    victim = cur;
  }
  double t0 = tr_now();
  ng_hood *h = &e->hood[victim];
  if (h->plane >= 0) ng_hood_idx_del(e, h); /* stale key (incl. same-key
                                             * different-grid rebuilds) */
  h->plane = plane;
  h->slice = slice;
  h->qx = qx;
  h->qy = qy;
  h->g = g;
  h->nseg = 0;
  h->use = ++e->htick;
  for (uint32_t p = ng_hood_hash(plane, slice, qx, qy);; p = (p + 1) & 2047u)
    if (!e->hidx[p]) {
      e->hidx[p] = (uint16_t)(victim + 1);
      break;
    }
  double cx = ((double)qx + 0.5) * 16.0, cy = ((double)qy + 0.5) * 16.0;
  double R = 92.0; /* query radius 80 + 16-px quantization slack */
  uint32_t paths[NG_QMAX];
  uint32_t np = ng_query(g, cx, cy, R, paths, NG_QMAX);
  static _Thread_local float *dec = NULL;
  static _Thread_local uint32_t dec_cap = 0;
  for (uint32_t p = 0; p < np && h->nseg < NG_HOODSEG; p++) {
    /* size the scratch from the record header — xy cross-sections are
     * long spirals and MUST NOT be truncated */
    uint32_t want = 1 + ng_be32(g->blob + g->precoff[paths[p]] + 8) / 2;
    if (dec_cap < want) {
      uint32_t nc2 = dec_cap ? dec_cap : 4096;
      while (nc2 < want) nc2 *= 2;
      float *nd = realloc(dec, (size_t)nc2 * 2 * sizeof *nd);
      if (!nd) break;
      dec = nd;
      dec_cap = nc2;
    }
    uint32_t npts = ng_path_decode(g, paths[p], dec, dec_cap);
    for (uint32_t s2 = 0; s2 + 1 < npts && h->nseg < NG_HOODSEG; s2++) {
      float ax = dec[(size_t)s2 * 2], ay = dec[(size_t)s2 * 2 + 1];
      float bx = dec[(size_t)(s2 + 1) * 2], by = dec[(size_t)(s2 + 1) * 2 + 1];
      double smx = ((double)ax + (double)bx) * 0.5,
             smy = ((double)ay + (double)by) * 0.5;
      double dx = smx - cx, dy = smy - cy;
      if (dx * dx + dy * dy > R * R) continue;
      float tx = bx - ax, ty = by - ay;
      float tl = sqrtf(tx * tx + ty * ty);
      if (tl < 1e-6f) continue;
      float *sg = h->sg[h->nseg];
      sg[0] = ax;
      sg[1] = ay;
      sg[2] = bx;
      sg[3] = by;
      sg[4] = -ty / tl;
      sg[5] = tx / tl;
      sg[10] = tx;
      sg[11] = ty;
      sg[12] = 1.0f / (tl * tl);
      uint8_t nb = 0;
      if (s2 > 0) {
        sg[6] = dec[(size_t)(s2 - 1) * 2];
        sg[7] = dec[(size_t)(s2 - 1) * 2 + 1];
        nb |= 1;
      }
      if (s2 + 2 < npts) {
        sg[8] = dec[(size_t)(s2 + 2) * 2];
        sg[9] = dec[(size_t)(s2 + 2) * 2 + 1];
        nb |= 2;
      }
      h->nb[h->nseg] = nb;
      h->nseg++;
    }
  }
  tr_tm_add(5, t0);
  return h;
}

/* env-level grid handle cache: refs released by tr_env_flush */
static ng_grid *ng_eget(tr_env *e, int plane, int slice, int *slot_out) {
  for (uint32_t i = 0; i < e->gcn; i++)
    if (e->gc[i].plane == plane && e->gc[i].slice == slice) {
      *slot_out = e->gc[i].slot;
      return e->gc[i].g;
    }
  int slot;
  ng_grid *g = ng_get(e->ngv, plane, slice, &slot);
  if (e->gcn == 12) { /* full: release the oldest */
    ng_put(e->ngv, e->gc[0].slot, e->gc[0].g);
    memmove(e->gc, e->gc + 1, 11 * sizeof e->gc[0]);
    e->gcn = 11;
  }
  e->gc[e->gcn].plane = plane;
  e->gc[e->gcn].slice = slice;
  e->gc[e->gcn].slot = slot;
  e->gc[e->gcn].g = g; /* NULL cached too */
  e->gcn++;
  *slot_out = slot;
  return g;
}

static void tr_env_flush(tr_env *e) {
  for (uint32_t i = 0; i < e->gcn; i++) ng_put(e->ngv, e->gc[i].slot, e->gc[i].g);
  e->gcn = 0;
}

/* ============ NormalConstraintPlane (vc3d CostFunctions.hpp) ==========
 * Scalar residual per (quad corner-rotation, plane family): where the
 * quad straddles the plane through A, compare the surface's in-slice
 * chord A->E against the grid polylines — orientation term (inverse-
 * square-distance weighted 1-|dot|) + snap term (consecutive segments of
 * ONE polyline near both endpoints; none => fixed penalty 1). */
#define NCP_W_NORMAL 10.0
#define NCP_W_SNAP 0.1
#define NCP_ROI2 4096.0   /* 64^2 */
#define NCP_QUERY_R 80.0
#define NCP_SNAP_TRIG 4.0
#define NCP_SNAP_RANGE 16.0

static double ncp_pt_seg_d2(double px, double py, double ax, double ay, double bx,
                            double by) {
  double dx = bx - ax, dy = by - ay;
  double L2 = dx * dx + dy * dy;
  double t = L2 > 1e-12 ? ((px - ax) * dx + (py - ay) * dy) / L2 : 0.0;
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  double qx = ax + t * dx - px, qy = ay + t * dy - py;
  return qx * qx + qy * qy;
}

/* project world point onto plane family: 0=xy(x,y) 1=xz(x,z) 2=yz(y,z) */
static inline void ncp_2d(int plane, const double P[3], double o[2]) {
  o[0] = plane == 2 ? P[1] : P[0];
  o[1] = plane == 0 ? P[1] : P[2];
}

static double ncp_residual(tr_env *e, int plane, const double A[3],
                           const double B1[3], const double B2[3], const double C[3]) {
  int axis = 2 - plane; /* world coord held constant by the slice */
  double b1r = B1[axis] - A[axis], b2r = B2[axis] - A[axis], cr = C[axis] - A[axis];
  if ((b1r > 0 && b2r > 0 && cr > 0) || (b1r < 0 && b2r < 0 && cr < 0)) return 0.0;
  const double *Bn = NULL;
  double bnr = 0.0;
  if (fabs(cr) < 1e-9) {
    if (fabs(b1r) > 1e-9) {
      Bn = B1;
      bnr = b1r;
    } else if (fabs(b2r) > 1e-9) {
      Bn = B2;
      bnr = b2r;
    }
  } else {
    if (b1r * cr <= 0.0) {
      Bn = B1;
      bnr = b1r;
    } else if (b2r * cr <= 0.0) {
      Bn = B2;
      bnr = b2r;
    }
  }
  if (!Bn || fabs(bnr - cr) < 1e-9) return 0.0;
  double t = -cr / (bnr - cr);
  double E[3];
  for (int a = 0; a < 3; a++) E[a] = C[a] + t * (Bn[a] - C[a]);
  int slice = (int)llround(A[axis]);
  int slot;
  ng_grid *g = ng_eget(e, plane, slice, &slot);
  if (!g || g->empty) return 0.0;
  double a2[2], e2[2];
  ncp_2d(plane, A, a2);
  ncp_2d(plane, E, e2);
  double ex = e2[0] - a2[0], ey = e2[1] - a2[1];
  double el = sqrt(ex * ex + ey * ey);
  if (el < 1e-9) return 0.0;
  double enx = ey / el, eny = -ex / el; /* edge normal */
  double mx = (a2[0] + e2[0]) * 0.5, my = (a2[1] + e2[1]) * 0.5;
  ng_hood *hd = ng_hood_get(e, g, plane, slice, mx, my);
  /* no segments: normal term 0, snap takes the fixed no-target penalty */
  double wsum = 0.0, nsum = 0.0;
  double best_snap = -1.0, best_score = -1.0;
  float fmx = (float)mx, fmy = (float)my;
  float fex = (float)e2[0], fey = (float)e2[1];
  float fenx = (float)enx, feny = (float)eny;
  if (hd)
    for (uint32_t k = 0; k < hd->nseg; k++) {
      const float *sg = hd->sg[k];
      float dmx = (sg[0] + sg[2]) * 0.5f - fmx, dmy = (sg[1] + sg[3]) * 0.5f - fmy;
      float d2 = dmx * dmx + dmy * dmy; /* midpoint-to-midpoint, vc3d */
      if (d2 <= (float)NCP_ROI2) {
        float dd = d2 < 10.0f ? 10.0f : d2;
        float dot = fabsf(fenx * sg[4] + feny * sg[5]);
        wsum += (double)(1.0f / dd);
        nsum += (double)((1.0f - dot) / dd);
      }
      /* snap: E near this segment, A near a NEIGHBOR segment of the same
       * polyline (prev/next captured at hood build) */
      float tt = ((fex - sg[0]) * sg[10] + (fey - sg[1]) * sg[11]) * sg[12];
      tt = tt < 0.0f ? 0.0f : (tt > 1.0f ? 1.0f : tt);
      float qex = sg[0] + tt * sg[10] - fex, qey = sg[1] + tt * sg[11] - fey;
      float d2e = qex * qex + qey * qey;
      if ((double)d2e < NCP_SNAP_TRIG * NCP_SNAP_TRIG) {
        for (int dsn = 0; dsn < 2; dsn++) {
          if (!(hd->nb[k] & (dsn ? 2u : 1u))) continue;
          double qax = dsn ? (double)sg[2] : (double)sg[6];
          double qay = dsn ? (double)sg[3] : (double)sg[7];
          double qbx = dsn ? (double)sg[8] : (double)sg[0];
          double qby = dsn ? (double)sg[9] : (double)sg[1];
          double d2a = ncp_pt_seg_d2(a2[0], a2[1], qax, qay, qbx, qby);
          if (d2a < NCP_SNAP_RANGE * NCP_SNAP_RANGE) {
            double d1n = sqrt(d2a) / NCP_SNAP_RANGE,
                   d2n = sqrt((double)d2e) / NCP_SNAP_TRIG;
            double score = 0.5 * (d1n + d2n); /* vc3d: min by score... */
            if (best_score < 0.0 || score < best_score) {
              best_score = score;
              best_snap = d1n * (1.0 - d2n) + d2n; /* ...keep the value */
            }
          }
        }
      }
    }
  double normal_loss = wsum > 1e-9 ? nsum / wsum : 0.0;
  double snap_loss = best_snap >= 0.0 ? best_snap : 1.0;
  /* angular weight: cross-section validity of the quad in this plane */
  double v1_[3], v2_[3], nq[3];
  for (int a = 0; a < 3; a++) {
    v1_[a] = Bn[a] - A[a];
    v2_[a] = Bn[a] - C[a];
  }
  nq[0] = v1_[1] * v2_[2] - v1_[2] * v2_[1];
  nq[1] = v1_[2] * v2_[0] - v1_[0] * v2_[2];
  nq[2] = v1_[0] * v2_[1] - v1_[1] * v2_[0];
  double nl2 = nq[0] * nq[0] + nq[1] * nq[1] + nq[2] * nq[2];
  (void)slot;
  if (nl2 < 1e-18) return 0.0;
  double na = nq[axis] / sqrt(nl2);
  double aw = 0.5 * (1.0 - na * na);
  return (NCP_W_NORMAL * normal_loss + NCP_W_SNAP * snap_loss) * aw;
}

/* Fused residual + forward-difference gradient: the base evaluation and
 * its three +h probes differ by half a voxel — same slice, same hood,
 * near-identical chords — so all four run through ONE pass over the
 * segment neighborhood with 4-wide math instead of four passes with
 * four hood lookups. out[0]=r(x), out[1..3]=r(x + h e_k). */
static void ncp_residual4(tr_env *e, int plane, const double Qr[4][3], int fxr,
                          double h, double out[4]) {
  int axis = 2 - plane;
  double V[4][3]; /* free-corner variants */
  for (int v = 0; v < 4; v++) {
    memcpy(V[v], Qr[fxr], sizeof V[v]);
    if (v) V[v][v - 1] += h;
  }
  /* per-variant chord prep (cheap scalar) */
  bool ok[4];
  int slice[4];
  float vmx[4], vmy[4], vax[4], vay[4], vex4[4], vey4[4], venx[4], veny[4];
  double aw[4];
  int nok = 0;
  for (int v = 0; v < 4; v++) {
    const double *cor[4];
    for (int c = 0; c < 4; c++) cor[c] = c == fxr ? V[v] : Qr[c];
    const double *A = cor[0], *B1 = cor[1], *B2 = cor[2], *C = cor[3];
    ok[v] = false;
    out[v] = 0.0;
    double b1r = B1[axis] - A[axis], b2r = B2[axis] - A[axis], cr = C[axis] - A[axis];
    if ((b1r > 0 && b2r > 0 && cr > 0) || (b1r < 0 && b2r < 0 && cr < 0)) continue;
    const double *Bn = NULL;
    double bnr = 0.0;
    if (fabs(cr) < 1e-9) {
      if (fabs(b1r) > 1e-9) {
        Bn = B1;
        bnr = b1r;
      } else if (fabs(b2r) > 1e-9) {
        Bn = B2;
        bnr = b2r;
      }
    } else {
      if (b1r * cr <= 0.0) {
        Bn = B1;
        bnr = b1r;
      } else if (b2r * cr <= 0.0) {
        Bn = B2;
        bnr = b2r;
      }
    }
    if (!Bn || fabs(bnr - cr) < 1e-9) continue;
    double t = -cr / (bnr - cr);
    double E[3];
    for (int a = 0; a < 3; a++) E[a] = C[a] + t * (Bn[a] - C[a]);
    double a2[2], e2[2];
    ncp_2d(plane, A, a2);
    ncp_2d(plane, E, e2);
    double ex = e2[0] - a2[0], ey = e2[1] - a2[1];
    double el = sqrt(ex * ex + ey * ey);
    if (el < 1e-9) continue;
    double v1_[3], v2_[3], nq[3];
    for (int a = 0; a < 3; a++) {
      v1_[a] = Bn[a] - A[a];
      v2_[a] = Bn[a] - C[a];
    }
    nq[0] = v1_[1] * v2_[2] - v1_[2] * v2_[1];
    nq[1] = v1_[2] * v2_[0] - v1_[0] * v2_[2];
    nq[2] = v1_[0] * v2_[1] - v1_[1] * v2_[0];
    double nl2 = nq[0] * nq[0] + nq[1] * nq[1] + nq[2] * nq[2];
    if (nl2 < 1e-18) continue;
    double na = nq[axis] / sqrt(nl2);
    aw[v] = 0.5 * (1.0 - na * na);
    slice[v] = (int)llround(A[axis]);
    vax[v] = (float)a2[0];
    vay[v] = (float)a2[1];
    vex4[v] = (float)e2[0];
    vey4[v] = (float)e2[1];
    venx[v] = (float)(ey / el);
    veny[v] = (float)(-ex / el);
    vmx[v] = (float)((a2[0] + e2[0]) * 0.5);
    vmy[v] = (float)((a2[1] + e2[1]) * 0.5);
    ok[v] = true;
    nok++;
  }
  if (!nok) return;
  /* one hood serves every variant (mids differ by <= h; the hood radius
   * carries 12 px of quantization slack). Variants on a different slice
   * (free coord crossing a .5 boundary) fall back to scalar. */
  int s0 = -1;
  for (int v = 0; v < 4; v++)
    if (ok[v]) {
      if (s0 < 0) s0 = slice[v];
      else if (slice[v] != s0) {
        double Qs[4][3];
        for (int c = 0; c < 4; c++)
          memcpy(Qs[c], c == fxr ? V[v] : Qr[c], sizeof Qs[c]);
        out[v] = ncp_residual(e, plane, Qs[0], Qs[1], Qs[2], Qs[3]);
        ok[v] = false;
      }
    }
  int slot;
  ng_grid *g = ng_eget(e, plane, s0, &slot);
  (void)slot;
  float wsum[4] = {0}, nsum[4] = {0};
  float best_snap[4] = {-1, -1, -1, -1}, best_score4[4] = {-1, -1, -1, -1};
  if (g && !g->empty) {
    double m0x = 0, m0y = 0;
    int nv = 0;
    for (int v = 0; v < 4; v++)
      if (ok[v]) {
        m0x += (double)vmx[v];
        m0y += (double)vmy[v];
        nv++;
      }
    ng_hood *hd = ng_hood_get(e, g, plane, s0, m0x / nv, m0y / nv);
    if (hd)
      for (uint32_t k = 0; k < hd->nseg; k++) {
        const float *sg = hd->sg[k];
        float smx = (sg[0] + sg[2]) * 0.5f, smy = (sg[1] + sg[3]) * 0.5f;
        for (int v = 0; v < 4; v++) { /* 4-wide: clang vectorizes this */
          float dmx = smx - vmx[v], dmy = smy - vmy[v];
          float d2 = dmx * dmx + dmy * dmy;
          if (d2 <= (float)NCP_ROI2) {
            float dd = d2 < 10.0f ? 10.0f : d2;
            float dot = fabsf(venx[v] * sg[4] + veny[v] * sg[5]);
            wsum[v] += 1.0f / dd;
            nsum[v] += (1.0f - dot) / dd;
          }
          float tt = ((vex4[v] - sg[0]) * sg[10] + (vey4[v] - sg[1]) * sg[11]) * sg[12];
          tt = tt < 0.0f ? 0.0f : (tt > 1.0f ? 1.0f : tt);
          float qex = sg[0] + tt * sg[10] - vex4[v], qey = sg[1] + tt * sg[11] - vey4[v];
          float d2e = qex * qex + qey * qey;
          if (d2e < (float)(NCP_SNAP_TRIG * NCP_SNAP_TRIG)) {
            for (int dsn = 0; dsn < 2; dsn++) {
              if (!(hd->nb[k] & (dsn ? 2u : 1u))) continue;
              double qax = dsn ? (double)sg[2] : (double)sg[6];
              double qay = dsn ? (double)sg[3] : (double)sg[7];
              double qbx = dsn ? (double)sg[8] : (double)sg[0];
              double qby = dsn ? (double)sg[9] : (double)sg[1];
              double d2a =
                  ncp_pt_seg_d2((double)vax[v], (double)vay[v], qax, qay, qbx, qby);
              if (d2a < NCP_SNAP_RANGE * NCP_SNAP_RANGE) {
                double d1n = sqrt(d2a) / NCP_SNAP_RANGE,
                       d2n = sqrt((double)d2e) / NCP_SNAP_TRIG;
                float score = (float)(0.5 * (d1n + d2n));
                if (best_score4[v] < 0.0f || score < best_score4[v]) {
                  best_score4[v] = score;
                  best_snap[v] = (float)(d1n * (1.0 - d2n) + d2n);
                }
              }
            }
          }
        }
      }
  }
  for (int v = 0; v < 4; v++) {
    if (!ok[v]) continue;
    double normal_loss = wsum[v] > 1e-9f ? (double)(nsum[v] / wsum[v]) : 0.0;
    double snap_loss = best_snap[v] >= 0.0f ? (double)best_snap[v] : 1.0;
    out[v] = (NCP_W_NORMAL * normal_loss + NCP_W_SNAP * snap_loss) * aw[v];
  }
}

/* ===================== tiny LM over one 3-vector ===================== */
typedef struct tr_nlsq {
  double JTJ[9], JTr[3], cost;
} tr_nlsq;

static void nq_begin(tr_nlsq *a) { memset(a, 0, sizeof *a); }

static void nq_add(tr_nlsq *a, double r, const double J[3]) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) a->JTJ[i * 3 + j] += J[i] * J[j];
    a->JTr[i] += J[i] * r;
  }
  a->cost += 0.5 * r * r;
}

static int nq_step(const double A[9], const double b[3], double lm, double d[3]) {
  double M[9];
  memcpy(M, A, sizeof M);
  for (int i = 0; i < 3; i++) {
    double di = A[i * 3 + i];
    M[i * 3 + i] += lm * (di > 1e-12 ? di : 1e-12);
  }
  double det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
               M[2] * (M[3] * M[7] - M[4] * M[6]);
  if (fabs(det) < 1e-30) return -1;
  double inv[9];
  inv[0] = (M[4] * M[8] - M[5] * M[7]) / det;
  inv[1] = (M[2] * M[7] - M[1] * M[8]) / det;
  inv[2] = (M[1] * M[5] - M[2] * M[4]) / det;
  inv[3] = (M[5] * M[6] - M[3] * M[8]) / det;
  inv[4] = (M[0] * M[8] - M[2] * M[6]) / det;
  inv[5] = (M[2] * M[3] - M[0] * M[5]) / det;
  inv[6] = (M[3] * M[7] - M[4] * M[6]) / det;
  inv[7] = (M[1] * M[6] - M[0] * M[7]) / det;
  inv[8] = (M[0] * M[4] - M[1] * M[3]) / det;
  for (int i = 0; i < 3; i++)
    d[i] = -(inv[i * 3 + 0] * b[0] + inv[i * 3 + 1] * b[1] + inv[i * 3 + 2] * b[2]);
  return 0;
}

/* ====================== spiral winding frame ======================
 * The useful core of vc3d's spiral service (scripts/spiral), sized to fit
 * inside the tracer: an umbilicus-anchored winding number per grid cell
 * plus a global Archimedean fit rho ~ r0(z) + omega*w refit every
 * generation on our own points. A point that slips onto an adjacent wrap
 * changes rho by ~omega without changing its (combinatorial) winding
 * number, so the prior pulls it back — the wrap-jump immunity the torch
 * fit provides upstream, without its inputs. */
#define TR_SP_KMAX 64    /* r0(z) knots (piecewise linear) */
#define TR_SP_KNOT 256.0 /* knot spacing, slices */
#define TR_SP_SIGMA 75.0 /* umbilicus smoothing, slices (vc3d) */

/* smoothed centerline + slope at z (clamped); false when no umbilicus */
static bool tr_uc_at(const r3d_tracer *t, double z, double *cx, double *cy,
                     double *dxz, double *dyz) {
  if (!t->uc || t->ucn < 2) return false;
  double zc = z < 0 ? 0 : (z > (double)t->ucn - 1 ? (double)t->ucn - 1 : z);
  uint32_t i0 = (uint32_t)zc;
  if (i0 >= t->ucn - 1) i0 = t->ucn - 2;
  double f = zc - (double)i0;
  const double *a = t->uc + (size_t)i0 * 2, *b = a + 2;
  *cx = a[0] * (1 - f) + b[0] * f;
  *cy = a[1] * (1 - f) + b[1] * f;
  if (dxz) *dxz = b[0] - a[0];
  if (dyz) *dyz = b[1] - a[1];
  return true;
}

/* angle about the umbilicus at P's slice; false without an umbilicus */
static bool tr_theta_of(const r3d_tracer *t, const double P[3], double *th) {
  double cx, cy;
  if (!tr_uc_at(t, P[2], &cx, &cy, NULL, NULL)) return false;
  double ux = P[0] - cx, uy = P[1] - cy;
  if (ux * ux + uy * uy < 1e-12) return false;
  *th = atan2(uy, ux);
  return true;
}

/* interpolate the umbilicus polyline per slice, then Gaussian-smooth */
static void tr_uc_build(r3d_tracer *t, uint32_t nz) {
  free(t->uc);
  t->uc = NULL;
  t->ucn = 0;
  const r3d_umbilicus *u = &t->umb;
  if (u->count < 2 || nz < 2) return;
  double *raw = malloc((size_t)nz * 2 * sizeof *raw);
  double *sm = malloc((size_t)nz * 2 * sizeof *sm);
  if (!raw || !sm) {
    free(raw);
    free(sm);
    return;
  }
  for (uint32_t z = 0; z < nz; z++) { /* points are z-ascending */
    double zz = (double)z;
    size_t s = 0;
    while (s + 2 < u->count && u->points[s + 1].z < zz) s++;
    const r3d_umbilicus_point *a = &u->points[s], *b = &u->points[s + 1];
    double dz = b->z - a->z;
    double f = dz > 1e-9 ? (zz - a->z) / dz : 0.0;
    if (f < 0) f = 0;
    if (f > 1) f = 1;
    raw[(size_t)z * 2 + 0] = a->x * (1 - f) + b->x * f;
    raw[(size_t)z * 2 + 1] = a->y * (1 - f) + b->y * f;
  }
  int rad = (int)(3.0 * TR_SP_SIGMA);
  for (uint32_t z = 0; z < nz; z++) {
    double acc[2] = {0, 0}, wsum = 0;
    for (int d = -rad; d <= rad; d++) {
      long zz = (long)z + d;
      if (zz < 0 || zz >= (long)nz) continue;
      double g = exp(-0.5 * (double)d * (double)d / (TR_SP_SIGMA * TR_SP_SIGMA));
      acc[0] += g * raw[zz * 2 + 0];
      acc[1] += g * raw[zz * 2 + 1];
      wsum += g;
    }
    sm[(size_t)z * 2 + 0] = acc[0] / wsum;
    sm[(size_t)z * 2 + 1] = acc[1] / wsum;
  }
  free(raw);
  t->uc = sm;
  t->ucn = nz;
}

/* r0(z) + slope from the current fit */
static void tr_sp_r0_at(const r3d_tracer *t, double z, double *r0, double *slope) {
  double u2 = (z - t->sp_z0) / t->sp_dz;
  if (u2 < 0) u2 = 0;
  if (u2 > (double)t->sp_k - 1) u2 = (double)t->sp_k - 1;
  uint32_t i0 = (uint32_t)u2;
  if (i0 >= t->sp_k - 1) i0 = t->sp_k - 2;
  double f = u2 - (double)i0;
  *r0 = t->sp_r0[i0] * (1 - f) + t->sp_r0[i0 + 1] * f;
  if (slope) *slope = (t->sp_r0[i0 + 1] - t->sp_r0[i0]) / t->sp_dz;
}

/* Cholesky solve (in place), n <= TR_SP_KMAX+1 */
static int tr_chol_solve(double *A, double *b, int n) {
  for (int i = 0; i < n; i++) {
    for (int j2 = 0; j2 <= i; j2++) {
      double s = A[i * n + j2];
      for (int k = 0; k < j2; k++) s -= A[i * n + k] * A[j2 * n + k];
      if (i == j2) {
        if (s <= 1e-12) return -1;
        A[i * n + i] = sqrt(s);
      } else {
        A[i * n + j2] = s / A[j2 * n + j2];
      }
    }
  }
  for (int i = 0; i < n; i++) {
    double s = b[i];
    for (int k = 0; k < i; k++) s -= A[i * n + k] * b[k];
    b[i] = s / A[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double s = b[i];
    for (int k = i + 1; k < n; k++) s -= A[k * n + i] * b[k];
    b[i] = s / A[i * n + i];
  }
  return 0;
}

/* refit rho ~ r0(z) + omega*w over all SET cells (IRLS, Cauchy). Joint
 * omega is identifiable only once the patch spans a full winding; below
 * that the measured inter-sheet gap (sp_om_meas, from radial DT rays)
 * substitutes and only r0(z) is solved — both signs are tried. Runs on
 * the coordinator between generations (pool idle) — solver threads read
 * the published fit without locking. */
static void tr_spiral_fit(r3d_tracer *t) {
  if (!t->uc || !(t->cfg.wind_weight > 0)) return;
  uint32_t nz = t->ucn;
  uint32_t K = (uint32_t)((double)nz / TR_SP_KNOT) + 2;
  if (K < 2) K = 2;
  if (K > TR_SP_KMAX) K = TR_SP_KMAX;
  double z0 = 0.0, dz = (double)nz / (double)(K - 1);
  uint64_t N = (uint64_t)t->W * t->H;
  /* winding span decides the omega mode */
  double wmin = 1e30, wmax = -1e30;
  uint32_t nobs = 0;
  for (uint64_t k = 0; k < N; k++) {
    if (t->state[k] != R3D_TR_SET) continue;
    double wd = (double)t->wind[k];
    if (wd < wmin) wmin = wd;
    if (wd > wmax) wmax = wd;
    nobs++;
  }
  double om_meas = t->sp_om_meas;
  int ncand;
  double cand[2];
  bool joint = wmax - wmin >= 1.0;
  double psign = t->sp_valid && t->sp_omega < 0 ? -1.0 : 1.0;
  if (nobs < 200) {
    return;
  } else if (joint) {
    ncand = 1;
    cand[0] = 0.0; /* solved */
  } else if (om_meas >= 4.0) {
    ncand = 2; /* incumbent sign first; challenger must clearly win */
    cand[0] = psign * om_meas;
    cand[1] = -psign * om_meas;
  } else {
    return;
  }
  int nmax = (int)K + 5; /* r0 knots + 4 theta harmonics + omega */
  double *A = malloc((size_t)nmax * (size_t)nmax * sizeof *A);
  double *x = malloc((size_t)nmax * sizeof *x);
  double *r0b = malloc((size_t)K * sizeof *r0b);
  double *best_r0 = malloc((size_t)K * sizeof *best_r0);
  if (!A || !x || !r0b || !best_r0) {
    free(A);
    free(x);
    free(r0b);
    free(best_r0);
    return;
  }
  double best_om = 0.0, best_rms = 1e30, best_ab[4] = {0, 0, 0, 0};
  bool have = false;
  for (int ci = 0; ci < ncand; ci++) {
    double om_fix = cand[ci];
    int n = om_fix != 0.0 ? (int)K + 4 : (int)K + 5; /* [r0.. ab[4] (omega)] */
    double ab[4] = {0, 0, 0, 0};
    double omega = om_fix != 0.0 ? om_fix : (t->sp_valid ? t->sp_omega : 0.0);
    bool warm = false; /* r0b holds a previous sweep's solution */
    double rms = 1e30;
    for (int sweep = 0; sweep < 4; sweep++) {
      memset(A, 0, (size_t)n * (size_t)n * sizeof *A);
      memset(x, 0, (size_t)n * sizeof *x);
      double wtot = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        if (t->state[k] != R3D_TR_SET) continue;
        const double *P = t->pos + k * 3;
        double cx, cy;
        if (!tr_uc_at(t, P[2], &cx, &cy, NULL, NULL)) continue;
        double uxv = P[0] - cx, uyv = P[1] - cy;
        double rho = hypot(uxv, uyv);
        if (rho < 1e-9) continue;
        double th = atan2(uyv, uxv);
        double wd = (double)t->wind[k];
        double zc = (P[2] - z0) / dz;
        if (zc < 0) zc = 0;
        if (zc > (double)K - 1) zc = (double)K - 1;
        uint32_t i0 = (uint32_t)zc;
        if (i0 >= K - 1) i0 = K - 2;
        double f = zc - (double)i0;
        double hh[4] = {cos(th), sin(th), cos(2 * th), sin(2 * th)};
        double wgt = 1.0;
        if (warm && fabs(omega) > 1e-9) { /* Cauchy, scale omega/4 */
          double r0v = r0b[i0] * (1 - f) + r0b[i0 + 1] * f + ab[0] * hh[0] +
                       ab[1] * hh[1] + ab[2] * hh[2] + ab[3] * hh[3];
          double r = (rho - (r0v + omega * wd)) / (0.25 * fabs(omega));
          wgt = 1.0 / (1.0 + r * r);
        }
        double tgt = om_fix != 0.0 ? rho - om_fix * wd : rho;
        double phi[7] = {1 - f, f, hh[0], hh[1], hh[2], hh[3], wd};
        int idx[7] = {(int)i0,      (int)i0 + 1,  (int)K,     (int)K + 1,
                      (int)K + 2,   (int)K + 3,   (int)K + 4};
        int nb = om_fix != 0.0 ? 6 : 7;
        for (int a2 = 0; a2 < nb; a2++) {
          for (int b2 = 0; b2 < nb; b2++)
            A[idx[a2] * n + idx[b2]] += wgt * phi[a2] * phi[b2];
          x[idx[a2]] += wgt * phi[a2] * tgt;
        }
        wtot += wgt;
      }
      double lam = 1e-2 * wtot / (double)K; /* smoothness ridge on r0 */
      for (uint32_t kk = 1; kk + 1 < K; kk++) {
        int ids[3] = {(int)kk - 1, (int)kk, (int)kk + 1};
        double co[3] = {-1.0, 2.0, -1.0};
        for (int a2 = 0; a2 < 3; a2++)
          for (int b2 = 0; b2 < 3; b2++)
            A[ids[a2] * n + ids[b2]] += lam * co[a2] * co[b2];
      }
      for (int i = 0; i < n; i++) A[i * n + i] += 1e-6 * (wtot / (double)n + 1.0);
      if (tr_chol_solve(A, x, n) != 0) break;
      for (uint32_t kk = 0; kk < K; kk++) r0b[kk] = x[kk];
      for (int a2 = 0; a2 < 4; a2++) ab[a2] = x[(int)K + a2];
      if (om_fix == 0.0) omega = x[n - 1];
      warm = true;
      /* robust rms over inliers */
      double se = 0.0;
      uint32_t ni = 0;
      for (uint64_t k = 0; k < N; k++) {
        if (t->state[k] != R3D_TR_SET) continue;
        const double *P = t->pos + k * 3;
        double cx, cy;
        if (!tr_uc_at(t, P[2], &cx, &cy, NULL, NULL)) continue;
        double rho = hypot(P[0] - cx, P[1] - cy);
        double zc = (P[2] - z0) / dz;
        if (zc < 0) zc = 0;
        if (zc > (double)K - 1) zc = (double)K - 1;
        uint32_t i0 = (uint32_t)zc;
        if (i0 >= K - 1) i0 = K - 2;
        double f = zc - (double)i0;
        double thv = atan2(P[1] - cy, P[0] - cx);
        double r = rho - (r0b[i0] * (1 - f) + r0b[i0 + 1] * f + ab[0] * cos(thv) +
                          ab[1] * sin(thv) + ab[2] * cos(2 * thv) +
                          ab[3] * sin(2 * thv) + omega * (double)t->wind[k]);
        if (fabs(r) < 0.5 * fabs(omega) + 1e-9) {
          se += r * r;
          ni++;
        }
      }
      rms = ni ? sqrt(se / (double)ni) : 1e30;
    }
    /* sign hysteresis: the challenger (ci==1) must beat the incumbent
     * by 5% — at small winding spans the sign is nearly unidentifiable
     * and free flapping would jitter the prior every generation */
    double need = ci == 1 && have ? 0.95 * best_rms : best_rms;
    if (warm && rms < need) {
      best_rms = rms;
      best_om = omega;
      memcpy(best_r0, r0b, (size_t)K * sizeof *best_r0);
      memcpy(best_ab, ab, sizeof best_ab);
      have = true;
    }
  }
  if (have) { /* publish (worker thread only; solver reads race benignly) */
    for (uint32_t kk = 0; kk < K; kk++) t->sp_r0[kk] = best_r0[kk];
    for (int a2 = 0; a2 < 4; a2++) t->sp_ab[a2] = best_ab[a2];
    t->sp_z0 = z0;
    t->sp_dz = dz;
    t->sp_k = K;
  }
  pthread_mutex_lock(&t->mu);
  if (have) {
    t->sp_omega = best_om;
    t->sp_rms = best_rms;
  }
  t->sp_valid = have && fabs(best_om) >= 2.0 && best_rms <= 0.5 * fabs(best_om);
  pthread_mutex_unlock(&t->mu);
  free(A);
  free(x);
  free(r0b);
  free(best_r0);
}

/* after a fit: distrust cells the spiral model calls wrong-wrap */
static void tr_spiral_flag(r3d_tracer *t) {
  if (!t->sp_valid) return;
  uint64_t N = (uint64_t)t->W * t->H;
  for (uint64_t k = 0; k < N; k++) {
    if (t->state[k] != R3D_TR_SET) continue;
    const double *P = t->pos + k * 3;
    double cx, cy;
    if (!tr_uc_at(t, P[2], &cx, &cy, NULL, NULL)) continue;
    double ux = P[0] - cx, uy = P[1] - cy;
    double rho = hypot(ux, uy);
    double th = atan2(uy, ux);
    double r0v;
    tr_sp_r0_at(t, P[2], &r0v, NULL);
    double H = t->sp_ab[0] * cos(th) + t->sp_ab[1] * sin(th) +
               t->sp_ab[2] * cos(2 * th) + t->sp_ab[3] * sin(2 * th);
    double r = rho - (r0v + H + t->sp_omega * (double)t->wind[k]);
    /* only clamp genuinely wrap-like errors: past half a sheet gap AND
     * past 2.5x the fit's own noise floor (a rigid model on a squashed
     * scroll cannot justify distrusting 2-sigma cells) */
    double thr = 0.5 * fabs(t->sp_omega);
    if (2.5 * t->sp_rms > thr) thr = 2.5 * t->sp_rms;
    if (fabs(r) > thr && t->conf[k] > 0.25f && !(t->dsup && t->dsup[k] >= 2))
      t->conf[k] = 0.25f;
  }
}

/* ====================== donor segments (fusion) ======================
 * The tracer-side half of vc3d's grow_surf_from_surfs: saved tifxyz
 * patches become donors; candidates near a donor are pulled onto it
 * (SurfaceLossD, w=0.1) and their agreement is counted, while cells no
 * donor covers grow by raw tracing exactly as before. The spiral frame
 * supplies what vc3d's fusion lacks: a donor whose registered winding
 * disagrees with the cell is never consulted, so adjacent wraps cannot
 * fuse no matter how close they pass. */
static int tr_dcmp(const void *a, const void *b);

#define TR_FUS_TH 2.0      /* same_surface_th (vc3d) */
#define TR_FUS_PULL 8.0    /* capture radius for the donor pull, vox */
#define TR_FUS_WIND_TH 0.3 /* winding agreement gate, windings */
#define TR_W_SURF 0.1      /* SurfaceLossD weight (vc3d hardcoded) */

typedef struct tr_donor {
  r3d_tifxyz s;
  float *dwind;  /* optional winding.tif channel, w*h; < -1e29 = none */
  double woff;   /* registered: global_wind = donor_wind + woff */
  bool woff_ok;
} tr_donor;

typedef struct tr_didx { /* uniform hash grid over donor quads */
  double cs;
  uint32_t nbuck; /* power of two */
  uint32_t *head; /* bucket -> entry+1 */
  uint32_t *next;
  uint64_t *qid; /* donor<<48 | j<<24 | i */
  uint32_t nent, cap;
} tr_didx;

typedef struct tr_dons {
  tr_donor *d;
  uint32_t n;
  tr_didx ix;
} tr_dons;

static uint32_t tr_didx_h(const tr_didx *ix, long cx, long cy, long cz) {
  uint64_t h = (uint64_t)cx * 0x9E3779B185EBCA87ull ^
               (uint64_t)cy * 0xC2B2AE3D27D4EB4Full ^ (uint64_t)cz * 0x165667B19E3779F9ull;
  return (uint32_t)(h >> 32) & (ix->nbuck - 1);
}

static int tr_didx_put(tr_didx *ix, long cx, long cy, long cz, uint64_t qid) {
  if (ix->nent == ix->cap) {
    uint32_t nc = ix->cap ? ix->cap * 2 : 4096;
    uint32_t *nn = realloc(ix->next, (size_t)nc * sizeof *nn);
    uint64_t *nq = realloc(ix->qid, (size_t)nc * sizeof *nq);
    if (!nn || !nq) {
      free(nn ? nn : ix->next);
      ix->next = NULL;
      return -1;
    }
    ix->next = nn;
    ix->qid = nq;
    ix->cap = nc;
  }
  uint32_t b = tr_didx_h(ix, cx, cy, cz);
  ix->qid[ix->nent] = qid;
  ix->next[ix->nent] = ix->head[b];
  ix->head[b] = ix->nent + 1;
  ix->nent++;
  return 0;
}

/* closest point on triangle abc to p (Eberly); returns squared distance,
 * fills q and barycentric (u for b, v for c) */
static double tr_tri_close(const double p[3], const double a[3], const double b[3],
                           const double c[3], double q[3], double *ub, double *vc) {
  double ab[3], ac[3], ap[3];
  for (int k = 0; k < 3; k++) {
    ab[k] = b[k] - a[k];
    ac[k] = c[k] - a[k];
    ap[k] = p[k] - a[k];
  }
  double d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
  double d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];
  double u = 0, v = 0;
  if (d1 <= 0 && d2 <= 0) {
    u = 0;
    v = 0;
  } else {
    double bp[3], cp[3];
    for (int k = 0; k < 3; k++) {
      bp[k] = p[k] - b[k];
      cp[k] = p[k] - c[k];
    }
    double d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
    double d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];
    double d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
    double d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
    if (d3 >= 0 && d4 <= d3) {
      u = 1;
      v = 0;
    } else if (d6 >= 0 && d5 <= d6) {
      u = 0;
      v = 1;
    } else {
      double vc0 = d1 * d4 - d3 * d2;
      double vb = d5 * d2 - d1 * d6;
      double va = d3 * d6 - d5 * d4;
      if (vc0 <= 0 && d1 >= 0 && d3 <= 0) {
        u = d1 / (d1 - d3);
        v = 0;
      } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        u = 0;
        v = d2 / (d2 - d6);
      } else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        u = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        v = 1 - u;
      } else {
        double den = va + vb + vc0;
        u = vb / den;
        v = vc0 / den;
      }
    }
  }
  double d2s = 0;
  for (int k = 0; k < 3; k++) {
    q[k] = a[k] + u * ab[k] + v * ac[k];
    double dd = p[k] - q[k];
    d2s += dd * dd;
  }
  *ub = u;
  *vc = v;
  return d2s;
}

/* nearest donor surface point to p within `rad`; returns donor index or
 * -1; fills q (closest point) and wnd (interpolated donor winding in the
 * GLOBAL frame; < -1e29 when the donor has no registered winding) */
static int tr_don_closest(const tr_dons *dn, const double p[3], double rad, double q[3],
                          double *wnd) {
  if (!dn || !dn->ix.nent) return -1;
  const tr_didx *ix = &dn->ix;
  long c0[3], c1[3];
  for (int k = 0; k < 3; k++) {
    c0[k] = (long)floor((p[k] - rad) / ix->cs);
    c1[k] = (long)floor((p[k] + rad) / ix->cs);
  }
  double bd2 = rad * rad;
  int bdon = -1;
  for (long cz = c0[2]; cz <= c1[2]; cz++)
    for (long cy = c0[1]; cy <= c1[1]; cy++)
      for (long cx = c0[0]; cx <= c1[0]; cx++)
        for (uint32_t e = ix->head[tr_didx_h(ix, cx, cy, cz)]; e; e = ix->next[e - 1]) {
          uint64_t id = ix->qid[e - 1];
          uint32_t di = (uint32_t)(id >> 48);
          uint32_t j = (uint32_t)(id >> 24) & 0xFFFFFFu;
          uint32_t i = (uint32_t)(id & 0xFFFFFFu);
          const tr_donor *d = &dn->d[di];
          const float *p00 = r3d_tifxyz_at(&d->s, i, j);
          const float *p10 = r3d_tifxyz_at(&d->s, i + 1, j);
          const float *p01 = r3d_tifxyz_at(&d->s, i, j + 1);
          const float *p11 = r3d_tifxyz_at(&d->s, i + 1, j + 1);
          double A[3] = {(double)p00[0], (double)p00[1], (double)p00[2]};
          double B[3] = {(double)p10[0], (double)p10[1], (double)p10[2]};
          double C[3] = {(double)p01[0], (double)p01[1], (double)p01[2]};
          double D[3] = {(double)p11[0], (double)p11[1], (double)p11[2]};
          double qq[3], ub, vc2;
          for (int tri = 0; tri < 2; tri++) {
            double d2 = tri == 0 ? tr_tri_close(p, A, B, C, qq, &ub, &vc2)
                                 : tr_tri_close(p, D, C, B, qq, &ub, &vc2);
            if (d2 >= bd2) continue;
            bd2 = d2;
            bdon = (int)di;
            memcpy(q, qq, sizeof qq);
            if (wnd) {
              *wnd = -1e30;
              if (d->dwind && d->woff_ok) {
                uint64_t k00 = (uint64_t)j * d->s.w + i;
                float w00 = d->dwind[k00], w10 = d->dwind[k00 + 1];
                float w01 = d->dwind[k00 + d->s.w], w11 = d->dwind[k00 + d->s.w + 1];
                if (w00 > -1e29f && w10 > -1e29f && w01 > -1e29f && w11 > -1e29f) {
                  double wA, wB, wC;
                  if (tri == 0) {
                    wA = (double)w00;
                    wB = (double)w10;
                    wC = (double)w01;
                  } else {
                    wA = (double)w11;
                    wB = (double)w01;
                    wC = (double)w10;
                  }
                  *wnd = wA + ub * (wB - wA) + vc2 * (wC - wA) + d->woff;
                }
              }
            }
          }
        }
  return bdon;
}

/* count donors within TR_FUS_TH of p (winding-gated) — the consensus
 * number vc3d scores candidates by */
static int tr_don_support(const tr_dons *dn, const double p[3], double cellw,
                          bool have_w) {
  if (!dn || !dn->ix.nent || dn->n > 64) return 0;
  double mind2[64], mwnd[64];
  for (uint32_t i = 0; i < dn->n; i++) {
    mind2[i] = 1e30;
    mwnd[i] = -1e30;
  }
  const tr_didx *ix = &dn->ix;
  double rad = TR_FUS_TH;
  long c0[3], c1[3];
  for (int k = 0; k < 3; k++) {
    c0[k] = (long)floor((p[k] - rad) / ix->cs);
    c1[k] = (long)floor((p[k] + rad) / ix->cs);
  }
  for (long cz = c0[2]; cz <= c1[2]; cz++)
    for (long cy = c0[1]; cy <= c1[1]; cy++)
      for (long cx = c0[0]; cx <= c1[0]; cx++)
        for (uint32_t e = ix->head[tr_didx_h(ix, cx, cy, cz)]; e; e = ix->next[e - 1]) {
          uint64_t id = ix->qid[e - 1];
          uint32_t di = (uint32_t)(id >> 48);
          uint32_t j = (uint32_t)(id >> 24) & 0xFFFFFFu;
          uint32_t i = (uint32_t)(id & 0xFFFFFFu);
          const tr_donor *d = &dn->d[di];
          const float *p00 = r3d_tifxyz_at(&d->s, i, j);
          const float *p10 = r3d_tifxyz_at(&d->s, i + 1, j);
          const float *p01 = r3d_tifxyz_at(&d->s, i, j + 1);
          const float *p11 = r3d_tifxyz_at(&d->s, i + 1, j + 1);
          double A[3] = {(double)p00[0], (double)p00[1], (double)p00[2]};
          double B[3] = {(double)p10[0], (double)p10[1], (double)p10[2]};
          double C[3] = {(double)p01[0], (double)p01[1], (double)p01[2]};
          double D[3] = {(double)p11[0], (double)p11[1], (double)p11[2]};
          double qq[3], ub, vc2;
          for (int tri = 0; tri < 2; tri++) {
            double d2 = tri == 0 ? tr_tri_close(p, A, B, C, qq, &ub, &vc2)
                                 : tr_tri_close(p, D, C, B, qq, &ub, &vc2);
            if (d2 >= mind2[di]) continue;
            mind2[di] = d2;
            mwnd[di] = -1e30;
            if (d->dwind && d->woff_ok) {
              uint64_t k00 = (uint64_t)j * d->s.w + i;
              float w00 = d->dwind[k00], w10 = d->dwind[k00 + 1];
              float w01 = d->dwind[k00 + d->s.w], w11 = d->dwind[k00 + d->s.w + 1];
              if (w00 > -1e29f && w10 > -1e29f && w01 > -1e29f && w11 > -1e29f) {
                double wA = (double)(tri == 0 ? w00 : w11);
                double wB = (double)(tri == 0 ? w10 : w01);
                double wC = (double)(tri == 0 ? w01 : w10);
                mwnd[di] = wA + ub * (wB - wA) + vc2 * (wC - wA) + d->woff;
              }
            }
          }
        }
  int sup = 0;
  for (uint32_t i = 0; i < dn->n; i++) {
    if (mind2[i] > TR_FUS_TH * TR_FUS_TH) continue;
    if (have_w && mwnd[i] > -1e29 && fabs(mwnd[i] - cellw) > TR_FUS_WIND_TH) continue;
    sup++;
  }
  return sup;
}

/* single float32 TIFF plane, w*h expected (winding.tif channel) */
static float *tr_read_plane(const char *path, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, "r");
  if (!tf) return NULL;
  uint32_t tw = 0, th2 = 0;
  uint16_t bps = 0, fmt = 0;
  TIFFGetField(tf, TIFFTAG_IMAGEWIDTH, &tw);
  TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &th2);
  TIFFGetField(tf, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(tf, TIFFTAG_SAMPLEFORMAT, &fmt);
  float *v = NULL;
  if (tw == w && th2 == h && bps == 32 && fmt == SAMPLEFORMAT_IEEEFP &&
      !TIFFIsTiled(tf)) {
    v = malloc((size_t)w * h * sizeof *v);
    if (v)
      for (uint32_t j = 0; j < h; j++)
        if (TIFFReadScanline(tf, v + (size_t)j * w, j, 0) < 0) {
          free(v);
          v = NULL;
          break;
        }
  }
  TIFFClose(tf);
  return v;
}

static void tr_dons_free(tr_dons *dn) {
  if (!dn) return;
  for (uint32_t i = 0; i < dn->n; i++) {
    r3d_tifxyz_free(&dn->d[i].s);
    free(dn->d[i].dwind);
  }
  free(dn->d);
  free(dn->ix.head);
  free(dn->ix.next);
  free(dn->ix.qid);
  free(dn);
}

/* load donors + build the quad hash index */
static tr_dons *tr_dons_load(const char *const *dirs, uint32_t n, double step) {
  if (!n) return NULL;
  tr_dons *dn = calloc(1, sizeof *dn);
  if (!dn) return NULL;
  dn->d = calloc(n, sizeof *dn->d);
  if (!dn->d) {
    free(dn);
    return NULL;
  }
  uint64_t nquad = 0;
  for (uint32_t i = 0; i < n; i++) {
    tr_donor *d = &dn->d[dn->n];
    if (r3d_tifxyz_load(&d->s, dirs[i]) != 0) {
      printf("tracer: donor %s failed to load (skipping)\n", dirs[i]);
      continue;
    }
    char wp[1200];
    snprintf(wp, sizeof wp, "%s/winding.tif", dirs[i]);
    d->dwind = tr_read_plane(wp, d->s.w, d->s.h); /* NULL = none */
    dn->n++;
    nquad += (uint64_t)(d->s.w - 1) * (d->s.h - 1);
  }
  if (!dn->n) {
    tr_dons_free(dn);
    return NULL;
  }
  tr_didx *ix = &dn->ix;
  ix->cs = 4.0 * step; /* quads span ~step vox; a few per cell */
  uint32_t nb = 1;
  while (nb < nquad / 2 + 64) nb <<= 1;
  if (nb > 1u << 22) nb = 1u << 22;
  ix->nbuck = nb;
  ix->head = calloc(nb, sizeof *ix->head);
  if (!ix->head) {
    tr_dons_free(dn);
    return NULL;
  }
  for (uint32_t di = 0; di < dn->n; di++) {
    const r3d_tifxyz *s = &dn->d[di].s;
    for (uint32_t j = 0; j + 1 < s->h; j++)
      for (uint32_t i = 0; i + 1 < s->w; i++) {
        const float *p00 = r3d_tifxyz_at(s, i, j), *p10 = r3d_tifxyz_at(s, i + 1, j);
        const float *p01 = r3d_tifxyz_at(s, i, j + 1),
                    *p11 = r3d_tifxyz_at(s, i + 1, j + 1);
        if (!r3d_tifxyz_valid(p00) || !r3d_tifxyz_valid(p10) ||
            !r3d_tifxyz_valid(p01) || !r3d_tifxyz_valid(p11))
          continue;
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        const float *ps[4] = {p00, p10, p01, p11};
        for (int q = 0; q < 4; q++)
          for (int k = 0; k < 3; k++) {
            double vv = (double)ps[q][k];
            if (vv < lo[k]) lo[k] = vv;
            if (vv > hi[k]) hi[k] = vv;
          }
        uint64_t qid = (uint64_t)di << 48 | (uint64_t)j << 24 | i;
        long c0[3], c1[3];
        for (int k = 0; k < 3; k++) {
          c0[k] = (long)floor((lo[k] - TR_FUS_PULL) / ix->cs);
          c1[k] = (long)floor((hi[k] + TR_FUS_PULL) / ix->cs);
        }
        for (long cz = c0[2]; cz <= c1[2]; cz++)
          for (long cy = c0[1]; cy <= c1[1]; cy++)
            for (long cx = c0[0]; cx <= c1[0]; cx++)
              if (tr_didx_put(ix, cx, cy, cz, qid) != 0) {
                tr_dons_free(dn);
                return NULL;
              }
      }
  }
  printf("tracer: %u donor segment%s, %u indexed quad-cells\n", dn->n,
         dn->n == 1 ? "" : "s", ix->nent);
  return dn;
}

/* register donor windings into the global spiral frame: offset = robust
 * median of (spiral-model winding estimate - donor winding) over donor
 * vertices. Runs on the coordinator once the fit is valid. */
static void tr_don_register(r3d_tracer *t) {
  tr_dons *dn = t->don;
  if (!dn || !t->sp_valid) return;
  for (uint32_t di = 0; di < dn->n; di++) {
    tr_donor *d = &dn->d[di];
    if (!d->dwind || d->woff_ok) continue;
    double diffs[513];
    int nd = 0;
    uint64_t nv = (uint64_t)d->s.w * d->s.h;
    for (uint64_t k = 0; k < nv && nd < 512; k += 131) {
      const float *P = d->s.xyz + k * 3;
      if (!r3d_tifxyz_valid(P) || d->dwind[k] < -1e29f) continue;
      double cx, cy;
      double Pd[3] = {(double)P[0], (double)P[1], (double)P[2]};
      if (!tr_uc_at(t, Pd[2], &cx, &cy, NULL, NULL)) continue;
      double ux = Pd[0] - cx, uy = Pd[1] - cy;
      double rho = hypot(ux, uy), th = atan2(uy, ux);
      double r0v;
      tr_sp_r0_at(t, Pd[2], &r0v, NULL);
      double H = t->sp_ab[0] * cos(th) + t->sp_ab[1] * sin(th) +
                 t->sp_ab[2] * cos(2 * th) + t->sp_ab[3] * sin(2 * th);
      double west = (rho - r0v - H) / t->sp_omega;
      diffs[nd++] = west - (double)d->dwind[k];
    }
    if (nd < 32) continue;
    qsort(diffs, (size_t)nd, sizeof *diffs, tr_dcmp);
    d->woff = diffs[nd / 2];
    d->woff_ok = true;
    printf("tracer: donor %u registered at winding offset %+.2f\n", di, d->woff);
  }
}

static int tr_dcmp(const void *a, const void *b) {
  double d = *(const double *)a - *(const double *)b;
  return d < 0 ? -1 : d > 0 ? 1 : 0;
}

/* measure the inter-sheet gap: march umbilicus-radial rays (both ways)
 * from a subsample of grown points through the prediction DT; the first
 * return to the sheet after clearly leaving it is the adjacent wrap.
 * Median over samples = omega for the fixed-omega fit. */
static double tr_omega_measure(r3d_tracer *t, td_cache *dt) {
  if (!t->uc || !dt) return 0.0;
  double gaps[512];
  int ngap = 0;
  uint64_t N = (uint64_t)t->W * t->H;
  for (uint64_t k = 0; k < N && ngap < 512; k += 97) { /* coprime stride */
    if (t->state[k] != R3D_TR_SET) continue;
    const double *P = t->pos + k * 3;
    double cx, cy;
    if (!tr_uc_at(t, P[2], &cx, &cy, NULL, NULL)) continue;
    double ux = P[0] - cx, uy = P[1] - cy;
    double rho = hypot(ux, uy);
    if (rho < 8.0) continue;
    ux /= rho;
    uy /= rho;
    double best = 0.0;
    for (int dir = -1; dir <= 1; dir += 2) {
      double m = 0.0;
      for (int s = 1; s <= 64; s++) {
        double q[3] = {P[0] + dir * s * ux, P[1] + dir * s * uy, P[2]};
        double d = td_tri(dt, q, NULL);
        if (d > m) m = d;
        if (d < 1.5 && m > 3.0) { /* left the sheet, hit the next one */
          if (best == 0.0 || (double)s < best) best = (double)s;
          break;
        }
      }
    }
    if (best >= 4.0 && best <= 60.0) gaps[ngap++] = best;
  }
  if (ngap < 32) return 0.0;
  qsort(gaps, (size_t)ngap, sizeof *gaps, tr_dcmp);
  return gaps[ngap / 2];
}

/* ======================== residual evaluation ======================== */
enum {
  TRF_DIST = 1,
  TRF_STRAIGHT = 2,
  TRF_SDIR = 4,
  TRF_SPACE = 8,
  TRF_NCP = 16,
  TRF_WIND = 32,
  TRF_SURF = 64
};
#define TRF_ALL \
  (TRF_DIST | TRF_STRAIGHT | TRF_SDIR | TRF_SPACE | TRF_NCP | TRF_WIND | TRF_SURF)

typedef struct tr_ctx {
  r3d_tracer *t;
  tr_env *e;
  int i, j; /* free cell */
  uint32_t flags;
} tr_ctx;

static inline bool tr_valid(const r3d_tracer *t, int i, int j) {
  if (i < 0 || j < 0 || i >= (int)t->W || j >= (int)t->H) return false;
  return t->state[(size_t)j * t->W + (size_t)i] == R3D_TR_SET;
}

/* position of cell (i,j), substituting x for the free cell */
static inline const double *tr_at(const tr_ctx *c, int i, int j, const double x[3]) {
  if (i == c->i && j == c->j) return x;
  return c->t->pos + ((size_t)j * c->t->W + (size_t)i) * 3;
}

/* DistLoss: r = w*(D/L - 1) or w*(L/D - 1); free point is `a` */
static void tr_res_dist(tr_nlsq *acc, const double a[3], const double b[3], double D,
                        double w) {
  double d[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  double L2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  if (L2 < 1e-18) { /* vc3d dist_sq<=0 branch: r = w*(L2 - 1) */
    double J[3] = {2 * w * d[0], 2 * w * d[1], 2 * w * d[2]};
    nq_add(acc, w * (L2 - 1.0), J);
    return;
  }
  double L = sqrt(L2);
  double r, s; /* dr/dL */
  if (L2 < D * D) {
    r = w * (D / L - 1.0);
    s = -w * D / L2;
  } else {
    r = w * (L / D - 1.0);
    s = w / D;
  }
  double J[3] = {s * d[0] / L, s * d[1] / L, s * d[2] / L};
  nq_add(acc, r, J);
}

/* StraightLoss over triple (a,b,c); role = which point is free (0,1,2) */
static void tr_res_straight(tr_nlsq *acc, const double a[3], const double b[3],
                            const double c[3], int role, double w) {
  double d1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  double d2[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
  double l1s = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
  double l2s = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
  if (l1s <= 1e-24 || l2s <= 1e-24) return;
  double l1 = sqrt(l1s), l2 = sqrt(l2s);
  double dot = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (l1 * l2);
  double r, g; /* g = dr/ddot */
  if (dot <= TR_KINK_COS) {
    double pen = TR_KINK_COS - dot;
    r = w * (1.0 - dot) + 8.0 * w * pen * pen;
    g = -w - 16.0 * w * pen;
  } else {
    r = w * (1.0 - dot);
    g = -w;
  }
  double gd1[3], gd2[3]; /* ddot/dd1, ddot/dd2 */
  for (int k = 0; k < 3; k++) {
    gd1[k] = d2[k] / (l1 * l2) - dot * d1[k] / l1s;
    gd2[k] = d1[k] / (l1 * l2) - dot * d2[k] / l2s;
  }
  double J[3];
  for (int k = 0; k < 3; k++) {
    double dd; /* ddot/dfree_k */
    if (role == 0) dd = -gd1[k];
    else if (role == 1) dd = gd1[k] - gd2[k];
    else dd = gd2[k];
    J[k] = g * dd;
  }
  nq_add(acc, r, J);
}

/* SymmetricDirichletLoss over (p, pu, pv); role: 0=p 1=pu 2=pv.
 * Reference metric identity; Cauchy(1.0) robustifier applied as
 * sqrt(rho') scaling of residual + Jacobian. */
static void tr_res_sdir(tr_nlsq *acc, const double p[3], const double pu[3],
                        const double pv[3], int role, double unit, double w) {
  double eu[3], ev[3];
  double lu = 0, lv = 0;
  for (int k = 0; k < 3; k++) {
    eu[k] = (pu[k] - p[k]) / unit;
    ev[k] = (pv[k] - p[k]) / unit;
    lu += eu[k] * eu[k];
    lv += ev[k] * ev[k];
  }
  if (lu * unit * unit < 1e-12 || lv * unit * unit < 1e-12) return;
  double a = lu, c2 = lv;
  double b = eu[0] * ev[0] + eu[1] * ev[1] + eu[2] * ev[2];
  double trg = a + c2, detg = a * c2 - b * b;
  double ds = detg + TR_SDIR_EPS_ABS + TR_SDIR_EPS_REL * fabs(trg);
  if (fabs(ds) < 1e-30) return;
  double E = trg + trg / ds;
  if (fabs(E) > 1e30) return; /* range guard (isfinite is UB-checked away
                               * under fast-math) */
  double r = w * (E - 4.0);
  /* dE/da etc. */
  double sgn = trg >= 0 ? 1.0 : -1.0;
  double dE_dtr = 1.0 + 1.0 / ds;
  double dE_dds = -trg / (ds * ds);
  double dE[3]; /* wrt a, c2, b */
  dE[0] = dE_dtr * 1.0 + dE_dds * (c2 + TR_SDIR_EPS_REL * sgn * 1.0);
  dE[1] = dE_dtr * 1.0 + dE_dds * (a + TR_SDIR_EPS_REL * sgn * 1.0);
  dE[2] = dE_dds * (-2.0 * b);
  double J[3];
  for (int k = 0; k < 3; k++) {
    double da, dc, db; /* d{a,c2,b}/dfree_k */
    if (role == 1) { /* pu */
      da = 2.0 * eu[k] / unit;
      dc = 0.0;
      db = ev[k] / unit;
    } else if (role == 2) { /* pv */
      da = 0.0;
      dc = 2.0 * ev[k] / unit;
      db = eu[k] / unit;
    } else { /* p */
      da = -2.0 * eu[k] / unit;
      dc = -2.0 * ev[k] / unit;
      db = -(eu[k] + ev[k]) / unit;
    }
    J[k] = w * (dE[0] * da + dE[1] * dc + dE[2] * db);
  }
  double s = sqrt(1.0 / (1.0 + r * r)); /* Cauchy(1) */
  double Js[3] = {J[0] * s, J[1] * s, J[2] * s};
  nq_add(acc, r * s, Js);
}

/* space-line data term along edge (x -> nb): mean DT over interior
 * samples; free point is x */
static void tr_res_space(tr_nlsq *acc, td_cache *dt, const double x[3],
                         const double nb[3], double w) {
  double rsum = 0.0, J[3] = {0, 0, 0};
  const int n = TR_SPACE_STEPS;
  for (int s = 1; s < n; s++) {
    double f = (double)s / n;
    double q[3] = {x[0] * (1 - f) + nb[0] * f, x[1] * (1 - f) + nb[1] * f,
                   x[2] * (1 - f) + nb[2] * f};
    double g[3];
    rsum += td_tri(dt, q, g);
    for (int k = 0; k < 3; k++) J[k] += (1 - f) * g[k];
  }
  double sc = w / (double)(n - 1);
  double Js[3] = {J[0] * sc, J[1] * sc, J[2] * sc};
  nq_add(acc, rsum * sc, Js);
}

/* spiral winding prior: r = w * (rho - (r0(z) + omega*wind)) / |omega|,
 * Cauchy(1)-robustified; the free point is x. Normalizing by omega makes
 * one whole wrap of error cost ~w — commensurate with the other O(1)
 * residuals. */
static void tr_res_wind(tr_nlsq *acc, const r3d_tracer *t, const double x[3],
                        double wnd, double wgt) {
  double cx, cy, dxz, dyz;
  if (!tr_uc_at(t, x[2], &cx, &cy, &dxz, &dyz)) return;
  double ux = x[0] - cx, uy = x[1] - cy;
  double rho = sqrt(ux * ux + uy * uy);
  if (rho < 1e-6) return;
  double r0v, r0s;
  tr_sp_r0_at(t, x[2], &r0v, &r0s);
  double om = fabs(t->sp_omega);
  if (om < 1e-9) return;
  double th = atan2(uy, ux);
  const double *ab = t->sp_ab;
  double H = ab[0] * cos(th) + ab[1] * sin(th) + ab[2] * cos(2 * th) +
             ab[3] * sin(2 * th);
  double Hd = -ab[0] * sin(th) + ab[1] * cos(th) - 2 * ab[2] * sin(2 * th) +
              2 * ab[3] * cos(2 * th);
  double r = wgt * (rho - (r0v + H + t->sp_omega * wnd)) / om;
  double r2 = rho * rho;
  double J[3] = {wgt * (ux / rho + Hd * uy / r2) / om,
                 wgt * (uy / rho - Hd * ux / r2) / om,
                 wgt * (-(ux * dxz + uy * dyz) / rho - r0s -
                        Hd * (uy * dxz - ux * dyz) / r2) /
                     om};
  double s = sqrt(1.0 / (1.0 + r * r)); /* Cauchy(1) */
  double Js[3] = {J[0] * s, J[1] * s, J[2] * s};
  nq_add(acc, r * s, Js);
}

/* all residuals touching free cell (i,j) evaluated at trial position x */
static void tr_eval(tr_ctx *c, const double x[3], tr_nlsq *acc) {
  r3d_tracer *t = c->t;
  tr_env *e = c->e;
  const double unit = t->cfg.step;
  int i = c->i, j = c->j;
  static const int n8[8][2] = {{1, 0},  {-1, 0}, {0, 1},  {0, -1},
                               {1, 1},  {1, -1}, {-1, 1}, {-1, -1}};
  if (c->flags & TRF_DIST)
    for (int o = 0; o < 8; o++) {
      int ii = i + n8[o][0], jj = j + n8[o][1];
      if (!tr_valid(t, ii, jj)) continue;
      double D = unit * sqrt((double)(n8[o][0] * n8[o][0] + n8[o][1] * n8[o][1]));
      tr_res_dist(acc, x, tr_at(c, ii, jj, x), D, TR_W_DIST);
    }
  if (c->flags & TRF_STRAIGHT) {
    static const int ax[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    for (int a = 0; a < 4; a++)
      for (int w0 = -2; w0 <= 0; w0++) { /* triple starts at p + w0*axis */
        int ai = i + w0 * ax[a][0], aj = j + w0 * ax[a][1];
        int bi = ai + ax[a][0], bj = aj + ax[a][1];
        int ci = bi + ax[a][0], cj = bj + ax[a][1];
        if (!tr_valid(t, ai, aj) || !tr_valid(t, bi, bj) || !tr_valid(t, ci, cj))
          continue;
        tr_res_straight(acc, tr_at(c, ai, aj, x), tr_at(c, bi, bj, x),
                        tr_at(c, ci, cj, x), -w0, TR_W_STRAIGHT);
      }
  }
  if (c->flags & TRF_SDIR) {
    /* cells whose (p,pu,pv) triangle involves (i,j): base at p, p-(0,1),
     * p-(1,0) — roles p/pu/pv respectively (pu = col+1, pv = row+1) */
    static const int cell[3][2] = {{0, 0}, {-1, 0}, {0, -1}};
    for (int q = 0; q < 3; q++) {
      int pi = i + cell[q][0], pj = j + cell[q][1];
      if (!tr_valid(t, pi, pj) || !tr_valid(t, pi + 1, pj) || !tr_valid(t, pi, pj + 1))
        continue;
      tr_res_sdir(acc, tr_at(c, pi, pj, x), tr_at(c, pi + 1, pj, x),
                  tr_at(c, pi, pj + 1, x), q, unit, TR_W_SDIR);
    }
  }
  bool ng_on = e->ngv && e->ngv->active;
  if ((c->flags & TRF_SPACE) && !ng_on) { /* DT fallback only without grids
                                           * (vc3d ships space-line off) */
    static const int n4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int o = 0; o < 4; o++) {
      int ii = i + n4[o][0], jj = j + n4[o][1];
      if (!tr_valid(t, ii, jj)) continue;
      tr_res_space(acc, e->dt, x, tr_at(c, ii, jj, x), TR_W_SPACE);
    }
  }
  if ((c->flags & TRF_NCP) && ng_on) {
    double tt0 = tr_now();
    /* every quad containing the free cell x 3 planes x 4 corner
     * rotations (vc3d gen_normal_loss); gradients by central difference
     * — vc3d autodiffs through the same frozen path choice */
    static const int qoff[4][2] = {{0, 0}, {-1, 0}, {0, -1}, {-1, -1}};
    static const int rot[4][4] = {{0, 1, 2, 3}, {3, 2, 1, 0}, {1, 3, 0, 2}, {2, 0, 3, 1}};
    for (int q = 0; q < 4; q++) {
      int qi = i + qoff[q][0], qj = j + qoff[q][1];
      if (!tr_valid(t, qi, qj) || !tr_valid(t, qi + 1, qj) ||
          !tr_valid(t, qi, qj + 1) || !tr_valid(t, qi + 1, qj + 1))
        continue;
      const double *ptrs[4] = {tr_at(c, qi, qj, x), tr_at(c, qi + 1, qj, x),
                               tr_at(c, qi, qj + 1, x), tr_at(c, qi + 1, qj + 1, x)};
      double Q[4][3];
      int fx = -1;
      for (int k2 = 0; k2 < 4; k2++) {
        memcpy(Q[k2], ptrs[k2], sizeof Q[k2]);
        if (ptrs[k2] == x) fx = k2;
      }
      if (fx < 0) continue;
      for (int rr = 0; rr < 4; rr++)
        for (int pl = 0; pl < 3; pl++) {
          double Qrot[4][3];
          int fxrot = -1;
          for (int c2 = 0; c2 < 4; c2++) {
            memcpy(Qrot[c2], Q[rot[rr][c2]], sizeof Qrot[c2]);
            if (rot[rr][c2] == fx) fxrot = c2;
          }
          double r4[4];
          ncp_residual4(e, pl, Qrot, fxrot, 0.5, r4);
          if (r4[0] == 0.0) continue; /* non-straddling plane */
          double J[3];
          for (int a = 0; a < 3; a++) J[a] = (r4[1 + a] - r4[0]) / 0.5;
          nq_add(acc, r4[0], J);
        }
    }
    tr_tm_add(4, tt0);
  }
  if ((c->flags & TRF_WIND) && t->sp_valid && t->cfg.wind_weight > 0)
    tr_res_wind(acc, t, x, (double)t->wind[(size_t)j * t->W + (size_t)i],
                t->cfg.wind_weight);
  if ((c->flags & TRF_SURF) && t->don) {
    /* donor anchor (vc3d SurfaceLossD, w=0.1): pull toward the nearest
     * donor surface point, frozen for this evaluation; donors on a
     * different registered winding never attract */
    double q[3], dwnd;
    size_t kc = (size_t)j * t->W + (size_t)i;
    int di = tr_don_closest(t->don, x, TR_FUS_PULL, q, &dwnd);
    if (di >= 0 &&
        !(t->uc && dwnd > -1e29 &&
          fabs(dwnd - (double)t->wind[kc]) > TR_FUS_WIND_TH)) {
      for (int a = 0; a < 3; a++) {
        double J[3] = {0, 0, 0};
        J[a] = TR_W_SURF;
        nq_add(acc, TR_W_SURF * (x[a] - q[a]), J);
      }
    }
  }
}

/* LM solve for one cell; updates pos in place (under mu), returns cost */
static double tr_solve_cell(r3d_tracer *t, tr_env *e, int i, int j, uint32_t flags,
                            int max_iter) {
  tr_ctx c = {.t = t, .e = e, .i = i, .j = j, .flags = flags};
  double x[3];
  memcpy(x, t->pos + ((size_t)j * t->W + (size_t)i) * 3, sizeof x);
  tr_nlsq a;
  nq_begin(&a);
  tr_eval(&c, x, &a);
  double cost = a.cost, lm = 1e-3;
  for (int it = 0; it < max_iter; it++) {
    double d[3];
    if (nq_step(a.JTJ, a.JTr, lm, d) != 0) break;
    double nx[3] = {x[0] + d[0], x[1] + d[1], x[2] + d[2]};
    tr_nlsq b;
    nq_begin(&b);
    tr_eval(&c, nx, &b);
    if (b.cost < cost) {
      bool conv = cost - b.cost < 1e-3 * cost ||
                  d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 1e-8;
      memcpy(x, nx, sizeof x);
      cost = b.cost;
      a = b;
      lm = lm > 1e-9 ? lm * 0.5 : lm;
      if (conv) break;
    } else {
      lm *= 4.0;
      if (lm > 1e8) break;
    }
  }
  pthread_mutex_lock(&t->mu);
  memcpy(t->pos + ((size_t)j * t->W + (size_t)i) * 3, x, sizeof x);
  pthread_mutex_unlock(&t->mu);
  if (e->ngv) tr_env_flush(e);
  return cost;
}

static void tr_update_conf(r3d_tracer *t, tr_env *e, int i, int j) {
  size_t k = (size_t)j * t->W + (size_t)i;
  if (t->state[k] != R3D_TR_SET) return;
  double tt0 = tr_now();
  const double *P = t->pos + k * 3;
  double d = TR_CONF_R;
  if (e->ngv && e->ngv->active) {
    /* distance to the nearest xy-plane polyline — no prediction volume,
     * no EDT: the grids ARE the data now */
    int slot;
    ng_grid *g = ng_eget(e, 0, (int)llround(P[2]), &slot);
    if (g && !g->empty) {
      ng_hood *hd = ng_hood_get(e, g, 0, (int)llround(P[2]), P[0], P[1]);
      if (hd && hd->nseg) {
        double d2min = 1e30;
        for (uint32_t q = 0; q < hd->nseg; q++) {
          double d2 = ncp_pt_seg_d2(P[0], P[1], (double)hd->sg[q][0],
                                    (double)hd->sg[q][1], (double)hd->sg[q][2],
                                    (double)hd->sg[q][3]);
          if (d2 < d2min) d2min = d2;
        }
        d = sqrt(d2min);
      }
    }
    tr_env_flush(e);
    if (d >= TR_CONF_R && e->dt) /* no polyline within reach: the grids
        * prune short traces — ask the raw predictions before declaring
        * the point unsupported */
      d = td_tri(e->dt, P, NULL);
  } else if (e->dt) {
    d = td_tri(e->dt, P, NULL);
  }
  tr_tm_add(2, tt0);
  double cf = 1.0 - (d > TR_CONF_R ? TR_CONF_R : d) / TR_CONF_R;
  if (t->dsup && t->dsup[k]) { /* donor-vouched cells keep a conf floor:
                                * 2+ donors agreeing beats a weak DT */
    double fl = t->dsup[k] >= 2 ? 0.95 : 0.75;
    if (cf < fl) cf = fl;
  }
  t->conf[k] = (float)cf;
}

#define TR_NTHREADS 10
struct tr_pool;
static uint32_t tr_pool_run2(struct tr_pool *pl, tr_env *cenv, const uint32_t *items,
                             uint32_t n, uint32_t *out, int mode);

/* vc3d local_optimization(radius, p): free = SET cells within Euclidean
 * `radius` of center, boundary ring fixed; here solved as alternating
 * Gauss-Seidel sweeps of per-cell LM instead of one joint sparse solve. */
static void tr_local_opt(r3d_tracer *t, tr_env *e, int ci, int cj, int radius,
                         int sweeps, bool do_conf) {
  double tt0 = tr_now();
  int r0i = ci - radius, r1i = ci + radius, r0j = cj - radius, r1j = cj + radius;
  if (r0i < 0) r0i = 0;
  if (r0j < 0) r0j = 0;
  if (r1i >= (int)t->W) r1i = (int)t->W - 1;
  if (r1j >= (int)t->H) r1j = (int)t->H - 1;
  if (e->ngv && e->ngv->nfth) /* planned prefetch: cells sit ~unit apart, so the
                         * slice sequence jumps ~20 per cell — neighbor-
                         * radius prefetch never catches it; enqueue every
                         * slice this disc will actually touch */
    for (int jj = r0j; jj <= r1j; jj++)
      for (int ii = r0i; ii <= r1i; ii++) {
        if (t->state[(size_t)jj * t->W + (size_t)ii] != R3D_TR_SET) continue;
        const double *P = t->pos + ((size_t)jj * t->W + (size_t)ii) * 3;
        ng_prefetch(e->ngv, 0, (int)llround(P[2]));
        ng_prefetch(e->ngv, 1, (int)llround(P[1]));
        ng_prefetch(e->ngv, 2, (int)llround(P[0]));
      }
  if (radius >= 16 && e->ngv && e->ngv->active && !e->in_pool && e->pl) {
    /* big disc (global solves, final polish): run each sweep through the
     * candidate pool — >= 7-cell separation keeps concurrent radius-2
     * residual stencils disjoint */
    uint32_t cap = (uint32_t)(r1i - r0i + 1) * (uint32_t)(r1j - r0j + 1);
    uint32_t *cells = malloc((size_t)cap * sizeof *cells);
    if (cells) {
      uint32_t nc2 = 0;
      for (int jj = r0j; jj <= r1j; jj++)
        for (int ii = r0i; ii <= r1i; ii++) {
          long di = ii - ci, dj = jj - cj;
          if (di * di + dj * dj > (long)radius * radius) continue;
          if (t->state[(size_t)jj * t->W + (size_t)ii] != R3D_TR_SET) continue;
          cells[nc2++] = (uint32_t)((size_t)jj * t->W + (size_t)ii);
        }
      for (int s = 0; s < sweeps && !t->quit; s++)
        tr_pool_run2(e->pl, e, cells, nc2, NULL, 2);
      if (do_conf)
        for (uint32_t c2 = 0; c2 < nc2 && !t->quit; c2++)
          tr_update_conf(t, e, (int)(cells[c2] % t->W), (int)(cells[c2] / t->W));
      free(cells);
      tr_tm_add(1, tt0);
      return;
    }
  }
  for (int s = 0; s < sweeps && !t->quit; s++) {
    bool rev = (s & 1) != 0;
    for (int jj = r0j; jj <= r1j; jj++)
      for (int ii = r0i; ii <= r1i; ii++) {
        int i = rev ? r1i - (ii - r0i) : ii;
        int j = rev ? r1j - (jj - r0j) : jj;
        long di = i - ci, dj = j - cj;
        if (di * di + dj * dj > (long)radius * radius) continue;
        if (t->state[(size_t)j * t->W + (size_t)i] != R3D_TR_SET) continue;
        tr_solve_cell(t, e, i, j, TRF_ALL, 4);
        if (do_conf && s + 1 == sweeps) tr_update_conf(t, e, i, j);
      }
  }
  tr_tm_add(1, tt0);
}

static double tr_urand(unsigned *st) { /* U(-0.05, 0.05), vc3d perturbation */
  *st = *st * 1664525u + 1013904223u;
  return ((double)(*st >> 8) / (double)(1u << 24) - 0.5) * 0.1;
}

/* =========================== growth loop =========================== */
/* one candidate: pick the best parent, commit on top of it, solve into
 * place (vc3d per-candidate sequence). Returns true when placed. */
static bool tr_place_cand(r3d_tracer *t, tr_env *e, uint32_t cell, unsigned *rng) {
  uint32_t W = t->W;
  int i = (int)(cell % W), j = (int)(cell / W);
  /* best parent: the 3x3-neighbor with the most valid 3x3-neighbors */
  int bi = -1, bj = -1, bcnt = -1;
  for (int dj = -1; dj <= 1; dj++)
    for (int di = -1; di <= 1; di++) {
      int ii = i + di, jj = j + dj;
      if (!tr_valid(t, ii, jj)) continue;
      int cnt = 0;
      for (int qj = -1; qj <= 1; qj++)
        for (int qi = -1; qi <= 1; qi++)
          if (tr_valid(t, ii + qi, jj + qj)) cnt++;
      if (cnt > bcnt) {
        bcnt = cnt;
        bi = ii;
        bj = jj;
      }
    }
  if (bi < 0) return false;
  size_t k = (size_t)j * W + (size_t)i;
  if (t->state[k] == R3D_TR_SET) return false;
  const double *bp = t->pos + ((size_t)bj * W + (size_t)bi) * 3;
  pthread_mutex_lock(&t->mu);
  for (int a = 0; a < 3; a++) t->pos[k * 3 + (size_t)a] = bp[a] + tr_urand(rng);
  t->state[k] = R3D_TR_SET; /* committed before the solve (vc3d) */
  t->nset++;
  pthread_mutex_unlock(&t->mu);
  /* placement: geometric + data terms, then radius-1 and radius-3 */
  double tt0 = tr_now();
  tr_solve_cell(t, e, i, j, TRF_DIST | TRF_STRAIGHT | TRF_SPACE, 50);
  tr_tm_add(0, tt0);
  tr_local_opt(t, e, i, j, 1, 2, false);
  tr_local_opt(t, e, i, j, 3, 3, false);
  const double *fp = t->pos + k * 3; /* the ONLY legitimate hole: the
                                      * point left the scroll volume */
  if (t->uc) { /* winding: neighbor wind + unwrapped angle step, averaged
                * over the placed 3x3 ring — combinatorial, so later
                * solves can move the point but never re-wind it */
    double th;
    if (tr_theta_of(t, fp, &th)) {
      double sum = 0.0;
      int nn = 0;
      for (int dj = -1; dj <= 1; dj++)
        for (int di = -1; di <= 1; di++) {
          if (!di && !dj) continue;
          int ii = i + di, jj = j + dj;
          if (!tr_valid(t, ii, jj)) continue;
          size_t nk = (size_t)jj * t->W + (size_t)ii;
          double thn;
          if (!tr_theta_of(t, t->pos + nk * 3, &thn)) continue;
          double d = th - thn;
          while (d > M_PI) d -= 2.0 * M_PI;
          while (d < -M_PI) d += 2.0 * M_PI;
          sum += (double)t->wind[nk] + d / (2.0 * M_PI);
          nn++;
        }
      if (nn) t->wind[k] = (float)(sum / nn);
    }
  }
  if (t->don && t->dsup) { /* consensus count (vc3d inlier vote, gated by
                            * the winding frame) */
    int sup = tr_don_support(t->don, fp, (double)t->wind[k], t->uc != NULL);
    t->dsup[k] = (uint8_t)(sup > 255 ? 255 : sup);
  }
  if (t->vdim[0] > 0 &&
      (fp[0] < 0 || fp[1] < 0 || fp[2] < 0 || fp[0] >= t->vdim[0] ||
       fp[1] >= t->vdim[1] || fp[2] >= t->vdim[2])) {
    pthread_mutex_lock(&t->mu);
    t->state[k] = R3D_TR_FAIL;
    if (t->nset) t->nset--;
    pthread_mutex_unlock(&t->mu);
    return false;
  }
  return true;
}

/* persistent candidate pool: vc3d runs candidates under OMP with >= 7
 * grid cells between concurrently-processed points, so simultaneous
 * radius-3 discs never overlap. Threads, solve envs, and neighborhood
 * caches live for the whole trace — jobs are posted per phase. */
typedef struct tr_pool {
  r3d_tracer *t;
  ng_vol *ngv;
  pthread_t th[TR_NTHREADS];
  tr_env env[TR_NTHREADS];
  uint32_t nth;
  /* current job, guarded by mu */
  const uint32_t *items;
  uint32_t n;
  uint8_t *st; /* 0 pending 1 active 2 done */
  uint32_t scan0;
  int mode; /* 0 placement (sep 7), 1 radius-8 opt (sep 17), 2 sweep visit */
  int act[TR_NTHREADS][2];
  uint32_t nact;
  uint32_t *out;
  uint32_t nout;
  uint64_t job;
  uint32_t done;
  uint32_t nwait; /* claim-loop waiters (gate cv wakeups) */
  int klass;              /* mode 3: (i%7) + 7*(j%7) residue class */
  _Atomic uint32_t aidx;  /* mode 3: lock-free work index */
  bool quit;
  pthread_mutex_t mu;
  pthread_cond_t cv, idle_cv;
} tr_pool;

static void *tr_pool_thread(void *ud) {
  tr_env *e = ud;
  tr_pool *pl = e->pl;
  r3d_tracer *t = pl->t;
  uint64_t seen = 0;
  pthread_mutex_lock(&pl->mu);
  for (;;) {
    while (!pl->quit && pl->job == seen) pthread_cond_wait(&pl->cv, &pl->mu);
    if (pl->quit) break;
    seen = pl->job;
    if (pl->mode == 3) { /* sweep visit, residue-class scheduled: cells of
                          * one (i%7, j%7) class are pairwise >= 7 apart —
                          * no claims, no conflict scans, no cv traffic */
      int kls = pl->klass;
      pthread_mutex_unlock(&pl->mu);
      for (;;) {
        uint32_t c = atomic_fetch_add(&pl->aidx, 1);
        if (c >= pl->n || t->quit) break;
        uint32_t cell = pl->items[c];
        int i = (int)(cell % t->W), j = (int)(cell / t->W);
        if (i % 7 + 7 * (j % 7) != kls) continue;
        tr_solve_cell(t, e, i, j, TRF_ALL, 4);
      }
      pthread_mutex_lock(&pl->mu);
      pl->done++;
      pthread_cond_signal(&pl->idle_cv);
      continue;
    }
    for (;;) { /* claim loop; mu held at the top of each iteration */
      if (t->quit) break;
      while (pl->scan0 < pl->n && pl->st[pl->scan0] != 0) pl->scan0++;
      long sep2 = pl->mode == 1 ? 289 : 49;
      int pick = -1;
      bool pending = false;
      for (uint32_t c = pl->scan0; c < pl->n; c++) {
        if (pl->st[c] != 0) continue;
        pending = true;
        int i = (int)(pl->items[c] % t->W), j = (int)(pl->items[c] / t->W);
        bool clear = true;
        for (uint32_t a = 0; a < pl->nact && clear; a++) {
          long di = i - pl->act[a][0], dj = j - pl->act[a][1];
          clear = di * di + dj * dj >= sep2;
        }
        if (clear) {
          pick = (int)c;
          break;
        }
      }
      if (pick < 0) {
        if (!pending || pl->nact == 0) break;
        pl->nwait++;
        pthread_cond_wait(&pl->cv, &pl->mu);
        pl->nwait--;
        continue;
      }
      uint32_t cell = pl->items[pick];
      pl->st[pick] = 1;
      pl->act[pl->nact][0] = (int)(cell % t->W);
      pl->act[pl->nact][1] = (int)(cell / t->W);
      uint32_t slot = pl->nact++;
      int mode = pl->mode;
      pthread_mutex_unlock(&pl->mu);
      bool placed = false;
      if (mode == 0) {
        placed = tr_place_cand(t, e, cell, &e->rng);
      } else if (mode == 1) {
        tr_local_opt(t, e, (int)(cell % t->W), (int)(cell / t->W), 8, 3, false);
      } else { /* one sweep visit of a big-radius solve */
        tr_solve_cell(t, e, (int)(cell % t->W), (int)(cell / t->W), TRF_ALL, 4);
      }
      pthread_mutex_lock(&pl->mu);
      pl->st[pick] = 2;
      pl->act[slot][0] = pl->act[--pl->nact][0];
      pl->act[slot][1] = pl->act[pl->nact][1];
      if (placed) pl->out[pl->nout++] = cell;
      if (pl->nwait) pthread_cond_broadcast(&pl->cv);
    }
    pl->done++;
    pthread_cond_signal(&pl->idle_cv);
  }
  pthread_mutex_unlock(&pl->mu);
  tr_env_flush(e);
  return NULL;
}

static void tr_pool_init(tr_pool *pl, r3d_tracer *t, ng_vol *ngv) {
  memset(pl, 0, sizeof *pl);
  pl->t = t;
  pl->ngv = ngv;
  pl->st = malloc((size_t)t->W * t->H);
  if (!pl->st) return;
  pthread_mutex_init(&pl->mu, NULL);
  pthread_cond_init(&pl->cv, NULL);
  pthread_cond_init(&pl->idle_cv, NULL);
  uint32_t want = TR_NTHREADS;
  const char *tenv = getenv("R3D_TRACE_THREADS");
  if (tenv) { /* 0/1 = serial (deterministic quality A/B runs) */
    long tv = strtol(tenv, NULL, 10);
    want = tv < 2 ? 0 : (tv > TR_NTHREADS ? TR_NTHREADS : (uint32_t)tv);
  }
  for (uint32_t i = 0; i < want; i++) {
    tr_env *e = &pl->env[pl->nth];
    memset(e, 0, sizeof *e);
    e->ngv = ngv;
    e->in_pool = true;
    e->pl = pl;
    e->rng = t->rng ^ (0x9E3779B9u * (i + 1));
    e->hood = calloc(NG_NHOOD, sizeof *e->hood);
    if (!e->hood) break;
    for (int hi = 0; hi < NG_NHOOD; hi++) e->hood[hi].plane = -1;
    if (pthread_create(&pl->th[pl->nth], NULL, tr_pool_thread, e) != 0) {
      free(e->hood);
      break;
    }
    pl->nth++;
  }
}

static void tr_pool_destroy(tr_pool *pl) {
  if (pl->st) {
    pthread_mutex_lock(&pl->mu);
    pl->quit = true;
    pthread_cond_broadcast(&pl->cv);
    pthread_mutex_unlock(&pl->mu);
    for (uint32_t i = 0; i < pl->nth; i++) {
      pthread_join(pl->th[i], NULL);
      free(pl->env[i].hood);
    }
    pthread_mutex_destroy(&pl->mu);
    pthread_cond_destroy(&pl->cv);
    pthread_cond_destroy(&pl->idle_cv);
    free(pl->st);
  }
  pl->st = NULL;
  pl->nth = 0;
}

/* run items through the pool; cenv drains anything left (safety: items
 * are PROC-marked before the pool sees them). Returns placed count. */
static uint32_t tr_pool_run2(tr_pool *pl, tr_env *cenv, const uint32_t *items,
                             uint32_t n, uint32_t *out, int mode) {
  r3d_tracer *t = pl->t;
  uint32_t nout = 0;
  if (false && n && pl->nth && mode == 2) { /* residue-class scheduling:
      * lock-free but stride-7 iteration wrecks slice/hood locality —
      * kept for reference, claim order wins in practice */
    for (int kls = 0; kls < 49 && !t->quit; kls++) {
      pthread_mutex_lock(&pl->mu);
      pl->items = items;
      pl->n = n;
      pl->mode = 3;
      pl->klass = kls;
      atomic_store(&pl->aidx, 0);
      pl->done = 0;
      pl->job++;
      pthread_cond_broadcast(&pl->cv);
      while (pl->done < pl->nth) pthread_cond_wait(&pl->idle_cv, &pl->mu);
      pl->n = 0;
      pthread_mutex_unlock(&pl->mu);
    }
    return 0;
  }
  if (n && pl->nth) {
    memset(pl->st, 0, n);
    pthread_mutex_lock(&pl->mu);
    pl->items = items;
    pl->n = n;
    pl->scan0 = 0;
    pl->mode = mode;
    pl->nact = 0;
    pl->out = out;
    pl->nout = 0;
    pl->done = 0;
    pl->job++;
    pthread_cond_broadcast(&pl->cv);
    while (pl->done < pl->nth) pthread_cond_wait(&pl->idle_cv, &pl->mu);
    nout = pl->nout;
    pl->n = 0;
    pthread_mutex_unlock(&pl->mu);
  }
  for (uint32_t c = 0; c < n && !t->quit; c++) { /* serial drain */
    if (pl->nth && pl->st[c] != 0) continue;
    if (mode == 0) {
      if (tr_place_cand(t, cenv, items[c], &t->rng)) out[nout++] = items[c];
    } else if (mode == 1) {
      tr_local_opt(t, cenv, (int)(items[c] % t->W), (int)(items[c] / t->W), 8, 3,
                   false);
    } else {
      tr_solve_cell(t, cenv, (int)(items[c] % t->W), (int)(items[c] / t->W), TRF_ALL,
                    4);
    }
  }
  return nout;
}

static void *tr_worker(void *ud) {
  r3d_tracer *t = ud;
  r3d_cpuvol vol;
  if (r3d_cpuvol_open(&vol, t->root, 96) != 0) goto fail_open;
  td_cache *dt = td_open(&vol, t->cfg.level);
  if (!dt) {
    r3d_cpuvol_close(&vol);
    goto fail_open;
  }
  for (int k = 0; k < 6; k++) atomic_store(&tr_tm_ns[k], 0);
  double tr_t_start = tr_now();
  t->vdim[0] = (double)vol.nx;
  t->vdim[1] = (double)vol.ny;
  t->vdim[2] = (double)vol.nz;
  tr_uc_build(t, (uint32_t)vol.nz); /* spiral frame origin per slice */
  if (t->cfg.wind_weight > 0 && !t->uc)
    printf("tracer: spiral prior requested but no usable umbilicus (needs "
           ">= 2 control points) — growing without it\n");
  ng_vol ng; /* vc3d's real data term when the store exists upstream */
  ng_open(&ng, t->root);
  tr_env cenv = {.dt = dt, .ngv = &ng}; /* coordinator's solve env */
  cenv.hood = calloc(NG_NHOOD, sizeof *cenv.hood);
  if (cenv.hood)
    for (int hi = 0; hi < NG_NHOOD; hi++) cenv.hood[hi].plane = -1;
  tr_pool pool = {0};
  if (ng.active) tr_pool_init(&pool, t, &ng);
  cenv.pl = pool.nth ? &pool : NULL;
  if (ng.active && fabs(ng.spiral_step - t->cfg.step) > 1e-6)
    printf("tracer: grid step %.1f != normal-grid spiral-step %.1f (vc3d "
           "errors here; proceeding)\n",
           t->cfg.step, ng.spiral_step);
  uint32_t W = t->W, H = t->H;
  int x0 = (int)W / 2, y0 = (int)H / 2;
  uint32_t *fringe = malloc((size_t)W * H * sizeof *fringe);
  uint32_t *nfringe = malloc((size_t)W * H * sizeof *nfringe);
  uint32_t *cands = malloc((size_t)W * H * sizeof *cands);
  if (!fringe || !nfringe || !cands) {
    free(fringe);
    free(nfringe);
    free(cands);
    tr_pool_destroy(&pool);
    free(cenv.hood);
    ng_close(&ng);
    td_close(dt);
    r3d_cpuvol_close(&vol);
    goto fail_open;
  }
  uint32_t nf = 0;
  bool resume = t->nset > 0;

  if (!resume) {
    /* vc3d seed: one 2x2 quad at the origin, 0.1-voxel extent; the first
     * solve inflates it to `unit` and pulls it onto the sheet */
    /* probe down the pyramid so a sparsely cached prediction tree still
     * seeds (locally cached data may start at a coarser level) */
    for (uint32_t lv = t->cfg.level; lv + 1 < vol.nlev; lv++) {
      double v = td_tri(dt, t->cfg.seed, NULL);
      if (v < 64.0) break;
      dt->level = lv + 1;
      dt->memo_key = 0;
      for (int s2 = 0; s2 < TD_SLOTS; s2++) dt->s[s2].key = 0;
    }
    static const int off4[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    pthread_mutex_lock(&t->mu);
    for (int q = 0; q < 4; q++) {
      int i = x0 + off4[q][0], j = y0 + off4[q][1];
      size_t k = (size_t)j * W + (size_t)i;
      double *P = t->pos + k * 3;
      P[0] = t->cfg.seed[0] + 0.1 * off4[q][0];
      P[1] = t->cfg.seed[1] + 0.1 * off4[q][1];
      P[2] = t->cfg.seed[2];
      t->state[k] = R3D_TR_SET;
      t->conf[k] = 1.0f;
      fringe[nf++] = (uint32_t)k;
    }
    t->nset = 4;
    t->gen++;
    pthread_mutex_unlock(&t->mu);
    tr_local_opt(t, &cenv, x0, y0, 8, 6, true);
    pthread_mutex_lock(&t->mu);
    t->gen++;
    pthread_mutex_unlock(&t->mu);
  } else {
    /* resume: fringe = every SET cell with an EMPTY 8-neighbor */
    for (int j = 0; j < (int)H; j++)
      for (int i = 0; i < (int)W; i++) {
        if (!tr_valid(t, i, j)) continue;
        bool edge = false;
        for (int dj = -1; dj <= 1 && !edge; dj++)
          for (int di = -1; di <= 1; di++) {
            int ii = i + di, jj = j + dj;
            if (ii < 0 || jj < 0 || ii >= (int)W || jj >= (int)H) continue;
            if (t->state[(size_t)jj * W + (size_t)ii] == R3D_TR_EMPTY) {
              edge = true;
              break;
            }
          }
        if (edge) fringe[nf++] = (uint32_t)((size_t)j * W + (size_t)i);
      }
  }

  static const int n8[8][2] = {{1, 0},  {0, 1},  {-1, 0}, {0, -1},
                               {1, 1},  {1, -1}, {-1, 1}, {-1, -1}};
  uint32_t budget = t->cfg.max_ring > t->gens_done ? t->cfg.max_ring - t->gens_done : 0;
  uint32_t gens_run = 0;
  while (nf > 0 && gens_run < budget && !t->quit) {
    gens_run++;
    uint32_t generation = t->gens_done + gens_run;
    bool global_opt = generation <= 10 && !resume;
    /* candidates: 8-neighborhood of the fringe */
    uint32_t nc = 0;
    for (uint32_t f = 0; f < nf; f++) {
      int i = (int)(fringe[f] % W), j = (int)(fringe[f] / W);
      for (int o = 0; o < 8; o++) {
        int ii = i + n8[o][0], jj = j + n8[o][1];
        if (ii < 2 || jj < 2 || ii >= (int)W - 2 || jj >= (int)H - 2) continue;
        size_t k = (size_t)jj * W + (size_t)ii;
        if (t->state[k] != R3D_TR_EMPTY) continue;
        t->state[k] = R3D_TR_PROC; /* offered once, ever (vc3d) */
        cands[nc++] = (uint32_t)k;
      }
    }
    ng_prefetch_reset(&ng);
    if (ng.nfth) /* a generation of lead time: enqueue the grid slices
                  * every candidate will touch before any solving starts */
      for (uint32_t ci = 0; ci < nc; ci++) {
        int i = (int)(cands[ci] % W), j = (int)(cands[ci] / W);
        for (int dj = -1; dj <= 1; dj++)
          for (int di = -1; di <= 1; di++) {
            if (!tr_valid(t, i + di, j + dj)) continue;
            const double *P = t->pos + ((size_t)(j + dj) * W + (size_t)(i + di)) * 3;
            int rng = (int)t->cfg.step + 4; /* placement travels a full
                                             * grid step from its parent */
            for (int dd = -rng; dd <= rng; dd++) {
              ng_prefetch(&ng, 0, (int)llround(P[2]) + dd);
              ng_prefetch(&ng, 1, (int)llround(P[1]) + dd);
              ng_prefetch(&ng, 2, (int)llround(P[0]) + dd);
            }
          }
      }
    uint32_t nnew = 0;
    if (pool.nth && nc >= 8 && !t->quit) { /* parallel (grids are the slow
                                             * path and are thread-safe) */
      nnew = tr_pool_run2(&pool, &cenv, cands, nc, nfringe, 0);
    } else {
      unsigned rng = t->rng;
      for (uint32_t ci = 0; ci < nc && !t->quit; ci++)
        if (tr_place_cand(t, &cenv, cands[ci], &rng)) nfringe[nnew++] = cands[ci];
      t->rng = rng;
    }
    for (uint32_t f = 0; f < nnew; f++) /* conf: coordinator only (DT) */
      tr_update_conf(t, &cenv, (int)(nfringe[f] % W), (int)(nfringe[f] / W));
    if (t->cfg.wind_weight > 0 && t->uc && t->nset >= 200 &&
        (t->sp_om_meas == 0.0 || generation % 8 == 2))
      t->sp_om_meas = tr_omega_measure(t, cenv.dt);
    tr_spiral_fit(t); /* refit the global spiral before the anneal so the
                       * winding prior joins the big solves */
    tr_spiral_flag(t);
    tr_don_register(t);
    /* schedule: vc3d runs global solves only in the first 10 generations
     * (Ceres cost); our parallel sweeps are cheap enough to keep them
     * coming — every 8th generation, forever. This anneals away the
     * candidate-order nondeterminism that tears weak outskirt regions
     * (same-build 60-gen QC swung 0.06%..9.7% v-edges >30 vox). */
    if (!resume && generation % 8 == 0) {
      tr_local_opt(t, &cenv, x0, y0, (int)W + (int)H, global_opt ? 6 : 3, true);
    } else if (!global_opt) {
      uint32_t nsub = 0;
      for (uint32_t f = 0; f < nnew; f++) {
        int i = (int)(nfringe[f] % W), j = (int)(nfringe[f] / W);
        if (i % 4 == 0 && j % 4 == 0) cands[nsub++] = nfringe[f];
      }
      if (pool.nth && nsub >= 4) {
        tr_pool_run2(&pool, &cenv, cands, nsub, cands + nsub, 1);
      } else {
        for (uint32_t f = 0; f < nsub && !t->quit; f++)
          tr_local_opt(t, &cenv, (int)(cands[f] % W), (int)(cands[f] / W), 8, 3, true);
      }
    }
    memcpy(fringe, nfringe, (size_t)nnew * sizeof *fringe);
    nf = nnew;
    pthread_mutex_lock(&t->mu);
    t->ring = generation;
    t->gen++;
    pthread_mutex_unlock(&t->mu);
  }
  t->gens_done += gens_run;
  /* final polish: one bounded pass so late cells see settled neighbors */
  if (!t->quit) tr_local_opt(t, &cenv, x0, y0, (int)W + (int)H, 4, true);
  free(fringe);
  free(nfringe);
  free(cands);
  tr_pool_destroy(&pool);
  tr_env_flush(&cenv);
  free(cenv.hood);
  printf("tracer: finished at generation %u with %u point%s (level L%u%s)\n", t->ring,
         t->nset, t->nset == 1 ? "" : "s", t->cfg.level,
         ng.active ? ", normal grids" : "");
  if (t->sp_valid)
    printf("tracer: spiral fit omega %.2f vox/winding, rms %.2f vox\n", t->sp_omega,
           t->sp_rms);
  printf("tracer: %.1fs total | place %.1fs lopt %.1fs (ncp %.1fs hood %.1fs) "
         "conf %.1fs gridfetch %.1fs\n",
         tr_now() - tr_t_start, TR_TM(0), TR_TM(1), TR_TM(4), TR_TM(5), TR_TM(2),
         TR_TM(3));
  ng_close(&ng);
  td_close(dt);
  r3d_cpuvol_close(&vol);
  pthread_mutex_lock(&t->mu);
  t->done = true;
  t->running = false;
  t->gen++;
  pthread_mutex_unlock(&t->mu);
  return NULL;

fail_open:
  pthread_mutex_lock(&t->mu);
  t->done = true;
  t->running = false;
  pthread_mutex_unlock(&t->mu);
  return NULL;
}

/* ============================ lifecycle ============================ */
int r3d_tracer_start(r3d_tracer *t, const char *pred_root, const r3d_tracer_cfg *cfg,
                     const r3d_umbilicus *umb) {
  return r3d_tracer_start_fused(t, pred_root, cfg, umb, NULL, 0);
}

int r3d_tracer_start_fused(r3d_tracer *t, const char *pred_root,
                           const r3d_tracer_cfg *cfg, const r3d_umbilicus *umb,
                           const char *const *donor_dirs, uint32_t ndonors) {
  memset(t, 0, sizeof *t);
  t->cfg = *cfg;
  if (t->cfg.max_ring < 4) t->cfg.max_ring = 4;
  if (t->cfg.max_ring > 400) t->cfg.max_ring = 400;
  if (t->cfg.step < 1.0) t->cfg.step = 20.0;
  snprintf(t->root, sizeof t->root, "%s", pred_root);
  t->W = t->H = 2 * t->cfg.max_ring + 50;
  t->pos = calloc((size_t)t->W * t->H * 3, sizeof *t->pos);
  t->state = calloc((size_t)t->W * t->H, 1);
  t->conf = calloc((size_t)t->W * t->H, sizeof *t->conf);
  t->wind = calloc((size_t)t->W * t->H, sizeof *t->wind);
  t->rng = 0x1234567u;
  if (!t->pos || !t->state || !t->conf || !t->wind) {
    r3d_tracer_free(t);
    return -1;
  }
  r3d_umbilicus_init(&t->umb);
  if (umb)
    for (size_t k = 0; k < umb->count; k++)
      r3d_umbilicus_set(&t->umb, umb->points[k].x, umb->points[k].y, umb->points[k].z);
  if (ndonors) {
    t->don = tr_dons_load(donor_dirs, ndonors, t->cfg.step);
    if (t->don) {
      t->ndon = ((tr_dons *)t->don)->n;
      t->dsup = calloc((size_t)t->W * t->H, 1);
      if (!t->dsup) {
        r3d_tracer_free(t);
        return -1;
      }
    }
  }
  pthread_mutex_init(&t->mu, NULL);
  t->running = true;
  if (pthread_create(&t->th, NULL, tr_worker, t) != 0) {
    pthread_mutex_destroy(&t->mu);
    r3d_tracer_free(t);
    return -1;
  }
  return 0;
}

int r3d_tracer_grow(r3d_tracer *t, uint32_t extra) {
  if (t->running || !t->pos || !extra) return -1;
  uint32_t nr = t->cfg.max_ring + extra;
  if (nr > 400) nr = 400;
  if (nr == t->cfg.max_ring) return -1;
  uint32_t NW = 2 * nr + 50, off = nr - t->cfg.max_ring;
  double *np = calloc((size_t)NW * NW * 3, sizeof *np);
  uint8_t *ns = calloc((size_t)NW * NW, 1);
  float *nc = calloc((size_t)NW * NW, sizeof *nc);
  float *nw = calloc((size_t)NW * NW, sizeof *nw);
  uint8_t *nd = t->dsup ? calloc((size_t)NW * NW, 1) : NULL;
  if (!np || !ns || !nc || !nw || (t->dsup && !nd)) {
    free(np);
    free(ns);
    free(nc);
    free(nw);
    free(nd);
    return -1;
  }
  for (uint32_t j = 0; j < t->H; j++)
    for (uint32_t i = 0; i < t->W; i++) {
      size_t ok = (size_t)j * t->W + i;
      size_t nk = (size_t)(j + off) * NW + (i + off);
      if (t->state[ok] == R3D_TR_SET) { /* PROC cells reset for another try */
        ns[nk] = R3D_TR_SET;
        nc[nk] = t->conf[ok];
        nw[nk] = t->wind[ok];
        if (nd) nd[nk] = t->dsup[ok];
        memcpy(np + nk * 3, t->pos + ok * 3, 3 * sizeof(double));
      }
    }
  free(t->pos);
  free(t->state);
  free(t->conf);
  free(t->wind);
  free(t->dsup);
  t->pos = np;
  t->state = ns;
  t->conf = nc;
  t->wind = nw;
  t->dsup = nd;
  t->W = t->H = NW;
  t->cfg.max_ring = nr;
  t->quit = false;
  t->done = false;
  t->gen++;
  t->running = true;
  if (pthread_create(&t->th, NULL, tr_worker, t) != 0) {
    t->running = false;
    return -1;
  }
  return 0;
}

void r3d_tracer_stop(r3d_tracer *t) {
  if (!t->pos) return;
  t->quit = true;
  if (t->running || t->th) pthread_join(t->th, NULL);
  t->th = 0;
  t->running = false;
}

void r3d_tracer_free(r3d_tracer *t) {
  free(t->pos);
  free(t->state);
  free(t->conf);
  free(t->wind);
  free(t->dsup);
  free(t->uc);
  tr_dons_free(t->don);
  r3d_umbilicus_free(&t->umb);
  memset(t, 0, sizeof *t);
}

uint64_t r3d_tracer_snapshot(r3d_tracer *t, double *pos, uint8_t *state, float *conf,
                             uint32_t *ring, uint32_t *nset, bool *done) {
  pthread_mutex_lock(&t->mu);
  if (pos) memcpy(pos, t->pos, (size_t)t->W * t->H * 3 * sizeof *pos);
  if (state) memcpy(state, t->state, (size_t)t->W * t->H);
  if (conf) memcpy(conf, t->conf, (size_t)t->W * t->H * sizeof *conf);
  if (ring) *ring = t->ring;
  if (nset) *nset = t->nset;
  if (done) *done = t->done;
  uint64_t g = t->gen;
  pthread_mutex_unlock(&t->mu);
  return g;
}

/* ============================== save ============================== */
static int tr_write_plane(const char *path, const float *v, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, "w8");
  if (!tf) return -1;
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 64);
  int rc = 0;
  for (uint32_t j = 0; j < h && rc == 0; j++)
    if (TIFFWriteScanline(tf, (void *)(v + (size_t)j * w), j, 0) < 0) rc = -1;
  TIFFClose(tf);
  return rc;
}

int r3d_tracer_save(r3d_tracer *t, const char *dir, float cutoff, bool fill) {
  if (!t->pos) return -1;
  uint64_t n = (uint64_t)t->W * t->H;
  float *pl = malloc(n * sizeof *pl);
  if (!pl) return -1;
  static const char *nm[3] = {"x.tif", "y.tif", "z.tif"};
  int rc = 0;
  pthread_mutex_lock(&t->mu);
  /* tear mask: a cell whose edge to any 4-neighbor is way off the unit
   * is mis-seated (usually a wrong-wrap capture in ambiguous data) — an
   * honest hole beats a committed discontinuity */
  uint8_t *keep = malloc(n);
  if (keep) {
    double lim = 1.75 * t->cfg.step;
    for (uint64_t k = 0; k < n; k++) {
      keep[k] = t->state[k] == R3D_TR_SET && t->conf[k] >= cutoff;
      if (!keep[k]) continue;
      int i = (int)(k % t->W), j = (int)(k / t->W);
      static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (int o = 0; o < 4 && keep[k]; o++) {
        int ii = i + o4[o][0], jj = j + o4[o][1];
        if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
        size_t k2 = (size_t)jj * t->W + (size_t)ii;
        if (t->state[k2] != R3D_TR_SET) continue;
        double d2 = 0;
        for (int a = 0; a < 3; a++) {
          double dd = t->pos[k * 3 + (size_t)a] - t->pos[k2 * 3 + (size_t)a];
          d2 += dd * dd;
        }
        if (d2 > lim * lim) keep[k] = 0;
      }
    }
    /* infill: an isolated low-conf cell whose neighbors are trusted and
     * whose own edges are sane is geometrically vouched for. Scattered
     * pinholes otherwise cost 4x their area on screen — the bilinear
     * rule blacks every quad touching an invalid corner. */
    for (uint64_t k = 0; k < n; k++) {
      if (keep[k] || t->state[k] != R3D_TR_SET) continue;
      int i = (int)(k % t->W), j = (int)(k / t->W);
      int good = 0;
      bool sane = true;
      for (int dj = -1; dj <= 1 && sane; dj++)
        for (int di = -1; di <= 1; di++) {
          if (!di && !dj) continue;
          int ii = i + di, jj = j + dj;
          if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
          size_t k2 = (size_t)jj * t->W + (size_t)ii;
          if (keep[k2] != 1) continue; /* originals only: no cascades */
          good++;
          double d2 = 0;
          for (int a = 0; a < 3; a++) {
            double dd = t->pos[k * 3 + (size_t)a] - t->pos[k2 * 3 + (size_t)a];
            d2 += dd * dd;
          }
          double lim2 = lim * (di && dj ? 1.4142135623730951 : 1.0);
          if (d2 > lim2 * lim2) sane = false;
        }
      if (good >= 6 && sane) keep[k] = 2; /* 2: infilled (not a seed for
                                           * further infill) */
    }
    if (fill) {
      /* no holes: every grown cell gets a point. Unkept cells (low conf
       * or tear-cut) are re-seated by membrane interpolation anchored on
       * kept neighbors — smooth continuation instead of a skip. */
      double *fp = malloc(n * 3 * sizeof *fp);
      if (fp) {
        memcpy(fp, t->pos, n * 3 * sizeof *fp);
        for (int it = 0; it < 64; it++) {
          double moved = 0.0;
          for (uint64_t k = 0; k < n; k++) {
            if (keep[k] || t->state[k] != R3D_TR_SET) continue;
            int i = (int)(k % t->W), j = (int)(k / t->W);
            double avg[3] = {0, 0, 0};
            int na = 0;
            static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int o = 0; o < 4; o++) {
              int ii = i + o4[o][0], jj = j + o4[o][1];
              if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
              size_t k2 = (size_t)jj * t->W + (size_t)ii;
              if (t->state[k2] != R3D_TR_SET) continue;
              const double *src = keep[k2] ? t->pos + k2 * 3 : fp + k2 * 3;
              for (int a = 0; a < 3; a++) avg[a] += src[a];
              na++;
            }
            if (na < 2) continue;
            for (int a = 0; a < 3; a++) {
              double nv2 = avg[a] / na;
              moved += fabs(nv2 - fp[k * 3 + (size_t)a]);
              fp[k * 3 + (size_t)a] = nv2;
            }
          }
          if (moved < 0.01 * (double)n) break;
        }
        for (uint64_t k = 0; k < n; k++)
          if (!keep[k] && t->state[k] == R3D_TR_SET) {
            memcpy(t->pos + k * 3, fp + k * 3, 3 * sizeof(double));
            keep[k] = 3; /* filled */
          }
        free(fp);
      }
    }
  }
  for (int a = 0; a < 3 && rc == 0; a++) {
    for (uint64_t k = 0; k < n; k++)
      pl[k] = (keep ? keep[k]
                    : t->state[k] == R3D_TR_SET && t->conf[k] >= cutoff)
                  ? (float)t->pos[k * 3 + (size_t)a]
                  : -1.0f;
    char path[1200];
    snprintf(path, sizeof path, "%s/%s", dir, nm[a]);
    rc = tr_write_plane(path, pl, t->W, t->H);
  }
  if (rc == 0 && t->uc && t->wind) { /* winding channel (vc3d ecosystem:
                                      * float winding per cell, NAN off) */
    for (uint64_t k = 0; k < n; k++)
      pl[k] = (keep ? keep[k] != 0
                    : t->state[k] == R3D_TR_SET && t->conf[k] >= cutoff)
                  ? t->wind[k]
                  : -1e30f; /* invalid marker (validity lives in x.tif;
                             * NAN is unusable under -ffast-math) */
    char path[1200];
    snprintf(path, sizeof path, "%s/winding.tif", dir);
    rc = tr_write_plane(path, pl, t->W, t->H);
  }
  pthread_mutex_unlock(&t->mu);
  free(keep);
  free(pl);
  if (rc != 0) return -1;
  char mp[1200];
  snprintf(mp, sizeof mp, "%s/meta.json", dir);
  FILE *mf = fopen(mp, "w");
  if (!mf) return -1;
  double sc = 1.0 / t->cfg.step;
  fprintf(mf,
          "{\n  \"format\": \"tifxyz\",\n  \"type\": \"seg\",\n  \"scale\": [\n"
          "    %.6f,\n    %.6f\n  ],\n  \"source\": \"render3d-tracer\"\n}\n",
          sc, sc);
  fclose(mf);
  if (t->sp_valid) { /* spiral model, for fusion / winding registration */
    snprintf(mp, sizeof mp, "%s/spiral.json", dir);
    FILE *sf = fopen(mp, "w");
    if (sf) {
      fprintf(sf, "{\n  \"omega\": %.6f,\n  \"omega_measured\": %.3f,\n"
                  "  \"rms\": %.6f,\n  \"z0\": %.3f,\n"
                  "  \"dz\": %.3f,\n  \"r0\": [",
              t->sp_omega, t->sp_om_meas, t->sp_rms, t->sp_z0, t->sp_dz);
      for (uint32_t kk = 0; kk < t->sp_k; kk++)
        fprintf(sf, "%s%.3f", kk ? ", " : "", t->sp_r0[kk]);
      fprintf(sf, "]\n}\n");
      fclose(sf);
    }
  }
  return 0;
}

/* ========================= spiral selftest =========================
 * Pure-synthetic check of the winding frame: exact unwrap along a strip,
 * omega/r0 recovery by the global fit, and wrong-wrap flagging. */
int r3d_tracer_spiral_selftest(void) {
  int rc = -1;
  r3d_tracer t = {0};
  t.W = 64;
  t.H = 24;
  t.cfg.step = 20.0;
  t.cfg.wind_weight = 0.5;
  uint64_t N = (uint64_t)t.W * t.H;
  t.pos = calloc(N * 3, sizeof *t.pos);
  t.state = calloc(N, 1);
  t.conf = calloc(N, sizeof *t.conf);
  t.wind = calloc(N, sizeof *t.wind);
  pthread_mutex_init(&t.mu, NULL);
  r3d_umbilicus_init(&t.umb);
  if (!t.pos || !t.state || !t.conf || !t.wind) goto out;
  r3d_umbilicus_set(&t.umb, 500.0, 400.0, 0.0);
  r3d_umbilicus_set(&t.umb, 500.0, 400.0, 199.0);
  tr_uc_build(&t, 200);
  if (!t.uc) goto out;
  const double om = 18.0, r0 = 30.0;
  for (uint32_t j = 0; j < t.H; j++)
    for (uint32_t i = 0; i < t.W; i++) {
      size_t k = (size_t)j * t.W + i;
      double w = (double)i * 0.05;
      double th = 2.0 * M_PI * w, rho = r0 + om * w;
      t.pos[k * 3 + 0] = 500.0 + rho * cos(th);
      t.pos[k * 3 + 1] = 400.0 + rho * sin(th);
      t.pos[k * 3 + 2] = 40.0 + (double)j * 6.0;
      t.state[k] = R3D_TR_SET;
      t.conf[k] = 1.0f;
    }
  /* incremental unwrap along each row (the placement rule, 1-D chain) */
  for (uint32_t j = 0; j < t.H; j++) {
    t.wind[(size_t)j * t.W] = 0.0f;
    for (uint32_t i = 1; i < t.W; i++) {
      size_t k = (size_t)j * t.W + i, kp = k - 1;
      double th, thp;
      if (!tr_theta_of(&t, t.pos + k * 3, &th) ||
          !tr_theta_of(&t, t.pos + kp * 3, &thp))
        goto out;
      double d = th - thp;
      while (d > M_PI) d -= 2.0 * M_PI;
      while (d < -M_PI) d += 2.0 * M_PI;
      t.wind[k] = (float)((double)t.wind[kp] + d / (2.0 * M_PI));
      if (fabs((double)t.wind[k] - (double)i * 0.05) > 1e-4) goto out;
    }
  }
  tr_spiral_fit(&t);
  if (!t.sp_valid || fabs(t.sp_omega - om) > 0.5 || t.sp_rms > 1.0) goto out;
  { /* narrow-span patch (0.16 windings): joint omega is unidentifiable,
     * the measured-gap path must carry it */
    for (uint32_t j = 0; j < t.H; j++)
      for (uint32_t i = 0; i < t.W; i++) {
        size_t k = (size_t)j * t.W + i;
        double w = (double)i * 0.0025;
        double th = 2.0 * M_PI * w, rho = r0 + om * w;
        t.pos[k * 3 + 0] = 500.0 + rho * cos(th);
        t.pos[k * 3 + 1] = 400.0 + rho * sin(th);
        t.wind[k] = (float)w;
        t.conf[k] = 1.0f;
      }
    t.sp_valid = false;
    tr_spiral_fit(&t);
    if (t.sp_valid) goto out; /* must refuse without a measured omega */
    t.sp_om_meas = om;
    tr_spiral_fit(&t);
    if (!t.sp_valid || fabs(t.sp_omega - om) > 1e-6 || t.sp_rms > 1.0) goto out;
    /* restore the wide-span spiral for the flag check */
    for (uint32_t j = 0; j < t.H; j++)
      for (uint32_t i = 0; i < t.W; i++) {
        size_t k = (size_t)j * t.W + i;
        double w = (double)i * 0.05;
        double th = 2.0 * M_PI * w, rho = r0 + om * w;
        t.pos[k * 3 + 0] = 500.0 + rho * cos(th);
        t.pos[k * 3 + 1] = 400.0 + rho * sin(th);
        t.wind[k] = (float)w;
      }
    t.sp_om_meas = 0.0;
    tr_spiral_fit(&t);
    if (!t.sp_valid) goto out;
  }
  { /* shove one interior cell a full wrap outward: flag must catch it */
    size_t k = (size_t)12 * t.W + 40;
    double th = 2.0 * M_PI * 40.0 * 0.05, rho = r0 + om * (40.0 * 0.05 + 1.0);
    t.pos[k * 3 + 0] = 500.0 + rho * cos(th);
    t.pos[k * 3 + 1] = 400.0 + rho * sin(th);
    tr_spiral_flag(&t);
    if (t.conf[k] > 0.25f) goto out;
    size_t k2 = (size_t)12 * t.W + 39; /* honest neighbor stays trusted */
    if (t.conf[k2] < 0.9f) goto out;
  }
  rc = 0;
out:
  free(t.pos);
  free(t.state);
  free(t.conf);
  free(t.wind);
  free(t.uc);
  r3d_umbilicus_free(&t.umb);
  pthread_mutex_destroy(&t.mu);
  return rc;
}

/* ========================= fusion selftest =========================
 * Synthetic donor round-trip: write a flat tifxyz + winding channel,
 * load + index it, and check closest-point queries, the winding
 * interpolation, and the support vote. */
int r3d_tracer_fusion_selftest(void) {
  int rc = -1;
  char dir[] = "/tmp/r3d_test_donor_XXXXXX";
  if (!mkdtemp(dir)) return -1;
  const uint32_t W = 12, H = 10;
  float *pl = malloc((size_t)W * H * sizeof *pl);
  tr_dons *dn = NULL;
  if (!pl) goto out;
  static const char *nm[4] = {"x.tif", "y.tif", "z.tif", "winding.tif"};
  for (int a = 0; a < 4; a++) { /* plane z=50, spacing 20, wind = i*0.05 */
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++) {
        float v = a == 0 ? (float)i * 20.0f
                  : a == 1 ? (float)j * 20.0f
                  : a == 2 ? 50.0f
                           : (float)i * 0.05f;
        pl[(size_t)j * W + i] = v;
      }
    char path[1300];
    snprintf(path, sizeof path, "%s/%s", dir, nm[a]);
    if (tr_write_plane(path, pl, W, H) != 0) goto out;
  }
  {
    char mp[1300];
    snprintf(mp, sizeof mp, "%s/meta.json", dir);
    FILE *mf = fopen(mp, "w");
    if (!mf) goto out;
    fprintf(mf, "{\"format\":\"tifxyz\",\"scale\":[0.05,0.05]}\n");
    fclose(mf);
  }
  const char *dirs[1] = {dir};
  dn = tr_dons_load(dirs, 1, 20.0);
  if (!dn || dn->n != 1 || !dn->d[0].dwind) goto out;
  {
    double q[3], wnd;
    double p[3] = {35.0, 42.0, 53.0};
    int di = tr_don_closest(dn, p, TR_FUS_PULL, q, &wnd);
    if (di != 0 || fabs(q[0] - 35.0) > 1e-6 || fabs(q[1] - 42.0) > 1e-6 ||
        fabs(q[2] - 50.0) > 1e-6)
      goto out;
    if (wnd > -1e29) goto out; /* not registered yet: no winding */
    dn->d[0].woff = 2.0;
    dn->d[0].woff_ok = true;
    di = tr_don_closest(dn, p, TR_FUS_PULL, q, &wnd);
    if (di != 0 || fabs(wnd - (35.0 / 20.0 * 0.05 + 2.0)) > 1e-4) goto out;
    /* support: on-surface within threshold, winding-gated */
    double ps[3] = {35.0, 42.0, 51.0};
    if (tr_don_support(dn, ps, wnd, true) != 1) goto out;
    if (tr_don_support(dn, ps, wnd + 1.0, true) != 0) goto out; /* wrong wrap */
    double pf[3] = {35.0, 42.0, 58.0}; /* beyond same-surface threshold */
    if (tr_don_support(dn, pf, wnd, true) != 0) goto out;
  }
  rc = 0;
out:
  if (dn) tr_dons_free(dn);
  free(pl);
  {
    char cmd[1400];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
    if (system(cmd) != 0) rc = rc == 0 ? 0 : rc;
  }
  return rc;
}
