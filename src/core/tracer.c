#include "core/tracer.h"

#include <curl/curl.h>
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__GLIBC__) || defined(__linux__)
#include <malloc.h>
#endif
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
#define TR_W_FOLD 2.0     /* anti-double-back hinge (inactive above 90 deg) */
#define TR_FOLD_COS 0.0   /* hinge threshold: cos(90 deg) */
#define TR_W_PLANAR 0.25  /* quad twist: free corner vs other-3 plane */
#define TR_W_CTSNAP 2.0   /* CT edge pull along the frozen normal (ctsnap) */
#define TR_W_ANC 2.0      /* user-anchor pull: "must pass through" — an order
                           * stronger than the donor pull (0.1) so it beats
                           * the data term when the sheet picked wrong */
#define TR_ANC_CAPTURE 3.0 /* capture radius, grid steps: an anchor farther
                            * than this from every grown cell stays idle
                            * until the front approaches (a hard pull across
                            * half the scroll would rip the grid) */
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

static void td_build_raw(r3d_cpuvol *vol, uint32_t level, int64_t cx, int64_t cy,
                         int64_t cz, float *sq, uint8_t *out);

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
  if (lru->key != 0 && c->memo_d == lru->d) c->memo_key = 0; /* the memo
      * must die with the evicted chunk: the d buffer is reused, and a
      * stale memo would return the NEW chunk's data under the OLD key */
  td_build_raw(c->vol, c->level, cx, cy, cz, c->sq, lru->d);
  lru->key = key;
  lru->use = ++c->tick;
  c->memo_key = key;
  c->memo_d = lru->d;
  return lru->d;
}

/* compute one DT chunk standalone: thread-safe (cpuvol reads only, no
 * cache mutation) — the parallel conf prewarm builds chunks on worker
 * threads and inserts them on the coordinator */
static void td_build_raw(r3d_cpuvol *vol, uint32_t level, int64_t cx, int64_t cy,
                         int64_t cz, float *sq, uint8_t *out) {
  /* occupancy of the extended block, squared-EDT seeds */
  const double sc = (double)vol->lev[level].scale;
  for (int64_t lz = 0; lz < TD_EXT; lz++)
    for (int64_t ly = 0; ly < TD_EXT; ly++)
      for (int64_t lx = 0; lx < TD_EXT; lx++) {
        double bx = ((double)(cx * TD_CORE - TD_BORD + lx) + 0.5) * sc;
        double by = ((double)(cy * TD_CORE - TD_BORD + ly) + 0.5) * sc;
        double bz = ((double)(cz * TD_CORE - TD_BORD + lz) + 0.5) * sc;
        uint8_t v = r3d_cpuvol_at(vol, level, bx, by, bz);
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
        out[((size_t)z2 * TD_CORE + (size_t)y2) * TD_CORE + (size_t)x2] =
            dd >= 255.0f ? 255 : (uint8_t)(dd + 0.5f);
      }
}

static double td_vox(td_cache *c, int64_t vx, int64_t vy, int64_t vz) {
  if (vx < 0 || vy < 0 || vz < 0) return 255.0;
  const uint8_t *d = td_chunk(c, vx / TD_CORE, vy / TD_CORE, vz / TD_CORE);
  if (!d) return 255.0;
  return (double)d[(((size_t)(vz % TD_CORE)) * TD_CORE + (size_t)(vy % TD_CORE)) * TD_CORE +
                   (size_t)(vx % TD_CORE)];
}

/* is the chunk already cached? (coordinator-only, like all cache access) */
static bool td_have(td_cache *c, uint64_t key) {
  uint64_t h = key * 0x9E3779B97F4A7C15ull;
  uint32_t base = (uint32_t)(h >> 32) % TD_SLOTS;
  for (uint32_t p = 0; p < 8; p++)
    if (c->s[(base + p) % TD_SLOTS].key == key) return true;
  return false;
}

/* insert a ready-built chunk, taking ownership of d */
static void td_insert(td_cache *c, uint64_t key, uint8_t *d) {
  uint64_t h = key * 0x9E3779B97F4A7C15ull;
  uint32_t base = (uint32_t)(h >> 32) % TD_SLOTS;
  td_slot *lru = NULL;
  for (uint32_t p = 0; p < 8; p++) {
    td_slot *sl = &c->s[(base + p) % TD_SLOTS];
    if (sl->key == key) { /* raced with a direct build: keep the existing */
      free(d);
      return;
    }
    if (sl->key == 0) {
      if (!lru || lru->key != 0) lru = sl;
    } else if (!lru || (lru->key != 0 && sl->use < lru->use)) {
      lru = sl;
    }
  }
  if (c->memo_d && c->memo_d == lru->d) c->memo_key = 0;
  free(lru->d);
  lru->d = d;
  lru->key = key;
  lru->use = ++c->tick;
}

/* parallel conf prewarm: build the missing DT chunks for a set of world
 * points on ad-hoc threads (each with its own scratch; cpuvol is
 * thread-safe), then insert on the calling thread. The per-generation
 * conf pass was 35% of trace wall time, nearly all of it serial chunk
 * builds. */
#define TD_PW_MAX 512
typedef struct td_pw_job {
  r3d_cpuvol *vol;
  uint32_t level;
  const uint64_t *keys;
  uint8_t **out;
  _Atomic uint32_t *next;
  uint32_t n;
} td_pw_job;

static void *td_pw_thread(void *ud) {
  td_pw_job *j = ud;
  float *sq = malloc((size_t)TD_EXT * TD_EXT * TD_EXT * sizeof *sq);
  if (!sq) return NULL;
  for (;;) {
    uint32_t i = atomic_fetch_add(j->next, 1);
    if (i >= j->n) break;
    uint64_t k = j->keys[i] - 1u;
    uint8_t *d = malloc((size_t)TD_CORE * TD_CORE * TD_CORE);
    if (!d) break;
    td_build_raw(j->vol, j->level, (int64_t)(k & 0xFFFFFu),
                 (int64_t)((k >> 20) & 0xFFFFFu), (int64_t)(k >> 40), sq, d);
    j->out[i] = d;
  }
  free(sq);
  return NULL;
}

static void td_prewarm(td_cache *c, const double *pts3, const uint32_t *cells,
                       uint32_t ncell, uint32_t stride_w) {
  if (!c || !ncell) return;
  uint64_t keys[TD_PW_MAX];
  uint32_t nk = 0;
  const double sc = (double)c->vol->lev[c->level].scale;
  for (uint32_t f = 0; f < ncell && nk < TD_PW_MAX; f++) {
    const double *P = pts3 + (size_t)cells[f] * 3;
    (void)stride_w;
    for (int corner = 0; corner < 8; corner++) {
      int64_t vx = (int64_t)floor(P[0] / sc) + (corner & 1);
      int64_t vy = (int64_t)floor(P[1] / sc) + ((corner >> 1) & 1);
      int64_t vz = (int64_t)floor(P[2] / sc) + (corner >> 2);
      if (vx < 0 || vy < 0 || vz < 0) continue;
      uint64_t key = 1u + (((uint64_t)(vz / TD_CORE) << 40) |
                           ((uint64_t)(vy / TD_CORE) << 20) | (uint64_t)(vx / TD_CORE));
      bool dup = false;
      for (uint32_t q = 0; q < nk && !dup; q++) dup = keys[q] == key;
      if (dup || td_have(c, key)) continue;
      if (nk < TD_PW_MAX) keys[nk++] = key;
    }
  }
  if (nk < 2) return; /* not worth the fan-out */
  uint8_t *out[TD_PW_MAX] = {0};
  _Atomic uint32_t next = 0;
  td_pw_job job = {c->vol, c->level, keys, out, &next, nk};
  pthread_t th[8];
  uint32_t nth = nk < 8 ? nk : 8; /* 16 measured slower (bandwidth-bound) */
  uint32_t started = 0;
  for (uint32_t i = 0; i < nth; i++)
    if (pthread_create(&th[started], NULL, td_pw_thread, &job) == 0) started++;
  if (!started) { /* fall back to serial builds on demand */
    return;
  }
  for (uint32_t i = 0; i < started; i++) pthread_join(th[i], NULL);
  for (uint32_t i = 0; i < nk; i++)
    if (out[i]) td_insert(c, keys[i], out[i]);
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

static _Atomic uint64_t ng_blob_bytes, ng_idx_bytes; /* footprint split */

static uint32_t ng_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

/* Decoded-slice RAM budget. A real trace's working set (the xz/yz slices
 * its frontier spans) commonly exceeds 1.5 GB, and an undersized budget
 * thrashes: profiling showed 28% of trace CPU re-parsing evicted grid
 * files inside residual evals. Default 6 GB; R3D_NG_BUDGET_MB overrides
 * for smaller machines. */
static size_t ng_budget(void) {
  static _Atomic size_t v;
  size_t b = atomic_load_explicit(&v, memory_order_relaxed);
  if (!b) {
    const char *ev = getenv("R3D_NG_BUDGET_MB");
    long mb = ev ? atol(ev) : 0;
    b = mb > 0 ? (size_t)mb << 20 : (size_t)6u << 30;
    atomic_store_explicit(&v, b, memory_order_relaxed);
  }
  return b;
}
/* concurrent tracers (GUI: up to 8) each own an ng_vol; the budget is
 * per-process, split evenly, so a seed-queue burst cannot OOM the box */
static _Atomic int ng_nopen;
#define NG_BUDGET (ng_budget() / (size_t)(atomic_load(&ng_nopen) > 1 ? atomic_load(&ng_nopen) : 1))

typedef struct ng_grid {
  size_t bytes;         /* footprint (budget accounting) */
  int32_t bx, by;       /* bounds origin (always 0 in published grids) */
  uint32_t w, h;        /* slice extent, px */
  uint32_t cell, gw, gh; /* bucket grid */
  uint8_t *blob;        /* paths region verbatim: {u32be sx, sy, noff,
                         * int8 deltas[noff]} records — 4x smaller than
                         * decoded float pairs; decoded per hood build */
  size_t blob_n;
  uint32_t *bidx;       /* [gw*gh+1] CSR into boff */
  uint32_t *boff;       /* blob-relative record OFFSET per bucket entry,
                         * verbatim from the file. Offsets are unique per
                         * path, so queries dedup and decode on them
                         * directly — no path-index table (a precoff[]
                         * array was half an xz grid's index RAM, and
                         * building it needed a full record walk + hash
                         * at every load) */
  bool empty;
} ng_grid;

static void ng_grid_free(ng_grid *g) {
  if (!g) return;
  free(g->blob);
  free(g->bidx);
  free(g->boff);
  free(g);
}

/* decode the path record at blob offset `po` into pts (caller cap >=
 * npts); returns point count, 0 on a malformed offset */
static uint32_t ng_path_decode(const ng_grid *g, uint32_t po, float *pts,
                               uint32_t cap) {
  if ((size_t)po + 12 > g->blob_n) return 0;
  const uint8_t *r = g->blob + po;
  double cx = ng_be32(r), cy = ng_be32(r + 4);
  uint32_t noff = ng_be32(r + 8), n = 0;
  if ((size_t)po + 12 + noff > g->blob_n || (noff & 1u)) return 0;
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
  if (po > pend) goto bad;
  g->blob_n = pend - po;
  g->blob = malloc(g->blob_n ? g->blob_n : 1);
  if (!g->blob) goto bad;
  memcpy(g->blob, d + po, g->blob_n);
  /* bucket CSR + record offsets, verbatim (offsets are validated at
   * decode time against blob_n) */
  uint32_t nboff = ng_be32(d + bio + 4u * nbuck);
  if ((size_t)po - bio - 4u * (nbuck + 1) < 4u * (size_t)nboff) goto bad;
  g->bidx = malloc((nbuck + 1) * sizeof *g->bidx);
  g->boff = malloc((nboff ? nboff : 1) * sizeof *g->boff);
  if (!g->bidx || !g->boff) goto bad;
  for (uint32_t i = 0; i <= nbuck; i++) g->bidx[i] = ng_be32(d + bio + 4u * i);
  for (uint32_t i = 0; i < nboff; i++)
    g->boff[i] = ng_be32(d + bio + 4u * (nbuck + 1) + 4u * i);
  free(d);
  g->bytes = sizeof *g + g->blob_n + (size_t)(nbuck + 1 + nboff) * sizeof(uint32_t);
  atomic_fetch_add_explicit(&ng_blob_bytes, g->blob_n, memory_order_relaxed);
  atomic_fetch_add_explicit(&ng_idx_bytes,
                            (size_t)(nbuck + 1 + nboff) * sizeof(uint32_t),
                            memory_order_relaxed);
  return g;
bad:
  free(d);
  ng_grid_free(g);
  return NULL;
}

/* vc3d get(center, radius): truncating rect, inclusive bucket range;
 * appends unique path indices to out[] (caller-sized), returns count */
static uint32_t ng_query(const ng_grid *g, double cx, double cy, double radius,
                         uint32_t *out, uint32_t max) {
  if (!g || g->empty || !g->blob_n) return 0;
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
        uint32_t po = g->boff[k]; /* blob-relative record offset: unique
                                   * per path, so it IS the identity */
        bool dup = false;
        for (uint32_t q = 0; q < nout && !dup; q++) dup = out[q] == po;
        if (!dup && nout < max) out[nout++] = po;
      }
    }
  return nout;
}

/* --- slice cache + demand fetch ------------------------------------- */
/* Slice-table capacity. This must cover the trace's DISTINCT slice
 * working set: a 60-gen trace touches ~10k (plane,slice) keys, and with
 * only 2048 slots the LRU replacement thrashed (403k loads for <16k
 * distinct files, 25x re-parse). 16384 covers a whole store; the byte
 * budget below is the actual RAM cap. idx[] entries hold slot+1 in a
 * u16, so NG_SLOTS must stay <= 16384. */
#define NG_SLOTS 16384
#define NG_IDXN 32768u /* open-addressed index size (2x slots) */
#define NG_IDXMASK (NG_IDXN - 1u)
#define NG_QMAX 64    /* paths per neighborhood query */
#define NG_NHOOD 2048 /* cached segment neighborhoods (per solver thread; ~40 MB) */
#define NG_HOODSEG 128 /* segments per neighborhood (measured: mean 55,
                         * max 104 on p343 — ng_hoodseg_binds counts any
                         * store that saturates this) */

/* phase timers, atomic ns: accumulated from every worker thread */
static _Atomic uint64_t tr_tm_ns[6]; /* 0 place 1 lopt 2 conf 3 ngfetch
                                      * 4 ngeval 5 hoodbuild */
/* truncation-cap bind counters: how often a neighborhood query filled
 * NG_QMAX paths or NG_HOODSEG segments (decides whether the caps ever
 * cost accuracy on real slices before we spend on distance-ranked cuts) */
static _Atomic uint64_t ng_qmax_binds, ng_hoodseg_binds;
/* load-traffic counters: how often slice grids are (re)parsed and the
 * peak decoded footprint — decides whether the budget or the parse cost
 * is the lever */
static _Atomic uint64_t ng_loads, ng_evicts, ng_bytes_peak;
static _Atomic uint64_t ng_eget_hit, ng_eget_miss;
static _Atomic uint64_t ng_hood_hits, ng_hood_builds;
static _Atomic uint64_t ng_hood_seg_sum, ng_hood_seg_max;
/* coordinator serial-phase wall timers (one thread, plain doubles) */
static double co_tm[8]; /* 0 dead 1 conf 2 omega 3 fit/don/sfx 4 windrelax
                         * 5 qc2 6 foldrepair */
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
  /* SoA sidecars for the hot residual pass: contiguous lanes vectorize
   * where the 13-float AoS records could not */
  float smidx[NG_HOODSEG], smidy[NG_HOODSEG]; /* segment midpoint */
  float snx[NG_HOODSEG], sny[NG_HOODSEG];     /* unit normal */
  float sgr2[NG_HOODSEG]; /* snap gate: (halflen + trig + slack)^2 — a
                           * segment farther than this from the chord end
                           * cannot pass the 4-px snap trigger */
  uint8_t nb[NG_HOODSEG];   /* bit0: prev exists, bit1: next exists */
  uint64_t use;
} ng_hood;

typedef struct ng_vol {
  char root[1200];  /* local cache dir (<pred_root>/ngrids) */
  char url[1400];   /* remote base; empty = local only */
  double spiral_step;
  int sparse;       /* slice stride of the store (1 = dense) */
  char lvldir[8];   /* multiscale: "0/" level directory, else "" */
  bool active;
  void *curl;
  uint64_t net_cool;
  struct {
    int plane, slice; /* plane -1 = free */
    ng_grid *g;       /* NULL = known missing */
    uint64_t use;
    uint32_t ref;     /* pinned by in-flight residual evals */
  } s[NG_SLOTS];
  uint16_t idx[NG_IDXN]; /* open-addressed (plane,slice) -> slot+1 */
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

/* G14: sparse stores publish every Nth slice; querying the in-between
 * slices hits nonexistent files that get cached as "known missing" and
 * the data term dies silently. Snap every requested slice to the grid. */
static inline int ng_snap(const struct ng_vol *v, int slice);

static size_t ng_curl_write(const void *data, size_t sz, size_t nm, void *ud) {
  FILE *f = ud;
  return fwrite(data, sz, nm, f);
}

static void *ng_prefetch_worker(void *ud);

/* open from the prediction tree: <root>/source.json url with ".zarr"
 * swapped for ".normal-grids"; metadata.json fetched once for the step */
static void ng_open(ng_vol *v, const char *pred_root) {
  atomic_fetch_add(&ng_nopen, 1);
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
    const char *sv2 = strstr(mj, "\"sparse-volume\"");
    v->sparse = sv2 ? atoi(sv2 + 16) : 1;
    if (v->sparse < 1) v->sparse = 1;
    if (strstr(mj, "\"normal-grid-multiscale\"")) {
      /* multiscale store: use the native level (0); its per-level
       * metadata may rescale coordinates, which we do not support yet */
      snprintf(v->lvldir, sizeof v->lvldir, "0/");
      printf("tracer: multiscale normal-grid store: using level 0\n");
    }
    if (v->sparse > 1)
      printf("tracer: sparse normal-grid store (every %d slices)\n", v->sparse);
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

static inline int ng_snap(const ng_vol *v, int slice) {
  if (v->sparse <= 1) return slice;
  return (int)((long)(slice + v->sparse / 2) / v->sparse * v->sparse);
}

static uint32_t ng_ih(int plane, int slice) {
  uint64_t key = (((uint64_t)(unsigned)plane << 32) | (unsigned)slice);
  return (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 40) & NG_IDXMASK;
}

/* find slot for (plane,slice) via the index; -1 = absent. gmu held. */
static int ng_islot(ng_vol *v, int plane, int slice) {
  for (uint32_t h = ng_ih(plane, slice), n = 0; n < NG_IDXN;
       h = (h + 1) & NG_IDXMASK, n++) {
    uint16_t e = v->idx[h];
    if (!e) return -1;
    uint32_t sl = (uint32_t)e - 1;
    if (v->s[sl].plane == plane && v->s[sl].slice == slice) return (int)sl;
  }
  return -1;
}

static void ng_idx_put(ng_vol *v, int plane, int slice, uint32_t slot) {
  for (uint32_t h = ng_ih(plane, slice), n = 0; n < NG_IDXN;
       h = (h + 1) & NG_IDXMASK, n++)
    if (!v->idx[h]) {
      v->idx[h] = (uint16_t)(slot + 1);
      return;
    }
}

static void ng_idx_del(ng_vol *v, int plane, int slice) {
  for (uint32_t h = ng_ih(plane, slice), n = 0; n < NG_IDXN;
       h = (h + 1) & NG_IDXMASK, n++) {
    uint16_t e = v->idx[h];
    if (!e) return;
    uint32_t sl = (uint32_t)e - 1;
    if (v->s[sl].plane == plane && v->s[sl].slice == slice) {
      v->idx[h] = 0;
      /* re-insert the probe run after the hole (open addressing) */
      for (uint32_t h2 = (h + 1) & NG_IDXMASK; v->idx[h2]; h2 = (h2 + 1) & NG_IDXMASK) {
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
  atomic_fetch_sub(&ng_nopen, 1);
  /* the decoded-grid gigabytes free here, but glibc keeps the arena
   * pages resident (measured: RSS unchanged after eviction); hand them
   * back so a GUI session recovers the RAM between traces */
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
#if defined(__GLIBC__)
  malloc_trim(0); /* see the comment at the top of this function */
#endif
}

/* fetch one slice file to disk with the given handle; thread-safe
 * (unique tmp names, atomic rename). Returns 0 when the file (or a
 * .missing sentinel) now exists. */
static int ng_fetch_file(ng_vol *v, CURL *c, int plane, int slice, bool sync) {
  char dir[1300], path[1500], miss[1520];
  snprintf(dir, sizeof dir, "%s/%s%s", v->root, v->lvldir, ng_plane_dir[plane]);
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
  snprintf(url, sizeof url, "%s/%s%s/%06d.grid", v->url, v->lvldir, ng_plane_dir[plane], slice);
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
  slice = ng_snap(v, slice);
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
  snprintf(path, sizeof path, "%s/%s%s/%06d.grid", v->root, v->lvldir, ng_plane_dir[plane], slice);
  double tt0 = tr_now();
  atomic_fetch_add(&ng_loads, 1);
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
    atomic_fetch_add(&ng_evicts, 1);
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
    atomic_fetch_add(&ng_evicts, 1);
  }
  {
    uint64_t b = atomic_load(&v->bytes), pk = atomic_load(&ng_bytes_peak);
    while (b > pk && !atomic_compare_exchange_weak(&ng_bytes_peak, &pk, b)) {}
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
  uint16_t hidx[4096]; /* open-addressed hood key -> slot+1 (the linear
                        * scan was 40%% of trace CPU) */
  uint32_t hclock;     /* clock-hand eviction cursor */
  uint64_t htick;
  bool in_pool;   /* set inside pool workers: never nest pools */
  struct tr_pool *pl; /* coordinator: the persistent worker pool */
  unsigned rng;
  /* held grid refs, open-addressed by (plane,slice): residual evals hit
   * this ~100M times per trace, so the lookup must be O(1) and almost
   * always avoid ng_get's global mutex (a linear FIFO here churned at an
   * 88% miss rate = 1.5M mutex acquisitions/s across the pool). used=0
   * (zero-init) marks an empty entry; refs released by tr_env_flush or
   * by LRU replacement within the 8-probe window. */
  struct { int plane, slice, slot; ng_grid *g; uint32_t use; uint8_t used; } gc[256];
  uint32_t gtick;
  int gl_plane, gl_slice, gl_slot; /* last-answer memo (gl_plane -1 = unset):
      * consecutive residuals overwhelmingly hit the same slice */
  ng_grid *gl_g;
  /* staged weight relaxation (G9, vc3d correction/inpaint continuation
   * schedule): scale factors on DistLoss, StraightLoss(+planar) and the
   * ncp snap component. 0 means "not set" and reads as 1.0 — envs are
   * zero-initialized all over. */
  double ws_dist, ws_straight, ws_snap;
} tr_env;

/* 0 = unset (1.0); negative = literal zero (a schedule pass with the
 * term fully off) */
static inline double tr_ws(double v) { return v == 0.0 ? 1.0 : (v < 0.0 ? 0.0 : v); }
static inline int tr_geom_terms(void) { /* R3D_GEOM_TERMS=0: A/B the
    * anti-fold + planarity residuals */
  static _Atomic int on = -1;
  int v = atomic_load_explicit(&on, memory_order_relaxed);
  if (v < 0) {
    const char *ev = getenv("R3D_GEOM_TERMS");
    v = ev ? atoi(ev) : 1; /* adaptive planarity on (sparse regions only) */
    atomic_store(&on, v);
  }
  return v;
}

/* segment neighborhood lookup/build; center quantized to 16 px so the
 * full-path scan amortizes across residual evals and LM iterations */
static uint32_t ng_hood_hash(int plane, int slice, int qx, int qy) {
  uint64_t key = ((uint64_t)(unsigned)plane << 60) ^ ((uint64_t)(unsigned)slice << 32) ^
                 ((uint64_t)(unsigned)qx << 16) ^ (uint64_t)(unsigned)qy;
  return (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 44) & 4095u;
}

static void ng_hood_idx_del(tr_env *e, const ng_hood *h) {
  for (uint32_t p = ng_hood_hash(h->plane, h->slice, h->qx, h->qy), n = 0; n < 4096;
       p = (p + 1) & 4095u, n++) {
    uint16_t ent = e->hidx[p];
    if (!ent) return;
    if (&e->hood[ent - 1] == h) {
      e->hidx[p] = 0;
      for (uint32_t p2 = (p + 1) & 4095u; e->hidx[p2]; p2 = (p2 + 1) & 4095u) {
        uint16_t ent2 = e->hidx[p2];
        e->hidx[p2] = 0;
        const ng_hood *h2 = &e->hood[ent2 - 1];
        for (uint32_t p3 = ng_hood_hash(h2->plane, h2->slice, h2->qx, h2->qy);;
             p3 = (p3 + 1) & 4095u)
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
  for (uint32_t p = ng_hood_hash(plane, slice, qx, qy), n = 0; n < 4096;
       p = (p + 1) & 4095u, n++) {
    uint16_t ent = e->hidx[p];
    if (!ent) break;
    ng_hood *h = &e->hood[ent - 1];
    if (h->plane == plane && h->slice == slice && h->qx == qx && h->qy == qy &&
        h->g == g) {
      h->use = ++e->htick;
      atomic_fetch_add_explicit(&ng_hood_hits, 1, memory_order_relaxed);
      return h;
    }
  }
  /* victim: least-recently-used of 8 clock-sampled entries. Bounded cost
   * (a full LRU scan and a staleness clock were both measured slower) and
   * empty/old entries win immediately. */
  uint32_t victim = e->hclock;
  uint64_t vuse = UINT64_MAX;
  for (uint32_t nprobe = 0; nprobe < 8; nprobe++) {
    ng_hood *h = &e->hood[e->hclock];
    uint32_t cur = e->hclock;
    e->hclock = (e->hclock + 1) & (NG_NHOOD - 1);
    if (h->plane < 0) {
      victim = cur;
      break;
    }
    if (h->use < vuse) {
      vuse = h->use;
      victim = cur;
    }
  }
  double t0 = tr_now();
  atomic_fetch_add_explicit(&ng_hood_builds, 1, memory_order_relaxed);
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
  for (uint32_t p = ng_hood_hash(plane, slice, qx, qy);; p = (p + 1) & 4095u)
    if (!e->hidx[p]) {
      e->hidx[p] = (uint16_t)(victim + 1);
      break;
    }
  double cx = ((double)qx + 0.5) * 16.0, cy = ((double)qy + 0.5) * 16.0;
  double R = 92.0; /* query radius 80 + 16-px quantization slack */
  uint32_t paths[NG_QMAX];
  uint32_t np = ng_query(g, cx, cy, R, paths, NG_QMAX);
  if (np == NG_QMAX) atomic_fetch_add(&ng_qmax_binds, 1);
  static _Thread_local float *dec = NULL;
  static _Thread_local uint32_t dec_cap = 0;
  for (uint32_t p = 0; p < np && h->nseg < NG_HOODSEG; p++) {
    /* size the scratch from the record header — xy cross-sections are
     * long spirals and MUST NOT be truncated */
    if ((size_t)paths[p] + 12 > g->blob_n) continue;
    uint32_t want = 1 + ng_be32(g->blob + paths[p] + 8) / 2;
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
      h->smidx[h->nseg] = (float)smx;
      h->smidy[h->nseg] = (float)smy;
      h->snx[h->nseg] = sg[4];
      h->sny[h->nseg] = sg[5];
      float gr = 0.5f * tl + 4.5f; /* NCP_SNAP_TRIG + slack */
      h->sgr2[h->nseg] = gr * gr;
      h->nseg++;
    }
  }
  if (h->nseg == NG_HOODSEG) atomic_fetch_add(&ng_hoodseg_binds, 1);
  { /* occupancy stats: size NG_HOODSEG from data, not guesswork */
    atomic_fetch_add_explicit(&ng_hood_seg_sum, h->nseg, memory_order_relaxed);
    uint64_t pmx = atomic_load_explicit(&ng_hood_seg_max, memory_order_relaxed);
    while (h->nseg > pmx &&
           !atomic_compare_exchange_weak(&ng_hood_seg_max, &pmx, h->nseg)) {}
  }
  tr_tm_add(5, t0);
  return h;
}

/* env-level grid handle cache: refs released by tr_env_flush */
static ng_grid *ng_eget(tr_env *e, int plane, int slice, int *slot_out) {
  slice = ng_snap(e->ngv, slice); /* sparse store: snap to published */
  if (plane == e->gl_plane && slice == e->gl_slice) {
    *slot_out = e->gl_slot;
    return e->gl_g;
  }
  uint32_t h = (((uint32_t)plane << 30) ^ (uint32_t)slice * 2654435761u) >> 24;
  uint32_t victim = h & 255u;
  uint32_t vuse = UINT32_MAX;
  for (uint32_t pr = 0; pr < 8; pr++) {
    uint32_t k = (h + pr) & 255u;
    if (!e->gc[k].used) { /* empty before any match: not cached */
      victim = k;
      vuse = 0;
      break;
    }
    if (e->gc[k].plane == plane && e->gc[k].slice == slice) {
      e->gc[k].use = ++e->gtick;
      *slot_out = e->gc[k].slot;
      e->gl_plane = plane;
      e->gl_slice = slice;
      e->gl_slot = e->gc[k].slot;
      e->gl_g = e->gc[k].g;
      atomic_fetch_add_explicit(&ng_eget_hit, 1, memory_order_relaxed);
      return e->gc[k].g;
    }
    if (e->gc[k].use < vuse) {
      vuse = e->gc[k].use;
      victim = k;
    }
  }
  atomic_fetch_add_explicit(&ng_eget_miss, 1, memory_order_relaxed);
  int slot;
  ng_grid *g = ng_get(e->ngv, plane, slice, &slot);
  if (e->gc[victim].used) {
    ng_put(e->ngv, e->gc[victim].slot, e->gc[victim].g);
    if (e->gl_plane == e->gc[victim].plane && e->gl_slice == e->gc[victim].slice)
      e->gl_plane = -9; /* memo died with its ref */
  }
  e->gc[victim].plane = plane;
  e->gc[victim].slice = slice;
  e->gc[victim].slot = slot;
  e->gc[victim].g = g; /* NULL cached too */
  e->gc[victim].use = ++e->gtick;
  e->gc[victim].used = 1;
  e->gl_plane = plane;
  e->gl_slice = slice;
  e->gl_slot = slot;
  e->gl_g = g;
  *slot_out = slot;
  return g;
}

static void tr_env_flush(tr_env *e) {
  e->gl_plane = -9; /* refs release below: the memo must not outlive them */
  e->gl_g = NULL;
  for (uint32_t i = 0; i < 256; i++)
    if (e->gc[i].used) {
      ng_put(e->ngv, e->gc[i].slot, e->gc[i].g);
      e->gc[i].used = 0;
    }
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
  return (NCP_W_NORMAL * normal_loss + NCP_W_SNAP * tr_ws(e->ws_snap) * snap_loss) *
         aw;
}

/* Fused residual + forward-difference gradient: the base evaluation and
 * its three +h probes differ by half a voxel — same slice, same hood,
 * near-identical chords — so all four run through ONE pass over the
 * segment neighborhood with 4-wide math instead of four passes with
 * four hood lookups. out[0]=r(x), out[1..3]=r(x + h e_k). */
/* G2/G3 (vc3d parity): the finite difference must differentiate ONLY what
 * vc3d's autodiff differentiates. vc3d freezes every configuration-
 * selecting quantity on the scalar part (.val()): the straddle decision,
 * the corner Bn, the slice, the angular weight, ROI membership, the 1/d^2
 * distance weights, and the snap-target identity. Differentiating through
 * those (as the legacy path below does) adds a translational pull toward
 * whichever nearby polyline is best aligned — a direct wrap-jump force —
 * plus a quad-flattening torque through aw and step discontinuities
 * where ROI/snap identities flip between probes. And where the grids have
 * no coverage at all, the residual must be exactly 0, not a constant
 * orientation penalty. R3D_NCP_LEGACY=1 restores the old behavior. */
static void ncp_residual4_legacy(tr_env *e, int plane, const double Qr[4][3],
                                 int fxr, double h, double out[4]);

static void ncp_residual4(tr_env *e, int plane, const double Qr[4][3], int fxr,
                          double h, double out[4]) {
  static _Atomic int legacy = -1;
  int lg = atomic_load_explicit(&legacy, memory_order_relaxed);
  if (lg < 0) {
    lg = getenv("R3D_NCP_LEGACY") ? 1 : 0;
    atomic_store(&legacy, lg);
  }
  if (lg) {
    ncp_residual4_legacy(e, plane, Qr, fxr, h, out);
    return;
  }
  int axis = 2 - plane;
  for (int v = 0; v < 4; v++) out[v] = 0.0;
  /* base-pose configuration (variant 0 = unperturbed) */
  const double *A0 = Qr[0], *B10 = Qr[1], *B20 = Qr[2], *C0 = Qr[3];
  double b1r = B10[axis] - A0[axis], b2r = B20[axis] - A0[axis],
         cr0 = C0[axis] - A0[axis];
  if ((b1r > 0 && b2r > 0 && cr0 > 0) || (b1r < 0 && b2r < 0 && cr0 < 0)) return;
  int bn_idx = 0; /* frozen straddle corner: 1=B1, 2=B2 */
  if (fabs(cr0) < 1e-9) {
    if (fabs(b1r) > 1e-9) bn_idx = 1;
    else if (fabs(b2r) > 1e-9) bn_idx = 2;
  } else {
    if (b1r * cr0 <= 0.0) bn_idx = 1;
    else if (b2r * cr0 <= 0.0) bn_idx = 2;
  }
  if (!bn_idx) return;
  double aw_base;
  { /* frozen angular weight from the base quad */
    const double *Bn0 = bn_idx == 1 ? B10 : B20;
    double v1_[3], v2_[3], nq[3];
    for (int a = 0; a < 3; a++) {
      v1_[a] = Bn0[a] - A0[a];
      v2_[a] = Bn0[a] - C0[a];
    }
    nq[0] = v1_[1] * v2_[2] - v1_[2] * v2_[1];
    nq[1] = v1_[2] * v2_[0] - v1_[0] * v2_[2];
    nq[2] = v1_[0] * v2_[1] - v1_[1] * v2_[0];
    double nl2 = nq[0] * nq[0] + nq[1] * nq[1] + nq[2] * nq[2];
    if (nl2 < 1e-18) return;
    double na = nq[axis] / sqrt(nl2);
    aw_base = 0.5 * (1.0 - na * na);
  }
  double aw0 = aw_base;
  int s0 = (int)llround(A0[axis]); /* frozen slice */
  /* per-variant chord geometry with the FROZEN Bn identity */
  bool ok[4];
  float vax[4], vay[4], vex4[4], vey4[4], venx[4], veny[4];
  double m0x = 0, m0y = 0;
  for (int v = 0; v < 4; v++) {
    double V[3];
    memcpy(V, Qr[fxr], sizeof V);
    if (v) V[v - 1] += h;
    const double *cor[4];
    for (int c = 0; c < 4; c++) cor[c] = c == fxr ? V : Qr[c];
    const double *A = cor[0], *Bn = cor[bn_idx], *C = cor[3];
    ok[v] = false;
    double bnr = Bn[axis] - A[axis], cr = C[axis] - A[axis];
    if (fabs(bnr - cr) < 1e-9) continue;
    double t = -cr / (bnr - cr);
    double E[3];
    for (int a = 0; a < 3; a++) E[a] = C[a] + t * (Bn[a] - C[a]);
    double a2[2], e2[2];
    ncp_2d(plane, A, a2);
    ncp_2d(plane, E, e2);
    double ex = e2[0] - a2[0], ey = e2[1] - a2[1];
    double el = sqrt(ex * ex + ey * ey);
    if (el < 1e-9) continue;
    vax[v] = (float)a2[0];
    vay[v] = (float)a2[1];
    vex4[v] = (float)e2[0];
    vey4[v] = (float)e2[1];
    venx[v] = (float)(ey / el);
    veny[v] = (float)(-ex / el);
    if (v == 0) {
      m0x = (a2[0] + e2[0]) * 0.5;
      m0y = (a2[1] + e2[1]) * 0.5;
    }
    ok[v] = true;
  }
  if (!ok[0]) return; /* config comes from the base: no base, no residual */
  int slot;
  ng_grid *g = ng_eget(e, plane, s0, &slot);
  (void)slot;
  if (!g || g->empty) return; /* G3: no coverage -> exactly 0 */
  ng_hood *hd = ng_hood_get(e, g, plane, s0, m0x, m0y);
  if (!hd || !hd->nseg) return; /* G3 */
  float fm0x = (float)m0x, fm0y = (float)m0y;
  float wsum = 0.0f; /* frozen: identical for every variant */
  float nsum[4] = {0, 0, 0, 0};
  float naccf = 0.0f;
  const float *bsg = NULL; /* frozen snap target */
  int bdsn = 0;
  float bscore = -1.0f;
  /* invalid variants keep venx/veny = 0 so the unconditional 4-lane
   * accumulation below stays garbage-free; their nsum lanes are unused */
  for (int v = 0; v < 4; v++)
    if (!ok[v]) venx[v] = veny[v] = 0.0f;
  const uint32_t ns = hd->nseg;
  const float *restrict midx = hd->smidx, *restrict midy = hd->smidy;
  const float *restrict snx = hd->snx, *restrict sny = hd->sny;
  /* pass 1 — branchless SoA sweep: ROI weight + normal alignment.
   * (The AoS record loop with branches would not vectorize; this pass
   * was the tracer's single hottest loop.) */
  for (uint32_t k = 0; k < ns; k++) {
    float dmx = midx[k] - fm0x, dmy = midy[k] - fm0y;
    float d2 = dmx * dmx + dmy * dmy; /* BASE distance: frozen ROI + weight */
    float in = d2 <= (float)NCP_ROI2 ? 1.0f : 0.0f;
    float dd = d2 < 10.0f ? 10.0f : d2;
    float w = in / dd;
    naccf += in;
    wsum += w;
    float d0 = fabsf(venx[0] * snx[k] + veny[0] * sny[k]);
    float d1 = fabsf(venx[1] * snx[k] + veny[1] * sny[k]);
    float d2v = fabsf(venx[2] * snx[k] + veny[2] * sny[k]);
    float d3 = fabsf(venx[3] * snx[k] + veny[3] * sny[k]);
    nsum[0] += (1.0f - d0) * w;
    nsum[1] += (1.0f - d1) * w;
    nsum[2] += (1.0f - d2v) * w;
    nsum[3] += (1.0f - d3) * w;
  }
  uint32_t nacc = (uint32_t)naccf;
  /* pass 2 — snap-target selection from the BASE pose only, gated by
   * midpoint distance: a segment farther than halflen+trig from the
   * chord end cannot pass the 4-px trigger, and that excludes nearly
   * every segment */
  const float ex0 = vex4[0], ey0 = vey4[0];
  const float *restrict gr2 = hd->sgr2;
  for (uint32_t k = 0; k < ns; k++) {
    float gdx = midx[k] - ex0, gdy = midy[k] - ey0;
    if (gdx * gdx + gdy * gdy > gr2[k]) continue;
    const float *sg = hd->sg[k];
    float tt = ((ex0 - sg[0]) * sg[10] + (ey0 - sg[1]) * sg[11]) * sg[12];
    tt = tt < 0.0f ? 0.0f : (tt > 1.0f ? 1.0f : tt);
    float qex = sg[0] + tt * sg[10] - ex0, qey = sg[1] + tt * sg[11] - ey0;
    float d2e = qex * qex + qey * qey;
    if (d2e < (float)(NCP_SNAP_TRIG * NCP_SNAP_TRIG)) {
      for (int dsn = 0; dsn < 2; dsn++) {
        if (!(hd->nb[k] & (dsn ? 2u : 1u))) continue;
        double qax = dsn ? (double)sg[2] : (double)sg[6];
        double qay = dsn ? (double)sg[3] : (double)sg[7];
        double qbx = dsn ? (double)sg[8] : (double)sg[0];
        double qby = dsn ? (double)sg[9] : (double)sg[1];
        double d2a = ncp_pt_seg_d2((double)vax[0], (double)vay[0], qax, qay, qbx, qby);
        if (d2a < NCP_SNAP_RANGE * NCP_SNAP_RANGE) {
          double d1n = sqrt(d2a) / NCP_SNAP_RANGE,
                 d2n = sqrt((double)d2e) / NCP_SNAP_TRIG;
          float score = (float)(0.5 * (d1n + d2n));
          if (bscore < 0.0f || score < bscore) {
            bscore = score;
            bsg = sg;
            bdsn = dsn;
          }
        }
      }
    }
  }
  if (!nacc) return; /* G3: nothing in the ROI -> no evidence, no penalty */
  for (int v = 0; v < 4; v++) {
    if (!ok[v]) continue;
    double normal_loss = wsum > 1e-9f ? (double)nsum[v] / (double)wsum : 0.0;
    double snap_loss = 1.0;
    if (bsg) { /* distances to the FROZEN segment carry the derivative */
      const float *sg = bsg;
      float tt = ((vex4[v] - sg[0]) * sg[10] + (vey4[v] - sg[1]) * sg[11]) * sg[12];
      tt = tt < 0.0f ? 0.0f : (tt > 1.0f ? 1.0f : tt);
      float qex = sg[0] + tt * sg[10] - vex4[v], qey = sg[1] + tt * sg[11] - vey4[v];
      double d2e = (double)(qex * qex + qey * qey);
      double qax = bdsn ? (double)sg[2] : (double)sg[6];
      double qay = bdsn ? (double)sg[3] : (double)sg[7];
      double qbx = bdsn ? (double)sg[8] : (double)sg[0];
      double qby = bdsn ? (double)sg[9] : (double)sg[1];
      double d2a = ncp_pt_seg_d2((double)vax[v], (double)vay[v], qax, qay, qbx, qby);
      double d1n = sqrt(d2a) / NCP_SNAP_RANGE, d2n = sqrt(d2e) / NCP_SNAP_TRIG;
      snap_loss = d1n * (1.0 - d2n) + d2n;
    }
    out[v] = (NCP_W_NORMAL * normal_loss +
              NCP_W_SNAP * tr_ws(e->ws_snap) * snap_loss) * aw0;
  }
}

static void ncp_residual4_legacy(tr_env *e, int plane, const double Qr[4][3],
                                 int fxr, double h, double out[4]) {
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
    out[v] = (NCP_W_NORMAL * normal_loss +
              NCP_W_SNAP * tr_ws(e->ws_snap) * snap_loss) * aw[v];
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
typedef struct tr_wf tr_wf; /* winding-potential field (defined below) */
static bool tr_wf_at(const tr_wf *w, double x, double y, double *val, double g2[2]);
static double tr_om_eff(const r3d_tracer *t);
static inline bool tr_valid(const r3d_tracer *t, int i, int j);

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
  /* joint omega only when clearly identifiable: 2.5+ windings of span.
   * Between 1 and 2.5 the column is near-collinear with r0 and the
   * "fit" happily explains radial sheet-hopping as a huge omega
   * (rib4: omega 2302 at span 1.13) — measured omega stays in charge. */
  bool joint = wmax - wmin >= 2.5;
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
  t->sp_valid = have && fabs(best_om) >= 2.0 && fabs(best_om) <= 120.0 &&
                best_rms <= 0.5 * fabs(best_om);
  pthread_mutex_unlock(&t->mu);
  free(A);
  free(x);
  free(r0b);
  free(best_r0);
}

/* after a fit: distrust cells the spiral model calls wrong-wrap */
static void tr_spiral_flag(r3d_tracer *t) {
  const tr_wf *wfp = t->wf;
  double ome = tr_om_eff(t);
  if (wfp && ome > 0) {
    uint64_t N2 = (uint64_t)t->W * t->H;
    for (uint64_t k = 0; k < N2; k++) {
      if (t->state[k] != R3D_TR_SET) continue;
      double v;
      if (!tr_wf_at(wfp, t->pos[k * 3], t->pos[k * 3 + 1], &v, NULL)) continue;
      double r = v - t->wf_base - ome * (double)t->wind[k];
      double thr = 0.5 * ome;
      if (2.5 * t->sp_rms > thr && t->sp_valid) thr = 2.5 * t->sp_rms;
      if (fabs(r) > thr && t->conf[k] > 0.25f && !(t->dsup && t->dsup[k] >= 2))
        t->conf[k] = 0.25f;
    }
    return;
  }
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
    double thr = 0.5 * fabs(atomic_load(&t->sp_omega));
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
                          double *wnd, double uv[2]) {
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
            if (uv) { /* donor-grid uv of the closest point (G12) */
              if (tri == 0) {
                uv[0] = (double)i + ub;
                uv[1] = (double)j + vc2;
              } else {
                uv[0] = (double)i + 1.0 - ub;
                uv[1] = (double)j + 1.0 - vc2;
              }
            }
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
    { /* donor tiering: a `defective` marker file excludes the segment
       * from fusion without deleting it (vc3d approved/defective tiers) */
      char mp[1200];
      snprintf(mp, sizeof mp, "%s/defective", dirs[i]);
      FILE *f = fopen(mp, "r");
      if (f) {
        fclose(f);
        printf("tracer: donor %s marked defective (skipping)\n", dirs[i]);
        continue;
      }
    }
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

/* bilinear donor-surface sample at fractional donor-grid uv; false when
 * any corner is invalid or uv leaves the grid */
static bool tr_don_bilerp(const tr_dons *dn, int di, const double uv[2],
                          double out[3]) {
  if (!dn || di < 0 || di >= (int)dn->n) return false;
  const tr_donor *d = &dn->d[di];
  if (uv[0] < 0.0 || uv[1] < 0.0 || uv[0] > (double)d->s.w - 1.001 ||
      uv[1] > (double)d->s.h - 1.001)
    return false;
  uint32_t i = (uint32_t)uv[0], j = (uint32_t)uv[1];
  double fu = uv[0] - i, fv = uv[1] - j;
  const float *p00 = r3d_tifxyz_at(&d->s, i, j);
  const float *p10 = r3d_tifxyz_at(&d->s, i + 1, j);
  const float *p01 = r3d_tifxyz_at(&d->s, i, j + 1);
  const float *p11 = r3d_tifxyz_at(&d->s, i + 1, j + 1);
  if (!r3d_tifxyz_valid(p00) || !r3d_tifxyz_valid(p10) || !r3d_tifxyz_valid(p01) ||
      !r3d_tifxyz_valid(p11))
    return false;
  for (int a = 0; a < 3; a++)
    out[a] = ((double)p00[a] * (1 - fu) + (double)p10[a] * fu) * (1 - fv) +
             ((double)p01[a] * (1 - fu) + (double)p11[a] * fu) * fv;
  return true;
}

/* donor uv membership refresh (G12, generation boundary): nearest donor
 * id + uv per SET cell, then a discontinuity pass — where a patch folds
 * back or two of its wraps pass close, nearest-point matching flips fold
 * mid-neighbourhood and the uv map jumps; those cells are marked -2 and
 * the donor pull/adoption skip them (the winding gate cannot catch a
 * same-patch fold: both folds carry similar winding). */
static void tr_don_members(r3d_tracer *t) {
  if (!t->don || !t->dcell_id || !t->dcell_uv) return;
  int W = (int)t->W, H = (int)t->H;
  for (int j = 0; j < H; j++)
    for (int i = 0; i < W; i++) {
      size_t k = (size_t)j * (size_t)W + (size_t)i;
      t->dcell_id[k] = -1;
      if (t->state[k] != R3D_TR_SET) continue;
      double q[3], dwnd, uv[2];
      int di = tr_don_closest(t->don, t->pos + k * 3, TR_FUS_PULL, q, &dwnd, uv);
      if (di < 0 || di > 63) continue;
      t->dcell_id[k] = (int8_t)di;
      t->dcell_uv[k * 2] = (float)uv[0];
      t->dcell_uv[k * 2 + 1] = (float)uv[1];
    }
  uint32_t nfold = 0;
  for (int j = 0; j < H; j++)
    for (int i = 0; i < W; i++) {
      size_t k = (size_t)j * (size_t)W + (size_t)i;
      if (t->dcell_id[k] < 0) continue;
      static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (int o = 0; o < 4; o++) {
        int ii = i + o4[o][0], jj = j + o4[o][1];
        if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue;
        size_t k2 = (size_t)jj * (size_t)W + (size_t)ii;
        if (t->dcell_id[k2] != t->dcell_id[k] || t->dcell_id[k2] < 0) continue;
        float du = t->dcell_uv[k * 2] - t->dcell_uv[k2 * 2];
        float dv = t->dcell_uv[k * 2 + 1] - t->dcell_uv[k2 * 2 + 1];
        if (du * du + dv * dv > 16.0f) { /* > 4 donor cells per step: the
                                          * membership jumped a fold */
          t->dcell_id[k] = -2;
          nfold++;
          break;
        }
      }
    }
  (void)nfold; /* vetoed silently; visible via donor QC coverage */
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
  double gaps[512], gpos[512][3];
  int ngap = 0;
  uint64_t N = (uint64_t)t->W * t->H;
  for (uint64_t k = 0; k < N && ngap < 512; k += 97) { /* coprime stride */
    if (t->state[k] != R3D_TR_SET) continue;
    const double *P = t->pos + k * 3;
    /* gap ray along the LOCAL GRID NORMAL (G5b): the umbilicus radial
     * overestimates by 1/cos on pancaked sections where the sheet is not
     * perpendicular to the radial */
    double ux, uy, uz = 0.0;
    int i = (int)(k % t->W), j = (int)(k / t->W);
    bool have_n = false;
    if (tr_valid(t, i - 1, j) && tr_valid(t, i + 1, j) && tr_valid(t, i, j - 1) &&
        tr_valid(t, i, j + 1)) {
      const double *pu0 = t->pos + ((size_t)j * t->W + (size_t)i - 1) * 3;
      const double *pu1 = t->pos + ((size_t)j * t->W + (size_t)i + 1) * 3;
      const double *pv0 = t->pos + ((size_t)(j - 1) * t->W + (size_t)i) * 3;
      const double *pv1 = t->pos + ((size_t)(j + 1) * t->W + (size_t)i) * 3;
      double eu[3], ev[3], n[3];
      for (int a = 0; a < 3; a++) {
        eu[a] = pu1[a] - pu0[a];
        ev[a] = pv1[a] - pv0[a];
      }
      n[0] = eu[1] * ev[2] - eu[2] * ev[1];
      n[1] = eu[2] * ev[0] - eu[0] * ev[2];
      n[2] = eu[0] * ev[1] - eu[1] * ev[0];
      double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (l > 1e-9) {
        ux = n[0] / l;
        uy = n[1] / l;
        uz = n[2] / l;
        have_n = true;
      }
    }
    if (!have_n) { /* fallback: the radial */
      double cx, cy;
      if (!tr_uc_at(t, P[2], &cx, &cy, NULL, NULL)) continue;
      ux = P[0] - cx;
      uy = P[1] - cy;
      double rho = hypot(ux, uy);
      if (rho < 8.0) continue;
      ux /= rho;
      uy /= rho;
      uz = 0.0;
    }
    double best = 0.0;
    for (int dir = -1; dir <= 1; dir += 2) {
      double m = 0.0;
      for (int s = 1; s <= 64; s++) {
        double q[3] = {P[0] + dir * s * ux, P[1] + dir * s * uy,
                       P[2] + dir * s * uz};
        double d = td_tri(dt, q, NULL);
        if (d > m) m = d;
        if (d < 1.5 && m > 3.0) { /* left the sheet, hit the next one */
          if (best == 0.0 || (double)s < best) best = (double)s;
          break;
        }
      }
    }
    if (best >= 4.0 && best <= 60.0) {
      memcpy(gpos[ngap], P, sizeof gpos[ngap]);
      gaps[ngap++] = best;
    }
  }
  if (ngap < 32) return 0.0;
  { /* coarse gap field (G5a): 64-vox cells over the sample bbox,
     * cell-median filled, nearest-filled, one 3^3 box smooth */
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int g = 0; g < ngap; g++)
      for (int a = 0; a < 3; a++) {
        if (gpos[g][a] < lo[a]) lo[a] = gpos[g][a];
        if (gpos[g][a] > hi[a]) hi[a] = gpos[g][a];
      }
    double cs = 64.0;
    uint32_t nn[3];
    for (int a = 0; a < 3; a++) {
      nn[a] = (uint32_t)((hi[a] - lo[a]) / cs) + 1;
      if (nn[a] > 64) nn[a] = 64;
    }
    size_t nc = (size_t)nn[0] * nn[1] * nn[2];
    float *acc = calloc(nc, sizeof *acc);
    uint16_t *cnt = calloc(nc, sizeof *cnt);
    float *fld = malloc(nc * sizeof *fld);
    if (acc && cnt && fld) {
      for (int g = 0; g < ngap; g++) {
        uint32_t ci[3];
        for (int a = 0; a < 3; a++) {
          long v = (long)((gpos[g][a] - lo[a]) / cs);
          ci[a] = v < 0 ? 0 : (v >= (long)nn[a] ? nn[a] - 1 : (uint32_t)v);
        }
        size_t idx = ((size_t)ci[2] * nn[1] + ci[1]) * nn[0] + ci[0];
        acc[idx] += (float)gaps[g];
        cnt[idx]++;
      }
      for (size_t c = 0; c < nc; c++) fld[c] = cnt[c] ? acc[c] / (float)cnt[c] : 0.0f;
      for (int pass = 0; pass < 4; pass++) { /* nearest-fill by dilation */
        bool any = false;
        for (uint32_t z = 0; z < nn[2]; z++)
          for (uint32_t y = 0; y < nn[1]; y++)
            for (uint32_t x2 = 0; x2 < nn[0]; x2++) {
              size_t idx = ((size_t)z * nn[1] + y) * nn[0] + x2;
              if (fld[idx] > 0.0f) continue;
              float s2 = 0.0f;
              int n2 = 0;
              for (int a = 0; a < 6; a++) {
                static const int o6[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                             {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
                long qx = (long)x2 + o6[a][0], qy = (long)y + o6[a][1],
                     qz = (long)z + o6[a][2];
                if (qx < 0 || qy < 0 || qz < 0 || qx >= (long)nn[0] ||
                    qy >= (long)nn[1] || qz >= (long)nn[2])
                  continue;
                float v = fld[((size_t)qz * nn[1] + (size_t)qy) * nn[0] + (size_t)qx];
                if (v > 0.0f) {
                  s2 += v;
                  n2++;
                }
              }
              if (n2) {
                acc[idx] = s2 / (float)n2; /* stage into acc to keep the
                                            * dilation isotropic */
                any = true;
              } else {
                acc[idx] = 0.0f;
              }
            }
        for (size_t c = 0; c < nc; c++)
          if (fld[c] <= 0.0f && acc[c] > 0.0f) fld[c] = acc[c];
        if (!any) break;
      }
      pthread_mutex_lock(&t->mu);
      free(t->omf);
      t->omf = fld;
      memcpy(t->omf_o, lo, sizeof lo);
      t->omf_cs = cs;
      memcpy(t->omf_n, nn, sizeof nn);
      pthread_mutex_unlock(&t->mu);
      fld = NULL;
    }
    free(acc);
    free(cnt);
    free(fld);
  }
  qsort(gaps, (size_t)ngap, sizeof *gaps, tr_dcmp);
  return gaps[ngap / 2];
}

/* position-dependent sheet gap: field value when covered, global scalar
 * otherwise */
static double tr_om_at(const r3d_tracer *t, const double p[3]) {
  if (t->omf) {
    long ci[3];
    bool in = true;
    for (int a = 0; a < 3; a++) {
      ci[a] = (long)((p[a] - t->omf_o[a]) / t->omf_cs);
      if (ci[a] < 0 || ci[a] >= (long)t->omf_n[a]) in = false;
    }
    if (in) {
      float v = t->omf[((size_t)ci[2] * t->omf_n[1] + (size_t)ci[1]) * t->omf_n[0] +
                       (size_t)ci[0]];
      if (v > 0.0f) return (double)v;
    }
  }
  return tr_om_eff(t);
}

/* ==================== self-overlap index + hinge ====================
 * Lasagna's min_dist idea in LM form: a cell may not come closer than
 * ~half a sheet gap to any part of the SAME grown surface on a DIFFERENT
 * winding. One-sided hinge — it separates wraps and preserves their
 * ordering without auxiliary variables. */
#define TR_W_SELF 1.0
#define TR_SELF_DW 0.5 /* windings apart to count as "another wrap" */

typedef struct tr_sfx {
  double cs;
  uint32_t nbuck; /* pow2 */
  uint32_t *head, *next, *cell;
  uint32_t nent, cap;
} tr_sfx;

static uint32_t tr_sfx_h(const tr_sfx *x, long cx, long cy, long cz) {
  uint64_t h = (uint64_t)cx * 0x9E3779B185EBCA87ull ^
               (uint64_t)cy * 0xC2B2AE3D27D4EB4Full ^ (uint64_t)cz * 0x165667B19E3779F9ull;
  return (uint32_t)(h >> 32) & (x->nbuck - 1);
}

static void tr_sfx_free(tr_sfx *x) {
  if (!x) return;
  free(x->head);
  free(x->next);
  free(x->cell);
  free(x);
}

/* rebuild from all SET cells (coordinator, pool idle) */
static void tr_sfx_build(r3d_tracer *t) {
  if (tr_om_eff(t) <= 0) return;
  tr_sfx *old = t->sfx;
  uint64_t N = (uint64_t)t->W * t->H;
  tr_sfx *x = calloc(1, sizeof *x);
  if (!x) return;
  x->cs = 2.0 * tr_om_eff(t);
  if (x->cs < 8.0) x->cs = 8.0;
  uint32_t nb = 1;
  while (nb < t->nset / 2 + 64) nb <<= 1;
  if (nb > 1u << 21) nb = 1u << 21;
  x->nbuck = nb;
  x->head = calloc(nb, sizeof *x->head);
  x->cap = t->nset + 16;
  x->next = malloc((size_t)x->cap * sizeof *x->next);
  x->cell = malloc((size_t)x->cap * sizeof *x->cell);
  if (!x->head || !x->next || !x->cell) {
    tr_sfx_free(x);
    return;
  }
  for (uint64_t k = 0; k < N && x->nent < x->cap; k++) {
    if (t->state[k] != R3D_TR_SET) continue;
    const double *P = t->pos + k * 3;
    uint32_t b = tr_sfx_h(x, (long)floor(P[0] / x->cs), (long)floor(P[1] / x->cs),
                          (long)floor(P[2] / x->cs));
    x->cell[x->nent] = (uint32_t)k;
    x->next[x->nent] = x->head[b];
    x->head[b] = x->nent + 1;
    x->nent++;
  }
  t->sfx = x;
  tr_sfx_free(old);
}

/* hinge residual against the nearest other-wrap cell; free point is x */
/* unit surface normal at a grid cell from central differences; false when
 * the 4-neighbourhood is incomplete or degenerate */
static bool tr_cell_normal(const r3d_tracer *t, int i, int j, double n[3]) {
  if (!tr_valid(t, i - 1, j) || !tr_valid(t, i + 1, j) || !tr_valid(t, i, j - 1) ||
      !tr_valid(t, i, j + 1))
    return false;
  const double *pu0 = t->pos + ((size_t)j * t->W + (size_t)i - 1) * 3;
  const double *pu1 = t->pos + ((size_t)j * t->W + (size_t)i + 1) * 3;
  const double *pv0 = t->pos + ((size_t)(j - 1) * t->W + (size_t)i) * 3;
  const double *pv1 = t->pos + ((size_t)(j + 1) * t->W + (size_t)i) * 3;
  double eu[3], ev[3];
  for (int a = 0; a < 3; a++) {
    eu[a] = pu1[a] - pu0[a];
    ev[a] = pv1[a] - pv0[a];
  }
  n[0] = eu[1] * ev[2] - eu[2] * ev[1];
  n[1] = eu[2] * ev[0] - eu[0] * ev[2];
  n[2] = eu[0] * ev[1] - eu[1] * ev[0];
  double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (l < 1e-9) return false;
  for (int a = 0; a < 3; a++) n[a] /= l;
  return true;
}

static void tr_res_self(tr_nlsq *acc, const r3d_tracer *t, const double x[3],
                        size_t self_k) {
  const tr_sfx *sx = t->sfx;
  if (!sx || !sx->nent) return;
  double rmin = 0.55 * tr_om_at(t, x); /* per-region gap (G5a) */
  double wself = (double)t->wind[self_k];
  long c0[3], c1[3];
  for (int a = 0; a < 3; a++) {
    c0[a] = (long)floor((x[a] - rmin) / sx->cs);
    c1[a] = (long)floor((x[a] + rmin) / sx->cs);
  }
  double bd2 = rmin * rmin;
  const double *bq = NULL;
  uint32_t bk_hinge = 0;
  for (long cz = c0[2]; cz <= c1[2]; cz++)
    for (long cy = c0[1]; cy <= c1[1]; cy++)
      for (long cx = c0[0]; cx <= c1[0]; cx++)
        for (uint32_t e = sx->head[tr_sfx_h(sx, cx, cy, cz)]; e; e = sx->next[e - 1]) {
          uint32_t k = sx->cell[e - 1];
          if (k == self_k || t->state[k] != R3D_TR_SET) continue;
          { /* grid-distance gate (was winding-gated, which silently
             * DISABLED this hinge in every umbilicus-less session — all
             * windings read 0 and nothing ever counted as another wrap).
             * Grid-close cells are the same local sheet patch; anything
             * grid-far that comes this close in 3D is an overlap — a
             * neighbouring wrap OR the sheet doubling back on itself,
             * which the winding gate could never see. */
            int di2 = abs((int)(k % t->W) - (int)(self_k % t->W));
            int dj2 = abs((int)(k / t->W) - (int)(self_k / t->W));
            if (di2 <= 8 && dj2 <= 8) continue;
          }
          (void)wself;
          const double *Q = t->pos + (size_t)k * 3;
          double d2 = 0;
          for (int a = 0; a < 3; a++) {
            double dd = x[a] - Q[a];
            d2 += dd * dd;
          }
          if (d2 < bd2) {
            bd2 = d2;
            bq = Q;
            bk_hinge = k;
          }
        }
  if (bq) { /* too close to another part of the surface: decide by the
             * NORMALS whether this is a fold-back (opposing) before
             * pushing - a legitimately tight curl brings same-wrap cells
             * near each other at moderate angles and must not be pried
             * open (the taco failure). Opposing normals within half a
             * sheet gap can only be a flap folded back onto the sheet;
             * near-parallel normals at large grid distance are a wrap
             * interpenetration. The ambiguous middle band is left to the
             * data and fold-hinge terms. */
    double n1[3], n2[3];
    bool ok1 = tr_cell_normal(t, (int)(self_k % t->W), (int)(self_k / t->W), n1);
    bool ok2 = tr_cell_normal(t, (int)(bk_hinge % t->W), (int)(bk_hinge / t->W), n2);
    double ndot = ok1 && ok2 ? n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] : 0.0;
    int gdi = abs((int)(bk_hinge % t->W) - (int)(self_k % t->W));
    int gdj = abs((int)(bk_hinge / t->W) - (int)(self_k / t->W));
    int gd = gdi > gdj ? gdi : gdj;
    bool fold_back = ok1 && ok2 && ndot < -0.3;
    bool interpen = ok1 && ok2 && ndot > 0.3 && gd > 20;
    if (fold_back || interpen || (!ok1 || !ok2)) {
      double d = sqrt(bd2);
      if (d > 1e-6) {
        double r = TR_W_SELF * (rmin - d) / rmin;
        double J[3];
        for (int a = 0; a < 3; a++) J[a] = -TR_W_SELF * (x[a] - bq[a]) / (d * rmin);
        nq_add(acc, r, J);
      }
    }
  }
  /* two-sided wrap spacing (geometry-agnostic, unlike the global rho
   * model): the nearest cell exactly one winding away should sit ~omega
   * from here. Weak Cauchy pull — real gaps vary. */
  double om = tr_om_at(t, x);
  double rad2 = 1.6 * om;
  long d0[3], d1[3];
  for (int a = 0; a < 3; a++) {
    d0[a] = (long)floor((x[a] - rad2) / sx->cs);
    d1[a] = (long)floor((x[a] + rad2) / sx->cs);
  }
  double nd2 = rad2 * rad2;
  const double *nq = NULL;
  for (long cz = d0[2]; cz <= d1[2]; cz++)
    for (long cy = d0[1]; cy <= d1[1]; cy++)
      for (long cx = d0[0]; cx <= d1[0]; cx++)
        for (uint32_t e = sx->head[tr_sfx_h(sx, cx, cy, cz)]; e; e = sx->next[e - 1]) {
          uint32_t k = sx->cell[e - 1];
          if (k == self_k || t->state[k] != R3D_TR_SET) continue;
          double dw = fabs((double)t->wind[k] - wself);
          if (dw < 0.6 || dw > 1.6) continue;
          const double *Q = t->pos + (size_t)k * 3;
          double d2 = 0;
          for (int a = 0; a < 3; a++) {
            double dd = x[a] - Q[a];
            d2 += dd * dd;
          }
          if (d2 < nd2) {
            nd2 = d2;
            nq = Q;
          }
        }
  if (nq) {
    double d = sqrt(nd2);
    if (d > 1e-6) {
      /* signed spacing (G5c, vc3d spiral_ceres): unsigned |d| holds a
       * cell that slipped PAST its inner neighbour at perfect spacing on
       * the wrong side — the exact failure this term exists to catch.
       * Sign the distance by radial order: outward reference x the
       * winding order; when inverted the residual is ~-2 instead of 0,
       * and the Cauchy robustifier is skipped so the repair force is not
       * robustified away. R3D_SIGNED_SPACING=0 restores unsigned. */
      static _Atomic int signed_on = -1;
      int sg = atomic_load_explicit(&signed_on, memory_order_relaxed);
      if (sg < 0) {
        const char *ev = getenv("R3D_SIGNED_SPACING");
        sg = ev ? atoi(ev) : 1;
        atomic_store(&signed_on, sg);
      }
      double s = 1.0;
      if (sg && t->uc) {
        double cx3, cy3;
        if (tr_uc_at(t, x[2], &cx3, &cy3, NULL, NULL)) {
          double rx = x[0] - cx3, ry = x[1] - cy3;
          double rl = hypot(rx, ry);
          if (rl > 1e-6) {
            /* winding grows inward or outward consistently: the sign of
             * (x-Q).radial must match the sign of (w_self - w_nb) times
             * the spiral's handedness (om_eff sign carries it via the
             * fit; use the observed order: higher winding = smaller rho
             * for an inward spiral — take the fit's slope sign) */
            double dot = ((x[0] - nq[0]) * rx + (x[1] - nq[1]) * ry) / rl;
            double worder = (double)t->wind[self_k] - (double)t->wind[
                (size_t)(nq - t->pos) / 3];
            double hand = t->sp_valid && t->sp_omega < 0 ? -1.0 : 1.0;
            s = (dot > 0 ? 1.0 : -1.0) * (worder > 0 ? 1.0 : -1.0) * hand;
          }
        }
      }
      double w2 = 0.3;
      double r = w2 * (s * d - om) / om;
      double sc = s < 0 ? 1.0 : sqrt(1.0 / (1.0 + r * r)); /* Cauchy(1) */
      double J[3];
      for (int a = 0; a < 3; a++) J[a] = sc * s * w2 * (x[a] - nq[a]) / (d * om);
      nq_add(acc, r * sc, J);
    }
  }
}

/* ================ winding-potential field (evolutor port) ================
 * KhartesViewer/evolutor's core idea, fed by our normal grids instead of
 * a structure tensor: solve one sparse linear system per cross-section
 * slab for a scalar field r1 with  n.grad(r1) = 1 (unit advance per voxel
 * crossed along the sheet normal) and  n x grad(r1) = 0.  r1 / sheet-gap
 * is then a winding coordinate valid on ANY cross-section shape — the
 * pancake scrolls where the Archimedean rho model fails. Normal sign is
 * disambiguated by a preliminary r0 solve (umbilicus-direction signs are
 * wrong on crushed scrolls). Solved matrix-free by CG on the normal
 * equations; seconds per slab. */
#define WF_MAX 1100

struct tr_wf {
  float *r1;
  uint32_t W, H;
  double dec; /* fullres vox per field cell */
  bool ok;
};

typedef struct wf_sys { /* one least-squares stencil system */
  uint32_t W, H;
  const float *nx, *ny, *coh;
  double wdot, wcross, ws1, ws2;
  uint32_t anchor;
} wf_sys;

/* y = A^T (A x): accumulate every residual row's value, scatter back */
static void wf_apply(const wf_sys *S, const double *x, double *y) {
  uint32_t W = S->W, H = S->H;
  memset(y, 0, (size_t)W * H * sizeof *y);
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      size_t k = (size_t)j * W + i;
      if (i + 1 < W && j + 1 < H && S->coh[k] > 0) {
        double gx = x[k + 1] - x[k], gy = x[k + W] - x[k];
        double cnx = (double)S->nx[k], cny = (double)S->ny[k];
        double c = (double)S->coh[k];
        if (S->wdot > 0) { /* r = a*(cnx*gx + cny*gy) */
          double a = S->wdot * c;
          double r = a * (cnx * gx + cny * gy);
          y[k + 1] += r * a * cnx;
          y[k + W] += r * a * cny;
          y[k] -= r * a * (cnx + cny);
        }
        if (S->wcross > 0) { /* r = a*(cnx*gy - cny*gx) */
          double a = S->wcross * c;
          double r = a * (cnx * gy - cny * gx);
          y[k + W] += r * a * cnx;
          y[k + 1] -= r * a * cny;
          y[k] += r * a * (cny - cnx);
        }
      }
      if (S->ws1 > 0) { /* first-difference smoothing rows */
        if (i + 1 < W) {
          double r = S->ws1 * (x[k + 1] - x[k]);
          y[k + 1] += r * S->ws1;
          y[k] -= r * S->ws1;
        }
        if (j + 1 < H) {
          double r = S->ws1 * (x[k + W] - x[k]);
          y[k + W] += r * S->ws1;
          y[k] -= r * S->ws1;
        }
      }
      if (S->ws2 > 0) { /* second-difference smoothing rows */
        if (i > 0 && i + 1 < W) {
          double r = S->ws2 * (x[k + 1] - 2 * x[k] + x[k - 1]);
          y[k + 1] += r * S->ws2;
          y[k - 1] += r * S->ws2;
          y[k] -= 2 * r * S->ws2;
        }
        if (j > 0 && j + 1 < H) {
          double r = S->ws2 * (x[k + W] - 2 * x[k] + x[k - W]);
          y[k + W] += r * S->ws2;
          y[k - W] += r * S->ws2;
          y[k] -= 2 * r * S->ws2;
        }
      }
    }
  y[S->anchor] += x[S->anchor]; /* weight-1 gauge pin */
}

/* y = A^T b for targets: tdot (per-cell dot target), ts1 (per-cell pair
 * targets for first-difference smoothing, from a base field), others 0 */
static void wf_rhs(const wf_sys *S, const double *tdot, const double *base,
                   double *y) {
  uint32_t W = S->W, H = S->H;
  memset(y, 0, (size_t)W * H * sizeof *y);
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      size_t k = (size_t)j * W + i;
      if (tdot && S->wdot > 0 && i + 1 < W && j + 1 < H && S->coh[k] > 0) {
        double a = S->wdot * (double)S->coh[k];
        double br = a * tdot[k];
        y[k + 1] += br * a * (double)S->nx[k];
        y[k + W] += br * a * (double)S->ny[k];
        y[k] -= br * a * ((double)S->nx[k] + (double)S->ny[k]);
      }
      if (base && S->ws1 > 0) {
        if (i + 1 < W) {
          double br = S->ws1 * (base[k + 1] - base[k]);
          y[k + 1] += br * S->ws1;
          y[k] -= br * S->ws1;
        }
        if (j + 1 < H) {
          double br = S->ws1 * (base[k + W] - base[k]);
          y[k + W] += br * S->ws1;
          y[k] -= br * S->ws1;
        }
      }
    }
}

/* conjugate gradient on the (SPD) normal equations */
static void wf_cg(const wf_sys *S, const double *b, double *x, int iters) {
  size_t N = (size_t)S->W * S->H;
  double *r = malloc(N * sizeof *r), *p = malloc(N * sizeof *p),
         *ap = malloc(N * sizeof *ap);
  if (!r || !p || !ap) {
    free(r);
    free(p);
    free(ap);
    return;
  }
  wf_apply(S, x, ap);
  double rr = 0;
  for (size_t k = 0; k < N; k++) {
    r[k] = b[k] - ap[k];
    p[k] = r[k];
    rr += r[k] * r[k];
  }
  double rr0 = rr > 0 ? rr : 1.0;
  for (int it = 0; it < iters && rr > 1e-8 * rr0; it++) {
    wf_apply(S, p, ap);
    double pap = 0;
    for (size_t k = 0; k < N; k++) pap += p[k] * ap[k];
    if (pap <= 0) break;
    double alpha = rr / pap;
    double nrr = 0;
    for (size_t k = 0; k < N; k++) {
      x[k] += alpha * p[k];
      r[k] -= alpha * ap[k];
      nrr += r[k] * r[k];
    }
    double beta = nrr / rr;
    rr = nrr;
    for (size_t k = 0; k < N; k++) p[k] = r[k] + beta * p[k];
  }
  free(r);
  free(p);
  free(ap);
}

static void tr_wf_free(tr_wf *w) {
  if (!w) return;
  free(w->r1);
  free(w);
}

/* build the field for the seed slab from the xy normal-grid slice */
static tr_wf *tr_wf_build(r3d_tracer *t, ng_vol *ngv, double nx_vox, double ny_vox) {
  if (!ngv || !ngv->active || !t->uc) return NULL;
  int slice = (int)llround(t->cfg.seed[2]);
  int slot;
  ng_grid *g = ng_get(ngv, 0, slice, &slot);
  if (!g || g->empty || !g->blob_n) {
    ng_put(ngv, slot, g);
    return NULL;
  }
  double dec = 8.0;
  while (nx_vox / dec > WF_MAX || ny_vox / dec > WF_MAX) dec *= 2.0;
  uint32_t W = (uint32_t)(nx_vox / dec) + 1, H = (uint32_t)(ny_vox / dec) + 1;
  size_t N = (size_t)W * H;
  double t0 = tr_now();
  float *c2 = calloc(N, sizeof *c2), *s2 = calloc(N, sizeof *s2);
  float *fnx = calloc(N, sizeof *fnx), *fny = calloc(N, sizeof *fny);
  float *coh = calloc(N, sizeof *coh);
  double *xs = calloc(N, sizeof *xs), *b = calloc(N, sizeof *b);
  double *rg = calloc(N, sizeof *rg);
  tr_wf *w = calloc(1, sizeof *w);
  float *r1f = calloc(N, sizeof *r1f);
  uint32_t maxpts = 4096, npaths = 0;
  float *pts = malloc((size_t)maxpts * 2 * sizeof *pts);
  bool oom = !c2 || !s2 || !fnx || !fny || !coh || !xs || !b || !rg || !w ||
             !r1f || !pts;
  if (!oom) {
    /* rasterize doubled-angle normals along every polyline (records are
     * consecutive in the blob: walk them, no path table needed) */
    npaths = 0;
    for (size_t po2 = 0; po2 + 12 <= g->blob_n;) {
      const uint8_t *rec = g->blob + po2;
      uint32_t noff = (uint32_t)rec[8] << 24 | (uint32_t)rec[9] << 16 |
                      (uint32_t)rec[10] << 8 | rec[11];
      if (po2 + 12 + noff > g->blob_n || (noff & 1u)) break;
      uint32_t pi = (uint32_t)po2;
      po2 += 12 + noff;
      npaths++;
      if (noff + 2 > maxpts) {
        maxpts = noff + 2;
        float *np = realloc(pts, (size_t)maxpts * 2 * sizeof *np);
        if (!np) continue;
        pts = np;
      }
      uint32_t n = ng_path_decode(g, pi, pts, maxpts);
      for (uint32_t q = 0; q + 1 < n; q++) {
        double x0 = (double)pts[q * 2], y0 = (double)pts[q * 2 + 1];
        double x1 = (double)pts[q * 2 + 2], y1 = (double)pts[q * 2 + 3];
        double tx = x1 - x0, ty = y1 - y0;
        double L = hypot(tx, ty);
        if (L < 1e-6) continue;
        double phi = atan2(tx / L, -ty / L); /* normal angle */
        float cc = (float)(cos(2 * phi) * L), ss = (float)(sin(2 * phi) * L);
        int nsp = (int)(L / (dec * 0.5)) + 1;
        for (int sp = 0; sp <= nsp; sp++) {
          double f = (double)sp / nsp;
          long ci = (long)((x0 + f * tx) / dec), cj = (long)((y0 + f * ty) / dec);
          if (ci < 0 || cj < 0 || ci >= (long)W || cj >= (long)H) continue;
          c2[(size_t)cj * W + (size_t)ci] += cc;
          s2[(size_t)cj * W + (size_t)ci] += ss;
        }
      }
    }
    double cx, cy;
    tr_uc_at(t, (double)slice, &cx, &cy, NULL, NULL);
    float cmax = 0;
    for (size_t k = 0; k < N; k++) {
      double m = hypot((double)c2[k], (double)s2[k]);
      if (m > 1e-9) {
        double phi = 0.5 * atan2((double)s2[k], (double)c2[k]);
        (void)0;
        fnx[k] = (float)cos(phi);
        fny[k] = (float)sin(phi);
        coh[k] = (float)m;
        if (coh[k] > cmax) cmax = coh[k];
      }
    }
    if (cmax > 0)
      for (size_t k = 0; k < N; k++) coh[k] /= cmax;
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++)
        rg[(size_t)j * W + i] = hypot(i * dec - cx, j * dec - cy) / dec;
    uint32_t anchor =
        (uint32_t)((size_t)(cy / dec) * W + (size_t)(cx / dec)) % (uint32_t)N;
    /* stage 1: r0 — cross rows only + smoothing toward geometric r */
    wf_sys S0 = {.W = W, .H = H, .nx = fnx, .ny = fny, .coh = coh,
                 .wdot = 0, .wcross = 0.95, .ws1 = 0.1, .ws2 = 0,
                 .anchor = anchor};
    wf_rhs(&S0, NULL, rg, b);
    memcpy(xs, rg, N * sizeof *xs);
    wf_cg(&S0, b, xs, 300);
    /* sign disambiguation: n . grad(r0) >= 0 */
    for (uint32_t j = 0; j + 1 < H; j++)
      for (uint32_t i = 0; i + 1 < W; i++) {
        size_t k = (size_t)j * W + i;
        double gx = xs[k + 1] - xs[k], gy = xs[k + W] - xs[k];
        if ((double)fnx[k] * gx + (double)fny[k] * gy < 0) {
          fnx[k] = -fnx[k];
          fny[k] = -fny[k];
        }
      }
    /* stage 2: r1 — unit advance along n per voxel */
    double *tdot = rg; /* reuse */
    for (size_t k = 0; k < N; k++) tdot[k] = dec;
    wf_sys S1 = {.W = W, .H = H, .nx = fnx, .ny = fny, .coh = coh,
                 .wdot = 0.05, .wcross = 0.95, .ws1 = 0, .ws2 = 0.2,
                 .anchor = anchor};
    wf_rhs(&S1, tdot, NULL, b);
    memset(xs, 0, N * sizeof *xs);
    wf_cg(&S1, b, xs, 400);
    for (size_t k = 0; k < N; k++) r1f[k] = (float)xs[k];
    w->r1 = r1f;
    w->W = W;
    w->H = H;
    w->dec = dec;
    w->ok = true;
    printf("tracer: winding field %ux%u (dec %.0f) from %u polylines in %.1fs\n",
           W, H, dec, npaths, tr_now() - t0);
  }
  ng_put(ngv, slot, g);
  free(c2);
  free(s2);
  free(fnx);
  free(fny);
  free(coh);
  free(xs);
  free(b);
  free(rg);
  free(pts);
  if (oom || !w->ok) {
    if (w && !w->ok) free(r1f);
    free(w);
    return NULL;
  }
  return w;
}

/* bilinear field sample + gradient (per fullres voxel) */
static bool tr_wf_at(const tr_wf *w, double x, double y, double *val, double g2[2]) {
  double u = x / w->dec, v = y / w->dec;
  if (u < 0 || v < 0 || u >= w->W - 1 || v >= w->H - 1) return false;
  uint32_t i = (uint32_t)u, j = (uint32_t)v;
  double fu = u - i, fv = v - j;
  size_t k = (size_t)j * w->W + i;
  double a = (double)w->r1[k], bb = (double)w->r1[k + 1];
  double c = (double)w->r1[k + w->W], d = (double)w->r1[k + w->W + 1];
  *val = a * (1 - fu) * (1 - fv) + bb * fu * (1 - fv) + c * (1 - fu) * fv + d * fu * fv;
  if (g2) {
    g2[0] = ((bb - a) * (1 - fv) + (d - c) * fv) / w->dec;
    g2[1] = ((c - a) * (1 - fu) + (d - bb) * fu) / w->dec;
  }
  return true;
}

/* seed columns along one winding-field iso-contour (the wrap's whole
 * perimeter, the outline "worked inward"): march the level set of r1
 * through p0, dropping a 2x2 seed column every `every` grid cells of
 * arc. Fronts then close the small gaps between columns instead of
 * relaying around the entire cross section from one start. */
#define TR_RIB_SEEDEVERY 32
static void tr_rib_seed_contour(r3d_tracer *t, const tr_wf *wf, td_cache *dt,
                                const double p0[2], int y0b, float w0, int x0,
                                uint32_t *fringe, uint32_t *nf) {
  double lev;
  if (!tr_wf_at(wf, p0[0], p0[1], &lev, NULL)) return;
  int placed = 0;
  for (int dir = 0; dir < 2; dir++) {
    double p[2] = {p0[0], p0[1]};
    double heading[2] = {0, 0};
    double arc = 0;
    int lastu = INT32_MIN;
    bool closed = false;
    for (int stp = 0; stp < 4000; stp++) {
      double v, g[2];
      if (!tr_wf_at(wf, p[0], p[1], &v, g)) break;
      double gl = hypot(g[0], g[1]);
      if (gl < 1e-4) break; /* field flat: outside data */
      double tx = -g[1] / gl, ty = g[0] / gl;
      if (dir == 1) {
        tx = -tx;
        ty = -ty;
      }
      if (heading[0] != 0 || heading[1] != 0)
        if (tx * heading[0] + ty * heading[1] < 0) {
          tx = -tx;
          ty = -ty;
        }
      heading[0] = tx;
      heading[1] = ty;
      double sl = t->cfg.step;
      p[0] += tx * sl;
      p[1] += ty * sl;
      for (int nw = 0; nw < 2; nw++) { /* Newton back onto the level set */
        if (!tr_wf_at(wf, p[0], p[1], &v, g)) break;
        double g2 = g[0] * g[0] + g[1] * g[1];
        if (g2 < 1e-8) break;
        double c = (v - lev) / g2;
        p[0] -= c * g[0];
        p[1] -= c * g[1];
      }
      arc += sl;
      if (arc > 8 * sl && hypot(p[0] - p0[0], p[1] - p0[1]) < sl) {
        closed = true;
        break;
      }
      int du = (int)llround(arc / sl);
      int u = dir == 0 ? x0 + du : x0 - du;
      if (u < 2 || u + 1 >= (int)t->W - 2) break;
      if (du % TR_RIB_SEEDEVERY || u == lastu) continue;
      lastu = u;
      if (dt) { /* the field is wrong in sheared zones — only seed where
                 * the predictions confirm a sheet actually lives here */
        double q[3] = {p[0], p[1], t->cfg.seed[2]};
        if (td_tri(dt, q, NULL) > 3.0) continue;
      }
      /* place a 2x2 column if the spot is free */
      bool free4 = true;
      for (int q = 0; q < 4 && free4; q++)
        if (t->state[(size_t)(y0b + q / 2) * t->W + (size_t)(u + q % 2)] !=
            R3D_TR_EMPTY)
          free4 = false;
      if (!free4) continue;
      pthread_mutex_lock(&t->mu);
      for (int q = 0; q < 4; q++) {
        int i = u + q % 2, j = y0b + q / 2;
        size_t k = (size_t)j * t->W + (size_t)i;
        double *P = t->pos + k * 3;
        P[0] = p[0] + 0.1 * (q % 2) * tx;
        P[1] = p[1] + 0.1 * (q % 2) * ty;
        P[2] = t->cfg.seed[2] + 0.1 * (q / 2);
        t->state[k] = R3D_TR_SET;
        if (t->gen_of) t->gen_of[k] = 1;
        t->conf[k] = 1.0f;
        t->wind[k] = w0;
        fringe[(*nf)++] = (uint32_t)k;
      }
      t->nset += 4;
      pthread_mutex_unlock(&t->mu);
      placed++;
    }
    if (closed) break;
  }
  if (placed)
    printf("tracer: wrap %+d seeded at %d perimeter columns\n", (int)w0, placed + 1);
}

/* ======================== residual evaluation ======================== */
enum {
  TRF_DIST = 1,
  TRF_STRAIGHT = 2,
  TRF_SDIR = 4,
  TRF_SPACE = 8,
  TRF_NCP = 16,
  TRF_WIND = 32,
  TRF_SURF = 64,
  TRF_SELF = 128
};
#define TRF_ALL \
  (TRF_DIST | TRF_STRAIGHT | TRF_SDIR | TRF_SPACE | TRF_NCP | TRF_WIND | TRF_SURF | \
   TRF_SELF)

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

/* Hard hinge against doubling back: consecutive grid edges must not turn
 * past TR_FOLD_COS (90 deg). tr_res_straight's curvature cost is gentle
 * enough that a 180-degree fold can still win when the data term prefers
 * the wrong sheet; this hinge makes a fold an order of magnitude more
 * expensive while costing nothing on a healthy grid. Same triple/role
 * layout as tr_res_straight. */
static void tr_res_fold(tr_nlsq *acc, const double a[3], const double b[3],
                        const double c[3], int role, double w) {
  double d1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  double d2[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
  double l1s = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
  double l2s = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
  if (l1s <= 1e-24 || l2s <= 1e-24) return;
  double l1 = sqrt(l1s), l2 = sqrt(l2s);
  double dot = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (l1 * l2);
  if (dot >= TR_FOLD_COS) return; /* inactive on a healthy grid */
  double r = w * (TR_FOLD_COS - dot), g = -w;
  double gd1[3], gd2[3];
  for (int k = 0; k < 3; k++) {
    gd1[k] = d2[k] / (l1 * l2) - dot * d1[k] / l1s;
    gd2[k] = d1[k] / (l1 * l2) - dot * d2[k] / l2s;
  }
  double J[3];
  for (int k = 0; k < 3; k++) {
    double dd;
    if (role == 0) dd = -gd1[k];
    else if (role == 1) dd = gd1[k] - gd2[k];
    else dd = gd2[k];
    J[k] = g * dd;
  }
  nq_add(acc, r, J);
}

/* Curvature continuation (sparse-region prior): penalize the CHANGE of
 * curvature along a grid line — the third difference a - 3b + 3c - d of
 * four consecutive cells. Where the data term is silent, the sheet keeps
 * bending at its established rate instead of going straight (a
 * straightness prior sends the front off tangent to a curl, which then
 * re-latches onto the wrong wrap when data resumes). role = index of the
 * free cell in the 4-tuple. */
static void tr_res_curv(tr_nlsq *acc, const double a[3], const double b[3],
                        const double c[3], const double d[3], int role, double w) {
  static const double coef[4] = {1.0, -3.0, 3.0, -1.0};
  for (int k = 0; k < 3; k++) {
    double r = w * (a[k] - 3.0 * b[k] + 3.0 * c[k] - d[k]);
    double J[3] = {0, 0, 0};
    J[k] = w * coef[role];
    nq_add(acc, r, J);
  }
}

/* Quad planarity (twist): distance of the free corner from the plane of
 * the quad's other three corners, frozen for this evaluation (Jacobian =
 * w*n). Curvature terms act along grid lines; a quad can still twist
 * about its diagonal without paying — this keeps each quad near-planar
 * without resisting the sheet's real bending (adjacent quads may tilt
 * against each other freely). */
static void tr_res_planar(tr_nlsq *acc, const double x[3], const double a[3],
                          const double b[3], const double c[3], double w) {
  double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0]};
  double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (ln < 1e-12) return;
  for (int k = 0; k < 3; k++) n[k] /= ln;
  double r = w * (n[0] * (x[0] - a[0]) + n[1] * (x[1] - a[1]) + n[2] * (x[2] - a[2]));
  double J[3] = {w * n[0], w * n[1], w * n[2]};
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

/* effective sheet gap: joint-fit omega when trusted, else measured */
static double tr_om_eff(const r3d_tracer *t) {
  if (t->sp_valid) return fabs(atomic_load(&t->sp_omega));
  return t->sp_om_meas >= 4.0 ? t->sp_om_meas : 0.0;
}

/* spiral winding prior: r = w * (rho - (r0(z) + omega*wind)) / |omega|,
 * Cauchy(1)-robustified; the free point is x. Normalizing by omega makes
 * one whole wrap of error cost ~w — commensurate with the other O(1)
 * residuals. */
/* ==================== spiral fill (fit_spiral analogue) ====================
 * Populate every EMPTY grid cell from the global spiral fit. (winding, z)
 * flood outward from the traced cells — along i the winding advances by the
 * arc one grid step subtends (step / (2 pi rho)), along j the z advances by
 * the row pitch measured from existing rows — and each cell lands at the
 * fit's forward position rho = r0(z) + harmonics(theta) + omega*w about the
 * umbilicus centerline, theta phased to the traced cells. The refine pass
 * that follows pulls the analytic sheet onto evidence. */
static uint32_t tr_spiral_populate(r3d_tracer *t) {
  if (!atomic_load(&t->sp_valid) || !t->uc || !t->wind || !t->pos) return 0;
  uint32_t W = t->W, H = t->H;
  double om = atomic_load(&t->sp_omega);
  if (fabs(om) < 1e-9) return 0;
  /* grid direction calibration from existing SET pairs: z per +j step and
   * the winding sign per +i step (magnitudes come from the fit) */
  double dzj = 0.0, dwi = 0.0;
  uint64_t nzj = 0, nwi = 0;
  for (uint32_t j = 0; j + 1 < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      size_t k = (size_t)j * W + i;
      if (t->state[k] == R3D_TR_SET && t->state[k + W] == R3D_TR_SET) {
        dzj += t->pos[(k + W) * 3 + 2] - t->pos[k * 3 + 2];
        nzj++;
      }
      if (i + 1 < W && t->state[k] == R3D_TR_SET && t->state[k + 1] == R3D_TR_SET) {
        dwi += (double)t->wind[k + 1] - (double)t->wind[k];
        nwi++;
      }
    }
  if (!nzj || !nwi) return 0;
  dzj /= (double)nzj;
  double wsign = dwi >= 0.0 ? 1.0 : -1.0;
  if (fabs(dzj) < 1e-6) dzj = t->cfg.step; /* single-row patch: assume +z */
  /* theta phase: any traced cell ties winding to absolute angle */
  double thr = 0.0, wr = 0.0;
  bool have_ref = false;
  for (size_t k = 0; k < (size_t)W * H && !have_ref; k++)
    if (t->state[k] == R3D_TR_SET && tr_theta_of(t, t->pos + k * 3, &thr)) {
      wr = (double)t->wind[k];
      have_ref = true;
    }
  if (!have_ref) return 0;
  double *wf = malloc((size_t)W * H * sizeof *wf);
  double *zf = malloc((size_t)W * H * sizeof *zf);
  uint8_t *hav = calloc((size_t)W * H, 1);
  if (!wf || !zf || !hav) {
    free(wf);
    free(zf);
    free(hav);
    return 0;
  }
  for (size_t k = 0; k < (size_t)W * H; k++)
    if (t->state[k] == R3D_TR_SET) {
      wf[k] = (double)t->wind[k];
      zf[k] = t->pos[k * 3 + 2];
      hav[k] = 1;
    }
  /* multi-pass flood: an empty cell takes (w, z) from any settled
   * 4-neighbour; i-steps advance the winding, j-steps advance z */
  bool grew = true;
  uint32_t pass = 0;
  while (grew && pass++ < W + H) {
    grew = false;
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++) {
        size_t k = (size_t)j * W + i;
        if (hav[k]) continue;
        static const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int o = 0; o < 4; o++) {
          int ii = (int)i + nb[o][0], jj = (int)j + nb[o][1];
          if (ii < 0 || jj < 0 || ii >= (int)W || jj >= (int)H) continue;
          size_t n = (size_t)jj * W + (size_t)ii;
          if (!hav[n]) continue;
          if (nb[o][1] == 0) { /* horizontal: winding advances by the arc
                                * one grid step subtends at this radius */
            double th = thr + 2.0 * M_PI * (wf[n] - wr);
            double r0v;
            tr_sp_r0_at(t, zf[n], &r0v, NULL);
            double rho = r0v + t->sp_ab[0] * cos(th) + t->sp_ab[1] * sin(th) +
                         t->sp_ab[2] * cos(2 * th) + t->sp_ab[3] * sin(2 * th) +
                         om * wf[n];
            if (fabs(rho) < 4.0) rho = rho < 0 ? -4.0 : 4.0;
            double dw = wsign * t->cfg.step / (2.0 * M_PI * fabs(rho));
            /* the neighbour sits at i + nb0: stepping back to k moves
             * -nb0 along i */
            wf[k] = wf[n] - (double)nb[o][0] * dw;
            zf[k] = zf[n];
          } else { /* vertical: z advances by the measured row pitch */
            wf[k] = wf[n];
            zf[k] = zf[n] - (double)nb[o][1] * dzj;
          }
          hav[k] = 1;
          grew = true;
          break;
        }
      }
  }
  /* stamp: forward map for every still-EMPTY cell inside the growth
   * margins, respecting the tracer's z window and the volume extent */
  uint32_t filled = 0;
  uint32_t mvj = t->cfg.rib_rows ? 0u : 2u;
  uint16_t g = (uint16_t)(t->gens_done ? (t->gens_done > 65535 ? 65535 : t->gens_done)
                                       : 1u);
  pthread_mutex_lock(&t->mu);
  for (uint32_t j = mvj; j + mvj < H; j++)
    for (uint32_t i = 2; i + 2 < W; i++) {
      size_t k = (size_t)j * W + i;
      if (t->state[k] != R3D_TR_EMPTY || !hav[k]) continue;
      double z = zf[k], w = wf[k];
      if (t->cfg.z_max > t->cfg.z_min && (z < t->cfg.z_min || z > t->cfg.z_max))
        continue;
      if (z < 1.0 || (t->vdim[2] > 0 && z > t->vdim[2] - 2.0)) continue;
      double cx, cy;
      if (!tr_uc_at(t, z, &cx, &cy, NULL, NULL)) continue;
      double th = thr + 2.0 * M_PI * (w - wr);
      double r0v;
      tr_sp_r0_at(t, z, &r0v, NULL);
      double rho = r0v + t->sp_ab[0] * cos(th) + t->sp_ab[1] * sin(th) +
                   t->sp_ab[2] * cos(2 * th) + t->sp_ab[3] * sin(2 * th) + om * w;
      if (rho < 4.0) continue; /* inside the core: off the physical sheet */
      double x = cx + rho * cos(th), y = cy + rho * sin(th);
      if (x < 1.0 || y < 1.0 || (t->vdim[0] > 0 && x > t->vdim[0] - 2.0) ||
          (t->vdim[1] > 0 && y > t->vdim[1] - 2.0))
        continue;
      t->pos[k * 3 + 0] = x;
      t->pos[k * 3 + 1] = y;
      t->pos[k * 3 + 2] = z;
      t->state[k] = R3D_TR_SET;
      t->conf[k] = 0.4f; /* analytic guess: below traced cells' confidence */
      t->wind[k] = (float)w;
      if (t->gen_of) t->gen_of[k] = g;
      t->nset++;
      filled++;
    }
  pthread_mutex_unlock(&t->mu);
  t->gen++;
  free(wf);
  free(zf);
  free(hav);
  return filled;
}

static void tr_res_wind(tr_nlsq *acc, const r3d_tracer *t, const double x[3],
                        double wnd, double wgt) {
  const tr_wf *wfp = t->wf;
  double ome = tr_om_eff(t);
  if (wfp && ome > 0) { /* geometry-agnostic winding-potential target */
    double v, g2[2];
    if (tr_wf_at(wfp, x[0], x[1], &v, g2)) {
      double r = wgt * (v - t->wf_base - ome * wnd) / ome;
      double s = sqrt(1.0 / (1.0 + r * r)); /* Cauchy(1) */
      double J[3] = {s * wgt * g2[0] / ome, s * wgt * g2[1] / ome, 0.0};
      nq_add(acc, r * s, J);
      return;
    }
  }
  if (!t->sp_valid) return;
  double cx, cy, dxz, dyz;
  if (!tr_uc_at(t, x[2], &cx, &cy, &dxz, &dyz)) return;
  double ux = x[0] - cx, uy = x[1] - cy;
  double rho = sqrt(ux * ux + uy * uy);
  if (rho < 1e-6) return;
  double r0v, r0s;
  tr_sp_r0_at(t, x[2], &r0v, &r0s);
  double om = fabs(atomic_load(&t->sp_omega));
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
      tr_res_dist(acc, x, tr_at(c, ii, jj, x), D, TR_W_DIST * tr_ws(e->ws_dist));
    }
  /* sheetness is confidence-adaptive (sparse predictions are where the
   * front doubles back): where the data term anchors the sheet the
   * geometry stays supple (a stiff prior held cells ~8 vox off-sheet and
   * collapsed conf — the measured planarity regression), but where the
   * predictions run out the sheet's own inertia must take over. Local
   * evidence = the best confidence in the 4-neighbourhood (the cell's own
   * conf is 0 during its placement solve). */
  double stiff = 1.0, sparse = 0.0;
  {
    float bc = t->conf[(size_t)j * t->W + (size_t)i];
    static const int o4s[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int o = 0; o < 4; o++) {
      int ii = i + o4s[o][0], jj = j + o4s[o][1];
      if (!tr_valid(t, ii, jj)) continue;
      float cn = t->conf[(size_t)jj * t->W + (size_t)ii];
      if (cn > bc) bc = cn;
    }
    sparse = 1.0 - (bc > 1.0f ? 1.0 : (double)bc);
    if (sparse < 0.0) sparse = 0.0;
    /* NOTE deliberately NO straightness boost: a scroll sheet in a gap
     * must keep CURVING at its established rate - a straightness prior
     * sent the front off tangent to the curl (then it re-latched onto a
     * different wrap when data resumed: the fork/taco failure). The
     * sparse prior is curvature CONTINUATION (below) + 2D planarity. */
    (void)stiff;
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
                        tr_at(c, ci, cj, x), -w0,
                        TR_W_STRAIGHT * tr_ws(e->ws_straight));
        if (a < 2) /* anti-double-back hinge: inert on healthy geometry
                    * (activates only past 90 degrees) */
          tr_res_fold(acc, tr_at(c, ai, aj, x), tr_at(c, bi, bj, x),
                      tr_at(c, ci, cj, x), -w0, TR_W_FOLD);
      }
    if (sparse > 0.05) /* gap prior: continue the established curvature */
      for (int a = 0; a < 2; a++)
        for (int w0 = -2; w0 <= -1; w0++) { /* 4-tuple starts at p + w0*axis */
          static const int axc[2][2] = {{1, 0}, {0, 1}};
          int ai = i + w0 * axc[a][0], aj = j + w0 * axc[a][1];
          int bi = ai + axc[a][0], bj = aj + axc[a][1];
          int ci2 = bi + axc[a][0], cj2 = bj + axc[a][1];
          int di2 = ci2 + axc[a][0], dj2 = cj2 + axc[a][1];
          if (!tr_valid(t, ai, aj) || !tr_valid(t, bi, bj) ||
              !tr_valid(t, ci2, cj2) || !tr_valid(t, di2, dj2))
            continue;
          tr_res_curv(acc, tr_at(c, ai, aj, x), tr_at(c, bi, bj, x),
                      tr_at(c, ci2, cj2, x), tr_at(c, di2, dj2, x), -w0,
                      0.6 * sparse * tr_ws(e->ws_straight));
        }
    /* quad planarity: for each full quad containing the free cell, keep the
     * free corner on the plane of the other three */
    static const int qoff2[4][2] = {{0, 0}, {-1, 0}, {0, -1}, {-1, -1}};
    for (int q = 0; q < 4; q++) {
      int qi = i + qoff2[q][0], qj = j + qoff2[q][1];
      if (!tr_valid(t, qi, qj) || !tr_valid(t, qi + 1, qj) || !tr_valid(t, qi, qj + 1) ||
          !tr_valid(t, qi + 1, qj + 1))
        continue;
      const double *o3[3];
      int no3 = 0;
      static const int cc[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
      for (int k2 = 0; k2 < 4 && no3 < 3; k2++) {
        int ci2 = qi + cc[k2][0], cj2 = qj + cc[k2][1];
        if (ci2 == i && cj2 == j) continue;
        o3[no3++] = tr_at(c, ci2, cj2, x);
      }
      if (no3 == 3 && sparse > 0.05 && tr_geom_terms())
        /* planarity ONLY where evidence is missing: the blanket version
         * held cells off the sheet everywhere (measured regression); the
         * adaptive version supplies 2D sheet inertia exactly where the
         * data term is silent and the front used to curl. */
        tr_res_planar(acc, x, o3[0], o3[1], o3[2],
                      TR_W_PLANAR * 2.0 * sparse * tr_ws(e->ws_straight));
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
  if ((c->flags & TRF_WIND) && (t->sp_valid || t->wf) && t->cfg.wind_weight > 0)
    tr_res_wind(acc, t, x, (double)t->wind[(size_t)j * t->W + (size_t)i],
                t->cfg.wind_weight);
  if (t->cfg.rib_rows) {
    /* ribbon rows are fixed-z cross-section curves (Lasagna's per-slice
     * mesh): anchor z to the row's plane so u follows sheet-x-plane —
     * a free 3D ribbon geodesically walks out of its slab instead.
     * Multi-wrap grids stack sibling blocks; every block spans the SAME
     * z slab (row index is block-local). */
    int rr = (int)t->cfg.rib_rows;
    int jl = t->cfg.rib_wraps > 1 ? j % (rr + 1) : j;
    double zrow = t->cfg.seed[2] + ((double)jl - (double)rr / 2) * t->cfg.step;
    double Jz[3] = {0, 0, 10.0};
    nq_add(acc, 10.0 * (x[2] - zrow), Jz);
  }
  if ((c->flags & TRF_SELF) && t->sfx)
    tr_res_self(acc, t, x, (size_t)j * t->W + (size_t)i);
  if ((c->flags & TRF_SURF) && t->don) {
    /* donor anchor (vc3d SurfaceLossD, w=0.1): pull toward the nearest
     * donor surface point, frozen for this evaluation; donors on a
     * different registered winding never attract */
    double q[3], dwnd;
    size_t kc = (size_t)j * t->W + (size_t)i;
    if (t->dcell_id && t->dcell_id[kc] == -2) goto skip_donor; /* fold */
    int di = tr_don_closest(t->don, x, TR_FUS_PULL, q, &dwnd, NULL);
    if (di >= 0 &&
        !(t->uc && dwnd > -1e29 &&
          fabs(dwnd - (double)t->wind[kc]) > TR_FUS_WIND_TH)) {
      for (int a = 0; a < 3; a++) {
        double J[3] = {0, 0, 0};
        J[a] = TR_W_SURF;
        nq_add(acc, TR_W_SURF * (x[a] - q[a]), J);
      }
    }
  skip_donor:;
  }
  if (t->nanc) {
    /* user anchor owned by this cell (G10b, vc3d PointsCorrectionLoss):
     * pull along the LOCAL SHEET NORMAL only — a correction is about
     * which sheet, not which uv. The old per-axis pull fought
     * Dist/Straight/Dirichlet directly: either it won and left a dimple,
     * or lost and the anchor never engaged. Normal frozen from the
     * current grid tangents; falls back to the full pull when the
     * neighbourhood is too degenerate to define one. */
    int32_t k32 = (int32_t)((size_t)j * t->W + (size_t)i);
    for (uint32_t a = 0; a < t->nanc; a++) {
      if (t->anc_cell[a] != k32) continue;
      const double *P = t->anc + (size_t)a * 3;
      double nrm[3] = {0, 0, 0};
      bool have_n = false;
      if (tr_valid(t, i - 1, j) && tr_valid(t, i + 1, j) && tr_valid(t, i, j - 1) &&
          tr_valid(t, i, j + 1)) {
        const double *pu0 = t->pos + ((size_t)j * t->W + (size_t)i - 1) * 3;
        const double *pu1 = t->pos + ((size_t)j * t->W + (size_t)i + 1) * 3;
        const double *pv0 = t->pos + ((size_t)(j - 1) * t->W + (size_t)i) * 3;
        const double *pv1 = t->pos + ((size_t)(j + 1) * t->W + (size_t)i) * 3;
        double eu[3], ev[3];
        for (int a2 = 0; a2 < 3; a2++) {
          eu[a2] = pu1[a2] - pu0[a2];
          ev[a2] = pv1[a2] - pv0[a2];
        }
        nrm[0] = eu[1] * ev[2] - eu[2] * ev[1];
        nrm[1] = eu[2] * ev[0] - eu[0] * ev[2];
        nrm[2] = eu[0] * ev[1] - eu[1] * ev[0];
        double l = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        if (l > 1e-9) {
          for (int a2 = 0; a2 < 3; a2++) nrm[a2] /= l;
          have_n = true;
        }
      }
      double dv[3] = {x[0] - P[0], x[1] - P[1], x[2] - P[2]};
      double dist = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
      double cap = TR_ANC_CAPTURE * t->cfg.step;
      double fall = dist < cap ? 1.0 - dist / cap * 0.5 : 0.5; /* gentle */
      if (have_n) {
        double dn = dv[0] * nrm[0] + dv[1] * nrm[1] + dv[2] * nrm[2];
        double w = TR_W_ANC * fall;
        double J[3] = {w * nrm[0], w * nrm[1], w * nrm[2]};
        nq_add(acc, w * dn, J);
      } else {
        for (int ax = 0; ax < 3; ax++) {
          double J[3] = {0, 0, 0};
          J[ax] = TR_W_ANC * fall;
          nq_add(acc, TR_W_ANC * fall * (x[ax] - P[ax]), J);
        }
      }
    }
  }
  if (t->reopt_on && t->reopt_pos && t->reopt_nrm) {
    /* tangential position memory during re-optimisation (G10a, vc3d
     * NormalOnlyPenalty): each cell in a refine disc remembers where it
     * was, tangentially only — free to move along its own normal (the
     * sheet-choice direction), pinned in uv so a 6-sweep anneal cannot
     * translate the disc along the sheet and shear the parameterisation
     * against the untouched region. */
    size_t kc = (size_t)j * t->W + (size_t)i;
    const float *n0f = t->reopt_nrm + kc * 3;
    double n0[3] = {(double)n0f[0], (double)n0f[1], (double)n0f[2]};
    double nl = n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2];
    if (nl > 0.25) {
      const double *p0 = t->reopt_pos + kc * 3;
      double d[3] = {x[0] - p0[0], x[1] - p0[1], x[2] - p0[2]};
      double dn = d[0] * n0[0] + d[1] * n0[1] + d[2] * n0[2];
      const double w = 10.0;
      for (int a2 = 0; a2 < 3; a2++) {
        double r = w * (d[a2] - dn * n0[a2]);
        double J[3];
        for (int b2 = 0; b2 < 3; b2++)
          J[b2] = w * ((a2 == b2 ? 1.0 : 0.0) - n0[a2] * n0[b2]);
        nq_add(acc, r, J);
      }
      if (t->ctsnap_tgt) {
        float s = t->ctsnap_tgt[kc];
        if (s < 1e29f) { /* edge found: pull onto it along the frozen normal */
          double J[3] = {TR_W_CTSNAP * n0[0], TR_W_CTSNAP * n0[1],
                         TR_W_CTSNAP * n0[2]};
          nq_add(acc, TR_W_CTSNAP * (dn - (double)s), J);
        }
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
  /* grid refs stay pinned in e->gc across solves: neighbor cells reuse
   * the same slices, and releasing per solve made every next solve
   * reacquire through the global table mutex (11% of trace CPU in lock
   * traffic). The FIFO in ng_eget bounds pins to 12/thread; workers
   * flush on pool idle. */
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

#define TR_NTHREADS 10 /* 16 measured slower: claims are separation-limited */
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

/* winding for cell k: neighbor wind + unwrapped angle step, averaged
 * over the placed 3x3 ring — combinatorial, so later solves can move
 * the point but never re-wind it */
static void tr_wind_assign(r3d_tracer *t, size_t k, int i, int j) {
  if (!t->uc) return;
  const double *fp = t->pos + k * 3;
  double th;
  if (!tr_theta_of(t, fp, &th)) return;
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
  { /* pre-solve veto (vc3d GrowPatch.cpp:4851): test the mean of the
     * candidate's valid 3x3 neighbours against the xyz boxes, the volume,
     * and (ribbon) CT validity BEFORE committing — a doomed candidate must
     * never be placed, solved, and allowed to drag its disc first */
    double pm[3] = {0, 0, 0};
    int pn = 0;
    for (int dj = -1; dj <= 1; dj++)
      for (int di = -1; di <= 1; di++) {
        if (!tr_valid(t, i + di, j + dj)) continue;
        const double *q = t->pos + ((size_t)(j + dj) * W + (size_t)(i + di)) * 3;
        for (int a = 0; a < 3; a++) pm[a] += q[a];
        pn++;
      }
    if (pn) {
      for (int a = 0; a < 3; a++) pm[a] /= pn;
      bool veto = false;
      if (t->cfg.z_max > t->cfg.z_min && (pm[2] < t->cfg.z_min || pm[2] > t->cfg.z_max))
        veto = true;
      if (t->cfg.x_max > t->cfg.x_min && (pm[0] < t->cfg.x_min || pm[0] > t->cfg.x_max))
        veto = true;
      if (t->cfg.y_max > t->cfg.y_min && (pm[1] < t->cfg.y_min || pm[1] > t->cfg.y_max))
        veto = true;
      if (t->vdim[0] > 0 && (pm[0] < 0 || pm[1] < 0 || pm[2] < 0 ||
                             pm[0] >= t->vdim[0] || pm[1] >= t->vdim[1] ||
                             pm[2] >= t->vdim[2]))
        veto = true;
      if (!veto && t->cfg.rib_rows) {
        /* ribbon boundary: stop at TRUE nothingness at the parent position
         * (was a whole generation late as a post-placement pass) */
        if (t->bnd_ct) {
          double v = r3d_cpuvol_tri(t->bnd_ct, t->bnd_lv, pm, NULL);
          for (int o = 0; o < 6 && v < t->bnd_min; o++) {
            static const double off[6][3] = {{2, 0, 0}, {-2, 0, 0}, {0, 2, 0},
                                             {0, -2, 0}, {0, 0, 2}, {0, 0, -2}};
            double q2[3] = {pm[0] + off[o][0], pm[1] + off[o][1], pm[2] + off[o][2]};
            double v2 = r3d_cpuvol_tri(t->bnd_ct, t->bnd_lv, q2, NULL);
            if (v2 > v) v = v2;
          }
          veto = v < t->bnd_min;
        } else if (e->dt) {
          veto = td_tri(e->dt, pm, NULL) > 50.0 + t->cfg.step;
        }
      }
      if (veto) {
        pthread_mutex_lock(&t->mu);
        t->state[k] = R3D_TR_FAIL; /* pos untouched, nset untouched */
        pthread_mutex_unlock(&t->mu);
        return false;
      }
    }
  }
  /* snapshot the radius-3 disc so a post-solve failure can retract without
   * leaving a scar (neighbours pulled toward a point that then dies) */
  double dsnap[49 * 3];
  uint32_t dsnap_k[49], dsnap_n = 0;
  for (int dj = -3; dj <= 3; dj++)
    for (int di = -3; di <= 3; di++) {
      int ii = i + di, jj = j + dj;
      if ((di == 0 && dj == 0) || !tr_valid(t, ii, jj)) continue;
      size_t k2 = (size_t)jj * W + (size_t)ii;
      memcpy(dsnap + (size_t)dsnap_n * 3, t->pos + k2 * 3, 3 * sizeof(double));
      dsnap_k[dsnap_n++] = (uint32_t)k2;
    }
  const double *bp = t->pos + ((size_t)bj * W + (size_t)bi) * 3;
  /* fused initial guess (G12, vc3d affine uv extrapolation): when the
   * parent and the cell behind it both sit on the same donor, extrapolate
   * the donor-grid uv linearly and start AT the donor surface — nearest-
   * point matching from parent+jitter is order-0 and direction-blind at
   * folds, where the nearest point can be behind the front. */
  double guess[3];
  bool have_guess = false;
  if (t->dcell_id && t->dcell_uv) {
    int hi2 = 2 * bi - i, hj2 = 2 * bj - j;
    size_t kp = (size_t)bj * W + (size_t)bi;
    if (hi2 >= 0 && hj2 >= 0 && hi2 < (int)W && hj2 < (int)t->H &&
        t->state[(size_t)hj2 * W + (size_t)hi2] == R3D_TR_SET) {
      size_t kb = (size_t)hj2 * W + (size_t)hi2;
      if (t->dcell_id[kp] >= 0 && t->dcell_id[kp] == t->dcell_id[kb]) {
        double uv[2] = {2.0 * (double)t->dcell_uv[kp * 2] -
                            (double)t->dcell_uv[kb * 2],
                        2.0 * (double)t->dcell_uv[kp * 2 + 1] -
                            (double)t->dcell_uv[kb * 2 + 1]};
        have_guess = tr_don_bilerp(t->don, t->dcell_id[kp], uv, guess);
      }
    }
  }
  pthread_mutex_lock(&t->mu);
  for (int a = 0; a < 3; a++)
    t->pos[k * 3 + (size_t)a] =
        (have_guess ? guess[a] : bp[a]) + tr_urand(rng);
  t->state[k] = R3D_TR_SET; /* committed before the solve (vc3d) */
  if (t->gen_of) t->gen_of[k] = t->cur_gen ? t->cur_gen : 1;
  t->nset++;
  pthread_mutex_unlock(&t->mu);
  tr_wind_assign(t, k, i, j); /* preliminary: the donor gates in the
                               * placement solve read it */
  /* placement: geometric + data terms, then radius-1 and radius-3 */
  double tt0 = tr_now();
  double pcost =
      tr_solve_cell(t, e, i, j, TRF_DIST | TRF_STRAIGHT | TRF_SPACE | TRF_SURF, 50);
  if (t->don) { /* adoption (vc3d commit semantics): a candidate that
                 * lands within same_surface_th of a same-winding donor
                 * takes the donor's exact geometry before refinement */
    double q[3], dwnd;
    double *P = t->pos + k * 3;
    int di = tr_don_closest(t->don, P, TR_FUS_TH, q, &dwnd, NULL);
    if (di >= 0 &&
        !(t->uc && dwnd > -1e29 &&
          fabs(dwnd - (double)t->wind[k]) > TR_FUS_WIND_TH)) {
      pthread_mutex_lock(&t->mu);
      memcpy(P, q, 3 * sizeof *P);
      pthread_mutex_unlock(&t->mu);
    }
  }
  tr_tm_add(0, tt0);
  tr_local_opt(t, e, i, j, 1, 2, false);
  tr_local_opt(t, e, i, j, 3, 3, false);
  const double *fp = t->pos + k * 3; /* the ONLY legitimate hole: the
                                      * point left the scroll volume */
  static _Atomic int fold_gate = -1;
  int fg = atomic_load_explicit(&fold_gate, memory_order_relaxed);
  if (fg < 0) {
    const char *ev = getenv("R3D_FOLD_GATE");
    fg = ev ? atoi(ev) : 0; /* DEFAULT OFF: A/B at 60 gens showed the
                             * deferral leaves mis-seated re-placements
                             * the tear mask then cuts - visually a far
                             * worse surface despite better fold counts.
                             * The final-QC fold clamp (save honesty)
                             * carries the do-not-ship-folds guarantee. */
    atomic_store(&fold_gate, fg);
  }
  if (fg) { /* fold gate: a placement that STILL leaves a >90-degree turn through
     * this cell after its radius-1/radius-3 solves is measurably wrong
     * right now — defer it (EMPTY = retried when the neighbourhood
     * improves) instead of letting the next generation build on top of a
     * fold and hold it in place. Growth-never-rejects predates the retry
     * machinery; with retries, committing a known fold buys nothing. */
    bool folded = false;
    static const int ax2[2][2] = {{1, 0}, {0, 1}};
    for (int a2 = 0; a2 < 2 && !folded; a2++) {
      int ai = i - ax2[a2][0], aj = j - ax2[a2][1];
      int ci2 = i + ax2[a2][0], cj2 = j + ax2[a2][1];
      if (!tr_valid(t, ai, aj) || !tr_valid(t, ci2, cj2)) continue;
      const double *pa = t->pos + ((size_t)aj * W + (size_t)ai) * 3;
      const double *pc = t->pos + ((size_t)cj2 * W + (size_t)ci2) * 3;
      const double *pb = t->pos + k * 3;
      double d1[3], d2v[3], l1 = 0, l2 = 0, dot = 0;
      for (int c2 = 0; c2 < 3; c2++) {
        d1[c2] = pb[c2] - pa[c2];
        d2v[c2] = pc[c2] - pb[c2];
        l1 += d1[c2] * d1[c2];
        l2 += d2v[c2] * d2v[c2];
        dot += d1[c2] * d2v[c2];
      }
      if (l1 > 1e-12 && l2 > 1e-12 && dot / sqrt(l1 * l2) < 0.0) folded = true;
    }
    if (folded) {
      pthread_mutex_lock(&t->mu);
      t->state[k] = R3D_TR_EMPTY; /* retryable, not FAIL */
      /* (gated by R3D_FOLD_GATE) */
      if (t->gen_of) t->gen_of[k] = 0;
      if (t->nset) t->nset--;
      for (uint32_t s2 = 0; s2 < dsnap_n; s2++)
        memcpy(t->pos + (size_t)dsnap_k[s2] * 3, dsnap + (size_t)s2 * 3,
               3 * sizeof(double));
      pthread_mutex_unlock(&t->mu);
      return false;
    }
  }
  tr_wind_assign(t, k, i, j); /* final: settled position */
  if (t->don && t->dsup) { /* consensus count (vc3d inlier vote, gated by
                            * the winding frame) */
    int sup = tr_don_support(t->don, fp, (double)t->wind[k], t->uc != NULL);
    t->dsup[k] = (uint8_t)(sup > 255 ? 255 : sup);
    /* consensus gate (G6, vc3d GrowSurface commit path): fused growth
     * exists to inherit the donors; a cell no donor vouches for AND whose
     * solve stayed expensive is deferred — back to EMPTY (retryable, and
     * re-offered when its neighbourhood improves or inl_th anneals down),
     * never permanent geometry that seeds the next generation. */
    if (sup == 0 && pcost > t->inl_th) {
      pthread_mutex_lock(&t->mu);
      t->state[k] = R3D_TR_EMPTY;
      if (t->gen_of) t->gen_of[k] = 0;
      if (t->nset) t->nset--;
      for (uint32_t s2 = 0; s2 < dsnap_n; s2++)
        memcpy(t->pos + (size_t)dsnap_k[s2] * 3, dsnap + (size_t)s2 * 3,
               3 * sizeof(double));
      pthread_mutex_unlock(&t->mu);
      return false;
    }
  }
  bool zclamp = t->cfg.z_max > t->cfg.z_min &&
                (fp[2] < t->cfg.z_min || fp[2] > t->cfg.z_max);
  if (!zclamp && t->cfg.rib_wraps > 1 && t->sfx) {
    /* sibling-wrap fronts stop where the sheet is already traced: a
     * candidate landing on an existing SAME-winding cell that is not a
     * grid neighbor is coverage overlap, not new surface */
    const tr_sfx *sx = t->sfx;
    double rad = 3.0;
    long c0[3], c1[3];
    for (int a = 0; a < 3; a++) {
      c0[a] = (long)floor((fp[a] - rad) / sx->cs);
      c1[a] = (long)floor((fp[a] + rad) / sx->cs);
    }
    for (long cz = c0[2]; cz <= c1[2] && !zclamp; cz++)
      for (long cy = c0[1]; cy <= c1[1] && !zclamp; cy++)
        for (long cx = c0[0]; cx <= c1[0] && !zclamp; cx++)
          for (uint32_t en = sx->head[tr_sfx_h(sx, cx, cy, cz)]; en;
               en = sx->next[en - 1]) {
            uint32_t k2 = sx->cell[en - 1];
            if (t->state[k2] != R3D_TR_SET) continue;
            long di = (long)(k2 % t->W) - i, dj = (long)(k2 / t->W) - j;
            if (labs(di) <= 5 && labs(dj) <= (long)t->cfg.rib_rows + 1) continue;
            if (fabs((double)t->wind[k2] - (double)t->wind[k]) > 0.3) continue;
            double d2 = 0;
            for (int a = 0; a < 3; a++) {
              double dd = fp[a] - t->pos[(size_t)k2 * 3 + (size_t)a];
              d2 += dd * dd;
            }
            if (d2 < rad * rad) {
              zclamp = true;
              break;
            }
          }
  }
  if (zclamp || (t->vdim[0] > 0 &&
      (fp[0] < 0 || fp[1] < 0 || fp[2] < 0 || fp[0] >= t->vdim[0] ||
       fp[1] >= t->vdim[1] || fp[2] >= t->vdim[2]))) {
    pthread_mutex_lock(&t->mu);
    t->state[k] = R3D_TR_FAIL;
    if (t->nset) t->nset--;
    /* retract: restore the radius-3 disc the placement solve dragged
     * toward the now-dead point (vc3d never lets the point exist; our
     * second line of defence must at least not leave a scar) */
    for (uint32_t s2 = 0; s2 < dsnap_n; s2++)
      memcpy(t->pos + (size_t)dsnap_k[s2] * 3, dsnap + (size_t)s2 * 3,
             3 * sizeof(double));
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
  if (t->cfg.max_threads && t->cfg.max_threads < want)
    want = t->cfg.max_threads < 2 ? 0 : t->cfg.max_threads;
  const char *tenv = getenv("R3D_TRACE_THREADS");
  if (tenv) { /* 0/1 = serial (deterministic quality A/B runs) */
    long tv = strtol(tenv, NULL, 10);
    want = tv < 2 ? 0 : (tv > TR_NTHREADS ? TR_NTHREADS : (uint32_t)tv);
  }
  for (uint32_t i = 0; i < want; i++) {
    tr_env *e = &pl->env[pl->nth];
    memset(e, 0, sizeof *e);
    e->gl_plane = -9; /* eget memo starts invalid (0 is a real plane) */
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

/* Mesh QC scan (generation boundary): count consecutive-edge pairs that
 * turned past 90 deg (folds — the sheet doubling back) and past 30 deg
 * (kinks), and measure quad twist as the rms distance of each quad's 4th
 * corner from the plane of the other three. Display-only; the solve keeps
 * these down via tr_res_fold / tr_res_planar. */
static bool tr_qc_ok(const r3d_tracer *t, int i, int j) {
  return tr_valid(t, i, j) && t->conf[(size_t)j * t->W + (size_t)i] > 0.25f;
}

static _Atomic uint64_t tr_excise_total; /* stats: fold cells cut per trace */

/* 3D path length along the straight grid line between two cells, over
 * their direct 3D distance. A legitimately tight curl reads ~1.5-3 (the
 * path goes around the apex); a flap lying back over the sheet reads
 * far higher (the path runs out to the crease and back). A broken path
 * (invalid cell on the line) reads as suspect too. */
static double tr_grid_path_ratio(const r3d_tracer *t, uint32_t ka, uint32_t kb) {
  int i0 = (int)(ka % t->W), j0 = (int)(ka / t->W);
  int i1 = (int)(kb % t->W), j1 = (int)(kb / t->W);
  int di = i1 - i0, dj = j1 - j0;
  int steps = abs(di) > abs(dj) ? abs(di) : abs(dj);
  if (steps < 1) return 1.0;
  double path = 0.0;
  const double *prev = t->pos + (size_t)ka * 3;
  for (int s2 = 1; s2 <= steps; s2++) {
    int ci = i0 + (di * s2 + (di >= 0 ? steps / 2 : -steps / 2)) / steps;
    int cj = j0 + (dj * s2 + (dj >= 0 ? steps / 2 : -steps / 2)) / steps;
    if (!tr_valid(t, ci, cj)) return 1e9; /* broken path: suspect */
    const double *cur = t->pos + ((size_t)cj * t->W + (size_t)ci) * 3;
    double dd = 0;
    for (int a = 0; a < 3; a++) dd += (cur[a] - prev[a]) * (cur[a] - prev[a]);
    path += sqrt(dd);
    prev = cur;
  }
  const double *A = t->pos + (size_t)ka * 3, *B = t->pos + (size_t)kb * 3;
  double d2 = 0;
  for (int a = 0; a < 3; a++) d2 += (B[a] - A[a]) * (B[a] - A[a]);
  double d = sqrt(d2);
  return d > 1e-9 ? path / d : 1e9;
}

/* cell normal for the excision scan: central differences when possible,
 * one-sided when a neighbor is missing — a flap cell must not become
 * invisible to the overlap test just because its neighbors were cut
 * first (grid-edge rows too). False only with no valid neighbor on an
 * axis or a degenerate frame. */
static bool tr_cell_normal_f(const r3d_tracer *t, int i, int j, double n[3]) {
  const double *c = t->pos + ((size_t)j * t->W + (size_t)i) * 3;
  const double *u0 = tr_valid(t, i - 1, j)
                         ? t->pos + ((size_t)j * t->W + (size_t)i - 1) * 3 : c;
  const double *u1 = tr_valid(t, i + 1, j)
                         ? t->pos + ((size_t)j * t->W + (size_t)i + 1) * 3 : c;
  const double *v0 = tr_valid(t, i, j - 1)
                         ? t->pos + ((size_t)(j - 1) * t->W + (size_t)i) * 3 : c;
  const double *v1 = tr_valid(t, i, j + 1)
                         ? t->pos + ((size_t)(j + 1) * t->W + (size_t)i) * 3 : c;
  if (u0 == u1 || v0 == v1) return false;
  double eu[3], ev[3];
  for (int a = 0; a < 3; a++) {
    eu[a] = u1[a] - u0[a];
    ev[a] = v1[a] - v0[a];
  }
  n[0] = eu[1] * ev[2] - eu[2] * ev[1];
  n[1] = eu[2] * ev[0] - eu[0] * ev[2];
  n[2] = eu[0] * ev[1] - eu[1] * ev[0];
  double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (l < 1e-9) return false;
  for (int a = 0; a < 3; a++) n[a] /= l;
  return true;
}

/* pick the excision victim of a confirmed fold pair: the newer cell
 * (the flap was placed on top of older sheet), ties by lower conf;
 * anchored cells are never cut. Returns SIZE_MAX when both anchored. */
static size_t tr_fold_victim(const r3d_tracer *t, size_t k1, size_t k2) {
  bool a1 = false, a2 = false;
  for (uint32_t an = 0; an < t->nanc; an++) {
    if (t->anc_cell[an] >= 0 && (size_t)t->anc_cell[an] == k1) a1 = true;
    if (t->anc_cell[an] >= 0 && (size_t)t->anc_cell[an] == k2) a2 = true;
  }
  if (a1 && a2) return (size_t)-1;
  if (a1) return k2;
  if (a2) return k1;
  uint16_t g1 = t->gen_of ? t->gen_of[k1] : 0, g2 = t->gen_of ? t->gen_of[k2] : 0;
  if (g1 != g2) return g1 > g2 ? k1 : k2;
  return t->conf[k1] <= t->conf[k2] ? k1 : k2;
}

/* Zero-tolerance fold excision (generation boundary, pool idle): any
 * cell still on a >90-degree turn AFTER the repair anneal is cut back
 * to EMPTY (retryable) right now, newest arm first — the sheet doubling
 * back on itself must never survive into the next generation or the
 * live view. Because this runs every generation, a fold flap can never
 * grow deeper than the ring that created it: the crease flags on the
 * flap's first cells, each excision exposes the next flap cell as a new
 * crease, and the pass loop unwinds the whole chain. User anchors are
 * never excised. Returns the number of cells cut. */
static uint32_t tr_fold_excise(r3d_tracer *t) {
  static const int ax2[2][2] = {{1, 0}, {0, 1}};
  int W = (int)t->W, H = (int)t->H;
  uint32_t total = 0;
  /* victims are COLLECTED against a consistent snapshot and applied
   * after the scan: cutting in-place made the outcome depend on scan
   * order (a cell whose neighbors were cut earlier in the same pass
   * loses its stencil and turns invisible — which cells survived then
   * varied with the optimizer's floating-point choices) */
  uint8_t *cutmap = calloc((size_t)W * (size_t)H, 1);
  if (!cutmap) return 0;
  for (int pass = 0; pass < 16; pass++) {
    uint32_t nex = 0;
    pthread_mutex_lock(&t->mu);
    for (int j = 0; j < H; j++)
      for (int i = 0; i < W; i++) {
        size_t kb = (size_t)j * t->W + (size_t)i;
        if (t->state[kb] != R3D_TR_SET) continue;
        for (int a = 0; a < 2; a++) {
          int ai = i - ax2[a][0], aj = j - ax2[a][1];
          int ci = i + ax2[a][0], cj = j + ax2[a][1];
          if (!tr_valid(t, ai, aj) || !tr_valid(t, ci, cj)) continue;
          size_t ka = (size_t)aj * t->W + (size_t)ai;
          size_t kc = (size_t)cj * t->W + (size_t)ci;
          const double *pa = t->pos + ka * 3, *pb = t->pos + kb * 3,
                       *pc = t->pos + kc * 3;
          double d1[3], d2[3], l1 = 0, l2 = 0, dot = 0;
          for (int c2 = 0; c2 < 3; c2++) {
            d1[c2] = pb[c2] - pa[c2];
            d2[c2] = pc[c2] - pb[c2];
            l1 += d1[c2] * d1[c2];
            l2 += d2[c2] * d2[c2];
            dot += d1[c2] * d2[c2];
          }
          if (l1 < 1e-12 || l2 < 1e-12 || dot >= 0.0) continue;
          /* victim: the newest of the three (the doubling-back arm was
           * placed on top of older sheet), skipping anchored cells */
          size_t cand[3] = {kb, ka, kc};
          size_t victim = (size_t)-1;
          uint32_t vgen = 0;
          for (int v = 0; v < 3; v++) {
            bool anch = false;
            for (uint32_t an = 0; an < t->nanc && !anch; an++)
              anch = t->anc_cell[an] >= 0 && (size_t)t->anc_cell[an] == cand[v];
            if (anch) continue;
            uint32_t g2 = t->gen_of ? t->gen_of[cand[v]] : 0;
            if (victim == (size_t)-1 || g2 > vgen) {
              victim = cand[v];
              vgen = g2;
            }
          }
          if (victim == (size_t)-1 || cutmap[victim]) continue;
          cutmap[victim] = 1;
          nex++;
        }
      }
    for (size_t k = 0; nex && k < (size_t)W * (size_t)H; k++) {
      if (!cutmap[k]) continue;
      cutmap[k] = 0;
      if (t->state[k] != R3D_TR_SET) continue;
      t->state[k] = R3D_TR_EMPTY; /* retryable, not FAIL: the region
          * regrows from better parents next generations */
      t->conf[k] = 0.0f;
      if (t->gen_of) t->gen_of[k] = 0;
      if (t->nset) t->nset--;
    }
    pthread_mutex_unlock(&t->mu);
    total += nex;
    if (!nex) break;
  }
  /* phase 2 — smooth doubling: a 180 spread over 2-3 cells never trips
   * the per-edge turn test above, yet the flap then lies within a
   * fraction of the sheet gap of the older sheet with OPPOSING normals.
   * The solve-time hinge (tr_res_self) skips grid-close pairs to protect
   * legitimate tight curls, which is exactly where a young flap lives —
   * so here the grid-PATH ratio arbitrates: a curl's path goes around
   * the apex (~1.5-3x the direct distance) while a fold's path runs out
   * to the crease and back (far higher, or broken). Far pairs use the
   * hinge's own fold-back criterion directly. */
  if (t->sfx) {
    const tr_sfx *sx = t->sfx;
    for (int pass = 0; pass < 16; pass++) {
      uint32_t nex = 0;
      pthread_mutex_lock(&t->mu);
      for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
          size_t kb = (size_t)j * t->W + (size_t)i;
          if (t->state[kb] != R3D_TR_SET) continue;
          double n1[3];
          if (!tr_cell_normal_f(t, i, j, n1)) continue;
          const double *P = t->pos + kb * 3;
          double rmin = 0.5 * tr_om_at(t, P);
          if (rmin <= 1.0) continue;
          long c0[3], c1[3];
          for (int a = 0; a < 3; a++) {
            c0[a] = (long)floor((P[a] - rmin) / sx->cs);
            c1[a] = (long)floor((P[a] + rmin) / sx->cs);
          }
          for (long cz = c0[2]; cz <= c1[2]; cz++)
            for (long cy = c0[1]; cy <= c1[1]; cy++)
              for (long cx = c0[0]; cx <= c1[0]; cx++)
                for (uint32_t e = sx->head[tr_sfx_h(sx, cx, cy, cz)]; e;
                     e = sx->next[e - 1]) {
                  uint32_t k2 = sx->cell[e - 1];
                  if (k2 <= kb || t->state[k2] != R3D_TR_SET) continue;
                  int di2 = abs((int)(k2 % t->W) - i);
                  int dj2 = abs((int)(k2 / t->W) - j);
                  int gd = di2 > dj2 ? di2 : dj2;
                  if (gd < 2) continue; /* immediate neighbors: same patch */
                  const double *Q = t->pos + (size_t)k2 * 3;
                  double d2 = 0;
                  for (int a = 0; a < 3; a++) d2 += (P[a] - Q[a]) * (P[a] - Q[a]);
                  if (d2 >= rmin * rmin) continue;
                  double n2[3];
                  if (!tr_cell_normal_f(t, (int)(k2 % t->W), (int)(k2 / t->W), n2))
                    continue;
                  double ndot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
                  if (ndot >= -0.3) continue;
                  if (gd <= 10 &&
                      tr_grid_path_ratio(t, (uint32_t)kb, k2) < 4.0)
                    continue; /* tight curl: the path hugs the apex */
                  size_t victim = tr_fold_victim(t, kb, (size_t)k2);
                  if (victim == (size_t)-1 || cutmap[victim]) continue;
                  cutmap[victim] = 1; /* collected; applied below so the
                      * whole pass detects against one consistent grid */
                  nex++;
                }
        }
      for (size_t k = 0; nex && k < (size_t)W * (size_t)H; k++) {
        if (!cutmap[k]) continue;
        cutmap[k] = 0;
        if (t->state[k] != R3D_TR_SET) continue;
        t->state[k] = R3D_TR_EMPTY;
        t->conf[k] = 0.0f;
        if (t->gen_of) t->gen_of[k] = 0;
        if (t->nset) t->nset--;
      }
      pthread_mutex_unlock(&t->mu);
      total += nex;
      if (!nex) break;
    }
  }
  free(cutmap);
  if (total && t->gen_of) {
    /* orphan cleanup: excising a crease DISCONNECTS the doubled-back
     * flap rather than exposing it cell by cell (the flap's interior is
     * locally smooth and passes the turn test). Anything no longer
     * 4-connected to a seed cell (gen 1) or an anchor only ever grew
     * from folded parents — cut it too; it regrows clean. */
    uint64_t n = (uint64_t)t->W * t->H;
    uint8_t *reach = calloc(n, 1);
    uint32_t *q = malloc(n * sizeof *q);
    if (reach && q) {
      uint32_t qn = 0;
      pthread_mutex_lock(&t->mu);
      for (uint64_t k = 0; k < n; k++)
        if (t->state[k] == R3D_TR_SET && t->gen_of[k] <= 1) {
          reach[k] = 1;
          q[qn++] = (uint32_t)k;
        }
      for (uint32_t a = 0; a < t->nanc; a++)
        if (t->anc_cell[a] >= 0 && t->state[t->anc_cell[a]] == R3D_TR_SET &&
            !reach[t->anc_cell[a]]) {
          reach[t->anc_cell[a]] = 1;
          q[qn++] = (uint32_t)t->anc_cell[a];
        }
      for (uint32_t h2 = 0; h2 < qn; h2++) {
        uint32_t k = q[h2];
        int i = (int)(k % t->W), j = (int)(k / t->W);
        static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int o = 0; o < 4; o++) {
          int ii = i + o4[o][0], jj = j + o4[o][1];
          if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
          uint32_t k2 = (uint32_t)jj * t->W + (uint32_t)ii;
          if (reach[k2] || t->state[k2] != R3D_TR_SET) continue;
          reach[k2] = 1;
          q[qn++] = k2;
        }
      }
      if (qn) /* no seed marked = a loaded grid without gen data: skip */
        for (uint64_t k = 0; k < n; k++)
          if (t->state[k] == R3D_TR_SET && !reach[k]) {
            t->state[k] = R3D_TR_EMPTY;
            t->conf[k] = 0.0f;
            t->gen_of[k] = 0;
            if (t->nset) t->nset--;
            total++;
          }
      pthread_mutex_unlock(&t->mu);
    }
    free(reach);
    free(q);
  }
  if (total) atomic_fetch_add(&tr_excise_total, total);
  return total;
}

uint32_t r3d_tracer_fold_excise(r3d_tracer *t) {
  tr_sfx_build(t); /* overlap phase needs current positions (no-op
                    * without a sheet-gap estimate) */
  return tr_fold_excise(t);
}

/* Flood-fill the exterior: mark every cell reachable 4-connected from the
 * grid border without crossing a blocked cell. What remains unblocked and
 * unmarked is enclosed (a hole). Shared by the QC hole metric and the
 * inpaint interiority gate. */
static void tr_flood_exterior(uint32_t W, uint32_t H, const uint8_t *blocked,
                              uint8_t *ext) {
  memset(ext, 0, (size_t)W * H);
  uint32_t *q = malloc((size_t)W * H * sizeof *q);
  if (!q) return;
  uint32_t qn = 0;
  for (uint32_t i = 0; i < W; i++) {
    if (!blocked[i] && !ext[i]) ext[i] = 1, q[qn++] = i;
    size_t b = (size_t)(H - 1) * W + i;
    if (!blocked[b] && !ext[b]) ext[b] = 1, q[qn++] = (uint32_t)b;
  }
  for (uint32_t j = 0; j < H; j++) {
    size_t l = (size_t)j * W, r = l + W - 1;
    if (!blocked[l] && !ext[l]) ext[l] = 1, q[qn++] = (uint32_t)l;
    if (!blocked[r] && !ext[r]) ext[r] = 1, q[qn++] = (uint32_t)r;
  }
  while (qn) {
    uint32_t k = q[--qn];
    uint32_t i = k % W, j = k / W;
    static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int o = 0; o < 4; o++) {
      int ii = (int)i + o4[o][0], jj = (int)j + o4[o][1];
      if (ii < 0 || jj < 0 || ii >= (int)W || jj >= (int)H) continue;
      size_t k2 = (size_t)jj * W + (size_t)ii;
      if (blocked[k2] || ext[k2]) continue;
      ext[k2] = 1;
      q[qn++] = (uint32_t)k2;
    }
  }
  free(q);
}

static int tr_fcmp(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return x < y ? -1 : x > y ? 1 : 0;
}
static int tr_u64cmp(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : x > y ? 1 : 0;
}

/* Donor-agreement QC (fused runs): distance from <=2000 sampled trusted
 * cells to their nearest donor point. The donors are free ground truth —
 * a trace drifting a wrap away from them shows up here first. */
static void tr_qc_donor(r3d_tracer *t);
static void tr_qc2(r3d_tracer *t, bool clamp_folds) {
  uint32_t folds = 0, kinks = 0, nsharp = 0;
  double tw2 = 0.0, area = 0.0;
  size_t ntw = 0, ntrust = 0;
  uint32_t bb[4] = {t->W, t->H, 0, 0}; /* i0,j0,i1,j1 */
  int W = (int)t->W, H = (int)t->H;
  float *slant = malloc((size_t)W * (size_t)H * sizeof *slant);
  size_t nsl = 0;
  static const int ax2[2][2] = {{1, 0}, {0, 1}};
  for (int j = 0; j < H; j++)
    for (int i = 0; i < W; i++) {
      if (!tr_qc_ok(t, i, j)) continue;
      ntrust++;
      if ((uint32_t)i < bb[0]) bb[0] = (uint32_t)i;
      if ((uint32_t)j < bb[1]) bb[1] = (uint32_t)j;
      if ((uint32_t)i > bb[2]) bb[2] = (uint32_t)i;
      if ((uint32_t)j > bb[3]) bb[3] = (uint32_t)j;
      const double *b = t->pos + ((size_t)j * t->W + (size_t)i) * 3;
      for (int a = 0; a < 2; a++) {
        int ai = i - ax2[a][0], aj = j - ax2[a][1];
        int ci = i + ax2[a][0], cj = j + ax2[a][1];
        if (!tr_qc_ok(t, ai, aj) || !tr_qc_ok(t, ci, cj)) continue;
        const double *pa = t->pos + ((size_t)aj * t->W + (size_t)ai) * 3;
        const double *pc = t->pos + ((size_t)cj * t->W + (size_t)ci) * 3;
        double d1[3], d2[3], l1 = 0, l2 = 0, dot = 0;
        for (int k = 0; k < 3; k++) {
          d1[k] = b[k] - pa[k];
          d2[k] = pc[k] - b[k];
          l1 += d1[k] * d1[k];
          l2 += d2[k] * d2[k];
          dot += d1[k] * d2[k];
        }
        if (l1 < 1e-12 || l2 < 1e-12) continue;
        dot /= sqrt(l1 * l2);
        if (dot < 0.0) {
          if (folds < 16)
            t->qc_fold_cell[folds] = (uint32_t)((size_t)j * t->W + (size_t)i);
          folds++;
          if (clamp_folds && t->conf[(size_t)j * t->W + (size_t)i] > 0.25f)
            t->conf[(size_t)j * t->W + (size_t)i] = 0.25f;
        } else if (dot < TR_KINK_COS) {
          if (dot < 0.64 && nsharp < 16) /* sharper than ~50 deg */
            t->qc_kink_cell[nsharp++] = (uint32_t)((size_t)j * t->W + (size_t)i);
          kinks++;
        }
      }
      if (tr_qc_ok(t, i + 1, j) && tr_qc_ok(t, i, j + 1) && tr_qc_ok(t, i + 1, j + 1)) {
        const double *p10 = t->pos + ((size_t)j * t->W + (size_t)i + 1) * 3;
        const double *p01 = t->pos + ((size_t)(j + 1) * t->W + (size_t)i) * 3;
        const double *p11 = t->pos + ((size_t)(j + 1) * t->W + (size_t)i + 1) * 3;
        double e1[3], e2[3], n[3], d[3];
        for (int k = 0; k < 3; k++) {
          e1[k] = p10[k] - b[k];
          e2[k] = p01[k] - b[k];
          d[k] = p11[k] - b[k];
        }
        n[0] = e1[1] * e2[2] - e1[2] * e2[1];
        n[1] = e1[2] * e2[0] - e1[0] * e2[2];
        n[2] = e1[0] * e2[1] - e1[1] * e2[0];
        double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (ln > 1e-12) {
          double dist = (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / ln;
          tw2 += dist * dist;
          ntw++;
        }
        /* two-triangle quad area: (b,p10,p01) + (p10,p11,p01) */
        double f1[3], f2[3], cx[3];
        for (int k = 0; k < 3; k++) {
          f1[k] = p11[k] - p10[k];
          f2[k] = p01[k] - p10[k];
        }
        cx[0] = f1[1] * f2[2] - f1[2] * f2[1];
        cx[1] = f1[2] * f2[0] - f1[0] * f2[2];
        cx[2] = f1[0] * f2[1] - f1[1] * f2[0];
        area += 0.5 * (ln + sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]));
        /* slant: shear of the local uv frame */
        double luu = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
        if (slant && luu > 1e-12)
          slant[nsl++] =
              (float)(fabs(e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / luu);
      }
    }
  float sl95 = 0.0f;
  if (slant && nsl) {
    qsort(slant, nsl, sizeof *slant, tr_fcmp);
    sl95 = slant[nsl - 1 - nsl / 20];
  }
  free(slant);
  /* enclosed holes: untrusted cells unreachable from the grid border */
  uint32_t holes = 0;
  uint64_t bba = 0;
  if (ntrust) {
    bba = (uint64_t)(bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1);
    uint8_t *blocked = malloc((size_t)W * (size_t)H), *ext = malloc((size_t)W * (size_t)H);
    if (blocked && ext) {
      for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++)
          blocked[(size_t)j * (size_t)W + (size_t)i] = tr_qc_ok(t, i, j) ? 1 : 0;
      tr_flood_exterior(t->W, t->H, blocked, ext);
      for (size_t k = 0; k < (size_t)W * (size_t)H; k++)
        if (!blocked[k] && !ext[k]) holes++;
    }
    free(blocked);
    free(ext);
  }
  pthread_mutex_lock(&t->mu);
  t->qc_folds = folds;
  t->qc_nfoldc = folds < 16 ? folds : 16;
  t->qc_nkinkc = nsharp;
  t->qc_kinks = kinks;
  t->qc_twist = ntw ? (float)sqrt(tw2 / (double)ntw) : 0.0f;
  t->qc_area_vx2 = area;
  memcpy(t->qc_bbox, bb, sizeof bb);
  t->qc_fill = bba ? (float)((double)ntrust / (double)bba) : 0.0f;
  t->qc_hole = bba ? (float)((double)holes / (double)bba) : 0.0f;
  t->qc_slant_p95 = sl95;
  pthread_mutex_unlock(&t->mu);
  if (t->don) tr_qc_donor(t);
}

static void tr_qc_donor(r3d_tracer *t) {
  size_t n = (size_t)t->W * t->H;
  float d[2000];
  uint32_t nd = 0, tried = 0;
  double rad = 2.0 * t->cfg.step;
  size_t stride = n / 2000 + 1;
  for (size_t k = 0; k < n && tried < 2000; k += stride) {
    if (!tr_qc_ok(t, (int)(k % t->W), (int)(k / t->W))) continue;
    tried++;
    double q[3], dwnd;
    if (tr_don_closest(t->don, t->pos + k * 3, rad, q, &dwnd, NULL) < 0) continue;
    double s = 0;
    for (int a = 0; a < 3; a++) {
      double dd = t->pos[k * 3 + (size_t)a] - q[a];
      s += dd * dd;
    }
    d[nd++] = (float)sqrt(s);
  }
  float mean = 0, rms = 0, p95 = 0;
  if (nd) {
    double s1 = 0, s2 = 0;
    for (uint32_t i = 0; i < nd; i++) {
      s1 += (double)d[i];
      s2 += (double)d[i] * (double)d[i];
    }
    mean = (float)(s1 / nd);
    rms = (float)sqrt(s2 / nd);
    qsort(d, nd, sizeof *d, tr_fcmp);
    p95 = d[nd - 1 - nd / 20];
  }
  pthread_mutex_lock(&t->mu);
  t->qc_don_mean = mean;
  t->qc_don_rms = rms;
  t->qc_don_p95 = p95;
  t->qc_don_cov = tried ? (float)nd / (float)tried : 0.0f;
  pthread_mutex_unlock(&t->mu);
}

/* G4 (vc3d vc_tifxyz_winding relax): the causal winding is assigned at
 * placement and never follows the cell as later solves move it — one cell
 * dragged across a wrap boundary keeps a stale winding forever, blinding
 * the self-overlap hinge, the spacing pull, the donor gates and the
 * sibling stop all at once. Relax winding as a FIELD over the grown grid:
 * w[k] <- mean over 4-neighbours of (w[n] + dtheta/2pi), the seed pinned
 * at 0, edges with |dtheta/2pi| > 0.25 rejected (a quarter winding per
 * grid step is a cross-wrap link, not a sheet step). |wind - relaxed| is
 * then an intrinsic wrong-wrap detector needing no spiral model; with
 * R3D_WERR_CLAMP=1 cells with werr > 0.3 have conf clamped to 0.25 so
 * the save cutoff and the QC trust gate both see the capture. */
static void tr_wind_relax(r3d_tracer *t, int iters) {
  if (!t->uc || !t->werr || t->nset < 16) return;
  size_t n = (size_t)t->W * t->H;
  float *th = malloc(n * sizeof *th);
  float *w0 = malloc(n * sizeof *w0);
  float *w1 = malloc(n * sizeof *w1);
  if (!th || !w0 || !w1) {
    free(th);
    free(w0);
    free(w1);
    return;
  }
  for (size_t k = 0; k < n; k++) {
    th[k] = -1e30f;
    w0[k] = t->state[k] == R3D_TR_SET ? t->wind[k] : 0.0f;
    if (t->state[k] == R3D_TR_SET) {
      double a;
      if (tr_theta_of(t, t->pos + k * 3, &a)) th[k] = (float)a;
    }
  }
  uint32_t seed_k = (t->H / 2) * t->W + t->W / 2;
  int W = (int)t->W, H = (int)t->H;
  for (int it = 0; it < iters; it++) {
    for (int j = 0; j < H; j++)
      for (int i = 0; i < W; i++) {
        size_t k = (size_t)j * (size_t)W + (size_t)i;
        w1[k] = w0[k];
        if (t->state[k] != R3D_TR_SET || th[k] < -1e29f || k == seed_k) continue;
        double acc = 0.0;
        int cnt = 0;
        static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int o = 0; o < 4; o++) {
          int ii = i + o4[o][0], jj = j + o4[o][1];
          if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue;
          size_t k2 = (size_t)jj * (size_t)W + (size_t)ii;
          if (t->state[k2] != R3D_TR_SET || th[k2] < -1e29f) continue;
          double d = (double)th[k] - (double)th[k2];
          while (d > M_PI) d -= 2.0 * M_PI;
          while (d < -M_PI) d += 2.0 * M_PI;
          double dw = d / (2.0 * M_PI);
          if (fabs(dw) > 0.25) continue; /* cross-wrap link */
          acc += (double)w0[k2] + dw;
          cnt++;
        }
        if (cnt) w1[k] = (float)(acc / (double)cnt);
      }
    float *tmp = w0;
    w0 = w1;
    w1 = tmp;
  }
  static _Atomic int clamp_on = -1;
  int cl = atomic_load_explicit(&clamp_on, memory_order_relaxed);
  if (cl < 0) {
    const char *ev = getenv("R3D_WERR_CLAMP");
    cl = ev ? atoi(ev) : 1;
    atomic_store(&clamp_on, cl);
  }
  float we[4096];
  size_t nwe = 0, ntr = 0, nwrap = 0;
  pthread_mutex_lock(&t->mu);
  for (size_t k = 0; k < n; k++) {
    if (t->state[k] != R3D_TR_SET || th[k] < -1e29f) {
      t->werr[k] = 0.0f;
      continue;
    }
    float err = fabsf(t->wind[k] - w0[k]);
    t->werr[k] = err;
    if (t->conf[k] > 0.25f) {
      ntr++;
      if (err > 0.3f) nwrap++;
      if (nwe < sizeof we / sizeof *we && (k % 3) == 0) we[nwe++] = err;
    }
    if (cl && err > 0.3f && t->conf[k] > 0.25f) t->conf[k] = 0.25f;
  }
  if (nwe) {
    qsort(we, nwe, sizeof *we, tr_fcmp);
    t->qc_werr_p95 = we[nwe - 1 - nwe / 20];
  }
  t->qc_wrap_frac = ntr ? (float)nwrap / (float)ntr : 0.0f;
  pthread_mutex_unlock(&t->mu);
  free(th);
  free(w0);
  free(w1);
}

/* G8 (vc3d inpaint): enclosed holes (FAIL cells the growth left inside
 * the patch) are re-solved with the REAL loss stack instead of being
 * membrane-filled blind at save time. Membrane = initial guess only
 * (vc3d's masked_blur role); then the data term ramps in over three
 * staged passes so the fill finds the sheet instead of cutting through a
 * neighbouring wrap; conf is recomputed afterwards so a hole that found
 * no evidence stays below the save cutoff. Coordinator-only. */
static void tr_inpaint(r3d_tracer *t, tr_env *e) {
  size_t n = (size_t)t->W * t->H;
  uint8_t *blocked = malloc(n), *ext = malloc(n);
  uint32_t *holes = malloc(n * sizeof *holes);
  uint32_t nh = 0;
  if (!blocked || !ext || !holes) goto out;
  for (size_t k = 0; k < n; k++) blocked[k] = t->state[k] == R3D_TR_SET;
  tr_flood_exterior(t->W, t->H, blocked, ext);
  for (size_t k = 0; k < n; k++)
    if (!blocked[k] && !ext[k]) holes[nh++] = (uint32_t)k;
  if (!nh || nh > t->nset / 5) goto out; /* none, or something is wrong */
  /* membrane seed over the hole set (anchored on SET neighbours) */
  pthread_mutex_lock(&t->mu);
  for (uint32_t h = 0; h < nh; h++) { /* first guess: neighbour average */
    uint32_t k = holes[h];
    int i = (int)(k % t->W), j = (int)(k / t->W);
    double avg[3] = {0, 0, 0};
    int na = 0;
    for (int dj = -1; dj <= 1; dj++)
      for (int di = -1; di <= 1; di++) {
        if (!tr_valid(t, i + di, j + dj)) continue;
        const double *q = t->pos + ((size_t)(j + dj) * t->W + (size_t)(i + di)) * 3;
        for (int a = 0; a < 3; a++) avg[a] += q[a];
        na++;
      }
    if (na)
      for (int a = 0; a < 3; a++) t->pos[(size_t)k * 3 + (size_t)a] = avg[a] / na;
  }
  for (uint32_t h = 0; h < nh; h++) {
    uint32_t k = holes[h];
    t->state[k] = R3D_TR_SET;
    if (t->gen_of) t->gen_of[k] = t->cur_gen ? t->cur_gen : 1;
    t->nset++;
  }
  pthread_mutex_unlock(&t->mu);
  for (int it = 0; it < 32; it++) { /* relax the membrane */
    for (uint32_t h = 0; h < nh; h++) {
      uint32_t k = holes[h];
      int i = (int)(k % t->W), j = (int)(k / t->W);
      double avg[3] = {0, 0, 0};
      int na = 0;
      static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (int o = 0; o < 4; o++) {
        if (!tr_valid(t, i + o4[o][0], j + o4[o][1])) continue;
        const double *q =
            t->pos + ((size_t)(j + o4[o][1]) * t->W + (size_t)(i + o4[o][0])) * 3;
        for (int a = 0; a < 3; a++) avg[a] += q[a];
        na++;
      }
      if (na >= 2)
        for (int a = 0; a < 3; a++) t->pos[(size_t)k * 3 + (size_t)a] = avg[a] / na;
    }
  }
  { /* staged data-term ramp: geometry only -> weak snap -> full */
    static const double sched[3][3] = {{0.3, 0.1, -1.0}, {0.3, 0.1, 0.1}, {1, 1, 1}};
    for (int pass = 0; pass < 3 && !t->quit; pass++) {
      e->ws_dist = sched[pass][0];
      e->ws_straight = sched[pass][1];
      e->ws_snap = sched[pass][2];
      for (int sweep = 0; sweep < 3; sweep++)
        for (uint32_t h = 0; h < nh && !t->quit; h++)
          tr_solve_cell(t, e, (int)(holes[h] % t->W), (int)(holes[h] / t->W),
                        TRF_ALL, 4);
    }
    e->ws_dist = e->ws_straight = e->ws_snap = 0.0;
  }
  for (uint32_t h = 0; h < nh; h++) /* honest confidence for the fills */
    tr_update_conf(t, e, (int)(holes[h] % t->W), (int)(holes[h] / t->W));
  printf("tracer: inpainted %u enclosed hole cell%s (data-term ramp)\n", nh,
         nh == 1 ? "" : "s");
out:
  free(blocked);
  free(ext);
  free(holes);
}

/* Generation boundary (pool idle): adopt a staged anchor set and assign
 * each anchor to the nearest SET cell within the capture radius. Distant
 * anchors stay unassigned and are retried every generation, so an anchor
 * dropped ahead of the front engages the moment growth reaches it. */
static void tr_anc_assign(r3d_tracer *t) {
  pthread_mutex_lock(&t->mu);
  if (t->anc_dirty) {
    memcpy(t->anc, t->anc_new, sizeof(double) * 3 * t->nanc_new);
    t->nanc = t->nanc_new;
    t->anc_dirty = false;
  }
  uint32_t n = t->nanc;
  pthread_mutex_unlock(&t->mu);
  if (!n) return;
  double cap = TR_ANC_CAPTURE * t->cfg.step;
  double cap2 = cap * cap;
  for (uint32_t a = 0; a < n; a++) {
    const double *P = t->anc + (size_t)a * 3;
    double best = cap2;
    int32_t bk = -1;
    for (size_t k = 0; k < (size_t)t->W * t->H; k++) {
      if (t->state[k] != R3D_TR_SET) continue;
      const double *q = t->pos + k * 3;
      double dx = q[0] - P[0], dy = q[1] - P[1], dz = q[2] - P[2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        bk = (int32_t)k;
      }
    }
    if (t->anc_cell[a] != bk && bk >= 0)
      printf("tracer: anchor %u (%.0f,%.0f,%.0f) -> cell (%u,%u), %.1f vox away\n",
             a, P[0], P[1], P[2], (uint32_t)bk % t->W, (uint32_t)bk / t->W,
             sqrt(best));
    t->anc_cell[a] = bk;
  }
}

void r3d_tracer_set_anchors(r3d_tracer *t, const double *pts, uint32_t n) {
  if (n > R3D_TR_MAX_ANCHORS) n = R3D_TR_MAX_ANCHORS;
  pthread_mutex_lock(&t->mu);
  memcpy(t->anc_new, pts, sizeof(double) * 3 * n);
  t->nanc_new = n;
  t->anc_dirty = true;
  if (!t->running) { /* no grow thread to adopt: take effect immediately
                      * (r3d_tracer_grow reuses the set on resume) */
    memcpy(t->anc, t->anc_new, sizeof(double) * 3 * n);
    t->nanc = n;
    for (uint32_t a = 0; a < R3D_TR_MAX_ANCHORS; a++) t->anc_cell[a] = -1;
    t->anc_dirty = false;
  }
  pthread_mutex_unlock(&t->mu);
}

/* tangential position memory (G10a): snapshot pos + per-cell grid normals
 * so a solve pass can move cells across wraps but not along the sheet */
static bool tr_reopt_snapshot(r3d_tracer *t) {
  uint32_t W = t->W, H = t->H;
  size_t n2 = (size_t)W * H;
  free(t->reopt_pos);
  free(t->reopt_nrm);
  t->reopt_pos = malloc(n2 * 3 * sizeof *t->reopt_pos);
  t->reopt_nrm = calloc(n2 * 3, sizeof *t->reopt_nrm);
  if (!t->reopt_pos || !t->reopt_nrm) return false;
  memcpy(t->reopt_pos, t->pos, n2 * 3 * sizeof *t->reopt_pos);
  for (int j2 = 0; j2 < (int)H; j2++)
    for (int i2 = 0; i2 < (int)W; i2++) {
      if (!tr_valid(t, i2 - 1, j2) || !tr_valid(t, i2 + 1, j2) ||
          !tr_valid(t, i2, j2 - 1) || !tr_valid(t, i2, j2 + 1))
        continue;
      const double *pu0 = t->pos + ((size_t)j2 * W + (size_t)i2 - 1) * 3;
      const double *pu1 = t->pos + ((size_t)j2 * W + (size_t)i2 + 1) * 3;
      const double *pv0 = t->pos + ((size_t)(j2 - 1) * W + (size_t)i2) * 3;
      const double *pv1 = t->pos + ((size_t)(j2 + 1) * W + (size_t)i2) * 3;
      double eu[3], ev[3], nr[3];
      for (int a = 0; a < 3; a++) {
        eu[a] = pu1[a] - pu0[a];
        ev[a] = pv1[a] - pv0[a];
      }
      nr[0] = eu[1] * ev[2] - eu[2] * ev[1];
      nr[1] = eu[2] * ev[0] - eu[0] * ev[2];
      nr[2] = eu[0] * ev[1] - eu[1] * ev[0];
      double l = sqrt(nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]);
      if (l > 1e-9)
        for (int a = 0; a < 3; a++)
          t->reopt_nrm[((size_t)j2 * W + (size_t)i2) * 3 + (size_t)a] =
              (float)(nr[a] / l);
    }
  t->reopt_on = true;
  return true;
}

static void *tr_worker(void *ud) {
  r3d_tracer *t = ud;
  r3d_cpuvol vol;
  /* 256 bricks (512 MB): conf/DT sampling over a 60-gen trace re-decoded
   * bricks constantly at 96 (b_decode_sub was 17% of trace CPU) */
  if (r3d_cpuvol_open(&vol, t->root, 256) != 0) goto fail_open;
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
  if (t->cfg.rib_rows && !(t->cfg.z_max > t->cfg.z_min)) {
    double hh = (double)t->cfg.rib_rows / 2 + 2.0;
    t->cfg.z_min = t->cfg.seed[2] - hh * t->cfg.step;
    t->cfg.z_max = t->cfg.seed[2] + hh * t->cfg.step;
  }
  if (t->cfg.wind_weight > 0 && !t->uc)
    printf("tracer: spiral prior requested but no usable umbilicus (needs "
           ">= 2 control points) — growing without it\n");
  r3d_cpuvol ctv; /* optional raw CT for the boundary pass */
  bool ctv_ok = false;
  uint32_t ct_lv = 1; /* L2 partial-volumes thin sheets with their air
                       * gaps and mid-sheet reads fall below the cutoff —
                       * L1 keeps sheets ~6 px thick */
  double ct_min = t->cfg.ct_min > 0 ? t->cfg.ct_min : 64.0;
  /* default from measured traced-point histograms (PHerc0125 L1):
   * real surface peaks 96..143 and ~90% sits above 64; 128 rejected
   * three quarters of legitimate points */
  if (t->cfg.ct_root[0]) {
    ctv_ok = r3d_cpuvol_open(&ctv, t->cfg.ct_root, 64) == 0;
    if (ctv_ok && ct_lv >= ctv.nlev) ct_lv = ctv.nlev - 1;
    t->bnd_ct = ctv_ok ? &ctv : NULL; /* placement pre-veto CT access */
    t->bnd_lv = ct_lv;
    t->bnd_min = ct_min;
    if (!ctv_ok)
      printf("tracer: CT tree %s failed to open (boundary pass falls back "
             "to predictions)\n",
             t->cfg.ct_root);
  }
  ng_vol ng; /* vc3d's real data term when the store exists upstream */
  ng_open(&ng, t->root);
  if (t->cfg.rib_rows && t->cfg.wind_weight > 0 && !t->wf) {
    t->wf = tr_wf_build(t, &ng, (double)vol.nx, (double)vol.ny);
    if (t->wf) {
      double v0;
      if (tr_wf_at(t->wf, t->cfg.seed[0], t->cfg.seed[1], &v0, NULL)) {
        t->wf_base = v0;
      } else {
        tr_wf_free(t->wf);
        t->wf = NULL;
      }
    }
  }
  tr_env cenv = {.dt = dt, .ngv = &ng, .gl_plane = -9}; /* coordinator's
      * solve env (gl_plane -9: eget memo starts invalid) */
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
  if (t->spiral_fill) { /* fill the grid from the fit; the refine branch
                         * below polishes the analytic sheet onto evidence */
    uint32_t filled = tr_spiral_populate(t);
    printf("tracer: spiral fill placed %u cells from the fit\n", filled);
    t->spiral_fill = false;
    resume = t->nset > 0;
  }

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
    { /* seed snap (vc3d vc_grow_seg_from_seed): the GUI seed is wherever
       * the camera focus landed — often a voxel or two off mid-sheet,
       * sometimes in inter-sheet air, and the first ~10 generations of
       * global solves lock in the sheet choice made from that start.
       * Walk to the nearest DT local minimum < 2.5 along the umbilicus
       * radial (or the coordinate axes) and start there instead. */
      double sv = td_tri(dt, t->cfg.seed, NULL);
      if (sv > 1.5) {
        double dirs[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        int ndir = 3;
        double cx2, cy2;
        if (t->uc && tr_uc_at(t, t->cfg.seed[2], &cx2, &cy2, NULL, NULL)) {
          double ux = t->cfg.seed[0] - cx2, uy = t->cfg.seed[1] - cy2;
          double rr = hypot(ux, uy);
          if (rr > 1e-6) {
            dirs[0][0] = ux / rr;
            dirs[0][1] = uy / rr;
            dirs[0][2] = 0;
            ndir = 1; /* radial crosses the wraps perpendicular-ish */
          }
        }
        double bs = 1e30, bq[3];
        for (int d = 0; d < ndir; d++) {
          double prev = 1e30, pprev = 1e30;
          for (double sm = -40.0; sm <= 40.0; sm += 1.0) {
            double q[3] = {t->cfg.seed[0] + sm * dirs[d][0],
                           t->cfg.seed[1] + sm * dirs[d][1],
                           t->cfg.seed[2] + sm * dirs[d][2]};
            double dv = td_tri(dt, q, NULL);
            if (pprev > prev && dv >= prev && prev < 2.5 &&
                fabs(sm - 1.0) < fabs(bs)) {
              bs = sm - 1.0;
              for (int a = 0; a < 3; a++)
                bq[a] = t->cfg.seed[a] + (sm - 1.0) * dirs[d][a];
            }
            pprev = prev;
            prev = dv;
          }
        }
        if (bs < 1e29) {
          printf("tracer: seed snapped %.1f vox onto the sheet (DT %.2f -> %.2f)\n",
                 fabs(bs), sv, td_tri(dt, bq, NULL));
          memcpy(t->cfg.seed, bq, sizeof bq);
        } else {
          printf("tracer: seed is not on a sheet (DT %.2f, no crossing within "
                 "40 vox) — aborting instead of growing garbage\n", sv);
          free(fringe);
          free(nfringe);
          free(cands);
          tr_pool_destroy(&pool);
          tr_env_flush(&cenv);
          free(cenv.hood);
          ng_close(&ng);
          td_close(dt);
          t->bnd_ct = NULL;
          if (ctv_ok) r3d_cpuvol_close(&ctv);
          r3d_cpuvol_close(&vol);
          pthread_mutex_lock(&t->mu);
          t->done = true;
          t->running = false;
          pthread_mutex_unlock(&t->mu);
          return NULL;
        }
      }
    }
    static const int off4[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    /* multi-wrap ribbons: one seed per radial sheet crossing */
    double cross_s[256];
    int ncr = 0, seed_rank = 0;
    uint32_t nwr = t->cfg.rib_rows ? t->cfg.rib_wraps : 1;
    if (nwr > 1 && t->uc) {
      double cx2, cy2;
      if (tr_uc_at(t, t->cfg.seed[2], &cx2, &cy2, NULL, NULL)) {
        double ux2 = t->cfg.seed[0] - cx2, uy2 = t->cfg.seed[1] - cy2;
        double rr2 = hypot(ux2, uy2);
        if (rr2 > 1e-6) {
          ux2 /= rr2;
          uy2 /= rr2;
          double range = 60.0 * nwr;
          double all_s[256];
          int nall = 0;
          double prev = 1e30, pprev = 1e30, last_s = -1e30;
          for (double sm = -range; sm <= range && nall < 256; sm += 1.0) {
            double q[3] = {t->cfg.seed[0] + sm * ux2, t->cfg.seed[1] + sm * uy2,
                           t->cfg.seed[2]};
            double dv = td_tri(dt, q, NULL);
            /* local DT minimum on a sheet, min separation 5 vox */
            if (pprev > prev && dv >= prev && prev < 2.5 && sm - 1.0 - last_s >= 5.0) {
              all_s[nall++] = sm - 1.0;
              last_s = sm - 1.0;
            }
            pprev = prev;
            prev = dv;
          }
          if (nall) { /* window of nwr crossings centered on the seed's */
            int i0 = 0;
            for (int q2 = 1; q2 < nall; q2++)
              if (fabs(all_s[q2]) < fabs(all_s[i0])) i0 = q2;
            int lo = i0 - (int)(nwr - 1) / 2;
            if (lo < 0) lo = 0;
            if (lo + (int)nwr > nall) lo = nall - (int)nwr;
            if (lo < 0) lo = 0;
            for (int q2 = lo; q2 < nall && ncr < (int)nwr; q2++)
              cross_s[ncr++] = all_s[q2];
            seed_rank = i0 - lo;
            if (seed_rank >= ncr) seed_rank = ncr ? ncr - 1 : 0;
          }
        }
      }
      if (ncr < 2) { /* not enough sheet crossings found — single wrap */
        ncr = 0;
        nwr = 1;
      } else {
        printf("tracer: %d sibling wraps seeded on radial crossings\n", ncr);
        nwr = (uint32_t)ncr;
      }
    }
    double ax[3] = {1, 0, 0}, ay[3] = {0, 1, 0}; /* seed quad axes */
    if (t->cfg.rib_rows) {
      /* ribbon: u along the in-plane spiral tangent, v along z — the
       * quad's orientation is what the straight losses propagate, and a
       * ribbon that starts tilted walks into its own z clamp */
      double cx2, cy2;
      if (tr_uc_at(t, t->cfg.seed[2], &cx2, &cy2, NULL, NULL)) {
        double ux2 = t->cfg.seed[0] - cx2, uy2 = t->cfg.seed[1] - cy2;
        double rr = hypot(ux2, uy2);
        if (rr > 1e-6) {
          ax[0] = -uy2 / rr;
          ax[1] = ux2 / rr;
          ax[2] = 0;
        }
      }
      ay[0] = ay[1] = 0;
      ay[2] = t->cfg.step; /* rows sit one z-plane (= step vox) apart */
    }
    double rdir[2] = {0, 0};
    if (ncr > 1 && t->uc) {
      double cx2, cy2;
      tr_uc_at(t, t->cfg.seed[2], &cx2, &cy2, NULL, NULL);
      double ux2 = t->cfg.seed[0] - cx2, uy2 = t->cfg.seed[1] - cy2;
      double rr2 = hypot(ux2, uy2);
      rdir[0] = ux2 / rr2;
      rdir[1] = uy2 / rr2;
    }
    int nblk = ncr > 1 ? ncr : 1;
    int rr = (int)t->cfg.rib_rows;
    pthread_mutex_lock(&t->mu);
    if (nblk > 1) /* spacer rows: never grown, never coupled */
      for (int b = 0; b + 1 < nblk; b++) {
        size_t row = (size_t)(b * (rr + 1) + rr);
        for (uint32_t i2 = 0; i2 < W; i2++) t->state[row * W + i2] = R3D_TR_FAIL;
      }
    for (int b = 0; b < nblk; b++) {
      double s_off = nblk > 1 ? cross_s[b] : 0.0;
      int y0b = t->cfg.rib_rows ? (nblk > 1 ? b * (rr + 1) + rr / 2 : y0) : y0;
      float w0 = nblk > 1 ? (float)(b - seed_rank) : 0.0f;
      for (int q = 0; q < 4; q++) {
        int i = x0 + off4[q][0], j = y0b + off4[q][1];
        size_t k = (size_t)j * W + (size_t)i;
        double *P = t->pos + k * 3;
        for (int a = 0; a < 3; a++)
          P[a] = t->cfg.seed[a] +
                 0.1 * (off4[q][0] * ax[a] + off4[q][1] * ay[a]);
        P[0] += s_off * rdir[0];
        P[1] += s_off * rdir[1];
        t->state[k] = R3D_TR_SET;
        if (t->gen_of) t->gen_of[k] = 1;
        t->conf[k] = 1.0f;
        t->wind[k] = w0;
        fringe[nf++] = (uint32_t)k;
      }
      t->nset += 4;
    }
    t->gen++;
    pthread_mutex_unlock(&t->mu);
    if (nblk > 1 && t->wf) /* perimeter seeding along each wrap's
                            * iso-contour (outline worked inward) */
      for (int b = 0; b < nblk; b++) {
        double pb[2] = {t->cfg.seed[0] + cross_s[b] * rdir[0],
                        t->cfg.seed[1] + cross_s[b] * rdir[1]};
        tr_rib_seed_contour(t, t->wf, cenv.dt, pb, b * (rr + 1) + rr / 2,
                            (float)(b - seed_rank), x0, fringe, &nf);
      }
    for (int b = 0; b < nblk; b++) {
      int y0b = nblk > 1 ? b * (rr + 1) + rr / 2 : y0;
      tr_local_opt(t, &cenv, x0, y0b, 8, 6, true);
    }
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
  uint32_t budget =
      (t->refine || t->ctsnap)
          ? 0
          : (t->cfg.max_ring > t->gens_done ? t->cfg.max_ring - t->gens_done : 0);
  uint32_t gens_run = 0;
  while (nf > 0 && gens_run < budget && !t->quit) {
    gens_run++;
    uint32_t generation = t->gens_done + gens_run;
    t->cur_gen = (uint16_t)(generation > 65535 ? 65535 : generation);
    bool global_opt = generation <= 10 && !resume;
    /* candidates: 8-neighborhood of the fringe */
    uint32_t nc = 0;
    for (uint32_t f = 0; f < nf; f++) {
      int i = (int)(fringe[f] % W), j = (int)(fringe[f] / W);
      for (int o = 0; o < 8; o++) {
        int ii = i + n8[o][0], jj = j + n8[o][1];
        int mv = t->cfg.rib_rows ? 0 : 2; /* ribbons use every row */
        if (ii < 2 || jj < mv || ii >= (int)W - 2 || jj >= (int)H - mv) continue;
        size_t k = (size_t)jj * W + (size_t)ii;
        if (t->state[k] != R3D_TR_EMPTY) continue;
        if (t->cfg.grow_dirs && !(t->cfg.grow_dirs & (1u << o))) continue;
        if (t->grow_mask && !t->grow_mask[k]) continue;
        if (generation > 30) {
          /* spur rule (vc3d L-shape, adapted): a candidate with a single
           * SET neighbour is a 1-cell spur the anti-fold term would
           * otherwise fight; left EMPTY it is re-offered once its
           * neighbourhood fills in. (The literal close-a-2x2 test blocks
           * every fresh front column in grow-from-seed mode — a new
           * column's candidates never have 3 placed quad-mates.) */
          int nbr2 = 0;
          for (int o2 = 0; o2 < 8 && nbr2 < 2; o2++)
            if (tr_valid(t, ii + n8[o2][0], jj + n8[o2][1])) nbr2++;
          if (nbr2 < 2) continue;
        }
        t->state[k] = R3D_TR_PROC; /* offered once, ever (vc3d) */
        cands[nc++] = (uint32_t)k;
      }
    }
    static _Atomic int ord_on = -1;
    int od2 = atomic_load_explicit(&ord_on, memory_order_relaxed);
    if (od2 < 0) {
      const char *ev = getenv("R3D_CAND_ORDER");
      od2 = ev ? atoi(ev) : 1;
      atomic_store(&ord_on, od2);
    }
    if (od2 && nc > 1) {
      /* certain territory first (vc3d CandidateOrdering): candidates with
       * more SET neighbours are solved first, so the front's schedule is
       * support-driven instead of raster-accidental — the source of the
       * same-build QC variance the periodic global solve pays to anneal */
      uint64_t *keys = malloc((size_t)nc * sizeof *keys);
      if (keys) {
        for (uint32_t c = 0; c < nc; c++) {
          int i2 = (int)(cands[c] % W), j2 = (int)(cands[c] / W);
          uint32_t nbr = 0;
          for (int o = 0; o < 8; o++)
            if (tr_valid(t, i2 + n8[o][0], j2 + n8[o][1])) nbr++;
          keys[c] = ((uint64_t)(8u - nbr) << 32) | cands[c];
        }
        qsort(keys, nc, sizeof *keys, tr_u64cmp);
        for (uint32_t c = 0; c < nc; c++) cands[c] = (uint32_t)keys[c];
        free(keys);
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
    if (t->cfg.rib_rows && nnew) {
      /* boundary pass: fronts stop at TRUE nothingness. Masked CT reads
       * exactly zero in the padding outside the scroll — that is proof.
       * Weak/missing predictions are not (papyrus often continues), so
       * the prediction-DT test is only the no-CT fallback. */
      uint32_t kept = 0;
      for (uint32_t f = 0; f < nnew; f++) {
        size_t k = nfringe[f];
        const double *P = t->pos + k * 3;
        bool dead = false;
        if (ctv_ok) {
          double v = r3d_cpuvol_tri(&ctv, ct_lv, P, NULL);
          if (v < ct_min) { /* try a slim neighborhood before declaring
                             * non-volume: traced points ride a voxel or
                             * two off mid-sheet routinely */
            for (int o = 0; o < 6 && v < ct_min; o++) {
              static const double off[6][3] = {{2, 0, 0}, {-2, 0, 0}, {0, 2, 0},
                                               {0, -2, 0}, {0, 0, 2}, {0, 0, -2}};
              double q[3] = {P[0] + off[o][0], P[1] + off[o][1], P[2] + off[o][2]};
              double v2 = r3d_cpuvol_tri(&ctv, ct_lv, q, NULL);
              if (v2 > v) v = v2;
            }
          }
          dead = v < ct_min;
        } else {
          dead = td_tri(cenv.dt, P, NULL) > 50.0;
        }
        if (dead) {
          pthread_mutex_lock(&t->mu);
          t->state[k] = R3D_TR_FAIL;
          if (t->nset) t->nset--;
          pthread_mutex_unlock(&t->mu);
        } else {
          nfringe[kept++] = (uint32_t)k;
        }
      }
      nnew = kept;
    }
    double co0 = tr_now();
    if (cenv.dt) /* build this ring's DT chunks in parallel first */
      td_prewarm(cenv.dt, t->pos, nfringe, nnew, W);
    for (uint32_t f = 0; f < nnew; f++) /* conf: coordinator only (DT) */
      tr_update_conf(t, &cenv, (int)(nfringe[f] % W), (int)(nfringe[f] / W));
    double co1 = tr_now();
    co_tm[1] += co1 - co0;
    if (t->cfg.wind_weight > 0 && t->uc && t->nset >= 200 &&
        (t->sp_om_meas == 0.0 || generation % 8 == 2))
      t->sp_om_meas = tr_omega_measure(t, cenv.dt);
    double co2 = tr_now();
    co_tm[2] += co2 - co1;
    tr_spiral_fit(t); /* refit the global spiral before the anneal so the
                       * winding prior joins the big solves */
    tr_spiral_flag(t);
    tr_don_register(t);
    tr_don_members(t); /* refresh donor uv membership + fold veto */
    tr_sfx_build(t); /* refresh the self-overlap index (pool is idle) */
    double co3 = tr_now();
    co_tm[3] += co3 - co2;
    tr_wind_relax(t, 30); /* winding follows the moved cells; werr = the
                           * wrong-wrap detector (clamps conf when on) */
    double co4 = tr_now();
    co_tm[4] += co4 - co3;
    tr_qc2(t, false); /* refresh the mesh QC counters for the panel */
    co_tm[5] += tr_now() - co4;
    if ((t->qc_nfoldc || t->qc_nkinkc) && !t->quit) {
      double co5 = tr_now();
      /* active fold repair: a fold that survives its own generation
       * becomes the parent geometry of the next ring. Anneal each fold's
       * disc with the staged schedule - data term off first so the fold
       * can cross back over the barrier holding it, then full weights.
       * (The placement-time gate was measured worse: deferral re-placed
       * cells from worse parents. Repair-in-place keeps the cell and
       * fixes it.) */
      static const double fsched[2][3] = {{0.3, 0.1, -1.0}, {1.0, 1.0, 1.0}};
      for (int pass = 0; pass < 2 && !t->quit; pass++) {
        cenv.ws_dist = fsched[pass][0];
        cenv.ws_straight = fsched[pass][1];
        cenv.ws_snap = fsched[pass][2];
        for (uint32_t pe = 0; pe < pool.nth; pe++) {
          pool.env[pe].ws_dist = fsched[pass][0];
          pool.env[pe].ws_straight = fsched[pass][1];
          pool.env[pe].ws_snap = fsched[pass][2];
        }
        for (uint32_t f = 0; f < t->qc_nfoldc && !t->quit; f++)
          tr_local_opt(t, &cenv, (int)(t->qc_fold_cell[f] % W),
                       (int)(t->qc_fold_cell[f] / W), 4, 3, true);
        for (uint32_t f = 0; f < t->qc_nkinkc && !t->quit; f++)
          tr_local_opt(t, &cenv, (int)(t->qc_kink_cell[f] % W),
                       (int)(t->qc_kink_cell[f] / W), 3, 3, true);
      }
      cenv.ws_dist = cenv.ws_straight = cenv.ws_snap = 0.0;
      for (uint32_t pe = 0; pe < pool.nth; pe++)
        pool.env[pe].ws_dist = pool.env[pe].ws_straight = pool.env[pe].ws_snap = 0.0;
      co_tm[6] += tr_now() - co5;
    }
    if (!t->quit) {
      /* zero tolerance: any fold the anneal could not flatten — or that
       * hid below the QC trust threshold — is CUT now. A fold never
       * survives a generation boundary, so the live view and every
       * downstream consumer only ever see fold-free geometry; the cells
       * retry from better parents as growth continues. */
      uint32_t nex = tr_fold_excise(t);
      if (nex) {
        printf("tracer: gen %u: excised %u fold cell%s the anneal could not "
               "flatten\n", generation, nex, nex == 1 ? "" : "s");
        tr_qc2(t, false); /* panel counters reflect the cut */
      }
    }
    tr_anc_assign(t); /* adopt/assign user anchors, then re-seat their
                       * neighborhoods so a correction shows immediately
                       * instead of waiting for the every-8th global solve */
    for (uint32_t a = 0; a < t->nanc; a++)
      if (t->anc_cell[a] >= 0)
        tr_local_opt(t, &cenv, (int)((uint32_t)t->anc_cell[a] % W),
                     (int)((uint32_t)t->anc_cell[a] / W), 3, 3, true);
    /* schedule: vc3d runs global solves only in the first 10 generations
     * (Ceres cost); our parallel sweeps are cheap enough to keep them
     * coming — every 8th generation, forever. This anneals away the
     * candidate-order nondeterminism that tears weak outskirt regions
     * (same-build 60-gen QC swung 0.06%..9.7% v-edges >30 vox). */
    if (!resume && generation % 8 == 0) {
      if (t->cfg.rib_rows && nnew) {
        /* ribbon: whole-grid anneals are O(len) x O(len/8) — solve a
         * sliding window around each growth front instead (Lasagna
         * opt_window / vc3d sliding_w) */
        int umin = (int)W, umax = 0;
        for (uint32_t f = 0; f < nnew; f++) {
          int iu = (int)(nfringe[f] % W);
          if (iu < umin) umin = iu;
          if (iu > umax) umax = iu;
        }
        int win = 48;
        tr_local_opt(t, &cenv, umin, y0, win, global_opt ? 6 : 3, true);
        if (umax - umin > win)
          tr_local_opt(t, &cenv, umax, y0, win, global_opt ? 6 : 3, true);
      } else {
        tr_local_opt(t, &cenv, x0, y0, (int)W + (int)H, global_opt ? 6 : 3, true);
      }
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
    if (t->don && nnew == 0 && t->inl_th > 2.0 && gens_run < budget && !t->quit) {
      /* fringe starved under the consensus gate: anneal the threshold and
       * reseed the fringe from every boundary SET cell (vc3d curr_best_
       * inl_th schedule) instead of terminating */
      t->inl_th = t->inl_th > 12.0 ? t->inl_th - 2.0 : (t->inl_th > 4.0 ? t->inl_th - 2.0 : 2.0);
      nnew = 0;
      for (uint32_t jr = 1; jr + 1 < H; jr++)
        for (uint32_t ir = 1; ir + 1 < W; ir++) {
          size_t kr = (size_t)jr * W + ir;
          if (t->state[kr] != R3D_TR_SET) continue;
          bool edge2 = false;
          for (int o = 0; o < 8 && !edge2; o++)
            if (t->state[(size_t)((int)jr + n8[o][1]) * W + (size_t)((int)ir + n8[o][0])] == R3D_TR_EMPTY)
              edge2 = true;
          if (edge2) nfringe[nnew++] = (uint32_t)kr;
        }
      printf("tracer: fusion gate annealed to %.0f, fringe reseeded (%u cells)\n",
             t->inl_th, nnew);
    }
    memcpy(fringe, nfringe, (size_t)nnew * sizeof *fringe);
    nf = nnew;
    pthread_mutex_lock(&t->mu);
    t->ring = generation;
    t->gen++;
    pthread_mutex_unlock(&t->mu);
  }
  t->gens_done += gens_run;
  if (t->refine && !t->quit) {
    /* solve-only pass (no growth): fix the existing grid through the user
     * anchors. Anchor neighborhoods are annealed hardest, re-assigning
     * ownership between passes so the pull follows the sheet as it crosses
     * to the right one; the global polish below then smooths the seams. */
    tr_spiral_fit(t);
    tr_spiral_flag(t);
    tr_sfx_build(t);
    tr_anc_assign(t);
    tr_reopt_snapshot(t); /* tangential position memory (G10a): snapshot
                           * pos + normals so the anneal can move cells
                           * across wraps but not along the sheet */
    /* staged weight relaxation (G9, vc3d correction schedule): a
     * correction asking the sheet to move a wrap over must cross a
     * barrier — the snap term holds it on its current polyline and
     * DIST/STRAIGHT resist the transition stretch. At full weights the
     * refine converges straight back into the wrong-sheet basin. */
    static const double sched[3][3] = {/* dist, straight, snap */
                                       {0.3, 0.1, -1.0}, /* -1 = off */
                                       {0.3, 0.1, 1.0},
                                       {1.0, 1.0, 1.0}};
    int rad = 8; /* widen to cover the anchor spread */
    for (uint32_t a = 0; a < t->nanc; a++)
      for (uint32_t b = a + 1; b < t->nanc; b++) {
        if (t->anc_cell[a] < 0 || t->anc_cell[b] < 0) continue;
        int dai = abs((int)((uint32_t)t->anc_cell[a] % W) -
                      (int)((uint32_t)t->anc_cell[b] % W));
        int daj = abs((int)((uint32_t)t->anc_cell[a] / W) -
                      (int)((uint32_t)t->anc_cell[b] / W));
        int sp = (dai > daj ? dai : daj) / 2;
        if (8 + sp > rad) rad = 8 + sp;
      }
    if (rad > 48) rad = 48;
    for (int pass = 0; pass < 3 && !t->quit; pass++) {
      cenv.ws_dist = sched[pass][0];
      cenv.ws_straight = sched[pass][1];
      cenv.ws_snap = sched[pass][2];
      for (uint32_t pe = 0; pe < pool.nth; pe++) {
        pool.env[pe].ws_dist = sched[pass][0];
        pool.env[pe].ws_straight = sched[pass][1];
        pool.env[pe].ws_snap = sched[pass][2];
      }
      for (uint32_t a = 0; a < t->nanc && !t->quit; a++)
        if (t->anc_cell[a] >= 0)
          tr_local_opt(t, &cenv, (int)((uint32_t)t->anc_cell[a] % W),
                       (int)((uint32_t)t->anc_cell[a] / W), rad, 6, true);
      tr_anc_assign(t);
      pthread_mutex_lock(&t->mu);
      t->gen++; /* live view: show each pass */
      pthread_mutex_unlock(&t->mu);
    }
    cenv.ws_dist = cenv.ws_straight = cenv.ws_snap = 0.0; /* back to 1.0 */
    for (uint32_t pe = 0; pe < pool.nth; pe++)
      pool.env[pe].ws_dist = pool.env[pe].ws_straight = pool.env[pe].ws_snap = 0.0;
    t->reopt_on = false;
    free(t->reopt_pos);
    free(t->reopt_nrm);
    t->reopt_pos = NULL;
    t->reopt_nrm = NULL;
    t->refine = false;
  }
  if (t->ctsnap && !t->quit) {
    /* CT edge snap (solve-only): the traced sheet follows the predictions,
     * which ride NEAR the papyrus but not on it. Pull every trusted cell
     * onto the actual papyrus/void interface in the raw CT — bsurf's snap
     * rule as a soft LM term: move only along the frozen normal, never
     * farther than ctsnap_dist, onto the nearest papyrus->void crossing of
     * ctsnap_cut (one face for the whole sheet: the normal field is
     * consistent, so every cell lands on the same side). Cells with no
     * edge in reach get no pull and ride the smoothness terms. Retarget
     * between passes so edges found from settled neighbours extend the
     * snap into cells that started out of range. */
    if (!ctv_ok) {
      printf("tracer: CT snap requested but no raw CT tree — skipped\n");
    } else if (tr_reopt_snapshot(t)) {
      size_t n2 = (size_t)W * H;
      free(t->ctsnap_tgt);
      t->ctsnap_tgt = malloc(n2 * sizeof *t->ctsnap_tgt);
      const double cut = t->ctsnap_cut, reach = t->ctsnap_dist;
      for (int pass = 0; pass < 3 && t->ctsnap_tgt && !t->quit; pass++) {
        uint32_t hit = 0, miss = 0;
        double moved = 0.0;
        for (size_t k = 0; k < n2; k++) {
          t->ctsnap_tgt[k] = 1e30f; /* sentinel: no edge (NaN is UB here) */
          if (t->state[k] != R3D_TR_SET) continue;
          const float *nrf = t->reopt_nrm + k * 3;
          double n0[3] = {(double)nrf[0], (double)nrf[1], (double)nrf[2]};
          if (n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2] <= 0.25) continue;
          const double *p0 = t->reopt_pos + k * 3;
          const double *pc = t->pos + k * 3;
          double dn = (pc[0] - p0[0]) * n0[0] + (pc[1] - p0[1]) * n0[1] +
                      (pc[2] - p0[2]) * n0[2];
          double best = 1e30, vp = -1.0;
          for (double s = dn - reach; s <= dn + reach + 1e-9; s += 0.5) {
            double q[3] = {p0[0] + s * n0[0], p0[1] + s * n0[1],
                           p0[2] + s * n0[2]};
            double v = r3d_cpuvol_tri(&ctv, 0, q, NULL);
            if (vp >= cut && v < cut) { /* papyrus behind, void ahead */
              double sc = s - 0.5 + 0.5 * (vp - cut) / (vp - v);
              if (fabs(sc - dn) < fabs(best - dn)) best = sc;
            }
            vp = v;
          }
          if (best < 1e29) {
            t->ctsnap_tgt[k] = (float)best;
            moved += fabs(best - dn);
            hit++;
          } else {
            miss++;
          }
        }
        printf("tracer: CT snap pass %d — edge for %u/%u cells, mean pull "
               "%.2f vox\n",
               pass + 1, hit, hit + miss, hit ? moved / hit : 0.0);
        tr_local_opt(t, &cenv, x0, y0, (int)W + (int)H, 4, true);
        pthread_mutex_lock(&t->mu);
        t->gen++; /* live view: show each pass */
        pthread_mutex_unlock(&t->mu);
      }
    }
    /* targets + tangential pin stay live through the final polish below:
     * released after, or the polish's prediction data term (the sheet we
     * just left) would drag the cells straight back off the CT edge */
  }
  /* final polish: one bounded pass so late cells see settled neighbors */
  if (!t->quit) tr_local_opt(t, &cenv, x0, y0, (int)W + (int)H, 4, true);
  if (!t->quit) tr_inpaint(t, &cenv); /* re-solve enclosed holes with the
                                       * real losses (G8) */
  if (t->ctsnap) {
    free(t->ctsnap_tgt);
    t->ctsnap_tgt = NULL;
    t->reopt_on = false;
    free(t->reopt_pos);
    free(t->reopt_nrm);
    t->reopt_pos = NULL;
    t->reopt_nrm = NULL;
    t->ctsnap = false;
  }
  tr_wind_relax(t, 30);
  if (!t->quit) { /* zero tolerance at finish too: the refine/ctsnap/
      * inpaint paths do not pass through the generation loop's per-ring
      * excision, and inpaint re-seating can fold in pathological spots */
    tr_sfx_build(t); /* the overlap phase needs post-polish positions */
    uint32_t nex = tr_fold_excise(t);
    if (nex)
      printf("tracer: final fold excision cut %u cell%s\n", nex,
             nex == 1 ? "" : "s");
  }
  /* final QC: a >90-degree fold is never legitimate papyrus � surviving
   * folds are marked untrusted so the save writes an honest hole there
   * instead of doubled-back geometry */
  tr_qc2(t, true);
  printf("tracer: QC %u folds, %u kinks, twist %.2f | area %.3g vx2, "
         "bbox %ux%u, fill %.2f, holes %.3f, slant p95 %.3f\n",
         t->qc_folds, t->qc_kinks, (double)t->qc_twist, t->qc_area_vx2,
         t->qc_bbox[2] >= t->qc_bbox[0] ? t->qc_bbox[2] - t->qc_bbox[0] + 1 : 0,
         t->qc_bbox[3] >= t->qc_bbox[1] ? t->qc_bbox[3] - t->qc_bbox[1] + 1 : 0,
         (double)t->qc_fill, (double)t->qc_hole, (double)t->qc_slant_p95);
  if (t->uc)
    printf("tracer: QC wrap-jump fraction %.4f, werr p95 %.3f\n",
           (double)t->qc_wrap_frac, (double)t->qc_werr_p95);
  if (t->don)
    printf("tracer: QC donor mean %.2f rms %.2f p95 %.2f vox, coverage %.2f\n",
           (double)t->qc_don_mean, (double)t->qc_don_rms, (double)t->qc_don_p95,
           (double)t->qc_don_cov);
  {
    uint64_t qb = atomic_load(&ng_qmax_binds), hb = atomic_load(&ng_hoodseg_binds);
    printf("tracer: ng loads %llu evicts %llu peak %.2f GB eget hit %llu miss %llu\n",
           (unsigned long long)atomic_load(&ng_loads),
           (unsigned long long)atomic_load(&ng_evicts),
           (double)atomic_load(&ng_bytes_peak) / 1073741824.0,
           (unsigned long long)atomic_load(&ng_eget_hit),
           (unsigned long long)atomic_load(&ng_eget_miss));
    {
      uint64_t hbld = atomic_load(&ng_hood_builds);
      printf("tracer: hood hits %llu builds %llu segs mean %.0f max %llu\n",
             (unsigned long long)atomic_load(&ng_hood_hits),
             (unsigned long long)hbld,
             hbld ? (double)atomic_load(&ng_hood_seg_sum) / (double)hbld : 0.0,
             (unsigned long long)atomic_load(&ng_hood_seg_max));
    }
    printf("tracer: ng bytes loaded: blob %.2f GB, indexes %.2f GB\n",
           (double)atomic_load(&ng_blob_bytes) / 1073741824.0,
           (double)atomic_load(&ng_idx_bytes) / 1073741824.0);
    if (atomic_load(&tr_excise_total))
      printf("tracer: %llu fold cell(s) excised across the trace\n",
             (unsigned long long)atomic_load(&tr_excise_total));
    printf("tracer: coordinator serial: conf %.1fs omega %.1fs fit/don/sfx %.1fs "
           "windrelax %.1fs qc2 %.1fs foldrepair %.1fs\n",
           co_tm[1], co_tm[2], co_tm[3], co_tm[4], co_tm[5], co_tm[6]);
    if (qb || hb)
      printf("tracer: ng caps bound: NG_QMAX %llu, NG_HOODSEG %llu times\n",
             (unsigned long long)qb, (unsigned long long)hb);
  }
  free(fringe);
  free(nfringe);
  free(cands);
  tr_pool_destroy(&pool);
  tr_env_flush(&cenv);
  free(cenv.hood);
  printf("tracer: finished at generation %u with %u point%s (level L%u%s)\n", t->ring,
         t->nset, t->nset == 1 ? "" : "s", t->cfg.level,
         ng.active ? ", normal grids" : "");
  for (uint32_t a = 0; a < t->nanc; a++) { /* how well each anchor held */
    if (t->anc_cell[a] < 0) {
      printf("tracer: anchor %u never captured (no cell within %.0f vox)\n", a,
             TR_ANC_CAPTURE * t->cfg.step);
      continue;
    }
    const double *q = t->pos + (size_t)t->anc_cell[a] * 3;
    const double *P = t->anc + (size_t)a * 3;
    double dx = q[0] - P[0], dy = q[1] - P[1], dz = q[2] - P[2];
    /* the pull is sheet-normal only (the cell slides tangentially), so
     * report the normal component separately — that is the "does the
     * sheet pass through the point" number */
    double nd = -1.0;
    int ai = (int)((uint32_t)t->anc_cell[a] % t->W),
        aj = (int)((uint32_t)t->anc_cell[a] / t->W);
    if (tr_valid(t, ai - 1, aj) && tr_valid(t, ai + 1, aj) &&
        tr_valid(t, ai, aj - 1) && tr_valid(t, ai, aj + 1)) {
      const double *pu0 = t->pos + ((size_t)aj * t->W + (size_t)ai - 1) * 3;
      const double *pu1 = t->pos + ((size_t)aj * t->W + (size_t)ai + 1) * 3;
      const double *pv0 = t->pos + ((size_t)(aj - 1) * t->W + (size_t)ai) * 3;
      const double *pv1 = t->pos + ((size_t)(aj + 1) * t->W + (size_t)ai) * 3;
      double eu[3], ev[3], nr[3];
      for (int c2 = 0; c2 < 3; c2++) {
        eu[c2] = pu1[c2] - pu0[c2];
        ev[c2] = pv1[c2] - pv0[c2];
      }
      nr[0] = eu[1] * ev[2] - eu[2] * ev[1];
      nr[1] = eu[2] * ev[0] - eu[0] * ev[2];
      nr[2] = eu[0] * ev[1] - eu[1] * ev[0];
      double l = sqrt(nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]);
      if (l > 1e-9) nd = fabs((dx * nr[0] + dy * nr[1] + dz * nr[2]) / l);
    }
    if (nd >= 0.0)
      printf("tracer: anchor %u off-sheet %.1f vox (3D %.1f, rest tangential)\n",
             a, nd, sqrt(dx * dx + dy * dy + dz * dz));
    else
      printf("tracer: anchor %u final distance %.1f vox\n", a,
             sqrt(dx * dx + dy * dy + dz * dz));
  }
  if (t->sp_valid)
    printf("tracer: spiral fit omega %.2f vox/winding, rms %.2f vox\n", t->sp_omega,
           t->sp_rms);
  printf("tracer: %.1fs total | place %.1fs lopt %.1fs (ncp %.1fs hood %.1fs) "
         "conf %.1fs gridfetch %.1fs\n",
         tr_now() - tr_t_start, TR_TM(0), TR_TM(1), TR_TM(4), TR_TM(5), TR_TM(2),
         TR_TM(3));
  ng_close(&ng);
  td_close(dt);
  t->bnd_ct = NULL; /* worker-owned sampler dies with the worker */
  if (t->mask_once) { /* reopt region regrown: lift the confinement */
    free(t->grow_mask);
    t->grow_mask = NULL;
    t->mask_once = false;
  }
  if (ctv_ok) r3d_cpuvol_close(&ctv);
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
  for (uint32_t a = 0; a < R3D_TR_MAX_ANCHORS; a++) t->anc_cell[a] = -1;
  t->inl_th = 20.0; /* fusion consensus gate, annealed on fringe starve */
  t->cfg = *cfg;
  if (t->cfg.max_ring < 4) t->cfg.max_ring = 4;
  if (t->cfg.max_ring > (t->cfg.rib_rows ? 40000u : 400u)) 
    t->cfg.max_ring = t->cfg.rib_rows ? 40000u : 400u;
  if (t->cfg.step < 1.0) t->cfg.step = 20.0;
  snprintf(t->root, sizeof t->root, "%s", pred_root);
  if (t->cfg.rib_rows) { /* ribbon: long in u, a few rows of v */
    if (t->cfg.rib_rows < 4) t->cfg.rib_rows = 4;
    if (t->cfg.rib_wraps < 1) t->cfg.rib_wraps = 1;
    if (t->cfg.rib_wraps > 256) t->cfg.rib_wraps = 256;
    t->W = 2 * t->cfg.max_ring + 10;
    t->H = t->cfg.rib_wraps * t->cfg.rib_rows + (t->cfg.rib_wraps - 1);
  } else {
    t->W = t->H = 2 * t->cfg.max_ring + 50;
  }
  t->pos = calloc((size_t)t->W * t->H * 3, sizeof *t->pos);
  t->state = calloc((size_t)t->W * t->H, 1);
  t->conf = calloc((size_t)t->W * t->H, sizeof *t->conf);
  t->wind = calloc((size_t)t->W * t->H, sizeof *t->wind);
  t->werr = calloc((size_t)t->W * t->H, sizeof *t->werr);
  t->gen_of = calloc((size_t)t->W * t->H, sizeof *t->gen_of);
  t->rng = 0x1234567u;
  if (!t->pos || !t->state || !t->conf || !t->wind || !t->werr || !t->gen_of) {
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
      t->dcell_id = malloc((size_t)t->W * t->H);
      t->dcell_uv = calloc((size_t)t->W * t->H * 2, sizeof *t->dcell_uv);
      if (!t->dsup || !t->dcell_id || !t->dcell_uv) {
        r3d_tracer_free(t);
        return -1;
      }
      memset(t->dcell_id, 0xff, (size_t)t->W * t->H); /* -1 = none */
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

/* ==================== tracer.json state sidecar ====================
 * tifxyz alone is a display artifact: it carries a CROPPED grid, no
 * solver configuration, no anchors and no per-vertex confidence, so a
 * load could only guess and the guess made resume a no-op. r3d_tracer_save
 * writes <dir>/tracer.json beside the planes with everything needed to
 * reconstruct the solver state; the object is deliberately FLAT with
 * unique keys so strstr + sscanf is a correct parser (no nesting to get
 * lost in) and a missing/short/garbage sidecar simply degrades to the
 * legacy assumed-configuration path. */
#define TR_STATE_FMT "r3d-tracer-state"
#define TR_STATE_VER 1
#define TR_STATE_MAX 262144       /* sidecar byte cap (anchors bound it) */
#define TR_LOAD_SLACK 16u         /* generations of grow budget assumed for
                                   * a sidecar-less load, so resume is not
                                   * silently a no-op */
#define TR_MAX_CELLS (1ull << 26) /* grid-frame sanity cap */

/* value text after "key": , or NULL */
static const char *tr_js_find(const char *buf, const char *key) {
  char pat[64];
  int k = snprintf(pat, sizeof pat, "\"%s\"", key);
  if (k <= 0 || (size_t)k >= sizeof pat) return NULL;
  const char *p = strstr(buf, pat);
  if (!p) return NULL;
  p = strchr(p + k, ':');
  return p ? p + 1 : NULL;
}

static bool tr_js_num(const char *buf, const char *key, double *out) {
  const char *p = tr_js_find(buf, key);
  double v;
  if (!p || sscanf(p, " %lf", &v) != 1) return false;
  *out = v;
  return true;
}

/* accepts only finite values inside [0, max] */
static bool tr_js_u32(const char *buf, const char *key, uint32_t *out,
                      uint32_t max) {
  double v;
  if (!tr_js_num(buf, key, &v) || !(v >= 0.0) || !(v <= (double)max))
    return false;
  *out = (uint32_t)v;
  return true;
}

/* finite doubles only: a sentinel/garbage bound must not become a clamp */
static bool tr_js_fin(const char *buf, const char *key, double *out) {
  double v;
  if (!tr_js_num(buf, key, &v) || !(v > -1e29) || !(v < 1e29)) return false;
  *out = v;
  return true;
}

static bool tr_js_bool(const char *buf, const char *key) {
  const char *p = tr_js_find(buf, key);
  while (p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  return p && strncmp(p, "true", 4) == 0;
}

/* only \\ and \" are ever produced by the writer for printable text;
 * any other escape keeps its payload character (control characters are
 * written as \uXXXX and degrade to that literal text — paths do not
 * contain them) */
static bool tr_js_str(const char *buf, const char *key, char *out, size_t cap) {
  const char *p = tr_js_find(buf, key);
  while (p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  if (!p || *p != '"' || cap == 0) return false;
  p++;
  size_t o = 0;
  while (*p && *p != '"' && o + 1 < cap) {
    if (*p == '\\' && p[1]) p++;
    out[o++] = *p++;
  }
  out[o] = 0;
  return *p == '"';
}

/* read a whole small file; NULL when absent/oversized */
static char *tr_slurp(const char *dir, const char *name, size_t cap) {
  char path[1200];
  int k = snprintf(path, sizeof path, "%s/%s", dir, name);
  if (k <= 0 || (size_t)k >= sizeof path) return NULL;
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  char *b = malloc(cap + 1);
  if (!b) {
    fclose(f);
    return NULL;
  }
  size_t rd = fread(b, 1, cap, f);
  bool over = fread(path, 1, 1, f) == 1; /* larger than the cap: refuse */
  fclose(f);
  if (over) {
    free(b);
    return NULL;
  }
  b[rd] = 0;
  return b;
}

/* probe a plane's dimensions */
static bool tr_plane_dims(const char *path, uint32_t *w, uint32_t *h) {
  TIFF *tf = TIFFOpen(path, "r");
  if (!tf) return false;
  uint32_t tw = 0, th2 = 0;
  TIFFGetField(tf, TIFFTAG_IMAGEWIDTH, &tw);
  TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &th2);
  TIFFClose(tf);
  if (!tw || !th2) return false;
  *w = tw;
  *h = th2;
  return true;
}

int r3d_tracer_load(r3d_tracer *t, const char *dir, const char *pred_root) {
  char path[1200];
  snprintf(path, sizeof path, "%s/x.tif", dir);
  uint32_t w, h;
  if (!tr_plane_dims(path, &w, &h)) return -1;
  memset(t, 0, sizeof *t);
  for (uint32_t a = 0; a < R3D_TR_MAX_ANCHORS; a++) t->anc_cell[a] = -1;
  snprintf(t->root, sizeof t->root, "%s", pred_root ? pred_root : "");
  t->cfg.step = 20.0;
  t->cfg.level = 1;
  t->cfg.thresh = 0.35f;
  t->cfg.max_ring = w > 50 ? (w - 50) / 2 : 4;
  { /* scale from meta.json (scale = 1/step) */
    snprintf(path, sizeof path, "%s/meta.json", dir);
    FILE *f = fopen(path, "r");
    if (f) {
      char buf[4096];
      size_t rd = fread(buf, 1, sizeof buf - 1, f);
      buf[rd] = 0;
      fclose(f);
      const char *sc = strstr(buf, "\"scale\"");
      double sv;
      if (sc && sscanf(sc, "\"scale\": [ %lf", &sv) == 1 && sv > 1e-6)
        t->cfg.step = 1.0 / sv;
      else if (sc) {
        const char *br = strchr(sc, '[');
        if (br && sscanf(br + 1, " %lf", &sv) == 1 && sv > 1e-6)
          t->cfg.step = 1.0 / sv;
      }
    }
  }
  /* ---- versioned state sidecar: the only faithful source of the solver
   * configuration, the uncropped grid frame, anchors and confidence ---- */
  char *sj = tr_slurp(dir, "tracer.json", TR_STATE_MAX);
  bool side = false;
  uint32_t GW = w, GH = h, oi = 0, oj = 0;
  if (sj) {
    char fmt[64] = {0};
    uint32_t ver = 0;
    if (tr_js_str(sj, "format", fmt, sizeof fmt) &&
        strcmp(fmt, TR_STATE_FMT) == 0 &&
        tr_js_u32(sj, "version", &ver, 4096) && ver >= 1 &&
        ver <= TR_STATE_VER) {
      side = true;
    } else {
      printf("tracer: %s/tracer.json is not a supported state sidecar "
             "(ignored)\n",
             dir);
      free(sj);
      sj = NULL;
    }
  }
  if (side) {
    double d;
    uint32_t u;
    if (tr_js_num(sj, "step", &d) && d > 1e-6 && d < 1e6) t->cfg.step = d;
    if (tr_js_num(sj, "thresh", &d) && d >= 0.0 && d <= 1.0)
      t->cfg.thresh = (float)d;
    if (tr_js_u32(sj, "max_ring", &u, 40000u) && u >= 4) t->cfg.max_ring = u;
    if (tr_js_u32(sj, "level", &u, 16u)) t->cfg.level = u;
    if (tr_js_u32(sj, "rib_rows", &u, 4096u)) t->cfg.rib_rows = u;
    if (tr_js_u32(sj, "rib_wraps", &u, 256u)) t->cfg.rib_wraps = u;
    if (tr_js_u32(sj, "grow_dirs", &u, 255u)) t->cfg.grow_dirs = (uint8_t)u;
    if (tr_js_u32(sj, "max_threads", &u, 4096u)) t->cfg.max_threads = u;
    if (tr_js_fin(sj, "seed_x", &d)) t->cfg.seed[0] = d;
    if (tr_js_fin(sj, "seed_y", &d)) t->cfg.seed[1] = d;
    if (tr_js_fin(sj, "seed_z", &d)) t->cfg.seed[2] = d;
    if (tr_js_fin(sj, "z_min", &d)) t->cfg.z_min = d;
    if (tr_js_fin(sj, "z_max", &d)) t->cfg.z_max = d;
    if (tr_js_fin(sj, "x_min", &d)) t->cfg.x_min = d;
    if (tr_js_fin(sj, "x_max", &d)) t->cfg.x_max = d;
    if (tr_js_fin(sj, "y_min", &d)) t->cfg.y_min = d;
    if (tr_js_fin(sj, "y_max", &d)) t->cfg.y_max = d;
    if (tr_js_fin(sj, "ct_min", &d)) t->cfg.ct_min = d;
    if (tr_js_fin(sj, "wind_weight", &d) && d >= 0.0) t->cfg.wind_weight = d;
    if (tr_js_fin(sj, "tear_lim", &d) && d >= 0.0) t->tear_lim = d;
    if (tr_js_fin(sj, "sp_om_meas", &d) && d >= 0.0) t->sp_om_meas = d;
    if (tr_js_fin(sj, "vdim_x", &d) && d >= 0.0) t->vdim[0] = d;
    if (tr_js_fin(sj, "vdim_y", &d) && d >= 0.0) t->vdim[1] = d;
    if (tr_js_fin(sj, "vdim_z", &d) && d >= 0.0) t->vdim[2] = d;
    tr_js_str(sj, "ct_root", t->cfg.ct_root, sizeof t->cfg.ct_root);
    if (!t->root[0]) tr_js_str(sj, "pred_root", t->root, sizeof t->root);
    { /* anchors: flat [x,y,z,...] */
      const char *ap = tr_js_find(sj, "anchors");
      if (ap) ap = strchr(ap, '[');
      if (ap) {
        ap++;
        uint32_t na = 0;
        while (na < R3D_TR_MAX_ANCHORS) {
          double v[3];
          int used = 0, got = 0;
          const char *q = ap;
          for (int c = 0; c < 3; c++) {
            if (sscanf(q, " %lf%n", &v[c], &used) != 1) break;
            q += used;
            while (*q == ' ' || *q == ',' || *q == '\n' || *q == '\t') q++;
            got++;
          }
          if (got != 3) break;
          for (int c = 0; c < 3; c++) t->anc[na * 3 + (uint32_t)c] = v[c];
          na++;
          ap = q;
        }
        t->nanc = na;
      }
    }
    { /* uncropped grid frame: restore the ORIGINAL solver lattice and
       * drop the crop back at its offset, so max_ring and the grid stay
       * consistent and a later grow re-centers correctly */
      uint32_t fw = 0, fh = 0, ai = 0, aj = 0;
      if (tr_js_u32(sj, "full_w", &fw, 1u << 22) &&
          tr_js_u32(sj, "full_h", &fh, 1u << 22) &&
          tr_js_u32(sj, "offset_i", &ai, 1u << 22) &&
          tr_js_u32(sj, "offset_j", &aj, 1u << 22) && fw >= w && fh >= h &&
          ai <= fw - w && aj <= fh - h &&
          (uint64_t)fw * (uint64_t)fh <= TR_MAX_CELLS) {
        GW = fw;
        GH = fh;
        oi = ai;
        oj = aj;
      } else if (fw || fh) {
        printf("tracer: sidecar grid frame %ux%u+%u+%u is unusable; keeping "
               "the cropped %ux%u frame\n",
               fw, fh, ai, aj, w, h);
      }
    }
  }
  t->W = GW;
  t->H = GH;
  uint64_t n = (uint64_t)GW * GH;
  t->pos = calloc(n * 3, sizeof *t->pos);
  t->state = calloc(n, 1);
  t->conf = calloc(n, sizeof *t->conf);
  t->wind = calloc(n, sizeof *t->wind);
  t->werr = calloc(n, sizeof *t->werr);
  t->gen_of = calloc(n, sizeof *t->gen_of);
  if (!t->pos || !t->state || !t->conf || !t->wind || !t->werr || !t->gen_of) {
    free(sj);
    r3d_tracer_free(t);
    return -1;
  }
  float *px = NULL, *py = NULL, *pz = NULL, *pw = NULL, *pg = NULL, *pc = NULL;
  snprintf(path, sizeof path, "%s/x.tif", dir);
  px = tr_read_plane(path, w, h);
  snprintf(path, sizeof path, "%s/y.tif", dir);
  py = tr_read_plane(path, w, h);
  snprintf(path, sizeof path, "%s/z.tif", dir);
  pz = tr_read_plane(path, w, h);
  snprintf(path, sizeof path, "%s/winding.tif", dir);
  pw = tr_read_plane(path, w, h); /* optional */
  snprintf(path, sizeof path, "%s/generations.tif", dir);
  pg = tr_read_plane(path, w, h); /* optional */
  if (side && tr_js_bool(sj, "confidence")) {
    snprintf(path, sizeof path, "%s/confidence.tif", dir);
    pc = tr_read_plane(path, w, h); /* optional */
  }
  if (!px || !py || !pz) {
    free(px);
    free(py);
    free(pz);
    free(pw);
    free(pg);
    free(pc);
    free(sj);
    r3d_tracer_free(t);
    return -1;
  }
  uint32_t maxgen = 1;
  for (uint32_t j = 0; j < h; j++)
    for (uint32_t i = 0; i < w; i++) {
      size_t sk = (size_t)j * w + i;
      if (px[sk] <= 0.0f) continue; /* tifxyz invalid encoding */
      size_t k = (size_t)(oj + j) * GW + (oi + i);
      t->pos[k * 3 + 0] = (double)px[sk];
      t->pos[k * 3 + 1] = (double)py[sk];
      t->pos[k * 3 + 2] = (double)pz[sk];
      t->state[k] = R3D_TR_SET;
      t->conf[k] = pc && pc[sk] >= 0.0f && pc[sk] <= 1.0f ? pc[sk] : 1.0f;
      t->wind[k] = pw && pw[sk] > -1e29f ? pw[sk] : 0.0f;
      uint16_t g = pg && pg[sk] > 0.0f && pg[sk] < 65535.0f ? (uint16_t)pg[sk]
                                                            : (uint16_t)1;
      t->gen_of[k] = g;
      if (g > maxgen) maxgen = g;
      t->nset++;
    }
  free(px);
  free(py);
  free(pz);
  free(pw);
  free(pg);
  free(pc);
  t->gens_done = maxgen;
  if (side) {
    uint32_t v;
    /* the true generation count: cropping can hide the outermost ring */
    if (tr_js_u32(sj, "gens_done", &v, 1u << 20) && v >= maxgen)
      t->gens_done = v;
  } else if (t->cfg.max_ring < maxgen + TR_LOAD_SLACK) {
    /* No sidecar: max_ring inferred from the CROPPED width is routinely
     * below the generations already grown, which made the worker's
     * budget (max_ring - gens_done) zero and resume a silent no-op.
     * Assume a resumable budget and say so. */
    uint32_t want = maxgen + TR_LOAD_SLACK;
    if (want > 40000u) want = 40000u;
    t->cfg.max_ring = want;
  }
  t->ring = t->gens_done;
  pthread_mutex_init(&t->mu, NULL);
  r3d_umbilicus_init(&t->umb);
  t->done = true; /* loaded = a finished, stopped trace */
  printf("tracer: loaded %s (%ux%u", dir, w, h);
  if (GW != w || GH != h) printf(" in a %ux%u frame at +%u+%u", GW, GH, oi, oj);
  printf(", %u points, gen %u/%u)\n", t->nset, t->gens_done, t->cfg.max_ring);
  if (side)
    printf("tracer: state sidecar restored: step %.3f level %u thresh %.2f "
           "max_ring %u, %u anchor%s, confidence %s\n",
           t->cfg.step, t->cfg.level, (double)t->cfg.thresh, t->cfg.max_ring,
           t->nanc, t->nanc == 1 ? "" : "s", pc ? "restored" : "assumed 1.0");
  else
    printf("tracer: no tracer.json sidecar - IMPORT ONLY, configuration is "
           "ASSUMED: step %.3f level %u thresh %.2f max_ring %u (grow budget "
           "%u generations), confidence 1.0, no anchors\n",
           t->cfg.step, t->cfg.level, (double)t->cfg.thresh, t->cfg.max_ring,
           t->cfg.max_ring > t->gens_done ? t->cfg.max_ring - t->gens_done : 0);
  free(sj);
  return 0;
}

int r3d_tracer_rewind(r3d_tracer *t, uint32_t gen) {
  if (t->running || !t->pos || !t->gen_of || !gen) return -1;
  uint32_t dropped = 0;
  pthread_mutex_lock(&t->mu);
  uint64_t n = (uint64_t)t->W * t->H;
  for (uint64_t k = 0; k < n; k++) {
    if (t->state[k] == R3D_TR_FAIL || t->state[k] == R3D_TR_PROC) {
      t->state[k] = R3D_TR_EMPTY; /* everything is retryable after rewind */
      t->gen_of[k] = 0;
      continue;
    }
    if (t->state[k] != R3D_TR_SET || t->gen_of[k] <= gen) continue;
    t->state[k] = R3D_TR_EMPTY;
    t->gen_of[k] = 0;
    t->conf[k] = 0.0f;
    if (t->nset) t->nset--;
    dropped++;
  }
  if (t->gens_done > gen) t->gens_done = gen;
  if (t->ring > gen) t->ring = gen;
  t->gen++;
  pthread_mutex_unlock(&t->mu);
  printf("tracer: rewound to generation %u (%u cells dropped)\n", gen, dropped);
  return 0;
}

int r3d_tracer_derive(r3d_tracer *t, const char *pred_root, int dir,
                      const char *out_dir) {
  if (t->running || !t->pos || !t->nset || !dir) return -1;
  r3d_cpuvol vol;
  if (r3d_cpuvol_open(&vol, pred_root, 256) != 0) return -1;
  td_cache *dt = td_open(&vol, t->cfg.level);
  if (!dt) {
    r3d_cpuvol_close(&vol);
    return -1;
  }
  /* gap field over this grid (tr_om_at reads it; median fallback) */
  double om_med = tr_omega_measure(t, dt);
  if (om_med > 0.0) t->sp_om_meas = om_med;
  if (tr_om_eff(t) <= 0.0) t->sp_om_meas = 20.0; /* last resort */
  int W = (int)t->W, H = (int)t->H;
  size_t n = (size_t)W * (size_t)H;
  double *np = malloc(n * 3 * sizeof *np);
  uint8_t *hit = calloc(n, 1);
  int rc = -1, sgn = dir > 0 ? 1 : -1;
  uint32_t nhit = 0, nmiss = 0;
  if (!np || !hit) goto out;
  for (int j = 0; j < H; j++)
    for (int i = 0; i < W; i++) {
      size_t k = (size_t)j * (size_t)W + (size_t)i;
      if (!tr_qc_ok(t, i, j)) continue;
      /* local normal (consistent orientation from the grid frame) */
      if (!tr_valid(t, i - 1, j) || !tr_valid(t, i + 1, j) ||
          !tr_valid(t, i, j - 1) || !tr_valid(t, i, j + 1))
        continue;
      const double *P = t->pos + k * 3;
      const double *pu0 = t->pos + ((size_t)j * (size_t)W + (size_t)i - 1) * 3;
      const double *pu1 = t->pos + ((size_t)j * (size_t)W + (size_t)i + 1) * 3;
      const double *pv0 = t->pos + ((size_t)(j - 1) * (size_t)W + (size_t)i) * 3;
      const double *pv1 = t->pos + ((size_t)(j + 1) * (size_t)W + (size_t)i) * 3;
      double eu[3], ev[3], nr[3];
      for (int a = 0; a < 3; a++) {
        eu[a] = pu1[a] - pu0[a];
        ev[a] = pv1[a] - pv0[a];
      }
      nr[0] = eu[1] * ev[2] - eu[2] * ev[1];
      nr[1] = eu[2] * ev[0] - eu[0] * ev[2];
      nr[2] = eu[0] * ev[1] - eu[1] * ev[0];
      double l = sqrt(nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]);
      if (l < 1e-9) continue;
      for (int a = 0; a < 3; a++) nr[a] = nr[a] / l * sgn;
      double g = tr_om_at(t, P);
      double clear = g * 0.35 > 2.5 ? g * 0.35 : 2.5;
      /* clearance/exit machine (vc3d gen_neighbor): the ray must first
       * leave THIS sheet, then the first DT minimum is the neighbor */
      bool exited = false;
      double prev = 1e30, pprev = 1e30, hs = -1.0;
      for (double s2 = 1.0; s2 <= 2.2 * g; s2 += 1.0) {
        double q[3] = {P[0] + s2 * nr[0], P[1] + s2 * nr[1], P[2] + s2 * nr[2]};
        double dv = td_tri(dt, q, NULL);
        if (!exited && dv > clear) exited = true;
        if (exited && pprev > prev && dv >= prev && prev < 2.0) {
          hs = s2 - 1.0;
          break;
        }
        pprev = prev;
        prev = dv;
      }
      if (hs > 0.0) {
        for (int a = 0; a < 3; a++) np[k * 3 + (size_t)a] = P[a] + hs * nr[a];
        hit[k] = 1;
        nhit++;
      } else {
        nmiss++;
      }
    }
  if (nhit < 64) {
    printf("tracer: derive found only %u crossings — no neighbor sheet?\n", nhit);
    goto out;
  }
  /* fill misses: interpolate the OFFSET (np - pos) from hit neighbors so
   * the derived wrap stays parallel where the DT was ambiguous */
  for (int it = 0; it < 48; it++) {
    for (int j = 0; j < H; j++)
      for (int i = 0; i < W; i++) {
        size_t k = (size_t)j * (size_t)W + (size_t)i;
        if (hit[k] || !tr_qc_ok(t, i, j)) continue;
        double off[3] = {0, 0, 0};
        int na = 0;
        static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int o = 0; o < 4; o++) {
          int ii = i + o4[o][0], jj = j + o4[o][1];
          if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue;
          size_t k2 = (size_t)jj * (size_t)W + (size_t)ii;
          if (!(hit[k2] & 3)) continue;
          for (int a = 0; a < 3; a++)
            off[a] += np[k2 * 3 + (size_t)a] - t->pos[k2 * 3 + (size_t)a];
          na++;
        }
        if (na >= 2) {
          for (int a = 0; a < 3; a++)
            np[k * 3 + (size_t)a] = t->pos[k * 3 + (size_t)a] + off[a] / na;
          hit[k] = 2; /* interpolated (not a seed for the first sweeps) */
        }
      }
    for (size_t k2 = 0; k2 < n; k2++) /* interpolated cells join in */
      if (hit[k2] == 2) hit[k2] = 3;
  }
  { /* write the derived wrap as tifxyz via a shallow tracer */
    r3d_tracer d;
    memset(&d, 0, sizeof d);
    d.W = t->W;
    d.H = t->H;
    d.cfg = t->cfg;
    d.pos = np;
    d.state = calloc(n, 1);
    d.conf = calloc(n, sizeof *d.conf);
    d.wind = calloc(n, sizeof *d.wind);
    d.uc = t->uc; /* borrow: enables the winding channel */
    d.ucn = t->ucn;
    pthread_mutex_init(&d.mu, NULL);
    if (d.state && d.conf && d.wind) {
      for (size_t k = 0; k < n; k++) {
        if (!hit[k]) continue;
        d.state[k] = R3D_TR_SET;
        d.conf[k] = hit[k] == 1 ? 0.9f : 0.4f; /* interpolated = tentative */
        d.wind[k] = t->wind ? t->wind[k] + (float)sgn : (float)sgn;
        d.nset++;
      }
      rc = r3d_tracer_save(&d, out_dir, 0.35f, false);
      if (rc == 0)
        printf("tracer: derived %s wrap -> %s (%u cast + %u interpolated)\n",
               sgn > 0 ? "outer" : "inner", out_dir, nhit, d.nset - nhit);
    }
    free(d.state);
    free(d.conf);
    free(d.wind);
    pthread_mutex_destroy(&d.mu);
  }
out:
  free(np);
  free(hit);
  td_close(dt);
  r3d_cpuvol_close(&vol);
  (void)nmiss;
  return rc == 0 ? (int)nhit : -1;
}

int r3d_tracer_reopt(r3d_tracer *t, const double p[3], int radius) {
  if (t->running || !t->pos || !t->nset) return -1;
  if (radius < 3) radius = 3;
  uint64_t n = (uint64_t)t->W * t->H;
  /* nearest SET cell to the correction point */
  int64_t bk = -1;
  double bd2 = 1e30;
  for (uint64_t k = 0; k < n; k++) {
    if (t->state[k] != R3D_TR_SET) continue;
    const double *q = t->pos + k * 3;
    double d2 = 0;
    for (int a = 0; a < 3; a++) {
      double dd = q[a] - p[a];
      d2 += dd * dd;
    }
    if (d2 < bd2) {
      bd2 = d2;
      bk = (int64_t)k;
    }
  }
  if (bk < 0) return -1;
  int ci = (int)((uint64_t)bk % t->W), cj = (int)((uint64_t)bk / t->W);
  /* flood the suspect region: low conf or wrong-wrap werr, plus an
   * unconditional core of radius 2 around the correction */
  uint8_t *reg = calloc(n, 1);
  uint32_t *q2 = malloc(n * sizeof *q2);
  if (!reg || !q2) {
    free(reg);
    free(q2);
    return -1;
  }
  uint32_t qn = 0, nreg = 0;
  reg[bk] = 1;
  q2[qn++] = (uint32_t)bk;
  bool border = false;
  while (qn) {
    uint32_t k = q2[--qn];
    nreg++;
    int i = (int)(k % t->W), j = (int)(k / t->W);
    if (i <= 1 || j <= 1 || i >= (int)t->W - 2 || j >= (int)t->H - 2) border = true;
    static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int o = 0; o < 4; o++) {
      int ii = i + o4[o][0], jj = j + o4[o][1];
      if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
      size_t k2 = (size_t)jj * t->W + (size_t)ii;
      if (reg[k2] || t->state[k2] != R3D_TR_SET) continue;
      int chev = abs(ii - ci) > abs(jj - cj) ? abs(ii - ci) : abs(jj - cj);
      if (chev > radius) continue;
      bool core = chev <= 2;
      bool suspect = t->conf[k2] < 0.5f || (t->werr && t->werr[k2] > 0.3f);
      if (!core && !suspect) continue;
      reg[k2] = 1;
      q2[qn++] = (uint32_t)k2;
    }
  }
  free(q2);
  if (border) { /* reached the rim: this is a rewind case, not a reopt */
    free(reg);
    return -1;
  }
  pthread_mutex_lock(&t->mu);
  for (uint64_t k = 0; k < n; k++) {
    if (!reg[k]) continue;
    t->state[k] = R3D_TR_EMPTY;
    if (t->gen_of) t->gen_of[k] = 0;
    t->conf[k] = 0.0f;
    if (t->nset) t->nset--;
  }
  free(t->grow_mask);
  t->grow_mask = reg; /* growth confined to the reopened region */
  t->mask_once = true;
  /* budget: enough generations to cross the region */
  uint32_t need = (uint32_t)radius + 5;
  t->gens_done = t->cfg.max_ring > need ? t->cfg.max_ring - need : 0;
  t->quit = false;
  t->done = false;
  t->gen++;
  t->running = true;
  pthread_mutex_unlock(&t->mu);
  printf("tracer: reopened %u cells around (%.0f,%.0f,%.0f), regrowing\n", nreg,
         p[0], p[1], p[2]);
  if (pthread_create(&t->th, NULL, tr_worker, t) != 0) {
    t->running = false;
    return -1;
  }
  return 0;
}

int r3d_tracer_grow(r3d_tracer *t, uint32_t extra) {
  if (t->running || !t->pos || !extra) return -1;
  uint32_t nr = t->cfg.max_ring + extra;
  if (nr > 400) nr = 400;
  /* <=, not ==: a loaded/ribbon trace can carry max_ring above the cap
   * and `nr - max_ring` would wrap into a huge recenter offset */
  if (nr <= t->cfg.max_ring) return -1;
  uint32_t NW = 2 * nr + 50, off = nr - t->cfg.max_ring;
  if (off + t->W > NW || off + t->H > NW) return -1; /* would not fit */
  double *np = calloc((size_t)NW * NW * 3, sizeof *np);
  uint8_t *ns = calloc((size_t)NW * NW, 1);
  float *nc = calloc((size_t)NW * NW, sizeof *nc);
  float *nw = calloc((size_t)NW * NW, sizeof *nw);
  uint8_t *nd = t->dsup ? calloc((size_t)NW * NW, 1) : NULL;
  uint16_t *ng2 = calloc((size_t)NW * NW, sizeof *ng2);
  if (!np || !ns || !nc || !nw || (t->dsup && !nd)) {
    free(np);
    free(ns);
    free(nc);
    free(nw);
    free(nd);
    free(ng2);
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
        if (ng2 && t->gen_of) ng2[nk] = t->gen_of[ok];
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
  free(t->werr);
  t->werr = calloc((size_t)NW * NW, sizeof *t->werr);
  t->dsup = nd;
  if (t->dcell_id) {
    free(t->dcell_id);
    free(t->dcell_uv);
    t->dcell_id = malloc((size_t)NW * NW);
    t->dcell_uv = calloc((size_t)NW * NW * 2, sizeof *t->dcell_uv);
    if (t->dcell_id) memset(t->dcell_id, 0xff, (size_t)NW * NW);
  }
  free(t->gen_of);
  t->gen_of = ng2;
  t->W = t->H = NW;
  t->cfg.max_ring = nr;
  for (uint32_t a = 0; a < R3D_TR_MAX_ANCHORS; a++)
    t->anc_cell[a] = -1; /* grid indices changed: reassign next generation */
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

int r3d_tracer_refine(r3d_tracer *t) {
  if (t->running || !t->pos || !t->nset) return -1;
  t->refine = true;
  t->quit = false;
  t->done = false;
  t->gen++;
  t->running = true;
  if (pthread_create(&t->th, NULL, tr_worker, t) != 0) {
    t->running = false;
    t->refine = false;
    return -1;
  }
  return 0;
}

int r3d_tracer_spiral_fill(r3d_tracer *t) {
  if (t->running || !t->pos || !t->nset) return -1;
  if (!atomic_load(&t->sp_valid)) return -1; /* needs a trusted fit */
  t->spiral_fill = true;
  t->refine = true; /* fill, then the solve-only polish pass */
  t->quit = false;
  t->done = false;
  t->gen++;
  t->running = true;
  if (pthread_create(&t->th, NULL, tr_worker, t) != 0) {
    t->running = false;
    t->refine = false;
    t->spiral_fill = false;
    return -1;
  }
  return 0;
}

int r3d_tracer_ctsnap(r3d_tracer *t, const char *ct_root, double cutoff,
                      double low_cut) {
  if (t->running || !t->pos || !t->nset) return -1;
  if (ct_root && ct_root[0])
    snprintf(t->cfg.ct_root, sizeof t->cfg.ct_root, "%s", ct_root);
  if (!t->cfg.ct_root[0]) return -1;
  t->ctsnap = true;
  t->ctsnap_dist = cutoff > 0 ? cutoff : 6.0;
  t->ctsnap_cut = low_cut;
  t->quit = false;
  t->done = false;
  t->gen++;
  t->running = true;
  if (pthread_create(&t->th, NULL, tr_worker, t) != 0) {
    t->running = false;
    t->ctsnap = false;
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
  free(t->werr);
  free(t->omf);
  free(t->reopt_pos);
  free(t->reopt_nrm);
  free(t->ctsnap_tgt);
  free(t->gen_of);
  free(t->grow_mask);
  free(t->dcell_id);
  free(t->dcell_uv);
  free(t->dsup);
  free(t->uc);
  tr_sfx_free(t->sfx);
  tr_wf_free(t->wf);
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

/* ==================== transactional publication ====================
 * Every artifact of one export is written to a temporary name in the
 * destination directory and only renamed into place once ALL of them are
 * complete. A failed, out-of-space or interrupted export therefore never
 * truncates the artifact that is already there. Publication order is
 * registration order and meta.json is registered LAST, so a torn rename
 * (rare: same-directory rename) cannot leave a manifest describing
 * planes that were never written. */
typedef struct tr_pub {
  const char *dir;
  char pfx[48];
  const char *name[12];
  uint32_t n;
} tr_pub;

static void tr_pub_init(tr_pub *p, const char *dir) {
  p->dir = dir;
  p->n = 0;
  snprintf(p->pfx, sizeof p->pfx, ".r3dtmp%ld", (long)getpid());
}

static bool tr_pub_path(const tr_pub *p, const char *name, bool tmp, char *out,
                        size_t cap) {
  int k = tmp ? snprintf(out, cap, "%s/%s.%s", p->dir, p->pfx, name)
              : snprintf(out, cap, "%s/%s", p->dir, name);
  return k > 0 && (size_t)k < cap;
}

/* register + hand back the temporary path to write; false = no room */
static bool tr_pub_add(tr_pub *p, const char *name, char *out, size_t cap) {
  if (p->n >= sizeof p->name / sizeof *p->name) return false;
  if (!tr_pub_path(p, name, true, out, cap)) return false;
  p->name[p->n++] = name;
  return true;
}

static void tr_pub_abort(tr_pub *p) {
  char tp[1200];
  for (uint32_t i = 0; i < p->n; i++)
    if (tr_pub_path(p, p->name[i], true, tp, sizeof tp)) unlink(tp);
  p->n = 0;
}

static int tr_pub_commit(tr_pub *p) {
  char tp[1200], fp[1200];
  for (uint32_t i = 0; i < p->n; i++) {
    if (!tr_pub_path(p, p->name[i], true, tp, sizeof tp) ||
        !tr_pub_path(p, p->name[i], false, fp, sizeof fp) ||
        rename(tp, fp) != 0) {
      for (uint32_t j = i; j < p->n; j++) /* never publish a partial set */
        if (tr_pub_path(p, p->name[j], true, tp, sizeof tp)) unlink(tp);
      p->n = 0;
      return -1;
    }
  }
  p->n = 0;
  return 0;
}

/* stdio errors are export failures, not noise */
static int tr_fclose_checked(FILE *f) {
  bool bad = ferror(f) != 0;
  if (fflush(f) != 0) bad = true;
  if (fclose(f) != 0) bad = true;
  return bad ? -1 : 0;
}

static void tr_json_esc(const char *in, char *out, size_t cap) {
  size_t o = 0;
  if (!cap) return;
  for (const unsigned char *q = (const unsigned char *)in; *q && o + 7 < cap;
       q++) {
    if (*q == '"' || *q == '\\') {
      out[o++] = '\\';
      out[o++] = (char)*q;
    } else if (*q < 0x20) {
      int k = snprintf(out + o, cap - o, "\\u%04x", (unsigned)*q);
      if (k <= 0) break;
      o += (size_t)k;
    } else {
      out[o++] = (char)*q;
    }
  }
  out[o] = 0;
}

/* the state sidecar r3d_tracer_load needs for a faithful round trip */
static int tr_write_state(const r3d_tracer *t, const char *path, uint32_t cw,
                          uint32_t ch, uint32_t ci0, uint32_t cj0,
                          float cutoff, bool fill, uint32_t maxgen,
                          uint64_t nkept, bool have_conf) {
  FILE *f = fopen(path, "w");
  if (!f) return -1;
  char ctr[6200], prr[6200];
  tr_json_esc(t->cfg.ct_root, ctr, sizeof ctr);
  tr_json_esc(t->root, prr, sizeof prr);
  fprintf(f,
          "{\n  \"format\": \"" TR_STATE_FMT "\",\n  \"version\": %d,\n"
          "  \"grid_w\": %u,\n  \"grid_h\": %u,\n"
          "  \"full_w\": %u,\n  \"full_h\": %u,\n"
          "  \"offset_i\": %u,\n  \"offset_j\": %u,\n"
          "  \"step\": %.9g,\n  \"thresh\": %.6g,\n"
          "  \"max_ring\": %u,\n  \"level\": %u,\n"
          "  \"rib_rows\": %u,\n  \"rib_wraps\": %u,\n"
          "  \"grow_dirs\": %u,\n  \"max_threads\": %u,\n"
          "  \"seed_x\": %.9g,\n  \"seed_y\": %.9g,\n  \"seed_z\": %.9g,\n"
          "  \"x_min\": %.9g,\n  \"x_max\": %.9g,\n"
          "  \"y_min\": %.9g,\n  \"y_max\": %.9g,\n"
          "  \"z_min\": %.9g,\n  \"z_max\": %.9g,\n"
          "  \"ct_min\": %.9g,\n  \"wind_weight\": %.9g,\n"
          "  \"tear_lim\": %.9g,\n  \"sp_om_meas\": %.9g,\n"
          "  \"vdim_x\": %.9g,\n  \"vdim_y\": %.9g,\n  \"vdim_z\": %.9g,\n"
          "  \"gens_done\": %u,\n  \"max_gen\": %u,\n  \"ring\": %u,\n"
          "  \"nset\": %u,\n  \"nkept\": %llu,\n"
          "  \"save_cutoff\": %.6g,\n  \"save_fill\": %s,\n"
          "  \"confidence\": %s,\n"
          "  \"nanchors\": %u,\n  \"anchors\": [",
          TR_STATE_VER, cw, ch, t->W, t->H, ci0, cj0, t->cfg.step,
          (double)t->cfg.thresh, t->cfg.max_ring, t->cfg.level,
          t->cfg.rib_rows, t->cfg.rib_wraps, (unsigned)t->cfg.grow_dirs,
          t->cfg.max_threads, t->cfg.seed[0], t->cfg.seed[1], t->cfg.seed[2],
          t->cfg.x_min, t->cfg.x_max, t->cfg.y_min, t->cfg.y_max, t->cfg.z_min,
          t->cfg.z_max, t->cfg.ct_min, t->cfg.wind_weight, t->tear_lim,
          t->sp_om_meas, t->vdim[0], t->vdim[1], t->vdim[2], t->gens_done,
          maxgen, t->ring, t->nset, (unsigned long long)nkept, (double)cutoff,
          fill ? "true" : "false", have_conf ? "true" : "false", t->nanc);
  for (uint32_t a = 0; a < t->nanc && a < R3D_TR_MAX_ANCHORS; a++)
    fprintf(f, "%s%.9g, %.9g, %.9g", a ? ",\n    " : "\n    ",
            t->anc[a * 3 + 0], t->anc[a * 3 + 1], t->anc[a * 3 + 2]);
  /* the free-text paths go LAST: a path is the only field that could
   * contain a "key": sequence, and nothing is looked up after it */
  fprintf(f, "%s],\n  \"ct_root\": \"%s\",\n  \"pred_root\": \"%s\"\n}\n",
          t->nanc ? "\n  " : "", ctr, prr);
  return tr_fclose_checked(f);
}

int r3d_tracer_save(r3d_tracer *t, const char *dir, float cutoff, bool fill) {
  if (!t->pos) return -1;
  uint64_t n = (uint64_t)t->W * t->H;
  float *pl = malloc(n * sizeof *pl);
  uint8_t *keep = malloc(n);
  uint8_t *torn = calloc(n, 1);
  double *fp = NULL; /* membrane re-seating, kept OUT of t->pos */
  uint8_t *ext = NULL;
  tr_pub pub;
  tr_pub_init(&pub, dir);
  if (!pl || !keep || !torn) { /* a missing tear/fill buffer would silently
                                * change what "saved" means: refuse */
    free(pl);
    free(keep);
    free(torn);
    return -1;
  }
  static const char *nm[3] = {"x.tif", "y.tif", "z.tif"};
  int rc = 0;
  pthread_mutex_lock(&t->mu);
  /* tear mask: a cell whose edge to any 4-neighbor is way off the unit
   * is mis-seated (usually a wrong-wrap capture in ambiguous data) — an
   * honest hole beats a committed discontinuity */
  {
    double lim = t->tear_lim > 0.0 ? t->tear_lim : 1.75 * t->cfg.step;
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
        if (d2 > lim * lim) {
          keep[k] = 0;
          torn[k] = 1; /* wrong-wrap capture: never re-seated */
        }
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
      /* no holes: every ENCLOSED grown cell gets a point (G8 gate). Rim
       * low-conf cells stay honest holes (a membrane there extrapolates
       * into the unknown), and tear-cut cells are never re-seated: the
       * tear mask deliberately removed them as wrong-wrap captures and a
       * data-blind membrane would sew the wrap back in.
       * The re-seated positions live in `fp` and are consumed by the
       * plane writer below — the live grid is never touched. */
      ext = malloc(n);
      uint8_t *blocked = ext ? malloc(n) : NULL;
      fp = blocked ? malloc(n * 3 * sizeof *fp) : NULL;
      if (!fp) { /* silently skipping the fill would change what the
                  * caller asked for: fail the export instead */
        free(blocked);
        rc = -1;
      } else {
        for (uint64_t k = 0; k < n; k++) blocked[k] = keep[k] != 0;
        tr_flood_exterior(t->W, t->H, blocked, ext);
        free(blocked);
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
              if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H)
                continue;
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
          if (!keep[k] && t->state[k] == R3D_TR_SET && !torn[k] && !ext[k])
            keep[k] = 3; /* filled (enclosed, untorn): read from fp */
      }
    }
  }
  /* min-size gate + bbox crop (G15, vc3d min_area/vc_tifxyz_trim): a
   * seed that landed in noise must not leave a plausible tifxyz behind
   * that later gets loaded as a donor, and the grid is 2*max_ring+50
   * wide so an early-stopped run otherwise ships a large all-invalid
   * margin. Crop to the kept bbox + 2 and record grid_offset. */
  uint64_t nkept = 0;
  uint32_t maxgen = 0;
  uint32_t cb[4] = {t->W, t->H, 0, 0};
  for (uint64_t k = 0; rc == 0 && k < n; k++) {
    if (!keep[k]) continue;
    nkept++;
    if (t->gen_of && t->gen_of[k] > maxgen) maxgen = t->gen_of[k];
    uint32_t i = (uint32_t)(k % t->W), j = (uint32_t)(k / t->W);
    if (i < cb[0]) cb[0] = i;
    if (j < cb[1]) cb[1] = j;
    if (i > cb[2]) cb[2] = i;
    if (j > cb[3]) cb[3] = j;
  }
  if (rc == 0 && nkept < 64) {
    pthread_mutex_unlock(&t->mu);
    free(keep);
    free(torn);
    free(pl);
    free(fp);
    free(ext);
    printf("tracer: refusing to save %llu-point patch (< 64 trusted cells)\n",
           (unsigned long long)nkept);
    return -2;
  }
  uint32_t ci0 = 0, cj0 = 0, cw = 0, ch = 0;
  if (rc == 0) {
    ci0 = cb[0] > 2 ? cb[0] - 2 : 0;
    cj0 = cb[1] > 2 ? cb[1] - 2 : 0;
    uint32_t ci1 = cb[2] + 2 < t->W ? cb[2] + 2 : t->W - 1;
    uint32_t cj1 = cb[3] + 2 < t->H ? cb[3] + 2 : t->H - 1;
    cw = ci1 - ci0 + 1;
    ch = cj1 - cj0 + 1;
  }
  for (int a = 0; a < 3 && rc == 0; a++) {
    for (uint32_t j = 0; j < ch; j++)
      for (uint32_t i = 0; i < cw; i++) {
        size_t k = (size_t)(cj0 + j) * t->W + (size_t)(ci0 + i);
        const double *src = keep[k] == 3 && fp ? fp + k * 3 : t->pos + k * 3;
        pl[(size_t)j * cw + i] = keep[k] ? (float)src[(size_t)a] : -1.0f;
      }
    char path[1200];
    rc = tr_pub_add(&pub, nm[a], path, sizeof path) ? 0 : -1;
    if (rc == 0) rc = tr_write_plane(path, pl, cw, ch);
  }
  if (rc == 0 && t->uc && t->wind) { /* winding channel (vc3d ecosystem:
                                      * float winding per cell, NAN off) */
    for (uint32_t j = 0; j < ch; j++)
      for (uint32_t i = 0; i < cw; i++) {
        size_t k = (size_t)(cj0 + j) * t->W + (size_t)(ci0 + i);
        pl[(size_t)j * cw + i] = keep[k] ? t->wind[k]
                                         : -1e30f; /* invalid marker (validity
                                                    * lives in x.tif; NAN is
                                                    * unusable under
                                                    * -ffast-math) */
      }
    char path[1200];
    rc = tr_pub_add(&pub, "winding.tif", path, sizeof path) ? 0 : -1;
    if (rc == 0) rc = tr_write_plane(path, pl, cw, ch);
  }
  if (rc == 0 && t->gen_of) { /* generation stamps: the rewind substrate */
    for (uint32_t j = 0; j < ch; j++)
      for (uint32_t i = 0; i < cw; i++)
        pl[(size_t)j * cw + i] =
            (float)t->gen_of[(size_t)(cj0 + j) * t->W + (size_t)(ci0 + i)];
    char path[1200];
    rc = tr_pub_add(&pub, "generations.tif", path, sizeof path) ? 0 : -1;
    if (rc == 0) rc = tr_write_plane(path, pl, cw, ch);
  }
  bool have_conf = false;
  if (rc == 0 && t->conf) { /* per-vertex confidence: without it a reload
                             * has to invent 1.0 everywhere and the tear/
                             * infill/fill decisions stop reproducing */
    for (uint32_t j = 0; j < ch; j++)
      for (uint32_t i = 0; i < cw; i++) {
        size_t k = (size_t)(cj0 + j) * t->W + (size_t)(ci0 + i);
        pl[(size_t)j * cw + i] = keep[k] ? t->conf[k] : -1.0f;
      }
    char path[1200];
    rc = tr_pub_add(&pub, "confidence.tif", path, sizeof path) ? 0 : -1;
    if (rc == 0) rc = tr_write_plane(path, pl, cw, ch);
    have_conf = rc == 0;
  }
  /* spiral model + state sidecar, still under the lock: they describe the
   * exact generation the planes came from */
  if (rc == 0 && t->sp_valid) { /* fusion / winding registration */
    char path[1200];
    rc = tr_pub_add(&pub, "spiral.json", path, sizeof path) ? 0 : -1;
    if (rc == 0) {
      FILE *sf = fopen(path, "w");
      if (!sf) {
        rc = -1;
      } else {
        fprintf(sf, "{\n  \"omega\": %.6f,\n  \"omega_measured\": %.3f,\n"
                    "  \"rms\": %.6f,\n  \"z0\": %.3f,\n"
                    "  \"dz\": %.3f,\n  \"r0\": [",
                t->sp_omega, t->sp_om_meas, t->sp_rms, t->sp_z0, t->sp_dz);
        for (uint32_t kk = 0; kk < t->sp_k; kk++)
          fprintf(sf, "%s%.3f", kk ? ", " : "", t->sp_r0[kk]);
        fprintf(sf, "]\n}\n");
        rc = tr_fclose_checked(sf);
      }
    }
  }
  if (rc == 0) {
    char path[1200];
    rc = tr_pub_add(&pub, "tracer.json", path, sizeof path) ? 0 : -1;
    if (rc == 0)
      rc = tr_write_state(t, path, cw, ch, ci0, cj0, cutoff, fill, maxgen,
                          nkept, have_conf);
  }
  if (rc == 0) { /* meta.json LAST: it is what makes the directory a
                  * tifxyz, so it must never precede its own planes */
    char path[1200];
    rc = tr_pub_add(&pub, "meta.json", path, sizeof path) ? 0 : -1;
    if (rc == 0) {
      FILE *mf = fopen(path, "w");
      if (!mf) {
        rc = -1;
      } else {
        double sc = 1.0 / t->cfg.step;
        fprintf(mf,
                "{\n  \"format\": \"tifxyz\",\n  \"type\": \"seg\",\n"
                "  \"scale\": [\n    %.6f,\n    %.6f\n  ],\n"
                "  \"source\": \"render3d-tracer\",\n"
                "  \"donor_segments\": %u,\n"
                "  \"area_vx2\": %.1f,\n  \"bbox\": [%u, %u, %u, %u],\n"
                "  \"fill\": %.4f,\n  \"hole\": %.4f,\n"
                "  \"qc\": {\"folds\": %u, \"kinks\": %u, \"twist\": %.3f, "
                "\"slant_p95\": %.4f, \"wrap_frac\": %.4f, \"werr_p95\": "
                "%.4f},\n"
                "  \"donor_qc\": {\"mean\": %.2f, \"rms\": %.2f, \"p95\": "
                "%.2f, \"coverage\": %.3f},\n"
                "  \"grid_offset\": [%u, %u],\n  \"anchors\": %u,\n"
                "  \"max_gen\": %u,\n  \"state\": \"tracer.json\"\n}\n",
                sc, sc, t->ndon, t->qc_area_vx2, t->qc_bbox[0], t->qc_bbox[1],
                t->qc_bbox[2], t->qc_bbox[3], (double)t->qc_fill,
                (double)t->qc_hole, t->qc_folds, t->qc_kinks,
                (double)t->qc_twist, (double)t->qc_slant_p95,
                (double)t->qc_wrap_frac, (double)t->qc_werr_p95,
                (double)t->qc_don_mean, (double)t->qc_don_rms,
                (double)t->qc_don_p95, (double)t->qc_don_cov, ci0, cj0,
                t->nanc, t->gens_done);
        rc = tr_fclose_checked(mf);
      }
    }
  }
  pthread_mutex_unlock(&t->mu);
  free(keep);
  free(torn);
  free(pl);
  free(fp);
  free(ext);
  if (rc != 0) {
    tr_pub_abort(&pub); /* the previous export, if any, is untouched */
    return -1;
  }
  if (tr_pub_commit(&pub) != 0) return -1;
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
  { /* winding relaxation: clean spiral -> werr ~ 0 everywhere; a cell
     * with a corrupted causal winding must light up as a wrong-wrap */
    t.werr = calloc(N, sizeof *t.werr);
    if (!t.werr) goto out;
    t.nset = (uint32_t)N;
    for (uint64_t k = 0; k < N; k++) t.conf[k] = 1.0f;
    tr_wind_relax(&t, 40);
    for (uint64_t k = 0; k < N; k++)
      if (t.werr[k] > 0.05f) goto out; /* clean grid must read clean */
    if (t.qc_wrap_frac != 0.0f) goto out;
    size_t bad = (size_t)(t.H / 2) * t.W + 10;
    float saved_w = t.wind[bad];
    t.wind[bad] += 1.0f; /* a wrap-jump capture: same pos, wrong winding */
    float saved_conf = t.conf[bad];
    tr_wind_relax(&t, 40);
    if (t.werr[bad] < 0.5f) goto out;     /* must be detected */
    if (t.conf[bad] > 0.26f) goto out;    /* clamp must fire (default on) */
    if (t.qc_wrap_frac <= 0.0f) goto out;
    t.wind[bad] = saved_w;
    t.conf[bad] = saved_conf;
    for (uint64_t k = 0; k < N; k++) t.conf[k] = 1.0f;
  }
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
  free(t.werr); /* allocated by the winding-relaxation block above */
  free(t.omf);  /* allocated by tr_spiral_fit */
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
    int di = tr_don_closest(dn, p, TR_FUS_PULL, q, &wnd, NULL);
    if (di != 0 || fabs(q[0] - 35.0) > 1e-6 || fabs(q[1] - 42.0) > 1e-6 ||
        fabs(q[2] - 50.0) > 1e-6)
      goto out;
    if (wnd > -1e29) goto out; /* not registered yet: no winding */
    dn->d[0].woff = 2.0;
    dn->d[0].woff_ok = true;
    di = tr_don_closest(dn, p, TR_FUS_PULL, q, &wnd, NULL);
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
