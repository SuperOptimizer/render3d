#include "core/tracer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

#include "core/cpuvol.h"

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

/* ======================== residual evaluation ======================== */
enum { TRF_DIST = 1, TRF_STRAIGHT = 2, TRF_SDIR = 4, TRF_SPACE = 8 };
#define TRF_ALL (TRF_DIST | TRF_STRAIGHT | TRF_SDIR | TRF_SPACE)

typedef struct tr_ctx {
  r3d_tracer *t;
  td_cache *dt;
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
  if (!isfinite(E)) return;
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

/* all residuals touching free cell (i,j) evaluated at trial position x */
static void tr_eval(tr_ctx *c, const double x[3], tr_nlsq *acc) {
  r3d_tracer *t = c->t;
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
  if (c->flags & TRF_SPACE) {
    static const int n4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int o = 0; o < 4; o++) {
      int ii = i + n4[o][0], jj = j + n4[o][1];
      if (!tr_valid(t, ii, jj)) continue;
      tr_res_space(acc, c->dt, x, tr_at(c, ii, jj, x), TR_W_SPACE);
    }
  }
}

/* LM solve for one cell; updates pos in place (under mu), returns cost */
static double tr_solve_cell(r3d_tracer *t, td_cache *dt, int i, int j, uint32_t flags,
                            int max_iter) {
  tr_ctx c = {.t = t, .dt = dt, .i = i, .j = j, .flags = flags};
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
  return cost;
}

static void tr_update_conf(r3d_tracer *t, td_cache *dt, int i, int j) {
  size_t k = (size_t)j * t->W + (size_t)i;
  if (t->state[k] != R3D_TR_SET) return;
  double v = td_tri(dt, t->pos + k * 3, NULL);
  double cf = 1.0 - (v > TR_CONF_R ? TR_CONF_R : v) / TR_CONF_R;
  t->conf[k] = (float)cf;
}

/* vc3d local_optimization(radius, p): free = SET cells within Euclidean
 * `radius` of center, boundary ring fixed; here solved as alternating
 * Gauss-Seidel sweeps of per-cell LM instead of one joint sparse solve. */
static void tr_local_opt(r3d_tracer *t, td_cache *dt, int ci, int cj, int radius,
                         int sweeps) {
  int r0i = ci - radius, r1i = ci + radius, r0j = cj - radius, r1j = cj + radius;
  if (r0i < 0) r0i = 0;
  if (r0j < 0) r0j = 0;
  if (r1i >= (int)t->W) r1i = (int)t->W - 1;
  if (r1j >= (int)t->H) r1j = (int)t->H - 1;
  for (int s = 0; s < sweeps && !t->quit; s++) {
    bool rev = (s & 1) != 0;
    for (int jj = r0j; jj <= r1j; jj++)
      for (int ii = r0i; ii <= r1i; ii++) {
        int i = rev ? r1i - (ii - r0i) : ii;
        int j = rev ? r1j - (jj - r0j) : jj;
        long di = i - ci, dj = j - cj;
        if (di * di + dj * dj > (long)radius * radius) continue;
        if (t->state[(size_t)j * t->W + (size_t)i] != R3D_TR_SET) continue;
        tr_solve_cell(t, dt, i, j, TRF_ALL, 4);
        if (s + 1 == sweeps) tr_update_conf(t, dt, i, j);
      }
  }
}

static double tr_rand(r3d_tracer *t) { /* U(-0.05, 0.05), vc3d perturbation */
  t->rng = t->rng * 1664525u + 1013904223u;
  return ((double)(t->rng >> 8) / (double)(1u << 24) - 0.5) * 0.1;
}

/* =========================== growth loop =========================== */
static void *tr_worker(void *ud) {
  r3d_tracer *t = ud;
  r3d_cpuvol vol;
  if (r3d_cpuvol_open(&vol, t->root, 96) != 0) goto fail_open;
  td_cache *dt = td_open(&vol, t->cfg.level);
  if (!dt) {
    r3d_cpuvol_close(&vol);
    goto fail_open;
  }
  uint32_t W = t->W, H = t->H;
  int x0 = (int)W / 2, y0 = (int)H / 2;
  uint32_t *fringe = malloc((size_t)W * H * sizeof *fringe);
  uint32_t *nfringe = malloc((size_t)W * H * sizeof *nfringe);
  uint32_t *cands = malloc((size_t)W * H * sizeof *cands);
  if (!fringe || !nfringe || !cands) {
    free(fringe);
    free(nfringe);
    free(cands);
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
    tr_local_opt(t, dt, x0, y0, 8, 6);
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
    uint32_t nnew = 0;
    for (uint32_t ci = 0; ci < nc && !t->quit; ci++) {
      int i = (int)(cands[ci] % W), j = (int)(cands[ci] / W);
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
      if (bi < 0) continue;
      size_t k = (size_t)j * W + (size_t)i;
      const double *bp = t->pos + ((size_t)bj * W + (size_t)bi) * 3;
      pthread_mutex_lock(&t->mu);
      for (int a = 0; a < 3; a++) t->pos[k * 3 + (size_t)a] = bp[a] + tr_rand(t);
      t->state[k] = R3D_TR_SET; /* committed before the solve (vc3d) */
      t->nset++;
      pthread_mutex_unlock(&t->mu);
      /* placement: geometric + data terms, then radius-1 and radius-3 */
      tr_solve_cell(t, dt, i, j, TRF_DIST | TRF_STRAIGHT | TRF_SPACE, 50);
      tr_local_opt(t, dt, i, j, 1, 2);
      tr_local_opt(t, dt, i, j, 3, 3);
      tr_update_conf(t, dt, i, j);
      nfringe[nnew++] = (uint32_t)k;
    }
    /* schedule: early global solves, later subsampled radius-8 solves */
    if (global_opt) {
      if (generation % 8 == 0) tr_local_opt(t, dt, x0, y0, (int)W + (int)H, 6);
    } else {
      for (uint32_t f = 0; f < nnew && !t->quit; f++) {
        int i = (int)(nfringe[f] % W), j = (int)(nfringe[f] / W);
        if (i % 4 == 0 && j % 4 == 0) tr_local_opt(t, dt, i, j, 8, 3);
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
  if (!t->quit) tr_local_opt(t, dt, x0, y0, (int)W + (int)H, 2);
  free(fringe);
  free(nfringe);
  free(cands);
  td_close(dt);
  r3d_cpuvol_close(&vol);
  printf("tracer: finished at generation %u with %u point%s (level L%u)\n", t->ring,
         t->nset, t->nset == 1 ? "" : "s", t->cfg.level);
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
  t->rng = 0x1234567u;
  if (!t->pos || !t->state || !t->conf) {
    r3d_tracer_free(t);
    return -1;
  }
  r3d_umbilicus_init(&t->umb);
  if (umb)
    for (size_t k = 0; k < umb->count; k++)
      r3d_umbilicus_set(&t->umb, umb->points[k].x, umb->points[k].y, umb->points[k].z);
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
  if (!np || !ns || !nc) {
    free(np);
    free(ns);
    free(nc);
    return -1;
  }
  for (uint32_t j = 0; j < t->H; j++)
    for (uint32_t i = 0; i < t->W; i++) {
      size_t ok = (size_t)j * t->W + i;
      size_t nk = (size_t)(j + off) * NW + (i + off);
      if (t->state[ok] == R3D_TR_SET) { /* PROC cells reset for another try */
        ns[nk] = R3D_TR_SET;
        nc[nk] = t->conf[ok];
        memcpy(np + nk * 3, t->pos + ok * 3, 3 * sizeof(double));
      }
    }
  free(t->pos);
  free(t->state);
  free(t->conf);
  t->pos = np;
  t->state = ns;
  t->conf = nc;
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

int r3d_tracer_save(r3d_tracer *t, const char *dir, float cutoff) {
  if (!t->pos) return -1;
  uint64_t n = (uint64_t)t->W * t->H;
  float *pl = malloc(n * sizeof *pl);
  if (!pl) return -1;
  static const char *nm[3] = {"x.tif", "y.tif", "z.tif"};
  int rc = 0;
  pthread_mutex_lock(&t->mu);
  for (int a = 0; a < 3 && rc == 0; a++) {
    for (uint64_t k = 0; k < n; k++)
      pl[k] = t->state[k] == R3D_TR_SET && t->conf[k] >= cutoff
                  ? (float)t->pos[k * 3 + (size_t)a]
                  : -1.0f;
    char path[1200];
    snprintf(path, sizeof path, "%s/%s", dir, nm[a]);
    rc = tr_write_plane(path, pl, t->W, t->H);
  }
  pthread_mutex_unlock(&t->mu);
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
  return 0;
}
