#include "core/bsurf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- voxel classification ------------------------------------------- */

static inline bool bs_pap(const r3d_bsurf *t, int64_t x, int64_t y, int64_t z) {
  for (int a = 0; a < 3; a++)
    if (t->cfg.vdim[a] > 0.0) {
      int64_t c = a == 0 ? x : a == 1 ? y : z;
      if (c < 0 || (double)c >= t->cfg.vdim[a]) return false; /* outside = void */
    }
  return (double)t->sample(t->sctx, x, y, z) >= t->cfg.low_cut;
}

static inline void bs_round(const double p[3], int64_t q[3]) {
  for (int a = 0; a < 3; a++) q[a] = (int64_t)llround(p[a]);
}

/* papyrus voxel with at least one void 6-neighbour */
static bool bs_is_bnd(const r3d_bsurf *t, const int64_t q[3]) {
  if (!bs_pap(t, q[0], q[1], q[2])) return false;
  static const int o6[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                               {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int o = 0; o < 6; o++)
    if (!bs_pap(t, q[0] + o6[o][0], q[1] + o6[o][1], q[2] + o6[o][2])) return true;
  return false;
}

/* outward normal at a boundary voxel: mean direction of the void voxels
 * in the 5x5x5 neighbourhood (papyrus -> air). false when degenerate. */
static bool bs_normal(const r3d_bsurf *t, const int64_t q[3], double n[3]) {
  double s[3] = {0, 0, 0};
  for (int dz = -2; dz <= 2; dz++)
    for (int dy = -2; dy <= 2; dy++)
      for (int dx = -2; dx <= 2; dx++) {
        if (!dx && !dy && !dz) continue;
        if (bs_pap(t, q[0] + dx, q[1] + dy, q[2] + dz)) continue;
        double l = sqrt((double)(dx * dx + dy * dy + dz * dz));
        s[0] += (double)dx / l;
        s[1] += (double)dy / l;
        s[2] += (double)dz / l;
      }
  double l = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  if (l < 1e-6) return false;
  for (int a = 0; a < 3; a++) n[a] = s[a] / l;
  return true;
}

/* Snap along the normal line through pred: papyrus must lie on the -n
 * side and void on the +n side. If pred itself is papyrus, walk +n to the
 * exit; if void, walk -n to the entry. Anything else (void with a sheet
 * entering ahead, papyrus with void behind only) is the wrong face and
 * is not a landing. Returns the boundary voxel or false. */
static bool bs_snap_line(const r3d_bsurf *t, const double pred[3], const double n[3],
                         int64_t out[3]) {
  double snap = t->cfg.snap;
  int64_t q0[3];
  bs_round(pred, q0);
  bool p0 = bs_pap(t, q0[0], q0[1], q0[2]);
  double dir = p0 ? 1.0 : -1.0;
  int64_t last[3] = {q0[0], q0[1], q0[2]};
  bool have_last = p0; /* last papyrus voxel seen along the walk */
  for (double s = 0.5; s <= snap + 1e-9; s += 0.5) {
    double p[3] = {pred[0] + dir * s * n[0], pred[1] + dir * s * n[1],
                   pred[2] + dir * s * n[2]};
    int64_t q[3];
    bs_round(p, q);
    if (q[0] == last[0] && q[1] == last[1] && q[2] == last[2] && have_last) continue;
    bool pq = bs_pap(t, q[0], q[1], q[2]);
    if (p0) {
      if (pq) {
        memcpy(last, q, sizeof last);
        continue;
      }
      /* left the papyrus: last is the exit voxel */
      if (bs_is_bnd(t, last)) {
        memcpy(out, last, sizeof last);
        return true;
      }
      return false;
    }
    /* walking -n from void: first papyrus voxel is the entry face */
    if (pq) {
      if (bs_is_bnd(t, q)) {
        memcpy(out, q, sizeof(int64_t) * 3);
        return true;
      }
      return false;
    }
  }
  return false;
}

/* Radial fallback: nearest boundary voxel to pred (within snap) whose
 * outward normal agrees with n. Offsets are visited in distance order so
 * the first hit is the answer. */
typedef struct {
  int16_t d[3];
  float r;
} bs_off;
static int bs_off_cmp(const void *a, const void *b) {
  float ra = ((const bs_off *)a)->r, rb = ((const bs_off *)b)->r;
  return ra < rb ? -1 : ra > rb ? 1 : 0;
}
static bs_off *bs_offs = NULL;
static uint32_t bs_noffs = 0;
static int bs_offs_r = 0;
static pthread_mutex_t bs_offs_mu = PTHREAD_MUTEX_INITIALIZER;
static void bs_offs_ensure(int r) {
  pthread_mutex_lock(&bs_offs_mu);
  if (r > bs_offs_r) {
    uint32_t side = (uint32_t)(2 * r + 1);
    bs_off *o = malloc((size_t)side * side * side * sizeof *o);
    if (o) {
      uint32_t n = 0;
      for (int dz = -r; dz <= r; dz++)
        for (int dy = -r; dy <= r; dy++)
          for (int dx = -r; dx <= r; dx++) {
            o[n].d[0] = (int16_t)dx;
            o[n].d[1] = (int16_t)dy;
            o[n].d[2] = (int16_t)dz;
            o[n].r = sqrtf((float)(dx * dx + dy * dy + dz * dz));
            n++;
          }
      qsort(o, n, sizeof *o, bs_off_cmp);
      free(bs_offs);
      bs_offs = o;
      bs_noffs = n;
      bs_offs_r = r;
    }
  }
  pthread_mutex_unlock(&bs_offs_mu);
}

/* need_n: the landing's outward normal must agree with n (dot >= 0.5) and
 * its offset from pred must lie within `tmax` of the normal line (a cone:
 * the node may travel along the normal to find the face, not sideways
 * along it - sideways is what the grid step is for). */
static bool bs_snap_radial(const r3d_bsurf *t, const double pred[3], const double n[3],
                           bool need_n, double tmax, int64_t out[3], double onrm[3]) {
  int r = (int)ceil(t->cfg.snap);
  if (r < 1) r = 1;
  if (r > 24) r = 24;
  bs_offs_ensure(r);
  int64_t q0[3];
  bs_round(pred, q0);
  double best = 1e30;
  bool found = false;
  for (uint32_t i = 0; i < bs_noffs; i++) {
    const bs_off *o = &bs_offs[i];
    if ((double)o->r > t->cfg.snap + 0.5) break;
    if (found && (double)o->r > best + 1e-6) break;
    int64_t q[3] = {q0[0] + o->d[0], q0[1] + o->d[1], q0[2] + o->d[2]};
    if (need_n) {
      double e[3] = {(double)q[0] - pred[0], (double)q[1] - pred[1],
                     (double)q[2] - pred[2]};
      double en = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
      double dt2 = 0.0;
      for (int a = 0; a < 3; a++) dt2 += (e[a] - en * n[a]) * (e[a] - en * n[a]);
      if (dt2 > tmax * tmax) continue;
    }
    if (!bs_is_bnd(t, q)) continue;
    double nn[3];
    if (!bs_normal(t, q, nn)) continue;
    if (need_n && nn[0] * n[0] + nn[1] * n[1] + nn[2] * n[2] < 0.5) continue;
    double d2 = 0.0;
    for (int a = 0; a < 3; a++) {
      double dd = (double)q[a] - pred[a];
      d2 += dd * dd;
    }
    if (d2 < best) {
      best = d2;
      memcpy(out, q, sizeof(int64_t) * 3);
      memcpy(onrm, nn, sizeof(double) * 3);
      found = true;
    }
  }
  return found;
}

/* ---- overlap hash ------------------------------------------------------ */

static uint64_t bs_hkey(const r3d_bsurf *t, const double p[3], int dx, int dy, int dz) {
  double inv = 1.0 / t->cfg.step;
  int64_t cx = (int64_t)floor(p[0] * inv) + dx + (1 << 20);
  int64_t cy = (int64_t)floor(p[1] * inv) + dy + (1 << 20);
  int64_t cz = (int64_t)floor(p[2] * inv) + dz + (1 << 20);
  return ((uint64_t)cx & 0x1fffff) | (((uint64_t)cy & 0x1fffff) << 21) |
         (((uint64_t)cz & 0x1fffff) << 42);
}
static uint32_t bs_hslot(const r3d_bsurf *t, uint64_t key, bool insert) {
  uint64_t h = key * 0x9E3779B97F4A7C15ull;
  uint32_t s = (uint32_t)(h >> 32) & (t->hcap - 1);
  for (;;) {
    if (t->hkey[s] == key) return s;
    if (t->hkey[s] == UINT64_MAX) return insert ? s : UINT32_MAX;
    s = (s + 1) & (t->hcap - 1);
  }
}
static void bs_hinsert(r3d_bsurf *t, uint32_t cell) {
  uint64_t key = bs_hkey(t, t->pos + (size_t)cell * 3, 0, 0, 0);
  uint32_t s = bs_hslot(t, key, true);
  if (t->hkey[s] != key) {
    t->hkey[s] = key;
    t->hhead[s] = -1;
  }
  t->hnext[cell] = t->hhead[s];
  t->hhead[s] = (int32_t)cell;
}
/* true when a SET node that is not a grid neighbour sits within `lim`
 * of p: the fringe has come round onto covered territory */
static bool bs_overlaps(const r3d_bsurf *t, const double p[3], int32_t i, int32_t j,
                        double lim) {
  double lim2 = lim * lim;
  for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        uint32_t s = bs_hslot(t, bs_hkey(t, p, dx, dy, dz), false);
        if (s == UINT32_MAX) continue;
        for (int32_t c = t->hhead[s]; c >= 0; c = t->hnext[c]) {
          int32_t ci = c % (int32_t)t->W, cj = c / (int32_t)t->W;
          if (abs(ci - i) <= 4 && abs(cj - j) <= 4) continue;
          const double *q = t->pos + (size_t)c * 3;
          double d2 = 0.0;
          for (int a = 0; a < 3; a++) d2 += (q[a] - p[a]) * (q[a] - p[a]);
          if (d2 < lim2) return true;
        }
      }
  return false;
}

/* ---- growth ------------------------------------------------------------ */

static void bs_commit(r3d_bsurf *t, uint32_t k, const int64_t q[3], const double n[3],
                      uint16_t gen) {
  pthread_mutex_lock(&t->mu);
  for (int a = 0; a < 3; a++) {
    t->pos[(size_t)k * 3 + (size_t)a] = (double)q[a];
    t->nrm[(size_t)k * 3 + (size_t)a] = (float)n[a];
  }
  t->state[k] = R3D_BS_SET;
  t->gen_of[k] = gen;
  t->nset++;
  pthread_mutex_unlock(&t->mu);
  bs_hinsert(t, k);
}

/* Try to place grid node (i,j) from its SET neighbours. */
static bool bs_place(r3d_bsurf *t, int32_t i, int32_t j, uint16_t gen) {
  int32_t W = (int32_t)t->W, H = (int32_t)t->H;
  uint32_t k = (uint32_t)j * t->W + (uint32_t)i;
  static const int o4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
#define BS_AT(ii, jj) ((ii) >= 0 && (jj) >= 0 && (ii) < W && (jj) < H && \
                       t->state[(size_t)(jj) * (size_t)W + (size_t)(ii)] == R3D_BS_SET)
#define BS_P(ii, jj) (t->pos + ((size_t)(jj) * (size_t)W + (size_t)(ii)) * 3)
#define BS_N(ii, jj) (t->nrm + ((size_t)(jj) * (size_t)W + (size_t)(ii)) * 3)
  int nn = 0, n8 = 0;
  for (int dj = -1; dj <= 1; dj++)
    for (int di = -1; di <= 1; di++)
      if ((di || dj) && BS_AT(i + di, j + dj)) n8++;
  for (int o = 0; o < 4; o++)
    if (BS_AT(i + o4[o][0], j + o4[o][1])) nn++;
  if (!nn) return false;
  /* local affine frame: weighted least squares p ~ b + Au*di + Av*dj over
   * the SET cells of the 5x5 neighbourhood. Integer-voxel landings jitter
   * by up to half a voxel each, so a fit over ~10-24 cells predicts far
   * better than 2*p1 - p2 from two of them. The frame also gives the
   * sheet normal (Au x Av) independent of the noisy per-voxel normals. */
  double M[9] = {0}, R[9] = {0}; /* M: normal matrix over (1,di,dj); R: rhs per axis */
  double navg[3] = {0, 0, 0};
  int nfit = 0;
  for (int dj = -2; dj <= 2; dj++)
    for (int di = -2; di <= 2; di++) {
      if (!di && !dj) continue;
      if (!BS_AT(i + di, j + dj)) continue;
      double w = exp(-(double)(di * di + dj * dj) / 5.0);
      double bx[3] = {1.0, (double)di, (double)dj};
      const double *pc = BS_P(i + di, j + dj);
      const float *nc = BS_N(i + di, j + dj);
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) M[r * 3 + c] += w * bx[r] * bx[c];
        for (int a = 0; a < 3; a++) R[r * 3 + a] += w * bx[r] * pc[a];
      }
      for (int a = 0; a < 3; a++) navg[a] += w * (double)nc[a];
      nfit++;
    }
  double det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
               M[2] * (M[3] * M[7] - M[4] * M[6]);
  if (nfit < 3 || fabs(det) < 1e-6) return false; /* collinear support: wait */
  double inv[9] = {(M[4] * M[8] - M[5] * M[7]) / det, (M[2] * M[7] - M[1] * M[8]) / det,
                   (M[1] * M[5] - M[2] * M[4]) / det, (M[5] * M[6] - M[3] * M[8]) / det,
                   (M[0] * M[8] - M[2] * M[6]) / det, (M[2] * M[3] - M[0] * M[5]) / det,
                   (M[3] * M[7] - M[4] * M[6]) / det, (M[1] * M[6] - M[0] * M[7]) / det,
                   (M[0] * M[4] - M[1] * M[3]) / det};
  double B[3], Au[3], Av[3]; /* offset, d/di, d/dj per world axis */
  for (int a = 0; a < 3; a++) {
    B[a] = inv[0] * R[a] + inv[1] * R[3 + a] + inv[2] * R[6 + a];
    Au[a] = inv[3] * R[a] + inv[4] * R[3 + a] + inv[5] * R[6 + a];
    Av[a] = inv[6] * R[a] + inv[7] * R[3 + a] + inv[8] * R[6 + a];
  }
  double pred[3] = {B[0], B[1], B[2]}; /* the fit evaluated at (0,0) */
  double nfitv[3] = {Au[1] * Av[2] - Au[2] * Av[1], Au[2] * Av[0] - Au[0] * Av[2],
                     Au[0] * Av[1] - Au[1] * Av[0]};
  double nl = sqrt(nfitv[0] * nfitv[0] + nfitv[1] * nfitv[1] + nfitv[2] * nfitv[2]);
  double npred[3];
  double al = sqrt(navg[0] * navg[0] + navg[1] * navg[1] + navg[2] * navg[2]);
  if (al < 1e-9) return false;
  for (int a = 0; a < 3; a++) navg[a] /= al;
  if (nl > 1e-6) {
    double sgn = (nfitv[0] * navg[0] + nfitv[1] * navg[1] + nfitv[2] * navg[2]) < 0.0
                     ? -1.0 : 1.0;
    for (int a = 0; a < 3; a++) npred[a] = sgn * nfitv[a] / nl + navg[a];
    double l2 = sqrt(npred[0] * npred[0] + npred[1] * npred[1] + npred[2] * npred[2]);
    for (int a = 0; a < 3; a++) npred[a] /= l2;
  } else {
    memcpy(npred, navg, sizeof npred);
  }
  /* spacing spring: the fit reproduces whatever spacing the neighbourhood
   * has, and accepted landings up to the tolerance would ratchet it up
   * generation by generation. Re-seat the prediction at exactly one step
   * (in the sheet) from each grown 4-neighbour and average. */
  {
    double acc[3] = {0, 0, 0};
    int na = 0;
    for (int o = 0; o < 4; o++) {
      int32_t i1 = i + o4[o][0], j1 = j + o4[o][1];
      if (!BS_AT(i1, j1)) continue;
      const double *p1 = BS_P(i1, j1);
      double e[3] = {pred[0] - p1[0], pred[1] - p1[1], pred[2] - p1[2]};
      double en = e[0] * npred[0] + e[1] * npred[1] + e[2] * npred[2];
      double tv[3], tl = 0.0;
      for (int a = 0; a < 3; a++) {
        tv[a] = e[a] - en * npred[a];
        tl += tv[a] * tv[a];
      }
      tl = sqrt(tl);
      if (tl < 1e-6) continue;
      for (int a = 0; a < 3; a++)
        acc[a] += p1[a] + tv[a] * (t->cfg.step / tl) + en * npred[a];
      na++;
    }
    if (na)
      for (int a = 0; a < 3; a++) pred[a] = acc[a] / (double)na;
  }

  int64_t q[3];
  double n[3];
  int why = 0; /* 1 no landing, 2 face disagreement, 3 spacing, 4 overlap */
  bool ok = bs_snap_line(t, pred, npred, q) && bs_normal(t, q, n);
  if (ok && n[0] * npred[0] + n[1] * npred[1] + n[2] * npred[2] < 0.15) ok = false;
  if (!ok) {
    ok = bs_snap_radial(t, pred, npred, true, 0.5 * t->cfg.step, q, n);
    if (!ok) {
      /* distinguish "nothing there" from "only the wrong face there" */
      int64_t q2[3];
      double n2[3];
      why = bs_snap_radial(t, pred, npred, false, 0.0, q2, n2) ? 2 : 1;
      goto fail;
    }
  }
  /* spacing to every SET 4-neighbour, measured IN THE SHEET: the
   * tangential part of the edge must be [0.35, 1.7] step (shorter = the
   * node doubled back, longer = it ran off along the face); the normal
   * part is whatever the snap needed (bounded by snap; an undulating
   * face legitimately has 3-D edges longer than the grid step) */
  {
    double p[3] = {(double)q[0], (double)q[1], (double)q[2]};
    for (int o = 0; o < 4; o++) {
      int32_t i1 = i + o4[o][0], j1 = j + o4[o][1];
      if (!BS_AT(i1, j1)) continue;
      const double *p1 = BS_P(i1, j1);
      double e[3] = {p[0] - p1[0], p[1] - p1[1], p[2] - p1[2]};
      double en = e[0] * npred[0] + e[1] * npred[1] + e[2] * npred[2];
      double dt2 = 0.0;
      for (int a = 0; a < 3; a++) {
        double c = e[a] - en * npred[a];
        dt2 += c * c;
      }
      double dt = sqrt(dt2);
      double lo = nn >= 3 ? 0.25 : 0.5, hi = nn >= 3 ? 2.0 : 1.5; /* enclosed
       * cells (3+ grown sides) are interior holes where fronts met out of
       * phase: fill them rather than leave a seam */
      if (dt < lo * t->cfg.step || dt > hi * t->cfg.step) {
        if (getenv("R3D_BSURF_DEBUG") && t->nwhy[2] < 40)
          fprintf(stderr,
                  "bsurf dbg: (%d,%d) nb(%d,%d) dt %.2f en %.2f | pred (%.1f,%.1f,%.1f) "
                  "land (%lld,%lld,%lld) nb (%.0f,%.0f,%.0f) n (%.2f,%.2f,%.2f) nfit %d\n",
                  i, j, i1, j1, dt, en, pred[0], pred[1], pred[2], (long long)q[0],
                  (long long)q[1], (long long)q[2], p1[0], p1[1], p1[2], npred[0], npred[1],
                  npred[2], nfit);
        why = 3;
        goto fail;
      }
    }
    if (nn < 3 && bs_overlaps(t, p, i, j, 0.5 * t->cfg.step)) {
      why = 4;
      goto fail;
    }
  }
  /* blend the measured normal with the frame it was predicted from so a
   * single noisy landing does not swing the next generation */
  for (int a = 0; a < 3; a++) n[a] = 0.6 * n[a] + 0.4 * npred[a];
  nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (int a = 0; a < 3; a++) n[a] /= nl;
  bs_commit(t, k, q, n, gen);
  return true;
fail:
  pthread_mutex_lock(&t->mu);
  if (t->state[k] != R3D_BS_FAIL) t->nfail++;
  if (why >= 1 && why <= 4) t->nwhy[why - 1]++;
  t->state[k] = R3D_BS_FAIL;
  t->ftry[k] = (uint8_t)n8;
  pthread_mutex_unlock(&t->mu);
  return false;
#undef BS_AT
#undef BS_P
#undef BS_N
}

static int bs_seed(r3d_bsurf *t) {
  /* nearest boundary voxel to the click, any orientation */
  int64_t q0[3];
  double n0[3];
  double snap_save = t->cfg.snap;
  if (t->cfg.snap < 16.0) t->cfg.snap = 16.0; /* clicks are approximate */
  bool ok = bs_snap_radial(t, t->cfg.seed, n0, false, 0.0, q0, n0);
  t->cfg.snap = snap_save;
  {
    /* seed diagnostics: papyrus fraction in the 33^3 around the click */
    int64_t c[3];
    bs_round(t->cfg.seed, c);
    uint32_t np = 0, nt = 0, hist[256] = {0};
    for (int dz = -16; dz <= 16; dz += 2)
      for (int dy = -16; dy <= 16; dy += 2)
        for (int dx = -16; dx <= 16; dx += 2) {
          nt++;
          hist[t->sample(t->sctx, c[0] + dx, c[1] + dy, c[2] + dz)]++;
          if (bs_pap(t, c[0] + dx, c[1] + dy, c[2] + dz)) np++;
        }
    uint32_t pc[5] = {0}, want[5] = {nt / 10, nt / 4, nt / 2, nt * 3 / 4, nt * 9 / 10};
    uint32_t acc = 0;
    for (uint32_t v = 0; v < 256; v++) {
      acc += hist[v];
      for (int q = 0; q < 5; q++)
        if (!pc[q] && acc >= want[q]) pc[q] = v;
    }
    printf("bsurf: seed (%lld,%lld,%lld) low cut %.0f: %.0f%% papyrus nearby%s "
           "(value p10/25/50/75/90 = %u/%u/%u/%u/%u)\n",
           (long long)c[0], (long long)c[1], (long long)c[2], t->cfg.low_cut,
           100.0 * (double)np / (double)nt, ok ? "" : ", no edge found", pc[0], pc[1],
           pc[2], pc[3], pc[4]);
  }
  if (!ok) {
    snprintf(t->status, sizeof t->status,
             "no papyrus/void edge within %.0f vox of the click (low cut %.0f)",
             t->cfg.snap > 16.0 ? t->cfg.snap : 16.0, t->cfg.low_cut);
    return -1;
  }
  /* tangent frame: e1 = axis least aligned with n0, made orthogonal */
  int ax = fabs(n0[0]) < fabs(n0[1]) ? (fabs(n0[0]) < fabs(n0[2]) ? 0 : 2)
                                     : (fabs(n0[1]) < fabs(n0[2]) ? 1 : 2);
  double e1[3] = {0, 0, 0}, e2[3];
  e1[ax] = 1.0;
  double d = e1[0] * n0[0] + e1[1] * n0[1] + e1[2] * n0[2];
  for (int a = 0; a < 3; a++) e1[a] -= d * n0[a];
  double l = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  for (int a = 0; a < 3; a++) e1[a] /= l;
  e2[0] = n0[1] * e1[2] - n0[2] * e1[1];
  e2[1] = n0[2] * e1[0] - n0[0] * e1[2];
  e2[2] = n0[0] * e1[1] - n0[1] * e1[0];
  /* 3x3 seed block, each node snapped along n0 */
  int32_t ci = (int32_t)t->W / 2, cj = (int32_t)t->H / 2;
  uint32_t placed = 0;
  for (int b = -1; b <= 1; b++)
    for (int a2 = -1; a2 <= 1; a2++) {
      double pred[3], n[3];
      int64_t q[3];
      for (int a = 0; a < 3; a++)
        pred[a] = (double)q0[a] + t->cfg.step * ((double)a2 * e1[a] + (double)b * e2[a]);
      bool got = (!a2 && !b) ? (memcpy(q, q0, sizeof q), memcpy(n, n0, sizeof n), true)
                             : (bs_snap_line(t, pred, n0, q) && bs_normal(t, q, n) &&
                                n[0] * n0[0] + n[1] * n0[1] + n[2] * n0[2] > 0.3) ||
                                   bs_snap_radial(t, pred, n0, true, 0.5 * t->cfg.step, q, n);
      if (!got) continue;
      uint32_t k = (uint32_t)(cj + b) * t->W + (uint32_t)(ci + a2);
      bs_commit(t, k, q, n, 1);
      placed++;
    }
  if (placed < 4) {
    snprintf(t->status, sizeof t->status,
             "edge at (%lld,%lld,%lld) too small to seed a %.0f-vox grid",
             (long long)q0[0], (long long)q0[1], (long long)q0[2], t->cfg.step);
    return -1;
  }
  return 0;
}

static void bs_grow(r3d_bsurf *t) {
  if (bs_seed(t) != 0) {
    pthread_mutex_lock(&t->mu);
    t->failed = true;
    t->done = true;
    t->gen++;
    pthread_mutex_unlock(&t->mu);
    return;
  }
  int32_t W = (int32_t)t->W, H = (int32_t)t->H;
  uint32_t *cand = malloc((size_t)W * (size_t)H * sizeof *cand);
  uint8_t *score = malloc((size_t)W * (size_t)H);
  static const int o8[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1},
                               {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  for (uint32_t g = 0; g < t->cfg.gens && !t->quit && cand && score; g++) {
    /* candidates: EMPTY (or FAIL that gained support) with a SET
     * 8-neighbour; most-supported first, so weakly framed nodes see the
     * strong ones already committed */
    uint32_t nc = 0;
    for (int32_t j = 0; j < H; j++)
      for (int32_t i = 0; i < W; i++) {
        uint32_t k = (uint32_t)j * t->W + (uint32_t)i;
        if (t->state[k] == R3D_BS_SET) continue;
        int n4 = 0, n8 = 0;
        for (int o = 0; o < 8; o++) {
          int32_t ii = i + o8[o][0], jj = j + o8[o][1];
          if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue;
          if (t->state[(size_t)jj * (size_t)W + (size_t)ii] != R3D_BS_SET) continue;
          n8++;
          if (o < 4) n4++;
        }
        if (!n8) continue; /* diagonal-only cells join too: their L-corner
                            * usually completes earlier in this same
                            * generation (higher-scored cells go first) */
        if (t->state[k] == R3D_BS_FAIL && n8 <= t->ftry[k] && (g & 3) != 3) continue;
        cand[nc] = k;
        score[nc] = (uint8_t)(n4 * 4 + n8);
        nc++;
      }
    if (!nc) break;
    /* counting sort by score, descending */
    uint32_t cnt[40] = {0};
    for (uint32_t c = 0; c < nc; c++) cnt[score[c]]++;
    uint32_t start[40], acc = 0;
    for (int s = 39; s >= 0; s--) {
      start[s] = acc;
      acc += cnt[s];
    }
    uint32_t *order = malloc((size_t)nc * sizeof *order);
    if (!order) break;
    for (uint32_t c = 0; c < nc; c++) order[start[score[c]]++] = cand[c];
    uint32_t placed = 0;
    for (uint32_t c = 0; c < nc && !t->quit; c++) {
      uint32_t k = order[c];
      if (t->state[k] == R3D_BS_SET) continue;
      if (bs_place(t, (int32_t)(k % t->W), (int32_t)(k / t->W), (uint16_t)(g + 2)))
        placed++;
    }
    free(order);
    pthread_mutex_lock(&t->mu);
    t->ring = g + 1;
    t->gen++;
    snprintf(t->status, sizeof t->status,
             "gen %u/%u: %u nodes (+%u), %u failed (no edge %u, other face %u, "
             "spacing %u, overlap %u)",
             t->ring, t->cfg.gens, t->nset, placed, t->nfail, t->nwhy[0], t->nwhy[1],
             t->nwhy[2], t->nwhy[3]);
    pthread_mutex_unlock(&t->mu);
    if (!placed) break; /* fringe exhausted */
  }
  free(cand);
  free(score);
  /* face QC: papyrus 2 voxels behind each node, void 2 voxels ahead */
  uint32_t qok = 0;
  for (size_t k = 0; k < (size_t)W * (size_t)H; k++) {
    if (t->state[k] != R3D_BS_SET) continue;
    const double *p = t->pos + k * 3;
    const float *nn = t->nrm + k * 3;
    int64_t b[3], f[3];
    for (int a = 0; a < 3; a++) {
      b[a] = (int64_t)llround(p[a] - 2.0 * (double)nn[a]);
      f[a] = (int64_t)llround(p[a] + 2.0 * (double)nn[a]);
    }
    if (bs_pap(t, b[0], b[1], b[2]) && !bs_pap(t, f[0], f[1], f[2])) qok++;
  }
  pthread_mutex_lock(&t->mu);
  t->done = true;
  t->gen++;
  t->face_ok = t->nset ? (float)qok / (float)t->nset : 0.0f;
  snprintf(t->status, sizeof t->status,
           "done: %u generations, %u nodes, face ok %.0f%%; %u failed (no edge %u, "
           "other face %u, spacing %u, overlap %u)",
           t->ring, t->nset, 100.0 * (double)t->face_ok, t->nfail, t->nwhy[0], t->nwhy[1],
           t->nwhy[2], t->nwhy[3]);
  pthread_mutex_unlock(&t->mu);
}

static int bs_init(r3d_bsurf *t, const r3d_bsurf_cfg *cfg, r3d_bsurf_sample_fn fn,
                   void *ctx) {
  memset(t, 0, sizeof *t);
  t->cfg = *cfg;
  if (t->cfg.step < 2.0) t->cfg.step = 2.0;
  if (t->cfg.snap < 1.0) t->cfg.snap = 1.0;
  if (t->cfg.gens < 1) t->cfg.gens = 1;
  if (t->cfg.gens > 2000) t->cfg.gens = 2000;
  t->sample = fn;
  t->sctx = ctx;
  t->W = t->H = 2 * t->cfg.gens + 6;
  size_t n = (size_t)t->W * t->H;
  t->pos = malloc(n * 3 * sizeof *t->pos);
  t->nrm = calloc(n * 3, sizeof *t->nrm);
  t->state = calloc(n, 1);
  t->gen_of = calloc(n, sizeof *t->gen_of);
  t->ftry = calloc(n, 1);
  t->hcap = 1;
  while (t->hcap < 2 * n) t->hcap <<= 1;
  t->hkey = malloc((size_t)t->hcap * sizeof *t->hkey);
  t->hhead = malloc((size_t)t->hcap * sizeof *t->hhead);
  t->hnext = malloc(n * sizeof *t->hnext);
  if (!t->pos || !t->nrm || !t->state || !t->gen_of || !t->ftry || !t->hkey ||
      !t->hhead || !t->hnext) {
    r3d_bsurf_free(t);
    return -1;
  }
  for (size_t k = 0; k < n * 3; k++) t->pos[k] = -1.0;
  memset(t->hkey, 0xff, (size_t)t->hcap * sizeof *t->hkey);
  pthread_mutex_init(&t->mu, NULL);
  snprintf(t->status, sizeof t->status, "seeding");
  return 0;
}

static void *bs_thread(void *arg) {
  r3d_bsurf *t = arg;
  bs_grow(t);
  pthread_mutex_lock(&t->mu);
  t->running = false;
  pthread_mutex_unlock(&t->mu);
  return NULL;
}

int r3d_bsurf_start(r3d_bsurf *t, const r3d_bsurf_cfg *cfg, r3d_bsurf_sample_fn fn,
                    void *ctx) {
  if (bs_init(t, cfg, fn, ctx) != 0) return -1;
  t->running = true;
  if (pthread_create(&t->th, NULL, bs_thread, t) != 0) {
    t->running = false;
    r3d_bsurf_free(t);
    return -1;
  }
  return 0;
}

int r3d_bsurf_run(r3d_bsurf *t, const r3d_bsurf_cfg *cfg, r3d_bsurf_sample_fn fn,
                  void *ctx) {
  if (bs_init(t, cfg, fn, ctx) != 0) return -1;
  bs_grow(t);
  return t->failed ? -1 : 0;
}

void r3d_bsurf_stop(r3d_bsurf *t) {
  if (!t->pos) return;
  t->quit = true;
  if (t->th) pthread_join(t->th, NULL);
  t->th = 0;
  t->running = false;
  t->done = true;
}

void r3d_bsurf_free(r3d_bsurf *t) {
  if (t->th) r3d_bsurf_stop(t);
  free(t->pos);
  free(t->nrm);
  free(t->state);
  free(t->gen_of);
  free(t->ftry);
  free(t->hkey);
  free(t->hhead);
  free(t->hnext);
  if (t->pos) pthread_mutex_destroy(&t->mu);
  memset(t, 0, sizeof *t);
}

uint64_t r3d_bsurf_snapshot(r3d_bsurf *t, double *pos, uint32_t *ring, uint32_t *nset,
                            bool *done) {
  if (!t->pos) return 0;
  pthread_mutex_lock(&t->mu);
  if (pos) {
    size_t n = (size_t)t->W * t->H;
    for (size_t k = 0; k < n; k++)
      for (int a = 0; a < 3; a++)
        pos[k * 3 + (size_t)a] = t->state[k] == R3D_BS_SET ? t->pos[k * 3 + (size_t)a] : -1.0;
  }
  if (ring) *ring = t->ring;
  if (nset) *nset = t->nset;
  if (done) *done = t->done;
  uint64_t g = t->gen;
  pthread_mutex_unlock(&t->mu);
  return g;
}

/* ---- selftest ---------------------------------------------------------- */

typedef struct {
  int kind; /* 0 = sphere shell, 1 = tilted slab */
} bs_synth;
static uint8_t bs_synth_at(void *ctx, int64_t x, int64_t y, int64_t z) {
  const bs_synth *s = ctx;
  if (s->kind == 0) {
    double dx = (double)x - 64.0, dy = (double)y - 64.0, dz = (double)z - 64.0;
    double r = sqrt(dx * dx + dy * dy + dz * dz);
    return (r >= 32.0 && r < 40.0) ? 200 : 20;
  }
  /* slab: 0.3x + 0.2y + z in [60, 68) */
  double v = 0.3 * (double)x + 0.2 * (double)y + (double)z;
  return (v >= 60.0 && v < 68.0) ? 200 : 20;
}

int r3d_bsurf_selftest(void) {
  int rc = 0;
  /* sphere shell, seed just outside the outer face: every node must sit
   * on the outer face (r ~ 39..40), none on the inner face (r ~ 32) */
  {
    bs_synth s = {.kind = 0};
    r3d_bsurf_cfg c = {.seed = {64.0, 64.0, 106.0}, .step = 4.0, .gens = 40, .snap = 6.0,
                       .low_cut = 100.0, .vdim = {128, 128, 128}};
    r3d_bsurf t;
    if (r3d_bsurf_run(&t, &c, bs_synth_at, &s) != 0) {
      fprintf(stderr, "bsurf selftest: sphere seed failed: %s\n", t.status);
      rc = -1;
    } else {
      uint32_t bad = 0;
      double rmin = 1e30, rmax = 0.0;
      for (size_t k = 0; k < (size_t)t.W * t.H; k++) {
        if (t.state[k] != R3D_BS_SET) continue;
        double dx = t.pos[k * 3] - 64.0, dy = t.pos[k * 3 + 1] - 64.0,
               dz = t.pos[k * 3 + 2] - 64.0;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < rmin) rmin = r;
        if (r > rmax) rmax = r;
        if (r < 38.0 || r >= 40.0) bad++;
      }
      /* the outer face of a 40-radius sphere at 4-vox pitch is ~1250
       * nodes; growth must cover most of it from one seed */
      if (t.nset < 700 || bad) {
        fprintf(stderr, "bsurf selftest: sphere nset=%u bad=%u r=[%.2f,%.2f] (%s)\n",
                t.nset, bad, rmin, rmax, t.status);
        rc = -1;
      }
    }
    r3d_bsurf_free(&t);
  }
  /* tilted slab: seed inside the slab near its upper face; all nodes on
   * the upper face (v in [67,68)), spacing sane, grid extent reached */
  {
    bs_synth s = {.kind = 1};
    r3d_bsurf_cfg c = {.seed = {64.0, 64.0, 35.0}, .step = 3.0, .gens = 12, .snap = 5.0,
                       .low_cut = 100.0, .vdim = {128, 128, 128}};
    r3d_bsurf t;
    if (r3d_bsurf_run(&t, &c, bs_synth_at, &s) != 0) {
      fprintf(stderr, "bsurf selftest: slab seed failed: %s\n", t.status);
      rc = -1;
    } else {
      uint32_t bad = 0;
      for (size_t k = 0; k < (size_t)t.W * t.H; k++) {
        if (t.state[k] != R3D_BS_SET) continue;
        double v = 0.3 * t.pos[k * 3] + 0.2 * t.pos[k * 3 + 1] + t.pos[k * 3 + 2];
        if (v < 66.5 || v >= 68.0) bad++;
      }
      uint32_t full = (2 * c.gens + 6) * (2 * c.gens + 6);
      if (bad || t.nset < full * 8 / 10) {
        fprintf(stderr, "bsurf selftest: slab nset=%u/%u bad=%u (%s)\n", t.nset, full,
                bad, t.status);
        rc = -1;
      }
    }
    r3d_bsurf_free(&t);
  }
  return rc;
}
