#include "core/tracer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

#include "core/cpuvol.h"

/* umbilicus x/y at z (clamped linear; volume-center fallback handled by the
 * caller passing an empty umbilicus) */
static bool tr_umb(const r3d_umbilicus *u, double z, double *x, double *y) {
  if (u->count == 0) return false;
  const r3d_umbilicus_point *p = u->points;
  size_t n = u->count;
  if (z <= p[0].z) {
    *x = p[0].x;
    *y = p[0].y;
    return true;
  }
  if (z >= p[n - 1].z) {
    *x = p[n - 1].x;
    *y = p[n - 1].y;
    return true;
  }
  for (size_t i = 1; i < n; i++)
    if (p[i].z >= z) {
      double dz = p[i].z - p[i - 1].z;
      double t = dz > 0.0 ? (z - p[i - 1].z) / dz : 0.0;
      *x = p[i - 1].x + (p[i].x - p[i - 1].x) * t;
      *y = p[i - 1].y + (p[i].y - p[i - 1].y) * t;
      return true;
    }
  return false;
}

/* local sheet frame at P: n = radial from the winding axis (sheet normal),
 * t = tangential (u direction), z = (0,0,1) (v direction) */
static void tr_frame(const r3d_tracer *t, const double P[3], double nrm[3], double tan_[3]) {
  double cx = 0.0, cy = 0.0;
  if (!tr_umb(&t->umb, P[2], &cx, &cy)) {
    cx = P[0] - 1.0; /* no umbilicus: a degenerate but stable fallback frame */
    cy = P[1];
  }
  double rx = P[0] - cx, ry = P[1] - cy;
  double rl = sqrt(rx * rx + ry * ry);
  if (rl < 1e-6) {
    rx = 1.0;
    ry = 0.0;
    rl = 1.0;
  }
  nrm[0] = rx / rl;
  nrm[1] = ry / rl;
  nrm[2] = 0.0;
  tan_[0] = -nrm[1];
  tan_[1] = nrm[0];
  tan_[2] = 0.0;
}

#define TR_FLOOR 0.10 /* below this the sheet is truly gone: cell fails */
#define TR_STRONG 0.45 /* full ridge trust above this confidence */

/* local sheet normal from the traced grid itself: cross of the grid's u/v
 * tangents around cell (i,j). This lets z participate — the cylindrical
 * radial frame locked every row to constant z, which forced the trace to
 * slide onto neighboring wraps wherever the real sheet tilts. Falls back
 * to the radial frame until enough neighbors exist. */
static void tr_grid_normal(const r3d_tracer *t, int i, int j, const double P[3],
                           double n[3]) {
  double du[3] = {0, 0, 0}, dv[3] = {0, 0, 0};
  bool hu = false, hv = false;
  for (int s2 = -1; s2 <= 1 && !hu; s2 += 2) {
    int a2 = i + s2, b2 = i - s2;
    if (a2 < 0 || a2 >= (int)t->W) continue;
    size_t ka = (size_t)j * t->W + (size_t)a2;
    if (t->state[ka] != R3D_TR_SET) continue;
    const double *pa = t->pos + ka * 3;
    const double *pb = P;
    if (b2 >= 0 && b2 < (int)t->W) {
      size_t kb = (size_t)j * t->W + (size_t)b2;
      if (t->state[kb] == R3D_TR_SET) pb = t->pos + kb * 3;
    }
    for (int a = 0; a < 3; a++) du[a] = (pa[a] - pb[a]) * s2;
    hu = true;
  }
  for (int s2 = -1; s2 <= 1 && !hv; s2 += 2) {
    int a2 = j + s2, b2 = j - s2;
    if (a2 < 0 || a2 >= (int)t->H) continue;
    size_t ka = (size_t)a2 * t->W + (size_t)i;
    if (t->state[ka] != R3D_TR_SET) continue;
    const double *pa = t->pos + ka * 3;
    const double *pb = P;
    if (b2 >= 0 && b2 < (int)t->H) {
      size_t kb = (size_t)b2 * t->W + (size_t)i;
      if (t->state[kb] == R3D_TR_SET) pb = t->pos + kb * 3;
    }
    for (int a = 0; a < 3; a++) dv[a] = (pa[a] - pb[a]) * s2;
    hv = true;
  }
  if (hu && hv) {
    n[0] = du[1] * dv[2] - du[2] * dv[1];
    n[1] = du[2] * dv[0] - du[0] * dv[2];
    n[2] = du[0] * dv[1] - du[1] * dv[0];
    double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (l > 1e-6) {
      double rad[3], tn0[3]; /* orient outward like the radial frame */
      tr_frame(t, P, rad, tn0);
      double d2 = (n[0] * rad[0] + n[1] * rad[1] + n[2] * rad[2]) / l;
      double s3 = d2 < 0.0 ? -1.0 : 1.0;
      for (int a = 0; a < 3; a++) n[a] = n[a] / l * s3;
      return;
    }
  }
  double tn0[3];
  tr_frame(t, P, n, tn0); /* fallback: radial */
}

/* snap P to the prediction ridge along the sheet normal; returns best value
 * (0..1). Weak confidence trusts the extrapolated position over the noisy
 * ridge max (continuous blend, no binary accept). */
static double tr_snap(r3d_tracer *t, r3d_cpuvol *v, double P[3], double search,
                      const double nrm_in[3]) {
  double nrm[3], tn[3];
  if (nrm_in) memcpy(nrm, nrm_in, sizeof nrm);
  else tr_frame(t, P, nrm, tn);
  double best = -1.0, bs = 0.0;
  for (double s = -search; s <= search; s += 1.0) {
    double val = (double)r3d_cpuvol_at(v, t->cfg.level, P[0] + nrm[0] * s,
                                       P[1] + nrm[1] * s, P[2] + nrm[2] * s) /
                 255.0;
    if (val > best) {
      best = val;
      bs = s;
    }
  }
  for (double s = bs - 0.75; s <= bs + 0.75; s += 0.25) { /* sub-voxel refine */
    double val = (double)r3d_cpuvol_at(v, t->cfg.level, P[0] + nrm[0] * s,
                                       P[1] + nrm[1] * s, P[2] + nrm[2] * s) /
                 255.0;
    if (val > best) {
      best = val;
      bs = s;
    }
  }
  double trust = best >= TR_STRONG ? 1.0 : (best <= TR_FLOOR ? 0.0
                                            : (best - TR_FLOOR) / (TR_STRONG - TR_FLOOR));
  P[0] += nrm[0] * bs * trust;
  P[1] += nrm[1] * bs * trust;
  P[2] += nrm[2] * bs * trust;
  if (best <= 0.0) return best;
  /* ridge contrast: a point in the blur BETWEEN sheets has value but no
   * valley on either side — those are the wrong-sheet captures. Scale
   * confidence by how much the ridge stands above its surroundings. */
  double off = 0.0;
  for (int sgn = -1; sgn <= 1; sgn += 2) {
    double ov = (double)r3d_cpuvol_at(v, t->cfg.level, P[0] + nrm[0] * 7.0 * sgn,
                                      P[1] + nrm[1] * 7.0 * sgn,
                                      P[2] + nrm[2] * 7.0 * sgn) /
                255.0;
    if (ov > off) off = ov;
  }
  double contrast = best > 1e-6 ? (best - off) / best : 0.0;
  if (contrast < 0.0) contrast = 0.0;
  return best * (0.65 + 0.35 * contrast); /* gentle: shift the cutoff
                       * meaning, don't starve it */
}

static inline double *tr_p(r3d_tracer *t, uint32_t i, uint32_t j) {
  return t->pos + ((size_t)j * t->W + i) * 3;
}

/* predicted position of cell (i,j) from its parent one step along one grid
 * axis: linear continuation when the grandparent exists, else a frame step */
static bool tr_predict(r3d_tracer *t, int i, int j, int si, int sj, double out[3]) {
  int pi = i + si, pj = j + sj;
  if (pi < 0 || pj < 0 || pi >= (int)t->W || pj >= (int)t->H) return false;
  size_t k1 = (size_t)pj * t->W + (size_t)pi;
  if (t->state[k1] != R3D_TR_SET) return false;
  const double *p1 = t->pos + k1 * 3;
  int gi = i + 2 * si, gj = j + 2 * sj;
  if (gi >= 0 && gj >= 0 && gi < (int)t->W && gj < (int)t->H) {
    size_t k2 = (size_t)gj * t->W + (size_t)gi;
    if (t->state[k2] == R3D_TR_SET) {
      const double *p2 = t->pos + k2 * 3;
      for (int a = 0; a < 3; a++) out[a] = 2.0 * p1[a] - p2[a];
      return true;
    }
  }
  double nrm[3], tn[3];
  tr_frame(t, p1, nrm, tn);
  for (int a = 0; a < 3; a++)
    out[a] = p1[a] + (tn[a] * -si + (a == 2 ? (double)-sj : 0.0)) * t->cfg.step;
  return true;
}

/* march from a parent position toward a target one grid step away in
 * ~3-voxel sub-steps, re-snapping with a small search each time so the
 * point slides ALONG its sheet instead of leaping across the gap to a
 * neighboring one. Returns the final snap confidence. */
static double tr_advance(r3d_tracer *t, r3d_cpuvol *v, const double from[3],
                         const double target[3], const double nrm[3], double out[3]) {
  double d[3] = {target[0] - from[0], target[1] - from[1], target[2] - from[2]};
  double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  double ms = t->cfg.march > 0.5 ? t->cfg.march : 3.0;
  int n = (int)(L / ms) + 1;
  memcpy(out, from, 3 * sizeof(double));
  double val = 0.0;
  for (int s2 = 0; s2 < n; s2++) {
    for (int a = 0; a < 3; a++) out[a] += d[a] / n;
    double sr = ms + 1.0 < 3.0 ? 3.0 : ms + 1.0;
    val = tr_snap(t, v, out, sr, nrm);
    if (val < TR_FLOOR && s2 + 1 < n) { /* mid-gap: keep marching straight,
                                         * the far side may pick it back up */
      continue;
    }
  }
  return val;
}

/* quality control for cell (i,j): cut vertices that tore the grid —
 * either far from where ALL their neighbors expect them (spacing), or
 * with a winding-radius jump against both u-neighbors (wrap hop). An
 * honest hole beats a committed discontinuity. */
static void tr_qc_cell(r3d_tracer *t, int i, int j) {
  size_t k = (size_t)j * t->W + (size_t)i;
  if (t->state[k] != R3D_TR_SET) return;
  const double *pc = t->pos + k * 3;
  double cx, cy, rc = -1.0;
  if (tr_umb(&t->umb, pc[2], &cx, &cy))
    rc = sqrt((pc[0] - cx) * (pc[0] - cx) + (pc[1] - cy) * (pc[1] - cy));
  double avg[3] = {0, 0, 0};
  int na = 0, rbad = 0, ru = 0;
  static const int off[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (int o = 0; o < 4; o++) {
    int ii = i + off[o][0], jj = j + off[o][1];
    if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
    size_t kk = (size_t)jj * t->W + (size_t)ii;
    if (t->state[kk] != R3D_TR_SET) continue;
    double nrm[3], tn[3];
    const double *pn = t->pos + kk * 3;
    tr_frame(t, pn, nrm, tn);
    for (int a = 0; a < 3; a++)
      avg[a] += pn[a] -
                (tn[a] * off[o][0] + (a == 2 ? (double)off[o][1] : 0.0)) * t->cfg.step;
    na++;
    if (off[o][1] == 0 && rc >= 0.0) { /* u-neighbor: winding radius check */
      double nx2, ny2;
      if (tr_umb(&t->umb, pn[2], &nx2, &ny2)) {
        double rn2 = sqrt((pn[0] - nx2) * (pn[0] - nx2) + (pn[1] - ny2) * (pn[1] - ny2));
        ru++;
        if (fabs(rn2 - rc) > 10.0) rbad++;
      }
    }
  }
  if (na < 2) return;
  double err = 0.0;
  for (int a = 0; a < 3; a++) {
    double d = pc[a] - avg[a] / na;
    err += d * d;
  }
  bool tear = sqrt(err) > 0.6 * t->cfg.step;
  bool hop = ru >= 2 && rbad == ru; /* radius jumps against every u-neighbor */
  if (tear || hop) {
    pthread_mutex_lock(&t->mu);
    if (t->cfg.fill) { /* repair, don't cut: re-seat on the consensus */
      for (int a = 0; a < 3; a++) t->pos[k * 3 + (size_t)a] = avg[a] / na;
      t->conf[k] *= 0.25f;
    } else {
      t->state[k] = R3D_TR_FAIL;
      t->conf[k] *= 0.25f;
      if (t->nset) t->nset--;
    }
    pthread_mutex_unlock(&t->mu);
  }
}

/* one confidence-weighted relaxation of cell (i,j): pull toward the
 * positions its SET neighbors expect, then re-snap */
static void tr_relax_cell(r3d_tracer *t, r3d_cpuvol *vol, int i, int j) {
  size_t k = (size_t)j * t->W + (size_t)i;
  if (t->state[k] != R3D_TR_SET) return;
  double avg[3] = {0, 0, 0};
  int na = 0;
  static const int off[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (int o = 0; o < 4; o++) {
    int ii = i + off[o][0], jj = j + off[o][1];
    if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
    size_t kk = (size_t)jj * t->W + (size_t)ii;
    if (t->state[kk] != R3D_TR_SET) continue;
    double nrm[3], tn[3];
    const double *pn = t->pos + kk * 3;
    tr_frame(t, pn, nrm, tn);
    for (int a = 0; a < 3; a++)
      avg[a] += pn[a] -
                (tn[a] * off[o][0] + (a == 2 ? (double)off[o][1] : 0.0)) * t->cfg.step;
    na++;
  }
  if (na < 2) return;
  double P[3], gn2[3];
  const double *pc = t->pos + k * 3;
  double w = 0.25 + 0.55 * (1.0 - (double)t->conf[k]);
  for (int a = 0; a < 3; a++) P[a] = (1.0 - w) * pc[a] + w * avg[a] / na;
  tr_grid_normal(t, i, j, P, gn2);
  double val = tr_snap(t, vol, P, 4.0, gn2);
  if (val >= TR_FLOOR) {
    pthread_mutex_lock(&t->mu);
    memcpy(t->pos + k * 3, P, sizeof P);
    t->conf[k] = (float)val;
    pthread_mutex_unlock(&t->mu);
  }
}

static void *tr_worker(void *ud) {

  r3d_tracer *t = ud;
  r3d_cpuvol vol;
  if (r3d_cpuvol_open(&vol, t->root, 96) != 0) {
    pthread_mutex_lock(&t->mu);
    t->done = true;
    t->running = false;
    pthread_mutex_unlock(&t->mu);
    return NULL;
  }
  uint32_t c = t->cfg.max_ring; /* seed cell */
  bool dead = false;
  if (t->state[(size_t)c * t->W + c] != R3D_TR_SET) { /* fresh start */
    double *sp = tr_p(t, c, c);
    double sv = -1.0;
    for (uint32_t lv = t->cfg.level; lv < t->cfg.level + 3 && lv < vol.nlev; lv++) {
      /* locally-cached predictions may start at a coarser level: probe down */
      memcpy(sp, t->cfg.seed, 3 * sizeof(double));
      t->cfg.level = lv;
      sv = tr_snap(t, &vol, sp, t->cfg.search, NULL);
      if (sv >= (double)t->cfg.thresh) break;
    }
    pthread_mutex_lock(&t->mu);
    t->state[(size_t)c * t->W + c] = sv >= TR_FLOOR ? R3D_TR_SET : R3D_TR_FAIL;
    t->conf[(size_t)c * t->W + c] = (float)(sv > 0.0 ? sv : 0.0);
    t->nset = sv >= TR_FLOOR ? 1u : 0u;
    t->gen++;
    dead = t->nset == 0;
    pthread_mutex_unlock(&t->mu);
  }

  for (uint32_t R = 1; R <= t->cfg.max_ring && !dead && !t->quit; R++) {
    uint32_t grew = 0;
    for (int dj = -(int)R; dj <= (int)R; dj++)
      for (int di = -(int)R; di <= (int)R; di++) {
        if (abs(di) != (int)R && abs(dj) != (int)R) continue;
        int i = (int)c + di, j = (int)c + dj;
        if (i < 0 || j < 0 || i >= (int)t->W || j >= (int)t->H) continue;
        size_t k = (size_t)j * t->W + (size_t)i;
        if (t->state[k] != R3D_TR_EMPTY) continue;
        int si = di > 0 ? -1 : (di < 0 ? 1 : 0);
        int sj = dj > 0 ? -1 : (dj < 0 ? 1 : 0);
        /* average the predictions of BOTH inward-adjacent parents: single-
         * parent chains never mix across the diagonals and leave starburst
         * seams radiating from the seed */
        double P[3] = {0, 0, 0}, pr[3], mv3[3], gn[3];
        int np = 0;
        double vsum = 0.0;
        if (si && tr_predict(t, i, j, si, 0, pr)) {
          const double *pp = t->pos + ((size_t)j * t->W + (size_t)(i + si)) * 3;
          tr_grid_normal(t, i + si, j, pp, gn);
          vsum += tr_advance(t, &vol, pp, pr, gn, mv3);
          for (int a = 0; a < 3; a++) P[a] += mv3[a];
          np++;
        }
        if (sj && tr_predict(t, i, j, 0, sj, pr)) {
          const double *pp = t->pos + ((size_t)(j + sj) * t->W + (size_t)i) * 3;
          tr_grid_normal(t, i, j + sj, pp, gn);
          vsum += tr_advance(t, &vol, pp, pr, gn, mv3);
          for (int a = 0; a < 3; a++) P[a] += mv3[a];
          np++;
        }
        if (!np && si && sj && tr_predict(t, i, j, si, sj, pr)) {
          const double *pp = t->pos + ((size_t)(j + sj) * t->W + (size_t)(i + si)) * 3;
          tr_grid_normal(t, i + si, j + sj, pp, gn);
          vsum += tr_advance(t, &vol, pp, pr, gn, mv3); /* corner fallback */
          for (int a = 0; a < 3; a++) P[a] += mv3[a];
          np++;
        }
        if (!np) continue;
        for (int a = 0; a < 3; a++) P[a] /= np;
        tr_grid_normal(t, i, j, P, gn);
        double val = tr_snap(t, &vol, P, 4.0, gn);
        if (val < vsum / np) val = vsum / np; /* marched confidence counts */
        pthread_mutex_lock(&t->mu);
        if (val >= TR_FLOOR || t->cfg.fill) {
          /* fill mode: even prediction-dead cells take the continued
           * position — a complete grid with honest low confidence beats
           * a hole; the save cutoff decides what survives */
          memcpy(t->pos + k * 3, P, sizeof P);
          t->state[k] = R3D_TR_SET;
          t->conf[k] = (float)(val > 0.0 ? val : 0.0);
          t->nset++;
          grew++;
        } else {
          t->state[k] = R3D_TR_FAIL;
          t->conf[k] = (float)(val > 0.0 ? val : 0.0);
        }
        pthread_mutex_unlock(&t->mu);
      }
    /* relaxation: two passes over the band behind the frontier, plus a
     * full-disc pass every 8 rings to bleed out accumulated drift */
    uint32_t r0 = R > 3 ? R - 3 : 1;
    for (int it2 = 0; it2 < 2 && !t->quit; it2++)
      for (int dj = -(int)R; dj <= (int)R; dj++)
        for (int di = -(int)R; di <= (int)R; di++) {
          uint32_t cr = (uint32_t)(abs(di) > abs(dj) ? abs(di) : abs(dj));
          if (cr < r0 || cr > R) continue;
          int i = (int)c + di, j = (int)c + dj;
          if (i < 0 || j < 0 || i >= (int)t->W || j >= (int)t->H) continue;
          tr_relax_cell(t, &vol, i, j);
        }
    for (int dj = -(int)R; dj <= (int)R && !t->quit; dj++)
      for (int di = -(int)R; di <= (int)R; di++) {
        uint32_t cr = (uint32_t)(abs(di) > abs(dj) ? abs(di) : abs(dj));
        if (cr < r0 || cr > R) continue;
        int i = (int)c + di, j = (int)c + dj;
        if (i < 0 || j < 0 || i >= (int)t->W || j >= (int)t->H) continue;
        tr_qc_cell(t, i, j);
      }
    if ((R % 8u == 0u || R == t->cfg.max_ring) && !t->quit)
      for (int dj = -(int)R; dj <= (int)R; dj++)
        for (int di = -(int)R; di <= (int)R; di++) {
          int i = (int)c + di, j = (int)c + dj;
          if (i < 0 || j < 0 || i >= (int)t->W || j >= (int)t->H) continue;
          tr_relax_cell(t, &vol, i, j);
        }
    pthread_mutex_lock(&t->mu);
    t->ring = R;
    t->gen++;
    pthread_mutex_unlock(&t->mu);
    if (grew == 0) dead = true;
  }
  r3d_cpuvol_close(&vol);
  printf("tracer: finished at ring %u with %u point%s (level L%u)\n", t->ring, t->nset,
         t->nset == 1 ? "" : "s", t->cfg.level);
  pthread_mutex_lock(&t->mu);
  t->done = true;
  t->running = false;
  t->gen++;
  pthread_mutex_unlock(&t->mu);
  return NULL;
}

int r3d_tracer_start(r3d_tracer *t, const char *pred_root, const r3d_tracer_cfg *cfg,
                     const r3d_umbilicus *umb) {
  memset(t, 0, sizeof *t);
  t->cfg = *cfg;
  if (t->cfg.max_ring < 4) t->cfg.max_ring = 4;
  if (t->cfg.march < 1.0) t->cfg.march = 3.0;
  if (t->cfg.march > t->cfg.step) t->cfg.march = t->cfg.step;
  if (t->cfg.max_ring > 400) t->cfg.max_ring = 400;
  snprintf(t->root, sizeof t->root, "%s", pred_root);
  t->W = t->H = 2 * t->cfg.max_ring + 1;
  t->pos = calloc((size_t)t->W * t->H * 3, sizeof *t->pos);
  t->state = calloc((size_t)t->W * t->H, 1);
  t->conf = calloc((size_t)t->W * t->H, sizeof *t->conf);
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
  uint32_t NW = 2 * nr + 1, off = nr - t->cfg.max_ring;
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
      if (t->state[ok] == R3D_TR_SET) { /* FAILED cells retry as EMPTY */
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
  t->ring = 0;
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
