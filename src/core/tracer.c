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

/* snap P to the prediction ridge along the sheet normal; returns best value
 * (0..1), P updated in place */
static double tr_snap(r3d_tracer *t, r3d_cpuvol *v, double P[3]) {
  double nrm[3], tn[3];
  tr_frame(t, P, nrm, tn);
  double best = -1.0, bs = 0.0;
  for (double s = -t->cfg.search; s <= t->cfg.search; s += 1.0) {
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
  P[0] += nrm[0] * bs;
  P[1] += nrm[1] * bs;
  P[2] += nrm[2] * bs;
  return best;
}

static inline double *tr_p(r3d_tracer *t, uint32_t i, uint32_t j) {
  return t->pos + ((size_t)j * t->W + i) * 3;
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
      sv = tr_snap(t, &vol, sp);
      if (sv >= (double)t->cfg.thresh) break;
    }
    pthread_mutex_lock(&t->mu);
    t->state[(size_t)c * t->W + c] =
        sv >= (double)t->cfg.thresh ? R3D_TR_SET : R3D_TR_FAIL;
    t->nset = sv >= (double)t->cfg.thresh ? 1u : 0u;
    t->gen++;
    dead = t->nset == 0;
    pthread_mutex_unlock(&t->mu);
  }

  for (uint32_t R = 1; R <= t->cfg.max_ring && !dead && !t->quit; R++) {
    uint32_t grew = 0;
    /* ring cells at Chebyshev radius R, then one relaxation pass */
    for (int pass = 0; pass < 2 && !t->quit; pass++) {
      for (int dj = -(int)R; dj <= (int)R; dj++)
        for (int di = -(int)R; di <= (int)R; di++) {
          if (abs(di) != (int)R && abs(dj) != (int)R) continue;
          int i = (int)c + di, j = (int)c + dj;
          if (i < 0 || j < 0 || i >= (int)t->W || j >= (int)t->H) continue;
          size_t k = (size_t)j * t->W + (size_t)i;
          if (pass == 0) { /* grow: extrapolate from inward neighbors */
            if (t->state[k] != R3D_TR_EMPTY) continue;
            int si = di > 0 ? -1 : (di < 0 ? 1 : 0);
            int sj = dj > 0 ? -1 : (dj < 0 ? 1 : 0);
            size_t k1 = (size_t)(j + sj) * t->W + (size_t)(i + si);
            if (t->state[k1] != R3D_TR_SET) continue;
            double P[3];
            size_t k2 = (size_t)(j + 2 * sj) * t->W + (size_t)(i + 2 * si);
            const double *p1 = t->pos + k1 * 3;
            if (abs(di) <= (int)R - 2 || abs(dj) <= (int)R - 2 ||
                (i + 2 * si >= 0 && j + 2 * sj >= 0 && i + 2 * si < (int)t->W &&
                 j + 2 * sj < (int)t->H && t->state[k2] == R3D_TR_SET)) {
              const double *p2 = t->pos + k2 * 3;
              bool lin = i + 2 * si >= 0 && j + 2 * sj >= 0 && i + 2 * si < (int)t->W &&
                         j + 2 * sj < (int)t->H && t->state[k2] == R3D_TR_SET;
              if (lin) {
                for (int a = 0; a < 3; a++) P[a] = 2.0 * p1[a] - p2[a];
              } else {
                double nrm[3], tn[3];
                tr_frame(t, p1, nrm, tn);
                for (int a = 0; a < 3; a++)
                  P[a] = p1[a] + (tn[a] * -si + (a == 2 ? (double)-sj : 0.0)) * t->cfg.step;
              }
            } else {
              double nrm[3], tn[3];
              tr_frame(t, p1, nrm, tn);
              for (int a = 0; a < 3; a++)
                P[a] = p1[a] + (tn[a] * -si + (a == 2 ? (double)-sj : 0.0)) * t->cfg.step;
            }
            double val = tr_snap(t, &vol, P);
            pthread_mutex_lock(&t->mu);
            if (val >= (double)t->cfg.thresh) {
              memcpy(t->pos + k * 3, P, sizeof P);
              t->state[k] = R3D_TR_SET;
              t->nset++;
              grew++;
            } else {
              t->state[k] = R3D_TR_FAIL;
            }
            pthread_mutex_unlock(&t->mu);
          } else { /* relax: average toward neighbors, re-snap */
            if (t->state[k] != R3D_TR_SET) continue;
            double avg[3] = {0, 0, 0};
            int na = 0;
            static const int off[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int o = 0; o < 4; o++) {
              int ii = i + off[o][0], jj = j + off[o][1];
              if (ii < 0 || jj < 0 || ii >= (int)t->W || jj >= (int)t->H) continue;
              size_t kk = (size_t)jj * t->W + (size_t)ii;
              if (t->state[kk] != R3D_TR_SET) continue;
              /* expected position of THIS cell from that neighbor */
              double nrm[3], tn[3];
              const double *pn = t->pos + kk * 3;
              tr_frame(t, pn, nrm, tn);
              for (int a = 0; a < 3; a++)
                avg[a] += pn[a] - (tn[a] * off[o][0] + (a == 2 ? (double)off[o][1] : 0.0)) *
                                      t->cfg.step;
              na++;
            }
            if (na < 2) continue;
            double P[3];
            const double *pc = t->pos + k * 3;
            for (int a = 0; a < 3; a++) P[a] = 0.6 * pc[a] + 0.4 * avg[a] / na;
            double val = tr_snap(t, &vol, P);
            if (val >= (double)t->cfg.thresh) {
              pthread_mutex_lock(&t->mu);
              memcpy(t->pos + k * 3, P, sizeof P);
              pthread_mutex_unlock(&t->mu);
            }
          }
        }
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
  if (t->cfg.max_ring > 400) t->cfg.max_ring = 400;
  snprintf(t->root, sizeof t->root, "%s", pred_root);
  t->W = t->H = 2 * t->cfg.max_ring + 1;
  t->pos = calloc((size_t)t->W * t->H * 3, sizeof *t->pos);
  t->state = calloc((size_t)t->W * t->H, 1);
  if (!t->pos || !t->state) {
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
  if (!np || !ns) {
    free(np);
    free(ns);
    return -1;
  }
  for (uint32_t j = 0; j < t->H; j++)
    for (uint32_t i = 0; i < t->W; i++) {
      size_t ok = (size_t)j * t->W + i;
      size_t nk = (size_t)(j + off) * NW + (i + off);
      if (t->state[ok] == R3D_TR_SET) { /* FAILED cells retry as EMPTY */
        ns[nk] = R3D_TR_SET;
        memcpy(np + nk * 3, t->pos + ok * 3, 3 * sizeof(double));
      }
    }
  free(t->pos);
  free(t->state);
  t->pos = np;
  t->state = ns;
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
  r3d_umbilicus_free(&t->umb);
  memset(t, 0, sizeof *t);
}

uint64_t r3d_tracer_snapshot(r3d_tracer *t, double *pos, uint8_t *state, uint32_t *ring,
                             uint32_t *nset, bool *done) {
  pthread_mutex_lock(&t->mu);
  if (pos) memcpy(pos, t->pos, (size_t)t->W * t->H * 3 * sizeof *pos);
  if (state) memcpy(state, t->state, (size_t)t->W * t->H);
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

int r3d_tracer_save(r3d_tracer *t, const char *dir) {
  if (!t->pos) return -1;
  uint64_t n = (uint64_t)t->W * t->H;
  float *pl = malloc(n * sizeof *pl);
  if (!pl) return -1;
  static const char *nm[3] = {"x.tif", "y.tif", "z.tif"};
  int rc = 0;
  pthread_mutex_lock(&t->mu);
  for (int a = 0; a < 3 && rc == 0; a++) {
    for (uint64_t k = 0; k < n; k++)
      pl[k] = t->state[k] == R3D_TR_SET ? (float)t->pos[k * 3 + (size_t)a] : -1.0f;
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
