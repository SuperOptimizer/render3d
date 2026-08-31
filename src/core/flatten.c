#include "core/flatten.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FL_INVALID 1e30f
#define FL_SIG_MIN 1e-4 /* singular-value floor: keeps weights finite */

static bool fl_valid(const float *p) { return p[0] >= 0.0f; }

/* closed-form 2x2 SVD: A = U diag(s) V^T with det(U) = det(V) = 1 and
 * s0 >= |s1| (s1 carries the sign of det A, so s1 > 0 iff orientation
 * is preserved) — the "signed SVD" every flip-aware parameterizer uses */
static void fl_svd2(const double A[4], double U[4], double s[2], double V[4]) {
  double E = (A[0] + A[3]) * 0.5, F = (A[0] - A[3]) * 0.5;
  double G = (A[2] + A[1]) * 0.5, Hh = (A[2] - A[1]) * 0.5;
  double Q = sqrt(E * E + Hh * Hh), R = sqrt(F * F + G * G);
  s[0] = Q + R;
  s[1] = Q - R;
  double a1 = atan2(G, F), a2 = atan2(Hh, E);
  double th = (a1 - a2) * 0.5, ph = (a2 + a1) * 0.5;
  U[0] = cos(ph);
  U[1] = -sin(ph);
  U[2] = sin(ph);
  U[3] = cos(ph);
  V[0] = cos(th);
  V[1] = -sin(th);
  V[2] = sin(th);
  V[3] = cos(th);
}

struct fl_tri {
  uint32_t v[3]; /* vertex indices */
  double B[4];   /* inverse rest matrix: J = [x1-x0 | x2-x0] * B */
  double area;   /* rest area (3D) */
  /* per-iteration state */
  double WW[4];  /* W^T W (symmetric proxy weight, from the current SVD) */
  double WWR[4]; /* W^T W * R (target term of the proxy) */
};

struct fl_mesh {
  uint32_t nv, nt;
  uint32_t *vid;      /* w*h -> vertex index or UINT32_MAX */
  uint32_t *vgrid;    /* vertex -> grid k */
  struct fl_tri *tri;
  double *x;          /* 2*nv current UV */
  double area_sum;
};

/* rest frame from the triangle's 3D edge lengths (isometric target):
 * p0 = (0,0), p1 = (l01, 0), p2 from the law of cosines */
static bool fl_rest(const float *q0, const float *q1, const float *q2, double B[4],
                    double *area) {
  double e1[3], e2[3];
  for (int a = 0; a < 3; a++) {
    e1[a] = (double)q1[a] - (double)q0[a];
    e2[a] = (double)q2[a] - (double)q0[a];
  }
  double l1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  double d12 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
  double l2s = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
  if (l1 < 1e-9) return false;
  double px = d12 / l1;
  double py2 = l2s - px * px;
  if (py2 < 1e-12) return false; /* degenerate (collinear) */
  double py = sqrt(py2);
  /* rest matrix M = [ (l1, 0) | (px, py) ]; B = M^-1 */
  double det = l1 * py;
  B[0] = py / det;
  B[1] = -px / det;
  B[2] = 0.0;
  B[3] = l1 / det;
  *area = 0.5 * det;
  return true;
}

static void fl_mesh_free(struct fl_mesh *m) {
  free(m->vid);
  free(m->vgrid);
  free(m->tri);
  free(m->x);
  memset(m, 0, sizeof *m);
}

static int fl_mesh_build(struct fl_mesh *m, const float *xyz, uint32_t w, uint32_t h,
                         double step) {
  memset(m, 0, sizeof *m);
  size_t n = (size_t)w * h;
  m->vid = malloc(n * sizeof *m->vid);
  if (!m->vid) return -1;
  for (size_t k = 0; k < n; k++) m->vid[k] = UINT32_MAX;
  /* vertices = valid corners of at least one fully-valid quad */
  for (uint32_t j = 0; j + 1 < h; j++)
    for (uint32_t i = 0; i + 1 < w; i++) {
      size_t k = (size_t)j * w + i;
      if (fl_valid(xyz + k * 3) && fl_valid(xyz + (k + 1) * 3) &&
          fl_valid(xyz + (k + w) * 3) && fl_valid(xyz + (k + w + 1) * 3)) {
        m->vid[k] = 0;
        m->vid[k + 1] = 0;
        m->vid[k + w] = 0;
        m->vid[k + w + 1] = 0;
      }
    }
  for (size_t k = 0; k < n; k++)
    if (m->vid[k] == 0) m->vid[k] = m->nv++;
    else m->vid[k] = UINT32_MAX;
  if (m->nv < 4) return -1;
  m->vgrid = malloc((size_t)m->nv * sizeof *m->vgrid);
  m->tri = malloc((size_t)(w - 1) * (h - 1) * 2 * sizeof *m->tri);
  m->x = malloc((size_t)m->nv * 2 * sizeof *m->x);
  if (!m->vgrid || !m->tri || !m->x) return -1;
  /* Start from cumulative physical edge lengths.  A uniform i*step/j*step
   * embedding bakes tracing-grid drift into the texture before SLIM even
   * starts.  Column/row mean physical lengths are monotone and preserve the
   * regular-grid orientation while using every supported edge. */
  double *cu = calloc(w, sizeof *cu), *cv = calloc(h, sizeof *cv);
  uint32_t *nu = calloc(w, sizeof *nu), *nv = calloc(h, sizeof *nv);
  if (!cu || !cv || !nu || !nv) {
    free(cu); free(cv); free(nu); free(nv);
    return -1;
  }
  for (uint32_t j = 0; j < h; j++)
    for (uint32_t i = 0; i + 1 < w; i++) {
      size_t a = (size_t)j * w + i, b = a + 1;
      if (m->vid[a] == UINT32_MAX || m->vid[b] == UINT32_MAX) continue;
      double dx = (double)xyz[b * 3] - (double)xyz[a * 3];
      double dy = (double)xyz[b * 3 + 1] - (double)xyz[a * 3 + 1];
      double dz = (double)xyz[b * 3 + 2] - (double)xyz[a * 3 + 2];
      cu[i + 1] += sqrt(dx * dx + dy * dy + dz * dz);
      nu[i + 1]++;
    }
  for (uint32_t j = 0; j + 1 < h; j++)
    for (uint32_t i = 0; i < w; i++) {
      size_t a = (size_t)j * w + i, b = a + w;
      if (m->vid[a] == UINT32_MAX || m->vid[b] == UINT32_MAX) continue;
      double dx = (double)xyz[b * 3] - (double)xyz[a * 3];
      double dy = (double)xyz[b * 3 + 1] - (double)xyz[a * 3 + 1];
      double dz = (double)xyz[b * 3 + 2] - (double)xyz[a * 3 + 2];
      cv[j + 1] += sqrt(dx * dx + dy * dy + dz * dz);
      nv[j + 1]++;
    }
  for (uint32_t i = 1; i < w; i++) cu[i] = cu[i - 1] + (nu[i] ? cu[i] / nu[i] : step);
  for (uint32_t j = 1; j < h; j++) cv[j] = cv[j - 1] + (nv[j] ? cv[j] / nv[j] : step);
  for (size_t k = 0; k < n; k++)
    if (m->vid[k] != UINT32_MAX) {
      m->vgrid[m->vid[k]] = (uint32_t)k;
      m->x[(size_t)m->vid[k] * 2 + 0] = cu[k % w];
      m->x[(size_t)m->vid[k] * 2 + 1] = cv[k / w];
    }
  free(cu); free(cv); free(nu); free(nv);
  for (uint32_t j = 0; j + 1 < h; j++)
    for (uint32_t i = 0; i + 1 < w; i++) {
      size_t k = (size_t)j * w + i;
      uint32_t a = m->vid[k], b = m->vid[k + 1], c = m->vid[k + w], d = m->vid[k + w + 1];
      if (a == UINT32_MAX || b == UINT32_MAX || c == UINT32_MAX || d == UINT32_MAX)
        continue;
      if (!fl_valid(xyz + k * 3) || !fl_valid(xyz + (k + 1) * 3) ||
          !fl_valid(xyz + (k + w) * 3) || !fl_valid(xyz + (k + w + 1) * 3))
        continue;
      struct fl_tri *t1 = &m->tri[m->nt];
      t1->v[0] = a;
      t1->v[1] = b;
      t1->v[2] = c;
      if (fl_rest(xyz + k * 3, xyz + (k + 1) * 3, xyz + (k + w) * 3, t1->B, &t1->area))
        m->nt++;
      struct fl_tri *t2 = &m->tri[m->nt];
      t2->v[0] = b;
      t2->v[1] = d;
      t2->v[2] = c;
      if (fl_rest(xyz + (k + 1) * 3, xyz + (k + w + 1) * 3, xyz + (k + w) * 3, t2->B,
                  &t2->area))
        m->nt++;
    }
  if (!m->nt) return -1;
  for (uint32_t t = 0; t < m->nt; t++) m->area_sum += m->tri[t].area;
  return 0;
}

/* Jacobian of triangle t under vertex positions x */
static void fl_jac(const struct fl_tri *t, const double *x, double J[4]) {
  const double *p0 = x + (size_t)t->v[0] * 2, *p1 = x + (size_t)t->v[1] * 2,
               *p2 = x + (size_t)t->v[2] * 2;
  double X[4] = {p1[0] - p0[0], p2[0] - p0[0], p1[1] - p0[1], p2[1] - p0[1]};
  J[0] = X[0] * t->B[0] + X[1] * t->B[2];
  J[1] = X[0] * t->B[1] + X[1] * t->B[3];
  J[2] = X[2] * t->B[0] + X[3] * t->B[2];
  J[3] = X[2] * t->B[1] + X[3] * t->B[3];
}

/* symmetric Dirichlet energy (area-weighted mean per triangle); flipped or
 * near-degenerate triangles count as +inf via a huge sentinel */
static double fl_energy(const struct fl_mesh *m, const double *x, double *stretch) {
  double e = 0.0, st = 0.0;
  for (uint32_t ti = 0; ti < m->nt; ti++) {
    const struct fl_tri *t = &m->tri[ti];
    double J[4], U[4], s[2], V[4];
    fl_jac(t, x, J);
    fl_svd2(J, U, s, V);
    if (s[1] < FL_SIG_MIN) return 1e30; /* flipped/collapsed */
    double s0 = s[0], s1 = s[1];
    e += t->area * (s0 * s0 + s1 * s1 + 1.0 / (s0 * s0) + 1.0 / (s1 * s1));
    if (stretch) st += t->area * (fabs(s0 - 1.0) + fabs(s1 - 1.0)) * 0.5;
  }
  if (stretch) *stretch = st / m->area_sum;
  return e / m->area_sum;
}

/* per-iteration proxy: W^T W and W^T W R from the current Jacobian's SVD
 * (SLIM reweighting for symmetric Dirichlet: WW = U diag((s - s^-3)/(s-1)) U^T,
 * limit 4 at s = 1; R = closest rotation U V^T) */
static void fl_reweight(struct fl_mesh *m) {
  for (uint32_t ti = 0; ti < m->nt; ti++) {
    struct fl_tri *t = &m->tri[ti];
    double J[4], U[4], s[2], V[4];
    fl_jac(t, m->x, J);
    fl_svd2(J, U, s, V);
    double ww[2];
    for (int i = 0; i < 2; i++) {
      double sg = s[i] < FL_SIG_MIN ? FL_SIG_MIN : s[i];
      double num = sg - 1.0 / (sg * sg * sg), den = sg - 1.0;
      ww[i] = fabs(den) < 1e-8 ? 4.0 : num / den;
      if (ww[i] < 1e-8) ww[i] = 1e-8;
    }
    /* WW = U diag(ww) U^T */
    t->WW[0] = U[0] * ww[0] * U[0] + U[1] * ww[1] * U[1];
    t->WW[1] = U[0] * ww[0] * U[2] + U[1] * ww[1] * U[3];
    t->WW[2] = t->WW[1];
    t->WW[3] = U[2] * ww[0] * U[2] + U[3] * ww[1] * U[3];
    double R[4] = {U[0] * V[0] + U[1] * V[1], U[0] * V[2] + U[1] * V[3],
                   U[2] * V[0] + U[3] * V[1], U[2] * V[2] + U[3] * V[3]};
    t->WWR[0] = t->WW[0] * R[0] + t->WW[1] * R[2];
    t->WWR[1] = t->WW[0] * R[1] + t->WW[1] * R[3];
    t->WWR[2] = t->WW[2] * R[0] + t->WW[3] * R[2];
    t->WWR[3] = t->WW[2] * R[1] + t->WW[3] * R[3];
  }
}

/* accumulate the adjoint of M (2x2, in Jacobian space) into the vertex
 * gradient g: X_adj = M B^T, then p1 += col0, p2 += col1, p0 -= col0+col1 */
static void fl_scatter(const struct fl_tri *t, const double M[4], double *g) {
  double XA[4] = {M[0] * t->B[0] + M[1] * t->B[1], M[0] * t->B[2] + M[1] * t->B[3],
                  M[2] * t->B[0] + M[3] * t->B[1], M[2] * t->B[2] + M[3] * t->B[3]};
  double *g0 = g + (size_t)t->v[0] * 2, *g1 = g + (size_t)t->v[1] * 2,
         *g2 = g + (size_t)t->v[2] * 2;
  g1[0] += XA[0];
  g1[1] += XA[2];
  g2[0] += XA[1];
  g2[1] += XA[3];
  g0[0] -= XA[0] + XA[1];
  g0[1] -= XA[2] + XA[3];
}

/* out = A x with A = sum_t area * K_t^T WW_t K_t (K_t: vertex uvs -> J) */
static void fl_apply(const struct fl_mesh *m, const double *x, double *out) {
  memset(out, 0, (size_t)m->nv * 2 * sizeof *out);
  for (uint32_t ti = 0; ti < m->nt; ti++) {
    const struct fl_tri *t = &m->tri[ti];
    double J[4];
    fl_jac(t, x, J);
    double M[4] = {t->WW[0] * J[0] + t->WW[1] * J[2], t->WW[0] * J[1] + t->WW[1] * J[3],
                   t->WW[2] * J[0] + t->WW[3] * J[2], t->WW[2] * J[1] + t->WW[3] * J[3]};
    for (int i = 0; i < 4; i++) M[i] *= t->area;
    fl_scatter(t, M, out);
  }
}

/* Jacobi preconditioner: diagonal of A */
static void fl_diag(const struct fl_mesh *m, double *d) {
  for (size_t i = 0; i < (size_t)m->nv * 2; i++) d[i] = 1e-12;
  for (uint32_t ti = 0; ti < m->nt; ti++) {
    const struct fl_tri *t = &m->tri[ti];
    /* exact diagonal of A: the u_i^2 coefficient of x^T A x comes only
     * from the WW[0]|J row0|^2 term (v_i^2 from WW[3]), scaled by the
     * squared row weight |k_i|^2 each vertex carries through B */
    double c1 = t->B[0] * t->B[0] + t->B[1] * t->B[1]; /* |row for p1|^2 */
    double c2 = t->B[2] * t->B[2] + t->B[3] * t->B[3];
    double c0 = (t->B[0] + t->B[2]) * (t->B[0] + t->B[2]) +
                (t->B[1] + t->B[3]) * (t->B[1] + t->B[3]);
    double wu = t->area * t->WW[0], wv = t->area * t->WW[3];
    d[(size_t)t->v[1] * 2 + 0] += wu * c1;
    d[(size_t)t->v[1] * 2 + 1] += wv * c1;
    d[(size_t)t->v[2] * 2 + 0] += wu * c2;
    d[(size_t)t->v[2] * 2 + 1] += wv * c2;
    d[(size_t)t->v[0] * 2 + 0] += wu * c0;
    d[(size_t)t->v[0] * 2 + 1] += wv * c0;
  }
}

/* largest step in direction dxy keeping every triangle unflipped: per
 * triangle det(J + s dJ) = 0 is quadratic in s; take 0.8x the smallest
 * positive root (standard SLIM/Smith-Schaefer bound) */
static double fl_max_step(const struct fl_mesh *m, const double *x, const double *dx) {
  double smax = 1e30;
  for (uint32_t ti = 0; ti < m->nt; ti++) {
    const struct fl_tri *t = &m->tri[ti];
    double J[4], D[4];
    fl_jac(t, x, J);
    fl_jac(t, dx, D); /* dx as absolute positions: subtract? no — see call */
    /* here dx holds the DIRECTION (deltas), and fl_jac is linear, so D is
     * the Jacobian of the direction field directly */
    double a = D[0] * D[3] - D[1] * D[2];
    double b = J[0] * D[3] + D[0] * J[3] - J[1] * D[2] - D[1] * J[2];
    double c = J[0] * J[3] - J[1] * J[2];
    double roots[2];
    int nr = 0;
    if (fabs(a) < 1e-14) {
      if (fabs(b) > 1e-14) roots[nr++] = -c / b;
    } else {
      double disc = b * b - 4.0 * a * c;
      if (disc >= 0.0) {
        double sq = sqrt(disc);
        roots[nr++] = (-b - sq) / (2.0 * a);
        roots[nr++] = (-b + sq) / (2.0 * a);
      }
    }
    for (int r = 0; r < nr; r++)
      if (roots[r] > 1e-12 && roots[r] < smax) smax = roots[r];
  }
  return smax >= 1e30 ? 1.0 : 0.8 * smax;
}

int r3d_flatten_slim(const float *xyz, uint32_t w, uint32_t h, double step,
                     uint32_t max_iters, float *uv, r3d_flatten_stats *st) {
  if (!xyz || !uv || w < 2 || h < 2 || !(step > 0.0)) return -1;
  struct fl_mesh m;
  if (fl_mesh_build(&m, xyz, w, h, step) != 0) {
    fl_mesh_free(&m);
    return -1;
  }
  size_t n2 = (size_t)m.nv * 2;
  double *rhs = malloc(n2 * sizeof *rhs), *sol = malloc(n2 * sizeof *sol);
  double *r = malloc(n2 * sizeof *r), *z = malloc(n2 * sizeof *z);
  double *p = malloc(n2 * sizeof *p), *ap = malloc(n2 * sizeof *ap);
  double *dg = malloc(n2 * sizeof *dg), *xnew = malloc(n2 * sizeof *xnew);
  int rc = -1;
  if (!rhs || !sol || !r || !z || !p || !ap || !dg || !xnew) goto out;
  double stretch0 = 0.0, e = fl_energy(&m, m.x, &stretch0);
  double e0 = e;
  uint32_t it = 0;
  for (; it < max_iters; it++) {
    fl_reweight(&m);
    /* rhs = sum_t area * K^T (WW R) */
    memset(rhs, 0, n2 * sizeof *rhs);
    for (uint32_t ti = 0; ti < m.nt; ti++) {
      double M[4];
      for (int i2 = 0; i2 < 4; i2++) M[i2] = m.tri[ti].WWR[i2] * m.tri[ti].area;
      fl_scatter(&m.tri[ti], M, rhs);
    }
    /* PCG solve A sol = rhs, warm-started at the current embedding */
    memcpy(sol, m.x, n2 * sizeof *sol);
    fl_apply(&m, sol, ap);
    for (size_t i2 = 0; i2 < n2; i2++) r[i2] = rhs[i2] - ap[i2];
    fl_diag(&m, dg);
    double rz = 0.0;
    for (size_t i2 = 0; i2 < n2; i2++) {
      z[i2] = r[i2] / dg[i2];
      p[i2] = z[i2];
      rz += r[i2] * z[i2];
    }
    double rhs_n = 0.0;
    for (size_t i2 = 0; i2 < n2; i2++) rhs_n += rhs[i2] * rhs[i2];
    double tol2 = rhs_n * 1e-10 + 1e-30;
    for (uint32_t cg = 0; cg < 400; cg++) {
      double rr = 0.0;
      for (size_t i2 = 0; i2 < n2; i2++) rr += r[i2] * r[i2];
      if (rr < tol2) break;
      fl_apply(&m, p, ap);
      double pap = 0.0;
      for (size_t i2 = 0; i2 < n2; i2++) pap += p[i2] * ap[i2];
      if (pap <= 0.0) break;
      double alpha = rz / pap;
      for (size_t i2 = 0; i2 < n2; i2++) {
        sol[i2] += alpha * p[i2];
        r[i2] -= alpha * ap[i2];
      }
      double rz2 = 0.0;
      for (size_t i2 = 0; i2 < n2; i2++) {
        z[i2] = r[i2] / dg[i2];
        rz2 += r[i2] * z[i2];
      }
      double beta = rz2 / rz;
      rz = rz2;
      for (size_t i2 = 0; i2 < n2; i2++) p[i2] = z[i2] + beta * p[i2];
    }
    /* line search along d = sol - x: flip bound, then backtracking */
    for (size_t i2 = 0; i2 < n2; i2++) ap[i2] = sol[i2] - m.x[i2]; /* direction */
    double tmax = fl_max_step(&m, m.x, ap);
    double t = tmax < 1.0 ? tmax : 1.0;
    double enew = 1e30;
    int bt = 0;
    for (; bt < 24; bt++) {
      for (size_t i2 = 0; i2 < n2; i2++) xnew[i2] = m.x[i2] + t * ap[i2];
      enew = fl_energy(&m, xnew, NULL);
      if (enew < e) break;
      t *= 0.5;
    }
    if (bt == 24 || enew >= e) break; /* converged / no descent */
    memcpy(m.x, xnew, n2 * sizeof *m.x);
    double rel = (e - enew) / (e - 4.0 + 1e-12); /* 4.0 = isometry floor */
    e = enew;
    if (rel < 1e-4) {
      it++;
      break;
    }
  }
  double stretch1 = 0.0;
  double e1 = fl_energy(&m, m.x, &stretch1);
  /* write UVs (recentered so min corner is ~0) */
  double mnu = 1e30, mnv = 1e30;
  for (uint32_t vtx = 0; vtx < m.nv; vtx++) {
    if (m.x[(size_t)vtx * 2] < mnu) mnu = m.x[(size_t)vtx * 2];
    if (m.x[(size_t)vtx * 2 + 1] < mnv) mnv = m.x[(size_t)vtx * 2 + 1];
  }
  for (size_t k = 0; k < (size_t)w * h; k++) {
    uv[k * 2] = FL_INVALID;
    uv[k * 2 + 1] = FL_INVALID;
  }
  for (uint32_t vtx = 0; vtx < m.nv; vtx++) {
    size_t k = m.vgrid[vtx];
    uv[k * 2] = (float)(m.x[(size_t)vtx * 2] - mnu);
    uv[k * 2 + 1] = (float)(m.x[(size_t)vtx * 2 + 1] - mnv);
  }
  if (st) {
    st->iters = it;
    st->nvert = m.nv;
    st->ntri = m.nt;
    st->e0 = e0;
    st->e1 = e1;
    st->stretch0 = stretch0;
    st->stretch1 = stretch1;
  }
  rc = 0;
out:
  free(rhs);
  free(sol);
  free(r);
  free(z);
  free(p);
  free(ap);
  free(dg);
  free(xnew);
  fl_mesh_free(&m);
  return rc;
}

int r3d_flatten_resample(const float *xyz, const float *uv, uint32_t w, uint32_t h,
                         double step, float **out_xyz, uint32_t *ow, uint32_t *oh) {
  if (!xyz || !uv || !out_xyz || !(step > 0.0)) return -1;
  double mxu = 0.0, mxv = 0.0;
  bool any = false;
  for (size_t k = 0; k < (size_t)w * h; k++) {
    if (uv[k * 2] >= FL_INVALID) continue;
    if ((double)uv[k * 2] > mxu) mxu = (double)uv[k * 2];
    if ((double)uv[k * 2 + 1] > mxv) mxv = (double)uv[k * 2 + 1];
    any = true;
  }
  if (!any) return -1;
  uint32_t W = (uint32_t)(mxu / step) + 2, H = (uint32_t)(mxv / step) + 2;
  if ((uint64_t)W * H > 64ull * 1024 * 1024) return -1; /* runaway embedding */
  float *o = malloc((size_t)W * H * 3 * sizeof *o);
  if (!o) return -1;
  for (size_t k = 0; k < (size_t)W * H; k++) {
    o[k * 3] = -1.0f;
    o[k * 3 + 1] = -1.0f;
    o[k * 3 + 2] = -1.0f;
  }
  /* rasterize every valid quad's two UV triangles over the lattice */
  for (uint32_t j = 0; j + 1 < h; j++)
    for (uint32_t i = 0; i + 1 < w; i++) {
      size_t k = (size_t)j * w + i;
      size_t q[4] = {k, k + 1, k + w, k + w + 1};
      bool ok = true;
      for (int c = 0; c < 4; c++)
        if (!fl_valid(xyz + q[c] * 3) || uv[q[c] * 2] >= FL_INVALID) ok = false;
      if (!ok) continue;
      static const int tris[2][3] = {{0, 1, 2}, {1, 3, 2}};
      for (int tt = 0; tt < 2; tt++) {
        const float *u0 = uv + q[tris[tt][0]] * 2, *u1 = uv + q[tris[tt][1]] * 2,
                    *u2 = uv + q[tris[tt][2]] * 2;
        const float *p0 = xyz + q[tris[tt][0]] * 3, *p1 = xyz + q[tris[tt][1]] * 3,
                    *p2 = xyz + q[tris[tt][2]] * 3;
        double au = (double)u0[0] / step, av = (double)u0[1] / step;
        double bu = (double)u1[0] / step, bv = (double)u1[1] / step;
        double cu = (double)u2[0] / step, cv = (double)u2[1] / step;
        double lox = fmin(au, fmin(bu, cu)), hix = fmax(au, fmax(bu, cu));
        double loy = fmin(av, fmin(bv, cv)), hiy = fmax(av, fmax(bv, cv));
        int x0 = (int)ceil(lox - 1e-9), x1 = (int)floor(hix + 1e-9);
        int y0 = (int)ceil(loy - 1e-9), y1 = (int)floor(hiy + 1e-9);
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 >= (int)W) x1 = (int)W - 1;
        if (y1 >= (int)H) y1 = (int)H - 1;
        double det = (bu - au) * (cv - av) - (cu - au) * (bv - av);
        if (fabs(det) < 1e-12) continue;
        for (int y = y0; y <= y1; y++)
          for (int x = x0; x <= x1; x++) {
            double l1 = ((double)x - au) * (cv - av) - (cu - au) * ((double)y - av);
            double l2 = ((bu - au) * ((double)y - av) - ((double)x - au) * (bv - av));
            l1 /= det;
            l2 /= det;
            double l0 = 1.0 - l1 - l2;
            if (l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6) continue;
            size_t ko = (size_t)y * W + (size_t)x;
            for (int a3 = 0; a3 < 3; a3++)
              o[ko * 3 + (size_t)a3] =
                  (float)(l0 * (double)p0[a3] + l1 * (double)p1[a3] +
                          l2 * (double)p2[a3]);
          }
      }
    }
  *out_xyz = o;
  *ow = W;
  *oh = H;
  return 0;
}
