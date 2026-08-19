#include "core/regvol.h"

#include <fysics.h> /* fy_register_affine / fy_ncc_warped (angle: fysics dir) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RB 128u
#define RB3 ((size_t)RB * RB * RB)
#define REG_SCRATCH_CAP ((size_t)64 * 1024 * 1024)

/* ---- 3x4 affine helpers, row-major (x, y, z) order ---------------------- */

static void m34_identity(double *o) {
  memset(o, 0, 12 * sizeof(double));
  o[0] = o[5] = o[10] = 1.0;
}

/* o = a ∘ b  (apply b, then a) */
static void m34_mul(double *o, const double *a, const double *b) {
  double t[12];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++)
      t[r * 4 + c] =
          a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c];
    t[r * 4 + 3] = a[r * 4 + 0] * b[3] + a[r * 4 + 1] * b[7] + a[r * 4 + 2] * b[11] + a[r * 4 + 3];
  }
  memcpy(o, t, sizeof t);
}

static void m34_apply(const double *a, const double *p, double *q) {
  double t[3];
  for (int r = 0; r < 3; r++)
    t[r] = a[r * 4 + 0] * p[0] + a[r * 4 + 1] * p[1] + a[r * 4 + 2] * p[2] + a[r * 4 + 3];
  memcpy(q, t, sizeof t);
}

static double m34_det(const double *a) {
  return a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8]) +
         a[2] * (a[4] * a[9] - a[5] * a[8]);
}

static int m34_invert(const double *a, double *o) {
  double d = m34_det(a);
  if (fabs(d) < 1e-30) return -1;
  double inv = 1.0 / d;
  double l[9] = {
      (a[5] * a[10] - a[6] * a[9]) * inv,  (a[2] * a[9] - a[1] * a[10]) * inv,
      (a[1] * a[6] - a[2] * a[5]) * inv,   (a[6] * a[8] - a[4] * a[10]) * inv,
      (a[0] * a[10] - a[2] * a[8]) * inv,  (a[2] * a[4] - a[0] * a[6]) * inv,
      (a[4] * a[9] - a[5] * a[8]) * inv,   (a[1] * a[8] - a[0] * a[9]) * inv,
      (a[0] * a[5] - a[1] * a[4]) * inv};
  double t[12] = {l[0], l[1], l[2], 0, l[3], l[4], l[5], 0, l[6], l[7], l[8], 0};
  t[3] = -(l[0] * a[3] + l[1] * a[7] + l[2] * a[11]);
  t[7] = -(l[3] * a[3] + l[4] * a[7] + l[5] * a[11]);
  t[11] = -(l[6] * a[3] + l[7] * a[7] + l[8] * a[11]);
  memcpy(o, t, sizeof t);
  return 0;
}

static void m34_translate(double *o, double x, double y, double z) {
  m34_identity(o);
  o[3] = x;
  o[7] = y;
  o[11] = z;
}

/* fysics matrices are (z, y, x)-ordered; ours (x, y, z). Same op both ways. */
static void m34_swap_order(const double *p, double *q) {
  double t[12];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) t[i * 4 + j] = p[(2 - i) * 4 + (2 - j)];
    t[i * 4 + 3] = p[(2 - i) * 4 + 3];
  }
  memcpy(q, t, sizeof t);
}

/* ---- transform state ---------------------------------------------------- */

/* inverse of the interactive delta: the deltas move the MOVING volume within
 * the fixed frame (about the fixed volume center), so sampling composes the
 * inverse motion onto the pull map. D = T(t) C Rz Ry Rx S C^-1;
 * D^-1 = C S^-1 Rx^T Ry^T Rz^T C^-1 T(-t). */
static void reg_delta_inv(const r3d_regvol *rv, double *o) {
  double c[3] = {(double)rv->fdim[0] * 0.5, (double)rv->fdim[1] * 0.5,
                 (double)rv->fdim[2] * 0.5};
  double s = exp(-rv->d_lscale);
  double cx = cos(rv->d_rot[0]), sx = sin(rv->d_rot[0]);
  double cy = cos(rv->d_rot[1]), sy = sin(rv->d_rot[1]);
  double cz = cos(rv->d_rot[2]), sz = sin(rv->d_rot[2]);
  double rxT[12] = {1, 0, 0, 0, 0, cx, sx, 0, 0, -sx, cx, 0};
  double ryT[12] = {cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0};
  double rzT[12] = {cz, sz, 0, 0, -sz, cz, 0, 0, 0, 0, 1, 0};
  double t[12], u[12];
  m34_translate(t, -rv->d_tr[0], -rv->d_tr[1], -rv->d_tr[2]); /* T(-t) */
  m34_translate(u, -c[0], -c[1], -c[2]);                      /* C^-1 */
  m34_mul(t, u, t);
  m34_mul(t, rzT, t);
  m34_mul(t, ryT, t);
  m34_mul(t, rxT, t);
  double sc[12] = {s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0};
  m34_mul(t, sc, t);
  m34_translate(u, c[0], c[1], c[2]); /* C */
  m34_mul(o, u, t);
}

void r3d_regvol_pull(r3d_regvol *rv, double P[12]) {
  pthread_mutex_lock(&rv->mu);
  double d[12];
  reg_delta_inv(rv, d);
  m34_mul(P, rv->M, d);
  pthread_mutex_unlock(&rv->mu);
}

void r3d_regvol_bump(r3d_regvol *rv) { rv->gen++; }

void r3d_regvol_bake(r3d_regvol *rv) {
  pthread_mutex_lock(&rv->mu);
  double d[12];
  reg_delta_inv(rv, d);
  m34_mul(rv->M, rv->M, d);
  memset(rv->d_tr, 0, sizeof rv->d_tr);
  memset(rv->d_rot, 0, sizeof rv->d_rot);
  rv->d_lscale = 0.0;
  pthread_mutex_unlock(&rv->mu);
}

void r3d_regvol_reset_deltas(r3d_regvol *rv) {
  pthread_mutex_lock(&rv->mu);
  memset(rv->d_tr, 0, sizeof rv->d_tr);
  memset(rv->d_rot, 0, sizeof rv->d_rot);
  rv->d_lscale = 0.0;
  pthread_mutex_unlock(&rv->mu);
  rv->gen++;
}

void r3d_regvol_set_scale(r3d_regvol *rv, double s) {
  pthread_mutex_lock(&rv->mu);
  m34_identity(rv->M);
  rv->M[0] = rv->M[5] = rv->M[10] = s;
  pthread_mutex_unlock(&rv->mu);
  rv->gen++;
}

double r3d_regvol_scale(r3d_regvol *rv) {
  double P[12];
  r3d_regvol_pull(rv, P);
  return cbrt(fabs(m34_det(P)));
}

double r3d_regvol_parse_um(const char *path) {
  const char *p = path;
  while ((p = strstr(p, "um")) != NULL) {
    const char *q = p; /* walk back over the number ending at "um" */
    while (q > path && ((q[-1] >= '0' && q[-1] <= '9') || q[-1] == '.')) q--;
    if (q != p) { /* "volume" has no digits before its "um" and skips here */
      double v = strtod(q, NULL);
      if (v > 0.05 && v < 1000.0) return v;
    }
    p += 2;
  }
  return 0.0;
}

/* ---- open / close -------------------------------------------------------- */

int r3d_regvol_open(r3d_regvol *rv, const char *moving_root, const uint32_t fixed_dim[3]) {
  memset(rv, 0, sizeof *rv);
  if (r3d_cpuvol_open(&rv->mv, moving_root, 1024) != 0) return -1;
  snprintf(rv->root, sizeof rv->root, "%s", moving_root);
  memcpy(rv->fdim, fixed_dim, sizeof rv->fdim);
  m34_identity(rv->M);
  pthread_mutex_init(&rv->mu, NULL);
  rv->gen = 1;
  rv->open = true;
  printf("regvol: moving scan %s (%llu x %llu x %llu)\n", moving_root,
         (unsigned long long)rv->mv.nx, (unsigned long long)rv->mv.ny,
         (unsigned long long)rv->mv.nz);
  return 0;
}

void r3d_regvol_close(r3d_regvol *rv) {
  if (!rv->open) return;
  if (rv->th_up) { /* let a running job finish; its result is dropped */
    pthread_join(rv->th, NULL);
    rv->th_up = false;
  }
  r3d_cpuvol_close(&rv->mv);
  pthread_mutex_destroy(&rv->mu);
  free(rv->scratch);
  memset(rv, 0, sizeof *rv);
}

/* ---- renderer source: per-brick resample --------------------------------- */

/* moving mip level whose voxel pitch best matches one output texel */
static uint32_t reg_pick_level(const r3d_regvol *rv, const double *P, uint32_t level) {
  double s = cbrt(fabs(m34_det(P)));
  if (!(s > 1e-12)) s = 1.0;
  double step = s * (double)(1u << level);
  int ml = (int)floor(log2(step) + 0.5);
  if (ml < 0) ml = 0;
  if ((uint32_t)ml >= rv->mv.nlev) ml = (int)rv->mv.nlev - 1;
  return (uint32_t)ml;
}

/* moving-space AABB (base voxels) of one fixed brick */
static void reg_brick_aabb(const double *P, uint32_t level, uint32_t bx, uint32_t by,
                           uint32_t bz, double mn[3], double mx[3]) {
  for (int a = 0; a < 3; a++) {
    mn[a] = 1e300;
    mx[a] = -1e300;
  }
  for (int k = 0; k < 8; k++) {
    double w[3] = {(double)(((uint64_t)bx * RB + ((k & 1) ? RB : 0)) << level),
                   (double)(((uint64_t)by * RB + ((k & 2) ? RB : 0)) << level),
                   (double)(((uint64_t)bz * RB + ((k & 4) ? RB : 0)) << level)};
    double m[3];
    m34_apply(P, w, m);
    for (int a = 0; a < 3; a++) {
      if (m[a] < mn[a]) mn[a] = m[a];
      if (m[a] > mx[a]) mx[a] = m[a];
    }
  }
}

uint32_t r3d_regvol_srcgen(void *ctx, uint32_t level, uint32_t bx, uint32_t by, uint32_t bz) {
  r3d_regvol *rv = ctx;
  if (!rv->open) return 0;
  double P[12];
  r3d_regvol_pull(rv, P);
  double mn[3], mx[3];
  reg_brick_aabb(P, level, bx, by, bz, mn, mx);
  double dim[3] = {(double)rv->mv.nx, (double)rv->mv.ny, (double)rv->mv.nz};
  for (int a = 0; a < 3; a++)
    if (mx[a] < 0.0 || mn[a] >= dim[a]) return 0; /* outside the moving scan */
  return rv->gen;
}

static double tri_scratch(const uint8_t *s, int64_t nx, int64_t ny, int64_t nz, double x,
                          double y, double z) {
  if (x <= -0.5 || y <= -0.5 || z <= -0.5 || x >= (double)nx - 0.5 ||
      y >= (double)ny - 0.5 || z >= (double)nz - 0.5)
    return 0.0;
  int64_t x0 = (int64_t)floor(x), y0 = (int64_t)floor(y), z0 = (int64_t)floor(z);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (z0 < 0) z0 = 0;
  if (x0 > nx - 2) x0 = nx - 2;
  if (y0 > ny - 2) y0 = ny - 2;
  if (z0 > nz - 2) z0 = nz - 2;
  double fx = x - (double)x0, fy = y - (double)y0, fz = z - (double)z0;
  fx = fx < 0 ? 0 : fx > 1 ? 1 : fx;
  fy = fy < 0 ? 0 : fy > 1 ? 1 : fy;
  fz = fz < 0 ? 0 : fz > 1 ? 1 : fz;
  const uint8_t *p = s + (z0 * ny + y0) * nx + x0;
  double c00 = (double)p[0] + fx * ((double)p[1] - (double)p[0]);
  double c10 = (double)p[nx] + fx * ((double)p[nx + 1] - (double)p[nx]);
  const uint8_t *q = p + nx * ny;
  double c01 = (double)q[0] + fx * ((double)q[1] - (double)q[0]);
  double c11 = (double)q[nx] + fx * ((double)q[nx + 1] - (double)q[nx]);
  double c0 = c00 + fy * (c10 - c00), c1 = c01 + fy * (c11 - c01);
  return c0 + fz * (c1 - c0);
}

void r3d_regvol_srcfetch(void *ctx, uint32_t level, uint32_t bx, uint32_t by, uint32_t bz,
                         uint8_t *out) {
  r3d_regvol *rv = ctx;
  if (!rv->open) {
    memset(out, 0, RB3);
    return;
  }
  double P[12];
  r3d_regvol_pull(rv, P);
  uint32_t ml = reg_pick_level(rv, P, level);
  double linv = 1.0 / (double)(1u << ml);
  double mn[3], mx[3];
  reg_brick_aabb(P, level, bx, by, bz, mn, mx);
  const r3d_cpuvol_level *lv = &rv->mv.lev[ml];
  double ldim[3] = {(double)lv->vx, (double)lv->vy, (double)lv->vz};
  int64_t o0[3], on[3];
  bool empty = false;
  for (int a = 0; a < 3; a++) {
    double lo = mn[a] * linv - 1.5, hi = mx[a] * linv + 1.5;
    if (lo < -1.0) lo = -1.0;
    if (hi > ldim[a] + 1.0) hi = ldim[a] + 1.0;
    o0[a] = (int64_t)floor(lo);
    on[a] = (int64_t)ceil(hi) - o0[a] + 1;
    if (on[a] < 3) empty = true;
  }
  if (empty || mx[0] < 0.0 || mx[1] < 0.0 || mx[2] < 0.0 || mn[0] >= ldim[0] * (double)(1u << ml) ||
      mn[1] >= ldim[1] * (double)(1u << ml) || mn[2] >= ldim[2] * (double)(1u << ml)) {
    memset(out, 0, RB3);
    return;
  }
  size_t need = (size_t)on[0] * (size_t)on[1] * (size_t)on[2];
  double base_step = (double)(1u << level);
  if (need <= REG_SCRATCH_CAP) {
    if (need > rv->scratch_cap) {
      free(rv->scratch);
      rv->scratch = malloc(need);
      rv->scratch_cap = rv->scratch ? need : 0;
    }
    if (rv->scratch) {
      r3d_cpuvol_read_block(&rv->mv, ml, o0[0], o0[1], o0[2], (uint32_t)on[0], (uint32_t)on[1],
                            (uint32_t)on[2], rv->scratch);
      /* per-output-voxel moving coord via incremental adds along x */
      double dmx[3] = {P[0] * base_step, P[4] * base_step, P[8] * base_step};
      size_t o = 0;
      for (uint32_t oz = 0; oz < RB; oz++)
        for (uint32_t oy = 0; oy < RB; oy++) {
          double w[3] = {(double)(((uint64_t)bx * RB) << level),
                         (double)(((uint64_t)by * RB + oy) << level),
                         (double)(((uint64_t)bz * RB + oz) << level)};
          double m[3];
          m34_apply(P, w, m);
          double lx = m[0] * linv - (double)o0[0], ly = m[1] * linv - (double)o0[1],
                 lz = m[2] * linv - (double)o0[2];
          double sx = dmx[0] * linv, sy = dmx[1] * linv, sz = dmx[2] * linv;
          for (uint32_t ox = 0; ox < RB; ox++, o++) {
            double v = tri_scratch(rv->scratch, on[0], on[1], on[2], lx, ly, lz);
            out[o] = (uint8_t)(v + 0.5);
            lx += sx;
            ly += sy;
            lz += sz;
          }
        }
      return;
    }
  }
  /* oversized footprint (extreme rotation/scale): per-sample fallback */
  size_t o = 0;
  for (uint32_t oz = 0; oz < RB; oz++)
    for (uint32_t oy = 0; oy < RB; oy++)
      for (uint32_t ox = 0; ox < RB; ox++, o++) {
        double w[3] = {(double)(((uint64_t)bx * RB + ox) << level),
                       (double)(((uint64_t)by * RB + oy) << level),
                       (double)(((uint64_t)bz * RB + oz) << level)};
        double m[3];
        m34_apply(P, w, m);
        double v = r3d_cpuvol_tri(&rv->mv, ml, m, NULL);
        out[o] = (uint8_t)(v + 0.5);
      }
}

/* ---- transform.json ------------------------------------------------------ */

static char *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long ln = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *s = ln > 0 ? malloc((size_t)ln + 1) : NULL;
  if (s && fread(s, 1, (size_t)ln, f) != (size_t)ln) {
    free(s);
    s = NULL;
  }
  fclose(f);
  if (s) {
    s[ln] = 0;
    if (n) *n = (size_t)ln;
  }
  return s;
}

/* collect the first `want` numbers after `key` (tolerant of nesting/rows) */
static int json_nums(const char *s, const char *key, double *out, int want) {
  const char *p = strstr(s, key);
  if (!p) return -1;
  p += strlen(key);
  int got = 0;
  while (*p && got < want) {
    if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+') {
      char *e = NULL;
      double v = strtod(p, &e);
      if (e != p) {
        out[got++] = v;
        p = e;
        continue;
      }
    }
    if (*p == '}' && got > 0) break; /* ran off the array */
    p++;
  }
  return got == want ? 0 : -1;
}

int r3d_regvol_load_json(r3d_regvol *rv, const char *path) {
  char *s = read_file(path, NULL);
  if (!s) return -1;
  double m[12];
  int rc = -1;
  if (json_nums(s, "\"pull_matrix_xyz\"", m, 12) == 0) {
    pthread_mutex_lock(&rv->mu);
    memcpy(rv->M, m, sizeof m);
    pthread_mutex_unlock(&rv->mu);
    rc = 0;
  } else if (json_nums(s, "\"transformation_matrix\"", m, 12) == 0) {
    /* shipped convention: 3x4, (X,Y,Z) axis order, maps MOVING -> FIXED
     * voxels (REG_REFINE.md coordinate reconciliation); invert into the
     * pull map fixed -> moving */
    double inv[12];
    if (m34_invert(m, inv) == 0) {
      pthread_mutex_lock(&rv->mu);
      memcpy(rv->M, inv, sizeof inv);
      pthread_mutex_unlock(&rv->mu);
      rc = 0;
    }
  }
  free(s);
  if (rc == 0) {
    r3d_regvol_reset_deltas(rv);
    printf("regvol: transform loaded from %s (scale %.4f)\n", path,
           cbrt(fabs(m34_det(rv->M))));
  } else
    fprintf(stderr, "regvol: no usable matrix in %s\n", path);
  return rc;
}

int r3d_regvol_save_json(r3d_regvol *rv, const char *path) {
  double P[12], fwd[12];
  r3d_regvol_pull(rv, P);
  if (m34_invert(P, fwd) != 0) return -1;
  FILE *f = fopen(path, "w");
  if (!f) return -1;
  fprintf(f, "{\n  \"transformation_matrix\": [\n");
  for (int r = 0; r < 3; r++)
    fprintf(f, "    [%.12g, %.12g, %.12g, %.12g]%s\n", fwd[r * 4], fwd[r * 4 + 1],
            fwd[r * 4 + 2], fwd[r * 4 + 3], r < 2 ? "," : "");
  fprintf(f, "  ],\n  \"pull_matrix_xyz\": [\n");
  for (int r = 0; r < 3; r++)
    fprintf(f, "    [%.12g, %.12g, %.12g, %.12g]%s\n", P[r * 4], P[r * 4 + 1], P[r * 4 + 2],
            P[r * 4 + 3], r < 2 ? "," : "");
  fprintf(f,
          "  ],\n  \"axis_order\": \"xyz\",\n"
          "  \"direction\": \"transformation_matrix maps moving->fixed voxels; "
          "pull_matrix_xyz maps fixed->moving\",\n"
          "  \"moving_volume\": \"%s\"\n}\n",
          rv->root);
  fclose(f);
  printf("regvol: transform saved to %s\n", path);
  return 0;
}

/* ---- worker: NCC measure / auto-refine ----------------------------------- */

static void *reg_worker(void *a) {
  r3d_regvol *rv = a;
  uint32_t n = rv->job_half * 2u, Lr = rv->job_level;
  size_t vn = (size_t)n * n * n;
  float *ff = malloc(vn * sizeof(float)), *mf = malloc(vn * sizeof(float));
  uint8_t *u8 = malloc(vn);
  r3d_cpuvol fx;
  bool fx_ok = false;
  int st = 3;
  if (!ff || !mf || !u8) goto done;
  if (r3d_cpuvol_open(&fx, rv->fixed_root, 768) != 0) goto done;
  fx_ok = true;
  {
    int64_t o0[3];
    for (int q = 0; q < 3; q++)
      o0[q] = (int64_t)llround(rv->job_ctr[q] / (double)(1u << Lr)) - (int64_t)rv->job_half;
    r3d_cpuvol_read_block(&fx, Lr, o0[0], o0[1], o0[2], n, n, n, u8);
    for (size_t i = 0; i < vn; i++) ff[i] = (float)u8[i] * (1.0f / 255.0f);
    uint32_t ml = reg_pick_level(rv, rv->job_P, Lr);
    size_t i = 0;
    for (uint32_t z = 0; z < n; z++)
      for (uint32_t y = 0; y < n; y++)
        for (uint32_t x = 0; x < n; x++, i++) {
          double up0 = (double)(1u << Lr);
          double w[3] = {(double)(o0[0] + (int64_t)x) * up0,
                         (double)(o0[1] + (int64_t)y) * up0,
                         (double)(o0[2] + (int64_t)z) * up0};
          double m[3];
          m34_apply(rv->job_P, w, m);
          mf[i] = (float)(r3d_cpuvol_tri(&rv->mv, ml, m, NULL) * (1.0 / 255.0));
        }
    double ident[12];
    m34_identity(ident);
    rv->ncc0 = fy_ncc_warped(ff, mf, (int)n, (int)n, (int)n, ident);
    rv->ncc1 = rv->ncc0;
    if (rv->job_mode == 0) {
      st = 2;
      goto done;
    }
    double mc[12];
    m34_identity(mc); /* correction, ROI coords, fysics (z,y,x) order */
    if (fy_register_affine(ff, mf, (int)n, (int)n, (int)n, mc, rv->job_mode == 1) != 0)
      goto done;
    rv->ncc1 = fy_ncc_warped(ff, mf, (int)n, (int)n, (int)n, mc);
    /* compose the ROI-space correction into the pull map (base xyz coords):
     * P' = P ∘ S(2^Lr) ∘ T(+o0) ∘ C ∘ T(-o0) ∘ S(2^-Lr) */
    double cx[12], t[12], sc[12];
    m34_swap_order(mc, cx);
    double up = (double)(1u << Lr), dn = 1.0 / up;
    m34_translate(t, -(double)o0[0], -(double)o0[1], -(double)o0[2]);
    memcpy(sc, (double[12]){dn, 0, 0, 0, 0, dn, 0, 0, 0, 0, dn, 0}, sizeof sc);
    m34_mul(t, t, sc);   /* T(-o0) S(dn) */
    m34_mul(cx, cx, t);  /* C ... */
    m34_translate(t, (double)o0[0], (double)o0[1], (double)o0[2]);
    m34_mul(cx, t, cx);  /* T(+o0) ... */
    memcpy(sc, (double[12]){up, 0, 0, 0, 0, up, 0, 0, 0, 0, up, 0}, sizeof sc);
    m34_mul(cx, sc, cx); /* S(up) ... */
    m34_mul(rv->job_Mnew, rv->job_P, cx);
    st = 2;
  }
done:
  if (fx_ok) r3d_cpuvol_close(&fx);
  free(ff);
  free(mf);
  free(u8);
  atomic_store(&rv->state, st);
  return NULL;
}

int r3d_regvol_job_start(r3d_regvol *rv, const char *fixed_root, int mode, const double ctr[3],
                         uint32_t half, uint32_t level) {
  if (!rv->open || atomic_load(&rv->state) == 1) return -1;
  if (rv->th_up) {
    pthread_join(rv->th, NULL);
    rv->th_up = false;
  }
  snprintf(rv->fixed_root, sizeof rv->fixed_root, "%s", fixed_root);
  rv->job_mode = mode;
  memcpy(rv->job_ctr, ctr, sizeof rv->job_ctr);
  rv->job_half = half ? half : 64u;
  rv->job_level = level;
  r3d_regvol_pull(rv, rv->job_P);
  atomic_store(&rv->state, 1);
  if (pthread_create(&rv->th, NULL, reg_worker, rv) != 0) {
    atomic_store(&rv->state, 0);
    return -1;
  }
  rv->th_up = true;
  return 0;
}

int r3d_regvol_job_poll(r3d_regvol *rv, bool *ok) {
  int st = atomic_load(&rv->state);
  if (st == 1) return 1;
  if (ok) *ok = st != 3;
  if (st == 0) return 0;
  if (rv->th_up) {
    pthread_join(rv->th, NULL);
    rv->th_up = false;
  }
  if (st == 2 && rv->job_mode != 0) { /* apply the refined transform */
    pthread_mutex_lock(&rv->mu);
    memcpy(rv->M, rv->job_Mnew, sizeof rv->M);
    memset(rv->d_tr, 0, sizeof rv->d_tr);
    memset(rv->d_rot, 0, sizeof rv->d_rot);
    rv->d_lscale = 0.0;
    pthread_mutex_unlock(&rv->mu);
    rv->gen++;
    printf("regvol: refine applied — NCC %.4f -> %.4f\n", rv->ncc0, rv->ncc1);
  }
  atomic_store(&rv->state, 0);
  return 0;
}
