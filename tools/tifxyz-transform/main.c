/* tifxyz-transform -- apply a volume-registration transform.json affine to a
 * tifxyz segment surface, re-mapping it from one volume's voxel space to
 * another's. Reimplements volume-cartographer's vc_transform_geom /
 * transformSurfacePoints semantics exactly:
 *
 *   p' = scale_after * (M @ (scale_before * p))     (XYZ, column vector)
 *
 * transform.json stores a 3x4 row-major XYZ matrix meaning
 * p_fixed = M @ p_moving; check its "fixed_volume" id to decide direction —
 * mapping a segment traced on the FIXED volume onto the moving one needs
 * --invert. Invalid grid points (-1,-1,-1) pass through untouched. After
 * transforming, the grid is resampled by the measured median adjacent-point
 * spacing ratio (vc measureGridAxisSpacing: interior 10-90%, step 4) so one
 * grid cell again spans ~1/scale target voxels; meta.json "scale" is
 * preserved from the source and bbox is recomputed.
 *
 * Validated against the open-data bucket: transforming PHercParis4 GP
 * segment 20230702185753 from its 7.91um tifxyz with the 2.4um volume's
 * transform.json (--invert, since fixed_volume = the canonical 7.91um)
 * reproduces the published "-on-...-2.4um.tifxyz" exactly — same 1820x2530
 * grid, coordinates within 0.002 voxels mean / 0.016 max (f32 rounding). */
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>

#include "core/tifxyz.h"

typedef struct aff {
  double m[3][3], t[3];
} aff;

static int parse_transform(const char *path, aff *out) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "tifxyz-transform: cannot open %s\n", path);
    return -1;
  }
  char *buf = malloc(1 << 20);
  size_t n = fread(buf, 1, (1 << 20) - 1, f);
  buf[n] = 0;
  fclose(f);
  const char *p = strstr(buf, "\"transformation_matrix\"");
  if (!p) {
    fprintf(stderr, "tifxyz-transform: no transformation_matrix in %s\n", path);
    free(buf);
    return -1;
  }
  double v[12];
  int got = 0;
  while (got < 12 && *p) {
    char *end;
    double d = strtod(p, &end);
    if (end != p) {
      v[got++] = d;
      p = end;
    } else {
      if (*p == ']' && got && got % 4 == 0 && p[1] == ']') break;
      p++;
    }
  }
  free(buf);
  if (got != 12) {
    fprintf(stderr, "tifxyz-transform: matrix has %d values, want 12\n", got);
    return -1;
  }
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) out->m[r][c] = v[r * 4 + c];
    out->t[r] = v[r * 4 + 3];
  }
  return 0;
}

static int aff_invert(const aff *a, aff *out) {
  const double(*m)[3] = a->m;
  double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (fabs(det) < 1e-30) return -1;
  double inv[3][3] = {
      {(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
       (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
       (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det},
      {(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
       (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
       (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det},
      {(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
       (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
       (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det}};
  memcpy(out->m, inv, sizeof inv);
  for (int r = 0; r < 3; r++)
    out->t[r] = -(inv[r][0] * a->t[0] + inv[r][1] * a->t[1] + inv[r][2] * a->t[2]);
  return 0;
}

static void apply(const aff *a, double sb, double sa, const float *p, float *q) {
  double x = (double)p[0] * sb, y = (double)p[1] * sb, z = (double)p[2] * sb;
  for (int r = 0; r < 3; r++)
    q[r] = (float)((a->m[r][0] * x + a->m[r][1] * y + a->m[r][2] * z + a->t[r]) * sa);
}

static int dbl_cmp(const void *a, const void *b) {
  double x = *(const double *)a - *(const double *)b;
  return x < 0 ? -1 : (x > 0 ? 1 : 0);
}

/* vc measureGridAxisSpacing: median 3D distance between adjacent grid points
 * along an axis, interior 10-90% box, sampled at step 4 */
static double spacing(const float *xyz, uint32_t w, uint32_t h, int axis) {
  double *d = malloc((size_t)w * h * sizeof *d);
  size_t nd = 0;
  uint32_t i0 = w / 10, i1 = w - w / 10, j0 = h / 10, j1 = h - h / 10;
  for (uint32_t j = j0; j + 1 < j1; j += 4)
    for (uint32_t i = i0; i + 1 < i1; i += 4) {
      const float *p = xyz + ((size_t)j * w + i) * 3;
      const float *q = axis == 0 ? p + 3 : p + (size_t)w * 3;
      if (p[0] == -1.0f || q[0] == -1.0f) continue;
      double dx = (double)q[0] - (double)p[0], dy = (double)q[1] - (double)p[1],
             dz = (double)q[2] - (double)p[2];
      d[nd++] = sqrt(dx * dx + dy * dy + dz * dz);
    }
  double med = 0.0;
  if (nd) {
    qsort(d, nd, sizeof *d, dbl_cmp);
    med = d[nd / 2];
  }
  free(d);
  return med;
}

/* linear resample preserving invalids (vc resamplePointsLinearPreservingInvalids:
 * a blend never crosses an invalid; fall back to the nearest valid corner) */
static float *resample(const float *src, uint32_t w, uint32_t h, uint32_t nw, uint32_t nh) {
  float *out = malloc((size_t)nw * nh * 3 * sizeof *out);
  if (!out) return NULL;
  double fx = (double)w / nw, fy = (double)h / nh;
  for (uint32_t j = 0; j < nh; j++)
    for (uint32_t i = 0; i < nw; i++) {
      double u = ((double)i + 0.5) * fx - 0.5, v = ((double)j + 0.5) * fy - 0.5;
      if (u < 0) u = 0;
      if (v < 0) v = 0;
      if (u > (double)w - 1.0) u = (double)w - 1.0;
      if (v > (double)h - 1.0) v = (double)h - 1.0;
      uint32_t u0 = (uint32_t)u, v0 = (uint32_t)v;
      if (u0 > w - 2) u0 = w - 2;
      if (v0 > h - 2) v0 = h - 2;
      double du = u - u0, dv = v - v0;
      const float *c[4] = {src + ((size_t)v0 * w + u0) * 3, src + ((size_t)v0 * w + u0 + 1) * 3,
                           src + ((size_t)(v0 + 1) * w + u0) * 3,
                           src + ((size_t)(v0 + 1) * w + u0 + 1) * 3};
      float *o = out + ((size_t)j * nw + i) * 3;
      bool ok = c[0][0] != -1.0f && c[1][0] != -1.0f && c[2][0] != -1.0f && c[3][0] != -1.0f;
      if (ok) {
        for (int k = 0; k < 3; k++)
          o[k] = (float)(((double)c[0][k] * (1 - du) + (double)c[1][k] * du) * (1 - dv) +
                         ((double)c[2][k] * (1 - du) + (double)c[3][k] * du) * dv);
      } else {
        const float *near = c[(dv >= 0.5 ? 2 : 0) + (du >= 0.5 ? 1 : 0)];
        if (near[0] != -1.0f) memcpy(o, near, 3 * sizeof *o);
        else o[0] = o[1] = o[2] = -1.0f;
      }
    }
  return out;
}

static int write_plane(const char *path, const float *xyz, int comp, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, "w8");
  if (!tf) return -1;
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(tf, TIFFTAG_PREDICTOR, 3);
  TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 64);
  float *row = malloc((size_t)w * sizeof *row);
  int rc = 0;
  for (uint32_t j = 0; j < h && rc == 0; j++) {
    for (uint32_t i = 0; i < w; i++) row[i] = xyz[((size_t)j * w + i) * 3 + (size_t)comp];
    if (TIFFWriteScanline(tf, row, j, 0) < 0) rc = -1;
  }
  free(row);
  TIFFClose(tf);
  return rc;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: tifxyz-transform <in-tifxyz-dir> <out-dir> --affine transform.json "
            "[--invert] [--scale-before S] [--scale-after S]\n");
    return 2;
  }
  const char *in_dir = argv[1], *out_dir = argv[2], *aff_path = NULL;
  bool invert = false;
  double sb = 1.0, sa = 1.0;
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--affine") == 0 && i + 1 < argc) aff_path = argv[++i];
    else if (strcmp(argv[i], "--invert") == 0) invert = true;
    else if (strcmp(argv[i], "--scale-before") == 0 && i + 1 < argc) sb = atof(argv[++i]);
    else if (strcmp(argv[i], "--scale-after") == 0 && i + 1 < argc) sa = atof(argv[++i]);
    else {
      fprintf(stderr, "tifxyz-transform: unknown option %s\n", argv[i]);
      return 2;
    }
  }
  aff a = {.m = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, .t = {0, 0, 0}};
  if (aff_path && parse_transform(aff_path, &a) != 0) return 1;
  if (invert) {
    aff ai;
    if (aff_invert(&a, &ai) != 0) {
      fprintf(stderr, "tifxyz-transform: matrix is singular\n");
      return 1;
    }
    a = ai;
  }

  r3d_tifxyz s;
  if (r3d_tifxyz_load(&s, in_dir) != 0) return 1;
  double sp_before[2] = {spacing(s.xyz, s.w, s.h, 0), spacing(s.xyz, s.w, s.h, 1)};
  for (uint64_t k = 0; k < (uint64_t)s.w * s.h; k++) {
    float *p = s.xyz + k * 3;
    if (p[0] == -1.0f) continue; /* invalid passes through (vc semantics) */
    float q[3];
    apply(&a, sb, sa, p, q);
    if (!isfinite(q[0]) || !isfinite(q[1]) || !isfinite(q[2])) {
      p[0] = p[1] = p[2] = -1.0f;
      continue;
    }
    memcpy(p, q, sizeof q);
  }
  double sp_after[2] = {spacing(s.xyz, s.w, s.h, 0), spacing(s.xyz, s.w, s.h, 1)};
  double fx = sp_before[0] > 0 ? sp_after[0] / sp_before[0] : 1.0;
  double fy = sp_before[1] > 0 ? sp_after[1] / sp_before[1] : 1.0;
  uint32_t nw = s.w, nh = s.h;
  float *grid = s.xyz;
  if (fabs(fx - 1.0) > 1e-4 || fabs(fy - 1.0) > 1e-4) {
    nw = (uint32_t)llround((double)s.w * fx);
    nh = (uint32_t)llround((double)s.h * fy);
    if (nw < 2) nw = 2;
    if (nh < 2) nh = 2;
    grid = resample(s.xyz, s.w, s.h, nw, nh);
    if (!grid) return 1;
  }
  printf("tifxyz-transform: %ux%u -> %ux%u (spacing x%.4f/x%.4f), scale [%g, %g] kept\n",
         s.w, s.h, nw, nh, fx, fy, (double)s.sx, (double)s.sy);

  if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "tifxyz-transform: cannot create %s\n", out_dir);
    return 1;
  }
  char path[1024];
  static const char *pn[3] = {"x.tif", "y.tif", "z.tif"};
  for (int c = 0; c < 3; c++) {
    snprintf(path, sizeof path, "%s/%s", out_dir, pn[c]);
    /* invalid points are written as -1 (z <= 0 marks them invalid on load) */
    if (write_plane(path, grid, c, nw, nh) != 0) {
      fprintf(stderr, "tifxyz-transform: cannot write %s\n", path);
      return 1;
    }
  }
  double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
  uint64_t nvalid = 0;
  for (uint64_t k = 0; k < (uint64_t)nw * nh; k++) {
    const float *p = grid + k * 3;
    if (p[0] == -1.0f) continue;
    nvalid++;
    for (int c = 0; c < 3; c++) {
      if ((double)p[c] < lo[c]) lo[c] = (double)p[c];
      if ((double)p[c] > hi[c]) hi[c] = (double)p[c];
    }
  }
  snprintf(path, sizeof path, "%s/meta.json", out_dir);
  FILE *mf = fopen(path, "w");
  if (!mf) return 1;
  fprintf(mf,
          "{\n  \"bbox\": [[%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f]],\n"
          "  \"format\": \"tifxyz\",\n  \"scale\": [%g, %g],\n"
          "  \"type\": \"seg\",\n  \"uuid\": \"transformed\"\n}\n",
          lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], (double)s.sx, (double)s.sy);
  fclose(mf);
  printf("tifxyz-transform: wrote %s (%llu valid points)\n", out_dir,
         (unsigned long long)nvalid);
  if (grid != s.xyz) free(grid);
  r3d_tifxyz_free(&s);
  return 0;
}
