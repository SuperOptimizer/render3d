/* tifxyz-resample: re-grid a tifxyz surface to a finer parametric grid.
 *
 *   tifxyz-resample <in-tifxyz-dir> <out-tifxyz-dir> <factor> [--bilinear]
 *
 * Output has (W-1)*factor+1 x (H-1)*factor+1 points; meta.json "scale" is
 * multiplied by factor (published surfaces use 0.05 = one grid step per 20
 * voxels, so factor 20 reaches native voxel spacing). Coordinates are
 * interpolated per plane with Catmull-Rom bicubic (bilinear on request); an
 * output point is invalid when any of its interpolation taps is invalid, so
 * holes grow by up to one coarse cell. Rows stream to LZW float TIFFs, so
 * memory is the input grid plus a few output rows. */
#include <errno.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>

#include "core/tifxyz.h"

static TIFF *plane_open(const char *path, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, (uint64_t)w * h * 4 > (UINT64_C(3) << 30) ? "w8" : "w");
  if (!tf) return NULL;
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
  TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 64);
  return tf;
}

/* Catmull-Rom weights for fractional position t */
static void cubic_w(double t, double w[4]) {
  double t2 = t * t, t3 = t2 * t;
  w[0] = 0.5 * (-t3 + 2 * t2 - t);
  w[1] = 0.5 * (3 * t3 - 5 * t2 + 2);
  w[2] = 0.5 * (-3 * t3 + 4 * t2 + t);
  w[3] = 0.5 * (t3 - t2);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: tifxyz-resample <in-dir> <out-dir> <factor> [--bilinear]\n");
    return EXIT_FAILURE;
  }
  const char *in = argv[1], *out = argv[2];
  int k = atoi(argv[3]);
  bool bilinear = argc > 4 && !strcmp(argv[4], "--bilinear");
  if (k < 1 || k > 64) {
    fprintf(stderr, "factor must be 1..64\n");
    return EXIT_FAILURE;
  }
  r3d_tifxyz s;
  if (r3d_tifxyz_load(&s, in) != 0) return EXIT_FAILURE;
  if (s.w < 2 || s.h < 2) return EXIT_FAILURE;
  uint32_t W = (s.w - 1) * (uint32_t)k + 1, H = (s.h - 1) * (uint32_t)k + 1;
  if (mkdir(out, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "cannot create %s\n", out);
    return EXIT_FAILURE;
  }
  TIFF *tf[3];
  char path[2048];
  for (int a = 0; a < 3; a++) {
    snprintf(path, sizeof path, "%s/%c.tif", out, "xyz"[a]);
    tf[a] = plane_open(path, W, H);
    if (!tf[a]) return EXIT_FAILURE;
  }
  float *row = malloc((size_t)W * 3 * sizeof *row);
  if (!row) return EXIT_FAILURE;
  int taps = bilinear ? 2 : 4, off = bilinear ? 0 : -1;
  uint64_t nvalid = 0;
  for (uint32_t oy = 0; oy < H; oy++) {
    uint32_t j0 = oy / (uint32_t)k;
    double ty = (double)(oy % (uint32_t)k) / k;
    double wy[4], wx[4];
    if (bilinear) {
      wy[0] = 1 - ty;
      wy[1] = ty;
    } else
      cubic_w(ty, wy);
    for (uint32_t ox = 0; ox < W; ox++) {
      uint32_t i0 = ox / (uint32_t)k;
      double tx = (double)(ox % (uint32_t)k) / k;
      if (bilinear) {
        wx[0] = 1 - tx;
        wx[1] = tx;
      } else
        cubic_w(tx, wx);
      double acc[3] = {0, 0, 0};
      bool ok = true;
      for (int dj = 0; ok && dj < taps; dj++) {
        int jj = (int)j0 + dj + off;
        if (jj < 0) jj = 0;
        if (jj >= (int)s.h) jj = (int)s.h - 1;
        if (fabs(wy[dj]) < 1e-12) continue;
        for (int di = 0; di < taps; di++) {
          int ii = (int)i0 + di + off;
          if (ii < 0) ii = 0;
          if (ii >= (int)s.w) ii = (int)s.w - 1;
          if (fabs(wx[di]) < 1e-12) continue;
          const float *p = r3d_tifxyz_at(&s, (uint32_t)ii, (uint32_t)jj);
          if (!r3d_tifxyz_valid(p)) {
            ok = false;
            break;
          }
          double w = wy[dj] * wx[di];
          for (int a = 0; a < 3; a++) acc[a] += w * (double)p[a];
        }
      }
      for (int a = 0; a < 3; a++)
        row[(size_t)a * W + ox] = ok ? (float)acc[a] : -1.0f;
      if (ok) nvalid++;
    }
    for (int a = 0; a < 3; a++)
      if (TIFFWriteScanline(tf[a], row + (size_t)a * W, oy, 0) < 0) return EXIT_FAILURE;
  }
  for (int a = 0; a < 3; a++) TIFFClose(tf[a]);
  free(row);
  snprintf(path, sizeof path, "%s/meta.json", out);
  FILE *mf = fopen(path, "w");
  if (!mf) return EXIT_FAILURE;
  fprintf(mf,
          "{\n  \"format\": \"tifxyz\",\n  \"type\": \"seg\",\n  \"scale\": [\n    %.6f,\n    "
          "%.6f\n  ],\n  \"source\": \"tifxyz-resample x%d %s of %s\"\n}\n",
          (double)s.sx * k, (double)s.sy * k, k, bilinear ? "bilinear" : "bicubic", in);
  fclose(mf);
  printf("%s: %ux%u -> %ux%u (x%d %s), %llu valid of %llu\n", out, s.w, s.h, W, H, k,
         bilinear ? "bilinear" : "bicubic", (unsigned long long)nvalid,
         (unsigned long long)W * H);
  r3d_tifxyz_free(&s);
  return EXIT_SUCCESS;
}
