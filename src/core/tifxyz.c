#include "core/tifxyz.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

/* Read one single-channel float32 TIFF plane (striped or tiled). Returns a
 * malloc'd w*h float buffer, or NULL. The first call fixes the shared dims;
 * later calls must match. */
static float *read_plane(const char *path, uint32_t *w, uint32_t *h) {
  TIFF *tf = TIFFOpen(path, "r");
  if (!tf) {
    fprintf(stderr, "tifxyz: cannot open %s\n", path);
    return NULL;
  }
  uint32_t iw = 0, ih = 0;
  uint16_t bps = 0, fmt = 0, spp = 1;
  TIFFGetField(tf, TIFFTAG_IMAGEWIDTH, &iw);
  TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &ih);
  TIFFGetField(tf, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetField(tf, TIFFTAG_SAMPLEFORMAT, &fmt);
  TIFFGetFieldDefaulted(tf, TIFFTAG_SAMPLESPERPIXEL, &spp);
  if (!iw || !ih || bps != 32 || fmt != SAMPLEFORMAT_IEEEFP || spp != 1) {
    fprintf(stderr, "tifxyz: %s is not single-channel float32 (%ux%u bps %u fmt %u spp %u)\n",
            path, iw, ih, bps, fmt, spp);
    TIFFClose(tf);
    return NULL;
  }
  if (*w && (*w != iw || *h != ih)) {
    fprintf(stderr, "tifxyz: %s dims %ux%u mismatch %ux%u\n", path, iw, ih, *w, *h);
    TIFFClose(tf);
    return NULL;
  }
  *w = iw;
  *h = ih;
  float *out = malloc((uint64_t)iw * ih * sizeof *out);
  if (!out) {
    TIFFClose(tf);
    return NULL;
  }
  int ok = 1;
  if (TIFFIsTiled(tf)) {
    uint32_t tw = 0, th = 0;
    TIFFGetField(tf, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(tf, TIFFTAG_TILELENGTH, &th);
    float *tile = malloc((uint64_t)tw * th * sizeof *tile);
    if (!tile) ok = 0;
    for (uint32_t ty = 0; ok && ty < ih; ty += th)
      for (uint32_t tx = 0; ok && tx < iw; tx += tw) {
        if (TIFFReadTile(tf, tile, tx, ty, 0, 0) < 0) {
          ok = 0;
          break;
        }
        uint32_t cw = tw < iw - tx ? tw : iw - tx, ch = th < ih - ty ? th : ih - ty;
        for (uint32_t r = 0; r < ch; r++)
          memcpy(out + (uint64_t)(ty + r) * iw + tx, tile + (uint64_t)r * tw,
                 cw * sizeof *out);
      }
    free(tile);
  } else {
    for (uint32_t r = 0; ok && r < ih; r++)
      if (TIFFReadScanline(tf, out + (uint64_t)r * iw, r, 0) < 0) ok = 0;
  }
  TIFFClose(tf);
  if (!ok) {
    fprintf(stderr, "tifxyz: read error in %s\n", path);
    free(out);
    return NULL;
  }
  return out;
}

int r3d_tifxyz_load(r3d_tifxyz *s, const char *dir) {
  memset(s, 0, sizeof *s);
  TIFFSetWarningHandler(NULL); /* old-style JPEG etc. chatter */

  char path[1024];
  snprintf(path, sizeof path, "%s/meta.json", dir);
  FILE *mf = fopen(path, "r");
  if (!mf) {
    fprintf(stderr, "tifxyz: missing %s\n", path);
    return -1;
  }
  char meta[4096] = {0};
  size_t mn = fread(meta, 1, sizeof meta - 1, mf);
  fclose(mf);
  (void)mn;
  const char *sc = strstr(meta, "\"scale\"");
  double sx = 0, sy = 0;
  if (!sc || !(sc = strchr(sc, '[')) || sscanf(sc, "[ %lf , %lf", &sx, &sy) != 2 ||
      sx <= 0 || sy <= 0) {
    fprintf(stderr, "tifxyz: %s/meta.json lacks a valid \"scale\" [sx, sy]\n", dir);
    return -1;
  }
  s->sx = (float)sx;
  s->sy = (float)sy;

  static const char *plane_name[3] = {"x.tif", "y.tif", "z.tif"};
  float *plane[3] = {0};
  for (int c = 0; c < 3; c++) {
    snprintf(path, sizeof path, "%s/%s", dir, plane_name[c]);
    plane[c] = read_plane(path, &s->w, &s->h);
    if (!plane[c]) {
      for (int k = 0; k < c; k++) free(plane[k]);
      return -1;
    }
  }

  uint64_t n = (uint64_t)s->w * s->h;
  s->xyz = malloc(n * 3 * sizeof *s->xyz);
  if (!s->xyz) {
    for (int c = 0; c < 3; c++) free(plane[c]);
    return -1;
  }
  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (uint64_t i = 0; i < n; i++) {
    float x = plane[0][i], y = plane[1][i], z = plane[2][i];
    /* vc3d rule: unmapped points are stored as 0s; z <= 0 (or non-finite)
     * means the whole triplet is invalid */
    if (!(z > 0.0f) || !isfinite(x) || !isfinite(y)) {
      s->xyz[i * 3 + 0] = s->xyz[i * 3 + 1] = s->xyz[i * 3 + 2] = -1.0f;
      continue;
    }
    s->xyz[i * 3 + 0] = x;
    s->xyz[i * 3 + 1] = y;
    s->xyz[i * 3 + 2] = z;
    s->nvalid++;
    float p[3] = {x, y, z};
    for (int c = 0; c < 3; c++) {
      if (p[c] < lo[c]) lo[c] = p[c];
      if (p[c] > hi[c]) hi[c] = p[c];
    }
  }
  for (int c = 0; c < 3; c++) free(plane[c]);
  if (s->nvalid) {
    memcpy(s->bbox[0], lo, sizeof lo);
    memcpy(s->bbox[1], hi, sizeof hi);
  }
  return 0;
}

void r3d_tifxyz_free(r3d_tifxyz *s) {
  free(s->xyz);
  memset(s, 0, sizeof *s);
}
