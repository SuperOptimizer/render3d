/* Synthetic TSM annotation packets (header-only, test-side).
 *
 * Builds a packet directory that matches spec/annot.md exactly: a curved
 * "papyrus" sheet crossing the crop, with the two faces labelled on either
 * side of it, an ignore block, a source map, an rv_class banding and a pair
 * of deliberately-wrong predictions.  Tests use it in place of a real
 * dev/annot_export.py packet. */
#ifndef R3D_TESTS_ANNOTSYNTH_H
#define R3D_TESTS_ANNOTSYNTH_H

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Sheet surface: x position of the sheet centre at (z,y). */
static inline double annotsynth_sheet_x(uint32_t nx, uint32_t z, uint32_t y) {
  return (double)nx * 0.5 + 6.0 * sin((double)y * 0.15) + 2.0 * cos((double)z * 0.2);
}

static inline int annotsynth_write(const char *dir, const char *name, const void *p, size_t n) {
  char path[1024];
  if (snprintf(path, sizeof path, "%s/%s", dir, name) >= (int)sizeof path) return -1;
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  int ok = n == 0 || fwrite(p, 1, n, f) == n;
  return fclose(f) == 0 && ok ? 0 : -1;
}

/* Writes <dir>/{ct,faces_in,faces_out,ignore,source,rv_class,pred_in,pred_out}.u8
 * plus meta.json. `with_pred` = 0 omits the two prediction volumes. */
static inline int annotsynth_packet(const char *dir, uint32_t nz, uint32_t ny, uint32_t nx,
                                    const int64_t origin[3], int with_pred) {
  if (mkdir(dir, 0777) != 0 && errno != EEXIST) return -1;
  size_t n = (size_t)nz * ny * nx;
  uint8_t *ct = calloc(n, 1), *fi = calloc(n, 1), *fo = calloc(n, 1), *ig = calloc(n, 1),
          *src = calloc(n, 1), *rv = calloc(n, 1), *pi = calloc(n, 1), *po = calloc(n, 1);
  if (!ct || !fi || !fo || !ig || !src || !rv || !pi || !po) return -1;
  for (uint32_t z = 0; z < nz; z++)
    for (uint32_t y = 0; y < ny; y++) {
      double cx = annotsynth_sheet_x(nx, z, y);
      for (uint32_t x = 0; x < nx; x++) {
        size_t i = ((size_t)z * ny + y) * nx + x;
        double d = (double)x - cx; /* signed distance across the sheet */
        double a = fabs(d);
        ct[i] = (uint8_t)(a < 4.0 ? 200.0 - 30.0 * a : 40.0);
        if (d <= -3.0 && d > -4.5) { fi[i] = 1; rv[i] = 1; src[i] = 1; }
        if (d >= 3.0 && d < 4.5) { fo[i] = 1; rv[i] = 2; src[i] = 1; }
        if (a < 1.0) rv[i] = 3;
        if (z < 2u) { ig[i] = 1; src[i] = 0; }
        if (with_pred) { /* predictions shifted one voxel: something to fix */
          if (d <= -2.0 && d > -3.5) pi[i] = 1;
          if (d >= 4.0 && d < 5.5) po[i] = 1;
        }
      }
    }
  int rc = 0;
  rc |= annotsynth_write(dir, "ct.u8", ct, n);
  rc |= annotsynth_write(dir, "faces_in.u8", fi, n);
  rc |= annotsynth_write(dir, "faces_out.u8", fo, n);
  rc |= annotsynth_write(dir, "ignore.u8", ig, n);
  rc |= annotsynth_write(dir, "source.u8", src, n);
  rc |= annotsynth_write(dir, "rv_class.u8", rv, n);
  if (with_pred) {
    rc |= annotsynth_write(dir, "pred_in.u8", pi, n);
    rc |= annotsynth_write(dir, "pred_out.u8", po, n);
  }
  char meta[1024];
  int mn = snprintf(meta, sizeof meta,
                    "{\n  \"dims_zyx\": [%u, %u, %u],\n"
                    "  \"origin_zyx\": [%lld, %lld, %lld],\n"
                    "  \"voxel_um\": 2.4,\n"
                    "  \"layers\": [\"ct\", \"faces_in\", \"faces_out\", \"ignore\", "
                    "\"source\", \"rv_class\"%s]\n}\n",
                    nz, ny, nx, (long long)origin[0], (long long)origin[1],
                    (long long)origin[2], with_pred ? ", \"pred_in\", \"pred_out\"" : "");
  rc |= mn <= 0 || mn >= (int)sizeof meta;
  rc |= annotsynth_write(dir, "meta.json", meta, (size_t)mn);
  free(ct); free(fi); free(fo); free(ig); free(src); free(rv); free(pi); free(po);
  return rc;
}

/* Writes <root>/packet.json plus `count` packet directories p000.. under it.
 * `dims_step` grows each successive packet (0 = every packet identical), which
 * is what exercises a viewer's texture-resize path. */
static inline int annotsynth_corpus_varied(const char *root, uint32_t count, uint32_t nz,
                                           uint32_t ny, uint32_t nx, uint32_t dims_step) {
  if (mkdir(root, 0777) != 0 && errno != EEXIST) return -1;
  char json[8192];
  int off = snprintf(json, sizeof json,
                     "{\n  \"format\": \"tsm.annot.v1\",\n  \"voxel_um\": 2.4,\n"
                     "  \"palette\": {\n"
                     "    \"correction\": [\"untouched\", \"in\", \"out\", \"ignore\", \"erase\"],\n"
                     "    \"source\": [\"none\", \"rectoverso\", \"ct\", \"human\"],\n"
                     "    \"rv_class\": [\"bg\", \"recto\", \"verso\", \"contact\"]\n  },\n"
                     "  \"packets\": [\n");
  for (uint32_t i = 0; i < count; i++) {
    char sub[1024];
    if (snprintf(sub, sizeof sub, "%s/p%03u", root, i) >= (int)sizeof sub) return -1;
    int64_t origin[3] = {(int64_t)i * 100, 200, 300};
    uint32_t py = ny + i * dims_step, px = nx + i * dims_step;
    if (annotsynth_packet(sub, nz, py, px, origin, i % 2 == 0) != 0) return -1;
    off += snprintf(json + off, sizeof json - (size_t)off,
                    "    {\"path\": \"p%03u\", \"origin_zyx\": [%lld, 200, 300], "
                    "\"dims_zyx\": [%u, %u, %u]}%s\n",
                    i, (long long)origin[0], nz, py, px, i + 1 < count ? "," : "");
    if (off <= 0 || off >= (int)sizeof json) return -1;
  }
  off += snprintf(json + off, sizeof json - (size_t)off, "  ]\n}\n");
  if (off <= 0 || off >= (int)sizeof json) return -1;
  return annotsynth_write(root, "packet.json", json, (size_t)off);
}

static inline int annotsynth_corpus(const char *root, uint32_t count, uint32_t nz, uint32_t ny,
                                    uint32_t nx) {
  return annotsynth_corpus_varied(root, count, nz, ny, nx, 0);
}

#endif /* R3D_TESTS_ANNOTSYNTH_H */
