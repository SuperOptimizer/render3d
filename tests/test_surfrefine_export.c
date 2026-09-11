/* Label export: tifxyz2obj and surfsamples over a synthetic tifxyz surface.
 *
 * The surface is a W x H grid of a smooth, non-degenerate patch at scale
 * 0.05 (one grid step = 20 voxels) with a rectangular hole so the compaction
 * map, the "all four corners valid" quad rule and the confidence plane are
 * all exercised.  Both executables are run through system() with the paths
 * argv carries (as tests/test_surfconv.c does) and their output is parsed
 * back:
 *   OBJ      - `v`/`vt`/`vn` counts equal the valid-point count, `f` count
 *              equals 2 x the fully-valid quad count, `g` carries --name,
 *              every index is in range, positions match the grid, normals
 *              are unit, and a rerun is byte-identical (determinism).
 *   samples  - header magic and count, positions inside the surface bbox,
 *              unit normals, per-quad sample spacing close to --spacing,
 *              confidences honouring --conf-min, and the CSV form agreeing
 *              with the binary one. */
#include "core/surfconv.h"
#include "core/tifxyz.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <unistd.h>

#undef assert
#define assert(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define W 40u
#define H 30u
/* meta.json scale is 0.05: one grid step = 20 voxels */

static float gx(uint32_t i, uint32_t j) { return 1000.f + 20.f * (float)i + 0.5f * (float)j; }
static float gy(uint32_t i, uint32_t j) { return 2000.f + 0.5f * (float)i + 20.f * (float)j; }
static float gz(uint32_t i, uint32_t j) {
  return 500.f + 3.f * sinf((float)i * 0.2f) + 2.f * cosf((float)j * 0.15f);
}
static int hole(uint32_t i, uint32_t j) { return i >= 12 && i < 18 && j >= 9 && j < 14; }
static float gconf(uint32_t i) { return (i < 4) ? 0.1f : 0.8f; }

static void write_plane(const char *path, const float *px) {
  TIFF *tf = TIFFOpen(path, "w");
  assert(tf);
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, (uint32_t)W);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, (uint32_t)H);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 5);
  for (uint32_t y = 0; y < H; y++)
    assert(TIFFWriteScanline(tf, (void *)(px + (size_t)y * W), y, 0) >= 0);
  TIFFClose(tf);
}

static const char META[] = "{\"format\":\"tifxyz\",\"scale\":[0.05,0.05]}\n";

static void make_surface(const char *dir) {
  assert(!mkdir(dir, 0755));
  float *x = malloc(W * H * sizeof *x), *y = malloc(W * H * sizeof *y),
        *z = malloc(W * H * sizeof *z), *c = malloc(W * H * sizeof *c);
  assert(x && y && z && c);
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      size_t k = (size_t)j * W + i;
      int h = hole(i, j);
      x[k] = h ? 0.f : gx(i, j);
      y[k] = h ? 0.f : gy(i, j);
      z[k] = h ? 0.f : gz(i, j); /* z <= 0 is the tifxyz invalid convention */
      c[k] = gconf(i);
    }
  char p[600];
  snprintf(p, sizeof p, "%s/x.tif", dir);
  write_plane(p, x);
  snprintf(p, sizeof p, "%s/y.tif", dir);
  write_plane(p, y);
  snprintf(p, sizeof p, "%s/z.tif", dir);
  write_plane(p, z);
  snprintf(p, sizeof p, "%s/confidence.tif", dir);
  write_plane(p, c);
  snprintf(p, sizeof p, "%s/meta.json", dir);
  FILE *f = fopen(p, "wb");
  assert(f && fwrite(META, 1, strlen(META), f) == strlen(META));
  fclose(f);
  free(x);
  free(y);
  free(z);
  free(c);
}

static char *slurp(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  assert(f);
  assert(!fseek(f, 0, SEEK_END));
  long n = ftell(f);
  assert(n >= 0);
  assert(!fseek(f, 0, SEEK_SET));
  char *buf = malloc((size_t)n + 1);
  assert(buf);
  assert(fread(buf, 1, (size_t)n, f) == (size_t)n);
  buf[n] = 0;
  fclose(f);
  *len = (size_t)n;
  return buf;
}

static double le_f32(const uint8_t *p) {
  uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
                  ((uint32_t)p[3] << 24);
  float f;
  memcpy(&f, &bits, sizeof f);
  return (double)f;
}
static uint32_t le_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

int main(int argc, char **argv) {
  assert(argc > 2); /* <tifxyz2obj> <surfsamples> */
  char obj_exe[4096], smp_exe[4096];
  assert(realpath(argv[1], obj_exe));
  assert(realpath(argv[2], smp_exe));

  char tmp[] = "/tmp/r3d-sfexport-XXXXXX";
  assert(mkdtemp(tmp));
  char dir[600], p[900], cmd[9000];
  snprintf(dir, sizeof dir, "%s/seg.tifxyz", tmp);
  make_surface(dir);

  r3d_tifxyz s;
  assert(!r3d_surf_load(dir, &s));
  assert(s.w == W && s.h == H);
  uint64_t nvalid = 0, nquad = 0;
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++)
      if (r3d_tifxyz_valid(r3d_tifxyz_at(&s, i, j))) nvalid++;
  for (uint32_t j = 0; j + 1 < H; j++)
    for (uint32_t i = 0; i + 1 < W; i++)
      if (r3d_tifxyz_valid(r3d_tifxyz_at(&s, i, j)) &&
          r3d_tifxyz_valid(r3d_tifxyz_at(&s, i + 1, j)) &&
          r3d_tifxyz_valid(r3d_tifxyz_at(&s, i, j + 1)) &&
          r3d_tifxyz_valid(r3d_tifxyz_at(&s, i + 1, j + 1)))
        nquad++;
  assert(nvalid == s.nvalid && nvalid < (uint64_t)W * H && nquad > 0);

  /* ---------------- tifxyz2obj ---------------- */
  snprintf(cmd, sizeof cmd, "'%s' '%s' '%s/m.obj' --name testsheet > /dev/null",
           obj_exe, dir, tmp);
  assert(system(cmd) == 0);
  snprintf(p, sizeof p, "%s/m.obj", tmp);
  size_t olen = 0;
  char *obj = slurp(p, &olen);
  uint64_t nv = 0, nvt = 0, nvn = 0, nf = 0, ng = 0;
  double *vpos = malloc((size_t)nvalid * 3 * sizeof *vpos);
  assert(vpos);
  for (char *line = strtok(obj, "\n"); line; line = strtok(NULL, "\n")) {
    if (!strncmp(line, "v ", 2)) {
      double a, b, c;
      assert(sscanf(line + 2, "%lf %lf %lf", &a, &b, &c) == 3);
      assert(nv < nvalid);
      vpos[nv * 3] = a;
      vpos[nv * 3 + 1] = b;
      vpos[nv * 3 + 2] = c;
      nv++;
    } else if (!strncmp(line, "vt ", 3)) {
      double u, v;
      assert(sscanf(line + 3, "%lf %lf", &u, &v) == 2);
      assert(u >= -1e-9 && u <= 1 + 1e-9 && v >= -1e-9 && v <= 1 + 1e-9);
      nvt++;
    } else if (!strncmp(line, "vn ", 3)) {
      double a, b, c;
      assert(sscanf(line + 3, "%lf %lf %lf", &a, &b, &c) == 3);
      assert(fabs(sqrt(a * a + b * b + c * c) - 1.0) < 1e-4); /* no isolated verts here */
      nvn++;
    } else if (!strncmp(line, "f ", 2)) {
      unsigned long long i0, t0, n0, i1, t1, n1, i2, t2, n2;
      assert(sscanf(line + 2, "%llu/%llu/%llu %llu/%llu/%llu %llu/%llu/%llu", &i0, &t0,
                    &n0, &i1, &t1, &n1, &i2, &t2, &n2) == 9);
      assert(i0 == t0 && i0 == n0 && i1 == t1 && i1 == n1 && i2 == t2 && i2 == n2);
      assert(i0 >= 1 && i0 <= nvalid && i1 >= 1 && i1 <= nvalid && i2 >= 1 && i2 <= nvalid);
      assert(i0 != i1 && i1 != i2 && i0 != i2);
      nf++;
    } else if (!strncmp(line, "g ", 2)) {
      assert(!strcmp(line + 2, "testsheet"));
      ng++;
    }
  }
  assert(nv == nvalid && nvt == nvalid && nvn == nvalid);
  assert(nf == nquad * 2 && ng == 1);
  { /* vertices are the valid grid points, in row-major order */
    uint64_t k = 0;
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++) {
        const float *q = r3d_tifxyz_at(&s, i, j);
        if (!r3d_tifxyz_valid(q)) continue;
        for (uint64_t c = 0; c < 3; c++) assert(fabs(vpos[k * 3 + c] - (double)q[c]) < 1e-3);
        k++;
      }
    assert(k == nvalid);
  }
  free(vpos);
  free(obj);
  { /* deterministic: a second run is byte-identical */
    snprintf(cmd, sizeof cmd, "'%s' '%s' '%s/m2.obj' --name testsheet > /dev/null",
             obj_exe, dir, tmp);
    assert(system(cmd) == 0);
    snprintf(p, sizeof p, "%s/m2.obj", tmp);
    size_t l2 = 0;
    char *o2 = slurp(p, &l2);
    snprintf(p, sizeof p, "%s/m.obj", tmp);
    size_t l1 = 0;
    char *o1 = slurp(p, &l1);
    assert(l1 == l2 && l1 == olen && !memcmp(o1, o2, l1));
    free(o1);
    free(o2);
  }

  /* ---------------- surfsamples (binary) ---------------- */
  snprintf(cmd, sizeof cmd,
           "'%s' '%s' '%s/s.bin' --spacing 2.0 --conf-min 0.25 > /dev/null", smp_exe,
           dir, tmp);
  assert(system(cmd) == 0);
  snprintf(p, sizeof p, "%s/s.bin", tmp);
  size_t blen = 0;
  uint8_t *bin = (uint8_t *)slurp(p, &blen);
  assert(blen >= 16 && !memcmp(bin, "R3DS", 4));
  uint32_t count = le_u32(bin + 4);
  assert(le_u32(bin + 8) == 0 && le_u32(bin + 12) == 0);
  assert(blen == 16 + (size_t)count * 28);
  assert(count > 0);
  { /* --spacing 2 at scale 0.05: ceil(20/2) = 10 subsamples per grid step
     * in each direction, half-open per quad except on the closing row and
     * column; a sample is kept when its bilinear confidence (the plane's
     * only variation is along i) is at least --conf-min.  Enumerate exactly
     * that rule and require the tool to agree. */
    uint64_t want = 0;
    for (uint32_t j = 0; j + 1 < H; j++)
      for (uint32_t i = 0; i + 1 < W; i++) {
        if (!(r3d_tifxyz_valid(r3d_tifxyz_at(&s, i, j)) &&
              r3d_tifxyz_valid(r3d_tifxyz_at(&s, i + 1, j)) &&
              r3d_tifxyz_valid(r3d_tifxyz_at(&s, i, j + 1)) &&
              r3d_tifxyz_valid(r3d_tifxyz_at(&s, i + 1, j + 1))))
          continue;
        uint32_t ulim = (i + 2 == W) ? 11u : 10u, vlim = (j + 2 == H) ? 11u : 10u;
        for (uint32_t iu = 0; iu < ulim; iu++) {
          double u = (double)iu / 10.0;
          double cf = (1 - u) * (double)gconf(i) + u * (double)gconf(i + 1);
          if (cf >= 0.25) want += vlim;
        }
      }
    assert((uint64_t)count == want);
  }
  {
    double bad_norm = 0;
    for (uint32_t k = 0; k < count; k++) {
      const uint8_t *r = bin + 16 + (size_t)k * 28;
      double x = le_f32(r), y = le_f32(r + 4), z = le_f32(r + 8);
      double nx = le_f32(r + 12), ny = le_f32(r + 16), nz = le_f32(r + 20);
      double cf = le_f32(r + 24);
      for (int c = 0; c < 3; c++) {
        double v = c == 0 ? x : (c == 1 ? y : z);
        assert(v >= (double)s.bbox[0][c] - 1e-2 && v <= (double)s.bbox[1][c] + 1e-2);
      }
      double len = sqrt(nx * nx + ny * ny + nz * nz);
      if (fabs(len - 1.0) > bad_norm) bad_norm = fabs(len - 1.0);
      assert(cf >= 0.25 - 1e-6 && cf <= 1.0 + 1e-6);
    }
    assert(bad_norm < 1e-5); /* unit normals */
  }
  { /* spacing sanity: consecutive samples within one quad row are ~2 vox
     * apart (the patch is nearly planar over a 20-voxel step) */
    double lo = 1e30, hi = 0;
    uint32_t n = count < 200 ? count : 200;
    for (uint32_t k = 1; k < n; k++) {
      const uint8_t *a = bin + 16 + (size_t)(k - 1) * 28, *b = a + 28;
      double d = 0;
      for (int c = 0; c < 3; c++) {
        double v = le_f32(b + 4 * c) - le_f32(a + 4 * c);
        d += v * v;
      }
      d = sqrt(d);
      if (d < 6.0) { /* skip the jump at the end of a subsample row */
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
    }
    assert(lo > 1.5 && hi < 2.6);
  }

  /* ---------------- surfsamples (CSV) agrees ---------------- */
  snprintf(cmd, sizeof cmd,
           "'%s' '%s' '%s/s.csv' --spacing 2.0 --conf-min 0.25 > /dev/null", smp_exe,
           dir, tmp);
  assert(system(cmd) == 0);
  snprintf(p, sizeof p, "%s/s.csv", tmp);
  size_t clen = 0;
  char *csv = slurp(p, &clen);
  {
    char *line = strtok(csv, "\n");
    assert(line && !strcmp(line, "x,y,z,nx,ny,nz,conf"));
    uint32_t k = 0;
    for (line = strtok(NULL, "\n"); line; line = strtok(NULL, "\n"), k++) {
      double v[7];
      assert(sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", v, v + 1, v + 2, v + 3, v + 4,
                    v + 5, v + 6) == 7);
      assert(k < count);
      const uint8_t *r = bin + 16 + (size_t)k * 28;
      for (int c = 0; c < 7; c++) assert(fabs(v[c] - le_f32(r + 4 * c)) < 1e-4);
    }
    assert(k == count);
  }
  free(csv);
  free(bin);

  /* a .sfc input works too (confidence falls back to 1 there) */
  {
    char sfc[700];
    snprintf(sfc, sizeof sfc, "%s/seg.sfc", tmp);
    assert(!r3d_surf_encode(dir, sfc, 0.1, NULL, NULL));
    snprintf(cmd, sizeof cmd, "'%s' '%s' '%s/sfc.obj' > /dev/null", obj_exe, sfc, tmp);
    assert(system(cmd) == 0);
    snprintf(cmd, sizeof cmd, "'%s' '%s' '%s/sfc.bin' --spacing 5 > /dev/null", smp_exe,
             sfc, tmp);
    assert(system(cmd) == 0);
    snprintf(p, sizeof p, "%s/sfc.bin", tmp);
    size_t n2 = 0;
    uint8_t *b2 = (uint8_t *)slurp(p, &n2);
    assert(n2 >= 16 && !memcmp(b2, "R3DS", 4) && le_u32(b2 + 4) > 0);
    for (uint32_t k = 0; k < le_u32(b2 + 4); k++)
      assert(fabs(le_f32(b2 + 16 + (size_t)k * 28 + 24) - 1.0) < 1e-6);
    free(b2);
  }

  /* bad usage is refused */
  snprintf(cmd, sizeof cmd, "'%s' '%s' > /dev/null 2>&1", obj_exe, dir);
  assert(system(cmd) != 0);
  snprintf(cmd, sizeof cmd, "'%s' '%s/nope' '%s/x.bin' > /dev/null 2>&1", smp_exe, tmp,
           tmp);
  assert(system(cmd) != 0);

  r3d_tifxyz_free(&s);
  char rm[800];
  snprintf(rm, sizeof rm, "rm -rf '%s'", tmp);
  assert(system(rm) == 0);
  printf("surfrefine export passed: obj %llu v / %llu f, samples %u\n",
         (unsigned long long)nvalid, (unsigned long long)(nquad * 2), count);
  return 0;
}
