/* tifxyz <-> .sfc conversion: streaming encode (strips and tiles, a mask
 * gating validity, exact auxiliary channels), whole-surface loads of both
 * kinds, the resolver's freshness/fallback rules, decode round trip and the
 * surfconv CLI when its path is given. */
#include "core/surface.h"
#include "core/surfconv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
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

#define W 150u
#define H 100u

static TIFF *tiff_begin(const char *path, uint32_t w, uint32_t h, int bps, int fmt,
                        int spp, int tiled) {
  TIFF *tf = TIFFOpen(path, "w");
  assert(tf);
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, bps);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, fmt);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, spp);
  TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  if (tiled) {
    TIFFSetField(tf, TIFFTAG_TILEWIDTH, 32);
    TIFFSetField(tf, TIFFTAG_TILELENGTH, 48);
  } else
    TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 7);
  return tf;
}

static void write_image(const char *path, uint32_t w, uint32_t h, int bps, int fmt,
                        int spp, const void *px, int tiled) {
  TIFF *tf = tiff_begin(path, w, h, bps, fmt, spp, tiled);
  size_t z = (size_t)(bps / 8) * (size_t)spp;
  if (tiled) {
    uint8_t *tile = calloc(32 * 48, z);
    for (uint32_t ty = 0; ty < h; ty += 48)
      for (uint32_t tx = 0; tx < w; tx += 32) {
        memset(tile, 0, 32 * 48 * z);
        for (uint32_t y = 0; y < 48 && ty + y < h; y++)
          for (uint32_t x = 0; x < 32 && tx + x < w; x++)
            memcpy(tile + (y * 32 + x) * z,
                   (const uint8_t *)px + ((size_t)(ty + y) * w + tx + x) * z, z);
        assert(TIFFWriteTile(tf, tile, tx, ty, 0, 0) > 0);
      }
    free(tile);
  } else
    for (uint32_t y = 0; y < h; y++)
      assert(TIFFWriteScanline(tf, (void *)((const uint8_t *)px + (size_t)y * w * z),
                               y, 0) >= 0);
  TIFFClose(tf);
}

static void *read_image(const char *path, uint32_t w, uint32_t h, int bps, int spp) {
  TIFF *tf = TIFFOpen(path, "r");
  assert(tf);
  uint32_t iw = 0, ih = 0;
  uint16_t ibps = 0, ispp = 0;
  TIFFGetField(tf, TIFFTAG_IMAGEWIDTH, &iw);
  TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &ih);
  TIFFGetField(tf, TIFFTAG_BITSPERSAMPLE, &ibps);
  TIFFGetFieldDefaulted(tf, TIFFTAG_SAMPLESPERPIXEL, &ispp);
  assert(iw == w && ih == h && ibps == bps && ispp == spp);
  size_t z = (size_t)(bps / 8) * (size_t)spp;
  uint8_t *out = malloc((size_t)w * h * z);
  assert(out);
  assert(!TIFFIsTiled(tf));
  for (uint32_t y = 0; y < h; y++)
    assert(TIFFReadScanline(tf, out + (size_t)y * w * z, y, 0) >= 0);
  TIFFClose(tf);
  return out;
}

static float gx(uint32_t x, uint32_t y) { return 1000.f + 3.1f * (float)x + 0.2f * (float)y; }
static float gy(uint32_t x, uint32_t y) { return 2000.f + 0.1f * (float)x + 2.9f * (float)y; }
static float gz(uint32_t x, uint32_t y) {
  return 500.f + 0.05f * (float)x * (float)y / 10.f + 4.f * sinf((float)x * 0.1f);
}
/* holes in the tifxyz itself (z = 0, the vc3d convention) */
static int hole(uint32_t x, uint32_t y) { return (x > 60 && x < 70 && y > 40 && y < 50) || (x + y) % 37 == 0; }
/* points the mask channel removes on top of that */
static int masked(uint32_t x, uint32_t y) { return x >= 140 && y < 10; }

static const char META[] =
    "{\n  \"format\": \"tifxyz\",\n  \"scale\": [\n    0.05,\n    0.05\n  ],\n"
    "  \"custom\": \"kept verbatim\"\n}\n";

static void make_tifxyz(const char *dir) {
  assert(!mkdir(dir, 0755));
  float *x = malloc(W * H * sizeof *x), *y = malloc(W * H * sizeof *y),
        *z = malloc(W * H * sizeof *z), *wind = malloc(W * H * sizeof *wind);
  uint8_t *mask = malloc(W * H);
  uint16_t *gen = malloc(W * H * sizeof *gen);
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      size_t k = (size_t)j * W + i;
      int h = hole(i, j);
      x[k] = h ? 0 : gx(i, j);
      y[k] = h ? 0 : gy(i, j);
      z[k] = h ? 0 : gz(i, j);
      wind[k] = h ? -1e30f : (float)i * 0.01f;
      mask[k] = masked(i, j) ? 0 : 255;
      gen[k] = (uint16_t)(i + j);
    }
  char p[600];
  snprintf(p, sizeof p, "%s/x.tif", dir);
  write_image(p, W, H, 32, SAMPLEFORMAT_IEEEFP, 1, x, 1); /* tiled plane */
  snprintf(p, sizeof p, "%s/y.tif", dir);
  write_image(p, W, H, 32, SAMPLEFORMAT_IEEEFP, 1, y, 0);
  snprintf(p, sizeof p, "%s/z.tif", dir);
  write_image(p, W, H, 32, SAMPLEFORMAT_IEEEFP, 1, z, 0);
  snprintf(p, sizeof p, "%s/winding.tif", dir);
  write_image(p, W, H, 32, SAMPLEFORMAT_IEEEFP, 1, wind, 0);
  snprintf(p, sizeof p, "%s/mask.tif", dir);
  write_image(p, W, H, 8, SAMPLEFORMAT_UINT, 1, mask, 0);
  snprintf(p, sizeof p, "%s/generations.tif", dir);
  write_image(p, W, H, 16, SAMPLEFORMAT_UINT, 1, gen, 1);
  snprintf(p, sizeof p, "%s/meta.json", dir);
  FILE *f = fopen(p, "wb");
  assert(f && fwrite(META, 1, strlen(META), f) == strlen(META));
  fclose(f);
  free(x);
  free(y);
  free(z);
  free(wind);
  free(mask);
  free(gen);
}

static int progress_calls;
static int progress(void *ud, uint64_t done, uint64_t total) {
  (void)ud;
  assert(done <= total && total > 0);
  progress_calls++;
  return 0;
}
static int cancel(void *ud, uint64_t done, uint64_t total) {
  (void)ud;
  (void)done;
  (void)total;
  return 1;
}

/* every grid point: same validity (tifxyz validity AND the mask), and valid
 * points within the Euclidean budget */
static void compare(const r3d_tifxyz *ref, const r3d_tifxyz *got, double budget,
                    int mask_applied) {
  assert(ref->w == got->w && ref->h == got->h);
  assert(ref->sx == got->sx && ref->sy == got->sy);
  uint64_t nvalid = 0;
  double worst = 0;
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      const float *a = r3d_tifxyz_at(ref, i, j), *b = r3d_tifxyz_at(got, i, j);
      int want = r3d_tifxyz_valid(a) && !(mask_applied && masked(i, j));
      assert(r3d_tifxyz_valid(b) == want);
      if (!want) {
        assert(b[0] == -1 && b[1] == -1 && b[2] == -1);
        continue;
      }
      nvalid++;
      double d = 0;
      for (int c = 0; c < 3; c++) d += ((double)a[c] - (double)b[c]) * ((double)a[c] - (double)b[c]);
      if (sqrt(d) > worst) worst = sqrt(d);
    }
  assert(got->nvalid == nvalid);
  assert(worst <= budget + 1e-4);
  for (int c = 0; c < 3; c++) {
    assert(fabs((double)got->bbox[0][c] - (double)ref->bbox[0][c]) <= budget + 1e-3);
    assert(fabs((double)got->bbox[1][c] - (double)ref->bbox[1][c]) <= budget + 1e-3);
  }
}

static ino_t inode(const char *p) {
  struct stat st;
  assert(!stat(p, &st));
  return st.st_ino;
}

int main(int argc, char **argv) {
  char tmp[] = "/tmp/r3d-surfconv-XXXXXX";
  assert(mkdtemp(tmp));
  char cli[4096] = "";
  if (argc > 1) assert(realpath(argv[1], cli)); /* survives the chdir below */
  assert(!chdir(tmp)); /* the read-only fallback lands in ./cache/sfc */
  char dir[600], sfc[600], out[2048], p[700];
  snprintf(dir, sizeof dir, "%s/seg.tifxyz", tmp);
  make_tifxyz(dir);
  assert(r3d_surf_kind_of(dir) == R3D_SURF_TIFXYZ);
  assert(r3d_surf_kind_of(tmp) == R3D_SURF_NONE);
  snprintf(p, sizeof p, "%s/x.tif", dir);
  assert(r3d_surf_kind_of(p) == R3D_SURF_NONE); /* a TIFF is not a .sfc */

  /* sibling naming */
  assert(!r3d_surf_sibling("a/b.tifxyz/", out, sizeof out) && !strcmp(out, "a/b.sfc"));
  assert(!r3d_surf_sibling("a/b", out, sizeof out) && !strcmp(out, "a/b.sfc"));
  assert(!r3d_surf_sibling("a/b.sfc", out, sizeof out) && !strcmp(out, "a/b.sfc"));
  assert(r3d_surf_sibling("a/b", out, 5));

  /* resolve encodes the sibling once */
  assert(!r3d_surf_resolve(dir, 0.1, progress, NULL, out, sizeof out));
  snprintf(sfc, sizeof sfc, "%s/seg.sfc", tmp);
  assert(!strcmp(out, sfc) && progress_calls > 0);
  assert(r3d_surf_kind_of(sfc) == R3D_SURF_SFC);
  ino_t first = inode(sfc);
  assert(!r3d_surf_resolve(dir, 0.1, NULL, NULL, out, sizeof out));
  assert(!strcmp(out, sfc) && inode(sfc) == first); /* fresh: reused */
  assert(!r3d_surf_resolve(sfc, 0.1, NULL, NULL, out, sizeof out) && !strcmp(out, sfc));
  { /* a newer plane invalidates the sibling */
    struct timeval tv[2];
    struct stat st;
    assert(!stat(sfc, &st));
    tv[0].tv_sec = tv[1].tv_sec = st.st_mtime + 5;
    tv[0].tv_usec = tv[1].tv_usec = 0;
    snprintf(p, sizeof p, "%s/z.tif", dir);
    assert(!utimes(p, tv));
    assert(!r3d_surf_resolve(dir, 0.1, NULL, NULL, out, sizeof out));
    assert(inode(sfc) != first);
    assert(!stat(sfc, &st) && st.st_mtime >= tv[0].tv_sec);
  }
  { /* no temporary survives a cancelled encode */
    snprintf(p, sizeof p, "%s/cancelled.sfc", tmp);
    assert(r3d_surf_encode(dir, p, 0.1, cancel, NULL) == -1);
    struct stat st;
    assert(stat(p, &st) != 0);
    char t2[700];
    snprintf(t2, sizeof t2, "%s.tmp.%ld", p, (long)getpid());
    assert(stat(t2, &st) != 0);
  }
  assert(r3d_surf_encode(dir, "/nonexistent-dir/x.sfc", 0.1, NULL, NULL) == -2);

  /* whole-surface loads agree with the tifxyz reader within the budget */
  r3d_tifxyz ref, got;
  assert(!r3d_tifxyz_load(&ref, dir));
  assert(!r3d_surf_load_sfc(sfc, &got));
  compare(&ref, &got, 0.1, 1);
  r3d_tifxyz_free(&got);
  assert(!r3d_surf_load(sfc, &got));
  compare(&ref, &got, 0.1, 1);
  r3d_tifxyz_free(&got);
  assert(!r3d_surf_load(dir, &got)); /* direct: no conversion, no mask */
  compare(&ref, &got, 0, 0);
  r3d_tifxyz_free(&got);
  assert(r3d_surf_load(tmp, &got));
  { /* the paged reader reads the same file */
    r3d_surface_reader *r = NULL;
    assert(!r3d_surface_open(sfc, 4096 * 12 * 4, &r));
    r3d_surface_info info;
    r3d_surface_get_info(r, &info);
    assert(info.width == W && info.height == H && info.sx == 0.05f);
    float q[3];
    assert(!r3d_surface_point(r, 100, 70, q));
    const float *a = r3d_tifxyz_at(&ref, 100, 70);
    double d = 0;
    for (int c = 0; c < 3; c++) d += ((double)a[c] - (double)q[c]) * ((double)a[c] - (double)q[c]);
    assert(sqrt(d) <= 0.1 + 1e-4);
    assert(!r3d_surface_point(r, 145, 5, q) && q[0] == -1); /* masked */
    r3d_surface_close(r);
  }

  /* decode round trip: coordinates within budget, aux channels and metadata
   * exact, an existing destination refused */
  char back[600];
  snprintf(back, sizeof back, "%s/back.tifxyz", tmp);
  progress_calls = 0;
  assert(!r3d_surf_decode(sfc, back, progress, NULL) && progress_calls > 0);
  assert(r3d_surf_decode(sfc, back, NULL, NULL));
  assert(r3d_surf_kind_of(back) == R3D_SURF_TIFXYZ);
  assert(!r3d_tifxyz_load(&got, back));
  compare(&ref, &got, 0.1, 1);
  r3d_tifxyz_free(&got);
  {
    snprintf(p, sizeof p, "%s/winding.tif", back);
    float *wind = read_image(p, W, H, 32, 1);
    snprintf(p, sizeof p, "%s/mask.tif", back);
    uint8_t *mask = read_image(p, W, H, 8, 1);
    snprintf(p, sizeof p, "%s/generations.tif", back);
    uint16_t *gen = read_image(p, W, H, 16, 1);
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++) {
        size_t k = (size_t)j * W + i;
        assert(wind[k] == (hole(i, j) ? -1e30f : (float)i * 0.01f));
        assert(mask[k] == (masked(i, j) ? 0 : 255));
        assert(gen[k] == (uint16_t)(i + j));
      }
    free(wind);
    free(mask);
    free(gen);
    snprintf(p, sizeof p, "%s/meta.json", back);
    FILE *f = fopen(p, "rb");
    char buf[512] = {0};
    assert(f);
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    assert(n == strlen(META) && !memcmp(buf, META, n));
  }
  { /* re-encoding the decoded directory stays within budget of the source */
    snprintf(p, sizeof p, "%s/back.sfc", tmp);
    assert(!r3d_surf_resolve(back, 0.1, NULL, NULL, out, sizeof out) && !strcmp(out, p));
    assert(!r3d_surf_load_sfc(out, &got));
    compare(&ref, &got, 0.2, 1);
    r3d_tifxyz_free(&got);
  }

  /* read-only source location: the encode lands under cache/sfc */
  if (geteuid() != 0) {
    char ro[600], rodir[600];
    snprintf(ro, sizeof ro, "%s/ro", tmp);
    snprintf(rodir, sizeof rodir, "%s/ro/seg.tifxyz", tmp);
    assert(!mkdir(ro, 0755));
    assert(!rename(dir, rodir));
    assert(!chmod(ro, 0555));
    assert(!r3d_surf_resolve(rodir, 0.1, NULL, NULL, out, sizeof out));
    assert(!strcmp(out, "cache/sfc/seg.sfc"));
    ino_t fb = inode(out);
    assert(!r3d_surf_resolve(rodir, 0.1, NULL, NULL, out, sizeof out));
    assert(inode(out) == fb);
    assert(!r3d_surf_load_sfc(out, &got));
    compare(&ref, &got, 0.1, 1);
    r3d_tifxyz_free(&got);
    assert(!chmod(ro, 0755));
    assert(!rename(rodir, dir));
    assert(!rmdir(ro));
  }

  if (argc > 1) { /* the CLI wraps the same library */
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "'%s' encode '%s' '%s/cli.sfc' --error 0.05", cli, dir, tmp);
    assert(system(cmd) == 0);
    snprintf(p, sizeof p, "%s/cli.sfc", tmp);
    assert(!r3d_surf_load_sfc(p, &got));
    compare(&ref, &got, 0.05, 1);
    r3d_tifxyz_free(&got);
    snprintf(cmd, sizeof cmd, "'%s' info '%s/cli.sfc' > /dev/null", cli, tmp);
    assert(system(cmd) == 0);
    snprintf(cmd, sizeof cmd, "'%s' decode '%s/cli.sfc' '%s/cli.tifxyz'", cli, tmp, tmp);
    assert(system(cmd) == 0);
    snprintf(cmd, sizeof cmd, "'%s' resolve '%s/cli.tifxyz' > '%s/resolved.txt'", cli, tmp,
             tmp);
    assert(system(cmd) == 0);
    snprintf(p, sizeof p, "%s/resolved.txt", tmp);
    FILE *f = fopen(p, "r");
    char line[2048] = {0};
    assert(f && fgets(line, sizeof line, f));
    fclose(f);
    line[strcspn(line, "\n")] = 0;
    snprintf(p, sizeof p, "%s/cli.sfc", tmp);
    assert(!strcmp(line, p));
    snprintf(cmd, sizeof cmd, "'%s' encode '%s' > /dev/null 2>&1", cli, tmp);
    assert(system(cmd) != 0); /* not a tifxyz directory */
  }
  r3d_tifxyz_free(&ref);
  assert(!chdir("/"));
  char rm[700];
  snprintf(rm, sizeof rm, "rm -rf '%s'", tmp);
  assert(system(rm) == 0);
  puts("surfconv passed: encode, load, resolve, decode, cli");
  return 0;
}
