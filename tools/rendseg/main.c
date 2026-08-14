/* rendseg: flatten a traced tifxyz segment to an image. Surface positions
 * are bilinearly upsampled `up`x and sampled from a c5d LOD volume
 * (optionally averaged over +-span along the surface normal — vc3d's
 * layered flattened view collapsed to one image). Writes 8-bit grayscale
 * PNG (or PGM when the output path ends in .pgm). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#include "core/cpuvol.h"
#include "core/tifxyz.h"

static const uint32_t png_crc_init = 0xFFFFFFFFu;
static uint32_t png_crc_step(uint32_t c, const uint8_t *d, size_t n) {
  for (size_t i = 0; i < n; i++) {
    c ^= d[i];
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return c;
}

static void png_be32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)(v >> 16);
  p[2] = (uint8_t)(v >> 8);
  p[3] = (uint8_t)v;
}

static int png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
  uint8_t hdr[8];
  png_be32(hdr, n);
  memcpy(hdr + 4, type, 4);
  uint32_t crc = png_crc_step(png_crc_init, hdr + 4, 4);
  if (data && n) crc = png_crc_step(crc, data, n);
  crc ^= 0xFFFFFFFFu;
  uint8_t tail[4];
  png_be32(tail, crc);
  return fwrite(hdr, 1, 8, f) == 8 && (!n || fwrite(data, 1, n, f) == n) &&
                 fwrite(tail, 1, 4, f) == 4
             ? 0
             : -1;
}

/* 8-bit grayscale PNG via zlib */
static int png_write_gray(const char *path, const uint8_t *img, uint32_t w,
                          uint32_t h) {
  size_t rown = (size_t)w + 1;
  uint8_t *raw = malloc(rown * h);
  if (!raw) return -1;
  for (uint32_t j = 0; j < h; j++) {
    raw[j * rown] = 0; /* filter: none */
    memcpy(raw + j * rown + 1, img + (size_t)j * w, w);
  }
  uLongf zcap = compressBound((uLong)(rown * h));
  uint8_t *z = malloc(zcap);
  if (!z || compress2(z, &zcap, raw, (uLong)(rown * h), 6) != Z_OK) {
    free(raw);
    free(z);
    return -1;
  }
  free(raw);
  FILE *f = fopen(path, "wb");
  int rc = -1;
  if (f) {
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    uint8_t ihdr[13];
    png_be32(ihdr, w);
    png_be32(ihdr + 4, h);
    ihdr[8] = 8;  /* bit depth */
    ihdr[9] = 0;  /* grayscale */
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    rc = fwrite(sig, 1, 8, f) == 8 && png_chunk(f, "IHDR", ihdr, 13) == 0 &&
                 png_chunk(f, "IDAT", z, (uint32_t)zcap) == 0 &&
                 png_chunk(f, "IEND", NULL, 0) == 0
             ? 0
             : -1;
    if (fclose(f) != 0) rc = -1;
  }
  free(z);
  return rc;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: rendseg <vol-root> <tifxyz-dir> <out.png|.pgm> "
                    "[--level L] [--up N] [--span S]\n");
    return 2;
  }
  uint32_t level = 1, up = 8;
  double span = 0.0; /* +-span along the normal, averaged */
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--level") == 0 && i + 1 < argc)
      level = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--up") == 0 && i + 1 < argc)
      up = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--span") == 0 && i + 1 < argc)
      span = strtod(argv[++i], NULL);
  }
  r3d_tifxyz s;
  if (r3d_tifxyz_load(&s, argv[2]) != 0) return 1;
  r3d_cpuvol v;
  if (r3d_cpuvol_open(&v, argv[1], 512) != 0) {
    fprintf(stderr, "volume open failed\n");
    return 1;
  }
  uint32_t W = (s.w - 1) * up, H = (s.h - 1) * up;
  uint8_t *img = calloc((size_t)W * H, 1);
  if (!img) return 1;
  for (uint32_t oy = 0; oy < H; oy++) {
    double gv = (double)oy / up;
    uint32_t j = (uint32_t)gv;
    double fv = gv - j;
    for (uint32_t ox = 0; ox < W; ox++) {
      double gu = (double)ox / up;
      uint32_t i = (uint32_t)gu;
      double fu = gu - i;
      const float *p00 = r3d_tifxyz_at(&s, i, j), *p10 = r3d_tifxyz_at(&s, i + 1, j);
      const float *p01 = r3d_tifxyz_at(&s, i, j + 1),
                  *p11 = r3d_tifxyz_at(&s, i + 1, j + 1);
      if (!r3d_tifxyz_valid(p00) || !r3d_tifxyz_valid(p10) ||
          !r3d_tifxyz_valid(p01) || !r3d_tifxyz_valid(p11))
        continue;
      double p[3], du[3], dv[3];
      for (int a = 0; a < 3; a++) {
        double q0 = (double)p00[a] * (1 - fu) + (double)p10[a] * fu;
        double q1 = (double)p01[a] * (1 - fu) + (double)p11[a] * fu;
        p[a] = q0 * (1 - fv) + q1 * fv;
        du[a] = ((double)p10[a] - (double)p00[a]) * (1 - fv) +
                ((double)p11[a] - (double)p01[a]) * fv;
        dv[a] = q1 - q0;
      }
      double acc = 0.0;
      int ns = 0;
      if (span > 0) {
        double n[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2],
                       du[0] * dv[1] - du[1] * dv[0]};
        double nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nl > 1e-9)
          for (int a = 0; a < 3; a++) n[a] /= nl;
        else
          n[0] = n[1] = n[2] = 0;
        for (double o = -span; o <= span + 1e-9; o += span > 1 ? span / 2 : span) {
          double q[3] = {p[0] + o * n[0], p[1] + o * n[1], p[2] + o * n[2]};
          acc += r3d_cpuvol_tri(&v, level, q, NULL);
          ns++;
        }
      } else {
        acc = r3d_cpuvol_tri(&v, level, p, NULL);
        ns = 1;
      }
      double val = acc / ns;
      img[(size_t)oy * W + ox] = (uint8_t)(val < 0 ? 0 : val > 255 ? 255 : val);
    }
    if (oy % 64 == 0) fprintf(stderr, "\rrow %u/%u", oy, H);
  }
  fprintf(stderr, "\n");
  size_t ol = strlen(argv[3]);
  int rc2;
  if (ol > 4 && strcmp(argv[3] + ol - 4, ".pgm") == 0) {
    FILE *f = fopen(argv[3], "wb");
    if (!f) return 1;
    fprintf(f, "P5\n%u %u\n255\n", W, H);
    rc2 = fwrite(img, 1, (size_t)W * H, f) == (size_t)W * H ? 0 : -1;
    if (fclose(f) != 0) rc2 = -1;
  } else {
    rc2 = png_write_gray(argv[3], img, W, H);
  }
  if (rc2 != 0) {
    fprintf(stderr, "rendseg: write failed (%s)\n", argv[3]);
    return 1;
  }
  printf("wrote %s (%ux%u)\n", argv[3], W, H);
  free(img);
  r3d_tifxyz_free(&s);
  r3d_cpuvol_close(&v);
  return 0;
}
