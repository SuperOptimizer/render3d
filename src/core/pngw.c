#include "core/pngw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

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

int r3d_png_write_gray(const char *path, const uint8_t *img, uint32_t w,
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
