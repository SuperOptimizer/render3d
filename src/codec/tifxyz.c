#include "tifxyz.h"
#include "bytes.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <zlib.h>
/* Bound decoded allocations before touching untrusted compressed input. */
static size_t surface_bytes(uint32_t w, uint32_t h, size_t meta) {
  uint64_t n = (uint64_t)w * h;
  if (!w || !h || n > (UINT64_C(1) << 28) || meta > (1u << 26))
    return 0;
  return (size_t)n * 12 + meta;
}
void r3d_surface_free(r3d_surface_data *t) {
  if (!t)
    return;
  for (unsigned a = 0; a < 3; a++)
    free(t->plane[a]);
  free(t->meta);
  memset(t, 0, sizeof *t);
}
int r3d_surface_encode(const r3d_surface_data *t, int log2q, uint8_t **out,
                       size_t *n) {
  (void)log2q; /* Exact coordinates also satisfy every former quantization
                  bound. */
  if (!out || !n)
    return -1;
  *out = NULL;
  *n = 0;
  if (!t)
    return -1;
  size_t bytes = surface_bytes(t->w, t->h, t->meta_len),
         np = (size_t)t->w * t->h;
  if (!bytes || !t->plane[0] || !t->plane[1] || !t->plane[2] ||
      (t->meta_len && !t->meta))
    return -1;
  uint8_t *raw = malloc(bytes);
  if (!raw)
    return -1;
  for (size_t a = 0; a < 3; a++)
    for (size_t i = 0; i < np; i++) {
      uint32_t bits;
      memcpy(&bits, &t->plane[a][i], 4);
      r3c_put32(raw + 4 * (a * np + i), bits);
    }
  if (t->meta_len)
    memcpy(raw + 12 * np, t->meta, t->meta_len);
  uLongf cap = compressBound((uLong)bytes);
  uint8_t *buf = malloc((size_t)cap + 24);
  if (!buf) {
    free(raw);
    return -1;
  }
  memcpy(buf, "R3F1", 4);
  r3c_put32(buf + 4, t->w);
  r3c_put32(buf + 8, t->h);
  r3c_put32(buf + 12, 0);
  r3c_put64(buf + 16, t->meta_len);
  int rc = compress2(buf + 24, &cap, raw, (uLong)bytes, Z_BEST_SPEED);
  free(raw);
  if (rc != Z_OK) {
    free(buf);
    return -1;
  }
  *out = buf;
  *n = (size_t)cap + 24;
  return 0;
}
int r3d_surface_decode(const uint8_t *in, size_t n, r3d_surface_data *t) {
  if (!t)
    return -1;
  memset(t, 0, sizeof *t);
  if (!in || n <= 24 || memcmp(in, "R3F1", 4) || r3c_u64(in + 16) > SIZE_MAX)
    return -1;
  uint32_t w = r3c_u32(in + 4), h = r3c_u32(in + 8);
  size_t meta = (size_t)r3c_u64(in + 16);
  size_t bytes = surface_bytes(w, h, meta), np = (size_t)w * h;
  if (!bytes)
    return -1;
  uint8_t *raw = malloc(bytes);
  if (!raw)
    return -1;
  uLongf len = (uLong)bytes;
  uLong src = (uLong)(n - 24);
  if (uncompress2(raw, &len, in + 24, &src) != Z_OK || len != bytes ||
      src != n - 24) {
    free(raw);
    return -1;
  }
  t->w = w;
  t->h = h;
  t->meta_len = meta;
  for (size_t a = 0; a < 3; a++) {
    t->plane[a] = malloc(np * 4);
    if (!t->plane[a])
      goto fail;
    for (size_t i = 0; i < np; i++) {
      uint32_t bits = r3c_u32(raw + 4 * (a * np + i));
      memcpy(&t->plane[a][i], &bits, 4);
    }
  }
  if (meta) {
    t->meta = malloc(meta);
    if (!t->meta)
      goto fail;
    memcpy(t->meta, raw + 12 * np, meta);
  }
  free(raw);
  return 0;
fail:
  free(raw);
  r3d_surface_free(t);
  return -1;
}
int r3d_surface_load_dir(const char *dir, r3d_surface_data *t) {
  memset(t, 0, sizeof *t);
  char path[4096];
  for (unsigned a = 0; a < 3; a++) {
    if (snprintf(path, sizeof path, "%s/%c.tif", dir, "xyz"[a]) >=
        (int)sizeof path)
      goto fail;
    TIFF *f = TIFFOpen(path, "r");
    if (!f)
      goto fail;
    uint32_t w = 0, h = 0;
    uint16_t b = 0, s = 0, c = 0;
    TIFFGetField(f, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(f, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetFieldDefaulted(f, TIFFTAG_BITSPERSAMPLE, &b);
    TIFFGetFieldDefaulted(f, TIFFTAG_SAMPLEFORMAT, &s);
    TIFFGetFieldDefaulted(f, TIFFTAG_SAMPLESPERPIXEL, &c);
    if (!surface_bytes(w, h, 0) || b != 32 || s != SAMPLEFORMAT_IEEEFP ||
        c != 1 || (a && (w != t->w || h != t->h))) {
      TIFFClose(f);
      goto fail;
    }
    t->w = w;
    t->h = h;
    t->plane[a] = malloc((size_t)w * h * 4);
    if (!t->plane[a]) {
      TIFFClose(f);
      goto fail;
    }
    if (TIFFIsTiled(f)) {
      uint32_t tw = 0, th = 0;
      TIFFGetField(f, TIFFTAG_TILEWIDTH, &tw);
      TIFFGetField(f, TIFFTAG_TILELENGTH, &th);
      tmsize_t ts = TIFFTileSize(f);
      uint8_t *tile = ts > 0 ? malloc((size_t)ts) : NULL;
      if (!tile || !tw || !th || (uint64_t)tw * th * 4 > (uint64_t)ts) {
        free(tile);
        TIFFClose(f);
        goto fail;
      }
      for (uint32_t y = 0; y < h; y += th)
        for (uint32_t x = 0; x < w; x += tw) {
          if (TIFFReadTile(f, tile, x, y, 0, 0) < 0) {
            free(tile);
            TIFFClose(f);
            goto fail;
          }
          uint32_t nx = tw < w - x ? tw : w - x, ny = th < h - y ? th : h - y;
          for (uint32_t row = 0; row < ny; row++)
            memcpy(t->plane[a] + (size_t)(y + row) * w + x,
                   tile + (size_t)row * tw * 4, (size_t)nx * 4);
        }
      free(tile);
    } else {
      if (TIFFScanlineSize(f) != (tmsize_t)((size_t)w * 4)) {
        TIFFClose(f);
        goto fail;
      }
      for (uint32_t y = 0; y < h; y++)
        if (TIFFReadScanline(f, t->plane[a] + (size_t)y * w, y, 0) < 0) {
          TIFFClose(f);
          goto fail;
        }
    }
    TIFFClose(f);
  }
  if (snprintf(path, sizeof path, "%s/meta.json", dir) >= (int)sizeof path)
    goto fail;
  FILE *f = fopen(path, "rb");
  if (f) {
    if (fseek(f, 0, SEEK_END)) {
      fclose(f);
      goto fail;
    }
    long len = ftell(f);
    if (len < 0 || len > (1 << 26) || fseek(f, 0, SEEK_SET)) {
      fclose(f);
      goto fail;
    }
    t->meta_len = (size_t)len;
    t->meta = malloc(t->meta_len + 1);
    if (!t->meta || fread(t->meta, 1, t->meta_len, f) != t->meta_len) {
      fclose(f);
      goto fail;
    }
    fclose(f);
    t->meta[t->meta_len] = 0;
  }
  return 0;
fail:
  r3d_surface_free(t);
  return -1;
}
int r3d_surface_save_dir(const char *dir, const r3d_surface_data *t) {
  if (!t || !surface_bytes(t->w, t->h, t->meta_len) ||
      (mkdir(dir, 0755) != 0 && errno != EEXIST))
    return -1;
  char path[4096];
  for (unsigned a = 0; a < 3; a++) {
    if (!t->plane[a] || snprintf(path, sizeof path, "%s/%c.tif", dir,
                                 "xyz"[a]) >= (int)sizeof path)
      return -1;
    TIFF *f = TIFFOpen(path, "w8");
    if (!f)
      return -1;
    TIFFSetField(f, TIFFTAG_IMAGEWIDTH, t->w);
    TIFFSetField(f, TIFFTAG_IMAGELENGTH, t->h);
    TIFFSetField(f, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(f, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(f, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(f, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(f, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(f, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(f, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
    TIFFSetField(f, TIFFTAG_ROWSPERSTRIP, 32);
    for (uint32_t y = 0; y < t->h; y++)
      if (TIFFWriteScanline(f, t->plane[a] + (size_t)y * t->w, y, 0) < 0) {
        TIFFClose(f);
        return -1;
      }
    if (!TIFFWriteDirectory(f)) {
      TIFFClose(f);
      return -1;
    }
    TIFFClose(f);
  }
  if (snprintf(path, sizeof path, "%s/meta.json", dir) >= (int)sizeof path)
    return -1;
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  int rc = t->meta_len &&
           (!t->meta || fwrite(t->meta, 1, t->meta_len, f) != t->meta_len);
  if (fclose(f))
    rc = 1;
  return rc ? -1 : 0;
}
