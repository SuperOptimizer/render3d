#include "tifxyz.h"
#include "bytes.h"
#include <errno.h>
#include <stdbool.h>
#include <limits.h>
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
/* Stream planar little-endian floats through fixed scratch space. The view
 * stride permits direct interleaved headless I/O without full-volume copies. */
int r3d_surface_encode_strided(const r3d_surface_data *t, size_t stride,
                               uint8_t **out, size_t *n) {
  if (!out || !n) return -1;
  *out = NULL;
  *n = 0;
  if (!t || !stride) return -1;
  size_t bytes = surface_bytes(t->w, t->h, t->meta_len), np = (size_t)t->w * t->h;
  if (!bytes || !t->plane[0] || !t->plane[1] || !t->plane[2] ||
      (t->meta_len && !t->meta) || stride > SIZE_MAX / np / sizeof(float)) return -1;
  size_t capacity = (size_t)compressBound((uLong)bytes) + 24u;
  uint8_t *buf = malloc(capacity);
  if (!buf) return -1;
  memcpy(buf, "R3F1", 4);
  r3c_put32(buf + 4, t->w);
  r3c_put32(buf + 8, t->h);
  r3c_put32(buf + 12, 0);
  r3c_put64(buf + 16, t->meta_len);
  z_stream zs = {0};
  if (deflateInit(&zs, Z_BEST_SPEED) != Z_OK) { free(buf); return -1; }
  uint8_t scratch[65536];
  size_t pos = 0, written = 24;
  int rc = Z_OK;
  while (rc == Z_OK) {
    size_t count = bytes - pos;
    if (count > sizeof scratch) count = sizeof scratch;
    for (size_t k = 0; k < count;) {
      size_t at = pos + k;
      if (at < np * 12u) {
        size_t elem = at / 4u, axis = elem / np, first = elem % np;
        size_t take = (count - k) / 4u;
        if (take > np - first) take = np - first;
        for (size_t i = 0; i < take; i++) {
          uint32_t bits;
          memcpy(&bits, t->plane[axis] + (first + i) * stride, 4);
          r3c_put32(scratch + k + i * 4u, bits);
        }
        k += take * 4u;
      } else {
        memcpy(scratch + k, t->meta + at - np * 12u, count - k);
        break;
      }
    }
    pos += count;
    zs.next_in = scratch;
    zs.avail_in = (uInt)count;
    do {
      size_t avail = capacity - written;
      zs.next_out = buf + written;
      zs.avail_out = (uInt)(avail > UINT_MAX ? UINT_MAX : avail);
      uInt before = zs.avail_out;
      rc = deflate(&zs, pos == bytes ? Z_FINISH : Z_NO_FLUSH);
      written += before - zs.avail_out;
      if (!zs.avail_out && written == capacity && rc != Z_STREAM_END) {
        rc = Z_MEM_ERROR;
        break;
      }
    } while (rc == Z_OK && (zs.avail_in || pos == bytes));
  }
  deflateEnd(&zs);
  if (rc != Z_STREAM_END) { free(buf); return -1; }
  *out = buf;
  *n = written;
  return 0;
}
int r3d_surface_encode(const r3d_surface_data *t, int log2q, uint8_t **out, size_t *n) {
  (void)log2q; /* Exact coordinates satisfy every former quantization bound. */
  return r3d_surface_encode_strided(t, 1, out, n);
}
int r3d_surface_info(const uint8_t *in, size_t n, uint32_t *w, uint32_t *h, size_t *meta) {
  if (!in || !w || !h || !meta || n <= 24 || memcmp(in, "R3F1", 4) ||
      r3c_u64(in + 16) > SIZE_MAX)
    return -1;
  *w = r3c_u32(in + 4);
  *h = r3c_u32(in + 8);
  *meta = (size_t)r3c_u64(in + 16);
  return surface_bytes(*w, *h, *meta) ? 0 : -1;
}
int r3d_surface_decode_strided(const uint8_t *in, size_t n, r3d_surface_data *t,
                               size_t stride) {
  uint32_t w, h;
  size_t meta;
  if (!t || !stride || r3d_surface_info(in, n, &w, &h, &meta) != 0 ||
      t->w != w || t->h != h || t->meta_len != meta || !t->plane[0] ||
      !t->plane[1] || !t->plane[2] || (meta && !t->meta)) return -1;
  size_t np = (size_t)w * h, bytes = surface_bytes(w, h, meta);
  if (stride > SIZE_MAX / np / sizeof(float)) return -1;
  z_stream zs = {0};
  if (inflateInit(&zs) != Z_OK) return -1;
  uint8_t scratch[65536];
  size_t fed = 24, pos = 0;
  int rc = Z_OK;
  while (rc == Z_OK) {
    size_t count = bytes - pos;
    if (count > sizeof scratch) count = sizeof scratch;
    /* An extra output byte detects streams larger than the declared shape. */
    zs.next_out = scratch;
    zs.avail_out = (uInt)(count ? count : 1);
    uInt requested = zs.avail_out;
    while (zs.avail_out && rc == Z_OK) {
      if (!zs.avail_in && fed < n) {
        size_t avail = n - fed;
        zs.next_in = (Bytef *)in + fed;
        zs.avail_in = (uInt)(avail > UINT_MAX ? UINT_MAX : avail);
        fed += zs.avail_in;
      }
      rc = inflate(&zs, Z_NO_FLUSH);
    }
    size_t got = requested - zs.avail_out;
    if (got > count || (pos < np * 12u && got % 4u && pos + got < np * 12u)) {
      rc = Z_DATA_ERROR;
      break;
    }
    for (size_t k = 0; k < got;) {
      size_t at = pos + k;
      if (at < np * 12u) {
        if (got - k < 4u) { rc = Z_DATA_ERROR; break; }
        size_t elem = at / 4u, axis = elem / np, first = elem % np;
        size_t take = (got - k) / 4u;
        if (take > np - first) take = np - first;
        for (size_t i = 0; i < take; i++) {
          uint32_t bits = r3c_u32(scratch + k + i * 4u);
          memcpy(t->plane[axis] + (first + i) * stride, &bits, 4);
        }
        k += take * 4u;
      } else {
        memcpy(t->meta + at - np * 12u, scratch + k, got - k);
        break;
      }
    }
    pos += got;
  }
  bool valid = rc == Z_STREAM_END && pos == bytes && fed - zs.avail_in == n;
  inflateEnd(&zs);
  return valid ? 0 : -1;
}
int r3d_surface_decode(const uint8_t *in, size_t n, r3d_surface_data *t) {
  if (!t) return -1;
  memset(t, 0, sizeof *t);
  if (r3d_surface_info(in, n, &t->w, &t->h, &t->meta_len) != 0) return -1;
  size_t np = (size_t)t->w * t->h;
  for (unsigned a = 0; a < 3; a++) {
    t->plane[a] = malloc(np * sizeof(float));
    if (!t->plane[a]) goto fail;
  }
  if (t->meta_len) {
    t->meta = malloc(t->meta_len);
    if (!t->meta) goto fail;
  }
  if (r3d_surface_decode_strided(in, n, t, 1) != 0) goto fail;
  return 0;
fail:
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
