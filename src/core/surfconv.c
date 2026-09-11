#include "core/surfconv.h"
#include "surfcomp.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <tiffio.h>
#include <unistd.h>

#define SC_MAX_FILES 64u
#define SC_META_MAX (64u << 20)
#define SC_PATH 2048

/* ---------------------------------------------------------------- planes */
/* One TIFF image read in horizontal bands of `rows` rows: strips go through
 * libtiff's own strip cache, tiles through a cached tile row so every tile
 * decodes once per pass. Samples stay in their native type. */
typedef struct sc_plane {
  TIFF *tf;
  char stem[64];
  uint32_t w, h;
  uint16_t spp;
  sfc_dtype dtype;
  size_t ss;                /* bytes per sample */
  int tiled;
  uint32_t tw, th;
  uint8_t *band;            /* rows x w x spp samples */
  uint32_t rows, y0;
  int have;
  uint8_t *tile, *tilerow;  /* tiled only: one tile, one row of tiles */
  uint32_t tilerow_y;
  int have_tilerow;
} sc_plane;

static void sc_plane_close(sc_plane *p) {
  if (p->tf) TIFFClose(p->tf);
  free(p->band);
  free(p->tile);
  free(p->tilerow);
  memset(p, 0, sizeof *p);
}

static int sc_plane_open(sc_plane *p, const char *path, const char *stem) {
  memset(p, 0, sizeof *p);
  p->tf = TIFFOpen(path, "r");
  if (!p->tf) {
    fprintf(stderr, "surfconv: cannot open %s\n", path);
    return -1;
  }
  snprintf(p->stem, sizeof p->stem, "%s", stem);
  uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG;
  TIFFGetField(p->tf, TIFFTAG_IMAGEWIDTH, &p->w);
  TIFFGetField(p->tf, TIFFTAG_IMAGELENGTH, &p->h);
  TIFFGetField(p->tf, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(p->tf, TIFFTAG_SAMPLEFORMAT, &fmt);
  TIFFGetFieldDefaulted(p->tf, TIFFTAG_SAMPLESPERPIXEL, &p->spp);
  TIFFGetFieldDefaulted(p->tf, TIFFTAG_PLANARCONFIG, &planar);
  if (bps == 8 && (fmt == SAMPLEFORMAT_UINT || fmt == SAMPLEFORMAT_VOID))
    p->dtype = SFC_U8;
  else if (bps == 16 && (fmt == SAMPLEFORMAT_UINT || fmt == SAMPLEFORMAT_VOID))
    p->dtype = SFC_U16;
  else if (bps == 32 && fmt == SAMPLEFORMAT_IEEEFP)
    p->dtype = SFC_F32;
  else {
    fprintf(stderr, "surfconv: %s: unsupported samples (%u bits, format %u)\n",
            path, bps, fmt);
    goto fail;
  }
  if (!p->w || !p->h || !p->spp || p->spp > 16 ||
      (p->spp > 1 && planar != PLANARCONFIG_CONTIG)) {
    fprintf(stderr, "surfconv: %s: unsupported layout (%ux%u, %u samples)\n",
            path, p->w, p->h, p->spp);
    goto fail;
  }
  p->ss = sfc_sample_size(p->dtype);
  p->tiled = TIFFIsTiled(p->tf);
  if (p->tiled) {
    TIFFGetField(p->tf, TIFFTAG_TILEWIDTH, &p->tw);
    TIFFGetField(p->tf, TIFFTAG_TILELENGTH, &p->th);
    if (!p->tw || !p->th) goto fail;
    p->tile = malloc((size_t)TIFFTileSize(p->tf));
    p->tilerow = malloc((size_t)p->th * p->w * p->spp * p->ss);
    if (!p->tile || !p->tilerow) goto fail;
  }
  return 0;
fail:
  sc_plane_close(p);
  return -1;
}

static int sc_plane_rows(sc_plane *p, uint32_t rows) {
  free(p->band);
  p->band = malloc((size_t)rows * p->w * p->spp * p->ss);
  p->rows = rows;
  p->have = 0;
  return p->band ? 0 : -1;
}

static int sc_tilerow(sc_plane *p, uint32_t ty) {
  if (p->have_tilerow && p->tilerow_y == ty) return 0;
  size_t px = p->spp * p->ss;
  for (uint32_t tx = 0; tx < p->w; tx += p->tw) {
    if (TIFFReadTile(p->tf, p->tile, tx, ty * p->th, 0, 0) < 0) return -1;
    uint32_t cw = p->tw < p->w - tx ? p->tw : p->w - tx;
    for (uint32_t r = 0; r < p->th; r++)
      memcpy(p->tilerow + ((size_t)r * p->w + tx) * px,
             p->tile + (size_t)r * p->tw * px, cw * px);
  }
  p->have_tilerow = 1;
  p->tilerow_y = ty;
  return 0;
}

/* Load rows [y0, y0 + rows); rows past the image are zero. */
static int sc_plane_band(sc_plane *p, uint32_t y0) {
  if (p->have && p->y0 == y0) return 0;
  size_t px = p->spp * p->ss, rw = (size_t)p->w * px;
  for (uint32_t r = 0; r < p->rows; r++) {
    uint64_t y = (uint64_t)y0 + r;
    uint8_t *dst = p->band + (size_t)r * rw;
    if (y >= p->h) {
      memset(dst, 0, rw);
      continue;
    }
    if (p->tiled) {
      if (sc_tilerow(p, (uint32_t)(y / p->th))) return -1;
      memcpy(dst, p->tilerow + (size_t)(y % p->th) * rw, rw);
    } else if (TIFFReadScanline(p->tf, dst, (uint32_t)y, 0) < 0)
      return -1;
  }
  p->have = 1;
  p->y0 = y0;
  return 0;
}

static inline const uint8_t *sc_at(const sc_plane *p, uint32_t x, uint32_t y,
                                   unsigned s) {
  return p->band + (((size_t)(y - p->y0) * p->w + x) * p->spp + s) * p->ss;
}

static double sc_num(const sc_plane *p, const uint8_t *v) {
  if (p->dtype == SFC_U8) return *v;
  if (p->dtype == SFC_U16) {
    uint16_t u;
    memcpy(&u, v, 2);
    return u;
  }
  float f;
  memcpy(&f, v, 4);
  return (double)f;
}

/* ------------------------------------------------------------ small utils */
static int sc_read_file(const char *path, uint8_t **out, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END)) {
    fclose(f);
    return -1;
  }
  long sz = ftell(f);
  if (sz < 0 || (unsigned long)sz > SC_META_MAX) {
    fclose(f);
    return -1;
  }
  rewind(f);
  *out = malloc((size_t)sz + 1);
  if (!*out) {
    fclose(f);
    return -1;
  }
  *n = (size_t)sz;
  int ok = fread(*out, 1, *n, f) == *n;
  fclose(f);
  (*out)[*n] = 0;
  if (!ok) {
    free(*out);
    *out = NULL;
  }
  return ok ? 0 : -1;
}

static int sc_cmp(const void *a, const void *b) {
  return strcmp(*(char *const *)a, *(char *const *)b);
}

static int sc_parse_scale(const char *meta, float *sx, float *sy) {
  const char *p = strstr(meta, "\"scale\"");
  double x = 0, y = 0;
  if (p) p = strchr(p, '[');
  if (!p || sscanf(p, "[ %lf , %lf", &x, &y) != 2 || !(x > 0) || !(y > 0) ||
      !isfinite(x) || !isfinite(y))
    return -1;
  *sx = (float)x;
  *sy = (float)y;
  return 0;
}

static int sc_progress(r3d_surf_progress cb, void *ud, uint64_t done,
                       uint64_t total) {
  return cb ? cb(ud, done, total) : 0;
}

r3d_surf_kind r3d_surf_kind_of(const char *path) {
  struct stat st;
  if (!path || stat(path, &st) != 0) return R3D_SURF_NONE;
  if (S_ISDIR(st.st_mode)) {
    char p[SC_PATH];
    snprintf(p, sizeof p, "%s/meta.json", path);
    return stat(p, &st) == 0 && S_ISREG(st.st_mode) ? R3D_SURF_TIFXYZ
                                                      : R3D_SURF_NONE;
  }
  if (!S_ISREG(st.st_mode)) return R3D_SURF_NONE;
  sfc_reader *r = NULL;
  if (sfc_open_file(path, &r)) return R3D_SURF_NONE;
  sfc_close(r);
  return R3D_SURF_SFC;
}

double r3d_surf_default_error(void) {
  const char *e = getenv("R3D_SFC_ERROR");
  double v = e ? strtod(e, NULL) : 0.0;
  return v > 0 && isfinite(v) ? v : 0.1;
}

/* ---------------------------------------------------------------- encode */
typedef struct sc_chan {
  sfc_channel c;
  unsigned src, sample; /* plane index, sample within the plane */
} sc_chan;

int r3d_surf_encode(const char *dir, const char *out, double error,
                    r3d_surf_progress cb, void *ud) {
  if (!dir || !out) return -1;
  if (!(error > 0) || !isfinite(error)) error = 0.1;
  int rc = -1;
  uint8_t *meta = NULL;
  size_t ml = 0;
  sc_plane *pl = calloc(SC_MAX_FILES, sizeof *pl);
  sc_chan *ch = calloc(SC_MAX_FILES * 16, sizeof *ch);
  sfc_channel *descs = NULL;
  sfc_writer *writer = NULL;
  unsigned np = 0, nc = 0;
  int xyz[3] = {-1, -1, -1}, mask = -1;
  float (*joint)[3] = NULL;
  uint8_t *data = NULL, valid[4096];
  char path[SC_PATH], tmp[SC_PATH];
  snprintf(tmp, sizeof tmp, "%s.tmp.%ld", out, (long)getpid());
  int tmp_made = 0;
  if (!pl || !ch) goto done;

  snprintf(path, sizeof path, "%s/meta.json", dir);
  if (sc_read_file(path, &meta, &ml)) {
    fprintf(stderr, "surfconv: %s: missing meta.json\n", dir);
    goto done;
  }
  { /* every *.tif in the directory, sorted for a stable channel order */
    DIR *d = opendir(dir);
    if (!d) goto done;
    char *names[SC_MAX_FILES];
    unsigned nn = 0;
    struct dirent *de;
    int overflow = 0;
    while ((de = readdir(d))) {
      size_t len = strlen(de->d_name);
      if (len <= 4 || strcmp(de->d_name + len - 4, ".tif")) continue;
      if (len - 4 >= 64 || nn == SC_MAX_FILES || !(names[nn] = strdup(de->d_name))) {
        overflow = 1;
        break;
      }
      nn++;
    }
    closedir(d);
    if (overflow) {
      for (unsigned i = 0; i < nn; i++) free(names[i]);
      fprintf(stderr, "surfconv: %s: too many or too long TIFF names\n", dir);
      goto done;
    }
    qsort(names, nn, sizeof *names, sc_cmp);
    int failed = 0;
    for (unsigned i = 0; i < nn; i++) {
      char stem[64];
      size_t len = strlen(names[i]) - 4;
      memcpy(stem, names[i], len);
      stem[len] = 0;
      snprintf(path, sizeof path, "%s/%s", dir, names[i]);
      free(names[i]);
      if (failed) continue;
      if (sc_plane_open(pl + np, path, stem)) {
        failed = 1;
        continue;
      }
      for (int a = 0; a < 3; a++)
        if (stem[0] == "xyz"[a] && !stem[1]) xyz[a] = (int)np;
      if (!strcmp(stem, "mask")) mask = (int)np;
      np++;
    }
    if (failed) goto done;
  }
  for (int a = 0; a < 3; a++)
    if (xyz[a] < 0 || pl[xyz[a]].spp != 1) {
      fprintf(stderr, "surfconv: %s: needs single-channel x/y/z.tif\n", dir);
      goto done;
    }
  const uint32_t W = pl[xyz[0]].w, H = pl[xyz[0]].h;
  for (int a = 1; a < 3; a++)
    if (pl[xyz[a]].w != W || pl[xyz[a]].h != H) {
      fprintf(stderr, "surfconv: %s: x/y/z dimensions differ\n", dir);
      goto done;
    }
  /* a mask at an integer multiple of the grid resolution gates validity
   * (all mapped mask pixels must be >= 255), as the reference converter */
  uint32_t msx = 1, msy = 1;
  if (mask >= 0) {
    sc_plane *m = pl + mask;
    if (m->spp == 1 && m->w >= W && m->h >= H && m->w % W == 0 && m->h % H == 0) {
      msx = m->w / W;
      msy = m->h / H;
    } else
      mask = -1;
  }
  /* channel table: joint XYZ triple first, then every other plane sample */
  for (int a = 0; a < 3; a++) {
    sc_chan *c = ch + nc++;
    c->src = (unsigned)xyz[a];
    c->sample = 0;
    c->c.name[0] = "xyz"[a];
    c->c.width = W;
    c->c.height = H;
    c->c.dtype = SFC_F32;
    c->c.flags = SFC_COORDINATE | SFC_XYZ;
    c->c.components = 3;
    c->c.component = (uint32_t)a;
    c->c.tolerance = error;
  }
  for (unsigned i = 0; i < np; i++) {
    if ((int)i == xyz[0] || (int)i == xyz[1] || (int)i == xyz[2]) continue;
    for (unsigned s = 0; s < pl[i].spp; s++) {
      sc_chan *c = ch + nc++;
      c->src = i;
      c->sample = s;
      if (pl[i].spp == 1)
        snprintf(c->c.name, sizeof c->c.name, "%s", pl[i].stem);
      else if (snprintf(c->c.name, sizeof c->c.name, "%s.%u", pl[i].stem, s) >=
               (int)sizeof c->c.name)
        goto done;
      c->c.width = pl[i].w;
      c->c.height = pl[i].h;
      c->c.dtype = pl[i].dtype;
      c->c.flags = SFC_EXACT;
      c->c.components = pl[i].spp;
      c->c.component = s;
      c->c.tolerance = 0;
    }
  }
  descs = malloc(nc * sizeof *descs);
  if (!descs) goto done;
  for (unsigned i = 0; i < nc; i++) descs[i] = ch[i].c;
  { /* an unwritable destination is reported apart from codec failures */
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
      rc = -2;
      goto done;
    }
    close(fd);
    unlink(tmp);
  }
  if (sfc_create(tmp, descs, nc, meta, ml, &writer)) {
    fprintf(stderr, "surfconv: cannot create %s\n", tmp);
    goto done;
  }
  tmp_made = 1;
  uint64_t nbx = (W + 63u) / 64u, nby = (H + 63u) / 64u, done_blocks = 0,
           total = nbx * nby;
  for (unsigned i = 3; i < nc; i++)
    total += ((ch[i].c.width + 63u) / 64u) * ((ch[i].c.height + 63u) / 64u);
  /* joint XYZ patches */
  for (int a = 0; a < 3; a++)
    if (sc_plane_rows(pl + xyz[a], 64)) goto done;
  if (mask >= 0 && sc_plane_rows(pl + mask, 64 * msy)) goto done;
  joint = malloc(4096 * sizeof *joint);
  data = malloc(4096 * 4);
  if (!joint || !data) goto done;
  for (uint64_t by = 0; by < nby; by++) {
    uint32_t y0 = (uint32_t)(by * 64);
    for (int a = 0; a < 3; a++)
      if (sc_plane_band(pl + xyz[a], y0)) goto done;
    if (mask >= 0 && sc_plane_band(pl + mask, y0 * msy)) goto done;
    for (uint64_t bx = 0; bx < nbx; bx++) {
      memset(joint, 0, 4096 * sizeof *joint);
      memset(valid, 0, sizeof valid);
      for (unsigned k = 0; k < 4096; k++) {
        uint64_t x = bx * 64 + k % 64, y = by * 64 + k / 64;
        if (x >= W || y >= H) continue;
        int ok = 1;
        for (unsigned a = 0; a < 3; a++) {
          const sc_plane *p = pl + xyz[a];
          float v = (float)sc_num(p, sc_at(p, (uint32_t)x, (uint32_t)y, 0));
          joint[k][a] = v;
          if (!isfinite(v) || (a == 2 && v <= 0)) ok = 0;
        }
        if (ok && mask >= 0) {
          const sc_plane *m = pl + mask;
          for (uint32_t yy = 0; ok && yy < msy; yy++)
            for (uint32_t xx = 0; xx < msx; xx++)
              if (!(sc_num(m, sc_at(m, (uint32_t)x * msx + xx,
                                    (uint32_t)y * msy + yy, 0)) >= 255)) {
                ok = 0;
                break;
              }
        }
        valid[k] = (uint8_t)ok;
        if (!ok) joint[k][0] = joint[k][1] = joint[k][2] = 0;
      }
      if (sfc_write_xyz_block(writer, joint, 12, 768, valid)) goto done;
    }
    done_blocks += nbx;
    if (sc_progress(cb, ud, done_blocks, total)) goto done;
  }
  for (int a = 0; a < 3; a++) sc_plane_close(pl + xyz[a]);
  /* exact auxiliary channels */
  for (unsigned i = 3; i < nc; i++) {
    sc_plane *p = pl + ch[i].src;
    if (!p->tf) goto done;
    if (!p->band || p->rows != 64) {
      if (sc_plane_rows(p, 64)) goto done;
    }
    size_t z = p->ss;
    uint64_t cbx = (p->w + 63u) / 64u, cby = (p->h + 63u) / 64u;
    for (uint64_t by = 0; by < cby; by++) {
      if (sc_plane_band(p, (uint32_t)(by * 64))) goto done;
      for (uint64_t bx = 0; bx < cbx; bx++) {
        memset(data, 0, 4096 * z);
        memset(valid, 0, sizeof valid);
        for (unsigned k = 0; k < 4096; k++) {
          uint64_t x = bx * 64 + k % 64, y = by * 64 + k / 64;
          if (x >= p->w || y >= p->h) continue;
          memcpy(data + k * z, sc_at(p, (uint32_t)x, (uint32_t)y, ch[i].sample), z);
          valid[k] = 1;
        }
        if (sfc_write_block(writer, data, z, 64 * z, valid)) goto done;
      }
      done_blocks += cbx;
      if (sc_progress(cb, ud, done_blocks, total)) goto done;
    }
    if (ch[i].sample + 1 == p->spp) sc_plane_close(p);
  }
  rc = sfc_finish(writer);
  writer = NULL;
  if (rc) {
    fprintf(stderr, "surfconv: finishing %s failed\n", tmp);
    rc = -1;
    tmp_made = 0; /* finish removes its output on failure */
  } else if (rename(tmp, out)) {
    fprintf(stderr, "surfconv: cannot publish %s: %s\n", out, strerror(errno));
    rc = -1;
  } else
    tmp_made = 0;
done:
  if (writer) sfc_cancel(writer);
  else if (tmp_made) unlink(tmp);
  free(joint);
  free(data);
  for (unsigned i = 0; i < np; i++) sc_plane_close(pl + i);
  free(pl);
  free(ch);
  free(descs);
  free(meta);
  return rc;
}

/* ---------------------------------------------------------------- decode */
static TIFF *sc_tiff_create(const char *path, uint32_t w, uint32_t h,
                            sfc_dtype dtype, uint16_t spp) {
  uint64_t bytes = (uint64_t)w * h * spp * sfc_sample_size(dtype);
  TIFF *tf = TIFFOpen(path, bytes > (UINT64_C(3) << 30) ? "w8" : "w");
  if (!tf) return NULL;
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, (uint16_t)(8 * sfc_sample_size(dtype)));
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT,
               dtype == SFC_F32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, spp);
  TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(tf, TIFFTAG_PREDICTOR,
               dtype == SFC_F32 ? PREDICTOR_FLOATINGPOINT : PREDICTOR_HORIZONTAL);
  TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 64);
  return tf;
}

static int sc_write_rows(TIFF *tf, uint8_t *band, uint32_t y0, uint32_t rows,
                         size_t rowbytes) {
  for (uint32_t r = 0; r < rows; r++)
    if (TIFFWriteScanline(tf, band + (size_t)r * rowbytes, y0 + r, 0) < 0)
      return -1;
  return 0;
}

static int sc_rm_tree(const char *dir) {
  DIR *d = opendir(dir);
  if (d) {
    struct dirent *de;
    while ((de = readdir(d))) {
      if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
      char p[SC_PATH];
      snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
      unlink(p);
    }
    closedir(d);
  }
  return rmdir(dir);
}

int r3d_surf_decode(const char *sfc, const char *dir, r3d_surf_progress cb,
                    void *ud) {
  if (!sfc || !dir) return -1;
  struct stat st;
  if (stat(dir, &st) == 0) {
    fprintf(stderr, "surfconv: %s exists\n", dir);
    return -1;
  }
  sfc_reader *r = NULL;
  if (sfc_open_file(sfc, &r)) {
    fprintf(stderr, "surfconv: cannot open %s\n", sfc);
    return -1;
  }
  int rc = -1, tmp_made = 0;
  char tmp[SC_PATH], path[SC_PATH];
  snprintf(tmp, sizeof tmp, "%s.tmp.%ld", dir, (long)getpid());
  TIFF *tf[3] = {0};
  uint8_t *band = NULL, *blk = NULL;
  char *meta = NULL;
  int chx[3];
  uint32_t ncs = sfc_channel_count(r);
  for (int a = 0; a < 3; a++) {
    char nm[2] = {"xyz"[a], 0};
    chx[a] = sfc_find_channel(r, nm);
    if (chx[a] < 0 || sfc_channel_info(r, (uint32_t)chx[a])->dtype != SFC_F32) {
      fprintf(stderr, "surfconv: %s has no float32 %s channel\n", sfc, nm);
      goto done;
    }
  }
  const sfc_channel *cx = sfc_channel_info(r, (uint32_t)chx[0]);
  uint64_t W = cx->width, H = cx->height;
  for (int a = 1; a < 3; a++) {
    const sfc_channel *c = sfc_channel_info(r, (uint32_t)chx[a]);
    if (c->width != W || c->height != H) goto done;
  }
  if (W > UINT32_MAX || H > UINT32_MAX) goto done;
  if (mkdir(tmp, 0755)) {
    fprintf(stderr, "surfconv: cannot create %s: %s\n", tmp, strerror(errno));
    goto done;
  }
  tmp_made = 1;
  uint64_t nbx = (W + 63) / 64, nby = (H + 63) / 64, done_blocks = 0,
           total = nbx * nby;
  for (uint32_t i = 0; i < ncs; i++) {
    const sfc_channel *c = sfc_channel_info(r, i);
    if ((int)i == chx[0] || (int)i == chx[1] || (int)i == chx[2]) continue;
    total += ((c->width + 63) / 64) * ((c->height + 63) / 64);
  }
  /* coordinate planes */
  size_t rowf = (size_t)W * sizeof(float);
  band = malloc(3 * 64 * rowf);
  blk = malloc(4096 * 16);
  if (!band || !blk) goto done;
  for (int a = 0; a < 3; a++) {
    snprintf(path, sizeof path, "%s/%c.tif", tmp, "xyz"[a]);
    tf[a] = sc_tiff_create(path, (uint32_t)W, (uint32_t)H, SFC_F32, 1);
    if (!tf[a]) goto done;
  }
  bool joint = chx[1] == chx[0] + 1 && chx[2] == chx[0] + 2;
  for (uint64_t by = 0; by < nby; by++) {
    uint32_t rows = (uint32_t)(H - by * 64 < 64 ? H - by * 64 : 64);
    for (uint64_t bx = 0; bx < nbx; bx++) {
      float *xyz = (float *)blk;
      uint8_t valid[4096], m[4096];
      memset(valid, 1, sizeof valid);
      if (joint) {
        if (sfc_read_xyz(r, (uint32_t)chx[0], bx, by, xyz, 12, 768, valid)) goto done;
      } else
        for (int a = 0; a < 3; a++) {
          if (sfc_read_block(r, (uint32_t)chx[a], bx, by, xyz + a, 12, 768, m))
            goto done;
          for (unsigned k = 0; k < 4096; k++) valid[k] &= m[k];
        }
      uint32_t cols = (uint32_t)(W - bx * 64 < 64 ? W - bx * 64 : 64);
      for (uint32_t y = 0; y < rows; y++)
        for (uint32_t x = 0; x < cols; x++) {
          unsigned k = y * 64 + x;
          for (int a = 0; a < 3; a++) {
            float v = valid[k] && xyz[k * 3 + 2] > 0 ? xyz[k * 3 + (unsigned)a] : -1.0f;
            memcpy(band + ((size_t)a * 64 + y) * rowf + (bx * 64 + x) * 4, &v, 4);
          }
        }
    }
    for (int a = 0; a < 3; a++)
      if (sc_write_rows(tf[a], band + (size_t)a * 64 * rowf, (uint32_t)(by * 64),
                        rows, rowf))
        goto done;
    done_blocks += nbx;
    if (sc_progress(cb, ud, done_blocks, total)) goto done;
  }
  for (int a = 0; a < 3; a++) {
    TIFFClose(tf[a]);
    tf[a] = NULL;
  }
  free(band);
  band = NULL;
  /* auxiliary channels: contiguous "stem.k" components rejoin one TIFF */
  for (uint32_t i = 0; i < ncs; i++) {
    const sfc_channel *c = sfc_channel_info(r, i);
    if ((int)i == chx[0] || (int)i == chx[1] || (int)i == chx[2]) continue;
    uint32_t spp = c->components > 1 ? c->components : 1;
    if (c->component != 0 && spp > 1) continue; /* handled with component 0 */
    if (spp > 16 || i + spp > ncs || c->width > UINT32_MAX || c->height > UINT32_MAX)
      goto done;
    char stem[64];
    snprintf(stem, sizeof stem, "%s", c->name);
    if (spp > 1) {
      char *dot = strrchr(stem, '.');
      if (dot) *dot = 0;
    }
    size_t z = sfc_sample_size(c->dtype), rowb = (size_t)c->width * spp * z;
    band = malloc(64 * rowb);
    if (!band) goto done;
    snprintf(path, sizeof path, "%s/%s.tif", tmp, stem);
    TIFF *t = sc_tiff_create(path, (uint32_t)c->width, (uint32_t)c->height,
                             c->dtype, (uint16_t)spp);
    if (!t) goto done;
    uint64_t cbx = (c->width + 63) / 64, cby = (c->height + 63) / 64;
    int ok = 1;
    for (uint64_t by = 0; ok && by < cby; by++) {
      uint32_t rows = (uint32_t)(c->height - by * 64 < 64 ? c->height - by * 64 : 64);
      memset(band, 0, 64 * rowb);
      for (uint64_t bx = 0; ok && bx < cbx; bx++) {
        uint32_t cols = (uint32_t)(c->width - bx * 64 < 64 ? c->width - bx * 64 : 64);
        for (uint32_t s = 0; ok && s < spp; s++) {
          uint8_t m[4096];
          if (sfc_read_block(r, i + (uint32_t)s, bx, by, blk, z, 64 * z, m)) {
            ok = 0;
            break;
          }
          for (uint32_t y = 0; y < rows; y++)
            for (uint32_t x = 0; x < cols; x++)
              memcpy(band + (size_t)y * rowb + ((bx * 64 + x) * spp + s) * z,
                     blk + ((size_t)y * 64 + x) * z, z);
        }
      }
      if (ok && sc_write_rows(t, band, (uint32_t)(by * 64), rows, rowb)) ok = 0;
      done_blocks += cbx;
      if (ok && sc_progress(cb, ud, done_blocks, total)) ok = 0;
    }
    TIFFClose(t);
    free(band);
    band = NULL;
    if (!ok) goto done;
    i += spp - 1;
  }
  { /* metadata bytes verbatim */
    size_t n = (size_t)sfc_metadata_size(r);
    meta = malloc(n + 1);
    if (!meta || sfc_read_metadata(r, meta, n)) goto done;
    snprintf(path, sizeof path, "%s/meta.json", tmp);
    FILE *f = fopen(path, "wb");
    if (!f) goto done;
    int ok = fwrite(meta, 1, n, f) == n;
    if (fclose(f) || !ok) goto done;
  }
  if (rename(tmp, dir)) {
    fprintf(stderr, "surfconv: cannot publish %s: %s\n", dir, strerror(errno));
    goto done;
  }
  tmp_made = 0;
  rc = 0;
done:
  for (int a = 0; a < 3; a++)
    if (tf[a]) TIFFClose(tf[a]);
  free(band);
  free(blk);
  free(meta);
  if (tmp_made) sc_rm_tree(tmp);
  sfc_close(r);
  return rc;
}

/* ------------------------------------------------------------------ load */
int r3d_surf_load_sfc(const char *path, r3d_tifxyz *out) {
  if (!path || !out) return -1;
  memset(out, 0, sizeof *out);
  sfc_reader *r = NULL;
  if (sfc_open_file(path, &r)) {
    fprintf(stderr, "surfconv: cannot open %s\n", path);
    return -1;
  }
  int rc = -1, chx[3];
  float *xyz = NULL, *blk = NULL;
  char *meta = NULL;
  for (int a = 0; a < 3; a++) {
    char nm[2] = {"xyz"[a], 0};
    chx[a] = sfc_find_channel(r, nm);
    if (chx[a] < 0 || sfc_channel_info(r, (uint32_t)chx[a])->dtype != SFC_F32)
      goto done;
  }
  const sfc_channel *cx = sfc_channel_info(r, (uint32_t)chx[0]);
  uint64_t W = cx->width, H = cx->height;
  for (int a = 1; a < 3; a++) {
    const sfc_channel *c = sfc_channel_info(r, (uint32_t)chx[a]);
    if (c->width != W || c->height != H) goto done;
  }
  if (!W || !H || W > UINT32_MAX || H > UINT32_MAX || W * H > (UINT64_C(1) << 31)) {
    fprintf(stderr, "surfconv: %s: %llux%llu is too large to hold in memory\n", path,
            (unsigned long long)W, (unsigned long long)H);
    goto done;
  }
  size_t n = (size_t)sfc_metadata_size(r);
  meta = malloc(n + 1);
  if (!meta || sfc_read_metadata(r, meta, n)) goto done;
  meta[n] = 0;
  float sx, sy;
  if (sc_parse_scale(meta, &sx, &sy)) {
    fprintf(stderr, "surfconv: %s: metadata lacks a valid \"scale\"\n", path);
    goto done;
  }
  xyz = malloc((size_t)W * H * 3 * sizeof *xyz);
  blk = malloc(4096 * 3 * sizeof *blk);
  if (!xyz || !blk) goto done;
  bool joint = chx[1] == chx[0] + 1 && chx[2] == chx[0] + 2;
  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  uint64_t nvalid = 0;
  for (uint64_t by = 0; by < (H + 63) / 64; by++)
    for (uint64_t bx = 0; bx < (W + 63) / 64; bx++) {
      uint8_t valid[4096], m[4096];
      memset(valid, 1, sizeof valid);
      if (joint) {
        if (sfc_read_xyz(r, (uint32_t)chx[0], bx, by, blk, 12, 768, valid)) goto done;
      } else
        for (int a = 0; a < 3; a++) {
          if (sfc_read_block(r, (uint32_t)chx[a], bx, by, blk + a, 12, 768, m)) goto done;
          for (unsigned k = 0; k < 4096; k++) valid[k] &= m[k];
        }
      uint32_t rows = (uint32_t)(H - by * 64 < 64 ? H - by * 64 : 64),
               cols = (uint32_t)(W - bx * 64 < 64 ? W - bx * 64 : 64);
      for (uint32_t y = 0; y < rows; y++)
        for (uint32_t x = 0; x < cols; x++) {
          unsigned k = y * 64 + x;
          float *d = xyz + (((by * 64 + y) * W) + bx * 64 + x) * 3;
          const float *s = blk + k * 3;
          if (!valid[k] || !(s[2] > 0) || !isfinite(s[0]) || !isfinite(s[1]) ||
              !isfinite(s[2])) {
            d[0] = d[1] = d[2] = -1.0f;
            continue;
          }
          for (int a = 0; a < 3; a++) {
            d[a] = s[a];
            if (s[a] < lo[a]) lo[a] = s[a];
            if (s[a] > hi[a]) hi[a] = s[a];
          }
          nvalid++;
        }
    }
  out->w = (uint32_t)W;
  out->h = (uint32_t)H;
  out->sx = sx;
  out->sy = sy;
  out->xyz = xyz;
  out->nvalid = nvalid;
  if (nvalid) {
    memcpy(out->bbox[0], lo, sizeof lo);
    memcpy(out->bbox[1], hi, sizeof hi);
  }
  xyz = NULL;
  rc = 0;
done:
  free(xyz);
  free(blk);
  free(meta);
  sfc_close(r);
  return rc;
}

int r3d_surf_load(const char *path, r3d_tifxyz *out) {
  switch (r3d_surf_kind_of(path)) {
  case R3D_SURF_SFC:
    return r3d_surf_load_sfc(path, out);
  case R3D_SURF_TIFXYZ:
    return r3d_tifxyz_load(out, path);
  default:
    fprintf(stderr, "surfconv: %s is neither a .sfc file nor a tifxyz directory\n",
            path ? path : "(null)");
    if (out) memset(out, 0, sizeof *out);
    return -1;
  }
}

/* --------------------------------------------------------------- resolve */
int r3d_surf_sibling(const char *path, char *out, size_t n) {
  if (!path || !out || !n) return -1;
  size_t len = strlen(path);
  while (len > 1 && path[len - 1] == '/') len--;
  if (len >= 4 && !strncmp(path + len - 4, ".sfc", 4)) {
    if (len >= n) return -1;
    memmove(out, path, len);
    out[len] = 0;
    return 0;
  }
  if (len >= 7 && !strncmp(path + len - 7, ".tifxyz", 7)) len -= 7;
  if (len + 5 > n) return -1;
  memmove(out, path, len);
  memcpy(out + len, ".sfc", 5);
  return 0;
}

static time_t sc_source_mtime(const char *dir) {
  static const char *comp[4] = {"x.tif", "y.tif", "z.tif", "meta.json"};
  time_t mx = 0;
  for (int i = 0; i < 4; i++) {
    char p[SC_PATH];
    snprintf(p, sizeof p, "%s/%s", dir, comp[i]);
    struct stat st;
    if (stat(p, &st) == 0 && st.st_mtime > mx) mx = st.st_mtime;
  }
  return mx;
}

/* A source stamped in the future (clock skew, copied timestamps) must not
 * force a re-encode on every open: the fresh .sfc is never older than it. */
static void sc_stamp(const char *sfc, time_t src) {
  struct stat st;
  if (stat(sfc, &st) != 0 || st.st_mtime >= src) return;
  struct timeval tv[2] = {{src, 0}, {src, 0}};
  (void)utimes(sfc, tv);
}

static bool sc_fresh(const char *sfc, time_t src) {
  struct stat st;
  return stat(sfc, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0 &&
         st.st_mtime >= src;
}

int r3d_surf_resolve(const char *path, double error, r3d_surf_progress cb,
                     void *ud, char *out, size_t n) {
  if (!path || !out || !n) return -1;
  r3d_surf_kind k = r3d_surf_kind_of(path);
  if (k == R3D_SURF_SFC) {
    if (strlen(path) >= n) return -1;
    snprintf(out, n, "%s", path);
    return 0;
  }
  if (k != R3D_SURF_TIFXYZ) return -1;
  time_t src = sc_source_mtime(path);
  char sib[SC_PATH];
  if (r3d_surf_sibling(path, sib, sizeof sib)) return -1;
  if (sc_fresh(sib, src)) {
    if (strlen(sib) >= n) return -1;
    snprintf(out, n, "%s", sib);
    return 0;
  }
  int rc = r3d_surf_encode(path, sib, error, cb, ud);
  if (rc == 0) {
    sc_stamp(sib, src);
    if (strlen(sib) >= n) return -1;
    snprintf(out, n, "%s", sib);
    return 0;
  }
  if (rc != -2) return -1;
  /* read-only source location: keep the encode under cache/sfc */
  char base[SC_PATH], fb[SC_PATH];
  size_t len = strlen(path);
  while (len > 1 && path[len - 1] == '/') len--;
  size_t start = len;
  while (start > 0 && path[start - 1] != '/') start--;
  if (len - start >= sizeof base) return -1;
  memcpy(base, path + start, len - start);
  base[len - start] = 0;
  mkdir("cache", 0755);
  mkdir("cache/sfc", 0755);
  snprintf(fb, sizeof fb, "cache/sfc/%s", base);
  if (r3d_surf_sibling(fb, fb, sizeof fb)) return -1;
  if (!sc_fresh(fb, src)) {
    if (r3d_surf_encode(path, fb, error, cb, ud)) return -1;
    sc_stamp(fb, src);
  }
  if (strlen(fb) >= n) return -1;
  snprintf(out, n, "%s", fb);
  return 0;
}
