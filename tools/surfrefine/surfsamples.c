/* surfsamples: dense oriented point samples from a render3d surface, the
 * input `export_tsm_labels.py` scatters into a TSM label store.
 *
 *   surfsamples <surface> <out.bin|out.csv> [--spacing 1.0] [--conf-min 0.25]
 *
 * The surface is a `.sfc` container or a tifxyz directory.  Each grid quad
 * whose four corners are valid is bilinearly subdivided so that consecutive
 * samples are about `--spacing` voxels apart: one grid step spans 1/sx
 * voxels (meta.json `scale`; a published grid at scale 0.05 is 20 voxels per
 * step, so 20x20 subsamples per quad give 1-voxel spacing).  Each sample
 * carries the position, the unit normal of the bilinear patch at that
 * parameter and a confidence.
 *
 * Normal orientation.  For the bilinear patch P(u,v) over the quad corners
 * a=(i,j) b=(i+1,j) d=(i,j+1) c=(i+1,j+1), with u along +i and v along +j,
 * the normal is  n = (dP/dv) x (dP/du)  normalised -- the "+v x +u"
 * convention.  It is the same sense as the triangle winding tifxyz2obj
 * writes (a,b,c),(a,c,d), so the OBJ mesh and these samples agree, and it is
 * consistent across the whole grid because it is defined purely from the
 * grid parameterisation, not from any per-quad choice.  Which physical side
 * of the sheet that is depends on the surface's own grid handedness; the
 * label exporter re-orients it against the scroll axis (label_store.md's
 * n . r_hat >= 0 rule) before using the sign.
 *
 * Confidence is the bilinear value of `<dir>/confidence.tif` when the input
 * is a tifxyz directory that has one, else 1.  (A `.sfc` keeps confidence as
 * an auxiliary channel for which there is no public whole-surface reader;
 * decode it with `surfconv decode` first if you need real confidences.)
 * Samples whose confidence is below --conf-min are dropped.
 *
 * Binary output (default, or when the name does not end in .csv):
 *   16-byte header: "R3DS", uint32 count, uint32 reserved, uint32 reserved
 *   count records of 7 little-endian float32: x y z nx ny nz conf
 * CSV output (name ends in .csv) has a `x,y,z,nx,ny,nz,conf` header line.
 *
 * Memory is bounded: the grid is streamed one row of quads at a time and
 * samples are written as they are produced (the header count is patched at
 * the end for a seekable file). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

#include "core/surfconv.h"

#define SAMP_MAX_SUB 4096u /* per-quad subdivision cap, both axes */

static int usage(void) {
  fprintf(stderr, "usage: surfsamples <surface> <out.bin|out.csv> [--spacing 1.0] "
                  "[--conf-min 0.25]\n");
  return EXIT_FAILURE;
}

/* Single-channel float32 TIFF plane, striped or tiled, of exactly w*h.
 * Returns NULL when the file is missing or does not match (confidence is
 * optional, so that is not an error). */
static float *read_plane(const char *path, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, "r");
  if (!tf) return NULL;
  uint32_t iw = 0, ih = 0;
  uint16_t bps = 0, fmt = 0, spp = 1;
  TIFFGetField(tf, TIFFTAG_IMAGEWIDTH, &iw);
  TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &ih);
  TIFFGetField(tf, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(tf, TIFFTAG_SAMPLEFORMAT, &fmt);
  TIFFGetFieldDefaulted(tf, TIFFTAG_SAMPLESPERPIXEL, &spp);
  if (iw != w || ih != h || bps != 32 || fmt != SAMPLEFORMAT_IEEEFP || spp != 1) {
    TIFFClose(tf);
    return NULL;
  }
  float *out = malloc((size_t)w * h * sizeof *out);
  if (!out) {
    TIFFClose(tf);
    return NULL;
  }
  int ok = 1;
  if (TIFFIsTiled(tf)) {
    uint32_t tw = 0, th = 0;
    TIFFGetField(tf, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(tf, TIFFTAG_TILELENGTH, &th);
    float *tile = (tw && th) ? malloc((size_t)tw * th * sizeof *tile) : NULL;
    if (!tile) ok = 0;
    for (uint32_t ty = 0; ok && ty < ih; ty += th)
      for (uint32_t tx = 0; ok && tx < iw; tx += tw) {
        if (TIFFReadTile(tf, tile, tx, ty, 0, 0) < 0) {
          ok = 0;
          break;
        }
        uint32_t cw = tw < iw - tx ? tw : iw - tx, ch = th < ih - ty ? th : ih - ty;
        for (uint32_t r = 0; r < ch; r++)
          memcpy(out + (size_t)(ty + r) * iw + tx, tile + (size_t)r * tw,
                 (size_t)cw * sizeof *out);
      }
    free(tile);
  } else {
    for (uint32_t r = 0; ok && r < ih; r++)
      if (TIFFReadScanline(tf, out + (size_t)r * iw, r, 0) < 0) ok = 0;
  }
  TIFFClose(tf);
  if (!ok) {
    free(out);
    return NULL;
  }
  return out;
}

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)((v >> 8) & 0xffu);
  p[2] = (uint8_t)((v >> 16) & 0xffu);
  p[3] = (uint8_t)((v >> 24) & 0xffu);
}
static void put_f32(uint8_t *p, double v) {
  float f = (float)v;
  uint32_t bits;
  memcpy(&bits, &f, sizeof bits);
  put_le32(p, bits);
}

int main(int argc, char **argv) {
  const char *pos[2] = {0};
  int npos = 0;
  double spacing = 1.0, conf_min = 0.25;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--spacing") && i + 1 < argc) {
      spacing = strtod(argv[++i], NULL);
      if (!(spacing > 0) || !isfinite(spacing)) {
        fprintf(stderr, "surfsamples: invalid --spacing\n");
        return EXIT_FAILURE;
      }
    } else if (!strcmp(argv[i], "--conf-min") && i + 1 < argc) {
      conf_min = strtod(argv[++i], NULL);
      if (!isfinite(conf_min)) {
        fprintf(stderr, "surfsamples: invalid --conf-min\n");
        return EXIT_FAILURE;
      }
    } else if (argv[i][0] == '-' && argv[i][1]) {
      return usage();
    } else if (npos < 2) {
      pos[npos++] = argv[i];
    } else
      return usage();
  }
  if (npos != 2) return usage();

  r3d_surf_kind kind = r3d_surf_kind_of(pos[0]);
  r3d_tifxyz s;
  if (kind == R3D_SURF_NONE || r3d_surf_load(pos[0], &s)) {
    fprintf(stderr, "surfsamples: cannot load %s\n", pos[0]);
    return EXIT_FAILURE;
  }
  if (s.w < 2 || s.h < 2) {
    fprintf(stderr, "surfsamples: grid %ux%u has no quads\n", s.w, s.h);
    r3d_tifxyz_free(&s);
    return EXIT_FAILURE;
  }
  float *conf = NULL;
  if (kind == R3D_SURF_TIFXYZ) {
    char p[2100];
    snprintf(p, sizeof p, "%s/confidence.tif", pos[0]);
    conf = read_plane(p, s.w, s.h);
  }

  /* subsamples per grid step: one step is 1/sx voxels of arc length */
  double stepx = (s.sx > 0 && isfinite((double)s.sx)) ? 1.0 / (double)s.sx : 1.0;
  double stepy = (s.sy > 0 && isfinite((double)s.sy)) ? 1.0 / (double)s.sy : 1.0;
  double nud = ceil(stepx / spacing), nvd = ceil(stepy / spacing);
  uint32_t nu = (nud < 1) ? 1u : (nud > (double)SAMP_MAX_SUB ? SAMP_MAX_SUB : (uint32_t)nud);
  uint32_t nv = (nvd < 1) ? 1u : (nvd > (double)SAMP_MAX_SUB ? SAMP_MAX_SUB : (uint32_t)nvd);

  size_t nlen = strlen(pos[1]);
  int csv = nlen >= 4 && !strcmp(pos[1] + nlen - 4, ".csv");
  FILE *f = fopen(pos[1], "wb");
  if (!f) {
    fprintf(stderr, "surfsamples: cannot write %s\n", pos[1]);
    free(conf);
    r3d_tifxyz_free(&s);
    return EXIT_FAILURE;
  }
  uint8_t hdr[16] = {'R', '3', 'D', 'S', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  if (csv) fputs("x,y,z,nx,ny,nz,conf\n", f);
  else fwrite(hdr, 1, sizeof hdr, f);

  uint64_t nsamp = 0, ndrop = 0, nquad = 0;
  for (uint32_t j = 0; j + 1 < s.h; j++) {
    for (uint32_t i = 0; i + 1 < s.w; i++) {
      const float *a = r3d_tifxyz_at(&s, i, j);         /* u=0 v=0 */
      const float *b = r3d_tifxyz_at(&s, i + 1, j);     /* u=1 v=0 */
      const float *d = r3d_tifxyz_at(&s, i, j + 1);     /* u=0 v=1 */
      const float *c = r3d_tifxyz_at(&s, i + 1, j + 1); /* u=1 v=1 */
      if (!r3d_tifxyz_valid(a) || !r3d_tifxyz_valid(b) || !r3d_tifxyz_valid(c) ||
          !r3d_tifxyz_valid(d))
        continue;
      nquad++;
      size_t ka = (size_t)j * s.w + i, kb = ka + 1, kd = ka + s.w, kc = kd + 1;
      /* half-open in u and v so neighbouring quads do not duplicate a seam;
       * the last row/column closes the surface */
      uint32_t ulim = (i + 2 == s.w) ? nu + 1 : nu;
      uint32_t vlim = (j + 2 == s.h) ? nv + 1 : nv;
      for (uint32_t jv = 0; jv < vlim; jv++) {
        double v = (double)jv / (double)nv;
        for (uint32_t iu = 0; iu < ulim; iu++) {
          double u = (double)iu / (double)nu;
          double p[3], du[3], dv[3];
          for (int k = 0; k < 3; k++) {
            double A = (double)a[k], B = (double)b[k], C = (double)c[k], D = (double)d[k];
            p[k] = (1 - u) * (1 - v) * A + u * (1 - v) * B + u * v * C + (1 - u) * v * D;
            du[k] = (1 - v) * (B - A) + v * (C - D);
            dv[k] = (1 - u) * (D - A) + u * (C - B);
          }
          /* +v x +u (see the header comment) */
          double n[3] = {dv[1] * du[2] - dv[2] * du[1], dv[2] * du[0] - dv[0] * du[2],
                         dv[0] * du[1] - dv[1] * du[0]};
          double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
          if (!(len > 0) || !isfinite(len)) {
            ndrop++;
            continue;
          }
          for (int k = 0; k < 3; k++) n[k] /= len;
          double cv = 1.0;
          if (conf)
            cv = (1 - u) * (1 - v) * (double)conf[ka] + u * (1 - v) * (double)conf[kb] +
                 u * v * (double)conf[kc] + (1 - u) * v * (double)conf[kd];
          if (!isfinite(cv)) cv = 0.0;
          if (cv < conf_min) {
            ndrop++;
            continue;
          }
          if (csv) {
            fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", p[0], p[1], p[2], n[0],
                    n[1], n[2], cv);
          } else {
            uint8_t rec[28];
            put_f32(rec + 0, p[0]);
            put_f32(rec + 4, p[1]);
            put_f32(rec + 8, p[2]);
            put_f32(rec + 12, n[0]);
            put_f32(rec + 16, n[1]);
            put_f32(rec + 20, n[2]);
            put_f32(rec + 24, cv);
            fwrite(rec, 1, sizeof rec, f);
          }
          nsamp++;
        }
      }
    }
  }
  int err = ferror(f) != 0;
  if (!csv && !err) {
    if (nsamp > 0xffffffffull) {
      fprintf(stderr, "surfsamples: %llu samples exceeds the uint32 count\n",
              (unsigned long long)nsamp);
      err = 1;
    } else {
      put_le32(hdr + 4, (uint32_t)nsamp);
      if (fseek(f, 0, SEEK_SET) || fwrite(hdr, 1, sizeof hdr, f) != sizeof hdr) err = 1;
    }
  }
  if (fclose(f)) err = 1;
  free(conf);
  double sxv = stepx / (double)nu, syv = stepy / (double)nv;
  r3d_tifxyz_free(&s);
  if (err) {
    fprintf(stderr, "surfsamples: write failed\n");
    return EXIT_FAILURE;
  }
  printf("%s: %llu samples from %llu quads (%ux%u per quad, ~%.3fx%.3f vox), "
         "%llu dropped, confidence %s\n",
         pos[1], (unsigned long long)nsamp, (unsigned long long)nquad, nu, nv, sxv, syv,
         (unsigned long long)ndrop, conf ? "from confidence.tif" : "assumed 1");
  return EXIT_SUCCESS;
}
