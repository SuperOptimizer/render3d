/* rendseg: flatten a traced tifxyz segment to an image or a layered
 * surface volume. Surface positions are bilinearly upsampled `up`x and
 * sampled from a c5d LOD volume (optionally averaged over +-span along the
 * surface normal — vc3d's layered flattened view collapsed to one image).
 * Writes 8-bit grayscale PNG (or PGM when the output path ends in .pgm);
 * --layers N writes an N-layer surface volume (vc_render_tifxyz parity):
 * raw layer-major by default, or one PNG per layer with --stack png. Both
 * stack forms carry a sidecar JSON with everything a consumer needs to
 * register the stack against the segment grid and against masks/ink maps
 * (tools/ink9/raw2zarr.py converts the raw form to zarr). */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <zlib.h>

#include "core/cpuvol.h"
#include "core/tifxyz.h"
#include "core/umbilicus.h"

#include "core/pngw.h"
#define png_write_gray r3d_png_write_gray

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: rendseg <vol-root> <tifxyz-dir> <out.png|.pgm|.raw|dir> "
                    "[--level L] [--up N] [--span S] [--layers N] [--stack raw|png] "
                    "[--umbilicus F]\n"
                    "  --layers N: write an N-layer surface volume (offsets\n"
                    "  centered on the surface, 1 voxel apart along the normal)\n"
                    "  + a sidecar JSON with dims and registration metadata\n"
                    "  --stack raw: <out> is a raw u8 layer-major file (default)\n"
                    "  --stack png: <out> is a directory; layer_%%03u.png per\n"
                    "  layer + stack.json (raw2zarr.py converts raw to zarr)\n");
    return 2;
  }
  uint32_t level = 1, up = 8, layers = 0;
  double span = 0.0; /* +-span along the normal, averaged */
  const char *umbp = NULL;
  bool stack_png = false;
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--level") == 0 && i + 1 < argc)
      level = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--up") == 0 && i + 1 < argc)
      up = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--span") == 0 && i + 1 < argc)
      span = strtod(argv[++i], NULL);
    else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc)
      layers = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--umbilicus") == 0 && i + 1 < argc)
      umbp = argv[++i];
    else if (strcmp(argv[i], "--stack") == 0 && i + 1 < argc)
      stack_png = strcmp(argv[++i], "png") == 0;
  }
  r3d_tifxyz s;
  if (r3d_tifxyz_load(&s, argv[2]) != 0) return 1;
  /* with an umbilicus the layer normal is oriented AWAY from the scroll
   * core, so layer order (verso -> recto) is consistent across patches —
   * 2.5D ink models are trained on a fixed orientation */
  r3d_umbilicus umb;
  r3d_umbilicus_init(&umb);
  bool orient = umbp && r3d_umbilicus_load(&umb, umbp) == 0 && umb.count >= 1;
  r3d_cpuvol v;
  if (r3d_cpuvol_open(&v, argv[1], 512) != 0) {
    fprintf(stderr, "volume open failed\n");
    return 1;
  }
  uint32_t W = (s.w - 1) * up, H = (s.h - 1) * up;
  uint32_t nl = layers ? layers : 1;
  uint8_t *img = calloc((size_t)W * H * nl, 1);
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
      double n[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2],
                     du[0] * dv[1] - du[1] * dv[0]};
      double nrm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (nrm > 1e-9)
        for (int a = 0; a < 3; a++) n[a] /= nrm;
      else
        n[0] = n[1] = n[2] = 0;
      if (orient) { /* flip so n points radially outward at this slice */
        const r3d_umbilicus_point *ub = &umb.points[0];
        for (size_t q = 1; q < umb.count; q++)
          if (fabs(umb.points[q].z - p[2]) < fabs(ub->z - p[2])) ub = &umb.points[q];
        if (n[0] * (p[0] - ub->x) + n[1] * (p[1] - ub->y) < 0.0)
          for (int a = 0; a < 3; a++) n[a] = -n[a];
      }
      if (layers) { /* surface volume: one slice per voxel offset, layer
                     * (layers-1)/2 = the surface itself */
        for (uint32_t l = 0; l < nl; l++) {
          double o = (double)l - (double)(nl - 1) / 2.0;
          double q[3] = {p[0] + o * n[0], p[1] + o * n[1], p[2] + o * n[2]};
          double val = r3d_cpuvol_tri(&v, level, q, NULL);
          img[(size_t)l * W * H + (size_t)oy * W + ox] =
              (uint8_t)(val < 0 ? 0 : val > 255 ? 255 : val);
        }
        continue;
      }
      double acc = 0.0;
      int ns = 0;
      if (span > 0) {
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
  if (layers) {
    char jp[1400];
    if (stack_png) { /* <out> is a directory: one PNG per layer */
      if (mkdir(argv[3], 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "rendseg: mkdir %s failed\n", argv[3]);
        free(img);
        return 1;
      }
      rc2 = 0;
      for (uint32_t l = 0; l < nl && rc2 == 0; l++) {
        snprintf(jp, sizeof jp, "%s/layer_%03u.png", argv[3], l);
        rc2 = png_write_gray(jp, img + (size_t)l * W * H, W, H);
      }
      snprintf(jp, sizeof jp, "%s/stack.json", argv[3]);
    } else {
      FILE *f = fopen(argv[3], "wb");
      if (!f) {
        free(img);
        return 1;
      }
      rc2 = fwrite(img, 1, (size_t)W * H * nl, f) == (size_t)W * H * nl ? 0 : -1;
      if (fclose(f) != 0) rc2 = -1;
      int jn = snprintf(jp, sizeof jp, "%s.json", argv[3]);
      if (jn < 0 || (size_t)jn >= sizeof jp) rc2 = -1;
    }
    if (rc2 == 0) {
      /* registration metadata: enough for a consumer to map stack pixels
       * back to grid cells (pixel (x,y) <-> grid (x/up, y/up)) and to
       * validate a mask/ink map against the same surface (gw/gh/nvalid
       * mirror the ink-map staleness guard). Layer pitch is 1 voxel in
       * full-resolution space; intensities are read at LOD `level`. */
      FILE *jf = fopen(jp, "w");
      if (jf) {
        int jrc = fprintf(
            jf,
            "{\n  \"layers\": %u, \"height\": %u, \"width\": %u,\n"
            "  \"dtype\": \"u1\", \"order\": \"%s\", \"surface_layer\": %u,\n"
            "  \"up\": %u, \"grid_w\": %u, \"grid_h\": %u,\n"
            "  \"scale\": [%.9g, %.9g], \"level\": %u, \"layer_pitch_vox\": 1,\n"
            "  \"nvalid\": %llu,\n"
            "  \"bbox\": [[%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f]],\n"
            "  \"segment\": \"%s\", \"volume\": \"%s\",\n"
            "  \"umbilicus_oriented\": %s\n}\n",
            nl, H, W, stack_png ? "png-per-layer" : "layer-major", (nl - 1) / 2, up,
            s.w, s.h, (double)s.sx, (double)s.sy, level,
            (unsigned long long)s.nvalid, (double)s.bbox[0][0], (double)s.bbox[0][1],
            (double)s.bbox[0][2], (double)s.bbox[1][0], (double)s.bbox[1][1],
            (double)s.bbox[1][2], argv[2], argv[1], orient ? "true" : "false");
        if (jrc < 0 || fclose(jf) != 0) rc2 = -1;
      } else
        rc2 = -1;
    }
  } else if (ol > 4 && strcmp(argv[3] + ol - 4, ".pgm") == 0) {
    FILE *f = fopen(argv[3], "wb");
    if (!f) {
      free(img);
      return 1;
    }
    fprintf(f, "P5\n%u %u\n255\n", W, H);
    rc2 = fwrite(img, 1, (size_t)W * H, f) == (size_t)W * H ? 0 : -1;
    if (fclose(f) != 0) rc2 = -1;
  } else {
    rc2 = png_write_gray(argv[3], img, W, H);
  }
  free(img);
  if (rc2 != 0) {
    fprintf(stderr, "rendseg: write failed (%s)\n", argv[3]);
    return 1;
  }
  printf("wrote %s (%ux%u%s)\n", argv[3], W, H,
         layers ? (stack_png ? ", png stack" : ", raw stack") : "");
  r3d_umbilicus_free(&umb);
  r3d_tifxyz_free(&s);
  r3d_cpuvol_close(&v);
  return 0;
}
