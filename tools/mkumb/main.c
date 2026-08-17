/* Estimate a scroll's umbilicus (winding axis) from the masked CT: the
 * per-slice intensity centroid of scroll material at a coarse pyramid
 * level. Masked exports read 0 outside the scroll, so the centroid tracks
 * the roll's core to first order — exactly the frame the tracer's spiral
 * prior, winding numbers, wrap gates, and signed spacing need, without
 * hand annotation. Smoothed with a moving average; written in the Villa
 * control-point JSON the whole toolchain consumes.
 *
 *   mkumb <ct-lod-root> <out.json> [--step 256] [--th 24]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/cpuvol.h"
#include "core/umbilicus.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: mkumb <ct-lod-root> <out.json> [--step N] [--th V]\n");
    return 2;
  }
  const char *root = argv[1], *out = argv[2];
  double zstep = 256.0, th = 24.0;
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) zstep = atof(argv[++i]);
    else if (strcmp(argv[i], "--th") == 0 && i + 1 < argc) th = atof(argv[++i]);
  }
  r3d_cpuvol v;
  if (r3d_cpuvol_open(&v, root, 256) != 0) {
    fprintf(stderr, "mkumb: cannot open %s\n", root);
    return 1;
  }
  /* coarsest level with usable resolution (>= ~128 px across) */
  uint32_t li = v.nlev ? v.nlev - 1 : 0;
  while (li > 0 && (double)v.nx / v.lev[li].scale < 128.0) li--;
  double sc = v.lev[li].scale;
  double pitch = sc; /* one sample per coarse voxel */
  printf("mkumb: %s L%u (scale %.0f), slice step %.0f, threshold %.0f\n", root,
         li, sc, zstep, th);
  enum { MAXP = 512 };
  double px[MAXP], py[MAXP], pz[MAXP];
  int np = 0;
  for (double wz = zstep * 0.5; wz < (double)v.nz && np < MAXP; wz += zstep) {
    double sx = 0.0, sy = 0.0, sw = 0.0;
    for (double wy = pitch * 0.5; wy < (double)v.ny; wy += pitch)
      for (double wx = pitch * 0.5; wx < (double)v.nx; wx += pitch) {
        double p[3] = {wx, wy, wz};
        double val = r3d_cpuvol_tri(&v, li, p, NULL);
        if (val <= th) continue;
        sx += wx * val;
        sy += wy * val;
        sw += val;
      }
    if (sw < th * 16.0) continue; /* empty / off-scroll slice */
    px[np] = sx / sw;
    py[np] = sy / sw;
    pz[np] = wz;
    np++;
  }
  if (np < 2) {
    fprintf(stderr, "mkumb: found scroll material in only %d slices\n", np);
    r3d_cpuvol_close(&v);
    return 1;
  }
  /* moving-average smooth (the centroid wobbles with damage/void slices) */
  r3d_umbilicus u;
  r3d_umbilicus_init(&u);
  for (int i = 0; i < np; i++) {
    double ax = 0.0, ay = 0.0;
    int n = 0;
    for (int k = i - 2; k <= i + 2; k++) {
      if (k < 0 || k >= np) continue;
      ax += px[k];
      ay += py[k];
      n++;
    }
    r3d_umbilicus_set(&u, ax / n, ay / n, pz[i]);
  }
  int rc = r3d_umbilicus_save(&u, out, "mkumb-centroid", (uint32_t)v.nz,
                              (uint32_t)v.ny, (uint32_t)v.nx);
  printf("mkumb: %zu control points -> %s (x %.0f..%.0f, y %.0f..%.0f)\n",
         u.count, out, px[0], px[np - 1], py[0], py[np - 1]);
  r3d_umbilicus_free(&u);
  r3d_cpuvol_close(&v);
  return rc == 0 ? 0 : 1;
}
