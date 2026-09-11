/* tifxyz2obj: export a render3d surface (.sfc or tifxyz directory) as a
 * Wavefront OBJ mesh, the form villa's `voxelize_objs.py` consumes.
 *
 *   tifxyz2obj <surface> <out.obj> [--name NAME]
 *
 * Every valid grid point becomes one `v` (coordinates are the voxel units of
 * the source volume, exactly as the tifxyz stores them), one `vt` (the grid
 * index normalised to [0,1]: i/(w-1), j/(h-1)) and one `vn` (the area-free
 * average of the face normals of the triangles that touch it).  Each grid
 * quad whose four corners are all valid emits two triangles, split along the
 * (i,j)-(i+1,j+1) diagonal:  (a,b,c) and (a,c,d) with a=(i,j) b=(i+1,j)
 * c=(i+1,j+1) d=(i,j+1).  Face normals use that same winding, so the mesh
 * normal is +v x +u (the grid's j direction crossed with its i direction),
 * matching surfsamples.
 *
 * Indices are 1-based into the compacted vertex list (invalid grid points
 * take no slot).  The output is deterministic: row-major grid order, fixed
 * precision, LF line endings. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/surfconv.h"

static int usage(void) {
  fprintf(stderr, "usage: tifxyz2obj <surface> <out.obj> [--name NAME]\n");
  return EXIT_FAILURE;
}

/* Accumulate the unit normal of triangle (a,b,c) into the three slots. */
static void accum_tri(double *nrm, uint64_t ia, uint64_t ib, uint64_t ic,
                      const float *a, const float *b, const float *c) {
  double u[3], v[3], n[3];
  for (int k = 0; k < 3; k++) {
    u[k] = (double)b[k] - (double)a[k];
    v[k] = (double)c[k] - (double)a[k];
  }
  n[0] = u[1] * v[2] - u[2] * v[1];
  n[1] = u[2] * v[0] - u[0] * v[2];
  n[2] = u[0] * v[1] - u[1] * v[0];
  double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (!(len > 0) || !isfinite(len)) return;
  for (int k = 0; k < 3; k++) n[k] /= len;
  const uint64_t idx[3] = {ia, ib, ic};
  for (int t = 0; t < 3; t++)
    for (int k = 0; k < 3; k++) nrm[idx[t] * 3 + (uint64_t)k] += n[k];
}

int main(int argc, char **argv) {
  const char *pos[2] = {0}, *name = "surface";
  int npos = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--name") && i + 1 < argc) {
      name = argv[++i];
    } else if (argv[i][0] == '-' && argv[i][1]) {
      return usage();
    } else if (npos < 2) {
      pos[npos++] = argv[i];
    } else
      return usage();
  }
  if (npos != 2) return usage();

  r3d_tifxyz s;
  if (r3d_surf_kind_of(pos[0]) == R3D_SURF_NONE || r3d_surf_load(pos[0], &s)) {
    fprintf(stderr, "tifxyz2obj: cannot load %s\n", pos[0]);
    return EXIT_FAILURE;
  }
  if (s.w < 1 || s.h < 1) {
    fprintf(stderr, "tifxyz2obj: empty grid\n");
    r3d_tifxyz_free(&s);
    return EXIT_FAILURE;
  }
  uint64_t n = (uint64_t)s.w * s.h;

  /* compaction map: grid cell -> 1-based OBJ vertex index, 0 when invalid */
  uint64_t *vid = calloc(n, sizeof *vid);
  double *nrm = calloc(n * 3, sizeof *nrm);
  if (!vid || !nrm) {
    fprintf(stderr, "tifxyz2obj: out of memory\n");
    free(vid);
    free(nrm);
    r3d_tifxyz_free(&s);
    return EXIT_FAILURE;
  }
  uint64_t nv = 0;
  for (uint32_t j = 0; j < s.h; j++)
    for (uint32_t i = 0; i < s.w; i++) {
      const float *p = r3d_tifxyz_at(&s, i, j);
      if (r3d_tifxyz_valid(p)) vid[(uint64_t)j * s.w + i] = ++nv;
    }

  /* pass 1: accumulate per-vertex normals from the quads we will emit */
  uint64_t nquad = 0;
  for (uint32_t j = 0; j + 1 < s.h; j++)
    for (uint32_t i = 0; i + 1 < s.w; i++) {
      uint64_t ka = (uint64_t)j * s.w + i, kb = ka + 1;
      uint64_t kd = ka + s.w, kc = kd + 1;
      if (!vid[ka] || !vid[kb] || !vid[kc] || !vid[kd]) continue;
      nquad++;
      const float *a = s.xyz + ka * 3, *b = s.xyz + kb * 3;
      const float *c = s.xyz + kc * 3, *d = s.xyz + kd * 3;
      accum_tri(nrm, ka, kb, kc, a, b, c);
      accum_tri(nrm, ka, kc, kd, a, c, d);
    }

  FILE *f = fopen(pos[1], "wb");
  if (!f) {
    fprintf(stderr, "tifxyz2obj: cannot write %s\n", pos[1]);
    free(vid);
    free(nrm);
    r3d_tifxyz_free(&s);
    return EXIT_FAILURE;
  }
  fprintf(f, "# tifxyz2obj %s\n# grid %ux%u scale %g %g\n", pos[0], s.w, s.h,
          (double)s.sx, (double)s.sy);
  double du = s.w > 1 ? 1.0 / (double)(s.w - 1) : 0.0;
  double dv = s.h > 1 ? 1.0 / (double)(s.h - 1) : 0.0;
  for (uint32_t j = 0; j < s.h; j++)
    for (uint32_t i = 0; i < s.w; i++) {
      uint64_t k = (uint64_t)j * s.w + i;
      if (!vid[k]) continue;
      const float *p = s.xyz + k * 3;
      fprintf(f, "v %.6f %.6f %.6f\n", (double)p[0], (double)p[1], (double)p[2]);
    }
  for (uint32_t j = 0; j < s.h; j++)
    for (uint32_t i = 0; i < s.w; i++) {
      if (!vid[(uint64_t)j * s.w + i]) continue;
      fprintf(f, "vt %.6f %.6f\n", (double)i * du, (double)j * dv);
    }
  for (uint32_t j = 0; j < s.h; j++)
    for (uint32_t i = 0; i < s.w; i++) {
      uint64_t k = (uint64_t)j * s.w + i;
      if (!vid[k]) continue;
      double *v = nrm + k * 3;
      double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      if (len > 0 && isfinite(len))
        fprintf(f, "vn %.6f %.6f %.6f\n", v[0] / len, v[1] / len, v[2] / len);
      else /* isolated vertex: no adjacent quad */
        fprintf(f, "vn 0.000000 0.000000 0.000000\n");
    }
  fprintf(f, "g %s\n", name);
  for (uint32_t j = 0; j + 1 < s.h; j++)
    for (uint32_t i = 0; i + 1 < s.w; i++) {
      uint64_t ka = (uint64_t)j * s.w + i, kb = ka + 1;
      uint64_t kd = ka + s.w, kc = kd + 1;
      if (!vid[ka] || !vid[kb] || !vid[kc] || !vid[kd]) continue;
      unsigned long long a = (unsigned long long)vid[ka], b = (unsigned long long)vid[kb];
      unsigned long long c = (unsigned long long)vid[kc], d = (unsigned long long)vid[kd];
      fprintf(f, "f %llu/%llu/%llu %llu/%llu/%llu %llu/%llu/%llu\n", a, a, a, b, b, b, c,
              c, c);
      fprintf(f, "f %llu/%llu/%llu %llu/%llu/%llu %llu/%llu/%llu\n", a, a, a, c, c, c, d,
              d, d);
    }
  int err = ferror(f) != 0;
  if (fclose(f)) err = 1;
  free(vid);
  free(nrm);
  r3d_tifxyz_free(&s);
  if (err) {
    fprintf(stderr, "tifxyz2obj: write failed\n");
    return EXIT_FAILURE;
  }
  printf("%s: %llu vertices, %llu faces (%llu quads)\n", pos[1],
         (unsigned long long)nv, (unsigned long long)(nquad * 2),
         (unsigned long long)nquad);
  return EXIT_SUCCESS;
}
