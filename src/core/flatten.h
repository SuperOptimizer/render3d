/* SLIM re-flattening for traced segments (Rabinovich et al. 2017, grid-
 * specialized, dependency-free). A traced tifxyz grid is already a
 * parameterization, but not an isometric one: the flattened view stretches
 * where the tracer's grid spacing drifted from true surface distance. This
 * recomputes a 2D embedding that minimizes the symmetric Dirichlet energy
 * of the map from each quad's REAL 3D shape to the plane — the same
 * energy/iteration as libigl's SLIM, but exploiting the regular grid: the
 * mesh is implicit (two triangles per valid quad), rest shapes come from
 * 3D edge lengths, and the reweighted least-squares system is solved
 * matrix-free with Jacobi-preconditioned conjugate gradients.
 *
 * Pipeline: r3d_flatten_slim computes per-vertex UV (voxel units, free
 * boundary, flip-free line search); r3d_flatten_resample then rasterizes
 * the surface onto a regular UV lattice, producing a new grid whose cells
 * ARE near-isometric — saved as an ordinary tifxyz, every downstream
 * consumer (flat view, ink, stacks, masks) benefits unchanged. */
#ifndef R3D_FLATTEN_H
#define R3D_FLATTEN_H

#include <stdbool.h>
#include <stdint.h>

typedef struct r3d_flatten_stats {
  uint32_t iters;        /* SLIM iterations run */
  uint32_t nvert, ntri;  /* mesh size (valid grid points / triangles) */
  double e0, e1;         /* symmetric Dirichlet energy per triangle,
                          * initial and final (2.0 = perfect isometry) */
  double stretch0, stretch1; /* mean |singular value - 1| before/after */
} r3d_flatten_stats;

/* xyz: w*h*3 grid positions, invalid = x < 0 (tifxyz convention).
 * step: target UV pitch in voxels (the grid's nominal spacing, 1/scale).
 * uv (caller-allocated, w*h*2): filled for valid vertices; invalid cells
 * get 1e30f. Returns 0 on success (needs >= 1 valid quad). */
int r3d_flatten_slim(const float *xyz, uint32_t w, uint32_t h, double step,
                     uint32_t max_iters, float *uv, r3d_flatten_stats *st);

/* Resample the surface over the UV embedding onto a regular lattice of
 * `step`-voxel pitch: rasterizes every UV triangle, barycentrically
 * interpolating 3D positions. *out_xyz (malloc'd, *ow * *oh * 3) uses the
 * tifxyz invalid convention for uncovered lattice points. */
int r3d_flatten_resample(const float *xyz, const float *uv, uint32_t w, uint32_t h,
                         double step, float **out_xyz, uint32_t *ow, uint32_t *oh);

#endif /* R3D_FLATTEN_H */
