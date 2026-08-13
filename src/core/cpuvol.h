/* CPU-side sampler over a render3d c5d LOD tree (manifest.json + shard
 * files under c5d/L<l>, plus net-ingested .c5b files under bricks/L<l>)
 * with a brick-granular decode LRU. For worker-thread consumers — the
 * surface tracer reads prediction volumes through this; the renderer
 * keeps its own GPU path. Values are u8; no-data voxels read as 0. */
#ifndef R3D_CPUVOL_H
#define R3D_CPUVOL_H

#include <stdbool.h>
#include <stdint.h>

#define R3D_CPUVOL_LEVELS 8u

typedef struct r3d_cpuvol_level {
  uint32_t scale;          /* 1 << level */
  uint32_t vx, vy, vz;     /* level shape, voxels */
  uint32_t bx, by, bz;     /* brick grid dims */
  uint32_t sx, sy, sz;     /* shard grid dims */
  uint32_t shard_off;      /* first reader index */
} r3d_cpuvol_level;

typedef struct r3d_cpuvol {
  char root[1024];
  uint32_t nlev;
  uint64_t nx, ny, nz; /* base shape, voxels */
  r3d_cpuvol_level lev[R3D_CPUVOL_LEVELS];
  void *readers;   /* lazy shard readers */
  uint32_t nreaders;
  /* decode cache: slots of 128^3 bricks keyed (level, bx, by, bz) */
  uint8_t *slabs;
  uint64_t *keys;    /* key or UINT64_MAX */
  uint64_t *use;     /* LRU ticks */
  uint32_t nslots;
  uint64_t tick;
} r3d_cpuvol;

int r3d_cpuvol_open(r3d_cpuvol *v, const char *root, uint32_t cache_bricks);
void r3d_cpuvol_close(r3d_cpuvol *v);

/* Nearest-neighbor value at base-resolution voxel coords, sampled from
 * pyramid level li. Out-of-bounds or no-data reads 0. */
uint8_t r3d_cpuvol_at(r3d_cpuvol *v, uint32_t li, double x, double y, double z);

/* Trilinear value at base-resolution voxel coords sampled from level li,
 * 0..255 continuous; optionally its analytic gradient d value / d base
 * voxel (the trilinear surface is piecewise differentiable — solvers need
 * this the way vc3d gets interpolator derivatives from autodiff). */
double r3d_cpuvol_tri(r3d_cpuvol *v, uint32_t li, const double p[3], double grad[3]);

#endif /* R3D_CPUVOL_H */
