/* CPU-side sampler over a render3d c5d LOD tree (manifest.json + shard
 * files under c5d/L<l>, plus net-ingested .c5b files under bricks/L<l>)
 * with a brick-granular decode LRU. For worker-thread consumers — the
 * surface tracer reads prediction volumes through this; the renderer
 * keeps its own GPU path. Values are u8; no-data voxels read as 0. */
#ifndef R3D_CPUVOL_H
#define R3D_CPUVOL_H

#include <pthread.h>
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
  /* decode cache: refcounted slab pool of 128^3 bricks keyed
   * (level, bx, by, bz), pinned per reader lease. Opaque (cv_cache) and
   * separately refcounted so a lease outlives r3d_cpuvol_close. */
  void *cache;
  /* unique per successful open: a thread-local memo taken against a volume
   * that was closed and reopened at the same address must not validate */
  uint64_t id;
  /* negative cache: bricks known absent (empty cache file = air, permanent)
   * or unavailable right now (fetch/decode failed: expires) — trilinear
   * taps into empty space no longer cost 8 file probes per sample */
  uint64_t *neg_key;
  uint64_t *neg_exp;  /* expiry, seconds since epoch; UINT64_MAX = permanent */
  uint32_t nneg;      /* power of two */
  /* thread safety: the decode cache carries its own lock and per-slot pin
   * counts; mu guards the negative cache and the net backoff; io_mu
   * serializes shard reads + demand fetches. Decodes run outside all of
   * them. Concurrent r3d_cpuvol_tri/at/read_block callers on one volume are
   * supported and never observe a slot that eviction may rewrite: every
   * returned brick pointer is pinned until that thread asks for another
   * brick. The volume object itself is NOT concurrently closable — callers
   * must join their samplers before r3d_cpuvol_close (leases outstanding at
   * that point stay valid, but the r3d_cpuvol is gone). */
  pthread_mutex_t mu, io_mu;
  /* demand fetch (source.json): brick misses pull the owning zarr cell,
   * transcode, and land in <root>/bricks/L* — the same cache the
   * renderer's net ingest fills, so either side feeds the other. The
   * tracer must not go data-blind at its frontier just because the
   * viewer never looked there. */
  char url[1400];    /* empty = no net source */
  float q0;          /* c5d quality ladder base */
  uint32_t chsz[R3D_CPUVOL_LEVELS];
  bool raw[R3D_CPUVOL_LEVELS];
  void *curl;        /* lazy CURL handle */
  uint64_t net_cool; /* backoff: no fetches until this tick-time (s) */
  /* predict source (url predict://): misses are produced by the surface
   * predictor instead of fetched (see core/surfpred.h) */
  struct r3d_surfpred *sp;
} r3d_cpuvol;

int r3d_cpuvol_open(r3d_cpuvol *v, const char *root, uint32_t cache_bricks);
/* allow_predict=false opens a predict tree as plain files only (no
 * predictor, no recursion) — used by the predictor to read its own output */
int r3d_cpuvol_open_ex(r3d_cpuvol *v, const char *root, uint32_t cache_bricks,
                       bool allow_predict);
void r3d_cpuvol_close(r3d_cpuvol *v);

/* Nearest-neighbor value at base-resolution voxel coords, sampled from
 * pyramid level li. Out-of-bounds or no-data reads 0. */
uint8_t r3d_cpuvol_at(r3d_cpuvol *v, uint32_t li, double x, double y, double z);

/* Trilinear value at base-resolution voxel coords sampled from level li,
 * 0..255 continuous; optionally its analytic gradient d value / d base
 * voxel (the trilinear surface is piecewise differentiable — solvers need
 * this the way vc3d gets interpolator derivatives from autodiff). */
double r3d_cpuvol_tri(r3d_cpuvol *v, uint32_t li, const double p[3], double grad[3]);

/* Demand-fetch (in parallel, `threads` connections) the cells owning the
 * listed level-li bricks (x,y,z triples) that are neither cached nor on
 * disk. Fetched bricks land in the decode cache directly. Returns the
 * number of cells fetched, or -1. */
int r3d_cpuvol_prefetch(r3d_cpuvol *v, uint32_t li, const uint32_t *bxyz, uint32_t n,
                        uint32_t threads);

/* Copy an axis-aligned block of level-li voxels (base-level index space of
 * that level; may extend outside the volume — those voxels read 0; absent
 * bricks read 0). Bulk brick copies, not per-voxel sampling. */
void r3d_cpuvol_read_block(r3d_cpuvol *v, uint32_t li, int64_t x0, int64_t y0, int64_t z0,
                           uint32_t nx, uint32_t ny, uint32_t nz, uint8_t *out);

/* Insert a raw 128^3 brick into the decode cache (e.g. one just produced by
 * the predictor) so the next sample hits without touching the disk. */
void r3d_cpuvol_cache_put(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz,
                          const uint8_t *raw);

#endif /* R3D_CPUVOL_H */
