/* Void-boundary surface grower: an automatic sibling of the manual 'P'
 * surface. The user first sets the render low cut so papyrus stands
 * against air with a hard edge, then clicks on (or near) that edge. From
 * the nearest boundary voxel a quad grid grows generation by generation:
 * every fringe node is extrapolated from its grown neighbours and then
 * SNAPPED to the papyrus/void interface along the local outward normal
 * (papyrus on the inside, air outside), never farther than the snap
 * distance. Boundary normals measured at each landing keep the sheet on
 * ONE face: a node whose measured normal disagrees with its neighbours'
 * is a different face (or a sheet edge wrapping round) and fails; the
 * fringe stops there. Nodes are the papyrus voxels that touch the void,
 * so the saved tifxyz is exactly that voxel set, gridded.
 *
 * Volume access goes through a sample callback (integer voxel -> u8), so
 * the grower runs over a cpuvol tree in the GUI and over a synthetic
 * array in the tests. */
#ifndef R3D_BSURF_H
#define R3D_BSURF_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

typedef uint8_t (*r3d_bsurf_sample_fn)(void *ctx, int64_t x, int64_t y, int64_t z);

enum { R3D_BS_EMPTY = 0, R3D_BS_SET = 1, R3D_BS_FAIL = 2 };

typedef struct r3d_bsurf_cfg {
  double seed[3];   /* world voxels: on/near the papyrus-void edge */
  double step;      /* grid spacing, voxels (clamped >= 2) */
  uint32_t gens;    /* generations to grow (grid dim = 2*gens + 6) */
  double snap;      /* max distance a node may travel to reach the edge */
  double low_cut;   /* voxel value >= low_cut is papyrus, below is void */
  double vdim[3];   /* volume extent (0 = unbounded): growth stops there */
} r3d_bsurf_cfg;

typedef struct r3d_bsurf {
  r3d_bsurf_cfg cfg;
  r3d_bsurf_sample_fn sample;
  void *sctx;
  uint32_t W, H;
  double *pos;      /* [W*H*3] integer voxel centers of SET nodes */
  uint8_t *state;   /* [W*H] R3D_BS_* */
  float *nrm;       /* [W*H*3] outward (papyrus -> void) unit normal */
  uint16_t *gen_of; /* [W*H] generation placed (seed block = 1) */
  uint8_t *ftry;    /* [W*H] SET-neighbour count at the last failure:
                     * a failed node is retried once it gains support */
  /* self-overlap hash over SET node positions (bucket = step-sized cube) */
  uint64_t *hkey;
  int32_t *hhead;
  int32_t *hnext;
  uint32_t hcap;
  uint32_t ring, nset, nfail;
  uint32_t nwhy[4]; /* failure attempts: no edge / other face / spacing / overlap */
  float face_ok;    /* done: fraction of nodes with papyrus behind + void ahead */
  uint64_t gen;      /* bumps every generation (snapshot change key) */
  bool running, quit, done, failed;
  char status[200];
  pthread_t th;
  pthread_mutex_t mu;
} r3d_bsurf;

/* Grow on a worker thread. Returns 0 if the thread started (seed
 * validity is reported through failed/status once the worker looks). */
int r3d_bsurf_start(r3d_bsurf *t, const r3d_bsurf_cfg *cfg, r3d_bsurf_sample_fn fn,
                    void *ctx);
/* Same, synchronously on the calling thread (tests, CLI). Returns 0 when
 * a surface was grown, -1 when the seed found no boundary. */
int r3d_bsurf_run(r3d_bsurf *t, const r3d_bsurf_cfg *cfg, r3d_bsurf_sample_fn fn,
                  void *ctx);
void r3d_bsurf_stop(r3d_bsurf *t); /* request stop + join; result kept */
void r3d_bsurf_free(r3d_bsurf *t);

/* Copy the grid (pos: [W*H*3], invalid nodes = -1; may be NULL) and the
 * counters; returns gen so callers skip unchanged copies. */
uint64_t r3d_bsurf_snapshot(r3d_bsurf *t, double *pos, uint32_t *ring, uint32_t *nset,
                            bool *done);

int r3d_bsurf_selftest(void); /* synthetic sphere shell + slab; 0 = ok */

#endif /* R3D_BSURF_H */
