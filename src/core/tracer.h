/* Surface tracer: a faithful C reimplementation of vc3d's grow-from-seed
 * tracer (GrowPatch.cpp tracer()) over a surface-prediction volume. A quad
 * grid grows generation by generation from a 2x2 seed quad; every new cell
 * is committed on top of its best parent and pushed into place by nonlinear
 * least squares (Levenberg-Marquardt, hand-rolled — no Ceres): spacing
 * (DistLoss), curvature (StraightLoss), parameterization quality
 * (SymmetricDirichletLoss + Cauchy), and a data term pulling grid edges
 * onto the prediction sheet (vc3d's space-line loss over a Euclidean
 * distance transform of the thresholded predictions — vc3d's default
 * normal-grid term needs a preprocessed polyline volume we don't have).
 * There is no accept/reject: quality comes from the solve schedule
 * (per-point, radius-1, radius-3 after each placement; periodic global and
 * subsampled radius-8 solves). Runs on its own thread; the GUI polls
 * snapshots and draws growth live. Saves as tifxyz ready for segpack. */
#ifndef R3D_TRACER_H
#define R3D_TRACER_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "core/umbilicus.h"

enum { R3D_TR_EMPTY = 0, R3D_TR_SET = 1, R3D_TR_FAIL = 2, R3D_TR_PROC = 3 };

typedef struct r3d_tracer_cfg {
  double seed[3];    /* world voxels */
  double step;       /* grid unit, voxels (tifxyz scale = 1/step; vc3d
                      * convention is 20) */
  float thresh;      /* confidence cutoff for SAVING/display, 0..1; growth
                      * itself never rejects (vc3d semantics) */
  uint32_t max_ring; /* generations to grow (grid dim = 2*g + 50) */
  uint32_t level;    /* prediction pyramid level to sample */
} r3d_tracer_cfg;

typedef struct r3d_tracer {
  r3d_tracer_cfg cfg;
  char root[1024]; /* prediction LOD tree */
  uint32_t W, H;   /* grid dims */
  double *pos;     /* [W*H*3], guarded by mu */
  uint8_t *state;  /* [W*H] R3D_TR_* */
  float *conf;     /* [W*H] data-term proximity 0..1 (display/save mask) */
  r3d_umbilicus umb; /* copied winding guide (unused by the vc3d energy;
                      * kept for future fusion modes) */
  pthread_t th;
  pthread_mutex_t mu;
  bool running, quit, done;
  uint32_t ring, nset;   /* ring = generations grown so far */
  uint32_t gens_done;    /* completed generations across resumes */
  unsigned rng;          /* placement-perturbation PRNG state */
  uint64_t gen;
} r3d_tracer;

/* Copies cfg + umbilicus and starts the grow thread. */
int r3d_tracer_start(r3d_tracer *t, const char *pred_root, const r3d_tracer_cfg *cfg,
                     const r3d_umbilicus *umb);
void r3d_tracer_stop(r3d_tracer *t); /* request stop + join; result kept */
/* Enlarge a finished (stopped) trace and resume growth for `extra` more
 * generations (vc3d resume path: existing cells persist and anchor the
 * solve). Grid buffers reallocate — callers must refetch W/H. */
int r3d_tracer_grow(r3d_tracer *t, uint32_t extra);
void r3d_tracer_free(r3d_tracer *t);

/* Copy the current grid into caller buffers (each may be NULL); returns the
 * generation so callers can skip unchanged copies. */
uint64_t r3d_tracer_snapshot(r3d_tracer *t, double *pos, uint8_t *state, float *conf,
                             uint32_t *ring, uint32_t *nset, bool *done);

/* Write the traced grid as <dir>/{x,y,z}.tif + meta.json (failed/empty
 * cells = the tifxyz invalid encoding). fill = no holes: every grown
 * cell is written; untrusted cells are re-seated by membrane
 * interpolation anchored on trusted neighbors instead of skipped. */
int r3d_tracer_save(r3d_tracer *t, const char *dir, float cutoff, bool fill);

#endif /* R3D_TRACER_H */
