/* Seed-grown surface tracer over a surface-prediction volume (vc3d-style
 * grow-from-seed, reimplemented): a quad grid expands ring by ring from a
 * seed point, each new vertex extrapolated from its neighbors, snapped to
 * the prediction ridge along the local sheet normal (radial from the
 * umbilicus — scroll sheets wind around the core), then relaxed. Runs on
 * its own thread; the GUI polls snapshots and draws growth live. The
 * finished grid saves as a tifxyz dir ready for segpack. */
#ifndef R3D_TRACER_H
#define R3D_TRACER_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "core/umbilicus.h"

enum { R3D_TR_EMPTY = 0, R3D_TR_SET = 1, R3D_TR_FAIL = 2 };

typedef struct r3d_tracer_cfg {
  double seed[3];   /* world voxels */
  double step;      /* grid pitch, voxels (tifxyz scale = 1/step) */
  float thresh;     /* prediction acceptance, 0..1 */
  uint32_t max_ring; /* grid radius, cells */
  uint32_t level;   /* prediction pyramid level to sample */
  double search;    /* ridge-snap range along the normal, voxels */
} r3d_tracer_cfg;

typedef struct r3d_tracer {
  r3d_tracer_cfg cfg;
  char root[1024]; /* prediction LOD tree */
  uint32_t W, H;   /* 2*max_ring+1 square grid */
  double *pos;     /* [W*H*3], guarded by mu */
  uint8_t *state;  /* [W*H] R3D_TR_* */
  r3d_umbilicus umb; /* copied winding guide (may be empty) */
  pthread_t th;
  pthread_mutex_t mu;
  bool running, quit, done;
  uint32_t ring, nset;
  uint64_t gen;
} r3d_tracer;

/* Copies cfg + umbilicus and starts the grow thread. */
int r3d_tracer_start(r3d_tracer *t, const char *pred_root, const r3d_tracer_cfg *cfg,
                     const r3d_umbilicus *umb);
void r3d_tracer_stop(r3d_tracer *t); /* request stop + join; result kept */
/* Enlarge a finished (stopped) trace by extra rings and restart growth:
 * SET cells persist, FAILED cells get another chance (predictions may
 * have streamed in since). Grid buffers reallocate — callers must refetch
 * W/H before snapshotting. */
int r3d_tracer_grow(r3d_tracer *t, uint32_t extra);
void r3d_tracer_free(r3d_tracer *t);

/* Copy the current grid into caller buffers (each may be NULL); returns the
 * generation so callers can skip unchanged copies. */
uint64_t r3d_tracer_snapshot(r3d_tracer *t, double *pos, uint8_t *state, uint32_t *ring,
                             uint32_t *nset, bool *done);

/* Write the traced grid as <dir>/{x,y,z}.tif + meta.json (failed/empty
 * cells = the tifxyz invalid encoding). */
int r3d_tracer_save(r3d_tracer *t, const char *dir);

#endif /* R3D_TRACER_H */
