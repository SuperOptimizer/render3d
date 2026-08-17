/* On-demand surface prediction: fills a "predict tree" (a c5d LOD tree with
 * the CT's geometry, no data of its own, source.json url predict://host:port
 * + ct_root) by sampling CT blocks, running them through tools/surf/
 * surfserver.py (nnU-Net surface_m7), and writing the resulting bricks into
 * <root>/bricks/L{0,1}. Both consumers of prediction trees — the tracer's
 * cpuvol sampler and the renderer's overlay ingest — call r3d_surfpred_cell
 * where they would otherwise fetch a zarr cell over HTTP.
 *
 * Cell geometry: an L0 cell is 2x2x2 bricks (256^3 voxels) and is exactly
 * one L1 brick; predicting an L0 cell (with a margin for context) yields
 * eight L0 bricks and, by 2x box reduction, the L1 brick. Levels >= 2 are
 * not produced (the tracer runs at L0/L1; coarser overlay LODs stay empty). */
#pragma once
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "core/cpuvol.h"

typedef struct r3d_surfpred {
  char root[1024];    /* the prediction tree */
  char ct_root[1024]; /* the CT tree it predicts from */
  int port;
  uint32_t margin;    /* context voxels around the cell (default 32) */
  float th;           /* probabilities below th read as 0 (default 0.2) */
  float q;            /* c5d quality for the written bricks */
  r3d_cpuvol ct;
  bool ct_ok;
  int fd;             /* server connection, -1 = down */
  pthread_mutex_t mu; /* one prediction at a time (the server serializes too) */
  uint64_t cool_until; /* seconds: back off after a failure */
  uint32_t predicted_cells, failed_cells;
} r3d_surfpred;

/* true when the source url selects the predictor */
bool r3d_surfpred_url(const char *url);

/* Parse the predict tree's source.json (url predict://host:port, ct_root, and
 * optional margin/th) and open the CT sampler. Returns 0 on success. */
int r3d_surfpred_open(r3d_surfpred *sp, const char *pred_root);
void r3d_surfpred_close(r3d_surfpred *sp);

/* Predict the cell that owns brick (bx,by,bz) at level li (0 or 1): writes
 * the eight L0 bricks + the L1 brick as .c5b files (skipping ones already on
 * disk), optionally inserts them into `cache` (a cpuvol over the same tree),
 * and if `out` is non-NULL fills it with the requested level's cell content
 * (li 0: 256^3 voxels of the L0 cell; li 1: the 128^3 L1 brick).
 * Returns 1 = predicted, 0 = level not served (>= 2) or nothing to do,
 * -1 = failure (server unreachable / CT missing). */
int r3d_surfpred_cell(r3d_surfpred *sp, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz,
                      uint8_t *out, r3d_cpuvol *cache);
