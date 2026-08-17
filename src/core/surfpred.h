/* On-demand surface prediction: fills a "predict tree" (a c5d LOD tree with
 * the CT's geometry, no data of its own, source.json url predict://host:port
 * + ct_root) by sampling CT blocks, running them through tools/surf/
 * surfserver.py (nnU-Net surface_m7), and writing the resulting bricks into
 * <root>/bricks/L{0,1}. Both consumers of prediction trees — the tracer's
 * cpuvol sampler and the renderer's overlay ingest — call r3d_surfpred_cell
 * where they would otherwise fetch a zarr cell over HTTP.
 *
 * The model works at ~8-9 um voxels: pred_level P is the CT pyramid level
 * with that pitch (0 for the ESRF 8.6/9.4 um scans, 2 for 2.4 um volumes —
 * the bucket's m7-L0 / m7-L2 trees). A cell is 2x2x2 level-P bricks (256^3
 * level-P voxels) and is exactly one level-P+1 brick: predicting it (with a
 * context margin) yields eight P bricks and, by 2x box reduction, the P+1
 * brick. Requests for finer levels (li < P) are served by nearest upsampling
 * of the owning cell (predicted or taken from a small in-memory ring), one
 * brick at a time. Levels > P+1 are not produced. */
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
  uint32_t pred_level; /* CT level fed to the model (0 = 8-9um scans, 2 = 2.4um) */
  r3d_cpuvol ct;
  bool ct_ok;
  r3d_cpuvol self; /* the tree's own files (no predictor): cells already on disk */
  bool self_ok;
  int fd;             /* server connection, -1 = down */
  pthread_mutex_t mu; /* one prediction at a time (the server serializes too) */
  uint64_t cool_until; /* seconds: back off after a failure */
  uint32_t predicted_cells, failed_cells;
  /* ring of recent predicted cells (thresholded 256^3 level-P voxels) so
   * finer-level upsampled bricks of one cell don't re-run the model */
  struct {
    uint8_t *data;   /* SP_RING x 256^3 */
    uint64_t key[8]; /* cz<<40|cy<<20|cx, UINT64_MAX = free */
    uint32_t next;
  } ring;
} r3d_surfpred;

/* true when the source url selects the predictor */
bool r3d_surfpred_url(const char *url);

/* Parse the predict tree's source.json (url predict://host:port, ct_root, and
 * optional margin/th) and open the CT sampler. Returns 0 on success. */
int r3d_surfpred_open(r3d_surfpred *sp, const char *pred_root);
void r3d_surfpred_close(r3d_surfpred *sp);

/* Serve brick (bx,by,bz) at level li: predicts the owning level-P cell if
 * needed (writing its eight P bricks + the P+1 brick as .c5b, skipping ones
 * on disk), for li < P writes the requested upsampled brick, optionally
 * inserts produced bricks into `cache` (a cpuvol over the same tree), and if
 * `out` is non-NULL fills it with the requested cell content (li == P: the
 * 256^3 cell; otherwise the single 128^3 brick).
 * Returns 1 = produced, 0 = level not served (> P+1), -1 = failure. */
int r3d_surfpred_cell(r3d_surfpred *sp, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz,
                      uint8_t *out, r3d_cpuvol *cache);
