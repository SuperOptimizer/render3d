/* Live 2.5D ink detection for the flattened segment view: a worker thread
 * samples a 21-layer surface volume over the visible grid rect (cpuvol,
 * demand-fetching the same LOD cache the viewer streams), ships it to the
 * persistent inference server (tools/ink9/inkserver.py) over TCP, and
 * publishes the returned probability map for display. */
#pragma once
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "core/cpuvol.h"

#define R3D_INKLIVE_LAYERS 21u /* base slab depth (model 17 + 2 slack);
                                * a TTA depth-shift range S > 2 grows the
                                * sampled slab to 17 + 2*S layers so the
                                * server can slide the window that far */
#define R3D_INKLIVE_MAX_PX 2048u /* per-axis cap on one request's pixels */

typedef struct r3d_inklive {
  r3d_cpuvol vol;
  bool vol_ok;
  int port;
  int fd; /* server connection (worker-owned), -1 when down */

  pthread_t th;
  bool th_up;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  bool quit;

  /* request (owner: caller under mu; consumed by the worker) */
  _Atomic uint32_t req_gen; /* bump = new work (worker polls it to abandon stale sampling) */
  float *req_xyz;     /* (rw+1)*(rh+1)*3 grid corner positions, -1 = invalid */
  uint32_t rw, rh;    /* rect size in grid CELLS */
  uint32_t ri0, rj0;  /* rect origin in grid coords */
  uint32_t req_tta;   /* TTA/ensemble bitmask for the staged request */
  uint32_t up;        /* pixels per grid cell (= voxels per cell) */

  /* result (owner: worker under mu; consumed via r3d_inklive_poll) */
  uint32_t res_gen; /* matches the req_gen it answers */
  float *res;       /* rw*up x rh*up probabilities */
  uint32_t res_w, res_h, res_i0, res_j0, res_up;
  bool res_fresh;
  double res_ms;    /* sample+infer wall time, for the panel */
  char status[96];
} r3d_inklive;

/* Open the CT volume at vol_root and connect the worker; server may come up
 * later (the worker reconnects per request). Returns 0 on success. */
int r3d_inklive_start(r3d_inklive *il, const char *vol_root, int port);
void r3d_inklive_stop(r3d_inklive *il);

/* Ask for a prediction over grid cells [i0,i0+w)x[j0,j0+h). xyz is the
 * (w+1)x(h+1)x3 corner-position snapshot (row-major, invalid < 0); the call
 * copies nothing — ownership of the malloc'd xyz transfers to the module.
 * A request in flight is superseded (last write wins).
 * tta is the server's TTA/ensemble bitmask (bit0 flips, bit1 depth
 * shift, bit2 intensity, bit3 checkpoint ensemble; 0 = fast single pass;
 * TTA multiplies inference cost ~4-16x, so keep the live view at 0). */
void r3d_inklive_request(r3d_inklive *il, float *xyz, uint32_t i0, uint32_t j0,
                         uint32_t w, uint32_t h, uint32_t up, uint32_t tta);

/* True once per fresh result; hands out the module-owned buffer (valid until
 * the next result is published). */
bool r3d_inklive_poll(r3d_inklive *il, const float **pred, uint32_t *w, uint32_t *h,
                      uint32_t *i0, uint32_t *j0, uint32_t *up);
