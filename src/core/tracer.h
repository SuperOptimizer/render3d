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
  uint32_t rib_rows; /* >0: ribbon mode — grid is (2*max_ring+10) x rib_rows,
                      * growth runs along the sheet (whole-cross-section
                      * tracing in a thin z slab, Lasagna-style) */
  uint32_t rib_wraps; /* >1: grow this many wraps at once as sibling
                       * ribbons (one seed per radial sheet crossing,
                       * spacer rows between blocks, shared winding frame
                       * + mutual spacing; fronts stop where they meet a
                       * sibling on the same winding) */
  double z_min, z_max; /* z_max > z_min: hard z clamp (vc3d z_range);
                        * solved points outside FAIL like a volume exit */
  char ct_root[1024]; /* optional raw-CT LOD tree: ribbon fronts stop
                       * where the CT says there is no scroll. Predictions
                       * may be weak where papyrus continues — the CT is
                       * the ground truth for "nothing here". */
  double ct_min;      /* CT validity cutoff (default 128): masked volumes
                       * zero the outside but leave a gray halo that was
                       * never fully masked — intensities below this are
                       * not scroll volume and fronts must not cross. */
  double wind_weight; /* spiral winding prior weight (0 = off). Needs an
                       * umbilicus; the residual is normalized by the
                       * fitted sheet spacing so ~0.5 is a gentle prior. */
} r3d_tracer_cfg;

#define R3D_TR_MAX_ANCHORS 64

typedef struct r3d_tracer {
  r3d_tracer_cfg cfg;
  /* user anchors: world points the sheet must pass through (GUI
   * corrections for wrong-sheet jumps). Staged under mu; the grow thread
   * adopts them at each generation boundary, assigns each to the nearest
   * grown cell within a capture radius, and a strong pull on that cell
   * joins every solve until the sheet passes through the point. */
  double anc[R3D_TR_MAX_ANCHORS * 3];
  int32_t anc_cell[R3D_TR_MAX_ANCHORS]; /* grid index, -1 = unassigned */
  uint32_t nanc;
  double anc_new[R3D_TR_MAX_ANCHORS * 3]; /* staged replacement set */
  uint32_t nanc_new;
  bool anc_dirty;
  char root[1024]; /* prediction LOD tree */
  uint32_t W, H;   /* grid dims */
  double *pos;     /* [W*H*3], guarded by mu */
  uint8_t *state;  /* [W*H] R3D_TR_* */
  float *conf;     /* [W*H] data-term proximity 0..1 (display/save mask) */
  float *wind;     /* [W*H] winding number about the umbilicus (spiral
                    * frame; seed = 0, accumulated combinatorially at
                    * placement, never rewritten by solves) */
  r3d_umbilicus umb; /* winding guide (spiral frame origin per slice) */
  /* smoothed umbilicus centerline, x,y per z slice (sigma = 75 slices,
   * vc3d spiral-service convention); NULL when no/degenerate umbilicus */
  double *uc;
  uint32_t ucn;
  /* global spiral fit rho ~ r0(z) + omega*w over the grown points,
   * refit each generation (piecewise-linear r0, IRLS/Cauchy) */
  double sp_omega, sp_rms;
  double sp_om_meas; /* inter-sheet gap measured on radial DT rays; lets
                      * the fit run fixed-omega before the patch spans a
                      * full winding (where joint omega is unidentifiable) */
  double sp_r0[64];
  double sp_ab[4]; /* theta harmonics 1..2 (cos1,sin1,cos2,sin2): absorbs
                    * umbilicus offset + cross-section ellipticity */
  double sp_z0, sp_dz;
  uint32_t sp_k;
  bool sp_valid;
  void *wf;       /* winding-potential field r1(x,y) for the seed slab
                   * (evolutor port: geometry-agnostic winding coordinate
                   * from normal-grid polylines + one linear solve) */
  double wf_base;  /* field value at the seed (winding 0 reference) */
  void *sfx;      /* self-overlap hash (SET cell positions), rebuilt per
                   * generation; the anti-interpenetration hinge reads it */
  void *don;      /* donor segments + spatial index (fusion), owned */
  uint32_t ndon;
  uint8_t *dsup;  /* [W*H] donor-support count per cell (0 = raw-traced) */
  pthread_t th;
  pthread_mutex_t mu;
  bool running, quit, done;
  uint32_t ring, nset;   /* ring = generations grown so far */
  double vdim[3];        /* scroll volume extent (growth hard-stops there) */
  uint32_t gens_done;    /* completed generations across resumes */
  unsigned rng;          /* placement-perturbation PRNG state */
  uint64_t gen;
} r3d_tracer;

/* Copies cfg + umbilicus and starts the grow thread. */
int r3d_tracer_start(r3d_tracer *t, const char *pred_root, const r3d_tracer_cfg *cfg,
                     const r3d_umbilicus *umb);
/* Same, with donor segments (fusion): each dir is a saved tifxyz
 * (winding.tif + spiral.json consumed when present). Candidates near a
 * donor are pulled onto it during solves (vc3d grow_surf_from_surfs's
 * SurfaceLossD role); donors on a different winding never attract
 * (spiral-frame gate — vc3d has no such guard). Cells without donor
 * support fall back to raw tracing, so a set of patches plus growth in
 * the gaps fuses into one surface. */
int r3d_tracer_start_fused(r3d_tracer *t, const char *pred_root,
                           const r3d_tracer_cfg *cfg, const r3d_umbilicus *umb,
                           const char *const *donor_dirs, uint32_t ndonors);
void r3d_tracer_stop(r3d_tracer *t); /* request stop + join; result kept */

/* Replace the anchor set (world voxels, n <= R3D_TR_MAX_ANCHORS; excess is
 * dropped). Callable any time, including while growing — the grow thread
 * adopts the new set at the next generation boundary. */
void r3d_tracer_set_anchors(r3d_tracer *t, const double *pts, uint32_t n);
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

/* Synthetic self-check of the spiral winding frame + global fit (used by
 * the unit tests; no volume access). Returns 0 on success. */
int r3d_tracer_spiral_selftest(void);
int r3d_tracer_fusion_selftest(void);

#endif /* R3D_TRACER_H */
