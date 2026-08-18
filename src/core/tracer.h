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
  double x_min, x_max; /* optional x/y boxes, same convention as z */
  double y_min, y_max;
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
  uint8_t grow_dirs;  /* growth-direction bitmask over the 8-neighbour
                       * order {+u,+v,-u,-v,diag...}; 0 = all (G13) */
  uint32_t max_threads; /* solve-pool cap (0 = default); set when several
                         * tracers grow concurrently so they share cores */
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
  float *werr;     /* [W*H] |wind - relaxed wind field|: cells whose
                    * placement-time winding disagrees with the winding
                    * their neighbourhood geometry implies (wrong-wrap
                    * capture detector; refreshed with the spiral fit) */
  r3d_umbilicus umb; /* winding guide (spiral frame origin per slice) */
  /* smoothed umbilicus centerline, x,y per z slice (sigma = 75 slices,
   * vc3d spiral-service convention); NULL when no/degenerate umbilicus */
  double *uc;
  uint32_t ucn;
  /* global spiral fit rho ~ r0(z) + omega*w over the grown points,
   * refit each generation (piecewise-linear r0, IRLS/Cauchy) */
  double sp_omega, sp_rms;
  /* coarse per-region sheet-gap field (G5): real cross-sections vary
   * 2-3x in gap; one global median shoves genuine wraps apart where the
   * gap is tight and lets them interpenetrate where it is wide. Filled
   * from the same samples as sp_om_meas, nearest-filled + box-smoothed;
   * tr_om_at() falls back to the global scalar outside coverage. */
  float *omf;
  double omf_o[3], omf_cs;
  uint32_t omf_n[3];
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
  /* donor uv membership (G12): nearest-donor id per cell (-1 none,
   * -2 = fold-suspect: the uv map is discontinuous there and the donor
   * pull/adoption must not fire) + donor-grid uv of the closest point */
  int8_t *dcell_id;
  float *dcell_uv; /* [W*H*2] */
  uint16_t *gen_of; /* [W*H] generation each cell was placed (seed = 1);
                     * saved as generations.tif — the rewind substrate */
  uint8_t *grow_mask; /* optional: candidates allowed only where nonzero
                       * (region regrow); owned, freed on tracer_free */
  bool mask_once;     /* grow_mask set by reopt: cleared at worker finish */
  uint16_t cur_gen;   /* generation being grown (worker-internal) */
  /* fusion consensus gate (G6): candidates in fused mode must be donor-
   * supported or solve below inl_th; rejects go back to EMPTY
   * (retryable) and inl_th anneals down when the fringe starves */
  double inl_th;
  pthread_t th;
  pthread_mutex_t mu;
  bool running, quit, done;
  bool refine; /* solve-only pass in flight (no growth) */
  /* re-optimisation position memory (refine/inpaint): cells keep their
   * tangential position, moving only along their own frozen normal */
  bool reopt_on;
  double *reopt_pos; /* [W*H*3] positions at pass start */
  float *reopt_nrm;  /* [W*H*3] unit normals at pass start (0 = none) */
  /* worker-owned raw-CT sampler exposed to the placement pre-veto (ribbon
   * boundary test before commit instead of a generation later) */
  struct r3d_cpuvol *bnd_ct;
  uint32_t bnd_lv;
  double bnd_min;
  /* per-generation mesh QC (display + meta.json; refreshed with the
   * spiral fit, trusted cells only = SET && conf > 0.25):
   * folds = consecutive-edge pairs turned past 90 deg (doubling back),
   * kinks = pairs past 30 deg, twist = rms free-corner distance from the
   * quad plane in voxels (planarity) */
  uint32_t qc_folds, qc_kinks;
  float qc_twist;
  /* first fold locations this generation: fed to the active fold-repair
   * anneal so a fold is fixed while it is one generation old instead of
   * parenting the next ring */
  uint32_t qc_fold_cell[16];
  uint32_t qc_nfoldc;
  double qc_area_vx2;  /* two-triangle quad area over all-trusted quads */
  uint32_t qc_bbox[4]; /* i0,j0,i1,j1 inclusive over trusted cells */
  float qc_fill;       /* trusted cells / bbox area */
  float qc_hole;       /* enclosed (border-unreachable) untrusted / bbox */
  float qc_slant_p95;  /* |e_u.e_v|/|e_u|^2 — coherent shear detector */
  /* donor agreement (fused runs only; <=2000 sampled trusted cells) */
  float qc_don_mean, qc_don_rms, qc_don_p95; /* voxels */
  float qc_don_cov; /* fraction with a donor within 2 grid steps */
  /* winding consistency (needs an umbilicus): werr = |causal winding -
   * relaxed winding field|; wrap_frac = fraction of trusted cells with
   * werr > 0.3 (likely wrong-wrap captures) */
  float qc_werr_p95, qc_wrap_frac;
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

/* Re-solve a finished (stopped) trace in place without growing: anchor
 * neighborhoods are annealed through the user anchors, then a global
 * polish smooths the seams. Existing cells persist; grid dims unchanged.
 * Poll snapshots as usual; done goes true when the pass ends. */
int r3d_tracer_refine(r3d_tracer *t);

/* Load a saved trace (x/y/z.tif + optional winding/generations.tif) back
 * into a fresh tracer so it can be rewound, refined, or grown. pred_root
 * is the prediction tree growth would sample. Returns 0 on success. */
int r3d_tracer_load(r3d_tracer *t, const char *dir, const char *pred_root);

/* Drop every cell placed after `gen` (vc3d --rewind-gen): the standard
 * fix for a trace that went wrong at generation N of M — rewind past the
 * jump, drop an anchor, regrow. Stopped tracer only; follow with
 * r3d_tracer_grow to regrow. */
int r3d_tracer_rewind(r3d_tracer *t, uint32_t gen);

/* Unified-tracer stage 1 (vc3d gen_neighbor): derive the ADJACENT WRAP
 * of a finished (stopped/loaded) trace by casting every trusted cell
 * along its local normal by the local sheet gap — the ray must first
 * exit the current sheet, then the first prediction-DT minimum is the
 * neighbor crossing. Misses are filled by interpolating neighbor
 * offsets. dir = +1 (outward normal) or -1. Writes a tifxyz (x/y/z +
 * winding shifted by dir + meta) to out_dir, immediately usable as a
 * fusion donor for growing the neighbor wrap. Returns the number of
 * direct hits, or -1. */
int r3d_tracer_derive(r3d_tracer *t, const char *pred_root, int dir,
                      const char *out_dir);

/* Reopen and regrow the region around world point p (vc3d discard-and-
 * regrow): flood the suspect neighbourhood (low conf / high werr) of the
 * nearest cell out to `radius`, empty it against a frozen boundary ring,
 * and restart growth restricted to that region. Where a patch jumped a
 * wrap, the correct geometry is not a perturbation of the wrong one —
 * this is the clean fix. Aborts (-1) if the region reaches the grid
 * border (that is a rewind, not a reopt). Stopped tracer only. */
int r3d_tracer_reopt(r3d_tracer *t, const double p[3], int radius);
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
