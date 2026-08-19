/* Volume registration: overlay a second scan ("moving") of the same scroll
 * on the streamed volume ("fixed") and line them up. The moving scan is
 * resampled per fixed-volume brick through an affine pull map (fixed voxel
 * -> moving voxel), so the renderer can display it in every viewer via a
 * slot-parallel atlas. Auto-refinement (fysics): coarse-to-fine NCC affine /
 * rigid registration on an ROI, run on a worker thread.
 *
 * Transform model: an authoritative base pull map M (loaded from a shipped
 * transform.json, seeded from a voxel-size ratio, or identity) composed with
 * interactive deltas (translate / rotate / log-scale of the moving volume
 * within the fixed frame, about the fixed volume center). "Bake" folds the
 * deltas into M. All matrices here are 3x4 row-major in (x, y, z) order;
 * conversion to fysics' (z, y, x) convention happens at the call boundary. */
#ifndef R3D_REGVOL_H
#define R3D_REGVOL_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "core/cpuvol.h"

typedef struct r3d_regvol {
  r3d_cpuvol mv; /* moving scan (thread-safe sampler over its LOD tree) */
  bool open;
  char root[1024];
  uint32_t fdim[3]; /* fixed volume dims (x, y, z), base voxels */
  /* base pull map: fixed base voxel (xyz) -> moving base voxel (xyz) */
  double M[12];
  /* interactive deltas: move the MOVING volume within the fixed frame */
  double d_tr[3];  /* translation, fixed voxels (x, y, z) */
  double d_rot[3]; /* rotation about x/y/z axes, radians */
  double d_lscale; /* log isotropic scale */
  uint32_t gen;    /* display generation; bumps on any transform change */
  pthread_mutex_t mu;
  /* resample scratch (render-thread fetch only) */
  uint8_t *scratch;
  size_t scratch_cap;
  /* worker: NCC measure / auto-refine */
  pthread_t th;
  bool th_up;
  _Atomic int state; /* 0 idle, 1 running, 2 done, 3 failed */
  int job_mode;      /* 0 = measure NCC only, 1 = rigid refine, 2 = affine */
  char fixed_root[1024];
  double job_ctr[3]; /* ROI center, fixed base voxels (xyz) */
  uint32_t job_half; /* ROI half-extent in level voxels */
  uint32_t job_level;
  double job_P[12];   /* pull map snapshot the job ran with */
  double job_Mnew[12]; /* refined pull map (job result) */
  double ncc0, ncc1;  /* before / after (measure: ncc1 == ncc0) */
} r3d_regvol;

int r3d_regvol_open(r3d_regvol *rv, const char *moving_root, const uint32_t fixed_dim[3]);
void r3d_regvol_close(r3d_regvol *rv);

/* effective pull map: deltas composed onto M */
void r3d_regvol_pull(r3d_regvol *rv, double P[12]);
void r3d_regvol_bump(r3d_regvol *rv); /* display generation++ (call after edits) */
void r3d_regvol_bake(r3d_regvol *rv); /* fold deltas into M, zero them */
void r3d_regvol_reset_deltas(r3d_regvol *rv);
void r3d_regvol_set_scale(r3d_regvol *rv, double s); /* M = diag(s) (seed) */
double r3d_regvol_scale(r3d_regvol *rv); /* effective isotropic scale cbrt|det P| */
/* voxel pitch parsed from a path (bucket volume names embed "...-1.129um-...");
 * 0 when the path carries none */
double r3d_regvol_parse_um(const char *path);

/* transform.json: accepts the shipped Vesuvius format ("transformation_matrix",
 * 3x4 XYZ, moving -> fixed; inverted into the pull map) or this tool's own
 * saves ("pull_matrix_xyz", used directly). Save writes both. */
int r3d_regvol_load_json(r3d_regvol *rv, const char *path);
int r3d_regvol_save_json(r3d_regvol *rv, const char *path);

/* renderer source callbacks (r3d_label_src signature) */
uint32_t r3d_regvol_srcgen(void *rv, uint32_t level, uint32_t bx, uint32_t by, uint32_t bz);
void r3d_regvol_srcfetch(void *rv, uint32_t level, uint32_t bx, uint32_t by, uint32_t bz,
                         uint8_t *out);

/* worker job: measure NCC over an ROI (mode 0) or refine the transform there
 * (mode 1 rigid / 2 affine; result auto-applies on poll). ctr in fixed base
 * voxels (xyz); ROI = (2*half)^3 level-`level` voxels. Returns -1 if busy. */
int r3d_regvol_job_start(r3d_regvol *rv, const char *fixed_root, int mode,
                         const double ctr[3], uint32_t half, uint32_t level);
/* 1 = still running; 0 = idle/collected (a finished refine is applied and
 * gen bumped on the first poll that sees it); *ok=false if the job failed */
int r3d_regvol_job_poll(r3d_regvol *rv, bool *ok);

#endif /* R3D_REGVOL_H */
