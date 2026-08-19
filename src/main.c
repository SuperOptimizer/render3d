/* render3d — volumetric renderer for Vesuvius Challenge micro-CT volumes.
 * M1: SDL3 window + Vulkan compute raycaster (see spec/ and docs/measured.md). */
#include <SDL3/SDL.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "core/pngw.h"
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ; /* argv-spawned browser jobs inherit the environment */

#include "cimgui.h"
#include "core/camera.h"
#include "core/odbrowse.h"
#include "core/transfer.h"
#include "core/volume.h"
#include "core/input.h"
#include "core/mview.h"
#include "core/screenshot.h"
#include "core/segstore.h"
#include "core/tracer.h"
#include "core/bsurf.h"
#include "core/flatten.h"
#include "core/labelvol.h"
#include "core/regvol.h"
#include "core/cpuvol.h"
#include "core/segtrace.h"
#include "core/inklive.h"
#include "core/stats.h"
#include "core/tifxyz.h"
#include "core/umbilicus.h"
#include "render/render.h"
#include "vk/vkctx.h"

#ifndef R3D_SPV_DIR
#define R3D_SPV_DIR "spv" /* release fallback: exe-relative */
#endif
#ifndef R3D_C5D_REV
#define R3D_C5D_REV "unknown"
#endif

#define MOUSE_SENS 0.0025f
#define ORBIT_SENS 0.006f
#define BASE_SPEED 0.4f /* volume units per second */

enum { CAM_ORBIT = 0, CAM_FLY = 1 };

static const char PHERC1218_SOURCE[] =
    "https://dl.ash2txt.org/community-uploads/forrest/exports/PHerc1218/"
    "20250521120456-8.640um-1.2m-116keV-masked.zarr";
static const char PHERC1218_SHARDS[] =
    "https://dl.ash2txt.org/community-uploads/forrest/exports/PHerc1218/"
    "20250521120456-8.640um-1.2m-116keV-masked.zarr/0/c";
enum { PHERC1218_NX = 8192, PHERC1218_NY = 8192, PHERC1218_NZ = 23552 };

/* Full-surface ink map cache: for a SAVED segment the surface is frozen,
 * so 2.5D ink is deterministic — computed once (tiled, native resolution)
 * and cached beside the segment store as <store>/<name>.inkmap. Live
 * per-view inference remains only for changing (tracer) grids. */
/* Full-map tiling: each tile request carries a context margin so the model
 * sees real receptive field at tile edges, and finished tiles are blended
 * into the map with a linear ramp over the shared margin (prob/wsum
 * accumulation, the same scheme the server uses between its own patches).
 * The old 2-cell margin with a hard stitch left visible tile seams. */
#define INKMAP_MARGIN_CELLS 16u
static uint32_t inkmap_tile_cells(uint32_t up) {
  uint32_t its = R3D_INKLIVE_MAX_PX / (up ? up : 1);
  uint32_t m2 = 2u * INKMAP_MARGIN_CELLS;
  return its > m2 + 8u ? its - m2 : 8u;
}

/* Flattened-pane supervision mask (ink fine-tuning loop): a grid-space u8
 * class image at ink-map resolution ((gw-1)*up x (gh-1)*up), painted with
 * the mouse on the flattened pane. 0 unlabeled, 1 background (clearly no
 * ink), 2 ink. Exported as mask.png in the exact pixel grid of ink.png and
 * the rendseg stack, so stack + ink + mask register for fine-tuning. */
static uint8_t *g_smask = NULL;
static uint32_t g_smask_w, g_smask_h, g_smask_up, g_smask_gw, g_smask_gh;
static uint64_t g_smask_nvalid;
static bool g_smask_show = true, g_smask_paint = false;
static bool g_smask_gpu_dirty = false, g_smask_disk_dirty = false;
static int g_smask_class = 2; /* 2 ink, 1 background, 0 erase */
static float g_smask_brush = 2.0f; /* grid cells */
static bool g_smask_stroke = false;
static double g_smask_prev[2];

#define SMASK_MAGIC 0x314B534Du /* 'MSK1' */

static int smask_save_file(const char *path) {
  if (!g_smask) return -1;
  char tmp[1240];
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  uint32_t hdr[8] = {SMASK_MAGIC, g_smask_w, g_smask_h, g_smask_up, g_smask_gw,
                     g_smask_gh, (uint32_t)(g_smask_nvalid & 0xffffffffu),
                     (uint32_t)(g_smask_nvalid >> 32)};
  int ok = fwrite(hdr, sizeof hdr, 1, f) == 1 &&
           fwrite(g_smask, 1, (size_t)g_smask_w * g_smask_h, f) ==
               (size_t)g_smask_w * g_smask_h;
  if (fclose(f) != 0) ok = 0;
  if (!ok) {
    remove(tmp);
    return -1;
  }
  remove(path);
  return rename(tmp, path);
}

static bool smask_load_file(const char *path, uint32_t gw, uint32_t gh, uint64_t nvalid) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  uint32_t hdr[8];
  bool ok = false;
  if (fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == SMASK_MAGIC && hdr[1] && hdr[2] &&
      hdr[1] < 32768 && hdr[2] < 32768 && hdr[4] == gw && hdr[5] == gh &&
      ((uint64_t)hdr[6] | ((uint64_t)hdr[7] << 32)) == nvalid) {
    uint8_t *m = malloc((size_t)hdr[1] * hdr[2]);
    if (m && fread(m, 1, (size_t)hdr[1] * hdr[2], f) == (size_t)hdr[1] * hdr[2]) {
      free(g_smask);
      g_smask = m;
      g_smask_w = hdr[1];
      g_smask_h = hdr[2];
      g_smask_up = hdr[3] ? hdr[3] : 1;
      g_smask_gw = gw;
      g_smask_gh = gh;
      g_smask_nvalid = nvalid;
      g_smask_gpu_dirty = true;
      g_smask_disk_dirty = false;
      ok = true;
    } else
      free(m);
  }
  fclose(f);
  return ok;
}

/* stamp a class disc at grid coords (u, v), radius in grid cells */
static void smask_stamp(double u, double v, double rad_cells, uint8_t cls) {
  if (!g_smask) return;
  double cx = u * g_smask_up, cy = v * g_smask_up, r = rad_cells * g_smask_up;
  int64_t x0 = (int64_t)floor(cx - r), x1 = (int64_t)ceil(cx + r);
  int64_t y0 = (int64_t)floor(cy - r), y1 = (int64_t)ceil(cy + r);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= (int64_t)g_smask_w) x1 = (int64_t)g_smask_w - 1;
  if (y1 >= (int64_t)g_smask_h) y1 = (int64_t)g_smask_h - 1;
  double r2 = r * r;
  for (int64_t y = y0; y <= y1; y++)
    for (int64_t x = x0; x <= x1; x++) {
      double dx = (double)x - cx, dy = (double)y - cy;
      if (dx * dx + dy * dy > r2) continue;
      g_smask[(size_t)y * g_smask_w + (size_t)x] = cls;
    }
  g_smask_gpu_dirty = g_smask_disk_dirty = true;
}

/* autosave (when the segment has a traced dir) + free + clear the overlay;
 * call BEFORE mv_seg / the active-segment name changes */
static void smask_drop(r3d_renderer *renderer, const char *segname) {
  if (g_smask && g_smask_disk_dirty && segname && segname[0]) {
    char dir[320];
    snprintf(dir, sizeof dir, "cache/traced/%s", segname);
    struct stat st;
    if (stat(dir, &st) == 0) {
      char path[400];
      snprintf(path, sizeof path, "%s/mask.inkmask", dir);
      if (smask_save_file(path) == 0)
        printf("inklive: supervision mask saved -> %s\n", path);
    }
  }
  free(g_smask);
  g_smask = NULL;
  g_smask_gpu_dirty = g_smask_disk_dirty = false;
  g_smask_stroke = false;
  if (renderer) r3d_surfmask_clear(renderer);
}

/* SLIM flatten job: recompute a near-isometric parameterization of the
 * ACTIVE segment on a worker thread (core/flatten.c), resample onto a
 * regular lattice, and save the result as cache/traced/<name>-flat — an
 * ordinary tifxyz segment every downstream consumer uses unchanged. */
static _Atomic int g_flat_state; /* 0 idle, 1 running, 2 done, 3 failed */
static pthread_t g_flat_th;
static bool g_flat_th_up = false;
static float *g_flat_in;  /* snapshot of the segment grid (worker-owned) */
static uint32_t g_flat_in_w, g_flat_in_h;
static double g_flat_step;
static float *g_flat_out; /* resampled grid (worker result) */
static uint32_t g_flat_out_w, g_flat_out_h;
static r3d_flatten_stats g_flat_stats;
static char g_flat_name[240];

static void *flat_worker(void *arg) {
  (void)arg;
  float *uv = malloc((size_t)g_flat_in_w * g_flat_in_h * 2 * sizeof *uv);
  int st = 3;
  if (uv &&
      r3d_flatten_slim(g_flat_in, g_flat_in_w, g_flat_in_h, g_flat_step, 200, uv,
                       &g_flat_stats) == 0 &&
      r3d_flatten_resample(g_flat_in, uv, g_flat_in_w, g_flat_in_h, g_flat_step,
                           &g_flat_out, &g_flat_out_w, &g_flat_out_h) == 0)
    st = 2;
  free(uv);
  free(g_flat_in);
  g_flat_in = NULL;
  atomic_store(&g_flat_state, st);
  return NULL;
}

/* write the resampled grid as a normal segment via the tracer exporter */
static int flat_save(const char *dir, const float *xyz, uint32_t w, uint32_t h,
                     double step) {
  if (mkdir(dir, 0755) != 0 && errno != EEXIST) return -1;
  r3d_tracer t = {0};
  t.W = w;
  t.H = h;
  t.cfg.step = step;
  t.cfg.max_ring = w > 50 ? (w - 50) / 2 : 4;
  uint64_t n = (uint64_t)w * h;
  t.pos = calloc(n * 3, sizeof *t.pos);
  t.state = calloc(n, 1);
  t.conf = calloc(n, sizeof *t.conf);
  t.gen_of = calloc(n, sizeof *t.gen_of);
  int rc = -1;
  if (t.pos && t.state && t.conf && t.gen_of) {
    pthread_mutex_init(&t.mu, NULL);
    for (uint64_t k = 0; k < n; k++) {
      if (xyz[k * 3] < 0.0f) continue;
      for (int a = 0; a < 3; a++) t.pos[k * 3 + (uint64_t)a] = (double)xyz[k * 3 + (uint64_t)a];
      t.state[k] = R3D_TR_SET;
      t.conf[k] = 1.0f;
      t.gen_of[k] = 1;
      t.nset++;
    }
    rc = t.nset ? r3d_tracer_save(&t, dir, 0.0f, false) : -1;
  }
  r3d_tracer_free(&t);
  return rc;
}

/* Automatic post-save flattening: the 2.5D ink model is trained on
 * flattened segments, so every harvested trace is SLIM-flattened before
 * it enters the store (and thus before the ink queue sees it). Loads the
 * artifact just written by r3d_tracer_save (so fill/cutoff re-seating is
 * included), flattens + resamples, and saves <rawdir>-flat. The raw dir
 * keeps the tracer.json round trip for tracecli resume; only the flat
 * segment is packed/inked. Returns 0 and fills flatdir on success; on
 * any failure the caller falls back to the raw segment. */
static int trace_flatten_save(const char *rawdir, char *flatdir, size_t fn) {
  r3d_tifxyz s = {0};
  if (r3d_tifxyz_load(&s, rawdir) != 0 || s.w < 3 || s.h < 3) {
    r3d_tifxyz_free(&s);
    return -1;
  }
  double step = s.sx > 0.0f ? 1.0 / (double)s.sx : 20.0;
  float *uv = malloc((size_t)s.w * s.h * 2 * sizeof *uv);
  float *rxyz = NULL;
  uint32_t rw = 0, rh = 0;
  r3d_flatten_stats fst = {0};
  int rc = -1;
  if (uv && r3d_flatten_slim(s.xyz, s.w, s.h, step, 200, uv, &fst) == 0 &&
      r3d_flatten_resample(s.xyz, uv, s.w, s.h, step, &rxyz, &rw, &rh) == 0) {
    snprintf(flatdir, fn, "%s-flat", rawdir);
    rc = flat_save(flatdir, rxyz, rw, rh, step);
    if (rc == 0)
      printf("tracer: flattened %s -> %s (%ux%u, stretch %.4f -> %.4f, "
             "%u iters)\n", rawdir, flatdir, rw, rh, fst.stretch0,
             fst.stretch1, fst.iters);
  }
  free(rxyz);
  free(uv);
  r3d_tifxyz_free(&s);
  return rc;
}

static int inkmap_save(const char *path, const float *m, uint32_t w, uint32_t h,
                       uint32_t up, uint32_t gw, uint32_t gh, uint64_t nvalid) {
  char tmp[1240];
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  /* v2 header carries the source grid's dims + valid-point count so a
   * map is rejected when the segment under the same name was replaced
   * (discard + retrace): stale ink must never dress a new surface */
  uint32_t hdr[8] = {0x494B4E32u /* '2NKI' */, w, h, up, gw, gh,
                     (uint32_t)(nvalid & 0xffffffffu), (uint32_t)(nvalid >> 32)};
  int ok = fwrite(hdr, sizeof hdr, 1, f) == 1 &&
           fwrite(m, sizeof *m, (size_t)w * h, f) == (size_t)w * h;
  fclose(f);
  if (!ok) {
    remove(tmp);
    return -1;
  }
  remove(path);
  return rename(tmp, path);
}

static float *inkmap_load(const char *path, uint32_t *w, uint32_t *h, uint32_t *up,
                          uint32_t gw, uint32_t gh, uint64_t nvalid) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  uint32_t hdr[8];
  float *m = NULL;
  if (fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x494B4E32u && hdr[1] &&
      hdr[2] && hdr[1] < 32768 && hdr[2] < 32768) {
    uint64_t nv = (uint64_t)hdr[6] | ((uint64_t)hdr[7] << 32);
    if (hdr[4] != gw || hdr[5] != gh || nv != nvalid) {
      /* same name, different surface: the segment was replaced */
      fclose(f);
      printf("inklive: cached ink map is for a different surface "
             "(grid %ux%u/%llu vs %ux%u/%llu) - discarding it\n",
             hdr[4], hdr[5], (unsigned long long)nv, gw, gh,
             (unsigned long long)nvalid);
      remove(path);
      return NULL;
    }
    m = malloc((size_t)hdr[1] * hdr[2] * sizeof *m);
    if (m && fread(m, sizeof *m, (size_t)hdr[1] * hdr[2], f) !=
                 (size_t)hdr[1] * hdr[2]) {
      free(m);
      m = NULL;
    }
    if (m) {
      *w = hdr[1];
      *h = hdr[2];
      *up = hdr[3] ? hdr[3] : 1;
    }
  }
  fclose(f);
  return m;
}

/* Write the ink artifacts INTO a segment's tifxyz dir so the detection
 * travels with the surface: ink.inkmap (float, reload-exact) + ink.png
 * (8-bit view, scrollprize-style). Skips silently when the dir does not
 * exist (segments that came from elsewhere). */
static void inkmap_save_segdir(const char *name, const float *m, uint32_t w,
                               uint32_t h, uint32_t up, uint32_t gw, uint32_t gh,
                               uint64_t nvalid, bool verso);

/* CT sampler for the boundary-surface grower: cpuvol over the bricks tree,
 * base level (the papyrus/void edge is a voxel-scale feature) */
static uint8_t bsurf_ct_sample(void *ctx, int64_t x, int64_t y, int64_t z) {
  return r3d_cpuvol_at((r3d_cpuvol *)ctx, 0, (double)x, (double)y, (double)z);
}

/* 3D labelling: paint class ids (papyrus / ink / recto / ...) with the mouse
 * in the plane panes; the renderer mirrors the CPU volume into a slot-
 * parallel atlas (r3d_bricks_labels_sync), and C5L1 label bricks persist it
 * losslessly. State is file-scope because the paint gesture (event loop),
 * the panel (GUI), the per-pane routing and the dataset-loop cleanup all
 * touch it. */
static r3d_labelvol g_lblv;
static bool g_lbl_init = false;
static bool g_lbl_show = true, g_lbl_paint = false;
static int g_lbl_class = 1;
static float g_lbl_radius = 4.0f;
static char g_lbl_dir[640] = "";
/* last persistence outcome, shown in the labels panel: a failed save must be
 * visible and retryable, never silently swallowed */
static char g_lbl_status[200] = "";
static bool g_lbl_stroke = false; /* a drag is in progress */
static double g_lbl_prev[3];

static uint32_t lblsrc_gen(void *u, uint32_t level, uint32_t bx, uint32_t by, uint32_t bz) {
  return r3d_labelvol_gen((const r3d_labelvol *)u, level, bx, by, bz);
}
static void lblsrc_fetch(void *u, uint32_t level, uint32_t bx, uint32_t by, uint32_t bz,
                         uint8_t *out) {
  r3d_labelvol_fetch((const r3d_labelvol *)u, level, bx, by, bz, out);
}

/* Volume registration: a second scan of the same scroll overlaid through an
 * affine pull map (core/regvol.c) and displayed as a green/magenta fuse in
 * every pane. GUI mirrors hold the interactive deltas in friendly units. */
static r3d_regvol g_reg;
static bool g_reg_open = false;
static bool g_reg_show = true, g_reg_flat = false;
static float g_reg_alpha = 0.5f;
static char g_reg_root[640] = "";
static char g_reg_json[640] = "";
static float g_reg_tr[3], g_reg_rot[3]; /* voxels / degrees */
static float g_reg_scale = 1.0f;
static float g_reg_um_fix = 0.0f, g_reg_um_mov = 0.0f; /* voxel pitch, um */
static int g_reg_refmode = 0; /* 0 measure NCC, 1 rigid, 2 affine */
static int g_reg_reflevel = 1;
static bool g_reg_busy = false;
static double g_reg_ncc0 = -2.0, g_reg_ncc1 = -2.0;

static void reg_gui_reset_mirrors(void) {
  memset(g_reg_tr, 0, sizeof g_reg_tr);
  memset(g_reg_rot, 0, sizeof g_reg_rot);
  g_reg_scale = 1.0f;
}

static void reg_gui_apply_mirrors(void) {
  /* the renderer's sync worker reads the deltas through r3d_regvol_pull:
   * write them under the same lock */
  pthread_mutex_lock(&g_reg.mu);
  for (int a = 0; a < 3; a++) {
    g_reg.d_tr[a] = (double)g_reg_tr[a];
    g_reg.d_rot[a] = (double)g_reg_rot[a] * 0.017453292519943295;
  }
  g_reg.d_lscale = log(g_reg_scale > 0.01f ? (double)g_reg_scale : 0.01);
  pthread_mutex_unlock(&g_reg.mu);
  r3d_regvol_bump(&g_reg);
}

/* open (or replace) the registration moving volume: accepts the LOD root or
 * its manifest.json path. Scale seeding, best first: the voxel-pitch ratio
 * when both paths carry a "...um" token (bucket names do — crops differ
 * between scans, so this beats the shape ratio), else the shape ratio.
 * Shared by the panel and the data browser. */
static bool reg_open_moving(r3d_renderer *renderer, const char *root, const uint32_t fd[3],
                            const char *fixed_path) {
  char rr[640];
  snprintf(rr, sizeof rr, "%s", root);
  size_t rl = strlen(rr);
  if (rl > 14 && strcmp(rr + rl - 14, "/manifest.json") == 0) rr[rl - 14] = 0;
  if (g_reg_open) { /* replace the current moving volume: stop the renderer's
                     * sync worker before its source dies under it */
    g_reg_open = false;
    g_reg_busy = false;
    r3d_bricks_regatlas_detach(renderer);
    r3d_regvol_close(&g_reg);
  }
  if (r3d_regvol_open(&g_reg, rr, fd) != 0) return false;
  r3d_label_src ls = {r3d_regvol_srcgen, r3d_regvol_srcfetch, &g_reg};
  if (r3d_bricks_regatlas(renderer, &ls) != 0) {
    r3d_regvol_close(&g_reg); /* atlas never attached: no worker to stop */
    return false;
  }
  g_reg_open = true;
  g_reg_show = true;
  reg_gui_reset_mirrors();
  g_reg_ncc0 = g_reg_ncc1 = -2.0;
  snprintf(g_reg_root, sizeof g_reg_root, "%s", rr);
  double um_m = r3d_regvol_parse_um(rr);
  double um_f = fixed_path ? r3d_regvol_parse_um(fixed_path) : 0.0;
  if (um_m > 0.0) g_reg_um_mov = (float)um_m;
  if (um_f > 0.0) g_reg_um_fix = (float)um_f;
  if (um_m > 0.0 && um_f > 0.0) {
    if (fabs(um_f / um_m - 1.0) > 0.001) r3d_regvol_set_scale(&g_reg, um_f / um_m);
    printf("regvol: scale seeded from voxel pitch %.4gum / %.4gum = %.4f\n", um_f, um_m,
           um_f / um_m);
  } else {
    double sr = ((double)g_reg.mv.nx / (double)fd[0] + (double)g_reg.mv.ny / (double)fd[1] +
                 (double)g_reg.mv.nz / (double)fd[2]) /
                3.0;
    if (sr > 0.0 && fabs(sr - 1.0) > 0.01) r3d_regvol_set_scale(&g_reg, sr);
  }
  return true;
}

/* stamp spheres along the drag segment so fast strokes stay solid */
static void lbl_stroke_to(const double A[3], double radius, uint8_t cls) {
  if (g_lbl_stroke) {
    double d[3] = {A[0] - g_lbl_prev[0], A[1] - g_lbl_prev[1], A[2] - g_lbl_prev[2]};
    double dist = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    double step = radius * 0.5 > 0.75 ? radius * 0.5 : 0.75;
    int nstep = (int)ceil(dist / step);
    for (int k = 1; k <= nstep; k++) {
      double t = (double)k / (double)nstep;
      double q[3] = {g_lbl_prev[0] + d[0] * t, g_lbl_prev[1] + d[1] * t,
                     g_lbl_prev[2] + d[2] * t};
      r3d_labelvol_paint(&g_lblv, q, radius, cls);
    }
  } else
    r3d_labelvol_paint(&g_lblv, A, radius, cls);
  g_lbl_stroke = true;
  memcpy(g_lbl_prev, A, 3 * sizeof(double));
}

/* Fit a surface through hand-placed points with NO ordering assumptions:
 * the user hops between viewers, outlines roughly, then densifies inside.
 * PCA plane through all points -> (u,v) parameterization -> grid at the
 * requested step -> moving-least-squares height per node (order-1 fit,
 * Gaussian-weighted) from whatever points lie nearby, in any click order.
 * Nodes with no nearby support stay INVALID (x = -1), so the surface
 * spans exactly the clicked territory. Densifies until the grid clears
 * the min-size save gate. Works for patches up to roughly a quarter wrap
 * of curl (the plane parameterization folds beyond that). */
static double *msurf_fit(const double (*pts)[3], uint32_t n, double step,
                         uint32_t *ow, uint32_t *oh) {
  if (n < 4) return NULL;
  double c[3] = {0, 0, 0};
  for (uint32_t i = 0; i < n; i++)
    for (int a2 = 0; a2 < 3; a2++) c[a2] += pts[i][a2];
  for (int a2 = 0; a2 < 3; a2++) c[a2] /= n;
  double C[3][3] = {{0}};
  for (uint32_t i = 0; i < n; i++)
    for (int a2 = 0; a2 < 3; a2++)
      for (int b2 = 0; b2 < 3; b2++)
        C[a2][b2] += (pts[i][a2] - c[a2]) * (pts[i][b2] - c[b2]);
  /* dominant two axes by power iteration (+ deflation) */
  double e1[3] = {1, 0, 0}, e2[3] = {0, 1, 0};
  for (int it = 0; it < 48; it++) {
    double v[3] = {0, 0, 0};
    for (int a2 = 0; a2 < 3; a2++)
      for (int b2 = 0; b2 < 3; b2++) v[a2] += C[a2][b2] * e1[b2];
    double l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l < 1e-12) break;
    for (int a2 = 0; a2 < 3; a2++) e1[a2] = v[a2] / l;
  }
  double l1 = 0.0;
  {
    double v[3] = {0, 0, 0};
    for (int a2 = 0; a2 < 3; a2++)
      for (int b2 = 0; b2 < 3; b2++) v[a2] += C[a2][b2] * e1[b2];
    l1 = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }
  for (int a2 = 0; a2 < 3; a2++) /* deflate */
    for (int b2 = 0; b2 < 3; b2++) C[a2][b2] -= l1 * e1[a2] * e1[b2];
  for (int it = 0; it < 48; it++) {
    double v[3] = {0, 0, 0};
    for (int a2 = 0; a2 < 3; a2++)
      for (int b2 = 0; b2 < 3; b2++) v[a2] += C[a2][b2] * e2[b2];
    double d = v[0] * e1[0] + v[1] * e1[1] + v[2] * e1[2];
    for (int a2 = 0; a2 < 3; a2++) v[a2] -= d * e1[a2]; /* orthogonalize */
    double l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l < 1e-12) break;
    for (int a2 = 0; a2 < 3; a2++) e2[a2] = v[a2] / l;
  }
  double nrm[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                   e1[0] * e2[1] - e1[1] * e2[0]};
  /* project every point into the plane frame */
  double *uvw = malloc((size_t)n * 3 * sizeof *uvw);
  if (!uvw) return NULL;
  double u0 = 1e30, u1 = -1e30, v0 = 1e30, v1 = -1e30;
  for (uint32_t i = 0; i < n; i++) {
    double d[3] = {pts[i][0] - c[0], pts[i][1] - c[1], pts[i][2] - c[2]};
    double pu = d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2];
    double pv = d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2];
    double pw = d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2];
    uvw[i * 3] = pu;
    uvw[i * 3 + 1] = pv;
    uvw[i * 3 + 2] = pw;
    if (pu < u0) u0 = pu;
    if (pu > u1) u1 = pu;
    if (pv < v0) v0 = pv;
    if (pv > v1) v1 = pv;
  }
  if (u1 - u0 < 2.0 || v1 - v0 < 2.0) {
    free(uvw);
    return NULL; /* degenerate: points nearly collinear */
  }
  double *grid = NULL;
  for (int densify = 0; densify < 4 && !grid; densify++, step *= 0.5) {
    uint32_t W = (uint32_t)((u1 - u0) / step) + 2;
    uint32_t H = (uint32_t)((v1 - v0) / step) + 2;
    if (W < 2) W = 2;
    if (H < 2) H = 2;
    if (W > 1024) W = 1024;
    if (H > 1024) H = 1024;
    if ((uint64_t)W * H < 128 && densify < 3) continue;
    grid = malloc((size_t)W * H * 3 * sizeof *grid);
    if (!grid) break;
    /* support radius: generous around sparse clicks, tight when dense */
    double h2 = 3.0 * step;
    double area_per = (u1 - u0) * (v1 - v0) / (double)n;
    double spac = sqrt(area_per > 1.0 ? area_per : 1.0);
    if (spac * 1.5 > h2) h2 = spac * 1.5;
    uint32_t nvalid2 = 0;
    for (uint32_t gy = 0; gy < H; gy++)
      for (uint32_t gx = 0; gx < W; gx++) {
        double gu = u0 + (u1 - u0) * (double)gx / (double)(W - 1);
        double gv = v0 + (v1 - v0) * (double)gy / (double)(H - 1);
        /* order-1 MLS: w ~ a + b*du + c*dv, Gaussian weights */
        double A[9] = {0}, B[3] = {0};
        double wsum = 0.0;
        for (uint32_t i = 0; i < n; i++) {
          double du = uvw[i * 3] - gu, dv = uvw[i * 3 + 1] - gv;
          double d2 = du * du + dv * dv;
          if (d2 > 9.0 * h2 * h2) continue;
          double wgt = exp(-d2 / (2.0 * h2 * h2));
          double bx[3] = {1.0, du, dv};
          for (int r2 = 0; r2 < 3; r2++) {
            for (int c2 = 0; c2 < 3; c2++) A[r2 * 3 + c2] += wgt * bx[r2] * bx[c2];
            B[r2] += wgt * bx[r2] * uvw[i * 3 + 2];
          }
          wsum += wgt;
        }
        double *gp = grid + ((size_t)gy * W + gx) * 3;
        if (wsum < 0.35) { /* no nearby clicks: outside the surface */
          gp[0] = gp[1] = gp[2] = -1.0;
          continue;
        }
        /* solve the 3x3 (fallback to weighted mean when ill-posed) */
        double w_off;
        double det = A[0] * (A[4] * A[8] - A[5] * A[7]) -
                     A[1] * (A[3] * A[8] - A[5] * A[6]) +
                     A[2] * (A[3] * A[7] - A[4] * A[6]);
        if (fabs(det) > 1e-9 * (fabs(A[0]) + 1.0)) {
          w_off = (B[0] * (A[4] * A[8] - A[5] * A[7]) -
                   A[1] * (B[1] * A[8] - A[5] * B[2]) +
                   A[2] * (B[1] * A[7] - A[4] * B[2])) /
                  det;
        } else {
          w_off = B[0] / A[0];
        }
        for (int a2 = 0; a2 < 3; a2++)
          gp[a2] = c[a2] + gu * e1[a2] + gv * e2[a2] + w_off * nrm[a2];
        nvalid2++;
      }
    if (nvalid2 < 100 && densify < 3) {
      free(grid);
      grid = NULL;
      continue;
    }
    if (nvalid2 < 16) {
      free(grid);
      grid = NULL;
      break;
    }
    *ow = W;
    *oh = H;
  }
  free(uvw);
  return grid;
}

/* Trace segment name at FINISH time: yyyymmddhhmmss (sorts
 * chronologically = alphabetically), suffixed -2, -3... when several
 * traces finish within one second. */
static void trace_dir_now(char *out, size_t n) {
  time_t t = time(NULL);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char base[64];
  strftime(base, sizeof base, "%Y%m%d%H%M%S", &tmv);
  snprintf(out, n, "cache/traced/%s", base);
  struct stat st;
  for (int k = 2; stat(out, &st) == 0 && k < 100; k++) {
    char tmp[96];
    snprintf(tmp, sizeof tmp, "cache/traced/%s-%d", base, k);
    snprintf(out, n, "%s", tmp);
  }
}

static int inkmap_save(const char *path, const float *m, uint32_t w, uint32_t h,
                       uint32_t up, uint32_t gw, uint32_t gh, uint64_t nvalid);

static void inkmap_save_segdir(const char *name, const float *m, uint32_t w,
                               uint32_t h, uint32_t up, uint32_t gw, uint32_t gh,
                               uint64_t nvalid, bool verso) {
  char dir[320];
  snprintf(dir, sizeof dir, "cache/traced/%s", name);
  struct stat st;
  if (stat(dir, &st) != 0) return; /* not one of our traced segments */
  const char *side = verso ? "-verso" : "";
  char path[400];
  snprintf(path, sizeof path, "%s/ink%s.inkmap", dir, side);
  inkmap_save(path, m, w, h, up, gw, gh, nvalid);
  uint8_t *px = malloc((size_t)w * h);
  if (px) {
    for (size_t k = 0; k < (size_t)w * h; k++) {
      float v = m[k] * 255.0f;
      px[k] = (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
    }
    snprintf(path, sizeof path, "%s/ink%s.png", dir, side);
    if (r3d_png_write_gray(path, px, w, h) == 0)
      printf("inklive: ink image -> %s\n", path);
    free(px);
  }
}

static int annotation_z0(int z, uint32_t nz, uint32_t depth) {
  int64_t z0 = (int64_t)z - (int64_t)depth / 2;
  int64_t max = (int64_t)nz - depth;
  if (max < 0) max = 0;
  if (z0 < 0) z0 = 0;
  if (z0 > max) z0 = max;
  return (int)z0;
}

static bool annotation_pick(const r3d_camera *cam, r3d_v3 right, r3d_v3 up, r3d_v3 fwd,
                            r3d_m3 model, r3d_v3 vol_t, float sx, float sy, int view_w,
                            int view_h, uint32_t W, uint32_t H, uint32_t D, int64_t x0,
                            int64_t y0, int64_t z0, int z, double *x, double *y) {
  if (view_w <= 0 || view_h <= 0 || !W || !H || !D || !x || !y) return false;
  float ndcx = 2.0f * (sx + 0.5f) / (float)view_w - 1.0f;
  float ndcy = 1.0f - 2.0f * (sy + 0.5f) / (float)view_h;
  r3d_v3 dir = v3_norm(v3_add(fwd, v3_add(v3_scale(right, ndcx), v3_scale(up, ndcy))));
  float ey = (float)H / (float)W, ez = (float)D / (float)W;
  r3d_v3 center = v3(0.5f, ey * 0.5f, ez * 0.5f);
  r3d_v3 ro = v3_add(m3_tmul(model, v3_sub(v3_sub(cam->pos, vol_t), center)), center);
  r3d_v3 rd = m3_tmul(model, dir);
  if (fabsf(rd.z) < 1e-8f) return false;
  float plane = (float)((int64_t)z - z0) / (float)W;
  float t = (plane - ro.z) / rd.z;
  if (t < 0.0f) return false;
  r3d_v3 p = v3_add(ro, v3_scale(rd, t));
  if (p.x < 0.0f || p.x >= 1.0f || p.y < 0.0f || p.y >= ey) return false;
  *x = (double)x0 + (double)p.x * W;
  *y = (double)y0 + (double)p.y * W;
  return *x >= 0.0 && *x < (double)W && *y >= 0.0 && *y < (double)H;
}

static bool annotation_project(const r3d_umbilicus_point *point, const r3d_camera *cam,
                               r3d_v3 right, r3d_v3 up, r3d_v3 fwd, r3d_m3 model,
                               r3d_v3 vol_t, int view_w, int view_h, uint32_t W, uint32_t H,
                               uint32_t D, int64_t x0, int64_t y0, int64_t z0, ImVec2 *screen) {
  if (!point || !screen || view_w <= 0 || view_h <= 0) return false;
  float ey = (float)H / (float)W, ez = (float)D / (float)W;
  r3d_v3 center = v3(0.5f, ey * 0.5f, ez * 0.5f);
  r3d_v3 p = v3((float)(point->x - (double)x0) / (float)W,
                (float)(point->y - (double)y0) / (float)W,
                (float)(point->z - (double)z0) / (float)W);
  r3d_v3 world = v3_add(v3_add(m3_mul(model, v3_sub(p, center)), center), vol_t);
  r3d_v3 d = v3_sub(world, cam->pos);
  float rz = v3_dot(d, fwd), rl = v3_len(right), ul = v3_len(up);
  if (rz <= 0.0f || rl <= 0.0f || ul <= 0.0f) return false;
  float nx = v3_dot(d, v3_scale(right, 1.0f / rl)) / (rz * rl);
  float ny = v3_dot(d, v3_scale(up, 1.0f / ul)) / (rz * ul);
  screen->x = (nx + 1.0f) * 0.5f * (float)view_w;
  screen->y = (1.0f - ny) * 0.5f * (float)view_h;
  return screen->x >= 0.0f && screen->x < (float)view_w && screen->y >= 0.0f &&
         screen->y < (float)view_h;
}

static int save_umbilicus(r3d_umbilicus *u, const char *path, char status[256]) {
  if (r3d_umbilicus_save(u, path, PHERC1218_SOURCE, PHERC1218_NZ, PHERC1218_NY,
                         PHERC1218_NX) != 0) {
    snprintf(status, 256, "save failed: %s", strerror(errno));
    return -1;
  }
  snprintf(status, 256, "saved %zu point%s", u->count, u->count == 1 ? "" : "s");
  return 0;
}

/* Cached segment/plane intersection polylines (world uv + grid ij pairs). */
typedef struct mv_lines {
  float *w, *g; /* 4 floats per segment each */
  uint32_t n, cap;
} mv_lines;

static void mv_lines_emit(void *ud, float wu0, float wv0, float wu1, float wv1, float gi0,
                          float gj0, float gi1, float gj1) {
  mv_lines *l = ud;
  if (l->n == l->cap) {
    uint32_t nc = l->cap ? l->cap * 2 : 4096;
    float *nw = realloc(l->w, (size_t)nc * 4 * sizeof *nw);
    if (nw) l->w = nw;
    float *ng = realloc(l->g, (size_t)nc * 4 * sizeof *ng);
    if (ng) l->g = ng;
    if (!nw || !ng) return; /* drop segments on OOM */
    l->cap = nc;
  }
  float *w4 = l->w + (size_t)l->n * 4, *g4 = l->g + (size_t)l->n * 4;
  w4[0] = wu0;
  w4[1] = wv0;
  w4[2] = wu1;
  w4[3] = wv1;
  g4[0] = gi0;
  g4[1] = gj0;
  g4[2] = gi1;
  g4[3] = gj1;
  l->n++;
}

/* Draw a cached segment soup as chained polylines: marching squares emits
 * mostly head-to-tail runs, so consecutive segments sharing an endpoint merge
 * into one AddPolyline, sub-pixel steps collapse into the previous point, and
 * chains whose bounds miss the pane are dropped before tessellation. The
 * per-segment AddLine version tessellated + uploaded thousands of AA quads
 * every frame regardless of zoom (~25% of the render thread on the GP
 * banner). */
static void mv_draw_lines(ImDrawList *draw, const r3d_mview *v, const float *seg, uint32_t n,
                          ImU32 col, float thick, float collapse_px2) {
  static ImVec2 *pts;
  static uint32_t cap;
  if (!n) return;
  if (cap < n + 2) {
    uint32_t nc = n + 2;
    ImVec2 *np_ = realloc(pts, (size_t)nc * sizeof *np_);
    if (!np_) return;
    pts = np_;
    cap = nc;
  }
  float pxmin = (float)v->px, pymin = (float)v->py;
  float pxmax = pxmin + (float)v->pw, pymax = pymin + (float)v->ph;
  uint32_t np = 0;
  float bx0 = 0.0f, by0 = 0.0f, bx1 = 0.0f, by1 = 0.0f; /* chain screen bbox */
  float lu = 0.0f, lv = 0.0f;                           /* chain tail, source space */
  for (uint32_t k = 0; k <= n; k++) {
    float nu = 0.0f, nv = 0.0f;
    bool brk = k == n;
    if (!brk) {
      const float *s4 = seg + (size_t)k * 4;
      if (np == 0) {
        float x, y;
        r3d_mv_project(v, (double)s4[0], (double)s4[1], &x, &y);
        pts[np++] = (ImVec2){x, y};
        bx0 = bx1 = x;
        by0 = by1 = y;
        nu = s4[2];
        nv = s4[3];
      } else if (fabsf(s4[0] - lu) < 1e-2f && fabsf(s4[1] - lv) < 1e-2f) {
        nu = s4[2];
        nv = s4[3];
      } else if (fabsf(s4[2] - lu) < 1e-2f && fabsf(s4[3] - lv) < 1e-2f) {
        nu = s4[0];
        nv = s4[1];
      } else {
        brk = true;
        k--; /* flush, then revisit this segment as a new chain start */
      }
    }
    if (brk) {
      if (np >= 2 && bx1 >= pxmin && bx0 < pxmax && by1 >= pymin && by0 < pymax)
        ImDrawList_AddPolyline(draw, pts, (int)np, col, thick, 0);
      np = 0;
      continue;
    }
    float x, y;
    r3d_mv_project(v, (double)nu, (double)nv, &x, &y);
    float dx = x - pts[np - 1].x, dy = y - pts[np - 1].y;
    if (np >= 2 && dx * dx + dy * dy < collapse_px2)
      pts[np - 1] = (ImVec2){x, y}; /* sub-pixel step: slide the chain tail */
    else
      pts[np++] = (ImVec2){x, y};
    if (x < bx0) bx0 = x;
    if (x > bx1) bx1 = x;
    if (y < by0) by0 = y;
    if (y > by1) by1 = y;
    lu = nu;
    lv = nv;
  }
}

/* Build the surf-view GPU grids from a tifxyz segment: RGBA32F coords
 * (w = valid) and per-vertex normals (bilinear-tangent cross product, vc3d
 * grid_normal). Returns malloc'd w*h*4 float pairs via out params. */
static int mv_build_grids(const r3d_tifxyz *s, float **coords_out, float **normals_out) {
  uint64_t n = (uint64_t)s->w * s->h;
  float *co = malloc(n * 4 * sizeof *co), *no = calloc(n * 4, sizeof *no);
  if (!co || !no) {
    free(co);
    free(no);
    return -1;
  }
  for (uint64_t k = 0; k < n; k++) {
    const float *p = s->xyz + k * 3;
    bool ok = r3d_tifxyz_valid(p);
    co[k * 4 + 0] = p[0];
    co[k * 4 + 1] = p[1];
    co[k * 4 + 2] = p[2];
    co[k * 4 + 3] = ok ? 1.0f : 0.0f;
  }
  for (uint32_t j = 0; j < s->h; j++)
    for (uint32_t i = 0; i < s->w; i++) {
      uint64_t k = (uint64_t)j * s->w + i;
      if (co[k * 4 + 3] < 0.5f) continue;
      /* central differences, shrinking to one-sided at edges/invalid */
      uint32_t i0 = i > 0 ? i - 1 : i, i1 = i + 1 < s->w ? i + 1 : i;
      uint32_t j0 = j > 0 ? j - 1 : j, j1 = j + 1 < s->h ? j + 1 : j;
      const float *pu0 = r3d_tifxyz_at(s, i0, j), *pu1 = r3d_tifxyz_at(s, i1, j);
      const float *pv0 = r3d_tifxyz_at(s, i, j0), *pv1 = r3d_tifxyz_at(s, i, j1);
      const float *pc_ = r3d_tifxyz_at(s, i, j);
      if (!r3d_tifxyz_valid(pu0)) pu0 = pc_;
      if (!r3d_tifxyz_valid(pu1)) pu1 = pc_;
      if (!r3d_tifxyz_valid(pv0)) pv0 = pc_;
      if (!r3d_tifxyz_valid(pv1)) pv1 = pc_;
      r3d_v3 tu = v3(pu1[0] - pu0[0], pu1[1] - pu0[1], pu1[2] - pu0[2]);
      r3d_v3 tv = v3(pv1[0] - pv0[0], pv1[1] - pv0[1], pv1[2] - pv0[2]);
      r3d_v3 nn = v3_cross(tu, tv);
      float l = v3_len(nn);
      if (l < 1e-6f) continue;
      no[k * 4 + 0] = nn.x / l;
      no[k * 4 + 1] = nn.y / l;
      no[k * 4 + 2] = nn.z / l;
    }
  *coords_out = co;
  *normals_out = no;
  return 0;
}

/* nearest valid surface grid point to a world position (coarse step-2 scan);
 * true when it's within vc3d's 100-voxel focus tolerance */
static bool mv_nearest_surface(const r3d_tifxyz *s, const double f[3], uint32_t out_ij[2]) {
  double best = 1e30;
  uint32_t bi = 0, bj = 0;
  for (uint32_t gj = 0; gj < s->h; gj += 2)
    for (uint32_t gi = 0; gi < s->w; gi += 2) {
      const float *sp = r3d_tifxyz_at(s, gi, gj);
      if (!r3d_tifxyz_valid(sp)) continue;
      double dx = (double)sp[0] - f[0], dy = (double)sp[1] - f[1], dz = (double)sp[2] - f[2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        bi = gi;
        bj = gj;
      }
    }
  out_ij[0] = bi;
  out_ij[1] = bj;
  return best < 100.0 * 100.0;
}

/* Rebuild the segment-aligned frames for the XZ/YZ panes around surface grid
 * point ij: normal from the per-vertex normal grid, reference tangent = the
 * grid +i direction, origins at the focus. False (bases untouched) when the
 * normal is degenerate. */
static bool mv_seg_align(const r3d_tifxyz *s, const float *normals, const double focus[3],
                         const uint32_t ij[2], float theta_deg, double pb[4][3][3],
                         double po[4][3]) {
  size_t k = ((size_t)ij[1] * s->w + ij[0]) * 4;
  double n[3] = {(double)normals[k], (double)normals[k + 1], (double)normals[k + 2]};
  uint32_t i0 = ij[0] > 0 ? ij[0] - 1 : ij[0];
  uint32_t i1 = ij[0] + 1 < s->w ? ij[0] + 1 : ij[0];
  const float *pa = r3d_tifxyz_at(s, i0, ij[1]);
  const float *pb_ = r3d_tifxyz_at(s, i1, ij[1]);
  const float *pc_ = r3d_tifxyz_at(s, ij[0], ij[1]);
  if (!r3d_tifxyz_valid(pa)) pa = pc_;
  if (!r3d_tifxyz_valid(pb_)) pb_ = pc_;
  double tref[3] = {(double)pb_[0] - (double)pa[0], (double)pb_[1] - (double)pa[1],
                    (double)pb_[2] - (double)pa[2]};
  double ba[3][3], bb[3][3];
  if (r3d_mv_seg_frames(n, tref, (double)theta_deg * 0.017453292519943295, ba, bb) != 0)
    return false;
  memcpy(pb[R3D_MV_XZ], ba, sizeof ba);
  memcpy(pb[R3D_MV_YZ], bb, sizeof bb);
  for (int a = 0; a < 3; a++) po[R3D_MV_XZ][a] = po[R3D_MV_YZ][a] = focus[a];
  return true;
}

static void mv_axis_reset(double pb[4][3][3], double po[4][3]) {
  for (int i = R3D_MV_XZ; i <= R3D_MV_YZ; i++) {
    r3d_mv_axis_basis(i, pb[i]);
    for (int a = 0; a < 3; a++) po[i][a] = 0.0;
  }
}

/* ---- multi-segment corpus (--segments <store-dir>): a background worker
 * decodes decimated grids + row bounds into a RAM-budgeted cache; the GUI
 * thread only queries the store's tile index and traces cached grids, so
 * disk/decode never touches the frame loop. ---- */
enum { SGC_EMPTY = 0, SGC_QUEUED, SGC_READY, SGC_FAILED };

typedef struct sgc_ent {
  r3d_tifxyz s;      /* decimated grid (stride SGC_STRIDE) */
  r3d_segrows rows;  /* trace-skipping bounds for it */
  int state;         /* SGC_*; all access under mu */
  uint64_t last_use; /* LRU tick */
} sgc_ent;

#define SGC_STRIDE 4u

typedef struct sgcache {
  r3d_segstore st;
  sgc_ent *ent; /* [st.n] */
  pthread_t th;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  uint32_t *queue;
  uint32_t qn, qcap;
  bool quit, open;
  size_t bytes, budget;
  uint64_t tick;
  /* activation: the worker prepares a full-res grid + GPU grid arrays for
   * one segment; the GUI thread applies the swap when act_ready is set */
  uint32_t act_req, act_ready; /* UINT32_MAX = none */
  bool act_busy;
  r3d_tifxyz act_s;
  r3d_segrows act_rows;
  float *act_coords, *act_normals;
  /* overlap QC: ov[i] = fraction of surface i's tiles within ~8 voxels of
   * the active surface (worker-computed from the tile index alone) */
  float *ov;
  uint32_t ov_active, ov_req;
} sgcache;

static size_t sgc_ent_bytes(const sgc_ent *e) {
  return (size_t)e->s.w * e->s.h * 3 * sizeof(float) +
         ((size_t)e->rows.h * 3 + (size_t)e->rows.tw * e->rows.th * 3) * 2 * sizeof(float);
}

static void sgc_evict_lru(sgcache *c, uint32_t keep) { /* mu held */
  while (c->bytes > c->budget) {
    uint32_t victim = UINT32_MAX;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < c->st.n; i++)
      if (c->ent[i].state == SGC_READY && i != keep && c->ent[i].last_use < oldest) {
        oldest = c->ent[i].last_use;
        victim = i;
      }
    if (victim == UINT32_MAX) return;
    c->bytes -= sgc_ent_bytes(&c->ent[victim]);
    r3d_tifxyz_free(&c->ent[victim].s);
    r3d_segrows_free(&c->ent[victim].rows);
    c->ent[victim].state = SGC_EMPTY;
  }
}

static int mv_build_grids(const r3d_tifxyz *s, float **coords_out, float **normals_out);

static void *sgc_worker(void *ud) {
  sgcache *c = ud;
  pthread_mutex_lock(&c->mu);
  while (!c->quit) {
    if (c->act_req != UINT32_MAX && c->act_ready == UINT32_MAX) {
      uint32_t ai = c->act_req;
      c->act_req = UINT32_MAX;
      c->act_busy = true;
      pthread_mutex_unlock(&c->mu);
      r3d_tifxyz s;
      r3d_segrows rows = {0};
      float *co = NULL, *no = NULL;
      int ok = r3d_segstore_load(&c->st, ai, 1, &s) == 0;
      if (ok && (r3d_segrows_build(&s, &rows) != 0 || mv_build_grids(&s, &co, &no) != 0)) {
        r3d_tifxyz_free(&s);
        r3d_segrows_free(&rows);
        ok = 0;
      }
      pthread_mutex_lock(&c->mu);
      if (ok) {
        c->act_s = s;
        c->act_rows = rows;
        c->act_coords = co;
        c->act_normals = no;
        c->act_ready = ai;
      } else {
        c->act_busy = false;
      }
      continue;
    }
    if (c->ov_req != UINT32_MAX) {
      uint32_t oi = c->ov_req;
      c->ov_req = UINT32_MAX;
      pthread_mutex_unlock(&c->mu);
      float *tmp = malloc((c->st.n ? c->st.n : 1) * sizeof *tmp);
      if (tmp)
        for (uint32_t i = 0; i < c->st.n; i++)
          tmp[i] = i == oi ? 0.0f : (float)r3d_segstore_overlap(&c->st, i, oi, 8.0);
      pthread_mutex_lock(&c->mu);
      if (tmp && c->ov) {
        memcpy(c->ov, tmp, c->st.n * sizeof *c->ov);
        c->ov_active = oi;
      }
      free(tmp);
      continue;
    }
    if (c->qn == 0) {
      pthread_cond_wait(&c->cv, &c->mu);
      continue;
    }
    uint32_t i = c->queue[--c->qn];
    pthread_mutex_unlock(&c->mu);
    r3d_tifxyz s;
    r3d_segrows rows = {0};
    int ok = r3d_segstore_load(&c->st, i, SGC_STRIDE, &s) == 0;
    if (ok && r3d_segrows_build(&s, &rows) != 0) {
      r3d_tifxyz_free(&s);
      ok = 0;
    }
    pthread_mutex_lock(&c->mu);
    if (ok) {
      c->ent[i].s = s;
      c->ent[i].rows = rows;
      c->ent[i].state = SGC_READY;
      c->ent[i].last_use = ++c->tick;
      c->bytes += sgc_ent_bytes(&c->ent[i]);
      sgc_evict_lru(c, i);
    } else {
      c->ent[i].state = SGC_FAILED;
    }
  }
  pthread_mutex_unlock(&c->mu);
  return NULL;
}

static int sgc_open(sgcache *c, const char *store_dir, size_t budget) {
  memset(c, 0, sizeof *c);
  c->act_req = c->act_ready = UINT32_MAX;
  c->ov_active = c->ov_req = UINT32_MAX;
  if (r3d_segstore_open(&c->st, store_dir) != 0) {
    /* an older/incompatible manifest (the format is versioned and rejected
     * outright when it changes): regenerate it from the store's own .tfx
     * files, then retry — losing the store over a format bump is worse
     * than one rebuild pass */
    printf("segments: store manifest unreadable, rebuilding from %s\n", store_dir);
    if (r3d_segstore_build(store_dir, NULL, 0, 2, false) <= 0 ||
        r3d_segstore_open(&c->st, store_dir) != 0)
      return -1;
  }
  c->ent = calloc(c->st.n ? c->st.n : 1, sizeof *c->ent);
  c->queue = malloc((c->st.n ? c->st.n : 1) * sizeof *c->queue);
  c->ov = calloc(c->st.n ? c->st.n : 1, sizeof *c->ov);
  if (!c->ent || !c->queue || !c->ov) {
    free(c->ent);
    free(c->queue);
    free(c->ov);
    r3d_segstore_close(&c->st);
    return -1;
  }
  c->qcap = c->st.n;
  c->budget = budget;
  pthread_mutex_init(&c->mu, NULL);
  pthread_cond_init(&c->cv, NULL);
  if (pthread_create(&c->th, NULL, sgc_worker, c) != 0) {
    free(c->ent);
    free(c->queue);
    free(c->ov);
    r3d_segstore_close(&c->st);
    return -1;
  }
  c->open = true;
  return 0;
}

static void sgc_close(sgcache *c) {
  if (!c->open) return;
  pthread_mutex_lock(&c->mu);
  c->quit = true;
  pthread_cond_signal(&c->cv);
  pthread_mutex_unlock(&c->mu);
  pthread_join(c->th, NULL);
  if (c->act_ready != UINT32_MAX) {
    r3d_tifxyz_free(&c->act_s);
    r3d_segrows_free(&c->act_rows);
    free(c->act_coords);
    free(c->act_normals);
  }
  for (uint32_t i = 0; i < c->st.n; i++)
    if (c->ent[i].state == SGC_READY) {
      r3d_tifxyz_free(&c->ent[i].s);
      r3d_segrows_free(&c->ent[i].rows);
    }
  free(c->ent);
  free(c->queue);
  free(c->ov);
  pthread_mutex_destroy(&c->mu);
  pthread_cond_destroy(&c->cv);
  r3d_segstore_close(&c->st);
  c->open = false;
}

static void sgc_request(sgcache *c, uint32_t i) { /* mu held */
  if (c->ent[i].state != SGC_EMPTY || c->qn >= c->qcap) return;
  c->ent[i].state = SGC_QUEUED;
  c->queue[c->qn++] = i;
  pthread_cond_signal(&c->cv);
}

/* per-(plane view, segment) cached overlay polyline */
typedef struct sgc_line {
  mv_lines l;
  double slice;
  uint32_t gen;
  bool valid;
} sgc_line;

/* linear interpolation of the umbilicus curve at z (clamped to the ends);
 * false when no points exist */
static bool umb_interp(const r3d_umbilicus *u, double z, double *x, double *y) {
  if (u->count == 0) return false;
  const r3d_umbilicus_point *p = u->points;
  size_t n = u->count;
  if (z < p[0].z || z > p[n - 1].z) return false; /* no clamped extension:
                        * outside the curve's z-range there IS no umbilicus */
  if (z == p[0].z) {
    *x = p[0].x;
    *y = p[0].y;
    return true;
  }
  for (size_t i = 1; i < n; i++)
    if (p[i].z >= z) {
      double dz = p[i].z - p[i - 1].z;
      double t = dz > 0.0 ? (z - p[i - 1].z) / dz : 0.0;
      *x = p[i - 1].x + (p[i].x - p[i - 1].x) * t;
      *y = p[i - 1].y + (p[i].y - p[i - 1].y) * t;
      return true;
    }
  return false;
}

/* snapshot-based undo for umbilicus edits (point arrays are tiny) */
typedef struct umb_snap {
  r3d_umbilicus_point *pts;
  size_t count;
} umb_snap;
#define UMB_UNDO_MAX 64u

static void umb_snap_clear(umb_snap *st, uint32_t *n) {
  while (*n) free(st[--(*n)].pts);
}

/* replace the point set with a snapshot (consumes it; re-inserting via set
 * keeps the sorted invariant) */
static void umb_snap_apply(r3d_umbilicus *u, umb_snap s) {
  u->count = 0;
  for (size_t k = 0; k < s.count; k++)
    r3d_umbilicus_set(u, s.pts[k].x, s.pts[k].y, s.pts[k].z);
  free(s.pts);
}

static void umb_undo_push(umb_snap *st, uint32_t *n, const r3d_umbilicus *u) {
  if (*n == UMB_UNDO_MAX) { /* full: the oldest snapshot falls off */
    free(st[0].pts);
    memmove(st, st + 1, (UMB_UNDO_MAX - 1) * sizeof *st);
    (*n)--;
  }
  r3d_umbilicus_point *cp = NULL;
  if (u->count) {
    cp = malloc(u->count * sizeof *cp);
    if (!cp) return; /* OOM: this edit just isn't undoable */
    memcpy(cp, u->points, u->count * sizeof *cp);
  }
  st[*n].pts = cp;
  st[(*n)++].count = u->count;
}

/* first voxel value with nonzero TF alpha: below it, samples are invisible */
static float tf_min_visible(const uint8_t lut[256][4]) {
  for (uint32_t i = 0; i < 256; i++)
    if (lut[i][3] != 0) return (float)i;
  return 255.0f;
}

/* ---- open-data browser: browse the bucket, bootstrap + relaunch ---------- */

static const char OD_BUCKET[] = "https://vesuvius-challenge-open-data.s3.amazonaws.com";

#define OD_MAX_STEPS 6
#define OD_MAX_ARGS 12
#define OD_ARGBUF 8192

/* one argv-spawned step of a browser job (offset into od_state.argbuf) */
typedef struct {
  uint32_t off, nargs;
} od_step;

typedef struct od_state {
  r3d_odlist scrolls, vols, segs, variants;
  int sel_scroll, sel_vol, sel_seg, sel_variant;
  bool scrolls_ok, vols_ok, segs_ok, variants_ok;
  /* surface-prediction overlays (.zarr dirs under <scroll>/representations/
   * predictions/surfaces/) and 3D ink (.../ink-3d/): independent picks */
  r3d_odlist ovls, inks;
  int sel_ovl, sel_ink;
  bool ovls_ok, inks_ok;
  /* async listing worker: the GUI thread never blocks on the network or
   * touches the filesystem — it posts requests and polls results */
  pthread_t fth;
  bool fth_up, fbusy, fquit;
  pthread_mutex_t fmu;
  pthread_cond_t fcv;
  int freq, fdone; /* 1 scrolls, 2 vols+segs (of fscroll), 3 variants */
  char fscroll[300], fseg[300];
  r3d_odlist fres_a, fres_b;
  bool *fcached; /* per-volume [cached] flags, computed on the worker */
  bool *vol_cached;
  /* running subprocess: argv-spawned, never a shell (remote object names are
   * arguments, never syntax); stdout+stderr merged onto a non-blocking pipe */
  bool job_up;
  int jobfd;
  pid_t jobpid;
  char log[10][160];    /* rolling job output */
  int nlog;
  /* one pending action at a time: bootstrap/download the pick, then either
   * request a dataset swap (volume/segment) or a live attach (overlays) */
  int act; /* 0 idle, 1 volume, 2 segment, 3 surface preds, 4 3D ink */
  bool spawned;
  char tgt_dir[512]; /* local dir the action produces */
  /* the pending job as a sequence of argv-spawned steps, built at click time.
   * argv strings live NUL-separated in argbuf; a step names its first arg's
   * offset and its argument count. job_ovf latches a build overflow so the
   * action fails closed rather than running a truncated command. */
  od_step steps[OD_MAX_STEPS];
  int nsteps, curstep;
  char argbuf[OD_ARGBUF];
  uint32_t arglen;
  bool job_ovf;
  /* published only after every step succeeds: a partial download must never
   * look complete to the "does meta.json exist" probe */
  char fin_src[600], fin_dst[600];
  char linebuf[512];
  size_t linelen;
} od_state;

static void od_log(od_state *od, const char *line) {
  if (od->nlog == 10) {
    memmove(od->log[0], od->log[1], sizeof od->log - sizeof od->log[0]);
    od->nlog--;
  }
  snprintf(od->log[od->nlog++], sizeof od->log[0], "%s", line);
}

static void od_exe_dir(char out[512]) {
  ssize_t n = readlink("/proc/self/exe", out, 511);
  if (n <= 0) {
    snprintf(out, 512, ".");
    return;
  }
  out[n] = 0;
  char *slash = strrchr(out, '/');
  if (slash) *slash = 0;
}

static bool od_file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0;
}

/* Remote object names become both path components and process arguments.
 * Accept only a plain component: [A-Za-z0-9._-]+, no leading '-' (so it can
 * never be read as an option), no "..", bounded length. Everything else is
 * rejected before any path or argv is built. */
static bool od_name_ok(const char *s) {
  if (!s || !s[0] || s[0] == '-') return false;
  size_t n = strlen(s);
  if (n > 255) return false;
  if (strstr(s, "..")) return false;
  for (size_t i = 0; i < n; i++) {
    char c = s[i];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
        c == '.' || c == '_' || c == '-')
      continue;
    return false;
  }
  return true;
}

/* fail-closed guard for every remote name an action interpolates */
static bool od_names_ok(od_state *od, const char *const *names, int n) {
  for (int i = 0; i < n; i++)
    if (!od_name_ok(names[i])) {
      od_log(od, "rejected: unsafe remote name");
      fprintf(stderr, "odbrowse: rejected unsafe remote name \"%s\"\n",
              names[i] ? names[i] : "(null)");
      return false;
    }
  return true;
}

/* mkdir -p, in process: no shell, no quoting, and errors are visible */
static bool mkdir_p(const char *path) {
  char tmp[640];
  if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp) {
    errno = ENAMETOOLONG;
    return false;
  }
  for (char *q = tmp + 1; *q; q++) {
    if (*q != '/') continue;
    *q = 0;
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return false;
    *q = '/';
  }
  return mkdir(tmp, 0755) == 0 || errno == EEXIST;
}

static void od_job_reset(od_state *od) {
  od->nsteps = od->curstep = 0;
  od->arglen = 0;
  od->job_ovf = false;
  od->fin_src[0] = od->fin_dst[0] = 0;
}

/* append one argv-spawned step; on any overflow latch job_ovf so the caller
 * refuses the action rather than running a truncated command */
static void od_job_step(od_state *od, const char *const *args, int n) {
  if (od->job_ovf || od->nsteps >= OD_MAX_STEPS || n <= 0 || n > OD_MAX_ARGS) {
    od->job_ovf = true;
    return;
  }
  uint32_t off = od->arglen;
  for (int i = 0; i < n; i++) {
    if (!args[i]) {
      od->job_ovf = true;
      return;
    }
    size_t len = strlen(args[i]) + 1;
    if (len > sizeof od->argbuf - od->arglen) {
      od->job_ovf = true;
      return;
    }
    memcpy(od->argbuf + od->arglen, args[i], len);
    od->arglen += (uint32_t)len;
  }
  od->steps[od->nsteps].off = off;
  od->steps[od->nsteps].nargs = (uint32_t)n;
  od->nsteps++;
}

/* spawn steps[curstep] with merged stderr on a non-blocking pipe. argv only:
 * no /bin/sh, so names carrying quotes or metacharacters stay data. */
static int od_spawn_step(od_state *od) {
  if (od->job_ovf || od->curstep >= od->nsteps) return -1;
  const od_step *st = &od->steps[od->curstep];
  char *argv[OD_MAX_ARGS + 1];
  char *p = od->argbuf + st->off;
  for (uint32_t i = 0; i < st->nargs; i++) {
    argv[i] = p;
    p += strlen(p) + 1;
  }
  argv[st->nargs] = NULL;
  int fds[2];
  if (pipe2(fds, O_CLOEXEC) != 0) return -1; /* read end never reaches a child */
  posix_spawn_file_actions_t fa;
  if (posix_spawn_file_actions_init(&fa) != 0) {
    close(fds[0]);
    close(fds[1]);
    return -1;
  }
  int rc = posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
  if (rc == 0) rc = posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
  if (rc == 0 && fds[1] > 2) rc = posix_spawn_file_actions_addclose(&fa, fds[1]);
  pid_t pid = 0;
  if (rc == 0) rc = posix_spawnp(&pid, argv[0], &fa, NULL, argv, environ);
  posix_spawn_file_actions_destroy(&fa);
  close(fds[1]);
  if (rc != 0) {
    close(fds[0]);
    errno = rc;
    return -1;
  }
  fcntl(fds[0], F_SETFL, fcntl(fds[0], F_GETFL) | O_NONBLOCK);
  od->jobfd = fds[0];
  od->jobpid = pid;
  od->job_up = true;
  od->linelen = 0;
  return 0;
}

/* reap the finished child: 1 exited 0, -1 anything else */
static int od_job_reap(od_state *od) {
  int status = 0;
  pid_t r;
  do {
    r = waitpid(od->jobpid, &status, 0);
  } while (r < 0 && errno == EINTR);
  od->jobpid = 0;
  if (r < 0) return -1;
  return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 1 : -1;
}

/* pump the job pipe; returns 1 when the whole step sequence finished ok,
 * -1 failed, 0 busy (including "this step is done, next one started") */
static int od_pump(od_state *od) {
  if (!od->job_up) return 0;
  char buf[512];
  for (;;) {
    ssize_t n = read(od->jobfd, buf, sizeof buf);
    if (n > 0) {
      for (ssize_t i = 0; i < n; i++) {
        if (buf[i] == '\n' || od->linelen + 1 >= sizeof od->linebuf) {
          od->linebuf[od->linelen] = 0;
          if (od->linelen) od_log(od, od->linebuf);
          od->linelen = 0;
        } else if (buf[i] != '\r') {
          od->linebuf[od->linelen++] = buf[i];
        }
      }
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
    break; /* EOF or error */
  }
  close(od->jobfd);
  od->jobfd = -1;
  od->job_up = false;
  if (od_job_reap(od) < 0) return -1;
  if (++od->curstep < od->nsteps) {
    if (od_spawn_step(od) != 0) {
      od_log(od, "spawn failed");
      return -1;
    }
    return 0;
  }
  if (od->fin_src[0] && rename(od->fin_src, od->fin_dst) != 0) {
    od_log(od, "publish (rename) failed");
    fprintf(stderr, "odbrowse: rename %s -> %s failed: %s\n", od->fin_src, od->fin_dst,
            strerror(errno));
    return -1;
  }
  return 1;
}

/* zarr2c5d bootstrap: a single argv-spawned step */
static void od_job_bootstrap(od_state *od, const char *exe, const char *url) {
  char prog[600], meta[600];
  snprintf(prog, sizeof prog, "%s/zarr2c5d", exe);
  snprintf(meta, sizeof meta, "%s/meta", od->tgt_dir);
  const char *a[] = {prog,        meta,      od->tgt_dir, "--url",
                     url,         "--bootstrap", "--threads", "8"};
  od_job_reset(od);
  od_job_step(od, a, 8);
}

static void *od_fetch_worker(void *ud) {
  od_state *od = ud;
  for (;;) {
    pthread_mutex_lock(&od->fmu);
    while (!od->fquit && od->freq == 0) pthread_cond_wait(&od->fcv, &od->fmu);
    if (od->fquit) {
      pthread_mutex_unlock(&od->fmu);
      return NULL;
    }
    int req = od->freq;
    char scroll[300], seg[300];
    snprintf(scroll, sizeof scroll, "%s", od->fscroll);
    snprintf(seg, sizeof seg, "%s", od->fseg);
    pthread_mutex_unlock(&od->fmu);

    r3d_odlist a = {0}, b = {0};
    bool *cached = NULL;
    char pfx[700];
    if (req == 1) {
      r3d_odlist_fetch(OD_BUCKET, "", &a);
    } else if (req == 2) {
      snprintf(pfx, sizeof pfx, "%s/volumes/", scroll);
      r3d_odlist_fetch(OD_BUCKET, pfx, &a);
      snprintf(pfx, sizeof pfx, "%s/segments/", scroll);
      r3d_odlist_fetch(OD_BUCKET, pfx, &b);
      cached = calloc(a.ndirs ? a.ndirs : 1, sizeof *cached);
      for (uint32_t i = 0; cached && i < a.ndirs; i++) {
        char mp[900];
        snprintf(mp, sizeof mp, "cache/od/%s/%s/manifest.json", scroll, a.dirs[i]);
        cached[i] = od_file_exists(mp);
      }
    } else if (req == 3) {
      snprintf(pfx, sizeof pfx, "%s/segments/%s/mesh/", scroll, seg);
      r3d_odlist_fetch(OD_BUCKET, pfx, &a);
    } else if (req == 4) {
      snprintf(pfx, sizeof pfx, "%s/representations/predictions/surfaces/", scroll);
      r3d_odlist_fetch(OD_BUCKET, pfx, &a);
    } else if (req == 5) {
      snprintf(pfx, sizeof pfx, "%s/representations/predictions/ink-3d/", scroll);
      r3d_odlist_fetch(OD_BUCKET, pfx, &a);
    }
    pthread_mutex_lock(&od->fmu);
    r3d_odlist_free(&od->fres_a);
    r3d_odlist_free(&od->fres_b);
    free(od->fcached);
    od->fres_a = a;
    od->fres_b = b;
    od->fcached = cached;
    od->fdone = req;
    od->freq = 0;
    od->fbusy = false;
    pthread_mutex_unlock(&od->fmu);
  }
}

static void od_request(od_state *od, int req) {
  if (!od->fth_up) {
    pthread_mutex_init(&od->fmu, NULL);
    pthread_cond_init(&od->fcv, NULL);
    if (pthread_create(&od->fth, NULL, od_fetch_worker, od) != 0) return;
    od->fth_up = true;
  }
  pthread_mutex_lock(&od->fmu);
  if (!od->fbusy) {
    od->freq = req;
    od->fbusy = true;
    if (od->sel_scroll >= 0)
      snprintf(od->fscroll, sizeof od->fscroll, "%s", od->scrolls.dirs[od->sel_scroll]);
    if (od->sel_seg >= 0)
      snprintf(od->fseg, sizeof od->fseg, "%s", od->segs.dirs[od->sel_seg]);
    pthread_cond_signal(&od->fcv);
  }
  pthread_mutex_unlock(&od->fmu);
}

/* returns the completed request id (once) and moves its results out */
static int od_poll(od_state *od, r3d_odlist *a, r3d_odlist *b, bool **cached) {
  if (!od->fth_up) return 0;
  pthread_mutex_lock(&od->fmu);
  int done = od->fdone;
  if (done) {
    *a = od->fres_a;
    *b = od->fres_b;
    *cached = od->fcached;
    memset(&od->fres_a, 0, sizeof od->fres_a);
    memset(&od->fres_b, 0, sizeof od->fres_b);
    od->fcached = NULL;
    od->fdone = 0;
  }
  pthread_mutex_unlock(&od->fmu);
  return done;
}

/* The browser is an ordinary ImGui window in the frame loop. Opening a
 * dataset runs background jobs (zarr2c5d --bootstrap, tifxyz download);
 * when everything is local it writes the chosen paths and requests a
 * dataset swap — plain mutable state, reopened in the same session. */
static void od_browser_window(od_state *od, bool *open, char *next_bricks, size_t nb_cap,
                              char *next_seg, size_t ns_cap, char *next_ovl,
                              size_t no_cap, char *next_ink, size_t ni_cap,
                              char *next_reg, size_t nr_cap, bool *swap,
                              bool *attach_ovl, bool *attach_ink, bool *attach_reg,
                              const char *cur_bricks) {
  int prc = od_pump(od);
  if (prc < 0 && od->act) {
    od_log(od, "job FAILED");
    od->act = 0;
  }
  if (od->act && !od->job_up) { /* probe -> spawn -> finish the pending action */
    char probe[600];
    snprintf(probe, sizeof probe, "%s/%s", od->tgt_dir,
             od->act == 2 ? "meta.json" : "manifest.json");
    if (od_file_exists(probe)) {
      switch (od->act) {
      case 1: /* volume: full dataset swap, dropping segment + overlays */
        snprintf(next_bricks, nb_cap, "%s/manifest.json", od->tgt_dir);
        next_seg[0] = 0;
        next_ovl[0] = 0;
        next_ink[0] = 0;
        *swap = true;
        od_log(od, "volume ready - opening...");
        break;
      case 2: /* segment: re-swap the current volume with it; browser-picked
               * overlays (next_ovl/next_ink) survive the swap */
        if (cur_bricks) snprintf(next_bricks, nb_cap, "%s", cur_bricks);
        snprintf(next_seg, ns_cap, "%s", od->tgt_dir);
        *swap = true;
        od_log(od, "segment ready - opening...");
        break;
      case 3: /* surface predictions: live attach, no teardown */
        snprintf(next_ovl, no_cap, "%s", od->tgt_dir);
        *attach_ovl = true;
        od_log(od, "surface predictions ready - attaching...");
        break;
      case 4: /* 3D ink: live attach on the second overlay slot */
        snprintf(next_ink, ni_cap, "%s", od->tgt_dir);
        *attach_ink = true;
        od_log(od, "3D ink ready - attaching...");
        break;
      case 5: /* registration: live attach as the moving scan, no teardown */
        snprintf(next_reg, nr_cap, "%s", od->tgt_dir);
        *attach_reg = true;
        od_log(od, "registration volume ready - attaching...");
        break;
      }
      od->act = 0;
    } else if (od->spawned) {
      od_log(od, od->act == 2 ? "segment download incomplete"
                              : "bootstrap produced no manifest");
      od->act = 0;
    } else {
      od->spawned = true;
      if (od->job_ovf || od->nsteps == 0 || od_spawn_step(od) != 0) {
        od_log(od, "job could not be started");
        fprintf(stderr, "odbrowse: job spawn failed: %s\n", strerror(errno));
        od->act = 0;
      }
    }
  }

  {
    r3d_odlist ra = {0}, rb = {0};
    bool *rc_ = NULL;
    int done = od_poll(od, &ra, &rb, &rc_);
    if (done == 1) {
      r3d_odlist_free(&od->scrolls);
      od->scrolls = ra;
      od->scrolls_ok = true;
    } else if (done == 2) {
      r3d_odlist_free(&od->vols);
      r3d_odlist_free(&od->segs);
      free(od->vol_cached);
      od->vols = ra;
      od->segs = rb;
      od->vol_cached = rc_;
      rc_ = NULL;
      od->vols_ok = od->segs_ok = true;
    } else if (done == 3) {
      r3d_odlist_free(&od->variants);
      od->variants = ra;
      od->variants_ok = true;
    } else if (done == 4) {
      r3d_odlist_free(&od->ovls);
      od->ovls = ra;
      od->ovls_ok = true;
    } else if (done == 5) {
      r3d_odlist_free(&od->inks);
      od->inks = ra;
      od->inks_ok = true;
    } else if (done) {
      r3d_odlist_free(&ra);
      r3d_odlist_free(&rb);
    }
    if (done != 2) free(rc_);
  }

  if (!*open) return;
  igSetNextWindowSize((ImVec2){620, 640}, ImGuiCond_FirstUseEver);
  igSetNextWindowPos((ImVec2){380, 40}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
  if (!igBegin("open data browser", open, 0)) {
    igEnd();
    return;
  }
  if (!od->scrolls_ok) {
    od_request(od, 1);
    igTextDisabled("listing bucket...");
  }
  const char *cur = od->sel_scroll >= 0 ? od->scrolls.dirs[od->sel_scroll] : "<pick a scroll>";
  if (igBeginCombo("scroll", cur, 0)) {
    for (uint32_t i = 0; i < od->scrolls.ndirs; i++)
      if (igSelectable_Bool(od->scrolls.dirs[i], (int)i == od->sel_scroll, 0,
                            (ImVec2){0, 0})) {
        od->sel_scroll = (int)i;
        r3d_odlist_free(&od->vols);
        r3d_odlist_free(&od->segs);
        r3d_odlist_free(&od->variants);
        r3d_odlist_free(&od->ovls);
        r3d_odlist_free(&od->inks);
        od->vols_ok = od->segs_ok = od->variants_ok = od->ovls_ok = od->inks_ok = false;
        od->sel_vol = od->sel_seg = od->sel_variant = od->sel_ovl = od->sel_ink = -1;
      }
    igEndCombo();
  }
  if (od->sel_scroll >= 0 && !od->vols_ok) {
    od_request(od, 2);
    igTextDisabled("listing scroll...");
  }
  bool busy = od->act != 0;
  char volid[64] = "";
  if (od->sel_vol >= 0) { /* volume id = dir name up to the first '-' */
    snprintf(volid, sizeof volid, "%s", od->vols.dirs[od->sel_vol]);
    char *dash = strchr(volid, '-');
    if (dash) *dash = 0;
  }
  char exe[512];
  od_exe_dir(exe);
  /* ---- volume: its own list + button; opening swaps the whole dataset */
  if (od->vols_ok) {
    igText("volumes");
    igBeginChild_Str("odvols", (ImVec2){0, 120}, ImGuiChildFlags_Borders, 0);
    for (uint32_t i = 0; i < od->vols.ndirs; i++) {
      char label[800];
      snprintf(label, sizeof label, "%s%s",
               od->vol_cached && od->vol_cached[i] ? "[cached] " : "", od->vols.dirs[i]);
      if (igSelectable_Bool(label, (int)i == od->sel_vol, 0, (ImVec2){0, 0})) {
        od->sel_vol = (int)i;
        printf("odbrowse: volume %s selected\n", od->vols.dirs[i]);
      }
    }
    igEndChild();
    igBeginDisabled(busy || od->sel_vol < 0);
    if (igButton("open volume", (ImVec2){0, 0})) {
      const char *nm[2] = {od->scrolls.dirs[od->sel_scroll], od->vols.dirs[od->sel_vol]};
      if (od_names_ok(od, nm, 2)) {
        char url[1200];
        snprintf(od->tgt_dir, sizeof od->tgt_dir, "cache/od/%s/%s", nm[0], nm[1]);
        snprintf(url, sizeof url, "%s/%s/volumes/%s", OD_BUCKET, nm[0], nm[1]);
        od_job_bootstrap(od, exe, url);
        if (!od->job_ovf) {
          od->spawned = false;
          od->act = 1;
          od_log(od, "bootstrapping volume (coarsest level + source.json)...");
        } else {
          od_log(od, "rejected: job arguments too long");
        }
      }
    }
    igEndDisabled();
    igSameLine(0, 10);
    /* same bootstrap, but attaches live as the registration moving scan
     * (green/magenta fuse over the open volume) instead of swapping */
    igBeginDisabled(busy || od->sel_vol < 0 || !cur_bricks);
    if (igButton("open as registration volume", (ImVec2){0, 0})) {
      const char *nm[2] = {od->scrolls.dirs[od->sel_scroll], od->vols.dirs[od->sel_vol]};
      if (od_names_ok(od, nm, 2)) {
        char url[1200];
        snprintf(od->tgt_dir, sizeof od->tgt_dir, "cache/od/%s/%s", nm[0], nm[1]);
        snprintf(url, sizeof url, "%s/%s/volumes/%s", OD_BUCKET, nm[0], nm[1]);
        od_job_bootstrap(od, exe, url);
        if (!od->job_ovf) {
          od->spawned = false;
          od->act = 5;
          od_log(od, "bootstrapping registration volume...");
        } else {
          od_log(od, "rejected: job arguments too long");
        }
      }
    }
    igEndDisabled();
    if (od->sel_vol >= 0 && !cur_bricks && igIsItemHovered(0))
      igSetTooltip("open a volume first - the registration volume overlays it");
  }
  /* ---- segment: list + mesh variant + button; re-swaps the open volume */
  if (od->segs_ok && od->segs.ndirs) {
    igText("segments (%u)", od->segs.ndirs);
    igBeginChild_Str("odsegs", (ImVec2){0, 110}, ImGuiChildFlags_Borders, 0);
    for (uint32_t i = 0; i < od->segs.ndirs; i++)
      if (igSelectable_Bool(od->segs.dirs[i], (int)i == od->sel_seg, 0, (ImVec2){0, 0})) {
        od->sel_seg = (int)i;
        r3d_odlist_free(&od->variants);
        od->variants_ok = false;
        od->sel_variant = -1;
      }
    igEndChild();
    if (od->sel_seg >= 0 && !od->variants_ok) od_request(od, 3);
    if (od->variants_ok && od->variants.ndirs) {
      igText("segment mesh (tifxyz)");
      igBeginChild_Str("odvar", (ImVec2){0, 70}, ImGuiChildFlags_Borders, 0);
      char onvol[80];
      snprintf(onvol, sizeof onvol, "-on-%s", volid);
      for (uint32_t i = 0; i < od->variants.ndirs; i++) {
        bool match = volid[0] && strstr(od->variants.dirs[i], onvol) != NULL;
        char label[700];
        snprintf(label, sizeof label, "%s%s", match ? "[matches volume] " : "",
                 od->variants.dirs[i]);
        if (igSelectable_Bool(label, (int)i == od->sel_variant, 0, (ImVec2){0, 0}))
          od->sel_variant = (int)i;
      }
      igEndChild();
    }
    igBeginDisabled(busy || od->sel_variant < 0 || !cur_bricks);
    if (igButton("open segment", (ImVec2){0, 0})) {
      const char *nm[3] = {od->scrolls.dirs[od->sel_scroll], od->segs.dirs[od->sel_seg],
                           od->variants.dirs[od->sel_variant]};
      if (od_names_ok(od, nm, 3)) {
        char murl[1100];
        snprintf(od->tgt_dir, sizeof od->tgt_dir, "cache/od/segments/%s", nm[2]);
        snprintf(murl, sizeof murl, "%s/%s/segments/%s/mesh/%s", OD_BUCKET, nm[0], nm[1],
                 nm[2]);
        /* four argv-spawned curls; meta.json lands staged and is renamed into
         * place only after every file arrived, so the completion probe never
         * sees a partial download */
        od_job_reset(od);
        static const char *const parts[4] = {"x.tif", "y.tif", "z.tif", "meta.json"};
        for (int k = 0; k < 4; k++) {
          char dst[700], src[1300];
          snprintf(dst, sizeof dst, "%s/%s%s", od->tgt_dir, parts[k],
                   k == 3 ? ".dl" : "");
          snprintf(src, sizeof src, "%s/%s", murl, parts[k]);
          const char *a[5] = {"curl", "-fsS", "-o", dst, src};
          od_job_step(od, a, 5);
        }
        snprintf(od->fin_src, sizeof od->fin_src, "%s/meta.json.dl", od->tgt_dir);
        snprintf(od->fin_dst, sizeof od->fin_dst, "%s/meta.json", od->tgt_dir);
        if (od->job_ovf) {
          od_log(od, "rejected: job arguments too long");
        } else if (!mkdir_p(od->tgt_dir)) {
          od_log(od, "cannot create download dir");
          fprintf(stderr, "odbrowse: mkdir %s failed: %s\n", od->tgt_dir,
                  strerror(errno));
          od_job_reset(od);
        } else {
          od->spawned = false;
          od->act = 2;
          od_log(od, "downloading segment tifxyz...");
        }
      }
    }
    igEndDisabled();
    if (od->sel_variant >= 0 && !cur_bricks && igIsItemHovered(0))
      igSetTooltip("open a volume first");
  }
  /* ---- surface predictions: attach live as the blue overlay slot */
  if (od->sel_scroll >= 0 && od->vols_ok && !od->ovls_ok) od_request(od, 4);
  if (od->ovls_ok && od->ovls.ndirs) {
    igText("surface predictions");
    igBeginChild_Str("odovls", (ImVec2){0, 70}, ImGuiChildFlags_Borders, 0);
    for (uint32_t i = 0; i < od->ovls.ndirs; i++) {
      const char *nm = od->ovls.dirs[i];
      size_t nl = strlen(nm);
      if (nl < 5 || strcmp(nm + nl - 5, ".zarr") != 0)
        continue; /* .normal-grids siblings ride along with the tracer */
      bool match = volid[0] && strncmp(nm, volid, strlen(volid)) == 0;
      char label[700];
      snprintf(label, sizeof label, "%s%s", match ? "[matches volume] " : "", nm);
      if (igSelectable_Bool(label, (int)i == od->sel_ovl, 0, (ImVec2){0, 0}))
        od->sel_ovl = (int)i;
    }
    igEndChild();
    igBeginDisabled(busy || od->sel_ovl < 0 || !cur_bricks);
    if (igButton("open surface predictions", (ImVec2){0, 0})) {
      const char *nm[2] = {od->scrolls.dirs[od->sel_scroll], od->ovls.dirs[od->sel_ovl]};
      if (od_names_ok(od, nm, 2)) {
        char url[1200];
        snprintf(od->tgt_dir, sizeof od->tgt_dir, "cache/od/%s/overlays/%s", nm[0], nm[1]);
        snprintf(url, sizeof url, "%s/%s/representations/predictions/surfaces/%s",
                 OD_BUCKET, nm[0], nm[1]);
        od_job_bootstrap(od, exe, url);
        if (!od->job_ovf) {
          od->spawned = false;
          od->act = 3;
          od_log(od, "bootstrapping surface predictions...");
        } else {
          od_log(od, "rejected: job arguments too long");
        }
      }
    }
    igEndDisabled();
  }
  /* ---- 3D ink: attach live as the red overlay slot */
  if (od->sel_scroll >= 0 && od->vols_ok && !od->inks_ok) od_request(od, 5);
  if (od->inks_ok && od->inks.ndirs) {
    igText("3D ink");
    igBeginChild_Str("odinks", (ImVec2){0, 70}, ImGuiChildFlags_Borders, 0);
    for (uint32_t i = 0; i < od->inks.ndirs; i++) {
      const char *nm = od->inks.dirs[i];
      size_t nl = strlen(nm);
      if (nl < 5 || strcmp(nm + nl - 5, ".zarr") != 0) continue;
      bool match = volid[0] && strncmp(nm, volid, strlen(volid)) == 0;
      char label[700];
      snprintf(label, sizeof label, "%s%s", match ? "[matches volume] " : "", nm);
      if (igSelectable_Bool(label, (int)i == od->sel_ink, 0, (ImVec2){0, 0}))
        od->sel_ink = (int)i;
    }
    igEndChild();
    igBeginDisabled(busy || od->sel_ink < 0 || !cur_bricks);
    if (igButton("open 3D ink", (ImVec2){0, 0})) {
      const char *nm[2] = {od->scrolls.dirs[od->sel_scroll], od->inks.dirs[od->sel_ink]};
      if (od_names_ok(od, nm, 2)) {
        char url[1200];
        snprintf(od->tgt_dir, sizeof od->tgt_dir, "cache/od/%s/ink3d/%s", nm[0], nm[1]);
        snprintf(url, sizeof url, "%s/%s/representations/predictions/ink-3d/%s", OD_BUCKET,
                 nm[0], nm[1]);
        od_job_bootstrap(od, exe, url);
        if (!od->job_ovf) {
          od->spawned = false;
          od->act = 4;
          od_log(od, "bootstrapping 3D ink...");
        } else {
          od_log(od, "rejected: job arguments too long");
        }
      }
    }
    igEndDisabled();
  }
  if (busy) igTextDisabled("working...");
  igSeparator();
  for (int i = 0; i < od->nlog; i++) igTextDisabled("%s", od->log[i]);
  igEnd();
}

static void gui_event_hook(void *ud, const SDL_Event *ev) {
  r3d_gui_event((r3d_renderer *)ud, ev);
}

static void take_screenshot(r3d_renderer *renderer, uint64_t frame) {
  uint32_t w = 0, h = 0;
  if (r3d_read_frame(renderer, NULL, &w, &h) != 0) return;
  uint8_t *rgba = malloc((size_t)w * h * 4);
  if (!rgba) return;
  if (r3d_read_frame(renderer, rgba, &w, &h) == 0) {
    char path[64];
    snprintf(path, sizeof path, "render3d_%llu.ppm", (unsigned long long)frame);
    if (r3d_screenshot_ppm(path, rgba, w, h) == 0)
      printf("screenshot: %s (%ux%u)\n", path, w, h);
  }
  free(rgba);
}

static void profile_summary(const r3d_frame_stats *samples, uint64_t n, size_t field,
                            r3d_stats_summary *out) {
  memset(out, 0, sizeof *out);
  if (!samples || n == 0 || n > UINT32_MAX) return;
  uint64_t *v = malloc((size_t)n * sizeof *v);
  if (!v) return;
  for (uint64_t i = 0; i < n; i++) v[i] = ((const uint64_t *)&samples[i])[field];
  r3d_stats_summarize_values(v, (uint32_t)n, out);
  free(v);
}

static void json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (; s && *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
    else if (c == '\n') fputs("\\n", f);
    else if (c >= 0x20) fputc(c, f);
  }
  fputc('"', f);
}

static void json_timing(FILE *f, const char *name, const r3d_stats_summary *s, bool comma) {
  fprintf(f, "    \"%s\": {\"mean_ms\": %.6f, \"p50_ms\": %.6f, "
             "\"p95_ms\": %.6f, \"p99_ms\": %.6f, \"max_ms\": %.6f}%s\n",
          name, s->mean_ns / 1e6, (double)s->p50_ns / 1e6, (double)s->p95_ns / 1e6,
          (double)s->p99_ns / 1e6, (double)s->max_ns / 1e6, comma ? "," : "");
}

static int write_bench_json(const char *path, const char *scenario, int width, int height,
                            const char *quality, uint32_t warmup, const r3d_stats *stats,
                            const r3d_frame_stats *samples, uint64_t nsamples,
                            uint64_t pending_cell_frames, const r3d_bricks_stats *bricks) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "benchmark: cannot write %s\n", path);
    return -1;
  }
  r3d_stats_summary timing[10];
  r3d_stats_summarize(stats, &timing[0], &timing[1]);
  for (size_t i = 0; i < 8; i++) profile_summary(samples, nsamples, i, &timing[i + 2]);
  fputs("{\n  \"schema\": \"render3d-benchmark-v1\",\n  \"scenario\": ", f);
  json_string(f, scenario ? scenario : "static");
  fputs(",\n  \"quality\": ", f);
  json_string(f, quality);
  fprintf(f, ",\n  \"width\": %d,\n  \"height\": %d,\n  \"warmup_frames\": %u,\n"
             "  \"measured_frames\": %llu,\n  \"retained_frame_samples\": %u,\n"
             "  \"c5d_revision\": ",
          width, height, warmup, (unsigned long long)nsamples, stats->count);
  json_string(f, R3D_C5D_REV);
  fprintf(f, ",\n  \"pending_cell_frames\": %llu,\n"
             "  \"brick_stream\": {\"decoded\": %llu, \"jobs\": %llu, "
             "\"failures\": %u, \"mean_job_ms\": %.6f,\n"
             "    \"levels\": %u,\n"
             "    \"lod_wanted\": [%u, %u, %u, %u, %u, %u, %u, %u],\n"
             "    \"lod_requests\": [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu]\n"
             "  },\n  \"timings\": {\n",
          (unsigned long long)pending_cell_frames,
          (unsigned long long)(bricks ? bricks->decoded : 0),
          (unsigned long long)(bricks ? bricks->jobs : 0), bricks ? bricks->failures : 0,
          bricks && bricks->jobs ? (double)bricks->stream_ns / (double)bricks->jobs / 1e6 : 0.0,
          bricks ? bricks->nlevels : 0, bricks ? bricks->lod_wanted[0] : 0,
          bricks ? bricks->lod_wanted[1] : 0, bricks ? bricks->lod_wanted[2] : 0,
          bricks ? bricks->lod_wanted[3] : 0, bricks ? bricks->lod_wanted[4] : 0,
          bricks ? bricks->lod_wanted[5] : 0, bricks ? bricks->lod_wanted[6] : 0,
          bricks ? bricks->lod_wanted[7] : 0,
          (unsigned long long)(bricks ? bricks->lod_requests[0] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[1] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[2] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[3] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[4] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[5] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[6] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[7] : 0));
  static const char *names[10] = {"cpu_frame", "gpu_frame", "gpu_total", "gpu_raycast",
                                  "gpu_blit", "gpu_gui", "cpu_wait", "cpu_acquire",
                                  "cpu_record", "cpu_submit"};
  for (size_t i = 0; i < 10; i++) json_timing(f, names[i], &timing[i], i + 1 < 10);
  fputs("  }\n}\n", f);
  int rc = ferror(f) || fclose(f) != 0 ? -1 : 0;
  if (rc == 0) printf("benchmark json: %s\n", path);
  return rc;
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--probe") == 0) {
    r3d_vkctx vk;
    if (r3d_vkctx_create(&vk, NULL, 0, false) != 0) return EXIT_FAILURE;
    r3d_vkctx_print_caps(&vk);
    printf("c5d revision       : %s (GPU ABI %u)\n", R3D_C5D_REV,
           (unsigned)R3D_C5D_GPU_ABI);
    r3d_vkctx_destroy(&vk);
    return EXIT_SUCCESS;
  }

  /* automation flags (tests/CI): exit after N frames, dump a screenshot */
  uint32_t exit_frames = 0;
  uint32_t warmup_frames = 0;
  const char *shot_path = NULL;
  const char *bench_json = NULL;
  int force_mode = -1, tf_preset = -1;
  int win_w = 1280, win_h = 720;
  float cam0[5] = {0.5f, 0.5f, -1.5f, 0.0f, 0.0f}; /* pos, yaw, pitch */
  bool no_vsync = false;
  bool headless = false; /* --headless: no window/swapchain, offscreen only */
  bool mv_exercise_moving = false; /* R3D_MV_EXERCISE drives views without input events */
  double run_seconds = 0.0; /* --seconds S: exit after S s of wall time (with --frames: whichever first) */
  float lowcut0 = 0.0f;
  const char *bench = NULL; /* scripted camera path: orbit | zoom | fly */
  const char *bench_name = NULL;
  const char *quality_arg = "interactive";
  float volpos0[3] = {0, 0, 0}, volrot0[3] = {0, 0, 0};
  uint32_t slab_wz = 0;     /* nonzero = slab mode, max visible depth */
  int depth0 = 0;           /* initial visible depth; explicit 0 = full volume */
  bool depth_given = false;
  bool clip_mode = false;   /* clipmap over the shard band */
  const char *bricks_path = NULL; /* c5d shard for GPU-decoded bricks mode */
  int pool_bpa = 0, warm_mb = 0;  /* bricks hot-atlas slots/axis, warm-tier MB */
  int brick_z = -1, brick_depth = 0; /* global manifest XY slice; depth 0 = full volume */
  uint32_t brick_shape[3] = {0, 0, 0};
  bool brick_is_lod = false;
  uint64_t gpu_budget_bytes = 0;
  bool vslab_mode = false;        /* toroidal streaming window over the export */
  int vsw = 12096, vsh = 12096, vsd = 16; /* window dims (voxels) */
  long long vsz0 = 34288;         /* start z (world; default inside the local band) */
  uint32_t vsnx = 43008, vsny = 43008, vsnz = 68608;
  const char *vscache = "band", *vsurl = NULL;
  const char *umbilicus_path = NULL;
  const char *multiview_path = NULL; /* tifxyz dir: vc3d-style 2x2 viewer */
  int inklive_port = 0;             /* --inklive: live 2.5D ink on the seg view */
  r3d_inklive inklive = {.fd = -1};
  bool inklive_up = false, inklive_have = false;
  bool inklive_show = true; /* GUI toggle: display the ink overlay on the
                             * flattened pane (worker keeps predicting) */
  /* full-surface ink map state (see inkmap_save/load) */
  float *inkmap = NULL;
  float *inkmap_acc = NULL, *inkmap_wsum = NULL; /* tile-blend accumulators */
  uint32_t inkmap_w = 0, inkmap_h = 0, inkmap_up = 1;
  bool inkmap_have = false, inkmap_uploaded = false, inkmap_job = false;
  /* TTA/ensemble bitmask for full-map computes (bit0 flips, bit1 depth
   * shift, bit2 intensity, bit3 checkpoint ensemble); live view stays at
   * 0 unless explicitly opted in - TTA multiplies inference cost. */
  uint32_t inkmap_tta = 0x9; /* flips + ensemble: best quality/cost */
  bool ink_verso = false;    /* sample the slab from the reverse face
                              * (vc3d --direction both: verso ink) */
  /* background ink queue: harvested traces are ink-mapped automatically
   * (no TTA), one at a time, without touching the display */
  char ink_q[24][160];
  uint32_t ink_qn = 0;
  bool imq_active = false;
  char imq_path[1200];
  r3d_tifxyz imq_seg = {0};
  uint32_t im_tx = 0, im_ty = 0, im_ntx = 0, im_nty = 0, im_ts = 0;
  bool im_req_out = false;
  char inkmap_path[1200] = "";
  const char *seg_store_path = NULL; /* segpack store: draw ALL surfaces */
  const char *overlay_path = NULL;   /* active overlay c5d LOD root */
  const char *overlay_paths[8];      /* all --overlay trees (ink, surface preds...) */
  uint32_t n_overlays = 0;
  int overlay_sel = 0;
  char ink3d_root[640] = ""; /* --ink3d / browser: 3D-ink overlay (red, second
                              * atlas slot — shows WITH the surface preds) */
  bool ink3d_ok = false, ink3d_show = true;
  bool od_browse = false;            /* start with the open-data browser window */
  int sv_w = 2048, sv_h = 2048, sv_l = 128; /* flattened surface-volume window (1 GiB RG8) */
  int annotation_prefetch = 5; /* annotation steps ahead; one slot is kept behind */
  int annotation_z_prefetch = 32; /* contiguous GPU-resident fine-scroll margin */
  bool vsz_given = false;
  for (int i = 1; i < argc; i++) {
    if (i < argc - 1 && strcmp(argv[i], "--frames") == 0) exit_frames = (uint32_t)atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--seconds") == 0) run_seconds = atof(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--warmup") == 0)
      warmup_frames = (uint32_t)atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--bench-json") == 0) bench_json = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--shot") == 0) shot_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--mode") == 0) force_mode = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--tf") == 0) tf_preset = atoi(argv[i + 1]);
    if (i < argc - 2 && strcmp(argv[i], "--size") == 0) {
      win_w = atoi(argv[i + 1]);
      win_h = atoi(argv[i + 2]);
    }
    if (i < argc - 5 && strcmp(argv[i], "--cam") == 0)
      for (int k = 0; k < 5; k++) cam0[k] = (float)atof(argv[i + 1 + k]);
    if (strcmp(argv[i], "--no-vsync") == 0) no_vsync = true;
    if (strcmp(argv[i], "--headless") == 0) headless = true;
    if (i < argc - 1 && strcmp(argv[i], "--lowcut") == 0) lowcut0 = (float)atof(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--bench") == 0) bench = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--bench-name") == 0) bench_name = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--quality") == 0) quality_arg = argv[i + 1];
    if (i < argc - 3 && strcmp(argv[i], "--volpos") == 0)
      for (int k = 0; k < 3; k++) volpos0[k] = (float)atof(argv[i + 1 + k]);
    if (i < argc - 3 && strcmp(argv[i], "--volrot") == 0)
      for (int k = 0; k < 3; k++) volrot0[k] = (float)atof(argv[i + 1 + k]) / 57.29578f;
    if (i < argc - 1 && strcmp(argv[i], "--depth") == 0) {
      depth0 = atoi(argv[i + 1]);
      depth_given = true;
    }
    if (strcmp(argv[i], "--clipmap") == 0) clip_mode = true;
    if (i < argc - 1 && strcmp(argv[i], "--bricks") == 0) bricks_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--brick-z") == 0) brick_z = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--pool") == 0) pool_bpa = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--warm") == 0) warm_mb = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--gpu-mem") == 0)
      gpu_budget_bytes = (uint64_t)strtoull(argv[i + 1], NULL, 10) << 20;
    if (strcmp(argv[i], "--vslab") == 0) vslab_mode = true;
    if (i < argc - 1 && strcmp(argv[i], "--vsz") == 0) {
      vsz0 = atoll(argv[i + 1]);
      vsz_given = true;
    }
    if (i < argc - 3 && strcmp(argv[i], "--vswin") == 0) {
      vsw = atoi(argv[i + 1]);
      vsh = atoi(argv[i + 2]);
      vsd = atoi(argv[i + 3]);
    }
    if (i < argc - 1 && strcmp(argv[i], "--umbilicus") == 0) umbilicus_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--multiview") == 0) multiview_path = argv[i + 1];
    if (strcmp(argv[i], "--inklive") == 0) {
      inklive_port = 9743;
      if (i < argc - 1 && argv[i + 1][0] >= '1' && argv[i + 1][0] <= '9')
        inklive_port = atoi(argv[i + 1]);
    }
    if (i < argc - 1 && strcmp(argv[i], "--ink3d") == 0)
      snprintf(ink3d_root, sizeof ink3d_root, "%s", argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--overlay") == 0 &&
        n_overlays < sizeof overlay_paths / sizeof *overlay_paths)
      overlay_paths[n_overlays++] = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--segments") == 0) seg_store_path = argv[i + 1];
    if (strcmp(argv[i], "--browse") == 0) od_browse = true;
    if (i < argc - 3 && strcmp(argv[i], "--surfvol") == 0) {
      sv_w = atoi(argv[i + 1]);
      sv_h = atoi(argv[i + 2]);
      sv_l = atoi(argv[i + 3]);
    }
    if (i < argc - 1 && strcmp(argv[i], "--ann-prefetch") == 0)
      annotation_prefetch = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--ann-z-prefetch") == 0)
      annotation_z_prefetch = atoi(argv[i + 1]);
    if (strcmp(argv[i], "--slab") == 0) {
      slab_wz = 32;
      if (i < argc - 1 && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
        slab_wz = (uint32_t)atoi(argv[i + 1]);
    }
  }
  /* A bricks LOD session is always the 2x2 multiview: with no segment the
   * flattened pane starts blank (tracer-ready) and the plane views center on
   * the volume. The single-pane LOD slab remains reachable with --slab-view. */
  bool slab_view = false;
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "--slab-view") == 0) slab_view = true;
  if (bricks_path && !multiview_path && !slab_view && !umbilicus_path) multiview_path = "(none)";
  if (umbilicus_path && !multiview_path) { /* multiview annotates in-place;
                                            * standalone uses the vslab rig */
    vslab_mode = true;
    vsnx = PHERC1218_NX;
    vsny = PHERC1218_NY;
    vsnz = PHERC1218_NZ;
    vsw = PHERC1218_NX;
    vsh = PHERC1218_NY;
    vsd = 8;
    vscache = "cache/PHerc1218";
    vsurl = PHERC1218_SHARDS;
    if (!vsz_given) vsz0 = 0;
  }
  if (annotation_prefetch < 0 || annotation_prefetch >= R3D_VSLAB_PREFETCH_MAX) {
    fprintf(stderr, "--ann-prefetch must be between 0 and %u\n",
            R3D_VSLAB_PREFETCH_MAX - 1);
    return EXIT_FAILURE;
  }
  if (annotation_z_prefetch < 0 || annotation_z_prefetch > 128) {
    fprintf(stderr, "--ann-z-prefetch must be between 0 and 128\n");
    return EXIT_FAILURE;
  }
  if (bench && exit_frames == 0) exit_frames = 300;
  if (headless && !exit_frames && run_seconds <= 0.0)
    exit_frames = 1000; /* never run unattended forever */
  if (run_seconds > 0.0 && !exit_frames) exit_frames = 2000000u; /* time decides (cap keeps the per-frame sample buffer sane) */
  if (!exit_frames) warmup_frames = 0;
  if (exit_frames > UINT32_MAX - warmup_frames) {
    fprintf(stderr, "--frames + --warmup is too large\n");
    return EXIT_FAILURE;
  }
  uint32_t total_frames = exit_frames + warmup_frames;
  int quality_policy = 1; /* full=0, interactive=1, fast=2 */
  if (strcmp(quality_arg, "full") == 0) quality_policy = 0;
  else if (strcmp(quality_arg, "interactive") == 0) quality_policy = 1;
  else if (strcmp(quality_arg, "fast") == 0) quality_policy = 2;
  else {
    fprintf(stderr, "--quality must be full, interactive, or fast\n");
    return EXIT_FAILURE;
  }

  r3d_umbilicus umbilicus;
  r3d_umbilicus_init(&umbilicus);
  if (!umbilicus_path && bricks_path) {
    /* auto-discover: <bricks-root>/umbilicus.json (mkumb's output). The
     * winding frame powers the spiral prior, wrap gates, signed spacing
     * and werr QC - without it those defenses are silently inert. */
    static char auto_umb[1200];
    snprintf(auto_umb, sizeof auto_umb, "%s", bricks_path);
    char *sl = strrchr(auto_umb, '/');
    if (sl) {
      snprintf(sl + 1, sizeof auto_umb - (size_t)(sl + 1 - auto_umb),
               "umbilicus.json");
      FILE *tf = fopen(auto_umb, "r");
      if (tf) {
        fclose(tf);
        if (r3d_umbilicus_load(&umbilicus, auto_umb) == 0 && umbilicus.count >= 2)
          printf("umbilicus: auto-loaded %s (%zu points) - winding frame on\n",
                 auto_umb, umbilicus.count);
      }
    }
  }
  int annotation_step = 100;
  int annotation_z = (int)vsz0;
  int annotation_last_z = annotation_z, annotation_bench_z = annotation_z, annotation_dir = 1;
  char annotation_status[256] = "";
  if (umbilicus_path) {
    int urc = r3d_umbilicus_load(&umbilicus, umbilicus_path);
    if (urc < 0) {
      fprintf(stderr, "umbilicus: cannot load %s (expected Villa-compatible JSON)\n",
              umbilicus_path);
      r3d_umbilicus_free(&umbilicus);
      return EXIT_FAILURE;
    }
    if (urc == 0) {
      snprintf(annotation_status, sizeof annotation_status, "loaded %zu point%s",
               umbilicus.count, umbilicus.count == 1 ? "" : "s");
      if (!vsz_given && umbilicus.count) {
        double next = umbilicus.points[umbilicus.count - 1].z + annotation_step;
        annotation_z = (int)llround(next);
      }
    } else {
      snprintf(annotation_status, sizeof annotation_status, "new annotation");
    }
    if (annotation_z < 0) annotation_z = 0;
    if (annotation_z >= (int)vsnz) annotation_z = (int)vsnz - 1;
  }
  annotation_last_z = annotation_z;
  annotation_bench_z = annotation_z;

  /* headless: events subsystem only (the poll loop still runs), no window;
   * the renderer skips surface/swapchain/present and the GUI runs without a
   * platform backend. Everything else — streaming, bake, panes, GUI logic,
   * bench scripts, --shot, --bench-json — is identical to a windowed run. */
  if (!SDL_Init(headless ? SDL_INIT_EVENTS : SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }
  SDL_Window *win = NULL;
  if (!headless) {
    win = SDL_CreateWindow("render3d", win_w, win_h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);
    if (!win) {
      fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
      SDL_Quit();
      return EXIT_FAILURE;
    }
  }

  r3d_config cfg = {.validate = false,
                    .vsync = !no_vsync,
                    .spv_dir = R3D_SPV_DIR,
                    .gpu_budget_bytes = gpu_budget_bytes,
                    .headless = headless,
                    .headless_w = (uint32_t)win_w,
                    .headless_h = (uint32_t)win_h};
  r3d_renderer *renderer = NULL;
  if (r3d_create(win, &cfg, &renderer) != 0) {
    fprintf(stderr, "renderer init failed\n");
    if (win) SDL_DestroyWindow(win);
    SDL_Quit();
    return EXIT_FAILURE;
  }
  r3d_set_quality(renderer, quality_policy == 2 ? R3D_QUALITY_FAST : R3D_QUALITY_FULL);

  /* the dataset (volume + segment + overlay) is ordinary mutable state: the
   * open-data browser swaps it by tearing the bricks family down and
   * re-opening inside this loop, same window, same renderer */
  od_state od = {.sel_scroll = -1, .sel_vol = -1, .sel_seg = -1, .sel_variant = -1,
                 .sel_ovl = -1, .sel_ink = -1};
  bool od_window = od_browse;
  char od_next_bricks[640] = "", od_next_seg[560] = "", od_next_ovl[640] = "",
       od_next_ink[640] = "", od_next_reg[640] = "";
  bool od_swap = false, od_attach_ovl = false, od_attach_ink = false,
       od_attach_reg = false;
  for (;;) {
  if (od_swap) {
    od_swap = false;
    bricks_path = od_next_bricks[0] ? od_next_bricks : NULL;
    /* keep the 2x2 layout across a volume-only swap: with no segment the
     * multiview starts in its empty state (blank flattened pane, planes
     * centered on the volume, tracer ready) instead of dropping to the
     * single-pane slab view */
    /* volume-only swap keeps the 2x2 layout; CLOSE (no next volume)
     * drops to the plain empty view with the browser open instead of
     * exiting through the multiview-needs-bricks check */
    multiview_path =
        od_next_seg[0] ? od_next_seg : (od_next_bricks[0] ? "(none)" : NULL);
    if (!od_next_bricks[0]) od_window = true;
    overlay_path = NULL; /* else the old tree is reopened against the new
                          * volume: shape mismatch -> silent EXIT_FAILURE */
    if (od_next_ovl[0]) { /* browser-picked surface predictions */
      overlay_paths[0] = od_next_ovl;
      n_overlays = 1;
      overlay_sel = 0;
    } else {
      n_overlays = 0;
    }
    if (od_next_ink[0]) /* browser-picked 3D ink survives the swap */
      snprintf(ink3d_root, sizeof ink3d_root, "%s", od_next_ink);
    else
      ink3d_root[0] = 0; /* 3D ink attaches per volume (browser or --ink3d) */
    ink3d_ok = false;
    umbilicus_path = NULL;
    vslab_mode = false;
    clip_mode = false;
  }

  /* positional args: <volume.u8> <nx> <ny> <nz> */
  uint32_t mode = R3D_MODE_RAYDIR;
  r3d_volume slab_src = {0}; /* stays open in slab mode (window scrolls it) */
  uint32_t slab_z0 = 0, slab_z0_max = 0;
  if (argc >= 5 && argv[1][0] != '-') {
    r3d_volume vol;
    if (r3d_volume_open(&vol, argv[1], (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]),
                        (uint32_t)atoi(argv[4])) != 0)
      return EXIT_FAILURE;
    if (slab_wz) {
      /* --slab N = max visible depth; ring holds N+2 so face filtering never
       * crosses the seam and depth changes need no re-upload */
      uint32_t ring = slab_wz + 2;
      r3d_slab_desc sd = {.nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .wz = ring};
      if (r3d_slab_init(renderer, &sd) != 0) return EXIT_FAILURE;
      slab_src = vol;
      slab_z0 = (vol.nz - ring) / 2; /* start mid-depth */
      slab_z0_max = vol.nz - ring;
      if (r3d_slab_window(renderer, &slab_src, slab_z0) != 0) return EXIT_FAILURE;
    } else {
      r3d_volume_desc desc = {
          .nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .brick_dim = R3D_BRICK_DIM};
      int up = r3d_upload_volume(renderer, &desc, vol.voxels);
      r3d_volume_close(&vol); /* GPU has it; drop the mapping */
      if (up != 0) return EXIT_FAILURE;
    }
    mode = R3D_MODE_FULL;
  } else if (slab_wz) {
    fprintf(stderr, "--slab needs a volume argument\n");
    return EXIT_FAILURE;
  }

  if (bricks_path) {
    const char *pfe = getenv("R3D_POSTFILT"); /* headless/bench: force the
                                               * display filter (mode bits) */
    if (pfe) r3d_bricks_postfilter(renderer, (uint32_t)strtoul(pfe, NULL, 0), 1.0f, 1u);
    if (r3d_bricks_begin(renderer, bricks_path, (uint32_t)pool_bpa, (uint32_t)warm_mb) != 0)
      return EXIT_FAILURE;
    r3d_bricks_shape(renderer, brick_shape);
    r3d_bricks_stats initial_bst;
    r3d_bricks_get_stats(renderer, &initial_bst);
    brick_is_lod = initial_bst.nlevels > 1u;
    brick_depth = depth_given ? depth0 : (brick_is_lod ? 8 : 0);
    if (brick_depth < 0) brick_depth = 0;
    if (getenv("R3D_LBLTEST")) { /* headless/bench: enable 3D labelling and
                                  * stamp test blobs at the volume center */
      uint32_t ld[3] = {brick_shape[0], brick_shape[1], brick_shape[2]};
      if (r3d_labelvol_init(&g_lblv, ld) == 0) {
        r3d_label_src ls = {lblsrc_gen, lblsrc_fetch, &g_lblv};
        if (r3d_bricks_labels(renderer, &ls) == 0) {
          g_lbl_init = true;
          double c[3] = {(double)ld[0] * 0.5, (double)ld[1] * 0.5, (double)ld[2] * 0.5};
          for (uint8_t k = 1; k < R3D_LBL_NCLASS; k++) {
            double q[3] = {c[0] + (double)k * 40.0 - 180.0, c[1], c[2]};
            r3d_labelvol_paint(&g_lblv, q, 15.0, k);
          }
        } else
          r3d_labelvol_free(&g_lblv);
      }
    }
    const char *rte = getenv("R3D_REGTEST"); /* headless/bench: overlay a
                                              * moving scan; <root>[:tx,ty,tz] */
    if (rte && *rte) {
      char rr[640];
      snprintf(rr, sizeof rr, "%s", rte);
      char *co = strchr(rr, ':');
      double t3[3] = {0, 0, 0};
      if (co) {
        *co = 0;
        sscanf(co + 1, "%lf,%lf,%lf", &t3[0], &t3[1], &t3[2]);
      }
      uint32_t fd3[3] = {brick_shape[0], brick_shape[1], brick_shape[2]};
      if (reg_open_moving(renderer, rr, fd3, bricks_path)) {
        memcpy(g_reg.d_tr, t3, sizeof t3);
        r3d_regvol_bump(&g_reg);
      }
    }
    if (brick_depth > (int)brick_shape[2]) brick_depth = (int)brick_shape[2];
    if (brick_z < 0) brick_z = brick_depth ? ((int)brick_shape[2] - brick_depth) / 2 : 0;
    if (brick_z < 0) brick_z = 0;
    if (brick_z > (int)brick_shape[2] - brick_depth)
      brick_z = (int)brick_shape[2] - brick_depth;
    mode = R3D_MODE_FULL;
    if (n_overlays) overlay_path = overlay_paths[0];
    if (overlay_path && r3d_bricks_overlay(renderer, overlay_path) != 0) {
      fprintf(stderr, "overlay %s not usable with this volume; continuing without\n",
              overlay_path);
      overlay_path = NULL;
      n_overlays = 0;
    }
    if (ink3d_root[0]) {
      ink3d_ok = r3d_bricks_ink3d(renderer, ink3d_root) == 0;
      if (!ink3d_ok)
        fprintf(stderr, "3D ink %s not usable with this volume; continuing without\n",
                ink3d_root);
    }
  }
  bool overlay_show = overlay_path != NULL;
  float overlay_gain = 1.5f;

  /* vc3d-style 2x2 multi-view: flattened segment (TL, milestone C — an XY
   * overview until then) + XY/XZ/YZ ortho plane views, shared focus POI */
  r3d_tifxyz mv_seg = {0};
  r3d_mview mv[4] = {0};
  double mv_focus[3] = {0, 0, 0}; /* world voxels x,y,z */
  int mv_thick = 1;               /* plane-view slab thickness (voxels) */
  int mv_drag_view = -1;
  float *mv_normals = NULL; /* per-vertex normals kept for overlays/zoff shell */
  uint32_t mv_visible = 0xfu; /* per-view visibility; collapsed views cost nothing */
  uint32_t mv_ov_mask = 0xfu; /* which panes display the overlay tint */
  /* bottom panes pick from {XZ, YZ, 3D}: at most two of the three are live
   * (XZ/YZ carry similar info on a cylinder; XY is the real z-stack) */
  int mv_pane_kind[2] = {0, 1}; /* pane 2, pane 3: 0=XZ 1=YZ 2=3D volumetric */
  if (getenv("R3D_MV_3D")) mv_pane_kind[1] = 2; /* headless-shot default, like
                                                  * R3D_MV_STRETCH */
  /* 3D pane = a zoom-crop cube: wheel shrinks/grows the cube edge (higher
   * res as it shrinks — LOD follows the per-pixel footprint), right-drag
   * orbits around it, the camera keeps a fixed framing distance, and
   * Ctrl+click focus recenters it */
  double mv_crop_c[3] = {0, 0, 0}; /* cube center, world voxels */
  float mv_crop_d[3] = {0, 0, 0};  /* box extents, voxels; 0 = init to full.
                                    * GUI sliders make it any rectangle */
  float mv_crop_fit = 2.1f;        /* camera distance in cube edges
                                    * (Shift+wheel zooms the cube itself) */
  bool mv_vol_cam_init = false;
#define MV_KIND(i) ((i) == 2 ? mv_pane_kind[0] : (i) == 3 ? mv_pane_kind[1] : -1)
#define MV_IS3D(i) (MV_KIND(i) == 2)
  int mv_solo = -1;           /* Space on a hovered view maximizes it */
  bool mv_panel_open = true;  /* docked left panel; collapses to a slim bar */
  mv_lines mv_ol[4] = {0}, mv_ol_off[4] = {0}; /* intersection polylines */
  double mv_ol_slice[4] = {1e30, 1e30, 1e30, 1e30};
  double mv_ol_zoff = 1e30;
  r3d_segrows mv_rows = {0}; /* per-row coord bounds: segtrace row skipping */
  double mv_pb[4][3][3]; /* per-view plane frames (rows u, v, n) + origins */
  double mv_po[4][3] = {{0}};
  bool mv_aligned = false; /* XZ/YZ panes follow the segment normal */
  float mv_theta = 0.0f;   /* aligned-pane rotation around the normal (deg) */
  /* flattened view: distortion heatmap (env enables it for headless shots,
   * like R3D_FORCE_HALF) */
  bool mv_stretch = getenv("R3D_MV_STRETCH") != NULL;
  float mv_scrub = 1.0f; /* z-scrub sensitivity: voxels per wheel notch / R,F */
  /* umbilicus editing in the plane panes: plain click places the point for
   * its z (Shift+drag pans while on); off = clicks drag as usual */
  bool mv_umb_edit = umbilicus_path && multiview_path;
  uint32_t mv_umb_refoc = 0xfu; /* which panes recenter on placement (bit
                                 * per view: seg/XY/XZ/YZ); 0 = none */
  bool mv_umb_adv = false;  /* after placing, step the pane along its normal
                             * by the scrub-speed amount */
  bool mv_umb_back = false; /* ... in the negative direction */
  umb_snap mv_umb_undo[UMB_UNDO_MAX], mv_umb_redo[UMB_UNDO_MAX];
  uint32_t mv_umb_undo_n = 0, mv_umb_redo_n = 0;
  uint32_t mv_align_ij[2] = {0, 0}; /* surface grid point anchoring the frames */
  uint32_t mv_basis_gen = 0, mv_ol_gen[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) r3d_mv_axis_basis(i, mv_pb[i]);
  if (multiview_path) {
    if (!bricks_path || !brick_is_lod) {
      fprintf(stderr, "--multiview needs --bricks with a LOD manifest\n");
      return EXIT_FAILURE;
    }
    SDL_PumpEvents(); /* dataset swap: multi-second setup, stay responsive */
    if (r3d_tifxyz_load(&mv_seg, multiview_path) != 0) {
      /* virgin scroll, no segment yet: start with the close-segment empty
       * state — blank flattened pane, plane views centered on the volume,
       * ready for the tracer to grow the first patch */
      printf("multiview: no segment at %s — starting empty\n", multiview_path);
      mv_seg = (r3d_tifxyz){.w = 2, .h = 2, .sx = 0.05f, .sy = 0.05f};
      mv_seg.xyz = malloc(2 * 2 * 3 * sizeof *mv_seg.xyz);
      if (!mv_seg.xyz) return EXIT_FAILURE;
      for (int k = 0; k < 12; k++) mv_seg.xyz[k] = -1.0f;
    }
    if (r3d_segrows_build(&mv_seg, &mv_rows) != 0) return EXIT_FAILURE;
    SDL_PumpEvents();
    for (int a = 0; a < 3; a++)
      mv_focus[a] = ((double)mv_seg.bbox[0][a] + (double)mv_seg.bbox[1][a]) * 0.5;
    if (!mv_seg.nvalid) /* empty active surface: center on the volume */
      for (int a = 0; a < 3; a++) mv_focus[a] = (double)brick_shape[a] * 0.5;
    mv_align_ij[0] = mv_seg.w / 2;
    mv_align_ij[1] = mv_seg.h / 2;
    const float *mc = r3d_tifxyz_at(&mv_seg, mv_seg.w / 2, mv_seg.h / 2);
    if (r3d_tifxyz_valid(mc)) /* center the focus ON the sheet when possible */
      for (int a = 0; a < 3; a++) mv_focus[a] = (double)mc[a];
    brick_depth = 0; /* multiview owns per-view slab clips */
    mode = R3D_MODE_FULL; /* volumetric slabs in every quadrant (Tab: MIP etc.) */
    mv_thick = 24;
    for (int i = 0; i < 4; i++) {
      const uint8_t *ax = r3d_mv_axes[i];
      mv[i].cu = mv_focus[ax[0]];
      mv[i].cv = mv_focus[ax[1]];
      mv[i].slice = mv_focus[ax[2]];
      mv[i].zoom = 0.0; /* fitted on the first frame once quadrants exist */
    }
    /* TL = flattened segment view: grid-space camera, slice = normal offset */
    mv[R3D_MV_SEG].cu = (double)mv_seg.w * 0.5;
    mv[R3D_MV_SEG].cv = (double)mv_seg.h * 0.5;
    mv[R3D_MV_SEG].slice = 0.0;
    float *seg_coords = NULL;
    SDL_PumpEvents();
    if (mv_build_grids(&mv_seg, &seg_coords, &mv_normals) != 0 ||
        (SDL_PumpEvents(),
         r3d_surf_begin(renderer, mv_seg.w, mv_seg.h, seg_coords, mv_normals)) != 0) {
      fprintf(stderr, "multiview: surf grid upload failed\n");
      return EXIT_FAILURE;
    }
    free(seg_coords);
    /* flattened surface-volume window (RG8), resampled on the GPU from the
     * shared brick cache; --surfvol W H L overrides, device caps clamp */
    if (sv_w < 256) sv_w = 256;
    if (sv_h < 256) sv_h = 256;
    if (sv_l < 8) sv_l = 8;
    int dim3d = (int)r3d_max_dim3d(renderer); /* --surfvol vs device limit */
    if (sv_w > dim3d) sv_w = dim3d;
    if (sv_h > dim3d) sv_h = dim3d;
    if (sv_l > dim3d) sv_l = dim3d;
    if (r3d_surfvol_begin(renderer, (uint32_t)sv_w, (uint32_t)sv_h, (uint32_t)sv_l,
                          (uint32_t)sv_l / 2, mv_seg.sx, mv_seg.sy) != 0) {
      fprintf(stderr, "multiview: surface-volume window init failed\n");
      return EXIT_FAILURE;
    }
    if (inklive_up) { /* dataset swap: the sampler must follow the new CT tree */
      r3d_inklive_stop(&inklive);
      inklive_up = false;
      inklive_have = false;
      free(inkmap);
      free(inkmap_acc);
      inkmap_acc = NULL;
      free(inkmap_wsum);
      inkmap_wsum = NULL;
      inkmap = NULL;
      inkmap_have = inkmap_uploaded = inkmap_job = false;
      im_req_out = false;
      ink_qn = 0;
      if (imq_active) {
        r3d_tifxyz_free(&imq_seg);
        memset(&imq_seg, 0, sizeof imq_seg);
        imq_active = false;
      }
    }
    if (inklive_port) { /* live 2.5D ink worker (CT sampled via cpuvol on the
                         * same cache tree; predictions from inkserver.py) */
      char ilroot[1024];
      snprintf(ilroot, sizeof ilroot, "%s", bricks_path);
      char *mslash = strrchr(ilroot, '/');
      if (mslash && strcmp(mslash + 1, "manifest.json") == 0) *mslash = 0;
      if (r3d_inklive_start(&inklive, ilroot, inklive_port) == 0) {
        inklive_up = true;
        printf("inklive: worker up (CT %s, server port %d)\n", ilroot, inklive_port);
      } else {
        fprintf(stderr, "inklive: start failed (CT %s)\n", ilroot);
      }
    }
  }

  /* segment corpus: index + background decode cache + per-view polyline
   * caches; the frame loop only queries and traces, never decodes */
  sgcache sgc = {0};
  sgc_line *sgc_ln[4] = {NULL, NULL, NULL, NULL};
  uint32_t sgc_hits[4][128];
  uint32_t sgc_nhits[4] = {0, 0, 0, 0};
  double sgc_key_slice[4] = {1e30, 1e30, 1e30, 1e30};
  uint32_t sgc_key_gen[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
  char sgc_active[R3D_SEGSTORE_NAME] = {0};
  /* live surface tracers: several can grow CONCURRENTLY (each r3d_tracer
   * is self-contained - own worker + solve pool, thread-capped when
   * sharing cores); exactly ONE (gt_sel) is displayed in the seg pane
   * and receives anchors/refine/rewind/save. The rest grow in the
   * background and are switched to from the panel list. */
  #define GT_MAX 8
  struct gtrace {
    r3d_tracer tr;
    bool active, done, live_first;
    bool harvest; /* queue-started: auto-save + auto-ink on finish */
    double *pos;
    uint8_t *st;
    float *cf;
    uint64_t gen;
    uint32_t ring, nset;
  };
  static struct gtrace gts[GT_MAX]; /* static: r3d_tracer is large */
  memset(gts, 0, sizeof gts);
  int gt_sel = 0;
  #define GT (&gts[gt_sel])
  /* manual surface authoring ('P' over a plane view places a point along
   * a sheet; points cluster into z-rows and "create surface" fits a
   * tifxyz grid through them - which then flows through the normal
   * harvest pipeline: store + background ink map. The saved surfaces are
   * also hand-made training labels for the surface-prediction model.) */
  #define MSURF_MAX 1024
  double msurf_pts[MSURF_MAX][3];
  uint32_t msurf_n = 0;
  bool msurf_create = false;
  bool msurf_dirty = false; /* re-fit + preview in the flattened view */
  bool msurf_first = true;  /* fit-to-view on the first preview */
  /* boundary surface ('B' over a plane view seeds an automatic grower
   * that follows the papyrus/void edge defined by the render low cut;
   * grows on its own thread, previews live in the flattened pane, then
   * saves through the same harvest pipeline as a manual surface) */
  r3d_bsurf bsurf;
  memset(&bsurf, 0, sizeof bsurf);
  r3d_cpuvol bsurf_ct;
  memset(&bsurf_ct, 0, sizeof bsurf_ct);
  bool bsurf_ct_ok = false;
  bool bsurf_active = false; /* a grid exists (growing or finished) */
  bool bsurf_have_seed = false;
  double bsurf_seed[3] = {0, 0, 0};
  int bsurf_gens = 100;
  float bsurf_snap = 6.0f, bsurf_step = 4.0f;
  bool bsurf_go = false, bsurf_save = false, bsurf_discard = false;
  bool bsurf_first = true;
  uint64_t bsurf_gen = 0, bsurf_prev_ns = 0;
  /* queued seed points ('G' over a plane view). The queue is larger than
   * the concurrency: up to GT_MAX tracers run at once and the rest of
   * the queue drains automatically as slots free (discard or
   * save+activate a finished trace to free its slot). */
  #define SEEDS_MAX 100
  double mv_seeds[SEEDS_MAX * 3];
  uint32_t mv_seeds_n = 0;
  bool mv_seeds_go = false; /* draining: keep starting as slots free */
  float mv_tr_step = 20.0f, mv_tr_thresh = 0.35f;
  int mv_tr_rings = 60, mv_tr_nsaved = 0;
  bool mv_tr_spiral = true;
  bool mv_tr_fill = true;
  float mv_corpus_vis = 0.55f; /* corpus polyline alpha in the plane views */
  bool mv_tr_live = true;      /* render the growing grid in the seg pane */
  bool mv_tr_view = false; /* the flattened pane FOLLOWS the selected live
                            * trace; cleared when a store segment is
                            * activated so a finishing background trace
                            * never steals the view back */
  /* tracer anchors: world points the sheet must pass through (placed by
   * Ctrl+click in anchor mode; pushed to the tracer live) */
  double mv_anchor[R3D_TR_MAX_ANCHORS * 3];
  uint32_t mv_anchor_n = 0;
  uint64_t mv_tr_live_ns = 0;  /* last live swap (throttle) */
  double sgc_near_focus[3] = {1e30, 1e30, 1e30};
  uint32_t sgc_near[6];
  uint32_t sgc_nnear = 0;
  if (seg_store_path && !multiview_path)
    fprintf(stderr, "--segments needs --multiview; ignoring the store\n");
  if (seg_store_path && multiview_path) {
    if (sgc_open(&sgc, seg_store_path, (size_t)1536 << 20) != 0) {
      fprintf(stderr, "--segments: failed to open store %s\n", seg_store_path);
      return EXIT_FAILURE;
    }
    for (int i = 1; i < 4; i++) {
      sgc_ln[i] = calloc(sgc.st.n ? sgc.st.n : 1, sizeof *sgc_ln[i]);
      if (!sgc_ln[i]) return EXIT_FAILURE;
    }
    /* the active (flattened) segment draws in full color already */
    const char *sl_ = strrchr(multiview_path, '/');
    snprintf(sgc_active, sizeof sgc_active, "%s", sl_ && sl_[1] ? sl_ + 1 : multiview_path);
    printf("segments: %u surfaces, %llu index tiles from %s\n", sgc.st.n,
           (unsigned long long)sgc.st.ntiles, seg_store_path);
    for (uint32_t si = 0; si < sgc.st.n; si++) /* overlap-vs-active pass */
      if (strcmp(sgc.st.segs[si].name, sgc_active) == 0) {
        pthread_mutex_lock(&sgc.mu);
        sgc.ov_req = si;
        pthread_cond_signal(&sgc.cv);
        pthread_mutex_unlock(&sgc.mu);
        break;
      }
  }

  int64_t vs_z0 = umbilicus_path ? annotation_z0(annotation_z, vsnz, (uint32_t)vsd)
                                  : (int64_t)vsz0;
  double vs_fx = (double)vsnx * 0.5, vs_fy = (double)vsny * 0.5;
  bool vs_follow = true;
  uint64_t vs_pend_acc = 0;
  if (vslab_mode) {
    r3d_vslab_desc vd = {.nx = vsnx,
                         .ny = vsny,
                         .nz = vsnz,
                         .W = (uint32_t)vsw,
                         .H = (uint32_t)vsh,
                         .D = (uint32_t)vsd,
                         .cache_dir = vscache,
                         .shard_url = vsurl,
                         .z_prefetch = umbilicus_path ? (uint32_t)annotation_z_prefetch : 0u,
                         .prefetch_slots = umbilicus_path && annotation_prefetch
                                               ? (uint32_t)annotation_prefetch + 1u
                                               : 0u,
                         .prefetch_threads = umbilicus_path ? 6u : 0u};
    if (r3d_vslab_begin(renderer, &vd) != 0)
      return EXIT_FAILURE;
    mode = R3D_MODE_FULL;
    if (umbilicus_path) vs_follow = false; /* full XY plane is fixed at origin */
  }

  /* clipmap: 43k^2 cross sections from the shard band + pyramid */
  const uint32_t CLIP_NX = 43008, CLIP_BAND_Z = 33, CLIP_DEPTH_MAX = 32;
  uint64_t clip_z0 = 0, clip_z0_min = 0, clip_z0_max = 0;
  if (clip_mode) {
    if (r3d_clip_begin(renderer, "band", "pyramid", CLIP_BAND_Z, CLIP_DEPTH_MAX) != 0)
      return EXIT_FAILURE;
    clip_z0_min = (uint64_t)CLIP_BAND_Z * 1024;
    clip_z0_max = clip_z0_min + 1024 - CLIP_DEPTH_MAX;
    clip_z0 = clip_z0_min + 512 - CLIP_DEPTH_MAX / 2;
    mode = R3D_MODE_FULL;
  }
  if (force_mode >= 0) mode = (uint32_t)force_mode % R3D_MODE_COUNT;
  float tf_min_v = 1.0f; /* backend default ramp: alpha nonzero from value 1 */
  if (tf_preset >= 0) {
    r3d_tf tf;
    uint8_t lut[256][4];
    r3d_tf_preset((uint32_t)tf_preset, &tf);
    r3d_tf_build(&tf, lut);
    r3d_set_transfer(renderer, lut);
    tf_min_v = tf_min_visible(lut);
  }

  /* orbit (turntable around the volume) is the default; --cam implies fly */
  bool cam_given = false;
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "--cam") == 0) cam_given = true;
  int cam_mode = cam_given ? CAM_FLY : CAM_ORBIT;

  r3d_camera cam;
  r3d_camera_init(&cam, v3(cam0[0], cam0[1], cam0[2]));
  cam.yaw = cam0[3];
  cam.pitch = cam0[4];
  if (cam_mode == CAM_ORBIT) {
    if (clip_mode) {
      float ez = (float)CLIP_DEPTH_MAX / (float)CLIP_NX;
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, ez * 0.5f), 1.3f);
    } else if (vslab_mode) {
      float ey = (float)vsh / (float)vsw;
      float ez = (float)(vsd + 2) / (float)vsw;
      r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 1.4f);
    } else if (slab_wz) {
      /* orbit the thin slab: target its center, sit back along z */
      float ey = (float)slab_src.ny / (float)slab_src.nx;
      float ez = (float)slab_wz / (float)slab_src.nx;
      r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 1.4f);
    } else if (bricks_path) {
      float e[3];
      r3d_bricks_extent(renderer, e);
      float tz = e[2] * 0.5f;
      if (brick_depth > 0) {
        uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
        if (brick_shape[2] > md) md = brick_shape[2];
        tz = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
      }
      r3d_camera_orbit_set(&cam, v3(e[0] * 0.5f, e[1] * 0.5f, tz),
                           brick_depth > 0 ? 0.42f : 2.0f);
    } else {
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
    }
  }
  r3d_input in = {0};
  r3d_stats stats;
  r3d_stats_init(&stats);

  float step_voxels = 1.0f, density = 1.0f, lod_bias = 0.0f;
  float low_cut = lowcut0; /* voxel-value threshold, 0..255 */
  bool auto_scroll = false;
  float auto_speed = 10.0f; /* slices per second */
  float auto_accum = 0.0f;
  int slab_depth = depth0 >= 2 && depth0 <= (int)slab_wz ? depth0 : (int)slab_wz;
  int clip_depth = depth0 >= 2 && depth0 <= (int)CLIP_DEPTH_MAX ? depth0 : (int)CLIP_DEPTH_MAX;
  uint32_t clip_valid_disp = 0;

  /* volume (model) transform: translation in world units, rotation ypr */
  r3d_v3 vol_t = v3(volpos0[0], volpos0[1], volpos0[2]);
  float vol_rot[3] = {volrot0[0], volrot0[1], volrot0[2]}; /* yaw, pitch, roll (radians) */
  float fov_deg = 60.0f;
  uint32_t tf_idx = tf_preset > 0 ? (uint32_t)tf_preset : 0;
  uint32_t frame_index = 0;
  float fps_smooth = 60.0f;
  bool adaptive_res = quality_policy != 0; /* interactive/fast halve resolution while moving */
  int settle = 0;
  uint64_t last_gpu_ns = 0;
  /* main-thread phase profile: poll / nav / gui / stream / frame (ns) */
  enum { MT_POLL, MT_NAV, MT_GUI, MT_STREAM, MT_FRAME, MT_N };
  static const char *mt_name[MT_N] = {"poll", "nav", "gui", "stream", "frame"};
  uint64_t mt_sum[MT_N] = {0}, mt_max[MT_N] = {0}, mt_frames = 0;
  double mt_ema[MT_N] = {0};
  r3d_frame_stats prof = {0};   /* EMA-smoothed for display */
  r3d_frame_stats prof_sum = {0}; /* running sums for the exit report */
  uint64_t prof_frames = 0;
  r3d_frame_stats *prof_samples = exit_frames ? calloc(exit_frames, sizeof *prof_samples) : NULL;
  if (exit_frames && !prof_samples) return EXIT_FAILURE;
  uint64_t prev_ns = r3d_now_ns();

  bool running = true;
  uint32_t frame_fail = 0;
  uint64_t run_start_ns = r3d_now_ns();
  while (running) {
    uint64_t t0 = r3d_now_ns();
    if (run_seconds > 0.0 && (double)(t0 - run_start_ns) * 1e-9 >= run_seconds &&
        frame_index > warmup_frames)
      total_frames = frame_index; /* time is up: this frame is the last */
    if (exit_frames && frame_index == warmup_frames) {
      r3d_stats_init(&stats);
      memset(&prof_sum, 0, sizeof prof_sum);
      prof_frames = 0;
      vs_pend_acc = 0;
    }
    float dt = (float)((double)(t0 - prev_ns) / 1e9);
    prev_ns = t0;
    if (dt > 0.1f) dt = 0.1f;

    uint64_t mt_t[MT_N + 1];
    mt_t[0] = r3d_now_ns();
    ImGuiIO *io = igGetIO_Nil(); /* Want* flags reflect last frame — fine */
    r3d_input_poll(&in, win, gui_event_hook, renderer, !io->WantCaptureMouse,
                   cam_mode == CAM_FLY, umbilicus_path != NULL && !multiview_path,
                   multiview_path != NULL);
    mt_t[1] = r3d_now_ns();
    if (io->WantCaptureKeyboard && !in.captured)
      in.move[0] = in.move[1] = in.move[2] = 0.0f;
    if (in.quit) running = false;
    if (in.resized) r3d_resize(renderer);
    if (in.mode_delta) {
      mode = (mode + (uint32_t)in.mode_delta) % R3D_MODE_COUNT;
      printf("mode: %u\n", mode);
    }
    bool z_navigated = false;
    if (umbilicus_path && vslab_mode) { /* multiview panes own their wheel */
      int old_annotation_z = annotation_z;
      int64_t nz = annotation_z;
      nz += in.zdelta;
      nz += (int64_t)in.zpage * annotation_step;
      /* In annotation mode the ordinary wheel traverses z. Shift+wheel is
       * retained as camera zoom for inspecting a difficult center. Wheel is
       * deliberately fine-grained; buttons/pages retain annotation_step. */
      if (!in.fast && in.wheel != 0.0f && !io->WantCaptureMouse) {
        nz += in.wheel > 0.0f ? 1 : -1;
        in.wheel = 0.0f;
      }
      if (nz < 0) nz = 0;
      if (nz >= vsnz) nz = (int64_t)vsnz - 1;
      annotation_z = (int)nz;
      z_navigated = annotation_z != old_annotation_z;
      vs_z0 = annotation_z0(annotation_z, vsnz, (uint32_t)vsd);
    }
    if (bricks_path && brick_depth > 0 && !umbilicus_path && !multiview_path) {
      int old_brick_z = brick_z;
      int64_t nz = (int64_t)brick_z + in.zdelta + (int64_t)in.zpage * brick_depth;
      /* Global LOD slices use the same fine wheel semantics as annotation:
       * wheel changes one z slice, Shift+wheel retains camera zoom. */
      if (!in.fast && in.wheel != 0.0f && !io->WantCaptureMouse) {
        nz += in.wheel > 0.0f ? 1 : -1;
        in.wheel = 0.0f;
      }
      int64_t zmax = (int64_t)brick_shape[2] - brick_depth;
      if (nz < 0) nz = 0;
      if (nz > zmax) nz = zmax;
      brick_z = (int)nz;
      z_navigated = z_navigated || brick_z != old_brick_z;
      if (cam_mode == CAM_ORBIT && brick_z != old_brick_z) {
        uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
        if (brick_shape[2] > md) md = brick_shape[2];
        float dz = (float)(brick_z - old_brick_z) / (float)md;
        cam.target.z += dz;
        cam.pos.z += dz;
      }
    }
    if (clip_mode) {
      int64_t nz0 = (int64_t)clip_z0 + in.zdelta + (int64_t)in.zpage * (int64_t)clip_depth;
      if (nz0 < (int64_t)clip_z0_min) nz0 = (int64_t)clip_z0_min;
      if (nz0 > (int64_t)clip_z0_max) nz0 = (int64_t)clip_z0_max;
      clip_z0 = (uint64_t)nz0;
    }
    if (slab_wz) {
      int64_t nz0 = (int64_t)slab_z0 + in.zdelta + (int64_t)in.zpage * (int64_t)slab_depth;
      if (bench && strcmp(bench, "zsweep") == 0) /* scripted scroll for perf/tests */
        nz0 = (int64_t)((float)slab_z0_max *
                        (float)(frame_index < warmup_frames ? frame_index
                                                           : frame_index - warmup_frames) /
                        (float)(frame_index < warmup_frames && warmup_frames ? warmup_frames
                                                                            : exit_frames));
      if (auto_scroll) {
        auto_accum += auto_speed * dt;
        float whole = floorf(auto_accum);
        nz0 += (int64_t)whole;
        auto_accum -= whole;
        if (nz0 >= (int64_t)slab_z0_max) { /* bounce at the ends */
          nz0 = slab_z0_max;
          auto_speed = -auto_speed;
        } else if (nz0 <= 0 && auto_speed < 0) {
          nz0 = 0;
          auto_speed = -auto_speed;
        }
      }
      if (nz0 < 0) nz0 = 0;
      if (nz0 > (int64_t)slab_z0_max) nz0 = slab_z0_max;
      if ((uint32_t)nz0 != slab_z0) {
        slab_z0 = (uint32_t)nz0;
        r3d_slab_window(renderer, &slab_src, slab_z0);
      }
    }
    if (in.tf_delta) {
      tf_idx = (tf_idx + 1) % r3d_tf_preset(UINT32_MAX, NULL);
      r3d_tf tf;
      uint8_t lut[256][4];
      r3d_tf_preset(tf_idx, &tf);
      r3d_tf_build(&tf, lut);
      r3d_set_transfer(renderer, lut);
      tf_min_v = tf_min_visible(lut);
      printf("tf preset: %u\n", tf_idx);
    }
    step_voxels *= in.step_scale;
    density *= in.density_scale;
    lod_bias += in.lod_delta;
    cam.fov_y = fov_deg * 0.01745329f;
    if (multiview_path) {
      /* per-view interaction is handled after the quadrant layout below */
    } else if (cam_mode == CAM_ORBIT) {
      /* drag grabs the cube; Shift pans the camera; Ctrl translates the
       * volume; Ctrl+Shift rotates the volume */
      if (in.dragging && (in.ctrl || in.fast)) {
        r3d_v3 br, bu, bf;
        r3d_camera_basis(&cam, 1.0f, &br, &bu, &bf);
        r3d_v3 ru = v3_norm(br), uu = v3_norm(bu);
        float k = cam.dist * 0.0012f;
        if (in.ctrl && in.fast) { /* rotate volume */
          vol_rot[0] += in.look[0] * ORBIT_SENS;
          vol_rot[1] += in.look[1] * ORBIT_SENS;
        } else if (in.ctrl) { /* translate volume in the view plane */
          vol_t = v3_add(vol_t, v3_add(v3_scale(ru, in.look[0] * k),
                                       v3_scale(uu, -in.look[1] * k)));
        } else { /* pan camera (grab-the-world) */
          r3d_camera_orbit_pan(&cam, v3(-in.look[0], in.look[1], 0), k);
        }
      } else if (in.dragging) {
        r3d_camera_orbit_drag(&cam, -in.look[0] * ORBIT_SENS, -in.look[1] * ORBIT_SENS);
      }
      if (in.wheel != 0.0f && !io->WantCaptureMouse)
        /* shift+wheel zooms 3x faster: whole-scroll to fiber scale is a ~3200x
         * distance ratio, a long ride at 0.9/detent */
        r3d_camera_orbit_zoom(&cam, powf(in.fast ? 0.73f : 0.9f, in.wheel));
      float pan = BASE_SPEED * cam.dist * (in.fast ? 5.0f : 1.0f);
      if (in.move[0] != 0.0f || in.move[1] != 0.0f || in.move[2] != 0.0f)
        r3d_camera_orbit_pan(&cam, v3(in.move[0], in.move[1], in.move[2]), pan * dt);
    } else {
      r3d_camera_look(&cam, in.look[0] * MOUSE_SENS, -in.look[1] * MOUSE_SENS);
      float speed = BASE_SPEED * (in.fast ? 5.0f : 1.0f);
      r3d_camera_move(&cam, v3(in.move[0], in.move[1], in.move[2]), speed * dt);
    }

    int w = 0, h = 0;
    if (win) {
      SDL_GetWindowSizeInPixels(win, &w, &h);
    } else {
      w = win_w;
      h = win_h;
    }
    if (w <= 0 || h <= 0) {
      SDL_Delay(50);
      continue;
    }

    if (sgc.open && getenv("R3D_ACTIVATE_TEST") &&
        frame_index == (uint32_t)atoi(getenv("R3D_ACTIVATE_TEST")) && sgc.st.n) {
      /* headless: activate store segment 0 at this frame (repro/testing) */
      pthread_mutex_lock(&sgc.mu);
      if (sgc.act_req == UINT32_MAX && !sgc.act_busy) {
        sgc.act_req = 0;
        pthread_cond_signal(&sgc.cv);
      }
      pthread_mutex_unlock(&sgc.mu);
    }
    if (sgc.open) { /* apply a finished activation: swap the flattened view
       * to the worker-prepared segment (decode/grids were built off-thread;
       * only the GPU upload happens here) */
      pthread_mutex_lock(&sgc.mu);
      if (sgc.act_ready != UINT32_MAX) {
        uint32_t ai = sgc.act_ready;
        r3d_tifxyz ns = sgc.act_s;
        r3d_segrows nr = sgc.act_rows;
        float *co = sgc.act_coords, *no = sgc.act_normals;
        sgc.act_ready = UINT32_MAX;
        sgc.act_busy = false;
        sgc.act_coords = sgc.act_normals = NULL;
        pthread_mutex_unlock(&sgc.mu);
        if (r3d_surf_swap(renderer, ns.w, ns.h, co, no, ns.sx, ns.sy) == 0) {
          smask_drop(renderer, sgc_active); /* mask belongs to the old segment */
          r3d_tifxyz_free(&mv_seg);
          r3d_segrows_free(&mv_rows);
          free(mv_normals);
          mv_seg = ns;
          mv_rows = nr;
          mv_normals = no;
          snprintf(sgc_active, sizeof sgc_active, "%s", sgc.st.segs[ai].name);
          mv[R3D_MV_SEG].cu = (double)mv_seg.w * 0.5;
          mv[R3D_MV_SEG].cv = (double)mv_seg.h * 0.5;
          for (int i = 0; i < 4; i++) {
            mv_ol[i].n = 0;
            mv_ol_off[i].n = 0;
            mv_ol_slice[i] = 1e30;
          }
          mv_ol_zoff = 1e30;
          const float *mc2 = r3d_tifxyz_at(&mv_seg, mv_seg.w / 2, mv_seg.h / 2);
          if (r3d_tifxyz_valid(mc2))
            for (int a = 0; a < 3; a++) mv_focus[a] = (double)mc2[a];
          mv_align_ij[0] = mv_seg.w / 2;
          mv_align_ij[1] = mv_seg.h / 2;
          if (mv_aligned) {
            if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                              mv_po))
              mv_axis_reset(mv_pb, mv_po);
          }
          mv_basis_gen++; /* re-trace every cached overlay against the swap */
          for (int i = 1; i < 4; i++) { /* recenter planes on the new focus */
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
            mv[i].cu = fu;
            mv[i].cv = fv;
            mv[i].slice = fs;
          }
          mv_tr_view = false; /* the pane now shows a store segment; live
                               * traces keep growing without stealing it */
          inklive_have = false; /* prediction of the previous segment is stale */
          r3d_surfvol_inkpred_clear(renderer); /* and its texture must not
                                                * tint the new surface */
          if (!imq_active) { /* a queue job owns the buffer; leave it */
            free(inkmap);
            free(inkmap_acc);
            inkmap_acc = NULL;
            free(inkmap_wsum);
            inkmap_wsum = NULL;
            inkmap = NULL;
            inkmap_job = false;
            im_req_out = false;
          }
          inkmap_have = inkmap_uploaded = false;
          inkmap_path[0] = 0;
          if (seg_store_path && sgc_active[0] && !imq_active) {
            snprintf(inkmap_path, sizeof inkmap_path, "%s/%s%s.inkmap", seg_store_path,
                     sgc_active, ink_verso ? "-verso" : "");
            uint32_t lw, lh, lup;
            /* the copy living WITH the segment wins; store cache second */
            char sdp[400];
            snprintf(sdp, sizeof sdp, "cache/traced/%s/ink%s.inkmap", sgc_active,
                     ink_verso ? "-verso" : "");
            float *lm = inkmap_load(sdp, &lw, &lh, &lup, mv_seg.w, mv_seg.h,
                                    mv_seg.nvalid);
            if (!lm)
              lm = inkmap_load(inkmap_path, &lw, &lh, &lup, mv_seg.w,
                               mv_seg.h, mv_seg.nvalid);
            if (lm) {
              inkmap = lm;
              inkmap_w = lw;
              inkmap_h = lh;
              inkmap_up = lup;
              inkmap_have = true; /* upload happens in the frame loop */
              printf("inklive: cached ink map found (%ux%u)\n", lw, lh);
            }
            char smp[400]; /* the segment's saved supervision mask, if any */
            snprintf(smp, sizeof smp, "cache/traced/%s/mask.inkmask", sgc_active);
            if (smask_load_file(smp, mv_seg.w, mv_seg.h, mv_seg.nvalid))
              printf("inklive: supervision mask loaded (%ux%u)\n", g_smask_w,
                     g_smask_h);
          }
          if (r3d_tifxyz_valid(mc2) &&
              (mc2[0] >= (float)brick_shape[0] || mc2[1] >= (float)brick_shape[1] ||
               mc2[2] >= (float)brick_shape[2]))
            fprintf(stderr,
                    "warning: segment %s center (%.0f,%.0f,%.0f) lies outside this "
                    "volume (%ux%ux%u) - traced on a different scroll?\n",
                    sgc_active, (double)mc2[0], (double)mc2[1], (double)mc2[2],
                    brick_shape[0], brick_shape[1],
                    brick_shape[2]);
          printf("activated segment %s (%ux%u)\n", sgc_active, mv_seg.w, mv_seg.h);
          pthread_mutex_lock(&sgc.mu); /* refresh overlap-vs-active */
          sgc.ov_req = ai;
          pthread_cond_signal(&sgc.cv);
          pthread_mutex_unlock(&sgc.mu);
        } else {
          r3d_tifxyz_free(&ns);
          r3d_segrows_free(&nr);
          free(no);
          fprintf(stderr, "segment activation failed (surf grid swap)\n");
        }
        free(co);
      } else {
        pthread_mutex_unlock(&sgc.mu);
      }
    }
    uint32_t mv_mask = mv_solo >= 0 ? 1u << mv_solo : (mv_visible ? mv_visible : 0xfu);
    int mv_panel_w = multiview_path ? (mv_panel_open ? 360 : 26) : 0;
    if (mv_panel_w >= w) mv_panel_w = 0;
    if (multiview_path) {
      r3d_mview lay[4];
      r3d_mv_layout_mask(lay, w - mv_panel_w, h, mv_mask);
      for (int i = 0; i < 4; i++) {
        mv[i].px = lay[i].px + (lay[i].pw ? mv_panel_w : 0);
        mv[i].py = lay[i].py;
        mv[i].pw = lay[i].pw;
        mv[i].ph = lay[i].ph;
      }
      if (getenv("R3D_MV_EXERCISE")) {
        mv_exercise_moving = true; /* counts as interaction (adaptive res) */
        /* automated GUI interaction for profiling: scrub slices, wiggle
         * zoom and pan — drives overlay recompute, surfvol re-bakes and
         * streaming exactly like a user session, reproducibly */
        double ph = (double)frame_index * 0.02;
        for (int i = 1; i < 4; i++)
          mv[i].slice += (frame_index % 20) < 10 ? 1.0 : -1.0;
        mv[R3D_MV_SEG].slice = 24.0 * sin(ph);
        r3d_mview *ev = &mv[1 + (frame_index / 120) % 3];
        r3d_mv_zoom(ev, (float)(ev->px + ev->pw / 2), (float)(ev->py + ev->ph / 2),
                    (frame_index % 240) < 120 ? 1.01 : 0.99, 1.0 / 256.0, 10.0);
        ev->cu += 3.0 * sin(ph * 1.7);
        ev->cv += 3.0 * cos(ph * 1.3);
      }
      if (in.view_toggle) { /* Space: solo/restore the hovered view */
        int hv = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (mv_solo >= 0) mv_solo = -1;
        else if (hv >= 0) mv_solo = hv;
        mv_mask = mv_solo >= 0 ? 1u << mv_solo : (mv_visible ? mv_visible : 0xfu);
        r3d_mv_layout_mask(lay, w - mv_panel_w, h, mv_mask);
        for (int i = 0; i < 4; i++) {
          mv[i].px = lay[i].px + (lay[i].pw ? mv_panel_w : 0);
          mv[i].py = lay[i].py;
          mv[i].pw = lay[i].pw;
          mv[i].ph = lay[i].ph;
        }
      }
      if (mv[0].zoom <= 0.0) { /* first frame: fit */
        double fy2 = (double)mv[R3D_MV_SEG].ph / (double)(mv_seg.h ? mv_seg.h : 1);
        double fx2 = (double)mv[R3D_MV_SEG].pw / (double)(mv_seg.w ? mv_seg.w : 1);
        mv[R3D_MV_SEG].zoom = fy2 < fx2 ? fy2 : fx2;
        double fitv = getenv("R3D_MV_FIT") ? strtod(getenv("R3D_MV_FIT"), NULL) : 0.0;
        if (!(fitv >= 64.0)) fitv = 2048.0; /* headless: voxels shown vertically */
        for (int i = 1; i < 4; i++) mv[i].zoom = (double)mv[i].ph / fitv;
      }
      int hover = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
      if (in.dragging && mv_drag_view < 0) mv_drag_view = hover;
      if (!in.dragging) mv_drag_view = -1;
      if (mv_drag_view >= 0 && MV_IS3D(mv_drag_view) &&
          (in.look[0] != 0.0f || in.look[1] != 0.0f)) {
        r3d_camera_orbit_drag(&cam, in.look[0] * 0.005f, in.look[1] * 0.005f);
      } else if (mv_drag_view >= 0 && (in.look[0] != 0.0f || in.look[1] != 0.0f)) {
        r3d_mview *dv = &mv[mv_drag_view];
        dv->cu -= (double)in.look[0] / dv->zoom;
        dv->cv -= (double)in.look[1] / dv->zoom;
      }
      if (hover >= 0 && MV_IS3D(hover) && in.wheel != 0.0f && !io->WantCaptureMouse) {
        if (in.wheel_shift) { /* Shift+wheel: zoom the cube on screen */
          mv_crop_fit *= powf(0.92f, in.wheel);
          if (mv_crop_fit < 0.6f) mv_crop_fit = 0.6f;
          if (mv_crop_fit > 8.0f) mv_crop_fit = 8.0f;
        } else if (mv_crop_d[0] > 0.0f) { /* wheel: crop tighter/looser,
                                            * all axes in proportion */
          float sc = powf(0.9f, in.wheel);
          for (int a = 0; a < 3; a++) {
            mv_crop_d[a] *= sc;
            if (mv_crop_d[a] < 32.0f) mv_crop_d[a] = 32.0f;
            if (brick_shape[a] && mv_crop_d[a] > (float)brick_shape[a])
              mv_crop_d[a] = (float)brick_shape[a];
          }
        }
        in.wheel = 0.0f;
      }
      if (hover >= 0 && MV_IS3D(hover) && mv_crop_d[0] > 0.0f &&
          (in.adelta[0] || in.adelta[1] || in.zpage) && !io->WantCaptureKeyboard) {
        /* arrows pan the crop through the volume (x/y), PgUp/PgDn along z;
         * step = 5% of that axis's extent, so travel scales with zoom */
        mv_crop_c[0] += in.adelta[0] * (double)mv_crop_d[0] * 0.05;
        mv_crop_c[1] += in.adelta[1] * (double)mv_crop_d[1] * 0.05;
        mv_crop_c[2] += in.zpage * (double)mv_crop_d[2] * 0.05;
        in.zpage = 0; /* consumed: not a slice page */
        for (int a = 0; a < 3; a++) {
          if (mv_crop_c[a] < 0.0) mv_crop_c[a] = 0.0;
          if (brick_shape[a] && mv_crop_c[a] > (double)brick_shape[a])
            mv_crop_c[a] = (double)brick_shape[a];
        }
      }
      if (hover >= 0 && in.wheel != 0.0f && !io->WantCaptureMouse) {
        r3d_mview *hv = &mv[hover];
        if (in.wheel_shift) { /* scrub the slice along the view normal */
          hv->slice += (double)(in.wheel > 0.0f ? 1 : -1) * (double)mv_scrub;
        } else {
          /* max zoom ~10 screen pixels per source voxel; the segment view's
           * zoom unit is a grid step (1/scale voxels), so scale its cap */
          double zmax = hover == R3D_MV_SEG ? 10.0 / (double)mv_seg.sx : 10.0;
          r3d_mv_zoom(hv, in.mouse_xy[0], in.mouse_xy[1],
                      pow(1.05, (double)in.wheel * 2.0), 1.0 / 256.0, zmax);
        }
        in.wheel = 0.0f;
      }
      if (hover >= 0 && (in.zdelta || in.zpage))
        mv[hover].slice += (double)(in.zdelta + in.zpage * 10) * (double)mv_scrub;
      /* SEG slice = normal offset, symmetric around the sheet */
      if (mv[R3D_MV_SEG].slice < -512.0) mv[R3D_MV_SEG].slice = -512.0;
      if (mv[R3D_MV_SEG].slice > 512.0) mv[R3D_MV_SEG].slice = 512.0;
      for (int i = 1; i < 4; i++) { /* plane slices clamp to the volume */
        if (MV_IS3D(i)) continue;
        if (mv_aligned && i >= R3D_MV_XZ) {
          /* frame offset from the focus, not a world coordinate: keep it
           * within the volume diagonal so the pane can always scrub back */
          double lim = (double)brick_shape[0] + brick_shape[1] + brick_shape[2];
          if (mv[i].slice < -lim) mv[i].slice = -lim;
          if (mv[i].slice > lim) mv[i].slice = lim;
          continue;
        }
        uint32_t n = brick_shape[r3d_mv_axes[i][2]];
        if (mv[i].slice < 0.0) mv[i].slice = 0.0;
        if (n && mv[i].slice > (double)n - 1.0) mv[i].slice = (double)n - 1.0;
      }
      if (umbilicus_path && mv_umb_edit && in.umb_place && !io->WantCaptureKeyboard) {
        /* U places the umbilicus point at the cursor, in ANY plane pane —
         * the core can be ambiguous in one orientation and obvious in
         * another. Keyed by the position's (rounded) z; mouse buttons keep
         * their normal pan/zoom roles. */
        int cu_ = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (cu_ > 0 && !MV_IS3D(cu_)) {
          double uu, vv, W[3];
          r3d_mv_unproject(&mv[cu_], in.mouse_xy[0], in.mouse_xy[1], &uu, &vv);
          r3d_mv_b2w(mv_pb[cu_], mv_po[cu_], uu, vv, mv[cu_].slice, W);
          for (int a = 0; a < 3; a++) {
            if (W[a] < 0.0) W[a] = 0.0;
            if (brick_shape[a] && W[a] > (double)brick_shape[a] - 1.0)
              W[a] = (double)brick_shape[a] - 1.0;
          }
          umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
          umb_snap_clear(mv_umb_redo, &mv_umb_redo_n); /* a new edit forks */
          if (r3d_umbilicus_set(&umbilicus, W[0], W[1], (double)llround(W[2])) == 0)
            save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
          /* recenter the selected panes on it, like Ctrl+click */
          if (!mv_umb_refoc) goto umb_placed;
          memcpy(mv_focus, W, sizeof mv_focus);
          uint32_t ij[2];
          if (mv_nearest_surface(&mv_seg, mv_focus, ij)) {
            mv_align_ij[0] = ij[0];
            mv_align_ij[1] = ij[1];
            if (mv_umb_refoc & 1u) {
              mv[R3D_MV_SEG].cu = (double)ij[0];
              mv[R3D_MV_SEG].cv = (double)ij[1];
            }
          }
          if (mv_aligned) {
            if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                              mv_po))
              mv_axis_reset(mv_pb, mv_po);
            mv_basis_gen++;
          }
          memcpy(mv_crop_c, mv_focus, sizeof mv_crop_c); /* 3D crop follows */
          for (int i = 1; i < 4; i++) {
            if (!(mv_umb_refoc & (1u << i))) continue;
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
            mv[i].cu = fu;
            mv[i].cv = fv;
            mv[i].slice = fs;
          }
        umb_placed:;
          if (mv_umb_adv) /* step the placement pane by the scrub speed */
            mv[cu_].slice += (mv_umb_back ? -1.0 : 1.0) * (double)mv_scrub;
        }
      }
      if (umbilicus_path && mv_umb_edit && in.undo && !io->WantCaptureKeyboard &&
          mv_umb_undo_n) { /* Ctrl+Z: back to the previous point set */
        umb_undo_push(mv_umb_redo, &mv_umb_redo_n, &umbilicus);
        umb_snap_apply(&umbilicus, mv_umb_undo[--mv_umb_undo_n]);
        save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        snprintf(annotation_status, sizeof annotation_status, "undo (%u back, %u fwd)",
                 mv_umb_undo_n, mv_umb_redo_n);
      }
      if (umbilicus_path && mv_umb_edit && in.redo && !io->WantCaptureKeyboard &&
          mv_umb_redo_n) { /* Ctrl+Shift+Z: forward again */
        umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
        umb_snap_apply(&umbilicus, mv_umb_redo[--mv_umb_redo_n]);
        save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        snprintf(annotation_status, sizeof annotation_status, "redo (%u back, %u fwd)",
                 mv_umb_undo_n, mv_umb_redo_n);
      }
      if (in.anchor_place && !io->WantCaptureMouse) {
        /* X over a plane pane = drop a tracer anchor at the voxel under the
         * cursor (on that pane's slice); the sheet must pass through it */
        int av = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (av > 0 && !MV_IS3D(av) && av != R3D_MV_SEG &&
            mv_anchor_n < R3D_TR_MAX_ANCHORS) {
          double u, vq, A[3];
          r3d_mv_unproject(&mv[av], in.mouse_xy[0], in.mouse_xy[1], &u, &vq);
          r3d_mv_b2w(mv_pb[av], mv_po[av], u, vq, mv[av].slice, A);
          bool inside = true;
          for (int a = 0; a < 3; a++)
            if (A[a] < 0.0 || (brick_shape[a] && A[a] > (double)brick_shape[a] - 1.0))
              inside = false;
          if (inside) {
            memcpy(mv_anchor + (size_t)mv_anchor_n * 3, A, sizeof A);
            mv_anchor_n++;
            if (GT->active) r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
            printf("tracer: anchor %u placed at (%.0f, %.0f, %.0f)\n", mv_anchor_n,
                   A[0], A[1], A[2]);
          }
        }
      }
      if (in.seed_place && !io->WantCaptureMouse) {
        /* G over a plane pane = queue a trace seed at the voxel under the
         * cursor; "trace all seeds" grows them concurrently */
        int av = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (av > 0 && !MV_IS3D(av) && av != R3D_MV_SEG && mv_seeds_n < SEEDS_MAX) {
          double u, vq, A[3];
          r3d_mv_unproject(&mv[av], in.mouse_xy[0], in.mouse_xy[1], &u, &vq);
          r3d_mv_b2w(mv_pb[av], mv_po[av], u, vq, mv[av].slice, A);
          bool inside = true;
          for (int a = 0; a < 3; a++)
            if (A[a] < 0.0 || (brick_shape[a] && A[a] > (double)brick_shape[a] - 1.0))
              inside = false;
          if (inside) {
            memcpy(mv_seeds + (size_t)mv_seeds_n * 3, A, sizeof A);
            mv_seeds_n++;
            printf("tracer: seed %u queued at (%.0f, %.0f, %.0f)\n", mv_seeds_n,
                   A[0], A[1], A[2]);
          }
        }
      }
      if (in.surf_place && !io->WantCaptureMouse) {
        /* P over a plane pane = place a manual-surface point at the voxel
         * under the cursor */
        int av = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (av > 0 && !MV_IS3D(av) && av != R3D_MV_SEG && msurf_n < MSURF_MAX) {
          double u, vq, A[3];
          r3d_mv_unproject(&mv[av], in.mouse_xy[0], in.mouse_xy[1], &u, &vq);
          r3d_mv_b2w(mv_pb[av], mv_po[av], u, vq, mv[av].slice, A);
          bool inside = true;
          for (int a = 0; a < 3; a++)
            if (A[a] < 0.0 || (brick_shape[a] && A[a] > (double)brick_shape[a] - 1.0))
              inside = false;
          if (inside) {
            memcpy(msurf_pts[msurf_n], A, sizeof A);
            msurf_n++;
            msurf_dirty = true; /* live preview follows every point */
            mv_tr_view = false;
          }
        }
      }
      if (in.bnd_place && !io->WantCaptureMouse) {
        /* B over a plane pane = seed the boundary-surface grower at the
         * voxel under the cursor (replaces the previous seed) */
        int av = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (av > 0 && !MV_IS3D(av) && av != R3D_MV_SEG) {
          double u, vq, A[3];
          r3d_mv_unproject(&mv[av], in.mouse_xy[0], in.mouse_xy[1], &u, &vq);
          r3d_mv_b2w(mv_pb[av], mv_po[av], u, vq, mv[av].slice, A);
          bool inside = true;
          for (int a = 0; a < 3; a++)
            if (A[a] < 0.0 || (brick_shape[a] && A[a] > (double)brick_shape[a] - 1.0))
              inside = false;
          if (inside) {
            memcpy(bsurf_seed, A, sizeof A);
            bsurf_have_seed = true;
            printf("boundary surface: seed at (%.0f, %.0f, %.0f)\n", A[0], A[1], A[2]);
          }
        }
      }
      { /* label painting: plain LMB drag over a plane pane while the labels
         * panel's paint mode is on (plain LMB is otherwise idle in the
         * multiview; Ctrl+LMB stays the focus gesture) */
        bool painted = false;
        if (g_lbl_init && g_lbl_paint && in.lmb_held && !in.ctrl && !io->WantCaptureMouse) {
          int av = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
          if (av > 0 && !MV_IS3D(av) && av != R3D_MV_SEG) {
            double u, vq, A[3];
            r3d_mv_unproject(&mv[av], in.mouse_xy[0], in.mouse_xy[1], &u, &vq);
            r3d_mv_b2w(mv_pb[av], mv_po[av], u, vq, mv[av].slice, A);
            bool inside = true;
            for (int a = 0; a < 3; a++)
              if (A[a] < 0.0 || (brick_shape[a] && A[a] > (double)brick_shape[a] - 1.0))
                inside = false;
            if (inside) {
              lbl_stroke_to(A, (double)g_lbl_radius, (uint8_t)g_lbl_class);
              painted = true;
            }
          }
        }
        if (!painted) g_lbl_stroke = false;
      }
      { /* supervision-mask painting: plain LMB drag over the FLATTENED pane
         * while the mask paint mode is on (plain LMB is idle there too) */
        bool mpainted = false;
        if (g_smask_paint && mv_seg.w > 2 && in.lmb_held && !in.ctrl &&
            !io->WantCaptureMouse) {
          int av = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
          if (av == R3D_MV_SEG) {
            double su, sv;
            r3d_mv_unproject(&mv[av], in.mouse_xy[0], in.mouse_xy[1], &su, &sv);
            if (su >= 0.0 && sv >= 0.0 && su < (double)mv_seg.w - 1.0 &&
                sv < (double)mv_seg.h - 1.0) {
              if (!g_smask) { /* lazy alloc at the ink-map pixel convention */
                uint32_t iup = (uint32_t)lround(1.0 / (double)mv_seg.sx);
                if (iup < 1) iup = 1;
                g_smask_up = iup;
                g_smask_w = (mv_seg.w - 1) * iup;
                g_smask_h = (mv_seg.h - 1) * iup;
                g_smask_gw = mv_seg.w;
                g_smask_gh = mv_seg.h;
                g_smask_nvalid = mv_seg.nvalid;
                g_smask = calloc((size_t)g_smask_w * g_smask_h, 1);
              }
              if (g_smask) {
                double rad = (double)g_smask_brush;
                uint8_t cls = (uint8_t)g_smask_class;
                if (g_smask_stroke) { /* stamp along the drag */
                  double dx = su - g_smask_prev[0], dy = sv - g_smask_prev[1];
                  double dist = sqrt(dx * dx + dy * dy);
                  double step = rad * 0.5 > 0.2 ? rad * 0.5 : 0.2;
                  int nstep = (int)ceil(dist / step);
                  if (nstep < 1) nstep = 1;
                  for (int k = 1; k <= nstep; k++) {
                    double t = (double)k / (double)nstep;
                    smask_stamp(g_smask_prev[0] + dx * t, g_smask_prev[1] + dy * t,
                                rad, cls);
                  }
                } else
                  smask_stamp(su, sv, rad, cls);
                g_smask_stroke = true;
                g_smask_prev[0] = su;
                g_smask_prev[1] = sv;
                mpainted = true;
              }
            }
          }
        }
        if (!mpainted) g_smask_stroke = false;
      }
      if (in.annotate_click && in.click_ctrl) { /* Ctrl+click = set focus POI */
        int cv_ = r3d_mv_hit(mv, in.click_xy[0], in.click_xy[1]);
        bool focused = false, have_ij = false;
        if (MV_IS3D(cv_) && mv_crop_d[0] > 0.0f) {
          /* 3D pane: the click ray meets the view-facing plane through the
           * crop center — lateral pick at the cube's depth (iterate from a
           * plane view for exact depth) */
          uint32_t md3 = brick_shape[0];
          for (int a = 1; a < 3; a++)
            if (brick_shape[a] > md3) md3 = brick_shape[a];
          float mde = mv_crop_d[0] > mv_crop_d[1] ? mv_crop_d[0] : mv_crop_d[1];
          if (mv_crop_d[2] > mde) mde = mv_crop_d[2];
          double eb3 = (double)mde / md3;
          r3d_camera_orbit_set(&cam,
                               v3((float)(mv_crop_c[0] / md3), (float)(mv_crop_c[1] / md3),
                                  (float)(mv_crop_c[2] / md3)),
                               (float)(eb3 * (double)mv_crop_fit));
          r3d_v3 c_r, c_u, c_f;
          r3d_camera_basis(&cam, (float)mv[cv_].pw / (float)(mv[cv_].ph ? mv[cv_].ph : 1),
                           &c_r, &c_u, &c_f);
          float nx = ((in.click_xy[0] - (float)mv[cv_].px) / (float)mv[cv_].pw) * 2.0f - 1.0f;
          float ny = 1.0f - ((in.click_xy[1] - (float)mv[cv_].py) / (float)mv[cv_].ph) * 2.0f;
          r3d_v3 dir = v3_norm(v3_add(c_f, v3_add(v3_scale(c_r, nx), v3_scale(c_u, ny))));
          float dz = v3_dot(dir, c_f);
          if (dz > 1e-5f) {
            float tt = v3_dot(v3_sub(cam.target, cam.pos), c_f) / dz;
            r3d_v3 P = v3_add(cam.pos, v3_scale(dir, tt));
            double W3[3] = {(double)P.x * md3, (double)P.y * md3, (double)P.z * md3};
            for (int a = 0; a < 3; a++) { /* stay within the box */
              double half3 = (double)mv_crop_d[a] * 0.5;
              if (W3[a] < mv_crop_c[a] - half3) W3[a] = mv_crop_c[a] - half3;
              if (W3[a] > mv_crop_c[a] + half3) W3[a] = mv_crop_c[a] + half3;
              mv_focus[a] = W3[a];
            }
            focused = true;
          }
          cv_ = -1; /* the shared tail below handles the rest */
        } else if (MV_IS3D(cv_)) {
          cv_ = -1;
        }
        if (cv_ == R3D_MV_SEG) {
          /* focus at the surface point under the cursor (CPU bilinear tap) */
          double gu, gv;
          r3d_mv_unproject(&mv[cv_], in.click_xy[0], in.click_xy[1], &gu, &gv);
          if (gu >= 0.0 && gv >= 0.0 && gu <= (double)mv_seg.w - 1.0 &&
              gv <= (double)mv_seg.h - 1.0) {
            uint32_t gi = (uint32_t)gu, gj = (uint32_t)gv;
            if (gi > mv_seg.w - 2) gi = mv_seg.w - 2;
            if (gj > mv_seg.h - 2) gj = mv_seg.h - 2;
            const float *q00 = r3d_tifxyz_at(&mv_seg, gi, gj);
            const float *q10 = r3d_tifxyz_at(&mv_seg, gi + 1, gj);
            const float *q01 = r3d_tifxyz_at(&mv_seg, gi, gj + 1);
            const float *q11 = r3d_tifxyz_at(&mv_seg, gi + 1, gj + 1);
            if (r3d_tifxyz_valid(q00) && r3d_tifxyz_valid(q10) && r3d_tifxyz_valid(q01) &&
                r3d_tifxyz_valid(q11)) {
              double fx_ = gu - gi, fy_ = gv - gj;
              for (int a = 0; a < 3; a++)
                mv_focus[a] = ((double)q00[a] * (1 - fx_) + (double)q10[a] * fx_) * (1 - fy_) +
                              ((double)q01[a] * (1 - fx_) + (double)q11[a] * fx_) * fy_;
              focused = true;
              have_ij = true; /* the clicked cell anchors the aligned frames */
              mv_align_ij[0] = fx_ < 0.5 ? gi : gi + 1;
              mv_align_ij[1] = fy_ < 0.5 ? gj : gj + 1;
            }
          }
        } else if (cv_ > 0) {
          double u, vq;
          r3d_mv_unproject(&mv[cv_], in.click_xy[0], in.click_xy[1], &u, &vq);
          r3d_mv_b2w(mv_pb[cv_], mv_po[cv_], u, vq, mv[cv_].slice, mv_focus);
          focused = true;
        }
        if (focused) {
          for (int a = 0; a < 3; a++) { /* clamp into the volume */
            if (mv_focus[a] < 0.0) mv_focus[a] = 0.0;
            if (brick_shape[a] && mv_focus[a] > (double)brick_shape[a] - 1.0)
              mv_focus[a] = (double)brick_shape[a] - 1.0;
          }
          if (!have_ij) {
            /* recenter the segment view on the surface point nearest the
             * focus (vc3d 100-voxel tolerance); it also anchors the frames */
            uint32_t ij[2];
            if (mv_nearest_surface(&mv_seg, mv_focus, ij)) {
              mv_align_ij[0] = ij[0];
              mv_align_ij[1] = ij[1];
              mv[R3D_MV_SEG].cu = (double)ij[0];
              mv[R3D_MV_SEG].cv = (double)ij[1];
            }
          }
          if (mv_aligned) {
            if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                              mv_po))
              mv_axis_reset(mv_pb, mv_po);
            mv_basis_gen++;
          }
          memcpy(mv_crop_c, mv_focus, sizeof mv_crop_c); /* 3D crop follows */
          for (int i = 1; i < 4; i++) { /* recenter planes through the focus */
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
            mv[i].cu = fu;
            mv[i].cv = fv;
            mv[i].slice = fs;
          }
        }
      }

      if (mv_mask & 1u) { /* keep the flattened surface-volume window under the
         * view (snapped for hysteresis) and rebuild on residency arrivals;
         * a collapsed segment view skips baking entirely */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        double vox_per_px = 1.0 / (sv->zoom * (double)mv_seg.sx);
        double stepd = 1.0;
        while (stepd * 2.0 <= vox_per_px) stepd *= 2.0;
        double cu_vox = sv->cu / (double)mv_seg.sx, cv_vox = sv->cv / (double)mv_seg.sy;
        double snap = (double)sv_w / 8.0 * stepd; /* window W/8 */
        double u0 = floor((cu_vox - (double)sv_w * 0.5 * stepd) / snap) * snap;
        double v0 = floor((cv_vox - (double)sv_h * 0.5 * stepd) / snap) * snap;
        double zsnap = 24.0; /* layers are 1 voxel regardless of xy zoom */
        double z0 = floor(sv->slice / zsnap + 0.5) * zsnap;
        /* visible sub-box hint: full-window rebuilds bake this box in-frame
         * and refresh the rest progressively */
        double half_u = (double)sv->pw * 0.5 * vox_per_px;
        double half_v = (double)sv->ph * 0.5 / (sv->zoom * (double)mv_seg.sy);
        double tx0 = (cu_vox - half_u - u0) / stepd, tx1 = (cu_vox + half_u - u0) / stepd;
        double ty0 = (cv_vox - half_v - v0) / stepd, ty1 = (cv_vox + half_v - v0) / stepd;
        int lc = sv_l / 2 + (int)lround(sv->slice - z0);
        int lz0 = lc - mv_thick, lz1 = lc + mv_thick + 1;
        r3d_surfvol_visible(renderer, (uint32_t)(tx0 > 0.0 ? tx0 : 0.0),
                            (uint32_t)(ty0 > 0.0 ? ty0 : 0.0), (uint32_t)(lz0 > 0 ? lz0 : 0),
                            (uint32_t)(tx1 > 0.0 ? tx1 + 1.0 : 0.0),
                            (uint32_t)(ty1 > 0.0 ? ty1 + 1.0 : 0.0),
                            (uint32_t)(lz1 > 0 ? lz1 : 0));
        r3d_surfvol_window(renderer, u0, v0, (float)stepd, (float)z0);
        /* residency-arrival re-bakes are now triggered by the renderer at
         * page-table publication (the only moment the new bricks are actually
         * visible to the bake kernel); the old decode-counter poll here fired
         * a frame early, missed publications, and cost two stats locks/frame */
        if (inklive_up && !inkmap_job && !inkmap_have && getenv("R3D_INKMAP_TEST") &&
            mv_seg.w > 2 && mv_seg.nvalid > 16) {
          {
            const char *tm = getenv("R3D_INKMAP_TTA"); /* mask override */
            if (tm) inkmap_tta = (uint32_t)strtoul(tm, NULL, 0);
          }
          /* headless test: start the full-map pass as soon as a segment
           * is up (the GUI button lives in a collapsed panel) */
          uint32_t iup = (uint32_t)lround(1.0 / (double)mv_seg.sx);
          if (iup < 1) iup = 1;
          uint32_t its = inkmap_tile_cells(iup);
          inkmap_up = iup;
          inkmap_w = (mv_seg.w - 1) * iup;
          inkmap_h = (mv_seg.h - 1) * iup;
          free(inkmap);
          free(inkmap_acc);
          inkmap_acc = NULL;
          free(inkmap_wsum);
          inkmap_wsum = NULL;
          inkmap = calloc((size_t)inkmap_w * inkmap_h, sizeof *inkmap);
          if (inkmap) {
            im_ts = its;
            im_ntx = (mv_seg.w - 2) / its + 1;
            im_nty = (mv_seg.h - 2) / its + 1;
            im_tx = im_ty = 0;
            im_req_out = false;
            inkmap_job = true;
            printf("inklive: computing full ink map, %ux%u px in %u tiles\n",
                   inkmap_w, inkmap_h, im_ntx * im_nty);
          }
        }
        if (inklive_up && !inkmap_job && ink_qn && seg_store_path) {
          /* start the next background ink map (harvest chain, no TTA) */
          char dir2[320];
          snprintf(dir2, sizeof dir2, "cache/traced/%s", ink_q[0]);
          if (r3d_tifxyz_load(&imq_seg, dir2) == 0 && imq_seg.w > 2) {
            uint32_t iup = (uint32_t)lround(1.0 / (double)imq_seg.sx);
            if (iup < 1) iup = 1;
            uint32_t its = inkmap_tile_cells(iup);
            /* the display map (if any) yields its buffer; its disk cache
             * survives and reloads on the next activation */
            inkmap_have = inkmap_uploaded = false;
            inkmap_up = iup;
            inkmap_w = (imq_seg.w - 1) * iup;
            inkmap_h = (imq_seg.h - 1) * iup;
            free(inkmap);
            free(inkmap_acc);
            inkmap_acc = NULL;
            free(inkmap_wsum);
            inkmap_wsum = NULL;
            inkmap = calloc((size_t)inkmap_w * inkmap_h, sizeof *inkmap);
            if (inkmap) {
              im_ts = its;
              im_ntx = (imq_seg.w - 2) / its + 1;
              im_nty = (imq_seg.h - 2) / its + 1;
              im_tx = im_ty = 0;
              im_req_out = false;
              inkmap_job = true;
              imq_active = true;
              snprintf(imq_path, sizeof imq_path, "%s/%s.inkmap", seg_store_path,
                       ink_q[0]);
              printf("inklive: queued ink map for %s (%ux%u px, %u tiles, "
                     "%u more queued)\n", ink_q[0], inkmap_w, inkmap_h,
                     im_ntx * im_nty, ink_qn - 1);
            } else {
              r3d_tifxyz_free(&imq_seg);
            }
          } else {
            r3d_tifxyz_free(&imq_seg);
            fprintf(stderr, "inklive: cannot load %s for ink queue\n", dir2);
          }
          memmove(ink_q, ink_q + 1, (size_t)(ink_qn - 1) * sizeof ink_q[0]);
          ink_qn--;
        }
        if (inklive_up && inkmap_have && !inkmap_job) {
          /* a cached full-surface map exists: no live inference at all */
          if (!inkmap_uploaded &&
              r3d_surfvol_inkpred(renderer, inkmap, inkmap_w, inkmap_h, 0.0f, 0.0f,
                                  (float)inkmap_up) == 0) {
            inkmap_uploaded = true;
            inklive_have = true;
            printf("inklive: full ink map displayed (%ux%u)\n", inkmap_w, inkmap_h);
          }
          const float *pred; /* drain any stale in-flight prediction */
          uint32_t d0, d1, d2, d3, d4;
          if (r3d_inklive_poll(&inklive, &pred, &d0, &d1, &d2, &d3, &d4) &&
              inkmap_uploaded)
            r3d_surfvol_inkpred(renderer, inkmap, inkmap_w, inkmap_h, 0.0f, 0.0f,
                                (float)inkmap_up);
        } else if (inklive_up && inkmap_job) {
          /* full-map tile pass: one outstanding request at a time; each
           * finished tile stitches into the map (margins cropped) and the
           * partial map re-uploads so progress is visible */
          uint32_t up = inkmap_up;
          const r3d_tifxyz *IMS = imq_active ? &imq_seg : &mv_seg;
          const uint32_t m = INKMAP_MARGIN_CELLS; /* context margin, cells */
          if (!im_req_out) {
            uint32_t c0x = im_tx * im_ts, c0y = im_ty * im_ts;
            uint32_t c1x = c0x + im_ts, c1y = c0y + im_ts;
            if (c1x > IMS->w - 1) c1x = IMS->w - 1;
            if (c1y > IMS->h - 1) c1y = IMS->h - 1;
            uint32_t r0x = c0x > m ? c0x - m : 0, r0y = c0y > m ? c0y - m : 0;
            uint32_t r1x = c1x + m < IMS->w - 1 ? c1x + m : IMS->w - 1;
            uint32_t r1y = c1y + m < IMS->h - 1 ? c1y + m : IMS->h - 1;
            uint32_t rw = r1x - r0x, rh = r1y - r0y;
            float *xyz = malloc((size_t)(rw + 1) * (rh + 1) * 3 * sizeof *xyz);
            if (xyz && rw && rh) {
              for (uint32_t cj = 0; cj <= rh; cj++)
                for (uint32_t ci = 0; ci <= rw; ci++) {
                  const float *sp = r3d_tifxyz_at(IMS, r0x + ci, r0y + cj);
                  float *dst = xyz + ((size_t)cj * (rw + 1) + ci) * 3;
                  if (r3d_tifxyz_valid(sp)) {
                    dst[0] = sp[0];
                    dst[1] = sp[1];
                    dst[2] = sp[2];
                  } else {
                    dst[0] = dst[1] = dst[2] = -1.0f;
                  }
                }
              r3d_inklive_request(&inklive, xyz, r0x, r0y, rw, rh, up,
                                  imq_active ? 0u : inkmap_tta,
                                  imq_active ? false : ink_verso);
              im_req_out = true;
            } else {
              free(xyz);
              inkmap_job = false; /* degenerate tile: abort the pass */
            }
          }
          const float *pred;
          uint32_t pw2, ph2, pi0, pj0, pup;
          if (im_req_out &&
              r3d_inklive_poll(&inklive, &pred, &pw2, &ph2, &pi0, &pj0, &pup)) {
            { /* accept only THIS tile's result (a superseded request from
                 an aborted pass could still be in flight) */
              uint32_t e0x = im_tx * im_ts, e0y = im_ty * im_ts;
              uint32_t er0x = e0x > m ? e0x - m : 0, er0y = e0y > m ? e0y - m : 0;
              if (pi0 != er0x || pj0 != er0y) goto ink_poll_done;
            }
            im_req_out = false;
            uint32_t c0x = im_tx * im_ts, c0y = im_ty * im_ts;
            uint32_t c1x = c0x + im_ts, c1y = c0y + im_ts;
            if (c1x > IMS->w - 1) c1x = IMS->w - 1;
            if (c1y > IMS->h - 1) c1y = IMS->h - 1;
            /* blended stitch: weight 1 inside the tile core, linear ramp
             * to 0 across the context margin; overlapping margins
             * accumulate prob*w and w and normalize (the server's own
             * inter-patch scheme, applied between tiles) so boundaries
             * are seamless instead of hard-edged */
            if (!inkmap_acc)
              inkmap_acc = calloc((size_t)inkmap_w * inkmap_h, sizeof *inkmap_acc);
            if (!inkmap_wsum)
              inkmap_wsum = calloc((size_t)inkmap_w * inkmap_h, sizeof *inkmap_wsum);
            if (!inkmap_acc || !inkmap_wsum) {
              inkmap_job = false; /* allocation failed: abort the pass */
              goto ink_poll_done;
            }
            {
              double mpx = (double)INKMAP_MARGIN_CELLS * pup;
              double bx0 = (double)((uint64_t)c0x * pup);
              double bx1 = (double)((uint64_t)c1x * pup) - 1.0;
              double by0 = (double)((uint64_t)c0y * pup);
              double by1 = (double)((uint64_t)c1y * pup) - 1.0;
              for (uint32_t sy = 0; sy < ph2; sy++) {
                uint32_t py = pj0 * pup + sy;
                if (py >= inkmap_h) break;
                double wy = 1.0;
                if ((double)py < by0) wy = 1.0 - (by0 - (double)py) / mpx;
                else if ((double)py > by1) wy = 1.0 - ((double)py - by1) / mpx;
                if (wy <= 0.0) continue;
                for (uint32_t sx2 = 0; sx2 < pw2; sx2++) {
                  uint32_t px = pi0 * pup + sx2;
                  if (px >= inkmap_w) break;
                  double wx = 1.0;
                  if ((double)px < bx0) wx = 1.0 - (bx0 - (double)px) / mpx;
                  else if ((double)px > bx1) wx = 1.0 - ((double)px - bx1) / mpx;
                  if (wx <= 0.0) continue;
                  float wgt = (float)(wx * wy);
                  size_t k = (size_t)py * inkmap_w + px;
                  inkmap_acc[k] += pred[(size_t)sy * pw2 + sx2] * wgt;
                  inkmap_wsum[k] += wgt;
                  inkmap[k] = inkmap_acc[k] / inkmap_wsum[k];
                }
              }
            }
            if (!imq_active) { /* live progress on the displayed segment */
              r3d_surfvol_inkpred(renderer, inkmap, inkmap_w, inkmap_h, 0.0f, 0.0f,
                                  (float)pup);
              inklive_have = true;
            }
            if (++im_tx >= im_ntx) {
              im_tx = 0;
              im_ty++;
            }
            if (im_ty >= im_nty && imq_active) {
              /* queue job done: save + free, never display (it is another
               * segment's map) */
              inkmap_job = false;
              imq_active = false;
              if (inkmap_save(imq_path, inkmap, inkmap_w, inkmap_h, inkmap_up,
                              IMS->w, IMS->h, IMS->nvalid) == 0)
                printf("inklive: queued ink map saved -> %s\n", imq_path);
              { /* the detection travels with the surface */
                const char *bn = strrchr(imq_path, '/');
                char nm[200];
                snprintf(nm, sizeof nm, "%s", bn ? bn + 1 : imq_path);
                char *dot = strstr(nm, ".inkmap");
                if (dot) *dot = 0;
                inkmap_save_segdir(nm, inkmap, inkmap_w, inkmap_h, inkmap_up,
                                   IMS->w, IMS->h, IMS->nvalid, false);
              }
              free(inkmap);
              free(inkmap_acc);
              inkmap_acc = NULL;
              free(inkmap_wsum);
              inkmap_wsum = NULL;
              inkmap = NULL;
              r3d_tifxyz_free(&imq_seg);
              memset(&imq_seg, 0, sizeof imq_seg);
            } else if (im_ty >= im_nty) { /* done: cache + switch to map mode */
              inkmap_job = false;
              inkmap_have = true;
              inkmap_uploaded = true;
              if (inkmap_path[0] &&
                  inkmap_save(inkmap_path, inkmap, inkmap_w, inkmap_h, inkmap_up,
                              IMS->w, IMS->h, IMS->nvalid) == 0)
                printf("inklive: full ink map saved -> %s\n", inkmap_path);
              if (sgc_active[0]) /* the detection travels with the surface */
                inkmap_save_segdir(sgc_active, inkmap, inkmap_w, inkmap_h,
                                   inkmap_up, IMS->w, IMS->h, IMS->nvalid, ink_verso);
            } else {
              printf("inklive: ink map tile %u/%u\n", im_ty * im_ntx + im_tx,
                     im_ntx * im_nty);
            }
          }
        ink_poll_done:;
        }
        /* NOTE: there is deliberately no view-driven inference. 2.5D ink
         * runs only when the user asks for it (the full-map compute); the
         * old live mode re-inferred every viewport rect forever and threw
         * the results away. */
      }
    }

    /* scripted camera paths for reproducible perf runs (override user input) */
    if (bench) {
      uint32_t phase_frame = frame_index < warmup_frames ? frame_index
                                                         : frame_index - warmup_frames;
      uint32_t phase_count = frame_index < warmup_frames && warmup_frames ? warmup_frames
                                                                          : exit_frames;
      float ph = phase_count ? (float)phase_frame / (float)phase_count : 0.0f;
      float tau = 6.2831853f;
      if (strcmp(bench, "orbit") == 0) {
        cam.yaw = ph * tau;
        cam.pitch = 0.5f * sinf(ph * tau * 2.0f);
        r3d_v3 target = v3(0.5f, 0.5f, 0.5f);
        if (bricks_path) {
          float e[3];
          r3d_bricks_extent(renderer, e);
          target = v3(e[0] * 0.5f, e[1] * 0.5f, e[2] * 0.5f);
          if (brick_depth > 0) {
            uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
            if (brick_shape[2] > md) md = brick_shape[2];
            target.z = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
          }
        }
        r3d_camera_orbit_set(&cam, target, 2.0f);
      } else if (strcmp(bench, "zoom") == 0) {
        cam.yaw = ph * tau * 0.5f;
        cam.pitch = 0.2f;
        r3d_v3 target = v3(0.5f, 0.5f, 0.5f);
        if (bricks_path) {
          float e[3];
          r3d_bricks_extent(renderer, e);
          target = v3(e[0] * 0.5f, e[1] * 0.5f, e[2] * 0.5f);
          if (brick_depth > 0) {
            uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
            if (brick_shape[2] > md) md = brick_shape[2];
            target.z = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
          }
        }
        r3d_camera_orbit_set(&cam, target, 1.75f - 1.55f * sinf(ph * 3.14159265f));
      } else if (strcmp(bench, "fly") == 0) { /* weaving pass through the volume */
        cam.pos = v3(0.5f + 0.15f * sinf(ph * tau * 1.5f), 0.5f + 0.1f * sinf(ph * tau),
                     -0.3f + 1.6f * ph);
        cam.yaw = 0.15f * sinf(ph * tau);
        cam.pitch = 0.1f * cosf(ph * tau);
      } else if (strcmp(bench, "clippan") == 0) { /* sweep across the cross-section */
        cam.pos = v3(0.25f + 0.5f * ph, 0.5f, -0.02f);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
      } else if (strcmp(bench, "zoomio") == 0) {
        /* full zoom sweep: whole-composite view down to voxel scale and back
         * (log-space triangle wave) — walks every LOD band the mode has */
        float tx = 0.5f;
        float ey = slab_wz ? (float)slab_src.ny / (float)slab_src.nx : 1.0f;
        float ez = slab_wz ? (float)slab_wz / (float)slab_src.nx : 1.0f;
        if (vslab_mode) { ey = (float)vsh / (float)vsw; ez = (float)(vsd + 2) / (float)vsw; }
        if (bricks_path) {
          float e[3];
          r3d_bricks_extent(renderer, e);
          tx = e[0] * 0.5f; ey = e[1]; ez = e[2];
        }
        float tri = ph < 0.5f ? ph * 2.0f : 2.0f - ph * 2.0f; /* 0->1->0 */
        float d = 1.6f * powf(0.002f / 1.6f, tri);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
        float tz = ez * 0.5f;
        if (bricks_path && brick_depth > 0) {
          uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
          if (brick_shape[2] > md) md = brick_shape[2];
          tz = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
        }
        r3d_camera_orbit_set(&cam, v3(tx, ey * 0.5f, tz), d);
      } else if (strcmp(bench, "volrot") == 0) {
        /* mid-zoom + model rotation: worst case for anisotropic footprints */
        float ey = slab_wz ? (float)slab_src.ny / (float)slab_src.nx : 1.0f;
        float ez = slab_wz ? (float)slab_wz / (float)slab_src.nx : 1.0f;
        if (vslab_mode) { ey = (float)vsh / (float)vsw; ez = (float)(vsd + 2) / (float)vsw; }
        vol_rot[0] = 0.7f * sinf(ph * tau);
        vol_rot[1] = 0.45f * sinf(ph * tau * 0.7f);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
        r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 0.2f);
      } else if (umbilicus_path && strcmp(bench, "annstep") == 0) {
        int64_t bz = annotation_bench_z;
        if (ph >= 0.6f) bz += annotation_step;
        if (bz >= vsnz) bz = (int64_t)vsnz - 1;
        annotation_z = (int)bz;
      } else if (umbilicus_path && strcmp(bench, "annscroll") == 0) {
        /* Hold long enough to fill look-ahead, then make six consecutive
         * annotation-step moves 0.6 s apart in a 600-frame vsynced run. */
        int64_t n = ph < 0.6f ? 0 : 1 + (int64_t)((ph - 0.6f) / 0.06f);
        if (n > 6) n = 6;
        int64_t bz = (int64_t)annotation_bench_z + n * annotation_step;
        if (bz >= vsnz) bz = (int64_t)vsnz - 1;
        annotation_z = (int)bz;
      } else if (umbilicus_path && strcmp(bench, "annwheel") == 0) {
        /* Fine-scroll stress: after residency warmup, traverse 20 adjacent
         * slices at ten detents/second in a 600-frame vsynced run. */
        int64_t n = ph < 0.7f ? 0 : 1 + (int64_t)((ph - 0.7f) / 0.01f);
        if (n > 20) n = 20;
        int64_t bz = (int64_t)annotation_bench_z + n;
        if (bz >= vsnz) bz = (int64_t)vsnz - 1;
        annotation_z = (int)bz;
      } /* other bench names (zsweep) keep the default camera */
      if (vslab_mode && strcmp(bench, "zsweep") == 0) /* scroll z across the band */
        vs_z0 = 33792 + (int64_t)(ph * (1024.0f - (float)(vsd + 2)));
      if (bricks_path && brick_depth > 0 && strcmp(bench, "zsweep") == 0) {
        int bz = (int)(ph * (float)((int)brick_shape[2] - brick_depth));
        uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
        if (brick_shape[2] > md) md = brick_shape[2];
        float dz = (float)(bz - brick_z) / (float)md;
        brick_z = bz;
        cam.target.z += dz;
        cam.pos.z += dz;
      }
      if (vslab_mode && strcmp(bench, "clippan") == 0) { /* xy window churn */
        vs_follow = false;
        vs_fx = 12000.0 + (double)ph * 18000.0;
        vs_fy = 21504.0;
      }
    }
    r3d_v3 right, up, fwd;
    r3d_camera_basis(&cam, (float)w / (float)h, &right, &up, &fwd);
    r3d_m3 vm = m3_ypr(vol_rot[0], vol_rot[1], vol_rot[2]);

    if (umbilicus_path && vslab_mode && in.annotate_click) {
      int64_t origin[3] = {0, 0, vs_z0};
      r3d_vslab_get(renderer, origin, NULL);
      if (origin[0] < 0) origin[0] = 0;
      if (origin[1] < 0) origin[1] = 0;
      double ax = 0.0, ay = 0.0;
      if (annotation_pick(&cam, right, up, fwd, vm, vol_t, in.click_xy[0], in.click_xy[1],
                          w, h, (uint32_t)vsw, (uint32_t)vsh, (uint32_t)vsd, origin[0],
                          origin[1], vs_z0, annotation_z, &ax, &ay) &&
          r3d_umbilicus_set(&umbilicus, ax, ay, annotation_z) == 0) {
        save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        if (in.click_ctrl) {
          int64_t next = (int64_t)annotation_z + annotation_step;
          annotation_z = (int)(next < vsnz ? next : (int64_t)vsnz - 1);
          vs_z0 = annotation_z0(annotation_z, vsnz, (uint32_t)vsd);
        }
      } else {
        snprintf(annotation_status, sizeof annotation_status,
                 "click missed the displayed XY plane");
      }
    }

    /* adaptive resolution: drop to half res while interacting (4x fewer
     * rays), snap back to full once the camera settles */
    bool moving = in.dragging || in.captured || in.wheel != 0.0f || in.zdelta || in.zpage ||
                  z_navigated || auto_scroll || mv_exercise_moving ||
                  in.move[0] != 0.0f || in.move[1] != 0.0f || in.move[2] != 0.0f;
    settle = moving ? 15 : (settle > 0 ? settle - 1 : 0);
    bool half_res = adaptive_res && settle > 0 && !multiview_path;
    /* multiview: panes render one ray per 2x2 block while interacting and
     * snap back to full when the view settles (the pane cache keys on the
     * flag, so the settle frame redraws at full resolution) */
    /* Only worth it when the GPU is actually loaded: at light loads (1080p,
     * ~1 ms frames) the GPU is clock-gated and 4x fewer rays measured no
     * faster, while 4K interaction gained 30-70%. Gate on the smoothed GPU
     * frame time (R3D_MV_HALF_MS overrides the 5 ms budget). */
    static double mv_half_ms = -1.0;
    if (mv_half_ms < 0.0) {
      const char *hm = getenv("R3D_MV_HALF_MS");
      mv_half_ms = hm ? atof(hm) : 4.0;
    }
    static bool mv_half_latch = false; /* hysteresis: half frames measure cheaper */
    {
      double gms = (double)prof.gpu_ns * 1e-6;
      if (!mv_half_latch && gms > mv_half_ms) {
        mv_half_latch = true;
        if (getenv("R3D_MV_HALF_LOG")) fprintf(stderr, "mv: half-res on (gpu %.2f ms)\n", gms);
      } else if (mv_half_latch && gms < mv_half_ms / 3.0) { /* half frames cost ~1/2-1/3 */
        mv_half_latch = false;
        if (getenv("R3D_MV_HALF_LOG")) fprintf(stderr, "mv: half-res off (gpu %.2f ms)\n", gms);
      }
    }
    bool mv_half = adaptive_res && settle > 0 && multiview_path && !getenv("R3D_MV_NO_HALF") &&
                   mv_half_latch;
    if (multiview_path && getenv("R3D_MV_FORCE_HALF")) mv_half = true; /* A/B */
    if (getenv("R3D_FORCE_HALF")) half_res = true; /* testing/benching the path */
    else if (in.screenshot || (total_frames && shot_path && frame_index + 1 >= total_frames)) {
      half_res = false; /* captures always full res */
      mv_half = false;
    }
    uint32_t rvw = half_res ? (uint32_t)w / 2 : (uint32_t)w;
    uint32_t rvh = half_res ? (uint32_t)h / 2 : (uint32_t)h;

    mt_t[2] = r3d_now_ns();
    /* control panel: floating window normally; in multiview a docked left
     * side panel that collapses to a slim bar (views reflow to fill) */
    fps_smooth = fps_smooth * 0.95f + (dt > 0 ? 0.05f / dt : 0.0f);
    r3d_gui_begin(renderer);
    bool panel_content = true;
    if (multiview_path) {
      igSetNextWindowPos((ImVec2){0, 0}, ImGuiCond_Always, (ImVec2){0, 0});
      igSetNextWindowSize((ImVec2){(float)mv_panel_w, (float)h}, ImGuiCond_Always);
      igBegin("render3d", NULL,
              ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                  (mv_panel_open ? 0 : ImGuiWindowFlags_NoScrollbar));
      if (igButton(mv_panel_open ? "<<" : ">>", (ImVec2){0, 0}))
        mv_panel_open = !mv_panel_open;
      panel_content = mv_panel_open;
      if (panel_content) {
        igSameLine(0, 10);
        igText("%.0f fps  gpu %.1f ms", (double)fps_smooth, (double)last_gpu_ns / 1e6);
        igSeparator();
      }
    } else {
      igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
      igBegin("render3d", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    }
    if (panel_content) {
    if (!multiview_path)
      igText("%.0f fps   gpu %.2f ms", (double)fps_smooth, (double)last_gpu_ns / 1e6);
    if (igButton("data browser", (ImVec2){0, 0})) od_window = true;
    igSameLine(0, 8);
    if (igButton("open segment", (ImVec2){0, 0})) igOpenPopup_Str("r3d_open_seg", 0);
    igSameLine(0, 8);
    if (multiview_path && mv_seg.nvalid && igButton("close segment", (ImVec2){0, 0})) {
      /* swap in an empty segment: the flattened pane goes blank and no
       * active-segment overlays draw — a clean slate for the tracer */
      r3d_tifxyz es = {.w = 2, .h = 2, .sx = 0.05f, .sy = 0.05f};
      es.xyz = malloc(2 * 2 * 3 * sizeof *es.xyz);
      float *eco = NULL, *eno = NULL;
      r3d_segrows er = {0};
      if (es.xyz) {
        for (int k = 0; k < 12; k++) es.xyz[k] = -1.0f;
        memcpy(es.bbox[0], (float[3]){0, 0, 0}, sizeof es.bbox[0]);
        memcpy(es.bbox[1], (float[3]){0, 0, 0}, sizeof es.bbox[1]);
      }
      if (es.xyz && r3d_segrows_build(&es, &er) == 0 &&
          mv_build_grids(&es, &eco, &eno) == 0 &&
          r3d_surf_swap(renderer, es.w, es.h, eco, eno, es.sx, es.sy) == 0) {
        smask_drop(renderer, sgc_active); /* mask belongs to the old segment */
        r3d_tifxyz_free(&mv_seg);
        r3d_segrows_free(&mv_rows);
        free(mv_normals);
        mv_seg = es;
        mv_rows = er;
        mv_normals = eno;
        sgc_active[0] = 0;
        for (int oi2 = 0; oi2 < 4; oi2++) {
          mv_ol[oi2].n = 0;
          mv_ol_off[oi2].n = 0;
          mv_ol_slice[oi2] = 1e30;
        }
        mv_ol_zoff = 1e30;
        printf("segment closed (empty active surface)\n");
      } else {
        r3d_tifxyz_free(&es);
        r3d_segrows_free(&er);
        free(eno);
      }
      free(eco);
    }
    if (igButton("open volume", (ImVec2){0, 0})) igOpenPopup_Str("r3d_open_vol", 0);
    igSameLine(0, 8);
    if (bricks_path && igButton("close volume", (ImVec2){0, 0})) {
      od_next_bricks[0] = 0; /* teardown to the blank (no-dataset) state */
      od_next_seg[0] = 0;
      od_swap = true;
      running = false;
    }
    if (igBeginPopup("r3d_open_seg", 0)) {
      if (sgc.open) {
        for (uint32_t si = 0; si < sgc.st.n; si++) {
          char lbl[96];
          snprintf(lbl, sizeof lbl, "%.64s##os%u", sgc.st.segs[si].name, si);
          if (igSelectable_Bool(lbl, false, 0, (ImVec2){0, 0})) {
            pthread_mutex_lock(&sgc.mu);
            if (sgc.act_req == UINT32_MAX && !sgc.act_busy) {
              sgc.act_req = si;
              pthread_cond_signal(&sgc.cv);
            }
            pthread_mutex_unlock(&sgc.mu);
            igCloseCurrentPopup();
          }
        }
      } else {
        igTextDisabled("no segment store open (--segments <dir>)");
      }
      igEndPopup();
    }
    if (igBeginPopup("r3d_open_vol", 0)) {
      static char vol_list[24][640];
      static uint32_t vol_n = 0;
      if (igIsWindowAppearing()) { /* rescan local volume trees */
        vol_n = 0;
        static const char *roots[2] = {"cache", "cache/od"};
        for (int r0 = 0; r0 < 2 && vol_n < 24; r0++) {
          DIR *dp0 = opendir(roots[r0]);
          struct dirent *de0;
          while (dp0 && (de0 = readdir(dp0)) != NULL && vol_n < 24) {
            if (de0->d_name[0] == '.') continue;
            char mp0[720];
            snprintf(mp0, sizeof mp0, "%s/%s/manifest.json", roots[r0], de0->d_name);
            FILE *mf0 = fopen(mp0, "rb");
            if (!mf0 && r0 == 1) { /* od volumes nest one level deeper */
              char sub[700];
              snprintf(sub, sizeof sub, "%s/%s", roots[r0], de0->d_name);
              DIR *dp1 = opendir(sub);
              struct dirent *de1;
              while (dp1 && (de1 = readdir(dp1)) != NULL && vol_n < 24) {
                if (de1->d_name[0] == '.') continue;
                snprintf(mp0, sizeof mp0, "%s/%s/manifest.json", sub, de1->d_name);
                FILE *mf1 = fopen(mp0, "rb");
                if (mf1) {
                  fclose(mf1);
                  snprintf(vol_list[vol_n++], sizeof vol_list[0], "%s", mp0);
                }
              }
              if (dp1) closedir(dp1);
            } else if (mf0) {
              fclose(mf0);
              snprintf(vol_list[vol_n++], sizeof vol_list[0], "%s", mp0);
            }
          }
          if (dp0) closedir(dp0);
        }
      }
      for (uint32_t vi = 0; vi < vol_n; vi++) {
        char lbl[720];
        snprintf(lbl, sizeof lbl, "%.640s##ov%u", vol_list[vi], vi);
        if (igSelectable_Bool(lbl, false, 0, (ImVec2){0, 0})) {
          snprintf(od_next_bricks, sizeof od_next_bricks, "%s", vol_list[vi]);
          if (multiview_path)
            snprintf(od_next_seg, sizeof od_next_seg, "%s", multiview_path);
          od_swap = true;
          running = false;
          igCloseCurrentPopup();
        }
      }
      if (!vol_n) igTextDisabled("no local volume trees under cache/");
      igEndPopup();
    }
    if (igCollapsingHeader_TreeNodeFlags("rendering", 0)) {
      int m = (int)mode;
      if (igCombo_Str("mode", &m, "full\0mip\0depth\0heatmap\0raydir\0flat\0", 6))
        mode = (uint32_t)m;
      int prev_cm = cam_mode;
      igCombo_Str("camera", &cam_mode, "orbit\0fly\0", 2);
      if (cam_mode == CAM_ORBIT && prev_cm == CAM_FLY)
        r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
      int t = (int)tf_idx;
      if (igCombo_Str("transfer fn", &t, "gray\0scroll\0high-pass\0", 3)) {
        tf_idx = (uint32_t)t;
        r3d_tf tfp;
        uint8_t lut[256][4];
        r3d_tf_preset(tf_idx, &tfp);
        r3d_tf_build(&tfp, lut);
        r3d_set_transfer(renderer, lut);
        tf_min_v = tf_min_visible(lut);
      }
      igSliderFloat("step (voxels)", &step_voxels, 0.25f, 4.0f, "%.2f", 0);
      igSliderFloat("density", &density, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
      igSliderFloat("low cut", &low_cut, 0.0f, 255.0f, "%.0f", 0);
    }
    if (slab_wz) {
      igSliderInt("depth (voxels)", &slab_depth, 2, (int)slab_wz, "%d", 0);
      int z0i = (int)slab_z0;
      if (igSliderInt("z position", &z0i, 0, (int)slab_z0_max, "%d", 0)) {
        slab_z0 = (uint32_t)z0i;
        r3d_slab_window(renderer, &slab_src, slab_z0);
      }
      igCheckbox("auto-scroll", &auto_scroll);
      igSameLine(0, 10);
      igSliderFloat("slices/s", &auto_speed, -60.0f, 60.0f, "%.0f", 0);
    }
    if (clip_mode) {
      igSliderInt("depth (voxels)", &clip_depth, 2, (int)CLIP_DEPTH_MAX, "%d", 0);
      int z0i = (int)(clip_z0 - clip_z0_min);
      if (igSliderInt("z position", &z0i, 0, (int)(clip_z0_max - clip_z0_min), "%d", 0))
        clip_z0 = clip_z0_min + (uint64_t)z0i;
      igText("levels ready:%s%s%s%s%s%s", (clip_valid_disp & 1) ? " L0" : "",
             (clip_valid_disp & 2) ? " L1" : "", (clip_valid_disp & 4) ? " L2" : "",
             (clip_valid_disp & 8) ? " L3" : "", (clip_valid_disp & 16) ? " L4" : "",
             (clip_valid_disp & 32) ? " L5" : "");
    }
    if (vslab_mode) {
      if (umbilicus_path) {
        igSeparator();
        igText("umbilicus annotation   %zu point%s", umbilicus.count,
               umbilicus.count == 1 ? "" : "s");
        int zi = annotation_z;
        if (igSliderInt("slice z", &zi, 0, (int)vsnz - 1, "%d", 0)) annotation_z = zi;
        if (igInputInt("slice step", &annotation_step, 1, 100, 0)) {
          if (annotation_step < 1) annotation_step = 1;
          if (annotation_step > (int)vsnz - 1) annotation_step = (int)vsnz - 1;
        }
        if (igButton("< previous", (ImVec2){0, 0})) {
          annotation_z -= annotation_step;
          if (annotation_z < 0) annotation_z = 0;
        }
        igSameLine(0, 8);
        if (igButton("next >", (ImVec2){0, 0})) {
          annotation_z += annotation_step;
          if (annotation_z >= (int)vsnz) annotation_z = (int)vsnz - 1;
        }
        if (igButton("< annotated", (ImVec2){0, 0})) {
          for (size_t i = umbilicus.count; i > 0; i--)
            if (umbilicus.points[i - 1].z < annotation_z) {
              annotation_z = (int)llround(umbilicus.points[i - 1].z);
              break;
            }
        }
        igSameLine(0, 8);
        if (igButton("annotated >", (ImVec2){0, 0})) {
          for (size_t i = 0; i < umbilicus.count; i++)
            if (umbilicus.points[i].z > annotation_z) {
              annotation_z = (int)llround(umbilicus.points[i].z);
              break;
            }
        }
        const r3d_umbilicus_point *here = r3d_umbilicus_find(&umbilicus, annotation_z);
        if (here) {
          igText("current point: x %.1f   y %.1f", here->x, here->y);
          if (igButton("delete current", (ImVec2){0, 0}) &&
              r3d_umbilicus_remove(&umbilicus, annotation_z))
            save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
          igSameLine(0, 8);
        } else {
          igTextDisabled("current slice is not annotated");
        }
        if (igButton("save now", (ImVec2){0, 0}))
          save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        const char *annotation_name = strrchr(umbilicus_path, '/');
        igTextDisabled("%s | %s", annotation_name ? annotation_name + 1 : umbilicus_path,
                       annotation_status);
        r3d_vslab_prefetch_stats pcs;
        r3d_vslab_prefetch_get(renderer, &pcs);
        igText("decoded cache: %u/%u ready%s  hits %llu  last %.2f s", pcs.ready,
               pcs.capacity,
               pcs.filling ? " + filling" : "", (unsigned long long)pcs.hits,
               pcs.last_decode_ms / 1000.0);
        igTextDisabled("GPU fine-scroll neighborhood: +/- %d slices", annotation_z_prefetch);
      } else {
        int z0i = (int)vs_z0;
        int zmax = (int)vsnz - vsd;
        if (igSliderInt("z position (world)", &z0i, 0, zmax > 0 ? zmax : 0, "%d", 0))
          vs_z0 = z0i;
        igCheckbox("follow camera (x/y)", &vs_follow);
      }
      int64_t vo_[3];
      uint32_t pend;
      r3d_vslab_get(renderer, vo_, &pend);
      uint32_t rpend = r3d_vslab_resident_pending(renderer);
      igText("window @ (%lld, %lld, %lld)%s", (long long)vo_[0], (long long)vo_[1],
             (long long)vo_[2], pend ? "  streaming..." : (rpend ? "  prefetching..." : ""));
    }
    if (bricks_path) {
      r3d_bricks_stats bst;
      r3d_bricks_get_stats(renderer, &bst);
      if (brick_is_lod && brick_depth > 0) {
        int bz = brick_z;
        int zmax = (int)brick_shape[2] - brick_depth;
        if (igSliderInt("slice z", &bz, 0, zmax > 0 ? zmax : 0, "%d", 0) && bz != brick_z) {
          uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
          if (brick_shape[2] > md) md = brick_shape[2];
          float dz = (float)(bz - brick_z) / (float)md;
          brick_z = bz;
          if (cam_mode == CAM_ORBIT) {
            cam.target.z += dz;
            cam.pos.z += dz;
          }
        }
        int bd = brick_depth;
        int dmax = brick_shape[2] < 256u ? (int)brick_shape[2] : 256;
        if (igSliderInt("slice depth", &bd, 1, dmax, "%d", 0)) {
          brick_depth = bd;
          if (brick_z > (int)brick_shape[2] - brick_depth)
            brick_z = (int)brick_shape[2] - brick_depth;
        }
      }
      if (multiview_path) {
        igSeparator();
    if (igCollapsingHeader_TreeNodeFlags("views", ImGuiTreeNodeFlags_DefaultOpen)) {
          static const char *mv_name[4] = {"segment", "XY", "XZ", "YZ"};
          for (int i = 0; i < 4; i++) {
            bool vis = (mv_visible >> i) & 1u;
            if (i) igSameLine(0, 8);
            if (igCheckbox(mv_name[i], &vis))
              mv_visible = vis ? mv_visible | (1u << i) : mv_visible & ~(1u << i);
          }
          if (mv_solo >= 0) {
            igSameLine(0, 12);
            igTextDisabled("solo: %s (Space restores)", mv_name[mv_solo]);
          }
          igText("multiview focus  x %.0f  y %.0f  z %.0f", mv_focus[0], mv_focus[1],
                 mv_focus[2]);
          if (mv_aligned) /* side panes scrub offsets from the focus */
            igText("XY z %.0f | segA %+.0f | segB %+.0f", mv[R3D_MV_XY].slice,
                   mv[R3D_MV_XZ].slice, mv[R3D_MV_YZ].slice);
          else
            igText("XY z %.0f | XZ y %.0f | YZ x %.0f", mv[R3D_MV_XY].slice,
                   mv[R3D_MV_XZ].slice, mv[R3D_MV_YZ].slice);
          int th = mv_thick;
          if (igSliderInt("slice thickness", &th, 1, 128, "%d", 0)) mv_thick = th;
          igSliderFloat("scrub speed", &mv_scrub, 0.25f, 200.0f, "%.2f vox/notch",
                        ImGuiSliderFlags_Logarithmic);
          if (mv_pane_kind[0] == 2 || mv_pane_kind[1] == 2) {
            static const char *cl[3] = {"crop x (vox)", "crop y (vox)", "crop z (vox)"};
            for (int a = 0; a < 3; a++) {
              float cv2 = mv_crop_d[a];
              igSetNextItemWidth(180);
              if (igSliderFloat(cl[a], &cv2, 32.0f,
                                brick_shape[a] ? (float)brick_shape[a] : 65536.0f, "%.0f",
                                ImGuiSliderFlags_Logarithmic) &&
                  cv2 >= 32.0f)
                mv_crop_d[a] = cv2;
            }
          }
          static const char *pane_lbl[2] = {"bottom-left##pk", "bottom-right##pk"};
          for (int pk = 0; pk < 2; pk++) {
            int kv = mv_pane_kind[pk];
            igSetNextItemWidth(140);
            if (igCombo_Str(pane_lbl[pk], &kv, "XZ\0YZ\0" "3D volume\0\0", 3) &&
                kv != mv_pane_kind[pk]) {
              if (mv_pane_kind[1 - pk] == kv) /* keep the three kinds on two
                                               * panes distinct: swap */
                mv_pane_kind[1 - pk] = mv_pane_kind[pk];
              mv_pane_kind[pk] = kv;
              for (int pp = 0; pp < 2; pp++) { /* rebuild plane frames */
                int vi = 2 + pp;
                if (mv_pane_kind[pp] == 2) continue;
                r3d_mv_axis_basis(2 + mv_pane_kind[pp], mv_pb[vi]);
                for (int a = 0; a < 3; a++) mv_po[vi][a] = 0.0;
                double fu, fv, fs;
                r3d_mv_w2b(mv_pb[vi], mv_po[vi], mv_focus, &fu, &fv, &fs);
                mv[vi].cu = fu;
                mv[vi].cv = fv;
                mv[vi].slice = fs;
              }
              if (mv_aligned) mv_aligned = false; /* aligned pair owns 2+3 */
              mv_basis_gen++;
            }
          }
          bool alg = mv_aligned;
          if (igCheckbox("segment-aligned planes", &alg)) {
            mv_aligned = alg;
            if (mv_aligned) {
              uint32_t ij[2];
              if (mv_nearest_surface(&mv_seg, mv_focus, ij)) {
                mv_align_ij[0] = ij[0];
                mv_align_ij[1] = ij[1];
              }
              if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                                mv_po))
                mv_aligned = false; /* no usable surface normal near the focus */
            }
            if (!mv_aligned) mv_axis_reset(mv_pb, mv_po);
            for (int i = R3D_MV_XZ; i <= R3D_MV_YZ; i++) {
              double fu, fv, fs;
              r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
              mv[i].cu = fu;
              mv[i].cv = fv;
              mv[i].slice = fs;
            }
            mv_basis_gen++;
          }
          if (mv_aligned &&
              igSliderFloat("plane rotation", &mv_theta, 0.0f, 360.0f, "%.0f deg", 0) &&
              mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                           mv_po)) {
            for (int i = R3D_MV_XZ; i <= R3D_MV_YZ; i++) {
              mv[i].cu = mv[i].cv = 0.0; /* frames rotate about the focus */
              mv[i].slice = 0.0;
            }
            mv_basis_gen++;
          }
          float zo = (float)mv[R3D_MV_SEG].slice;
          if (igSliderFloat("segment offset", &zo, -64.0f, 64.0f, "%.0f vox", 0))
            mv[R3D_MV_SEG].slice = (double)zo;
          igCheckbox("stretch heatmap", &mv_stretch);
    }
        if (umbilicus_path &&
            igCollapsingHeader_TreeNodeFlags("umbilicus", ImGuiTreeNodeFlags_DefaultOpen)) {
          igText("%zu point%s", umbilicus.count, umbilicus.count == 1 ? "" : "s");
          igCheckbox("edit (U places; Ctrl+Z / Ctrl+Shift+Z)", &mv_umb_edit);
          igText("recenter on place:");
          static const char *umb_rf_name[4] = {"seg##urf", "XY##urf", "XZ##urf", "YZ##urf"};
          for (int rf = 0; rf < 4; rf++) {
            igSameLine(0, 8);
            bool on = (mv_umb_refoc >> rf) & 1u;
            if (igCheckbox(umb_rf_name[rf], &on))
              mv_umb_refoc = on ? mv_umb_refoc | (1u << rf) : mv_umb_refoc & ~(1u << rf);
          }
          igCheckbox("advance after placing (by scrub speed)", &mv_umb_adv);
          igSameLine(0, 10);
          igCheckbox("backwards", &mv_umb_back);
          double curz = mv[R3D_MV_XY].slice;
          if (igButton("< annotated", (ImVec2){0, 0})) {
            for (size_t k = umbilicus.count; k > 0; k--)
              if (umbilicus.points[k - 1].z < curz) {
                mv[R3D_MV_XY].slice = umbilicus.points[k - 1].z;
                mv[R3D_MV_XY].cu = umbilicus.points[k - 1].x;
                mv[R3D_MV_XY].cv = umbilicus.points[k - 1].y;
                break;
              }
          }
          igSameLine(0, 8);
          if (igButton("annotated >", (ImVec2){0, 0})) {
            for (size_t k = 0; k < umbilicus.count; k++)
              if (umbilicus.points[k].z > curz) {
                mv[R3D_MV_XY].slice = umbilicus.points[k].z;
                mv[R3D_MV_XY].cu = umbilicus.points[k].x;
                mv[R3D_MV_XY].cv = umbilicus.points[k].y;
                break;
              }
          }
          igSameLine(0, 8);
          const r3d_umbilicus_point *here =
              r3d_umbilicus_find(&umbilicus, (double)llround(curz));
          if (here && igButton("delete here", (ImVec2){0, 0})) {
            umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
            umb_snap_clear(mv_umb_redo, &mv_umb_redo_n);
            if (r3d_umbilicus_remove(&umbilicus, (double)llround(curz)))
              save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
          }
          if (igButton("start fresh (move to .bak)", (ImVec2){0, 0})) {
            char bak[1360];
            snprintf(bak, sizeof bak, "%s.bak", umbilicus_path);
            umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
            umb_snap_clear(mv_umb_redo, &mv_umb_redo_n);
            if (rename(umbilicus_path, bak) == 0 || errno == ENOENT) {
              r3d_umbilicus_free(&umbilicus);
              r3d_umbilicus_init(&umbilicus);
              snprintf(annotation_status, sizeof annotation_status,
                       "cleared; previous saved as %s", bak);
            } else {
              snprintf(annotation_status, sizeof annotation_status, "backup failed: %s",
                       strerror(errno));
            }
          }
          if (annotation_status[0]) igTextDisabled("%s", annotation_status);
        }
        igTextDisabled("segment %ux%u  %llu valid points", mv_seg.w, mv_seg.h,
                       (unsigned long long)mv_seg.nvalid);
    if (multiview_path && igCollapsingHeader_TreeNodeFlags("manual surface", 0)) {
      igTextDisabled("P over a plane view: place points on one sheet, in\n"
                     "any order, any pane - rough outline first, densify\n"
                     "inside later. Create fits a surface through them all;\n"
                     "it joins the store, gets its ink map, and doubles as\n"
                     "training data for surface prediction.");
      igText("%u point%s", msurf_n, msurf_n == 1 ? "" : "s");
      if (msurf_n) {
        igSameLine(0, 8);
        if (igSmallButton("undo##mp")) {
          msurf_n--;
          msurf_dirty = true;
        }
        igSameLine(0, 6);
        if (igSmallButton("clear##mp")) {
          msurf_n = 0;
          msurf_first = true;
        }
        igSameLine(0, 8);
        if (msurf_n >= 4 && igButton("create surface", (ImVec2){0, 0}))
          msurf_create = true;
      }
    }
    if (multiview_path && bricks_path &&
        igCollapsingHeader_TreeNodeFlags("boundary surface", 0)) {
      igTextDisabled("Set the render low cut so papyrus stands hard against\n"
                     "air, then B over a plane view on/near that edge. Grow\n"
                     "follows the papyrus/void interface outward from the\n"
                     "seed: each node is the papyrus voxel touching the void\n"
                     "along the local normal, never farther than the snap\n"
                     "distance; the fringe stops where the face ends.");
      igText("low cut %.0f (render panel)", (double)low_cut);
      if (bsurf_have_seed)
        igText("seed (%.0f, %.0f, %.0f)", bsurf_seed[0], bsurf_seed[1], bsurf_seed[2]);
      else
        igTextDisabled("no seed - press B over a plane view");
      igSliderInt("generations##bs", &bsurf_gens, 1, 600, "%d", 0);
      igSliderFloat("snap dist (vox)##bs", &bsurf_snap, 1.0f, 24.0f, "%.0f", 0);
      igSliderFloat("grid step (vox)##bs", &bsurf_step, 2.0f, 40.0f, "%.0f", 0);
      if (bsurf_active) {
        bool bdone = false;
        uint32_t bring = 0, bnset = 0;
        r3d_bsurf_snapshot(&bsurf, NULL, &bring, &bnset, &bdone);
        igText("%s", bsurf.status);
        if (!bdone) {
          if (igButton("stop##bs", (ImVec2){0, 0})) r3d_bsurf_stop(&bsurf);
        } else {
          if (!bsurf.failed && bnset > 64 && igButton("save surface##bs", (ImVec2){0, 0}))
            bsurf_save = true;
          if (!bsurf.failed && bnset > 64) igSameLine(0, 8);
          if (igButton("discard##bs", (ImVec2){0, 0})) bsurf_discard = true;
          igSameLine(0, 8);
          if (bsurf_have_seed && igButton("regrow##bs", (ImVec2){0, 0})) {
            bsurf_discard = true;
            bsurf_go = true;
          }
        }
      } else if (bsurf_have_seed && igButton("grow##bs", (ImVec2){0, 0})) {
        bsurf_go = true;
      }
    }
    if (multiview_path && n_overlays &&
        igCollapsingHeader_TreeNodeFlags("tracer", 0)) {
      const char *pr = overlay_paths[overlay_sel];
      const char *prb = strrchr(pr, '/');
      igTextDisabled("predictions: %s", prb ? prb + 1 : pr);
      igSliderFloat("grid step (vox)", &mv_tr_step, 1.0f, 40.0f, "%.0f", 0);
      igSliderFloat("confidence cutoff", &mv_tr_thresh, 0.05f, 0.9f, "%.2f", 0);
      igTextDisabled("growth never rejects (vc3d); the cutoff masks display + save");
      igSliderInt("generations", &mv_tr_rings, 8, 200, "%d", 0);
      igCheckbox("live in segment view", &mv_tr_live);
      igSameLine(0, 10);
      igCheckbox("fill holes", &mv_tr_fill);
      igTextDisabled("X over a plane view: anchor the sheet through the cursor");
      if (mv_anchor_n) {
        igText("%u anchor%s", mv_anchor_n, mv_anchor_n == 1 ? "" : "s");
        igSameLine(0, 10);
        if (igSmallButton("undo##anc")) {
          mv_anchor_n--;
          if (GT->active) r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
        }
        igSameLine(0, 6);
        if (igSmallButton("clear##anc")) {
          mv_anchor_n = 0;
          if (GT->active) r3d_tracer_set_anchors(&GT->tr, mv_anchor, 0);
        }
      }
      if (umbilicus.count >= 2) {
        igCheckbox("spiral prior", &mv_tr_spiral);
        if (GT->active && GT->tr.sp_valid) {
          igSameLine(0, 10);
          igTextDisabled("omega %.1f rms %.1f", GT->tr.sp_omega, GT->tr.sp_rms);
        }
      } else {
        igTextDisabled("spiral prior needs an umbilicus (U to annotate)");
      }
      /* concurrent traces: the seeds queue ('G' over a plane view) grows
       * one tracer per seed, all at once; the list below switches which
       * one the flattened pane displays */
      {
        uint32_t nact = 0;
        for (int ti = 0; ti < GT_MAX; ti++)
          if (gts[ti].active) nact++;
        igTextDisabled("G over a plane view: queue a trace seed");
        if (mv_seeds_n) {
          igText("%u seed%s queued%s", mv_seeds_n, mv_seeds_n == 1 ? "" : "s",
                 mv_seeds_go ? " (draining as slots free)" : "");
          igSameLine(0, 8);
          if (igSmallButton("clear##seeds")) {
            mv_seeds_n = 0;
            mv_seeds_go = false;
          }
          igSameLine(0, 8);
          if (igButton("trace all seeds", (ImVec2){0, 0})) mv_seeds_go = true;
          (void)nact;
        }
        for (int ti = 0; ti < GT_MAX; ti++) { /* the live trace list */
          struct gtrace *g = &gts[ti];
          if (!g->active) continue;
          igPushID_Int(ti);
          igText("%s trace %d: %u/%u, %u pts%s", ti == gt_sel ? ">" : " ", ti,
                 g->ring, g->tr.cfg.max_ring, g->nset, g->done ? " (done)" : "");
          if (ti != gt_sel) {
            igSameLine(0, 8);
            if (igSmallButton("show")) {
              gt_sel = ti;
              GT->gen = 0; /* force a fresh snapshot + live swap */
              GT->live_first = true;
              mv_tr_view = true;
            }
          }
          igSameLine(0, 6);
          if (igSmallButton(g->done ? "x" : "stop")) {
            r3d_tracer_stop(&g->tr);
            if (!g->done) {
              g->done = true; /* keep on screen; x again to discard */
            } else {
              r3d_tracer_free(&g->tr);
              free(g->pos);
              free(g->st);
              free(g->cf);
              g->pos = NULL;
              g->st = NULL;
              g->cf = NULL;
              g->active = false;
            }
          }
          igPopID();
        }
      }
      if (!GT->active) {
        bool have_free = false;
        for (int s3 = 0; s3 < GT_MAX && !have_free; s3++)
          if (!gts[s3].active) have_free = true;
        if (have_free && igButton("seed at focus", (ImVec2){0, 0})) {
          for (int s3 = 0; s3 < GT_MAX; s3++)
            if (!gts[s3].active) {
              gt_sel = s3;
              break;
            }
          r3d_tracer_cfg tc = {.seed = {mv_focus[0], mv_focus[1], mv_focus[2]},
                               .step = (double)mv_tr_step,
                               .thresh = mv_tr_thresh,
                               .max_ring = (uint32_t)mv_tr_rings,
                               .level = 1,
                               .wind_weight =
                                   mv_tr_spiral && umbilicus.count >= 2 ? 0.5 : 0.0};
          if (r3d_tracer_start(&GT->tr, pr, &tc, &umbilicus) == 0) {
            GT->harvest = false; /* interactive: stays resident when done */
            mv_tr_view = true;
            if (mv_anchor_n) r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
            free(GT->pos);
            free(GT->st);
            free(GT->cf);
            GT->pos = malloc((size_t)GT->tr.W * GT->tr.H * 3 * sizeof *GT->pos);
            GT->st = calloc((size_t)GT->tr.W * GT->tr.H, 1);
            GT->cf = calloc((size_t)GT->tr.W * GT->tr.H, sizeof *GT->cf);
            GT->gen = 0;
            GT->live_first = true;
            GT->active = GT->pos && GT->st && GT->cf;
          }
        }
        igSameLine(0, 8);
        igTextDisabled("(uses the SELECTED overlay as predictions)");
      } else {
        igText("ring %u/%u  %u point%s%s", GT->ring, GT->tr.cfg.max_ring, GT->nset,
               GT->nset == 1 ? "" : "s",
               GT->done ? "  done"
                          : (GT->ring >= GT->tr.cfg.max_ring ? "  optimizing..."
                                                              : "  growing..."));
        if (GT->tr.qc_folds || GT->tr.qc_kinks)
          igTextColored((ImVec4){1.0f, 0.55f, 0.3f, 1.0f},
                        "QC: %u fold%s, %u kink%s, twist %.2f vox", GT->tr.qc_folds,
                        GT->tr.qc_folds == 1 ? "" : "s", GT->tr.qc_kinks,
                        GT->tr.qc_kinks == 1 ? "" : "s", (double)GT->tr.qc_twist);
        else if (GT->nset > 8)
          igTextDisabled("QC clean: no folds/kinks, twist %.2f vox",
                         (double)GT->tr.qc_twist);
        if (igButton(GT->done ? "discard" : "stop", (ImVec2){0, 0})) {
          r3d_tracer_stop(&GT->tr);
          if (!GT->done) { /* stopped mid-grow: keep result on screen */
            GT->done = true;
          } else {
            r3d_tracer_free(&GT->tr);
            free(GT->pos);
            free(GT->st);
            free(GT->cf);
            GT->pos = NULL;
            GT->st = NULL;
            GT->cf = NULL;
            GT->active = false;
          }
        }
        if (GT->done && GT->nset > 0 && GT->tr.gen_of && GT->tr.gens_done > 4) {
          static int mv_tr_rewind_to = 0;
          if (mv_tr_rewind_to <= 0 || mv_tr_rewind_to >= (int)GT->tr.gens_done)
            mv_tr_rewind_to = (int)GT->tr.gens_done / 2;
          igSetNextItemWidth(140);
          igSliderInt("##rewind", &mv_tr_rewind_to, 1, (int)GT->tr.gens_done - 1,
                      "rewind to gen %d", 0);
          igSameLine(0, 6);
          if (igButton("rewind##tr", (ImVec2){0, 0})) {
            /* drop everything placed after the chosen generation; grow
             * then regrows it (drop an anchor first to steer) */
            r3d_tracer_stop(&GT->tr);
            r3d_tracer_rewind(&GT->tr, (uint32_t)mv_tr_rewind_to);
            GT->gen = 0; /* refresh the live view */
          }
        }
        if (GT->done && GT->nset > 0 && atomic_load(&GT->tr.sp_valid)) {
          igSameLine(0, 8);
          if (igButton("spiral fill", (ImVec2){0, 0})) {
            /* fit_spiral analogue: populate every empty grid cell from the
             * global spiral fit, then polish onto evidence (no growth) */
            r3d_tracer_stop(&GT->tr);
            if (r3d_tracer_spiral_fill(&GT->tr) == 0) {
              GT->done = false;
              GT->gen = 0; /* force a fresh snapshot/live swap */
            }
          }
          if (igIsItemHovered(0))
            igSetTooltip("fill the whole grid from the fitted spiral\n"
                         "(omega %.1f, rms %.2f) and refine onto evidence;\n"
                         "follow with 'snap to CT edge' for the final fit",
                         atomic_load(&GT->tr.sp_omega), atomic_load(&GT->tr.sp_rms));
        }
        if (GT->done && GT->nset > 0 && mv_anchor_n) {
          igSameLine(0, 8);
          if (igButton("re-solve", (ImVec2){0, 0})) {
            /* fix the segment in place: no growth, anneal the grid through
             * the placed anchors (grid dims unchanged, buffers stay) */
            r3d_tracer_stop(&GT->tr);
            r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
            if (r3d_tracer_refine(&GT->tr) == 0) {
              GT->done = false;
              GT->gen = 0; /* force a fresh snapshot/live swap */
            }
          }
          if (igIsItemHovered(0))
            igSetTooltip("re-run the solve through the %u anchor%s without growing",
                         mv_anchor_n, mv_anchor_n == 1 ? "" : "s");
        }
        if (GT->done && GT->nset > 0) {
          igSameLine(0, 8);
          if (igButton("grow", (ImVec2){0, 0})) { /* extend by half again;
             * failed cells retry (more predictions may be local now) */
            r3d_tracer_stop(&GT->tr);
            if (r3d_tracer_grow(&GT->tr, GT->tr.cfg.max_ring / 2 + 10) == 0) {
              free(GT->pos);
              free(GT->st);
              free(GT->cf);
              GT->pos = malloc((size_t)GT->tr.W * GT->tr.H * 3 * sizeof *GT->pos);
              GT->st = calloc((size_t)GT->tr.W * GT->tr.H, 1);
              GT->cf = calloc((size_t)GT->tr.W * GT->tr.H, sizeof *GT->cf);
              GT->gen = 0;
              GT->done = false;
              if (!GT->pos || !GT->st || !GT->cf) GT->active = false;
            }
          }
        }
        if (GT->done && GT->nset > 0 && bricks_path) {
          /* prediction-traced sheet -> actual papyrus: bsurf's edge rule as
           * a tracer refine (cells move along their normals only, capped) */
          static float mv_tr_snapd = 8.0f;
          igSetNextItemWidth(130);
          igSliderFloat("##ctsnapd", &mv_tr_snapd, 2.0f, 24.0f, "reach %.0f vox",
                        0);
          igSameLine(0, 6);
          if (igButton("snap to CT edge", (ImVec2){0, 0})) {
            char croot[1024];
            snprintf(croot, sizeof croot, "%s", bricks_path);
            char *ms3 = strrchr(croot, '/');
            if (ms3 && strcmp(ms3 + 1, "manifest.json") == 0) *ms3 = 0;
            r3d_tracer_stop(&GT->tr);
            if (r3d_tracer_ctsnap(&GT->tr, croot, (double)mv_tr_snapd,
                                  (double)low_cut) == 0) {
              GT->done = false;
              GT->gen = 0; /* force a fresh snapshot/live swap */
            }
          }
          if (igIsItemHovered(0))
            igSetTooltip("pull the sheet off the predictions onto the actual\n"
                         "papyrus/void edge (render low cut %.0f), each point\n"
                         "moving at most %.0f voxels along its own normal",
                         (double)low_cut, (double)mv_tr_snapd);
        }
        igSameLine(0, 8);
        if (GT->nset > 8 && igButton("save + activate", (ImVec2){0, 0})) {
          r3d_tracer_stop(&GT->tr);
          char td[256];
          trace_dir_now(td, sizeof td);
          errno = 0;
          if (mkdir_p(td) &&
              r3d_tracer_save(&GT->tr, td, mv_tr_thresh, mv_tr_fill) == 0) {
            ++mv_tr_nsaved; /* only a durable artifact counts as saved */
            printf("tracer: saved %s (%ux%u, %u pts)\n", td, GT->tr.W, GT->tr.H,
                   GT->nset);
            /* the STORE (and thus ink/display) gets the flattened segment;
             * the raw dir stays on disk for tracecli round trips */
            char fdtd[280];
            const char *sd = trace_flatten_save(td, fdtd, sizeof fdtd) == 0
                                 ? fdtd : td;
            if (sgc.open) { /* pack + reopen the corpus so it shows at once */
              const char *dirs1[1] = {sd};
              if (r3d_segstore_build(seg_store_path, dirs1, 1, 2, false) > 0) {
                for (int oi2 = 1; oi2 < 4; oi2++) {
                  if (!sgc_ln[oi2]) continue;
                  for (uint32_t si = 0; si < sgc.st.n; si++) {
                    free(sgc_ln[oi2][si].l.w);
                    free(sgc_ln[oi2][si].l.g);
                  }
                  free(sgc_ln[oi2]);
                  sgc_ln[oi2] = NULL;
                }
                sgc_close(&sgc);
                if (sgc_open(&sgc, seg_store_path, (size_t)1536 << 20) == 0) {
                  for (int oi2 = 1; oi2 < 4; oi2++)
                    sgc_ln[oi2] = calloc(sgc.st.n ? sgc.st.n : 1, sizeof *sgc_ln[oi2]);
                  for (int oi2 = 0; oi2 < 4; oi2++) {
                    sgc_key_slice[oi2] = 1e30;
                    sgc_key_gen[oi2] = UINT32_MAX;
                  }
                  memcpy(sgc_near_focus,
                         (double[3]){1e30, 1e30, 1e30}, sizeof sgc_near_focus);
                  printf("tracer: %s added to the store (%u surfaces)\n", sd, sgc.st.n);
                  const char *tb = strrchr(sd, '/');
                  tb = tb ? tb + 1 : sd;
                  pthread_mutex_lock(&sgc.mu); /* make it the ACTIVE surface:
                     * the grown segment replaces the flattened view */
                  for (uint32_t si = 0; si < sgc.st.n; si++)
                    if (strcmp(sgc.st.segs[si].name, tb) == 0) {
                      sgc.act_req = si;
                      pthread_cond_signal(&sgc.cv);
                      break;
                    }
                  pthread_mutex_unlock(&sgc.mu);
                  /* the orange preview served its purpose */
                  r3d_tracer_free(&GT->tr);
                  free(GT->pos);
                  free(GT->st);
                  free(GT->cf);
                  GT->pos = NULL;
                  GT->st = NULL;
                  GT->cf = NULL;
                  GT->active = false;
                }
              }
            }
          } else {
            /* nothing is freed here: the trace stays resident so the button
             * can be pressed again once the cause is cleared */
            fprintf(stderr, "tracer: SAVE FAILED for %s (%s) - trace kept in memory\n",
                    td, errno ? strerror(errno) : "write error");
          }
        }
      }
    }
    if (sgc.open && igCollapsingHeader_TreeNodeFlags("surfaces", 0)) {
      { /* SLIM re-flattening of the active segment */
        int fst = atomic_load(&g_flat_state);
        igBeginDisabled(fst == 1 || mv_seg.w <= 2 || sgc_active[0] == 0);
        if (igButton("flatten (SLIM)##flat", (ImVec2){0, 0}) && fst != 1 &&
            mv_seg.w > 2 && sgc_active[0]) {
          size_t nn = (size_t)mv_seg.w * mv_seg.h;
          g_flat_in = malloc(nn * 3 * sizeof *g_flat_in);
          if (g_flat_in) {
            memcpy(g_flat_in, mv_seg.xyz, nn * 3 * sizeof *g_flat_in);
            g_flat_in_w = mv_seg.w;
            g_flat_in_h = mv_seg.h;
            g_flat_step = 1.0 / (double)mv_seg.sx;
            snprintf(g_flat_name, sizeof g_flat_name, "%s-flat", sgc_active);
            if (g_flat_th_up) {
              pthread_join(g_flat_th, NULL);
              g_flat_th_up = false;
            }
            atomic_store(&g_flat_state, 1);
            if (pthread_create(&g_flat_th, NULL, flat_worker, NULL) == 0)
              g_flat_th_up = true;
            else {
              atomic_store(&g_flat_state, 0);
              free(g_flat_in);
              g_flat_in = NULL;
            }
          }
        }
        igEndDisabled();
        if (igIsItemHovered(0))
          igSetTooltip("recompute a near-isometric parameterization (SLIM,\n"
                       "symmetric Dirichlet) and save it as %s-flat -\n"
                       "removes the 2D stretch the distortion heatmap shows",
                       sgc_active[0] ? sgc_active : "<segment>");
        if (fst == 1) {
          igSameLine(0, 10);
          igText("flattening %ux%u...", g_flat_in_w, g_flat_in_h);
        } else if (g_flat_stats.nvert) {
          igSameLine(0, 10);
          igTextDisabled("last: stretch %.3f -> %.4f (%u iters)",
                         g_flat_stats.stretch0, g_flat_stats.stretch1,
                         g_flat_stats.iters);
        }
      }
          igSetNextItemWidth(160);
          igSliderFloat("corpus lines", &mv_corpus_vis, 0.0f, 1.0f, "%.2f", 0);
          {
            pthread_mutex_lock(&sgc.mu);
            uint32_t ready = 0;
            for (uint32_t si = 0; si < sgc.st.n; si++)
              if (sgc.ent[si].state == SGC_READY) ready++;
            igText("segment store: %u surfaces  %u cached (%zu MB)", sgc.st.n, ready,
                   sgc.bytes >> 20);
            igText("plane hits  XY %u  XZ %u  YZ %u", sgc_nhits[R3D_MV_XY],
                   sgc_nhits[R3D_MV_XZ], sgc_nhits[R3D_MV_YZ]);
            if (sgc_near_focus[0] != mv_focus[0] || sgc_near_focus[1] != mv_focus[1] ||
                sgc_near_focus[2] != mv_focus[2]) {
              memcpy(sgc_near_focus, mv_focus, sizeof sgc_near_focus);
              sgc_nnear = r3d_segstore_near_query(&sgc.st, mv_focus, 300.0, sgc_near, 6);
              if (sgc_nnear > 6) sgc_nnear = 6;
            }
            bool act_pending = sgc.act_req != UINT32_MAX || sgc.act_busy;
            if (act_pending) igTextDisabled("activating...");
            else igTextDisabled("near focus (click to activate):");
            for (uint32_t k = 0; k < sgc_nnear; k++) {
              uint32_t si = sgc_near[k];
              bool cur = strcmp(sgc.st.segs[si].name, sgc_active) == 0;
              char lbl[112];
              float ovf = sgc.ov_active != UINT32_MAX ? sgc.ov[si] : 0.0f;
              if (cur)
                snprintf(lbl, sizeof lbl, "  %.64s (active)", sgc.st.segs[si].name);
              else if (ovf > 0.02f)
                snprintf(lbl, sizeof lbl, "  %.64s  ov %.0f%%", sgc.st.segs[si].name,
                         (double)ovf * 100.0);
              else
                snprintf(lbl, sizeof lbl, "  %.64s", sgc.st.segs[si].name);
              if (igSelectable_Bool(lbl, cur, 0, (ImVec2){0, 0}) && !cur && !act_pending &&
                  sgc.act_ready == UINT32_MAX) {
                sgc.act_req = si;
                pthread_cond_signal(&sgc.cv);
              }
            }
            if (igCollapsingHeader_TreeNodeFlags("all surfaces", 0)) {
              /* alphabetical view (timestamp names sort chronologically);
               * the order array rebuilds when the store size changes */
              static uint32_t *surf_ord = NULL;
              static uint32_t surf_ord_n = 0;
              if (surf_ord_n != sgc.st.n) {
                free(surf_ord);
                surf_ord = malloc(sgc.st.n * sizeof *surf_ord);
                surf_ord_n = surf_ord ? sgc.st.n : 0;
                if (surf_ord) {
                  for (uint32_t si = 0; si < sgc.st.n; si++) surf_ord[si] = si;
                  for (uint32_t a = 1; a < sgc.st.n; a++) { /* insertion sort */
                    uint32_t v = surf_ord[a], b = a;
                    while (b > 0 && strcmp(sgc.st.segs[surf_ord[b - 1]].name,
                                           sgc.st.segs[v].name) > 0) {
                      surf_ord[b] = surf_ord[b - 1];
                      b--;
                    }
                    surf_ord[b] = v;
                  }
                }
              }
              for (uint32_t oi3 = 0; oi3 < sgc.st.n; oi3++) {
                uint32_t si = surf_ord ? surf_ord[oi3] : oi3;
                bool cur = strcmp(sgc.st.segs[si].name, sgc_active) == 0;
                char lbl[96];
                snprintf(lbl, sizeof lbl, "%.64s##s%u", sgc.st.segs[si].name, si);
                if (igSelectable_Bool(lbl, cur, 0, (ImVec2){0, 0}) && !cur && !act_pending &&
                    sgc.act_ready == UINT32_MAX) {
                  sgc.act_req = si;
                  pthread_cond_signal(&sgc.cv);
                }
              }
            }
            pthread_mutex_unlock(&sgc.mu);
          }
    }
      }
    if (inklive_up && igCollapsingHeader_TreeNodeFlags("live ink", 0)) {
      igCheckbox("show ink on flattened view", &inklive_show);
      igBeginDisabled(inkmap_job);
      if (igCheckbox("verso (reverse side)##inkverso", &ink_verso)) {
        /* side switch: drop the displayed map and pick up the other
         * side's cache if it exists; otherwise 'compute' builds it */
        inklive_have = false;
        r3d_surfvol_inkpred_clear(renderer);
        free(inkmap);
        free(inkmap_acc);
        inkmap_acc = NULL;
        free(inkmap_wsum);
        inkmap_wsum = NULL;
        inkmap = NULL;
        inkmap_have = inkmap_uploaded = false;
        inkmap_path[0] = 0;
        if (seg_store_path && sgc_active[0]) {
          snprintf(inkmap_path, sizeof inkmap_path, "%s/%s%s.inkmap", seg_store_path,
                   sgc_active, ink_verso ? "-verso" : "");
          uint32_t lw, lh, lup;
          char sdp[400];
          snprintf(sdp, sizeof sdp, "cache/traced/%s/ink%s.inkmap", sgc_active,
                   ink_verso ? "-verso" : "");
          float *lm = inkmap_load(sdp, &lw, &lh, &lup, mv_seg.w, mv_seg.h,
                                  mv_seg.nvalid);
          if (!lm)
            lm = inkmap_load(inkmap_path, &lw, &lh, &lup, mv_seg.w, mv_seg.h,
                             mv_seg.nvalid);
          if (lm) {
            inkmap = lm;
            inkmap_w = lw;
            inkmap_h = lh;
            inkmap_up = lup;
            inkmap_have = true;
          }
        }
      }
      igEndDisabled();
      if (igIsItemHovered(0))
        igSetTooltip("run ink detection from the sheet's other face: the\n"
                     "slab is sampled with reversed layer order (vc3d\n"
                     "--direction both). Each side keeps its own cached map");
      if (igTreeNode_Str("TTA / ensemble")) {
        /* applied to full-map computes; each option multiplies inference
         * time (flips x4, depth +2, intensity +2, ensemble x models) */
        bool b0 = inkmap_tta & 1u, b1 = inkmap_tta & 2u, b2 = inkmap_tta & 4u,
             b3 = inkmap_tta & 8u;
        if (igCheckbox("flips (x4)", &b0)) inkmap_tta ^= 1u;
        if (igCheckbox("depth shift (+4)", &b1)) inkmap_tta ^= 2u;
        if (inkmap_tta & 2u) {
          /* range S: the surface can sit several voxels off mid-sheet;
           * the sampler ships a deeper slab (17 + 2*S layers) and the
           * server slides the window to +-S and +-S/2 */
          int dsr = (int)((inkmap_tta >> 8) & 0xfu);
          if (dsr < 1) dsr = 1;
          igSetNextItemWidth(140);
          if (igSliderInt("depth range##tta", &dsr, 1, 10, "+-%d layers", 0))
            inkmap_tta = (inkmap_tta & 0xffu) | ((uint32_t)dsr << 8);
        }
        if (igCheckbox("intensity x0.9/1.1 (+2)", &b2)) inkmap_tta ^= 4u;
        if (igCheckbox("checkpoint ensemble (x2)", &b3)) inkmap_tta ^= 8u;
        igTextDisabled("used by 'compute full ink map'; recompute to re-run\n"
                       "an existing map with new settings");
        igTreePop();
      }
      if (inkmap_have) {
        igTextDisabled("full ink map: %ux%u (cached, one-time)", inkmap_w, inkmap_h);
        igSameLine(0, 8);
        if (igSmallButton("recompute##ink")) {
          inkmap_have = inkmap_uploaded = false;
          if (inkmap_path[0]) remove(inkmap_path);
        }
      } else if (inkmap_job) {
        igText("computing full ink map: tile %u/%u", im_ty * im_ntx + im_tx + 1,
               im_ntx * im_nty);
      } else if (mv_seg.w > 2 && mv_seg.nvalid > 16) {
        static bool inkmap_test_fired = false;
        bool auto_fire = getenv("R3D_INKMAP_TEST") && !inkmap_test_fired;
        if (auto_fire) inkmap_test_fired = true;
        if (igButton("compute full ink map (once)", (ImVec2){0, 0}) || auto_fire) {
          /* whole surface at native resolution, tiled; result cached on
           * disk so this only ever runs once per saved segment */
          uint32_t iup = (uint32_t)lround(1.0 / (double)mv_seg.sx);
          if (iup < 1) iup = 1;
          uint32_t its = inkmap_tile_cells(iup);
          inkmap_up = iup;
          inkmap_w = (mv_seg.w - 1) * iup;
          inkmap_h = (mv_seg.h - 1) * iup;
          free(inkmap);
          free(inkmap_acc);
          inkmap_acc = NULL;
          free(inkmap_wsum);
          inkmap_wsum = NULL;
          inkmap = calloc((size_t)inkmap_w * inkmap_h, sizeof *inkmap);
          if (inkmap) {
            im_ts = its;
            im_ntx = (mv_seg.w - 2) / its + 1;
            im_nty = (mv_seg.h - 2) / its + 1;
            im_tx = im_ty = 0;
            im_req_out = false;
            inkmap_job = true;
            printf("inklive: computing full ink map, %ux%u px in %u tiles\n",
                   inkmap_w, inkmap_h, im_ntx * im_nty);
          }
        }
        if (sgc_active[0] == 0)
          igTextDisabled("(unsaved trace: map won't be cached to disk)");
      }
      if (mv_seg.w > 2 && igTreeNode_Str("supervision mask")) {
        /* conservatively label clear strokes (ink) and trustworthy empty
         * areas (background) on the flattened view; exported masks share
         * ink.png's exact pixel grid, so stack + ink + mask register for
         * fine-tuning (the post's iterative-refinement loop, in-tool) */
        igCheckbox("paint (LMB drag on flattened pane)##smpaint", &g_smask_paint);
        igSameLine(0, 12);
        igCheckbox("show##smshow", &g_smask_show);
        if (igRadioButton_Bool("ink##smc2", g_smask_class == 2)) g_smask_class = 2;
        igSameLine(0, 10);
        if (igRadioButton_Bool("background##smc1", g_smask_class == 1))
          g_smask_class = 1;
        igSameLine(0, 10);
        if (igRadioButton_Bool("erase##smc0", g_smask_class == 0)) g_smask_class = 0;
        igSetNextItemWidth(160);
        igSliderFloat("brush##smbrush", &g_smask_brush, 0.25f, 16.0f, "%.2f cells", 0);
        if (g_smask) {
          static uint64_t smc_ink = 0, smc_bg = 0;
          static uint32_t smc_frame = 0;
          if (frame_index - smc_frame > 32 || smc_frame == 0) {
            smc_frame = frame_index ? frame_index : 1;
            smc_ink = smc_bg = 0;
            for (size_t k = 0; k < (size_t)g_smask_w * g_smask_h; k++) {
              smc_ink += g_smask[k] == 2;
              smc_bg += g_smask[k] == 1;
            }
          }
          igText("%llu ink px, %llu background px%s",
                 (unsigned long long)smc_ink, (unsigned long long)smc_bg,
                 g_smask_disk_dirty ? " (unsaved)" : "");
          if (igButton("export mask##smexp", (ImVec2){0, 0}) && sgc_active[0]) {
            char dir[320];
            snprintf(dir, sizeof dir, "cache/traced/%s", sgc_active);
            struct stat sst;
            if (stat(dir, &sst) == 0) {
              char pth[400];
              snprintf(pth, sizeof pth, "%s/mask.inkmask", dir);
              if (smask_save_file(pth) == 0) g_smask_disk_dirty = false;
              uint8_t *px8 = malloc((size_t)g_smask_w * g_smask_h);
              if (px8) { /* 0 / 128 / 255 = unlabeled / background / ink */
                for (size_t k = 0; k < (size_t)g_smask_w * g_smask_h; k++)
                  px8[k] = g_smask[k] == 2 ? 255 : g_smask[k] == 1 ? 128 : 0;
                snprintf(pth, sizeof pth, "%s/mask.png", dir);
                if (r3d_png_write_gray(pth, px8, g_smask_w, g_smask_h) == 0)
                  printf("inklive: supervision mask image -> %s\n", pth);
                free(px8);
              }
            } else
              printf("inklive: mask export needs a traced segment dir\n");
          }
          igSameLine(0, 10);
          if (igButton("clear all##smclr", (ImVec2){0, 0})) {
            memset(g_smask, 0, (size_t)g_smask_w * g_smask_h);
            g_smask_gpu_dirty = g_smask_disk_dirty = true;
          }
          if (sgc_active[0] == 0)
            igTextDisabled("(unsaved trace: export/autosave unavailable)");
        } else
          igTextDisabled("paint to create the mask layer");
        igTreePop();
      }
      igTextDisabled("server 127.0.0.1:%d", inklive_port);
      igTextDisabled("%s", inklive.status);
    }
    if (bricks_path && igCollapsingHeader_TreeNodeFlags("post process", 0)) {
      /* GPU post-decode display filter: runs once per streamed brick, so
       * pane rendering stays free; applying flushes the resident bricks so
       * they re-stream through the filter IN PLACE — no reload, and no
       * viewer state is touched */
      static int pf_sel = 0; /* combo index == mode low byte */
      static bool pf_sharp = false;
      static float pf_amt = 1.0f;
      static bool pf_ct = true, pf_preds = false, pf_ink = false;
      igSetNextItemWidth(170);
      igCombo_Str("3D filter", &pf_sel,
                  "none\0median 3x3x3\0median 5x5x5\0max pool 3x3x3\0"
                  "max pool 5x5x5\0",
                  5);
      igCheckbox("sharpen 3D##pfs", &pf_sharp);
      if (pf_sharp) {
        igSameLine(0, 10);
        igSetNextItemWidth(150);
        igSliderFloat("amount##pfamt", &pf_amt, 0.25f, 4.0f, "%.2f", 0);
      }
      igText("apply to:");
      igSameLine(0, 8);
      igCheckbox("raw CT##pft", &pf_ct);
      if (overlay_path) {
        igSameLine(0, 8);
        igCheckbox("surface preds##pft", &pf_preds);
      }
      if (ink3d_ok) {
        igSameLine(0, 8);
        igCheckbox("3D ink##pft", &pf_ink);
      }
      if (igButton("apply##pf", (ImVec2){0, 0})) {
        uint32_t tgts = (pf_ct ? 1u : 0u) | (pf_preds && overlay_path ? 2u : 0u) |
                        (pf_ink && ink3d_ok ? 4u : 0u);
        uint32_t pfmode = (uint32_t)pf_sel | (pf_sharp ? 0x100u : 0u);
        r3d_bricks_postfilter(renderer, pfmode, pf_amt, tgts);
        r3d_bricks_refilter(renderer);
      }
      if (igIsItemHovered(0))
        igSetTooltip("resident bricks re-stream through the filter in place;\n"
                     "the view falls back to the coarse level for a moment\n"
                     "while they refill — nothing else changes");
    }
    if (bricks_path && igCollapsingHeader_TreeNodeFlags("labels", 0)) {
      if (!g_lbl_init) {
        igTextWrapped("paint 3D class labels (papyrus, ink, recto/verso, ...) into "
                      "the volume; saved losslessly as C5L1 label bricks");
        if (igButton("enable 3D labelling##lblen", (ImVec2){0, 0})) {
          uint32_t ld[3] = {brick_shape[0], brick_shape[1], brick_shape[2]};
          if (r3d_labelvol_init(&g_lblv, ld) == 0) {
            r3d_label_src ls = {lblsrc_gen, lblsrc_fetch, &g_lblv};
            if (r3d_bricks_labels(renderer, &ls) == 0) {
              g_lbl_init = true;
              g_lbl_paint = true;
              if (!g_lbl_dir[0]) { /* default: <lod root>/labels */
                char croot[560];
                snprintf(croot, sizeof croot, "%s", bricks_path);
                char *ms = strrchr(croot, '/');
                if (ms) *ms = 0;
                snprintf(g_lbl_dir, sizeof g_lbl_dir, "%s/labels", croot);
              }
              char mp[704];
              snprintf(mp, sizeof mp, "%s/manifest.json", g_lbl_dir);
              FILE *mf = fopen(mp, "r");
              if (mf) { /* labels already on disk: pick them straight up */
                fclose(mf);
                r3d_labelvol_load(&g_lblv, g_lbl_dir);
              }
            } else {
              r3d_labelvol_free(&g_lblv);
              printf("labels: renderer atlas unavailable (needs streamed LOD bricks)\n");
            }
          }
        }
      } else {
        igCheckbox("show##lblshow", &g_lbl_show);
        igSameLine(0, 12);
        igCheckbox("paint (LMB drag in plane views)##lblpaint", &g_lbl_paint);
        igSetNextItemWidth(170);
        igSliderFloat("brush##lblrad", &g_lbl_radius, 0.5f, 32.0f, "%.1f vox", 0);
        for (uint32_t c = 0; c < R3D_LBL_NCLASS; c++) {
          if (c) {
            const float *rgb = r3d_lbl_class_rgb[c];
            char cid[16];
            snprintf(cid, sizeof cid, "##lblc%u", c);
            igColorButton(cid, (ImVec4){rgb[0], rgb[1], rgb[2], 1.0f},
                          ImGuiColorEditFlags_NoTooltip, (ImVec2){14, 14});
            igSameLine(0, 6);
          }
          char rl[64];
          if (c && g_lblv.nvox[c])
            snprintf(rl, sizeof rl, "%s (%llu vox)##lblr%u", r3d_lbl_class_name[c],
                     (unsigned long long)g_lblv.nvox[c], c);
          else
            snprintf(rl, sizeof rl, "%s##lblr%u", r3d_lbl_class_name[c], c);
          if (igRadioButton_Bool(rl, g_lbl_class == (int)c)) g_lbl_class = (int)c;
        }
        igSetNextItemWidth(280);
        igInputText("dir##lbldir", g_lbl_dir, sizeof g_lbl_dir, 0, NULL, NULL);
        uint32_t lbl_unsaved = r3d_labelvol_dirty(&g_lblv);
        if (igButton("save##lblsave", (ImVec2){0, 0}) && g_lbl_dir[0]) {
          errno = 0;
          if (r3d_labelvol_save(&g_lblv, g_lbl_dir) == 0) {
            snprintf(g_lbl_status, sizeof g_lbl_status, "saved to %s", g_lbl_dir);
          } else { /* nothing is dropped: the edits stay dirty and retryable */
            snprintf(g_lbl_status, sizeof g_lbl_status, "SAVE FAILED: %s (%s)",
                     g_lbl_dir, errno ? strerror(errno) : "write error");
            fprintf(stderr, "labels: %s\n", g_lbl_status);
          }
        }
        igSameLine(0, 10);
        if (igButton("load##lblload", (ImVec2){0, 0}) && g_lbl_dir[0]) {
          errno = 0;
          if (r3d_labelvol_load(&g_lblv, g_lbl_dir) == 0)
            snprintf(g_lbl_status, sizeof g_lbl_status, "loaded from %s", g_lbl_dir);
          else {
            snprintf(g_lbl_status, sizeof g_lbl_status, "LOAD FAILED: %s (%s)",
                     g_lbl_dir, errno ? strerror(errno) : "read error");
            fprintf(stderr, "labels: %s\n", g_lbl_status);
          }
        }
        if (lbl_unsaved) {
          igSameLine(0, 12);
          igTextColored((ImVec4){1.0f, 0.75f, 0.3f, 1.0f}, "%u brick(s) unsaved",
                        lbl_unsaved);
        }
        if (g_lbl_status[0]) {
          bool bad = strstr(g_lbl_status, "FAILED") != NULL;
          igTextColored(bad ? (ImVec4){1.0f, 0.4f, 0.35f, 1.0f}
                            : (ImVec4){0.6f, 0.8f, 0.6f, 1.0f},
                        "%s", g_lbl_status);
        }
      }
    }
    if (bricks_path && igCollapsingHeader_TreeNodeFlags("registration", 0)) {
      if (!g_reg_open) {
        igTextWrapped("overlay a second scan of the same scroll and line it up "
                      "(green = this volume, magenta = the other scan)");
        igSetNextItemWidth(280);
        igInputText("moving volume##regroot", g_reg_root, sizeof g_reg_root, 0, NULL, NULL);
        if (igButton("open moving volume##regopen", (ImVec2){0, 0}) && g_reg_root[0]) {
          uint32_t fd[3] = {brick_shape[0], brick_shape[1], brick_shape[2]};
          reg_open_moving(renderer, g_reg_root, fd, bricks_path);
        }
        igSameLine(0, 10);
        igTextDisabled("(or use the data browser)");
      } else {
        igText("moving: %s", g_reg.root);
        { /* voxel-pitch matching: the correct scale between two scans of the
           * same scroll is the pitch ratio (crops differ, shapes don't tell).
           * Prefilled when a path carries "...um"; editable otherwise. */
          igSetNextItemWidth(70);
          igInputFloat("fixed um##regumf", &g_reg_um_fix, 0.0f, 0.0f, "%.4g", 0);
          igSameLine(0, 10);
          igSetNextItemWidth(70);
          igInputFloat("moving um##regumm", &g_reg_um_mov, 0.0f, 0.0f, "%.4g", 0);
          igSameLine(0, 10);
          igBeginDisabled(!(g_reg_um_fix > 0.0f) || !(g_reg_um_mov > 0.0f));
          if (igButton("match voxel size##regmatch", (ImVec2){0, 0})) {
            r3d_regvol_set_scale(&g_reg, (double)g_reg_um_fix / (double)g_reg_um_mov);
            reg_gui_reset_mirrors();
          }
          igEndDisabled();
          if (igIsItemHovered(0))
            igSetTooltip("replace the transform with the pure scale\n"
                         "fixed um / moving um (translation/rotation reset)");
          igText("current scale: %.4f moving vox per fixed vox",
                 r3d_regvol_scale(&g_reg));
        }
        igCheckbox("show##regshow", &g_reg_show);
        igSameLine(0, 12);
        igCheckbox("flattened pane##regflat", &g_reg_flat);
        igSetNextItemWidth(170);
        igSliderFloat("opacity##rega", &g_reg_alpha, 0.0f, 1.0f, "%.2f", 0);
        bool rch = false;
        igSetNextItemWidth(240);
        rch |= igDragFloat3("translate##regt", g_reg_tr, 0.25f, -1e6f, 1e6f, "%.2f", 0);
        igSetNextItemWidth(240);
        rch |= igDragFloat3("rotate deg##regr", g_reg_rot, 0.02f, -180.0f, 180.0f, "%.3f", 0);
        igSetNextItemWidth(120);
        rch |= igDragFloat("scale##regs", &g_reg_scale, 0.0005f, 0.01f, 100.0f, "%.4f", 0);
        if (rch) reg_gui_apply_mirrors();
        if (igButton("bake##regbake", (ImVec2){0, 0})) {
          r3d_regvol_bake(&g_reg);
          reg_gui_reset_mirrors();
        }
        if (igIsItemHovered(0))
          igSetTooltip("fold the sliders into the base transform and zero them");
        igSameLine(0, 10);
        if (igButton("reset deltas##regreset", (ImVec2){0, 0})) {
          r3d_regvol_reset_deltas(&g_reg);
          reg_gui_reset_mirrors();
        }
        igSetNextItemWidth(280);
        igInputText("transform.json##regjson", g_reg_json, sizeof g_reg_json, 0, NULL, NULL);
        if (igButton("load transform##regjl", (ImVec2){0, 0}) && g_reg_json[0]) {
          if (r3d_regvol_load_json(&g_reg, g_reg_json) == 0) reg_gui_reset_mirrors();
        }
        if (igIsItemHovered(0))
          igSetTooltip("shipped Vesuvius transform.json (transformation_matrix,\n"
                       "XYZ, moving->fixed) or this tool's own saves");
        igSameLine(0, 10);
        if (igButton("save transform##regjs", (ImVec2){0, 0}) && g_reg_json[0])
          r3d_regvol_save_json(&g_reg, g_reg_json);
        igSeparator();
        igSetNextItemWidth(150);
        igCombo_Str("##regmode", &g_reg_refmode, "measure NCC\0refine rigid\0refine affine\0",
                    3);
        igSameLine(0, 8);
        igSetNextItemWidth(90);
        igSliderInt("level##reglvl", &g_reg_reflevel, 0, 4, "L%d", 0);
        igSameLine(0, 8);
        if (!g_reg_busy && igButton("run##regrun", (ImVec2){0, 0})) {
          char croot[1024];
          snprintf(croot, sizeof croot, "%s", bricks_path);
          char *ms = strrchr(croot, '/');
          if (ms) *ms = 0;
          double ctr[3] = {mv_crop_c[0], mv_crop_c[1], mv_crop_c[2]};
          if (r3d_regvol_job_start(&g_reg, croot, g_reg_refmode, ctr, 64,
                                   (uint32_t)g_reg_reflevel) == 0)
            g_reg_busy = true;
        }
        if (igIsItemHovered(0))
          igSetTooltip("128^3 ROI at level L around the focus point\n"
                       "(Ctrl+click a pane to set the focus); refines\n"
                       "the transform in the background and applies it");
        if (g_reg_busy) {
          igSameLine(0, 10);
          igText("running...");
        }
        if (g_reg_ncc0 > -1.5)
          igText("NCC %.4f%s", g_reg_ncc0,
                 g_reg_ncc1 > -1.5 && g_reg_ncc1 != g_reg_ncc0 ? "" : " (at focus ROI)");
        if (g_reg_ncc1 > -1.5 && g_reg_ncc1 != g_reg_ncc0) {
          igSameLine(0, 6);
          igText("-> %.4f (refined)", g_reg_ncc1);
        }
        if (igButton("close moving volume##regclose", (ImVec2){0, 0})) {
          g_reg_open = false;
          g_reg_flat = false;
          g_reg_busy = false;
          r3d_bricks_regatlas_detach(renderer); /* worker holds the source */
          r3d_regvol_close(&g_reg);
        }
      }
    }
    if ((overlay_path || ink3d_ok) && igCollapsingHeader_TreeNodeFlags("overlay", 0)) {
        {
          if (overlay_path) igCheckbox("show##ovshow", &overlay_show);
          if (ink3d_ok) {
            if (overlay_path) igSameLine(0, 10);
            igCheckbox("3D ink (red)##i3dshow", &ink3d_show);
          }
          igSameLine(0, 10);
          igSetNextItemWidth(140);
          igSliderFloat("gain", &overlay_gain, 0.25f, 8.0f, "%.2f",
                        ImGuiSliderFlags_Logarithmic);
          if (multiview_path) {
            igText("show in:");
            static const char *ov_pane[4] = {"seg##ovp", "XY##ovp", "XZ##ovp", "YZ##ovp"};
            for (int op = 0; op < 4; op++) {
              igSameLine(0, 8);
              bool on = (mv_ov_mask >> op) & 1u;
              if (igCheckbox(ov_pane[op], &on))
                mv_ov_mask = on ? mv_ov_mask | (1u << op) : mv_ov_mask & ~(1u << op);
            }
          }
          if (n_overlays > 1) {
            for (uint32_t k = 0; k < n_overlays; k++) {
              const char *sl_ = strrchr(overlay_paths[k], '/');
              char lbl[96];
              snprintf(lbl, sizeof lbl, "%.80s##ov%u", sl_ ? sl_ + 1 : overlay_paths[k], k);
              bool cur = (int)k == overlay_sel;
              if (igRadioButton_Bool(lbl, cur) && !cur &&
                  r3d_bricks_overlay_switch(renderer, overlay_paths[k]) == 0) {
                overlay_sel = (int)k;
                overlay_path = overlay_paths[k];
              }
            }
          }
        }
    }
    if (igCollapsingHeader_TreeNodeFlags("streaming", 0)) {
        igText("bricks: hot %u/%u slots  warm %u (%.0f/%llu MB)%s", bst.hot, bst.hot_cap,
               bst.warm_bricks, (double)bst.warm_bytes / 1048576.0,
               (unsigned long long)(bst.warm_cap >> 20), bst.inflight ? "  streaming..." : "");
        if (bst.net_pending || bst.net_fetched)
          igText("net ingest: %u pending  %llu chunks fetched  %llu bricks cached",
                 bst.net_pending, (unsigned long long)bst.net_fetched,
                 (unsigned long long)bst.net_encoded);
        if (bst.nlevels > 1)
          igText("wanted L0..L%u: %u %u %u %u %u %u %u %u", bst.nlevels - 1u,
                 bst.lod_wanted[0], bst.lod_wanted[1], bst.lod_wanted[2], bst.lod_wanted[3],
                 bst.lod_wanted[4], bst.lod_wanted[5], bst.lod_wanted[6], bst.lod_wanted[7]);
    }
    }
    if (igCollapsingHeader_TreeNodeFlags("quality", 0)) {
      igSliderFloat("lod bias", &lod_bias, -2.0f, 4.0f, "%.2f", 0);
      int qp = quality_policy;
      if (igCombo_Str("policy##quality", &qp, "full\0interactive\0fast\0\0", 3)) {
        quality_policy = qp;
        adaptive_res = quality_policy != 0;
        r3d_set_quality(renderer, quality_policy == 2 ? R3D_QUALITY_FAST : R3D_QUALITY_FULL);
      }
      igCheckbox("half-res while moving", &adaptive_res);
    }
    if (igCollapsingHeader_TreeNodeFlags("transform", 0)) {
      float vt[3] = {vol_t.x, vol_t.y, vol_t.z};
      if (igDragFloat3("volume pos", vt, 0.002f, -4.0f, 4.0f, "%.3f", 0))
        vol_t = v3(vt[0], vt[1], vt[2]);
      float vr[3] = {vol_rot[0] * 57.29578f, vol_rot[1] * 57.29578f, vol_rot[2] * 57.29578f};
      if (igDragFloat3("volume rot (ypr)", vr, 0.5f, -180.0f, 180.0f, "%.1f", 0))
        for (int k = 0; k < 3; k++) vol_rot[k] = vr[k] / 57.29578f;
      if (igButton("reset volume", (ImVec2){0, 0})) {
        vol_t = v3(0, 0, 0);
        vol_rot[0] = vol_rot[1] = vol_rot[2] = 0.0f;
      }
      igSeparator();
      float ct[3] = {cam.target.x, cam.target.y, cam.target.z};
      if (igDragFloat3("cam target", ct, 0.002f, -4.0f, 4.0f, "%.3f", 0) &&
          cam_mode == CAM_ORBIT)
        r3d_camera_orbit_set(&cam, v3(ct[0], ct[1], ct[2]), cam.dist);
      if (igDragFloat("cam dist", &cam.dist, 0.005f, 0.0005f, 20.0f, "%.4f", 0) &&
          cam_mode == CAM_ORBIT)
        r3d_camera_orbit_set(&cam, cam.target, cam.dist);
      igSliderFloat("fov", &fov_deg, 20.0f, 120.0f, "%.0f°", 0);
    }
    igText("cam (%.2f %.2f %.2f) yaw %.2f pitch %.2f", (double)cam.pos.x, (double)cam.pos.y,
           (double)cam.pos.z, (double)cam.yaw, (double)cam.pitch);
    if (igCollapsingHeader_TreeNodeFlags("profile", 0)) {
      igText("gpu total   %6.2f ms", (double)prof.gpu_ns / 1e6);
      igText("  raycast   %6.2f ms", (double)prof.gpu_raycast_ns / 1e6);
      igText("  blit      %6.2f ms", (double)prof.gpu_blit_ns / 1e6);
      igText("  gui       %6.2f ms", (double)prof.gpu_gui_ns / 1e6);
      igText("cpu wait    %6.2f ms", (double)prof.cpu_wait_ns / 1e6);
      igText("cpu acquire %6.2f ms", (double)prof.cpu_acquire_ns / 1e6);
      igText("cpu record  %6.2f ms", (double)prof.cpu_record_ns / 1e6);
      igText("cpu submit  %6.2f ms", (double)prof.cpu_submit_ns / 1e6);
      igSeparator();
      for (int ph = 0; ph < MT_N; ph++)
        igText("mt %-6s   %6.2f ms (max %6.1f)", mt_name[ph], mt_ema[ph] / 1e6,
               (double)mt_max[ph] / 1e6);
    }
    if (umbilicus_path && vslab_mode)
      igTextDisabled("click: set point | wheel, R/F: 1 slice\n"
                     "PgUp/PgDn: z step | Ctrl+click: set + advance\n"
                     "Shift+drag: pan | Shift+wheel: zoom | F12: shot");
    else if (multiview_path)
      igTextDisabled("right-drag: pan view | wheel: zoom | Shift+wheel: slice\n"
                     "R/F: slice | Ctrl+click: set focus | F12: shot\n"
                     "Space: solo hovered view | checkboxes hide views\n"
                     "3D pane: wheel crop | Shift+wheel zoom | arrows/PgUpDn move%s",
                     umbilicus_path ? "\nU: place umbilicus point at cursor" : "");
    else if (cam_mode == CAM_ORBIT)
      igTextDisabled("drag orbit | shift+drag pan cam | ctrl+drag move vol\n"
                     "ctrl+shift+drag rot vol | wheel zoom | WASD pan | F12 shot");
    else
      igTextDisabled("click: fly (Esc releases)   WASD+QE: move   F12: shot");
    } /* panel_content */
    igEnd();

    od_browser_window(&od, &od_window, od_next_bricks, sizeof od_next_bricks, od_next_seg,
                      sizeof od_next_seg, od_next_ovl, sizeof od_next_ovl, od_next_ink,
                      sizeof od_next_ink, od_next_reg, sizeof od_next_reg, &od_swap,
                      &od_attach_ovl, &od_attach_ink, &od_attach_reg, bricks_path);
    if (od_attach_ovl) { /* browser: attach surface predictions live (blue) */
      od_attach_ovl = false;
      if (bricks_path && od_next_ovl[0] &&
          r3d_bricks_overlay_switch(renderer, od_next_ovl) == 0) {
        overlay_paths[0] = od_next_ovl;
        n_overlays = 1;
        overlay_sel = 0;
        overlay_path = od_next_ovl;
        overlay_show = true;
      } else if (od_next_ovl[0]) {
        printf("odbrowse: overlay %s failed to attach\n", od_next_ovl);
        od_next_ovl[0] = 0;
      }
    }
    if (od_attach_ink) { /* browser: attach 3D ink live (red, second slot) */
      od_attach_ink = false;
      if (bricks_path && od_next_ink[0] &&
          r3d_bricks_ink3d_switch(renderer, od_next_ink) == 0) {
        snprintf(ink3d_root, sizeof ink3d_root, "%s", od_next_ink);
        ink3d_ok = true;
        ink3d_show = true;
      } else if (od_next_ink[0]) {
        printf("odbrowse: 3D ink %s failed to attach\n", od_next_ink);
        od_next_ink[0] = 0;
      }
    }
    if (od_attach_reg) { /* browser: attach a second scan as the moving
                          * registration volume (green/magenta fuse) */
      od_attach_reg = false;
      if (bricks_path && od_next_reg[0]) {
        uint32_t fd3[3] = {brick_shape[0], brick_shape[1], brick_shape[2]};
        if (reg_open_moving(renderer, od_next_reg, fd3, bricks_path))
          printf("odbrowse: registration volume %s attached\n", od_next_reg);
        else {
          printf("odbrowse: registration volume %s failed to attach\n", od_next_reg);
          od_next_reg[0] = 0;
        }
      }
    }
    if (getenv("R3D_SWAP_TEST") && frame_index == 300 && !od_swap) {
      /* headless repro of a browser dataset swap: R3D_SWAP_TEST=<manifest.json>
       * (volume-only swap; the segment store and --inklive stay) */
      snprintf(od_next_bricks, sizeof od_next_bricks, "%s", getenv("R3D_SWAP_TEST"));
      od_next_seg[0] = 0;
      od_swap = true;
      frame_index = 0; /* so later frame-keyed test hooks fire again */
    }
    if (od_swap) running = false; /* teardown + reopen in the dataset loop */

    if (umbilicus_path) {
      if (annotation_z < 0) annotation_z = 0;
      if (annotation_z >= (int)vsnz) annotation_z = (int)vsnz - 1;
      vs_z0 = annotation_z0(annotation_z, vsnz, (uint32_t)vsd);
      const r3d_umbilicus_point *here = r3d_umbilicus_find(&umbilicus, annotation_z);
      ImVec2 marker;
      if (annotation_project(here, &cam, right, up, fwd, vm, vol_t, w, h, (uint32_t)vsw,
                             (uint32_t)vsh, (uint32_t)vsd, 0, 0, vs_z0, &marker)) {
        ImDrawList *draw = igGetBackgroundDrawList_Nil();
        const ImU32 black = 0xff000000u, yellow = 0xff00ffffu;
        ImDrawList_AddCircle(draw, marker, 12.0f, black, 32, 5.0f);
        ImDrawList_AddCircle(draw, marker, 12.0f, yellow, 32, 2.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x - 18.0f, marker.y},
                           (ImVec2){marker.x + 18.0f, marker.y}, black, 5.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x, marker.y - 18.0f},
                           (ImVec2){marker.x, marker.y + 18.0f}, black, 5.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x - 18.0f, marker.y},
                           (ImVec2){marker.x + 18.0f, marker.y}, yellow, 2.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x, marker.y - 18.0f},
                           (ImVec2){marker.x, marker.y + 18.0f}, yellow, 2.0f);
      }
    }

    if (getenv("R3D_TRACE_TEST") && multiview_path && n_overlays && !GT->active &&
        frame_index == 120) { /* headless: seed a trace at the focus */
      if (getenv("R3D_TRACE_RINGS"))
        mv_tr_rings = atoi(getenv("R3D_TRACE_RINGS"));
      r3d_tracer_cfg tc = {.seed = {mv_focus[0], mv_focus[1], mv_focus[2]},
                           .step = (double)mv_tr_step,
                           .thresh = mv_tr_thresh,
                           .max_ring = (uint32_t)mv_tr_rings,
                           .level = 1,
                           .wind_weight = umbilicus.count >= 2 ? 0.5 : 0.0};
      if (r3d_tracer_start(&GT->tr, overlay_paths[overlay_sel], &tc, &umbilicus) == 0) {
        GT->pos = malloc((size_t)GT->tr.W * GT->tr.H * 3 * sizeof *GT->pos);
        GT->st = calloc((size_t)GT->tr.W * GT->tr.H, 1);
        GT->cf = calloc((size_t)GT->tr.W * GT->tr.H, sizeof *GT->cf);
        GT->active = GT->pos && GT->st && GT->cf;
        mv_tr_view = true; /* headless hooks watch the trace */
        if (GT->active && getenv("R3D_ANCHOR_TEST")) {
          /* headless: anchors as "x,y,z;x,y,z;..." (world voxels) */
          const char *s = getenv("R3D_ANCHOR_TEST");
          while (s && *s && mv_anchor_n < R3D_TR_MAX_ANCHORS) {
            double *A = mv_anchor + (size_t)mv_anchor_n * 3;
            if (sscanf(s, "%lf,%lf,%lf", &A[0], &A[1], &A[2]) == 3) mv_anchor_n++;
            s = strchr(s, ';');
            if (s) s++;
          }
          if (mv_anchor_n) r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
        }
      }
    }
    static bool refine_test_done = false;
    if (GT->active && GT->done && mv_anchor_n && getenv("R3D_REFINE_TEST") &&
        !refine_test_done) { /* headless: one solve-only pass after finish */
      refine_test_done = true;
      r3d_tracer_stop(&GT->tr);
      r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
      if (r3d_tracer_refine(&GT->tr) == 0) {
        GT->done = false;
        GT->gen = 0;
        printf("tracer: refine test started\n");
      }
    }
    if (getenv("R3D_SEEDS_TEST") && frame_index == 300 && !mv_seeds_n) {
      /* headless: seeds as "x,y,z;x,y,z;..." then trace-all */
      if (getenv("R3D_TRACE_RINGS")) mv_tr_rings = atoi(getenv("R3D_TRACE_RINGS"));
      const char *sp2 = getenv("R3D_SEEDS_TEST");
      while (sp2 && *sp2 && mv_seeds_n < SEEDS_MAX) {
        double *A = mv_seeds + (size_t)mv_seeds_n * 3;
        if (sscanf(sp2, "%lf,%lf,%lf", &A[0], &A[1], &A[2]) == 3) mv_seeds_n++;
        sp2 = strchr(sp2, ';');
        if (sp2) sp2++;
      }
      if (mv_seeds_n) mv_seeds_go = true;
    }
    if (getenv("R3D_BSURF_TEST") && frame_index == 300 && !bsurf_active &&
        !bsurf_have_seed) {
      /* headless: "x,y,z,lowcut,gens,step,snap" -> seed + grow; the
       * finished grid auto-saves through the harvest pipeline */
      double bx = 0, by = 0, bz = 0, blc = 0, bst = 4, bsn = 6;
      int bg = 60;
      if (sscanf(getenv("R3D_BSURF_TEST"), "%lf,%lf,%lf,%lf,%d,%lf,%lf", &bx, &by, &bz,
                 &blc, &bg, &bst, &bsn) >= 4) {
        bsurf_seed[0] = bx;
        bsurf_seed[1] = by;
        bsurf_seed[2] = bz;
        bsurf_have_seed = true;
        low_cut = (float)blc;
        bsurf_gens = bg;
        bsurf_step = (float)bst;
        bsurf_snap = (float)bsn;
        bsurf_go = true;
        printf("bsurf test: seed (%.0f,%.0f,%.0f) lowcut %.0f gens %d step %.0f snap %.0f\n",
               bx, by, bz, blc, bg, bst, bsn);
      }
    }
    if (getenv("R3D_BSURF_TEST") && bsurf_active && !bsurf_save) {
      bool bdone = false;
      uint32_t bn = 0;
      r3d_bsurf_snapshot(&bsurf, NULL, NULL, &bn, &bdone);
      if (bdone) {
        printf("bsurf test: %s\n", bsurf.status);
        if (!bsurf.failed && bn > 64) bsurf_save = true;
        else bsurf_discard = true;
      }
    }
    if (getenv("R3D_MSURF_TEST") && frame_index == 300 && !msurf_n) {
      /* headless: points as "x,y,z;..." then create */
      const char *sp3 = getenv("R3D_MSURF_TEST");
      while (sp3 && *sp3 && msurf_n < MSURF_MAX) {
        double *A = msurf_pts[msurf_n];
        if (sscanf(sp3, "%lf,%lf,%lf", &A[0], &A[1], &A[2]) == 3) msurf_n++;
        sp3 = strchr(sp3, ';');
        if (sp3) sp3++;
      }
      if (msurf_n >= 4) msurf_create = true;
    }
    if (bsurf_discard && bsurf_active) {
      r3d_bsurf_free(&bsurf);
      bsurf_active = false;
      bsurf_discard = false;
    }
    if (bsurf_go && !bsurf_active && bsurf_have_seed && bricks_path) {
      bsurf_go = false;
      if (!bsurf_ct_ok) {
        char croot[1024];
        snprintf(croot, sizeof croot, "%s", bricks_path);
        char *ms2 = strrchr(croot, '/');
        if (ms2 && strcmp(ms2 + 1, "manifest.json") == 0) *ms2 = 0;
        bsurf_ct_ok = r3d_cpuvol_open(&bsurf_ct, croot, 256) == 0;
        if (!bsurf_ct_ok) printf("boundary surface: cannot open CT tree %s\n", croot);
      }
      if (bsurf_ct_ok) {
        r3d_bsurf_cfg bc = {.step = (double)bsurf_step,
                            .gens = (uint32_t)bsurf_gens,
                            .snap = (double)bsurf_snap,
                            .low_cut = (double)low_cut};
        memcpy(bc.seed, bsurf_seed, sizeof bc.seed);
        for (int a = 0; a < 3; a++) bc.vdim[a] = (double)brick_shape[a];
        if (r3d_bsurf_start(&bsurf, &bc, bsurf_ct_sample, &bsurf_ct) == 0) {
          bsurf_active = true;
          bsurf_first = true;
          bsurf_gen = 0;
          bsurf_prev_ns = 0;
          mv_tr_view = false;
          printf("boundary surface: growing %u gens, step %.0f, snap %.0f, low cut %.0f\n",
                 bc.gens, bc.step, bc.snap, bc.low_cut);
        }
      }
    }
    /* live preview of an authored surface (manual points re-fit on every
     * click; boundary grower polled as it grows): swap the grid into the
     * flattened pane immediately */
    double *pg3 = NULL;
    uint32_t pw3 = 0, ph3 = 0;
    bool *pfirst = &msurf_first;
    const char *plabel = "(manual)";
    if (msurf_dirty && !msurf_create && msurf_n >= 4 && multiview_path) {
      msurf_dirty = false;
      pg3 = msurf_fit((const double(*)[3])msurf_pts, msurf_n, (double)mv_tr_step,
                      &pw3, &ph3);
    } else if (bsurf_active && multiview_path) {
      bool bdone = false;
      uint64_t bg = r3d_bsurf_snapshot(&bsurf, NULL, NULL, NULL, &bdone);
      uint64_t bnow = r3d_now_ns();
      if (bg != bsurf_gen && (bdone || bnow - bsurf_prev_ns > 400000000ull)) {
        bsurf_gen = bg;
        bsurf_prev_ns = bnow;
        pw3 = bsurf.W;
        ph3 = bsurf.H;
        pg3 = malloc((size_t)pw3 * ph3 * 3 * sizeof *pg3);
        if (pg3) r3d_bsurf_snapshot(&bsurf, pg3, NULL, NULL, NULL);
        pfirst = &bsurf_first;
        plabel = "(boundary)";
      }
    }
    if (pg3) {
      {
        double pstep = pfirst == &msurf_first ? (double)mv_tr_step : bsurf.cfg.step;
        r3d_tifxyz ts = {.w = pw3, .h = ph3};
        ts.sx = ts.sy = (float)(1.0 / pstep);
        ts.xyz = malloc((size_t)pw3 * ph3 * 3 * sizeof *ts.xyz);
        if (ts.xyz) {
          float bb[2][3] = {{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
          uint64_t nv = 0;
          uint32_t gi0 = pw3, gi1 = 0, gj0 = ph3, gj1 = 0;
          for (size_t k = 0; k < (size_t)pw3 * ph3; k++) {
            bool ok3 = pg3[k * 3] >= 0.0;
            for (int a = 0; a < 3; a++)
              ts.xyz[k * 3 + (size_t)a] = ok3 ? (float)pg3[k * 3 + (size_t)a] : -1.0f;
            if (ok3) {
              nv++;
              for (int a = 0; a < 3; a++) {
                float vv3 = ts.xyz[k * 3 + (size_t)a];
                if (vv3 < bb[0][a]) bb[0][a] = vv3;
                if (vv3 > bb[1][a]) bb[1][a] = vv3;
              }
              uint32_t gi = (uint32_t)(k % pw3), gj = (uint32_t)(k / pw3);
              if (gi < gi0) gi0 = gi;
              if (gi > gi1) gi1 = gi;
              if (gj < gj0) gj0 = gj;
              if (gj > gj1) gj1 = gj;
            }
          }
          memcpy(ts.bbox, bb, sizeof bb);
          ts.nvalid = nv;
          float *tco = NULL, *tno = NULL;
          r3d_segrows trr = {0};
          if (nv > 4 && r3d_segrows_build(&ts, &trr) == 0 &&
              mv_build_grids(&ts, &tco, &tno) == 0 &&
              r3d_surf_swap(renderer, ts.w, ts.h, tco, tno, ts.sx, ts.sy) == 0) {
            smask_drop(renderer, sgc_active); /* mask belongs to the old segment */
            r3d_tifxyz_free(&mv_seg);
            r3d_segrows_free(&mv_rows);
            free(mv_normals);
            mv_seg = ts;
            mv_rows = trr;
            mv_normals = tno;
            snprintf(sgc_active, sizeof sgc_active, "%s", plabel);
            for (int oi2 = 0; oi2 < 4; oi2++) {
              mv_ol[oi2].n = 0;
              mv_ol_off[oi2].n = 0;
              mv_ol_slice[oi2] = 1e30;
            }
            mv_ol_zoff = 1e30;
            if (*pfirst) {
              *pfirst = false;
              mv[R3D_MV_SEG].cu = ((double)gi0 + gi1) * 0.5;
              mv[R3D_MV_SEG].cv = ((double)gj0 + gj1) * 0.5;
              double bw = (double)(gi1 - gi0) + 6.0, bh = (double)(gj1 - gj0) + 6.0;
              double zx = (double)mv[R3D_MV_SEG].pw / bw;
              double zy = (double)mv[R3D_MV_SEG].ph / bh;
              double zf = zx < zy ? zx : zy;
              double zmax3 = 10.0 / (double)mv_seg.sx;
              if (zf > zmax3) zf = zmax3;
              if (zf > 0.05) mv[R3D_MV_SEG].zoom = zf;
            }
          } else {
            r3d_tifxyz_free(&ts);
            r3d_segrows_free(&trr);
            free(tno);
          }
          free(tco);
        }
        free(pg3);
      }
    }
    if (msurf_create || (bsurf_save && bsurf_active)) {
      /* fit the clicked points into a grid (or take the boundary grower's
       * finished grid) and inject it as a FINISHED synthetic trace with
       * the harvest flag: the normal pipeline then saves it (timestamp
       * name), packs it into the store, and computes its ink map - a
       * hand-made surface is a first-class segment and a training label
       * with zero extra plumbing */
      bool from_bsurf = !msurf_create;
      msurf_create = false;
      bsurf_save = false;
      uint32_t mw = 0, mh = 0;
      double *mg = NULL;
      double mstep = (double)mv_tr_step;
      if (from_bsurf) {
        r3d_bsurf_stop(&bsurf);
        mw = bsurf.W;
        mh = bsurf.H;
        mstep = bsurf.cfg.step;
        mg = malloc((size_t)mw * mh * 3 * sizeof *mg);
        if (mg) r3d_bsurf_snapshot(&bsurf, mg, NULL, NULL, NULL);
      } else {
        mg = msurf_fit((const double(*)[3])msurf_pts, msurf_n, mstep, &mw, &mh);
      }
      int slot = -1;
      for (int s3 = 0; s3 < GT_MAX; s3++)
        if (!gts[s3].active) {
          slot = s3;
          break;
        }
      if (mg && slot >= 0) {
        struct gtrace *g = &gts[slot];
        memset(&g->tr, 0, sizeof g->tr);
        for (uint32_t a = 0; a < R3D_TR_MAX_ANCHORS; a++) g->tr.anc_cell[a] = -1;
        g->tr.W = mw;
        g->tr.H = mh;
        g->tr.cfg.step = mstep;
        g->tr.cfg.thresh = mv_tr_thresh;
        if (from_bsurf) /* voxel-snapped nodes: edges carry the normal
                         * excursion on top of the grid step */
          g->tr.tear_lim = 1.75 * mstep + bsurf.cfg.snap;
        g->tr.pos = mg;
        g->tr.state = malloc((size_t)mw * mh);
        g->tr.conf = malloc((size_t)mw * mh * sizeof *g->tr.conf);
        g->tr.wind = calloc((size_t)mw * mh, sizeof *g->tr.wind);
        g->tr.werr = calloc((size_t)mw * mh, sizeof *g->tr.werr);
        g->tr.gen_of = malloc((size_t)mw * mh * sizeof *g->tr.gen_of);
        if (g->tr.state && g->tr.conf && g->tr.wind && g->tr.gen_of) {
          uint32_t nv2 = 0;
          for (size_t k = 0; k < (size_t)mw * mh; k++) {
            bool valid = mg[k * 3] >= 0.0;
            g->tr.state[k] = valid ? R3D_TR_SET : R3D_TR_EMPTY;
            g->tr.conf[k] = valid ? 1.0f : 0.0f;
            g->tr.gen_of[k] = valid ? (from_bsurf ? bsurf.gen_of[k] : 1) : 0;
            if (valid) nv2++;
          }
          g->tr.nset = nv2;
          pthread_mutex_init(&g->tr.mu, NULL);
          r3d_umbilicus_init(&g->tr.umb);
          g->pos = NULL;
          g->st = NULL;
          g->cf = NULL;
          g->active = true;
          g->done = true;
          g->harvest = true; /* the pipeline takes it from here */
          g->nset = g->tr.nset;
          g->ring = 0;
          if (from_bsurf) {
            printf("boundary surface: %ux%u grid, %u nodes -> harvest\n", mw, mh,
                   nv2);
            r3d_bsurf_free(&bsurf);
            bsurf_active = false;
          } else {
            printf("manual surface: %ux%u grid from %u points -> harvest\n", mw,
                   mh, msurf_n);
            msurf_n = 0;
          }
        } else {
          free(g->tr.pos);
          free(g->tr.state);
          free(g->tr.conf);
          free(g->tr.wind);
          free(g->tr.werr);
          free(g->tr.gen_of);
          memset(&g->tr, 0, sizeof g->tr);
        }
      } else {
        free(mg);
        if (!mg)
          printf("manual surface: need points on 2+ z slices (2+ per slice)\n");
        else
          printf("manual surface: no free trace slot - stop/save one first\n");
      }
    }
    { /* auto-harvest: a finished QUEUE-STARTED trace is
       * saved straight into the segment store (selectable in the surfaces
       * panel like any segment) and its slot freed so the next seed
       * starts immediately - no manual discard/save needed. With an empty
       * queue, finished traces stay resident for interactive work. */
      const char *saved_dirs[GT_MAX];
      char saved_names[GT_MAX][256];
      uint32_t nharv = 0;
      for (int ti = 0; ti < GT_MAX; ti++) {
        struct gtrace *g = &gts[ti];
        if (!g->active || !g->done || !g->harvest) continue;
        r3d_tracer_stop(&g->tr);
        if (g->nset > 64) {
          char td[256];
          trace_dir_now(td, sizeof td);
          errno = 0;
          bool ok = mkdir_p(td) &&
                    r3d_tracer_save(&g->tr, td, mv_tr_thresh, mv_tr_fill) == 0;
          if (!ok) {
            /* the in-memory trace is the only copy of this result: never free
             * it because persistence failed. Drop it out of the harvest queue
             * (so it does not retry every frame) and leave it resident and
             * selectable so the user can retry save from the panel. */
            fprintf(stderr,
                    "tracer: SAVE FAILED for %s (%s) - trace kept in memory, "
                    "retry with \"save + activate\" in the surfaces panel\n",
                    td, errno ? strerror(errno) : "write error");
            g->harvest = false;
            continue;
          }
          ++mv_tr_nsaved;
          /* flatten before the store/ink chain (the ink model expects
           * flattened segments); raw dir kept for tracecli round trips */
          char fdh[280];
          if (trace_flatten_save(td, fdh, sizeof fdh) == 0)
            snprintf(saved_names[nharv], sizeof saved_names[0], "%s", fdh);
          else
            snprintf(saved_names[nharv], sizeof saved_names[0], "%s", td);
          saved_dirs[nharv] = saved_names[nharv];
          const char *bn = strrchr(saved_names[nharv], '/');
          bn = bn ? bn + 1 : saved_names[nharv];
          if (ink_qn < 24) { /* chain the 2.5D ink map (no TTA) */
            snprintf(ink_q[ink_qn], sizeof ink_q[0], "%s", bn);
            ink_qn++;
          }
          nharv++;
        }
        r3d_tracer_free(&g->tr);
        free(g->pos);
        free(g->st);
        free(g->cf);
        g->pos = NULL;
        g->st = NULL;
        g->cf = NULL;
        g->active = g->done = false;
      }
      if (nharv && sgc.open && seg_store_path &&
          r3d_segstore_build(seg_store_path, saved_dirs, nharv, 2, false) <= 0)
        fprintf(stderr,
                "tracer: segment-store rebuild failed for %s - %u harvested trace(s) "
                "are on disk but not in the corpus\n",
                seg_store_path, nharv);
      else if (nharv && sgc.open && seg_store_path) {
        for (int oi2 = 1; oi2 < 4; oi2++) { /* rebuild the polyline caches */
          if (!sgc_ln[oi2]) continue;
          for (uint32_t si = 0; si < sgc.st.n; si++) {
            free(sgc_ln[oi2][si].l.w);
            free(sgc_ln[oi2][si].l.g);
          }
          free(sgc_ln[oi2]);
          sgc_ln[oi2] = NULL;
        }
        sgc_close(&sgc);
        if (sgc_open(&sgc, seg_store_path, (size_t)1536 << 20) == 0) {
          for (int oi2 = 1; oi2 < 4; oi2++)
            sgc_ln[oi2] = calloc(sgc.st.n ? sgc.st.n : 1, sizeof *sgc_ln[oi2]);
          for (int oi2 = 0; oi2 < 4; oi2++) {
            sgc_key_slice[oi2] = 1e30;
            sgc_key_gen[oi2] = UINT32_MAX;
          }
          memcpy(sgc_near_focus, (double[3]){1e30, 1e30, 1e30},
                 sizeof sgc_near_focus);
          printf("tracer: %u finished trace(s) harvested into the store "
                 "(%u surfaces)\n", nharv, sgc.st.n);
        } else {
          fprintf(stderr, "tracer: segment-store reopen failed for %s - the saved "
                          "trace(s) remain on disk\n", seg_store_path);
        }
      }
    }
    if (mv_seeds_go && mv_seeds_n && multiview_path && n_overlays) {
      /* draining queue: start into whatever slots are free this frame and
       * stay armed until the queue empties */
      const char *pr = overlay_paths[overlay_sel];
      uint32_t nact = 0;
      for (int ti = 0; ti < GT_MAX; ti++)
        if (gts[ti].active) nact++;
      {
            uint32_t nrun = mv_seeds_n < GT_MAX - nact ? mv_seeds_n : GT_MAX - nact;
            uint32_t cap = 20u / (nact + (nrun ? nrun : 1));
            if (cap < 2) cap = 2;
            for (uint32_t si2 = 0; si2 < nrun; si2++) {
              int slot = -1;
              for (int s3 = 0; s3 < GT_MAX; s3++)
                if (!gts[s3].active) {
                  slot = s3;
                  break;
                }
              if (slot < 0) break;
              gt_sel = slot;
              r3d_tracer_cfg tc = {.seed = {mv_seeds[si2 * 3], mv_seeds[si2 * 3 + 1],
                                            mv_seeds[si2 * 3 + 2]},
                                   .step = (double)mv_tr_step,
                                   .thresh = mv_tr_thresh,
                                   .max_ring = (uint32_t)mv_tr_rings,
                                   .level = 1,
                                   .max_threads = cap,
                                   .wind_weight = mv_tr_spiral && umbilicus.count >= 2
                                                      ? 0.5
                                                      : 0.0};
              if (r3d_tracer_start(&GT->tr, pr, &tc, &umbilicus) == 0) {
                GT->harvest = true; /* auto-save + auto-ink on finish */
                if (sgc_active[0] == 0) mv_tr_view = true;
                if (mv_anchor_n)
                  r3d_tracer_set_anchors(&GT->tr, mv_anchor, mv_anchor_n);
                free(GT->pos);
                free(GT->st);
                free(GT->cf);
                GT->pos = malloc((size_t)GT->tr.W * GT->tr.H * 3 * sizeof *GT->pos);
                GT->st = calloc((size_t)GT->tr.W * GT->tr.H, 1);
                GT->cf = calloc((size_t)GT->tr.W * GT->tr.H, sizeof *GT->cf);
                GT->gen = 0;
                GT->live_first = true;
                GT->active = GT->pos && GT->st && GT->cf;
              }
            }
            if (nrun) { /* drop the started seeds; keep the rest queued */
              memmove(mv_seeds, mv_seeds + (size_t)nrun * 3,
                      (size_t)(mv_seeds_n - nrun) * 3 * sizeof *mv_seeds);
              mv_seeds_n -= nrun;
              printf("tracer: growing %u concurrent trace(s), %u seed%s queued\n",
                     nrun, mv_seeds_n, mv_seeds_n == 1 ? "" : "s");
            }
            if (!mv_seeds_n) mv_seeds_go = false; /* queue drained */
      }
    } else if (!mv_seeds_n) {
      mv_seeds_go = false;
    }
    for (int ti = 0; ti < GT_MAX; ti++) /* background traces: light poll
        (ring/points/done for the panel; no buffer copy, no live swap) */
      if (ti != gt_sel && gts[ti].active)
        r3d_tracer_snapshot(&gts[ti].tr, NULL, NULL, NULL, &gts[ti].ring,
                            &gts[ti].nset, &gts[ti].done);
    if (GT->active) { /* live tracer snapshot when it grew */
      uint64_t g = r3d_tracer_snapshot(&GT->tr, NULL, NULL, NULL, &GT->ring,
                                       &GT->nset, &GT->done);
      if (g != GT->gen && GT->pos) {
        r3d_tracer_snapshot(&GT->tr, GT->pos, GT->st, GT->cf, NULL, NULL, NULL);
        GT->gen = g;
        uint64_t now2 = r3d_now_ns();
        if (mv_tr_live && mv_tr_view && GT->nset > 8 &&
            (now2 - mv_tr_live_ns > 400000000ull || GT->done)) {
          /* live growth in the segment pane: swap the trace grid into the
           * active surf machinery (tiny grids — the upload is trivial; the
           * surfvol rebakes progressively) */
          mv_tr_live_ns = now2;
          if ((inkmap_have || inkmap_job || inklive_have) && !imq_active) {
            /* discarded/replaced surface: the old segment's ink (map, job,
             * and the GPU texture) must all die with it. Background queue
             * jobs are independent of the display and keep running. */
            free(inkmap);
            free(inkmap_acc);
            inkmap_acc = NULL;
            free(inkmap_wsum);
            inkmap_wsum = NULL;
            inkmap = NULL;
            inkmap_have = inkmap_uploaded = inkmap_job = false;
            im_req_out = false;
            inkmap_path[0] = 0;
            inklive_have = false;
            r3d_surfvol_inkpred_clear(renderer);
          } else if (inklive_have && imq_active) {
            inklive_have = false; /* display state only */
            inkmap_have = inkmap_uploaded = false;
            r3d_surfvol_inkpred_clear(renderer);
          }
          r3d_tifxyz ts = {.w = GT->tr.W, .h = GT->tr.H};
          ts.sx = ts.sy = (float)(1.0 / GT->tr.cfg.step);
          ts.xyz = malloc((size_t)ts.w * ts.h * 3 * sizeof *ts.xyz);
          if (ts.xyz) {
            float bb[2][3] = {{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
            uint64_t nv = 0;
            uint32_t gi0 = ts.w, gi1 = 0, gj0 = ts.h, gj1 = 0; /* grid bbox */
            for (size_t k = 0; k < (size_t)ts.w * ts.h; k++) {
              bool ok2 = GT->st[k] == R3D_TR_SET; /* growth never skips a
                  * cell; the cutoff is a SAVE decision, not a display one */
              for (int a = 0; a < 3; a++) {
                float vv2 = ok2 ? (float)GT->pos[k * 3 + (size_t)a] : -1.0f;
                ts.xyz[k * 3 + (size_t)a] = vv2;
                if (ok2) {
                  if (vv2 < bb[0][a]) bb[0][a] = vv2;
                  if (vv2 > bb[1][a]) bb[1][a] = vv2;
                }
              }
              if (ok2) {
                nv++;
                uint32_t gi = (uint32_t)(k % ts.w), gj = (uint32_t)(k / ts.w);
                if (gi < gi0) gi0 = gi;
                if (gi > gi1) gi1 = gi;
                if (gj < gj0) gj0 = gj;
                if (gj > gj1) gj1 = gj;
              }
            }
            memcpy(ts.bbox, bb, sizeof bb);
            ts.nvalid = nv;
            float *tco = NULL, *tno = NULL;
            r3d_segrows trr = {0};
            if (nv > 8 && r3d_segrows_build(&ts, &trr) == 0 &&
                mv_build_grids(&ts, &tco, &tno) == 0 &&
                r3d_surf_swap(renderer, ts.w, ts.h, tco, tno, ts.sx, ts.sy) == 0) {
              r3d_tifxyz_free(&mv_seg);
              r3d_segrows_free(&mv_rows);
              free(mv_normals);
              mv_seg = ts;
              mv_rows = trr;
              mv_normals = tno;
              snprintf(sgc_active, sizeof sgc_active, "(tracing)");
              for (int oi2 = 0; oi2 < 4; oi2++) {
                mv_ol[oi2].n = 0;
                mv_ol_off[oi2].n = 0;
                mv_ol_slice[oi2] = 1e30;
              }
              mv_ol_zoff = 1e30;
              if (GT->live_first || !GT->done) {
                /* keep the pane fitted to the growing patch; once growth
                 * finishes the view is yours */
                GT->live_first = false;
                double bw = (double)(gi1 - gi0) + 6.0, bh = (double)(gj1 - gj0) + 6.0;
                mv[R3D_MV_SEG].cu = ((double)gi0 + gi1) * 0.5;
                mv[R3D_MV_SEG].cv = ((double)gj0 + gj1) * 0.5;
                double zx = (double)mv[R3D_MV_SEG].pw / bw;
                double zy = (double)mv[R3D_MV_SEG].ph / bh;
                double zf = zx < zy ? zx : zy;
                double zmax2 = 10.0 / (double)mv_seg.sx;
                if (zf > zmax2) zf = zmax2;
                if (zf > 0.05) mv[R3D_MV_SEG].zoom = zf;
              }
            } else {
              r3d_tifxyz_free(&ts);
              r3d_segrows_free(&trr);
              free(tno);
            }
            free(tco);
          }
        }
      }
    }
    if (multiview_path) { /* overlays: intersections, focus marker, borders */
      /* recompute intersection polylines only when their inputs move */
      double zoff = mv[R3D_MV_SEG].slice;
      for (int i = 1; i < 4; i++) {
        /* recompute when the pane is visible OR the segment view is (its
         * plane trace lines draw there even with the pane collapsed/solo) */
        if (MV_IS3D(i)) continue;
        if (!(mv_mask & (1u << i)) && !(mv_mask & 1u)) continue;
        bool stale = mv_ol_slice[i] != mv[i].slice || mv_ol_gen[i] != mv_basis_gen;
        if (stale) {
          mv_ol[i].n = 0;
          r3d_segtrace_basis(&mv_seg, &mv_rows, NULL, 0.0f, mv_po[i], mv_pb[i][0],
                             mv_pb[i][1], mv_pb[i][2], mv[i].slice, mv_lines_emit,
                             &mv_ol[i]);
        }
        if (stale || mv_ol_zoff != zoff) {
          mv_ol_off[i].n = 0;
          if (zoff != 0.0)
            r3d_segtrace_basis(&mv_seg, &mv_rows, mv_normals, (float)zoff, mv_po[i],
                               mv_pb[i][0], mv_pb[i][1], mv_pb[i][2], mv[i].slice,
                               mv_lines_emit, &mv_ol_off[i]);
        }
        mv_ol_slice[i] = mv[i].slice;
        mv_ol_gen[i] = mv_basis_gen;
      }
      mv_ol_zoff = zoff;

      ImDrawList *draw = igGetBackgroundDrawList_Nil();
      const ImU32 fc = 0xffd7ff32u; /* vc3d focus teal (50,255,215), ABGR */
      const ImU32 bc = 0x60808080u;
      const ImU32 seg_col = 0xff50dcffu;      /* active segment: warm yellow */
      const ImU32 seg_off_col = 0x9050dcffu;  /* zoff shell: same, translucent */
      /* plane trace colors on the segment view (vc3d): XY orange, XZ red,
       * YZ yellow */
      const ImU32 trace_col[4] = {0, 0xff008cffu, 0xff0000ffu, 0xff00ffffu};
      if (sgc.open) { /* corpus surfaces: dimmed intersection polylines under
         * the active segment's curve. Queries hit only the tile index;
         * grids come from the worker's cache (missing ones get queued) and
         * at most a few re-traces run per frame to amortize slice scrubs. */
        /* corpus line visibility: user-adjustable alpha; brighter teal
         * default (the old 0x50 gray vanished into the CT) */
        uint32_t ca8 = (uint32_t)(mv_corpus_vis * 255.0f + 0.5f);
        const ImU32 dim_col = (ca8 << 24) | 0x00c8b478u;
        const ImU32 ov_hi = 0x783c8ce6u; /* heavy overlap with active: orange */
        const ImU32 ov_lo = 0x5864b4d2u; /* light overlap: sand */
        pthread_mutex_lock(&sgc.mu);
        int traces = 0;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
          const double *bn = mv_pb[i][2];
          double sl = mv[i].slice + bn[0] * mv_po[i][0] + bn[1] * mv_po[i][1] +
                      bn[2] * mv_po[i][2];
          if (sgc_key_slice[i] != sl || sgc_key_gen[i] != mv_basis_gen) {
            sgc_key_slice[i] = sl;
            sgc_key_gen[i] = mv_basis_gen;
            sgc_nhits[i] = r3d_segstore_plane_query(&sgc.st, bn, sl, 1.0, NULL, NULL,
                                                    sgc_hits[i], 128);
            if (sgc_nhits[i] > 128) sgc_nhits[i] = 128;
            /* nearest-first by bbox center vs the pane center: zoomed-out
             * panes cross the whole corpus, and drawing every polyline
             * costs ~8 ms/frame — only the closest SGC_DRAW_MAX draw */
            double pcw[3];
            r3d_mv_b2w(mv_pb[i], mv_po[i], mv[i].cu, mv[i].cv, mv[i].slice, pcw);
            for (uint32_t a2 = 1; a2 < sgc_nhits[i]; a2++)
              for (uint32_t b2 = a2; b2 > 0; b2--) {
                double da = 0.0, db = 0.0;
                for (int c2 = 0; c2 < 3; c2++) {
                  const r3d_segmeta *sa = &sgc.st.segs[sgc_hits[i][b2]];
                  const r3d_segmeta *sb = &sgc.st.segs[sgc_hits[i][b2 - 1]];
                  double ca =
                      ((double)sa->bbox[0][c2] + (double)sa->bbox[1][c2]) * 0.5 - pcw[c2];
                  double cb =
                      ((double)sb->bbox[0][c2] + (double)sb->bbox[1][c2]) * 0.5 - pcw[c2];
                  da += ca * ca;
                  db += cb * cb;
                }
                if (da >= db) break;
                uint32_t tswap = sgc_hits[i][b2];
                sgc_hits[i][b2] = sgc_hits[i][b2 - 1];
                sgc_hits[i][b2 - 1] = tswap;
              }
          }
          ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
          ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
          ImDrawList_PushClipRect(draw, cmin, cmax, false);
          uint32_t draw_n = sgc_nhits[i] < 48 ? sgc_nhits[i] : 48;
          for (uint32_t k = 0; k < draw_n; k++) {
            uint32_t si = sgc_hits[i][k];
            if (strcmp(sgc.st.segs[si].name, sgc_active) == 0) continue;
            sgc_ent *e = &sgc.ent[si];
            if (e->state == SGC_EMPTY) {
              sgc_request(&sgc, si);
              continue;
            }
            sgc_line *ln = &sgc_ln[i][si];
            if (e->state == SGC_READY &&
                (!ln->valid || ln->slice != mv[i].slice || ln->gen != mv_basis_gen) &&
                traces < 2) {
              traces++;
              ln->l.n = 0;
              r3d_segtrace_basis(&e->s, &e->rows, NULL, 0.0f, mv_po[i], mv_pb[i][0],
                                 mv_pb[i][1], mv_pb[i][2], mv[i].slice, mv_lines_emit,
                                 &ln->l);
              ln->slice = mv[i].slice;
              ln->gen = mv_basis_gen;
              ln->valid = true;
              e->last_use = ++sgc.tick;
            }
            if (ln->valid && ln->slice == mv[i].slice && ln->gen == mv_basis_gen) {
              float ovf = sgc.ov_active != UINT32_MAX ? sgc.ov[si] : 0.0f;
              ImU32 col = ovf > 0.15f ? ov_hi : (ovf > 0.02f ? ov_lo : dim_col);
              /* corpus lines tolerate a 3 px collapse: half the tessellation */
              mv_draw_lines(draw, &mv[i], ln->l.w, ln->l.n, col, 1.0f, 9.0f);
            }
          }
          ImDrawList_PopClipRect(draw);
        }
        pthread_mutex_unlock(&sgc.mu);
      }
      for (int i = 1; i < 4; i++) { /* segment curve on each plane view */
        if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
        ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
        ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
        ImDrawList_PushClipRect(draw, cmin, cmax, false);
        mv_draw_lines(draw, &mv[i], mv_ol[i].w, mv_ol[i].n, seg_col, 1.6f, 1.0f);
        mv_draw_lines(draw, &mv[i], mv_ol_off[i].w, mv_ol_off[i].n, seg_off_col, 1.0f, 1.0f);
        ImDrawList_PopClipRect(draw);
      }
      if (mv_mask & 1u) { /* plane trace lines on the flattened segment view */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        ImVec2 cmin = {(float)sv->px, (float)sv->py};
        ImVec2 cmax = {(float)(sv->px + sv->pw), (float)(sv->py + sv->ph)};
        ImDrawList_PushClipRect(draw, cmin, cmax, false);
        for (int i = 1; i < 4; i++) /* even for collapsed panes: the slices
                                     * still exist and the lines orient you */
          mv_draw_lines(draw, sv, mv_ol[i].g, mv_ol[i].n, trace_col[i], 1.4f, 1.0f);
        ImDrawList_PopClipRect(draw);
      }
      for (int i = 1; i < 4; i++) { /* pane borders around visible views */
        if (!(mv_mask & (1u << i))) continue;
        if (mv[i].px > 0)
          ImDrawList_AddLine(draw, (ImVec2){(float)mv[i].px, (float)mv[i].py},
                             (ImVec2){(float)mv[i].px, (float)(mv[i].py + mv[i].ph)}, bc,
                             1.0f);
        if (mv[i].py > 0)
          ImDrawList_AddLine(draw, (ImVec2){(float)mv[i].px, (float)mv[i].py},
                             (ImVec2){(float)(mv[i].px + mv[i].pw), (float)mv[i].py}, bc,
                             1.0f);
      }
      for (int i = 1; i < 4; i++) {
        if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
        double fu, fv, fs;
        r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
        float fx_, fy_;
        r3d_mv_project(&mv[i], fu, fv, &fx_, &fy_);
        if (fx_ >= (float)mv[i].px && fx_ < (float)(mv[i].px + mv[i].pw) &&
            fy_ >= (float)mv[i].py && fy_ < (float)(mv[i].py + mv[i].ph))
          ImDrawList_AddCircle(draw, (ImVec2){fx_, fy_}, 10.0f, fc, 24, 2.0f);
      }
      if (msurf_n) { /* manual-surface points: magenta dots, consecutive
                      * same-row clicks connected so the sheet path reads */
        const ImU32 mb_ = 0xff000000u, mc_ = 0xffff40ffu, md_ = 0x80ff40ffu;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
          float lx = 0, ly = 0;
          bool have_prev = false;
          double prev_z = 1e30;
          for (uint32_t a = 0; a < msurf_n; a++) {
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], msurf_pts[a], &fu, &fv, &fs);
            float ax_, ay_;
            r3d_mv_project(&mv[i], fu, fv, &ax_, &ay_);
            double zd = fabs(fs - mv[i].slice);
            if (zd >= 3.0) continue; /* points live in a 3x3x3: on this
                                      * pane only within ~1.5 slices,
                                      * faint out to 3, gone beyond */
            bool on = zd < 1.5;
            bool inpane = ax_ >= (float)mv[i].px && ax_ < (float)(mv[i].px + mv[i].pw) &&
                          ay_ >= (float)mv[i].py && ay_ < (float)(mv[i].py + mv[i].ph);
            if (inpane) {
              ImDrawList_AddCircleFilled(draw, (ImVec2){ax_, ay_}, 4.0f, mb_, 12);
              ImDrawList_AddCircleFilled(draw, (ImVec2){ax_, ay_}, 3.0f,
                                         on ? mc_ : md_, 12);
            }
            (void)lx;
            (void)ly;
            (void)have_prev;
            (void)prev_z;
          }
        }
      }
      if (bsurf_have_seed) { /* boundary-surface seed: cyan ring */
        const ImU32 bb_ = 0xff000000u, bc_ = 0xffffd040u, bd_ = 0x80ffd040u;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
          double fu, fv, fs;
          r3d_mv_w2b(mv_pb[i], mv_po[i], bsurf_seed, &fu, &fv, &fs);
          float ax_, ay_;
          r3d_mv_project(&mv[i], fu, fv, &ax_, &ay_);
          double zd = fabs(fs - mv[i].slice);
          if (zd >= 6.0) continue;
          if (ax_ < (float)mv[i].px || ax_ >= (float)(mv[i].px + mv[i].pw) ||
              ay_ < (float)mv[i].py || ay_ >= (float)(mv[i].py + mv[i].ph))
            continue;
          ImDrawList_AddCircle(draw, (ImVec2){ax_, ay_}, 7.0f, bb_, 20, 4.0f);
          ImDrawList_AddCircle(draw, (ImVec2){ax_, ay_}, 7.0f, zd < 1.5 ? bc_ : bd_, 20,
                               2.0f);
        }
      }
      if (mv_seeds_n) { /* queued trace seeds: green circles */
        const ImU32 sb_ = 0xff000000u, sc_ = 0xff40d060u, sd_ = 0x8040d060u;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
          for (uint32_t a = 0; a < mv_seeds_n; a++) {
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_seeds + (size_t)a * 3, &fu, &fv, &fs);
            float ax_, ay_;
            r3d_mv_project(&mv[i], fu, fv, &ax_, &ay_);
            if (ax_ < (float)mv[i].px || ax_ >= (float)(mv[i].px + mv[i].pw) ||
                ay_ < (float)mv[i].py || ay_ >= (float)(mv[i].py + mv[i].ph))
              continue;
            ImU32 col = fabs(fs - mv[i].slice) < 6.0 ? sc_ : sd_;
            ImDrawList_AddCircle(draw, (ImVec2){ax_, ay_}, 8.0f, sb_, 20, 3.5f);
            ImDrawList_AddCircle(draw, (ImVec2){ax_, ay_}, 8.0f, col, 20, 1.8f);
          }
        }
      }
      if (mv_anchor_n) { /* tracer anchors: orange diamonds in the plane
                          * panes, dimmed when off this pane's slice */
        const ImU32 ab_ = 0xff000000u, ac_ = 0xff00a5ffu, ad_ = 0x8000a5ffu;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i)) || MV_IS3D(i)) continue;
          for (uint32_t a = 0; a < mv_anchor_n; a++) {
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_anchor + (size_t)a * 3, &fu, &fv, &fs);
            float ax_, ay_;
            r3d_mv_project(&mv[i], fu, fv, &ax_, &ay_);
            if (ax_ < (float)mv[i].px || ax_ >= (float)(mv[i].px + mv[i].pw) ||
                ay_ < (float)mv[i].py || ay_ >= (float)(mv[i].py + mv[i].ph))
              continue;
            ImU32 col = fabs(fs - mv[i].slice) < 6.0 ? ac_ : ad_;
            float r_ = 7.0f;
            ImVec2 q[4] = {{ax_, ay_ - r_}, {ax_ + r_, ay_}, {ax_, ay_ + r_},
                           {ax_ - r_, ay_}};
            ImDrawList_AddQuad(draw, q[0], q[1], q[2], q[3], ab_, 3.5f);
            ImDrawList_AddQuad(draw, q[0], q[1], q[2], q[3], col, 1.8f);
          }
        }
      }
      if (umbilicus_path && umbilicus.count) {
        /* umbilicus curve in every plane pane: connected control points
         * (magenta) + a crosshair at the XY pane's current-z interpolation */
        const ImU32 uc = 0xffff00ffu, ub = 0xff000000u;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i))) continue;
          if (MV_IS3D(i)) {
            if (mv_crop_d[0] <= 0.0f) continue;
            /* the curve inside the crop box, through the pane's camera
             * (recomputed here exactly as the frame build does) */
            uint32_t md3 = brick_shape[0];
            for (int a = 1; a < 3; a++)
              if (brick_shape[a] > md3) md3 = brick_shape[a];
            float mde = mv_crop_d[0] > mv_crop_d[1] ? mv_crop_d[0] : mv_crop_d[1];
            if (mv_crop_d[2] > mde) mde = mv_crop_d[2];
            double eb3 = (double)mde / md3;
            r3d_camera_orbit_set(&cam,
                                 v3((float)(mv_crop_c[0] / md3), (float)(mv_crop_c[1] / md3),
                                    (float)(mv_crop_c[2] / md3)),
                                 (float)(eb3 * (double)mv_crop_fit));
            r3d_v3 c_r, c_u, c_f;
            r3d_camera_basis(&cam, (float)mv[i].pw / (float)(mv[i].ph ? mv[i].ph : 1),
                             &c_r, &c_u, &c_f);
            float lr = v3_len(c_r), lu = v3_len(c_u);
            r3d_v3 rn = v3_scale(c_r, lr > 0 ? 1.0f / lr : 0.0f);
            r3d_v3 un = v3_scale(c_u, lu > 0 ? 1.0f / lu : 0.0f);
            double half3v[3] = {(double)mv_crop_d[0] * 0.5, (double)mv_crop_d[1] * 0.5,
                                (double)mv_crop_d[2] * 0.5};
            ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
            ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
            ImDrawList_PushClipRect(draw, cmin, cmax, false);
            for (size_t k = 0; k < umbilicus.count; k++) {
              /* clip segment k-1 -> k to the cube */
              ImVec2 s0 = {0, 0}, s1 = {0, 0};
              bool seg_ok = false;
              if (k > 0) {
                double ct0 = 0.0, ct1 = 1.0;
                const r3d_umbilicus_point *pa = &umbilicus.points[k - 1];
                const r3d_umbilicus_point *pb = &umbilicus.points[k];
                double pav[3] = {pa->x, pa->y, pa->z}, pbv[3] = {pb->x, pb->y, pb->z};
                for (int a = 0; a < 3 && ct0 < ct1; a++) {
                  double lo = mv_crop_c[a] - half3v[a], hi = mv_crop_c[a] + half3v[a];
                  double va = pav[a], vb = pbv[a];
                  if (va == vb) {
                    if (va < lo || va > hi) ct0 = 1.0;
                  } else {
                    double ta = (lo - va) / (vb - va), tb = (hi - va) / (vb - va);
                    if (ta > tb) {
                      double tt = ta;
                      ta = tb;
                      tb = tt;
                    }
                    if (ta > ct0) ct0 = ta;
                    if (tb < ct1) ct1 = tb;
                  }
                }
                if (ct0 < ct1) {
                  seg_ok = true;
                  for (int e2 = 0; e2 < 2; e2++) {
                    double tt = e2 ? ct1 : ct0;
                    r3d_v3 P = v3((float)((pav[0] + (pbv[0] - pav[0]) * tt) / md3),
                                  (float)((pav[1] + (pbv[1] - pav[1]) * tt) / md3),
                                  (float)((pav[2] + (pbv[2] - pav[2]) * tt) / md3));
                    r3d_v3 d = v3_sub(P, cam.pos);
                    float zf = v3_dot(d, c_f);
                    if (zf < 1e-5f) {
                      seg_ok = false;
                      break;
                    }
                    float nx = v3_dot(d, rn) / (zf * (lr > 0 ? lr : 1));
                    float ny = v3_dot(d, un) / (zf * (lu > 0 ? lu : 1));
                    ImVec2 sp = {(float)mv[i].px + (nx * 0.5f + 0.5f) * (float)mv[i].pw,
                                 (float)mv[i].py + (0.5f - ny * 0.5f) * (float)mv[i].ph};
                    if (e2) s1 = sp;
                    else s0 = sp;
                  }
                }
              }
              if (seg_ok) {
                ImDrawList_AddLine(draw, s0, s1, 0xff000000u, 3.5f);
                ImDrawList_AddLine(draw, s0, s1, 0xffff00ffu, 1.8f);
              }
            }
            ImDrawList_PopClipRect(draw);
            continue;
          }
          ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
          ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
          ImDrawList_PushClipRect(draw, cmin, cmax, false);
          /* only what actually intersects this pane's slab draws: clip each
           * curve segment to fs in [slice, slice+thickness] along the pane
           * normal, and markers to points inside it */
          double lo = mv[i].slice, hi = lo + (double)mv_thick;
          double pu = 0.0, pv = 0.0, pf = 0.0;
          for (size_t k = 0; k < umbilicus.count; k++) {
            double P[3] = {umbilicus.points[k].x, umbilicus.points[k].y,
                           umbilicus.points[k].z};
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], P, &fu, &fv, &fs);
            if (k) { /* parametric slab clip of the segment prev -> cur */
              double ct0 = 0.0, ct1 = 1.0;
              if (pf == fs) {
                if (pf < lo || pf > hi) ct0 = 1.0;
              } else {
                double ta = (lo - pf) / (fs - pf), tb = (hi - pf) / (fs - pf);
                if (ta > tb) {
                  double tt = ta;
                  ta = tb;
                  tb = tt;
                }
                if (ta > ct0) ct0 = ta;
                if (tb < ct1) ct1 = tb;
              }
              if (ct0 < ct1) {
                float ax, ay, bx, by;
                r3d_mv_project(&mv[i], pu + (fu - pu) * ct0, pv + (fv - pv) * ct0, &ax, &ay);
                r3d_mv_project(&mv[i], pu + (fu - pu) * ct1, pv + (fv - pv) * ct1, &bx, &by);
                ImDrawList_AddLine(draw, (ImVec2){ax, ay}, (ImVec2){bx, by}, uc, 1.5f);
              }
            }
            if (fs >= lo && fs <= hi) {
              float sx_, sy_;
              r3d_mv_project(&mv[i], fu, fv, &sx_, &sy_);
              ImDrawList_AddCircleFilled(draw, (ImVec2){sx_, sy_}, 3.5f, ub, 12);
              ImDrawList_AddCircleFilled(draw, (ImVec2){sx_, sy_}, 2.5f, uc, 12);
            }
            pu = fu;
            pv = fv;
            pf = fs;
          }
          if (i == R3D_MV_XY && !mv_aligned) {
            double ux, uy;
            if (umb_interp(&umbilicus, mv[i].slice, &ux, &uy)) {
              double fu, fv, fs, P[3] = {ux, uy, mv[i].slice};
              r3d_mv_w2b(mv_pb[i], mv_po[i], P, &fu, &fv, &fs);
              float sx_, sy_;
              r3d_mv_project(&mv[i], fu, fv, &sx_, &sy_);
              ImDrawList_AddCircle(draw, (ImVec2){sx_, sy_}, 9.0f, ub, 24, 4.0f);
              ImDrawList_AddCircle(draw, (ImVec2){sx_, sy_}, 9.0f, uc, 24, 2.0f);
            }
          }
          ImDrawList_PopClipRect(draw);
        }
      }
      if (GT->active && GT->done && getenv("R3D_TRACE_SAVE") && !mv_tr_nsaved) {
        /* TEMP verify: save the finished trace headlessly */
        mv_tr_nsaved = 1;
        r3d_tracer_stop(&GT->tr);
        errno = 0;
        if (mkdir_p("cache/traced/trace-test") &&
            r3d_tracer_save(&GT->tr, "cache/traced/trace-test", mv_tr_thresh, true) == 0)
          printf("tracer: saved cache/traced/trace-test\n");
        else
          fprintf(stderr, "tracer: SAVE FAILED for cache/traced/trace-test (%s)\n",
                  errno ? strerror(errno) : "write error");
      }
      if (GT->active && GT->pos) { /* growing trace: orange points */
        const ImU32 tc_ = 0xff2896ffu, tb_ = 0xff000000u;
        uint32_t TW = GT->tr.W, TH = GT->tr.H;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i))) continue;
          ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
          ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
          ImDrawList_PushClipRect(draw, cmin, cmax, false);
          if (MV_IS3D(i)) {
            if (mv_crop_d[0] > 0.0f) {
              uint32_t md3 = brick_shape[0];
              for (int a = 1; a < 3; a++)
                if (brick_shape[a] > md3) md3 = brick_shape[a];
              float mde = mv_crop_d[0] > mv_crop_d[1] ? mv_crop_d[0] : mv_crop_d[1];
              if (mv_crop_d[2] > mde) mde = mv_crop_d[2];
              r3d_camera_orbit_set(&cam,
                                   v3((float)(mv_crop_c[0] / md3),
                                      (float)(mv_crop_c[1] / md3),
                                      (float)(mv_crop_c[2] / md3)),
                                   (float)((double)mde / md3 * (double)mv_crop_fit));
              r3d_v3 c_r, c_u, c_f;
              r3d_camera_basis(&cam, (float)mv[i].pw / (float)(mv[i].ph ? mv[i].ph : 1),
                               &c_r, &c_u, &c_f);
              float lr = v3_len(c_r), lu = v3_len(c_u);
              r3d_v3 rn = v3_scale(c_r, lr > 0 ? 1.0f / lr : 0.0f);
              r3d_v3 un = v3_scale(c_u, lu > 0 ? 1.0f / lu : 0.0f);
              uint32_t strd = TW > 200 ? TW / 100 : 1;
              for (uint32_t j = 0; j < TH; j += strd)
                for (uint32_t ii = 0; ii < TW; ii += strd) {
                  size_t k = (size_t)j * TW + ii;
                  if (GT->st[k] != R3D_TR_SET) continue;
                  const double *P = GT->pos + k * 3;
                  bool inb = true;
                  for (int a = 0; a < 3; a++)
                    if (P[a] < mv_crop_c[a] - (double)mv_crop_d[a] * 0.5 ||
                        P[a] > mv_crop_c[a] + (double)mv_crop_d[a] * 0.5)
                      inb = false;
                  if (!inb) continue;
                  r3d_v3 Pb = v3((float)(P[0] / md3), (float)(P[1] / md3),
                                 (float)(P[2] / md3));
                  r3d_v3 d = v3_sub(Pb, cam.pos);
                  float zf = v3_dot(d, c_f);
                  if (zf < 1e-5f) continue;
                  float nx = v3_dot(d, rn) / (zf * (lr > 0 ? lr : 1));
                  float ny = v3_dot(d, un) / (zf * (lu > 0 ? lu : 1));
                  ImVec2 sp = {(float)mv[i].px + (nx * 0.5f + 0.5f) * (float)mv[i].pw,
                               (float)mv[i].py + (0.5f - ny * 0.5f) * (float)mv[i].ph};
                  ImDrawList_AddCircleFilled(
                      draw, sp, 2.0f,
                      GT->cf[k] < mv_tr_thresh ? 0x907070e0u : tc_, 6);
                }
            }
          } else {
            double lo = mv[i].slice - 1.0, hi = mv[i].slice + (double)mv_thick + 1.0;
            for (uint32_t j = 0; j < TH; j++)
              for (uint32_t ii = 0; ii < TW; ii++) {
                size_t k = (size_t)j * TW + ii;
                if (GT->st[k] != R3D_TR_SET) continue;
                const double *P = GT->pos + k * 3;
                double fu, fv, fs;
                r3d_mv_w2b(mv_pb[i], mv_po[i], P, &fu, &fv, &fs);
                if (fs < lo || fs > hi) continue;
                float sx_, sy_;
                r3d_mv_project(&mv[i], fu, fv, &sx_, &sy_);
                float cf2 = GT->cf[k];
                bool weak = cf2 < mv_tr_thresh;
                ImU32 pc_ = weak ? 0x907070e0u /* weak: translucent red */
                                 : tc_;
                ImDrawList_AddCircleFilled(draw, (ImVec2){sx_, sy_}, 2.6f, tb_, 6);
                ImDrawList_AddCircleFilled(draw, (ImVec2){sx_, sy_}, 1.8f, pc_, 6);
              }
          }
          ImDrawList_PopClipRect(draw);
        }
      }
      if (mv_mask & 1u) { /* flattened view: mark the surface grid point
                           * anchoring the focus (nearest valid point) */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        float fx_, fy_;
        r3d_mv_project(sv, (double)mv_align_ij[0], (double)mv_align_ij[1], &fx_, &fy_);
        if (fx_ >= (float)sv->px && fx_ < (float)(sv->px + sv->pw) && fy_ >= (float)sv->py &&
            fy_ < (float)(sv->py + sv->ph))
          ImDrawList_AddCircle(draw, (ImVec2){fx_, fy_}, 10.0f, fc, 24, 2.0f);
      }
    }

    mt_t[3] = r3d_now_ns();
    r3d_frame_params p = {
        .cam_origin = {cam.pos.x, cam.pos.y, cam.pos.z},
        .cam_right = {right.x, right.y, right.z},
        .cam_up = {up.x, up.y, up.z},
        .cam_forward = {fwd.x, fwd.y, fwd.z},
        .step_voxels = step_voxels,
        .density = density,
        .lod_bias = lod_bias,
        .max_mip = 10.0f,
        .viewport = {rvw, rvh},
        .mode = mode,
        .frame_index = frame_index++,
        .threshold = low_cut / 255.0f,
        .skip_gate = fmaxf(low_cut, tf_min_v - 0.5f) / 255.0f,
        .overlay_gain = overlay_gain,
        /* live 2.5D ink shows even without a 3D overlay tree. The palette
         * follows the source: bit 8 = surface-prediction cyan for the plane
         * views' overlay tree, bit 1 = the flat pane carries live ink and
         * tints green regardless of that tree's palette */
        .overlay_flags =
            ((overlay_path && overlay_show) || (inklive_have && inklive_show)
                 ? 1u
                 : 0u) |
            (inklive_have && inklive_show ? 2u : 0u) |
            (ink3d_ok && ink3d_show ? 4u : 0u) |
            (g_lbl_init && g_lbl_show ? 8u : 0u) |
            (g_reg_open && g_reg_show
                 ? (16u | ((uint32_t)(g_reg_alpha * 255.0f + 0.5f) << 16))
                 : 0u) |
            (overlay_path && !strstr(overlay_path, "ink") ? 256u : 0u),
    };
    memcpy(p.vol_r0, &vm.r0, 12);
    memcpy(p.vol_r1, &vm.r1, 12);
    memcpy(p.vol_r2, &vm.r2, 12);
    p.vol_tx = vol_t.x;
    p.vol_ty = vol_t.y;
    p.vol_tz = vol_t.z;
    if (slab_wz) {
      r3d_slab_params(renderer, &p);
      p.slab_depth = (uint32_t)slab_depth;
    }
    if (vslab_mode) {
      /* focus = view axis ^ window mid-plane, in WINDOW space -> world */
      float vey = (float)vsh / (float)vsw, vez = (float)(vsd + 2) / (float)vsw;
      r3d_v3 vc = v3(0.5f, vey * 0.5f, vez * 0.5f);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float tt = vd.z != 0.0f ? (vez * 0.5f - vo.z) / vd.z : 0.0f;
      if (tt < 0.0f) tt = 0.0f;
      float fxn = fclampf(vo.x + vd.x * tt, 0.0f, 1.0f);
      float fyn = fclampf((vo.y + vd.y * tt) / (vey > 0.0f ? vey : 1.0f), 0.0f, 1.0f);
      int64_t vo3[3];
      r3d_vslab_get(renderer, vo3, NULL);
      if (vs_follow && vo3[0] >= 0) {
        vs_fx = (double)vo3[0] + (double)fxn * vsw;
        vs_fy = (double)vo3[1] + (double)fyn * vsh;
      }
      r3d_vslab_frame(renderer, vs_fx, vs_fy, vs_z0, moving ? 1u : 3u, &p);
      if (umbilicus_path) {
        int64_t co[3];
        r3d_vslab_get(renderer, co, NULL);
        uint32_t cpending = r3d_vslab_resident_pending(renderer);
        if (annotation_z > annotation_last_z) annotation_dir = 1;
        else if (annotation_z < annotation_last_z) annotation_dir = -1;
        annotation_last_z = annotation_z;
        if (cpending == 0 && co[2] == vs_z0) {
          int64_t targets[R3D_VSLAB_PREFETCH_MAX];
          uint32_t ntargets = 0;
          for (int k = 1; k <= annotation_prefetch; k++) {
            int64_t az = (int64_t)annotation_z +
                         (int64_t)annotation_dir * annotation_step * k;
            if (az < 0) az = 0;
            if (az >= vsnz) az = (int64_t)vsnz - 1;
            int64_t pz = annotation_z0((int)az, vsnz, (uint32_t)vsd);
            if (pz != vs_z0) targets[ntargets++] = pz;
          }
          int64_t back = (int64_t)annotation_z -
                         (int64_t)annotation_dir * annotation_step;
          if (back < 0) back = 0;
          if (back >= vsnz) back = (int64_t)vsnz - 1;
          int64_t back_z0 = annotation_z0((int)back, vsnz, (uint32_t)vsd);
          if (back_z0 != vs_z0 && ntargets < R3D_VSLAB_PREFETCH_MAX)
            targets[ntargets++] = back_z0;
          r3d_vslab_prefetch(renderer, targets, ntargets);
        } else {
          r3d_vslab_prefetch(renderer, NULL, 0);
        }
      }
      /* residency-lag metric: cells short of full residency, second half of
       * the run only (the first half absorbs the initial window fill) */
      if (bench && frame_index > warmup_frames &&
          (frame_index - warmup_frames) * 2 >= exit_frames) {
        int64_t bo_[3];
        uint32_t pd = 0;
        r3d_vslab_get(renderer, bo_, &pd);
        vs_pend_acc += pd;
      }
    }
    if (bricks_path) {
      p.slab_z0 = (float)brick_z;
      p.slab_depth = (uint32_t)brick_depth;
      /* streaming pump: camera in VOLUME space (model transform inverted, like
       * the clip focus); smaller decode budget while moving so the pump's GPU
       * time shares the frame with half-res rendering */
      float bext[3];
      r3d_bricks_extent(renderer, bext);
      r3d_v3 vc = v3(bext[0] * 0.5f, bext[1] * 0.5f, bext[2] * 0.5f);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float be[3] = {vo.x, vo.y, vo.z}, bf[3] = {vd.x, vd.y, vd.z};
      float asp = (float)w / (float)h;
      float ht = tanf(cam.fov_y * 0.5f) * sqrtf(1.0f + asp * asp);
      float pixel_cone = 2.0f * tanf(cam.fov_y * 0.5f) / (float)(rvh ? rvh : 1) * exp2f(lod_bias);
      if (multiview_path) {
        /* per-view AABB collects: each plane view wants its visible rect at
         * its own magnification, +- the slab thickness along the normal */
        uint32_t mdim = brick_shape[0];
        for (int a = 1; a < 3; a++)
          if (brick_shape[a] > mdim) mdim = brick_shape[a];
        if (r3d_bricks_stream_begin(renderer)) {
          if (mv_mask & 1u) { /* segment view: walk the visible grid rect and
             * request the bricks its surface points (+ normal offset) touch */
            const r3d_mview *sv = &mv[R3D_MV_SEG];
            double hw = (double)sv->pw * 0.5 / sv->zoom, hh = (double)sv->ph * 0.5 / sv->zoom;
            int64_t g0 = (int64_t)(sv->cu - hw), g1 = (int64_t)(sv->cu + hw) + 1;
            int64_t j0 = (int64_t)(sv->cv - hh), j1 = (int64_t)(sv->cv + hh) + 1;
            if (g0 < 0) g0 = 0;
            if (j0 < 0) j0 = 0;
            if (g1 > (int64_t)mv_seg.w) g1 = mv_seg.w;
            if (j1 > (int64_t)mv_seg.h) j1 = mv_seg.h;
            double span = (double)((g1 - g0) * (j1 - j0));
            int64_t step = span > 0 ? (int64_t)(sqrt(span / 384.0) + 1.0) : 1;
            /* voxels per pixel: (grid units per px) / (grid units per voxel) */
            float vf = (float)(1.0 / (sv->zoom * (double)mv_seg.sx)) * exp2f(lod_bias);
            uint32_t lvl = 0;
            while (vf >= 2.0f && lvl < 7u) {
              vf *= 0.5f;
              lvl++;
            }
            for (int64_t gj = j0; gj < j1; gj += step)
              for (int64_t gi = g0; gi < g1; gi += step) {
                const float *sp = r3d_tifxyz_at(&mv_seg, (uint32_t)gi, (uint32_t)gj);
                if (!r3d_tifxyz_valid(sp)) continue;
                float pp[3] = {sp[0] / (float)mdim, sp[1] / (float)mdim,
                               sp[2] / (float)mdim};
                r3d_bricks_stream_point(renderer, pp, lvl, p.skip_gate);
              }
          }
          for (int i = 1; i < 4; i++) {
            if (!(mv_mask & (1u << i))) continue; /* collapsed: no streaming */
            if (MV_IS3D(i)) { /* volumetric pane: stream exactly the crop
               * cube; LOD = crop edge over pane pixels (the zoom IS the
               * crop, so the request stays bounded at every zoom) */
              if (mv_crop_d[0] <= 0.0f) continue; /* first frame inits it */
              float lo3[3], hi3[3];
              for (int a = 0; a < 3; a++) { /* small refit margin per axis */
                double half = (double)mv_crop_d[a] * 0.55;
                lo3[a] = (float)((mv_crop_c[a] - half) / mdim);
                hi3[a] = (float)((mv_crop_c[a] + half) / mdim);
              }
              int pmin = mv[i].pw < mv[i].ph ? mv[i].pw : mv[i].ph;
              /* stream_box wants BOX units per pixel (like the plane views'
               * 1/(zoom*mdim)), so voxels-per-pixel divides by mdim */
              float mde = mv_crop_d[0] > mv_crop_d[1] ? mv_crop_d[0] : mv_crop_d[1];
              if (mv_crop_d[2] > mde) mde = mv_crop_d[2];
              float vpp3 = (float)((double)mde / (double)(pmin > 0 ? pmin : 1) / mdim) *
                           exp2f(lod_bias);
              r3d_bricks_stream_box(renderer, lo3, hi3, vpp3, p.skip_gate);
              continue;
            }
            double hw = (double)mv[i].pw * 0.5 / mv[i].zoom;
            double hh = (double)mv[i].ph * 0.5 / mv[i].zoom;
            double th = (double)mv_thick;
            /* world AABB of the visible rect + slab: frame center at the
             * slab midpoint, per-axis extents from the |basis| components */
            double ctr[3];
            r3d_mv_b2w(mv_pb[i], mv_po[i], mv[i].cu, mv[i].cv, mv[i].slice + th * 0.5,
                       ctr);
            float lo[3], hi[3];
            for (int a = 0; a < 3; a++) {
              double he = hw * fabs(mv_pb[i][0][a]) + hh * fabs(mv_pb[i][1][a]) +
                          (th * 0.5 + 1.0) * fabs(mv_pb[i][2][a]);
              lo[a] = (float)((ctr[a] - he) / (double)mdim);
              hi[a] = (float)((ctr[a] + he) / (double)mdim);
            }
            float vpp = (float)(1.0 / (mv[i].zoom * (double)mdim)) * exp2f(lod_bias);
            r3d_bricks_stream_box(renderer, lo, hi, vpp, p.skip_gate);
          }
          r3d_bricks_stream_submit(renderer, moving ? 3u : 8u);
        }
      } else {
        r3d_bricks_stream(renderer, be, bf, ht, pixel_cone, (uint32_t)brick_z,
                          (uint32_t)brick_depth, p.skip_gate, moving ? 2u : 6u);
      }
      if (g_lbl_init) /* mirror paint edits / slot churn into the label atlas */
        r3d_bricks_labels_sync(renderer, 8u);
      if (g_smask && g_smask_gpu_dirty && (frame_index & 7u) == 0) {
        /* throttled: strokes re-upload the whole mask image at ~1/8 frames */
        if (r3d_surfmask(renderer, g_smask, g_smask_w, g_smask_h) == 0)
          g_smask_gpu_dirty = false;
      }
      if (g_reg_busy) { /* registration worker: collect NCC / refined map */
        bool rok = true;
        if (r3d_regvol_job_poll(&g_reg, &rok) == 0) {
          g_reg_busy = false;
          if (rok) {
            g_reg_ncc0 = g_reg.ncc0;
            g_reg_ncc1 = g_reg.ncc1;
            if (g_reg.job_mode != 0) reg_gui_reset_mirrors(); /* deltas folded */
          } else
            printf("regvol: job failed\n");
        }
      }
      r3d_surfvol_regtap(renderer, g_reg_open && g_reg_flat);
      r3d_bricks_regatlas_sync(renderer, 16u);
      { /* SLIM flatten completion: save + add to the store on this thread */
        int fst2 = atomic_load(&g_flat_state);
        if (fst2 == 2 || fst2 == 3) {
          if (g_flat_th_up) {
            pthread_join(g_flat_th, NULL);
            g_flat_th_up = false;
          }
          if (fst2 == 2 && g_flat_out) {
            char fd[560];
            snprintf(fd, sizeof fd, "cache/traced/%s", g_flat_name);
            if (flat_save(fd, g_flat_out, g_flat_out_w, g_flat_out_h, g_flat_step) == 0) {
              printf("flatten: %s saved (%ux%u, stretch %.3f -> %.4f, %u iters)\n",
                     fd, g_flat_out_w, g_flat_out_h, g_flat_stats.stretch0,
                     g_flat_stats.stretch1, g_flat_stats.iters);
              if (seg_store_path) { /* into the corpus, like a harvest */
                const char *fdirs[1] = {fd};
                if (r3d_segstore_build(seg_store_path, fdirs, 1, 2, false) > 0) {
                  for (int oi2 = 1; oi2 < 4; oi2++) {
                    if (!sgc_ln[oi2]) continue;
                    for (uint32_t si = 0; si < sgc.st.n; si++) {
                      free(sgc_ln[oi2][si].l.w);
                      free(sgc_ln[oi2][si].l.g);
                    }
                    free(sgc_ln[oi2]);
                    sgc_ln[oi2] = NULL;
                  }
                  sgc_close(&sgc);
                  if (sgc_open(&sgc, seg_store_path, (size_t)1536 << 20) == 0) {
                    for (int oi2 = 1; oi2 < 4; oi2++)
                      sgc_ln[oi2] = calloc(sgc.st.n ? sgc.st.n : 1, sizeof *sgc_ln[oi2]);
                    for (int oi2 = 0; oi2 < 4; oi2++) {
                      sgc_key_slice[oi2] = 1e30;
                      sgc_key_gen[oi2] = UINT32_MAX;
                    }
                    memcpy(sgc_near_focus, (double[3]){1e30, 1e30, 1e30},
                           sizeof sgc_near_focus);
                    printf("flatten: %s added to the store (%u surfaces)\n",
                           g_flat_name, sgc.st.n);
                  }
                }
              }
            } else
              printf("flatten: SAVE FAILED for %s - result discarded\n", fd);
          } else if (fst2 == 3)
            printf("flatten: job failed (degenerate segment?)\n");
          free(g_flat_out);
          g_flat_out = NULL;
          atomic_store(&g_flat_state, 0);
        }
      }
      r3d_bricks_params(renderer, &p);
    }
    if (clip_mode) {
      /* focus = where the view axis crosses the slab plane, computed in
       * VOLUME space so it tracks the model transform */
      float ezc = (float)CLIP_DEPTH_MAX / (float)CLIP_NX * 0.5f;
      r3d_v3 vc = v3(0.5f, 0.5f, ezc);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float tt = vd.z != 0.0f ? (ezc - vo.z) / vd.z : 0.0f;
      if (tt < 0.0f) tt = 0.0f;
      double fx = (double)(vo.x + vd.x * tt) * CLIP_NX;
      double fy = (double)(vo.y + vd.y * tt) * CLIP_NX;
      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx > CLIP_NX) fx = CLIP_NX;
      if (fy > CLIP_NX) fy = CLIP_NX;
      p.slab_depth = (uint32_t)clip_depth;
      if (r3d_clip_frame(renderer, fx, fy, clip_z0, &p) != 0) running = false;
      if (p.clip_valid != clip_valid_disp)
        printf("clip: valid=0x%02x z0=%llu focus=(%.0f,%.0f)\n", p.clip_valid,
               (unsigned long long)clip_z0, fx, fy);
      clip_valid_disp = p.clip_valid;
    }
    mt_t[4] = r3d_now_ns();
    r3d_frame_stats st = {0};
    int frc;
    if (multiview_path) {
      /* one FrameParams per quadrant: axis-aligned ortho cameras over the
       * bricks virtual volume, slab-clipped to each view's slice */
      uint32_t mdim = brick_shape[0];
      for (int a = 1; a < 3; a++)
        if (brick_shape[a] > mdim) mdim = brick_shape[a];
      r3d_frame_params vp4[4];
      uint32_t nvp = 0;
      /* the flattened bake samples the 3D overlay atlas only for ink
       * trees; surface predictions never dress the flattened segment */
      r3d_surfvol_overlay_enable(renderer,
                                 (g_reg_open && g_reg_flat) || ink3d_ok ||
                                     (overlay_path && strstr(overlay_path, "ink")));
      for (int i = 0; i < 4; i++) {
        if (!(mv_mask & (1u << i))) continue; /* collapsed: no dispatch at all */
        r3d_frame_params q = p;
        q.viewport[0] = (uint32_t)mv[i].pw;
        q.viewport[1] = (uint32_t)mv[i].ph;
        q.view_org = (uint32_t)mv[i].px | ((uint32_t)mv[i].py << 16);
        { /* strict per-source routing: SURFACE PREDICTIONS (overlay panel,
           * blue) show only on the plane views; 2.5D INK (mint) only on the
           * flattened segment; 3D INK (red, second atlas slot) on both. */
          bool tree_ink = overlay_path && strstr(overlay_path, "ink");
          q.overlay_flags &= ~(1u | 2u | 4u | 8u | 16u | 32u);
          if (i == R3D_MV_SEG) {
            bool ink25 = inklive_have && inklive_show;
            bool baked3d = (ink3d_ok && ink3d_show && (mv_ov_mask & (1u << i))) ||
                           (tree_ink && overlay_show && (mv_ov_mask & (1u << i)));
            if (g_smask && g_smask_show)
              q.overlay_flags |= 32u; /* supervision-mask tint */
            if (g_reg_open && g_reg_show && g_reg_flat) {
              /* the bake's G channel carries the moving scan: magenta tint */
              q.overlay_flags |= 1u | 16u;
            } else {
              if (ink25 || baked3d) q.overlay_flags |= 1u; /* G channel shows */
              if (ink25) q.overlay_flags |= 2u; /* ink-mint palette */
              else if (ink3d_ok && ink3d_show) q.overlay_flags |= 4u; /* red */
            }
          } else {
            if (g_reg_open && g_reg_show)
              q.overlay_flags |= 16u; /* second scan: green/magenta fuse */
            if (overlay_path && overlay_show && (mv_ov_mask & (1u << i)))
              q.overlay_flags |= 1u; /* tree overlay (surface preds or legacy
                                      * ink tree), palette by tree type */
            if (ink3d_ok && ink3d_show && (mv_ov_mask & (1u << i)))
              q.overlay_flags |= 4u; /* 3D ink rides on top in red */
            if (g_lbl_init && g_lbl_show)
              q.overlay_flags |= 8u; /* paint labels: plane + 3D panes only */
          }
        }
        if (i == R3D_MV_SEG) {
          /* flattened segment: raycast the surface-volume window. Camera in
           * FLATTENED VOXELS (grid / scale); window mapping from the backend. */
          q.view_flags = R3D_VIEW_SURF | (mv_stretch ? R3D_VIEW_STRETCH : 0u);
          q.cam_origin[0] = (float)(mv[i].cu / (double)mv_seg.sx);
          q.cam_origin[1] = (float)(mv[i].cv / (double)mv_seg.sy);
          q.cam_origin[2] = 0.0f;
          memset(q.cam_right, 0, sizeof q.cam_right);
          memset(q.cam_up, 0, sizeof q.cam_up);
          memset(q.cam_forward, 0, sizeof q.cam_forward);
          q.cam_right[0] = (float)((double)mv[i].pw * 0.5 / mv[i].zoom / (double)mv_seg.sx);
          q.cam_up[1] = (float)-((double)mv[i].ph * 0.5 / mv[i].zoom / (double)mv_seg.sy);
          r3d_surfvol_params(renderer, &q);
          q.slab_z0 = (float)mv[i].slice; /* render-time normal offset (voxels) */
          q.slab_depth = (uint32_t)mv_thick; /* marched thickness (voxels) */
          vp4[nvp++] = q;
          continue;
        }
        if (MV_IS3D(i)) { /* zoom-crop volumetric view */
          if (!mv_vol_cam_init) {
            cam.pitch = getenv("R3D_3D_PITCH") ? strtof(getenv("R3D_3D_PITCH"), NULL)
                                               : 1.25f; /* side-on default */
            for (int a = 0; a < 3; a++) mv_crop_c[a] = (double)brick_shape[a] * 0.5;
            mv_vol_cam_init = true;
          }
          if (mv_crop_d[0] <= 0.0f)
            for (int a = 0; a < 3; a++) /* start: the whole scroll, coarse */
              mv_crop_d[a] = (float)brick_shape[a];
          if (getenv("R3D_3D_CROP")) { /* "e" or "x,y,z" */
            const char *cs = getenv("R3D_3D_CROP");
            char *ce = NULL;
            for (int a = 0; a < 3; a++) {
              float cv2 = strtof(cs, &ce);
              if (ce == cs) break;
              mv_crop_d[a] = cv2;
              if (*ce != ',') {
                if (a == 0)
                  for (int b2 = 1; b2 < 3; b2++) mv_crop_d[b2] = cv2;
                break;
              }
              cs = ce + 1;
            }
          }
          /* the camera frames the box from a fixed relative distance */
          float mde = mv_crop_d[0] > mv_crop_d[1] ? mv_crop_d[0] : mv_crop_d[1];
          if (mv_crop_d[2] > mde) mde = mv_crop_d[2];
          double eb = (double)mde / mdim; /* largest extent, box units */
          r3d_camera_orbit_set(&cam,
                               v3((float)(mv_crop_c[0] / mdim), (float)(mv_crop_c[1] / mdim),
                                  (float)(mv_crop_c[2] / mdim)),
                               (float)(eb * (double)mv_crop_fit));
          r3d_v3 vr, vu, vf;
          r3d_camera_basis(&cam, (float)mv[i].pw / (float)(mv[i].ph ? mv[i].ph : 1), &vr,
                           &vu, &vf);
          q.cam_origin[0] = cam.pos.x;
          q.cam_origin[1] = cam.pos.y;
          q.cam_origin[2] = cam.pos.z;
          memcpy(q.cam_right, &vr, 12);
          memcpy(q.cam_up, &vu, 12);
          memcpy(q.cam_forward, &vf, 12);
          q.view_flags = R3D_VIEW_CROP; /* perspective, clipped to the cube */
          q.slab_x0 = (float)(mv_crop_c[0] - (double)mv_crop_d[0] * 0.5);
          q.slab_y0 = (float)(mv_crop_c[1] - (double)mv_crop_d[1] * 0.5);
          q.slab_z0 = (float)(mv_crop_c[2] - (double)mv_crop_d[2] * 0.5);
          q.slab_px = mv_crop_d[0]; /* per-axis extents for the box clip */
          q.slab_py = mv_crop_d[1];
          q.slab_nx = mv_crop_d[2];
          q.slab_depth = (uint32_t)mde;
          if (getenv("R3D_3D_MODE")) q.mode = (uint32_t)atoi(getenv("R3D_3D_MODE"));
          if (getenv("R3D_3D_BIAS")) q.lod_bias = strtof(getenv("R3D_3D_BIAS"), NULL);
          vp4[nvp++] = q;
          continue;
        }
        /* ortho camera from the view frame: origin 2 voxels before the slab
         * near face along the plane normal, half-extent vectors along u/v;
         * the slab clip rides the ray (oblique), which for axis frames is
         * identical to the old per-axis box clip */
        q.view_flags = R3D_VIEW_ORTHO | R3D_VIEW_OBLIQUE;
        double hw = (double)mv[i].pw * 0.5 / mv[i].zoom;
        double hh = (double)mv[i].ph * 0.5 / mv[i].zoom;
        double og[3];
        r3d_mv_b2w(mv_pb[i], mv_po[i], mv[i].cu, mv[i].cv, mv[i].slice - 2.0, og);
        for (int a = 0; a < 3; a++) {
          q.cam_origin[a] = (float)(og[a] / (double)mdim);
          q.cam_right[a] = (float)(mv_pb[i][0][a] * hw / (double)mdim); /* +x -> +u */
          q.cam_up[a] = (float)(-mv_pb[i][1][a] * hh / (double)mdim);   /* ndc +y -> -v */
          q.cam_forward[a] = (float)mv_pb[i][2][a];
        }
        q.slab_z0 = 2.0f; /* voxels from the ray origin to the slab near face */
        q.slab_depth = (uint32_t)mv_thick;
        vp4[nvp++] = q;
      }
      if (mv_half)
        for (uint32_t k = 0; k < nvp; k++) vp4[k].view_flags |= R3D_VIEW_HALF;
      frc = r3d_frame_views(renderer, vp4, nvp, &st);
    } else {
      frc = r3d_frame(renderer, &p, &st);
    }
    if (frc < 0) { /* device lost / unrecoverable frame: don't spin forever
                    * (Dozen can drop the device under heavy load, and every
                    * later frame then fails) */
      if (++frame_fail >= 4) {
        fprintf(stderr, "render3d: renderer failed %u frames in a row "
                        "(device lost?); exiting\n", frame_fail);
        break;
      }
    } else {
      frame_fail = 0;
    }
    bool measuring = !exit_frames || frame_index > warmup_frames;
    if (frc == 0 && measuring) {
      last_gpu_ns = st.gpu_ns;
      const uint64_t *sv = (const uint64_t *)&st;
      uint64_t *pv = (uint64_t *)&prof, *qv = (uint64_t *)&prof_sum;
      for (size_t k = 0; k < sizeof st / sizeof(uint64_t); k++) {
        pv[k] = (uint64_t)((double)pv[k] * 0.95 + (double)sv[k] * 0.05);
        qv[k] += sv[k];
      }
      if (prof_samples && prof_frames < exit_frames) prof_samples[prof_frames] = st;
      prof_frames++;
    }
    mt_t[5] = r3d_now_ns();
    for (int ph = 0; ph < MT_N; ph++) {
      uint64_t d = mt_t[ph + 1] - mt_t[ph];
      mt_sum[ph] += d;
      if (d > mt_max[ph]) mt_max[ph] = d;
      mt_ema[ph] = mt_ema[ph] * 0.95 + (double)d * 0.05;
    }
    mt_frames++;
    if (frc < 0) {
      fprintf(stderr, "r3d_frame failed\n");
      running = false;
    }
    if (in.screenshot) take_screenshot(renderer, stats.frame_index);
    if (total_frames && frame_index >= total_frames) {
      if (shot_path) {
        uint32_t sw = 0, sh = 0;
        r3d_read_frame(renderer, NULL, &sw, &sh);
        uint8_t *rgba = malloc((size_t)sw * sh * 4);
        if (rgba && r3d_read_frame(renderer, rgba, &sw, &sh) == 0)
          r3d_screenshot_ppm(shot_path, rgba, sw, sh);
        free(rgba);
      }
      running = false;
    }

    if (measuring) {
      r3d_stats_push(&stats, r3d_now_ns() - t0, st.gpu_ns);
      r3d_stats_report(&stats);
    }
  }

  r3d_stats_report_now(&stats);
  if (slab_src.voxels) r3d_volume_close(&slab_src);
  if (mt_frames > 2) {
    printf("mainthread avg:");
    for (int ph = 0; ph < MT_N; ph++)
      printf(" %s %.2f", mt_name[ph], (double)mt_sum[ph] / (double)mt_frames / 1e6);
    printf(" ms | max:");
    for (int ph = 0; ph < MT_N; ph++) printf(" %s %.1f", mt_name[ph], (double)mt_max[ph] / 1e6);
    printf(" ms\n");
  }
  if (prof_frames > 2) {
    /* skip warmup skew: averages include first frames with empty queries */
    double n = (double)prof_frames;
    printf("profile avg: gpu %.2f (raycast %.2f blit %.2f gui %.2f) | "
           "wait %.2f acquire %.2f record %.2f submit %.2f ms\n",
           (double)prof_sum.gpu_ns / n / 1e6, (double)prof_sum.gpu_raycast_ns / n / 1e6,
           (double)prof_sum.gpu_blit_ns / n / 1e6, (double)prof_sum.gpu_gui_ns / n / 1e6,
           (double)prof_sum.cpu_wait_ns / n / 1e6, (double)prof_sum.cpu_acquire_ns / n / 1e6,
           (double)prof_sum.cpu_record_ns / n / 1e6, (double)prof_sum.cpu_submit_ns / n / 1e6);
  }
  if (vslab_mode && bench) {
    int64_t final_o[3];
    uint32_t final_visible = 0;
    r3d_vslab_get(renderer, final_o, &final_visible);
    printf("vslab bench: pending cell-frames %llu, final visible %u resident %u\n",
           (unsigned long long)vs_pend_acc, final_visible,
           r3d_vslab_resident_pending(renderer));
  }
  if (umbilicus_path) {
    r3d_vslab_prefetch_stats pcs;
    r3d_vslab_prefetch_get(renderer, &pcs);
    printf("vslab decoded cache: %u ready, %llu hits, %llu misses, last %.0f ms\n", pcs.ready,
           (unsigned long long)pcs.hits, (unsigned long long)pcs.misses, pcs.last_decode_ms);
  }
  r3d_bricks_stats final_bst = {0};
  if (bricks_path) {
    r3d_bricks_flush(renderer);
    r3d_bricks_get_stats(renderer, &final_bst);
    printf("bricks bench: decoded %llu in %llu jobs, %.2f ms/job, %u failure(s), hot %u/%u\n",
           (unsigned long long)final_bst.decoded, (unsigned long long)final_bst.jobs,
           final_bst.jobs ? (double)final_bst.stream_ns / (double)final_bst.jobs / 1e6 : 0.0,
           final_bst.failures, final_bst.hot, final_bst.hot_cap);
    if (final_bst.nlevels > 1)
      printf("bricks LOD wanted: [%u,%u,%u,%u,%u,%u,%u,%u]\n", final_bst.lod_wanted[0],
             final_bst.lod_wanted[1], final_bst.lod_wanted[2], final_bst.lod_wanted[3],
             final_bst.lod_wanted[4], final_bst.lod_wanted[5], final_bst.lod_wanted[6],
             final_bst.lod_wanted[7]);
    if (final_bst.nlevels > 1)
      printf("bricks LOD requests: [%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
             (unsigned long long)final_bst.lod_requests[0],
             (unsigned long long)final_bst.lod_requests[1],
             (unsigned long long)final_bst.lod_requests[2],
             (unsigned long long)final_bst.lod_requests[3],
             (unsigned long long)final_bst.lod_requests[4],
             (unsigned long long)final_bst.lod_requests[5],
             (unsigned long long)final_bst.lod_requests[6],
             (unsigned long long)final_bst.lod_requests[7]);
  }
  if (bench_json)
    write_bench_json(bench_json, bench_name ? bench_name : bench, win_w, win_h, quality_arg,
                     warmup_frames, &stats, prof_samples, prof_frames, vs_pend_acc, &final_bst);
  if (umbilicus_path && umbilicus.dirty &&
      save_umbilicus(&umbilicus, umbilicus_path, annotation_status) != 0)
    fprintf(stderr, "umbilicus: FINAL SAVE FAILED for %s (%s) - %zu point(s) unsaved\n",
            umbilicus_path, annotation_status, umbilicus.count);
  r3d_umbilicus_free(&umbilicus);
  if (inklive_up) r3d_inklive_stop(&inklive);
  free(inkmap);
  free(inkmap_acc);
  inkmap_acc = NULL;
  free(inkmap_wsum);
  inkmap_wsum = NULL;
  if (multiview_path) {
    r3d_tifxyz_free(&mv_seg);
    r3d_segrows_free(&mv_rows);
    free(mv_normals);
    umb_snap_clear(mv_umb_undo, &mv_umb_undo_n);
    umb_snap_clear(mv_umb_redo, &mv_umb_redo_n);
    if (sgc.open) {
      pthread_mutex_lock(&sgc.mu);
      uint32_t ready = 0;
      uint64_t lines = 0;
      for (uint32_t si = 0; si < sgc.st.n; si++)
        if (sgc.ent[si].state == SGC_READY) ready++;
      for (int i = 1; i < 4; i++)
        for (uint32_t si = 0; si < sgc.st.n; si++)
          if (sgc_ln[i][si].valid) lines += sgc_ln[i][si].l.n;
      printf("segments: %u/%u cached (%zu MB), hits XY %u XZ %u YZ %u, %llu overlay "
             "segments traced\n",
             ready, sgc.st.n, sgc.bytes >> 20, sgc_nhits[R3D_MV_XY], sgc_nhits[R3D_MV_XZ],
             sgc_nhits[R3D_MV_YZ], (unsigned long long)lines);
      pthread_mutex_unlock(&sgc.mu);
    }
    for (int i = 1; i < 4; i++) {
      if (!sgc_ln[i]) continue;
      for (uint32_t si = 0; si < sgc.st.n; si++) {
        free(sgc_ln[i][si].l.w);
        free(sgc_ln[i][si].l.g);
      }
      free(sgc_ln[i]);
      sgc_ln[i] = NULL;
    }
    for (int ti = 0; ti < GT_MAX; ti++) { /* stop every background trace */
      struct gtrace *g = &gts[ti];
      if (g->active || g->tr.pos) {
        r3d_tracer_stop(&g->tr);
        r3d_tracer_free(&g->tr);
      }
      free(g->pos);
      free(g->st);
      free(g->cf);
      memset(g, 0, sizeof *g);
    }
    mv_seeds_n = 0;
    if (bsurf_active) r3d_bsurf_free(&bsurf);
    bsurf_active = false;
    if (bsurf_ct_ok) r3d_cpuvol_close(&bsurf_ct);
    bsurf_ct_ok = false;
    sgc_close(&sgc);
    for (int i = 0; i < 4; i++) {
      free(mv_ol[i].w);
      free(mv_ol[i].g);
      free(mv_ol_off[i].w);
      free(mv_ol_off[i].g);
    }
  }
  free(prof_samples);
  if (g_lbl_init) { /* labels are per volume: save unsaved edits, then drop */
    bool lbl_lost = false;
    if (g_lbl_dir[0] && r3d_labelvol_dirty(&g_lblv)) {
      printf("labels: auto-saving unsaved edits to %s\n", g_lbl_dir);
      errno = 0;
      if (r3d_labelvol_save(&g_lblv, g_lbl_dir) != 0) {
        lbl_lost = true;
        snprintf(g_lbl_status, sizeof g_lbl_status, "SAVE FAILED: %s (%s)", g_lbl_dir,
                 errno ? strerror(errno) : "write error");
        fprintf(stderr,
                "labels: AUTOSAVE FAILED for %s (%s) - edits kept in memory; "
                "fix the path and press save in the labels panel\n",
                g_lbl_dir, errno ? strerror(errno) : "write error");
      }
    }
    if (!lbl_lost) { /* only drop the volume once its edits are durable */
      r3d_labelvol_free(&g_lblv);
      g_lbl_init = false;
      g_lbl_paint = false;
      g_lbl_dir[0] = 0;
    }
  }
  smask_drop(renderer, sgc_active); /* mask autosaves with its segment */
  if (g_reg_open) { /* registration is per volume too */
    g_reg_open = false;
    g_reg_flat = false;
    g_reg_busy = false;
    r3d_bricks_regatlas_detach(renderer); /* worker holds the source */
    r3d_regvol_close(&g_reg);
  }
  if (!od_swap) break;
  r3d_bricks_end(renderer); /* dataset swap: teardown, then reopen */
  if (slab_src.voxels) r3d_volume_close(&slab_src);
  } /* dataset loop */
  if (od.job_up) { /* own the browser's child: terminate and reap it */
    close(od.jobfd);
    od.jobfd = -1;
    od.job_up = false;
    kill(od.jobpid, SIGTERM);
    int jst = 0;
    while (waitpid(od.jobpid, &jst, 0) < 0 && errno == EINTR) {}
    od.jobpid = 0;
  }
  if (od.fth_up) { /* stop the browser's listing worker */
    pthread_mutex_lock(&od.fmu);
    od.fquit = true;
    pthread_cond_broadcast(&od.fcv);
    pthread_mutex_unlock(&od.fmu);
    pthread_join(od.fth, NULL);
  }
  r3d_destroy(renderer);
  if (win) SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
