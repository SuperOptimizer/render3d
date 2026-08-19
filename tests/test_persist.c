/* Persistence + registration + CPU-cache tests (ctest label: quick).
 * Builds a small synthetic c5d LOD tree (256^3, two levels, textured) in a
 * temp dir so everything runs without scroll data:
 *  - cpuvol: decode correctness, and the pin/lease protocol under
 *    concurrent readers with a 2-slot cache (eviction churn) — exact-match
 *    against a single-threaded reference, so any unpinned-overwrite bug
 *    shows up as wrong bytes (and as a race under TSan)
 *  - labelvol: paint/save/load round trip, plus fail-closed behavior for
 *    unwritable dirs, truncated bricks, and mismatched manifests
 *  - regvol: identity resample == source at both levels, transform.json
 *    round trip, out-of-volume gen gating, and a rigid auto-refine
 *    recovering a known misalignment */
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "synthtree.h"

#include "core/cpuvol.h"
#include "core/labelvol.h"
#include "core/regvol.h"
#include "core/segstore.h"
#include "core/tracer.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

#define TB ST_B                  /* brick edge */
#define TDIM 256u               /* synthetic volume edge (2x2x2 bricks) */
#define TB3 ((size_t)TB * TB * TB)

/* ---- cpuvol: decode correctness + lease protocol under eviction churn --- */

static void test_cpuvol_basic(const char *root, const uint8_t *ref) {
  r3d_cpuvol v;
  CHECK(r3d_cpuvol_open(&v, root, 8) == 0);
  CHECK(v.nx == TDIM && v.ny == TDIM && v.nz == TDIM && v.nlev == 2);
  static uint8_t got[TB * TB]; /* one z=const plane slice of a brick */
  r3d_cpuvol_read_block(&v, 0, 100, 7, 200, TB, TB, 1, got);
  double mad = 0;
  for (uint32_t y = 0; y < TB; y++)
    for (uint32_t x = 0; x < TB; x++)
      mad += fabs((double)got[(size_t)y * TB + x] -
                  (double)ref[((size_t)200 * TDIM + (7 + y)) * TDIM + (100 + x)]);
  mad /= (double)(TB * TB);
  CHECK(mad < 0.01); /* decode is deterministic: must equal the reference */
  /* level 1 approximates the mean-downsampled pattern (lossy, so loose) */
  double p1[3] = {60.0, 60.0, 60.0};
  double lv = r3d_cpuvol_tri(&v, 1, p1, NULL);
  CHECK(fabs(lv - (double)st_pat_lvl(1, 30, 30, 30)) < 6.0);
  r3d_cpuvol_close(&v);
}

struct churn {
  r3d_cpuvol *v;
  const uint8_t *ref;
  uint32_t seed;
  int bad;
};

static void *churn_thread(void *a) {
  struct churn *c = a;
  uint32_t s = c->seed;
  for (int i = 0; i < 30000; i++) {
    s = s * 1664525u + 1013904223u;
    uint32_t x = (s >> 8) % TDIM;
    uint32_t y = (s >> 12) % TDIM;
    uint32_t z = (s >> 16) % TDIM;
    uint8_t got = r3d_cpuvol_at(c->v, 0, (double)x, (double)y, (double)z);
    uint8_t want = c->ref[((size_t)z * TDIM + y) * TDIM + x];
    if (got != want) c->bad++;
  }
  return NULL;
}

static void test_cpuvol_lease(const char *root, const uint8_t *ref) {
  /* 2-slot cache + 8 bricks + 4 threads = constant eviction under load.
   * Before the pin/lease protocol this returned bytes from bricks being
   * overwritten mid-read; now every sample must match the reference. */
  r3d_cpuvol v;
  CHECK(r3d_cpuvol_open(&v, root, 2) == 0);
  struct churn c[4];
  pthread_t th[4];
  for (int i = 0; i < 4; i++) {
    c[i] = (struct churn){.v = &v, .ref = ref, .seed = (uint32_t)(i * 7919 + 1)};
    CHECK(pthread_create(&th[i], NULL, churn_thread, &c[i]) == 0);
  }
  int bad = 0;
  for (int i = 0; i < 4; i++) {
    pthread_join(th[i], NULL);
    bad += c[i].bad;
  }
  CHECK(bad == 0);
  r3d_cpuvol_close(&v);
}

/* ---- labelvol: round trip + fail-closed persistence --------------------- */

static void test_labelvol(const char *tmp) {
  char dir[700];
  snprintf(dir, sizeof dir, "%s/labels", tmp);
  uint32_t dim[3] = {300, 260, 200};
  r3d_labelvol lv;
  CHECK(r3d_labelvol_init(&lv, dim) == 0);
  double p1[3] = {130.0, 120.0, 100.0}, p2[3] = {10.0, 10.0, 10.0};
  uint64_t c1 = r3d_labelvol_paint(&lv, p1, 6.0, 2);
  uint64_t c2 = r3d_labelvol_paint(&lv, p2, 4.0, 1);
  CHECK(c1 > 800 && c2 > 200);
  CHECK(lv.nvox[2] == c1 && lv.nvox[1] == c2);
  CHECK(r3d_labelvol_gen(&lv, 0, 1, 0, 0) > 0);
  CHECK(r3d_labelvol_gen(&lv, 1, 0, 0, 0) > 0);
  CHECK(r3d_labelvol_gen(&lv, 0, 2, 2, 1) == 0);
  /* save to an unwritable dir: fails, everything stays dirty */
  uint32_t dirty0 = r3d_labelvol_dirty(&lv);
  CHECK(dirty0 > 0);
  if (geteuid() != 0) { /* root ignores permissions */
    char ro[700];
    snprintf(ro, sizeof ro, "%s/ro", tmp);
    CHECK(mkdir(ro, 0555) == 0);
    char rod[750];
    snprintf(rod, sizeof rod, "%s/labels", ro);
    CHECK(r3d_labelvol_save(&lv, rod) != 0);
    CHECK(r3d_labelvol_dirty(&lv) == dirty0);
    chmod(ro, 0755);
    rmdir(ro);
  }
  /* real save + fresh-volume load round trip */
  CHECK(r3d_labelvol_save(&lv, dir) == 0);
  CHECK(r3d_labelvol_dirty(&lv) == 0);
  r3d_labelvol lw;
  CHECK(r3d_labelvol_init(&lw, dim) == 0);
  CHECK(r3d_labelvol_load(&lw, dir) == 0);
  CHECK(lw.nvox[1] == lv.nvox[1] && lw.nvox[2] == lv.nvox[2]);
  size_t nb = (size_t)lv.lnb[0][0] * lv.lnb[0][1] * lv.lnb[0][2];
  for (size_t i = 0; i < nb; i++) {
    const uint8_t *a = lv.data[i], *b = lw.data[i];
    if (!a && !b) continue;
    static uint8_t za[128 * 128 * 128];
    if (!a) CHECK(b && memcmp(b, memset(za, 0, sizeof za), sizeof za) == 0);
    else if (!b) CHECK(memcmp(a, memset(za, 0, sizeof za), sizeof za) == 0);
    else CHECK(memcmp(a, b, sizeof za) == 0);
  }
  /* coarse-LOD fetch stride-samples the paint grid: an even-coordinate
   * voxel painted at level 0 must appear at half coords in the level-1
   * brick (out[o] samples world voxel o<<1) */
  {
    double pv[3] = {50.0, 40.0, 30.0};
    CHECK(r3d_labelvol_paint(&lv, pv, 0.4, 7) == 1);
    static uint8_t l1[128 * 128 * 128];
    r3d_labelvol_fetch(&lv, 1, 0, 0, 0, l1);
    CHECK(l1[((size_t)15 * 128 + 20) * 128 + 25] == 7);
    CHECK(r3d_labelvol_gen(&lv, 1, 0, 0, 0) > 0);
    CHECK(r3d_labelvol_paint(&lv, pv, 0.4, 0) == 1); /* erase it again */
    CHECK(lv.nvox[7] == 0);
  }
  /* truncated brick: load fails closed, the live volume is untouched */
  {
    DIR *dp = opendir(dir);
    struct dirent *de;
    char victim[800] = "";
    while (dp && (de = readdir(dp)) != NULL)
      if (strncmp(de->d_name, "b_", 2) == 0) {
        snprintf(victim, sizeof victim, "%s/%s", dir, de->d_name);
        break;
      }
    if (dp) closedir(dp);
    CHECK(victim[0]);
    CHECK(truncate(victim, 10) == 0);
    uint64_t keep1 = lw.nvox[1], keep2 = lw.nvox[2];
    r3d_labelvol_load(&lw, dir); /* may report failure; must not corrupt */
    CHECK(lw.nvox[1] == keep1 && lw.nvox[2] == keep2);
  }
  /* mismatched manifest dims: refused before touching live bricks */
  {
    uint32_t dim2[3] = {512, 512, 512};
    r3d_labelvol lx;
    CHECK(r3d_labelvol_init(&lx, dim2) == 0);
    CHECK(r3d_labelvol_load(&lx, dir) != 0);
    CHECK(lx.nvox[1] == 0 && lx.nvox[2] == 0);
    r3d_labelvol_free(&lx);
  }
  /* erase everything, save: brick files disappear; empty dir loads as ok */
  double pc[3] = {150.0, 130.0, 100.0};
  CHECK(r3d_labelvol_paint(&lv, pc, 500.0, 0) > 0);
  CHECK(lv.nvox[1] == 0 && lv.nvox[2] == 0);
  CHECK(r3d_labelvol_save(&lv, dir) == 0);
  {
    DIR *dp = opendir(dir);
    struct dirent *de;
    int bricks = 0;
    while (dp && (de = readdir(dp)) != NULL)
      if (strncmp(de->d_name, "b_", 2) == 0) bricks++;
    if (dp) closedir(dp);
    CHECK(bricks == 0);
  }
  r3d_labelvol lz;
  CHECK(r3d_labelvol_init(&lz, dim) == 0);
  CHECK(r3d_labelvol_load(&lz, dir) == 0); /* valid manifest, zero bricks */
  r3d_labelvol_free(&lz);
  r3d_labelvol_free(&lv);
  r3d_labelvol_free(&lw);
}

/* ---- regvol: resample identity, json round trip, rigid refine ----------- */

static void test_regvol(const char *root, const char *tmp, const uint8_t *ref) {
  uint32_t fd[3] = {TDIM, TDIM, TDIM};
  r3d_regvol rv;
  CHECK(r3d_regvol_open(&rv, root, fd) == 0);
  /* identity fetch reproduces the source exactly at both levels */
  static uint8_t got[TB * TB * TB];
  r3d_regvol_srcfetch(&rv, 0, 1, 0, 1, got);
  double mad = 0;
  for (uint32_t z = 0; z < TB; z++)
    for (uint32_t y = 0; y < TB; y++)
      for (uint32_t x = 0; x < TB; x++)
        mad += fabs((double)got[((size_t)z * TB + y) * TB + x] -
                    (double)ref[((size_t)(TB + z) * TDIM + y) * TDIM + (TB + x)]);
  CHECK(mad / (double)TB3 < 0.01);
  CHECK(r3d_regvol_srcgen(&rv, 0, 1, 0, 1) > 0);
  /* voxel-pitch parsing */
  CHECK(fabs(r3d_regvol_parse_um("a/20231117-1.129um-b.zarr") - 1.129) < 1e-9);
  CHECK(r3d_regvol_parse_um("my/volumes/thing") == 0.0);
  /* a brick fully outside the moving volume gates to gen 0 */
  rv.d_tr[0] = 4096.0;
  r3d_regvol_bump(&rv);
  CHECK(r3d_regvol_srcgen(&rv, 0, 0, 0, 0) == 0);
  r3d_regvol_reset_deltas(&rv);
  /* transform.json round trip preserves the effective pull map */
  rv.d_tr[0] = 3.25;
  rv.d_rot[2] = 0.01;
  r3d_regvol_bump(&rv);
  char jp[700];
  snprintf(jp, sizeof jp, "%s/reg.json", tmp);
  CHECK(r3d_regvol_save_json(&rv, jp) == 0);
  double P0[12], P1[12];
  r3d_regvol_pull(&rv, P0);
  r3d_regvol_reset_deltas(&rv);
  CHECK(r3d_regvol_load_json(&rv, jp) == 0);
  r3d_regvol_pull(&rv, P1);
  for (int i = 0; i < 12; i++) CHECK(fabs(P0[i] - P1[i]) < 1e-6);
  /* shipped Vesuvius convention: transformation_matrix is moving->fixed in
   * XYZ; load must invert it into the pull map. f = 0.5*m + t  =>
   * m = 2*(f - t): check the exact inverse lands. */
  {
    char sp2[700];
    snprintf(sp2, sizeof sp2, "%s/shipped.json", tmp);
    FILE *f = fopen(sp2, "w");
    CHECK(f != NULL);
    if (f) {
      fprintf(f, "{\n  \"transformation_matrix\": [\n"
                 "    [0.5, 0, 0, 10],\n    [0, 0.5, 0, 20],\n"
                 "    [0, 0, 0.5, 30]\n  ]\n}\n");
      fclose(f);
      CHECK(r3d_regvol_load_json(&rv, sp2) == 0);
      double Ps[12];
      r3d_regvol_pull(&rv, Ps);
      CHECK(fabs(Ps[0] - 2.0) < 1e-9 && fabs(Ps[3] + 20.0) < 1e-9);
      CHECK(fabs(Ps[5] - 2.0) < 1e-9 && fabs(Ps[7] + 40.0) < 1e-9);
      CHECK(fabs(Ps[10] - 2.0) < 1e-9 && fabs(Ps[11] + 60.0) < 1e-9);
      /* malformed json: refused, and the transform is left untouched */
      FILE *g = fopen(sp2, "w");
      CHECK(g != NULL);
      if (g) {
        fprintf(g, "{ not json at all ]]");
        fclose(g);
        CHECK(r3d_regvol_load_json(&rv, sp2) != 0);
        double Pk[12];
        r3d_regvol_pull(&rv, Pk);
        for (int i = 0; i < 12; i++) CHECK(fabs(Pk[i] - Ps[i]) < 1e-12);
      }
      unlink(sp2);
    }
  }
  /* rigid refine recovers a known misalignment (64^3 ROI at the center) */
  r3d_regvol_set_scale(&rv, 1.0);
  rv.d_tr[0] = 2.5;
  rv.d_tr[1] = -1.5;
  r3d_regvol_bump(&rv);
  double ctr[3] = {128.0, 128.0, 128.0};
  CHECK(r3d_regvol_job_start(&rv, root, 1, ctr, 32, 0) == 0);
  bool ok = false;
  while (r3d_regvol_job_poll(&rv, &ok) == 1) usleep(50000);
  CHECK(ok);
  CHECK(rv.ncc1 > 0.9 && rv.ncc1 > rv.ncc0);
  double P[12];
  r3d_regvol_pull(&rv, P);
  CHECK(fabs(P[3]) < 0.5 && fabs(P[7]) < 0.5 && fabs(P[11]) < 0.5);
  r3d_regvol_close(&rv);
  unlink(jp);
  /* closing with a job still running must join it, not crash or hang
   * (the GUI can close/replace the moving volume at any time) */
  CHECK(r3d_regvol_open(&rv, root, fd) == 0);
  CHECK(r3d_regvol_job_start(&rv, root, 2, ctr, 32, 0) == 0);
  r3d_regvol_close(&rv); /* affine refine mid-flight */
  r3d_regvol_close(&rv); /* double close is a no-op */
}

/* ---- tracer: save -> load round trip through the versioned sidecar ----- */

/* a gently curved synthetic sheet, seed ring at the center: geometry whose
 * 4-neighbor edges match cfg.step so the tear mask keeps it all. zoff
 * varies the sheet so distinct "segments" differ. */
static int make_synth_trace(r3d_tracer *t, double zoff) {
  memset(t, 0, sizeof *t);
  t->W = 40;
  t->H = 30;
  t->cfg.step = 5.0;
  t->cfg.max_ring = 60;
  t->cfg.thresh = 0.42f;
  t->cfg.level = 2;
  uint64_t n = (uint64_t)t->W * t->H;
  t->pos = calloc(n * 3, sizeof *t->pos);
  t->state = calloc(n, 1);
  t->conf = calloc(n, sizeof *t->conf);
  t->wind = calloc(n, sizeof *t->wind);
  t->gen_of = calloc(n, sizeof *t->gen_of);
  if (!t->pos || !t->state || !t->conf || !t->wind || !t->gen_of) return -1;
  pthread_mutex_init(&t->mu, NULL);
  for (uint32_t j = 0; j < t->H; j++)
    for (uint32_t i = 0; i < t->W; i++) {
      size_t k = (size_t)j * t->W + i;
      t->pos[k * 3 + 0] = 100.0 + 5.0 * (double)i;
      t->pos[k * 3 + 1] = 200.0 + 5.0 * (double)j;
      t->pos[k * 3 + 2] = 50.0 + zoff + 3.0 * sin((double)i * 0.2);
      t->state[k] = R3D_TR_SET;
      t->conf[k] = 0.25f + 0.5f * (float)i / (float)t->W;
      t->wind[k] = 0.01f * (float)i;
      uint32_t ri = i > t->W / 2 ? i - t->W / 2 : t->W / 2 - i;
      uint32_t rj = j > t->H / 2 ? j - t->H / 2 : t->H / 2 - j;
      t->gen_of[k] = (uint16_t)(1u + (ri > rj ? ri : rj));
    }
  atomic_store(&t->gens_done, 15u);
  return 0;
}

static void test_tracer_roundtrip(const char *tmp) {
  char dir[700];
  snprintf(dir, sizeof dir, "%s/trace", tmp);
  CHECK(mkdir(dir, 0755) == 0); /* save expects the target dir to exist */
  r3d_tracer t;
  CHECK(make_synth_trace(&t, 0.0) == 0);
  uint64_t n = (uint64_t)t.W * t.H;
  t.anc[0] = 111.0;
  t.anc[1] = 222.0;
  t.anc[2] = 55.0;
  t.nanc = 1;
  CHECK(r3d_tracer_save(&t, dir, 0.0f, false) == 0);
  r3d_tracer u = {0};
  CHECK(r3d_tracer_load(&u, dir, NULL) == 0);
  CHECK(u.W == t.W && u.H == t.H);
  CHECK(fabs(u.cfg.step - t.cfg.step) < 1e-9);
  CHECK(u.cfg.max_ring == t.cfg.max_ring);
  CHECK(fabs((double)u.cfg.thresh - (double)t.cfg.thresh) < 1e-6);
  CHECK(u.cfg.level == t.cfg.level);
  CHECK(u.nanc == 1 && fabs(u.anc[0] - 111.0) < 1e-6 && fabs(u.anc[2] - 55.0) < 1e-6);
  double mad = 0;
  uint64_t set = 0;
  bool gen_ok = true;
  for (uint64_t k = 0; k < n; k++) {
    if (u.state[k] != R3D_TR_SET || t.state[k] != R3D_TR_SET) continue;
    set++;
    for (uint64_t a = 0; a < 3; a++)
      mad += fabs(u.pos[k * 3 + a] - t.pos[k * 3 + a]);
    if (u.gen_of[k] != t.gen_of[k]) gen_ok = false;
  }
  CHECK(set == n); /* nothing torn/cropped away */
  CHECK(mad / ((double)set * 3.0) < 0.01);
  CHECK(gen_ok);
  /* the restored budget must allow further growth (P0.2: no zero-budget
   * resumes): max_ring restored above already proves the sidecar path */
  r3d_tracer_free(&u);
  r3d_tracer_free(&t);
  DIR *dp = opendir(dir); /* cleanup: whatever artifact set save wrote */
  struct dirent *de;
  while (dp && (de = readdir(dp)) != NULL)
    if (de->d_name[0] != '.') {
      char p[900];
      snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
      unlink(p);
    }
  if (dp) closedir(dp);
  rmdir(dir);
}

/* ---- tracer: spiral fill repopulates an erased ribbon ------------------- */

static void test_spiral_fill(const char *root) {
  /* synthetic spiral ribbon (64 x 12) on rho = r0 + omega*w about a fixed
   * center; half the columns erased. With the fit parameters set to the
   * true generators, r3d_tracer_spiral_fill must repopulate the erased
   * region near the analytic sheet (the worker opens `root` as its
   * prediction volume; the follow-up polish runs over it). */
  r3d_tracer t = {0};
  t.W = 64;
  t.H = 12;
  t.cfg.step = 6.0;
  t.cfg.max_ring = 4;
  t.cfg.rib_rows = 12;
  t.cfg.wind_weight = 0.5;
  t.cfg.z_min = 1.0;
  t.cfg.z_max = 250.0;
  snprintf(t.root, sizeof t.root, "%s", root);
  uint64_t n = (uint64_t)t.W * t.H;
  t.pos = calloc(n * 3, sizeof *t.pos);
  t.state = calloc(n, 1);
  t.conf = calloc(n, sizeof *t.conf);
  t.wind = calloc(n, sizeof *t.wind);
  t.gen_of = calloc(n, sizeof *t.gen_of);
  CHECK(t.pos && t.state && t.conf && t.wind && t.gen_of);
  pthread_mutex_init(&t.mu, NULL);
  r3d_umbilicus_init(&t.umb);
  r3d_umbilicus_set(&t.umb, 128.0, 128.0, 0.0);
  r3d_umbilicus_set(&t.umb, 128.0, 128.0, 255.0);
  const double om = 14.0, r0 = 30.0, ucx = 128.0, ucy = 128.0;
  /* arc-length parameterization: one grid step = cfg.step voxels along the
   * sheet, so the winding advances by step/(2 pi rho) per column — the same
   * geometry the ribbon grower produces and the fill assumes */
  double wcol[64];
  {
    double w = 0.0;
    for (uint32_t i = 0; i < t.W; i++) {
      wcol[i] = w;
      w += t.cfg.step / (2.0 * M_PI * (r0 + om * w));
    }
  }
  uint32_t kept = 0;
  for (uint32_t j = 0; j < t.H; j++)
    for (uint32_t i = 0; i < t.W; i++) {
      size_t k = (size_t)j * t.W + i;
      double w = wcol[i];
      double th = 2.0 * M_PI * w, rho = r0 + om * w;
      t.pos[k * 3 + 0] = ucx + rho * cos(th);
      t.pos[k * 3 + 1] = ucy + rho * sin(th);
      t.pos[k * 3 + 2] = 60.0 + (double)j * 6.0;
      t.wind[k] = (float)w;
      if (i < 32) { /* the right half of the ribbon is missing */
        t.state[k] = R3D_TR_SET;
        t.conf[k] = 1.0f;
        t.gen_of[k] = 1;
        kept++;
      }
    }
  t.nset = kept;
  /* inject the (true) fit so sp_valid gates open */
  atomic_store(&t.sp_omega, om);
  atomic_store(&t.sp_rms, 0.1);
  t.sp_z0 = 0.0;
  t.sp_dz = 256.0;
  t.sp_k = 2;
  t.sp_r0[0] = t.sp_r0[1] = r0;
  memset(t.sp_ab, 0, sizeof t.sp_ab);
  atomic_store(&t.sp_valid, true);
  CHECK(r3d_tracer_spiral_fill(&t) == 0);
  for (int spin = 0; spin < 1200 && t.running; spin++) usleep(100000);
  CHECK(!t.running);
  CHECK(t.nset > kept + 200); /* the erased half substantially refilled */
  /* refilled cells sit near the analytic spiral (the polish may nudge
   * them, so the gate is loose: within two grid steps) */
  double dists[64 * 12];
  uint32_t filled = 0;
  for (uint32_t j = 0; j < t.H; j++)
    for (uint32_t i = 32; i < t.W; i++) {
      size_t k = (size_t)j * t.W + i;
      if (t.state[k] != R3D_TR_SET) continue;
      double w = (double)t.wind[k];
      double th = 2.0 * M_PI * w, rho = r0 + om * w;
      double ex = ucx + rho * cos(th), ey = ucy + rho * sin(th);
      dists[filled++] = hypot(t.pos[k * 3 + 0] - ex, t.pos[k * 3 + 1] - ey);
    }
  CHECK(filled > 200);
  /* sort for quantiles: the polish may fling a handful of low-support
   * border cells to a neighbouring wrap; the BODY of the fill must track
   * the analytic sheet (real flows follow with CT-snap + QC anyway) */
  for (uint32_t a = 1; a < filled; a++)
    for (uint32_t b = a; b > 0 && dists[b] < dists[b - 1]; b--) {
      double tmp2 = dists[b];
      dists[b] = dists[b - 1];
      dists[b - 1] = tmp2;
    }
  double med = dists[filled / 2], p90 = dists[(filled * 9) / 10];
  CHECK(med < t.cfg.step);
  CHECK(p90 < 2.0 * t.cfg.step);
  printf("spiral fill: %u refilled, median %.2f / p90 %.2f / max %.2f vox\n", filled,
         med, p90, dists[filled - 1]);
  r3d_tracer_stop(&t);
  r3d_tracer_free(&t);
}

/* ---- segstore: fail-closed rebuild + recovery ---------------------------- */

static void rmdir_all(const char *dir) {
  DIR *dp = opendir(dir);
  struct dirent *de;
  while (dp && (de = readdir(dp)) != NULL)
    if (de->d_name[0] != '.') {
      char p[900];
      snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
      unlink(p);
    }
  if (dp) closedir(dp);
  rmdir(dir);
}

static void test_segstore(const char *tmp) {
  char sa[700], sb[700], store[700];
  snprintf(sa, sizeof sa, "%s/segA", tmp);
  snprintf(sb, sizeof sb, "%s/segB", tmp);
  snprintf(store, sizeof store, "%s/store", tmp);
  CHECK(mkdir(sa, 0755) == 0 && mkdir(sb, 0755) == 0 && mkdir(store, 0755) == 0);
  r3d_tracer t;
  CHECK(make_synth_trace(&t, 0.0) == 0);
  CHECK(r3d_tracer_save(&t, sa, 0.0f, false) == 0);
  r3d_tracer_free(&t);
  CHECK(make_synth_trace(&t, 40.0) == 0);
  CHECK(r3d_tracer_save(&t, sb, 0.0f, false) == 0);
  r3d_tracer_free(&t);
  /* build with A, then add B: B's build carries A forward */
  const char *da[1] = {sa}, *db[1] = {sb};
  CHECK(r3d_segstore_build(store, da, 1, 2, false) == 1);
  CHECK(r3d_segstore_build(store, db, 1, 2, false) == 2);
  r3d_segstore st;
  CHECK(r3d_segstore_open(&st, store) == 0);
  CHECK(st.n == 2);
  r3d_segstore_close(&st);
  if (geteuid() != 0) {
    /* fail-closed: an unwritable store dir must abort the rebuild and
     * leave the previous corpus fully usable */
    CHECK(chmod(store, 0555) == 0);
    CHECK(r3d_segstore_build(store, da, 1, 2, true) < 0); /* force re-encode */
    CHECK(chmod(store, 0755) == 0);
    CHECK(r3d_segstore_open(&st, store) == 0);
    CHECK(st.n == 2);
    r3d_segstore_close(&st);
  }
  /* corrupt manifest: open refuses (no crash, no partial corpus), and a
   * source-less rebuild regenerates it from the store's own .tfx files */
  char mp[760];
  snprintf(mp, sizeof mp, "%s/segments.r3ds", store);
  CHECK(truncate(mp, 17) == 0);
  CHECK(r3d_segstore_open(&st, store) != 0);
  CHECK(r3d_segstore_build(store, NULL, 0, 2, false) == 2);
  CHECK(r3d_segstore_open(&st, store) == 0);
  CHECK(st.n == 2);
  /* the rebuilt corpus still decodes */
  r3d_tifxyz s0;
  CHECK(r3d_segstore_load(&st, 0, 1, &s0) == 0);
  CHECK(s0.nvalid > 500);
  r3d_tifxyz_free(&s0);
  r3d_segstore_close(&st);
  rmdir_all(sa);
  rmdir_all(sb);
  rmdir_all(store);
}

int main(void) {
  char tmp[512];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_persist_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) {
    fprintf(stderr, "mkdtemp failed\n");
    return 1;
  }
  char root[600];
  snprintf(root, sizeof root, "%s/tree", tmp);
  uint32_t tdim[3] = {TDIM, TDIM, TDIM};
  if (st_make_tree(root, tdim, 2, 0) != 0) {
    fprintf(stderr, "synthetic tree build failed\n");
    return 1;
  }
  /* single-threaded decode reference: what every reader must observe */
  uint8_t *ref = malloc((size_t)TDIM * TDIM * TDIM);
  if (!ref) return 1;
  {
    r3d_cpuvol v;
    if (r3d_cpuvol_open(&v, root, 16) != 0) {
      fprintf(stderr, "cpuvol open failed\n");
      return 1;
    }
    r3d_cpuvol_read_block(&v, 0, 0, 0, 0, TDIM, TDIM, TDIM, ref);
    r3d_cpuvol_close(&v);
  }
  test_cpuvol_basic(root, ref);
  test_cpuvol_lease(root, ref);
  test_labelvol(tmp);
  test_regvol(root, tmp, ref);
  test_tracer_roundtrip(tmp);
  {
    char sproot[600];
    snprintf(sproot, sizeof sproot, "%s/sptree", tmp);
    st_spiral_mode = 1;
    if (st_make_tree(sproot, tdim, 2, 0) == 0) test_spiral_fill(sproot);
    else CHECK(false);
    st_spiral_mode = 0;
    st_rm_tree(sproot, 2);
  }
  test_segstore(tmp);
  free(ref);
  st_rm_tree(root, 2);
  /* label/json leftovers live under tmp; remove what the tests created */
  char lp[700];
  snprintf(lp, sizeof lp, "%s/labels/manifest.json", tmp);
  unlink(lp);
  snprintf(lp, sizeof lp, "%s/labels", tmp);
  rmdir(lp);
  rmdir(tmp);
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("persist/registration/cache tests OK\n");
  return 0;
}
