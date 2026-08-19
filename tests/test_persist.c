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
}

/* ---- tracer: save -> load round trip through the versioned sidecar ----- */

static void test_tracer_roundtrip(const char *tmp) {
  char dir[700];
  snprintf(dir, sizeof dir, "%s/trace", tmp);
  CHECK(mkdir(dir, 0755) == 0); /* save expects the target dir to exist */
  r3d_tracer t = {0};
  t.W = 40;
  t.H = 30;
  t.cfg.step = 5.0;
  t.cfg.max_ring = 60;
  t.cfg.thresh = 0.42f;
  t.cfg.level = 2;
  uint64_t n = (uint64_t)t.W * t.H;
  t.pos = calloc(n * 3, sizeof *t.pos);
  t.state = calloc(n, 1);
  t.conf = calloc(n, sizeof *t.conf);
  t.wind = calloc(n, sizeof *t.wind);
  t.gen_of = calloc(n, sizeof *t.gen_of);
  CHECK(t.pos && t.state && t.conf && t.wind && t.gen_of);
  pthread_mutex_init(&t.mu, NULL);
  /* a gently curved synthetic sheet, seed ring at the center: geometry
   * whose 4-neighbor edges match cfg.step so the tear mask keeps it all */
  for (uint32_t j = 0; j < t.H; j++)
    for (uint32_t i = 0; i < t.W; i++) {
      size_t k = (size_t)j * t.W + i;
      t.pos[k * 3 + 0] = 100.0 + 5.0 * (double)i;
      t.pos[k * 3 + 1] = 200.0 + 5.0 * (double)j;
      t.pos[k * 3 + 2] = 50.0 + 3.0 * sin((double)i * 0.2);
      t.state[k] = R3D_TR_SET;
      t.conf[k] = 0.25f + 0.5f * (float)i / (float)t.W;
      t.wind[k] = 0.01f * (float)i;
      uint32_t ri = i > t.W / 2 ? i - t.W / 2 : t.W / 2 - i;
      uint32_t rj = j > t.H / 2 ? j - t.H / 2 : t.H / 2 - j;
      t.gen_of[k] = (uint16_t)(1u + (ri > rj ? ri : rj));
    }
  atomic_store(&t.gens_done, 15u);
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
