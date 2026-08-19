/* rendseg surface-volume stack export (ctest label: quick — offline).
 * A synthetic flat segment (written through the real tracer exporter) inside
 * the synthetic tree is rendered as a 5-layer stack in both forms:
 *  - raw layer-major: the center layer must match the CT at the surface
 *    (the segment lies at known analytic positions) and the +-2 layers must
 *    sample above/below it
 *  - png-per-layer: 5 layer files plus a stack.json sidecar carrying the
 *    registration metadata (up, grid dims, scale, nvalid)
 * Usage: test_stack <path-to-rendseg> */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "synthtree.h"

#include "core/tracer.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

/* a flat sheet at z = 128 fully inside the 256^3 tree (grid step 4) */
static int make_segment(const char *dir) {
  if (mkdir(dir, 0755) != 0) return -1;
  r3d_tracer t = {0};
  t.W = 40;
  t.H = 30;
  t.cfg.step = 4.0;
  t.cfg.max_ring = 40;
  uint64_t n = (uint64_t)t.W * t.H;
  t.pos = calloc(n * 3, sizeof *t.pos);
  t.state = calloc(n, 1);
  t.conf = calloc(n, sizeof *t.conf);
  t.gen_of = calloc(n, sizeof *t.gen_of);
  if (!t.pos || !t.state || !t.conf || !t.gen_of) return -1;
  pthread_mutex_init(&t.mu, NULL);
  for (uint32_t j = 0; j < t.H; j++)
    for (uint32_t i = 0; i < t.W; i++) {
      size_t k = (size_t)j * t.W + i;
      t.pos[k * 3 + 0] = 48.0 + 4.0 * (double)i;
      t.pos[k * 3 + 1] = 60.0 + 4.0 * (double)j;
      t.pos[k * 3 + 2] = 128.0;
      t.state[k] = R3D_TR_SET;
      t.conf[k] = 1.0f;
      t.gen_of[k] = 1;
    }
  int rc = r3d_tracer_save(&t, dir, 0.0f, false);
  r3d_tracer_free(&t);
  return rc;
}

static void rm_rf(const char *dir) { /* test-owned temp paths only */
  char cmd[900];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
  if (system(cmd) != 0) fprintf(stderr, "cleanup failed for %s\n", dir);
}

static uint8_t *read_all(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long ln = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = ln > 0 ? malloc((size_t)ln) : NULL;
  size_t got = b ? fread(b, 1, (size_t)ln, f) : 0;
  fclose(f);
  if (b && got != (size_t)ln) {
    free(b);
    return NULL;
  }
  *n = got;
  return b;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_stack <rendseg-binary>\n");
    return 77;
  }
  char tmp[512];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_stack_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[600], seg[600], rawp[620], pngd[620], cmd[2200];
  snprintf(root, sizeof root, "%s/tree", tmp);
  snprintf(seg, sizeof seg, "%s/seg", tmp);
  snprintf(rawp, sizeof rawp, "%s/stack.raw", tmp);
  snprintf(pngd, sizeof pngd, "%s/stackpng", tmp);
  uint32_t dim[3] = {256, 256, 256};
  if (st_make_tree(root, dim, 2, 0) != 0 || make_segment(seg) != 0) {
    fprintf(stderr, "fixture build failed\n");
    return 1;
  }
  /* raw stack at level 0, up 2: numeric verification of the layer axis */
  const uint32_t UP = 2, NL = 5;
  snprintf(cmd, sizeof cmd, "%s %s %s %s --level 0 --up %u --layers %u >%s/r.log 2>&1",
           argv[1], root, seg, rawp, UP, NL, tmp);
  CHECK(system(cmd) == 0);
  uint32_t W = 39 * UP, H = 29 * UP; /* (w-1)*up x (h-1)*up */
  size_t rn = 0;
  uint8_t *raw = read_all(rawp, &rn);
  CHECK(raw && rn == (size_t)W * H * NL);
  if (raw && rn == (size_t)W * H * NL) {
    /* pixel (ox,oy) -> grid (ox/up, oy/up) -> world (48 + 2*ox, 60 + 2*oy, 128);
     * layer l samples z = 128 + (l - 2) along the +z normal (flat sheet) */
    double mad_c = 0, mad_off = 0;
    uint32_t nchk = 0;
    for (uint32_t oy = 4; oy < H - 4; oy += 3)
      for (uint32_t ox = 4; ox < W - 4; ox += 3) {
        uint32_t wx = 48 + 2 * ox, wy = 60 + 2 * oy;
        double c = raw[(size_t)2 * W * H + (size_t)oy * W + ox];
        double lo = raw[(size_t)0 * W * H + (size_t)oy * W + ox];
        double hi = raw[(size_t)4 * W * H + (size_t)oy * W + ox];
        mad_c += fabs(c - (double)st_pat(wx, wy, 128));
        /* layer order may be +z or -z depending on the normal's sign:
         * accept the better of the two assignments per sample */
        double a = fabs(lo - (double)st_pat(wx, wy, 126)) +
                   fabs(hi - (double)st_pat(wx, wy, 130));
        double b = fabs(lo - (double)st_pat(wx, wy, 130)) +
                   fabs(hi - (double)st_pat(wx, wy, 126));
        mad_off += (a < b ? a : b) * 0.5;
        nchk++;
      }
    CHECK(nchk > 100);
    CHECK(mad_c / nchk < 4.0);   /* codec + bilinear tolerance */
    CHECK(mad_off / nchk < 5.0); /* off-surface layers track +-1 voxel */
    /* sidecar carries the registration metadata */
    char jp[680];
    snprintf(jp, sizeof jp, "%s.json", rawp);
    size_t jn = 0;
    uint8_t *js = read_all(jp, &jn);
    CHECK(js && jn > 0);
    if (js) {
      char *j = (char *)js;
      j[jn - 1] = 0;
      CHECK(strstr(j, "\"up\": 2") && strstr(j, "\"grid_w\": 40") &&
            strstr(j, "\"grid_h\": 30") && strstr(j, "\"layers\": 5") &&
            strstr(j, "\"surface_layer\": 2") && strstr(j, "\"nvalid\""));
      free(js);
    }
  }
  free(raw);
  /* png stack: one file per layer + stack.json */
  snprintf(cmd, sizeof cmd,
           "%s %s %s %s --level 0 --up %u --layers %u --stack png >%s/p.log 2>&1",
           argv[1], root, seg, pngd, UP, NL, tmp);
  CHECK(system(cmd) == 0);
  for (uint32_t l = 0; l < NL; l++) {
    char lp[700];
    snprintf(lp, sizeof lp, "%s/layer_%03u.png", pngd, l);
    struct stat st;
    CHECK(stat(lp, &st) == 0 && st.st_size > 100);
  }
  {
    char jp[700];
    snprintf(jp, sizeof jp, "%s/stack.json", pngd);
    struct stat st;
    CHECK(stat(jp, &st) == 0 && st.st_size > 100);
  }
  rm_rf(tmp);
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("rendseg stack export OK\n");
  return 0;
}
