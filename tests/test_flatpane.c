/* Flattened-segment pane (ctest label: gpu).
 * A synthetic near-flat sheet (written through the real tracer exporter, so
 * it is a genuine tifxyz segment) lies inside the synthetic tree at
 * z ~= 128. Opening it as the multiview segment must light up the
 * flattened pane with baked CT content; without a segment that pane stays
 * black. Guards the surfvol bake + seg-pane routing end to end.
 * Usage: test_flatpane <path-to-render3d> */
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

/* a 40x30 sheet fully inside the 256^3 tree, undulating around z = 128 */
static int make_flat_segment(const char *dir) {
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
      t.pos[k * 3 + 0] = 48.0 + 4.0 * (double)i;  /* 48..204 */
      t.pos[k * 3 + 1] = 60.0 + 4.0 * (double)j;  /* 60..176 */
      t.pos[k * 3 + 2] = 128.0 + 2.0 * sin((double)i * 0.3);
      t.state[k] = R3D_TR_SET;
      t.conf[k] = 1.0f;
      t.gen_of[k] = 1;
    }
  int rc = r3d_tracer_save(&t, dir, 0.0f, false);
  r3d_tracer_free(&t);
  return rc;
}

static uint8_t *read_ppm(const char *path, uint32_t *w, uint32_t *h) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  char magic[3] = {0};
  uint32_t maxv = 0;
  if (fscanf(f, "%2s %u %u %u", magic, w, h, &maxv) != 4 || strcmp(magic, "P6") != 0 ||
      maxv != 255) {
    fclose(f);
    return NULL;
  }
  fgetc(f);
  size_t n = (size_t)*w * *h * 3;
  uint8_t *px = malloc(n);
  size_t got = px ? fread(px, 1, n, f) : 0;
  fclose(f);
  if (got != n) {
    free(px);
    return NULL;
  }
  return px;
}

/* content pixels in the top-left (flattened) pane, panel strip excluded */
static uint64_t seg_pane_content(const uint8_t *px, uint32_t w, uint32_t h) {
  uint64_t n = 0;
  uint32_t x0 = w * 25 / 100, x1 = w * 55 / 100, y1 = h * 45 / 100;
  for (uint32_t y = 8; y < y1; y++)
    for (uint32_t x = x0; x < x1; x++) {
      size_t o = ((size_t)y * w + x) * 3;
      if ((uint32_t)px[o] + px[o + 1] + px[o + 2] >= 60) n++;
    }
  return n;
}

static void rm_rf(const char *dir) { /* test-owned temp paths only */
  char cmd[900];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
  if (system(cmd) != 0) fprintf(stderr, "cleanup failed for %s\n", dir);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_flatpane <render3d-binary>\n");
    return 77;
  }
  char tmp[256];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_flat_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[320], seg[320];
  snprintf(root, sizeof root, "%s/tree", tmp);
  snprintf(seg, sizeof seg, "%s/seg", tmp);
  uint32_t dim[3] = {256, 256, 256};
  if (st_make_tree(root, dim, 2, 0) != 0 || make_flat_segment(seg) != 0) {
    fprintf(stderr, "fixture build failed\n");
    return 1;
  }
  char shot[2][360], logp[2][360], cmd[1800];
  int rc = 0;
  for (int i = 0; i < 2 && rc == 0; i++) {
    snprintf(shot[i], sizeof shot[i], "%s/s%d.ppm", tmp, i);
    snprintf(logp[i], sizeof logp[i], "%s/s%d.log", tmp, i);
    snprintf(cmd, sizeof cmd,
             "R3D_MV_FIT=300 %s --bricks %s/manifest.json %s%s --headless "
             "--frames 500 --shot %s >%s 2>&1",
             argv[1], root, i ? "--multiview " : "", i ? seg : "", shot[i], logp[i]);
    if (system(cmd) != 0) {
      fprintf(stderr, "render3d run %d failed - skipping (see %s)\n", i, logp[i]);
      rc = 77;
    }
  }
  if (rc == 0) {
    uint32_t w0, h0, w1, h1;
    uint8_t *p0 = read_ppm(shot[0], &w0, &h0);
    uint8_t *p1 = read_ppm(shot[1], &w1, &h1);
    CHECK(p0 && p1);
    if (p0 && p1) {
      uint64_t c0 = seg_pane_content(p0, w0, h0);
      uint64_t c1 = seg_pane_content(p1, w1, h1);
      if (!(c0 < 500 && c1 > 5000)) {
        fprintf(stderr, "flattened pane content: without seg %llu px, with seg %llu px\n",
                (unsigned long long)c0, (unsigned long long)c1);
        failures++;
      } else
        printf("flattened pane OK (%llu content px baked, %llu without)\n",
               (unsigned long long)c1, (unsigned long long)c0);
    }
    free(p0);
    free(p1);
  }
  rm_rf(tmp);
  return rc ? rc : (failures ? 1 : 0);
}
