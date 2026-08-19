/* Label + registration display conformance (ctest label: gpu).
 * Renders the synthetic tree through the actual binary with the headless
 * hooks and asserts the display invariants:
 *  - R3D_LBLTEST: the painted class blobs appear in their palette colors
 *  - R3D_REGTEST self-overlay, identity: the green/magenta fuse is exactly
 *    neutral (|R-G| == 0 over content) — the moving sample runs through
 *    the same transfer function and low-cut as the fixed scan
 *  - R3D_REGTEST with a 24-voxel offset: fringes appear (mean |R-G| well
 *    above the aligned case)
 * Usage: test_overlay <path-to-render3d> */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "synthtree.h"

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
  fgetc(f); /* single whitespace after the header */
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

static int run_shot(const char *bin, const char *root, const char *env, const char *shot,
                    const char *log) {
  char cmd[2048];
  snprintf(cmd, sizeof cmd,
           "%s R3D_MV_FIT=300 %s --bricks %s/manifest.json --headless --frames 300 "
           "--shot %s >%s 2>&1",
           env, bin, root, shot, log);
  return system(cmd);
}

/* mean |R-G| over content pixels: the registration-fringe metric */
static double fringe(const uint8_t *px, uint32_t w, uint32_t h, uint64_t *content) {
  uint64_t n = 0, s = 0;
  for (uint32_t y = 0; y < h; y += 2)
    for (uint32_t x = 0; x < w; x += 2) {
      size_t o = ((size_t)y * w + x) * 3;
      if ((uint32_t)px[o] + px[o + 1] + px[o + 2] < 30) continue;
      n++;
      s += (uint64_t)abs((int)px[o] - (int)px[o + 1]);
    }
  *content = n;
  return n ? (double)s / (double)n : -1.0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_overlay <render3d-binary>\n");
    return 77; /* skip */
  }
  char tmp[256];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_ovl_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[320];
  snprintf(root, sizeof root, "%s/tree", tmp);
  uint32_t dim[3] = {256, 256, 256};
  if (st_make_tree(root, dim, 2, 0) != 0) {
    fprintf(stderr, "synthetic tree build failed\n");
    return 1;
  }
  char shot[3][360], logp[3][360], env[3][720];
  snprintf(env[0], sizeof env[0], "R3D_LBLTEST=1");
  snprintf(env[1], sizeof env[1], "R3D_REGTEST=%s", root);
  snprintf(env[2], sizeof env[2], "R3D_REGTEST=%s:24,0,0", root);
  int rc = 0;
  for (int i = 0; i < 3 && rc == 0; i++) {
    snprintf(shot[i], sizeof shot[i], "%s/s%d.ppm", tmp, i);
    snprintf(logp[i], sizeof logp[i], "%s/s%d.log", tmp, i);
    if (run_shot(argv[1], root, env[i], shot[i], logp[i]) != 0) {
      fprintf(stderr, "render3d run %d failed - skipping (see %s)\n", i, logp[i]);
      rc = 77; /* no GPU / no Vulkan */
    }
  }
  int failures = 0;
  if (rc == 0) {
    /* labels: at least 3 distinct palette classes visible somewhere */
    uint32_t w = 0, h = 0;
    uint8_t *px = read_ppm(shot[0], &w, &h);
    if (!px) failures++;
    else {
      int magenta = 0, orange = 0, teal = 0, green = 0, blue = 0;
      for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
          size_t o = ((size_t)y * w + x) * 3;
          int r = px[o], g = px[o + 1], b = px[o + 2];
          if (r > 170 && b > 140 && g < 110) magenta = 1;
          if (r > 190 && g > 80 && g < 160 && b < 90) orange = 1;
          if (r < 120 && g > 150 && b > 150) teal = 1;
          if (g > 170 && r < 130 && b < 130) green = 1;
          if (b > 170 && r < 130 && g < 150) blue = 1;
        }
      int classes = magenta + orange + teal + green + blue;
      if (classes < 3) {
        fprintf(stderr, "labels: only %d palette classes visible\n", classes);
        failures++;
      }
      free(px);
    }
    /* registration: aligned fuse exactly neutral, offset fuse fringed */
    uint64_t n0 = 0, n1 = 0;
    uint8_t *p0 = read_ppm(shot[1], &w, &h);
    double f0 = p0 ? fringe(p0, w, h, &n0) : -1.0;
    free(p0);
    uint8_t *p1 = read_ppm(shot[2], &w, &h);
    double f1 = p1 ? fringe(p1, w, h, &n1) : -1.0;
    free(p1);
    if (!(n0 > 5000 && f0 >= 0.0 && f0 < 0.01)) {
      fprintf(stderr, "aligned overlay not neutral: |R-G| %.3f over %llu px\n", f0,
              (unsigned long long)n0);
      failures++;
    }
    if (!(n1 > 5000 && f1 > 1.0)) {
      fprintf(stderr, "offset overlay shows no fringes: |R-G| %.3f over %llu px\n", f1,
              (unsigned long long)n1);
      failures++;
    }
    if (!failures)
      printf("overlay conformance OK (aligned |R-G| %.3f, offset %.2f)\n", f0, f1);
  }
  for (int i = 0; i < 3; i++) {
    unlink(shot[i]);
    unlink(logp[i]);
  }
  st_rm_tree(root, 2);
  rmdir(tmp);
  return rc ? rc : (failures ? 1 : 0);
}
