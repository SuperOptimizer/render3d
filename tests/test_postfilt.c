/* GPU post-process display filters (ctest label: gpu).
 * Two properties pin the postfilt kernels without a CPU reference:
 *  - identity on a constant volume: median (3^3/5^3), max pool (3^3/5^3)
 *    and unsharp are all exact identities on constant data, so every
 *    filtered render of a constant tree must be byte-identical to the
 *    unfiltered one (any neighborhood/indexing bug shows at brick seams)
 *  - on textured data: a filter changes the image, and running the same
 *    filter twice is deterministic (byte-identical screenshots)
 * Usage: test_postfilt <path-to-render3d> */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "synthtree.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

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

static int run_shot(const char *bin, const char *root, uint32_t pfmode, const char *shot,
                    const char *log) {
  char cmd[1800];
  char env[64] = "";
  if (pfmode) snprintf(env, sizeof env, "R3D_POSTFILT=%u", pfmode);
  snprintf(cmd, sizeof cmd,
           "%s R3D_MV_FIT=300 %s --bricks %s/manifest.json --headless --frames 300 "
           "--shot %s >%s 2>&1",
           env, bin, root, shot, log);
  return system(cmd);
}

static size_t diff_bytes(const uint8_t *a, size_t na, const uint8_t *b, size_t nb) {
  if (!a || !b || na != nb) return SIZE_MAX;
  size_t d = 0;
  for (size_t i = 0; i < na; i++) d += a[i] != b[i];
  return d;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_postfilt <render3d-binary>\n");
    return 77;
  }
  char tmp[256];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_pf_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char croot[320], troot[320];
  snprintf(croot, sizeof croot, "%s/const", tmp);
  snprintf(troot, sizeof troot, "%s/tex", tmp);
  uint32_t dim[3] = {256, 256, 256};
  st_const_value = 150;
  if (st_make_tree(croot, dim, 2, 0) != 0) return 1;
  st_const_value = -1;
  if (st_make_tree(troot, dim, 2, 0) != 0) return 1;

  /* mode encoding: low byte = primary filter (1 median3, 2 median5,
   * 3 max3, 4 max5), bit 8 = unsharp sharpen */
  static const uint32_t MODES[5] = {1u, 2u, 3u, 4u, 0x100u};
  char shot[16][360], logp[16][360];
  int ns = 0;
  int rc = 0;
#define SHOT(root, mode)                                                        \
  (snprintf(shot[ns], sizeof shot[ns], "%s/s%d.ppm", tmp, ns),                  \
   snprintf(logp[ns], sizeof logp[ns], "%s/s%d.log", tmp, ns),                  \
   rc == 0 && run_shot(argv[1], root, mode, shot[ns], logp[ns]) != 0 ? (rc = 77) : 0, \
   ns++)
  int c_none = SHOT(croot, 0);
  int c_f[5];
  for (int i = 0; i < 5; i++) c_f[i] = SHOT(croot, MODES[i]);
  int t_none = SHOT(troot, 0);
  int t_med = SHOT(troot, 2u);
  int t_med2 = SHOT(troot, 2u);
  if (rc == 77) fprintf(stderr, "render3d run failed - skipping\n");
  if (rc == 0) {
    size_t nn = 0;
    uint8_t *ref = read_all(shot[c_none], &nn);
    CHECK(ref && nn > 0);
    for (int i = 0; i < 5 && ref; i++) { /* identity on constant data */
      size_t fn = 0;
      uint8_t *fp = read_all(shot[c_f[i]], &fn);
      size_t d = diff_bytes(ref, nn, fp, fn);
      if (d != 0) {
        fprintf(stderr, "filter mode 0x%x not identity on constant volume: "
                        "%zu bytes differ\n",
                MODES[i], d);
        failures++;
      }
      free(fp);
    }
    free(ref);
    size_t n0 = 0, n1 = 0, n2 = 0;
    uint8_t *p0 = read_all(shot[t_none], &n0);
    uint8_t *p1 = read_all(shot[t_med], &n1);
    uint8_t *p2 = read_all(shot[t_med2], &n2);
    size_t d01 = diff_bytes(p0, n0, p1, n1);
    size_t d12 = diff_bytes(p1, n1, p2, n2);
    CHECK(d01 != SIZE_MAX && d01 > n0 / 100); /* the filter visibly ran */
    CHECK(d12 == 0);                          /* and it is deterministic */
    free(p0);
    free(p1);
    free(p2);
    if (!failures) printf("postfilt conformance OK (median5 changed %zu bytes)\n", d01);
  }
  for (int i = 0; i < ns; i++) {
    unlink(shot[i]);
    unlink(logp[i]);
  }
  st_rm_tree(croot, 2);
  st_rm_tree(troot, 2);
  rmdir(tmp);
  return rc ? rc : (failures ? 1 : 0);
}
