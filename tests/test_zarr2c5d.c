/* zarr2c5d converter end-to-end (ctest label: quick — offline, local mirror).
 * Builds a local zarr v2 mirror (raw u1 chunks, dimension_separator "/"),
 * runs the actual converter binary, and verifies the produced c5d LOD tree
 * decodes back to the source voxels through cpuvol. Also asserts the
 * amplification guard: a legal-but-pathological 1024^3 chunk edge is
 * rejected instead of allocating gigabytes.
 * Usage: test_zarr2c5d <path-to-zarr2c5d> */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/cpuvol.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

#define ZB 128u
#define ZDIM 256u
#define CHUNK_BYTES ((size_t)ZB * ZB * ZB)

static uint8_t zpat(uint32_t x, uint32_t y, uint32_t z) {
  double v = 120.0 + 70.0 * sin((double)x * 0.113) * sin((double)y * 0.131) *
                         sin((double)z * 0.171);
  if (v < 0.0) v = 0.0;
  if (v > 255.0) v = 255.0;
  return (uint8_t)(v + 0.5);
}

static int write_zarray(const char *mirror, uint32_t edge, uint32_t chunk) {
  char p[700];
  snprintf(p, sizeof p, "%s/0", mirror);
  if (mkdir(p, 0755) != 0 && errno != EEXIST) return -1;
  snprintf(p, sizeof p, "%s/0/.zarray", mirror);
  FILE *f = fopen(p, "w");
  if (!f) return -1;
  fprintf(f,
          "{\"shape\": [%u, %u, %u], \"chunks\": [%u, %u, %u], \"dtype\": \"|u1\","
          "\"fill_value\": 0, \"order\": \"C\", \"filters\": null,"
          "\"dimension_separator\": \"/\", \"compressor\": null, \"zarr_format\": 2}\n",
          edge, edge, edge, chunk, chunk, chunk);
  return fclose(f) == 0 ? 0 : -1;
}

static int write_chunks(const char *mirror) {
  uint8_t *buf = malloc(CHUNK_BYTES);
  if (!buf) return -1;
  int rc = 0;
  for (uint32_t cz = 0; cz < 2 && rc == 0; cz++)
    for (uint32_t cy = 0; cy < 2 && rc == 0; cy++)
      for (uint32_t cx = 0; cx < 2 && rc == 0; cx++) {
        for (uint32_t z = 0; z < ZB; z++)
          for (uint32_t y = 0; y < ZB; y++)
            for (uint32_t x = 0; x < ZB; x++)
              buf[((size_t)z * ZB + y) * ZB + x] =
                  zpat(cx * ZB + x, cy * ZB + y, cz * ZB + z);
        char p[760];
        snprintf(p, sizeof p, "%s/0/%u", mirror, cz);
        if (mkdir(p, 0755) != 0 && errno != EEXIST) rc = -1;
        snprintf(p, sizeof p, "%s/0/%u/%u", mirror, cz, cy);
        if (rc == 0 && mkdir(p, 0755) != 0 && errno != EEXIST) rc = -1;
        snprintf(p, sizeof p, "%s/0/%u/%u/%u", mirror, cz, cy, cx);
        FILE *f = rc == 0 ? fopen(p, "wb") : NULL;
        if (!f || fwrite(buf, 1, CHUNK_BYTES, f) != CHUNK_BYTES) rc = -1;
        if (f && fclose(f) != 0) rc = -1;
      }
  free(buf);
  return rc;
}

static void rm_rf(const char *dir) { /* test-owned temp paths only */
  char cmd[900];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
  if (system(cmd) != 0) fprintf(stderr, "cleanup failed for %s\n", dir);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_zarr2c5d <zarr2c5d-binary>\n");
    return 77;
  }
  char tmp[512];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_z2c_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char mirror[600], out[600];
  snprintf(mirror, sizeof mirror, "%s/mirror", tmp);
  snprintf(out, sizeof out, "%s/out", tmp);
  CHECK(mkdir(mirror, 0755) == 0);
  CHECK(write_zarray(mirror, ZDIM, ZB) == 0);
  CHECK(write_chunks(mirror) == 0);
  char cmd[2048];
  snprintf(cmd, sizeof cmd, "%s %s %s --threads 2 --c5d-quality 2 --full-from 0 >%s/z.log 2>&1",
           argv[1], mirror, out, tmp);
  CHECK(system(cmd) == 0);
  /* the produced tree decodes back to the source voxels */
  r3d_cpuvol v;
  CHECK(r3d_cpuvol_open(&v, out, 16) == 0);
  if (v.nx == ZDIM) {
    static uint8_t got[ZB * ZB];
    r3d_cpuvol_read_block(&v, 0, 60, 33, 200, ZB, ZB, 1, got);
    double mad = 0;
    for (uint32_t y = 0; y < ZB; y++)
      for (uint32_t x = 0; x < ZB; x++)
        mad += fabs((double)got[(size_t)y * ZB + x] -
                    (double)zpat(60 + x, 33 + y, 200));
    mad /= (double)(ZB * ZB);
    CHECK(mad < 2.0); /* q2 lossy encode: near-lossless on a smooth field */
    r3d_cpuvol_close(&v);
  } else {
    CHECK(v.nx == ZDIM);
    r3d_cpuvol_close(&v);
  }
  /* pathological chunk edge: rejected up front, no multi-GiB assembly */
  char bad[600];
  snprintf(bad, sizeof bad, "%s/bad", tmp);
  CHECK(mkdir(bad, 0755) == 0);
  CHECK(write_zarray(bad, 2048, 1024) == 0);
  snprintf(cmd, sizeof cmd, "%s %s %s/badout --threads 1 >%s/bad.log 2>&1", argv[1], bad,
           tmp, tmp);
  CHECK(system(cmd) != 0);
  rm_rf(tmp);
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("zarr2c5d converter OK\n");
  return 0;
}
