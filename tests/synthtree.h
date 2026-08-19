/* Synthetic c5d LOD tree builder shared by tests: a textured volume (waves
 * + blobs — gradient in every direction, near-lossless at q2) written as
 * real c5d shards + manifest.json, so cpuvol/regvol/the renderer operate on
 * it exactly like scroll data. Header-only; include from one TU per test. */
#ifndef R3D_TEST_SYNTHTREE_H
#define R3D_TEST_SYNTHTREE_H

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <brick.h> /* c5d (angle include: c5d src dir) */
#include <shard.h>

#define ST_B 128u /* brick edge */

/* >= 0: every voxel takes this value instead of the textured pattern —
 * constant volumes are the identity fixture for the display filters */
static int st_const_value = -1;
/* nonzero: bright spiral shells (rho = 30 + 14*w about (128,128), sheet
 * ~2 voxels thick) — a plausible surface-prediction volume for the tracer */
static int st_spiral_mode = 0;

static inline uint8_t st_pat(uint32_t x, uint32_t y, uint32_t z) {
  if (st_const_value >= 0) return (uint8_t)st_const_value;
  if (st_spiral_mode) {
    (void)z;
    double rho = hypot((double)x - 128.0, (double)y - 128.0);
    if (rho < 16.0) return 8; /* core: no sheet */
    double m = fmod(rho - 30.0, 14.0);
    if (m < 0.0) m += 14.0;
    double d = m < 7.0 ? m : 14.0 - m; /* radial distance to a wrap */
    return d < 1.5 ? 220 : 12;
  }
  double v = 120.0 + 70.0 * sin((double)x * 0.113) * sin((double)y * 0.131) *
                         sin((double)z * 0.171);
  double bx = (double)x - 90.0, by = (double)y - 140.0, bz = (double)z - 100.0;
  v += 80.0 * exp(-(bx * bx + by * by + bz * bz) / 900.0);
  bx = (double)x - 180.0;
  by = (double)y - 60.0;
  bz = (double)z - 170.0;
  v += 60.0 * exp(-(bx * bx + by * by + bz * bz) / 1600.0);
  if (v < 0.0) v = 0.0;
  if (v > 255.0) v = 255.0;
  return (uint8_t)(v + 0.5);
}

/* level-L voxel = mean of its (2^L)^3 base voxels (L in 0..1 supported) */
static inline uint8_t st_pat_lvl(uint32_t level, uint32_t x, uint32_t y, uint32_t z) {
  if (level == 0) return st_pat(x, y, z);
  uint32_t s = 0;
  for (uint32_t k = 0; k < 8; k++)
    s += st_pat(x * 2 + (k & 1), y * 2 + ((k >> 1) & 1), z * 2 + (k >> 2));
  return (uint8_t)((s + 4) / 8);
}

/* one 1024^3 shard covers up to 8^3 bricks: grids beyond that need more
 * shards, which the tests don't — keep nb* <= 8 */
static inline int st_write_level(const char *root, uint32_t level, const uint32_t nb[3],
                          bool content) {
  char dir[640], path[720], up[640];
  snprintf(up, sizeof up, "%s/c5d", root);
  snprintf(dir, sizeof dir, "%s/c5d/L%u", root, level);
  if ((mkdir(up, 0755) != 0 && errno != EEXIST) ||
      (mkdir(dir, 0755) != 0 && errno != EEXIST))
    return -1;
  snprintf(path, sizeof path, "%s/0_0_0.c5s", dir);
  c5d_shard_writer *w = c5d_shard_create(path, 1024, ST_B, level, 2.0f);
  if (!w) return -1;
  uint8_t *raw = malloc((size_t)ST_B * ST_B * ST_B);
  uint8_t *blob = NULL;
  int rc = raw ? 0 : -1;
  for (uint32_t bi = 0; bi < 512 && rc == 0; bi++) {
    uint32_t bx = bi & 7u, by = (bi >> 3) & 7u, bz = bi >> 6;
    if (!content || bx >= nb[0] || by >= nb[1] || bz >= nb[2]) {
      rc = c5d_shard_put_zero(w, bi);
      continue;
    }
    for (uint32_t z = 0; z < ST_B; z++)
      for (uint32_t y = 0; y < ST_B; y++)
        for (uint32_t x = 0; x < ST_B; x++)
          raw[((size_t)z * ST_B + y) * ST_B + x] =
              st_pat_lvl(level, bx * ST_B + x, by * ST_B + y, bz * ST_B + z);
    c5d_brick_params p = c5d_brick_defaults(2.0f);
    size_t n = 0;
    if (c5d_brick_encode(&p, raw, ST_B, &blob, &n) != 0) rc = -1;
    else {
      rc = c5d_shard_put(w, bi, blob, n);
      free(blob);
      blob = NULL;
    }
  }
  free(raw);
  if (c5d_shard_close(w) != 0) rc = -1;
  return rc;
}

/* dims in voxels (x, y, z), each a multiple of 128 covering <= 8 bricks per
 * axis; nlev 1 or 2 (level 1 halves the dims, rounding up per c5d-lod).
 * Levels below content_min are written all-zero — cheap trees whose only
 * real payload is the pinned coarsest level. */
static inline int st_make_tree(const char *root, const uint32_t dim[3], uint32_t nlev,
                        uint32_t content_min) {
  if (mkdir(root, 0755) != 0 && errno != EEXIST) return -1;
  char mp[640];
  snprintf(mp, sizeof mp, "%s/manifest.json", root);
  FILE *f = fopen(mp, "w");
  if (!f) return -1;
  fprintf(f,
          "{\n  \"format\": \"render3d.c5d-lod.v1\",\n"
          "  \"shape\": [%u, %u, %u],\n"
          "  \"shard_shape\": [1024, 1024, 1024],\n"
          "  \"brick_shape\": [128, 128, 128],\n  \"levels\": [\n",
          dim[2], dim[1], dim[0]);
  for (uint32_t l = 0; l < nlev; l++) {
    uint32_t d[3];
    for (int a = 0; a < 3; a++) d[a] = l ? (dim[a] + 1u) / 2u : dim[a];
    fprintf(f,
            "    {\"level\": %u, \"scale\": %u, \"shape\": [%u, %u, %u],"
            " \"shards\": [1, 1, 1], \"c5d\": \"c5d/L%u/{z}_{y}_{x}.c5s\"}%s\n",
            l, 1u << l, d[2], d[1], d[0], l, l + 1u < nlev ? "," : "");
  }
  fprintf(f, "  ]\n}\n");
  if (fclose(f) != 0) return -1;
  for (uint32_t l = 0; l < nlev; l++) {
    uint32_t d[3], nb[3];
    for (int a = 0; a < 3; a++) {
      d[a] = l ? (dim[a] + 1u) / 2u : dim[a];
      nb[a] = (d[a] + ST_B - 1u) / ST_B;
    }
    if (st_write_level(root, l, nb, l >= content_min) != 0) return -1;
  }
  return 0;
}

static inline void st_rm_tree(const char *root, uint32_t nlev) {
  char p[720];
  for (uint32_t l = 0; l < nlev; l++) {
    snprintf(p, sizeof p, "%s/c5d/L%u/0_0_0.c5s", root, l);
    unlink(p);
    snprintf(p, sizeof p, "%s/c5d/L%u", root, l);
    rmdir(p);
  }
  snprintf(p, sizeof p, "%s/c5d", root);
  rmdir(p);
  snprintf(p, sizeof p, "%s/manifest.json", root);
  unlink(p);
  snprintf(p, sizeof p, "%s/seed.raw", root);
  unlink(p); /* the renderer caches its coarsest-level decode here */
  rmdir(root);
}

#endif /* R3D_TEST_SYNTHTREE_H */
