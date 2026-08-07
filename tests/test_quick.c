/* CPU-only quick tests (ctest label: quick). No window, no GPU. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>
#include <unistd.h>

#include "core/camera.h"
#include "core/clip.h"
#include "core/lod.h"
#include "core/mathx.h"
#include "core/slab.h"
#include "core/stats.h"
#include "core/umbilicus.h"
#include "core/vslab.h"
#include "core/transfer.h"
#include "core/volume.h"

static int failures = 0;
#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      failures++;                                                    \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                \
  } while (0)

static int feq(float a, float b) { return fabsf(a - b) < 1e-6f; }

static void test_stats(void) {
  uint64_t values[100];
  for (uint32_t i = 0; i < 100; i++) values[i] = 100u - i; /* intentionally unsorted */
  r3d_stats_summary s;
  r3d_stats_summarize_values(values, 100, &s);
  CHECK(fabs(s.mean_ns - 50.5) < 1e-12);
  CHECK(s.p50_ns == 50 && s.p95_ns == 95 && s.p99_ns == 99 && s.max_ns == 100);
  s.max_ns = 123;
  r3d_stats_summarize_values(NULL, 0, &s);
  CHECK(s.mean_ns == 0.0 && s.max_ns == 0);
}

static void test_mathx(void) {
  r3d_v3 a = v3(1, 2, 3), b = v3(4, 5, 6);
  CHECK(feq(v3_dot(a, b), 32.0f));
  r3d_v3 c = v3_cross(v3(1, 0, 0), v3(0, 1, 0));
  CHECK(feq(c.x, 0) && feq(c.y, 0) && feq(c.z, 1));
  CHECK(feq(v3_len(v3(3, 4, 0)), 5.0f));
  r3d_v3 n = v3_norm(v3(0, 0, 9));
  CHECK(feq(n.z, 1.0f));
  CHECK(feq(v3_len(v3_norm(v3(0, 0, 0))), 0.0f)); /* zero vector stays zero */
  CHECK(feq(fclampf(5, 0, 1), 1.0f));
  CHECK(feq(flerpf(2, 4, 0.5f), 3.0f));
  r3d_v3 s = v3_add(v3_scale(a, 2.0f), v3_sub(b, a));
  CHECK(feq(s.x, 5) && feq(s.y, 7) && feq(s.z, 9));
}

static void test_lod(void) {
  CHECK(r3d_lod_pick(0.0f, 0.001f, 8192.0f, 8) == 0);
  CHECK(r3d_lod_pick(1.0f, 0.001f, 8192.0f, 8) == 3); /* 8.2 voxels/pixel */
  CHECK(r3d_lod_pick(100.0f, 0.01f, 8192.0f, 8) == 7); /* clamp coarsest */
  const uint32_t want[5] = {2, 3, 4, 1, 0};
  for (uint32_t i = 0; i < 5; i++) CHECK(r3d_lod_fallback(2, i, 5) == want[i]);
  CHECK(r3d_lod_fallback(2, 5, 5) == UINT32_MAX);
}

static void test_volume(void) {
  /* 4x3x2 volume with index-identifying values, via a temp file */
  char path[] = "/tmp/r3d_test_vol_XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  uint8_t data[4 * 3 * 2];
  for (unsigned i = 0; i < sizeof data; i++) data[i] = (uint8_t)(i * 3u);
  CHECK(write(fd, data, sizeof data) == (ssize_t)sizeof data);
  close(fd);

  r3d_volume v;
  CHECK(r3d_volume_open(&v, path, 4, 3, 2) == 0);
  CHECK(v.nbytes == sizeof data);
  CHECK(r3d_volume_at(&v, 0, 0, 0) == 0);
  CHECK(r3d_volume_at(&v, 3, 0, 0) == 9);      /* x-fastest */
  CHECK(r3d_volume_at(&v, 0, 1, 0) == 12);     /* +y skips nx */
  CHECK(r3d_volume_at(&v, 0, 0, 1) == 36);     /* +z skips nx*ny */
  CHECK(r3d_volume_at(&v, 3, 2, 1) == (uint8_t)(23 * 3));
  r3d_volume_close(&v);
  CHECK(v.voxels == NULL);

  /* wrong dims must be rejected */
  CHECK(r3d_volume_open(&v, path, 4, 3, 3) != 0);
  unlink(path);
}

static void test_camera(void) {
  r3d_camera c;
  r3d_camera_init(&c, v3(0, 0, 0));
  r3d_v3 r, u, f;
  r3d_camera_basis(&c, 2.0f, &r, &u, &f);
  /* yaw=0,pitch=0 looks +Z; right=-X? cross(fwd,up)=cross(+Z,+Y)=-X... check handedness */
  CHECK(feq(f.x, 0) && feq(f.y, 0) && feq(f.z, 1));
  CHECK(feq(v3_dot(r, f), 0.0f) && feq(v3_dot(u, f), 0.0f)); /* orthogonal basis */
  CHECK(feq(v3_len(u), tanf(c.fov_y * 0.5f)));               /* fov pre-scale */
  CHECK(feq(v3_len(r), 2.0f * tanf(c.fov_y * 0.5f)));        /* aspect pre-scale */
  /* pitch clamps */
  r3d_camera_look(&c, 0.0f, 10.0f);
  CHECK(c.pitch < 1.5708f);
  /* forward motion moves along view */
  r3d_camera_init(&c, v3(0, 0, 0));
  r3d_camera_move(&c, v3(0, 0, 1), 2.0f);
  CHECK(feq(c.pos.z, 2.0f) && feq(c.pos.x, 0.0f));
}

static void test_orbit(void) {
  r3d_camera c;
  r3d_camera_init(&c, v3(0, 0, 0));
  r3d_camera_orbit_set(&c, v3(0.5f, 0.5f, 0.5f), 2.0f);
  CHECK(feq(c.pos.x, 0.5f) && feq(c.pos.y, 0.5f) && feq(c.pos.z, -1.5f));
  r3d_camera_orbit_zoom(&c, 0.5f);
  CHECK(feq(c.dist, 1.0f) && feq(c.pos.z, -0.5f));
  /* quarter turn: camera ends on the -X side, same distance */
  r3d_camera_orbit_drag(&c, 3.14159265f / 2.0f, 0.0f);
  CHECK(feq(v3_len(v3_sub(c.pos, c.target)), 1.0f));
  CHECK(fabsf(c.pos.x - (-0.5f)) < 1e-5f && fabsf(c.pos.z - 0.5f) < 1e-5f);
  /* pan moves target and pos together */
  r3d_v3 before = v3_sub(c.pos, c.target);
  r3d_camera_orbit_pan(&c, v3(0, 1, 0), 0.25f);
  r3d_v3 after = v3_sub(c.pos, c.target);
  CHECK(feq(c.target.y, 0.75f) && feq(v3_len(v3_sub(after, before)), 0.0f));
}

static void test_transfer(void) {
  r3d_tf tf;
  uint8_t lut[256][4];
  CHECK(r3d_tf_preset(0, &tf) == 0);
  r3d_tf_build(&tf, lut);
  CHECK(lut[0][0] == 0 && lut[0][3] == 0);
  CHECK(lut[255][0] == 255 && lut[255][3] == 255);
  CHECK(lut[128][0] == 128); /* identity ramp */
  CHECK(r3d_tf_preset(UINT32_MAX, NULL) == 3); /* preset count */
  CHECK(r3d_tf_preset(1, &tf) == 1);
  r3d_tf_build(&tf, lut);
  CHECK(lut[20][3] == 0);   /* air transparent */
  CHECK(lut[255][3] == 255);
  /* monotone alpha between declared points */
  CHECK(lut[100][3] >= lut[60][3]);
}

static void test_umbilicus(void) {
  r3d_umbilicus u;
  r3d_umbilicus_init(&u);
  CHECK(r3d_umbilicus_set(&u, 20.25, 30.5, 100) == 0);
  CHECK(r3d_umbilicus_set(&u, 5, 6, 50) == 0);
  CHECK(r3d_umbilicus_set(&u, 21.25, 31.5, 100) == 0); /* replace */
  CHECK(u.count == 2 && u.points[0].z == 50 && u.points[1].z == 100);
  const r3d_umbilicus_point *p = r3d_umbilicus_find(&u, 100);
  CHECK(p && fabs(p->x - 21.25) < 1e-12 && fabs(p->y - 31.5) < 1e-12);
  CHECK(!r3d_umbilicus_remove(&u, 75));

  char path[] = "/tmp/r3d_test_umbilicus_XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  unlink(path); /* save creates the final file by atomic rename */
  CHECK(r3d_umbilicus_save(&u, path, "test-source", 300, 200, 100) == 0);
  CHECK(!u.dirty);
  r3d_umbilicus v;
  r3d_umbilicus_init(&v);
  CHECK(r3d_umbilicus_load(&v, path) == 0);
  CHECK(v.count == 2 && v.points[0].z == 50 && v.points[1].z == 100);
  CHECK(r3d_umbilicus_remove(&v, 50) && v.count == 1 && v.dirty);
  FILE *f = fopen(path, "w");
  CHECK(f != NULL);
  if (f) {
    fputs("[[7, 8, 9], {\"x\": 12, \"y\": 11, \"z\": 10}]\n", f);
    CHECK(fclose(f) == 0);
  }
  CHECK(r3d_umbilicus_load(&v, path) == 0);
  CHECK(v.count == 2 && v.points[0].x == 9 && v.points[0].y == 8 && v.points[0].z == 7);
  unlink(path);
  r3d_umbilicus_free(&v);
  r3d_umbilicus_free(&u);
}

static void test_slab(void) {
  r3d_slab_layout l;
  /* small cube: single tile */
  CHECK(r3d_slab_layout_init(&l, 1024, 1024, 1024, 32) == 0);
  CHECK(l.gx == 1 && l.gy == 1 && l.px == 1024 && l.py == 1024);
  CHECK(r3d_slab_tile_w(&l) == 1026);
  CHECK(r3d_slab_z0_max(&l) == 992);
  /* wide: 2x2 grid */
  CHECK(r3d_slab_layout_init(&l, 4092, 4092, 192, 32) == 0);
  CHECK(l.gx == 2 && l.gy == 2 && l.px == 2046 && r3d_slab_tile_w(&l) == 2048);
  /* 3-wide and 4x4 grids */
  CHECK(r3d_slab_layout_init(&l, 4093, 1024, 192, 32) == 0);
  CHECK(l.gx == 3 && l.gy == 1 && l.px == 1365);
  CHECK(r3d_slab_layout_init(&l, 8184, 8184, 48, 18) == 0);
  CHECK(l.gx == 4 && l.gy == 4 && l.px == 2046);
  /* 8x8 grid + overview downscale bump (one <=2046 overview texture) */
  CHECK(r3d_slab_layout_init(&l, 8184, 8184, 48, 18) == 0);
  CHECK(l.ovs == 4);
  CHECK(r3d_slab_layout_init(&l, 8185, 1024, 192, 32) == 0);
  CHECK(l.gx == 5 && l.ovs == 8);
  CHECK(r3d_slab_layout_init(&l, 16368, 16368, 48, 18) == 0);
  CHECK(l.gx == 8 && l.gy == 8 && l.px == 2046 && l.ovs == 8);
  /* 16x16 grid */
  CHECK(r3d_slab_layout_init(&l, 32736, 32736, 16, 10) == 0);
  CHECK(l.gx == 16 && l.gy == 16 && l.px == 2046 && l.ovs == 16);
  /* 22x22 grid: the whole 43k cross section */
  CHECK(r3d_slab_layout_init(&l, 43008, 43008, 16, 10) == 0);
  CHECK(l.gx == 22 && l.gy == 22 && l.px == 1955 && l.ovs == 32);
  /* too wide */
  CHECK(r3d_slab_layout_init(&l, 45013, 1024, 192, 32) == -1);
  /* bad ring */
  CHECK(r3d_slab_layout_init(&l, 1024, 1024, 16, 32) == -1);

  /* --- explicit base-tile cap (device maxImageDimension3D > 2048) --- */
  /* 3072^2 needs a 2x2 grid at the 2048 cap but fits ONE tile at 16384 */
  CHECK(r3d_slab_layout_init(&l, 3072, 3072, 96, 32) == 0);
  CHECK(l.gx == 2 && l.gy == 2 && l.px == 1536);
  CHECK(r3d_slab_layout_init_cap(&l, 3072, 3072, 96, 32, 16384) == 0);
  CHECK(l.gx == 1 && l.gy == 1 && l.px == 3072 && r3d_slab_tile_w(&l) == 3074);
  /* the overview scale is bound by R3D_SLAB_OV_TILE, NOT the base cap: the
   * shader's pyramid math hardcodes the 2046 payload, so a wider base tile
   * must not change ovs (16368/2046 = 8 either way) */
  CHECK(r3d_slab_layout_init(&l, 16368, 16368, 48, 18) == 0);
  uint32_t ovs_default = l.ovs;
  CHECK(r3d_slab_layout_init_cap(&l, 16368, 16368, 48, 18, 16384) == 0);
  CHECK(l.gx == 1 && l.gy == 1 && l.px == 16368);
  CHECK(l.ovs == ovs_default);
  /* a cap below 2048 still works (clamped-down device) */
  CHECK(r3d_slab_layout_init_cap(&l, 3072, 3072, 96, 32, 1024) == 0);
  CHECK(l.gx == 4 && l.gy == 4 && l.px == 768);
  /* degenerate caps rejected, not divided by zero */
  CHECK(r3d_slab_layout_init_cap(&l, 1024, 1024, 96, 32, 4) == -1);
  /* the 22-tile grid limit is still enforced against the supplied cap */
  CHECK(r3d_slab_layout_init_cap(&l, 43008, 1024, 192, 32, 1024) == -1);

  /* apron source mapping: tile 0 col 0 clamps to world 0; col 1 = world 0;
   * tile 1 col 0 duplicates world px-1 (tile 0's last payload col) */
  CHECK(r3d_slab_layout_init(&l, 4092, 4092, 192, 32) == 0);
  CHECK(r3d_slab_src_col(&l, 0, 0) == 0);
  CHECK(r3d_slab_src_col(&l, 0, 1) == 0);
  CHECK(r3d_slab_src_col(&l, 0, 2) == 1);
  CHECK(r3d_slab_src_col(&l, 0, 2047) == 2046); /* tile0 apron = tile1 col 0 */
  CHECK(r3d_slab_src_col(&l, 1, 0) == 2045);    /* tile1 apron = tile0 last */
  CHECK(r3d_slab_src_col(&l, 1, 1) == 2046);
  CHECK(r3d_slab_src_col(&l, 1, 2047) == 4091); /* clamped at volume edge */

  /* ring layers wrap */
  CHECK(r3d_slab_ring_layer(&l, 0) == 0);
  CHECK(r3d_slab_ring_layer(&l, 31) == 31);
  CHECK(r3d_slab_ring_layer(&l, 32) == 0);
  CHECK(r3d_slab_ring_layer(&l, 100) == 100 % 32);

  /* scroll ranges */
  uint32_t s0, s1;
  int full;
  r3d_slab_scroll_range(&l, -1, 0, &s0, &s1, &full); /* first upload */
  CHECK(full == 1 && s0 == 0 && s1 == 32);
  r3d_slab_scroll_range(&l, 10, 13, &s0, &s1, &full); /* +3 deeper */
  CHECK(full == 0 && s0 == 42 && s1 == 45);
  r3d_slab_scroll_range(&l, 13, 10, &s0, &s1, &full); /* -3 back */
  CHECK(full == 0 && s0 == 10 && s1 == 13);
  r3d_slab_scroll_range(&l, 10, 10, &s0, &s1, &full); /* no-op */
  CHECK(full == 0 && s0 == s1);
  r3d_slab_scroll_range(&l, 10, 100, &s0, &s1, &full); /* jump */
  CHECK(full == 1 && s0 == 100 && s1 == 132);
}

static void test_vslab(void) {
  r3d_vslab v;
  CHECK(r3d_vslab_init(&v, 43008, 43008, 68608, 12096, 12096, 16) == 0);
  /* payload adapts: 12096/8 rounded up to 32 -> 1536; + straddle + prefetch */
  CHECK(v.wz == 18 && v.px == 1536 && v.lv[0].gx == 10 && v.lv[0].gy == 10);
  /* pyramid present only while window/s >= 1024: L1, L2 yes; L3, L4 no */
  CHECK(v.lv[1].gx == 4 && v.lv[2].gx == 3 && v.lv[3].gx == 0 && v.lv[4].gx == 0);
  CHECK(v.ntiles == 100 + 16 + 9);
  CHECK(r3d_vs_cell(&v, 0) == 1536 && r3d_vs_cell(&v, 2) == 1536 * 8);
  int64_t c0, c1;
  r3d_vs_range(&v, 0, 10000, 43008, 12096, &c0, &c1);
  CHECK(c0 == 6 && c1 == 14); /* 10000/1536=6, 22095/1536=14 */
  r3d_vs_range(&v, 0, 43008 - 12096, 43008, 12096, &c0, &c1);
  CHECK(c1 == 27); /* clamped at the last cell: 43007/1536 */
  CHECK(r3d_vs_phys(10, 7) == 3);
  /* deep cube window: single ring image (D <= 2044), no pyramid needed */
  CHECK(r3d_vslab_init(&v, 43008, 43008, 68608, 2048, 2048, 2044) == 0);
  CHECK(v.px == 256 && v.wz == 2046 && v.lv[0].gx == 10);
  CHECK(v.lv[1].gx == 0 && v.ntiles == 100);
  /* whole-xy thin plane: immovable axes drop straddle + prefetch entirely */
  CHECK(r3d_vslab_init(&v, 43008, 43008, 68608, 43008, 43008, 4) == 0);
  CHECK(v.px == 2016 && v.wz == 6 && v.lv[0].gx == 22 && v.lv[0].gy == 22);
  CHECK(v.lv[1].gx == 6 && v.lv[4].gx == 1);
  CHECK(v.ntiles == 484 + 36 + 9 + 4 + 1);
  CHECK(r3d_vslab_init(&v, 43008, 43008, 68608, 12096, 12096, 2045) != 0); /* too deep */
  CHECK(r3d_vslab_init(&v, 43008, 43008, 68608, 43009, 4096, 16) != 0);    /* > volume */
  CHECK(r3d_vs_key(3, 5) == (3u | (5u << 10) | 0x80000000u));
  CHECK(r3d_vs_layer(&v, 34288) == 34288 % 6); /* v = whole-xy window, wz 6 */
  CHECK(r3d_vs_max0(68608, 18) == 68590);
  /* Visible depth stays 8 while a symmetric 16-slice GPU margin expands the
   * ring to 42; near a volume face the unavailable margin shifts across. */
  CHECK(r3d_vslab_init_margin(&v, 8192, 8192, 23552, 8192, 8192, 8, 16) == 0);
  CHECK(v.D == 8 && v.z_margin == 16 && v.wz == 42);
  int64_t za, zb;
  r3d_vs_zrange(&v, 100, &za, &zb);
  CHECK(za == 84 && zb == 126);
  r3d_vs_zrange(&v, 0, &za, &zb);
  CHECK(za == 0 && zb == 42);
  r3d_vs_zrange(&v, 23544, &za, &zb);
  CHECK(za == 23510 && zb == 23552);
  CHECK(r3d_vslab_init_margin(&v, 8192, 8192, 23552, 8192, 8192, 8, 1020) != 0);
}

static void test_clip(void) {
  r3d_clip c;
  r3d_clip_init(&c, 43008, 43008, 68608, 32, 21504, 21504);
  CHECK(c.lv[0].s == 1 && c.lv[5].s == 32);
  CHECK(r3d_clip_coverage(&c, 0) == 2046);
  CHECK(r3d_clip_coverage(&c, 5) == 65472); /* > whole 43k cross-section */
  /* ring depths: window/s + 3, min 4 */
  CHECK(c.lv[0].wzl == 35 && c.lv[3].wzl == 7 && c.lv[5].wzl == 4);
  /* origins snapped to 16*s and centered-ish on focus */
  CHECK(c.lv[0].ox % 16 == 0 && c.lv[2].ox % 64 == 0);
  CHECK(c.lv[0].ox <= 21504 - 900 && c.lv[0].ox >= 21504 - 1200);
  /* L5 covers everything -> clamped to 0, never recenters */
  CHECK(c.lv[5].ox == 0 && c.lv[5].oy == 0);
  CHECK(!r3d_clip_need_recenter(&c, 5, 0, 0));
  /* containment + recenter policy at L0 */
  CHECK(r3d_clip_contains(&c, 0, 21504, 21504, 1));
  CHECK(!r3d_clip_contains(&c, 0, 21504 + 2000, 21504, 1));
  CHECK(!r3d_clip_need_recenter(&c, 0, 21504 + 400, 21504));
  CHECK(r3d_clip_need_recenter(&c, 0, 21504 + 600, 21504));
  /* recenter clamps at the volume edge */
  r3d_clip_recenter(&c, 1, 0, 0);
  CHECK(c.lv[1].ox == 0 && c.lv[1].oy == 0 && !c.lv[1].valid);
  r3d_clip_recenter(&c, 1, 43008, 43008);
  CHECK(c.lv[1].ox + (int64_t)r3d_clip_coverage(&c, 1) <= 43008);
  /* ring layers */
  CHECK(r3d_clip_ring_layer(&c, 0, 35) == 0);
  CHECK(r3d_clip_ring_layer(&c, 5, 1056) == 1056 % 4);
}

int main(void) {
  test_mathx();
  test_lod();
  test_stats();
  test_slab();
  test_clip();
  test_volume();
  test_vslab();
  test_camera();
  test_orbit();
  test_transfer();
  test_umbilicus();
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  printf("test_quick: all ok\n");
  return EXIT_SUCCESS;
}
