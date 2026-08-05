/* CPU-only quick tests (ctest label: quick). No window, no GPU. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>
#include <unistd.h>

#include "core/camera.h"
#include "core/clip.h"
#include "core/mathx.h"
#include "core/slab.h"
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
  CHECK(v.wz == 18 && v.lv[0].gx == 7 && v.lv[0].gy == 7); /* 6 cells + straddle */
  CHECK(v.lv[1].gx == 3 && v.lv[4].gx == 2);               /* s=4, s=32 */
  CHECK(v.ntiles == 49 + 9 + 4 + 4 + 4);
  CHECK(r3d_vs_cell(&v, 0) == 2016 && r3d_vs_cell(&v, 4) == 2016 * 32);
  int64_t c0, c1;
  r3d_vs_range(&v, 0, 10000, 43008, 12096, &c0, &c1);
  CHECK(c0 == 4 && c1 == 10); /* 10000/2016=4, 22095/2016=10 -> 7 cells */
  r3d_vs_range(&v, 0, 43008 - 12096, 43008, 12096, &c0, &c1);
  CHECK(c1 == 21); /* clamped at the last cell */
  CHECK(r3d_vs_phys(10, 7) == 3);
  CHECK(r3d_vs_key(3, 5) == (3u | (5u << 10) | 0x80000000u));
  CHECK(r3d_vs_layer(&v, 34288) == 34288 % 18);
  CHECK(r3d_vs_max0(68608, 18) == 68590);
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
  test_slab();
  test_clip();
  test_volume();
  test_vslab();
  test_camera();
  test_orbit();
  test_transfer();
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  printf("test_quick: all ok\n");
  return EXIT_SUCCESS;
}
