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
#include "core/mview.h"
#include "core/segstore.h"
#include "core/segtrace.h"
#include "core/tifxyz.h"
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

#include <string.h>
#include <tiffio.h>

/* Write one plane the way vc3d/the AWS exports do: tiled BigTIFF, LZW,
 * floating-point predictor. */
static void tifxyz_write_plane(const char *path, const float *v, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, "w8");
  CHECK(tf != NULL);
  if (!tf) return;
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(tf, TIFFTAG_PREDICTOR, 3);
  TIFFSetField(tf, TIFFTAG_TILEWIDTH, 16);
  TIFFSetField(tf, TIFFTAG_TILELENGTH, 16);
  float tile[16 * 16];
  for (uint32_t ty = 0; ty < h; ty += 16)
    for (uint32_t tx = 0; tx < w; tx += 16) {
      memset(tile, 0, sizeof tile);
      for (uint32_t r = 0; r < 16 && ty + r < h; r++)
        for (uint32_t c = 0; c < 16 && tx + c < w; c++)
          tile[r * 16 + c] = v[(uint64_t)(ty + r) * w + tx + c];
      CHECK(TIFFWriteTile(tf, tile, tx, ty, 0, 0) >= 0);
    }
  TIFFClose(tf);
}

static void test_mview(void) {
  r3d_mview v[4] = {0};
  r3d_mv_layout_mask(v, 1281, 721, 0xf); /* odd pixels go right/bottom */
  CHECK(v[0].pw == 640 && v[0].ph == 360 && v[3].px == 640 && v[3].py == 360);
  CHECK(v[3].pw == 641 && v[3].ph == 361);
  CHECK(r3d_mv_hit(v, 1000.0f, 500.0f) == 3 && r3d_mv_hit(v, 10.0f, 10.0f) == 0);
  r3d_mv_layout_mask(v, 1280, 720, 1u << 2); /* solo XZ */
  CHECK(v[2].pw == 1280 && v[2].ph == 720 && v[0].pw == 0 && v[1].pw == 0);
  CHECK(r3d_mv_hit(v, 10.0f, 10.0f) == 2); /* hidden views never hit */
  r3d_mv_layout_mask(v, 1280, 720, 0xb); /* three visible: 0,1 top; 3 bottom */
  CHECK(v[0].ph == 360 && v[1].px == 640 && v[3].py == 360 && v[3].pw == 1280);
  CHECK(v[2].pw == 0);
  r3d_mv_layout_mask(v, 1280, 720, 0x6); /* two: 1 left, 2 right */
  CHECK(v[1].px == 0 && v[1].pw == 640 && v[2].px == 640 && v[2].ph == 720);
  /* zoom about a fixed point keeps it fixed */
  v[1].cu = 100.0;
  v[1].cv = 200.0;
  v[1].zoom = 1.0;
  double u0, w0;
  r3d_mv_unproject(&v[1], 400.0f, 300.0f, &u0, &w0);
  r3d_mv_zoom(&v[1], 400.0f, 300.0f, 2.0, 0.01, 100.0);
  double u1, w1;
  r3d_mv_unproject(&v[1], 400.0f, 300.0f, &u1, &w1);
  CHECK(fabs(u0 - u1) < 1e-9 && fabs(w0 - w1) < 1e-9 && v[1].zoom == 2.0);

  /* segment-aligned frames: orthonormal, both contain -n as screen-down,
   * pane normals are each other's horizontals, world<->frame round-trips */
  double n[3] = {0.36, 0.48, 0.8}, tref[3] = {1.0, 0.2, -0.1};
  double ba[3][3], bb[3][3];
  CHECK(r3d_mv_seg_frames(n, tref, 0.7, ba, bb) == 0);
  for (int f = 0; f < 2; f++) {
    double(*b)[3] = f ? bb : ba;
    for (int r = 0; r < 3; r++) {
      double l = b[r][0] * b[r][0] + b[r][1] * b[r][1] + b[r][2] * b[r][2];
      CHECK(fabs(l - 1.0) < 1e-12);
      for (int r2 = r + 1; r2 < 3; r2++)
        CHECK(fabs(b[r][0] * b[r2][0] + b[r][1] * b[r2][1] + b[r][2] * b[r2][2]) < 1e-12);
    }
    /* screen down = -n */
    CHECK(fabs(b[1][0] * n[0] + b[1][1] * n[1] + b[1][2] * n[2] + 1.0) < 1e-9);
  }
  for (int c = 0; c < 3; c++) {
    CHECK(fabs(ba[2][c] - bb[0][c]) < 1e-12); /* nA = uB */
    CHECK(fabs(bb[2][c] + ba[0][c]) < 1e-12); /* nB = -uA */
  }
  double o[3] = {10.0, 20.0, 30.0}, p3[3], ru, rv, rs;
  r3d_mv_b2w(ba, o, 3.0, -4.0, 5.0, p3);
  r3d_mv_w2b(ba, o, p3, &ru, &rv, &rs);
  CHECK(fabs(ru - 3.0) < 1e-9 && fabs(rv + 4.0) < 1e-9 && fabs(rs - 5.0) < 1e-9);
  /* degenerate tref (parallel to n) still yields a valid frame */
  double tpar[3] = {0.36, 0.48, 0.8};
  CHECK(r3d_mv_seg_frames(n, tpar, 0.0, ba, bb) == 0);
  double l0 = ba[0][0] * ba[0][0] + ba[0][1] * ba[0][1] + ba[0][2] * ba[0][2];
  CHECK(fabs(l0 - 1.0) < 1e-12);
}

typedef struct tr_acc {
  uint32_t n;
  float gi_min, gi_max, wu_min, wu_max;
} tr_acc;

static void tr_emit(void *ud, float wu0, float wv0, float wu1, float wv1, float gi0,
                    float gj0, float gi1, float gj1) {
  (void)wv0;
  (void)wv1;
  (void)gj0;
  (void)gj1;
  tr_acc *a = ud;
  a->n++;
  float gmin = gi0 < gi1 ? gi0 : gi1, gmax = gi0 < gi1 ? gi1 : gi0;
  float wmin = wu0 < wu1 ? wu0 : wu1, wmax = wu0 < wu1 ? wu1 : wu0;
  if (gmin < a->gi_min) a->gi_min = gmin;
  if (gmax > a->gi_max) a->gi_max = gmax;
  if (wmin < a->wu_min) a->wu_min = wmin;
  if (wmax > a->wu_max) a->wu_max = wmax;
}

static void test_tifxyz(void) {
  char dir[] = "/tmp/r3d_test_tifxyz_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  enum { W = 21, H = 13 };
  float px[W * H], py[W * H], pz[W * H];
  for (uint32_t j = 0; j < H; j++)
    for (uint32_t i = 0; i < W; i++) {
      uint64_t k = (uint64_t)j * W + i;
      px[k] = 100.0f + 20.0f * (float)i;
      py[k] = 200.0f + 20.0f * (float)j;
      pz[k] = 3000.0f + 0.25f * (float)i;
    }
  pz[5] = 0.0f;   /* unmapped: stored as zeros */
  px[5] = 0.0f;
  py[5] = 0.0f;
  pz[7] = -4.0f;  /* z <= 0 is invalid even with nonzero x/y */
  char path[300];
  snprintf(path, sizeof path, "%s/x.tif", dir);
  tifxyz_write_plane(path, px, W, H);
  snprintf(path, sizeof path, "%s/y.tif", dir);
  tifxyz_write_plane(path, py, W, H);
  snprintf(path, sizeof path, "%s/z.tif", dir);
  tifxyz_write_plane(path, pz, W, H);
  snprintf(path, sizeof path, "%s/meta.json", dir);
  FILE *f = fopen(path, "w");
  CHECK(f != NULL);
  if (f) {
    fputs("{\n  \"format\": \"tifxyz\",\n  \"scale\": [\n    0.05,\n    0.05\n  ],\n"
          "  \"type\": \"seg\"\n}\n", f);
    fclose(f);
  }

  r3d_tifxyz s;
  CHECK(r3d_tifxyz_load(&s, dir) == 0);
  CHECK(s.w == W && s.h == H);
  CHECK(fabsf(s.sx - 0.05f) < 1e-6f && fabsf(s.sy - 0.05f) < 1e-6f);
  CHECK(s.nvalid == W * H - 2);
  const float *p = r3d_tifxyz_at(&s, 2, 1);
  CHECK(r3d_tifxyz_valid(p));
  CHECK(p[0] == 140.0f && p[1] == 220.0f && fabsf(p[2] - 3000.5f) < 1e-3f);
  CHECK(!r3d_tifxyz_valid(r3d_tifxyz_at(&s, 5, 0))); /* zeros -> invalid */
  CHECK(!r3d_tifxyz_valid(r3d_tifxyz_at(&s, 7, 0))); /* z<0  -> invalid */
  CHECK(r3d_tifxyz_at(&s, 7, 0)[1] == -1.0f);        /* whole triplet forced */
  CHECK(s.bbox[0][0] == 100.0f && s.bbox[1][0] == 100.0f + 20.0f * (W - 1));
  CHECK(s.bbox[0][2] == 3000.0f);
  /* segtrace: the synthetic surface has z = 3000 + 0.25*i, so the z=3001.6
   * plane crosses at grid i = 6.4 — one cell-row segment per j except the
   * row poisoned by the invalid point at (7,0) */
  tr_acc acc = {0, 1e9f, -1e9f, 1e9f, -1e9f};
  uint32_t nseg = r3d_segtrace(&s, NULL, NULL, 0.0f, 2, 0, 1, 3001.6, tr_emit, &acc);
  CHECK(nseg == acc.n && nseg == H - 2);
  CHECK(acc.gi_min > 6.35f && acc.gi_max < 6.45f); /* crossing at i = 6.4 */
  CHECK(acc.wu_min > 227.9f && acc.wu_max < 228.1f); /* world x = 100 + 20*6.4 */

  /* row-bounds index: identical trace with skipping enabled, and a slice
   * outside every row's z range emits nothing */
  r3d_segrows rows;
  CHECK(r3d_segrows_build(&s, &rows) == 0);
  tr_acc acc2 = {0, 1e9f, -1e9f, 1e9f, -1e9f};
  uint32_t nseg2 = r3d_segtrace(&s, &rows, NULL, 0.0f, 2, 0, 1, 3001.6, tr_emit, &acc2);
  CHECK(nseg2 == nseg && acc2.n == acc.n);
  CHECK(fabsf(acc2.gi_min - acc.gi_min) < 1e-6f && fabsf(acc2.wu_max - acc.wu_max) < 1e-6f);
  tr_acc acc3 = {0, 1e9f, -1e9f, 1e9f, -1e9f};
  CHECK(r3d_segtrace(&s, &rows, NULL, 0.0f, 2, 0, 1, 9000.0, tr_emit, &acc3) == 0);

  /* arbitrary-basis segtrace: a unit-axis basis with a shifted origin must
   * reproduce the axis trace with (u, slice) offset by the origin */
  {
    double org[3] = {100.0, 0.0, 3000.0};
    double bu[3] = {1, 0, 0}, bv[3] = {0, 1, 0}, bn[3] = {0, 0, 1};
    tr_acc accb = {0, 1e9f, -1e9f, 1e9f, -1e9f};
    uint32_t nb = r3d_segtrace_basis(&s, &rows, NULL, 0.0f, org, bu, bv, bn, 1.6, tr_emit,
                                     &accb);
    CHECK(nb == nseg);
    CHECK(fabsf(accb.gi_min - acc.gi_min) < 1e-4f);
    CHECK(fabsf(accb.wu_min - (acc.wu_min - 100.0f)) < 1e-3f);
  }
  /* oblique basis: plane normal (1,0,1)/sqrt2 crosses the synthetic surface
   * (x = 100+20i, z = 3000+0.25i) at i = 6.4 for the matching slice; the
   * multi-axis dot bounds must not skip any crossing rows */
  {
    double rt = sqrt(0.5);
    double org[3] = {0, 0, 0};
    double bu[3] = {rt, 0, -rt}, bv[3] = {0, 1, 0}, bn[3] = {rt, 0, rt};
    double slice = (3100.0 + 20.25 * 6.4) * rt;
    tr_acc acco = {0, 1e9f, -1e9f, 1e9f, -1e9f};
    uint32_t no_ = r3d_segtrace_basis(&s, &rows, NULL, 0.0f, org, bu, bv, bn, slice,
                                      tr_emit, &acco);
    tr_acc acco2 = {0, 1e9f, -1e9f, 1e9f, -1e9f};
    uint32_t no2 = r3d_segtrace_basis(&s, NULL, NULL, 0.0f, org, bu, bv, bn, slice,
                                      tr_emit, &acco2);
    CHECK(no_ == H - 2 && no2 == no_); /* row/tile skipping changes nothing */
    CHECK(acco.gi_min > 6.35f && acco.gi_max < 6.45f);
  }
  r3d_segrows_free(&rows);
  CHECK(rows.mn == NULL);

  r3d_tifxyz_free(&s);
  CHECK(s.xyz == NULL);

  /* cleanup */
  /* segment store: pack the synthetic dir (lossless), reopen, query, load */
  char sdir[] = "/tmp/r3d_test_segstore_XXXXXX";
  CHECK(mkdtemp(sdir) != NULL);
  const char *sdirs[1] = {dir};
  CHECK(r3d_segstore_build(sdir, sdirs, 1, -1, false) == 1);
  r3d_segstore st;
  CHECK(r3d_segstore_open(&st, sdir) == 0);
  CHECK(st.n == 1 && st.segs[0].w == W && st.segs[0].h == H);
  CHECK(st.segs[0].nvalid == W * H - 2);
  CHECK(st.segs[0].bbox[0][0] == 100.0f && st.segs[0].bbox[1][1] == 200.0f + 20.0f * (H - 1));
  /* plane queries against the index alone: the z = 3001.6 plane crosses,
   * z = 9000 and an out-of-box ROI miss */
  double bnz[3] = {0, 0, 1}, roi_lo[3] = {0, 0, 0}, roi_hi[3] = {1e4, 1e4, 1e4};
  uint32_t qhit[4];
  CHECK(r3d_segstore_plane_query(&st, bnz, 3001.6, 0.0, roi_lo, roi_hi, qhit, 4) == 1);
  CHECK(qhit[0] == 0);
  CHECK(r3d_segstore_plane_query(&st, bnz, 9000.0, 0.0, NULL, NULL, NULL, 0) == 0);
  double far_lo[3] = {5000, 5000, 0}, far_hi[3] = {6000, 6000, 1e4};
  CHECK(r3d_segstore_plane_query(&st, bnz, 3001.6, 0.0, far_lo, far_hi, NULL, 0) == 0);
  /* oblique plane through the surface (multi-axis tile dot bounds) */
  double rt2 = sqrt(0.5);
  double bno[3] = {rt2, 0, rt2};
  CHECK(r3d_segstore_plane_query(&st, bno, (3100.0 + 20.25 * 6.4) * rt2, 0.0, NULL, NULL,
                                 NULL, 0) == 1);
  /* near query: on-surface point hits, a far point misses */
  double np_[3] = {140.0, 220.0, 3000.5}, farp[3] = {9000.0, 9000.0, 9000.0};
  CHECK(r3d_segstore_near_query(&st, np_, 10.0, qhit, 4) == 1 && qhit[0] == 0);
  CHECK(r3d_segstore_near_query(&st, farp, 100.0, NULL, 0) == 0);
  /* lossless load round-trips bit-exact vs the direct tifxyz load */
  r3d_tifxyz sl;
  CHECK(r3d_segstore_load(&st, 0, 1, &sl) == 0);
  CHECK(sl.w == W && sl.h == H && sl.nvalid == W * H - 2);
  {
    r3d_tifxyz ref;
    CHECK(r3d_tifxyz_load(&ref, dir) == 0);
    CHECK(memcmp(sl.xyz, ref.xyz, (size_t)W * H * 3 * sizeof(float)) == 0);
    CHECK(sl.sx == ref.sx && sl.sy == ref.sy);
    r3d_tifxyz_free(&ref);
  }
  r3d_tifxyz_free(&sl);
  /* decimated load: every 2nd point, halved scale, segtrace still works */
  r3d_tifxyz s2;
  CHECK(r3d_segstore_load(&st, 0, 2, &s2) == 0);
  CHECK(s2.w == (W + 1) / 2 && s2.h == (H + 1) / 2);
  CHECK(fabsf(s2.sx - 0.05f / 2.0f) < 1e-7f);
  const float *d00 = r3d_tifxyz_at(&s2, 1, 1); /* source point (2,2) */
  CHECK(d00[0] == 140.0f && d00[1] == 240.0f);
  tr_acc accd = {0, 1e9f, -1e9f, 1e9f, -1e9f};
  CHECK(r3d_segtrace(&s2, NULL, NULL, 0.0f, 2, 0, 1, 3001.6, tr_emit, &accd) > 0);
  CHECK(accd.wu_min > 227.0f && accd.wu_max < 229.0f); /* crossing near x=228 */
  r3d_tifxyz_free(&s2);
  /* stride-4 load rides the .tfx4 tier and must equal the full decode
   * subsampled (lossless store, same source points) */
  r3d_tifxyz s4;
  CHECK(r3d_segstore_load(&st, 0, 4, &s4) == 0);
  CHECK(s4.w == (W + 3) / 4 && s4.h == (H + 3) / 4);
  const float *d40 = r3d_tifxyz_at(&s4, 1, 1); /* source point (4,4) */
  CHECK(d40[0] == 180.0f && d40[1] == 280.0f);
  r3d_tifxyz_free(&s4);
  /* incremental rebuild with NO source dirs keeps the packed segment via
   * the manifest (sources deletable after packing) */
  CHECK(r3d_segstore_build(sdir, NULL, 0, -1, false) == 1); /* kept via manifest */
  r3d_segstore st2;
  CHECK(r3d_segstore_open(&st2, sdir) == 0);
  CHECK(st2.n == 1 && st2.segs[0].w == W && st2.segs[0].nvalid == W * H - 2);
  r3d_segstore_close(&st2);
  r3d_segstore_close(&st);
  CHECK(st.segs == NULL);
  {
    char spath[350];
    snprintf(spath, sizeof spath, "%s/segments.r3ds", sdir);
    unlink(spath);
    const char *base = strrchr(dir, '/') + 1;
    snprintf(spath, sizeof spath, "%s/%s.tfx", sdir, base);
    unlink(spath);
    snprintf(spath, sizeof spath, "%s/%s.tfx4", sdir, base);
    unlink(spath);
    rmdir(sdir);
  }

  static const char *nm[4] = {"x.tif", "y.tif", "z.tif", "meta.json"};
  for (int i = 0; i < 4; i++) {
    snprintf(path, sizeof path, "%s/%s", dir, nm[i]);
    unlink(path);
  }
  rmdir(dir);
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
  test_tifxyz();
  test_mview();
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  printf("test_quick: all ok\n");
  return EXIT_SUCCESS;
}
