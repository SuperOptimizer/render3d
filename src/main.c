/* render3d — volumetric renderer for Vesuvius Challenge micro-CT volumes.
 * M1: SDL3 window + Vulkan compute raycaster (see spec/ and docs/measured.md). */
#include <SDL3/SDL.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cimgui.h"
#include "core/camera.h"
#include "core/transfer.h"
#include "core/volume.h"
#include "core/input.h"
#include "core/mview.h"
#include "core/screenshot.h"
#include "core/segtrace.h"
#include "core/stats.h"
#include "core/tifxyz.h"
#include "core/umbilicus.h"
#include "render/render.h"
#include "vk/vkctx.h"

#ifndef R3D_SPV_DIR
#define R3D_SPV_DIR "spv" /* release fallback: exe-relative */
#endif
#ifndef R3D_C5D_REV
#define R3D_C5D_REV "unknown"
#endif

#define MOUSE_SENS 0.0025f
#define ORBIT_SENS 0.006f
#define BASE_SPEED 0.4f /* volume units per second */

enum { CAM_ORBIT = 0, CAM_FLY = 1 };

static const char PHERC1218_SOURCE[] =
    "https://dl.ash2txt.org/community-uploads/forrest/exports/PHerc1218/"
    "20250521120456-8.640um-1.2m-116keV-masked.zarr";
static const char PHERC1218_SHARDS[] =
    "https://dl.ash2txt.org/community-uploads/forrest/exports/PHerc1218/"
    "20250521120456-8.640um-1.2m-116keV-masked.zarr/0/c";
enum { PHERC1218_NX = 8192, PHERC1218_NY = 8192, PHERC1218_NZ = 23552 };

static int annotation_z0(int z, uint32_t nz, uint32_t depth) {
  int64_t z0 = (int64_t)z - (int64_t)depth / 2;
  int64_t max = (int64_t)nz - depth;
  if (max < 0) max = 0;
  if (z0 < 0) z0 = 0;
  if (z0 > max) z0 = max;
  return (int)z0;
}

static bool annotation_pick(const r3d_camera *cam, r3d_v3 right, r3d_v3 up, r3d_v3 fwd,
                            r3d_m3 model, r3d_v3 vol_t, float sx, float sy, int view_w,
                            int view_h, uint32_t W, uint32_t H, uint32_t D, int64_t x0,
                            int64_t y0, int64_t z0, int z, double *x, double *y) {
  if (view_w <= 0 || view_h <= 0 || !W || !H || !D || !x || !y) return false;
  float ndcx = 2.0f * (sx + 0.5f) / (float)view_w - 1.0f;
  float ndcy = 1.0f - 2.0f * (sy + 0.5f) / (float)view_h;
  r3d_v3 dir = v3_norm(v3_add(fwd, v3_add(v3_scale(right, ndcx), v3_scale(up, ndcy))));
  float ey = (float)H / (float)W, ez = (float)D / (float)W;
  r3d_v3 center = v3(0.5f, ey * 0.5f, ez * 0.5f);
  r3d_v3 ro = v3_add(m3_tmul(model, v3_sub(v3_sub(cam->pos, vol_t), center)), center);
  r3d_v3 rd = m3_tmul(model, dir);
  if (fabsf(rd.z) < 1e-8f) return false;
  float plane = (float)((int64_t)z - z0) / (float)W;
  float t = (plane - ro.z) / rd.z;
  if (t < 0.0f) return false;
  r3d_v3 p = v3_add(ro, v3_scale(rd, t));
  if (p.x < 0.0f || p.x >= 1.0f || p.y < 0.0f || p.y >= ey) return false;
  *x = (double)x0 + (double)p.x * W;
  *y = (double)y0 + (double)p.y * W;
  return *x >= 0.0 && *x < (double)W && *y >= 0.0 && *y < (double)H;
}

static bool annotation_project(const r3d_umbilicus_point *point, const r3d_camera *cam,
                               r3d_v3 right, r3d_v3 up, r3d_v3 fwd, r3d_m3 model,
                               r3d_v3 vol_t, int view_w, int view_h, uint32_t W, uint32_t H,
                               uint32_t D, int64_t x0, int64_t y0, int64_t z0, ImVec2 *screen) {
  if (!point || !screen || view_w <= 0 || view_h <= 0) return false;
  float ey = (float)H / (float)W, ez = (float)D / (float)W;
  r3d_v3 center = v3(0.5f, ey * 0.5f, ez * 0.5f);
  r3d_v3 p = v3((float)(point->x - (double)x0) / (float)W,
                (float)(point->y - (double)y0) / (float)W,
                (float)(point->z - (double)z0) / (float)W);
  r3d_v3 world = v3_add(v3_add(m3_mul(model, v3_sub(p, center)), center), vol_t);
  r3d_v3 d = v3_sub(world, cam->pos);
  float rz = v3_dot(d, fwd), rl = v3_len(right), ul = v3_len(up);
  if (rz <= 0.0f || rl <= 0.0f || ul <= 0.0f) return false;
  float nx = v3_dot(d, v3_scale(right, 1.0f / rl)) / (rz * rl);
  float ny = v3_dot(d, v3_scale(up, 1.0f / ul)) / (rz * ul);
  screen->x = (nx + 1.0f) * 0.5f * (float)view_w;
  screen->y = (1.0f - ny) * 0.5f * (float)view_h;
  return screen->x >= 0.0f && screen->x < (float)view_w && screen->y >= 0.0f &&
         screen->y < (float)view_h;
}

static int save_umbilicus(r3d_umbilicus *u, const char *path, char status[256]) {
  if (r3d_umbilicus_save(u, path, PHERC1218_SOURCE, PHERC1218_NZ, PHERC1218_NY,
                         PHERC1218_NX) != 0) {
    snprintf(status, 256, "save failed: %s", strerror(errno));
    return -1;
  }
  snprintf(status, 256, "saved %zu point%s", u->count, u->count == 1 ? "" : "s");
  return 0;
}

/* Cached segment/plane intersection polylines (world uv + grid ij pairs). */
typedef struct mv_lines {
  float *w, *g; /* 4 floats per segment each */
  uint32_t n, cap;
} mv_lines;

static void mv_lines_emit(void *ud, float wu0, float wv0, float wu1, float wv1, float gi0,
                          float gj0, float gi1, float gj1) {
  mv_lines *l = ud;
  if (l->n == l->cap) {
    uint32_t nc = l->cap ? l->cap * 2 : 4096;
    float *nw = realloc(l->w, (size_t)nc * 4 * sizeof *nw);
    if (nw) l->w = nw;
    float *ng = realloc(l->g, (size_t)nc * 4 * sizeof *ng);
    if (ng) l->g = ng;
    if (!nw || !ng) return; /* drop segments on OOM */
    l->cap = nc;
  }
  float *w4 = l->w + (size_t)l->n * 4, *g4 = l->g + (size_t)l->n * 4;
  w4[0] = wu0;
  w4[1] = wv0;
  w4[2] = wu1;
  w4[3] = wv1;
  g4[0] = gi0;
  g4[1] = gj0;
  g4[2] = gi1;
  g4[3] = gj1;
  l->n++;
}

/* Build the surf-view GPU grids from a tifxyz segment: RGBA32F coords
 * (w = valid) and per-vertex normals (bilinear-tangent cross product, vc3d
 * grid_normal). Returns malloc'd w*h*4 float pairs via out params. */
static int mv_build_grids(const r3d_tifxyz *s, float **coords_out, float **normals_out) {
  uint64_t n = (uint64_t)s->w * s->h;
  float *co = malloc(n * 4 * sizeof *co), *no = calloc(n * 4, sizeof *no);
  if (!co || !no) {
    free(co);
    free(no);
    return -1;
  }
  for (uint64_t k = 0; k < n; k++) {
    const float *p = s->xyz + k * 3;
    bool ok = r3d_tifxyz_valid(p);
    co[k * 4 + 0] = p[0];
    co[k * 4 + 1] = p[1];
    co[k * 4 + 2] = p[2];
    co[k * 4 + 3] = ok ? 1.0f : 0.0f;
  }
  for (uint32_t j = 0; j < s->h; j++)
    for (uint32_t i = 0; i < s->w; i++) {
      uint64_t k = (uint64_t)j * s->w + i;
      if (co[k * 4 + 3] < 0.5f) continue;
      /* central differences, shrinking to one-sided at edges/invalid */
      uint32_t i0 = i > 0 ? i - 1 : i, i1 = i + 1 < s->w ? i + 1 : i;
      uint32_t j0 = j > 0 ? j - 1 : j, j1 = j + 1 < s->h ? j + 1 : j;
      const float *pu0 = r3d_tifxyz_at(s, i0, j), *pu1 = r3d_tifxyz_at(s, i1, j);
      const float *pv0 = r3d_tifxyz_at(s, i, j0), *pv1 = r3d_tifxyz_at(s, i, j1);
      const float *pc_ = r3d_tifxyz_at(s, i, j);
      if (!r3d_tifxyz_valid(pu0)) pu0 = pc_;
      if (!r3d_tifxyz_valid(pu1)) pu1 = pc_;
      if (!r3d_tifxyz_valid(pv0)) pv0 = pc_;
      if (!r3d_tifxyz_valid(pv1)) pv1 = pc_;
      r3d_v3 tu = v3(pu1[0] - pu0[0], pu1[1] - pu0[1], pu1[2] - pu0[2]);
      r3d_v3 tv = v3(pv1[0] - pv0[0], pv1[1] - pv0[1], pv1[2] - pv0[2]);
      r3d_v3 nn = v3_cross(tu, tv);
      float l = v3_len(nn);
      if (l < 1e-6f) continue;
      no[k * 4 + 0] = nn.x / l;
      no[k * 4 + 1] = nn.y / l;
      no[k * 4 + 2] = nn.z / l;
    }
  *coords_out = co;
  *normals_out = no;
  return 0;
}

/* first voxel value with nonzero TF alpha: below it, samples are invisible */
static float tf_min_visible(const uint8_t lut[256][4]) {
  for (uint32_t i = 0; i < 256; i++)
    if (lut[i][3] != 0) return (float)i;
  return 255.0f;
}

static void gui_event_hook(void *ud, const SDL_Event *ev) {
  r3d_gui_event((r3d_renderer *)ud, ev);
}

static void take_screenshot(r3d_renderer *renderer, uint64_t frame) {
  uint32_t w = 0, h = 0;
  if (r3d_read_frame(renderer, NULL, &w, &h) != 0) return;
  uint8_t *rgba = malloc((size_t)w * h * 4);
  if (!rgba) return;
  if (r3d_read_frame(renderer, rgba, &w, &h) == 0) {
    char path[64];
    snprintf(path, sizeof path, "render3d_%llu.ppm", (unsigned long long)frame);
    if (r3d_screenshot_ppm(path, rgba, w, h) == 0)
      printf("screenshot: %s (%ux%u)\n", path, w, h);
  }
  free(rgba);
}

static void profile_summary(const r3d_frame_stats *samples, uint64_t n, size_t field,
                            r3d_stats_summary *out) {
  memset(out, 0, sizeof *out);
  if (!samples || n == 0 || n > UINT32_MAX) return;
  uint64_t *v = malloc((size_t)n * sizeof *v);
  if (!v) return;
  for (uint64_t i = 0; i < n; i++) v[i] = ((const uint64_t *)&samples[i])[field];
  r3d_stats_summarize_values(v, (uint32_t)n, out);
  free(v);
}

static void json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (; s && *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
    else if (c == '\n') fputs("\\n", f);
    else if (c >= 0x20) fputc(c, f);
  }
  fputc('"', f);
}

static void json_timing(FILE *f, const char *name, const r3d_stats_summary *s, bool comma) {
  fprintf(f, "    \"%s\": {\"mean_ms\": %.6f, \"p50_ms\": %.6f, "
             "\"p95_ms\": %.6f, \"p99_ms\": %.6f, \"max_ms\": %.6f}%s\n",
          name, s->mean_ns / 1e6, (double)s->p50_ns / 1e6, (double)s->p95_ns / 1e6,
          (double)s->p99_ns / 1e6, (double)s->max_ns / 1e6, comma ? "," : "");
}

static int write_bench_json(const char *path, const char *scenario, int width, int height,
                            const char *quality, uint32_t warmup, const r3d_stats *stats,
                            const r3d_frame_stats *samples, uint64_t nsamples,
                            uint64_t pending_cell_frames, const r3d_bricks_stats *bricks) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "benchmark: cannot write %s\n", path);
    return -1;
  }
  r3d_stats_summary timing[10];
  r3d_stats_summarize(stats, &timing[0], &timing[1]);
  for (size_t i = 0; i < 8; i++) profile_summary(samples, nsamples, i, &timing[i + 2]);
  fputs("{\n  \"schema\": \"render3d-benchmark-v1\",\n  \"scenario\": ", f);
  json_string(f, scenario ? scenario : "static");
  fputs(",\n  \"quality\": ", f);
  json_string(f, quality);
  fprintf(f, ",\n  \"width\": %d,\n  \"height\": %d,\n  \"warmup_frames\": %u,\n"
             "  \"measured_frames\": %llu,\n  \"retained_frame_samples\": %u,\n"
             "  \"c5d_revision\": ",
          width, height, warmup, (unsigned long long)nsamples, stats->count);
  json_string(f, R3D_C5D_REV);
  fprintf(f, ",\n  \"pending_cell_frames\": %llu,\n"
             "  \"brick_stream\": {\"decoded\": %llu, \"jobs\": %llu, "
             "\"failures\": %u, \"mean_job_ms\": %.6f,\n"
             "    \"levels\": %u,\n"
             "    \"lod_wanted\": [%u, %u, %u, %u, %u, %u, %u, %u],\n"
             "    \"lod_requests\": [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu]\n"
             "  },\n  \"timings\": {\n",
          (unsigned long long)pending_cell_frames,
          (unsigned long long)(bricks ? bricks->decoded : 0),
          (unsigned long long)(bricks ? bricks->jobs : 0), bricks ? bricks->failures : 0,
          bricks && bricks->jobs ? (double)bricks->stream_ns / (double)bricks->jobs / 1e6 : 0.0,
          bricks ? bricks->nlevels : 0, bricks ? bricks->lod_wanted[0] : 0,
          bricks ? bricks->lod_wanted[1] : 0, bricks ? bricks->lod_wanted[2] : 0,
          bricks ? bricks->lod_wanted[3] : 0, bricks ? bricks->lod_wanted[4] : 0,
          bricks ? bricks->lod_wanted[5] : 0, bricks ? bricks->lod_wanted[6] : 0,
          bricks ? bricks->lod_wanted[7] : 0,
          (unsigned long long)(bricks ? bricks->lod_requests[0] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[1] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[2] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[3] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[4] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[5] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[6] : 0),
          (unsigned long long)(bricks ? bricks->lod_requests[7] : 0));
  static const char *names[10] = {"cpu_frame", "gpu_frame", "gpu_total", "gpu_raycast",
                                  "gpu_blit", "gpu_gui", "cpu_wait", "cpu_acquire",
                                  "cpu_record", "cpu_submit"};
  for (size_t i = 0; i < 10; i++) json_timing(f, names[i], &timing[i], i + 1 < 10);
  fputs("  }\n}\n", f);
  int rc = ferror(f) || fclose(f) != 0 ? -1 : 0;
  if (rc == 0) printf("benchmark json: %s\n", path);
  return rc;
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--probe") == 0) {
    r3d_vkctx vk;
    if (r3d_vkctx_create(&vk, NULL, 0, false) != 0) return EXIT_FAILURE;
    r3d_vkctx_print_caps(&vk);
    printf("c5d revision       : %s (GPU ABI %u)\n", R3D_C5D_REV,
           (unsigned)R3D_C5D_GPU_ABI);
    r3d_vkctx_destroy(&vk);
    return EXIT_SUCCESS;
  }

  /* automation flags (tests/CI): exit after N frames, dump a screenshot */
  uint32_t exit_frames = 0;
  uint32_t warmup_frames = 0;
  const char *shot_path = NULL;
  const char *bench_json = NULL;
  int force_mode = -1, tf_preset = -1;
  int win_w = 1280, win_h = 720;
  float cam0[5] = {0.5f, 0.5f, -1.5f, 0.0f, 0.0f}; /* pos, yaw, pitch */
  bool no_vsync = false;
  float lowcut0 = 0.0f;
  const char *bench = NULL; /* scripted camera path: orbit | zoom | fly */
  const char *bench_name = NULL;
  const char *quality_arg = "interactive";
  float volpos0[3] = {0, 0, 0}, volrot0[3] = {0, 0, 0};
  uint32_t slab_wz = 0;     /* nonzero = slab mode, max visible depth */
  int depth0 = 0;           /* initial visible depth; explicit 0 = full volume */
  bool depth_given = false;
  bool clip_mode = false;   /* clipmap over the shard band */
  const char *bricks_path = NULL; /* c5d shard for GPU-decoded bricks mode */
  int pool_bpa = 0, warm_mb = 0;  /* bricks hot-atlas slots/axis, warm-tier MB */
  int brick_z = -1, brick_depth = 0; /* global manifest XY slice; depth 0 = full volume */
  uint32_t brick_shape[3] = {0, 0, 0};
  bool brick_is_lod = false;
  uint64_t gpu_budget_bytes = 0;
  bool vslab_mode = false;        /* toroidal streaming window over the export */
  int vsw = 12096, vsh = 12096, vsd = 16; /* window dims (voxels) */
  long long vsz0 = 34288;         /* start z (world; default inside the local band) */
  uint32_t vsnx = 43008, vsny = 43008, vsnz = 68608;
  const char *vscache = "band", *vsurl = NULL;
  const char *umbilicus_path = NULL;
  const char *multiview_path = NULL; /* tifxyz dir: vc3d-style 2x2 viewer */
  int annotation_prefetch = 5; /* annotation steps ahead; one slot is kept behind */
  int annotation_z_prefetch = 32; /* contiguous GPU-resident fine-scroll margin */
  bool vsz_given = false;
  for (int i = 1; i < argc; i++) {
    if (i < argc - 1 && strcmp(argv[i], "--frames") == 0) exit_frames = (uint32_t)atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--warmup") == 0)
      warmup_frames = (uint32_t)atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--bench-json") == 0) bench_json = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--shot") == 0) shot_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--mode") == 0) force_mode = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--tf") == 0) tf_preset = atoi(argv[i + 1]);
    if (i < argc - 2 && strcmp(argv[i], "--size") == 0) {
      win_w = atoi(argv[i + 1]);
      win_h = atoi(argv[i + 2]);
    }
    if (i < argc - 5 && strcmp(argv[i], "--cam") == 0)
      for (int k = 0; k < 5; k++) cam0[k] = (float)atof(argv[i + 1 + k]);
    if (strcmp(argv[i], "--no-vsync") == 0) no_vsync = true;
    if (i < argc - 1 && strcmp(argv[i], "--lowcut") == 0) lowcut0 = (float)atof(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--bench") == 0) bench = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--bench-name") == 0) bench_name = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--quality") == 0) quality_arg = argv[i + 1];
    if (i < argc - 3 && strcmp(argv[i], "--volpos") == 0)
      for (int k = 0; k < 3; k++) volpos0[k] = (float)atof(argv[i + 1 + k]);
    if (i < argc - 3 && strcmp(argv[i], "--volrot") == 0)
      for (int k = 0; k < 3; k++) volrot0[k] = (float)atof(argv[i + 1 + k]) / 57.29578f;
    if (i < argc - 1 && strcmp(argv[i], "--depth") == 0) {
      depth0 = atoi(argv[i + 1]);
      depth_given = true;
    }
    if (strcmp(argv[i], "--clipmap") == 0) clip_mode = true;
    if (i < argc - 1 && strcmp(argv[i], "--bricks") == 0) bricks_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--brick-z") == 0) brick_z = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--pool") == 0) pool_bpa = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--warm") == 0) warm_mb = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--gpu-mem") == 0)
      gpu_budget_bytes = (uint64_t)strtoull(argv[i + 1], NULL, 10) << 20;
    if (strcmp(argv[i], "--vslab") == 0) vslab_mode = true;
    if (i < argc - 1 && strcmp(argv[i], "--vsz") == 0) {
      vsz0 = atoll(argv[i + 1]);
      vsz_given = true;
    }
    if (i < argc - 3 && strcmp(argv[i], "--vswin") == 0) {
      vsw = atoi(argv[i + 1]);
      vsh = atoi(argv[i + 2]);
      vsd = atoi(argv[i + 3]);
    }
    if (i < argc - 1 && strcmp(argv[i], "--umbilicus") == 0) umbilicus_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--multiview") == 0) multiview_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--ann-prefetch") == 0)
      annotation_prefetch = atoi(argv[i + 1]);
    if (i < argc - 1 && strcmp(argv[i], "--ann-z-prefetch") == 0)
      annotation_z_prefetch = atoi(argv[i + 1]);
    if (strcmp(argv[i], "--slab") == 0) {
      slab_wz = 32;
      if (i < argc - 1 && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
        slab_wz = (uint32_t)atoi(argv[i + 1]);
    }
  }
  if (umbilicus_path) {
    vslab_mode = true;
    vsnx = PHERC1218_NX;
    vsny = PHERC1218_NY;
    vsnz = PHERC1218_NZ;
    vsw = PHERC1218_NX;
    vsh = PHERC1218_NY;
    vsd = 8;
    vscache = "cache/PHerc1218";
    vsurl = PHERC1218_SHARDS;
    if (!vsz_given) vsz0 = 0;
  }
  if (annotation_prefetch < 0 || annotation_prefetch >= R3D_VSLAB_PREFETCH_MAX) {
    fprintf(stderr, "--ann-prefetch must be between 0 and %u\n",
            R3D_VSLAB_PREFETCH_MAX - 1);
    return EXIT_FAILURE;
  }
  if (annotation_z_prefetch < 0 || annotation_z_prefetch > 128) {
    fprintf(stderr, "--ann-z-prefetch must be between 0 and 128\n");
    return EXIT_FAILURE;
  }
  if (bench && exit_frames == 0) exit_frames = 300;
  if (!exit_frames) warmup_frames = 0;
  if (exit_frames > UINT32_MAX - warmup_frames) {
    fprintf(stderr, "--frames + --warmup is too large\n");
    return EXIT_FAILURE;
  }
  uint32_t total_frames = exit_frames + warmup_frames;
  int quality_policy = 1; /* full=0, interactive=1, fast=2 */
  if (strcmp(quality_arg, "full") == 0) quality_policy = 0;
  else if (strcmp(quality_arg, "interactive") == 0) quality_policy = 1;
  else if (strcmp(quality_arg, "fast") == 0) quality_policy = 2;
  else {
    fprintf(stderr, "--quality must be full, interactive, or fast\n");
    return EXIT_FAILURE;
  }

  r3d_umbilicus umbilicus;
  r3d_umbilicus_init(&umbilicus);
  int annotation_step = 100;
  int annotation_z = (int)vsz0;
  int annotation_last_z = annotation_z, annotation_bench_z = annotation_z, annotation_dir = 1;
  char annotation_status[256] = "";
  if (umbilicus_path) {
    int urc = r3d_umbilicus_load(&umbilicus, umbilicus_path);
    if (urc < 0) {
      fprintf(stderr, "umbilicus: cannot load %s (expected Villa-compatible JSON)\n",
              umbilicus_path);
      r3d_umbilicus_free(&umbilicus);
      return EXIT_FAILURE;
    }
    if (urc == 0) {
      snprintf(annotation_status, sizeof annotation_status, "loaded %zu point%s",
               umbilicus.count, umbilicus.count == 1 ? "" : "s");
      if (!vsz_given && umbilicus.count) {
        double next = umbilicus.points[umbilicus.count - 1].z + annotation_step;
        annotation_z = (int)llround(next);
      }
    } else {
      snprintf(annotation_status, sizeof annotation_status, "new annotation");
    }
    if (annotation_z < 0) annotation_z = 0;
    if (annotation_z >= (int)vsnz) annotation_z = (int)vsnz - 1;
  }
  annotation_last_z = annotation_z;
  annotation_bench_z = annotation_z;

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }
  SDL_Window *win = SDL_CreateWindow("render3d", win_w, win_h,
                                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);
  if (!win) {
    fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
    SDL_Quit();
    return EXIT_FAILURE;
  }

  r3d_config cfg = {.validate = false,
                    .vsync = !no_vsync,
                    .spv_dir = R3D_SPV_DIR,
                    .gpu_budget_bytes = gpu_budget_bytes};
  r3d_renderer *renderer = NULL;
  if (r3d_create(win, &cfg, &renderer) != 0) {
    fprintf(stderr, "renderer init failed\n");
    SDL_DestroyWindow(win);
    SDL_Quit();
    return EXIT_FAILURE;
  }
  r3d_set_quality(renderer, quality_policy == 2 ? R3D_QUALITY_FAST : R3D_QUALITY_FULL);

  /* positional args: <volume.u8> <nx> <ny> <nz> */
  uint32_t mode = R3D_MODE_RAYDIR;
  r3d_volume slab_src = {0}; /* stays open in slab mode (window scrolls it) */
  uint32_t slab_z0 = 0, slab_z0_max = 0;
  if (argc >= 5 && argv[1][0] != '-') {
    r3d_volume vol;
    if (r3d_volume_open(&vol, argv[1], (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]),
                        (uint32_t)atoi(argv[4])) != 0)
      return EXIT_FAILURE;
    if (slab_wz) {
      /* --slab N = max visible depth; ring holds N+2 so face filtering never
       * crosses the seam and depth changes need no re-upload */
      uint32_t ring = slab_wz + 2;
      r3d_slab_desc sd = {.nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .wz = ring};
      if (r3d_slab_init(renderer, &sd) != 0) return EXIT_FAILURE;
      slab_src = vol;
      slab_z0 = (vol.nz - ring) / 2; /* start mid-depth */
      slab_z0_max = vol.nz - ring;
      if (r3d_slab_window(renderer, &slab_src, slab_z0) != 0) return EXIT_FAILURE;
    } else {
      r3d_volume_desc desc = {
          .nx = vol.nx, .ny = vol.ny, .nz = vol.nz, .brick_dim = R3D_BRICK_DIM};
      int up = r3d_upload_volume(renderer, &desc, vol.voxels);
      r3d_volume_close(&vol); /* GPU has it; drop the mapping */
      if (up != 0) return EXIT_FAILURE;
    }
    mode = R3D_MODE_FULL;
  } else if (slab_wz) {
    fprintf(stderr, "--slab needs a volume argument\n");
    return EXIT_FAILURE;
  }

  if (bricks_path) {
    if (r3d_bricks_begin(renderer, bricks_path, (uint32_t)pool_bpa, (uint32_t)warm_mb) != 0)
      return EXIT_FAILURE;
    r3d_bricks_shape(renderer, brick_shape);
    r3d_bricks_stats initial_bst;
    r3d_bricks_get_stats(renderer, &initial_bst);
    brick_is_lod = initial_bst.nlevels > 1u;
    brick_depth = depth_given ? depth0 : (brick_is_lod ? 8 : 0);
    if (brick_depth < 0) brick_depth = 0;
    if (brick_depth > (int)brick_shape[2]) brick_depth = (int)brick_shape[2];
    if (brick_z < 0) brick_z = brick_depth ? ((int)brick_shape[2] - brick_depth) / 2 : 0;
    if (brick_z < 0) brick_z = 0;
    if (brick_z > (int)brick_shape[2] - brick_depth)
      brick_z = (int)brick_shape[2] - brick_depth;
    mode = R3D_MODE_FULL;
  }

  /* vc3d-style 2x2 multi-view: flattened segment (TL, milestone C — an XY
   * overview until then) + XY/XZ/YZ ortho plane views, shared focus POI */
  r3d_tifxyz mv_seg = {0};
  r3d_mview mv[4] = {0};
  double mv_focus[3] = {0, 0, 0}; /* world voxels x,y,z */
  int mv_thick = 1;               /* plane-view slab thickness (voxels) */
  int mv_drag_view = -1;
  float *mv_normals = NULL; /* per-vertex normals kept for overlays/zoff shell */
  uint64_t mv_sv_decoded = 0; /* residency-driven surfvol rebuild bookkeeping */
  int mv_sv_cool = 0;
  mv_lines mv_ol[4] = {0}, mv_ol_off[4] = {0}; /* intersection polylines */
  double mv_ol_slice[4] = {1e30, 1e30, 1e30, 1e30};
  double mv_ol_zoff = 1e30;
  if (multiview_path) {
    if (!bricks_path || !brick_is_lod) {
      fprintf(stderr, "--multiview needs --bricks with a LOD manifest\n");
      return EXIT_FAILURE;
    }
    if (r3d_tifxyz_load(&mv_seg, multiview_path) != 0) return EXIT_FAILURE;
    for (int a = 0; a < 3; a++)
      mv_focus[a] = ((double)mv_seg.bbox[0][a] + (double)mv_seg.bbox[1][a]) * 0.5;
    const float *mc = r3d_tifxyz_at(&mv_seg, mv_seg.w / 2, mv_seg.h / 2);
    if (r3d_tifxyz_valid(mc)) /* center the focus ON the sheet when possible */
      for (int a = 0; a < 3; a++) mv_focus[a] = (double)mc[a];
    brick_depth = 0; /* multiview owns per-view slab clips */
    mode = R3D_MODE_FULL; /* volumetric slabs in every quadrant (Tab: MIP etc.) */
    mv_thick = 24;
    for (int i = 0; i < 4; i++) {
      const uint8_t *ax = r3d_mv_axes[i];
      mv[i].cu = mv_focus[ax[0]];
      mv[i].cv = mv_focus[ax[1]];
      mv[i].slice = mv_focus[ax[2]];
      mv[i].zoom = 0.0; /* fitted on the first frame once quadrants exist */
    }
    /* TL = flattened segment view: grid-space camera, slice = normal offset */
    mv[R3D_MV_SEG].cu = (double)mv_seg.w * 0.5;
    mv[R3D_MV_SEG].cv = (double)mv_seg.h * 0.5;
    mv[R3D_MV_SEG].slice = 0.0;
    float *seg_coords = NULL;
    if (mv_build_grids(&mv_seg, &seg_coords, &mv_normals) != 0 ||
        r3d_surf_begin(renderer, mv_seg.w, mv_seg.h, seg_coords, mv_normals) != 0) {
      fprintf(stderr, "multiview: surf grid upload failed\n");
      return EXIT_FAILURE;
    }
    free(seg_coords);
    /* flattened surface volume window: 1024^2 x 96 layers (48 behind) R8
     * = 96 MB, resampled on the GPU from the shared brick cache */
    if (r3d_surfvol_begin(renderer, 1024, 1024, 96, 48, mv_seg.sx, mv_seg.sy) != 0) {
      fprintf(stderr, "multiview: surface-volume window init failed\n");
      return EXIT_FAILURE;
    }
  }

  int64_t vs_z0 = umbilicus_path ? annotation_z0(annotation_z, vsnz, (uint32_t)vsd)
                                  : (int64_t)vsz0;
  double vs_fx = (double)vsnx * 0.5, vs_fy = (double)vsny * 0.5;
  bool vs_follow = true;
  uint64_t vs_pend_acc = 0;
  if (vslab_mode) {
    r3d_vslab_desc vd = {.nx = vsnx,
                         .ny = vsny,
                         .nz = vsnz,
                         .W = (uint32_t)vsw,
                         .H = (uint32_t)vsh,
                         .D = (uint32_t)vsd,
                         .cache_dir = vscache,
                         .shard_url = vsurl,
                         .z_prefetch = umbilicus_path ? (uint32_t)annotation_z_prefetch : 0u,
                         .prefetch_slots = umbilicus_path && annotation_prefetch
                                               ? (uint32_t)annotation_prefetch + 1u
                                               : 0u,
                         .prefetch_threads = umbilicus_path ? 6u : 0u};
    if (r3d_vslab_begin(renderer, &vd) != 0)
      return EXIT_FAILURE;
    mode = R3D_MODE_FULL;
    if (umbilicus_path) vs_follow = false; /* full XY plane is fixed at origin */
  }

  /* clipmap: 43k^2 cross sections from the shard band + pyramid */
  const uint32_t CLIP_NX = 43008, CLIP_BAND_Z = 33, CLIP_DEPTH_MAX = 32;
  uint64_t clip_z0 = 0, clip_z0_min = 0, clip_z0_max = 0;
  if (clip_mode) {
    if (r3d_clip_begin(renderer, "band", "pyramid", CLIP_BAND_Z, CLIP_DEPTH_MAX) != 0)
      return EXIT_FAILURE;
    clip_z0_min = (uint64_t)CLIP_BAND_Z * 1024;
    clip_z0_max = clip_z0_min + 1024 - CLIP_DEPTH_MAX;
    clip_z0 = clip_z0_min + 512 - CLIP_DEPTH_MAX / 2;
    mode = R3D_MODE_FULL;
  }
  if (force_mode >= 0) mode = (uint32_t)force_mode % R3D_MODE_COUNT;
  float tf_min_v = 1.0f; /* backend default ramp: alpha nonzero from value 1 */
  if (tf_preset >= 0) {
    r3d_tf tf;
    uint8_t lut[256][4];
    r3d_tf_preset((uint32_t)tf_preset, &tf);
    r3d_tf_build(&tf, lut);
    r3d_set_transfer(renderer, lut);
    tf_min_v = tf_min_visible(lut);
  }

  /* orbit (turntable around the volume) is the default; --cam implies fly */
  bool cam_given = false;
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "--cam") == 0) cam_given = true;
  int cam_mode = cam_given ? CAM_FLY : CAM_ORBIT;

  r3d_camera cam;
  r3d_camera_init(&cam, v3(cam0[0], cam0[1], cam0[2]));
  cam.yaw = cam0[3];
  cam.pitch = cam0[4];
  if (cam_mode == CAM_ORBIT) {
    if (clip_mode) {
      float ez = (float)CLIP_DEPTH_MAX / (float)CLIP_NX;
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, ez * 0.5f), 1.3f);
    } else if (vslab_mode) {
      float ey = (float)vsh / (float)vsw;
      float ez = (float)(vsd + 2) / (float)vsw;
      r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 1.4f);
    } else if (slab_wz) {
      /* orbit the thin slab: target its center, sit back along z */
      float ey = (float)slab_src.ny / (float)slab_src.nx;
      float ez = (float)slab_wz / (float)slab_src.nx;
      r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 1.4f);
    } else if (bricks_path) {
      float e[3];
      r3d_bricks_extent(renderer, e);
      float tz = e[2] * 0.5f;
      if (brick_depth > 0) {
        uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
        if (brick_shape[2] > md) md = brick_shape[2];
        tz = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
      }
      r3d_camera_orbit_set(&cam, v3(e[0] * 0.5f, e[1] * 0.5f, tz),
                           brick_depth > 0 ? 0.42f : 2.0f);
    } else {
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
    }
  }
  r3d_input in = {0};
  r3d_stats stats;
  r3d_stats_init(&stats);

  float step_voxels = 1.0f, density = 1.0f, lod_bias = 0.0f;
  float low_cut = lowcut0; /* voxel-value threshold, 0..255 */
  bool auto_scroll = false;
  float auto_speed = 10.0f; /* slices per second */
  float auto_accum = 0.0f;
  int slab_depth = depth0 >= 2 && depth0 <= (int)slab_wz ? depth0 : (int)slab_wz;
  int clip_depth = depth0 >= 2 && depth0 <= (int)CLIP_DEPTH_MAX ? depth0 : (int)CLIP_DEPTH_MAX;
  uint32_t clip_valid_disp = 0;

  /* volume (model) transform: translation in world units, rotation ypr */
  r3d_v3 vol_t = v3(volpos0[0], volpos0[1], volpos0[2]);
  float vol_rot[3] = {volrot0[0], volrot0[1], volrot0[2]}; /* yaw, pitch, roll (radians) */
  float fov_deg = 60.0f;
  uint32_t tf_idx = tf_preset > 0 ? (uint32_t)tf_preset : 0;
  uint32_t frame_index = 0;
  float fps_smooth = 60.0f;
  bool adaptive_res = quality_policy != 0; /* interactive/fast halve resolution while moving */
  int settle = 0;
  uint64_t last_gpu_ns = 0;
  r3d_frame_stats prof = {0};   /* EMA-smoothed for display */
  r3d_frame_stats prof_sum = {0}; /* running sums for the exit report */
  uint64_t prof_frames = 0;
  r3d_frame_stats *prof_samples = exit_frames ? calloc(exit_frames, sizeof *prof_samples) : NULL;
  if (exit_frames && !prof_samples) return EXIT_FAILURE;
  uint64_t prev_ns = r3d_now_ns();

  bool running = true;
  while (running) {
    uint64_t t0 = r3d_now_ns();
    if (exit_frames && frame_index == warmup_frames) {
      r3d_stats_init(&stats);
      memset(&prof_sum, 0, sizeof prof_sum);
      prof_frames = 0;
      vs_pend_acc = 0;
    }
    float dt = (float)((double)(t0 - prev_ns) / 1e9);
    prev_ns = t0;
    if (dt > 0.1f) dt = 0.1f;

    ImGuiIO *io = igGetIO_Nil(); /* Want* flags reflect last frame — fine */
    r3d_input_poll(&in, win, gui_event_hook, renderer, !io->WantCaptureMouse,
                   cam_mode == CAM_FLY, umbilicus_path != NULL, multiview_path != NULL);
    if (io->WantCaptureKeyboard && !in.captured)
      in.move[0] = in.move[1] = in.move[2] = 0.0f;
    if (in.quit) running = false;
    if (in.resized) r3d_resize(renderer);
    if (in.mode_delta) {
      mode = (mode + (uint32_t)in.mode_delta) % R3D_MODE_COUNT;
      printf("mode: %u\n", mode);
    }
    bool z_navigated = false;
    if (umbilicus_path) {
      int old_annotation_z = annotation_z;
      int64_t nz = annotation_z;
      nz += in.zdelta;
      nz += (int64_t)in.zpage * annotation_step;
      /* In annotation mode the ordinary wheel traverses z. Shift+wheel is
       * retained as camera zoom for inspecting a difficult center. Wheel is
       * deliberately fine-grained; buttons/pages retain annotation_step. */
      if (!in.fast && in.wheel != 0.0f && !io->WantCaptureMouse) {
        nz += in.wheel > 0.0f ? 1 : -1;
        in.wheel = 0.0f;
      }
      if (nz < 0) nz = 0;
      if (nz >= vsnz) nz = (int64_t)vsnz - 1;
      annotation_z = (int)nz;
      z_navigated = annotation_z != old_annotation_z;
      vs_z0 = annotation_z0(annotation_z, vsnz, (uint32_t)vsd);
    }
    if (bricks_path && brick_depth > 0 && !umbilicus_path && !multiview_path) {
      int old_brick_z = brick_z;
      int64_t nz = (int64_t)brick_z + in.zdelta + (int64_t)in.zpage * brick_depth;
      /* Global LOD slices use the same fine wheel semantics as annotation:
       * wheel changes one z slice, Shift+wheel retains camera zoom. */
      if (!in.fast && in.wheel != 0.0f && !io->WantCaptureMouse) {
        nz += in.wheel > 0.0f ? 1 : -1;
        in.wheel = 0.0f;
      }
      int64_t zmax = (int64_t)brick_shape[2] - brick_depth;
      if (nz < 0) nz = 0;
      if (nz > zmax) nz = zmax;
      brick_z = (int)nz;
      z_navigated = z_navigated || brick_z != old_brick_z;
      if (cam_mode == CAM_ORBIT && brick_z != old_brick_z) {
        uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
        if (brick_shape[2] > md) md = brick_shape[2];
        float dz = (float)(brick_z - old_brick_z) / (float)md;
        cam.target.z += dz;
        cam.pos.z += dz;
      }
    }
    if (clip_mode) {
      int64_t nz0 = (int64_t)clip_z0 + in.zdelta + (int64_t)in.zpage * (int64_t)clip_depth;
      if (nz0 < (int64_t)clip_z0_min) nz0 = (int64_t)clip_z0_min;
      if (nz0 > (int64_t)clip_z0_max) nz0 = (int64_t)clip_z0_max;
      clip_z0 = (uint64_t)nz0;
    }
    if (slab_wz) {
      int64_t nz0 = (int64_t)slab_z0 + in.zdelta + (int64_t)in.zpage * (int64_t)slab_depth;
      if (bench && strcmp(bench, "zsweep") == 0) /* scripted scroll for perf/tests */
        nz0 = (int64_t)((float)slab_z0_max *
                        (float)(frame_index < warmup_frames ? frame_index
                                                           : frame_index - warmup_frames) /
                        (float)(frame_index < warmup_frames && warmup_frames ? warmup_frames
                                                                            : exit_frames));
      if (auto_scroll) {
        auto_accum += auto_speed * dt;
        float whole = floorf(auto_accum);
        nz0 += (int64_t)whole;
        auto_accum -= whole;
        if (nz0 >= (int64_t)slab_z0_max) { /* bounce at the ends */
          nz0 = slab_z0_max;
          auto_speed = -auto_speed;
        } else if (nz0 <= 0 && auto_speed < 0) {
          nz0 = 0;
          auto_speed = -auto_speed;
        }
      }
      if (nz0 < 0) nz0 = 0;
      if (nz0 > (int64_t)slab_z0_max) nz0 = slab_z0_max;
      if ((uint32_t)nz0 != slab_z0) {
        slab_z0 = (uint32_t)nz0;
        r3d_slab_window(renderer, &slab_src, slab_z0);
      }
    }
    if (in.tf_delta) {
      tf_idx = (tf_idx + 1) % r3d_tf_preset(UINT32_MAX, NULL);
      r3d_tf tf;
      uint8_t lut[256][4];
      r3d_tf_preset(tf_idx, &tf);
      r3d_tf_build(&tf, lut);
      r3d_set_transfer(renderer, lut);
      tf_min_v = tf_min_visible(lut);
      printf("tf preset: %u\n", tf_idx);
    }
    step_voxels *= in.step_scale;
    density *= in.density_scale;
    lod_bias += in.lod_delta;
    cam.fov_y = fov_deg * 0.01745329f;
    if (multiview_path) {
      /* per-view interaction is handled after the quadrant layout below */
    } else if (cam_mode == CAM_ORBIT) {
      /* drag grabs the cube; Shift pans the camera; Ctrl translates the
       * volume; Ctrl+Shift rotates the volume */
      if (in.dragging && (in.ctrl || in.fast)) {
        r3d_v3 br, bu, bf;
        r3d_camera_basis(&cam, 1.0f, &br, &bu, &bf);
        r3d_v3 ru = v3_norm(br), uu = v3_norm(bu);
        float k = cam.dist * 0.0012f;
        if (in.ctrl && in.fast) { /* rotate volume */
          vol_rot[0] += in.look[0] * ORBIT_SENS;
          vol_rot[1] += in.look[1] * ORBIT_SENS;
        } else if (in.ctrl) { /* translate volume in the view plane */
          vol_t = v3_add(vol_t, v3_add(v3_scale(ru, in.look[0] * k),
                                       v3_scale(uu, -in.look[1] * k)));
        } else { /* pan camera (grab-the-world) */
          r3d_camera_orbit_pan(&cam, v3(-in.look[0], in.look[1], 0), k);
        }
      } else if (in.dragging) {
        r3d_camera_orbit_drag(&cam, -in.look[0] * ORBIT_SENS, -in.look[1] * ORBIT_SENS);
      }
      if (in.wheel != 0.0f && !io->WantCaptureMouse)
        /* shift+wheel zooms 3x faster: whole-scroll to fiber scale is a ~3200x
         * distance ratio, a long ride at 0.9/detent */
        r3d_camera_orbit_zoom(&cam, powf(in.fast ? 0.73f : 0.9f, in.wheel));
      float pan = BASE_SPEED * cam.dist * (in.fast ? 5.0f : 1.0f);
      if (in.move[0] != 0.0f || in.move[1] != 0.0f || in.move[2] != 0.0f)
        r3d_camera_orbit_pan(&cam, v3(in.move[0], in.move[1], in.move[2]), pan * dt);
    } else {
      r3d_camera_look(&cam, in.look[0] * MOUSE_SENS, -in.look[1] * MOUSE_SENS);
      float speed = BASE_SPEED * (in.fast ? 5.0f : 1.0f);
      r3d_camera_move(&cam, v3(in.move[0], in.move[1], in.move[2]), speed * dt);
    }

    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(win, &w, &h);
    if (w <= 0 || h <= 0) {
      SDL_Delay(50);
      continue;
    }

    if (multiview_path) {
      r3d_mview lay[4];
      r3d_mv_layout(lay, w, h);
      for (int i = 0; i < 4; i++) {
        mv[i].px = lay[i].px;
        mv[i].py = lay[i].py;
        mv[i].pw = lay[i].pw;
        mv[i].ph = lay[i].ph;
      }
      if (mv[0].zoom <= 0.0) { /* first frame: fit */
        double fy2 = (double)mv[R3D_MV_SEG].ph / (double)(mv_seg.h ? mv_seg.h : 1);
        double fx2 = (double)mv[R3D_MV_SEG].pw / (double)(mv_seg.w ? mv_seg.w : 1);
        mv[R3D_MV_SEG].zoom = fy2 < fx2 ? fy2 : fx2;
        for (int i = 1; i < 4; i++) mv[i].zoom = (double)mv[i].ph / 2048.0;
      }
      int hover = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
      if (in.dragging && mv_drag_view < 0) mv_drag_view = hover;
      if (!in.dragging) mv_drag_view = -1;
      if (mv_drag_view >= 0 && (in.look[0] != 0.0f || in.look[1] != 0.0f)) {
        r3d_mview *dv = &mv[mv_drag_view];
        dv->cu -= (double)in.look[0] / dv->zoom;
        dv->cv -= (double)in.look[1] / dv->zoom;
      }
      if (hover >= 0 && in.wheel != 0.0f && !io->WantCaptureMouse) {
        r3d_mview *hv = &mv[hover];
        if (in.wheel_shift) { /* scrub the slice along the view normal */
          hv->slice += (double)(in.wheel > 0.0f ? 1 : -1);
        } else {
          r3d_mv_zoom(hv, in.mouse_xy[0], in.mouse_xy[1],
                      pow(1.05, (double)in.wheel * 2.0), 1.0 / 256.0, 64.0);
        }
        in.wheel = 0.0f;
      }
      if (hover >= 0 && (in.zdelta || in.zpage))
        mv[hover].slice += (double)(in.zdelta + in.zpage * 10);
      /* SEG slice = normal offset, symmetric around the sheet */
      if (mv[R3D_MV_SEG].slice < -512.0) mv[R3D_MV_SEG].slice = -512.0;
      if (mv[R3D_MV_SEG].slice > 512.0) mv[R3D_MV_SEG].slice = 512.0;
      for (int i = 1; i < 4; i++) { /* plane slices clamp to the volume */
        uint32_t n = brick_shape[r3d_mv_axes[i][2]];
        if (mv[i].slice < 0.0) mv[i].slice = 0.0;
        if (n && mv[i].slice > (double)n - 1.0) mv[i].slice = (double)n - 1.0;
      }
      if (in.annotate_click && in.click_ctrl) { /* Ctrl+click = set focus POI */
        int cv_ = r3d_mv_hit(mv, in.click_xy[0], in.click_xy[1]);
        bool focused = false;
        if (cv_ == R3D_MV_SEG) {
          /* focus at the surface point under the cursor (CPU bilinear tap) */
          double gu, gv;
          r3d_mv_unproject(&mv[cv_], in.click_xy[0], in.click_xy[1], &gu, &gv);
          if (gu >= 0.0 && gv >= 0.0 && gu <= (double)mv_seg.w - 1.0 &&
              gv <= (double)mv_seg.h - 1.0) {
            uint32_t gi = (uint32_t)gu, gj = (uint32_t)gv;
            if (gi > mv_seg.w - 2) gi = mv_seg.w - 2;
            if (gj > mv_seg.h - 2) gj = mv_seg.h - 2;
            const float *q00 = r3d_tifxyz_at(&mv_seg, gi, gj);
            const float *q10 = r3d_tifxyz_at(&mv_seg, gi + 1, gj);
            const float *q01 = r3d_tifxyz_at(&mv_seg, gi, gj + 1);
            const float *q11 = r3d_tifxyz_at(&mv_seg, gi + 1, gj + 1);
            if (r3d_tifxyz_valid(q00) && r3d_tifxyz_valid(q10) && r3d_tifxyz_valid(q01) &&
                r3d_tifxyz_valid(q11)) {
              double fx_ = gu - gi, fy_ = gv - gj;
              for (int a = 0; a < 3; a++)
                mv_focus[a] = ((double)q00[a] * (1 - fx_) + (double)q10[a] * fx_) * (1 - fy_) +
                              ((double)q01[a] * (1 - fx_) + (double)q11[a] * fx_) * fy_;
              focused = true;
            }
          }
        } else if (cv_ > 0) {
          const uint8_t *ax = r3d_mv_axes[cv_];
          double u, vq;
          r3d_mv_unproject(&mv[cv_], in.click_xy[0], in.click_xy[1], &u, &vq);
          mv_focus[ax[0]] = u;
          mv_focus[ax[1]] = vq;
          mv_focus[ax[2]] = mv[cv_].slice;
          focused = true;
          /* recenter the segment view on the surface point nearest the focus */
          double best = 1e30;
          uint32_t bi = 0, bj = 0;
          for (uint32_t gj = 0; gj < mv_seg.h; gj += 2)
            for (uint32_t gi = 0; gi < mv_seg.w; gi += 2) {
              const float *sp = r3d_tifxyz_at(&mv_seg, gi, gj);
              if (!r3d_tifxyz_valid(sp)) continue;
              double dx = (double)sp[0] - mv_focus[0], dy = (double)sp[1] - mv_focus[1],
                     dz = (double)sp[2] - mv_focus[2];
              double d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < best) {
                best = d2;
                bi = gi;
                bj = gj;
              }
            }
          if (best < 100.0 * 100.0) { /* vc3d tolerance: 100 voxels */
            mv[R3D_MV_SEG].cu = (double)bi;
            mv[R3D_MV_SEG].cv = (double)bj;
          }
        }
        if (focused) {
          for (int a = 0; a < 3; a++) { /* clamp into the volume */
            if (mv_focus[a] < 0.0) mv_focus[a] = 0.0;
            if (brick_shape[a] && mv_focus[a] > (double)brick_shape[a] - 1.0)
              mv_focus[a] = (double)brick_shape[a] - 1.0;
          }
          for (int i = 1; i < 4; i++) { /* recenter planes through the focus */
            const uint8_t *a2 = r3d_mv_axes[i];
            mv[i].cu = mv_focus[a2[0]];
            mv[i].cv = mv_focus[a2[1]];
            mv[i].slice = mv_focus[a2[2]];
          }
        }
      }

      { /* keep the flattened surface-volume window under the view (snapped
         * for hysteresis) and rebuild it when brick residency improves */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        double vox_per_px = 1.0 / (sv->zoom * (double)mv_seg.sx);
        double stepd = 1.0;
        while (stepd * 2.0 <= vox_per_px) stepd *= 2.0;
        double cu_vox = sv->cu / (double)mv_seg.sx, cv_vox = sv->cv / (double)mv_seg.sy;
        double snap = 128.0 * stepd; /* window W/8 */
        double u0 = floor((cu_vox - 512.0 * stepd) / snap) * snap;
        double v0 = floor((cv_vox - 512.0 * stepd) / snap) * snap;
        double zsnap = 24.0; /* layers are 1 voxel regardless of xy zoom */
        double z0 = floor(sv->slice / zsnap + 0.5) * zsnap;
        r3d_surfvol_window(renderer, u0, v0, (float)stepd, (float)z0);
        r3d_bricks_stats svst;
        r3d_bricks_get_stats(renderer, &svst);
        if (mv_sv_cool > 0) mv_sv_cool--;
        if (svst.decoded != mv_sv_decoded && mv_sv_cool == 0) {
          mv_sv_decoded = svst.decoded;
          mv_sv_cool = 20; /* at most one residency rebuild per ~1/3 s */
          r3d_surfvol_mark(renderer);
        }
      }
    }

    /* scripted camera paths for reproducible perf runs (override user input) */
    if (bench) {
      uint32_t phase_frame = frame_index < warmup_frames ? frame_index
                                                         : frame_index - warmup_frames;
      uint32_t phase_count = frame_index < warmup_frames && warmup_frames ? warmup_frames
                                                                          : exit_frames;
      float ph = phase_count ? (float)phase_frame / (float)phase_count : 0.0f;
      float tau = 6.2831853f;
      if (strcmp(bench, "orbit") == 0) {
        cam.yaw = ph * tau;
        cam.pitch = 0.5f * sinf(ph * tau * 2.0f);
        r3d_v3 target = v3(0.5f, 0.5f, 0.5f);
        if (bricks_path) {
          float e[3];
          r3d_bricks_extent(renderer, e);
          target = v3(e[0] * 0.5f, e[1] * 0.5f, e[2] * 0.5f);
          if (brick_depth > 0) {
            uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
            if (brick_shape[2] > md) md = brick_shape[2];
            target.z = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
          }
        }
        r3d_camera_orbit_set(&cam, target, 2.0f);
      } else if (strcmp(bench, "zoom") == 0) {
        cam.yaw = ph * tau * 0.5f;
        cam.pitch = 0.2f;
        r3d_v3 target = v3(0.5f, 0.5f, 0.5f);
        if (bricks_path) {
          float e[3];
          r3d_bricks_extent(renderer, e);
          target = v3(e[0] * 0.5f, e[1] * 0.5f, e[2] * 0.5f);
          if (brick_depth > 0) {
            uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
            if (brick_shape[2] > md) md = brick_shape[2];
            target.z = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
          }
        }
        r3d_camera_orbit_set(&cam, target, 1.75f - 1.55f * sinf(ph * 3.14159265f));
      } else if (strcmp(bench, "fly") == 0) { /* weaving pass through the volume */
        cam.pos = v3(0.5f + 0.15f * sinf(ph * tau * 1.5f), 0.5f + 0.1f * sinf(ph * tau),
                     -0.3f + 1.6f * ph);
        cam.yaw = 0.15f * sinf(ph * tau);
        cam.pitch = 0.1f * cosf(ph * tau);
      } else if (strcmp(bench, "clippan") == 0) { /* sweep across the cross-section */
        cam.pos = v3(0.25f + 0.5f * ph, 0.5f, -0.02f);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
      } else if (strcmp(bench, "zoomio") == 0) {
        /* full zoom sweep: whole-composite view down to voxel scale and back
         * (log-space triangle wave) — walks every LOD band the mode has */
        float tx = 0.5f;
        float ey = slab_wz ? (float)slab_src.ny / (float)slab_src.nx : 1.0f;
        float ez = slab_wz ? (float)slab_wz / (float)slab_src.nx : 1.0f;
        if (vslab_mode) { ey = (float)vsh / (float)vsw; ez = (float)(vsd + 2) / (float)vsw; }
        if (bricks_path) {
          float e[3];
          r3d_bricks_extent(renderer, e);
          tx = e[0] * 0.5f; ey = e[1]; ez = e[2];
        }
        float tri = ph < 0.5f ? ph * 2.0f : 2.0f - ph * 2.0f; /* 0->1->0 */
        float d = 1.6f * powf(0.002f / 1.6f, tri);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
        float tz = ez * 0.5f;
        if (bricks_path && brick_depth > 0) {
          uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
          if (brick_shape[2] > md) md = brick_shape[2];
          tz = ((float)brick_z + 0.5f * (float)brick_depth) / (float)md;
        }
        r3d_camera_orbit_set(&cam, v3(tx, ey * 0.5f, tz), d);
      } else if (strcmp(bench, "volrot") == 0) {
        /* mid-zoom + model rotation: worst case for anisotropic footprints */
        float ey = slab_wz ? (float)slab_src.ny / (float)slab_src.nx : 1.0f;
        float ez = slab_wz ? (float)slab_wz / (float)slab_src.nx : 1.0f;
        if (vslab_mode) { ey = (float)vsh / (float)vsw; ez = (float)(vsd + 2) / (float)vsw; }
        vol_rot[0] = 0.7f * sinf(ph * tau);
        vol_rot[1] = 0.45f * sinf(ph * tau * 0.7f);
        cam.yaw = 0.0f;
        cam.pitch = 0.0f;
        r3d_camera_orbit_set(&cam, v3(0.5f, ey * 0.5f, ez * 0.5f), 0.2f);
      } else if (umbilicus_path && strcmp(bench, "annstep") == 0) {
        int64_t bz = annotation_bench_z;
        if (ph >= 0.6f) bz += annotation_step;
        if (bz >= vsnz) bz = (int64_t)vsnz - 1;
        annotation_z = (int)bz;
      } else if (umbilicus_path && strcmp(bench, "annscroll") == 0) {
        /* Hold long enough to fill look-ahead, then make six consecutive
         * annotation-step moves 0.6 s apart in a 600-frame vsynced run. */
        int64_t n = ph < 0.6f ? 0 : 1 + (int64_t)((ph - 0.6f) / 0.06f);
        if (n > 6) n = 6;
        int64_t bz = (int64_t)annotation_bench_z + n * annotation_step;
        if (bz >= vsnz) bz = (int64_t)vsnz - 1;
        annotation_z = (int)bz;
      } else if (umbilicus_path && strcmp(bench, "annwheel") == 0) {
        /* Fine-scroll stress: after residency warmup, traverse 20 adjacent
         * slices at ten detents/second in a 600-frame vsynced run. */
        int64_t n = ph < 0.7f ? 0 : 1 + (int64_t)((ph - 0.7f) / 0.01f);
        if (n > 20) n = 20;
        int64_t bz = (int64_t)annotation_bench_z + n;
        if (bz >= vsnz) bz = (int64_t)vsnz - 1;
        annotation_z = (int)bz;
      } /* other bench names (zsweep) keep the default camera */
      if (vslab_mode && strcmp(bench, "zsweep") == 0) /* scroll z across the band */
        vs_z0 = 33792 + (int64_t)(ph * (1024.0f - (float)(vsd + 2)));
      if (bricks_path && brick_depth > 0 && strcmp(bench, "zsweep") == 0) {
        int bz = (int)(ph * (float)((int)brick_shape[2] - brick_depth));
        uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
        if (brick_shape[2] > md) md = brick_shape[2];
        float dz = (float)(bz - brick_z) / (float)md;
        brick_z = bz;
        cam.target.z += dz;
        cam.pos.z += dz;
      }
      if (vslab_mode && strcmp(bench, "clippan") == 0) { /* xy window churn */
        vs_follow = false;
        vs_fx = 12000.0 + (double)ph * 18000.0;
        vs_fy = 21504.0;
      }
    }
    r3d_v3 right, up, fwd;
    r3d_camera_basis(&cam, (float)w / (float)h, &right, &up, &fwd);
    r3d_m3 vm = m3_ypr(vol_rot[0], vol_rot[1], vol_rot[2]);

    if (umbilicus_path && in.annotate_click) {
      int64_t origin[3] = {0, 0, vs_z0};
      r3d_vslab_get(renderer, origin, NULL);
      if (origin[0] < 0) origin[0] = 0;
      if (origin[1] < 0) origin[1] = 0;
      double ax = 0.0, ay = 0.0;
      if (annotation_pick(&cam, right, up, fwd, vm, vol_t, in.click_xy[0], in.click_xy[1],
                          w, h, (uint32_t)vsw, (uint32_t)vsh, (uint32_t)vsd, origin[0],
                          origin[1], vs_z0, annotation_z, &ax, &ay) &&
          r3d_umbilicus_set(&umbilicus, ax, ay, annotation_z) == 0) {
        save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        if (in.click_ctrl) {
          int64_t next = (int64_t)annotation_z + annotation_step;
          annotation_z = (int)(next < vsnz ? next : (int64_t)vsnz - 1);
          vs_z0 = annotation_z0(annotation_z, vsnz, (uint32_t)vsd);
        }
      } else {
        snprintf(annotation_status, sizeof annotation_status,
                 "click missed the displayed XY plane");
      }
    }

    /* adaptive resolution: drop to half res while interacting (4x fewer
     * rays), snap back to full once the camera settles */
    bool moving = in.dragging || in.captured || in.wheel != 0.0f || in.zdelta || in.zpage ||
                  z_navigated || auto_scroll ||
                  in.move[0] != 0.0f || in.move[1] != 0.0f || in.move[2] != 0.0f;
    settle = moving ? 15 : (settle > 0 ? settle - 1 : 0);
    bool half_res = adaptive_res && settle > 0 && !multiview_path;
    if (getenv("R3D_FORCE_HALF")) half_res = true; /* testing/benching the path */
    else if (in.screenshot || (total_frames && shot_path && frame_index + 1 >= total_frames))
      half_res = false; /* captures always full res */
    uint32_t rvw = half_res ? (uint32_t)w / 2 : (uint32_t)w;
    uint32_t rvh = half_res ? (uint32_t)h / 2 : (uint32_t)h;

    /* control panel */
    fps_smooth = fps_smooth * 0.95f + (dt > 0 ? 0.05f / dt : 0.0f);
    r3d_gui_begin(renderer);
    igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
    igBegin("render3d", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    igText("%.0f fps   gpu %.2f ms", (double)fps_smooth, (double)last_gpu_ns / 1e6);
    int m = (int)mode;
    if (igCombo_Str("mode", &m, "full\0mip\0depth\0heatmap\0raydir\0flat\0", 6))
      mode = (uint32_t)m;
    int prev_cm = cam_mode;
    igCombo_Str("camera", &cam_mode, "orbit\0fly\0", 2);
    if (cam_mode == CAM_ORBIT && prev_cm == CAM_FLY)
      r3d_camera_orbit_set(&cam, v3(0.5f, 0.5f, 0.5f), 2.0f);
    int t = (int)tf_idx;
    if (igCombo_Str("transfer fn", &t, "gray\0scroll\0high-pass\0", 3)) {
      tf_idx = (uint32_t)t;
      r3d_tf tfp;
      uint8_t lut[256][4];
      r3d_tf_preset(tf_idx, &tfp);
      r3d_tf_build(&tfp, lut);
      r3d_set_transfer(renderer, lut);
      tf_min_v = tf_min_visible(lut);
    }
    igSliderFloat("step (voxels)", &step_voxels, 0.25f, 4.0f, "%.2f", 0);
    igSliderFloat("density", &density, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    igSliderFloat("low cut", &low_cut, 0.0f, 255.0f, "%.0f", 0);
    if (slab_wz) {
      igSliderInt("depth (voxels)", &slab_depth, 2, (int)slab_wz, "%d", 0);
      int z0i = (int)slab_z0;
      if (igSliderInt("z position", &z0i, 0, (int)slab_z0_max, "%d", 0)) {
        slab_z0 = (uint32_t)z0i;
        r3d_slab_window(renderer, &slab_src, slab_z0);
      }
      igCheckbox("auto-scroll", &auto_scroll);
      igSameLine(0, 10);
      igSliderFloat("slices/s", &auto_speed, -60.0f, 60.0f, "%.0f", 0);
    }
    if (clip_mode) {
      igSliderInt("depth (voxels)", &clip_depth, 2, (int)CLIP_DEPTH_MAX, "%d", 0);
      int z0i = (int)(clip_z0 - clip_z0_min);
      if (igSliderInt("z position", &z0i, 0, (int)(clip_z0_max - clip_z0_min), "%d", 0))
        clip_z0 = clip_z0_min + (uint64_t)z0i;
      igText("levels ready:%s%s%s%s%s%s", (clip_valid_disp & 1) ? " L0" : "",
             (clip_valid_disp & 2) ? " L1" : "", (clip_valid_disp & 4) ? " L2" : "",
             (clip_valid_disp & 8) ? " L3" : "", (clip_valid_disp & 16) ? " L4" : "",
             (clip_valid_disp & 32) ? " L5" : "");
    }
    if (vslab_mode) {
      if (umbilicus_path) {
        igSeparator();
        igText("umbilicus annotation   %zu point%s", umbilicus.count,
               umbilicus.count == 1 ? "" : "s");
        int zi = annotation_z;
        if (igSliderInt("slice z", &zi, 0, (int)vsnz - 1, "%d", 0)) annotation_z = zi;
        if (igInputInt("slice step", &annotation_step, 1, 100, 0)) {
          if (annotation_step < 1) annotation_step = 1;
          if (annotation_step > (int)vsnz - 1) annotation_step = (int)vsnz - 1;
        }
        if (igButton("< previous", (ImVec2){0, 0})) {
          annotation_z -= annotation_step;
          if (annotation_z < 0) annotation_z = 0;
        }
        igSameLine(0, 8);
        if (igButton("next >", (ImVec2){0, 0})) {
          annotation_z += annotation_step;
          if (annotation_z >= (int)vsnz) annotation_z = (int)vsnz - 1;
        }
        if (igButton("< annotated", (ImVec2){0, 0})) {
          for (size_t i = umbilicus.count; i > 0; i--)
            if (umbilicus.points[i - 1].z < annotation_z) {
              annotation_z = (int)llround(umbilicus.points[i - 1].z);
              break;
            }
        }
        igSameLine(0, 8);
        if (igButton("annotated >", (ImVec2){0, 0})) {
          for (size_t i = 0; i < umbilicus.count; i++)
            if (umbilicus.points[i].z > annotation_z) {
              annotation_z = (int)llround(umbilicus.points[i].z);
              break;
            }
        }
        const r3d_umbilicus_point *here = r3d_umbilicus_find(&umbilicus, annotation_z);
        if (here) {
          igText("current point: x %.1f   y %.1f", here->x, here->y);
          if (igButton("delete current", (ImVec2){0, 0}) &&
              r3d_umbilicus_remove(&umbilicus, annotation_z))
            save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
          igSameLine(0, 8);
        } else {
          igTextDisabled("current slice is not annotated");
        }
        if (igButton("save now", (ImVec2){0, 0}))
          save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        const char *annotation_name = strrchr(umbilicus_path, '/');
        igTextDisabled("%s | %s", annotation_name ? annotation_name + 1 : umbilicus_path,
                       annotation_status);
        r3d_vslab_prefetch_stats pcs;
        r3d_vslab_prefetch_get(renderer, &pcs);
        igText("decoded cache: %u/%u ready%s  hits %llu  last %.2f s", pcs.ready,
               pcs.capacity,
               pcs.filling ? " + filling" : "", (unsigned long long)pcs.hits,
               pcs.last_decode_ms / 1000.0);
        igTextDisabled("GPU fine-scroll neighborhood: +/- %d slices", annotation_z_prefetch);
      } else {
        int z0i = (int)vs_z0;
        int zmax = (int)vsnz - vsd;
        if (igSliderInt("z position (world)", &z0i, 0, zmax > 0 ? zmax : 0, "%d", 0))
          vs_z0 = z0i;
        igCheckbox("follow camera (x/y)", &vs_follow);
      }
      int64_t vo_[3];
      uint32_t pend;
      r3d_vslab_get(renderer, vo_, &pend);
      uint32_t rpend = r3d_vslab_resident_pending(renderer);
      igText("window @ (%lld, %lld, %lld)%s", (long long)vo_[0], (long long)vo_[1],
             (long long)vo_[2], pend ? "  streaming..." : (rpend ? "  prefetching..." : ""));
    }
    if (bricks_path) {
      r3d_bricks_stats bst;
      r3d_bricks_get_stats(renderer, &bst);
      if (brick_is_lod && brick_depth > 0) {
        int bz = brick_z;
        int zmax = (int)brick_shape[2] - brick_depth;
        if (igSliderInt("slice z", &bz, 0, zmax > 0 ? zmax : 0, "%d", 0) && bz != brick_z) {
          uint32_t md = brick_shape[0] > brick_shape[1] ? brick_shape[0] : brick_shape[1];
          if (brick_shape[2] > md) md = brick_shape[2];
          float dz = (float)(bz - brick_z) / (float)md;
          brick_z = bz;
          if (cam_mode == CAM_ORBIT) {
            cam.target.z += dz;
            cam.pos.z += dz;
          }
        }
        int bd = brick_depth;
        int dmax = brick_shape[2] < 256u ? (int)brick_shape[2] : 256;
        if (igSliderInt("slice depth", &bd, 1, dmax, "%d", 0)) {
          brick_depth = bd;
          if (brick_z > (int)brick_shape[2] - brick_depth)
            brick_z = (int)brick_shape[2] - brick_depth;
        }
      }
      if (multiview_path) {
        igSeparator();
        igText("multiview focus  x %.0f  y %.0f  z %.0f", mv_focus[0], mv_focus[1],
               mv_focus[2]);
        igText("XY z %.0f | XZ y %.0f | YZ x %.0f", mv[R3D_MV_XY].slice,
               mv[R3D_MV_XZ].slice, mv[R3D_MV_YZ].slice);
        int th = mv_thick;
        if (igSliderInt("slice thickness", &th, 1, 128, "%d", 0)) mv_thick = th;
        float zo = (float)mv[R3D_MV_SEG].slice;
        if (igSliderFloat("segment offset", &zo, -64.0f, 64.0f, "%.0f vox", 0))
          mv[R3D_MV_SEG].slice = (double)zo;
        igTextDisabled("segment %ux%u  %llu valid points", mv_seg.w, mv_seg.h,
                       (unsigned long long)mv_seg.nvalid);
      }
      igText("bricks: hot %u/%u slots  warm %u (%.0f/%llu MB)%s", bst.hot, bst.hot_cap,
             bst.warm_bricks, (double)bst.warm_bytes / 1048576.0,
             (unsigned long long)(bst.warm_cap >> 20), bst.inflight ? "  streaming..." : "");
      if (bst.nlevels > 1)
        igText("wanted L0..L%u: %u %u %u %u %u %u %u %u", bst.nlevels - 1u,
               bst.lod_wanted[0], bst.lod_wanted[1], bst.lod_wanted[2], bst.lod_wanted[3],
               bst.lod_wanted[4], bst.lod_wanted[5], bst.lod_wanted[6], bst.lod_wanted[7]);
    }
    igSliderFloat("lod bias", &lod_bias, -2.0f, 4.0f, "%.2f", 0);
    int qp = quality_policy;
    if (igCombo_Str("quality", &qp, "full\0interactive\0fast\0\0", 3)) {
      quality_policy = qp;
      adaptive_res = quality_policy != 0;
      r3d_set_quality(renderer, quality_policy == 2 ? R3D_QUALITY_FAST : R3D_QUALITY_FULL);
    }
    igCheckbox("half-res while moving", &adaptive_res);
    if (igCollapsingHeader_TreeNodeFlags("transform", 0)) {
      float vt[3] = {vol_t.x, vol_t.y, vol_t.z};
      if (igDragFloat3("volume pos", vt, 0.002f, -4.0f, 4.0f, "%.3f", 0))
        vol_t = v3(vt[0], vt[1], vt[2]);
      float vr[3] = {vol_rot[0] * 57.29578f, vol_rot[1] * 57.29578f, vol_rot[2] * 57.29578f};
      if (igDragFloat3("volume rot (ypr)", vr, 0.5f, -180.0f, 180.0f, "%.1f", 0))
        for (int k = 0; k < 3; k++) vol_rot[k] = vr[k] / 57.29578f;
      if (igButton("reset volume", (ImVec2){0, 0})) {
        vol_t = v3(0, 0, 0);
        vol_rot[0] = vol_rot[1] = vol_rot[2] = 0.0f;
      }
      igSeparator();
      float ct[3] = {cam.target.x, cam.target.y, cam.target.z};
      if (igDragFloat3("cam target", ct, 0.002f, -4.0f, 4.0f, "%.3f", 0) &&
          cam_mode == CAM_ORBIT)
        r3d_camera_orbit_set(&cam, v3(ct[0], ct[1], ct[2]), cam.dist);
      if (igDragFloat("cam dist", &cam.dist, 0.005f, 0.0005f, 20.0f, "%.4f", 0) &&
          cam_mode == CAM_ORBIT)
        r3d_camera_orbit_set(&cam, cam.target, cam.dist);
      igSliderFloat("fov", &fov_deg, 20.0f, 120.0f, "%.0f°", 0);
    }
    igText("cam (%.2f %.2f %.2f) yaw %.2f pitch %.2f", (double)cam.pos.x, (double)cam.pos.y,
           (double)cam.pos.z, (double)cam.yaw, (double)cam.pitch);
    if (igCollapsingHeader_TreeNodeFlags("profile", 0)) {
      igText("gpu total   %6.2f ms", (double)prof.gpu_ns / 1e6);
      igText("  raycast   %6.2f ms", (double)prof.gpu_raycast_ns / 1e6);
      igText("  blit      %6.2f ms", (double)prof.gpu_blit_ns / 1e6);
      igText("  gui       %6.2f ms", (double)prof.gpu_gui_ns / 1e6);
      igText("cpu wait    %6.2f ms", (double)prof.cpu_wait_ns / 1e6);
      igText("cpu acquire %6.2f ms", (double)prof.cpu_acquire_ns / 1e6);
      igText("cpu record  %6.2f ms", (double)prof.cpu_record_ns / 1e6);
      igText("cpu submit  %6.2f ms", (double)prof.cpu_submit_ns / 1e6);
    }
    if (umbilicus_path)
      igTextDisabled("click: set point | wheel, R/F: 1 slice\n"
                     "PgUp/PgDn: z step | Ctrl+click: set + advance\n"
                     "Shift+drag: pan | Shift+wheel: zoom | F12: shot");
    else if (multiview_path)
      igTextDisabled("drag: pan view | wheel: zoom | Shift+wheel: slice\n"
                     "R/F: slice | Ctrl+click: set focus | F12: shot");
    else if (cam_mode == CAM_ORBIT)
      igTextDisabled("drag orbit | shift+drag pan cam | ctrl+drag move vol\n"
                     "ctrl+shift+drag rot vol | wheel zoom | WASD pan | F12 shot");
    else
      igTextDisabled("click: fly (Esc releases)   WASD+QE: move   F12: shot");
    igEnd();

    if (umbilicus_path) {
      if (annotation_z < 0) annotation_z = 0;
      if (annotation_z >= (int)vsnz) annotation_z = (int)vsnz - 1;
      vs_z0 = annotation_z0(annotation_z, vsnz, (uint32_t)vsd);
      const r3d_umbilicus_point *here = r3d_umbilicus_find(&umbilicus, annotation_z);
      ImVec2 marker;
      if (annotation_project(here, &cam, right, up, fwd, vm, vol_t, w, h, (uint32_t)vsw,
                             (uint32_t)vsh, (uint32_t)vsd, 0, 0, vs_z0, &marker)) {
        ImDrawList *draw = igGetBackgroundDrawList_Nil();
        const ImU32 black = 0xff000000u, yellow = 0xff00ffffu;
        ImDrawList_AddCircle(draw, marker, 12.0f, black, 32, 5.0f);
        ImDrawList_AddCircle(draw, marker, 12.0f, yellow, 32, 2.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x - 18.0f, marker.y},
                           (ImVec2){marker.x + 18.0f, marker.y}, black, 5.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x, marker.y - 18.0f},
                           (ImVec2){marker.x, marker.y + 18.0f}, black, 5.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x - 18.0f, marker.y},
                           (ImVec2){marker.x + 18.0f, marker.y}, yellow, 2.0f);
        ImDrawList_AddLine(draw, (ImVec2){marker.x, marker.y - 18.0f},
                           (ImVec2){marker.x, marker.y + 18.0f}, yellow, 2.0f);
      }
    }

    if (multiview_path) { /* overlays: intersections, focus marker, borders */
      /* recompute intersection polylines only when their inputs move */
      double zoff = mv[R3D_MV_SEG].slice;
      for (int i = 1; i < 4; i++) {
        const uint8_t *ax = r3d_mv_axes[i];
        if (mv_ol_slice[i] != mv[i].slice) {
          mv_ol[i].n = 0;
          r3d_segtrace(&mv_seg, NULL, 0.0f, ax[2], ax[0], ax[1], mv[i].slice, mv_lines_emit,
                       &mv_ol[i]);
        }
        if (mv_ol_slice[i] != mv[i].slice || mv_ol_zoff != zoff) {
          mv_ol_off[i].n = 0;
          if (zoff != 0.0)
            r3d_segtrace(&mv_seg, mv_normals, (float)zoff, ax[2], ax[0], ax[1], mv[i].slice,
                         mv_lines_emit, &mv_ol_off[i]);
        }
        mv_ol_slice[i] = mv[i].slice;
      }
      mv_ol_zoff = zoff;

      ImDrawList *draw = igGetBackgroundDrawList_Nil();
      const ImU32 fc = 0xffd7ff32u; /* vc3d focus teal (50,255,215), ABGR */
      const ImU32 bc = 0x60808080u;
      const ImU32 seg_col = 0xff50dcffu;      /* active segment: warm yellow */
      const ImU32 seg_off_col = 0x9050dcffu;  /* zoff shell: same, translucent */
      /* plane trace colors on the segment view (vc3d): XY orange, XZ red,
       * YZ yellow */
      const ImU32 trace_col[4] = {0, 0xff008cffu, 0xff0000ffu, 0xff00ffffu};
      for (int i = 1; i < 4; i++) { /* segment curve on each plane view */
        ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
        ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
        ImDrawList_PushClipRect(draw, cmin, cmax, false);
        for (uint32_t k = 0; k < mv_ol[i].n; k++) {
          const float *s4 = mv_ol[i].w + (size_t)k * 4;
          float x0, y0, x1, y1;
          r3d_mv_project(&mv[i], (double)s4[0], (double)s4[1], &x0, &y0);
          r3d_mv_project(&mv[i], (double)s4[2], (double)s4[3], &x1, &y1);
          ImDrawList_AddLine(draw, (ImVec2){x0, y0}, (ImVec2){x1, y1}, seg_col, 1.6f);
        }
        for (uint32_t k = 0; k < mv_ol_off[i].n; k++) {
          const float *s4 = mv_ol_off[i].w + (size_t)k * 4;
          float x0, y0, x1, y1;
          r3d_mv_project(&mv[i], (double)s4[0], (double)s4[1], &x0, &y0);
          r3d_mv_project(&mv[i], (double)s4[2], (double)s4[3], &x1, &y1);
          ImDrawList_AddLine(draw, (ImVec2){x0, y0}, (ImVec2){x1, y1}, seg_off_col, 1.0f);
        }
        ImDrawList_PopClipRect(draw);
      }
      { /* plane trace lines on the flattened segment view */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        ImVec2 cmin = {(float)sv->px, (float)sv->py};
        ImVec2 cmax = {(float)(sv->px + sv->pw), (float)(sv->py + sv->ph)};
        ImDrawList_PushClipRect(draw, cmin, cmax, false);
        for (int i = 1; i < 4; i++)
          for (uint32_t k = 0; k < mv_ol[i].n; k++) {
            const float *g4 = mv_ol[i].g + (size_t)k * 4;
            float x0, y0, x1, y1;
            r3d_mv_project(sv, (double)g4[0], (double)g4[1], &x0, &y0);
            r3d_mv_project(sv, (double)g4[2], (double)g4[3], &x1, &y1);
            ImDrawList_AddLine(draw, (ImVec2){x0, y0}, (ImVec2){x1, y1}, trace_col[i], 1.4f);
          }
        ImDrawList_PopClipRect(draw);
      }
      ImDrawList_AddLine(draw, (ImVec2){(float)(w / 2), 0}, (ImVec2){(float)(w / 2), (float)h},
                         bc, 1.0f);
      ImDrawList_AddLine(draw, (ImVec2){0, (float)(h / 2)}, (ImVec2){(float)w, (float)(h / 2)},
                         bc, 1.0f);
      for (int i = 1; i < 4; i++) {
        const uint8_t *ax = r3d_mv_axes[i];
        float fx_, fy_;
        r3d_mv_project(&mv[i], mv_focus[ax[0]], mv_focus[ax[1]], &fx_, &fy_);
        if (fx_ >= (float)mv[i].px && fx_ < (float)(mv[i].px + mv[i].pw) &&
            fy_ >= (float)mv[i].py && fy_ < (float)(mv[i].py + mv[i].ph))
          ImDrawList_AddCircle(draw, (ImVec2){fx_, fy_}, 10.0f, fc, 24, 2.0f);
      }
    }

    r3d_frame_params p = {
        .cam_origin = {cam.pos.x, cam.pos.y, cam.pos.z},
        .cam_right = {right.x, right.y, right.z},
        .cam_up = {up.x, up.y, up.z},
        .cam_forward = {fwd.x, fwd.y, fwd.z},
        .step_voxels = step_voxels,
        .density = density,
        .lod_bias = lod_bias,
        .max_mip = 10.0f,
        .viewport = {rvw, rvh},
        .mode = mode,
        .frame_index = frame_index++,
        .threshold = low_cut / 255.0f,
        .skip_gate = fmaxf(low_cut, tf_min_v - 0.5f) / 255.0f,
    };
    memcpy(p.vol_r0, &vm.r0, 12);
    memcpy(p.vol_r1, &vm.r1, 12);
    memcpy(p.vol_r2, &vm.r2, 12);
    p.vol_tx = vol_t.x;
    p.vol_ty = vol_t.y;
    p.vol_tz = vol_t.z;
    if (slab_wz) {
      r3d_slab_params(renderer, &p);
      p.slab_depth = (uint32_t)slab_depth;
    }
    if (vslab_mode) {
      /* focus = view axis ^ window mid-plane, in WINDOW space -> world */
      float vey = (float)vsh / (float)vsw, vez = (float)(vsd + 2) / (float)vsw;
      r3d_v3 vc = v3(0.5f, vey * 0.5f, vez * 0.5f);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float tt = vd.z != 0.0f ? (vez * 0.5f - vo.z) / vd.z : 0.0f;
      if (tt < 0.0f) tt = 0.0f;
      float fxn = fclampf(vo.x + vd.x * tt, 0.0f, 1.0f);
      float fyn = fclampf((vo.y + vd.y * tt) / (vey > 0.0f ? vey : 1.0f), 0.0f, 1.0f);
      int64_t vo3[3];
      r3d_vslab_get(renderer, vo3, NULL);
      if (vs_follow && vo3[0] >= 0) {
        vs_fx = (double)vo3[0] + (double)fxn * vsw;
        vs_fy = (double)vo3[1] + (double)fyn * vsh;
      }
      r3d_vslab_frame(renderer, vs_fx, vs_fy, vs_z0, moving ? 1u : 3u, &p);
      if (umbilicus_path) {
        int64_t co[3];
        r3d_vslab_get(renderer, co, NULL);
        uint32_t cpending = r3d_vslab_resident_pending(renderer);
        if (annotation_z > annotation_last_z) annotation_dir = 1;
        else if (annotation_z < annotation_last_z) annotation_dir = -1;
        annotation_last_z = annotation_z;
        if (cpending == 0 && co[2] == vs_z0) {
          int64_t targets[R3D_VSLAB_PREFETCH_MAX];
          uint32_t ntargets = 0;
          for (int k = 1; k <= annotation_prefetch; k++) {
            int64_t az = (int64_t)annotation_z +
                         (int64_t)annotation_dir * annotation_step * k;
            if (az < 0) az = 0;
            if (az >= vsnz) az = (int64_t)vsnz - 1;
            int64_t pz = annotation_z0((int)az, vsnz, (uint32_t)vsd);
            if (pz != vs_z0) targets[ntargets++] = pz;
          }
          int64_t back = (int64_t)annotation_z -
                         (int64_t)annotation_dir * annotation_step;
          if (back < 0) back = 0;
          if (back >= vsnz) back = (int64_t)vsnz - 1;
          int64_t back_z0 = annotation_z0((int)back, vsnz, (uint32_t)vsd);
          if (back_z0 != vs_z0 && ntargets < R3D_VSLAB_PREFETCH_MAX)
            targets[ntargets++] = back_z0;
          r3d_vslab_prefetch(renderer, targets, ntargets);
        } else {
          r3d_vslab_prefetch(renderer, NULL, 0);
        }
      }
      /* residency-lag metric: cells short of full residency, second half of
       * the run only (the first half absorbs the initial window fill) */
      if (bench && frame_index > warmup_frames &&
          (frame_index - warmup_frames) * 2 >= exit_frames) {
        int64_t bo_[3];
        uint32_t pd = 0;
        r3d_vslab_get(renderer, bo_, &pd);
        vs_pend_acc += pd;
      }
    }
    if (bricks_path) {
      p.slab_z0 = (float)brick_z;
      p.slab_depth = (uint32_t)brick_depth;
      /* streaming pump: camera in VOLUME space (model transform inverted, like
       * the clip focus); smaller decode budget while moving so the pump's GPU
       * time shares the frame with half-res rendering */
      float bext[3];
      r3d_bricks_extent(renderer, bext);
      r3d_v3 vc = v3(bext[0] * 0.5f, bext[1] * 0.5f, bext[2] * 0.5f);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float be[3] = {vo.x, vo.y, vo.z}, bf[3] = {vd.x, vd.y, vd.z};
      float asp = (float)w / (float)h;
      float ht = tanf(cam.fov_y * 0.5f) * sqrtf(1.0f + asp * asp);
      float pixel_cone = 2.0f * tanf(cam.fov_y * 0.5f) / (float)(rvh ? rvh : 1) * exp2f(lod_bias);
      if (multiview_path) {
        /* per-view AABB collects: each plane view wants its visible rect at
         * its own magnification, +- the slab thickness along the normal */
        uint32_t mdim = brick_shape[0];
        for (int a = 1; a < 3; a++)
          if (brick_shape[a] > mdim) mdim = brick_shape[a];
        if (r3d_bricks_stream_begin(renderer)) {
          { /* segment view: walk the visible grid rect decimated and request
             * the bricks its surface points (+ normal offset) touch */
            const r3d_mview *sv = &mv[R3D_MV_SEG];
            double hw = (double)sv->pw * 0.5 / sv->zoom, hh = (double)sv->ph * 0.5 / sv->zoom;
            int64_t g0 = (int64_t)(sv->cu - hw), g1 = (int64_t)(sv->cu + hw) + 1;
            int64_t j0 = (int64_t)(sv->cv - hh), j1 = (int64_t)(sv->cv + hh) + 1;
            if (g0 < 0) g0 = 0;
            if (j0 < 0) j0 = 0;
            if (g1 > (int64_t)mv_seg.w) g1 = mv_seg.w;
            if (j1 > (int64_t)mv_seg.h) j1 = mv_seg.h;
            double span = (double)((g1 - g0) * (j1 - j0));
            int64_t step = span > 0 ? (int64_t)(sqrt(span / 384.0) + 1.0) : 1;
            /* voxels per pixel: (grid units per px) / (grid units per voxel) */
            float vf = (float)(1.0 / (sv->zoom * (double)mv_seg.sx)) * exp2f(lod_bias);
            uint32_t lvl = 0;
            while (vf >= 2.0f && lvl < 7u) {
              vf *= 0.5f;
              lvl++;
            }
            for (int64_t gj = j0; gj < j1; gj += step)
              for (int64_t gi = g0; gi < g1; gi += step) {
                const float *sp = r3d_tifxyz_at(&mv_seg, (uint32_t)gi, (uint32_t)gj);
                if (!r3d_tifxyz_valid(sp)) continue;
                float pp[3] = {sp[0] / (float)mdim, sp[1] / (float)mdim,
                               sp[2] / (float)mdim};
                r3d_bricks_stream_point(renderer, pp, lvl, p.skip_gate);
              }
          }
          for (int i = 1; i < 4; i++) {
            const uint8_t *ax = r3d_mv_axes[i];
            double hw = (double)mv[i].pw * 0.5 / mv[i].zoom;
            double hh = (double)mv[i].ph * 0.5 / mv[i].zoom;
            double th = (double)mv_thick;
            double wlo[3], whi[3];
            wlo[ax[0]] = mv[i].cu - hw;
            whi[ax[0]] = mv[i].cu + hw;
            wlo[ax[1]] = mv[i].cv - hh;
            whi[ax[1]] = mv[i].cv + hh;
            wlo[ax[2]] = mv[i].slice - 1.0;
            whi[ax[2]] = mv[i].slice + th + 1.0;
            float lo[3], hi[3];
            for (int a = 0; a < 3; a++) {
              lo[a] = (float)(wlo[a] / (double)mdim);
              hi[a] = (float)(whi[a] / (double)mdim);
            }
            float vpp = (float)(1.0 / (mv[i].zoom * (double)mdim)) * exp2f(lod_bias);
            r3d_bricks_stream_box(renderer, lo, hi, vpp, p.skip_gate);
          }
          r3d_bricks_stream_submit(renderer, moving ? 3u : 8u);
        }
      } else {
        r3d_bricks_stream(renderer, be, bf, ht, pixel_cone, (uint32_t)brick_z,
                          (uint32_t)brick_depth, p.skip_gate, moving ? 2u : 6u);
      }
      r3d_bricks_params(renderer, &p);
    }
    if (clip_mode) {
      /* focus = where the view axis crosses the slab plane, computed in
       * VOLUME space so it tracks the model transform */
      float ezc = (float)CLIP_DEPTH_MAX / (float)CLIP_NX * 0.5f;
      r3d_v3 vc = v3(0.5f, 0.5f, ezc);
      r3d_v3 vo = v3_add(m3_tmul(vm, v3_sub(v3_sub(cam.pos, vol_t), vc)), vc);
      r3d_v3 vd = m3_tmul(vm, fwd);
      float tt = vd.z != 0.0f ? (ezc - vo.z) / vd.z : 0.0f;
      if (tt < 0.0f) tt = 0.0f;
      double fx = (double)(vo.x + vd.x * tt) * CLIP_NX;
      double fy = (double)(vo.y + vd.y * tt) * CLIP_NX;
      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx > CLIP_NX) fx = CLIP_NX;
      if (fy > CLIP_NX) fy = CLIP_NX;
      p.slab_depth = (uint32_t)clip_depth;
      if (r3d_clip_frame(renderer, fx, fy, clip_z0, &p) != 0) running = false;
      if (p.clip_valid != clip_valid_disp)
        printf("clip: valid=0x%02x z0=%llu focus=(%.0f,%.0f)\n", p.clip_valid,
               (unsigned long long)clip_z0, fx, fy);
      clip_valid_disp = p.clip_valid;
    }
    r3d_frame_stats st = {0};
    int frc;
    if (multiview_path) {
      /* one FrameParams per quadrant: axis-aligned ortho cameras over the
       * bricks virtual volume, slab-clipped to each view's slice */
      uint32_t mdim = brick_shape[0];
      for (int a = 1; a < 3; a++)
        if (brick_shape[a] > mdim) mdim = brick_shape[a];
      static const uint32_t axis_code[3] = {1u, 2u, 0u}; /* world axis -> view_flags code */
      r3d_frame_params vp4[4];
      for (int i = 0; i < 4; i++) {
        const uint8_t *ax = r3d_mv_axes[i];
        r3d_frame_params q = p;
        q.viewport[0] = (uint32_t)mv[i].pw;
        q.viewport[1] = (uint32_t)mv[i].ph;
        q.view_org = (uint32_t)mv[i].px | ((uint32_t)mv[i].py << 16);
        if (i == R3D_MV_SEG) {
          /* flattened segment: raycast the surface-volume window. Camera in
           * FLATTENED VOXELS (grid / scale); window mapping from the backend. */
          q.view_flags = R3D_VIEW_SURF;
          q.cam_origin[0] = (float)(mv[i].cu / (double)mv_seg.sx);
          q.cam_origin[1] = (float)(mv[i].cv / (double)mv_seg.sy);
          q.cam_origin[2] = 0.0f;
          memset(q.cam_right, 0, sizeof q.cam_right);
          memset(q.cam_up, 0, sizeof q.cam_up);
          memset(q.cam_forward, 0, sizeof q.cam_forward);
          q.cam_right[0] = (float)((double)mv[i].pw * 0.5 / mv[i].zoom / (double)mv_seg.sx);
          q.cam_up[1] = (float)-((double)mv[i].ph * 0.5 / mv[i].zoom / (double)mv_seg.sy);
          r3d_surfvol_params(renderer, &q);
          q.slab_z0 = (float)mv[i].slice; /* render-time normal offset (voxels) */
          q.slab_depth = (uint32_t)mv_thick; /* marched thickness (voxels) */
          vp4[i] = q;
          continue;
        }
        q.view_flags = R3D_VIEW_ORTHO | R3D_VIEW_AXIS(axis_code[ax[2]]);
        double hw = (double)mv[i].pw * 0.5 / mv[i].zoom / (double)mdim;
        double hh = (double)mv[i].ph * 0.5 / mv[i].zoom / (double)mdim;
        float org[3], rgt[3] = {0, 0, 0}, upv[3] = {0, 0, 0}, fwdv[3] = {0, 0, 0};
        org[ax[0]] = (float)(mv[i].cu / (double)mdim);
        org[ax[1]] = (float)(mv[i].cv / (double)mdim);
        org[ax[2]] = (float)((mv[i].slice - 2.0) / (double)mdim);
        rgt[ax[0]] = (float)hw;   /* screen +x -> +u */
        upv[ax[1]] = (float)-hh;  /* ndc +y (screen top) -> smaller v */
        fwdv[ax[2]] = 1.0f;
        memcpy(q.cam_origin, org, sizeof org);
        memcpy(q.cam_right, rgt, sizeof rgt);
        memcpy(q.cam_up, upv, sizeof upv);
        memcpy(q.cam_forward, fwdv, sizeof fwdv);
        q.slab_z0 = (float)mv[i].slice;
        q.slab_depth = (uint32_t)mv_thick;
        vp4[i] = q;
      }
      frc = r3d_frame_views(renderer, vp4, 4, &st);
    } else {
      frc = r3d_frame(renderer, &p, &st);
    }
    bool measuring = !exit_frames || frame_index > warmup_frames;
    if (frc == 0 && measuring) {
      last_gpu_ns = st.gpu_ns;
      const uint64_t *sv = (const uint64_t *)&st;
      uint64_t *pv = (uint64_t *)&prof, *qv = (uint64_t *)&prof_sum;
      for (size_t k = 0; k < sizeof st / sizeof(uint64_t); k++) {
        pv[k] = (uint64_t)((double)pv[k] * 0.95 + (double)sv[k] * 0.05);
        qv[k] += sv[k];
      }
      if (prof_samples && prof_frames < exit_frames) prof_samples[prof_frames] = st;
      prof_frames++;
    }
    if (frc < 0) {
      fprintf(stderr, "r3d_frame failed\n");
      running = false;
    }
    if (in.screenshot) take_screenshot(renderer, stats.frame_index);
    if (total_frames && frame_index >= total_frames) {
      if (shot_path) {
        uint32_t sw = 0, sh = 0;
        r3d_read_frame(renderer, NULL, &sw, &sh);
        uint8_t *rgba = malloc((size_t)sw * sh * 4);
        if (rgba && r3d_read_frame(renderer, rgba, &sw, &sh) == 0)
          r3d_screenshot_ppm(shot_path, rgba, sw, sh);
        free(rgba);
      }
      running = false;
    }

    if (measuring) {
      r3d_stats_push(&stats, r3d_now_ns() - t0, st.gpu_ns);
      r3d_stats_report(&stats);
    }
  }

  r3d_stats_report_now(&stats);
  if (slab_src.voxels) r3d_volume_close(&slab_src);
  if (prof_frames > 2) {
    /* skip warmup skew: averages include first frames with empty queries */
    double n = (double)prof_frames;
    printf("profile avg: gpu %.2f (raycast %.2f blit %.2f gui %.2f) | "
           "wait %.2f acquire %.2f record %.2f submit %.2f ms\n",
           (double)prof_sum.gpu_ns / n / 1e6, (double)prof_sum.gpu_raycast_ns / n / 1e6,
           (double)prof_sum.gpu_blit_ns / n / 1e6, (double)prof_sum.gpu_gui_ns / n / 1e6,
           (double)prof_sum.cpu_wait_ns / n / 1e6, (double)prof_sum.cpu_acquire_ns / n / 1e6,
           (double)prof_sum.cpu_record_ns / n / 1e6, (double)prof_sum.cpu_submit_ns / n / 1e6);
  }
  if (vslab_mode && bench) {
    int64_t final_o[3];
    uint32_t final_visible = 0;
    r3d_vslab_get(renderer, final_o, &final_visible);
    printf("vslab bench: pending cell-frames %llu, final visible %u resident %u\n",
           (unsigned long long)vs_pend_acc, final_visible,
           r3d_vslab_resident_pending(renderer));
  }
  if (umbilicus_path) {
    r3d_vslab_prefetch_stats pcs;
    r3d_vslab_prefetch_get(renderer, &pcs);
    printf("vslab decoded cache: %u ready, %llu hits, %llu misses, last %.0f ms\n", pcs.ready,
           (unsigned long long)pcs.hits, (unsigned long long)pcs.misses, pcs.last_decode_ms);
  }
  r3d_bricks_stats final_bst = {0};
  if (bricks_path) {
    r3d_bricks_flush(renderer);
    r3d_bricks_get_stats(renderer, &final_bst);
    printf("bricks bench: decoded %llu in %llu jobs, %.2f ms/job, %u failure(s), hot %u/%u\n",
           (unsigned long long)final_bst.decoded, (unsigned long long)final_bst.jobs,
           final_bst.jobs ? (double)final_bst.stream_ns / (double)final_bst.jobs / 1e6 : 0.0,
           final_bst.failures, final_bst.hot, final_bst.hot_cap);
    if (final_bst.nlevels > 1)
      printf("bricks LOD wanted: [%u,%u,%u,%u,%u,%u,%u,%u]\n", final_bst.lod_wanted[0],
             final_bst.lod_wanted[1], final_bst.lod_wanted[2], final_bst.lod_wanted[3],
             final_bst.lod_wanted[4], final_bst.lod_wanted[5], final_bst.lod_wanted[6],
             final_bst.lod_wanted[7]);
    if (final_bst.nlevels > 1)
      printf("bricks LOD requests: [%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
             (unsigned long long)final_bst.lod_requests[0],
             (unsigned long long)final_bst.lod_requests[1],
             (unsigned long long)final_bst.lod_requests[2],
             (unsigned long long)final_bst.lod_requests[3],
             (unsigned long long)final_bst.lod_requests[4],
             (unsigned long long)final_bst.lod_requests[5],
             (unsigned long long)final_bst.lod_requests[6],
             (unsigned long long)final_bst.lod_requests[7]);
  }
  if (bench_json)
    write_bench_json(bench_json, bench_name ? bench_name : bench, win_w, win_h, quality_arg,
                     warmup_frames, &stats, prof_samples, prof_frames, vs_pend_acc, &final_bst);
  if (umbilicus_path && umbilicus.dirty)
    save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
  r3d_umbilicus_free(&umbilicus);
  if (multiview_path) {
    r3d_tifxyz_free(&mv_seg);
    free(mv_normals);
    for (int i = 0; i < 4; i++) {
      free(mv_ol[i].w);
      free(mv_ol[i].g);
      free(mv_ol_off[i].w);
      free(mv_ol_off[i].g);
    }
  }
  free(prof_samples);
  r3d_destroy(renderer);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
