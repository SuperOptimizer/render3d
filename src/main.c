/* render3d — volumetric renderer for Vesuvius Challenge micro-CT volumes.
 * M1: SDL3 window + Vulkan compute raycaster (see spec/ and docs/measured.md). */
#include <SDL3/SDL.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cimgui.h"
#include "core/camera.h"
#include "core/odbrowse.h"
#include "core/transfer.h"
#include "core/volume.h"
#include "core/input.h"
#include "core/mview.h"
#include "core/screenshot.h"
#include "core/segstore.h"
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

/* Draw a cached segment soup as chained polylines: marching squares emits
 * mostly head-to-tail runs, so consecutive segments sharing an endpoint merge
 * into one AddPolyline, sub-pixel steps collapse into the previous point, and
 * chains whose bounds miss the pane are dropped before tessellation. The
 * per-segment AddLine version tessellated + uploaded thousands of AA quads
 * every frame regardless of zoom (~25% of the render thread on the GP
 * banner). */
static void mv_draw_lines(ImDrawList *draw, const r3d_mview *v, const float *seg, uint32_t n,
                          ImU32 col, float thick, float collapse_px2) {
  static ImVec2 *pts;
  static uint32_t cap;
  if (!n) return;
  if (cap < n + 2) {
    uint32_t nc = n + 2;
    ImVec2 *np_ = realloc(pts, (size_t)nc * sizeof *np_);
    if (!np_) return;
    pts = np_;
    cap = nc;
  }
  float pxmin = (float)v->px, pymin = (float)v->py;
  float pxmax = pxmin + (float)v->pw, pymax = pymin + (float)v->ph;
  uint32_t np = 0;
  float bx0 = 0.0f, by0 = 0.0f, bx1 = 0.0f, by1 = 0.0f; /* chain screen bbox */
  float lu = 0.0f, lv = 0.0f;                           /* chain tail, source space */
  for (uint32_t k = 0; k <= n; k++) {
    float nu = 0.0f, nv = 0.0f;
    bool brk = k == n;
    if (!brk) {
      const float *s4 = seg + (size_t)k * 4;
      if (np == 0) {
        float x, y;
        r3d_mv_project(v, (double)s4[0], (double)s4[1], &x, &y);
        pts[np++] = (ImVec2){x, y};
        bx0 = bx1 = x;
        by0 = by1 = y;
        nu = s4[2];
        nv = s4[3];
      } else if (fabsf(s4[0] - lu) < 1e-2f && fabsf(s4[1] - lv) < 1e-2f) {
        nu = s4[2];
        nv = s4[3];
      } else if (fabsf(s4[2] - lu) < 1e-2f && fabsf(s4[3] - lv) < 1e-2f) {
        nu = s4[0];
        nv = s4[1];
      } else {
        brk = true;
        k--; /* flush, then revisit this segment as a new chain start */
      }
    }
    if (brk) {
      if (np >= 2 && bx1 >= pxmin && bx0 < pxmax && by1 >= pymin && by0 < pymax)
        ImDrawList_AddPolyline(draw, pts, (int)np, col, thick, 0);
      np = 0;
      continue;
    }
    float x, y;
    r3d_mv_project(v, (double)nu, (double)nv, &x, &y);
    float dx = x - pts[np - 1].x, dy = y - pts[np - 1].y;
    if (np >= 2 && dx * dx + dy * dy < collapse_px2)
      pts[np - 1] = (ImVec2){x, y}; /* sub-pixel step: slide the chain tail */
    else
      pts[np++] = (ImVec2){x, y};
    if (x < bx0) bx0 = x;
    if (x > bx1) bx1 = x;
    if (y < by0) by0 = y;
    if (y > by1) by1 = y;
    lu = nu;
    lv = nv;
  }
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

/* nearest valid surface grid point to a world position (coarse step-2 scan);
 * true when it's within vc3d's 100-voxel focus tolerance */
static bool mv_nearest_surface(const r3d_tifxyz *s, const double f[3], uint32_t out_ij[2]) {
  double best = 1e30;
  uint32_t bi = 0, bj = 0;
  for (uint32_t gj = 0; gj < s->h; gj += 2)
    for (uint32_t gi = 0; gi < s->w; gi += 2) {
      const float *sp = r3d_tifxyz_at(s, gi, gj);
      if (!r3d_tifxyz_valid(sp)) continue;
      double dx = (double)sp[0] - f[0], dy = (double)sp[1] - f[1], dz = (double)sp[2] - f[2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        bi = gi;
        bj = gj;
      }
    }
  out_ij[0] = bi;
  out_ij[1] = bj;
  return best < 100.0 * 100.0;
}

/* Rebuild the segment-aligned frames for the XZ/YZ panes around surface grid
 * point ij: normal from the per-vertex normal grid, reference tangent = the
 * grid +i direction, origins at the focus. False (bases untouched) when the
 * normal is degenerate. */
static bool mv_seg_align(const r3d_tifxyz *s, const float *normals, const double focus[3],
                         const uint32_t ij[2], float theta_deg, double pb[4][3][3],
                         double po[4][3]) {
  size_t k = ((size_t)ij[1] * s->w + ij[0]) * 4;
  double n[3] = {(double)normals[k], (double)normals[k + 1], (double)normals[k + 2]};
  uint32_t i0 = ij[0] > 0 ? ij[0] - 1 : ij[0];
  uint32_t i1 = ij[0] + 1 < s->w ? ij[0] + 1 : ij[0];
  const float *pa = r3d_tifxyz_at(s, i0, ij[1]);
  const float *pb_ = r3d_tifxyz_at(s, i1, ij[1]);
  const float *pc_ = r3d_tifxyz_at(s, ij[0], ij[1]);
  if (!r3d_tifxyz_valid(pa)) pa = pc_;
  if (!r3d_tifxyz_valid(pb_)) pb_ = pc_;
  double tref[3] = {(double)pb_[0] - (double)pa[0], (double)pb_[1] - (double)pa[1],
                    (double)pb_[2] - (double)pa[2]};
  double ba[3][3], bb[3][3];
  if (r3d_mv_seg_frames(n, tref, (double)theta_deg * 0.017453292519943295, ba, bb) != 0)
    return false;
  memcpy(pb[R3D_MV_XZ], ba, sizeof ba);
  memcpy(pb[R3D_MV_YZ], bb, sizeof bb);
  for (int a = 0; a < 3; a++) po[R3D_MV_XZ][a] = po[R3D_MV_YZ][a] = focus[a];
  return true;
}

static void mv_axis_reset(double pb[4][3][3], double po[4][3]) {
  for (int i = R3D_MV_XZ; i <= R3D_MV_YZ; i++) {
    r3d_mv_axis_basis(i, pb[i]);
    for (int a = 0; a < 3; a++) po[i][a] = 0.0;
  }
}

/* ---- multi-segment corpus (--segments <store-dir>): a background worker
 * decodes decimated grids + row bounds into a RAM-budgeted cache; the GUI
 * thread only queries the store's tile index and traces cached grids, so
 * disk/decode never touches the frame loop. ---- */
enum { SGC_EMPTY = 0, SGC_QUEUED, SGC_READY, SGC_FAILED };

typedef struct sgc_ent {
  r3d_tifxyz s;      /* decimated grid (stride SGC_STRIDE) */
  r3d_segrows rows;  /* trace-skipping bounds for it */
  int state;         /* SGC_*; all access under mu */
  uint64_t last_use; /* LRU tick */
} sgc_ent;

#define SGC_STRIDE 4u

typedef struct sgcache {
  r3d_segstore st;
  sgc_ent *ent; /* [st.n] */
  pthread_t th;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  uint32_t *queue;
  uint32_t qn, qcap;
  bool quit, open;
  size_t bytes, budget;
  uint64_t tick;
  /* activation: the worker prepares a full-res grid + GPU grid arrays for
   * one segment; the GUI thread applies the swap when act_ready is set */
  uint32_t act_req, act_ready; /* UINT32_MAX = none */
  bool act_busy;
  r3d_tifxyz act_s;
  r3d_segrows act_rows;
  float *act_coords, *act_normals;
  /* overlap QC: ov[i] = fraction of surface i's tiles within ~8 voxels of
   * the active surface (worker-computed from the tile index alone) */
  float *ov;
  uint32_t ov_active, ov_req;
} sgcache;

static size_t sgc_ent_bytes(const sgc_ent *e) {
  return (size_t)e->s.w * e->s.h * 3 * sizeof(float) +
         ((size_t)e->rows.h * 3 + (size_t)e->rows.tw * e->rows.th * 3) * 2 * sizeof(float);
}

static void sgc_evict_lru(sgcache *c, uint32_t keep) { /* mu held */
  while (c->bytes > c->budget) {
    uint32_t victim = UINT32_MAX;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < c->st.n; i++)
      if (c->ent[i].state == SGC_READY && i != keep && c->ent[i].last_use < oldest) {
        oldest = c->ent[i].last_use;
        victim = i;
      }
    if (victim == UINT32_MAX) return;
    c->bytes -= sgc_ent_bytes(&c->ent[victim]);
    r3d_tifxyz_free(&c->ent[victim].s);
    r3d_segrows_free(&c->ent[victim].rows);
    c->ent[victim].state = SGC_EMPTY;
  }
}

static int mv_build_grids(const r3d_tifxyz *s, float **coords_out, float **normals_out);

static void *sgc_worker(void *ud) {
  sgcache *c = ud;
  pthread_mutex_lock(&c->mu);
  while (!c->quit) {
    if (c->act_req != UINT32_MAX && c->act_ready == UINT32_MAX) {
      uint32_t ai = c->act_req;
      c->act_req = UINT32_MAX;
      c->act_busy = true;
      pthread_mutex_unlock(&c->mu);
      r3d_tifxyz s;
      r3d_segrows rows = {0};
      float *co = NULL, *no = NULL;
      int ok = r3d_segstore_load(&c->st, ai, 1, &s) == 0;
      if (ok && (r3d_segrows_build(&s, &rows) != 0 || mv_build_grids(&s, &co, &no) != 0)) {
        r3d_tifxyz_free(&s);
        r3d_segrows_free(&rows);
        ok = 0;
      }
      pthread_mutex_lock(&c->mu);
      if (ok) {
        c->act_s = s;
        c->act_rows = rows;
        c->act_coords = co;
        c->act_normals = no;
        c->act_ready = ai;
      } else {
        c->act_busy = false;
      }
      continue;
    }
    if (c->ov_req != UINT32_MAX) {
      uint32_t oi = c->ov_req;
      c->ov_req = UINT32_MAX;
      pthread_mutex_unlock(&c->mu);
      float *tmp = malloc((c->st.n ? c->st.n : 1) * sizeof *tmp);
      if (tmp)
        for (uint32_t i = 0; i < c->st.n; i++)
          tmp[i] = i == oi ? 0.0f : (float)r3d_segstore_overlap(&c->st, i, oi, 8.0);
      pthread_mutex_lock(&c->mu);
      if (tmp && c->ov) {
        memcpy(c->ov, tmp, c->st.n * sizeof *c->ov);
        c->ov_active = oi;
      }
      free(tmp);
      continue;
    }
    if (c->qn == 0) {
      pthread_cond_wait(&c->cv, &c->mu);
      continue;
    }
    uint32_t i = c->queue[--c->qn];
    pthread_mutex_unlock(&c->mu);
    r3d_tifxyz s;
    r3d_segrows rows = {0};
    int ok = r3d_segstore_load(&c->st, i, SGC_STRIDE, &s) == 0;
    if (ok && r3d_segrows_build(&s, &rows) != 0) {
      r3d_tifxyz_free(&s);
      ok = 0;
    }
    pthread_mutex_lock(&c->mu);
    if (ok) {
      c->ent[i].s = s;
      c->ent[i].rows = rows;
      c->ent[i].state = SGC_READY;
      c->ent[i].last_use = ++c->tick;
      c->bytes += sgc_ent_bytes(&c->ent[i]);
      sgc_evict_lru(c, i);
    } else {
      c->ent[i].state = SGC_FAILED;
    }
  }
  pthread_mutex_unlock(&c->mu);
  return NULL;
}

static int sgc_open(sgcache *c, const char *store_dir, size_t budget) {
  memset(c, 0, sizeof *c);
  c->act_req = c->act_ready = UINT32_MAX;
  c->ov_active = c->ov_req = UINT32_MAX;
  if (r3d_segstore_open(&c->st, store_dir) != 0) return -1;
  c->ent = calloc(c->st.n ? c->st.n : 1, sizeof *c->ent);
  c->queue = malloc((c->st.n ? c->st.n : 1) * sizeof *c->queue);
  c->ov = calloc(c->st.n ? c->st.n : 1, sizeof *c->ov);
  if (!c->ent || !c->queue || !c->ov) {
    free(c->ent);
    free(c->queue);
    free(c->ov);
    r3d_segstore_close(&c->st);
    return -1;
  }
  c->qcap = c->st.n;
  c->budget = budget;
  pthread_mutex_init(&c->mu, NULL);
  pthread_cond_init(&c->cv, NULL);
  if (pthread_create(&c->th, NULL, sgc_worker, c) != 0) {
    free(c->ent);
    free(c->queue);
    free(c->ov);
    r3d_segstore_close(&c->st);
    return -1;
  }
  c->open = true;
  return 0;
}

static void sgc_close(sgcache *c) {
  if (!c->open) return;
  pthread_mutex_lock(&c->mu);
  c->quit = true;
  pthread_cond_signal(&c->cv);
  pthread_mutex_unlock(&c->mu);
  pthread_join(c->th, NULL);
  if (c->act_ready != UINT32_MAX) {
    r3d_tifxyz_free(&c->act_s);
    r3d_segrows_free(&c->act_rows);
    free(c->act_coords);
    free(c->act_normals);
  }
  for (uint32_t i = 0; i < c->st.n; i++)
    if (c->ent[i].state == SGC_READY) {
      r3d_tifxyz_free(&c->ent[i].s);
      r3d_segrows_free(&c->ent[i].rows);
    }
  free(c->ent);
  free(c->queue);
  free(c->ov);
  pthread_mutex_destroy(&c->mu);
  pthread_cond_destroy(&c->cv);
  r3d_segstore_close(&c->st);
  c->open = false;
}

static void sgc_request(sgcache *c, uint32_t i) { /* mu held */
  if (c->ent[i].state != SGC_EMPTY || c->qn >= c->qcap) return;
  c->ent[i].state = SGC_QUEUED;
  c->queue[c->qn++] = i;
  pthread_cond_signal(&c->cv);
}

/* per-(plane view, segment) cached overlay polyline */
typedef struct sgc_line {
  mv_lines l;
  double slice;
  uint32_t gen;
  bool valid;
} sgc_line;

/* linear interpolation of the umbilicus curve at z (clamped to the ends);
 * false when no points exist */
static bool umb_interp(const r3d_umbilicus *u, double z, double *x, double *y) {
  if (u->count == 0) return false;
  const r3d_umbilicus_point *p = u->points;
  size_t n = u->count;
  if (z < p[0].z || z > p[n - 1].z) return false; /* no clamped extension:
                        * outside the curve's z-range there IS no umbilicus */
  if (z == p[0].z) {
    *x = p[0].x;
    *y = p[0].y;
    return true;
  }
  for (size_t i = 1; i < n; i++)
    if (p[i].z >= z) {
      double dz = p[i].z - p[i - 1].z;
      double t = dz > 0.0 ? (z - p[i - 1].z) / dz : 0.0;
      *x = p[i - 1].x + (p[i].x - p[i - 1].x) * t;
      *y = p[i - 1].y + (p[i].y - p[i - 1].y) * t;
      return true;
    }
  return false;
}

/* snapshot-based undo for umbilicus edits (point arrays are tiny) */
typedef struct umb_snap {
  r3d_umbilicus_point *pts;
  size_t count;
} umb_snap;
#define UMB_UNDO_MAX 64u

static void umb_undo_push(umb_snap *st, uint32_t *n, const r3d_umbilicus *u) {
  if (*n == UMB_UNDO_MAX) { /* full: the oldest snapshot falls off */
    free(st[0].pts);
    memmove(st, st + 1, (UMB_UNDO_MAX - 1) * sizeof *st);
    (*n)--;
  }
  r3d_umbilicus_point *cp = NULL;
  if (u->count) {
    cp = malloc(u->count * sizeof *cp);
    if (!cp) return; /* OOM: this edit just isn't undoable */
    memcpy(cp, u->points, u->count * sizeof *cp);
  }
  st[*n].pts = cp;
  st[(*n)++].count = u->count;
}

/* first voxel value with nonzero TF alpha: below it, samples are invisible */
static float tf_min_visible(const uint8_t lut[256][4]) {
  for (uint32_t i = 0; i < 256; i++)
    if (lut[i][3] != 0) return (float)i;
  return 255.0f;
}

/* ---- open-data browser: browse the bucket, bootstrap + relaunch ---------- */

static const char OD_BUCKET[] = "https://vesuvius-challenge-open-data.s3.amazonaws.com";

typedef struct od_state {
  r3d_odlist scrolls, vols, segs, variants;
  int sel_scroll, sel_vol, sel_seg, sel_variant;
  bool scrolls_ok, vols_ok, segs_ok, variants_ok;
  /* async listing worker: the GUI thread never blocks on the network or
   * touches the filesystem — it posts requests and polls results */
  pthread_t fth;
  bool fth_up, fbusy, fquit;
  pthread_mutex_t fmu;
  pthread_cond_t fcv;
  int freq, fdone; /* 1 scrolls, 2 vols+segs (of fscroll), 3 variants */
  char fscroll[300], fseg[300];
  r3d_odlist fres_a, fres_b;
  bool *fcached; /* per-volume [cached] flags, computed on the worker */
  bool *vol_cached;
  FILE *job;            /* running subprocess (popen, non-blocking reads) */
  char log[10][160];    /* rolling job output */
  int nlog;
  int stage;            /* 0 idle, 1 volume bootstrap, 2 segment download */
  char vol_dir[512], seg_dir[512];
  bool with_segment, spawned_vol, spawned_seg;
  char linebuf[512];
  size_t linelen;
} od_state;

static void od_log(od_state *od, const char *line) {
  if (od->nlog == 10) {
    memmove(od->log[0], od->log[1], sizeof od->log - sizeof od->log[0]);
    od->nlog--;
  }
  snprintf(od->log[od->nlog++], sizeof od->log[0], "%s", line);
}

static void od_exe_dir(char out[512]) {
  ssize_t n = readlink("/proc/self/exe", out, 511);
  if (n <= 0) {
    snprintf(out, 512, ".");
    return;
  }
  out[n] = 0;
  char *slash = strrchr(out, '/');
  if (slash) *slash = 0;
}

static bool od_file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0;
}

/* start a shell command with merged stderr, non-blocking stdout */
static int od_spawn(od_state *od, const char *cmd) {
  char full[4096];
  snprintf(full, sizeof full, "%s 2>&1", cmd);
  od->job = popen(full, "r");
  if (!od->job) return -1;
  int fd = fileno(od->job);
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
  od->linelen = 0;
  return 0;
}

/* pump the job pipe; returns 1 when finished (status ok), -1 failed, 0 busy */
static int od_pump(od_state *od) {
  if (!od->job) return 0;
  char buf[512];
  for (;;) {
    ssize_t n = read(fileno(od->job), buf, sizeof buf);
    if (n > 0) {
      for (ssize_t i = 0; i < n; i++) {
        if (buf[i] == '\n' || od->linelen + 1 >= sizeof od->linebuf) {
          od->linebuf[od->linelen] = 0;
          if (od->linelen) od_log(od, od->linebuf);
          od->linelen = 0;
        } else if (buf[i] != '\r') {
          od->linebuf[od->linelen++] = buf[i];
        }
      }
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
    break; /* EOF or error */
  }
  int status = pclose(od->job);
  od->job = NULL;
  return status == 0 ? 1 : -1;
}

static void *od_fetch_worker(void *ud) {
  od_state *od = ud;
  for (;;) {
    pthread_mutex_lock(&od->fmu);
    while (!od->fquit && od->freq == 0) pthread_cond_wait(&od->fcv, &od->fmu);
    if (od->fquit) {
      pthread_mutex_unlock(&od->fmu);
      return NULL;
    }
    int req = od->freq;
    char scroll[300], seg[300];
    snprintf(scroll, sizeof scroll, "%s", od->fscroll);
    snprintf(seg, sizeof seg, "%s", od->fseg);
    pthread_mutex_unlock(&od->fmu);

    r3d_odlist a = {0}, b = {0};
    bool *cached = NULL;
    char pfx[700];
    if (req == 1) {
      r3d_odlist_fetch(OD_BUCKET, "", &a);
    } else if (req == 2) {
      snprintf(pfx, sizeof pfx, "%s/volumes/", scroll);
      r3d_odlist_fetch(OD_BUCKET, pfx, &a);
      snprintf(pfx, sizeof pfx, "%s/segments/", scroll);
      r3d_odlist_fetch(OD_BUCKET, pfx, &b);
      cached = calloc(a.ndirs ? a.ndirs : 1, sizeof *cached);
      for (uint32_t i = 0; cached && i < a.ndirs; i++) {
        char mp[900];
        snprintf(mp, sizeof mp, "cache/od/%s/%s/manifest.json", scroll, a.dirs[i]);
        cached[i] = od_file_exists(mp);
      }
    } else if (req == 3) {
      snprintf(pfx, sizeof pfx, "%s/segments/%s/mesh/", scroll, seg);
      r3d_odlist_fetch(OD_BUCKET, pfx, &a);
    }
    pthread_mutex_lock(&od->fmu);
    r3d_odlist_free(&od->fres_a);
    r3d_odlist_free(&od->fres_b);
    free(od->fcached);
    od->fres_a = a;
    od->fres_b = b;
    od->fcached = cached;
    od->fdone = req;
    od->freq = 0;
    od->fbusy = false;
    pthread_mutex_unlock(&od->fmu);
  }
}

static void od_request(od_state *od, int req) {
  if (!od->fth_up) {
    pthread_mutex_init(&od->fmu, NULL);
    pthread_cond_init(&od->fcv, NULL);
    if (pthread_create(&od->fth, NULL, od_fetch_worker, od) != 0) return;
    od->fth_up = true;
  }
  pthread_mutex_lock(&od->fmu);
  if (!od->fbusy) {
    od->freq = req;
    od->fbusy = true;
    if (od->sel_scroll >= 0)
      snprintf(od->fscroll, sizeof od->fscroll, "%s", od->scrolls.dirs[od->sel_scroll]);
    if (od->sel_seg >= 0)
      snprintf(od->fseg, sizeof od->fseg, "%s", od->segs.dirs[od->sel_seg]);
    pthread_cond_signal(&od->fcv);
  }
  pthread_mutex_unlock(&od->fmu);
}

/* returns the completed request id (once) and moves its results out */
static int od_poll(od_state *od, r3d_odlist *a, r3d_odlist *b, bool **cached) {
  if (!od->fth_up) return 0;
  pthread_mutex_lock(&od->fmu);
  int done = od->fdone;
  if (done) {
    *a = od->fres_a;
    *b = od->fres_b;
    *cached = od->fcached;
    memset(&od->fres_a, 0, sizeof od->fres_a);
    memset(&od->fres_b, 0, sizeof od->fres_b);
    od->fcached = NULL;
    od->fdone = 0;
  }
  pthread_mutex_unlock(&od->fmu);
  return done;
}

/* The browser is an ordinary ImGui window in the frame loop. Opening a
 * dataset runs background jobs (zarr2c5d --bootstrap, tifxyz download);
 * when everything is local it writes the chosen paths and requests a
 * dataset swap — plain mutable state, reopened in the same session. */
static void od_browser_window(od_state *od, bool *open, char *next_bricks, size_t nb_cap,
                              char *next_seg, size_t ns_cap, bool *swap) {
  int prc = od_pump(od);
  if (prc < 0 && od->stage) {
    od_log(od, "job FAILED");
    od->stage = 0;
  }
  if (od->stage == 1 && !od->job) { /* ensure the volume is bootstrapped */
    char mp[600];
    snprintf(mp, sizeof mp, "%s/manifest.json", od->vol_dir);
    if (od_file_exists(mp)) {
      od_log(od, "volume ready");
      od->stage = od->with_segment ? 2 : 3;
    } else if (od->spawned_vol) {
      od_log(od, "bootstrap produced no manifest");
      od->stage = 0;
    } else {
      od->spawned_vol = true;
      char exe[512], cmd[4096];
      od_exe_dir(exe);
      snprintf(cmd, sizeof cmd,
               "'%s/zarr2c5d' '%s/meta' '%s' --url '%s/%s/volumes/%s' --bootstrap "
               "--threads 8",
               exe, od->vol_dir, od->vol_dir, OD_BUCKET, od->scrolls.dirs[od->sel_scroll],
               od->vols.dirs[od->sel_vol]);
      od_log(od, "bootstrapping volume (coarsest level + source.json)...");
      if (od_spawn(od, cmd) != 0) od->stage = 0;
    }
  }
  if (od->stage == 2 && !od->job) { /* ensure the segment tifxyz is local */
    char probe[600];
    snprintf(probe, sizeof probe, "%s/meta.json", od->seg_dir);
    if (od_file_exists(probe)) {
      od_log(od, "segment ready");
      od->stage = 3;
    } else if (od->spawned_seg) {
      od_log(od, "segment download incomplete");
      od->stage = 0;
    } else {
      od->spawned_seg = true;
      char murl[1100], cmd[4096];
      snprintf(murl, sizeof murl, "%s/%s/segments/%s/mesh/%s", OD_BUCKET,
               od->scrolls.dirs[od->sel_scroll], od->segs.dirs[od->sel_seg],
               od->variants.dirs[od->sel_variant]);
      snprintf(cmd, sizeof cmd,
               "mkdir -p '%s' && curl -fsS -o '%s/x.tif' '%s/x.tif' && "
               "curl -fsS -o '%s/y.tif' '%s/y.tif' && "
               "curl -fsS -o '%s/z.tif' '%s/z.tif' && "
               "curl -fsS -o '%s/meta.json.dl' '%s/meta.json' && "
               "mv '%s/meta.json.dl' '%s/meta.json' && echo downloaded",
               od->seg_dir, od->seg_dir, murl, od->seg_dir, murl, od->seg_dir, murl,
               od->seg_dir, murl, od->seg_dir, od->seg_dir);
      od_log(od, "downloading segment tifxyz...");
      if (od_spawn(od, cmd) != 0) od->stage = 0;
    }
  }
  if (od->stage == 3) { /* everything local: request the dataset swap */
    snprintf(next_bricks, nb_cap, "%s/manifest.json", od->vol_dir);
    if (od->with_segment)
      snprintf(next_seg, ns_cap, "%s", od->seg_dir);
    else
      next_seg[0] = 0;
    *swap = true;
    od->stage = 0;
    od_log(od, "opening...");
  }

  {
    r3d_odlist ra = {0}, rb = {0};
    bool *rc_ = NULL;
    int done = od_poll(od, &ra, &rb, &rc_);
    if (done == 1) {
      r3d_odlist_free(&od->scrolls);
      od->scrolls = ra;
      od->scrolls_ok = true;
    } else if (done == 2) {
      r3d_odlist_free(&od->vols);
      r3d_odlist_free(&od->segs);
      free(od->vol_cached);
      od->vols = ra;
      od->segs = rb;
      od->vol_cached = rc_;
      rc_ = NULL;
      od->vols_ok = od->segs_ok = true;
    } else if (done == 3) {
      r3d_odlist_free(&od->variants);
      od->variants = ra;
      od->variants_ok = true;
    } else if (done) {
      r3d_odlist_free(&ra);
      r3d_odlist_free(&rb);
    }
    if (done != 2) free(rc_);
  }

  if (!*open) return;
  igSetNextWindowSize((ImVec2){620, 640}, ImGuiCond_FirstUseEver);
  igSetNextWindowPos((ImVec2){380, 40}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
  if (!igBegin("open data browser", open, 0)) {
    igEnd();
    return;
  }
  if (!od->scrolls_ok) {
    od_request(od, 1);
    igTextDisabled("listing bucket...");
  }
  const char *cur = od->sel_scroll >= 0 ? od->scrolls.dirs[od->sel_scroll] : "<pick a scroll>";
  if (igBeginCombo("scroll", cur, 0)) {
    for (uint32_t i = 0; i < od->scrolls.ndirs; i++)
      if (igSelectable_Bool(od->scrolls.dirs[i], (int)i == od->sel_scroll, 0,
                            (ImVec2){0, 0})) {
        od->sel_scroll = (int)i;
        r3d_odlist_free(&od->vols);
        r3d_odlist_free(&od->segs);
        r3d_odlist_free(&od->variants);
        od->vols_ok = od->segs_ok = od->variants_ok = false;
        od->sel_vol = od->sel_seg = od->sel_variant = -1;
      }
    igEndCombo();
  }
  if (od->sel_scroll >= 0 && !od->vols_ok) {
    od_request(od, 2);
    igTextDisabled("listing scroll...");
  }
  if (od->vols_ok) {
    igText("volumes");
    igBeginChild_Str("odvols", (ImVec2){0, 140}, ImGuiChildFlags_Borders, 0);
    for (uint32_t i = 0; i < od->vols.ndirs; i++) {
      char label[800];
      snprintf(label, sizeof label, "%s%s",
               od->vol_cached && od->vol_cached[i] ? "[cached] " : "", od->vols.dirs[i]);
      if (igSelectable_Bool(label, (int)i == od->sel_vol, 0, (ImVec2){0, 0})) {
        od->sel_vol = (int)i;
        printf("odbrowse: volume %s selected\n", od->vols.dirs[i]);
      }
    }
    igEndChild();
  }
  if (od->segs_ok && od->segs.ndirs) {
    igText("segments (%u)", od->segs.ndirs);
    igBeginChild_Str("odsegs", (ImVec2){0, 150}, ImGuiChildFlags_Borders, 0);
    for (uint32_t i = 0; i < od->segs.ndirs; i++)
      if (igSelectable_Bool(od->segs.dirs[i], (int)i == od->sel_seg, 0, (ImVec2){0, 0})) {
        od->sel_seg = (int)i;
        r3d_odlist_free(&od->variants);
        od->variants_ok = false;
        od->sel_variant = -1;
      }
    igEndChild();
  }
  if (od->sel_seg >= 0 && !od->variants_ok) od_request(od, 3);
  char volid[64] = "";
  if (od->sel_vol >= 0) { /* volume id = dir name up to the first '-' */
    snprintf(volid, sizeof volid, "%s", od->vols.dirs[od->sel_vol]);
    char *dash = strchr(volid, '-');
    if (dash) *dash = 0;
  }
  if (od->variants_ok && od->variants.ndirs) {
    igText("segment mesh (tifxyz)");
    igBeginChild_Str("odvar", (ImVec2){0, 90}, ImGuiChildFlags_Borders, 0);
    char onvol[80];
    snprintf(onvol, sizeof onvol, "-on-%s", volid);
    for (uint32_t i = 0; i < od->variants.ndirs; i++) {
      bool match = volid[0] && strstr(od->variants.dirs[i], onvol) != NULL;
      char label[700];
      snprintf(label, sizeof label, "%s%s", match ? "[matches volume] " : "",
               od->variants.dirs[i]);
      if (igSelectable_Bool(label, (int)i == od->sel_variant, 0, (ImVec2){0, 0}))
        od->sel_variant = (int)i;
    }
    igEndChild();
  }
  bool busy = od->stage != 0;
  igTextDisabled("sel: scroll %d vol %d seg %d variant %d stage %d", od->sel_scroll,
                 od->sel_vol, od->sel_seg, od->sel_variant, od->stage);
  igBeginDisabled(busy || od->sel_vol < 0);
  bool go_vol = igButton("open volume", (ImVec2){0, 0});
  igEndDisabled();
  igSameLine(0, 10);
  igBeginDisabled(busy || od->sel_vol < 0 || od->sel_variant < 0);
  bool go_seg = igButton("open volume + segment", (ImVec2){0, 0});
  igEndDisabled();
  if ((go_vol || go_seg) && od->sel_vol >= 0) {
    od->with_segment = go_seg && od->sel_variant >= 0;
    od->spawned_vol = od->spawned_seg = false;
    snprintf(od->vol_dir, sizeof od->vol_dir, "cache/od/%s/%s",
             od->scrolls.dirs[od->sel_scroll], od->vols.dirs[od->sel_vol]);
    if (od->with_segment)
      snprintf(od->seg_dir, sizeof od->seg_dir, "cache/od/segments/%s",
               od->variants.dirs[od->sel_variant]);
    od->stage = 1;
  }
  if (busy) {
    igSameLine(0, 12);
    igTextDisabled("working...");
  }
  igSeparator();
  for (int i = 0; i < od->nlog; i++) igTextDisabled("%s", od->log[i]);
  igEnd();
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
  const char *seg_store_path = NULL; /* segpack store: draw ALL surfaces */
  const char *overlay_path = NULL;   /* second c5d LOD root (ink predictions) */
  bool od_browse = false;            /* start with the open-data browser window */
  int sv_w = 2048, sv_h = 2048, sv_l = 96; /* flattened surface-volume window */
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
    if (i < argc - 1 && strcmp(argv[i], "--overlay") == 0) overlay_path = argv[i + 1];
    if (i < argc - 1 && strcmp(argv[i], "--segments") == 0) seg_store_path = argv[i + 1];
    if (strcmp(argv[i], "--browse") == 0) od_browse = true;
    if (i < argc - 3 && strcmp(argv[i], "--surfvol") == 0) {
      sv_w = atoi(argv[i + 1]);
      sv_h = atoi(argv[i + 2]);
      sv_l = atoi(argv[i + 3]);
    }
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
  if (umbilicus_path && !multiview_path) { /* multiview annotates in-place;
                                            * standalone uses the vslab rig */
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

  /* the dataset (volume + segment + overlay) is ordinary mutable state: the
   * open-data browser swaps it by tearing the bricks family down and
   * re-opening inside this loop, same window, same renderer */
  od_state od = {.sel_scroll = -1, .sel_vol = -1, .sel_seg = -1, .sel_variant = -1};
  bool od_window = od_browse;
  char od_next_bricks[640] = "", od_next_seg[560] = "";
  bool od_swap = false;
  for (;;) {
  if (od_swap) {
    od_swap = false;
    bricks_path = od_next_bricks;
    multiview_path = od_next_seg[0] ? od_next_seg : NULL;
    overlay_path = NULL; /* browser-opened datasets have no overlay tree yet */
    umbilicus_path = NULL;
    vslab_mode = false;
    clip_mode = false;
  }

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
    if (overlay_path && r3d_bricks_overlay(renderer, overlay_path) != 0)
      return EXIT_FAILURE;
  }
  bool overlay_show = overlay_path != NULL;
  float overlay_gain = 1.5f;

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
  uint32_t mv_visible = 0xfu; /* per-view visibility; collapsed views cost nothing */
  int mv_solo = -1;           /* Space on a hovered view maximizes it */
  bool mv_panel_open = true;  /* docked left panel; collapses to a slim bar */
  mv_lines mv_ol[4] = {0}, mv_ol_off[4] = {0}; /* intersection polylines */
  double mv_ol_slice[4] = {1e30, 1e30, 1e30, 1e30};
  double mv_ol_zoff = 1e30;
  r3d_segrows mv_rows = {0}; /* per-row coord bounds: segtrace row skipping */
  double mv_pb[4][3][3]; /* per-view plane frames (rows u, v, n) + origins */
  double mv_po[4][3] = {{0}};
  bool mv_aligned = false; /* XZ/YZ panes follow the segment normal */
  float mv_theta = 0.0f;   /* aligned-pane rotation around the normal (deg) */
  /* flattened view: distortion heatmap (env enables it for headless shots,
   * like R3D_FORCE_HALF) */
  bool mv_stretch = getenv("R3D_MV_STRETCH") != NULL;
  /* umbilicus editing in the plane panes: plain click places the point for
   * its z (Shift+drag pans while on); off = clicks drag as usual */
  bool mv_umb_edit = umbilicus_path && multiview_path;
  umb_snap mv_umb_undo[UMB_UNDO_MAX];
  uint32_t mv_umb_undo_n = 0;
  uint32_t mv_align_ij[2] = {0, 0}; /* surface grid point anchoring the frames */
  uint32_t mv_basis_gen = 0, mv_ol_gen[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) r3d_mv_axis_basis(i, mv_pb[i]);
  if (multiview_path) {
    if (!bricks_path || !brick_is_lod) {
      fprintf(stderr, "--multiview needs --bricks with a LOD manifest\n");
      return EXIT_FAILURE;
    }
    SDL_PumpEvents(); /* dataset swap: multi-second setup, stay responsive */
    if (r3d_tifxyz_load(&mv_seg, multiview_path) != 0) return EXIT_FAILURE;
    if (r3d_segrows_build(&mv_seg, &mv_rows) != 0) return EXIT_FAILURE;
    SDL_PumpEvents();
    for (int a = 0; a < 3; a++)
      mv_focus[a] = ((double)mv_seg.bbox[0][a] + (double)mv_seg.bbox[1][a]) * 0.5;
    mv_align_ij[0] = mv_seg.w / 2;
    mv_align_ij[1] = mv_seg.h / 2;
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
    SDL_PumpEvents();
    if (mv_build_grids(&mv_seg, &seg_coords, &mv_normals) != 0 ||
        (SDL_PumpEvents(),
         r3d_surf_begin(renderer, mv_seg.w, mv_seg.h, seg_coords, mv_normals)) != 0) {
      fprintf(stderr, "multiview: surf grid upload failed\n");
      return EXIT_FAILURE;
    }
    free(seg_coords);
    /* flattened surface-volume window (RG8), resampled on the GPU from the
     * shared brick cache; --surfvol W H L overrides, device caps clamp */
    if (sv_w < 256) sv_w = 256;
    if (sv_h < 256) sv_h = 256;
    if (sv_l < 8) sv_l = 8;
    int dim3d = (int)r3d_max_dim3d(renderer); /* --surfvol vs device limit */
    if (sv_w > dim3d) sv_w = dim3d;
    if (sv_h > dim3d) sv_h = dim3d;
    if (sv_l > dim3d) sv_l = dim3d;
    if (r3d_surfvol_begin(renderer, (uint32_t)sv_w, (uint32_t)sv_h, (uint32_t)sv_l,
                          (uint32_t)sv_l / 2, mv_seg.sx, mv_seg.sy) != 0) {
      fprintf(stderr, "multiview: surface-volume window init failed\n");
      return EXIT_FAILURE;
    }
  }

  /* segment corpus: index + background decode cache + per-view polyline
   * caches; the frame loop only queries and traces, never decodes */
  sgcache sgc = {0};
  sgc_line *sgc_ln[4] = {NULL, NULL, NULL, NULL};
  uint32_t sgc_hits[4][128];
  uint32_t sgc_nhits[4] = {0, 0, 0, 0};
  double sgc_key_slice[4] = {1e30, 1e30, 1e30, 1e30};
  uint32_t sgc_key_gen[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
  char sgc_active[R3D_SEGSTORE_NAME] = {0};
  double sgc_near_focus[3] = {1e30, 1e30, 1e30};
  uint32_t sgc_near[6];
  uint32_t sgc_nnear = 0;
  if (seg_store_path && !multiview_path)
    fprintf(stderr, "--segments needs --multiview; ignoring the store\n");
  if (seg_store_path && multiview_path) {
    if (sgc_open(&sgc, seg_store_path, (size_t)1536 << 20) != 0) {
      fprintf(stderr, "--segments: failed to open store %s\n", seg_store_path);
      return EXIT_FAILURE;
    }
    for (int i = 1; i < 4; i++) {
      sgc_ln[i] = calloc(sgc.st.n ? sgc.st.n : 1, sizeof *sgc_ln[i]);
      if (!sgc_ln[i]) return EXIT_FAILURE;
    }
    /* the active (flattened) segment draws in full color already */
    const char *sl_ = strrchr(multiview_path, '/');
    snprintf(sgc_active, sizeof sgc_active, "%s", sl_ && sl_[1] ? sl_ + 1 : multiview_path);
    printf("segments: %u surfaces, %llu index tiles from %s\n", sgc.st.n,
           (unsigned long long)sgc.st.ntiles, seg_store_path);
    for (uint32_t si = 0; si < sgc.st.n; si++) /* overlap-vs-active pass */
      if (strcmp(sgc.st.segs[si].name, sgc_active) == 0) {
        pthread_mutex_lock(&sgc.mu);
        sgc.ov_req = si;
        pthread_cond_signal(&sgc.cv);
        pthread_mutex_unlock(&sgc.mu);
        break;
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
  /* main-thread phase profile: poll / nav / gui / stream / frame (ns) */
  enum { MT_POLL, MT_NAV, MT_GUI, MT_STREAM, MT_FRAME, MT_N };
  static const char *mt_name[MT_N] = {"poll", "nav", "gui", "stream", "frame"};
  uint64_t mt_sum[MT_N] = {0}, mt_max[MT_N] = {0}, mt_frames = 0;
  double mt_ema[MT_N] = {0};
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

    uint64_t mt_t[MT_N + 1];
    mt_t[0] = r3d_now_ns();
    ImGuiIO *io = igGetIO_Nil(); /* Want* flags reflect last frame — fine */
    r3d_input_poll(&in, win, gui_event_hook, renderer, !io->WantCaptureMouse,
                   cam_mode == CAM_FLY, umbilicus_path != NULL && !multiview_path,
                   multiview_path != NULL);
    mt_t[1] = r3d_now_ns();
    if (io->WantCaptureKeyboard && !in.captured)
      in.move[0] = in.move[1] = in.move[2] = 0.0f;
    if (in.quit) running = false;
    if (in.resized) r3d_resize(renderer);
    if (in.mode_delta) {
      mode = (mode + (uint32_t)in.mode_delta) % R3D_MODE_COUNT;
      printf("mode: %u\n", mode);
    }
    bool z_navigated = false;
    if (umbilicus_path && vslab_mode) { /* multiview panes own their wheel */
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

    if (sgc.open) { /* apply a finished activation: swap the flattened view
       * to the worker-prepared segment (decode/grids were built off-thread;
       * only the GPU upload happens here) */
      pthread_mutex_lock(&sgc.mu);
      if (sgc.act_ready != UINT32_MAX) {
        uint32_t ai = sgc.act_ready;
        r3d_tifxyz ns = sgc.act_s;
        r3d_segrows nr = sgc.act_rows;
        float *co = sgc.act_coords, *no = sgc.act_normals;
        sgc.act_ready = UINT32_MAX;
        sgc.act_busy = false;
        sgc.act_coords = sgc.act_normals = NULL;
        pthread_mutex_unlock(&sgc.mu);
        if (r3d_surf_swap(renderer, ns.w, ns.h, co, no, ns.sx, ns.sy) == 0) {
          r3d_tifxyz_free(&mv_seg);
          r3d_segrows_free(&mv_rows);
          free(mv_normals);
          mv_seg = ns;
          mv_rows = nr;
          mv_normals = no;
          snprintf(sgc_active, sizeof sgc_active, "%s", sgc.st.segs[ai].name);
          mv[R3D_MV_SEG].cu = (double)mv_seg.w * 0.5;
          mv[R3D_MV_SEG].cv = (double)mv_seg.h * 0.5;
          for (int i = 0; i < 4; i++) {
            mv_ol[i].n = 0;
            mv_ol_off[i].n = 0;
            mv_ol_slice[i] = 1e30;
          }
          mv_ol_zoff = 1e30;
          const float *mc2 = r3d_tifxyz_at(&mv_seg, mv_seg.w / 2, mv_seg.h / 2);
          if (r3d_tifxyz_valid(mc2))
            for (int a = 0; a < 3; a++) mv_focus[a] = (double)mc2[a];
          mv_align_ij[0] = mv_seg.w / 2;
          mv_align_ij[1] = mv_seg.h / 2;
          if (mv_aligned) {
            if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                              mv_po))
              mv_axis_reset(mv_pb, mv_po);
          }
          mv_basis_gen++; /* re-trace every cached overlay against the swap */
          for (int i = 1; i < 4; i++) { /* recenter planes on the new focus */
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
            mv[i].cu = fu;
            mv[i].cv = fv;
            mv[i].slice = fs;
          }
          printf("activated segment %s (%ux%u)\n", sgc_active, mv_seg.w, mv_seg.h);
          pthread_mutex_lock(&sgc.mu); /* refresh overlap-vs-active */
          sgc.ov_req = ai;
          pthread_cond_signal(&sgc.cv);
          pthread_mutex_unlock(&sgc.mu);
        } else {
          r3d_tifxyz_free(&ns);
          r3d_segrows_free(&nr);
          free(no);
          fprintf(stderr, "segment activation failed (surf grid swap)\n");
        }
        free(co);
      } else {
        pthread_mutex_unlock(&sgc.mu);
      }
    }
    uint32_t mv_mask = mv_solo >= 0 ? 1u << mv_solo : (mv_visible ? mv_visible : 0xfu);
    int mv_panel_w = multiview_path ? (mv_panel_open ? 360 : 26) : 0;
    if (mv_panel_w >= w) mv_panel_w = 0;
    if (multiview_path) {
      r3d_mview lay[4];
      r3d_mv_layout_mask(lay, w - mv_panel_w, h, mv_mask);
      for (int i = 0; i < 4; i++) {
        mv[i].px = lay[i].px + (lay[i].pw ? mv_panel_w : 0);
        mv[i].py = lay[i].py;
        mv[i].pw = lay[i].pw;
        mv[i].ph = lay[i].ph;
      }
      if (getenv("R3D_MV_EXERCISE")) {
        /* automated GUI interaction for profiling: scrub slices, wiggle
         * zoom and pan — drives overlay recompute, surfvol re-bakes and
         * streaming exactly like a user session, reproducibly */
        double ph = (double)frame_index * 0.02;
        for (int i = 1; i < 4; i++)
          mv[i].slice += (frame_index % 20) < 10 ? 1.0 : -1.0;
        mv[R3D_MV_SEG].slice = 24.0 * sin(ph);
        r3d_mview *ev = &mv[1 + (frame_index / 120) % 3];
        r3d_mv_zoom(ev, (float)(ev->px + ev->pw / 2), (float)(ev->py + ev->ph / 2),
                    (frame_index % 240) < 120 ? 1.01 : 0.99, 1.0 / 256.0, 10.0);
        ev->cu += 3.0 * sin(ph * 1.7);
        ev->cv += 3.0 * cos(ph * 1.3);
      }
      if (in.view_toggle) { /* Space: solo/restore the hovered view */
        int hv = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (mv_solo >= 0) mv_solo = -1;
        else if (hv >= 0) mv_solo = hv;
        mv_mask = mv_solo >= 0 ? 1u << mv_solo : (mv_visible ? mv_visible : 0xfu);
        r3d_mv_layout_mask(lay, w - mv_panel_w, h, mv_mask);
        for (int i = 0; i < 4; i++) {
          mv[i].px = lay[i].px + (lay[i].pw ? mv_panel_w : 0);
          mv[i].py = lay[i].py;
          mv[i].pw = lay[i].pw;
          mv[i].ph = lay[i].ph;
        }
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
          /* max zoom ~10 screen pixels per source voxel; the segment view's
           * zoom unit is a grid step (1/scale voxels), so scale its cap */
          double zmax = hover == R3D_MV_SEG ? 10.0 / (double)mv_seg.sx : 10.0;
          r3d_mv_zoom(hv, in.mouse_xy[0], in.mouse_xy[1],
                      pow(1.05, (double)in.wheel * 2.0), 1.0 / 256.0, zmax);
        }
        in.wheel = 0.0f;
      }
      if (hover >= 0 && (in.zdelta || in.zpage))
        mv[hover].slice += (double)(in.zdelta + in.zpage * 10);
      /* SEG slice = normal offset, symmetric around the sheet */
      if (mv[R3D_MV_SEG].slice < -512.0) mv[R3D_MV_SEG].slice = -512.0;
      if (mv[R3D_MV_SEG].slice > 512.0) mv[R3D_MV_SEG].slice = 512.0;
      for (int i = 1; i < 4; i++) { /* plane slices clamp to the volume */
        if (mv_aligned && i >= R3D_MV_XZ) {
          /* frame offset from the focus, not a world coordinate: keep it
           * within the volume diagonal so the pane can always scrub back */
          double lim = (double)brick_shape[0] + brick_shape[1] + brick_shape[2];
          if (mv[i].slice < -lim) mv[i].slice = -lim;
          if (mv[i].slice > lim) mv[i].slice = lim;
          continue;
        }
        uint32_t n = brick_shape[r3d_mv_axes[i][2]];
        if (mv[i].slice < 0.0) mv[i].slice = 0.0;
        if (n && mv[i].slice > (double)n - 1.0) mv[i].slice = (double)n - 1.0;
      }
      if (umbilicus_path && mv_umb_edit && in.umb_place && !io->WantCaptureKeyboard) {
        /* U places the umbilicus point at the cursor, in ANY plane pane —
         * the core can be ambiguous in one orientation and obvious in
         * another. Keyed by the position's (rounded) z; mouse buttons keep
         * their normal pan/zoom roles. */
        int cu_ = r3d_mv_hit(mv, in.mouse_xy[0], in.mouse_xy[1]);
        if (cu_ > 0) {
          double uu, vv, W[3];
          r3d_mv_unproject(&mv[cu_], in.mouse_xy[0], in.mouse_xy[1], &uu, &vv);
          r3d_mv_b2w(mv_pb[cu_], mv_po[cu_], uu, vv, mv[cu_].slice, W);
          for (int a = 0; a < 3; a++) {
            if (W[a] < 0.0) W[a] = 0.0;
            if (brick_shape[a] && W[a] > (double)brick_shape[a] - 1.0)
              W[a] = (double)brick_shape[a] - 1.0;
          }
          umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
          if (r3d_umbilicus_set(&umbilicus, W[0], W[1], (double)llround(W[2])) == 0)
            save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        }
      }
      if (umbilicus_path && mv_umb_edit && in.undo && !io->WantCaptureKeyboard &&
          mv_umb_undo_n) { /* Ctrl+Z: restore the previous point set */
        umb_snap s = mv_umb_undo[--mv_umb_undo_n];
        umbilicus.count = 0; /* re-inserting keeps the sorted invariant */
        for (size_t k = 0; k < s.count; k++)
          r3d_umbilicus_set(&umbilicus, s.pts[k].x, s.pts[k].y, s.pts[k].z);
        free(s.pts);
        save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
        snprintf(annotation_status, sizeof annotation_status, "undo (%u step%s left)",
                 mv_umb_undo_n, mv_umb_undo_n == 1 ? "" : "s");
      }
      if (in.annotate_click && in.click_ctrl) { /* Ctrl+click = set focus POI */
        int cv_ = r3d_mv_hit(mv, in.click_xy[0], in.click_xy[1]);
        bool focused = false, have_ij = false;
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
              have_ij = true; /* the clicked cell anchors the aligned frames */
              mv_align_ij[0] = fx_ < 0.5 ? gi : gi + 1;
              mv_align_ij[1] = fy_ < 0.5 ? gj : gj + 1;
            }
          }
        } else if (cv_ > 0) {
          double u, vq;
          r3d_mv_unproject(&mv[cv_], in.click_xy[0], in.click_xy[1], &u, &vq);
          r3d_mv_b2w(mv_pb[cv_], mv_po[cv_], u, vq, mv[cv_].slice, mv_focus);
          focused = true;
        }
        if (focused) {
          for (int a = 0; a < 3; a++) { /* clamp into the volume */
            if (mv_focus[a] < 0.0) mv_focus[a] = 0.0;
            if (brick_shape[a] && mv_focus[a] > (double)brick_shape[a] - 1.0)
              mv_focus[a] = (double)brick_shape[a] - 1.0;
          }
          if (!have_ij) {
            /* recenter the segment view on the surface point nearest the
             * focus (vc3d 100-voxel tolerance); it also anchors the frames */
            uint32_t ij[2];
            if (mv_nearest_surface(&mv_seg, mv_focus, ij)) {
              mv_align_ij[0] = ij[0];
              mv_align_ij[1] = ij[1];
              mv[R3D_MV_SEG].cu = (double)ij[0];
              mv[R3D_MV_SEG].cv = (double)ij[1];
            }
          }
          if (mv_aligned) {
            if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                              mv_po))
              mv_axis_reset(mv_pb, mv_po);
            mv_basis_gen++;
          }
          for (int i = 1; i < 4; i++) { /* recenter planes through the focus */
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
            mv[i].cu = fu;
            mv[i].cv = fv;
            mv[i].slice = fs;
          }
        }
      }

      if (mv_mask & 1u) { /* keep the flattened surface-volume window under the
         * view (snapped for hysteresis) and rebuild on residency arrivals;
         * a collapsed segment view skips baking entirely */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        double vox_per_px = 1.0 / (sv->zoom * (double)mv_seg.sx);
        double stepd = 1.0;
        while (stepd * 2.0 <= vox_per_px) stepd *= 2.0;
        double cu_vox = sv->cu / (double)mv_seg.sx, cv_vox = sv->cv / (double)mv_seg.sy;
        double snap = (double)sv_w / 8.0 * stepd; /* window W/8 */
        double u0 = floor((cu_vox - (double)sv_w * 0.5 * stepd) / snap) * snap;
        double v0 = floor((cv_vox - (double)sv_h * 0.5 * stepd) / snap) * snap;
        double zsnap = 24.0; /* layers are 1 voxel regardless of xy zoom */
        double z0 = floor(sv->slice / zsnap + 0.5) * zsnap;
        /* visible sub-box hint: full-window rebuilds bake this box in-frame
         * and refresh the rest progressively */
        double half_u = (double)sv->pw * 0.5 * vox_per_px;
        double half_v = (double)sv->ph * 0.5 / (sv->zoom * (double)mv_seg.sy);
        double tx0 = (cu_vox - half_u - u0) / stepd, tx1 = (cu_vox + half_u - u0) / stepd;
        double ty0 = (cv_vox - half_v - v0) / stepd, ty1 = (cv_vox + half_v - v0) / stepd;
        int lc = sv_l / 2 + (int)lround(sv->slice - z0);
        int lz0 = lc - mv_thick, lz1 = lc + mv_thick + 1;
        r3d_surfvol_visible(renderer, (uint32_t)(tx0 > 0.0 ? tx0 : 0.0),
                            (uint32_t)(ty0 > 0.0 ? ty0 : 0.0), (uint32_t)(lz0 > 0 ? lz0 : 0),
                            (uint32_t)(tx1 > 0.0 ? tx1 + 1.0 : 0.0),
                            (uint32_t)(ty1 > 0.0 ? ty1 + 1.0 : 0.0),
                            (uint32_t)(lz1 > 0 ? lz1 : 0));
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

    if (umbilicus_path && vslab_mode && in.annotate_click) {
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

    mt_t[2] = r3d_now_ns();
    /* control panel: floating window normally; in multiview a docked left
     * side panel that collapses to a slim bar (views reflow to fill) */
    fps_smooth = fps_smooth * 0.95f + (dt > 0 ? 0.05f / dt : 0.0f);
    r3d_gui_begin(renderer);
    bool panel_content = true;
    if (multiview_path) {
      igSetNextWindowPos((ImVec2){0, 0}, ImGuiCond_Always, (ImVec2){0, 0});
      igSetNextWindowSize((ImVec2){(float)mv_panel_w, (float)h}, ImGuiCond_Always);
      igBegin("render3d", NULL,
              ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                  (mv_panel_open ? 0 : ImGuiWindowFlags_NoScrollbar));
      if (igButton(mv_panel_open ? "<<" : ">>", (ImVec2){0, 0}))
        mv_panel_open = !mv_panel_open;
      panel_content = mv_panel_open;
      if (panel_content) {
        igSameLine(0, 10);
        igText("%.0f fps  gpu %.1f ms", (double)fps_smooth, (double)last_gpu_ns / 1e6);
        igSeparator();
      }
    } else {
      igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
      igBegin("render3d", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    }
    if (panel_content) {
    if (!multiview_path)
      igText("%.0f fps   gpu %.2f ms", (double)fps_smooth, (double)last_gpu_ns / 1e6);
    if (igButton("data browser", (ImVec2){0, 0})) od_window = true;
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
        static const char *mv_name[4] = {"segment", "XY", "XZ", "YZ"};
        for (int i = 0; i < 4; i++) {
          bool vis = (mv_visible >> i) & 1u;
          if (i) igSameLine(0, 8);
          if (igCheckbox(mv_name[i], &vis))
            mv_visible = vis ? mv_visible | (1u << i) : mv_visible & ~(1u << i);
        }
        if (mv_solo >= 0) {
          igSameLine(0, 12);
          igTextDisabled("solo: %s (Space restores)", mv_name[mv_solo]);
        }
        igText("multiview focus  x %.0f  y %.0f  z %.0f", mv_focus[0], mv_focus[1],
               mv_focus[2]);
        if (mv_aligned) /* side panes scrub offsets from the focus */
          igText("XY z %.0f | segA %+.0f | segB %+.0f", mv[R3D_MV_XY].slice,
                 mv[R3D_MV_XZ].slice, mv[R3D_MV_YZ].slice);
        else
          igText("XY z %.0f | XZ y %.0f | YZ x %.0f", mv[R3D_MV_XY].slice,
                 mv[R3D_MV_XZ].slice, mv[R3D_MV_YZ].slice);
        int th = mv_thick;
        if (igSliderInt("slice thickness", &th, 1, 128, "%d", 0)) mv_thick = th;
        bool alg = mv_aligned;
        if (igCheckbox("segment-aligned planes", &alg)) {
          mv_aligned = alg;
          if (mv_aligned) {
            uint32_t ij[2];
            if (mv_nearest_surface(&mv_seg, mv_focus, ij)) {
              mv_align_ij[0] = ij[0];
              mv_align_ij[1] = ij[1];
            }
            if (!mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                              mv_po))
              mv_aligned = false; /* no usable surface normal near the focus */
          }
          if (!mv_aligned) mv_axis_reset(mv_pb, mv_po);
          for (int i = R3D_MV_XZ; i <= R3D_MV_YZ; i++) {
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
            mv[i].cu = fu;
            mv[i].cv = fv;
            mv[i].slice = fs;
          }
          mv_basis_gen++;
        }
        if (mv_aligned &&
            igSliderFloat("plane rotation", &mv_theta, 0.0f, 360.0f, "%.0f deg", 0) &&
            mv_seg_align(&mv_seg, mv_normals, mv_focus, mv_align_ij, mv_theta, mv_pb,
                         mv_po)) {
          for (int i = R3D_MV_XZ; i <= R3D_MV_YZ; i++) {
            mv[i].cu = mv[i].cv = 0.0; /* frames rotate about the focus */
            mv[i].slice = 0.0;
          }
          mv_basis_gen++;
        }
        float zo = (float)mv[R3D_MV_SEG].slice;
        if (igSliderFloat("segment offset", &zo, -64.0f, 64.0f, "%.0f vox", 0))
          mv[R3D_MV_SEG].slice = (double)zo;
        igCheckbox("stretch heatmap", &mv_stretch);
        if (umbilicus_path) {
          igSeparator();
          igText("umbilicus  %zu point%s", umbilicus.count,
                 umbilicus.count == 1 ? "" : "s");
          igCheckbox("edit (U places at cursor, Ctrl+Z undoes)", &mv_umb_edit);
          double curz = mv[R3D_MV_XY].slice;
          if (igButton("< annotated", (ImVec2){0, 0})) {
            for (size_t k = umbilicus.count; k > 0; k--)
              if (umbilicus.points[k - 1].z < curz) {
                mv[R3D_MV_XY].slice = umbilicus.points[k - 1].z;
                mv[R3D_MV_XY].cu = umbilicus.points[k - 1].x;
                mv[R3D_MV_XY].cv = umbilicus.points[k - 1].y;
                break;
              }
          }
          igSameLine(0, 8);
          if (igButton("annotated >", (ImVec2){0, 0})) {
            for (size_t k = 0; k < umbilicus.count; k++)
              if (umbilicus.points[k].z > curz) {
                mv[R3D_MV_XY].slice = umbilicus.points[k].z;
                mv[R3D_MV_XY].cu = umbilicus.points[k].x;
                mv[R3D_MV_XY].cv = umbilicus.points[k].y;
                break;
              }
          }
          igSameLine(0, 8);
          const r3d_umbilicus_point *here =
              r3d_umbilicus_find(&umbilicus, (double)llround(curz));
          if (here && igButton("delete here", (ImVec2){0, 0})) {
            umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
            if (r3d_umbilicus_remove(&umbilicus, (double)llround(curz)))
              save_umbilicus(&umbilicus, umbilicus_path, annotation_status);
          }
          if (igButton("start fresh (move to .bak)", (ImVec2){0, 0})) {
            char bak[1360];
            snprintf(bak, sizeof bak, "%s.bak", umbilicus_path);
            umb_undo_push(mv_umb_undo, &mv_umb_undo_n, &umbilicus);
            if (rename(umbilicus_path, bak) == 0 || errno == ENOENT) {
              r3d_umbilicus_free(&umbilicus);
              r3d_umbilicus_init(&umbilicus);
              snprintf(annotation_status, sizeof annotation_status,
                       "cleared; previous saved as %s", bak);
            } else {
              snprintf(annotation_status, sizeof annotation_status, "backup failed: %s",
                       strerror(errno));
            }
          }
          if (annotation_status[0]) igTextDisabled("%s", annotation_status);
        }
        igTextDisabled("segment %ux%u  %llu valid points", mv_seg.w, mv_seg.h,
                       (unsigned long long)mv_seg.nvalid);
        if (sgc.open) {
          pthread_mutex_lock(&sgc.mu);
          uint32_t ready = 0;
          for (uint32_t si = 0; si < sgc.st.n; si++)
            if (sgc.ent[si].state == SGC_READY) ready++;
          igText("segment store: %u surfaces  %u cached (%zu MB)", sgc.st.n, ready,
                 sgc.bytes >> 20);
          igText("plane hits  XY %u  XZ %u  YZ %u", sgc_nhits[R3D_MV_XY],
                 sgc_nhits[R3D_MV_XZ], sgc_nhits[R3D_MV_YZ]);
          if (sgc_near_focus[0] != mv_focus[0] || sgc_near_focus[1] != mv_focus[1] ||
              sgc_near_focus[2] != mv_focus[2]) {
            memcpy(sgc_near_focus, mv_focus, sizeof sgc_near_focus);
            sgc_nnear = r3d_segstore_near_query(&sgc.st, mv_focus, 300.0, sgc_near, 6);
            if (sgc_nnear > 6) sgc_nnear = 6;
          }
          bool act_pending = sgc.act_req != UINT32_MAX || sgc.act_busy;
          if (act_pending) igTextDisabled("activating...");
          else igTextDisabled("near focus (click to activate):");
          for (uint32_t k = 0; k < sgc_nnear; k++) {
            uint32_t si = sgc_near[k];
            bool cur = strcmp(sgc.st.segs[si].name, sgc_active) == 0;
            char lbl[112];
            float ovf = sgc.ov_active != UINT32_MAX ? sgc.ov[si] : 0.0f;
            if (cur)
              snprintf(lbl, sizeof lbl, "  %.64s (active)", sgc.st.segs[si].name);
            else if (ovf > 0.02f)
              snprintf(lbl, sizeof lbl, "  %.64s  ov %.0f%%", sgc.st.segs[si].name,
                       (double)ovf * 100.0);
            else
              snprintf(lbl, sizeof lbl, "  %.64s", sgc.st.segs[si].name);
            if (igSelectable_Bool(lbl, cur, 0, (ImVec2){0, 0}) && !cur && !act_pending &&
                sgc.act_ready == UINT32_MAX) {
              sgc.act_req = si;
              pthread_cond_signal(&sgc.cv);
            }
          }
          if (igCollapsingHeader_TreeNodeFlags("all surfaces", 0))
            for (uint32_t si = 0; si < sgc.st.n; si++) {
              bool cur = strcmp(sgc.st.segs[si].name, sgc_active) == 0;
              char lbl[96];
              snprintf(lbl, sizeof lbl, "%.64s##s%u", sgc.st.segs[si].name, si);
              if (igSelectable_Bool(lbl, cur, 0, (ImVec2){0, 0}) && !cur && !act_pending &&
                  sgc.act_ready == UINT32_MAX) {
                sgc.act_req = si;
                pthread_cond_signal(&sgc.cv);
              }
            }
          pthread_mutex_unlock(&sgc.mu);
        }
      }
      if (overlay_path) {
        igCheckbox("ink overlay", &overlay_show);
        igSameLine(0, 10);
        igSetNextItemWidth(140);
        igSliderFloat("gain", &overlay_gain, 0.25f, 8.0f, "%.2f",
                      ImGuiSliderFlags_Logarithmic);
      }
      igText("bricks: hot %u/%u slots  warm %u (%.0f/%llu MB)%s", bst.hot, bst.hot_cap,
             bst.warm_bricks, (double)bst.warm_bytes / 1048576.0,
             (unsigned long long)(bst.warm_cap >> 20), bst.inflight ? "  streaming..." : "");
      if (bst.net_pending || bst.net_fetched)
        igText("net ingest: %u pending  %llu chunks fetched  %llu bricks cached",
               bst.net_pending, (unsigned long long)bst.net_fetched,
               (unsigned long long)bst.net_encoded);
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
      igSeparator();
      for (int ph = 0; ph < MT_N; ph++)
        igText("mt %-6s   %6.2f ms (max %6.1f)", mt_name[ph], mt_ema[ph] / 1e6,
               (double)mt_max[ph] / 1e6);
    }
    if (umbilicus_path && vslab_mode)
      igTextDisabled("click: set point | wheel, R/F: 1 slice\n"
                     "PgUp/PgDn: z step | Ctrl+click: set + advance\n"
                     "Shift+drag: pan | Shift+wheel: zoom | F12: shot");
    else if (multiview_path)
      igTextDisabled("drag: pan view | wheel: zoom | Shift+wheel: slice\n"
                     "R/F: slice | Ctrl+click: set focus | F12: shot\n"
                     "Space: solo hovered view | checkboxes hide views%s",
                     umbilicus_path ? "\nU: place umbilicus point at cursor" : "");
    else if (cam_mode == CAM_ORBIT)
      igTextDisabled("drag orbit | shift+drag pan cam | ctrl+drag move vol\n"
                     "ctrl+shift+drag rot vol | wheel zoom | WASD pan | F12 shot");
    else
      igTextDisabled("click: fly (Esc releases)   WASD+QE: move   F12: shot");
    } /* panel_content */
    igEnd();

    od_browser_window(&od, &od_window, od_next_bricks, sizeof od_next_bricks, od_next_seg,
                      sizeof od_next_seg, &od_swap);
    if (od_swap) running = false; /* teardown + reopen in the dataset loop */

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
        /* recompute when the pane is visible OR the segment view is (its
         * plane trace lines draw there even with the pane collapsed/solo) */
        if (!(mv_mask & (1u << i)) && !(mv_mask & 1u)) continue;
        bool stale = mv_ol_slice[i] != mv[i].slice || mv_ol_gen[i] != mv_basis_gen;
        if (stale) {
          mv_ol[i].n = 0;
          r3d_segtrace_basis(&mv_seg, &mv_rows, NULL, 0.0f, mv_po[i], mv_pb[i][0],
                             mv_pb[i][1], mv_pb[i][2], mv[i].slice, mv_lines_emit,
                             &mv_ol[i]);
        }
        if (stale || mv_ol_zoff != zoff) {
          mv_ol_off[i].n = 0;
          if (zoff != 0.0)
            r3d_segtrace_basis(&mv_seg, &mv_rows, mv_normals, (float)zoff, mv_po[i],
                               mv_pb[i][0], mv_pb[i][1], mv_pb[i][2], mv[i].slice,
                               mv_lines_emit, &mv_ol_off[i]);
        }
        mv_ol_slice[i] = mv[i].slice;
        mv_ol_gen[i] = mv_basis_gen;
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
      if (sgc.open) { /* corpus surfaces: dimmed intersection polylines under
         * the active segment's curve. Queries hit only the tile index;
         * grids come from the worker's cache (missing ones get queued) and
         * at most a few re-traces run per frame to amortize slice scrubs. */
        const ImU32 dim_col = 0x505a5a5au;
        const ImU32 ov_hi = 0x783c8ce6u; /* heavy overlap with active: orange */
        const ImU32 ov_lo = 0x5864b4d2u; /* light overlap: sand */
        pthread_mutex_lock(&sgc.mu);
        int traces = 0;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i))) continue;
          const double *bn = mv_pb[i][2];
          double sl = mv[i].slice + bn[0] * mv_po[i][0] + bn[1] * mv_po[i][1] +
                      bn[2] * mv_po[i][2];
          if (sgc_key_slice[i] != sl || sgc_key_gen[i] != mv_basis_gen) {
            sgc_key_slice[i] = sl;
            sgc_key_gen[i] = mv_basis_gen;
            sgc_nhits[i] = r3d_segstore_plane_query(&sgc.st, bn, sl, 1.0, NULL, NULL,
                                                    sgc_hits[i], 128);
            if (sgc_nhits[i] > 128) sgc_nhits[i] = 128;
            /* nearest-first by bbox center vs the pane center: zoomed-out
             * panes cross the whole corpus, and drawing every polyline
             * costs ~8 ms/frame — only the closest SGC_DRAW_MAX draw */
            double pcw[3];
            r3d_mv_b2w(mv_pb[i], mv_po[i], mv[i].cu, mv[i].cv, mv[i].slice, pcw);
            for (uint32_t a2 = 1; a2 < sgc_nhits[i]; a2++)
              for (uint32_t b2 = a2; b2 > 0; b2--) {
                double da = 0.0, db = 0.0;
                for (int c2 = 0; c2 < 3; c2++) {
                  const r3d_segmeta *sa = &sgc.st.segs[sgc_hits[i][b2]];
                  const r3d_segmeta *sb = &sgc.st.segs[sgc_hits[i][b2 - 1]];
                  double ca =
                      ((double)sa->bbox[0][c2] + (double)sa->bbox[1][c2]) * 0.5 - pcw[c2];
                  double cb =
                      ((double)sb->bbox[0][c2] + (double)sb->bbox[1][c2]) * 0.5 - pcw[c2];
                  da += ca * ca;
                  db += cb * cb;
                }
                if (da >= db) break;
                uint32_t tswap = sgc_hits[i][b2];
                sgc_hits[i][b2] = sgc_hits[i][b2 - 1];
                sgc_hits[i][b2 - 1] = tswap;
              }
          }
          ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
          ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
          ImDrawList_PushClipRect(draw, cmin, cmax, false);
          uint32_t draw_n = sgc_nhits[i] < 48 ? sgc_nhits[i] : 48;
          for (uint32_t k = 0; k < draw_n; k++) {
            uint32_t si = sgc_hits[i][k];
            if (strcmp(sgc.st.segs[si].name, sgc_active) == 0) continue;
            sgc_ent *e = &sgc.ent[si];
            if (e->state == SGC_EMPTY) {
              sgc_request(&sgc, si);
              continue;
            }
            sgc_line *ln = &sgc_ln[i][si];
            if (e->state == SGC_READY &&
                (!ln->valid || ln->slice != mv[i].slice || ln->gen != mv_basis_gen) &&
                traces < 2) {
              traces++;
              ln->l.n = 0;
              r3d_segtrace_basis(&e->s, &e->rows, NULL, 0.0f, mv_po[i], mv_pb[i][0],
                                 mv_pb[i][1], mv_pb[i][2], mv[i].slice, mv_lines_emit,
                                 &ln->l);
              ln->slice = mv[i].slice;
              ln->gen = mv_basis_gen;
              ln->valid = true;
              e->last_use = ++sgc.tick;
            }
            if (ln->valid && ln->slice == mv[i].slice && ln->gen == mv_basis_gen) {
              float ovf = sgc.ov_active != UINT32_MAX ? sgc.ov[si] : 0.0f;
              ImU32 col = ovf > 0.15f ? ov_hi : (ovf > 0.02f ? ov_lo : dim_col);
              /* corpus lines tolerate a 3 px collapse: half the tessellation */
              mv_draw_lines(draw, &mv[i], ln->l.w, ln->l.n, col, 1.0f, 9.0f);
            }
          }
          ImDrawList_PopClipRect(draw);
        }
        pthread_mutex_unlock(&sgc.mu);
      }
      for (int i = 1; i < 4; i++) { /* segment curve on each plane view */
        if (!(mv_mask & (1u << i))) continue;
        ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
        ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
        ImDrawList_PushClipRect(draw, cmin, cmax, false);
        mv_draw_lines(draw, &mv[i], mv_ol[i].w, mv_ol[i].n, seg_col, 1.6f, 1.0f);
        mv_draw_lines(draw, &mv[i], mv_ol_off[i].w, mv_ol_off[i].n, seg_off_col, 1.0f, 1.0f);
        ImDrawList_PopClipRect(draw);
      }
      if (mv_mask & 1u) { /* plane trace lines on the flattened segment view */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        ImVec2 cmin = {(float)sv->px, (float)sv->py};
        ImVec2 cmax = {(float)(sv->px + sv->pw), (float)(sv->py + sv->ph)};
        ImDrawList_PushClipRect(draw, cmin, cmax, false);
        for (int i = 1; i < 4; i++) /* even for collapsed panes: the slices
                                     * still exist and the lines orient you */
          mv_draw_lines(draw, sv, mv_ol[i].g, mv_ol[i].n, trace_col[i], 1.4f, 1.0f);
        ImDrawList_PopClipRect(draw);
      }
      for (int i = 1; i < 4; i++) { /* pane borders around visible views */
        if (!(mv_mask & (1u << i))) continue;
        if (mv[i].px > 0)
          ImDrawList_AddLine(draw, (ImVec2){(float)mv[i].px, (float)mv[i].py},
                             (ImVec2){(float)mv[i].px, (float)(mv[i].py + mv[i].ph)}, bc,
                             1.0f);
        if (mv[i].py > 0)
          ImDrawList_AddLine(draw, (ImVec2){(float)mv[i].px, (float)mv[i].py},
                             (ImVec2){(float)(mv[i].px + mv[i].pw), (float)mv[i].py}, bc,
                             1.0f);
      }
      for (int i = 1; i < 4; i++) {
        if (!(mv_mask & (1u << i))) continue;
        double fu, fv, fs;
        r3d_mv_w2b(mv_pb[i], mv_po[i], mv_focus, &fu, &fv, &fs);
        float fx_, fy_;
        r3d_mv_project(&mv[i], fu, fv, &fx_, &fy_);
        if (fx_ >= (float)mv[i].px && fx_ < (float)(mv[i].px + mv[i].pw) &&
            fy_ >= (float)mv[i].py && fy_ < (float)(mv[i].py + mv[i].ph))
          ImDrawList_AddCircle(draw, (ImVec2){fx_, fy_}, 10.0f, fc, 24, 2.0f);
      }
      if (umbilicus_path && umbilicus.count) {
        /* umbilicus curve in every plane pane: connected control points
         * (magenta) + a crosshair at the XY pane's current-z interpolation */
        const ImU32 uc = 0xffff00ffu, ub = 0xff000000u;
        for (int i = 1; i < 4; i++) {
          if (!(mv_mask & (1u << i))) continue;
          ImVec2 cmin = {(float)mv[i].px, (float)mv[i].py};
          ImVec2 cmax = {(float)(mv[i].px + mv[i].pw), (float)(mv[i].py + mv[i].ph)};
          ImDrawList_PushClipRect(draw, cmin, cmax, false);
          /* only what actually intersects this pane's slab draws: clip each
           * curve segment to fs in [slice, slice+thickness] along the pane
           * normal, and markers to points inside it */
          double lo = mv[i].slice, hi = lo + (double)mv_thick;
          double pu = 0.0, pv = 0.0, pf = 0.0;
          for (size_t k = 0; k < umbilicus.count; k++) {
            double P[3] = {umbilicus.points[k].x, umbilicus.points[k].y,
                           umbilicus.points[k].z};
            double fu, fv, fs;
            r3d_mv_w2b(mv_pb[i], mv_po[i], P, &fu, &fv, &fs);
            if (k) { /* parametric slab clip of the segment prev -> cur */
              double ct0 = 0.0, ct1 = 1.0;
              if (pf == fs) {
                if (pf < lo || pf > hi) ct0 = 1.0;
              } else {
                double ta = (lo - pf) / (fs - pf), tb = (hi - pf) / (fs - pf);
                if (ta > tb) {
                  double tt = ta;
                  ta = tb;
                  tb = tt;
                }
                if (ta > ct0) ct0 = ta;
                if (tb < ct1) ct1 = tb;
              }
              if (ct0 < ct1) {
                float ax, ay, bx, by;
                r3d_mv_project(&mv[i], pu + (fu - pu) * ct0, pv + (fv - pv) * ct0, &ax, &ay);
                r3d_mv_project(&mv[i], pu + (fu - pu) * ct1, pv + (fv - pv) * ct1, &bx, &by);
                ImDrawList_AddLine(draw, (ImVec2){ax, ay}, (ImVec2){bx, by}, uc, 1.5f);
              }
            }
            if (fs >= lo && fs <= hi) {
              float sx_, sy_;
              r3d_mv_project(&mv[i], fu, fv, &sx_, &sy_);
              ImDrawList_AddCircleFilled(draw, (ImVec2){sx_, sy_}, 3.5f, ub, 12);
              ImDrawList_AddCircleFilled(draw, (ImVec2){sx_, sy_}, 2.5f, uc, 12);
            }
            pu = fu;
            pv = fv;
            pf = fs;
          }
          if (i == R3D_MV_XY && !mv_aligned) {
            double ux, uy;
            if (umb_interp(&umbilicus, mv[i].slice, &ux, &uy)) {
              double fu, fv, fs, P[3] = {ux, uy, mv[i].slice};
              r3d_mv_w2b(mv_pb[i], mv_po[i], P, &fu, &fv, &fs);
              float sx_, sy_;
              r3d_mv_project(&mv[i], fu, fv, &sx_, &sy_);
              ImDrawList_AddCircle(draw, (ImVec2){sx_, sy_}, 9.0f, ub, 24, 4.0f);
              ImDrawList_AddCircle(draw, (ImVec2){sx_, sy_}, 9.0f, uc, 24, 2.0f);
            }
          }
          ImDrawList_PopClipRect(draw);
        }
      }
      if (mv_mask & 1u) { /* flattened view: mark the surface grid point
                           * anchoring the focus (nearest valid point) */
        const r3d_mview *sv = &mv[R3D_MV_SEG];
        float fx_, fy_;
        r3d_mv_project(sv, (double)mv_align_ij[0], (double)mv_align_ij[1], &fx_, &fy_);
        if (fx_ >= (float)sv->px && fx_ < (float)(sv->px + sv->pw) && fy_ >= (float)sv->py &&
            fy_ < (float)(sv->py + sv->ph))
          ImDrawList_AddCircle(draw, (ImVec2){fx_, fy_}, 10.0f, fc, 24, 2.0f);
      }
    }

    mt_t[3] = r3d_now_ns();
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
        .overlay_gain = overlay_gain,
        .overlay_flags = overlay_path && overlay_show ? 1u : 0u,
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
          if (mv_mask & 1u) { /* segment view: walk the visible grid rect and
             * request the bricks its surface points (+ normal offset) touch */
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
            if (!(mv_mask & (1u << i))) continue; /* collapsed: no streaming */
            double hw = (double)mv[i].pw * 0.5 / mv[i].zoom;
            double hh = (double)mv[i].ph * 0.5 / mv[i].zoom;
            double th = (double)mv_thick;
            /* world AABB of the visible rect + slab: frame center at the
             * slab midpoint, per-axis extents from the |basis| components */
            double ctr[3];
            r3d_mv_b2w(mv_pb[i], mv_po[i], mv[i].cu, mv[i].cv, mv[i].slice + th * 0.5,
                       ctr);
            float lo[3], hi[3];
            for (int a = 0; a < 3; a++) {
              double he = hw * fabs(mv_pb[i][0][a]) + hh * fabs(mv_pb[i][1][a]) +
                          (th * 0.5 + 1.0) * fabs(mv_pb[i][2][a]);
              lo[a] = (float)((ctr[a] - he) / (double)mdim);
              hi[a] = (float)((ctr[a] + he) / (double)mdim);
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
    mt_t[4] = r3d_now_ns();
    r3d_frame_stats st = {0};
    int frc;
    if (multiview_path) {
      /* one FrameParams per quadrant: axis-aligned ortho cameras over the
       * bricks virtual volume, slab-clipped to each view's slice */
      uint32_t mdim = brick_shape[0];
      for (int a = 1; a < 3; a++)
        if (brick_shape[a] > mdim) mdim = brick_shape[a];
      r3d_frame_params vp4[4];
      uint32_t nvp = 0;
      for (int i = 0; i < 4; i++) {
        if (!(mv_mask & (1u << i))) continue; /* collapsed: no dispatch at all */
        r3d_frame_params q = p;
        q.viewport[0] = (uint32_t)mv[i].pw;
        q.viewport[1] = (uint32_t)mv[i].ph;
        q.view_org = (uint32_t)mv[i].px | ((uint32_t)mv[i].py << 16);
        if (i == R3D_MV_SEG) {
          /* flattened segment: raycast the surface-volume window. Camera in
           * FLATTENED VOXELS (grid / scale); window mapping from the backend. */
          q.view_flags = R3D_VIEW_SURF | (mv_stretch ? R3D_VIEW_STRETCH : 0u);
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
          vp4[nvp++] = q;
          continue;
        }
        /* ortho camera from the view frame: origin 2 voxels before the slab
         * near face along the plane normal, half-extent vectors along u/v;
         * the slab clip rides the ray (oblique), which for axis frames is
         * identical to the old per-axis box clip */
        q.view_flags = R3D_VIEW_ORTHO | R3D_VIEW_OBLIQUE;
        double hw = (double)mv[i].pw * 0.5 / mv[i].zoom;
        double hh = (double)mv[i].ph * 0.5 / mv[i].zoom;
        double og[3];
        r3d_mv_b2w(mv_pb[i], mv_po[i], mv[i].cu, mv[i].cv, mv[i].slice - 2.0, og);
        for (int a = 0; a < 3; a++) {
          q.cam_origin[a] = (float)(og[a] / (double)mdim);
          q.cam_right[a] = (float)(mv_pb[i][0][a] * hw / (double)mdim); /* +x -> +u */
          q.cam_up[a] = (float)(-mv_pb[i][1][a] * hh / (double)mdim);   /* ndc +y -> -v */
          q.cam_forward[a] = (float)mv_pb[i][2][a];
        }
        q.slab_z0 = 2.0f; /* voxels from the ray origin to the slab near face */
        q.slab_depth = (uint32_t)mv_thick;
        vp4[nvp++] = q;
      }
      frc = r3d_frame_views(renderer, vp4, nvp, &st);
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
    mt_t[5] = r3d_now_ns();
    for (int ph = 0; ph < MT_N; ph++) {
      uint64_t d = mt_t[ph + 1] - mt_t[ph];
      mt_sum[ph] += d;
      if (d > mt_max[ph]) mt_max[ph] = d;
      mt_ema[ph] = mt_ema[ph] * 0.95 + (double)d * 0.05;
    }
    mt_frames++;
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
  if (mt_frames > 2) {
    printf("mainthread avg:");
    for (int ph = 0; ph < MT_N; ph++)
      printf(" %s %.2f", mt_name[ph], (double)mt_sum[ph] / (double)mt_frames / 1e6);
    printf(" ms | max:");
    for (int ph = 0; ph < MT_N; ph++) printf(" %s %.1f", mt_name[ph], (double)mt_max[ph] / 1e6);
    printf(" ms\n");
  }
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
    r3d_segrows_free(&mv_rows);
    free(mv_normals);
    if (sgc.open) {
      pthread_mutex_lock(&sgc.mu);
      uint32_t ready = 0;
      uint64_t lines = 0;
      for (uint32_t si = 0; si < sgc.st.n; si++)
        if (sgc.ent[si].state == SGC_READY) ready++;
      for (int i = 1; i < 4; i++)
        for (uint32_t si = 0; si < sgc.st.n; si++)
          if (sgc_ln[i][si].valid) lines += sgc_ln[i][si].l.n;
      printf("segments: %u/%u cached (%zu MB), hits XY %u XZ %u YZ %u, %llu overlay "
             "segments traced\n",
             ready, sgc.st.n, sgc.bytes >> 20, sgc_nhits[R3D_MV_XY], sgc_nhits[R3D_MV_XZ],
             sgc_nhits[R3D_MV_YZ], (unsigned long long)lines);
      pthread_mutex_unlock(&sgc.mu);
    }
    for (int i = 1; i < 4; i++) {
      if (!sgc_ln[i]) continue;
      for (uint32_t si = 0; si < sgc.st.n; si++) {
        free(sgc_ln[i][si].l.w);
        free(sgc_ln[i][si].l.g);
      }
      free(sgc_ln[i]);
      sgc_ln[i] = NULL;
    }
    sgc_close(&sgc);
    for (int i = 0; i < 4; i++) {
      free(mv_ol[i].w);
      free(mv_ol[i].g);
      free(mv_ol_off[i].w);
      free(mv_ol_off[i].g);
    }
  }
  free(prof_samples);
  if (!od_swap) break;
  r3d_bricks_end(renderer); /* dataset swap: teardown, then reopen */
  if (slab_src.voxels) r3d_volume_close(&slab_src);
  } /* dataset loop */
  if (od.fth_up) { /* stop the browser's listing worker */
    pthread_mutex_lock(&od.fmu);
    od.fquit = true;
    pthread_cond_broadcast(&od.fcv);
    pthread_mutex_unlock(&od.fmu);
    pthread_join(od.fth, NULL);
  }
  r3d_destroy(renderer);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
