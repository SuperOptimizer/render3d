#include "viewui.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "cimgui.h"
#include "core/camera.h"
#include "core/pngw.h"
#include "core/screenshot.h"
#include "core/view.h"
#include "core/volume.h"
#include "render/render.h"

typedef struct view_ui {
  r3d_view_manifest man;
  uint32_t index;
  r3d_view_packet pkt;
  r3d_view_opts opts;

  uint8_t *rgba; /* w*h*4 composite of the current slice */
  ImTextureData *tex;
  /* A texture the backend has seen may not be freed until it has destroyed
   * its own GPU resources, which only happens inside a later draw — so an
   * axis change or a resized packet retires the old one here. */
  ImTextureData *retired[8];
  uint32_t n_retired;
  uint32_t tex_w, tex_h;
  bool image_stale;

  float pan[2], zoom;
  bool fit_pending;

  r3d_view_live live;
  bool live_up;
  int64_t nudge[3];    /* arrow-key box offset in scroll voxels, z/y/x */
  int nudge_step;
  char want[128];
  int tta;             /* 0 none, 1 flip8, 2 flip8_rot4 */

  uint32_t shot_seq;
  char status[256];
  char hover[2048];
} view_ui;

static const char *const k_tta_name[3] = {"none", "flip8", "flip8_rot4"};

__attribute__((format(printf, 2, 3))) static void ui_status(view_ui *a, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(a->status, sizeof a->status, fmt, ap);
  va_end(ap);
}

/* ===================== texture plumbing ================================== */

static void ui_retire_texture(view_ui *a) {
  if (!a->tex) return;
  a->tex->WantDestroyNextFrame = true;
  if (a->n_retired < 8u) a->retired[a->n_retired++] = a->tex;
  a->tex = NULL;
}

static void ui_reap_textures(view_ui *a) {
  for (uint32_t i = 0; i < a->n_retired;) {
    if (a->retired[i]->Status == ImTextureStatus_Destroyed) {
      igUnregisterUserTexture(a->retired[i]);
      ImTextureData_destroy(a->retired[i]);
      a->retired[i] = a->retired[--a->n_retired];
    } else {
      i++;
    }
  }
}

static bool ui_make_texture(view_ui *a, uint32_t w, uint32_t h) {
  ui_retire_texture(a);
  a->tex = ImTextureData_ImTextureData();
  if (!a->tex) return false;
  ImTextureData_Create(a->tex, ImTextureFormat_RGBA32, (int)w, (int)h);
  a->tex->UseColors = true;
  ImTextureData_SetStatus(a->tex, ImTextureStatus_WantCreate);
  igRegisterUserTexture(a->tex);
  a->tex_w = w;
  a->tex_h = h;
  return true;
}

/* Size the pixel buffer and the texture to the current axis' slice plane. */
static bool ui_resize(view_ui *a) {
  uint32_t w, h;
  r3d_view_slice_dims(&a->pkt, a->opts.axis, &w, &h);
  if (w != a->tex_w || h != a->tex_h || !a->tex) {
    free(a->rgba);
    a->rgba = malloc((size_t)w * h * 4u);
    if (!a->rgba) return false;
    if (!ui_make_texture(a, w, h)) return false;
    a->fit_pending = true;
  }
  a->image_stale = true;
  return true;
}

static void ui_refresh_image(view_ui *a) {
  if (!a->tex || !a->rgba) return;
  r3d_view_composite(&a->pkt, &a->opts, a->rgba);
  memcpy(ImTextureData_GetPixels(a->tex), a->rgba, (size_t)a->tex_w * a->tex_h * 4u);
  if (a->tex->Status != ImTextureStatus_WantCreate) {
    a->tex->UpdateRect = (ImTextureRect){0, 0, (unsigned short)a->tex_w, (unsigned short)a->tex_h};
    ImTextureData_SetStatus(a->tex, ImTextureStatus_WantUpdates);
  }
  a->image_stale = false;
}

/* ===================== packet switching ================================== */

static int ui_open(view_ui *a, r3d_renderer *renderer, uint32_t index) {
  if (index >= a->man.count) return -1;
  r3d_view_packet np = {0};
  if (r3d_view_packet_load(&np, a->man.ent[index].path) != 0) return -1;
  r3d_view_packet_free(&a->pkt);
  a->pkt = np;
  a->index = index;
  int keep_axis = a->opts.axis;
  r3d_view_opts_default(&a->opts);
  a->opts.axis = keep_axis;
  a->opts.slice = r3d_view_slice_count(&a->pkt, a->opts.axis) / 2u;
  a->tex_w = a->tex_h = 0; /* force the buffer/texture to be re-sized */
  if (!ui_resize(a)) return -1;
  memset(a->nudge, 0, sizeof a->nudge);
  if (renderer && a->pkt.ct >= 0) {
    r3d_volume_desc d = {.nx = a->pkt.nx, .ny = a->pkt.ny, .nz = a->pkt.nz,
                         .brick_dim = R3D_BRICK_DIM, .voxel_um = (float)a->pkt.voxel_um};
    (void)r3d_upload_volume(renderer, &d, a->pkt.layer[a->pkt.ct].data);
  }
  ui_status(a, "packet %u/%u  %s  %u layers", index + 1u, a->man.count, a->man.ent[index].name,
            a->pkt.count);
  return 0;
}

static void ui_set_axis(view_ui *a, int axis) {
  if (axis == a->opts.axis) return;
  double frac = (double)a->opts.slice /
                (double)(r3d_view_slice_count(&a->pkt, a->opts.axis) - 1u + 1u);
  a->opts.axis = axis;
  uint32_t n = r3d_view_slice_count(&a->pkt, axis);
  uint32_t s = (uint32_t)(frac * (double)n);
  a->opts.slice = s < n ? s : n - 1u;
  (void)ui_resize(a);
}

static void ui_set_slice(view_ui *a, int64_t s) {
  int64_t n = (int64_t)r3d_view_slice_count(&a->pkt, a->opts.axis);
  if (s < 0) s = 0;
  if (s >= n) s = n - 1;
  if ((uint32_t)s == a->opts.slice) return;
  a->opts.slice = (uint32_t)s;
  a->image_stale = true;
}

/* ===================== screenshot ======================================== */

static void ui_screenshot(view_ui *a) {
  char path[256];
  snprintf(path, sizeof path, "view_%s_%c%u_%03u.png",
           a->man.ent[a->index].name[0] ? a->man.ent[a->index].name : "packet",
           "zyx"[a->opts.axis], a->opts.slice, a->shot_seq);
  for (char *c = path; *c; c++)
    if (*c == '/') *c = '_';
  if (a->image_stale) ui_refresh_image(a);
  if (r3d_png_write_rgba(path, a->rgba, a->tex_w, a->tex_h) == 0) {
    a->shot_seq++;
    ui_status(a, "wrote %s (%ux%u)", path, a->tex_w, a->tex_h);
  } else {
    ui_status(a, "screenshot to %s FAILED", path);
  }
}

/* ===================== live mode ========================================= */

static void ui_predict(view_ui *a) {
  if (!a->live_up) return;
  int64_t origin[3] = {a->pkt.origin[0] + a->nudge[0], a->pkt.origin[1] + a->nudge[1],
                       a->pkt.origin[2] + a->nudge[2]};
  int64_t dims[3] = {a->pkt.nz, a->pkt.ny, a->pkt.nx};
  r3d_view_live_request(&a->live, origin, dims, a->want, k_tta_name[a->tta]);
  ui_status(a, "predict z%lld y%lld x%lld  %lldx%lldx%lld", (long long)origin[0],
            (long long)origin[1], (long long)origin[2], (long long)dims[0], (long long)dims[1],
            (long long)dims[2]);
}

/* Fold a server reply into the packet: new layers appear in the panel, layers
 * that already exist keep their display state and take the new bytes. */
static void ui_poll_live(view_ui *a, r3d_renderer *renderer) {
  if (!a->live_up) return;
  r3d_view_packet got = {0};
  if (!r3d_view_live_poll(&a->live, &got)) return;
  uint32_t was = a->pkt.count;
  bool same_box = got.nz == a->pkt.nz && got.ny == a->pkt.ny && got.nx == a->pkt.nx &&
                  memcmp(got.origin, a->pkt.origin, sizeof got.origin) == 0;
  if (r3d_view_merge(&a->pkt, &got) != 0) {
    r3d_view_packet_free(&got);
    ui_status(a, "merging the server's layers failed");
    return;
  }
  if (!same_box) { /* a nudged box replaced the packet outright */
    memset(a->nudge, 0, sizeof a->nudge);
    a->opts.cmp_a = a->opts.cmp_b = -1;
    a->opts.slice = r3d_view_slice_count(&a->pkt, a->opts.axis) / 2u;
    a->tex_w = a->tex_h = 0;
    if (renderer && a->pkt.ct >= 0) {
      r3d_volume_desc d = {.nx = a->pkt.nx, .ny = a->pkt.ny, .nz = a->pkt.nz,
                           .brick_dim = R3D_BRICK_DIM, .voxel_um = (float)a->pkt.voxel_um};
      (void)r3d_upload_volume(renderer, &d, a->pkt.layer[a->pkt.ct].data);
    }
  }
  (void)ui_resize(a);
  ui_status(a, "server: %u layers (%u new)", a->pkt.count,
            a->pkt.count > was ? a->pkt.count - was : 0u);
}

/* ===================== panel ============================================= */

static void ui_layer_row(view_ui *a, uint32_t i) {
  r3d_view_layer *l = &a->pkt.layer[i];
  igPushID_Int((int)i);
  bool changed = false;
  changed |= igCheckbox(l->name, &l->show);
  igSameLine(0, 6);
  igTextDisabled("%s", r3d_view_kind_name[l->kind]);
  igSameLine(0, 6);
  if (igSmallButton(l->solo ? "unsolo" : "solo")) {
    bool want_solo = !l->solo;
    for (uint32_t k = 0; k < a->pkt.count; k++) a->pkt.layer[k].solo = false;
    l->solo = want_solo;
    changed = true;
  }
  changed |= igColorEdit3("colour", l->color, ImGuiColorEditFlags_NoInputs);
  igSameLine(0, 6);
  changed |= igSliderFloat("alpha", &l->opacity, 0.0f, 1.0f, "%.2f", 0);
  switch (l->kind) {
    case R3D_VIEW_PROB:
      changed |= igSliderFloat("threshold", &l->threshold, 0.0f, 1.0f, "%.2f", 0);
      break;
    case R3D_VIEW_DENSITY:
      changed |= igSliderFloat("threshold", &l->threshold, 0.0f, 0.255f, "%.3f", 0);
      break;
    case R3D_VIEW_COUNT:
      changed |= igSliderFloat("threshold", &l->threshold, 0.0f, 255.0f, "%.0f", 0);
      break;
    case R3D_VIEW_SDF:
      changed |= igSliderFloat("tol (vox)", &l->tol, 0.0f, 8.0f, "%.2f", 0);
      igSameLine(0, 6);
      changed |= igCheckbox("heat", &l->heatmap);
      break;
    case R3D_VIEW_SIGNED:
      if (l->vec[0]) igTextDisabled("vec %s axis %d", l->vec, l->axis);
      break;
    case R3D_VIEW_CLASS:
      if (l->npalette) {
        char names[256] = "";
        size_t at = 0;
        for (uint32_t k = 0; k < l->npalette && at + 1u < sizeof names; k++) {
          int n = snprintf(names + at, sizeof names - at, "%s%u:%s", k ? " " : "", k,
                           l->palette[k]);
          if (n < 0) break;
          at += (size_t)n < sizeof names - at ? (size_t)n : sizeof names - at - 1u;
        }
        igTextDisabled("%s", names);
      }
      break;
    case R3D_VIEW_CT:
    case R3D_VIEW_NKIND: break;
  }
  igPopID();
  if (changed) a->image_stale = true;
}

static void ui_compare_picker(view_ui *a) {
  if (!igCollapsingHeader_TreeNodeFlags("compare", 0)) return;
  char preview[160];
  for (int side = 0; side < 2; side++) {
    int *sel = side == 0 ? &a->opts.cmp_a : &a->opts.cmp_b;
    int other = side == 0 ? a->opts.cmp_b : a->opts.cmp_a;
    if (*sel >= 0 && (uint32_t)*sel < a->pkt.count)
      snprintf(preview, sizeof preview, "%s.%s", a->pkt.layer[*sel].group,
               a->pkt.layer[*sel].name);
    else
      snprintf(preview, sizeof preview, "(none)");
    if (igBeginCombo(side == 0 ? "A" : "B", preview, 0)) {
      if (igSelectable_Bool("(none)", *sel < 0, 0, (ImVec2){0, 0})) {
        *sel = -1;
        a->image_stale = true;
      }
      for (uint32_t i = 0; i < a->pkt.count; i++) {
        const r3d_view_layer *l = &a->pkt.layer[i];
        if (l->kind == R3D_VIEW_CT) continue;
        /* only same-kind pairs may be compared (spec item 3) */
        if (other >= 0 && (uint32_t)other < a->pkt.count &&
            a->pkt.layer[other].kind != l->kind)
          continue;
        char label[160];
        snprintf(label, sizeof label, "%s.%s (%s)", l->group, l->name,
                 r3d_view_kind_name[l->kind]);
        if (igSelectable_Bool(label, *sel == (int)i, 0, (ImVec2){0, 0})) {
          *sel = (int)i;
          a->image_stale = true;
        }
      }
      igEndCombo();
    }
  }
  r3d_view_cmp cm;
  if (r3d_view_compare(&a->pkt, &a->opts, &cm) == 0) {
    if (a->pkt.layer[a->opts.cmp_a].kind == R3D_VIEW_CLASS) {
      if (igSliderInt("class (-1 = any)", &a->opts.cmp_class, -1, 15, "%d", 0))
        a->image_stale = true;
    }
    igColorEdit3("agree", a->opts.cmp_col[0], ImGuiColorEditFlags_NoInputs);
    igSameLine(0, 6);
    igColorEdit3("A only", a->opts.cmp_col[1], ImGuiColorEditFlags_NoInputs);
    igSameLine(0, 6);
    igColorEdit3("B only", a->opts.cmp_col[2], ImGuiColorEditFlags_NoInputs);
    igText("agree %llu  A-only %llu  B-only %llu", (unsigned long long)cm.agree,
           (unsigned long long)cm.a_only, (unsigned long long)cm.b_only);
    igText("Dice (this slice) %.4f", cm.dice);
  } else {
    igTextDisabled("pick two layers of the same kind");
  }
}

static void ui_live_panel(view_ui *a) {
  if (!a->live_up) return;
  if (!igCollapsingHeader_TreeNodeFlags("live (tsm serve)", ImGuiTreeNodeFlags_DefaultOpen))
    return;
  char st[256];
  bool busy = false;
  r3d_view_live_status(&a->live, st, sizeof st, &busy);
  r3d_view_hello hello;
  r3d_view_live_hello(&a->live, &hello);
  igText("%s:%d", a->live.host, a->live.port);
  if (hello.valid) {
    igTextWrapped("groups %s", hello.groups);
    if (hello.checkpoint[0]) igTextWrapped("checkpoint %s", hello.checkpoint);
    if (hello.max_dims[0])
      igText("max dims %lld %lld %lld", (long long)hello.max_dims[0],
             (long long)hello.max_dims[1], (long long)hello.max_dims[2]);
  }
  igInputText("want", a->want, sizeof a->want, 0, NULL, NULL);
  igCombo_Str_arr("tta", &a->tta, k_tta_name, 3, -1);
  igSliderInt("nudge step", &a->nudge_step, 1, 512, "%d vox", 0);
  igText("box offset z%lld y%lld x%lld", (long long)a->nudge[0], (long long)a->nudge[1],
         (long long)a->nudge[2]);
  igTextDisabled("arrows nudge y/x, PgUp/PgDn nudge z, Home resets");
  if (igButton("nudge reset", (ImVec2){120, 0})) memset(a->nudge, 0, sizeof a->nudge);
  igSameLine(0, 8);
  if (igButton(busy ? "predicting..." : "predict", (ImVec2){140, 0}) && !busy) ui_predict(a);
  igTextWrapped("%s", st);
}

static void ui_panel(view_ui *a, r3d_renderer *renderer, bool *quit) {
  igText("packet %u / %u   %s", a->index + 1u, a->man.count, a->man.ent[a->index].name);
  igText("dims %u x %u x %u (z,y,x)", a->pkt.nz, a->pkt.ny, a->pkt.nx);
  igText("origin z%lld y%lld x%lld", (long long)a->pkt.origin[0], (long long)a->pkt.origin[1],
         (long long)a->pkt.origin[2]);
  if (a->pkt.scroll[0]) igText("scroll %s", a->pkt.scroll);
  if (a->pkt.voxel_um > 0.0) igText("%.3g um/voxel", a->pkt.voxel_um);
  if (igButton("prev (P)", (ImVec2){110, 0}) && a->index > 0)
    (void)ui_open(a, renderer, a->index - 1u);
  igSameLine(0, 8);
  if (igButton("next (N)", (ImVec2){110, 0}) && a->index + 1u < a->man.count)
    (void)ui_open(a, renderer, a->index + 1u);
  igSameLine(0, 8);
  if (igButton("quit", (ImVec2){70, 0})) *quit = true;
  igSeparator();

  int axis = a->opts.axis;
  igText("axis");
  igSameLine(0, 8);
  for (int k = 0; k < 3; k++) {
    char lbl[8] = {(char)("zyx"[k]), 0};
    if (igRadioButton_IntPtr(lbl, &axis, k)) ui_set_axis(a, k);
    if (k < 2) igSameLine(0, 8);
  }
  int slice = (int)a->opts.slice;
  int nslice = (int)r3d_view_slice_count(&a->pkt, a->opts.axis);
  if (igSliderInt("slice", &slice, 0, nslice - 1, "%d", 0)) ui_set_slice(a, slice);
  if (igSliderFloat("CT gain", &a->opts.ct_gain, 0.1f, 4.0f, "%.2f", 0)) a->image_stale = true;
  igTextDisabled("R/F +-1  PgUp/PgDn +-16  X/Y/Z axis  F12 png");
  igSeparator();

  if (igCollapsingHeader_TreeNodeFlags("layers", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (uint32_t i = 0; i < a->pkt.count;) {
      const char *group = a->pkt.layer[i].group;
      uint32_t end = i;
      while (end < a->pkt.count && strcmp(a->pkt.layer[end].group, group) == 0) end++;
      igPushID_Int((int)i);
      if (igCollapsingHeader_TreeNodeFlags(group, ImGuiTreeNodeFlags_DefaultOpen))
        for (uint32_t k = i; k < end; k++) {
          if (a->pkt.layer[k].kind == R3D_VIEW_CT) {
            igTextDisabled("%s (base image)", a->pkt.layer[k].name);
            continue;
          }
          ui_layer_row(a, k);
        }
      igPopID();
      i = end;
    }
  }
  igSeparator();
  ui_compare_picker(a);
  ui_live_panel(a);
  igSeparator();
  if (a->pkt.provenance[0] && igCollapsingHeader_TreeNodeFlags("provenance", 0))
    igTextWrapped("%s", a->pkt.provenance);
  igSeparator();
  igTextWrapped("%s", a->status);
}

/* ===================== canvas ============================================ */

static void ui_canvas(view_ui *a) {
  ImVec2 p0 = igGetCursorScreenPos();
  ImVec2 avail = igGetContentRegionAvail();
  if (avail.x < 16.0f) avail.x = 16.0f;
  if (avail.y < 48.0f) avail.y = 48.0f;
  avail.y -= 32.0f; /* leave room for the hover readout below */
  if (a->fit_pending) {
    float zx = avail.x / (float)a->tex_w, zy = avail.y / (float)a->tex_h;
    a->zoom = zx < zy ? zx : zy;
    if (a->zoom <= 0.0f) a->zoom = 1.0f;
    a->pan[0] = (avail.x - (float)a->tex_w * a->zoom) * 0.5f;
    a->pan[1] = (avail.y - (float)a->tex_h * a->zoom) * 0.5f;
    a->fit_pending = false;
  }
  igInvisibleButton("##view-canvas", avail,
                    ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
  bool hovered = igIsItemHovered(0);
  ImDrawList *dl = igGetWindowDrawList();
  ImVec2 img0 = {p0.x + a->pan[0], p0.y + a->pan[1]};
  ImVec2 img1 = {img0.x + (float)a->tex_w * a->zoom, img0.y + (float)a->tex_h * a->zoom};
  ImDrawList_PushClipRect(dl, p0, (ImVec2){p0.x + avail.x, p0.y + avail.y}, true);
  ImDrawList_AddRectFilled(dl, p0, (ImVec2){p0.x + avail.x, p0.y + avail.y}, 0xff101010u, 0.0f, 0);
  if (a->tex)
    ImDrawList_AddImage(dl, ImTextureData_GetTexRef(a->tex), img0, img1, (ImVec2){0, 0},
                        (ImVec2){1, 1}, 0xffffffffu);
  ImDrawList_AddRect(dl, img0, img1, 0xff606060u, 0.0f, 1.0f, 0);
  ImDrawList_PopClipRect(dl);

  ImGuiIO *io = igGetIO_Nil();
  ImVec2 mp = igGetMousePos();
  double vx = ((double)mp.x - (double)img0.x) / (double)a->zoom;
  double vy = ((double)mp.y - (double)img0.y) / (double)a->zoom;
  /* ImGui reports +-FLT_MAX when there is no mouse: clamp before narrowing */
  if (!(vx > -1e6) || !(vx < 1e6)) vx = -1e6;
  if (!(vy > -1e6) || !(vy < 1e6)) vy = -1e6;
  int32_t px = (int32_t)floor(vx), py = (int32_t)floor(vy);
  bool inside = hovered && vx >= 0.0 && vy >= 0.0 && px < (int32_t)a->tex_w &&
                py < (int32_t)a->tex_h;

  if (hovered && io->MouseWheel != 0.0f) {
    if (io->KeyShift) {
      ui_set_slice(a, (int64_t)a->opts.slice + (io->MouseWheel > 0.0f ? 1 : -1));
    } else { /* zoom about the cursor: the voxel under it stays put */
      float f = io->MouseWheel > 0.0f ? 1.25f : 1.0f / 1.25f;
      float nz = a->zoom * f;
      if (nz < 0.05f) nz = 0.05f;
      if (nz > 256.0f) nz = 256.0f;
      a->pan[0] = (float)((double)mp.x - (double)p0.x - vx * (double)nz);
      a->pan[1] = (float)((double)mp.y - (double)p0.y - vy * (double)nz);
      a->zoom = nz;
    }
  }
  if (hovered && (igIsMouseDragging(ImGuiMouseButton_Middle, -1.0f) ||
                  igIsMouseDragging(ImGuiMouseButton_Left, -1.0f))) {
    a->pan[0] += io->MouseDelta.x;
    a->pan[1] += io->MouseDelta.y;
  }

  if (inside)
    (void)r3d_view_hover(&a->pkt, &a->opts, (uint32_t)px, (uint32_t)py, a->hover,
                        sizeof a->hover);
  igTextUnformatted(inside ? a->hover : "wheel zoom, shift+wheel slice, drag pan", NULL);
}

/* ===================== keys ============================================== */

static void ui_keys(view_ui *a, r3d_renderer *renderer, bool *quit) {
  ImGuiIO *io = igGetIO_Nil();
  if (io->WantCaptureKeyboard) return;
  if (igIsKeyPressed_Bool(ImGuiKey_R, true)) ui_set_slice(a, (int64_t)a->opts.slice + 1);
  if (igIsKeyPressed_Bool(ImGuiKey_F, true)) ui_set_slice(a, (int64_t)a->opts.slice - 1);
  if (igIsKeyPressed_Bool(ImGuiKey_Z, false)) ui_set_axis(a, R3D_VIEW_AXIS_Z);
  if (igIsKeyPressed_Bool(ImGuiKey_Y, false)) ui_set_axis(a, R3D_VIEW_AXIS_Y);
  if (igIsKeyPressed_Bool(ImGuiKey_X, false)) ui_set_axis(a, R3D_VIEW_AXIS_X);
  if (igIsKeyPressed_Bool(ImGuiKey_F12, false)) ui_screenshot(a);
  if (igIsKeyPressed_Bool(ImGuiKey_N, false) && a->index + 1u < a->man.count)
    (void)ui_open(a, renderer, a->index + 1u);
  if (igIsKeyPressed_Bool(ImGuiKey_P, false) && a->index > 0)
    (void)ui_open(a, renderer, a->index - 1u);
  if (igIsKeyPressed_Bool(ImGuiKey_Q, false) && io->KeyCtrl) *quit = true;
  if (a->live_up) { /* arrow keys walk the request box through the scroll */
    int64_t step = a->nudge_step;
    if (igIsKeyPressed_Bool(ImGuiKey_LeftArrow, true)) a->nudge[2] -= step;
    if (igIsKeyPressed_Bool(ImGuiKey_RightArrow, true)) a->nudge[2] += step;
    if (igIsKeyPressed_Bool(ImGuiKey_UpArrow, true)) a->nudge[1] -= step;
    if (igIsKeyPressed_Bool(ImGuiKey_DownArrow, true)) a->nudge[1] += step;
    if (igIsKeyPressed_Bool(ImGuiKey_PageUp, true)) a->nudge[0] += step;
    if (igIsKeyPressed_Bool(ImGuiKey_PageDown, true)) a->nudge[0] -= step;
    if (igIsKeyPressed_Bool(ImGuiKey_Home, false)) memset(a->nudge, 0, sizeof a->nudge);
    if (igIsKeyPressed_Bool(ImGuiKey_Enter, false)) ui_predict(a);
  } else {
    if (igIsKeyPressed_Bool(ImGuiKey_PageUp, true)) ui_set_slice(a, (int64_t)a->opts.slice + 16);
    if (igIsKeyPressed_Bool(ImGuiKey_PageDown, true)) ui_set_slice(a, (int64_t)a->opts.slice - 16);
  }
}

/* ===================== driver ============================================ */

int r3d_viewui_run(const r3d_viewui_opts *o) {
  view_ui a = {0};
  a.pkt.ct = -1;
  a.nudge_step = 64;
  a.zoom = 1.0f;
  snprintf(a.want, sizeof a.want, "student");
  r3d_view_opts_default(&a.opts);
  if (r3d_view_manifest_load(&a.man, o->path) != 0) {
    fprintf(stderr, "view: %s is not a packet or a view.json manifest\n", o->path);
    return EXIT_FAILURE;
  }
  if (o->serve && *o->serve) a.live_up = r3d_view_live_start(&a.live, o->serve) == 0;

  if (!SDL_Init(o->headless ? SDL_INIT_EVENTS : SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    r3d_view_manifest_free(&a.man);
    return EXIT_FAILURE;
  }
  SDL_Window *win = NULL;
  if (!o->headless) {
    win = SDL_CreateWindow("render3d - model view", o->win_w, o->win_h,
                           SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);
    if (!win) {
      fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
      SDL_Quit();
      r3d_view_manifest_free(&a.man);
      return EXIT_FAILURE;
    }
  }
  r3d_config cfg = {.vsync = !o->no_vsync,
                    .spv_dir = R3D_SPV_DIR,
                    .headless = o->headless,
                    .headless_w = (uint32_t)o->win_w,
                    .headless_h = (uint32_t)o->win_h};
  r3d_renderer *renderer = NULL;
  if (r3d_create(win, &cfg, &renderer) != 0) {
    fprintf(stderr, "renderer init failed\n");
    if (win) SDL_DestroyWindow(win);
    SDL_Quit();
    r3d_view_manifest_free(&a.man);
    return EXIT_FAILURE;
  }

  int rc = EXIT_SUCCESS;
  bool quit = false;
  if (ui_open(&a, renderer, 0) != 0) {
    fprintf(stderr, "view: cannot open %s\n", a.man.ent[0].path);
    quit = true;
    rc = EXIT_FAILURE;
  }
  printf("view: %u packet%s from %s\n", a.man.count, a.man.count == 1 ? "" : "s",
         a.man.path[0] ? a.man.path : a.man.ent[0].path);

  /* test hook: step through packets and axes so an automated run covers the
   * packet switch, the axis change and the texture resize behind it */
  const char *cyc = getenv("R3D_VIEW_CYCLE");
  uint32_t cycle = cyc ? (uint32_t)strtoul(cyc, NULL, 10) : 0u;

  r3d_camera cam;
  r3d_camera_init(&cam, (r3d_v3){0.5f, 0.5f, -1.6f});
  uint32_t frame = 0;
  while (!quit) {
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      r3d_gui_event(renderer, &ev);
      if (ev.type == SDL_EVENT_QUIT || ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) quit = true;
    }
    if (quit) break;
    if (r3d_gui_begin(renderer) != 0) break;
    ui_reap_textures(&a);
    ui_poll_live(&a, renderer);
    ui_keys(&a, renderer, &quit);

    ImGuiViewport *vp = igGetMainViewport();
    const float panel_w = 380.0f;
    igSetNextWindowPos(vp->WorkPos, ImGuiCond_Always, (ImVec2){0, 0});
    igSetNextWindowSize((ImVec2){panel_w, vp->WorkSize.y}, ImGuiCond_Always);
    if (igBegin("model view", NULL,
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoCollapse))
      ui_panel(&a, renderer, &quit);
    igEnd();
    igSetNextWindowPos((ImVec2){vp->WorkPos.x + panel_w, vp->WorkPos.y}, ImGuiCond_Always,
                       (ImVec2){0, 0});
    igSetNextWindowSize((ImVec2){vp->WorkSize.x - panel_w, vp->WorkSize.y}, ImGuiCond_Always);
    if (igBegin("slice", NULL,
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_NoScrollWithMouse))
      ui_canvas(&a);
    igEnd();
    if (a.image_stale) ui_refresh_image(&a);

    r3d_v3 right, up, fwd;
    r3d_camera_basis(&cam, (float)o->win_w / (float)(o->win_h ? o->win_h : 1), &right, &up, &fwd);
    r3d_frame_params fp = {
        .cam_origin = {cam.pos.x, cam.pos.y, cam.pos.z},
        .cam_right = {right.x, right.y, right.z},
        .cam_up = {up.x, up.y, up.z},
        .cam_forward = {fwd.x, fwd.y, fwd.z},
        .step_voxels = 1.0f,
        .density = 1.0f,
        .max_mip = 10.0f,
        .viewport = {(uint32_t)o->win_w, (uint32_t)o->win_h},
        .mode = R3D_MODE_MIP,
        .frame_index = frame,
        .vol_r0 = {1, 0, 0}, .vol_r1 = {0, 1, 0}, .vol_r2 = {0, 0, 1},
    };
    r3d_frame_stats fs;
    if (r3d_frame(renderer, &fp, &fs) < 0) {
      fprintf(stderr, "view: frame failed\n");
      rc = EXIT_FAILURE;
      break;
    }
    frame++;
    if (cycle && frame % cycle == 0) {
      ui_set_axis(&a, (a.opts.axis + 1) % 3);
      if (a.opts.axis == 0) (void)ui_open(&a, renderer, (a.index + 1u) % a.man.count);
    }
    if (o->exit_frames && frame >= o->exit_frames) quit = true;
  }

  if (o->shot_path) { /* headless verification: the composited panes as PPM */
    uint32_t w = 0, h = 0;
    if (r3d_read_frame(renderer, NULL, &w, &h) == 0) {
      uint8_t *px = malloc((size_t)w * h * 4u);
      if (px && r3d_read_frame(renderer, px, &w, &h) == 0 &&
          r3d_screenshot_ppm(o->shot_path, px, w, h) == 0)
        printf("view: wrote %s (%ux%u)\n", o->shot_path, w, h);
      free(px);
    }
  }

  if (a.live_up) r3d_view_live_stop(&a.live);
  free(a.rgba);
  r3d_view_packet_free(&a.pkt);
  r3d_view_manifest_free(&a.man);
  /* the ImGui Vulkan backend walks the texture list during ITS shutdown, so
   * the buffers outlive the renderer and are released only afterwards */
  ui_retire_texture(&a);
  r3d_destroy(renderer);
  for (uint32_t i = 0; i < a.n_retired; i++) ImTextureData_destroy(a.retired[i]);
  if (win) SDL_DestroyWindow(win);
  SDL_Quit();
  return rc;
}
