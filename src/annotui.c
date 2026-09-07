#include "annotui.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "cimgui.h"
#include "core/annot.h"
#include "core/camera.h"
#include "core/screenshot.h"
#include "core/transfer.h"
#include "core/volume.h"
#include "render/render.h"

/* ===================== headless CLI ====================================== */

static int resolve_packet(const char *path, char *out, size_t cap) {
  r3d_annot_manifest m = {0};
  if (r3d_annot_manifest_load(&m, path) != 0) {
    fprintf(stderr, "annot: %s is not a packet or a packet.json manifest\n", path);
    return -1;
  }
  snprintf(out, cap, "%s", m.ent[0].path);
  r3d_annot_manifest_free(&m);
  return 0;
}

int r3d_annot_apply_cli(const char *packet_path, const char *strokes_path) {
  char dir[900];
  if (resolve_packet(packet_path, dir, sizeof dir) != 0) return -1;
  r3d_annot_packet p = {0};
  if (r3d_annot_packet_load(&p, dir) != 0) return -1;
  r3d_annot_stroke *s = NULL;
  uint32_t n = 0;
  if (r3d_annot_strokes_load(strokes_path, &s, &n) != 0) {
    fprintf(stderr, "annot: cannot read stroke script %s\n", strokes_path);
    r3d_annot_packet_free(&p);
    return -1;
  }
  r3d_annot_undo *u = r3d_annot_undo_create(64);
  int rc = r3d_annot_run_strokes(&p, s, n, u);
  if (rc == 0) rc = r3d_annot_correction_save(&p);
  if (rc == 0) {
    printf("annot: %s %ux%ux%u  %u stroke%s ->", dir, p.nz, p.ny, p.nx, n, n == 1 ? "" : "s");
    for (int c = 1; c < R3D_ANNOT_NCLASS; c++)
      printf(" %s=%llu", r3d_annot_class_name[c], (unsigned long long)p.counts[c]);
    printf("\n");
  } else {
    fprintf(stderr, "annot: applying %s to %s failed\n", strokes_path, dir);
  }
  r3d_annot_undo_destroy(u);
  r3d_annot_strokes_free(s, n);
  r3d_annot_packet_free(&p);
  return rc;
}

/* ===================== interactive mode ================================== */

typedef struct annot_ui {
  r3d_annot_manifest man;
  uint32_t index;
  r3d_annot_packet pkt;
  r3d_annot_undo *undo;
  r3d_annot_view view;
  uint8_t lut[256][4];
  bool use_lut;

  uint8_t *rgba;       /* nx*ny*4 composite of the current slice */
  ImTextureData *tex;
  /* A texture the backend has seen may not be freed until it has destroyed
   * its own GPU resources, which only happens inside a later draw (or the
   * backend's shutdown) — so a resized packet retires the old one here. */
  ImTextureData *retired[8];
  uint32_t n_retired;
  uint32_t tex_w, tex_h;
  bool image_stale;

  int32_t z;
  int cls;             /* 1..4; 0 = pan tool */
  int radius;
  int depth_k;
  bool depth_on;
  float pan[2], zoom;
  bool fit_pending;

  bool stroking;
  int32_t last_xy[2];
  bool have_anchor;
  int32_t anchor[2];
  char status[256];
} annot_ui;

__attribute__((format(printf, 2, 3))) static void ui_status(annot_ui *a, const char *fmt,
                                                          ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(a->status, sizeof a->status, fmt, ap);
  va_end(ap);
}

static void ui_retire_texture(annot_ui *a) {
  if (!a->tex) return;
  a->tex->WantDestroyNextFrame = true; /* ImGui -> WantDestroy -> the backend */
  if (a->n_retired < 8u) a->retired[a->n_retired++] = a->tex;
  a->tex = NULL;
}

/* Free the retired textures the backend has finished with. */
static void ui_reap_textures(annot_ui *a) {
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

static bool ui_make_texture(annot_ui *a) {
  ui_retire_texture(a);
  a->tex = ImTextureData_ImTextureData();
  if (!a->tex) return false;
  ImTextureData_Create(a->tex, ImTextureFormat_RGBA32, (int)a->pkt.nx, (int)a->pkt.ny);
  a->tex->UseColors = true;
  ImTextureData_SetStatus(a->tex, ImTextureStatus_WantCreate);
  igRegisterUserTexture(a->tex);
  a->tex_w = a->pkt.nx;
  a->tex_h = a->pkt.ny;
  return true;
}

static void ui_refresh_image(annot_ui *a) {
  if (!a->tex || !a->rgba) return;
  r3d_annot_composite(&a->pkt, (uint32_t)a->z, &a->view, a->rgba);
  size_t bytes = (size_t)a->pkt.nx * a->pkt.ny * 4u;
  memcpy(ImTextureData_GetPixels(a->tex), a->rgba, bytes);
  if (a->tex->Status != ImTextureStatus_WantCreate) {
    a->tex->UpdateRect = (ImTextureRect){0, 0, (unsigned short)a->pkt.nx,
                                         (unsigned short)a->pkt.ny};
    ImTextureData_SetStatus(a->tex, ImTextureStatus_WantUpdates);
  }
  a->image_stale = false;
}

static void ui_save(annot_ui *a) {
  if (!a->pkt.dirty) return;
  if (r3d_annot_correction_save(&a->pkt) != 0)
    ui_status(a, "SAVE FAILED for %s", a->pkt.dir);
}

static int ui_open(annot_ui *a, r3d_renderer *renderer, uint32_t index) {
  if (index >= a->man.count) return -1;
  if (a->pkt.correction) {
    ui_save(a);
    r3d_annot_packet_free(&a->pkt);
  }
  if (r3d_annot_packet_load(&a->pkt, a->man.ent[index].path) != 0) return -1;
  a->index = index;
  free(a->rgba);
  a->rgba = malloc((size_t)a->pkt.nx * a->pkt.ny * 4u);
  if (!a->rgba) return -1;
  if (a->tex_w != a->pkt.nx || a->tex_h != a->pkt.ny || !a->tex)
    if (!ui_make_texture(a)) return -1;
  a->z = (int32_t)(a->pkt.nz / 2u);
  a->fit_pending = true;
  a->image_stale = true;
  while (r3d_annot_undo_pop(&a->pkt, a->undo) > 0) {} /* journal is per packet */
  if (renderer) {
    r3d_volume_desc d = {.nx = a->pkt.nx, .ny = a->pkt.ny, .nz = a->pkt.nz,
                         .brick_dim = R3D_BRICK_DIM, .voxel_um = (float)a->pkt.voxel_um};
    (void)r3d_upload_volume(renderer, &d, a->pkt.layer[R3D_ANNOT_L_CT]);
  }
  ui_status(a, "packet %u/%u  %s  origin z%lld y%lld x%lld", index + 1u, a->man.count,
            a->man.ent[index].name, (long long)a->pkt.origin[0], (long long)a->pkt.origin[1],
            (long long)a->pkt.origin[2]);
  return 0;
}

static void ui_set_z(annot_ui *a, int64_t z) {
  if (z < 0) z = 0;
  if (z >= (int64_t)a->pkt.nz) z = (int64_t)a->pkt.nz - 1;
  if ((int32_t)z == a->z) return;
  a->z = (int32_t)z;
  a->image_stale = true;
}

/* One brush/line application, folded into the currently open undo group. */
static void ui_paint(annot_ui *a, r3d_annot_op op, uint8_t cls, const int32_t *pts,
                     uint32_t npts) {
  r3d_annot_stroke s = {.op = op,
                        .cls = cls,
                        .radius = a->radius,
                        .depth = a->depth_on ? a->depth_k : 0,
                        .z = a->z,
                        .pts = pts,
                        .npts = npts};
  if (r3d_annot_apply(&a->pkt, &s, a->undo) > 0) a->image_stale = true;
}

static void ui_panel(annot_ui *a, r3d_renderer *renderer, bool *quit) {
  igText("packet %u / %u", a->index + 1u, a->man.count);
  igText("%s", a->man.ent[a->index].name);
  igText("origin z%lld y%lld x%lld", (long long)a->pkt.origin[0],
         (long long)a->pkt.origin[1], (long long)a->pkt.origin[2]);
  igText("dims %u x %u x %u (z,y,x)", a->pkt.nz, a->pkt.ny, a->pkt.nx);
  if (a->pkt.voxel_um > 0.0) igText("%.3g um/voxel", a->pkt.voxel_um);
  igSeparator();

  int z = a->z;
  if (igSliderInt("slice z", &z, 0, (int)a->pkt.nz - 1, "%d", 0)) ui_set_z(a, z);
  igText("R/F +-1   PgUp/PgDn +-16   N/P packet");
  igSeparator();

  igText("class (1-4, 0 = pan)");
  for (int c = 0; c < R3D_ANNOT_NCLASS; c++) {
    char label[64];
    snprintf(label, sizeof label, "%d %s", c, c == 0 ? "pan / no paint" : r3d_annot_class_name[c]);
    if (igRadioButton_IntPtr(label, &a->cls, c) && c == 0) a->have_anchor = false;
  }
  igSliderInt("brush radius", &a->radius, 0, 64, "%d vox  ([ / ])", 0);
  igCheckbox("depth mode (D)", &a->depth_on);
  igSliderInt("depth +-k", &a->depth_k, 1, 32, "%d slices  (, / .)", 0);
  igSeparator();

  igText("layers");
  igCheckboxFlags_UintPtr("faces_in / faces_out", &a->view.show, R3D_ANNOT_SHOW_FACES);
  igCheckboxFlags_UintPtr("ignore", &a->view.show, R3D_ANNOT_SHOW_IGNORE);
  igCheckboxFlags_UintPtr("rv_class bands", &a->view.show, R3D_ANNOT_SHOW_RV);
  igCheckboxFlags_UintPtr("pred_in / pred_out", &a->view.show, R3D_ANNOT_SHOW_PRED);
  igCheckboxFlags_UintPtr("correction", &a->view.show, R3D_ANNOT_SHOW_CORRECTION);
  igSliderFloat("CT gain", &a->view.ct_gain, 0.25f, 4.0f, "%.2f", 0);
  igSliderFloat("label alpha", &a->view.label_alpha, 0.0f, 1.0f, "%.2f", 0);
  igSliderFloat("correction alpha", &a->view.corr_alpha, 0.1f, 1.0f, "%.2f", 0);
  igSeparator();

  igText("painted voxels");
  for (int c = 1; c < R3D_ANNOT_NCLASS; c++)
    igText("  %-9s %llu", r3d_annot_class_name[c], (unsigned long long)a->pkt.counts[c]);
  igText("undo levels: %u", r3d_annot_undo_depth(a->undo));
  igText("correction: %s", a->pkt.dirty ? "UNSAVED" : "saved");
  igSeparator();

  bool done = a->pkt.done;
  if (igCheckbox("mark packet done", &done)) {
    ui_save(a);
    if (r3d_annot_set_done(&a->pkt, done) != 0) ui_status(a, "cannot write meta.json");
    else ui_status(a, "meta.json done=%s", done ? "true" : "false");
  }
  if (igButton("prev packet (P)", (ImVec2){140, 0}) && a->index > 0)
    (void)ui_open(a, renderer, a->index - 1u);
  igSameLine(0, 8);
  if (igButton("next packet (N)", (ImVec2){140, 0}) && a->index + 1u < a->man.count)
    (void)ui_open(a, renderer, a->index + 1u);
  if (igButton("quit (saves)", (ImVec2){140, 0})) *quit = true;
  igSeparator();
  igText("%s", a->status);
}

static void ui_canvas(annot_ui *a) {
  ImVec2 p0 = igGetCursorScreenPos();
  ImVec2 avail = igGetContentRegionAvail();
  if (avail.x < 16.0f) avail.x = 16.0f;
  if (avail.y < 16.0f) avail.y = 16.0f;
  if (a->fit_pending) {
    float zx = avail.x / (float)a->pkt.nx, zy = avail.y / (float)a->pkt.ny;
    a->zoom = zx < zy ? zx : zy;
    if (a->zoom <= 0.0f) a->zoom = 1.0f;
    a->pan[0] = (avail.x - (float)a->pkt.nx * a->zoom) * 0.5f;
    a->pan[1] = (avail.y - (float)a->pkt.ny * a->zoom) * 0.5f;
    a->fit_pending = false;
  }
  igInvisibleButton("##annot-canvas", avail, ImGuiButtonFlags_MouseButtonLeft |
                                                 ImGuiButtonFlags_MouseButtonRight |
                                                 ImGuiButtonFlags_MouseButtonMiddle);
  bool hovered = igIsItemHovered(0);
  ImDrawList *dl = igGetWindowDrawList();
  ImVec2 img0 = {p0.x + a->pan[0], p0.y + a->pan[1]};
  ImVec2 img1 = {img0.x + (float)a->pkt.nx * a->zoom, img0.y + (float)a->pkt.ny * a->zoom};
  ImDrawList_PushClipRect(dl, p0, (ImVec2){p0.x + avail.x, p0.y + avail.y}, true);
  ImDrawList_AddRectFilled(dl, p0, (ImVec2){p0.x + avail.x, p0.y + avail.y}, 0xff101010u, 0.0f, 0);
  if (a->tex)
    ImDrawList_AddImage(dl, ImTextureData_GetTexRef(a->tex), img0, img1, (ImVec2){0, 0},
                        (ImVec2){1, 1}, 0xffffffffu);
  ImDrawList_AddRect(dl, img0, img1, 0xff606060u, 0.0f, 1.0f, 0);

  ImGuiIO *io = igGetIO_Nil();
  ImVec2 mp = igGetMousePos();
  double vx = ((double)mp.x - (double)img0.x) / (double)a->zoom;
  double vy = ((double)mp.y - (double)img0.y) / (double)a->zoom;
  /* ImGui reports +-FLT_MAX when there is no mouse: clamp before narrowing */
  if (!(vx > -1e6) || !(vx < 1e6)) vx = -1e6;
  if (!(vy > -1e6) || !(vy < 1e6)) vy = -1e6;
  int32_t px = (int32_t)floor(vx), py = (int32_t)floor(vy);
  bool inside = hovered && vx >= 0.0 && vy >= 0.0 && px < (int32_t)a->pkt.nx &&
                py < (int32_t)a->pkt.ny;

  if (hovered && io->MouseWheel != 0.0f) {
    if (io->KeyShift) {
      ui_set_z(a, (int64_t)a->z + (int64_t)(io->MouseWheel > 0.0f ? 1 : -1));
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
  /* pan: middle drag always, left drag when the pan tool is selected */
  bool pan_drag = igIsMouseDragging(ImGuiMouseButton_Middle, -1.0f) ||
                  (a->cls == 0 && igIsMouseDragging(ImGuiMouseButton_Left, -1.0f));
  if (hovered && pan_drag) {
    a->pan[0] += io->MouseDelta.x;
    a->pan[1] += io->MouseDelta.y;
  }

  if (a->cls > 0 && inside) {
    int32_t pt[2] = {px, py};
    if (io->KeyShift && igIsMouseClicked_Bool(ImGuiMouseButton_Left, false)) {
      if (!a->have_anchor) {
        a->anchor[0] = px;
        a->anchor[1] = py;
        a->have_anchor = true;
        ui_status(a, "fill line: anchor (%d,%d) - shift-click the far end", px, py);
      } else {
        int32_t seg[4] = {a->anchor[0], a->anchor[1], px, py};
        r3d_annot_undo_begin(a->undo);
        ui_paint(a, R3D_ANNOT_OP_LINE, (uint8_t)a->cls, seg, 2);
        r3d_annot_undo_end(a->undo);
        a->have_anchor = false;
        ui_save(a);
        ui_status(a, "fill line -> (%d,%d)", px, py);
      }
    } else if (!io->KeyShift &&
               (igIsMouseClicked_Bool(ImGuiMouseButton_Left, false) ||
                igIsMouseClicked_Bool(ImGuiMouseButton_Right, false))) {
      a->stroking = true;
      r3d_annot_undo_begin(a->undo);
      uint8_t cls = igIsMouseDown_Nil(ImGuiMouseButton_Right) ? (uint8_t)R3D_ANNOT_UNTOUCHED
                                                             : (uint8_t)a->cls;
      ui_paint(a, R3D_ANNOT_OP_BRUSH, cls, pt, 1);
      a->last_xy[0] = px;
      a->last_xy[1] = py;
    } else if (a->stroking && (igIsMouseDown_Nil(ImGuiMouseButton_Left) ||
                               igIsMouseDown_Nil(ImGuiMouseButton_Right))) {
      if (px != a->last_xy[0] || py != a->last_xy[1]) {
        int32_t seg[4] = {a->last_xy[0], a->last_xy[1], px, py};
        uint8_t cls = igIsMouseDown_Nil(ImGuiMouseButton_Right) ? (uint8_t)R3D_ANNOT_UNTOUCHED
                                                                : (uint8_t)a->cls;
        ui_paint(a, R3D_ANNOT_OP_BRUSH, cls, seg, 2);
        a->last_xy[0] = px;
        a->last_xy[1] = py;
      }
    }
    /* brush footprint */
    float r = ((float)a->radius + 0.5f) * a->zoom;
    ImVec2 c = {img0.x + ((float)px + 0.5f) * a->zoom, img0.y + ((float)py + 0.5f) * a->zoom};
    ImDrawList_AddCircle(dl, c, r < 2.0f ? 2.0f : r, 0xffffffffu, 0, 1.5f);
    if (a->have_anchor) {
      ImVec2 a0 = {img0.x + ((float)a->anchor[0] + 0.5f) * a->zoom,
                   img0.y + ((float)a->anchor[1] + 0.5f) * a->zoom};
      ImDrawList_AddLine(dl, a0, c, 0xff40ffffu, 1.5f);
    }
  }
  if (a->stroking && !igIsMouseDown_Nil(ImGuiMouseButton_Left) &&
      !igIsMouseDown_Nil(ImGuiMouseButton_Right)) {
    a->stroking = false;
    r3d_annot_undo_end(a->undo);
    ui_save(a); /* never lose a stroke: correction.u8 lands on release */
  }
  ImDrawList_PopClipRect(dl);

  if (inside) {
    size_t vi = ((size_t)(uint32_t)a->z * a->pkt.ny + (uint32_t)py) * a->pkt.nx + (uint32_t)px;
    const uint8_t *srcl = a->pkt.layer[R3D_ANNOT_L_SOURCE];
    const uint8_t *rvl = a->pkt.layer[R3D_ANNOT_L_RV_CLASS];
    igText("voxel x %d  y %d  z %d   scroll z %lld  y %lld  x %lld   ct %u   corr %s"
           "   label %s   source %s   rv %s",
           px, py, a->z, (long long)(a->pkt.origin[0] + a->z),
           (long long)(a->pkt.origin[1] + py), (long long)(a->pkt.origin[2] + px),
           a->pkt.layer[R3D_ANNOT_L_CT][vi], r3d_annot_class_name[a->pkt.correction[vi]],
           a->pkt.layer[R3D_ANNOT_L_FACES_IN] && a->pkt.layer[R3D_ANNOT_L_FACES_IN][vi]
               ? "in"
               : a->pkt.layer[R3D_ANNOT_L_FACES_OUT] && a->pkt.layer[R3D_ANNOT_L_FACES_OUT][vi]
                     ? "out"
                     : a->pkt.layer[R3D_ANNOT_L_IGNORE] && a->pkt.layer[R3D_ANNOT_L_IGNORE][vi]
                           ? "ignore"
                           : "-",
           srcl && srcl[vi] < 4u ? r3d_annot_source_name[srcl[vi]] : "-",
           rvl && rvl[vi] < 4u ? r3d_annot_rv_name[rvl[vi]] : "-");
  } else
    igText("zoom %.2f px/voxel   wheel zoom, shift+wheel slice, middle-drag pan", (double)a->zoom);
}

static void ui_keys(annot_ui *a, r3d_renderer *renderer, bool *quit) {
  ImGuiIO *io = igGetIO_Nil();
  if (io->WantCaptureKeyboard) return;
  if (igIsKeyPressed_Bool(ImGuiKey_R, true)) ui_set_z(a, (int64_t)a->z + 1);
  if (igIsKeyPressed_Bool(ImGuiKey_F, true)) ui_set_z(a, (int64_t)a->z - 1);
  if (igIsKeyPressed_Bool(ImGuiKey_PageUp, true)) ui_set_z(a, (int64_t)a->z + 16);
  if (igIsKeyPressed_Bool(ImGuiKey_PageDown, true)) ui_set_z(a, (int64_t)a->z - 16);
  for (int c = 0; c < R3D_ANNOT_NCLASS; c++)
    if (igIsKeyPressed_Bool((ImGuiKey)(ImGuiKey_0 + c), false)) {
      a->cls = c;
      a->have_anchor = false;
    }
  if (igIsKeyPressed_Bool(ImGuiKey_Escape, false)) {
    a->cls = 0;
    a->have_anchor = false;
  }
  if (igIsKeyPressed_Bool(ImGuiKey_LeftBracket, true) && a->radius > 0) a->radius--;
  if (igIsKeyPressed_Bool(ImGuiKey_RightBracket, true) && a->radius < 64) a->radius++;
  if (igIsKeyPressed_Bool(ImGuiKey_D, false)) a->depth_on = !a->depth_on;
  if (igIsKeyPressed_Bool(ImGuiKey_Comma, true) && a->depth_k > 1) a->depth_k--;
  if (igIsKeyPressed_Bool(ImGuiKey_Period, true) && a->depth_k < 32) a->depth_k++;
  if (io->KeyCtrl && igIsKeyPressed_Bool(ImGuiKey_Z, true)) {
    int64_t n = r3d_annot_undo_pop(&a->pkt, a->undo);
    if (n > 0) {
      a->image_stale = true;
      ui_save(a);
      ui_status(a, "undo: %lld voxels (%u left)", (long long)n, r3d_annot_undo_depth(a->undo));
    } else {
      ui_status(a, "nothing to undo");
    }
  }
  if (io->KeyCtrl && igIsKeyPressed_Bool(ImGuiKey_S, false)) {
    a->pkt.dirty = true;
    ui_save(a);
    ui_status(a, "saved %s/correction.u8", a->pkt.dir);
  }
  if (igIsKeyPressed_Bool(ImGuiKey_N, false) && a->index + 1u < a->man.count)
    (void)ui_open(a, renderer, a->index + 1u);
  if (igIsKeyPressed_Bool(ImGuiKey_P, false) && a->index > 0)
    (void)ui_open(a, renderer, a->index - 1u);
  if (igIsKeyPressed_Bool(ImGuiKey_Q, false) && io->KeyCtrl) *quit = true;
}

int r3d_annot_run(const r3d_annot_opts *o) {
  annot_ui a = {0};
  if (r3d_annot_manifest_load(&a.man, o->path) != 0) {
    fprintf(stderr, "annot: %s is not a packet or a packet.json manifest\n", o->path);
    return EXIT_FAILURE;
  }
  a.undo = r3d_annot_undo_create(64);
  if (!a.undo) {
    r3d_annot_manifest_free(&a.man);
    return EXIT_FAILURE;
  }
  r3d_annot_view_default(&a.view);
  a.cls = R3D_ANNOT_IN;
  a.radius = 2;
  a.depth_k = 2;
  a.zoom = 1.0f;
  if (o->tf_preset >= 0) {
    r3d_tf tf;
    if (r3d_tf_preset((uint32_t)o->tf_preset, &tf) == 0) {
      r3d_tf_build(&tf, a.lut);
      a.view.lut = (const uint8_t (*)[4])a.lut;
      a.use_lut = true;
    }
  }

  if (!SDL_Init(o->headless ? SDL_INIT_EVENTS : SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }
  SDL_Window *win = NULL;
  if (!o->headless) {
    win = SDL_CreateWindow("render3d - faces annotation", o->win_w, o->win_h,
                           SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);
    if (!win) {
      fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
      SDL_Quit();
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
    return EXIT_FAILURE;
  }
  if (a.use_lut) r3d_set_transfer(renderer, a.lut);

  int rc = EXIT_SUCCESS;
  bool quit = false;
  if (ui_open(&a, renderer, 0) != 0) {
    fprintf(stderr, "annot: cannot open %s\n", a.man.ent[0].path);
    quit = true;
    rc = EXIT_FAILURE;
  }
  printf("annot: %u packet%s from %s\n", a.man.count, a.man.count == 1 ? "" : "s",
         a.man.path[0] ? a.man.path : a.man.ent[0].path);

  /* test hook: step to the next packet every N frames so an automated run
   * covers packet switching, the save-on-switch and the texture resize */
  const char *cyc = getenv("R3D_ANNOT_CYCLE");
  uint32_t cycle = cyc ? (uint32_t)strtoul(cyc, NULL, 10) : 0u;

  r3d_camera cam;
  r3d_camera_init(&cam, (r3d_v3){0.5f, 0.5f, -1.6f});
  uint32_t frame = 0;
  while (!quit) {
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      r3d_gui_event(renderer, &ev);
      if (ev.type == SDL_EVENT_QUIT) quit = true;
      if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) quit = true;
    }
    if (quit) break;
    if (r3d_gui_begin(renderer) != 0) break;
    ui_reap_textures(&a);

    ui_keys(&a, renderer, &quit);
    ImGuiViewport *vp = igGetMainViewport();
    const float panel_w = 340.0f;
    igSetNextWindowPos(vp->WorkPos, ImGuiCond_Always, (ImVec2){0, 0});
    igSetNextWindowSize((ImVec2){panel_w, vp->WorkSize.y}, ImGuiCond_Always);
    if (igBegin("faces annotation", NULL,
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

    /* secondary 3D view: the packet CT as an ordinary volume, behind the
     * panes. It costs one upload per packet and no per-frame work here. */
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
    r3d_frame_stats st;
    if (r3d_frame(renderer, &fp, &st) < 0) {
      fprintf(stderr, "annot: frame failed\n");
      rc = EXIT_FAILURE;
      break;
    }
    frame++;
    if (cycle && frame % cycle == 0)
      (void)ui_open(&a, renderer, (a.index + 1u) % a.man.count);
    if (o->exit_frames && frame >= o->exit_frames) quit = true;
  }

  if (o->shot_path) { /* headless verification: the composited panes as PPM */
    uint32_t w = 0, h = 0;
    if (r3d_read_frame(renderer, NULL, &w, &h) == 0) {
      uint8_t *px = malloc((size_t)w * h * 4u);
      if (px && r3d_read_frame(renderer, px, &w, &h) == 0 &&
          r3d_screenshot_ppm(o->shot_path, px, w, h) == 0)
        printf("annot: wrote %s (%ux%u)\n", o->shot_path, w, h);
      free(px);
    }
  }

  ui_save(&a);
  free(a.rgba);
  r3d_annot_packet_free(&a.pkt);
  r3d_annot_undo_destroy(a.undo);
  r3d_annot_manifest_free(&a.man);
  /* the ImGui Vulkan backend walks the texture list during ITS shutdown, so
   * the buffers outlive the renderer and are released only afterwards */
  ui_retire_texture(&a);
  r3d_destroy(renderer);
  for (uint32_t i = 0; i < a.n_retired; i++) ImTextureData_destroy(a.retired[i]);
  if (win) SDL_DestroyWindow(win);
  SDL_Quit();
  return rc;
}
