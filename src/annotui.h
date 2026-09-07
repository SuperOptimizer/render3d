/* `--annot` faces-annotation mode: a self-contained window + renderer over
 * one TSM annotation packet at a time (core/annot.h holds all the state and
 * geometry; this layer is only input, compositing hand-off and the panel),
 * and the `--annot-apply` headless CLI the tests drive. */
#ifndef R3D_ANNOTUI_H
#define R3D_ANNOTUI_H

#include <stdbool.h>
#include <stdint.h>

typedef struct r3d_annot_opts {
  const char *path;     /* packet.json, its directory, or one packet dir */
  int tf_preset;        /* -1 = plain grey ramp */
  int win_w, win_h;
  bool headless;        /* offscreen renderer, no window (CI/verification) */
  bool no_vsync;
  uint32_t exit_frames; /* 0 = run until quit */
  const char *shot_path;
} r3d_annot_opts;

int r3d_annot_run(const r3d_annot_opts *o);

/* Load a packet, apply a stroke script, write correction.u8. No GPU. */
int r3d_annot_apply_cli(const char *packet_path, const char *strokes_path);

#endif /* R3D_ANNOTUI_H */
