/* `--view` model-view mode: a self-contained window over one TSM view packet
 * at a time (core/view.h holds every byte of state and all the compositing;
 * this layer is only input, texture hand-off and the panel), plus the
 * `--view-shot` headless entry point the tests drive. */
#ifndef R3D_VIEWUI_H
#define R3D_VIEWUI_H

#include <stdbool.h>
#include <stdint.h>

typedef struct r3d_viewui_opts {
  const char *path;   /* view.json, its directory, or one packet directory */
  const char *serve;  /* "host:port" for tsm serve, or NULL */
  int win_w, win_h;
  bool headless;      /* offscreen renderer, no window (CI/verification) */
  bool no_vsync;
  uint32_t exit_frames; /* 0 = run until quit */
  const char *shot_path; /* PPM of the final frame (automation) */
} r3d_viewui_opts;

int r3d_viewui_run(const r3d_viewui_opts *o);

#endif /* R3D_VIEWUI_H */
