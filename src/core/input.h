/* SDL3 events -> renderer actions. Click captures the mouse for look; Esc
 * releases the capture, or quits when not captured. */
#ifndef R3D_INPUT_H
#define R3D_INPUT_H

#include <SDL3/SDL.h>
#include <stdbool.h>

typedef struct r3d_input {
  /* per-frame outputs */
  bool quit;
  bool screenshot;   /* F12 pressed this frame */
  bool resized;      /* pixel size changed this frame */
  float move[3];     /* right, up, forward in -1..1 (WASD + Q/E) */
  float look[2];     /* accumulated mouse delta this frame (pixels) */
  bool fast;         /* shift held */
  bool ctrl;         /* ctrl held (volume-transform drag modifier) */
  int mode_delta;    /* Tab cycles debug mode */
  int tf_delta;      /* T cycles transfer-function preset */
  float step_scale;  /* '['=×1.25 ']'=×0.8, 1.0 otherwise */
  float density_scale; /* ','=×0.8 '.'=×1.25 */
  float lod_delta;   /* '-'/'=' adjust lod bias by ∓0.25 */
  int zdelta;        /* slab window scroll in slices: R/F = ±1 (repeats) */
  int zpage;         /* slab window scroll in pages: PgUp/PgDn = ±1 */
  int adelta[2];     /* arrow keys: x (left/right), y (up/down); repeats */
  float wheel;       /* scroll this frame (+away from user) */
  bool wheel_shift;  /* Shift held during any wheel event this frame */
  bool view_toggle;  /* Space pressed (multiview: solo/restore hovered view) */
  bool umb_place;    /* U pressed (umbilicus edit: place point at cursor) */
  bool undo;         /* Ctrl+Z pressed */
  bool redo;         /* Ctrl+Shift+Z pressed */
  bool annotate_click; /* uncaptured LMB press in annotation mode */
  bool click_ctrl;     /* Ctrl was held for annotate_click */
  float click_xy[2];   /* window pixel coordinates of annotate_click */
  float mouse_xy[2];   /* current pointer position (window pixels) */
  /* persistent */
  bool captured;     /* fly mode: pointer grabbed until Esc */
  bool dragging;     /* orbit mode: LMB held */
} r3d_input;

/* hook (may be NULL) sees every event first (e.g. GUI). allow_capture=false
 * suppresses click-to-capture/drag (e.g. pointer over GUI). fly_mode picks
 * click semantics: capture-and-fly vs hold-drag-to-rotate. In annotation
 * mode, plain LMB produces annotate_click while Shift+LMB still pans. In
 * multiview mode, plain LMB drags (pan) and Ctrl+LMB produces
 * annotate_click (focus gesture; click_ctrl is set). */
void r3d_input_poll(r3d_input *in, SDL_Window *win,
                    void (*hook)(void *ud, const SDL_Event *ev), void *ud, bool allow_capture,
                    bool fly_mode, bool annotation_mode, bool multiview_mode);

#endif /* R3D_INPUT_H */
