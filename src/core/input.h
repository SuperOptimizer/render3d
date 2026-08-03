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
  int mode_delta;    /* Tab cycles debug mode */
  int tf_delta;      /* T cycles transfer-function preset */
  float step_scale;  /* '['=×1.25 ']'=×0.8, 1.0 otherwise */
  float density_scale; /* ','=×0.8 '.'=×1.25 */
  float lod_delta;   /* '-'/'=' adjust lod bias by ∓0.25 */
  /* persistent */
  bool captured;
} r3d_input;

/* hook (may be NULL) sees every event first (e.g. GUI). allow_capture=false
 * suppresses click-to-capture (e.g. pointer over GUI). */
void r3d_input_poll(r3d_input *in, SDL_Window *win,
                    void (*hook)(void *ud, const SDL_Event *ev), void *ud, bool allow_capture);

#endif /* R3D_INPUT_H */
