#include "core/input.h"

void r3d_input_poll(r3d_input *in, SDL_Window *win,
                    void (*hook)(void *ud, const SDL_Event *ev), void *ud, bool allow_capture,
                    bool fly_mode) {
  in->quit = false;
  in->screenshot = false;
  in->resized = false;
  in->look[0] = in->look[1] = 0.0f;
  in->mode_delta = 0;
  in->tf_delta = 0;
  in->step_scale = 1.0f;
  in->density_scale = 1.0f;
  in->lod_delta = 0.0f;
  in->wheel = 0.0f;

  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    if (hook) hook(ud, &ev);
    switch (ev.type) {
    case SDL_EVENT_QUIT:
      in->quit = true;
      break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
      in->resized = true;
      break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
      if (ev.button.button != SDL_BUTTON_LEFT || !allow_capture) break;
      if (fly_mode && !in->captured) {
        SDL_SetWindowRelativeMouseMode(win, true);
        in->captured = true;
      } else if (!fly_mode && !in->dragging) {
        SDL_SetWindowRelativeMouseMode(win, true); /* endless drag, hidden cursor */
        in->dragging = true;
      }
      break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (ev.button.button == SDL_BUTTON_LEFT && in->dragging) {
        SDL_SetWindowRelativeMouseMode(win, false);
        in->dragging = false;
      }
      break;
    case SDL_EVENT_MOUSE_MOTION:
      if (in->captured || in->dragging) {
        in->look[0] += ev.motion.xrel;
        in->look[1] += ev.motion.yrel;
      }
      break;
    case SDL_EVENT_MOUSE_WHEEL:
      in->wheel += ev.wheel.y;
      break;
    case SDL_EVENT_KEY_DOWN:
      if (ev.key.repeat) break;
      switch (ev.key.key) {
      case SDLK_ESCAPE:
        if (in->captured || in->dragging) {
          SDL_SetWindowRelativeMouseMode(win, false);
          in->captured = false;
          in->dragging = false;
        } else {
          in->quit = true;
        }
        break;
      case SDLK_F12: in->screenshot = true; break;
      case SDLK_TAB: in->mode_delta = 1; break;
      case SDLK_T: in->tf_delta = 1; break;
      case SDLK_LEFTBRACKET: in->step_scale = 1.25f; break;
      case SDLK_RIGHTBRACKET: in->step_scale = 0.8f; break;
      case SDLK_COMMA: in->density_scale = 0.8f; break;
      case SDLK_PERIOD: in->density_scale = 1.25f; break;
      case SDLK_MINUS: in->lod_delta = -0.25f; break;
      case SDLK_EQUALS: in->lod_delta = 0.25f; break;
      default: break;
      }
      break;
    default:
      break;
    }
  }

  const bool *keys = SDL_GetKeyboardState(NULL);
  in->move[0] = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
  in->move[1] = (keys[SDL_SCANCODE_E] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_Q] ? 1.0f : 0.0f);
  in->move[2] = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
  in->fast = keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT];
}
