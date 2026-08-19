#include "core/input.h"

void r3d_input_poll(r3d_input *in, SDL_Window *win,
                    void (*hook)(void *ud, const SDL_Event *ev), void *ud, bool allow_capture,
                    bool fly_mode, bool annotation_mode, bool multiview_mode) {
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
  in->wheel_shift = false;
  in->zdelta = 0;
  in->zpage = 0;
  in->adelta[0] = in->adelta[1] = 0;
  in->view_toggle = false;
  in->umb_place = false;
  in->anchor_place = false;
  in->seed_place = false;
  in->surf_place = false;
  in->bnd_place = false;
  in->undo = false;
  in->redo = false;
  in->annotate_click = false;
  in->click_ctrl = false;

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
      if (ev.button.button == SDL_BUTTON_LEFT) in->lmb_held = true;
      if (!allow_capture) break;
      if (multiview_mode && ev.button.button == SDL_BUTTON_RIGHT) {
        /* multiview pans with the RIGHT button. No relative-mouse (pointer
         * lock) here: a pan is bounded, and under WSLg/Wayland the warp-based
         * relative mode reports huge bogus deltas that threw every pane off
         * the volume. SDL still delivers xrel/yrel per motion event. */
        in->dragging = true;
        break;
      }
      if (ev.button.button != SDL_BUTTON_LEFT) break;
      if (multiview_mode && (SDL_GetModState() & SDL_KMOD_CTRL)) {
        in->annotate_click = true; /* focus gesture; plain LMB stays a drag */
        in->click_ctrl = true;
        in->click_xy[0] = ev.button.x;
        in->click_xy[1] = ev.button.y;
        break;
      }
      if (annotation_mode && !(SDL_GetModState() & SDL_KMOD_SHIFT)) {
        in->annotate_click = true;
        in->click_ctrl = (SDL_GetModState() & SDL_KMOD_CTRL) != 0;
        in->click_xy[0] = ev.button.x;
        in->click_xy[1] = ev.button.y;
        break;
      }
      if (fly_mode && !in->captured) {
        SDL_SetWindowRelativeMouseMode(win, true);
        in->captured = true;
      } else if (!fly_mode && !multiview_mode && !in->dragging) {
        SDL_SetWindowRelativeMouseMode(win, true); /* endless drag, hidden cursor */
        in->dragging = true;
      }
      break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (ev.button.button == SDL_BUTTON_LEFT) in->lmb_held = false;
      if ((ev.button.button == SDL_BUTTON_LEFT ||
           (multiview_mode && ev.button.button == SDL_BUTTON_RIGHT)) &&
          in->dragging) {
        if (!multiview_mode) SDL_SetWindowRelativeMouseMode(win, false);
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
      if (SDL_GetModState() & SDL_KMOD_SHIFT) in->wheel_shift = true;
      break;
    case SDL_EVENT_KEY_DOWN:
      /* z-scroll keys repeat on hold */
      if (ev.key.key == SDLK_R) in->zdelta += 1;
      if (ev.key.key == SDLK_F) in->zdelta -= 1;
      if (ev.key.key == SDLK_PAGEUP) in->zpage += 1;
      if (ev.key.key == SDLK_PAGEDOWN) in->zpage -= 1;
      if (ev.key.key == SDLK_LEFT) in->adelta[0] -= 1;
      if (ev.key.key == SDLK_RIGHT) in->adelta[0] += 1;
      if (ev.key.key == SDLK_UP) in->adelta[1] -= 1;
      if (ev.key.key == SDLK_DOWN) in->adelta[1] += 1;
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
      case SDLK_SPACE: in->view_toggle = true; break;
      case SDLK_U: in->umb_place = true; break;
      case SDLK_X: in->anchor_place = true; break;
      case SDLK_G: in->seed_place = true; break;
      case SDLK_P: in->surf_place = true; break;
      case SDLK_B: in->bnd_place = true; break;
      case SDLK_Z:
        if (SDL_GetModState() & SDL_KMOD_CTRL) {
          if (SDL_GetModState() & SDL_KMOD_SHIFT) in->redo = true;
          else in->undo = true;
        }
        break;
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

  SDL_GetMouseState(&in->mouse_xy[0], &in->mouse_xy[1]);

  const bool *keys = SDL_GetKeyboardState(NULL);
  in->move[0] = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
  in->move[1] = (keys[SDL_SCANCODE_E] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_Q] ? 1.0f : 0.0f);
  in->move[2] = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
  in->fast = keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT];
  in->ctrl = keys[SDL_SCANCODE_LCTRL] || keys[SDL_SCANCODE_RCTRL];
}
