/* render3d — volumetric renderer for Vesuvius Challenge micro-CT volumes.
 * M1: SDL3 window + Vulkan compute raycaster (see spec/ and docs/measured.md). */
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/stats.h"
#include "render/render.h"
#include "vk/vkctx.h"

#ifndef R3D_SPV_DIR
#define R3D_SPV_DIR "spv" /* release fallback: exe-relative */
#endif

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--probe") == 0) {
    r3d_vkctx vk;
    if (r3d_vkctx_create(&vk, NULL, 0, false) != 0) return EXIT_FAILURE;
    r3d_vkctx_print_caps(&vk);
    r3d_vkctx_destroy(&vk);
    return EXIT_SUCCESS;
  }

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }

  SDL_Window *win = SDL_CreateWindow("render3d", 1280, 720,
                                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);
  if (!win) {
    fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
    SDL_Quit();
    return EXIT_FAILURE;
  }

  r3d_config cfg = {.validate = false, .vsync = true, .spv_dir = R3D_SPV_DIR};
  r3d_renderer *renderer = NULL;
  if (r3d_create(win, &cfg, &renderer) != 0) {
    fprintf(stderr, "renderer init failed\n");
    SDL_DestroyWindow(win);
    SDL_Quit();
    return EXIT_FAILURE;
  }

  r3d_stats stats;
  r3d_stats_init(&stats);
  uint32_t frame_index = 0;

  bool running = true;
  while (running) {
    uint64_t t0 = r3d_now_ns();
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_EVENT_QUIT) running = false;
      if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_ESCAPE) running = false;
      if (ev.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) r3d_resize(renderer);
    }

    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(win, &w, &h);
    r3d_frame_params p = {
        .cam_origin = {0.5f, 0.5f, -1.5f},
        .cam_right = {1, 0, 0},
        .cam_up = {0, 1, 0},
        .cam_forward = {0, 0, 1},
        .step_voxels = 1.0f,
        .density = 1.0f,
        .max_mip = 10.0f,
        .viewport = {(uint32_t)w, (uint32_t)h},
        .mode = R3D_MODE_RAYDIR,
        .frame_index = frame_index++,
    };
    r3d_frame_stats st = {0};
    int rc = r3d_frame(renderer, &p, &st);
    if (rc < 0) {
      fprintf(stderr, "r3d_frame failed\n");
      running = false;
    }
    r3d_stats_push(&stats, r3d_now_ns() - t0, st.gpu_ns);
    r3d_stats_report(&stats);
  }

  r3d_destroy(renderer);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
