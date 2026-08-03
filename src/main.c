/* render3d — volumetric renderer for Vesuvius Challenge micro-CT volumes.
 * M1: SDL3 window + Vulkan compute raycaster (see spec/ and docs/measured.md). */
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/stats.h"
#include "vk/vkctx.h"

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

  r3d_stats stats;
  r3d_stats_init(&stats);

  bool running = true;
  while (running) {
    uint64_t t0 = r3d_now_ns();
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_EVENT_QUIT) running = false;
      if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_ESCAPE) running = false;
    }
    /* Rendering lands in later milestone steps; don't spin the CPU meanwhile. */
    SDL_Delay(8);
    r3d_stats_push(&stats, r3d_now_ns() - t0, 0);
    r3d_stats_report(&stats);
  }

  SDL_DestroyWindow(win);
  SDL_Quit();
  return EXIT_SUCCESS;
}
