/* SDL3 surface + swapchain management. Swapchain images are TRANSFER_DST only
 * (we blit the offscreen storage image into them; storage usage on swapchain
 * images is never assumed). */
#ifndef R3D_VKSWAP_H
#define R3D_VKSWAP_H

#include <SDL3/SDL.h>
#include <stdbool.h>

#include "vk/vkctx.h"

#define R3D_MAX_SWAP_IMAGES 8

typedef struct r3d_vkswap {
  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;
  VkFormat format;
  VkExtent2D extent;
  uint32_t nimages;
  VkImage images[R3D_MAX_SWAP_IMAGES];
  VkSemaphore render_done[R3D_MAX_SWAP_IMAGES]; /* per-image, signaled by submit */
} r3d_vkswap;

int r3d_vkswap_create(r3d_vkctx *c, SDL_Window *win, bool vsync, r3d_vkswap *s);
/* Recreate after resize/out-of-date. Caller must ensure the device is idle. */
int r3d_vkswap_recreate(r3d_vkctx *c, SDL_Window *win, bool vsync, r3d_vkswap *s);
void r3d_vkswap_destroy(r3d_vkctx *c, r3d_vkswap *s);

#endif /* R3D_VKSWAP_H */
