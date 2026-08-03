/* Dear ImGui (via cimgui) drawn as a dynamic-rendering color pass directly on
 * the swapchain image, after the raycast blit. Backend-internal header. */
#ifndef R3D_VKGUI_H
#define R3D_VKGUI_H

#include <SDL3/SDL.h>

#include "vk/vkctx.h"

int r3d_vkgui_init(r3d_vkctx *c, SDL_Window *win, VkFormat color_format, uint32_t image_count);
void r3d_vkgui_shutdown(void);

void r3d_vkgui_event(const SDL_Event *ev);
void r3d_vkgui_new_frame(void); /* backend NewFrame + igNewFrame */
void r3d_vkgui_discard(void);   /* end an opened frame without drawing (skipped frame) */
/* igRender + draw into `view` (already COLOR_ATTACHMENT_OPTIMAL, loadOp=LOAD). */
void r3d_vkgui_render(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent);

#endif /* R3D_VKGUI_H */
