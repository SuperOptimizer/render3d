#include "vk/vkswap.h"

#include <SDL3/SDL_vulkan.h>
#include <stdio.h>
#include <string.h>

static int create_swapchain(r3d_vkctx *c, SDL_Window *win, bool vsync, r3d_vkswap *s,
                            VkSwapchainKHR old) {
  VkSurfaceCapabilitiesKHR caps;
  if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, s->surface, &caps) != VK_SUCCESS) {
    fprintf(stderr, "vkswap: surface caps query failed\n");
    return -1;
  }

  /* pick format: prefer BGRA8/RGBA8 UNORM, else first offered */
  uint32_t nfmt = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, s->surface, &nfmt, NULL);
  VkSurfaceFormatKHR fmts[32];
  if (nfmt > 32) nfmt = 32;
  vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, s->surface, &nfmt, fmts);
  VkSurfaceFormatKHR pick = fmts[0];
  for (uint32_t i = 0; i < nfmt; i++)
    if ((fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM || fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM) &&
        fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      pick = fmts[i];
      break;
    }

  /* present mode: FIFO always exists; MAILBOX (uncapped, no tear) if !vsync */
  VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  if (!vsync) {
    uint32_t npm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(c->phys, s->surface, &npm, NULL);
    VkPresentModeKHR pms[8];
    if (npm > 8) npm = 8;
    vkGetPhysicalDeviceSurfacePresentModesKHR(c->phys, s->surface, &npm, pms);
    for (uint32_t i = 0; i < npm; i++)
      if (pms[i] == VK_PRESENT_MODE_MAILBOX_KHR) mode = VK_PRESENT_MODE_MAILBOX_KHR;
    if (mode == VK_PRESENT_MODE_FIFO_KHR)
      for (uint32_t i = 0; i < npm; i++)
        if (pms[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
  }

  VkExtent2D extent = caps.currentExtent;
  if (extent.width == UINT32_MAX) { /* surface lets us choose */
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(win, &w, &h);
    extent.width = (uint32_t)w;
    extent.height = (uint32_t)h;
  }
  if (extent.width == 0 || extent.height == 0) return 1; /* minimized; try later */

  uint32_t count = caps.minImageCount + 1;
  if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sci = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = s->surface,
      .minImageCount = count,
      .imageFormat = pick.format,
      .imageColorSpace = pick.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = caps.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = mode,
      .clipped = VK_TRUE,
      .oldSwapchain = old,
  };
  VkSwapchainKHR sc;
  if (vkCreateSwapchainKHR(c->dev, &sci, NULL, &sc) != VK_SUCCESS) {
    fprintf(stderr, "vkswap: swapchain create failed\n");
    return -1;
  }
  if (old) vkDestroySwapchainKHR(c->dev, old, NULL);
  s->swapchain = sc;
  s->format = pick.format;
  s->extent = extent;

  for (uint32_t i = 0; i < R3D_MAX_SWAP_IMAGES; i++)
    if (s->views[i]) {
      vkDestroyImageView(c->dev, s->views[i], NULL);
      s->views[i] = VK_NULL_HANDLE;
    }
  s->nimages = 0;
  vkGetSwapchainImagesKHR(c->dev, sc, &s->nimages, NULL);
  if (s->nimages > R3D_MAX_SWAP_IMAGES) s->nimages = R3D_MAX_SWAP_IMAGES;
  vkGetSwapchainImagesKHR(c->dev, sc, &s->nimages, s->images);
  for (uint32_t i = 0; i < s->nimages; i++) {
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = s->images[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = s->format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if (vkCreateImageView(c->dev, &vci, NULL, &s->views[i]) != VK_SUCCESS) return -1;
  }

  for (uint32_t i = 0; i < s->nimages; i++) {
    if (s->render_done[i]) continue;
    VkSemaphoreCreateInfo semci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (vkCreateSemaphore(c->dev, &semci, NULL, &s->render_done[i]) != VK_SUCCESS) return -1;
  }
  return 0;
}

int r3d_vkswap_create(r3d_vkctx *c, SDL_Window *win, bool vsync, r3d_vkswap *s) {
  memset(s, 0, sizeof *s);
  if (!SDL_Vulkan_CreateSurface(win, c->instance, NULL, &s->surface)) {
    fprintf(stderr, "vkswap: SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
    return -1;
  }
  VkBool32 sup = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(c->phys, c->qfam, s->surface, &sup);
  if (!sup) {
    fprintf(stderr, "vkswap: queue family %u cannot present\n", c->qfam);
    return -1;
  }
  return create_swapchain(c, win, vsync, s, VK_NULL_HANDLE);
}

int r3d_vkswap_recreate(r3d_vkctx *c, SDL_Window *win, bool vsync, r3d_vkswap *s) {
  vkDeviceWaitIdle(c->dev);
  return create_swapchain(c, win, vsync, s, s->swapchain);
}

void r3d_vkswap_destroy(r3d_vkctx *c, r3d_vkswap *s) {
  for (uint32_t i = 0; i < R3D_MAX_SWAP_IMAGES; i++) {
    if (s->views[i]) vkDestroyImageView(c->dev, s->views[i], NULL);
    if (s->render_done[i]) vkDestroySemaphore(c->dev, s->render_done[i], NULL);
  }
  if (s->swapchain) vkDestroySwapchainKHR(c->dev, s->swapchain, NULL);
  if (s->surface) vkDestroySurfaceKHR(c->instance, s->surface, NULL);
  memset(s, 0, sizeof *s);
}
