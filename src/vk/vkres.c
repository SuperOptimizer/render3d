#include "vk/vkres.h"

#include <stdio.h>
#include <string.h>

uint32_t r3d_vk_find_mem(const r3d_vkctx *c, uint32_t type_bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(c->phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
  return UINT32_MAX;
}

int r3d_vkbuf_create_host(r3d_vkctx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                          r3d_vkbuf *b) {
  memset(b, 0, sizeof *b);
  VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
  if (vkCreateBuffer(c->dev, &bci, NULL, &b->buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(c->dev, b->buf, &mr);
  uint32_t idx = r3d_vk_find_mem(c, mr.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (idx == UINT32_MAX) return -1;
  VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = mr.size,
                              .memoryTypeIndex = idx};
  if (vkAllocateMemory(c->dev, &mai, NULL, &b->mem) != VK_SUCCESS ||
      vkBindBufferMemory(c->dev, b->buf, b->mem, 0) != VK_SUCCESS ||
      vkMapMemory(c->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->mapped) != VK_SUCCESS) {
    fprintf(stderr, "vkres: host buffer alloc failed (%llu bytes)\n", (unsigned long long)size);
    return -1;
  }
  b->size = size;
  return 0;
}

void r3d_vkbuf_destroy(r3d_vkctx *c, r3d_vkbuf *b) {
  if (b->buf) vkDestroyBuffer(c->dev, b->buf, NULL);
  if (b->mem) vkFreeMemory(c->dev, b->mem, NULL);
  memset(b, 0, sizeof *b);
}

int r3d_vkimage_create(r3d_vkctx *c, VkFormat format, VkExtent3D extent, uint32_t mips,
                       VkImageUsageFlags usage, r3d_vkimage *im) {
  memset(im, 0, sizeof *im);
  bool is3d = extent.depth > 1;
  VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = extent,
      .mipLevels = mips,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  if (vkCreateImage(c->dev, &ici, NULL, &im->img) != VK_SUCCESS) {
    fprintf(stderr, "vkres: image create failed (%ux%ux%u m%u)\n", extent.width, extent.height,
            extent.depth, mips);
    return -1;
  }
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(c->dev, im->img, &mr);
  uint32_t idx = r3d_vk_find_mem(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (idx == UINT32_MAX) return -1;
  VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                       .image = im->img};
  VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .pNext = &ded,
                              .allocationSize = mr.size,
                              .memoryTypeIndex = idx};
  if (vkAllocateMemory(c->dev, &mai, NULL, &im->mem) != VK_SUCCESS ||
      vkBindImageMemory(c->dev, im->img, im->mem, 0) != VK_SUCCESS) {
    fprintf(stderr, "vkres: image alloc failed (%.2f MiB)\n", (double)mr.size / (1u << 20));
    return -1;
  }
  VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = im->img,
      .viewType = is3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1},
  };
  if (vkCreateImageView(c->dev, &vci, NULL, &im->view) != VK_SUCCESS) return -1;
  im->format = format;
  im->extent = extent;
  im->mips = mips;
  return 0;
}

void r3d_vkimage_destroy(r3d_vkctx *c, r3d_vkimage *im) {
  if (im->view) vkDestroyImageView(c->dev, im->view, NULL);
  if (im->img) vkDestroyImage(c->dev, im->img, NULL);
  if (im->mem) vkFreeMemory(c->dev, im->mem, NULL);
  memset(im, 0, sizeof *im);
}

VkCommandBuffer r3d_vk_oneshot_begin(r3d_vkctx *c, VkCommandPool pool) {
  VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(c->dev, &cai, &cmd) != VK_SUCCESS) return VK_NULL_HANDLE;
  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cmd, &bi);
  return cmd;
}

int r3d_vk_oneshot_end(r3d_vkctx *c, VkCommandPool pool, VkCommandBuffer cmd) {
  vkEndCommandBuffer(cmd);
  VkCommandBufferSubmitInfo csi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                   .commandBuffer = cmd};
  VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                      .commandBufferInfoCount = 1,
                      .pCommandBufferInfos = &csi};
  VkResult r = vkQueueSubmit2(c->queue, 1, &si, VK_NULL_HANDLE);
  if (r == VK_SUCCESS) r = vkQueueWaitIdle(c->queue);
  vkFreeCommandBuffers(c->dev, pool, 1, &cmd);
  return r == VK_SUCCESS ? 0 : -1;
}

void r3d_vk_image_barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                          uint32_t base_mip, uint32_t mip_count) {
  VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_mip, mip_count, 0, 1},
  };
  VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
  vkCmdPipelineBarrier2(cmd, &dep);
}
