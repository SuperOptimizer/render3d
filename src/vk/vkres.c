#include "vk/vkres.h"

#include <stdio.h>
#include <stdlib.h>
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
  /* dimensionality heuristic: depth>1 -> 3D, height>1 -> 2D, else 1D
   * (callers wanting a 1-slice 3D image must pass depth>=2) */
  VkImageType type = extent.depth > 1   ? VK_IMAGE_TYPE_3D
                     : extent.height > 1 ? VK_IMAGE_TYPE_2D
                                         : VK_IMAGE_TYPE_1D;
  VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = type,
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
      .viewType = type == VK_IMAGE_TYPE_3D   ? VK_IMAGE_VIEW_TYPE_3D
                  : type == VK_IMAGE_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D
                                             : VK_IMAGE_VIEW_TYPE_1D,
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

int r3d_vkcomp_create(r3d_vkctx *c, const char *spv_path, const VkDescriptorType *types,
                      uint32_t ntypes, uint32_t push_size, r3d_vkcomp *out) {
  memset(out, 0, sizeof *out);
  VkDescriptorSetLayoutBinding binds[16];
  VkDescriptorPoolSize sizes[16];
  for (uint32_t i = 0; i < ntypes && i < 16; i++) {
    binds[i] = (VkDescriptorSetLayoutBinding){.binding = i,
                                              .descriptorType = types[i],
                                              .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    sizes[i] = (VkDescriptorPoolSize){types[i], 1};
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = ntypes,
      .pBindings = binds};
  if (vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, &out->dsl) != VK_SUCCESS) return -1;
  VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = push_size};
  VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1,
                                     .pSetLayouts = &out->dsl,
                                     .pushConstantRangeCount = push_size ? 1u : 0u,
                                     .pPushConstantRanges = &pcr};
  if (vkCreatePipelineLayout(c->dev, &plci, NULL, &out->layout) != VK_SUCCESS) return -1;

  FILE *f = fopen(spv_path, "rb");
  if (!f) {
    fprintf(stderr, "vkcomp: missing %s\n", spv_path);
    return -1;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint32_t *code = malloc((size_t)n);
  if (!code || fread(code, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(code);
    return -1;
  }
  fclose(f);
  VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = (size_t)n,
                                   .pCode = code};
  VkShaderModule mod;
  VkResult r = vkCreateShaderModule(c->dev, &smci, NULL, &mod);
  free(code);
  if (r != VK_SUCCESS) return -1;
  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main"},
      .layout = out->layout};
  r = vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &out->pipe);
  vkDestroyShaderModule(c->dev, mod, NULL);
  if (r != VK_SUCCESS) return -1;

  VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 1,
                                     .poolSizeCount = ntypes,
                                     .pPoolSizes = sizes};
  if (vkCreateDescriptorPool(c->dev, &dpci, NULL, &out->dpool) != VK_SUCCESS) return -1;
  VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = out->dpool,
                                      .descriptorSetCount = 1,
                                      .pSetLayouts = &out->dsl};
  return vkAllocateDescriptorSets(c->dev, &dsai, &out->dset) == VK_SUCCESS ? 0 : -1;
}

void r3d_vkcomp_destroy(r3d_vkctx *c, r3d_vkcomp *p) {
  if (p->dpool) vkDestroyDescriptorPool(c->dev, p->dpool, NULL);
  if (p->pipe) vkDestroyPipeline(c->dev, p->pipe, NULL);
  if (p->layout) vkDestroyPipelineLayout(c->dev, p->layout, NULL);
  if (p->dsl) vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
  memset(p, 0, sizeof *p);
}

void r3d_vkcomp_bind_image(r3d_vkctx *c, r3d_vkcomp *p, uint32_t binding, VkDescriptorType type,
                           VkImageView view, VkSampler sampler, VkImageLayout layout) {
  VkDescriptorImageInfo ii = {.sampler = sampler, .imageView = view, .imageLayout = layout};
  VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = p->dset,
                            .dstBinding = binding,
                            .descriptorCount = 1,
                            .descriptorType = type,
                            .pImageInfo = &ii};
  vkUpdateDescriptorSets(c->dev, 1, &w, 0, NULL);
}

void r3d_vkcomp_dispatch(VkCommandBuffer cmd, r3d_vkcomp *p, const void *push,
                         uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->dset, 0,
                          NULL);
  if (push_size) vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
  vkCmdDispatch(cmd, gx, gy, gz);
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
