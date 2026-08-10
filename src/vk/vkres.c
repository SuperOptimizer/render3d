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

static bool mem_type_is_device_local(const r3d_vkctx *c, uint32_t idx) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(c->phys, &mp);
  return idx < mp.memoryTypeCount &&
         (mp.memoryTypes[idx].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

/* Reserve before vkAllocateMemory so concurrent render/stream workers cannot
 * both pass a stale budget/count check. The reservation becomes the live
 * allocation accounting on success and is rolled back on failure. */
static bool reserve_mem(r3d_vkctx *c, VkDeviceSize device_local_bytes) {
  uint32_t count = atomic_load(&c->allocation_count);
  do {
    if (count >= c->caps.max_allocations) return false;
  } while (!atomic_compare_exchange_weak(&c->allocation_count, &count, count + 1));
  if (!device_local_bytes) return true;
  VkDeviceSize used = atomic_load(&c->allocated_bytes);
  do {
    if (used > c->budget_bytes || device_local_bytes > c->budget_bytes - used) {
      atomic_fetch_sub(&c->allocation_count, 1);
      return false;
    }
  } while (!atomic_compare_exchange_weak(&c->allocated_bytes, &used,
                                          used + device_local_bytes));
  return true;
}

static void release_mem_reservation(r3d_vkctx *c, VkDeviceSize device_local_bytes) {
  if (device_local_bytes) atomic_fetch_sub(&c->allocated_bytes, device_local_bytes);
  atomic_fetch_sub(&c->allocation_count, 1);
}

int r3d_vkbuf_create_host(r3d_vkctx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                          r3d_vkbuf *b) {
  memset(b, 0, sizeof *b);
  VkDeviceSize charged = 0;
  bool reserved = false;
  VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
  if (vkCreateBuffer(c->dev, &bci, NULL, &b->buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(c->dev, b->buf, &mr);
  uint32_t idx = r3d_vk_find_mem(c, mr.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (idx == UINT32_MAX) goto fail;
  bool device_local = mem_type_is_device_local(c, idx);
  charged = device_local ? mr.size : 0;
  if (mr.size > c->caps.max_alloc_bytes || !(reserved = reserve_mem(c, charged))) {
    fprintf(stderr, "vkres: host buffer exceeds allocation/budget limit "
                    "(%llu requested, %llu device-local free)\n",
            (unsigned long long)mr.size,
            (unsigned long long)r3d_vkctx_budget_available(c));
    goto fail;
  }
  VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = mr.size,
                              .memoryTypeIndex = idx};
  if (vkAllocateMemory(c->dev, &mai, NULL, &b->mem) != VK_SUCCESS ||
      vkBindBufferMemory(c->dev, b->buf, b->mem, 0) != VK_SUCCESS ||
      vkMapMemory(c->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->mapped) != VK_SUCCESS) {
    fprintf(stderr, "vkres: host buffer alloc failed (%llu bytes)\n", (unsigned long long)size);
    goto fail;
  }
  b->size = size;
  b->alloc_size = charged;
  return 0;
fail:
  if (b->buf) vkDestroyBuffer(c->dev, b->buf, NULL);
  if (b->mem) vkFreeMemory(c->dev, b->mem, NULL);
  if (reserved) release_mem_reservation(c, charged);
  memset(b, 0, sizeof *b);
  return -1;
}

void r3d_vkbuf_destroy(r3d_vkctx *c, r3d_vkbuf *b) {
  if (b->buf) vkDestroyBuffer(c->dev, b->buf, NULL);
  if (b->mem) vkFreeMemory(c->dev, b->mem, NULL);
  if (b->mem) release_mem_reservation(c, b->alloc_size);
  memset(b, 0, sizeof *b);
}

int r3d_vkimage_create(r3d_vkctx *c, VkFormat format, VkExtent3D extent, uint32_t mips,
                       VkImageUsageFlags usage, r3d_vkimage *im) {
  memset(im, 0, sizeof *im);
  bool reserved = false;
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
  if (idx == UINT32_MAX) goto fail;
  VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                       .image = im->img};
  VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .pNext = &ded,
                              .allocationSize = mr.size,
                              .memoryTypeIndex = idx};
  if (mr.size > c->caps.max_alloc_bytes || !(reserved = reserve_mem(c, mr.size))) {
    fprintf(stderr,
            "vkres: image exceeds allocation/budget limit (%.2f MiB requested, %.2f MiB free)\n",
            (double)mr.size / (1u << 20),
            (double)r3d_vkctx_budget_available(c) / (1u << 20));
    goto fail;
  }
  if (vkAllocateMemory(c->dev, &mai, NULL, &im->mem) != VK_SUCCESS ||
      vkBindImageMemory(c->dev, im->img, im->mem, 0) != VK_SUCCESS) {
    fprintf(stderr, "vkres: image alloc failed (%.2f MiB)\n", (double)mr.size / (1u << 20));
    goto fail;
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
  if (vkCreateImageView(c->dev, &vci, NULL, &im->view) != VK_SUCCESS) goto fail;
  im->format = format;
  im->extent = extent;
  im->mips = mips;
  im->alloc_size = mr.size;
  im->owns_mem = true;
  return 0;
fail:
  if (im->img) vkDestroyImage(c->dev, im->img, NULL);
  if (im->mem) vkFreeMemory(c->dev, im->mem, NULL);
  if (reserved) release_mem_reservation(c, mr.size);
  memset(im, 0, sizeof *im);
  return -1;
}

void r3d_vkarena_init(r3d_vkarena *a, VkDeviceSize block_size) {
  memset(a, 0, sizeof *a);
  a->block_size = block_size ? block_size : ((VkDeviceSize)256 << 20);
}

void r3d_vkarena_destroy(r3d_vkctx *c, r3d_vkarena *a) {
  if (!a) return;
  for (uint32_t i = 0; i < a->count; i++) {
    if (a->blocks[i].mem) {
      vkFreeMemory(c->dev, a->blocks[i].mem, NULL);
      release_mem_reservation(c, a->blocks[i].size);
    }
  }
  free(a->blocks);
  memset(a, 0, sizeof *a);
}

int r3d_vkimage_create_arena(r3d_vkctx *c, r3d_vkarena *a, VkFormat format,
                             VkExtent3D extent, uint32_t mips, VkImageUsageFlags usage,
                             r3d_vkimage *im) {
  if (!a) return r3d_vkimage_create(c, format, extent, mips, usage, im);
  memset(im, 0, sizeof *im);
  VkImageType type = extent.depth > 1   ? VK_IMAGE_TYPE_3D
                     : extent.height > 1 ? VK_IMAGE_TYPE_2D
                                         : VK_IMAGE_TYPE_1D;
  VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                           .imageType = type,
                           .format = format,
                           .extent = extent,
                           .mipLevels = mips,
                           .arrayLayers = 1,
                           .samples = VK_SAMPLE_COUNT_1_BIT,
                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                           .usage = usage,
                           .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                           .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  if (vkCreateImage(c->dev, &ici, NULL, &im->img) != VK_SUCCESS) return -1;
  VkMemoryDedicatedRequirements dr = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
  VkMemoryRequirements2 mr2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                               .pNext = &dr};
  VkImageMemoryRequirementsInfo2 mri = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                        .image = im->img};
  vkGetImageMemoryRequirements2(c->dev, &mri, &mr2);
  VkMemoryRequirements mr = mr2.memoryRequirements;
  if (dr.requiresDedicatedAllocation) {
    vkDestroyImage(c->dev, im->img, NULL);
    memset(im, 0, sizeof *im);
    return r3d_vkimage_create(c, format, extent, mips, usage, im);
  }
  uint32_t type_idx = r3d_vk_find_mem(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (type_idx == UINT32_MAX || mr.size > c->caps.max_alloc_bytes) goto fail;

  r3d_vkarena_block *block = NULL;
  VkDeviceSize offset = 0;
  for (uint32_t i = 0; i < a->count; i++) {
    r3d_vkarena_block *b = &a->blocks[i];
    if (b->memory_type != type_idx) continue;
    VkDeviceSize off = (b->used + mr.alignment - 1) & ~(mr.alignment - 1);
    if (off <= b->size && mr.size <= b->size - off) {
      block = b;
      offset = off;
      break;
    }
  }
  if (!block) {
    VkDeviceSize block_size = a->block_size > mr.size ? a->block_size : mr.size;
    if (block_size > c->caps.max_alloc_bytes) block_size = c->caps.max_alloc_bytes;
    VkDeviceSize available = r3d_vkctx_budget_available(c);
    if (block_size > available) block_size = available;
    if (block_size < mr.size) goto fail;
    if (a->count == a->capacity) {
      uint32_t nc = a->capacity ? a->capacity * 2u : 8u;
      r3d_vkarena_block *nb = realloc(a->blocks, (size_t)nc * sizeof *nb);
      if (!nb) goto fail;
      a->blocks = nb;
      a->capacity = nc;
    }
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = block_size,
                                .memoryTypeIndex = type_idx};
    block = &a->blocks[a->count];
    memset(block, 0, sizeof *block);
    if (!reserve_mem(c, block_size)) goto fail;
    if (vkAllocateMemory(c->dev, &mai, NULL, &block->mem) != VK_SUCCESS) {
      release_mem_reservation(c, block_size);
      goto fail;
    }
    block->size = block_size;
    block->memory_type = type_idx;
    a->count++;
  }
  if (vkBindImageMemory(c->dev, im->img, block->mem, offset) != VK_SUCCESS) goto fail;
  block->used = offset + mr.size;
  VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = im->img,
      .viewType = type == VK_IMAGE_TYPE_3D   ? VK_IMAGE_VIEW_TYPE_3D
                  : type == VK_IMAGE_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D
                                             : VK_IMAGE_VIEW_TYPE_1D,
      .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1}};
  if (vkCreateImageView(c->dev, &vci, NULL, &im->view) != VK_SUCCESS) goto fail;
  im->mem = block->mem; /* borrowed; the arena frees it after image destruction */
  im->format = format;
  im->extent = extent;
  im->mips = mips;
  im->owns_mem = false;
  return 0;
fail:
  if (im->view) vkDestroyImageView(c->dev, im->view, NULL);
  if (im->img) vkDestroyImage(c->dev, im->img, NULL);
  memset(im, 0, sizeof *im);
  return -1;
}

void r3d_vkimage_destroy(r3d_vkctx *c, r3d_vkimage *im) {
  if (im->view) vkDestroyImageView(c->dev, im->view, NULL);
  if (im->img) vkDestroyImage(c->dev, im->img, NULL);
  if (im->owns_mem && im->mem) vkFreeMemory(c->dev, im->mem, NULL);
  if (im->owns_mem && im->mem) release_mem_reservation(c, im->alloc_size);
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

void r3d_vkcomp_bind_buffer(r3d_vkctx *c, r3d_vkcomp *p, uint32_t binding, VkBuffer buf,
                            VkDeviceSize offset, VkDeviceSize range) {
  VkDescriptorBufferInfo bi = {.buffer = buf, .offset = offset, .range = range};
  VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = p->dset,
                            .dstBinding = binding,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &bi};
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
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(c->dev, &fci, NULL, &fence) != VK_SUCCESS) {
    vkFreeCommandBuffers(c->dev, pool, 1, &cmd);
    return -1;
  }
  VkCommandBufferSubmitInfo csi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                   .commandBuffer = cmd};
  VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                      .commandBufferInfoCount = 1,
                      .pCommandBufferInfos = &csi};
  VkResult r = r3d_vkctx_queue_submit2(c, 1, &si, fence);
  if (r == VK_SUCCESS) r = vkWaitForFences(c->dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(c->dev, fence, NULL);
  vkFreeCommandBuffers(c->dev, pool, 1, &cmd);
  return r == VK_SUCCESS ? 0 : -1;
}

int r3d_vk_image_to_general(r3d_vkctx *c, VkCommandPool pool, r3d_vkimage *img) {
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(c, pool);
  if (!cmd) return -1;
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0, img->mips);
  return r3d_vk_oneshot_end(c, pool, cmd);
}

int r3d_vk_upload_image_staged_buf_pitch(r3d_vkctx *c, VkCommandPool pool, r3d_vkbuf *stage,
                                         r3d_vkimage *img, const void *host,
                                         uint32_t row_length, uint32_t image_height,
                                         VkOffset3D offset, VkExtent3D extent) {
  if (!host || !row_length || image_height < extent.height || !extent.width ||
      !extent.height || !extent.depth)
    return -1;
  VkDeviceSize plane = (VkDeviceSize)row_length * image_height;
  VkDeviceSize bytes = plane * (extent.depth - 1) +
                       (VkDeviceSize)row_length * (extent.height - 1) + extent.width;
  if (stage->size < bytes) {
    r3d_vkbuf_destroy(c, stage);
    VkDeviceSize cap = 1u << 20;
    while (cap < bytes && cap <= UINT64_MAX / 2) cap *= 2;
    if (cap < bytes || r3d_vkbuf_create_host(c, cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stage) != 0)
      return -1;
  }
  for (uint32_t z = 0; z < extent.depth; z++)
    for (uint32_t y = 0; y < extent.height; y++)
      memcpy((uint8_t *)stage->mapped + (size_t)z * plane + (size_t)y * row_length,
             (const uint8_t *)host + (size_t)z * plane + (size_t)y * row_length,
             extent.width);

  VkCommandBuffer cmd = r3d_vk_oneshot_begin(c, pool);
  if (!cmd) return -1;
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
  VkBufferImageCopy reg = {
      .bufferRowLength = row_length,
      .bufferImageHeight = image_height,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = offset,
      .imageExtent = extent,
  };
  vkCmdCopyBufferToImage(cmd, stage->buf, img->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &reg);
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
  return r3d_vk_oneshot_end(c, pool, cmd);
}

int r3d_vk_upload_image_staged_buf(r3d_vkctx *c, VkCommandPool pool, r3d_vkbuf *stage,
                                   r3d_vkimage *img, const void *host, uint32_t row_length,
                                   VkOffset3D offset, VkExtent3D extent) {
  return r3d_vk_upload_image_staged_buf_pitch(c, pool, stage, img, host, row_length,
                                               extent.height, offset, extent);
}

int r3d_vk_upload_image_staged(r3d_vkctx *c, VkCommandPool pool, r3d_vkimage *img,
                               const void *host, uint32_t row_length, VkOffset3D offset,
                               VkExtent3D extent) {
  r3d_vkbuf stage = {0};
  int rc = r3d_vk_upload_image_staged_buf(c, pool, &stage, img, host, row_length, offset, extent);
  r3d_vkbuf_destroy(c, &stage);
  return rc;
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
