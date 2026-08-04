/* Small hand-rolled resource helpers (no VMA; c5d vk.c style). */
#ifndef R3D_VKRES_H
#define R3D_VKRES_H

#include "vk/vkctx.h"

typedef struct r3d_vkbuf {
  VkBuffer buf;
  VkDeviceMemory mem;
  void *mapped; /* persistently mapped iff host-visible */
  VkDeviceSize size;
} r3d_vkbuf;

typedef struct r3d_vkimage {
  VkImage img;
  VkDeviceMemory mem;
  VkImageView view;
  VkFormat format;
  VkExtent3D extent;
  uint32_t mips;
} r3d_vkimage;

uint32_t r3d_vk_find_mem(const r3d_vkctx *c, uint32_t type_bits, VkMemoryPropertyFlags flags);

/* Host-visible|coherent, persistently mapped. */
int r3d_vkbuf_create_host(r3d_vkctx *c, VkDeviceSize size, VkBufferUsageFlags usage,
                          r3d_vkbuf *b);
void r3d_vkbuf_destroy(r3d_vkctx *c, r3d_vkbuf *b);

/* Device-local image + view (2D if extent.depth==1, else 3D). Dedicated allocation. */
int r3d_vkimage_create(r3d_vkctx *c, VkFormat format, VkExtent3D extent, uint32_t mips,
                       VkImageUsageFlags usage, r3d_vkimage *im);
void r3d_vkimage_destroy(r3d_vkctx *c, r3d_vkimage *im);

/* One-shot command buffer helpers (submit + wait idle; init/upload paths only). */
VkCommandBuffer r3d_vk_oneshot_begin(r3d_vkctx *c, VkCommandPool pool);
int r3d_vk_oneshot_end(r3d_vkctx *c, VkCommandPool pool, VkCommandBuffer cmd);

/* Small generic compute pipeline (descriptor types given per binding). */
typedef struct r3d_vkcomp {
  VkDescriptorSetLayout dsl;
  VkPipelineLayout layout;
  VkPipeline pipe;
  VkDescriptorPool dpool;
  VkDescriptorSet dset;
} r3d_vkcomp;

int r3d_vkcomp_create(r3d_vkctx *c, const char *spv_path, const VkDescriptorType *types,
                      uint32_t ntypes, uint32_t push_size, r3d_vkcomp *out);
void r3d_vkcomp_destroy(r3d_vkctx *c, r3d_vkcomp *p);
void r3d_vkcomp_bind_image(r3d_vkctx *c, r3d_vkcomp *p, uint32_t binding, VkDescriptorType type,
                           VkImageView view, VkSampler sampler, VkImageLayout layout);
void r3d_vkcomp_dispatch(VkCommandBuffer cmd, r3d_vkcomp *p, const void *push,
                         uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz);

/* sync2 image barrier convenience */
void r3d_vk_image_barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                          uint32_t base_mip, uint32_t mip_count);

#endif /* R3D_VKRES_H */
