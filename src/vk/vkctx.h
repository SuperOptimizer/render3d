/* Vulkan instance/device bring-up + capability probe. Every capability the
 * renderer relies on is queried here and gates a fallback — never assumed
 * (docs/measured.md records the target hardware's answers). */
#ifndef R3D_VKCTX_H
#define R3D_VKCTX_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

typedef struct r3d_vkcaps {
  uint32_t api_version;
  uint32_t max_dim_3d;
  uint32_t max_push_bytes;
  uint32_t max_wg_invocations;
  uint32_t subgroup_size;
  VkDeviceSize max_alloc_bytes;   /* maintenance3 maxMemoryAllocationSize */
  VkFormatFeatureFlags r8_optimal; /* R8_UNORM optimal-tiling features */
  bool host_image_copy;            /* VK_EXT_host_image_copy usable */
  bool timestamps;                 /* compute-queue timestamps supported */
  double ts_period_ns;
  char dev_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
} r3d_vkcaps;

typedef struct r3d_vkctx {
  VkInstance instance;
  VkDebugUtilsMessengerEXT messenger; /* VK_NULL_HANDLE unless validating */
  VkPhysicalDevice phys;
  VkDevice dev;
  VkQueue queue; /* one graphics+compute queue */
  uint32_t qfam;
  r3d_vkcaps caps;
} r3d_vkctx;

/* inst_exts: extra instance extensions (e.g. from SDL_Vulkan_GetInstanceExtensions);
 * may be NULL/0 for headless probing. validate: enable VK_LAYER_KHRONOS_validation
 * + debug messenger (dev builds; also via env R3D_VALIDATE=1). Returns 0 on success. */
int r3d_vkctx_create(r3d_vkctx *c, const char *const *inst_exts, uint32_t n_inst_exts,
                     bool validate);
void r3d_vkctx_destroy(r3d_vkctx *c);

/* Human-readable capability report (the `render3d --probe` output). */
void r3d_vkctx_print_caps(const r3d_vkctx *c);

#endif /* R3D_VKCTX_H */
