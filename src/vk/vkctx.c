#include "vk/vkctx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_TRY(expr)                                                       \
  do {                                                                     \
    VkResult vk_try_r = (expr);                                            \
    if (vk_try_r != VK_SUCCESS) {                                          \
      fprintf(stderr, "vk: %s failed (%d) at %s:%d\n", #expr, (int)vk_try_r, \
              __FILE__, __LINE__);                                         \
      return -1;                                                           \
    }                                                                      \
  } while (0)

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
  (void)types;
  (void)user;
  const char *tag = sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "warn";
  fprintf(stderr, "vk-validation [%s]: %s\n", tag, data->pMessage);
  return VK_FALSE;
}

static bool has_layer(const char *name) {
  uint32_t n = 0;
  vkEnumerateInstanceLayerProperties(&n, NULL);
  VkLayerProperties *props = calloc(n, sizeof *props);
  if (!props) return false;
  vkEnumerateInstanceLayerProperties(&n, props);
  bool found = false;
  for (uint32_t i = 0; i < n; i++)
    if (strcmp(props[i].layerName, name) == 0) found = true;
  free(props);
  return found;
}

static bool dev_has_ext(VkPhysicalDevice pd, const char *name) {
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
  VkExtensionProperties *props = calloc(n, sizeof *props);
  if (!props) return false;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &n, props);
  bool found = false;
  for (uint32_t i = 0; i < n; i++)
    if (strcmp(props[i].extensionName, name) == 0) found = true;
  free(props);
  return found;
}

int r3d_vkctx_create(r3d_vkctx *c, const char *const *inst_exts, uint32_t n_inst_exts,
                     bool validate) {
  memset(c, 0, sizeof *c);
  validate = validate || (getenv("R3D_VALIDATE") && *getenv("R3D_VALIDATE") == '1');
  if (validate && !has_layer("VK_LAYER_KHRONOS_validation")) {
    fprintf(stderr, "vk: validation requested but layer not installed; continuing without\n");
    validate = false;
  }

  /* --- instance --- */
  const char *exts[16];
  uint32_t nexts = 0;
  for (uint32_t i = 0; i < n_inst_exts && nexts < 15; i++) exts[nexts++] = inst_exts[i];
  if (validate) exts[nexts++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

  VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "render3d",
      .apiVersion = VK_API_VERSION_1_3,
  };
  const char *layer = "VK_LAYER_KHRONOS_validation";
  VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = nexts,
      .ppEnabledExtensionNames = exts,
      .enabledLayerCount = validate ? 1u : 0u,
      .ppEnabledLayerNames = validate ? &layer : NULL,
  };
  VK_TRY(vkCreateInstance(&ici, NULL, &c->instance));

  if (validate) {
    VkDebugUtilsMessengerCreateInfoEXT dci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_cb,
    };
    PFN_vkCreateDebugUtilsMessengerEXT create_messenger =
        (PFN_vkCreateDebugUtilsMessengerEXT)(void (*)(void))vkGetInstanceProcAddr(
            c->instance, "vkCreateDebugUtilsMessengerEXT");
    if (create_messenger) create_messenger(c->instance, &dci, NULL, &c->messenger);
  }

  /* --- physical device: prefer discrete > integrated > anything-not-CPU --- */
  uint32_t npd = 0;
  VK_TRY(vkEnumeratePhysicalDevices(c->instance, &npd, NULL));
  if (npd == 0) {
    fprintf(stderr, "vk: no devices\n");
    return -1;
  }
  VkPhysicalDevice pds[8];
  if (npd > 8) npd = 8;
  VK_TRY(vkEnumeratePhysicalDevices(c->instance, &npd, pds));
  int best_score = -1;
  for (uint32_t i = 0; i < npd; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pds[i], &p);
    int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 0
                                                                         : 1;
    if (score > best_score) {
      best_score = score;
      c->phys = pds[i];
    }
  }

  /* --- queue family: one graphics+compute queue --- */
  uint32_t nqf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nqf, NULL);
  VkQueueFamilyProperties qf[16];
  if (nqf > 16) nqf = 16;
  vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nqf, qf);
  c->qfam = UINT32_MAX;
  for (uint32_t i = 0; i < nqf; i++) {
    VkQueueFlags want = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if ((qf[i].queueFlags & want) == want) {
      c->qfam = i;
      c->caps.timestamps = qf[i].timestampValidBits > 0;
      break;
    }
  }
  if (c->qfam == UINT32_MAX) {
    fprintf(stderr, "vk: no graphics+compute queue family\n");
    return -1;
  }

  /* --- capabilities --- */
  VkPhysicalDeviceMaintenance3Properties m3 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties sg = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = &m3};
  VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                    .pNext = &sg};
  vkGetPhysicalDeviceProperties2(c->phys, &p2);
  c->caps.api_version = p2.properties.apiVersion;
  c->caps.max_dim_3d = p2.properties.limits.maxImageDimension3D;
  c->caps.max_push_bytes = p2.properties.limits.maxPushConstantsSize;
  c->caps.max_wg_invocations = p2.properties.limits.maxComputeWorkGroupInvocations;
  c->caps.subgroup_size = sg.subgroupSize;
  c->caps.max_alloc_bytes = m3.maxMemoryAllocationSize;
  c->caps.ts_period_ns = (double)p2.properties.limits.timestampPeriod;
  memcpy(c->caps.dev_name, p2.properties.deviceName, sizeof c->caps.dev_name);
  if (p2.properties.apiVersion < VK_API_VERSION_1_3) {
    fprintf(stderr, "vk: %s is Vulkan %u.%u, need 1.3\n", c->caps.dev_name,
            VK_API_VERSION_MAJOR(p2.properties.apiVersion),
            VK_API_VERSION_MINOR(p2.properties.apiVersion));
    return -1;
  }

  VkFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
  vkGetPhysicalDeviceFormatProperties2(c->phys, VK_FORMAT_R8_UNORM, &fp);
  c->caps.r8_optimal = fp.formatProperties.optimalTilingFeatures;

  bool want_hic = dev_has_ext(c->phys, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
  VkPhysicalDeviceHostImageCopyFeaturesEXT hicf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT};
  VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = want_hic ? &hicf : NULL};
  VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13};
  VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                  .pNext = &f12};
  vkGetPhysicalDeviceFeatures2(c->phys, &f2);
  if (!f12.timelineSemaphore || !f13.synchronization2 || !f13.maintenance4) {
    fprintf(stderr, "vk: missing core features (timeline=%u sync2=%u maint4=%u)\n",
            f12.timelineSemaphore, f13.synchronization2, f13.maintenance4);
    return -1;
  }
  c->caps.host_image_copy = want_hic && hicf.hostImageCopy;

  /* --- device --- */
  const char *dev_exts[4];
  uint32_t ndev_exts = 0;
  if (dev_has_ext(c->phys, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
    dev_exts[ndev_exts++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  if (c->caps.host_image_copy) dev_exts[ndev_exts++] = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME;

  VkPhysicalDeviceHostImageCopyFeaturesEXT en_hic = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
      .hostImageCopy = VK_TRUE};
  VkPhysicalDeviceVulkan13Features en13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = c->caps.host_image_copy ? &en_hic : NULL,
      .synchronization2 = VK_TRUE,
      .maintenance4 = VK_TRUE,
  };
  VkPhysicalDeviceVulkan12Features en12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &en13,
      .timelineSemaphore = VK_TRUE,
      .hostQueryReset = f12.hostQueryReset,
  };
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = c->qfam,
      .queueCount = 1,
      .pQueuePriorities = &prio,
  };
  VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &en12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = ndev_exts,
      .ppEnabledExtensionNames = dev_exts,
  };
  VK_TRY(vkCreateDevice(c->phys, &dci, NULL, &c->dev));
  vkGetDeviceQueue(c->dev, c->qfam, 0, &c->queue);
  return 0;
}

void r3d_vkctx_destroy(r3d_vkctx *c) {
  if (c->dev) vkDestroyDevice(c->dev, NULL);
  if (c->messenger) {
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger =
        (PFN_vkDestroyDebugUtilsMessengerEXT)(void (*)(void))vkGetInstanceProcAddr(
            c->instance, "vkDestroyDebugUtilsMessengerEXT");
    if (destroy_messenger) destroy_messenger(c->instance, c->messenger, NULL);
  }
  if (c->instance) vkDestroyInstance(c->instance, NULL);
  memset(c, 0, sizeof *c);
}

void r3d_vkctx_print_caps(const r3d_vkctx *c) {
  const r3d_vkcaps *k = &c->caps;
  printf("device            : %s (Vulkan %u.%u.%u)\n", k->dev_name,
         VK_API_VERSION_MAJOR(k->api_version), VK_API_VERSION_MINOR(k->api_version),
         VK_API_VERSION_PATCH(k->api_version));
  printf("queue family      : %u (graphics+compute, timestamps=%s, period %.2f ns)\n", c->qfam,
         k->timestamps ? "yes" : "no", k->ts_period_ns);
  printf("maxImageDimension3D    : %u\n", k->max_dim_3d);
  printf("maxMemoryAllocation    : %.2f GiB\n", (double)k->max_alloc_bytes / (1u << 30));
  printf("maxPushConstants       : %u B\n", k->max_push_bytes);
  printf("maxWorkgroupInvocations: %u (subgroup %u)\n", k->max_wg_invocations, k->subgroup_size);
  printf("host_image_copy        : %s\n", k->host_image_copy ? "yes" : "no");
  VkFormatFeatureFlags r8 = k->r8_optimal;
  printf("R8_UNORM optimal       :%s%s%s%s%s\n",
         (r8 & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? " sampled" : "",
         (r8 & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? " linear-filter" : "",
         (r8 & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? " storage" : "",
         (r8 & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? " blit-src" : "",
         (r8 & VK_FORMAT_FEATURE_BLIT_DST_BIT) ? " blit-dst" : "");
  uint32_t vol = 1024;
  printf("verdict: 1024^3 R8 + mips %s (dim %u <= %u, ~1.14 GiB <= alloc max)\n",
         (vol <= k->max_dim_3d) ? "OK" : "TOO BIG", vol, k->max_dim_3d);
}
