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
  c->caps.vendor_id = p2.properties.vendorID;
  c->caps.device_id = p2.properties.deviceID;
  c->caps.max_dim_3d = p2.properties.limits.maxImageDimension3D;
  c->caps.max_push_bytes = p2.properties.limits.maxPushConstantsSize;
  c->caps.max_wg_invocations = p2.properties.limits.maxComputeWorkGroupInvocations;
  c->caps.min_ubo_alignment = p2.properties.limits.minUniformBufferOffsetAlignment;
  c->caps.max_allocations = p2.properties.limits.maxMemoryAllocationCount;
  c->caps.max_sampled_images = p2.properties.limits.maxDescriptorSetSampledImages;
  if (c->caps.max_sampled_images > p2.properties.limits.maxPerStageDescriptorSampledImages)
    c->caps.max_sampled_images = p2.properties.limits.maxPerStageDescriptorSampledImages;
  if (c->caps.max_sampled_images > p2.properties.limits.maxDescriptorSetSamplers)
    c->caps.max_sampled_images = p2.properties.limits.maxDescriptorSetSamplers;
  if (c->caps.max_sampled_images > p2.properties.limits.maxPerStageDescriptorSamplers)
    c->caps.max_sampled_images = p2.properties.limits.maxPerStageDescriptorSamplers;
  c->caps.max_stage_resources = p2.properties.limits.maxPerStageResources;
  c->caps.subgroup_size = sg.subgroupSize;
  c->caps.max_alloc_bytes = m3.maxMemoryAllocationSize;
  c->caps.ts_period_ns = (double)p2.properties.limits.timestampPeriod;
  memcpy(c->caps.dev_name, p2.properties.deviceName, sizeof c->caps.dev_name);
  /* 1.3 native, or 1.2 + the KHR extensions we actually use (sync2 +
   * dynamic rendering — e.g. Mesa Dozen on WSL2, a 1.2 driver) */
  bool vk12_compat = p2.properties.apiVersion < VK_API_VERSION_1_3;
  if (p2.properties.apiVersion < VK_API_VERSION_1_2 ||
      (vk12_compat && (!dev_has_ext(c->phys, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) ||
                       !dev_has_ext(c->phys, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)))) {
    fprintf(stderr, "vk: %s is Vulkan %u.%u, need 1.3 (or 1.2 + sync2/dynrender)\n",
            c->caps.dev_name, VK_API_VERSION_MAJOR(p2.properties.apiVersion),
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
  VkPhysicalDeviceSynchronization2FeaturesKHR fs2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .pNext = want_hic ? &hicf : NULL};
  VkPhysicalDeviceDynamicRenderingFeaturesKHR fdr = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
      .pNext = &fs2};
  VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = vk12_compat ? (void *)&fdr : (void *)&f13};
  VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                  .pNext = &f12};
  vkGetPhysicalDeviceFeatures2(c->phys, &f2);
  bool have_sync2 = vk12_compat ? fs2.synchronization2 : f13.synchronization2;
  bool have_dynrender = vk12_compat ? fdr.dynamicRendering : f13.dynamicRendering;
  if (!f12.timelineSemaphore || !have_sync2 || !have_dynrender ||
      (!vk12_compat && !f13.maintenance4)) {
    fprintf(stderr, "vk: missing core features (timeline=%u sync2=%u dynrender=%u)\n",
            f12.timelineSemaphore, have_sync2, have_dynrender);
    return -1;
  }
  c->caps.host_image_copy = want_hic && hicf.hostImageCopy;
  if (getenv("R3D_NO_HOST_COPY") && *getenv("R3D_NO_HOST_COPY") == '1')
    c->caps.host_image_copy = false; /* conformance-test the staged fallback */
  c->caps.descriptor_indexing = f12.shaderSampledImageArrayNonUniformIndexing;
  c->caps.runtime_descriptor_array = f12.runtimeDescriptorArray;
  c->caps.memory_budget = dev_has_ext(c->phys, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

  VkPhysicalDeviceMemoryBudgetPropertiesEXT mb = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 mp2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
      .pNext = c->caps.memory_budget ? &mb : NULL};
  vkGetPhysicalDeviceMemoryProperties2(c->phys, &mp2);
  for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++) {
    if (!(mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
    VkDeviceSize hs = mp2.memoryProperties.memoryHeaps[i].size;
    if (hs <= c->caps.device_heap_bytes) continue;
    c->caps.device_heap_bytes = hs;
    c->caps.heap_budget_bytes = c->caps.memory_budget ? mb.heapBudget[i] : hs;
    c->caps.heap_usage_bytes = c->caps.memory_budget ? mb.heapUsage[i] : 0;
  }

  /* --- device --- */
  const char *dev_exts[6];
  uint32_t ndev_exts = 0;
  if (dev_has_ext(c->phys, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
    dev_exts[ndev_exts++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  if (c->caps.host_image_copy) dev_exts[ndev_exts++] = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME;
  if (c->caps.memory_budget) dev_exts[ndev_exts++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
  if (vk12_compat) {
    dev_exts[ndev_exts++] = VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME;
    dev_exts[ndev_exts++] = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
  }

  VkPhysicalDeviceHostImageCopyFeaturesEXT en_hic = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
      .hostImageCopy = VK_TRUE};
  VkPhysicalDeviceVulkan13Features en13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = c->caps.host_image_copy ? &en_hic : NULL,
      .synchronization2 = VK_TRUE,
      .maintenance4 = VK_TRUE,
      .dynamicRendering = VK_TRUE, /* GUI pass renders without render passes */
  };
  VkPhysicalDeviceSynchronization2FeaturesKHR en_s2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .pNext = c->caps.host_image_copy ? &en_hic : NULL,
      .synchronization2 = VK_TRUE};
  VkPhysicalDeviceDynamicRenderingFeaturesKHR en_dr = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
      .pNext = &en_s2,
      .dynamicRendering = VK_TRUE};
  VkPhysicalDeviceVulkan12Features en12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = vk12_compat ? (void *)&en_dr : (void *)&en13,
      .timelineSemaphore = VK_TRUE,
      .hostQueryReset = f12.hostQueryReset,
      /* slab tiles are dynamically indexed (a 544-case literal switch costs
       * ~20x per sample vs native non-uniform indexing on this hardware) */
      .shaderSampledImageArrayNonUniformIndexing = f12.shaderSampledImageArrayNonUniformIndexing,
      .runtimeDescriptorArray = f12.runtimeDescriptorArray,
      /* R3D_BINDLESS=1: descriptorIndexing makes Dozen go bindless. Measured
       * on the RTX 5080/WSL2: record 0.37 -> 2.07 ms per 4-dispatch frame and
       * no GPU win, so it stays opt-in. */
      .descriptorIndexing = f12.descriptorIndexing && getenv("R3D_BINDLESS") != NULL,
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
  if (pthread_mutex_init(&c->queue_mu, NULL) != 0) return -1;
  c->queue_mu_up = true;
  r3d_vkctx_set_budget(c, 0);
  return 0;
}

void r3d_vkctx_set_budget(r3d_vkctx *c, VkDeviceSize requested) {
  VkDeviceSize reported = c->caps.heap_budget_bytes;
  VkDeviceSize reserve = reported / 10;
  if (reserve < ((VkDeviceSize)1 << 30)) reserve = (VkDeviceSize)1 << 30;
  VkDeviceSize derived = reported > reserve ? reported - reserve : reported * 7 / 10;
  if (!c->caps.memory_budget) derived = c->caps.device_heap_bytes * 7 / 10;
  c->budget_bytes = requested && requested < derived ? requested : derived;
}

VkDeviceSize r3d_vkctx_budget_available(const r3d_vkctx *c) {
  VkDeviceSize used = atomic_load(&c->allocated_bytes);
  return used < c->budget_bytes ? c->budget_bytes - used : 0;
}

void r3d_vkctx_destroy(r3d_vkctx *c) {
  if (c->dev) vkDestroyDevice(c->dev, NULL);
  if (c->queue_mu_up) pthread_mutex_destroy(&c->queue_mu);
  if (c->messenger) {
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger =
        (PFN_vkDestroyDebugUtilsMessengerEXT)(void (*)(void))vkGetInstanceProcAddr(
            c->instance, "vkDestroyDebugUtilsMessengerEXT");
    if (destroy_messenger) destroy_messenger(c->instance, c->messenger, NULL);
  }
  if (c->instance) vkDestroyInstance(c->instance, NULL);
  memset(c, 0, sizeof *c);
}

VkResult r3d_vkctx_queue_submit(r3d_vkctx *c, uint32_t n, const VkSubmitInfo *info,
                                VkFence fence) {
  if (!c->queue_mu_up) return vkQueueSubmit(c->queue, n, info, fence);
  pthread_mutex_lock(&c->queue_mu);
  VkResult rc = vkQueueSubmit(c->queue, n, info, fence);
  pthread_mutex_unlock(&c->queue_mu);
  return rc;
}

VkResult r3d_vkctx_queue_submit2(r3d_vkctx *c, uint32_t n, const VkSubmitInfo2 *info,
                                 VkFence fence) {
  if (!c->queue_mu_up) return vkQueueSubmit2(c->queue, n, info, fence);
  pthread_mutex_lock(&c->queue_mu);
  VkResult rc = vkQueueSubmit2(c->queue, n, info, fence);
  pthread_mutex_unlock(&c->queue_mu);
  return rc;
}

VkResult r3d_vkctx_queue_present(r3d_vkctx *c, const VkPresentInfoKHR *info) {
  if (!c->queue_mu_up) return vkQueuePresentKHR(c->queue, info);
  pthread_mutex_lock(&c->queue_mu);
  VkResult rc = vkQueuePresentKHR(c->queue, info);
  pthread_mutex_unlock(&c->queue_mu);
  return rc;
}

VkResult r3d_vkctx_queue_wait_idle(r3d_vkctx *c) {
  if (!c->queue_mu_up) return vkQueueWaitIdle(c->queue);
  pthread_mutex_lock(&c->queue_mu);
  VkResult rc = vkQueueWaitIdle(c->queue);
  pthread_mutex_unlock(&c->queue_mu);
  return rc;
}

VkResult r3d_vkctx_device_wait_idle(r3d_vkctx *c) {
  if (!c->queue_mu_up) return vkDeviceWaitIdle(c->dev);
  pthread_mutex_lock(&c->queue_mu);
  VkResult rc = vkDeviceWaitIdle(c->dev);
  pthread_mutex_unlock(&c->queue_mu);
  return rc;
}

void r3d_vkctx_print_caps(const r3d_vkctx *c) {
  const r3d_vkcaps *k = &c->caps;
  printf("device            : %s (Vulkan %u.%u.%u, %04x:%08x)\n", k->dev_name,
         VK_API_VERSION_MAJOR(k->api_version), VK_API_VERSION_MINOR(k->api_version),
         VK_API_VERSION_PATCH(k->api_version), k->vendor_id, k->device_id);
  printf("queue family      : %u (graphics+compute, timestamps=%s, period %.2f ns)\n", c->qfam,
         k->timestamps ? "yes" : "no", k->ts_period_ns);
  printf("maxImageDimension3D    : %u\n", k->max_dim_3d);
  if (k->max_alloc_bytes == UINT64_MAX)
    printf("maxMemoryAllocation    : unbounded (driver sentinel)\n");
  else
    printf("maxMemoryAllocation    : %.2f GiB\n", (double)k->max_alloc_bytes / (1ull << 30));
  printf("device-local heap      : %.2f GiB\n", (double)k->device_heap_bytes / (1ull << 30));
  printf("heap budget/usage      : %.2f / %.2f GiB%s\n",
         (double)k->heap_budget_bytes / (1ull << 30),
         (double)k->heap_usage_bytes / (1ull << 30), k->memory_budget ? "" : " (estimated)");
  printf("renderer budget        : %.2f GiB\n", (double)c->budget_bytes / (1ull << 30));
  printf("descriptor indexing    : nonuniform=%s runtime-array=%s, "
         "max combined images %u, stage resources %u\n",
         k->descriptor_indexing ? "yes" : "no", k->runtime_descriptor_array ? "yes" : "no",
         k->max_sampled_images, k->max_stage_resources);
  if (k->max_allocations == UINT32_MAX)
    printf("max memory allocations : unbounded (driver sentinel)\n");
  else
    printf("max memory allocations : %u\n", k->max_allocations);
  printf("maxPushConstants       : %u B\n", k->max_push_bytes);
  printf("maxWorkgroupInvocations: %u (subgroup %u)\n", k->max_wg_invocations, k->subgroup_size);
  printf("uniformBufferAlignment : %llu B\n",
         (unsigned long long)k->min_ubo_alignment);
  printf("host_image_copy        : %s\n", k->host_image_copy ? "yes" : "no");
  VkFormatFeatureFlags r8 = k->r8_optimal;
  printf("R8_UNORM optimal       :%s%s%s%s%s\n",
         (r8 & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? " sampled" : "",
         (r8 & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? " linear-filter" : "",
         (r8 & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? " storage" : "",
         (r8 & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? " blit-src" : "",
         (r8 & VK_FORMAT_FEATURE_BLIT_DST_BIT) ? " blit-dst" : "");
  uint32_t vol = 1024;
  VkDeviceSize need = (VkDeviceSize)1228 << 20; /* R8 mip chain + conservative alignment */
  bool ok = vol <= k->max_dim_3d && need <= k->max_alloc_bytes && need <= c->budget_bytes;
  printf("verdict: 1024^3 R8 + mips %s (dim %u/%u, estimate %.2f GiB, budget %.2f GiB)\n",
         ok ? "OK" : "UNSUPPORTED", vol, k->max_dim_3d, (double)need / (1ull << 30),
         (double)c->budget_bytes / (1ull << 30));
}
