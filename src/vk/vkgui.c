#include "vk/vkgui.h"

#include <stdio.h>

/* CIMGUI_DEFINE_ENUMS_AND_STRUCTS + CIMGUI_USE_SDL3/VULKAN come from the build */
#include "cimgui.h"
#include "cimgui_impl.h"

static VkFormat g_color_format; /* must outlive Init (pipeline info points at it) */
static bool g_headless;         /* no SDL platform backend: synthetic io */
static float g_hw = 1280.0f, g_hh = 720.0f;

void r3d_vkgui_set_display(uint32_t w, uint32_t h) {
  g_hw = (float)(w ? w : 1u);
  g_hh = (float)(h ? h : 1u);
}

int r3d_vkgui_init(r3d_vkctx *c, SDL_Window *win, VkFormat color_format, uint32_t image_count) {
  g_color_format = color_format;
  g_headless = win == NULL;
  igCreateContext(NULL);
  if (g_headless) {
    ImGuiIO *io = igGetIO_Nil();
    io->DisplaySize = (ImVec2){g_hw, g_hh};
    io->DeltaTime = 1.0f / 60.0f;
    /* the Vulkan backend needs the font atlas either way */
  } else if (!ImGui_ImplSDL3_InitForVulkan(win)) {
    fprintf(stderr, "vkgui: SDL3 backend init failed\n");
    return -1;
  }
  ImGui_ImplVulkan_InitInfo ii = {
      .ApiVersion = VK_API_VERSION_1_3,
      .Instance = c->instance,
      .PhysicalDevice = c->phys,
      .Device = c->dev,
      .QueueFamily = c->qfam,
      .Queue = c->queue,
      .DescriptorPool = VK_NULL_HANDLE,
      .DescriptorPoolSize = 64, /* backend creates its own pool */
      .MinImageCount = 2,
      .ImageCount = image_count,
      .UseDynamicRendering = true,
      .PipelineInfoMain = {
          .PipelineRenderingCreateInfo = {
              .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
              .colorAttachmentCount = 1,
              .pColorAttachmentFormats = &g_color_format,
          },
      },
  };
  if (!ImGui_ImplVulkan_Init(&ii)) {
    fprintf(stderr, "vkgui: Vulkan backend init failed\n");
    return -1;
  }
  return 0;
}

void r3d_vkgui_shutdown(void) {
  ImGui_ImplVulkan_Shutdown();
  if (!g_headless) ImGui_ImplSDL3_Shutdown();
  igDestroyContext(NULL);
}

void r3d_vkgui_event(const SDL_Event *ev) {
  if (!g_headless) ImGui_ImplSDL3_ProcessEvent(ev);
}

void r3d_vkgui_new_frame(void) {
  ImGui_ImplVulkan_NewFrame();
  if (g_headless) {
    ImGuiIO *io = igGetIO_Nil();
    io->DisplaySize = (ImVec2){g_hw, g_hh};
    io->DeltaTime = 1.0f / 60.0f;
  } else {
    ImGui_ImplSDL3_NewFrame();
  }
  igNewFrame();
}

void r3d_vkgui_discard(void) { igEndFrame(); }

void r3d_vkgui_render(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent) {
  igRender();
  VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, /* composite over the raycast blit */
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
  };
  VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {.extent = extent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &att,
  };
  vkCmdBeginRendering(cmd, &ri);
  ImGui_ImplVulkan_RenderDrawData(igGetDrawData(), cmd, VK_NULL_HANDLE);
  vkCmdEndRendering(cmd);
}
