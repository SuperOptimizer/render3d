/* POD types crossing the core <-> backend boundary. r3d_frame_params is the
 * push-constant block and must mirror FrameParams in src/shaders/common.slang
 * byte-for-byte (std430 rules: float3 aligns to 16). */
#ifndef R3D_RENDER_TYPES_H
#define R3D_RENDER_TYPES_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct r3d_config {
  bool validate;       /* Vulkan validation layer + debug messenger */
  bool vsync;          /* FIFO (true) vs MAILBOX/IMMEDIATE if available */
  const char *spv_dir; /* directory holding compiled .spv shaders */
} r3d_config;

/* Debug/render modes (mirrors shader `pc.mode`). */
enum {
  R3D_MODE_FULL = 0,
  R3D_MODE_MIP = 1,     /* max-intensity projection */
  R3D_MODE_DEPTH = 2,   /* first-hit depth */
  R3D_MODE_HEATMAP = 3, /* step-count heatmap */
  R3D_MODE_RAYDIR = 4,  /* ray-direction color (wiring check) */
  R3D_MODE_FLAT = 5,    /* full compositing, shading off (perf A/B + tests) */
  R3D_MODE_COUNT = 6,
};

typedef struct r3d_volume_desc {
  uint32_t nx, ny, nz;
  uint32_t brick_dim; /* 128; carried now so streaming doesn't change the ABI */
  float voxel_um;
} r3d_volume_desc;

typedef struct r3d_frame_params {
  float cam_origin[3];  float step_voxels;
  float cam_right[3];   float density;
  float cam_up[3];      float lod_bias;
  float cam_forward[3]; float max_mip;
  uint32_t viewport[2];
  uint32_t mode;
  uint32_t frame_index;
  float threshold; /* voxels below this (normalized 0..1) are zeroed/transparent */
} r3d_frame_params;
static_assert(sizeof(r3d_frame_params) == 84, "must mirror shader FrameParams");

typedef struct r3d_frame_stats {
  /* GPU zones from timestamp queries (0 if unsupported); lag 2 frames */
  uint64_t gpu_ns;         /* whole submitted command buffer */
  uint64_t gpu_raycast_ns; /* compute dispatch */
  uint64_t gpu_blit_ns;    /* offscreen -> swapchain blit */
  uint64_t gpu_gui_ns;     /* ImGui color pass (0 when no GUI drawn) */
  /* CPU phases inside r3d_frame, current frame */
  uint64_t cpu_wait_ns;    /* timeline wait for slot reuse */
  uint64_t cpu_acquire_ns; /* vkAcquireNextImageKHR */
  uint64_t cpu_record_ns;  /* command buffer recording */
  uint64_t cpu_submit_ns;  /* queue submit + present */
} r3d_frame_stats;

#endif /* R3D_RENDER_TYPES_H */
