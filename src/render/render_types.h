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
} r3d_frame_params;
static_assert(sizeof(r3d_frame_params) == 80, "must mirror shader FrameParams");

typedef struct r3d_frame_stats {
  uint64_t gpu_ns; /* raycast dispatch time (0 if timestamps unsupported) */
} r3d_frame_stats;

#endif /* R3D_RENDER_TYPES_H */
