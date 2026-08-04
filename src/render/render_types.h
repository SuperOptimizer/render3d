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
  /* slab mode (tiled thin window); slab_grid == 0 selects cube mode */
  uint32_t slab_grid; /* gx | gy<<8 */
  uint32_t slab_wz;   /* ring depth (slices) */
  float slab_z0;      /* window start slice */
  float slab_nx, slab_ny; /* source xy dims (voxels) */
  float slab_px, slab_py; /* tile payload (voxels) */
  uint32_t slab_depth;    /* rendered thickness in voxels (<= ring - 2) */
  /* clip mode (nested-octave clipmap); clip_valid == 0 selects slab/cube.
   * Reuses slab_nx/ny (volume dims), slab_z0 (window start, world), slab_wz
   * (max depth -> per-level ring size), slab_depth (visible depth). */
  uint32_t clip_valid;    /* bit ℓ set = level ℓ filled and sampleable */
  float clip_orig[6][2];  /* world voxel of payload texel (0,0) per level */
  /* model (volume) transform: rays are moved into volume space by
   * p_v = R^T (p_w - vol_t - c) + c with c = extent/2. Rows of R here. */
  float vol_r0[3]; float vol_tx;
  float vol_r1[3]; float vol_ty;
  float vol_r2[3]; float vol_tz;
  /* bricks mode (c5d GPU-decoded atlas): 0 = off; else bpa | atlas_bpa<<8 */
  uint32_t brick_mode;
  /* skip gate (0..1): voxels below this are invisible under the CURRENT
   * transfer function + low-cut, so empty-space skipping may leap them.
   * Exact, not approximate: TF alpha is zero below it by construction. */
  float skip_gate;
} r3d_frame_params;
static_assert(sizeof(r3d_frame_params) == 224, "must mirror shader FrameParams");

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
