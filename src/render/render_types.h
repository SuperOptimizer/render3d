/* POD types crossing the core <-> backend boundary. r3d_frame_params is copied
 * to the per-frame uniform-buffer ring and must mirror FrameParams in
 * src/shaders/common.slang byte-for-byte (float3 aligns to 16). */
#ifndef R3D_RENDER_TYPES_H
#define R3D_RENDER_TYPES_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct r3d_config {
  bool validate;       /* Vulkan validation layer + debug messenger */
  bool vsync;          /* FIFO (true) vs MAILBOX/IMMEDIATE if available */
  const char *spv_dir; /* directory holding compiled .spv shaders */
  uint64_t gpu_budget_bytes; /* 0 = derive conservatively from the device heap */
  /* headless: no window, surface, swapchain or present. Frames render into
   * the offscreen only (r3d_read_frame captures it); ImGui runs without a
   * platform backend so the panel code still executes. For automated perf
   * runs and CI on machines without a display. */
  bool headless;
  uint32_t headless_w, headless_h;
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

enum {
  R3D_QUALITY_FULL = 0, /* six-tap central-difference gradient */
  R3D_QUALITY_FAST = 1, /* four-tap tetrahedral gradient */
  R3D_QUALITY_COUNT = 2,
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
  /* virtual-slab mode (slab_grid bit 24): window world origin x/y (z is
   * slab_z0); coords <= 43008 are exact in f32 */
  float slab_x0, slab_y0;
  /* multi-view: offscreen pixel origin of this viewport (x | y<<16) */
  uint32_t view_org;
  /* bit 0: orthographic rays (cam_right/up lengths = half-extents in box
   * units, cam_forward unit); bit 2: clip the slab along the ray instead of
   * a world axis (slab_z0 = voxels from the ray origin to the slab near
   * face); bits 8..9: bricks slab-clip axis 0=z 1=x 2=y */
  uint32_t view_flags;
  /* overlay volume (ink predictions): bit 0 of flags enables the tint,
   * gain scales the overlay value before it drives the blend */
  float overlay_gain;
  uint32_t overlay_flags;
} r3d_frame_params;
static_assert(sizeof(r3d_frame_params) == 248, "must mirror shader FrameParams");

#define R3D_VIEW_ORTHO 1u
#define R3D_VIEW_SURF 2u /* flattened tifxyz segment view (needs r3d_surf_begin) */
#define R3D_VIEW_OBLIQUE 4u /* slab clip along the ray, not a world axis */
#define R3D_VIEW_STRETCH 8u /* surf view: flattening-distortion heatmap */
#define R3D_VIEW_CROP 16u   /* bricks: clip to the cube at slab_x0/y0/z0,
                             * edge slab_depth voxels (3D pane zoom-crop) */
#define R3D_VIEW_HALF 32u   /* interaction: one ray per 2x2 block, written to
                             * all four pixels (dispatch at half size) */
#define R3D_VIEW_AXIS(a) ((uint32_t)(a) << 8) /* 0=z 1=x 2=y */
#define R3D_MAX_VIEWS 4u

typedef struct r3d_frame_stats {
  /* GPU zones from timestamp queries (0 if unsupported); lag 2 frames */
  uint64_t gpu_ns;         /* whole submitted command buffer */
  uint64_t gpu_raycast_ns; /* compute dispatch */
  uint64_t gpu_blit_ns;    /* offscreen -> swapchain blit */
  uint64_t gpu_gui_ns;     /* ImGui color pass (0 when no GUI drawn) */
  uint32_t panes_drawn;    /* views re-raycast this frame (pane cache misses) */
  /* CPU phases inside r3d_frame, current frame */
  uint64_t cpu_wait_ns;    /* timeline wait for slot reuse */
  uint64_t cpu_acquire_ns; /* vkAcquireNextImageKHR */
  uint64_t cpu_record_ns;  /* command buffer recording */
  uint64_t cpu_submit_ns;  /* queue submit + present */
} r3d_frame_stats;

#endif /* R3D_RENDER_TYPES_H */
