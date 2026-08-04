// Shared shader definitions. FrameParams must mirror r3d_frame_params in
// src/render/render_types.h byte-for-byte (guarded by _Static_assert there).
// Push-constant budget: 128 bytes (Vulkan guaranteed minimum).

struct FrameParams {
  float3 cam_origin;   float step_voxels;   // ray origin (volume-normalized)
  float3 cam_right;    float density;       // basis pre-scaled by tan(fov/2)*aspect
  float3 cam_up;       float lod_bias;      // basis pre-scaled by tan(fov/2)
  float3 cam_forward;  float max_mip;
  uint2 viewport;                            // drawable size in pixels
  uint mode;                                 // 0=full 1=MIP 2=depth 3=heatmap 4=raydir
  uint frame_index;
  float threshold;                           // voxels below this (0..1) are zeroed/transparent
  // slab mode (tiled thin z-window); slab_grid == 0 selects cube mode
  uint slab_grid;                            // gx | gy<<8
  uint slab_wz;                              // ring depth (slices)
  float slab_z0;                             // window start slice
  float slab_nx, slab_ny;                    // source xy dims (voxels)
  float slab_px, slab_py;                    // tile payload (voxels)
};

[[vk::push_constant]] FrameParams pc;

// Cheap per-pixel hash for jitter (PCG-ish); stable across drivers.
float hash12(uint2 p, uint frame) {
  uint h = p.x * 374761393u + p.y * 668265263u + frame * 2246822519u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return float(h >> 8) * (1.0 / 16777216.0);
}
