// Volume raycaster: fixed-step marching through a monolithic R8 3D texture.
// M1 assumes a cubic volume in [0,1]^3 (spec/volume.md). The loop is written
// so a later outer brick-DDA can wrap the inner fixed-step march unchanged.
import common;

[[vk::binding(0, 0)]] Sampler3D<float> volume;
[[vk::binding(1, 0)]] Sampler1D<float4> tf;
[[vk::binding(2, 0)]] RWTexture2D<float4> out_image;

static const uint MODE_FULL = 0;
static const uint MODE_MIP = 1;
static const uint MODE_DEPTH = 2;
static const uint MODE_HEATMAP = 3;
static const uint MODE_RAYDIR = 4;
static const uint MAX_STEPS = 4096; // hard driver-safety clamp
static const float ALPHA_CUTOFF = 0.98;

float3 ray_dir(uint2 pix) {
  float2 uv = (float2(pix) + 0.5) / float2(pc.viewport);
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return normalize(pc.cam_forward + ndc.x * pc.cam_right + ndc.y * pc.cam_up);
}

// slab test against [0,1]^3; returns (tnear, tfar), tnear<=tfar means hit
float2 box_hit(float3 o, float3 d) {
  float3 inv = 1.0 / d;
  float3 t0 = (float3(0) - o) * inv;
  float3 t1 = (float3(1) - o) * inv;
  float3 tmin = min(t0, t1), tmax = max(t0, t1);
  return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
  if (tid.x >= pc.viewport.x || tid.y >= pc.viewport.y) return;
  float3 dir = ray_dir(tid.xy);

  if (pc.mode == MODE_RAYDIR) {
    out_image[tid.xy] = float4(dir * 0.5 + 0.5, 1);
    return;
  }

  uint3 vdim;
  uint levels;
  volume.GetDimensions(0, vdim.x, vdim.y, vdim.z, levels);
  float voxel = 1.0 / float(max(vdim.x, max(vdim.y, vdim.z)));

  float2 hit = box_hit(pc.cam_origin, dir);
  float tn = max(hit.x, 0.0), tf_ = hit.y;
  if (tn >= tf_) {
    out_image[tid.xy] = float4(0.02, 0.02, 0.03, 1); // background
    return;
  }

  float dt = pc.step_voxels * voxel;
  float jitter = hash12(tid.xy, pc.frame_index);
  float t = tn + jitter * dt;

  float mip_max = 0.0;   // MODE_MIP accumulator
  float first_hit = -1.0;
  uint steps = 0;

  for (; t < tf_ && steps < MAX_STEPS; steps++) {
    float lod = clamp(pc.lod_bias, 0.0, pc.max_mip);
    float3 p = pc.cam_origin + t * dir;
    float v = volume.SampleLevel(p, lod);
    if (v > mip_max) mip_max = v;
    if (first_hit < 0.0 && v > 0.05) first_hit = t;
    t += dt;
  }

  float3 color;
  if (pc.mode == MODE_HEATMAP) {
    float x = float(steps) / 1024.0;
    color = float3(x, 1.0 - abs(x - 0.5) * 2.0, 1.0 - x); // blue->green->red
  } else if (pc.mode == MODE_DEPTH) {
    color = float3(first_hit < 0.0 ? 0.0 : 1.0 - saturate(first_hit / 2.0));
  } else { // MODE_MIP (and MODE_FULL until step 9)
    color = float3(mip_max * pc.density);
  }
  out_image[tid.xy] = float4(color, 1);
}
