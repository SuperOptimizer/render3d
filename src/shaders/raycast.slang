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
static const uint MODE_FLAT = 5; // full compositing, shading off (perf A/B + tests)
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

  // ray-cone footprint: pixel angular size (cam_up is pre-scaled by tan(fov/2))
  float cone = 2.0 * length(pc.cam_up) / float(pc.viewport.y);
  float max_lod = min(pc.max_mip, float(levels - 1));

  float dt = pc.step_voxels * voxel;
  float jitter = hash12(tid.xy, pc.frame_index);
  float t = tn + jitter * dt;

  float mip_max = 0.0;   // MODE_MIP accumulator
  float first_hit = -1.0;
  float4 acc = float4(0); // MODE_FULL front-to-back accumulator
  uint steps = 0;
  bool composite = pc.mode == MODE_FULL || pc.mode == MODE_FLAT;
  bool shaded = pc.mode == MODE_FULL;

  for (; t < tf_ && steps < MAX_STEPS; steps++) {
    // LOD-on-distance: mip where the ray cone's width matches the voxel pitch;
    // step size scales with it so far-field marching is proportionally cheap.
    float lod = clamp(log2(max(t * cone / voxel, 1.0)) + pc.lod_bias, 0.0, max_lod);
    float lod_step = exp2(floor(lod));
    float3 p = pc.cam_origin + t * dir;
    float v = volume.SampleLevel(p, lod);
    if (v > mip_max) mip_max = v;
    if (first_hit < 0.0 && v > 0.05) first_hit = t;

    if (composite) {
      float4 s = tf.SampleLevel(v, 0);
      // opacity correction: appearance invariant under step-size changes
      float a = 1.0 - pow(1.0 - s.a, pc.density * pc.step_voxels * lod_step);
      if (a > 0.003) {
        float3 col = s.rgb;
        if (shaded) {
          // gradient-modulated Blinn-Phong; homogeneous regions stay unshaded
          float e = voxel * exp2(floor(lod));
          float3 g = float3(
              volume.SampleLevel(p + float3(e, 0, 0), lod) - volume.SampleLevel(p - float3(e, 0, 0), lod),
              volume.SampleLevel(p + float3(0, e, 0), lod) - volume.SampleLevel(p - float3(0, e, 0), lod),
              volume.SampleLevel(p + float3(0, 0, e), lod) - volume.SampleLevel(p - float3(0, 0, e), lod));
          float gm = length(g);
          if (gm > 1e-4) {
            float3 n = g / gm;
            float3 l = -dir; // headlight
            float3 key = normalize(float3(0.4, 0.8, -0.4));
            float diff = 0.7 * abs(dot(n, l)) + 0.3 * abs(dot(n, key));
            float spec = pow(abs(dot(n, normalize(l - dir))), 24.0) * 0.3;
            float lit = 0.25 + 0.75 * diff;
            float w = saturate(gm * 6.0); // gradient-magnitude modulation
            col = col * lerp(1.0, lit, w) + spec * w;
          }
        }
        acc.rgb += (1.0 - acc.a) * a * col;
        acc.a += (1.0 - acc.a) * a;
        if (acc.a >= ALPHA_CUTOFF) break;
      }
    }
    t += dt * lod_step;
  }

  float3 color;
  if (pc.mode == MODE_HEATMAP) {
    float x = float(steps) / 1024.0;
    color = float3(x, 1.0 - abs(x - 0.5) * 2.0, 1.0 - x); // blue->green->red
  } else if (pc.mode == MODE_DEPTH) {
    color = float3(first_hit < 0.0 ? 0.0 : 1.0 - saturate(first_hit / 2.0));
  } else if (composite) {
    color = acc.rgb + (1.0 - acc.a) * float3(0.02, 0.02, 0.03);
  } else { // MODE_MIP
    color = float3(mip_max * pc.density);
  }
  out_image[tid.xy] = float4(color, 1);
}
