// Volume raycaster. Volume marcher lands in M1 steps 8-11; currently renders
// camera-driven debug modes to prove the push-constant/camera wiring.
import common;

[[vk::binding(2, 0)]] RWTexture2D<float4> out_image;

static const uint MODE_FULL = 0;
static const uint MODE_MIP = 1;
static const uint MODE_DEPTH = 2;
static const uint MODE_HEATMAP = 3;
static const uint MODE_RAYDIR = 4;

float3 ray_dir(uint2 pix) {
  float2 uv = (float2(pix) + 0.5) / float2(pc.viewport);
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return normalize(pc.cam_forward + ndc.x * pc.cam_right + ndc.y * pc.cam_up);
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
  if (tid.x >= pc.viewport.x || tid.y >= pc.viewport.y) return;
  float3 dir = ray_dir(tid.xy);

  float3 color;
  if (pc.mode == MODE_RAYDIR) {
    color = dir * 0.5 + 0.5;
  } else {
    // placeholder animated pattern until the marcher lands
    float2 uv = float2(tid.xy) / float2(pc.viewport);
    float t = float(pc.frame_index) * 0.02;
    color = 0.5 + 0.5 * cos(t + uv.xyx * 6.28318 + float3(0, 2, 4));
  }
  out_image[tid.xy] = float4(color, 1);
}
