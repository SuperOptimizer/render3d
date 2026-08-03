// Volume raycaster. M1 step 2/6 placeholder: animated UV pattern proving the
// slangc -> SPIR-V -> compute -> storage-image path; the marcher lands in
// steps 8-11 per the milestone plan.
import common;

[[vk::binding(2, 0)]] RWTexture2D<float4> out_image;

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
  if (tid.x >= pc.viewport.x || tid.y >= pc.viewport.y) return;
  float2 uv = float2(tid.xy) / float2(pc.viewport);
  float t = float(pc.frame_index) * 0.02;
  float3 c = 0.5 + 0.5 * cos(t + uv.xyx * 6.28318 + float3(0, 2, 4));
  out_image[tid.xy] = float4(c, 1);
}
