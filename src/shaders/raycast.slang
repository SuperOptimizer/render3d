// Volume raycaster: fixed-step marching, two data modes.
//  - cube: one monolithic 3D texture in [0,1]^3 (volumes[0], all 4 slots alias)
//  - slab: up to 2x2 tile textures (payload + 1-voxel apron) covering a wide
//    thin z-window; z is a ring buffer sampled with a REPEAT-W sampler
//    (see src/core/slab.h for the layout math mirrored here).
import common;

// slab tiles (element j*4+i, up to 4x4); cube mode binds the monolithic x16
[[vk::binding(0, 0)]] Sampler3D<float> volumes[16];
[[vk::binding(1, 0)]] Sampler1D<float4> tf;
[[vk::binding(2, 0)]] RWTexture2D<float4> out_image;
// per-8^3-block max, dilated 1 block (nearest sampler) — empty-space skipping
[[vk::binding(3, 0)]] Sampler3D<float> occ;

static const uint MODE_FULL = 0;
static const uint MODE_MIP = 1;
static const uint MODE_DEPTH = 2;
static const uint MODE_HEATMAP = 3;
static const uint MODE_RAYDIR = 4;
static const uint MODE_FLAT = 5; // full compositing, shading off (perf A/B + tests)
static const uint MAX_STEPS = 4096; // hard driver-safety clamp
static const float ALPHA_CUTOFF = 0.98;
static const float OCC_BLOCK = 8.0;

bool is_slab() { return pc.slab_grid != 0; }
bool is_clip() { return pc.clip_valid != 0; }

// mirror of core/clip.h ring sizing: depth_max/2^l + 3, min 4
float clip_wzl(uint l) {
  uint w = (pc.slab_wz >> l) + 3;
  return float(max(w, 4u));
}

// World box the ray marches: cube = [0,1]^3; slab = thin box whose z spans the
// wz-1 seam-safe slices, all in units of 1/nx.
float3 vol_extent() {
  if (!is_slab() && !is_clip()) return float3(1);
  // box is exactly slab_depth voxels thick; the ring holds depth+2 slices so
  // filtering at the faces never crosses the ring seam
  return float3(1.0, pc.slab_ny / pc.slab_nx, float(pc.slab_depth) / pc.slab_nx);
}

float sample_tile(float3 uvw, uint idx) {
  switch (idx) { // literal indices: no descriptor-indexing features needed
  case 1: return volumes[1].SampleLevel(uvw, 0);
  case 2: return volumes[2].SampleLevel(uvw, 0);
  case 3: return volumes[3].SampleLevel(uvw, 0);
  case 4: return volumes[4].SampleLevel(uvw, 0);
  case 5: return volumes[5].SampleLevel(uvw, 0);
  case 6: return volumes[6].SampleLevel(uvw, 0);
  case 7: return volumes[7].SampleLevel(uvw, 0);
  case 8: return volumes[8].SampleLevel(uvw, 0);
  case 9: return volumes[9].SampleLevel(uvw, 0);
  case 10: return volumes[10].SampleLevel(uvw, 0);
  case 11: return volumes[11].SampleLevel(uvw, 0);
  case 12: return volumes[12].SampleLevel(uvw, 0);
  case 13: return volumes[13].SampleLevel(uvw, 0);
  case 14: return volumes[14].SampleLevel(uvw, 0);
  case 15: return volumes[15].SampleLevel(uvw, 0);
  default: return volumes[0].SampleLevel(uvw, 0);
  }
}

// --- clip mode: level pick is separated from sampling so the marcher picks
// once per sample and gradient taps reuse the level (7x fewer walks) ---

float sample_clip_level(uint l, float3 p) {
  float s = float(1u << l);
  float wx = p.x * pc.slab_nx, wy = p.y * pc.slab_nx;
  float zw = pc.slab_z0 + 0.5 + p.z * pc.slab_nx;
  float3 uvw = float3((wx - pc.clip_orig[l].x) / s + 1.0, (wy - pc.clip_orig[l].y) / s + 1.0,
                      0.0) /
               float3(2048.0, 2048.0, 1.0);
  uvw.z = frac(zw / s / clip_wzl(l)); // REPEAT-W ring per level
  switch (l) {                        // literal indices, 6 clip levels only
  case 0: return volumes[0].SampleLevel(uvw, 0);
  case 1: return volumes[1].SampleLevel(uvw, 0);
  case 2: return volumes[2].SampleLevel(uvw, 0);
  case 3: return volumes[3].SampleLevel(uvw, 0);
  case 4: return volumes[4].SampleLevel(uvw, 0);
  default: return volumes[5].SampleLevel(uvw, 0);
  }
}

// Finest usable level for p (coarser-preferred from `desired`); -1 = none.
int pick_clip_level(float3 p, float lod) {
  float wx = p.x * pc.slab_nx, wy = p.y * pc.slab_nx;
  int desired = int(clamp(lod, 0.0, 5.0));
  for (int step = 0; step < 6; step++) {
    int li = desired + step;
    if (li > 5) li = 5 - step; /* wrapped: try coarser first, then finer */
    if (li < 0) continue;
    if ((pc.clip_valid & (1u << uint(li))) == 0) continue;
    float s = float(1u << uint(li));
    float tx = (wx - pc.clip_orig[li].x) / s + 1.0;
    float ty = (wy - pc.clip_orig[li].y) / s + 1.0;
    if (tx < 1.0 || tx >= 2047.0 || ty < 1.0 || ty >= 2047.0) continue;
    return li;
  }
  return -1;
}

// p in world units (see vol_extent).
float vol_sample(float3 p, float lod) {
  if (is_clip()) {
    int l = pick_clip_level(p, lod);
    return l < 0 ? 0.0 : sample_clip_level(uint(l), p);
  }
  if (!is_slab()) return volumes[0].SampleLevel(p, lod);
  // voxel coords (units of 1/nx per axis)
  float wx = p.x * pc.slab_nx;
  float wy = p.y * pc.slab_nx;
  float zc = pc.slab_z0 + 0.5 + p.z * pc.slab_nx; // ring z, seam-safe by box extent
  float w = frac(zc / float(pc.slab_wz)); // REPEAT sampler wraps the ring
  if (lod >= 1.75) {
    // far field: 1/4-res overview of the whole composite (binding 3 in slab
    // mode) — pre-filtered, so wide zoom-outs neither alias nor cost full res
    float ox = ceil(pc.slab_nx * 0.25), oy = ceil(pc.slab_ny * 0.25);
    return occ.SampleLevel(float3(wx / (4.0 * ox), wy / (4.0 * oy), w), 0);
  }
  uint gx = pc.slab_grid & 0xffu, gy = pc.slab_grid >> 8;
  uint ix = min(uint(max(wx, 0.0) / pc.slab_px), gx - 1);
  uint iy = min(uint(max(wy, 0.0) / pc.slab_py), gy - 1);
  float tw = pc.slab_px + 2.0, th = pc.slab_py + 2.0;
  float u = (wx - float(ix) * pc.slab_px + 1.0) / tw;
  float v = (wy - float(iy) * pc.slab_py + 1.0) / th;
  return sample_tile(float3(u, v, w), iy * 4 + ix);
}

float3 ray_dir(uint2 pix) {
  float2 uv = (float2(pix) + 0.5) / float2(pc.viewport);
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return normalize(pc.cam_forward + ndc.x * pc.cam_right + ndc.y * pc.cam_up);
}

// slab test against [0, ext]; returns (tnear, tfar), tnear<=tfar means hit
float2 box_hit(float3 o, float3 d, float3 ext) {
  float3 inv = 1.0 / d;
  float3 t0 = (float3(0) - o) * inv;
  float3 t1 = (ext - o) * inv;
  float3 tmin = min(t0, t1), tmax = max(t0, t1);
  return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// Workgroup size: 16x8 = one Adreno wave (subgroup 128). An 8x8 variant is
// built alongside for the occupancy sweep (R3D_WG env; docs/measured.md).
#ifndef WG_X
#define WG_X 16
#define WG_Y 8
#endif

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
  if (tid.x >= pc.viewport.x || tid.y >= pc.viewport.y) return;
  float3 dir = ray_dir(tid.xy);

  if (pc.mode == MODE_RAYDIR) {
    out_image[tid.xy] = float4(dir * 0.5 + 0.5, 1);
    return;
  }

  float voxel;
  float max_lod;
  if (is_clip()) {
    voxel = 1.0 / pc.slab_nx;
    max_lod = 7.0; // levels end at 5; extra lod keeps widening far-field steps
  } else if (is_slab()) {
    voxel = 1.0 / pc.slab_nx;
    max_lod = 4.0; // lod >= 1.75 switches to the 1/4-res overview texture
  } else {
    uint3 vdim;
    uint levels;
    volumes[0].GetDimensions(0, vdim.x, vdim.y, vdim.z, levels);
    voxel = 1.0 / float(max(vdim.x, max(vdim.y, vdim.z)));
    max_lod = min(pc.max_mip, float(levels - 1));
  }
  float3 ext = vol_extent();
  /* move the ray into volume space: the model transform rotates the box about
   * its center and translates it; marching/shading all happen in this space */
  float3 ro = to_volume_point(pc.cam_origin, ext * 0.5);
  dir = to_volume_dir(dir);

  float2 hit = box_hit(ro, dir, ext);
  float tn = max(hit.x, 0.0), tf_ = hit.y;
  if (tn >= tf_) {
    out_image[tid.xy] = float4(0.02, 0.02, 0.03, 1); // background
    return;
  }

  // ray-cone footprint: pixel angular size (cam_up is pre-scaled by tan(fov/2))
  float cone = 2.0 * length(pc.cam_up) / float(pc.viewport.y);

  float dt = pc.step_voxels * voxel;
  float jitter = hash12(tid.xy, pc.frame_index);
  float t = tn + jitter * dt;

  float mip_max = 0.0;   // MODE_MIP accumulator
  float first_hit = -1.0;
  float4 acc = float4(0); // MODE_FULL front-to-back accumulator
  uint steps = 0;
  bool composite = pc.mode == MODE_FULL || pc.mode == MODE_FLAT;
  bool shaded = pc.mode == MODE_FULL;

  float bs = OCC_BLOCK * voxel;                     // skip-block edge, volume units
  float occ_gate = max(pc.threshold, 0.5 / 255.0);  // block max below this = void

  while (t < tf_ && steps < MAX_STEPS) {
    steps++;
    // LOD-on-distance: mip where the ray cone's width matches the voxel pitch;
    // step size scales with it so far-field marching is proportionally cheap.
    float lod = clamp(log2(max(t * cone / voxel, 1.0)) + pc.lod_bias, 0.0, max_lod);
    float lod_step = exp2(lod); /* continuous: fractional lod still widens dt */
    if (is_slab() || is_clip()) // keep >=4 samples along THIS ray's box segment
      lod_step = min(lod_step, max(1.0, (tf_ - tn) / (4.0 * dt)));
    float3 p = ro + t * dir;

    int cl_lvl = -1; // clip: pick the level once; gradient taps reuse it
    float v;
    if (is_clip()) {
      cl_lvl = pick_clip_level(p, lod);
      v = cl_lvl < 0 ? 0.0 : sample_clip_level(uint(cl_lvl), p);
    } else {
      v = vol_sample(p, lod);
    }
    bool cut = v < pc.threshold; // low-cut: below-threshold voxels are void
    if (cut) v = 0.0;

    // empty-space skip (cube mode; slab occ is a dummy full texture), checked
    // only where the fine sample is already void so dense marching pays nothing
    if (!is_slab() && v < occ_gate && occ.SampleLevel(p, 0) < occ_gate) {
      float3 nextp = (floor(p / bs) + max(sign(dir), float3(0))) * bs;
      float3 tex = select(abs(dir) > 1e-8, (nextp - ro) / dir, float3(1e30));
      t = max(min(min(tex.x, tex.y), tex.z), t) + bs * 0.01;
      continue;
    }

    if (v > mip_max) mip_max = v;
    if (pc.mode == MODE_MIP && mip_max >= 0.999) break; // saturated: nothing to gain
    if (first_hit < 0.0 && v > 0.05) first_hit = t;

    if (composite && !cut) {
      float4 s = tf.SampleLevel(v, 0);
      // opacity correction: appearance invariant under step-size changes
      float a = 1.0 - pow(1.0 - s.a, pc.density * pc.step_voxels * lod_step);
      if (a > 0.003) {
        float3 col = s.rgb;
        if (shaded) {
          // gradient-modulated Blinn-Phong; homogeneous regions stay unshaded
          float e = voxel * lod_step;
          float3 g;
          if (cl_lvl >= 0) {
            uint gl = uint(cl_lvl);
            g = float3(sample_clip_level(gl, p + float3(e, 0, 0)) -
                           sample_clip_level(gl, p - float3(e, 0, 0)),
                       sample_clip_level(gl, p + float3(0, e, 0)) -
                           sample_clip_level(gl, p - float3(0, e, 0)),
                       sample_clip_level(gl, p + float3(0, 0, e)) -
                           sample_clip_level(gl, p - float3(0, 0, e)));
          } else {
            g = float3(
                vol_sample(p + float3(e, 0, 0), lod) - vol_sample(p - float3(e, 0, 0), lod),
                vol_sample(p + float3(0, e, 0), lod) - vol_sample(p - float3(0, e, 0), lod),
                vol_sample(p + float3(0, 0, e), lod) - vol_sample(p - float3(0, 0, e), lod));
          }
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
