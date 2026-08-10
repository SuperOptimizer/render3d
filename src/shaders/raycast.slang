// Volume raycaster: fixed-step marching, two data modes.
//  - cube: one monolithic 3D texture in [0,1]^3 (volumes[0], all 4 slots alias)
//  - slab: up to 22x22 tile textures (payload + 1-voxel apron) covering a wide
//    thin z-window; z is a ring buffer sampled with a REPEAT-W sampler
//    (see src/core/slab.h for the layout math mirrored here).
import common;

// slab tiles: base grid (compact j*gx+i) then overview-pyramid levels;
// cube mode binds the monolithic view in every slot
[[vk::binding(0, 0)]] Sampler3D<float> volumes[1024];
[[vk::binding(1, 0)]] Sampler1D<float4> tf;
[[vk::binding(2, 0)]] RWTexture2D<float4> out_image;
// per-8^3-block max, dilated 1 block (nearest sampler) — empty-space skipping
[[vk::binding(3, 0)]] Sampler3D<float> occ;
// bricks mode: c5d GPU-decoded brick atlas + page table
// page[bi] = slot_index | brick_max<<24, 0xFFFFFFFF = not resident
[[vk::binding(4, 0)]] Sampler3D<float> brick_atlas;
[[vk::binding(5, 0)]] StructuredBuffer<uint> page;
// surf mode (R3D_MODE 5): tifxyz segment grid — volume-voxel coords
// (w = 1 valid / 0 invalid) and unit surface normals, manually bilerped so
// invalid grid points poison their quad (vc3d semantics)
[[vk::binding(7, 0)]] Texture2D<float4> surf_coords;
[[vk::binding(8, 0)]] Texture2D<float4> surf_normals;

static const uint MODE_FULL = 0;
static const uint MODE_MIP = 1;
static const uint MODE_DEPTH = 2;
static const uint MODE_HEATMAP = 3;
static const uint MODE_RAYDIR = 4;
static const uint MODE_FLAT = 5; // full compositing, shading off (perf A/B + tests)
static const uint MAX_STEPS = 4096; // hard driver-safety clamp
static const float ALPHA_CUTOFF = 0.98;
static const float OCC_BLOCK = 8.0;

// Mode is a COMPILE-TIME specialization (slangc -DR3D_MODE=n): carrying all
// four sampling architectures in one kernel cost 6x in register pressure.
// 0 = cube, 1 = slab, 2 = clip, 3 = bricks; the backend picks the pipeline.
#ifndef R3D_MODE
#define R3D_MODE 0
#endif
#ifndef R3D_FAST_GRADIENT
#define R3D_FAST_GRADIENT 0
#endif
bool is_slab() { return R3D_MODE == 1; }
bool is_clip() { return R3D_MODE == 2; }
bool is_bricks() { return R3D_MODE == 3; }
bool is_vslab() { return R3D_MODE == 4; } /* virtual slab: toroidal window */
bool is_surf() { return R3D_MODE == 5; }  /* flattened tifxyz segment */

/* bricks mode: one page-table resolve per sample position; the fetch (and
 * the 6 gradient taps reusing the context) are then plain atlas samples. */
struct BrickCtx {
  bool ok;
  float maxv;     /* brick max (page high byte) for empty skip */
  float3 bl;      /* voxel coords local to the brick */
  float3 slot_org; /* atlas mip0 texel origin of the slot */
  float atlas_lod; /* residual filtering after choosing an on-disk LOD */
  float grad_vox;  /* one base-space gradient radius in chosen-level voxels */
  float world_brick; /* chosen brick edge in renderer/world units */
};

// slot geometry: 132^3 = 128 payload + 2-voxel apron per side (seamless
// trilinear across bricks; clamp only guards coarse mips)
static const float BR_SLOT = 128.0;
static const float BR_AP = 0.0;
static const uint BR_LOD_FLAG = 0x20000u;
static const uint BR_PAGE_HEADER = 64u;

bool bricks_is_lod() { return (pc.brick_mode & BR_LOD_FLAG) != 0u; }

BrickCtx bricks_locate(float3 p) {
  uint bpa = pc.brick_mode & 0xffu;
  uint abpa = (pc.brick_mode >> 8) & 0xffu;
  float3 vp = saturate(p) * float(bpa * 128u);
  uint3 bc = min(uint3(vp) >> 7u, uint3(bpa - 1u));
  uint e = page[(bc.z * bpa + bc.y) * bpa + bc.x];
  BrickCtx c;
  c.ok = e != 0xFFFFFFFFu;
  c.maxv = c.ok ? float(e >> 24u) / 255.0 : 0.0;
  uint slot = e & 0xFFFFFFu;
  uint3 sc = uint3(slot % abpa, (slot / abpa) % abpa, slot / (abpa * abpa));
  c.slot_org = float3(sc) * BR_SLOT + BR_AP; /* payload texel origin */
  c.bl = vp - float3(bc << 7u);
  c.atlas_lod = 0.0;
  c.grad_vox = 1.0;
  c.world_brick = 1.0 / float(bpa);
  return c;
}

/* Resolve the level whose voxel pitch matches this sample's projected
 * footprint.  The streamer keeps the immediate parent warm, so desired,
 * parent, grandparent, and the pinned coarsest level cover both steady-state
 * and cold-start sampling with at most four page probes. */
BrickCtx bricks_lod_locate(float3 p, float lod) {
  BrickCtx c;
  c.ok = false;
  c.maxv = 0.0;
  c.bl = float3(0);
  c.slot_org = float3(0);
  c.atlas_lod = 0.0;
  c.grad_vox = 1.0;
  c.world_brick = 1.0;
  uint nlev = page[0];
  float3 base_shape = float3(page[1], page[2], page[3]);
  float maxdim = max(base_shape.x, max(base_shape.y, base_shape.z));
  float3 base_v = clamp(p * maxdim, float3(0), max(base_shape - 0.001, float3(0)));
  int desired = int(clamp(floor(lod), 0.0, float(nlev - 1u)));
  uint abpa = pc.brick_mode & 0xffu;
  int last = int(nlev) - 1;
  for (uint step = 0; step < 4u; step++) {
      int li = step < 3u ? desired + int(step) : last;
      if (li < 0 || li > last || (step == 3u && li <= desired + 2)) continue;
      uint h = 4u + uint(li) * 4u;
      uint off = page[h], packed = page[h + 1u], scale = page[h + 2u];
      uint3 bd = uint3(packed & 1023u, (packed >> 10u) & 1023u, packed >> 20u);
      float3 lv = base_v / float(scale);
      uint3 bc = min(uint3(lv) >> 7u, bd - 1u);
      uint bi = off + (bc.z * bd.y + bc.y) * bd.x + bc.x;
      uint e = page[bi];
      if (e == 0xFFFFFFFFu) continue;
      uint slot = e & 0xFFFFFFu;
      uint3 sc = uint3(slot % abpa, (slot / abpa) % abpa, slot / (abpa * abpa));
      c.ok = true;
      c.maxv = float(e >> 24u) / 255.0;
      c.slot_org = float3(sc) * BR_SLOT + BR_AP;
      c.bl = lv - float3(bc << 7u);
      /* The disk pyramid is the LOD hierarchy.  Per-slot atlas mips are
       * intentionally disabled: independently replaced slots cannot form a
       * globally continuous mip image. */
      c.atlas_lod = 0.0;
      c.grad_vox = exp2(lod) / float(scale);
      c.world_brick = BR_SLOT * float(scale) / maxdim;
      return c;
  }
  return c;
}

/* bl_off: brick-local voxel coords (already offset for gradient taps).
 * inset is negative at fine mips (the apron covers filtering) and only
 * clamps once a mip's footprint would pass the 2-voxel apron. */
float bricks_fetch(float3 slot_org, float3 bl_off, float lod, float inv_adim) {
  float inset = max(0.5 * exp2(lod) - BR_AP, 0.5);
  float3 l = clamp(bl_off, float3(inset), float3(128.0 - inset));
  return brick_atlas.SampleLevel((slot_org + l) * inv_adim, lod);
}

float bricks_sample(float3 p) { /* generic path (non-marcher callers) */
  BrickCtx c = bricks_is_lod() ? bricks_lod_locate(p, 0.0) : bricks_locate(p);
  uint abpa = bricks_is_lod() ? (pc.brick_mode & 0xffu) : ((pc.brick_mode >> 8) & 0xffu);
  float inv = 1.0 / (float(abpa) * BR_SLOT);
  return c.ok ? bricks_fetch(c.slot_org, c.bl, c.atlas_lod, inv) : 0.0;
}

// mirror of core/clip.h ring sizing: depth_max/2^l + 3, min 4
float clip_wzl(uint l) {
  uint w = (pc.slab_wz >> l) + 3;
  return float(max(w, 4u));
}

// World box the ray marches: cube = [0,1]^3; slab = thin box whose z spans the
// wz-1 seam-safe slices, all in units of 1/nx.
float3 vol_extent() {
  if (is_bricks() && bricks_is_lod()) {
    float3 shape = float3(page[1], page[2], page[3]);
    return shape / max(shape.x, max(shape.y, shape.z));
  }
  if (!is_slab() && !is_clip() && !is_vslab()) return float3(1);
  // box is exactly slab_depth voxels thick; the ring holds depth+2 slices so
  // filtering at the faces never crosses the ring seam
  return float3(1.0, pc.slab_ny / pc.slab_nx, float(pc.slab_depth) / pc.slab_nx);
}

float sample_tile(float3 uvw, uint idx) {
  // native non-uniform descriptor indexing (Turnip: ...NonUniformIndexingNative)
  // — replaces the old literal-index switch, whose branch tree cost ~20x per
  // sample once the tile pool grew past a few dozen entries
  return volumes[NonUniformResourceIndex(idx)].SampleLevel(uvw, 0);
}

/* --- virtual slab: world-anchored toroidal tile grids over the full export.
 * World coords = window origin + p * W. Cell (cx,cy) of level l lives in
 * physical tile (cx mod gx_l, cy mod gy_l); a validity table (binding 5,
 * base-level granularity) gates sampling — pyramid content is written by the
 * same fill as its base cell, so it inherits base validity. All geometry in
 * floats: payload pc.slab_px (runtime, <= 2016), cells exact in f32; no int
 * divides. Pyramid levels are OPTIONAL (grid byte 0 = absent, small windows
 * skip levels they don't need — sampling steps down to the finest present). */
float vs_sample(float3 p, float lod) {
  float P = pc.slab_px, T = P + 2.0;
  float wx = pc.slab_x0 + p.x * pc.slab_nx;
  float wy = pc.slab_y0 + p.y * pc.slab_nx; /* extent y = H/W: p.y*W = offset */
  float zc = pc.slab_z0 + 0.5 + p.z * pc.slab_nx;
  float w = frac(zc / float(pc.slab_wz)); /* world-anchored z ring */
  float gx0 = float(pc.slab_grid & 63u), gy0 = float((pc.slab_grid >> 6) & 63u);
  float cbx = floor(wx / P), cby = floor(wy / P);
  float pbx = cbx - floor(cbx / gx0) * gx0, pby = cby - floor(cby / gy0) * gy0;
  uint want = (uint(cbx) & 1023u) | ((uint(cby) & 1023u) << 10) | 0x80000000u;
  /* validity = 4 words/slot {key, za, zb, pad}: the async fill worker
   * publishes ranges as they land, so partially filled cells render their
   * resident z span and in-progress writes are never sampled */
  uint s4 = uint(pby * gx0 + pbx) * 4u;
  if (page[s4] != want) return 0.0; /* not resident (yet) */
  if (zc - 0.5 < float(page[s4 + 1]) || zc + 0.5 > float(page[s4 + 2])) return 0.0;
  int lev = min(int(floor(lod)) - 2, 3); /* prefilter 4<<lev matches footprint */
  while (lev >= 0 && ((pc.brick_mode >> (uint(lev) * 8u)) & 0xffu) == 0u)
    lev--; /* absent level: step down to the finest present */
  if (lev < 0) {
    float lx = wx - cbx * P + 1.0, ly = wy - cby * P + 1.0;
    return sample_tile(float3(lx / T, ly / T, w), uint(pby * gx0 + pbx));
  }
  float sc = float(4 << lev);
  uint gp = pc.brick_mode >> (uint(lev) * 8u); /* packed pyramid grids */
  float gxl = float(gp & 15u), gyl = float((gp >> 4) & 15u);
  uint bse = uint(gx0) * uint(gy0);
  for (int k = 0; k < lev; k++) {
    uint gk = pc.brick_mode >> (uint(k) * 8u);
    bse += (gk & 15u) * ((gk >> 4) & 15u);
  }
  float tx = wx / sc, ty = wy / sc; /* absolute level texel coords */
  float cx = floor(tx / P), cy = floor(ty / P);
  float px_ = cx - floor(cx / gxl) * gxl, py_ = cy - floor(cy / gyl) * gyl;
  float lx = tx - cx * P + 1.0, ly = ty - cy * P + 1.0;
  return sample_tile(float3(lx / T, ly / T, w), bse + uint(py_ * gxl + px_));
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
  if (is_bricks()) return bricks_sample(p);
  if (is_vslab()) return vs_sample(p, lod);
  if (!is_slab()) return volumes[0].SampleLevel(p, lod);
  // voxel coords (units of 1/nx per axis)
  float wx = p.x * pc.slab_nx;
  float wy = p.y * pc.slab_nx;
  float zc = pc.slab_z0 + 0.5 + p.z * pc.slab_nx; // ring z, seam-safe by box extent
  float w = frac(zc / float(pc.slab_wz)); // REPEAT sampler wraps the ring
  uint gx = pc.slab_grid & 0xffu, gy = (pc.slab_grid >> 8) & 0xffu;
  uint ovs = (pc.slab_grid >> 16) & 0xffu;
  // overview pyramid: pick the level whose prefilter (4<<lev) matches the
  // ray-cone footprint (mip-style) — mid/far zooms sample filtered data with
  // texture-cache locality instead of thrashing full-res tiles
  int lev = int(floor(lod)) - 2;
  if (lev >= 0) {
    /* all-float geometry (mirrors slab.h exactly: quotient fractions are
     * >= 1/2046, far above f32 eps) — Adreno has no integer divide */
    int nlev = int(log2(float(ovs)) + 0.5) - 1; /* levels s = 4 .. ovs */
    lev = min(lev, nlev - 1);
    float sc = float(4 << lev);
    float obase = float(gx * gy); /* tile-pool offset of this level */
    for (int k = 0; k < lev; k++) {
      float sk = float(4 << k);
      obase += ceil(ceil(pc.slab_nx / sk) / 2046.0) * ceil(ceil(pc.slab_ny / sk) / 2046.0);
    }
    float vnx = ceil(pc.slab_nx / sc), vny = ceil(pc.slab_ny / sc);
    float gxl = ceil(vnx / 2046.0), gyl = ceil(vny / 2046.0);
    float pxl = ceil(vnx / gxl), pyl = ceil(vny / gyl);
    float wxl = wx / sc, wyl = wy / sc;
    float oix = min(floor(max(wxl, 0.0) / pxl), gxl - 1.0);
    float oiy = min(floor(max(wyl, 0.0) / pyl), gyl - 1.0);
    float ou = (wxl - oix * pxl + 1.0) / (pxl + 2.0);
    float ov = (wyl - oiy * pyl + 1.0) / (pyl + 2.0);
    return sample_tile(float3(ou, ov, w), uint(obase + oiy * gxl + oix));
  }
  uint ix = min(uint(max(wx, 0.0) / pc.slab_px), gx - 1);
  uint iy = min(uint(max(wy, 0.0) / pc.slab_py), gy - 1);
  float tw = pc.slab_px + 2.0, th = pc.slab_py + 2.0;
  float u = (wx - float(ix) * pc.slab_px + 1.0) / tw;
  float v = (wy - float(iy) * pc.slab_py + 1.0) / th;
  return sample_tile(float3(u, v, w), iy * gx + ix);
}

bool is_ortho() { return (pc.view_flags & 1u) != 0u; }

float2 pix_ndc(uint2 pix) {
  float2 uv = (float2(pix) + 0.5) / float2(pc.viewport);
  return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

float3 ray_dir(uint2 pix) {
  if (is_ortho()) return normalize(pc.cam_forward);
  float2 ndc = pix_ndc(pix);
  return normalize(pc.cam_forward + ndc.x * pc.cam_right + ndc.y * pc.cam_up);
}

/* ortho views offset the origin across the image plane instead of fanning
 * directions; cam_right/up carry the half-extents of the visible rect */
float3 ray_origin(uint2 pix) {
  if (!is_ortho()) return pc.cam_origin;
  float2 ndc = pix_ndc(pix);
  return pc.cam_origin + ndc.x * pc.cam_right + ndc.y * pc.cam_up;
}

// slab test against [lo, hi]; returns (tnear, tfar), tnear<=tfar means hit
float2 box_hit(float3 o, float3 d, float3 lo, float3 hi) {
  float3 inv = 1.0 / d;
  float3 t0 = (lo - o) * inv;
  float3 t1 = (hi - o) * inv;
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
  uint2 opix = tid.xy + uint2(pc.view_org & 0xffffu, pc.view_org >> 16);

  if (is_surf()) {
    /* Flattened segment: view plane IS the tifxyz grid. cam_origin.xy =
     * grid-space view center, cam_right.x / cam_up.y = half-extents in grid
     * units, slab_z0 = offset along the per-pixel surface normal (voxels),
     * slab_depth = slab thickness marched along the normal (voxels),
     * slab_px = voxels per grid step (1/scale). Per pixel this raymarches a
     * REAL volumetric slab centered zoff along the local surface normal —
     * the flattened analogue of the plane views' ortho slabs. */
    float2 ndc = pix_ndc(tid.xy);
    float2 g = pc.cam_origin.xy + ndc.x * pc.cam_right.xy + ndc.y * pc.cam_up.xy;
    uint2 dim;
    surf_coords.GetDimensions(dim.x, dim.y);
    float4 bg = float4(0.02, 0.02, 0.03, 1);
    if (g.x < 0.0 || g.y < 0.0 || g.x > float(dim.x - 1) || g.y > float(dim.y - 1)) {
      out_image[opix] = bg;
      return;
    }
    int2 i0 = int2(min(uint2(g), dim - 2u));
    float2 fr = g - float2(i0);
    float4 c00 = surf_coords.Load(int3(i0, 0));
    float4 c10 = surf_coords.Load(int3(i0 + int2(1, 0), 0));
    float4 c01 = surf_coords.Load(int3(i0 + int2(0, 1), 0));
    float4 c11 = surf_coords.Load(int3(i0 + int2(1, 1), 0));
    if (min(min(c00.w, c10.w), min(c01.w, c11.w)) < 0.5) { /* unmapped quad */
      out_image[opix] = bg;
      return;
    }
    float3 wp = lerp(lerp(c00.xyz, c10.xyz, fr.x), lerp(c01.xyz, c11.xyz, fr.x), fr.y);
    float3 n00 = surf_normals.Load(int3(i0, 0)).xyz;
    float3 n10 = surf_normals.Load(int3(i0 + int2(1, 0), 0)).xyz;
    float3 n01 = surf_normals.Load(int3(i0 + int2(0, 1), 0)).xyz;
    float3 n11 = surf_normals.Load(int3(i0 + int2(1, 1), 0)).xyz;
    float3 nrm = lerp(lerp(n00, n10, fr.x), lerp(n01, n11, fr.x), fr.y);
    float nl = length(nrm);
    if (nl < 1e-4) {
      out_image[opix] = bg;
      return;
    }
    nrm /= nl;
    float3 shape = float3(page[1], page[2], page[3]);
    float md = max(shape.x, max(shape.y, shape.z));
    /* footprint: grid-units per pixel * voxels per grid unit */
    float vpp = 2.0 * length(pc.cam_right.xy) * pc.slab_px / float(pc.viewport.x);
    float lod = clamp(log2(max(vpp, 1.0)) + pc.lod_bias, 0.0, float(page[0] - 1u));
    float inv = 1.0 / (float((pc.brick_mode & 0xffu)) * BR_SLOT);

    float depth = max(float(pc.slab_depth), 1.0);
    float dt = pc.step_voxels * max(exp2(lod) * 0.5, 1.0); /* voxels along n */
    float t = -0.5 * depth + pc.slab_z0 + hash12(tid.xy, pc.frame_index) * dt;
    float tf_end = 0.5 * depth + pc.slab_z0;
    float mip_max = 0.0;
    float4 acc = float4(0);
    bool composite = pc.mode != MODE_MIP;
    for (uint k = 0; k < 256u && t < tf_end; k++, t += dt) {
      float3 sp = wp + nrm * t;
      if (any(sp < float3(0)) || any(sp >= shape)) continue;
      BrickCtx c = bricks_lod_locate(sp / md, lod);
      if (!c.ok) continue;
      float v = bricks_fetch(c.slot_org, c.bl, c.atlas_lod, inv);
      if (v < pc.threshold) continue;
      if (v > mip_max) mip_max = v;
      if (!composite) continue;
      float4 s = tf.SampleLevel(v, 0);
      float a = 1.0 - pow(1.0 - s.a, pc.density * dt);
      if (a <= 0.003) continue;
      float3 col = s.rgb;
      if (pc.mode == MODE_FULL) { /* gradient-lit like the box marcher */
        float ev = c.grad_vox;
        float3 gr = float3(
            bricks_fetch(c.slot_org, c.bl + float3(ev, 0, 0), c.atlas_lod, inv) -
                bricks_fetch(c.slot_org, c.bl + float3(-ev, 0, 0), c.atlas_lod, inv),
            bricks_fetch(c.slot_org, c.bl + float3(0, ev, 0), c.atlas_lod, inv) -
                bricks_fetch(c.slot_org, c.bl + float3(0, -ev, 0), c.atlas_lod, inv),
            bricks_fetch(c.slot_org, c.bl + float3(0, 0, ev), c.atlas_lod, inv) -
                bricks_fetch(c.slot_org, c.bl + float3(0, 0, -ev), c.atlas_lod, inv));
        float gm = length(gr);
        if (gm > 1e-4) {
          float3 nn = gr / gm;
          float diff = 0.7 * abs(dot(nn, nrm)) + 0.3;
          float lit = 0.25 + 0.75 * diff;
          col = col * lerp(1.0, lit, saturate(gm * 6.0));
        }
      }
      acc.rgb += (1.0 - acc.a) * a * col;
      acc.a += (1.0 - acc.a) * a;
      if (acc.a >= ALPHA_CUTOFF) break;
    }
    float3 color = composite ? acc.rgb + (1.0 - acc.a) * bg.rgb
                             : float3(mip_max * pc.density);
    out_image[opix] = float4(color, 1);
    return;
  }

  float3 dir = ray_dir(tid.xy);

  if (pc.mode == MODE_RAYDIR) {
    out_image[opix] = float4(dir * 0.5 + 0.5, 1);
    return;
  }

  float voxel;
  float max_lod;
  if (is_clip()) {
    voxel = 1.0 / pc.slab_nx;
    max_lod = 7.0; // levels end at 5; extra lod keeps widening far-field steps
  } else if (is_bricks()) {
    if (bricks_is_lod()) {
      float md = float(max(page[1], max(page[2], page[3])));
      voxel = 1.0 / md;
      max_lod = float(page[0] - 1u); // complete on-disk hierarchy
    } else {
      voxel = 1.0 / float((pc.brick_mode & 0xffu) * 128u);
      max_lod = 3.0;
    }
  } else if (is_vslab()) {
    voxel = 1.0 / pc.slab_nx;
    /* +2 lod past the finest PRESENT pyramid level keeps far steps widening;
     * small windows may carry no pyramid at all (base only -> 3.0) */
    max_lod = 3.0;
    for (int l = 3; l >= 0; l--)
      if (((pc.brick_mode >> (uint(l) * 8u)) & 0xffu) != 0u) {
        max_lod = float(l) + 4.0; /* scale 4<<l: log2 = l+2, +2 headroom */
        break;
      }
  } else if (is_slab()) {
    voxel = 1.0 / pc.slab_nx;
    // must exceed the overview switch (log2(ovs)-0.25) or wide views would be
    // clamped below it and march full-res tiles forever; +2 keeps far-field
    // steps widening past the switch like the old fixed 4.0 did for ovs=4
    max_lod = log2(float((pc.slab_grid >> 16) & 0xffu)) + 2.0;
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
  float3 ro = to_volume_point(ray_origin(tid.xy), ext * 0.5);
  dir = to_volume_dir(dir);

  float3 box_lo = float3(0), box_hi = ext;
  if (is_bricks() && bricks_is_lod() && pc.slab_depth > 0u) {
    float md = float(max(page[1], max(page[2], page[3])));
    uint axc = (pc.view_flags >> 8) & 3u; /* 0=z 1=x 2=y */
    int a = axc == 1u ? 0 : (axc == 2u ? 1 : 2);
    box_lo[a] = clamp(pc.slab_z0 / md, 0.0, ext[a]);
    box_hi[a] = clamp((pc.slab_z0 + float(pc.slab_depth)) / md, box_lo[a], ext[a]);
  }
  float2 hit = box_hit(ro, dir, box_lo, box_hi);
  float tn = max(hit.x, 0.0), tf_ = hit.y;
  if (tn >= tf_) {
    out_image[opix] = float4(0.02, 0.02, 0.03, 1); // background
    return;
  }

  // ray-cone footprint: pixel angular size (cam_up is pre-scaled by tan(fov/2));
  // ortho views have a constant per-pixel footprint instead (see lod below)
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
  /* A deep rectangular volume can contain tens of thousands of voxels along
   * one ray.  Bound and fully cover that interval; without this depth-step
   * floor close views truncated at MAX_STEPS and still took hundreds of ms. */
  uint ray_budget = is_bricks() && bricks_is_lod() ? (shaded ? 192u : 256u) : MAX_STEPS;
  float span_step = max((tf_ - tn) / (float(ray_budget) * dt), 1.0);

  float bs = OCC_BLOCK * voxel; // skip-block edge, volume units
  // block max below this contributes nothing: low-cut OR zero TF alpha
  float occ_gate = max(pc.skip_gate, 0.5 / 255.0);

  // bricks mode: page entry cached across steps (rays cross bricks rarely)
  uint b_bpa = pc.brick_mode & 0xffu;
  float b_N = float(b_bpa * 128u);
  uint b_abpa = bricks_is_lod() ? (pc.brick_mode & 0xffu) : ((pc.brick_mode >> 8) & 0xffu);
  float b_inv = 1.0 / (float(b_abpa) * BR_SLOT);
  uint b_last = 0xFFFFFFFFu;
  bool b_ok = false;
  float b_max = 0.0;
  float3 b_slot = float3(0);
  float b_grad_vox = 1.0;
  float b_world_brick = 128.0 * voxel;

  while (t < tf_ && steps < ray_budget) {
    steps++;
    // LOD-on-distance: mip where the ray cone's width matches the voxel pitch;
    // step size scales with it so far-field marching is proportionally cheap.
    // Ortho: the footprint is depth-independent (parallel rays).
    float foot = is_ortho() ? cone : t * cone;
    float lod = clamp(log2(max(foot / voxel, 1.0)) + pc.lod_bias, 0.0, max_lod);
    /* Spatial filtering follows projected magnification.  The independent
     * depth step floor guarantees traversal of the whole ray without forcing
     * an isotropic coarse LOD (which visibly destroyed close-up XY detail). */
    float lod_step = max(exp2(lod), span_step);
    if (is_slab() || is_clip() || is_vslab()) // >=4 samples along the box segment
      lod_step = min(lod_step, max(1.0, (tf_ - tn) / (4.0 * dt)));
    float3 p = ro + t * dir;

    int cl_lvl = -1; // clip: pick the level once; gradient taps reuse it
    float3 b_bl = float3(0); // bricks: local coords this sample
    float blod = 0.0;
    float v;
    if (is_clip()) {
      cl_lvl = pick_clip_level(p, lod);
      v = cl_lvl < 0 ? 0.0 : sample_clip_level(uint(cl_lvl), p);
    } else if (is_bricks()) {
      if (bricks_is_lod()) {
        BrickCtx bc = bricks_lod_locate(p, lod);
        b_ok = bc.ok;
        b_max = bc.maxv;
        b_slot = bc.slot_org;
        b_bl = bc.bl;
        blod = bc.atlas_lod;
        b_grad_vox = bc.grad_vox;
        b_world_brick = bc.world_brick;
        v = b_ok ? bricks_fetch(b_slot, b_bl, blod, b_inv) : 0.0;
      } else if ((pc.brick_mode & 0x10000u) != 0) {
        blod = clamp(lod, 0.0, 3.0); /* atlas has 4 mips */
        /* identity residency (v1 full shard): atlas layout == world layout,
         * so the page indirection is a no-op — sample like cube mode */
        v = brick_atlas.SampleLevel(p, blod);
      } else {
        blod = clamp(lod, 0.0, 3.0);
        float3 vp = saturate(p) * b_N;
        uint3 bc = min(uint3(vp) >> 7u, uint3(b_bpa - 1u));
        uint bi = (bc.z * b_bpa + bc.y) * b_bpa + bc.x;
        if (bi != b_last) { /* page resolve only on brick crossings */
          b_last = bi;
          uint e = page[bi];
          b_ok = e != 0xFFFFFFFFu;
          b_max = b_ok ? float(e >> 24u) / 255.0 : 0.0;
          uint abpa = (pc.brick_mode >> 8) & 0xffu;
          uint slot = e & 0xFFFFFFu;
          b_slot = float3(uint3(slot % abpa, (slot / abpa) % abpa, slot / (abpa * abpa))) *
                       BR_SLOT + BR_AP;
        }
        b_bl = vp - float3(bc << 7u);
        v = b_ok ? bricks_fetch(b_slot, b_bl, blod, b_inv) : 0.0;
      }
    } else {
      v = vol_sample(p, lod);
    }
    bool cut = v < pc.threshold; // low-cut: below-threshold voxels are void
    if (cut) v = 0.0;

    // empty-space skip, checked only where the fine sample is already void so
    // dense marching pays nothing. Cube mode: 8^3 occupancy blocks; bricks
    // mode: whole 128^3 bricks via the page-table max byte.
    bool void_blk = false;
    float bsz = bs;
    if (is_bricks()) {
      /* decoded air carries 1-3 LSB codec noise, so exact-zero gating never
       * fires; a ~4 LSB skip floor loses at most ~1% accumulated alpha per
       * skipped block (invisible) and restores cube-grade skipping */
      float bgate = max(occ_gate, 4.5 / 255.0);
      if (v < bgate) {
        if (bricks_is_lod()) {
          if (b_max < bgate) {
            void_blk = true;
            bsz = b_world_brick;
          }
        } else if ((pc.brick_mode & 0x10000u) != 0) { /* identity: 8^3 occ only */
          if (occ.SampleLevel(p, 0) < bgate) void_blk = true;
        } else if (b_max < bgate) { /* whole brick void (cached page entry) */
          void_blk = true;
          bsz = 128.0 * voxel;
        } else if (occ.SampleLevel(p, 0) < bgate) { /* 8^3 block void */
          void_blk = true;
        }
      }
    } else if (!is_slab() && !is_clip() && v < occ_gate && occ.SampleLevel(p, 0) < occ_gate) {
      void_blk = true;
    }
    if (void_blk) {
      float3 nextp = (floor(p / bsz) + max(sign(dir), float3(0))) * bsz;
      float3 tex = select(abs(dir) > 1e-8, (nextp - ro) / dir, float3(1e30));
      t = max(min(min(tex.x, tex.y), tex.z), t) + bsz * 0.01;
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
#if R3D_FAST_GRADIENT
          /* Four tetrahedron-corner taps instead of six central differences.
           * The radius is unchanged (e), and sqrt(3)/2 restores the response
           * of a linear field to the same 2*e magnitude. This is an explicit
           * quality/performance mode: real papyrus retains structure but its
           * fine surface speckle differs measurably from central differences. */
          const float ir3 = 0.57735026919;
          const float renorm = 0.86602540378;
          float q0, q1, q2, q3;
          if (is_bricks() && !bricks_is_lod() && (pc.brick_mode & 0x10000u) != 0) {
            float ee = e * ir3;
            q0 = brick_atlas.SampleLevel(p + float3( ee, -ee, -ee), blod);
            q1 = brick_atlas.SampleLevel(p + float3(-ee, -ee,  ee), blod);
            q2 = brick_atlas.SampleLevel(p + float3(-ee,  ee, -ee), blod);
            q3 = brick_atlas.SampleLevel(p + float3( ee,  ee,  ee), blod);
          } else if (is_bricks()) {
            float ev = (bricks_is_lod() ? b_grad_vox : lod_step) * ir3;
            q0 = bricks_fetch(b_slot, b_bl + float3( ev, -ev, -ev), blod, b_inv);
            q1 = bricks_fetch(b_slot, b_bl + float3(-ev, -ev,  ev), blod, b_inv);
            q2 = bricks_fetch(b_slot, b_bl + float3(-ev,  ev, -ev), blod, b_inv);
            q3 = bricks_fetch(b_slot, b_bl + float3( ev,  ev,  ev), blod, b_inv);
          } else if (cl_lvl >= 0) {
            uint gl = uint(cl_lvl);
            float ee = e * ir3;
            q0 = sample_clip_level(gl, p + float3( ee, -ee, -ee));
            q1 = sample_clip_level(gl, p + float3(-ee, -ee,  ee));
            q2 = sample_clip_level(gl, p + float3(-ee,  ee, -ee));
            q3 = sample_clip_level(gl, p + float3( ee,  ee,  ee));
          } else {
            float ee = e * ir3;
            q0 = vol_sample(p + float3( ee, -ee, -ee), lod);
            q1 = vol_sample(p + float3(-ee, -ee,  ee), lod);
            q2 = vol_sample(p + float3(-ee,  ee, -ee), lod);
            q3 = vol_sample(p + float3( ee,  ee,  ee), lod);
          }
          g = float3(q0 - q1 - q2 + q3, -q0 - q1 + q2 + q3,
                     -q0 + q1 - q2 + q3) * renorm;
#else
          if (is_bricks() && !bricks_is_lod() && (pc.brick_mode & 0x10000u) != 0) {
            g = float3(brick_atlas.SampleLevel(p + float3(e, 0, 0), blod) -
                           brick_atlas.SampleLevel(p - float3(e, 0, 0), blod),
                       brick_atlas.SampleLevel(p + float3(0, e, 0), blod) -
                           brick_atlas.SampleLevel(p - float3(0, e, 0), blod),
                       brick_atlas.SampleLevel(p + float3(0, 0, e), blod) -
                           brick_atlas.SampleLevel(p - float3(0, 0, e), blod));
          } else if (is_bricks()) {
            /* taps reuse the cached brick (offsets in voxels) */
            float ev = bricks_is_lod() ? b_grad_vox : lod_step;
            g = float3(bricks_fetch(b_slot, b_bl + float3(ev, 0, 0), blod, b_inv) -
                           bricks_fetch(b_slot, b_bl + float3(-ev, 0, 0), blod, b_inv),
                       bricks_fetch(b_slot, b_bl + float3(0, ev, 0), blod, b_inv) -
                           bricks_fetch(b_slot, b_bl + float3(0, -ev, 0), blod, b_inv),
                       bricks_fetch(b_slot, b_bl + float3(0, 0, ev), blod, b_inv) -
                           bricks_fetch(b_slot, b_bl + float3(0, 0, -ev), blod, b_inv));
          } else if (cl_lvl >= 0) {
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
#endif
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
  out_image[opix] = float4(color, 1);
}
