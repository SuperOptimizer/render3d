// Volume raycaster: fixed-step marching, two data modes.
//  - cube: one monolithic 3D texture in [0,1]^3 (volumes[0], all 4 slots alias)
//  - slab: up to 16x16 tile textures (payload + 1-voxel apron) covering a wide
//    thin z-window; z is a ring buffer sampled with a REPEAT-W sampler
//    (see src/core/slab.h for the layout math mirrored here).
import common;

// slab tiles (element j*16+i, up to 16x16); cube mode binds the monolithic x256
[[vk::binding(0, 0)]] Sampler3D<float> volumes[256];
[[vk::binding(1, 0)]] Sampler1D<float4> tf;
[[vk::binding(2, 0)]] RWTexture2D<float4> out_image;
// per-8^3-block max, dilated 1 block (nearest sampler) — empty-space skipping
[[vk::binding(3, 0)]] Sampler3D<float> occ;
// bricks mode: c5d GPU-decoded brick atlas + page table
// page[bi] = slot_index | brick_max<<24, 0xFFFFFFFF = not resident
[[vk::binding(4, 0)]] Sampler3D<float> brick_atlas;
[[vk::binding(5, 0)]] StructuredBuffer<uint> page;

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
bool is_slab() { return R3D_MODE == 1; }
bool is_clip() { return R3D_MODE == 2; }
bool is_bricks() { return R3D_MODE == 3; }

/* bricks mode: one page-table resolve per sample position; the fetch (and
 * the 6 gradient taps reusing the context) are then plain atlas samples. */
struct BrickCtx {
  bool ok;
  float maxv;     /* brick max (page high byte) for empty skip */
  float3 bl;      /* voxel coords local to the brick */
  float3 slot_org; /* atlas mip0 texel origin of the slot */
};

// slot geometry: 132^3 = 128 payload + 2-voxel apron per side (seamless
// trilinear across bricks; clamp only guards coarse mips)
static const float BR_SLOT = 128.0;
static const float BR_AP = 0.0;

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
  BrickCtx c = bricks_locate(p);
  float inv = 1.0 / (float((pc.brick_mode >> 8) & 0xffu) * BR_SLOT);
  return c.ok ? bricks_fetch(c.slot_org, c.bl, 0.0, inv) : 0.0;
}

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
  case 16: return volumes[16].SampleLevel(uvw, 0);
  case 17: return volumes[17].SampleLevel(uvw, 0);
  case 18: return volumes[18].SampleLevel(uvw, 0);
  case 19: return volumes[19].SampleLevel(uvw, 0);
  case 20: return volumes[20].SampleLevel(uvw, 0);
  case 21: return volumes[21].SampleLevel(uvw, 0);
  case 22: return volumes[22].SampleLevel(uvw, 0);
  case 23: return volumes[23].SampleLevel(uvw, 0);
  case 24: return volumes[24].SampleLevel(uvw, 0);
  case 25: return volumes[25].SampleLevel(uvw, 0);
  case 26: return volumes[26].SampleLevel(uvw, 0);
  case 27: return volumes[27].SampleLevel(uvw, 0);
  case 28: return volumes[28].SampleLevel(uvw, 0);
  case 29: return volumes[29].SampleLevel(uvw, 0);
  case 30: return volumes[30].SampleLevel(uvw, 0);
  case 31: return volumes[31].SampleLevel(uvw, 0);
  case 32: return volumes[32].SampleLevel(uvw, 0);
  case 33: return volumes[33].SampleLevel(uvw, 0);
  case 34: return volumes[34].SampleLevel(uvw, 0);
  case 35: return volumes[35].SampleLevel(uvw, 0);
  case 36: return volumes[36].SampleLevel(uvw, 0);
  case 37: return volumes[37].SampleLevel(uvw, 0);
  case 38: return volumes[38].SampleLevel(uvw, 0);
  case 39: return volumes[39].SampleLevel(uvw, 0);
  case 40: return volumes[40].SampleLevel(uvw, 0);
  case 41: return volumes[41].SampleLevel(uvw, 0);
  case 42: return volumes[42].SampleLevel(uvw, 0);
  case 43: return volumes[43].SampleLevel(uvw, 0);
  case 44: return volumes[44].SampleLevel(uvw, 0);
  case 45: return volumes[45].SampleLevel(uvw, 0);
  case 46: return volumes[46].SampleLevel(uvw, 0);
  case 47: return volumes[47].SampleLevel(uvw, 0);
  case 48: return volumes[48].SampleLevel(uvw, 0);
  case 49: return volumes[49].SampleLevel(uvw, 0);
  case 50: return volumes[50].SampleLevel(uvw, 0);
  case 51: return volumes[51].SampleLevel(uvw, 0);
  case 52: return volumes[52].SampleLevel(uvw, 0);
  case 53: return volumes[53].SampleLevel(uvw, 0);
  case 54: return volumes[54].SampleLevel(uvw, 0);
  case 55: return volumes[55].SampleLevel(uvw, 0);
  case 56: return volumes[56].SampleLevel(uvw, 0);
  case 57: return volumes[57].SampleLevel(uvw, 0);
  case 58: return volumes[58].SampleLevel(uvw, 0);
  case 59: return volumes[59].SampleLevel(uvw, 0);
  case 60: return volumes[60].SampleLevel(uvw, 0);
  case 61: return volumes[61].SampleLevel(uvw, 0);
  case 62: return volumes[62].SampleLevel(uvw, 0);
  case 63: return volumes[63].SampleLevel(uvw, 0);
  case 64: return volumes[64].SampleLevel(uvw, 0);
  case 65: return volumes[65].SampleLevel(uvw, 0);
  case 66: return volumes[66].SampleLevel(uvw, 0);
  case 67: return volumes[67].SampleLevel(uvw, 0);
  case 68: return volumes[68].SampleLevel(uvw, 0);
  case 69: return volumes[69].SampleLevel(uvw, 0);
  case 70: return volumes[70].SampleLevel(uvw, 0);
  case 71: return volumes[71].SampleLevel(uvw, 0);
  case 72: return volumes[72].SampleLevel(uvw, 0);
  case 73: return volumes[73].SampleLevel(uvw, 0);
  case 74: return volumes[74].SampleLevel(uvw, 0);
  case 75: return volumes[75].SampleLevel(uvw, 0);
  case 76: return volumes[76].SampleLevel(uvw, 0);
  case 77: return volumes[77].SampleLevel(uvw, 0);
  case 78: return volumes[78].SampleLevel(uvw, 0);
  case 79: return volumes[79].SampleLevel(uvw, 0);
  case 80: return volumes[80].SampleLevel(uvw, 0);
  case 81: return volumes[81].SampleLevel(uvw, 0);
  case 82: return volumes[82].SampleLevel(uvw, 0);
  case 83: return volumes[83].SampleLevel(uvw, 0);
  case 84: return volumes[84].SampleLevel(uvw, 0);
  case 85: return volumes[85].SampleLevel(uvw, 0);
  case 86: return volumes[86].SampleLevel(uvw, 0);
  case 87: return volumes[87].SampleLevel(uvw, 0);
  case 88: return volumes[88].SampleLevel(uvw, 0);
  case 89: return volumes[89].SampleLevel(uvw, 0);
  case 90: return volumes[90].SampleLevel(uvw, 0);
  case 91: return volumes[91].SampleLevel(uvw, 0);
  case 92: return volumes[92].SampleLevel(uvw, 0);
  case 93: return volumes[93].SampleLevel(uvw, 0);
  case 94: return volumes[94].SampleLevel(uvw, 0);
  case 95: return volumes[95].SampleLevel(uvw, 0);
  case 96: return volumes[96].SampleLevel(uvw, 0);
  case 97: return volumes[97].SampleLevel(uvw, 0);
  case 98: return volumes[98].SampleLevel(uvw, 0);
  case 99: return volumes[99].SampleLevel(uvw, 0);
  case 100: return volumes[100].SampleLevel(uvw, 0);
  case 101: return volumes[101].SampleLevel(uvw, 0);
  case 102: return volumes[102].SampleLevel(uvw, 0);
  case 103: return volumes[103].SampleLevel(uvw, 0);
  case 104: return volumes[104].SampleLevel(uvw, 0);
  case 105: return volumes[105].SampleLevel(uvw, 0);
  case 106: return volumes[106].SampleLevel(uvw, 0);
  case 107: return volumes[107].SampleLevel(uvw, 0);
  case 108: return volumes[108].SampleLevel(uvw, 0);
  case 109: return volumes[109].SampleLevel(uvw, 0);
  case 110: return volumes[110].SampleLevel(uvw, 0);
  case 111: return volumes[111].SampleLevel(uvw, 0);
  case 112: return volumes[112].SampleLevel(uvw, 0);
  case 113: return volumes[113].SampleLevel(uvw, 0);
  case 114: return volumes[114].SampleLevel(uvw, 0);
  case 115: return volumes[115].SampleLevel(uvw, 0);
  case 116: return volumes[116].SampleLevel(uvw, 0);
  case 117: return volumes[117].SampleLevel(uvw, 0);
  case 118: return volumes[118].SampleLevel(uvw, 0);
  case 119: return volumes[119].SampleLevel(uvw, 0);
  case 120: return volumes[120].SampleLevel(uvw, 0);
  case 121: return volumes[121].SampleLevel(uvw, 0);
  case 122: return volumes[122].SampleLevel(uvw, 0);
  case 123: return volumes[123].SampleLevel(uvw, 0);
  case 124: return volumes[124].SampleLevel(uvw, 0);
  case 125: return volumes[125].SampleLevel(uvw, 0);
  case 126: return volumes[126].SampleLevel(uvw, 0);
  case 127: return volumes[127].SampleLevel(uvw, 0);
  case 128: return volumes[128].SampleLevel(uvw, 0);
  case 129: return volumes[129].SampleLevel(uvw, 0);
  case 130: return volumes[130].SampleLevel(uvw, 0);
  case 131: return volumes[131].SampleLevel(uvw, 0);
  case 132: return volumes[132].SampleLevel(uvw, 0);
  case 133: return volumes[133].SampleLevel(uvw, 0);
  case 134: return volumes[134].SampleLevel(uvw, 0);
  case 135: return volumes[135].SampleLevel(uvw, 0);
  case 136: return volumes[136].SampleLevel(uvw, 0);
  case 137: return volumes[137].SampleLevel(uvw, 0);
  case 138: return volumes[138].SampleLevel(uvw, 0);
  case 139: return volumes[139].SampleLevel(uvw, 0);
  case 140: return volumes[140].SampleLevel(uvw, 0);
  case 141: return volumes[141].SampleLevel(uvw, 0);
  case 142: return volumes[142].SampleLevel(uvw, 0);
  case 143: return volumes[143].SampleLevel(uvw, 0);
  case 144: return volumes[144].SampleLevel(uvw, 0);
  case 145: return volumes[145].SampleLevel(uvw, 0);
  case 146: return volumes[146].SampleLevel(uvw, 0);
  case 147: return volumes[147].SampleLevel(uvw, 0);
  case 148: return volumes[148].SampleLevel(uvw, 0);
  case 149: return volumes[149].SampleLevel(uvw, 0);
  case 150: return volumes[150].SampleLevel(uvw, 0);
  case 151: return volumes[151].SampleLevel(uvw, 0);
  case 152: return volumes[152].SampleLevel(uvw, 0);
  case 153: return volumes[153].SampleLevel(uvw, 0);
  case 154: return volumes[154].SampleLevel(uvw, 0);
  case 155: return volumes[155].SampleLevel(uvw, 0);
  case 156: return volumes[156].SampleLevel(uvw, 0);
  case 157: return volumes[157].SampleLevel(uvw, 0);
  case 158: return volumes[158].SampleLevel(uvw, 0);
  case 159: return volumes[159].SampleLevel(uvw, 0);
  case 160: return volumes[160].SampleLevel(uvw, 0);
  case 161: return volumes[161].SampleLevel(uvw, 0);
  case 162: return volumes[162].SampleLevel(uvw, 0);
  case 163: return volumes[163].SampleLevel(uvw, 0);
  case 164: return volumes[164].SampleLevel(uvw, 0);
  case 165: return volumes[165].SampleLevel(uvw, 0);
  case 166: return volumes[166].SampleLevel(uvw, 0);
  case 167: return volumes[167].SampleLevel(uvw, 0);
  case 168: return volumes[168].SampleLevel(uvw, 0);
  case 169: return volumes[169].SampleLevel(uvw, 0);
  case 170: return volumes[170].SampleLevel(uvw, 0);
  case 171: return volumes[171].SampleLevel(uvw, 0);
  case 172: return volumes[172].SampleLevel(uvw, 0);
  case 173: return volumes[173].SampleLevel(uvw, 0);
  case 174: return volumes[174].SampleLevel(uvw, 0);
  case 175: return volumes[175].SampleLevel(uvw, 0);
  case 176: return volumes[176].SampleLevel(uvw, 0);
  case 177: return volumes[177].SampleLevel(uvw, 0);
  case 178: return volumes[178].SampleLevel(uvw, 0);
  case 179: return volumes[179].SampleLevel(uvw, 0);
  case 180: return volumes[180].SampleLevel(uvw, 0);
  case 181: return volumes[181].SampleLevel(uvw, 0);
  case 182: return volumes[182].SampleLevel(uvw, 0);
  case 183: return volumes[183].SampleLevel(uvw, 0);
  case 184: return volumes[184].SampleLevel(uvw, 0);
  case 185: return volumes[185].SampleLevel(uvw, 0);
  case 186: return volumes[186].SampleLevel(uvw, 0);
  case 187: return volumes[187].SampleLevel(uvw, 0);
  case 188: return volumes[188].SampleLevel(uvw, 0);
  case 189: return volumes[189].SampleLevel(uvw, 0);
  case 190: return volumes[190].SampleLevel(uvw, 0);
  case 191: return volumes[191].SampleLevel(uvw, 0);
  case 192: return volumes[192].SampleLevel(uvw, 0);
  case 193: return volumes[193].SampleLevel(uvw, 0);
  case 194: return volumes[194].SampleLevel(uvw, 0);
  case 195: return volumes[195].SampleLevel(uvw, 0);
  case 196: return volumes[196].SampleLevel(uvw, 0);
  case 197: return volumes[197].SampleLevel(uvw, 0);
  case 198: return volumes[198].SampleLevel(uvw, 0);
  case 199: return volumes[199].SampleLevel(uvw, 0);
  case 200: return volumes[200].SampleLevel(uvw, 0);
  case 201: return volumes[201].SampleLevel(uvw, 0);
  case 202: return volumes[202].SampleLevel(uvw, 0);
  case 203: return volumes[203].SampleLevel(uvw, 0);
  case 204: return volumes[204].SampleLevel(uvw, 0);
  case 205: return volumes[205].SampleLevel(uvw, 0);
  case 206: return volumes[206].SampleLevel(uvw, 0);
  case 207: return volumes[207].SampleLevel(uvw, 0);
  case 208: return volumes[208].SampleLevel(uvw, 0);
  case 209: return volumes[209].SampleLevel(uvw, 0);
  case 210: return volumes[210].SampleLevel(uvw, 0);
  case 211: return volumes[211].SampleLevel(uvw, 0);
  case 212: return volumes[212].SampleLevel(uvw, 0);
  case 213: return volumes[213].SampleLevel(uvw, 0);
  case 214: return volumes[214].SampleLevel(uvw, 0);
  case 215: return volumes[215].SampleLevel(uvw, 0);
  case 216: return volumes[216].SampleLevel(uvw, 0);
  case 217: return volumes[217].SampleLevel(uvw, 0);
  case 218: return volumes[218].SampleLevel(uvw, 0);
  case 219: return volumes[219].SampleLevel(uvw, 0);
  case 220: return volumes[220].SampleLevel(uvw, 0);
  case 221: return volumes[221].SampleLevel(uvw, 0);
  case 222: return volumes[222].SampleLevel(uvw, 0);
  case 223: return volumes[223].SampleLevel(uvw, 0);
  case 224: return volumes[224].SampleLevel(uvw, 0);
  case 225: return volumes[225].SampleLevel(uvw, 0);
  case 226: return volumes[226].SampleLevel(uvw, 0);
  case 227: return volumes[227].SampleLevel(uvw, 0);
  case 228: return volumes[228].SampleLevel(uvw, 0);
  case 229: return volumes[229].SampleLevel(uvw, 0);
  case 230: return volumes[230].SampleLevel(uvw, 0);
  case 231: return volumes[231].SampleLevel(uvw, 0);
  case 232: return volumes[232].SampleLevel(uvw, 0);
  case 233: return volumes[233].SampleLevel(uvw, 0);
  case 234: return volumes[234].SampleLevel(uvw, 0);
  case 235: return volumes[235].SampleLevel(uvw, 0);
  case 236: return volumes[236].SampleLevel(uvw, 0);
  case 237: return volumes[237].SampleLevel(uvw, 0);
  case 238: return volumes[238].SampleLevel(uvw, 0);
  case 239: return volumes[239].SampleLevel(uvw, 0);
  case 240: return volumes[240].SampleLevel(uvw, 0);
  case 241: return volumes[241].SampleLevel(uvw, 0);
  case 242: return volumes[242].SampleLevel(uvw, 0);
  case 243: return volumes[243].SampleLevel(uvw, 0);
  case 244: return volumes[244].SampleLevel(uvw, 0);
  case 245: return volumes[245].SampleLevel(uvw, 0);
  case 246: return volumes[246].SampleLevel(uvw, 0);
  case 247: return volumes[247].SampleLevel(uvw, 0);
  case 248: return volumes[248].SampleLevel(uvw, 0);
  case 249: return volumes[249].SampleLevel(uvw, 0);
  case 250: return volumes[250].SampleLevel(uvw, 0);
  case 251: return volumes[251].SampleLevel(uvw, 0);
  case 252: return volumes[252].SampleLevel(uvw, 0);
  case 253: return volumes[253].SampleLevel(uvw, 0);
  case 254: return volumes[254].SampleLevel(uvw, 0);
  case 255: return volumes[255].SampleLevel(uvw, 0);
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
  if (is_bricks()) return bricks_sample(p);
  if (!is_slab()) return volumes[0].SampleLevel(p, lod);
  // voxel coords (units of 1/nx per axis)
  float wx = p.x * pc.slab_nx;
  float wy = p.y * pc.slab_nx;
  float zc = pc.slab_z0 + 0.5 + p.z * pc.slab_nx; // ring z, seam-safe by box extent
  float w = frac(zc / float(pc.slab_wz)); // REPEAT sampler wraps the ring
  float ovs = float((pc.slab_grid >> 16) & 0xffu); // overview downscale (4 or 8)
  if (lod >= log2(ovs) - 0.25) {
    // far field: 1/ovs-res overview of the whole composite (binding 3 in slab
    // mode) — pre-filtered, so wide zoom-outs neither alias nor cost full res
    float ox = ceil(pc.slab_nx / ovs), oy = ceil(pc.slab_ny / ovs);
    return occ.SampleLevel(float3(wx / (ovs * ox), wy / (ovs * oy), w), 0);
  }
  uint gx = pc.slab_grid & 0xffu, gy = (pc.slab_grid >> 8) & 0xffu;
  uint ix = min(uint(max(wx, 0.0) / pc.slab_px), gx - 1);
  uint iy = min(uint(max(wy, 0.0) / pc.slab_py), gy - 1);
  float tw = pc.slab_px + 2.0, th = pc.slab_py + 2.0;
  float u = (wx - float(ix) * pc.slab_px + 1.0) / tw;
  float v = (wy - float(iy) * pc.slab_py + 1.0) / th;
  return sample_tile(float3(u, v, w), iy * 16 + ix);
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
  } else if (is_bricks()) {
    voxel = 1.0 / float((pc.brick_mode & 0xffu) * 128u);
    max_lod = 3.0; // atlas has no mips; lod only widens far-field steps
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

  float bs = OCC_BLOCK * voxel; // skip-block edge, volume units
  // block max below this contributes nothing: low-cut OR zero TF alpha
  float occ_gate = max(pc.skip_gate, 0.5 / 255.0);

  // bricks mode: page entry cached across steps (rays cross bricks rarely)
  uint b_bpa = pc.brick_mode & 0xffu;
  float b_N = float(b_bpa * 128u);
  float b_inv = 1.0 / (float((pc.brick_mode >> 8) & 0xffu) * BR_SLOT);
  uint b_last = 0xFFFFFFFFu;
  bool b_ok = false;
  float b_max = 0.0;
  float3 b_slot = float3(0);

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
    float3 b_bl = float3(0); // bricks: local coords this sample
    float blod = 0.0;
    float v;
    if (is_clip()) {
      cl_lvl = pick_clip_level(p, lod);
      v = cl_lvl < 0 ? 0.0 : sample_clip_level(uint(cl_lvl), p);
    } else if (is_bricks()) {
      blod = clamp(lod, 0.0, 3.0); /* atlas has 4 mips */
      if ((pc.brick_mode & 0x10000u) != 0) {
        /* identity residency (v1 full shard): atlas layout == world layout,
         * so the page indirection is a no-op — sample like cube mode */
        v = brick_atlas.SampleLevel(p, blod);
      } else {
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
        if ((pc.brick_mode & 0x10000u) != 0) { /* identity: 8^3 occ only */
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
          if (is_bricks() && (pc.brick_mode & 0x10000u) != 0) {
            g = float3(brick_atlas.SampleLevel(p + float3(e, 0, 0), blod) -
                           brick_atlas.SampleLevel(p - float3(e, 0, 0), blod),
                       brick_atlas.SampleLevel(p + float3(0, e, 0), blod) -
                           brick_atlas.SampleLevel(p - float3(0, e, 0), blod),
                       brick_atlas.SampleLevel(p + float3(0, 0, e), blod) -
                           brick_atlas.SampleLevel(p - float3(0, 0, e), blod));
          } else if (is_bricks()) {
            /* taps reuse the cached brick (offsets in voxels) */
            float ev = lod_step;
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
