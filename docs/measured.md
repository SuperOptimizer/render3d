# measured.md — decision ledger

Every non-obvious decision gets a line here with the numbers (or probe output)
behind it. Convention inherited from ~/compressor.

## 2026-08-03 — target hardware capability dump (Turnip / Adreno X1-85)

Queried via vulkaninfo + design-phase probe on the dev machine (Snapdragon X
Elite, Mesa 26.0.3, Vulkan 1.4.335 device, kernel 7.0.0-32-qcom-x1e):

- `maxImageDimension3D` = 2048 → 1024³ monolithic texture OK; multi-shard views
  must use a brick atlas (never one giant texture).
- `maxMemoryAllocationSize` ≈ 4 GiB → 1024³ R8 + 11 mips (~1.14 GiB) fits one
  dedicated allocation.
- subgroup size = 128 → workgroup candidates 16×8 (one wave) vs 8×8; decided by
  spec-constant sweep, not folklore.
- `maxComputeWorkGroupInvocations` = 2048, `maxStorageBufferRange` = 128 MiB.
- Features present: synchronization2, timelineSemaphore, maintenance4,
  hostQueryReset, **VK_EXT_host_image_copy** (R8_UNORM optimal tiling reports
  HOST_IMAGE_TRANSFER | SAMPLED_FILTER_LINEAR | BLIT_SRC | BLIT_DST).
- OpenGL 4.6 core with ARB_gl_spirv + ARB_spirv_extensions (GL backend can share
  SPIR-V or take Slang's GLSL output).

## 2026-08-03 — shader language: Slang

User decision (over GLSL/glslc and HLSL/dxc): one `.slang` source targeting
SPIR-V for Vulkan now; GLSL/Metal/DXIL targets keep future backends open.
Note for the record: graphics shader compilation is not part of clang/LLVM —
LLVM's SPIR-V backend serves compute/OpenCL kernels; graphics goes through
offline compilers (slangc/glslc/dxc). slangc pinned + vendored via
`tools/fetch_slang.sh` (not packaged for Ubuntu aarch64).

## 2026-08-03 — M1 data source

Reuse ~/compressor/corpus/full: verified contiguous 8×8×8 grid of 128³ u8
bricks (PHercParis4 z230-237/y136-143/x122-129, all 512 files present) →
`tools/assemble` concatenates into 1024³ `volume.u8`. No network needed.

## 2026-08-03 — GUI: cimgui

User decision: GUI via cimgui (C bindings over Dear ImGui) + imgui SDL3/Vulkan
backends, vendored pinned like slang. Project code stays pure C; only the
vendored imgui core compiles as C++.

## 2026-08-03 — M1 step 11: 1024³ scale-up (dev preset, 1280×720 FIFO)

- 1 GiB R8 + 11 mips uploads in 339 ms via host-image-copy (3.17 GB/s) vs
  **251 ms via staging slabs (4.28 GB/s)** — staging wins on Turnip today;
  host-image-copy kept as default anyway (no staging RAM spike, and the gap
  is one-time init cost). Revisit if upload becomes interactive-path.
- GPU mip chain (10 3D blits): 22–29 ms.
- Exterior view full shading: 2.3 ms GPU; interior dense-papyrus view: 8.0 ms.
  62 fps vsync-locked at 720p. Ray-cone LOD + adaptive dt active.

## 2026-08-03 — M1 step 12: perf sweep (release/ThinLTO, 1920×1080, uncapped, 1024³ scroll)

Workgroup sweep (mode FULL, scroll TF):
- **16×8 (one wave): 4.33 ms exterior / 3.62 ms interior — winner, default**
- 8×8: 4.40–4.50 ms; 16×16: 4.39–4.50 ms (all within ~5%; wave-sized wins)

Mode cost (exterior view):
- FULL (shaded): 4.34 ms · FLAT (no shading): 2.24 ms → gradient shading ≈ +2 ms
- HEATMAP: 7.1 ms · MIP: 17.2 ms (no early termination — expected worst case)

Verdict: 60 fps @ 1080p target exceeded ~4× (225 fps shaded). CPU frame cost
4.4 ms in lockstep with GPU (single queue, 2-in-flight pacing works).
Thermal soak (cold vs 10-min) still to be recorded before claiming sustained.

## 2026-08-03 — cimgui integration (M1 addendum)

cimgui 1.92.9 vendored (tools/fetch_cimgui.sh) with bundled imgui + SDL3/Vulkan
backends compiled as C++ (`IMGUI_IMPL_API=extern "C"`,
IMGUI_DISABLE_OBSOLETE_FUNCTIONS to avoid C-linkage overload clash); all
project code stays C via cimgui.h. GUI draws as a dynamic-rendering LOAD pass
directly on the swapchain image after the raycast blit (swapchain gained
COLOR_ATTACHMENT usage; dynamicRendering feature now required). GUI cost is
lost in the noise at 60 fps vsync (gpu 3.8 ms with panel vs 3.6 without).

## 2026-08-03 — profiling infrastructure + optimization pass (release, 1080p uncapped, real 1024³)

Instrumentation: GPU timestamp zones (raycast/blit/gui) + CPU phases
(wait/acquire/record/submit) in r3d_frame_stats; GUI "profile" section;
`profile avg` exit line; scripted camera benches (`--bench orbit|zoom|fly`) +
tools/perf.sh suite.

Baseline finding: 100% raycast-bound (blit 0.07 ms, gui 0.01, CPU phases <0.5).
Low-cut made frames SLOWER (4.8 vs 4.4 ms) — rays sampled voxels just to cut
them. Bench orbit was the true worst case: 35.6 ms.

Optimizations, measured one at a time:
1. **8³-block occupancy skip** (CPU-built max pyramid, 26-neighbor dilation for
   trilinear safety, ~0.4 s build at upload; nearest-sampled binding 3).
   First cut checked occ EVERY step → dense views regressed 15% (16.6→19.2 ms).
   Fix: consult occ only when the fine sample is already below the gate —
   dense paths pay zero. fly 9.6→8.6, zoom 7.8→7.0, dense views back at
   baseline. Verified conservative: gpu conformance test unchanged, lowcut
   image diff 0.13 LSB.
2. **Continuous LOD step scaling** (`exp2(lod)` not `exp2(floor(lod))`):
   fractional ray-cone lod now widens dt smoothly. orbit 33.2→**19.7 ms**
   (−41%), exterior 4.6→**3.3 ms**. Image diff vs pre-pass render: 0.16 LSB.
3. MIP saturation early-exit (mip_max≥0.999 break): no measurable change on
   scroll data (max rarely saturates); kept, it's free.

Result: orbit 35.6→19.7, exterior 4.5→3.3, fly 9.6→8.8, zoom 7.8→7.2 ms.
Static interior dense stays ~16.5-17.6 ms — genuinely dense near field at
step=1; the honest fix is quality-trading lod_bias (slider) or future
per-brick adaptive sampling, not marching tricks. Run-to-run thermal noise
is ±10% — trust deltas above that.

## 2026-08-04 — tiled slab renderer with z-scrolling (release, 1080p uncapped)

Wide thin z-window over big volumes without atlas/page-table machinery: up to
2x2 tile textures (payload 2046 + 1-voxel apron = 2048 cap; composite max
4092²), ring-buffered z (REPEAT-W sampler, wz-1 seam-safe slices), scrolled by
host_image_copy into permanently-GENERAL tiles after a timeline wait.

- Apron correctness: seam 2nd-difference at tile boundary == reference regions
  (synthetic continuous pattern, 2x2 @ 3072²).
- Real data: fetched 3072x3072x96 (0.84 GiB) from PHercParis4 zarr in ~3 min
  (tools/fetch_slab.py); renders seamlessly across tiles.
- Perf (zsweep bench, continuous scrolling): 1-tile 1024² = 4.3 ms; 2x2 3072²
  = 12.2 ms. Window jump (32 slices, 4 tiles) ~105 ms dev / ~50 ms release;
  per-slice scroll a few ms.
- Known gap: slab tiles have no mips (max_lod forced 0) → zoomed-out wide
  views march at full voxel pitch (21.7 ms at 720p dev on 3072² overview).
  Follow-up: per-tile mip chain or coarse overview texture.

## 2026-08-04 — 4x4 slab grid (8184² composites) + overview LOD

- Grid extended to 4x4 (16-slot descriptor array, literal-index switch);
  composite cap now 8184². Seam metrics clean across all 3 internal
  boundaries (synthetic pattern).
- **Overview LOD**: whole composite at 1/4 res always fits ONE texture
  (8184/4 = 2046), same ring z, bound at the occupancy slot (unused in slab
  mode); shader switches to it at ray-cone lod >= 1.75. Kills far-field
  aliasing AND cost: full-field view of real 8184² = **4.4 ms GPU / 62 fps**
  (3072² overview was ~20 ms before this). Slab lod_step clamped to
  depth/4 so thin slabs keep >=4 samples through their thickness.
- Real data: 8184x8184x48 (3.2 GB) fetched from PHercParis4 zarr (~13 min).
  Initial window (34 slices x 16 tiles + overview) 3.2 s; scroll ~95 ms/slice
  at this width — CPU assembly/downsample is single-threaded; threading it is
  the next lever if wide-slab scrolling needs to be smoother.

## 2026-08-04 — streaming clipmap: full 43k² PHercParis3 cross-sections

Data path: user's dct3d-sharded zarr v3 export (1024³ shards, 16³ chunks,
~56× compression) → local band Z=33 (1764 shards, 33 GB, ~75 min download)
→ mkpyramid L2-L5 (33 GB, 9 min, 0.3 s/shard) → 6-level XY clipmap
(2046-payload octaves, per-level ring z, async worker fill: L0/L1 live
threaded dct3d decode at 19.4 µs/chunk, L2-L5 pyramid mmap).

Bugs caught during bring-up: fills enqueued fine-to-coarse starved L5 behind
~10 s of L0/L1 decode (fix: coarse-first, screen never empty); fly camera had
no focus (fix: view-axis × slab-plane intersection); per-sample 6-level walk
with a 16-case switch inside the march loop cost 79-112 ms zoomed (fix: pick
level once per sample, gradient taps reuse it → 3.2× faster).

Perf (release, 1080p uncapped): whole cross-section 2.6 ms; zoomed full-res
32 ms; clippan sweep (recenter churn incl.) 33 ms. Default 720p vsync: 60 fps
everywhere. Recenter refill: L5..L2 < 200 ms; L1 ~1 s, L0 ~0.5 s (visible as
brief blur, never a stall). Known gaps: dct3d 16³ block seams visible at L0
(codec quality 32, no deblocking — c5d's normative deblock will fix);
L0 slice fills decode each 16-slice chunk row per slice (16× redundant work,
chunk-cache would cut recenter to ~60 ms).

Scaling: architecture handles the full 68608-slice volume once more bands are
fetched; textures stay 6×~150 MB regardless of extent.

## 2026-08-04 — model transform + camera controls

Volume (model) transform in push constants (R rows + translation, scalars to
dodge std430 float3 16-alignment; 216 B total, device max 256 checked at clip
init). Rays transformed world->volume once per pixel; works in cube/slab/clip;
clip focus intersection done in volume space. Identity transform keeps the
gpu conformance test bit-stable. Gestures: drag=orbit, shift=pan camera,
ctrl=translate volume, ctrl+shift=rotate volume; GUI numeric transform
section + fov; --volpos/--volrot for headless runs.

## 2026-08-04 — c5d GPU decode + GPU-resident brick cache (--bricks)

Target architecture proven end to end: .c5s shard -> compressed brick blobs
(only bytes that cross to the GPU) -> batched GPU decode (c5d's entropy/
dequant-IDCT/deblock kernels from the compressor working tree, flat batched
entropy dispatch) -> our pack.comp imageStores u8 into an R8 atlas 3D image
(the GPU cache) -> raycast samples via a page-table SSBO (entry = slot |
brick_max<<24; the max byte gives whole-brick empty skipping).

- Conformance (test_c5dgpu, both paths): 16.7M voxels, 10 at 1 LSB, 0 above.
- volume.u8 1 GiB -> volume.c5s 44.4 MB (24.2x, q=2) at 3.6 GB/s encode (12T).
- Bricks-mode render matches cube mode at 0.875 mean LSB (codec error, not
  renderer error); 60 fps vsync, GPU 10.2 ms vs cube's ~4.5 (page-table
  lookup + per-sample clamp; optimization headroom noted).
- Atlas fill, 512 bricks (1 GiB raw): full-GPU 7.9-8.4 s (61-64 bricks/s,
  0.14 GB/s — upstream's naive 32-lane entropy kernel is the bottleneck);
  hybrid (R3D_C5D_HYBRID=1) 10.8 s here because our he_decode calls are
  serial per brick (gputest's hybrid numbers pre-stage; threading he_decode
  across bricks would flip this). Either way: one-time fill cost.
- Real data: c5dpack band 33/20/20 (3ddct -> c5d transcode) = 16.3 MB for
  1 GiB raw (65.8x on already-lossy source; generational loss accepted for
  the bridge period until native c5d exports exist).
- Known gaps -> follow-ups: brick-border seams (no aprons, brick-local
  deblock), no streaming/LRU residency yet (v1 = full static shard),
  atlas has no mips, entropy kernel speed is upstream work.

## 2026-08-04 — perf regression: mega-shader → per-mode pipeline variants

Adding bricks mode made raycast.slang carry FOUR sampling architectures in
one kernel; register pressure cost cube mode 6.2x (3.3 -> 20.5 ms exterior
@1080p). Fix: R3D_MODE compile-time specialization — four SPIR-V variants
(cube/slab/clip/bricks), backend binds by frame params. After: cube 3.41 ms
(baseline restored), bricks 28.3 -> 10.5 ms, clip whole-view 2.6 -> 1.9 ms.
Lesson recorded: every new sampling mode gets its own pipeline variant.

Same day: resynced vkc5d to upstream's committed sparse-entropy kernels
(7a9909c, pairs+counts output, 6.5x entropy speedup): our conformance
gate stays at 10/16.7M voxels at 1 LSB, 0 above; engine decode 8 bricks
45 -> 15 ms full-GPU.
