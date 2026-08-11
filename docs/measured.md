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

## 2026-08-04 — resync to compressor HEAD (75938e1): packed-u8 + v1.4 nsub=128

Upstream changes absorbed: Vol buffer now packed u8 (4 voxels/word — our
pack.comp reads words, vol buffer 4x smaller), deblock dispatch total/4 on
axes y/z, sub/status sized for nsub<=128, he_gpu slot2sym may be NULL
(shared-mem binary search path). c5dpack + test_c5dgpu now encode with
p.nsub=128 (v1.4 GPU knob). Conformance unchanged (10/16.7M at 1 LSB).
512-brick atlas fill: full-GPU 8.1 -> 3.3 s (157 bricks/s, 0.33 GB/s) and
now FASTER than hybrid (4.8 s) — compressed-bytes-to-GPU wins as intended.
Gap to upstream's 2.35 GB/s streaming bench = their 3-deep pipelined
submission vs our serial batch+fence fills; follow-up when fills need to
be interactive (streaming residency milestone).

## 2026-08-04 — bricks-mode optimization pass (release, 1080p uncapped)

Baselines: exterior 10.7 ms, orbit 19.2, close/dense 58.9 (vs cube ~17).
1. Atlas mip chain (4 levels, whole-atlas blits; sampling inset grows
   0.5*2^lod texels so coarse mips never bleed across slots) + lod-aware
   fetch: exterior 10.7 -> 4.6, close 58.9 -> 32.6.
2. Atlas SHADER_READ_ONLY_OPTIMAL after fill: no measurable change on
   Turnip (kept — correct for drivers where GENERAL defeats compression).
3. Brick-context caching across marcher steps (page resolve only on brick
   crossings; gradient taps reuse slot+local coords): close 32.6 -> 25.7,
   exterior -> 4.05.
Net: exterior 2.6x, close 2.3x; default-720p worst view ~11 ms -> 60 fps.
Bricks-mode diff vs cube render improved 0.875 -> 0.317 mean LSB (mips
filter like cube's now). Remaining gap vs cube on dense interiors
(25.7 vs ~17 @1080p): cube's 8^3 occupancy skips intra-brick gaps; bricks
skip is 128^3-granular. Proper fix = hierarchical occupancy in the page
table — belongs to the streaming-residency milestone.

## 2026-08-04 — optimization round 2: GPU occupancy + TF-aware skip gate

Added: GPU-built 8^3 occupancy for bricks mode (occmax + occdilate kernels
reduce the atlas post-fill; valid under v1 identity slot mapping) wired into
the marcher with a 4.5-LSB codec-noise floor, and a TF-aware skip gate
(pc.skip_gate = max(low_cut, first-nonzero-TF-alpha)) — an EXACT skip that
benefits all modes and all future TFs. Measured: lowcut-style gating fires
well (dense close 26.3 -> 21.1 ms); the default scroll-TF interior view does
NOT improve (~26-28 ms) because its inter-sheet gaps (10-20 vox) are below
the dilated 8^3 skip granularity (~24 vox) — rays legitimately march there.
Remaining bricks-vs-cube gap on dense interiors ~1.5x = per-sample page-cache
branch + clamp/slot ALU on 7 fetches; further gains need aprons (drop the
clamp) or finer skip trees. At the default 720p window every view holds
60 fps. Thermal drift across long bench sessions is +-10-15% — compare
paired runs only. Generic r3d_vkcomp helper added to vkres for one-off
compute passes.

## 2026-08-04 — optimization round 3: apron experiment FAILED, reverted

Attempted 132^3 slots (2-voxel aprons copied from neighbors via a new
apron.comp; goal: seamless cross-brick trilinear + drop the fetch clamp).
Result: deterministic ~16-32 voxel tiling artifacts whenever sampling
lod > 0, spatially patterned, identical with aprons on/off and in both
GENERAL and READ_ONLY layouts. Bisection: mip0-only sampling clean (0.874);
any mip1 use corrupt (7.4); yet mip1 CONTENT dumps clean at three z planes
(boundary + interior + deep). Sampling math hand-verified repeatedly.
Root cause not found — suspected Turnip interaction with the non-pow2
slot pitch (132) in the 3D mip chain. Reverted to 128-slot/4-mip geometry
(diff back to 0.317; exterior 4.3 ms, close ~29 ms). Scaffolding kept
behind APRON>0 + R3D_NO_APRON + R3D_DUMP_MIP1 for a future retry; next
attempt should try 136 (8-aligned) slots or per-slot compute-generated
mips instead of whole-atlas blits.

## 2026-08-04 — optimization round 4: adaptive resolution

Half-res rendering while the camera moves (drag/pan/zoom/scroll/keys), full
res once settled (~1/4 s): p.viewport drives dispatch + blit-source region,
LINEAR upscale blit (identity at full res). Captures and benches always run
full res; R3D_FORCE_HALF tests the path. Worst-case dense view during
interaction: 29 -> 8.4 ms @1080p (4x fewer rays) — navigation is 60 fps
everywhere including 1080p windows; stills render at full quality.

## 2026-08-04 — optimization round 5: identity residency fast path

v1 full-shard residency maps slot == brick index, making the atlas layout
IDENTICAL to world layout — the page-table indirection is provably a no-op.
Backend detects this (brick_mode bit 16) and the shader samples the atlas
directly at p, cube-style; the indirected path remains for future streaming
(non-identity) residency. Dense close 29 -> 21.9 ms @1080p — parity with
cube mode; exterior 4.0 ms; diff vs cube improved 0.317 -> 0.196 and brick
seams vanish (continuous sampling, no slot borders). The indirected-path
costs return only when streaming remaps slots — at which point aprons
(retry plan in round 3) become the relevant fix.

## 2026-08-05 — streaming milestone: two-tier GPU cache (warm compressed / hot atlas)

Bricks mode now runs volumes larger than the atlas. WARM tier: compressed
c5d blobs cached in a host-visible device buffer (default 256 MB, --warm MB;
offset-sorted first-fit allocator, 64 B granules, LRU eviction; blobs bigger
than the tier or a thrashing tier fall back to the mmap'd shard). HOT tier:
the R8 atlas as pool_bpa^3 LRU slots (--pool N, default 8; identity full
residency + direct sampling whenever the volume fits). Per frame BEFORE
r3d_frame, r3d_bricks_stream() computes the desired brick set (view cone
widened 1.15x + brick radius, plus a near-camera sphere; camera transformed
to volume space so model transforms are honored), stamps LRU on residents,
sorts requests nearest-first, and decodes up to budget bricks (2 while the
camera moves, 6 idle) warm->hot. Decoded-empty bricks (max < 5 LSB codec
floor, or below the TF-aware gate) never occupy slots and are never
re-requested. Post-fill per decoded slot: 3 slot-region mip blits +
region-form occupancy (occmax slot->world, occdilate brick+1-block halo —
neighbor borders self-heal as fill order interleaves). Occupancy images are
world-indexed and persist across evictions (stale entries are safe: the
page-table whole-brick skip short-circuits first). Atlas stays GENERAL for
life in bricks mode (measured identical to READ_ONLY in round 3).

Measured (volume.c5s 512 bricks, pool 5^3 = 125 slots, warm 16 MB < 44 MB
shard, 720p, tf 1, no-vsync):
- close-up fill: ~20 fps during the ~20-frame initial stream-in (budgeted
  synchronous decode, ~6.4 ms/brick GPU), then 111 fps settled; screenshot
  vs cube mode mean 2.16 LSB, 0.05% pixels > 32 — all of it the documented
  non-identity brick-border seam cross + amplified codec noise; zero missing
  or misplaced bricks. (Identity path after the refactor: mean 0.100.)
- zoom bench (continuous residency churn): 72-110 fps throughout.
- orbit with whole volume visible (working set 4x the pool): thrash guard
  (never evict a slot wanted this same frame) stops decode cleanly; nearest
  125 bricks render, 440 fps, no churn. Proper answer for this case is
  LOD-fallback bricks (coarse-resolution residency for far bricks) — next.

Decodes drain the queue (vkQueueWaitIdle) before overwriting slots/pages
sampled by in-flight frames — pipelined handoff via the timeline semaphore
is the known follow-up, as is zero-copy entropy reads straight from the warm
buffer (needs c5d to expose the payload offset in he_gpu; today he_gpu_setup
stages a CPU copy of the payload per decode, ~50 KB/brick, negligible).

## 2026-08-05 — slab mode: 8x8 tile grid (16368^2 composites)

Max slab composite doubled per axis: 8184^2 -> 16368^2 x up to 32-deep
(R3D_SLAB_MAX_GRID 4 -> 8; vol[] descriptor array 16 -> 64, literal-index
switch; tile stride j*8+i). The single-texture far-field overview
generalizes from fixed 1/4-res to a computed downscale (ovs = 4 while the
composite fits 2046, else 8), carried in slab_grid bits 16+; the shader
switches to it at lod >= log2(ovs)-0.25. Descriptor pool sized 64+3.
New tools/bandcut: threaded dct3d region decode of the local band ->
flat .u8 (16368^2 x 32 of real PHercParis3 = 8.6 GB in ~2 min).

Measured (bigslab.u8 16368^2x32, --slab 16, 720p, tf 1, no-vsync):
- initial 18-slice window fill: 7.4 s (host_image_copy across 64 tiles +
  1/8-res overview build); GPU memory ~4.9 GB of the 23 GiB heap.
- whole-composite view after fill: 329 fps (1.9 ms GPU) — the overview
  path keeps zoom-outs at 8184-slab cost.
- z-scroll: 133 ms/slice (267 MB assemble+upload+downsample per slice,
  4x the 8184 cost, CPU-bound in the slice assembly); continuous zsweep
  averages ~37 fps. Threading the per-tile assembly is the lever if
  faster scrolling at 16k matters.
- 8184 regression: unchanged (312 fps, fill 1.8 s); quick-test layout
  cases extended to 8x8 + ovs; all 4 suites green.

## 2026-08-05 — slab mode: 16x16 grid (32736^2 — the whole scroll in one slab)

Grid cap 8 -> 16 (256-tile array + 256-case literal switch; everything else
was already R3D_SLAB_TILES-driven). ovs reaches 16 at 32k. Measured
(bigslab32k.u8 = 32736^2 x 16 real band crop, 17.1 GB, --slab 8, 720p):
window fill 10 slices in 17.9 s; whole-composite view 384 fps (1.6 ms —
the ENTIRE PHercParis3 cross-section is visible in one view); GPU ~10.7 GB
of 23 GiB. 32k x 16-deep (ring 18, ~19.3 GB) fits arithmetically but was
deliberately not defaulted: it leaves ~3 GB for OS + source page cache.
Scroll at 32k = ~1 GB/slice assembly (untested rate; threading the
assembly is the known lever). Suites green; 8184/16368 layouts unchanged.

## 2026-08-05 — slab: whole-scroll plane via bbox trim; max_lod bug at ovs>16

Goal "whole 43008^2 plane, 8 deep" (grid cap now 22): the untrimmed plane
(18.5 GB GPU + 30 GB source mmap on a 30 GB machine) filled in 24 s but
crashed intermittently — memory-pressure territory, not pursued. The
export has wide empty margins: occupied bbox at the band is only
35616 x 23840. Trimmed extract scrollplane.u8 (35712 x 24000 x 16,
13.7 GB; numpy crop of the full-plane file) renders the ENTIRE scroll:
depth 4 = 5.2 GB GPU, depth 8 = 8.6 GB, whole-composite view 0.6-1.0 ms.

Bug found on the way: slab max_lod was hardcoded 4.0 (ovs=4 era). With
ovs=32 the overview switch sits at lod 4.75 — unreachable past the clamp,
so wide views marched full-res tiles forever (24.6 ms). At ovs=16 (32k)
the 3.75 threshold cleared it by luck. max_lod is now log2(ovs)+2.
Lesson: thresholds derived from a scale knob must be checked against
every clamp on the same quantity.

## 2026-08-05 — slab optimization: overview pyramid + native descriptor indexing

New benches: zoomio (whole-composite -> voxel scale -> back, log-space) and
volrot (model rotation at mid-zoom). Baselines on the whole-scroll slab
(35712x24000x8, 720p full-res): zoomio 96.2 ms avg, volrot 74.1 ms — the
band between full-res tiles (lod 0) and the single 1/ovs overview had no
prefiltered data, so mid-zoom samples thrashed the texture cache. Occlusion/
viewport culling would not help: rays already exit on miss and terminate at
0.98 alpha; the cost was unfiltered sampling.

Fix 1 — tiled overview pyramid: prefiltered levels at scale 4<<lev up to
ovs (full ring z, never downsampled in depth), each a virtual slab layout
appended to the tiles[] pool; shader picks the level matching the ray-cone
footprint (all-float geometry math — Adreno has no integer divide). Fill
builds levels by 4x box + 2x chain per slice (~7% extra CPU, ~700 MB GPU
at this size).

Fix 2 — the real assassin: the "literal indices" sample_tile switch. Fine
at 16 cases (M1), it silently became a ~20x per-sample tax as the pool
grew (544 texture instructions in a branch tree; whole view 1.0 -> 21 ms
just from routing through it). Turnip reports
shaderSampledImageArrayNonUniformIndexingNative, so the switch is now one
NonUniformResourceIndex sample and the 1.2/DI features are enabled at
device create. Lesson: re-audit "no feature needed" workarounds when the
constant that motivated them grows.

Results (same bench, same data): zoomio 96.2 -> 2.83 ms (34x, worst frame
7.5 ms), volrot 74.1 -> 2.17 ms (34x), whole view unchanged 1.0 ms,
mid-zoom stills ~2.9 ms and visually clean (no seams/banding). Every zoom
level of the whole scroll now renders at 60 fps with headroom. Suites
green.

## 2026-08-05 — upstream c5d sync: tau corrections on GPU + free dq speedup

Pulled ~/compressor HEAD (73934ef): GPU tau support (corrections.comp,
post-deblock CAS scatter; he_gpu/he_decoded expose (voxel,delta+512)
pairs; he_gpu_setup no longer rejects FLAG_TAU), dq flat-chunk fast path,
NEON CPU work, flat-chunk IDCT skip. render3d side: corrections pipeline +
6 MB/brick pair staging in vkc5d (both legs), kernel added to the SPIR-V
build, test_c5dgpu now encodes half its bricks q4/tau2 (~700 pairs each,
verified flowing) with the amended contract (<=1 LSB plus <=8 bounded
gate-flip voxels). Existing push structs unchanged this time.

Results: all suites green both legs; bricks render vs cube unchanged
(mean 0.100); full-GPU atlas fill 144 -> 228 bricks/s (0.30 -> 0.48 GB/s)
free from upstream's dq fast path. GPU-path encodes may now use tau
(lossless still rejected); c5dpack stays tau=0 by default.

## 2026-08-05 — virtual slab: 3-axis streaming window over the ENTIRE export

New mode --vslab [--vswin W H D] [--vsz Z]: a W x H x D window positioned
anywhere in the full 43008^2 x 68608 volume. World-anchored TOROIDAL tile
grids per level (base + pyramid 4/8/16/32; payload 2016 = 63*32 so every
pyramid texel's source box lies in ONE base cell): scrolling any axis only
decodes ENTERING cells/strips — resident texels never move or re-upload.
Pyramid content is a by-product of base fills (4x + 2x chain per layer,
scattered into overlapped pyramid tiles with sub-rect host copies) and
inherits base validity: a key table on binding 5 gates sampling per base
cell, so unfilled regions render empty and refine as jobs land (R3D_MODE=4
pipeline variant; slab_grid bit 24; window origin = new push floats
slab_x0/y0, 224 -> 232 B). Sources: local shard cache (band/<Z>_<Y>_<X>)
with REMOTE FETCH on miss from the dl.ash2txt.org export (curl per shard,
404 -> .missing marker; R3D_VSLAB_NOFETCH=1 disables). Fills are
synchronous, budgeted in slices (~1 full cell per frame; z strips chunk-16
aligned since dct3d decodes whole chunks). Window follows the camera focus
(GUI toggle) + world-z slider over all 68608 slices.

Measured (12096^2 x 16 window = 70 tiles / 5.1 GB, 720p, band z=33):
- initial fill ~4 s (49 cells x ~80 ms decode), then 0.6 ms whole-window.
- xy pan bench (window slides 18k voxels): 51 fps avg, worst frame 104 ms
  (synchronous fills — async worker is the known follow-up).
- z sweep at 2.6 slices/frame (whole band in 7 s — far beyond real
  scrubbing): ~11 fps, throttled by the slice budget; validity blanks
  outrun cells until fills catch up.
- remote: window dropped into never-fetched z-row 32 pulled 49 shards from
  the export automatically and rendered fresh scroll data end to end.

Follow-ups: async fill worker (+ prefetch margin), parallel shard fetch,
multi-window z rings > 62, vslab zoomio-class benches.

## 2026-08-06 — vslab async fill worker + z-range validity + parallel fetch

Fills (fetch + decode + downsample + host_image_copy) moved off the render
thread onto ONE worker (pthread, vkclip pattern): the render thread now only
enumerates needed cells nearest-first, hands up to `budget` jobs to a 4-slot
queue, and folds completed jobs into the cell ledger. Write safety without
render-thread waits: a fresh cell's page key is cleared at enqueue and the
worker drains then-in-flight frames (timeline wait at the ENQUEUE value)
before touching the tile, so later frames see key=0 and never sample it;
z-strip refills only write layers outside the published validity range.
Validity grew from 1 to 4 words per base slot {key, za, zb, pad}: the shader
gates each sample's z taps against [za, zb), so PARTIAL z coverage renders
immediately (z scrolling shows data as strips land instead of stale ring
layers — also fixes the pre-existing stale-z artifact during fast sweeps).
Missing shards now download in batches of up to 6 concurrent curls.

Measured (12096^2 x 16 window, 720p, band z=33, same benches as 08-05):
- z sweep: ~11 fps -> vsync-locked 60 fps (worst frame 566 -> 50 ms).
- xy pan: 51 fps / worst 104 ms -> vsync-locked 60 fps / worst 38 ms.
- remote: 9 never-fetched z-row-31 shards pulled in two concurrent batches
  (6 + 3) mid-bench with zero frame stalls (gpu avg 1.05 ms throughout).
- decode itself was already threaded (r3d_shard_decode_region nthreads=0 ->
  all cores); the win is purely taking it off the frame loop.

Follow-ups: prefetch margin (fill one cell ring beyond the window in the
motion direction), multi-window z rings > 62, vslab zoomio-class benches.

## 2026-08-06 — vslab prefetch ring + zoomio/volrot benches

Prefetch margin: while the window origin is moving, the render thread also
enqueues (lowest priority, only into idle queue slots) the cell ring ONE step
beyond the window edge in the direction of motion, so pans arrive on
already-resident data. The base grid grew a second spare column/row
(ceil(W/cs)+2; 12096^2x16 window: 7x7 -> 8x8 base tiles, 70 -> 85 total,
5.1 -> 6.2 GB) because the original +1 straddle column is consumed by any
unaligned window — without it the toroidal span check (which stops a prefetch
from evicting a wanted cell) could never pass. Pyramid grids stay +1 (their
content rides along with base fills). R3D_VSLAB_NOPREF=1 disables.

New metric: benches accumulate `pending` (cells short of full residency) over
the SECOND half of the run — steady state, past the initial window fill.
clippan 1200 frames: 475 pending cell-frames without prefetch -> 0 with.
The sweep never touches a non-resident cell.

zoomio/volrot now run in vslab mode (window extents instead of slab dims):
both vsync-locked at 720p — zoomio gpu avg 9.1 ms, volrot 6.2 ms. The
overview pyramid + validity gating hold through full zoom sweeps and model
rotation over the streamed window.

Follow-up remaining: windows deeper than D=62 (multi-ring z).

## 2026-08-06 — vslab: arbitrary window shapes (adaptive payload + band fills)

The window is now any W x H x D that fits in RAM: tile payload adapts at
runtime (~extent/8, 32-aligned, capped 2016) instead of the compile-time
2016^2, D lifted from 62 to 2044 (ring wz = D+2 <= maxImageDimension3D 2048;
px auto-shrinks if a tile would pass the 4 GB allocation cap), straddle +
prefetch margins are PER AXIS and vanish when an axis spans the whole volume
(it cannot move), and pyramid levels exist only while window/scale >= 1024
(the shader steps down to the finest present level; max_lod follows). Tile
pool 544 -> 1024 descriptors (device limit is 16M); >20 GB tile totals are
refused at begin.

Deep windows exposed decode amplification: a fresh 258^2-cell fill re-decodes
whole 1024^2 dct3d chunks — ~60x waste, minutes for a 2048^3 fill. Small-xy
windows (exactly the no-pyramid case, max extent < 4096) therefore switch to
BAND fills: per-cell jobs sharing a z range group into one window-wide
16-slice decode scattered to every cell tile (~2x amplification, the chunk
row is decoded exactly once).

Measured (720p, band z=33 local):
- 2048x2048x1024 (payload 256, 100 tiles, 6.8 GB): full fill = exactly 65
  band jobs (= 1026/16 chunk rows, zero redundant decode) in ~13 s,
  renders real scroll data, 60 fps.
- 43008x43008x4 whole cross-section (payload 2016, 534 tiles, 13.0 GB):
  streams the ENTIRE scroll plane, whole-view gpu 2.2 ms.
- 12096^2x16 default regression: payload adapts to 1536 (125 tiles, 5.3 GB),
  clippan steady-state pending still 0 with prefetch, zoomio 9.1 -> 7.8 ms.

Note: camera-follow slides small windows continuously during orbits (the
focus intersection swings); fills keep up asynchronously but a follow
deadband for small windows is a possible refinement. R3D_VSLAB_DEBUG=1
prints per-job traces.

## 2026-08-07 — PHerc1218 annotation z-prefetch

The umbilicus workflow jumps between thin, complete 8192x8192 planes, usually
100 slices apart. Before this change each jump waited for the 3ddct shards to
be decoded and then uploaded. A separate CPU worker now prepares ordered nearby
z windows after the current GPU window becomes resident: five annotation steps
in the last navigation direction, then one behind it. Six LRU slots retain the
decoded 10-slice windows (the visible 8 plus filtering apron), 640 MiB each /
3.75 GiB maximum. The foreground upload worker copies a cache hit directly into
its cell staging box. Fetch publication remains serialized between the workers;
decoding runs with six threads. `--ann-prefetch N` selects 0-5 forward steps;
`R3D_VSLAB_NOPC=1` is the comparison/low-memory escape hatch.

Measured on the complete local PHerc1218 mirror, 420-frame `annstep` benchmark,
z=1600 to z=1700:

- no decoded cache: 2053 pending cell-frames;
- decoded cache: 1357 pending cell-frames (33.9% fewer), 64/64 destination
  cells served by cache;
- cached and uncached final renders were byte-identical (matching SHA-256);
- nearby 10-slice full-plane decodes usually completed in 200-500 ms, with a
  2.0 s cold outlier in this run;
- a cold standalone 8192x8192x16 decode took 2.07 s and about 1.0 GiB RSS.

The automated `tools/bench_ann_prefetch.sh` stress test holds the initial plane
until look-ahead is ready, then makes six +100 moves 0.6 seconds apart. Identical
600-frame runs on this machine:

| planes ahead | pending cell-frames | cache hits | cache misses |
|-------------:|--------------------:|-----------:|-------------:|
| 0 | 12609 | 0 | 0 |
| 1 | 12300 | 56 | 193 |
| 3 | 11326 | 169 | 128 |
| 5 | 8938 | 273 | 127 |

Five-ahead reduces pending work 27.3% relative to one-ahead and 29.1% relative
to no decoded cache. It is therefore the default on this 30 GiB machine.

Fine one-slice wheel movement exposed a separate issue: the original GPU ring
contained only the visible 8 slices plus 2 filtering layers. Even when 3ddct
data was cached elsewhere, each detent had to upload a new layer to all 64 base
cells and their overview tiles. Annotation mode now keeps a symmetric
contiguous GPU z margin and distinguishes visible residency from background
margin completion. Visible jobs always win; travel-side margin jobs fill in
16-slice breadth-first rounds, followed by the opposite side. Base and pyramid
z stacks are submitted as 3-D batches instead of one Vulkan submission per
slice.

`annwheel` holds the initial view, then traverses 20 adjacent slices at ten
detents/second. With the CPU jump cache disabled to isolate GPU residency,
identical 600-frame runs measured:

| GPU margin per side | ring depth | tile memory | visible pending cell-frames |
|--------------------:|-----------:|------------:|----------------------------:|
| 0 | 10 | 0.7 GB | 8577 |
| 32 | 74 | 5.4 GB | 0 |

The fully resident static outputs are visually equivalent (SSIM 0.9965). Their
non-identical bytes come from normalized 3-D sampling with different ring
periods, not missing cells: both runs ended with visible=0 and resident=0.
`--ann-z-prefetch N` controls the margin (default 32, 0-128); the six decoded
jump windows plus this ring use at most about 9.2 GB here.

c5d is not the right first-stage cache for this view. Its GPU path works in
128-cubed bricks, so an 8-10-slice plane would decode 13-16x the requested z
depth. The current c5d renderer also addresses one 1024-cubed `.c5s` shard at
a time, while PHerc1218 contains 1472 shards; using it here requires a global
multi-shard GPU page table/atlas rather than only a transcode. The benchmark
shows the remaining cache-hit delay is GPU upload and pyramid construction.
Batching those uploads, or adding disjoint GPU-resident z windows, is the next
useful optimization if navigation still needs to become instantaneous.

## 2026-08-06 — x86_64 / RTX 4060 port: upload path + a benchmark-validity bug

First run on a second GPU (RTX 4060 Laptop, NVIDIA 595.84, Ubuntu 26.04,
clang 21, Wayland). Probe: maxImageDimension3D **16384** (vs 2048 on Turnip),
subgroup **32** (vs 128), maxWorkgroupInvocations 1024, host_image_copy yes.

**Benchmark validity first — the old perf.sh numbers were not measuring what
they claimed.** This GPU idles at 210 MHz and settles ~2055 MHz under load. A
cold 300-frame process spends its first frames ramping, and `profile avg`
averages them in. The same scenario, same binary, three consecutive runs:
1.90 / 4.70 / 5.98 ms. Repeatability was accidental, not real: 8 back-to-back
runs agreed to 1.5% purely because they shared one clock state.
perf.sh now does a 400-frame throwaway warmup then measures 1200 frames —
spread drops under 1%. Any pre-2026-08-06 number in this file taken on a
discrete GPU should be assumed inflated by the cold ramp; the Adreno figures
are unaffected (no comparable idle/boost swing).

**Upload: staging is the default now** (was host-image-copy). Interleaved, 4
reps, 1 GiB volume: staging **294 ms** (3.6 GB/s) vs host-image-copy **547 ms**
(1.9 GB/s) — 1.86x. Staging also won on Turnip (251 vs 339), so the old
default was never the faster one; it was chosen to avoid a staging RAM spike,
but the staging path reuses a bounded <=128 MiB buffer, so there is no spike
to avoid. `R3D_STAGING=0` restores host-image-copy.
Verified the choice is upload-only: dropping VK_IMAGE_USAGE_HOST_TRANSFER_BIT
does **not** change sampling throughput (interior 5.97 vs 5.97, exterior 1.21
vs 1.21 ms) — no texture-compression penalty from that usage bit here.
Unaffected: slab/vslab/clip scrolling still requires host image copy outright.

**Workgroup size is not a lever here — keep 16x8.** The 16x8 default was
chosen as one 128-wide Adreno wave; this GPU's subgroup is 32, so the rationale
does not carry, but the outcome does. Interleaved medians of 3 (raycast ms):

| scenario | 8x8  | 16x8* | 16x16 |
|----------|------|-------|-------|
| exterior | 1.22 | 1.21  | 1.23  |
| orbit    | 1.49 | 1.48  | 1.52  |
| zoom     | 4.51 | 4.44  | 4.46  |
| fly      | 3.26 | 3.19  | 3.18  |
| interior | 6.02 | 5.97  | 5.98  |
| mip      | 4.22 | 4.09  | 3.93  |

16x16 wins MIP by 3.9% and loses orbit/exterior slightly; all six share one
cube pipeline so only one choice exists. Not worth churning the default.

Baseline after the upload change (warmed perf.sh, 1080p, 1024^3 gyroid —
note: synthetic, NOT the real scroll region the pre-08-06 rows used, so these
are not comparable to them):
orbit 1.47 · zoom 4.43 · fly 3.19 · orbit lowcut 1.24 · fly lowcut 2.30 ·
exterior 1.21 · interior dense 5.98 · MIP 4.14 · slab zsweep 1-tile 0.36 ·
slab zsweep 2x2 3072^2 (real PHercParis4) 0.53 ms.

Not done, sized but deferred: R3D_SLAB_MAX_TILE is hardcoded 2048 ("target
hardware") and clip/vslab inherit 2046/2044 payload caps from it. This GPU
allows 16384, which would cut a 3072^2 slab from 2x2 tiles to 1 and the whole
43k plane from 22x22 to 3x3. Blocked on two things: the shader hardcodes
2046.0 in the overview-pyramid level math (raycast.slang:222,225), so the cap
must become a push constant; and tile edge cannot simply follow the device
limit — one 16382^2 x 34 tile is 8.6 GB, so the bound is really VRAM, not
maxImageDimension3D. Needs a min(device limit, memory budget) rule.

## 2026-08-06 — gradient shading: one failed idea, one real-but-unadopted 11%

Shaded compositing costs ~30% of frame time on the RTX 4060 (mode 0 vs mode 5,
warmed harness): interior 5.96 -> 4.24, fly 3.19 -> 2.09, exterior 1.21 -> 0.84
ms. That is the 6 central-difference taps per contributing sample.

**FAILED — skip shading on low-contribution samples.** Gate the taps on the
sample's front-to-back weight (1-acc.a)*a instead of just `a > 0.003`, on the
theory that a dense ray's tail cannot move the pixel. At a visually lossless
threshold (0.004) it bought **nothing** (interior 5.96 vs 5.98); at 0.02 it was
*slower* (6.49) AND wrecked the image (mean 71 LSB, 99.8% of subpixels). The
lesson is SIMT, not thresholds: a warp executes the taps if ANY lane needs
them, so a per-sample divergent skip saves no work and adds a branch. Only
whole-warp-coherent skips (like the occupancy gate) pay off. Reverted.

**REAL but NOT adopted — 4-tap tetrahedral gradient, -11%.** Replace the 6
central differences with 4 taps at the tetrahedron corners; cost is uniform so
SIMT divergence is not an issue. Offsets scaled by 1/sqrt(3) to keep tap radius
equal to the central differences (sampling further out over-smooths noisy
scroll data) and renormalised by sqrt(3)/2 so |g| still matches — gm drives
saturate(gm*6.0), so magnitude matters, not just direction.

True A/B (same binary, only raycast_cube.spv swapped, medians of 4):
interior 5.97 -> 5.42 (-9.2%) · exterior 1.22 -> 1.09 (-10.7%) ·
fly 3.19 -> 2.82 (-11.6%) · zoom 4.45 -> 3.95 (-11.2%).

Not adopted because it changes the look on REAL data more than this project
has previously accepted. Synthetic gyroid is nearly free (interior p99.9 = 2
LSB, nothing over 4 LSB) but real PHercParis4 slab: mean 3.5 LSB, 14.4% of
pixels >4 LSB, 8.4% >16 LSB, 0.9% >64 LSB. Structure is visually preserved —
layers, fibres and boundaries read identically, the delta is fine surface
speckle — but prior accepted image diffs here are 0.13-0.16 LSB mean. The two
estimators only agree on smooth fields, and papyrus is not smooth. It is also
not a GPU-specific change: it would alter Adreno output identically. Patch
kept out of tree; flip is one block in the shader's gradient `else` branch.

**Harness note for whoever automates this next.** Two separate measurements in
this session were invalidated by writing `C="--size 1920 1080 ..."` and passing
`$C` unquoted: zsh does not word-split parameter expansions, so render3d
received ONE junk argument, silently ignored it, and rendered at default
size/TF — which reads as a 6x speedup. Scripts that build flag lists in
variables must run under sh/dash, or write the flags literally. Verify A/B
shots differ by md5 before trusting any image diff.

## 2026-08-06 — slab base-tile cap from the device, not a hardcoded 2048

R3D_SLAB_MAX_TILE was 2048 with the comment "maxImageDimension3D on target
hardware". That is an Adreno fact, not a portable one: this GPU reports 16384.
r3d_slab_layout_init_cap() now takes the cap explicitly and the backend derives
it as min(device maxImageDimension3D, edge that fits a 1 GiB per-tile
allocation), growing from the old 2048 so it can never do worse.

The bound is per-ALLOCATION, not total. Slab payload is nx*ny*wz however it is
tiled, so a wider cap costs no memory (it duplicates marginally fewer aprons —
measured 0.1%, not the several percent guessed before checking). What changes
is that the same bytes land in fewer, larger images, and one 4.8 GB image can
fail where 64 small ones succeed.

Overview levels keep a SEPARATE fixed cap (R3D_SLAB_OV_TILE = 2048): the
shader's pyramid math hardcodes the 2046 payload (raycast.slang "ceil(../
2046.0)"), while the base grid already reads slab_px/py and slab_grid from
push constants. Splitting the two caps is what keeps this a pure CPU-side
change — no shader edit, no push-constant growth (the block is ~232 B of the
256 this device allows, and the file still claims a 128 B budget).

Measured on the real 3072^2 PHercParis4 slab: 2x2 grid -> 1x1, one 306 MiB
tile. A/B against the old binary (medians of 3, warmed): raycast 0.53 vs 0.53
ms, full window fill 336.4 vs 336.9 ms, 1-slice scroll 10.2 vs 10.0 ms —
**no measurable speedup**, and output is byte-identical (md5 match at the
default view and at --lod-bias -2 where former tile seams would show). It is
kept as a correctness/portability fix, not a perf win: the payoff is fewer
descriptors and tiles at large grids (the 22x22 whole-plane case drops well
under the 1024-slot pool), and one less wrong hardware assumption.

Note on the c5d coupling: render3d compiles ${R3D_C5D_DIR}/src directly, i.e.
against that checkout's WORKING TREE, not a pinned commit. Edits there change
the host/kernel contract under render3d with no version signal — a 6->7
widening of the entropy subinfo stride (in the worktree while this was
written, since committed as c5d d094007) made test_c5dgpu fail
intermittently, then hang, then decode garbage, all with render3d unchanged.
Isolated by extracting c5d HEAD to a scratch dir (git archive, no mutation of
the checkout) and rebuilding: 4/4 pass, exact 16777189 voxels, proving
render3d was not at fault. If c5dgpu misbehaves, diff the c5d checkout before
suspecting render3d. Resync in the next entry.

## 2026-08-06 — resync vkc5d.c to the c5d GPU decode contract (d094007)

c5d d094007 ("Optimize Vesuvius codec performance and quality") changes the
GPU decode ABI; render3d's own glue in src/vk/vkc5d.c was still on the
6bd75e4-era contract, so full-GPU decode produced garbage while the hybrid
path (CPU entropy) stayed correct. The delta, taken from c5d's own driver
(src/gpu/codec.c) rather than guessed:

- subinfo widens 6 -> 7 uints/substream (adds pair_base). Now SUBW, used for
  the buffer size, the per-brick base, the memcpy and the pay_off patch —
  previously five independent literal 6s, which is how it drifted.
- entropy PC gains pairinfo_stride; PairInfo is now offset+count PER CHUNK,
  so that buffer doubles (NCHUNK*2 uints/brick, was NCHUNK).
- dequant_idct gains a 4th binding (QMap, Q2.6 per-chunk scales) and four PC
  fields: qweights (packed Q2.6 z|y|x, 0 = isotropic), pairinfo_base, qmap,
  qmap_base. Missing qweights was what silently mis-dequantised everything:
  the stride fix alone made the decode structurally valid but numerically
  wrong (16.4M voxels off by >1).
- qmap is written for EVERY brick (default 64 = 1.0) before the per-brick
  fill, so a previous brick's map cannot leak into one that carries none.

Result: full-GPU 5/5 exact 16777185 voxels, 31 at 1 LSB, 0 worse — identical
to hybrid, and deterministic where the mismatched build hung or varied run to
run. ctest --preset dev is 4/4 again (shard skipped: no local band data).

Worth noting for the next time this drifts: the failure signature moved
through three stages as the mismatch narrowed — hang (rANS looping on garbage
counts), then "truncated substream" (structurally invalid), then clean decode
with wrong values (valid but mis-dequantised). Only the last is quiet; treat
a c5dgpu hang as an ABI mismatch, not a driver bug.

## 2026-08-06 — performance-plan implementation and warmed final matrix

The full implementation record, percentile matrix, portability decisions,
quality A/B, validation scope, and prioritized residual work are in
`docs/performance-review-20260806.md`. This entry supersedes the historical
notes above which say that c5d uses an unpinned sibling working tree, streaming
requires host image copy, four-tap shading was not available, or one-shot work
uses queue-wide idle.

Headline results on the RTX 4060 at 1080p (400 warm-up + 1,200 measured):
raycast mean/p99 is orbit 1.455/1.846 ms, zoom 4.464/7.540, fly
3.295/7.252, exterior 1.198/1.400, dense interior 6.147/6.245, MIP
4.033/4.832, and real 3072^2 slab 0.531/0.534. Default image quality is
unchanged. The opt-in four-tap `fast` policy reduces the dense mean from 6.141
to 5.539 ms (-9.8%).

The principal latency result is c5d streaming: on a 384^3, 27-brick fixture
through a 2^3 hot atlas, GPU decode jobs average 89.43 ms while render-loop CPU
frames stay at or below 4.42 ms. Page-table publication is timeline-drained and
image access is explicitly synchronized; the decode no longer blocks the
render thread.

## 2026-08-07 — Snapdragon X Elite / Adreno X1-85 local tuning

Requalified the RTX-focused 2026-08-06 changes on the development laptop
(12-core Qualcomm Oryon, Adreno X1-85, Turnip 26.0.3). Release CPU, shard,
GPU-reference, and c5d-GPU conformance tests all pass. Bulk staging remains the
right upload default here: cached 1024^3 startup was 0.80-0.90 s versus 1.00 s
with `R3D_STAGING=0` host image copy.

The global workgroup sweep exposed a view-dependent split. At 1080p full
quality, 8x8 is poor for the dense static view (22.75 ms versus 19.05 ms for
16x8), but markedly better on the divergent orbit path. Under the default
half-resolution interaction policy, two 250-warmup/750-measured passes gave
raycast means of 2.303/2.307 ms for 8x8 versus 2.839/2.835 ms for 16x8: an
18.7% reduction. The backend therefore keeps both pipelines only on the exact
X1-85 device and selects 8x8 when the render viewport is reduced; full
resolution remains 16x8. `R3D_WG` still forces a fixed variant and disables
the automatic choice.

The portable release preset compiled generic AArch64 host code. A separate
`native` preset adds `-mcpu=native` without making release binaries
machine-specific. Across three cached 1024^3 starts, occupancy construction
and upload averaged 122 ms with portable release and 95 ms native (-22%);
steady-state GPU time was unchanged, as expected.

## 2026-08-07 — PHerc1218 global Zarr/c5d LOD pyramid

The 23552x8192x8192 mirror is now processed directly from its 1472 dct3d Zarr
shards. `lodpack` combines eight decoded parent 16³ chunks into each 2x-reduced
child chunk, writes standard Zarr v3 `sharding_indexed` objects, decodes each
new chunk closed-loop, and assembles c5d 128³ bricks. The level shapes are
23552x8192x8192, 11776x4096x4096, 5888x2048x2048, 2944x1024x1024,
1472x512x512, 736x256x256, 368x128x128, and 184x64x64. The resulting shard
counts are 1472, 192, 24, 3, 2, 1, 1, and 1.

The first implementation used c5d's target-ratio search. Those streams passed
CPU decode and a full 1.07-billion-voxel Zarr comparison but exposed a sparse
substream failure in the current GPU entropy kernel. Production therefore
uses the already-conformant fixed-quality path with an inverted q ladder
(2, 1, 0.5, then 0.25). This is an important distinction: valid CPU c5d is
not sufficient evidence for the renderer; the GPU conformance path is part of
the format gate.

`lodcheck` compared all 512 bricks in representative production shards against
decoded Zarr (1,073,741,824 voxels each): L0 measured MAE 0.721, PSNR 44.35 dB,
max error 57; L1 measured MAE 1.814, PSNR 40.17 dB, max error 45. Container
CRCs, zero sentinels, and global brick/chunk coordinates were all exercised.
All generated levels have matching Zarr/c5d shard counts and no partial output
files. The production path also passes `test_c5dgpu` in native and sanitizer
builds.

The renderer manifest creates one logical page table over every true brick at
all eight levels (861,341 entries for PHerc1218). The complete coarsest level
is decoded synchronously and pinned as the no-hole fallback. The CPU streamer
chooses a desired level from `distance * pixel_cone * base_max_dimension`,
while the ray shader repeats the choice at sample depth. Lookup is bounded to
the desired level, two immediate parents, and the pinned fallback. Atlas slots
can therefore contain bricks from any level, and adjacent regions compose even
when their requests complete on different frames without scanning unrelated
finer levels.

The original whole-depth MIP path exposed two correctness and scheduling
problems: it produced visible nested brick rectangles while a 5^3 atlas
thrashed, and asynchronous c5d compute shared the Adreno X1-85's only graphics
and compute queue with raycasting. A close 640x480 view measured about 149 ms
GPU. Manifests now default to the actual annotation workload: a shallow
8-slice XY slab, with z-range-pruned requests and a 8^3 hot atlas. Manifest
c5d is decoded on four CPU threads into a persistently mapped staging batch;
the GPU only performs the final atlas copy. Disk Zarr/c5d levels are the only
LOD hierarchy, so the atlas no longer allocates or filters a second per-slot
mip chain.

On the same machine, a settled 960x720 midpoint slice measured 1.59 ms GPU,
with 12 streamed bricks and no failures; the captured image has no brick-grid
discontinuities. A 600-frame 640x480 no-vsync `zoomio` sweep averaged 0.81 ms
GPU / 0.88 ms CPU, exercised four requested LODs, and completed with zero
failures. A 600-frame full-z `zsweep` averaged 0.90 ms GPU, decoded 144 bricks
in 28 batches, and also recorded zero failures. Benchmark JSON records the
desired-level set and cumulative per-level request counts so the LOD walk is
machine-checkable.

## 2026-08-10 — vc3d-style 2x2 multi-view on AWS open-data PHerc0172

Data: the `vesuvius-challenge-open-data` S3 bucket is the one place where
tifxyz segments and their exact source volume coexist — PHerc0172 has 53
segments traced on volume 20241024131838 (21000x6700x9100, zarr v2 u8,
blosc-zstd, 128³ chunks == c5d bricks, levels 0-5 prebuilt). `zarr2c5d`
transcodes a local chunk mirror into the `--bricks` c5d LOD tree without any
zarr output; selection is CHUNK-granular around a segment surface (a first
shard-granular cut selected 254 L0 shards ≈ 254 GB — chunk granularity cut
that to 26,977 L0 chunks; the w062 fetch totalled 33,167 chunks / 38 GB
mirrored, transcoded to an 8.9 GB c5d tree in ~10 min at 12 threads).
Absent-on-S3 (masked air, zero-fill) is distinguished from not-yet-fetched by
`.missing` markers written by the 404s in `tools/fetch_chunks.sh`. `--verify`
PSNR: 54.05 dB on L3-L5 (box-filtered), 39.32 dB on L0 (q2 on raw scan
noise) — consistent with the codec's known quality at those rates.

Renderer: `r3d_frame_views` records up to four dispatches into `view_org`
rects of the one offscreen image; per-frame state stays in the dynamic-offset
UBO ring (now FRAMES_IN_FLIGHT x R3D_MAX_VIEWS slots) — no push constants
were added and no descriptor duplication. Orthographic rays are a FrameParams
flag (origin offset across the image plane, constant per-pixel LOD footprint)
and the bricks slab clip generalized from z-only to any axis, so the XY/XZ/YZ
quadrants are plain ortho MIP slice views over the same bricks cache. The
streaming pump split into begin/collect/submit: each plane view submits an
auto-coarsening AABB collect at its own px/voxel magnification, and the
flattened view requests bricks point-wise along the visible decimated grid.
4-view GPU cost at 1280x720: raycast ~2.0 ms avg (single-view XY slab was
~1.0 ms) — well inside 60 fps.

Flattened segment view (R3D_MODE=5): per pixel, a manual bilinear tap of the
RGBA32F coords grid (any invalid corner poisons the quad — vc3d semantics),
per-pixel normal offset (Shift+wheel zoff), then one bricks-LOD virtual-volume
sample. Cross-view sync follows vc3d: Ctrl+click sets the shared focus POI;
plane views recenter and re-slice through it; the segment view recenters via
nearest-surface search (100-voxel tolerance). Intersection overlays are one
marching-squares pass per plane over the tifxyz grid (`core/segtrace`,
unit-tested), doubly parameterized so the same segments draw the surface
curve on plane views (+ translucent zoff shell) and the plane trace lines on
the flattened view (vc3d colors); w062 at mid-scroll yields 704/985/1244
segments for XY/XZ/YZ, recomputed only on slice/zoff change.

Follow-ups: perspective 3D quadrant option, vc3d segment-aligned plane
orientation + rotation handles, per-view GPU timestamps, ink-detection
overlay textures, multi-segment display.

## 2026-08-10 — flattened surface volume (vc3d composite, volumetric)

The segment quadrant no longer marches bricks per pixel: a `surfvol.comp`
kernel resamples the shared bricks cache into an R8 3D **flattened surface
volume** — a 1024x1024x96 window over (u, v, layer) where texel (x,y,l)
samples `surface(u,v) + normal(u,v) * (zoff0 + (l - 48))`. This is vc3d's
"composite" layer stack materialized as a texture instead of collapsed to
2D, so the view renders it with the real volumetric raycaster (trilinear
across layers, gradients in flattened space). Layers stay at native 1-voxel
pitch regardless of xy zoom — an early version scaled layer pitch with the
xy LOD and produced sub-sample speckle once the marched thickness dropped
below one layer. xy pitch follows the view (power-of-two voxels/texel), so
the 96 MB window covers a screenful at full resolution or the whole segment
zoomed out. Rebuilds are one 100M-texel dispatch triggered by hysteresis
(window origin snapped to W/8, zoff0 to 24 voxels) or by brick-residency
arrivals (cooldown 20 frames); render-time zoff slides within the baked
±48-voxel range for free. Depth knobs match vc3d composite: N layers
in front/behind at bake, marched thickness at render.

## 2026-08-10 — tifxyz-transform: volume-to-volume segment remapping

PHercParis4's 2.4um volume (75784x32693x32693, uncompressed zarr v2 chunks)
ships a transform.json: schema 1.0.0, a single 3x4 XYZ affine meaning
p_fixed = M @ p_moving in per-volume voxel units (the landmarks are
provenance, not an applied warp — volume-cartographer has no non-affine
transform type). fixed_volume here is the CANONICAL 7.91um scan, so mapping
canonical-traced segments onto the 2.4um volume applies the INVERSE
(landmark check: 1.9 vox RMS in fixed space). `tifxyz-transform`
reimplements vc_transform_geom: p' = s_after * (M @ (s_before * p)),
invalids pass through, grid resampled by the measured median adjacent-point
spacing ratio (interior 10-90%, step 4), meta scale preserved, bbox
recomputed. Validation is exact: our output of the GP banner (7.91um ->
2.4um) matches the bucket's published -on-2.4um.tifxyz on the same
1820x2530 grid at 0.002 vox mean / 0.016 max error — pure f32 rounding.
(A first comparison suggested a ~60-voxel "refinement" residual; that was a
half-grid-cell misregistration in the comparison itself, worth remembering:
1 grid cell = 1/scale = 20 voxels, so tiny index-convention errors read as
huge coordinate errors.) Any canonical Scroll 1 segment can now be carried
onto the 2.4um (or 1.129um) volumes locally.

## 2026-08-10 — PHercParis4 2.4um GP banner + streaming ingest

The 2023 Grand Prize banner (20230702185753, 1820x2530 grid, 4.03M valid
points) now renders in the 2x2 viewer on the 2.4um volume (75784x32693x32693,
6 zarr levels, uncompressed chunks). Tiered ingest: L1-L4 follow the whole
surface (pad 32), L5 full (pinned), L0 confined to a central 400x500-grid
--rect (~21 GB); 36,158 chunks / 72 GB total, 12 GB c5d tree, verify 51 dB.
Two systemic fixes fell out: (1) the pinned coarsest level here is 1216
bricks, which silently filled the whole 8^3 atlas pool — every slot
permanently pinned, zero decodes, black fine levels; the pool now auto-grows
(cap 12^3) to hold the coarsest level plus streaming headroom. (2) a
chained "wait for fetch then transcode" shell loop deadlocked on
`pgrep -f` matching its own command line — the pkill self-match rule
applies to pgrep too.

`zarr2c5d --url` is now a single-pass streaming ingest: worker threads
fetch chunks over HTTP straight into memory (per-thread libcurl handles,
connection reuse, 4-attempt backoff, 404 = fill) and only transcoded c5d
shards are written — no raw mirror. The mirror dir holds just the per-level
.zarray metadata. Validated byte-identical to the mirror-based path on L4
(12/12 shards, 198 chunks fetched, 0.4 GB). Mirror mode remains for
pre-fetched data and re-runs.

## 2026-08-10 — on-demand raw-volume streaming (fetch -> transcode -> cache)

`--bricks` volumes no longer require a full offline transcode. A bootstrap
run (`zarr2c5d --url U --full-from 5`, ~1 min: coarsest level + manifest +
source.json) is enough; the renderer then streams everything else on demand.
On a brick miss the pick loop enqueues the owning RAW zarr chunk
(nearest-first, deduped); a 4-thread fetch pool downloads it (retry/backoff,
404 = air), transcodes its (chsz/128)^3 bricks with the same quality ladder
as the offline tool, and writes per-brick c5d blobs to
<root>/bricks/L<l>/<z>_<y>_<x>.c5b (empty file = definitively absent, so a
brick is never re-requested; each chunk downloads exactly once, ever).
bricks_source_blob gained that cache as a tier between the shard tree and
"absent"; pending bricks stay candidates instead of being poisoned with
maxk=0. Verified on PHerc0172: bootstrap 33 chunks, then a 900-frame
headless run fetched+cached 83 bricks live from S3 and rendered them (0
failures); the second run served the same view from disk (10 fast decode
jobs, no fetches). Offline zarr2c5d remains the bulk pre-warm path and both
write layouts the renderer reads interchangeably.

## 2026-08-10 — main-thread profiling: phase timers + two hotspot fixes

Added a per-frame main-thread phase profiler (poll / nav / gui / stream /
frame timers; EMA in the profile GUI section, sum/max printed at exit as
`mainthread avg: ...`) and ran perf (dwarf call graphs, steady-state tail
sliced with `perf report --time 60%-100%`) on the static GP-banner multiview
with ink overlay. Two real hotspots surfaced, both fixed:

1. `warm_evict_one` scanned ALL virtual bricks per eviction — 44.4M for
   PHercParis4 — inside `warm_get`'s retry loop; ~10% of the render thread
   at steady state and the source of `stream max 84 ms` spikes. Now the warm
   tier keeps a compact list of resident brick ids (thousands) and the LRU
   scan walks only that.
2. Intersection overlays drew one anti-aliased AddLine per marching-squares
   cell (~6000/frame): AddPolyline tessellation ~11% + ImGui vertex upload
   memcpy ~15% of the main thread. mv_draw_lines now chains head-to-tail
   segments into single AddPolyline calls, collapses sub-pixel steps, and
   drops chains whose screen bbox misses the pane.

Same 900-frame static scenario, before -> after: gui 2.93 -> 1.52 ms, stream
0.43 -> 0.09 ms (max 84 -> 10 ms); ~58 fps with frame-phase time now vsync
wait. Remaining main-thread steady-state cost is AddPolyline 5.3% +
mv_draw_lines 3.3% (projection) + ~1.4% driver/allocator munmap+fault noise
— all sub-0.5 ms/frame, left alone. Automation note: `--frames N --warmup M
--shot F` headless runs are the repeatable scenarios; perf attach to a
running instance failed (rc=255, environment), record-with-`--delay` + time
slicing works. All 5 test suites green.

## 2026-08-10 — hitch hunting round 2: progressive surfvol re-bake + segtrace tiles

Two frame-spike sources removed on the static GP multiview scenario:

1. Residency-arrival surfvol re-bakes dispatched the full 2048^2x96 window in
   one frame (~75 ms GPU hitch, recurring while bricks stream in). Arrival
   re-bakes keep the window mapping unchanged, so rewriting any texel subset
   in place is exactly correct: they now bake progressively, SV_PROG_ROWS=128
   rows per frame (~5 ms GPU each, 16 frames to converge — under the 20-frame
   arrival cooldown). Window moves (pan/zoom/zoff snaps) keep the immediate
   full rebuild, since a partial rebuild there would mix two mappings.
2. Slice-change segtrace marched the full 1820x2530 grid per plane (~25 ms
   each; gui max 73 ms with the zoff shell). r3d_segrows now caches per-row
   and per-16x16-tile coordinate bounds per axis (built once at segment load,
   ~1 MB); traces skip rows and hop whole tiles that cannot straddle the
   slice, with a |zoff| margin for the shell since normals are unit length.
   GP mid-volume slices: 27 -> 4.5 / 1.7 / 0.3 ms (x / y / z axis, 6-103x),
   byte-identical output (unit-tested).

Same 900-frame scenario: gui max 73 -> 5.4 ms, cpu p99 58 -> ~27 ms, cpu max
90 -> ~50 ms (remaining spikes are the one-time initial full bake + startup),
gpu max 7.1 -> 6.8 ms steady. All 5 suites green; screenshot pixel-identical.

## 2026-08-10 — surfvol window moves: shift-in-place instead of full re-bake

Pan snaps (W/8 = 256 texels) and zoff scrubs (24-layer snaps) in the segment
view each cost the full ~75 ms window bake. The window origin only ever moves
in integer texel multiples, so r3d_surfvol_window now detects same-pitch
integer moves and shifts the surviving ~85% of the 3D window in place —
same-image vkCmdCopyImage strips along the moving axis (strip length =
|shift| keeps each copy's src/dst disjoint; strips ordered so every strip is
read before a later copy overwrites it, transfer barriers between) — then
bakes only the exposed bands (~10-20 ms total vs 75). uv moves and zoff
moves shift independently; combined moves and zoom (pitch changes) still
full-bake.

Verified in-run: shift one snap mid-run, then force a full progressive
re-bake with identical params and diff the frames before/after — u-only,
v-only, z-only, u+v, all+ink each match the no-shift control exactly
(seg-view residual mean 0.70 gray levels, identical across all cases; it is
the natural refresh of bricks that arrived after the interior's last bake).
A scary first diff turned out to be a stale surfvol.spv from mid-build
testing, not the copies. Steady scenario unchanged (gui max 5.4 ms, stream
max 11 ms); all 5 suites green.

## 2026-08-10 — startup: brick-parallel CPU decode + decoded-seed disk cache

Startup on GP multiview + ink was 39.4 s wall, dominated by decoding the same
pinned-coarsest bricks every launch (555 CT + 555 ink dense bricks, ~95 ms
CPU each — the coarsest level is the whole scroll, so its bricks carry near
max entropy). Tried GPU c5d decode for the seed first: 25 ms/brick on the
Adreno (40 bricks/s vs the 228 in the ledger for typical bricks) — WORSE
than CPU for these dense bricks; reverted. Two things that did work:

1. Brick-parallel CPU decode: bricks_decode_batch (and the overlay backfill)
   now run one single-threaded c5d_brick_decode per brick across all cores
   (brdec_run: atomic cursor, caller participates) instead of sequential
   bricks x 4-lane inner parallel_for. Steady streaming jobs 328 -> 133
   ms/batch; also the decode target moved from the write-combined staging
   buffer to a heap buffer (decode passes re-read dst; WC reads are ~100
   MB/s) with a memcpy into staging before upload — overlay backfill
   3.6 -> 1.9 s from that alone.
2. Decoded-seed cache: first launch writes the decoded slabs + maxes to
   <root>/seed.raw (~1.1 GB per tree, header/table/slabs, tmp+rename,
   guarded by manifest size+mtime); later launches stream it into the atlas.
   Seed 7.8 s -> 185 ms, ink backfill 1.9 s -> 302 ms.

Launch-to-render: cold (first ever) 13.5 s, warm 4.6 s — 8.5x vs the 39.4 s
baseline. Verified: screenshot identical, truncated-file and stale-manifest
fallbacks re-decode and rewrite the cache, all 5 suites green.

## 2026-08-10 — surfvol full rebuilds: visible-box-first + progressive rest

The last full-window bake trigger was a zoom's pitch change (once per octave,
~75 ms), plus the initial bake. r3d_surfvol_visible now hints the window
sub-box the view can actually see (main.c: view rect in window texels, zoff
+- mv_thick in layers, +64-texel/+8-layer margin); a full rebuild bakes just
that box in-frame (typically 3-15% of 2048^2x96 -> a few ms; gpu max 13.5 ms
including the initial bake, was ~75) and restarts the progressive pass to
refresh the remainder at 128 rows/frame. Content outside the box keeps the
old mapping for the ~quarter second the pass takes — only visible if the
user pans immediately after zooming. Window shifts now restart (rather than
cancel) an in-flight progressive pass so stale regions can't survive a
zoom-then-pan. Static 900-frame scenario, best numbers yet: cpu max 37 ms,
frame max 34 ms, no main-thread phase max above 33 ms; converged frame
bit-identical to the previous build. All 5 suites green.

## 2026-08-10 — ImGui Vulkan backend: persistently mapped vertex buffers

The steady-tail profile showed ~20% of main-thread samples in page-fault
handling under ImGui_ImplVulkan_RenderDrawData's upload memcpy: the stock
backend vkMapMemory/vkUnmapMemory's its vertex+index buffers every frame,
and on msm/Turnip the unmap drops the pages — every upload byte re-faults
through msm_gem_fault each frame. Patched the vendored backend
(tools/cimgui/imgui/backends/imgui_impl_vulkan.cpp, [render3d] comments) to
map each frame-render buffer once at (re)creation and keep it mapped for the
buffer's lifetime (flush stays; vkFreeMemory implicitly unmaps). Command
recording: 1.73 -> 0.45 ms/frame; RenderDrawData fell from 33% to 3.6% of
steady main-thread samples. Remaining active main-thread work is AddPolyline
tessellation + mv_draw_lines projection (~1.4 ms gui phase in a 16.7 ms
vsync frame) — left alone. All 5 suites green.

## 2026-08-10 — memory budgets derive from the device, not constants

Component sizes now scale to the reported Vulkan memory budget instead of
hard-coded values, for UMA (this 32 GB X1E: one 23 GiB shared heap, live
budget ~15.8 GiB via VK_EXT_memory_budget) and discrete 8/16 GB targets:

- Warm tier default: budget/8 clamped to [256 MB, 3 GiB] (u32-offset cap);
  explicit --warm still wins. This machine: 256 MB -> ~2.6 GB.
- Atlas ceiling: was a hard 12^3; now min(maxImageDimension3D/128, largest
  cube under maxMemoryAllocationSize, budget-derived share assuming a second
  overlay atlas + warm + surfvol + 2 GiB slack). Same 12^3 = 3.6 GiB here
  (the ~4 GiB per-allocation limit binds); a simulated 8 GiB card
  (--gpu-mem 6200) degrades to 9^3 + 774 MB warm and renders with 0
  failures instead of over-committing; 16 GiB class keeps 12^3 + 1.7 GB.
- --surfvol W/H/L clamp to maxImageDimension3D (new r3d_max_dim3d query).

GPU total here: ~10 GiB of the ~15 GiB budget (2x 3.6 GiB atlases + 2.6 GB
warm + 768 MB surfvol). CPU side stays implicit (per-brick bookkeeping ~1.4
GB virtual; page cache covers seed.raw + shard mmaps). 900-frame scenario
unchanged (frame max ~40 ms); all 5 suites green.

## 2026-08-11 — segment-aligned plane views (vc3d seg-xz/seg-yz)

The XZ/YZ panes can now reorient to the segment's local frame instead of the
world axes ("segment-aligned planes" checkbox + rotation slider in the
multiview panel). A plane view was generalized from two world axes to an
orthonormal frame {u, v, n} + origin (mview.h r3d_mv_seg_frames et al.):
both aligned panes contain the surface normal at the focus (screen up = +n,
sheet edge-on and horizontal), horizontals are the grid tangent rotated
theta and theta+90 around n, so the pair stays perpendicular and scrubbing
one pane slides along the other's horizontal. Ctrl+click re-anchors the
frames at the new focus (nearest surface point supplies the normal); frames
fall back to world axes when no valid normal is within vc3d's 100-voxel
tolerance.

Decisions/mechanics:
- Shader: R3D_VIEW_OBLIQUE clips the marched t-range along the (ortho) ray
  instead of an axis-aligned box — equivalent for axis frames, correct for
  any orientation; multiview now uses it for all three plane panes. The
  axis-box clip path stays for the perspective bricks slice mode.
- r3d_segtrace_basis: marching-squares overlays for arbitrary planes; the
  row/tile skip bounds dot(p, n) from the per-axis boxes (sign-aware sum,
  zero components skipped so empty +-inf boxes can't NaN). Axis form is now
  a wrapper; unit tests cover shifted-origin equivalence and an oblique
  45-degree normal against the synthetic surface.
- Streaming: per-pane want-AABB from |basis| component extents around the
  slab midpoint (superset of the visible oblique rect).

Verified on the GP banner: aligned panes render sheets horizontal at the
focus (screenshots), gpu ~4.2 ms vs ~5.0 axis-aligned at 1080p (fewer
sheet-parallel empty rays), 0 decode failures, all 5 suites green (gpu
conformance covers the shader change).

## 2026-08-11 — segment store: c5d-compressed corpus + tile spatial index

Groundwork for multi-segment viewing (all of a scroll's surfaces at once,
bigger than RAM). c5d pin bumped d094007 -> 84274f8, which brings the c5d
tifxyz surface codec (masked parallelogram predictor + static rANS; lossless
bit-exact f32 ~2.75x, quantized 2^-q voxel steps up to ~12x) and a turnip
encode_tokenize miscompile fix; gpu conformance suite validates the new rev.

New src/core/segstore.{h,c} + tools/segpack: a store directory holds one
.tfx per segment plus segments.r3ds — a binary manifest that doubles as the
spatial index (per-segment world AABB, then a u16-quantized AABB per 16x16
grid tile, quantized over the segment bbox; floor/ceil so dequantized boxes
only grow). Two-level queries (segment bbox -> tile scan with the sign-aware
dot-range straddle test from segtrace) answer "which segments cross this
plane / sit near this POI" without touching compressed data — the vc3d
SurfacePatchIndex idea (R-tree over grid tiles) flattened into rebuild-from-
arrays C. r3d_segstore_load decodes with an optional power-of-two stride
(decimated grids for overview polylines; scale rescales so segtrace works
unchanged).

Measured (4 real PHercParis4 segments: gp 1820x2530 + three 2.4um od
downloads up to 9196x3744): 988 MB tifxyz -> 59 MB store at q=2 (1/4-voxel
max error; ~17x incl. uncompressed sources), pack 7.8 s. 328k index tiles =
3.9 MB manifest. Plane query 331 us; full gp decode 237 ms, stride-4 201 ms
(decode-bound). Lossless mode round-trips bit-exact vs r3d_tifxyz_load
(unit-tested, incl. oblique-plane and near queries + decimated segtrace).

## 2026-08-11 — multiview draws the whole segment corpus (--segments)

The 2x2 viewer now overlays every store surface crossing a plane view as a
dimmed polyline under the active segment's curve (works in axis and
segment-aligned modes — the query normal is the pane's frame normal).
Mechanics: per pane, a (slice, basis-generation) key gates one
r3d_segstore_plane_query (~0.3 ms on the 4-segment/328k-tile store); hits
whose decimated grid isn't cached are queued to a background worker
(pthread, decode stride 4 + segrows build off-thread — the frame loop never
decodes); at most 2 re-traces run per frame to amortize slice scrubs, and
traced polylines persist per (pane, segment) so scrubbing back is free.
Cache is LRU-evicted against a 768 MB budget; the active segment is never
decoded (it already renders full-res). Panel: cache state, per-plane hit
counts, nearest-to-focus surfaces (near_query on focus change).

Measured (gp active + three 2.4um od segments, 400 frames): all 4 surfaces
hit each pane, 57 MB cached after the initial burst, gui phase 2.2 ms avg /
13.7 max (trace catch-up; was 24.6 at 4 traces/frame, budget lowered to 2),
frame max 32.7 ms, all 5 suites green. Exit line reports cache/hit/trace
stats for headless verification (ImGui overlays don't land in --shot
captures, which read the offscreen image).

## 2026-08-11 — activate any store surface from the panel (in-place swap)

Clicking a surface in the multiview panel (near-focus list or "all
surfaces") makes it the active flattened segment without restarting the
app. The corpus worker decodes the full-res grid and builds segrows +
coords/normals grids off-thread; the GUI thread then calls the new
r3d_surf_swap (device-idle, recreate + upload the two RGBA32F grid images,
rewrite bindings 7/8, rebind the surfvol taps and reset its window state so
the next _window call fully rebakes at the new segment's scale — the 768 MB
window texture itself survives). The viewer swaps mv_seg/rows/normals,
re-anchors the focus and aligned frames on the new segment's center, and
recenters the planes; the old active segment becomes a dimmed overlay.

Measured (gp -> 8508x2248 od segment, 19M points): apply hitch 405 ms in
one frame (vkDeviceWaitIdle + 2x306 MB grid uploads; user-triggered and
rare — acceptable, noted for a staged-upload follow-up), decode+grids fully
off-thread, gui-phase catch-up max 60.8 ms while re-traces amortize, steady
state back to ~2 ms. Verified headless via a temporary frame-150 activation
hook (removed): swap log line + screenshot show the new banner in the
flattened pane with ink overlay and recentered planes. All 5 suites green.

## 2026-08-11 — segstore: incremental packs + persistent stride-4 tier

r3d_segstore_build is now incremental: segments already in the store whose
source dirs aren't given are kept by copying their previous manifest entry
(or rebuilt from the .tfx itself, scale parsed from the carried meta.json)
— so packed tifxyz sources can be deleted, bounding fetch-all disk use to
one segment at a time. Each pack also writes <name>.tfx4, the same grid
pre-decimated by 4 (exact source points, so tier decode == full decode
subsampled — unit-tested); r3d_segstore_load with stride divisible by 4
decodes the 16x-smaller tier, cutting overview decodes from decode-bound
~200-800 ms to tens of ms per segment. tools/fetch_segments.sh streams a
whole scroll: list bucket -> curl the 4 files -> segpack -> delete source,
resumable (skips existing .tfx).

## 2026-08-11 — whole-scroll corpus ingested + stretch heatmap + overlap QC

Fetch-all completed: all 81 PHercParis4 segments (the -on-20260411134726-
2.4um variants; dir names key by mesh id, fixed in fetch_segments.sh)
packed into cache/PHercParis4-segstore — 82 surfaces incl. gp, 1.3 GB
total (~25 GB of source tifxyz, deleted after packing), 6.65M index tiles
(76 MB manifest), 0 failures. Viewer at full scale: 82 surfaces hit each
pane, worker cache settles at ~47 ready / 666 MB (only drawn surfaces
decode), frame 9.7 ms avg / 34.6 max.

New QC features (user-picked from the vc3d gap list):
- Stretch heatmap (R3D_VIEW_STRETCH, "stretch heatmap" checkbox or
  R3D_MV_STRETCH=1): surf view colors each pixel by local flattening
  distortion — coords-grid derivative length per cell vs the ideal 1/scale
  voxels, log ratio at +-26% full scale, warm = stretched / cool =
  compressed, luminance modulated by the CT. GP banner shows compression
  concentrated along fold seams. vol_tx/ty carry the grid scale into surf
  views (they ignore the model transform).
- Overlap QC: r3d_segstore_overlap = fraction of one surface's index tiles
  within ~8 voxels of another's (coarse 64^3 bitmap over the shared bbox,
  index-only); the corpus worker recomputes overlap-vs-active on every
  activation. Plane views tint overlapping polylines orange (>15%) / sand
  (>2%); near-focus list shows ov %. Unit-tested (identical twin ~1.0,
  +9000-voxel translate 0.0).
- Draw scaling: pane hits sort nearest-first (bbox center vs pane center)
  and at most 48 draw; corpus polylines collapse at 3 px (vs 1 px for the
  active segment). gui phase 8.5 -> 6.7 ms avg at full corpus; projected-
  polyline caching noted as the next lever. All 5 suites green.

## 2026-08-11 — net-ingest queue: newest view first

Chunk fetch requests used to append to the queue, so after a pan/zoom the
previous viewpoint's backlog (up to 256 chunks) downloaded before anything
under the cursor. Requests now insert at a cursor that resets to the queue
head each pump pass: the current view's chunks fetch first (keeping the
pass's own nearest-first order), re-requested backlog entries promote into
the head block, the stalest tail entry drops when the queue is full, and
in-flight transfers are never cancelled. Also: tiered ingests want a
source.json next to manifest.json (copied from the od bootstrap tree for
PHercParis4-lod) — without it, regions outside the tiered shards read as
permanently absent instead of demand-fetching (256 MB of bricks pulled
interactively in the first session with it enabled).

## 2026-08-11 — solo pane rendered through the adaptive-res blit (fix)

Space-solo made a multiview pane the frame's ONLY view, and r3d_frame_views
treated every nviews==1 submission as the classic single-view path: blit
offscreen (0,0)-(viewport) stretched to the whole window. A solo pane
renders at view_org (panel_w, 0) with the panel-shrunk width, so the blit
sampled the wrong rect and upscaled ~1493->1853 px — the CT image sheared
right/stretched under the (correct) ImGui overlays, converging toward the
cursor as you zoomed. The adaptive path is now gated on view_org == 0
(true for real single-view frames, never for panes beside the panel);
anything else renders in place and blits 1:1 with the multiview clear.
Also: the segment view's plane trace lines (orange/red/yellow) now
recompute and draw even when the plane panes are collapsed or solo'd —
they were gated on pane visibility, so SEG solo lost them. 2x2 regression
run unchanged (0 failures), all 5 suites green.

## 2026-08-11 — umbilicus annotation in the multiview panes

--umbilicus <json> now composes with --multiview instead of forcing the
PHerc1218 vslab rig. With editing on (panel checkbox, default on), a plain
click in ANY plane pane places the control point for that click's
(rounded) z — the core can be ambiguous in one orientation and obvious in
another — and Shift+drag pans meanwhile; Ctrl+click stays the focus
gesture. The curve draws in all three panes (magenta control points +
connecting polyline, projected through each pane's frame so it works in
segment-aligned mode too) with a crosshair at the XY pane's current-z
interpolation (new umb_interp, clamped linear). Panel: point count, edit
toggle, prev/next-annotated jumps (recenter XY on the point), delete-here,
autosave on every set (same Villa/VC-compatible JSON writer as vslab
mode). The vslab annotation workflow is untouched.

## 2026-08-11 — umbilicus annotation UX round (user-driven)

Interaction iterations after using the mode for real: U places at the
cursor instead of plain click (clicks keep pan/zoom; the vslab z-nav block
was also eating non-shift wheel under --umbilicus and is now vslab-gated);
Ctrl+Z / Ctrl+Shift+Z snapshot undo/redo (64 deep, new edits clear redo);
start-fresh button renames the JSON to .bak and clears; overlays draw only
where the curve intersects each pane's [slice, slice+thickness] slab
(parametric segment clip; the XY crosshair no longer extends past the
curve's z-range); scrub-speed slider (0.25-200 vox/notch, log) for
Shift+wheel and R/F; placing a point optionally refocuses all panes like
Ctrl+click (toggle, default on); panel section is a collapsing header.
