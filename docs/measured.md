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
