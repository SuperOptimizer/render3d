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
