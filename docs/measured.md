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
