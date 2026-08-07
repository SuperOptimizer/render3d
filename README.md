# render3d

High-performance volumetric renderer for Vesuvius Challenge micro-CT volumes.
Pure C23 (clang/LLVM), SDL3 + Vulkan compute raycasting (GL 4.6 backend planned),
Slang shaders → SPIR-V. Consumer of [c5d](https://github.com/SuperOptimizer/c5d) and its
16³ chunk / 128³ brick / 1024³ shard hierarchy.

Milestone 1 status: 1024³ real scroll region (PHercParis4) at **225 fps /
1920×1080** (full gradient shading, ray-cone LOD, release build, Adreno X1-85).

- `spec/volume.md` — data conventions
- `docs/measured.md` — every design decision with the numbers behind it
- `docs/performance-review-20260806.md` — implemented performance plan, benchmark matrix,
  portability work, and prioritized residual bottlenecks

## Quick start

Build dependencies are CMake 3.28+, Ninja, clang, SDL3, Vulkan headers/tools,
libcurl, and pthreads. CMake fetches c5d at an audited commit; Slang remains a
one-time pinned tool download.

```sh
tools/fetch_slang.sh                    # one-time: vendor pinned slangc
cmake --preset release && cmake --build --preset release
./build/release/assemble ~/compressor/corpus/full volume.u8   # 1024³ scroll region
./build/release/render3d volume.u8 1024 1024 1024 --tf 1
```

## Controls

Default **orbit** camera: click-hold-drag rotates the cube (turntable around
its center), scroll wheel zooms, WASD + Q/E pans the pivot (Shift = 5×).
Switch to **fly** in the GUI (or start with `--cam`): click captures the
mouse (Esc releases; Esc again quits), WASD + Q/E fly.

Tab cycles render mode (full / MIP / depth / step-heatmap / ray-dir / flat).
T cycles transfer-function presets. `[` `]` step size, `,` `.` density,
`-` `=` LOD bias. F12 screenshot (PPM).

**Slab mode** (`--slab [wz]`): renders a thin, wide z-window (default 32
slices) of a large volume from up to 2×2 tile textures (max 4092² XY) with a
ring-buffered, scrollable z axis. R/F scroll by slice (hold to repeat),
PgUp/PgDn by window, GUI has a z slider + auto-scroll. Fetch wide regions with
`tools/fetch_slab.py out.u8 Z0 NZ Y0 NY X0 NX`, e.g.
`tools/fetch_slab.py slab3072.u8 29824 96 16384 3072 14592 3072` then
`./build/release/render3d slab3072.u8 3072 3072 96 --slab 32 --tf 1`.

Useful flags: `--size W H`, `--cam x y z yaw pitch`, `--tf N`, `--mode N`,
`--no-vsync`, `--frames N --shot out.ppm` (headless capture), `--probe`
(print device capabilities and exit), `--gpu-mem MB` (lower the automatically
derived renderer allocation budget). Reproducible runs use `--warmup N`
(same process, excluded from metrics) and `--bench-json out.json`; `tools/perf.sh`
runs the standard suite and writes mean/p50/p95/p99/max timing files.
`--quality full|interactive|fast` selects fixed full-resolution six-tap shading,
the default six-tap shading with half-resolution motion, or the measured
four-tap tetrahedral gradient plus half-resolution motion.

## Development

Presets: `dev` (ASan+UBSan RelWithDebInfo) / `release` (hardened+ThinLTO) /
`tsan`. `ctest --preset quick` runs the CPU suite; `ctest --preset dev` adds
the GPU conformance test (renders vs a CPU reference raymarcher, needs the
real GPU). Warnings are errors.

c5d is fetched at the audited revision recorded by CMake. Codec developers may
point `R3D_C5D_DIR` at a checkout; non-audited revisions additionally require
`-DR3D_ALLOW_UNPINNED_C5D=ON` because the GPU glue and kernels share a private
ABI.

Under the dev preset run with
`LSAN_OPTIONS=suppressions=lsan.supp` — it silences leak reports from system
libraries (fontconfig/wayland/vulkan loader); our own code must stay leak-free
without suppressions.

Shader workgroup variants for occupancy tuning: `R3D_WG=8x8|16x8|16x16`.
Upload path override: `R3D_STAGING=0` forces VK_EXT_host_image_copy (default
is staging — 1.86× faster for the bulk upload, see docs/measured.md; foreground
slab/clip prefer host image copy and otherwise reuse a staged-upload buffer;
the asynchronous vslab worker always uses queue-ordered staging).
Vulkan validation: `R3D_VALIDATE=1` (needs vulkan-validationlayers installed).
`R3D_NO_HOST_COPY=1` forces the portable staged streaming-upload fallback for
slab/clip/vslab testing.
