# render3d

High-performance volumetric renderer for Vesuvius Challenge micro-CT volumes.
Pure C23 (clang/LLVM), SDL3 + Vulkan compute raycasting (GL 4.6 backend planned),
Slang shaders → SPIR-V. Sibling of [c5d](../compressor) and consumer of its
16³ chunk / 128³ brick / 1024³ shard hierarchy.

Milestone 1 status: 1024³ real scroll region (PHercParis4) at **225 fps /
1920×1080** (full gradient shading, ray-cone LOD, release build, Adreno X1-85).

- `spec/volume.md` — data conventions
- `docs/measured.md` — every design decision with the numbers behind it

## Quick start

```sh
tools/fetch_slang.sh                    # one-time: vendor pinned slangc
cmake --preset release && cmake --build --preset release
./build/release/assemble ~/compressor/corpus/full volume.u8   # 1024³ scroll region
./build/release/render3d volume.u8 1024 1024 1024 --tf 1
```

## Controls

Click captures the mouse (Esc releases; Esc again quits). WASD + Q/E fly,
Shift = 5×. Tab cycles render mode (full / MIP / depth / step-heatmap /
ray-dir / flat). T cycles transfer-function presets. `[` `]` step size,
`,` `.` density, `-` `=` LOD bias. F12 screenshot (PPM).

Useful flags: `--size W H`, `--cam x y z yaw pitch`, `--tf N`, `--mode N`,
`--no-vsync`, `--frames N --shot out.ppm` (headless capture), `--probe`
(print device capabilities and exit).

## Development

Presets: `dev` (ASan+UBSan RelWithDebInfo) / `release` (hardened+ThinLTO) /
`tsan`. `ctest --preset quick` runs the CPU suite; `ctest --preset dev` adds
the GPU conformance test (renders vs a CPU reference raymarcher, needs the
real GPU). Warnings are errors.

Under the dev preset run with
`LSAN_OPTIONS=suppressions=lsan.supp` — it silences leak reports from system
libraries (fontconfig/wayland/vulkan loader); our own code must stay leak-free
without suppressions.

Shader workgroup variants for occupancy tuning: `R3D_WG=8x8|16x8|16x16`.
Upload path override: `R3D_STAGING=1` (default is VK_EXT_host_image_copy).
Vulkan validation: `R3D_VALIDATE=1` (needs vulkan-validationlayers installed).
