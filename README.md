# render3d

High-performance volumetric renderer for Vesuvius Challenge micro-CT volumes.
Pure C23 (clang/LLVM), SDL3 + Vulkan compute raycasting (GL 4.6 backend planned),
Slang shaders → SPIR-V. Sibling of [c5d](../compressor) and consumer of its
16³ chunk / 128³ brick / 1024³ shard hierarchy.

- `spec/volume.md` — data conventions
- `docs/measured.md` — every design decision with the numbers behind it

## Quick start

```sh
tools/fetch_slang.sh                    # one-time: vendor pinned slangc (aarch64)
cmake --preset dev && cmake --build --preset dev
ctest --preset quick
./build/dev/tools/assemble ~/compressor/corpus/full volume.u8   # 1024³ real scroll region
LSAN_OPTIONS=suppressions=lsan.supp ./build/dev/render3d volume.u8 1024 1024 1024
```

`LSAN_OPTIONS=suppressions=lsan.supp` silences leak reports from system libraries
(fontconfig/pango/wayland/vulkan loader) under the dev (ASan) preset; our own code
must stay leak-free without suppressions.

## Development

Presets: `dev` (ASan+UBSan RelWithDebInfo) / `release` (hardened+ThinLTO) / `tsan`.
`ctest --preset quick` runs the CPU test suite; the `gpu` label needs the real GPU
and is run manually. Warnings are errors; keep clang-tidy-clean.
