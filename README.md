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

**Umbilicus annotation mode** streams the complete 8192×8192 XY plane of the
[PHerc1218 export](https://dl.ash2txt.org/community-uploads/forrest/exports/PHerc1218/)
in an 8-slice-deep window and writes an annotation after every click:

```sh
./build/native/render3d --umbilicus PHerc1218-umbilicus.json --tf 1
```

Plain click sets/replaces the `{x,y,z}` point on the current slice. Wheel and
R/F move one slice. Previous/Next and PageUp/PageDown move by the configurable
annotation step (default 100), and Ctrl+click saves then advances by that step.
Shift-drag pans; Shift-wheel zooms. The local shard cache is created at
`cache/PHerc1218`. Output JSON contains both Villa `control_points` and Volume
Cartographer `points` arrays, uses full-resolution voxel coordinates, and is
atomically autosaved so an interrupted session can be resumed.

Once the current plane is resident, a background worker decodes five annotation
steps in the direction of travel and one behind it. Six full-plane decoded
windows are retained (up to 3.75 GiB for this volume), so consecutive
Previous/Next or PageUp/PageDown moves normally avoid 3ddct decode latency. The
annotation panel reports cache readiness and hits. `--ann-prefetch N` selects 0-5 forward
steps (default 5); set it to 0 or use `R3D_VSLAB_NOPC=1` for memory-constrained
runs. `tools/bench_ann_prefetch.sh` compares look-ahead depths with a repeatable
six-jump navigation test.

Fine wheel scrolling has a separate GPU-resident neighborhood: by default 32
slices before and after the visible 8-slice plane are kept in a 74-layer ring.
Z stacks and their overview pyramids are uploaded in batches, so nearby slices
can be displayed without per-tile pop-in. This ring uses about 5.4 GB here;
`--ann-z-prefetch N` selects 0-128 slices per side. Together with the decoded
jump cache, the maximum annotation-mode cache footprint is about 9.2 GB on this
machine. The benchmark script also runs a repeatable 20-slice fine-scroll test.

### 2x2 multi-view (vc3d-style) on AWS open data

`--multiview <tifxyz-dir>` opens the volume-cartographer layout: top-left the
flattened segment, top-right XY, bottom-left XZ, bottom-right YZ — all
orthographic slice views over the `--bricks` LOD cache with a shared focus.
Drag pans a view, wheel zooms about the cursor, Shift+wheel (or R/F) scrubs
the hovered view's slice — on the segment view it slides the sampling shell
along the local surface normal. **Ctrl+click sets the focus point**: all
plane views recenter and re-slice through it, and the segment view recenters
on the nearest surface point. Plane views draw the segment intersection curve
(+ a translucent copy at the current normal offset); the segment view draws
the three plane traces (orange XY / red XZ / yellow YZ).

The **segment-aligned planes** checkbox reorients the two side panes to the
surface normal at the focus (vc3d's seg-xz/seg-yz): both panes contain the
normal with "up" pointing off the recto side, so the sheet lies edge-on and
roughly horizontal, and their horizontals stay perpendicular — the rotation
slider spins the pair around the normal. Scrubbing moves in signed offsets
from the focus. Ctrl+click re-anchors the frames at the new focus; XY stays
axis-aligned.

Test data comes straight from the `vesuvius-challenge-open-data` S3 bucket
(PHerc0172 pairs tifxyz segments with their exact source volume):

```sh
aws s3 cp --no-sign-request --recursive \
  "s3://vesuvius-challenge-open-data/PHerc0172/segments/<seg>/mesh/<id>.tifxyz/" \
  cache/PHerc0172-segments/w062/
# mirror coarse levels fully + fine chunks near the surface, then transcode
./build/release/zarr2c5d cache/PHerc0172-zarr cache/PHerc0172-lod \
  --surface cache/PHerc0172-segments/w062 --pad 64 --dry-run   # plan/estimate
./build/release/zarr2c5d ... --list-missing missing.txt        # chunk list
tools/fetch_chunks.sh <volume-zarr-url> cache/PHerc0172-zarr missing.txt 24
./build/release/zarr2c5d cache/PHerc0172-zarr cache/PHerc0172-lod \
  --surface cache/PHerc0172-segments/w062 --pad 64 --verify 64
./build/release/render3d --bricks cache/PHerc0172-lod/manifest.json \
  --multiview cache/PHerc0172-segments/w062
```

### Multiresolution Zarr + c5d bricks

`lodpack` builds the PHerc1218 pyramid without ever expanding the 18 GiB
mirror into a terabyte-scale raw intermediate. L0 is hard-linked into a
standard Zarr v3 hierarchy; L1-L7 are rounded isotropic 2x box reductions with
true array shapes, 1024³ shards and 16³ dct3d chunks. Every completed Zarr
chunk is decoded again before its 128³ brick is encoded to c5d, so the c5d
tree is a closed-loop transcode of the Zarr tree. Output shard pairs are
atomic and reruns skip completed work.

```sh
./build/native/lodpack cache/PHerc1218 cache/PHerc1218-lod \
  23552 8192 8192 7 --threads 8

# CPU/container/coordinate/fidelity check for any shard pair
./build/native/lodcheck cache/PHerc1218-lod/zarr/L1/c/1/1/1 \
  cache/PHerc1218-lod/c5d/L1/1_1_1.c5s

# Global multi-shard renderer (use --pool/--warm to size the GPU caches)
./build/native/render3d --bricks cache/PHerc1218-lod/manifest.json \
  --pool 8 --warm 512 --tf 1
```

The c5d quality ladder is inverted to spend more bits per voxel at coarse
levels: q2 at L0, q1 at L1, q0.5 at L2 and q0.25 thereafter. The renderer
opens a multilevel manifest as an 8-slice XY slab at mid-z. Wheel, R/F, and
PageUp/PageDown move through z; Shift+wheel zooms. `--brick-z Z` selects the
initial slice, `--depth N` changes its thickness, and explicit `--depth 0`
enables the expensive whole-volume diagnostic view.

The complete coarsest level remains resident. The streamer bounds its search
to the visible slice and view cone, estimates each brick's projected
base-voxel footprint, and requests the corresponding on-disk LOD. Sampling
tries the desired level and at most two immediate parents before the pinned
fallback, so independently arriving LODs compose without holes or arbitrary
fine/coarse jumps. Manifest c5d decode is asynchronous on four CPU threads by
default, avoiding contention with raycasting on this GPU's single compute and
graphics queue; `R3D_BRICKS_GPU_DECODE=1` retains the diagnostic GPU path.

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

Presets: `dev` (ASan+UBSan RelWithDebInfo) / `release` (portable ThinLTO) /
`native` (release+ThinLTO tuned for the build CPU) / `tsan`.
`ctest --preset quick` runs the CPU suite; `ctest --preset dev` adds
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
Without an override, Adreno X1-85 uses the measured 8x8 variant only during
reduced-resolution interaction and retains 16x8 for full-resolution rendering.
Upload path override: `R3D_STAGING=0` forces VK_EXT_host_image_copy (default
is staging — 1.86× faster for the bulk upload, see docs/measured.md; foreground
slab/clip prefer host image copy and otherwise reuse a staged-upload buffer;
the asynchronous vslab worker always uses queue-ordered staging).
Vulkan validation: `R3D_VALIDATE=1` (needs vulkan-validationlayers installed).
`R3D_NO_HOST_COPY=1` forces the portable staged streaming-upload fallback for
slab/clip/vslab testing.
