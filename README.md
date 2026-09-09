# render3d

High-performance volumetric renderer for Vesuvius Challenge micro-CT volumes.
C23 (clang/LLVM), SDL3, and Vulkan compute raycasting through MoltenVK on
macOS. [volume-compressor](https://github.com/SuperOptimizer/volume-compressor)
provides CPU-only volume decoding. Both the CPU sampling cache and GPU atlas
use **16³ blocks (4 KiB of u8 voxels)**; disk chunks remain 128³.

## macOS build and run

```sh
brew install cmake ninja llvm ccache sdl3 vulkan-headers vulkan-loader molten-vk shaderc c-blosc libtiff
export PATH="$(brew --prefix llvm)/bin:$PATH"
./tools/fetch_slang.sh
cmake --preset macos
cmake --build --preset macos
ctest --preset macos -L quick
./build/macos/render3d --probe
./build/macos/render3d
```

The last command opens the viewer with its synthetic volume. To open data:

```sh
./build/macos/volcomppack raw volume.u8 256 256 256 volume.vcs 2
./build/macos/render3d --bricks volume.vcs
./build/macos/render3d --bricks path/to/tree/manifest.json
```

CMake fetches pinned volume-compressor and fysics revisions. The Slang fetch
script selects Linux/macOS and x86-64/ARM64. On macOS use Homebrew LLVM for C23;
Apple's bundled compiler may be too old. MoltenVK is discovered through
Homebrew's Vulkan ICD configuration. `--headless --frames 1 --shot out.ppm`
checks rendering without opening a window.

## Storage and caches

`volcomppack`, `lodpack`, and `zarr2volcomp` write native volume-compressor
payloads. `.volc` files contain one native 128³ chunk. `.vcs` files use
render3d's VCS1 indexed container, with CRC32C and explicit missing/zero
entries; this is not upstream's zarr `sharding_indexed` container.
LOD trees use `render3d.volcomp-lod.v1` and `volcomp/L*/…vcs`.

A CPU miss reconstructs only the requested 16³ block. The codec may entropy
read up to 16 blocks within its substream but only runs the inverse transform
for the requested block. CPU cache sizes count 4 KiB blocks (default 4096,
16 MiB). The GPU page table and atlas also address 16³ blocks, uploaded after
CPU decode. Compressed chunks are shared by their block requests in the warm
cache. Fetch/transcode units remain 128³ chunks or the owning upstream Zarr
cell; sparse decode does not imply 4 KiB network requests.

`--pool N` now means N³ **16³** atlas slots. Painted class IDs remain exact
in zlib R3L1 files; surface float coordinates and metadata remain exact in
zlib R3F1 files. These are distinct from lossy u8 prediction volumes.
Old c5d shards, brick caches, labels and TFX1 surface files are incompatible;
rebuild from raw/Zarr/TIFF sources into a fresh output directory. `q` is the
volume-compressor quantizer, 1–255; the old error-bound/tau knob is removed.
The headless library ABI is version 2, with `surface_encode`/`surface_decode`
entry points replacing the old format-specific functions. Historical benchmark
documents describe the old implementation.

MoltenVK runs cube, block-volume and surface views. The older tiled
slab/clip/vslab modes require more combined samplers than this Mac exposes
and are disabled by the capability check. The existing metadata budget still
limits the number of virtual block pages; excessively large manifests fail
with an explicit page-budget error.

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

**Whole-corpus surfaces**: pack a scroll's tifxyz segments with
`segpack <store-dir> [-q log2q] <tifxyz-dir>...` (volcomp-compressed `.tfx`
grids + a manifest with per-tile AABBs), then add `--segments <store-dir>`:
every surface crossing a plane view draws as a dimmed polyline under the
active segment's curve. The frame loop only queries the tile index; a
background worker decodes decimated grids into a RAM-budgeted LRU cache,
and the panel lists cache state, per-plane hit counts, and the surfaces
nearest the focus.

**Concurrent traces**: press **G** over a plane view to queue a trace seed
(green circles), then *trace all seeds* grows one tracer per seed at the
same time (each self-contained with its own solve pool, thread-capped to
share cores; 8 run in parallel from a queue of up to 100 seeds). Queue-
started traces are fully automatic: on finish each is saved into the
segment store (selectable in the surfaces panel), its 2.5D ink map is
computed in the background (no TTA) and cached, and its slot immediately
starts the next queued seed - drop 21 seeds, come back to 21 packed,
ink-mapped segments. Manually seeded traces stay resident for
interactive refine/rewind as before. Exactly one trace is displayed in the flattened
pane; the panel lists the rest with live ring/point counts and *show* /
*stop* per trace - saved traces enter the segment store/browser as usual.
Headless: `R3D_SEEDS_TEST="x,y,z;x,y,z"`.

**Tracer anchors**: when the live tracer wanders onto the wrong sheet, hover
the sheet the trace *should* pass through in any plane view and press **X**
to drop an anchor at the cursor (orange diamonds; dimmed when off the pane's
slice). Each anchor pulls the nearest traced cell through the
point with a strong solve term (an order above the fusion donor pull), so the
local neighborhood re-seats onto the correct sheet — anchors placed ahead of
the growth front engage automatically once growth comes within 3 grid steps.
Anchors work at any stage: place them before seeding, live while growing, or
on a finished trace — the **re-solve** button (appears when a done trace has
anchors) re-runs the solve over the existing grid without growing, annealing
each anchor's neighborhood through its point and finishing with a global
polish. Undo/clear from the panel. Headless: `R3D_ANCHOR_TEST="x,y,z;x,y,z"`
with `R3D_TRACE_TEST`, plus `R3D_REFINE_TEST=1` for a post-finish re-solve.

Test data comes straight from the `vesuvius-challenge-open-data` S3 bucket
(PHerc0172 pairs tifxyz segments with their exact source volume):

```sh
aws s3 cp --no-sign-request --recursive \
  "s3://vesuvius-challenge-open-data/PHerc0172/segments/<seg>/mesh/<id>.tifxyz/" \
  cache/PHerc0172-segments/w062/
# mirror coarse levels fully + fine chunks near the surface, then transcode
./build/release/zarr2volcomp cache/PHerc0172-zarr cache/PHerc0172-lod \
  --surface cache/PHerc0172-segments/w062 --pad 64 --dry-run   # plan/estimate
./build/release/zarr2volcomp ... --list-missing missing.txt        # chunk list
tools/fetch_chunks.sh <volume-zarr-url> cache/PHerc0172-zarr missing.txt 24
./build/release/zarr2volcomp cache/PHerc0172-zarr cache/PHerc0172-lod \
  --surface cache/PHerc0172-segments/w062 --pad 64 --verify 64
./build/release/render3d --bricks cache/PHerc0172-lod/manifest.json \
  --multiview cache/PHerc0172-segments/w062
```

### Multiresolution Zarr + volcomp bricks

`lodpack` builds the PHerc1218 pyramid without ever expanding the 18 GiB
mirror into a terabyte-scale raw intermediate. L0 is hard-linked into a
standard Zarr v3 hierarchy; L1-L7 are rounded isotropic 2x box reductions with
true array shapes, 1024³ shards and 16³ dct3d chunks. Every completed Zarr
chunk is decoded again before its 128³ brick is encoded to volcomp, so the volcomp
tree is a closed-loop transcode of the Zarr tree. Output shard pairs are
atomic and reruns skip completed work.

```sh
./build/native/lodpack cache/PHerc1218 cache/PHerc1218-lod \
  23552 8192 8192 7 --threads 8

# CPU/container/coordinate/fidelity check for any shard pair
./build/native/lodcheck cache/PHerc1218-lod/zarr/L1/c/1/1/1 \
  cache/PHerc1218-lod/volcomp/L1/1_1_1.vcs

# Global multi-shard renderer (use --pool/--warm to size the GPU caches)
./build/native/render3d --bricks cache/PHerc1218-lod/manifest.json \
  --pool 8 --warm 512 --tf 1
```

The volcomp quality ladder is inverted to spend more bits per voxel at coarse
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
fine/coarse jumps. Volume-compressor decode runs on a persistent CPU worker pool, leaving the
graphics queue to upload and render requested 16³ blocks.

Useful flags: `--size W H`, `--cam x y z yaw pitch`, `--tf N`, `--mode N`,
`--no-vsync`, `--frames N --shot out.ppm` (headless capture), `--probe`
(print device capabilities and exit), `--gpu-mem MB` (lower the automatically
derived renderer allocation budget). Reproducible runs use `--warmup N`
(same process, excluded from metrics) and `--bench-json out.json`; `tools/perf.sh`
runs the standard suite and writes mean/p50/p95/p99/max timing files.
`--quality full|interactive|fast` selects fixed full-resolution six-tap shading,
the default six-tap shading with half-resolution motion, or the measured
four-tap tetrahedral gradient plus half-resolution motion.

### Live 2.5D ink detection in the viewer

`tools/ink9/inkserver.py` is a persistent inference server for the scrollprize
`ink_9um` hybrid_3d2d checkpoints (run it from villa's `ink-detection` env:
`uv run python tools/ink9/inkserver.py step-075000.pth`). Add `--inklive` to a
multiview session and the flattened pane gets a green ink-probability layer:
a worker samples the visible segment rect as a 21-layer surface volume,
sends it over TCP, and the prediction is painted into the surface-volume
window (re-run whenever the view or the growing trace changes). The "live
ink" panel section shows status; port defaults to 9743.

**TTA / ensembling**: the live-ink panel's *TTA / ensemble* section picks
deterministic test-time augmentations applied by the ink server: in-plane
flips (x4), the model's 17-layer depth window slid +-1 inside the sampled
21-layer slab (+2), intensity x0.9/1.1 (+2), and a checkpoint ensemble
(seed 42 + 43, x2) - probabilities averaged after inverse transforms.
Defaults to flips+ensemble; "recompute" re-runs an existing map with new
settings. Start the server with several checkpoints to enable the
ensemble: `inkserver.py seed42.pth seed43.pth`. Ink inference only ever
runs when asked: the one-time full-surface map is the sole path (results
cached per segment; there is no automatic per-view inference).

### On-demand surface prediction (volumes without published predictions)

`tools/surf/surfserver.py` serves nnU-Net `scrollprize/surface_m7_nnunet` — the
model behind the bucket's `surface-m7-L0` trees — over TCP (run it from villa's
`ink-detection` env with `nnunetv2` installed: `uv run python
tools/surf/surfserver.py <model-dir>`; ~3 s per 8-brick cell on an RTX 5080).
`tools/surf/mkpredtree.py <ct-lod> <out>` creates a *predict tree*: a volcomp LOD
tree with the CT's geometry and no data whose `source.json` points at the
server (`predict://127.0.0.1:9744` + `ct_root`). Use it anywhere a prediction
tree goes — `--overlay <out>` for the tint, `tracecli <out>` / the GUI tracer
for growing segments: bricks are predicted on demand from CT blocks (with a
32-voxel context margin), thresholded like the bucket (p >= 0.2), and cached
under `<out>/bricks`. The model's pitch is ~8-9 um, so `mkpredtree` picks the
CT level to feed it (`pred_level` 0 for the ESRF 8.6/9.4 um scans, 2 for
2.4 um volumes — the same choice as the bucket's m7-L0 / m7-L2 trees; the
tracer should run at that level). Finer levels are served by upsampling the
predicted cell (so the plane-view tint works at any zoom); levels above
`pred_level`+1 are not produced. Together with
`--inklive`, a fresh volume goes: empty 2x2 -> trace on live surface
predictions -> live 2.5D ink on the growing trace.

### Headless benchmarks

`--headless` renders without a window, surface, swapchain or present (the
offscreen is still captured by `--shot`); `--seconds S` ends a run by wall
time. `tools/perf_headless.sh <manifest> <tifxyz> [overlay] [segstore]` runs
the multiview scenario matrix unattended and diffs against a baseline dir.

## Development

Presets: `dev` (ASan+UBSan RelWithDebInfo) / `release` (portable ThinLTO) /
`native` (release+ThinLTO tuned for the build CPU) / `tsan`.
`ctest --preset quick` runs the CPU suite; `ctest --preset dev` adds
the GPU conformance test (renders vs a CPU reference raymarcher, needs the
real GPU). Warnings are errors.

The codec revision is pinned in CMake. `R3D_VOLCOMP_DIR` can point at a local
volume-compressor checkout for development; `--probe` reports its revision.

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
