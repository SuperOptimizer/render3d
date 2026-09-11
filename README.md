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

## Real volume download

Download PHerc1218's complete 8×/16×/32× overview pyramid and one 1024³
full-resolution region (about 186 MiB downloaded; about twice that on disk
with both source and renderer containers):

```sh
python3 tools/fetch_volcomp.py \
  https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr \
  cache/PHerc1218-volcomp
./build/macos/render3d --bricks cache/PHerc1218-volcomp/overview/manifest.json
```

The source is Zarr v3 `sharding_indexed` with native volume-compressor chunks.
`volcomppack zarr-shard` validates the index CRC and bounds and repackages
chunks as VCS1; the fetcher verifies every compressed payload is unchanged.
Missing Zarr chunks become zero-fill entries. Overview levels are rebased
so renderer L0 is source level 3 (69.12 µm voxels). The interior region is
`cache/PHerc1218-volcomp/interior-1024.vcs`, at original Z/Y/X origin
`11264/3072/3072`, with 8.64 µm voxels. The full-resolution scan is not fully
mirrored. Download metadata and original shards remain under `source/`.

## Storage and caches

`volcomppack`, `lodpack`, and `zarr2volcomp` write native volume-compressor
payloads. `.volc` files contain one native 128³ chunk. `.vcs` files use
render3d's VCS1 indexed container, with CRC32C and explicit missing/zero
entries; this is not upstream's zarr `sharding_indexed` container.
LOD trees use `render3d.volcomp-lod.v1` and `volcomp/L*/…vcs`.

A CPU miss reconstructs only the requested 16³ block. The codec may entropy
read up to 16 blocks within its substream but only dequantizes and transforms
the requested block. Bounded entropy checkpoints avoid repeating earlier reads
for nearby requests, using about 263 KiB per decoder thread for compressed bytes
and restart state. CPU cache sizes count 4 KiB blocks (default 4096,
16 MiB). The GPU page table and atlas also address 16³ blocks, uploaded after
CPU decode. Compressed chunks are shared by block requests through immutable
file mappings or a bounded CPU cache. `--warm` defaults to zero; a positive
value enables an optional CPU compressed-data cache in MiB. Fetch/transcode
units remain 128³ chunks or the owning upstream Zarr
cell; sparse decode does not imply 4 KiB network requests.

CT display volumes are deblocked on the CPU before upload by default. The filter
uses volume-compressor's gated four-tap kernel at every 16³ block face, including
128³ chunk seams, with the encoded block's quantizer. A bounded 32 MiB raw-block
cache supplies the two-voxel neighborhood. Missing local neighbors are skipped
and seams are retried from raw data as files arrive; filtering never initiates
extra downloads. Fully filtered coarsest levels are cached in `seed-deblock.raw`.
Set `R3D_DEBLOCK=0` before launching to disable it. This is a CT display filter:
raw CPU sampling, registration inputs, labels, and prediction overlays retain
their original values. Use the disabled setting for raw registration comparisons.
Upstream parity assumes a consistent quantizer within each pyramid level.

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
and are disabled by the capability check. GPU page tables and resident-block
metadata are sparse and sized to the atlas. Logical block IDs are 64-bit;
shaders use paired 32-bit words without requiring GPU 64-bit integer support.
Source availability uses a bounded cache, and CPU decode remains 16³-granular.
Pyramid draws use dedicated shaders to eliminate unused sampling paths while
preserving the selected quality. The full-resolution PHerc1667 volume
(5.29 billion virtual blocks) is tested,
including rendering blocks beyond the 32-bit ID range. Dimensions remain
32-bit per axis, shard-reader metadata is limited to 4 million entries, and
large volumes need a pyramid with a manageable coarsest level. Invalid browser
selections leave the current volume open.

## Data browser

Click **data browser** and choose a **source**:

- **S3 open data** lists the existing Vesuvius open-data bucket.
- **Compressed volumes (Forrest)** lists
  [the native compressed-volume collection](https://dl.ash2txt.org/community-uploads/forrest/volcomp/).

Choose a scroll, select a volume, and click **open volume**. You can switch
sources in the same window; opening another volume replaces the current dataset.
**Open as registration volume** also works with compressed volumes.

Compressed volumes download the coarsest level for the initial preview, then
fetch native 128³ payloads using HTTP ranges as you navigate. Payloads are
preserved without re-encoding; CPU decoding and decoded residency remain 16³.
Caches live under `cache/volcomp/`, separately from S3's `cache/od/`.
The compressed source currently supplies volumes; segment and prediction
browsing remains available from S3. Use **refresh** to reload a listing.
Bootstrap requires `python3`; CMake places its scripts beside the viewer.

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

**Faces annotation mode** (`--annot`) corrects TSM's two-face voxel labels on
small exported crops. A packet is a directory of raw `(z,y,x)` uint8 volumes —
CT plus the exporter's `faces_in`/`faces_out`/`ignore`/`source`/`rv_class` and
optionally `pred_in`/`pred_out` — described by `meta.json`, and `packet.json`
lists a whole batch of them (see [spec/annot.md](spec/annot.md), which is the
contract `~/tsm/dev/annot_export.py` writes to). The annotation output is a
`correction.u8` of identical shape written back into the packet directory:
0 untouched, 1 in-face, 2 out-face, 3 ignore, 4 erase.

```sh
./build/release/render3d --annot /path/to/packets/packet.json --tf 1
./build/release/render3d --annot /path/to/packets/p000        # one packet
```

The primary view is a single z slice of the CT with the exporter's labels
composited over it (faces red/blue, ignore dimmed grey, `rv_class` as tinted
bands, predictions as a dashed variant, each toggleable) and the correction on
top in saturated colours. The status line under the slice reads out the voxel
under the cursor in both packet and scroll coordinates, with its CT value, its
exporter label, and its `source` and `rv_class` values. The 3D raycast of the
same crop renders behind the panes as a secondary view.

| input | action |
| --- | --- |
| `1`-`4` | paint class: in-face, out-face, ignore, erase |
| `0` / Esc | pan tool (no painting) |
| left-drag | paint the current class with a circular brush |
| right-drag | erase the correction back to 0 (untouched) |
| Shift+click ×2 | fill line: a 1-voxel line of the current class between the two points |
| `[` / `]` | brush radius |
| `D`, `,` / `.` | depth mode on/off, and ±k slices — a stroke paints a short cylinder through z |
| `R` / `F`, PgUp/PgDn | slice ±1 / ±16 |
| wheel, Shift+wheel | zoom about the cursor, slice ±1 |
| middle-drag | pan |
| `N` / `P` | next / previous packet (saves first) |
| Ctrl+Z, Ctrl+S | undo one stroke (64 levels), force a save |

`correction.u8` is republished (temp file + rename) after every stroke, on
packet switch and at exit, so an interrupted session loses nothing and reopening
a packet resumes from what is on disk. The panel carries the packet index and
origin, the slice, class, brush, layer toggles, per-class painted-voxel counts,
and a "mark packet done" checkbox that sets `"done": true` in `meta.json`
without disturbing any other key.

The same stroke code runs without a GPU:

```sh
./build/release/render3d --annot-apply <packet> <strokes.json>
```

applies a stroke script — `{"strokes":[{"op":"brush","class":1,"z":10,
"radius":3,"depth":1,"points":[[x,y],...]}, {"op":"line",...}, {"op":"undo"}]}`
— and writes `correction.u8`, which is how `tests/test_annot.c` covers brush
geometry, the depth window, undo, atomic writes and manifest parsing against
synthetic packets from `tests/annotsynth.h`.

### Model view mode

`--view` inspects TSM **view packets** — one CT crop plus any number of named
uint8 layers (student heads, teacher outputs, label stores, human bands), each
tagged with a *kind* that says how to decode and draw it. The layer list is
data-driven: `meta.json`'s `layers[]` is the authority, so a packet gains a new
head without a line of viewer code. [spec/view.md](spec/view.md) is the
contract `~/tsm/dev/view_export.py` and `tsm serve` write to.

```sh
./build/release/render3d --view /path/to/view/view.json      # a batch
./build/release/render3d --view /path/to/view/v000           # one packet
./build/release/render3d --view /path/to/view --view-serve localhost:9760
```

The slice pane composites the layers over the CT on the CPU and hands the RGBA
result to ImGui as a texture, exactly as `--annot` does. Axis toggle (z/y/x),
slice slider, wheel zoom, shift+wheel slice, drag pan; the panel groups every
layer by its `group` with show / colour / opacity, a threshold for `prob`,
`density` and `count`, a tolerance and signed-heatmap toggle for `sdf`, and a
one-click **solo**. Three `signed` layers sharing a `vec` name draw as one RGB
normal image. **Compare** picks any two layers of the same kind and paints
agreement / A-only / B-only, with the counts and a Dice number for the current
slice. Hovering reads out every visible layer's decoded value (class layers by
palette name) plus the scroll-space coordinate. F12 writes a PNG of the
composited slice; N/P step through the packets in `view.json`.

With `--view-serve host:port` the panel gains a **predict** button that ships
the packet's box — or one walked by the arrow keys — to `tsm serve` over the
`TSV1`/`TSVR`/`TSVE` protocol on a worker thread, and merges the returned
layers back into the packet without blocking the UI.

The compositor also runs with no window and no GPU:

```sh
./build/release/render3d --view-shot <packet> <out.png> \
    [--axis z|y|x] [--slice N] [--show group.name,...] \
    [--compare group.name,group.name]
```

which is what `tests/test_view.c` drives end to end alongside the decodings,
the compositor's pixel colours, the spec's hard-error cases and the live client
against an in-test TCP server.

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

# Global multi-shard renderer (--pool sizes the GPU atlas; --warm is CPU MiB)
./build/native/render3d --bricks cache/PHerc1218-lod/manifest.json \
  --pool 8 --warm 512 --tf 1
```

The volcomp quality ladder spends more bits per voxel at coarse
levels: q8 at L0, q4 at L1, q2 at L2 and q1 thereafter. The renderer
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
For a local synthetic macOS streaming suite, run `python3 tools/bench_macos.py`.
Benchmark JSON v2 separates startup, warmup, measured and final-flush counters,
and includes measured-end cache/page-table state. GPU timestamps returned in
the measured window describe submissions two frames earlier.
See [the macOS performance measurements](docs/performance-macos-20260909.md)
for decode costs, detail arrival, memory use, and remaining bottlenecks.
The [performance review fixes](docs/performance-review-fixes-20260909.md)
describe sparse residency, CPU caching, queued updates and bounded conversion.
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

`R3D_TRACE_STARTUP=1` logs renderer creation, dataset setup, cached seed
upload/wait versus read/metadata, and total startup times.

`R3D_TRACE_WAITS=1` logs frame waits above 20 ms, separating semaphore wait
time from timestamp-query retrieval and including the completed slot's GPU time.

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

### Atlas startup and GPU update performance

Standalone `.vcs` volumes now start with an empty atlas and stream visible
16³ blocks. LOD manifests immediately show their coarsest fallback while
finer data arrives. For bulk-load throughput validation only, set
`R3D_BRICKS_EAGER=1` with a pool large enough to hold the standalone volume.
Mips and occupancy update in batched GPU compute work after CPU decoding.
The flattened-surface window is allocated on demand and released when no
segment remains open.

See [the full-codebase review](docs/performance-review-20260909.md) for the
confirmed follow-up ledger and required measurements.
The [optimization measurements](docs/performance-optimized-20260909.md)
compare real-data startup, navigation and memory before and after this pass.

### TSM inference overlays

Run multi-head 3D inference on visible CT regions with the Paris4 TSM checkpoint.
The GUI supports independent blue/red head selection, including ink, surfaces,
fibres and signed-distance fields. See [setup, controls and validation](docs/tsm-inference.md).


## Compressed parametric surfaces (read-only)

Build the standalone [surface-compressor](https://github.com/SuperOptimizer/surface-compressor)
converter and encode a tifxyz directory, then open the result with the matching
CT volume:

```sh
surface-compressor encode surface.tifxyz surface.sfc --error 0.1
build/macos/render3d --bricks /path/to/manifest.json --multiview surface.sfc
# Optional initial position, in global surface grid indices:
build/macos/render3d --bricks /path/to/manifest.json --multiview surface.sfc --surface-center 12000 8000
```

You can also use **open segment → open compressed surface** in the GUI.

The codec uses independent 64×64 floating-point DCT blocks. New files share entropy
tables across blocks while retaining independent coefficient streams. The
reader caches tables separately from decoded geometry and supports SFC
container versions 1 through 4. The viewer decodes
XYZ on the CPU into a 64 MiB block cache and uploads a maximum 1024×1024 geometry
window. Dragging the flattened view moves that window through the surface. The
**compressed surface** panel shows full dimensions, resident origin, cache use,
and global grid coordinates for jumping to another area. Whole surface
allocation is avoided; source dimensions and cache keys use 64-bit integers.

This first viewer integration is read-only. Plane intersections cover the
resident window, and zooming out does not yet build a coarse whole-surface
preview. A worker decodes and prepares new windows while the UI remains interactive. Ink-map editing
and SLIM flattening remain available for ordinary tifxyz surfaces; their controls
stay visible but are disabled for compressed windows. Existing tifxyz and .tfx
readers are unchanged. The bundled surface-compressor 1.0.0 snapshot is pinned
to its upstream revision and per-file hashes in `tools/surface-compressor/snapshot.json`.

Compressed surfaces also support version-4 joint XYZ packets. The renderer
requests all three components in one decode per 64×64 cache miss; affine/rotated
packets carry their own parameters and entropy tables, with no neighboring
patch dependency. Existing scalar surface files remain readable.
