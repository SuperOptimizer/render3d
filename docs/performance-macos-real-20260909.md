# Real PHerc1218 data — 2026-09-09

This report records the baseline. See the subsequent
[optimization measurements](performance-optimized-20260909.md) for the
batched GPU updates, demand startup and reduced memory footprint.

Downloaded from [the native volcomp export](https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr/).
The full original scan is 23,247×7,593×7,593 voxels (Z/Y/X).

The local selection is about 186 MiB of downloaded native compressed data:

- Complete source levels 3, 4, and 5: 2,906×950×950, 1,453×475×475,
  and 727×238×238 voxels. These are the complete 8×/16×/32× overview pyramid.
- One full-resolution 1024³ region at Z/Y/X origin 11,264/3,072/3,072.
  Original voxel pitch is 8.64 µm; source q=8 is preserved.

Files are under `cache/PHerc1218-volcomp/`. The directory uses approximately
415 MiB including original Zarr shards, renderer containers and the decoded
fallback seed. This is a selection, not a full-resolution scan mirror.
`overview/manifest.json` rebases source level 3 as render3d level 0;
`interior-1024.vcs` contains the original-resolution region.

`tools/fetch_volcomp.py` validates the supported Zarr layout, fetches with
three concurrent downloads, and invokes the new `volcomppack zarr-shard`
mode. The importer checks index CRC32C, payload bounds and native magic,
then atomically publishes a VCS1 container. No re-encoding occurs. Every
imported native payload was compared byte-for-byte with its source; omitted
Zarr chunks map to zero fill. Import tests also cover corrupt index CRC,
out-of-bounds offsets, invalid magic, truncation and preservation of an
existing output after failure.

## Navigation benchmark

Apple M4, 16 GiB, macOS 26.3 / MoltenVK, same renderer changes as the
[synthetic benchmark](performance-macos-20260909.md). Complete overview
pyramid, three repetitions, headless 1920×1080, fixed full quality,
`--tf 1 --pool 32 --warm 128 --warmup 400 --frames 1200`, with
`R3D_MV_EXERCISE=1 R3D_MV_FIT=512`. No segment is loaded. The scripted
navigation displays three CT planes and leaves the flattened pane empty.
Network and source transcoding are not in the measured loop; the local
fallback seed and OS file cache are warm across repetitions.

Median run mean: **4.423 ms/frame** (~226 uncapped FPS).
Median run p99: **22.200 ms/frame**. Every run had zero decode failures.
Raw timing/counter summaries are in
[benchmarks/macos-real-20260909.json](benchmarks/macos-real-20260909.json),
and full logs/screenshots are in `build/bench-real/` locally.
This is an overview-resolution navigation measurement, not full-resolution
whole-scroll navigation or a loaded traced-surface workload.

## Full-resolution bulk-load finding

Opening the 1024³ region with `--pool 64` eagerly populated 237,568 nonzero
16³ blocks and took **142.068 seconds** before rendering. The startup log
labels this interval "CPU decoded", but the interval also includes GPU
uploads and post-fill work; it is not a CPU-codec throughput measurement.
A two-second macOS `sample` profile taken during the load showed the main
thread primarily in MoltenVK waits from upload completion and
`bricks_post_fill`, with relatively little time in the codec. The profile
is `build/bench-real/full-load.sample.txt`.

This confirms a large eager-atlas initialization bottleneck that the smaller
synthetic test did not expose. Many small mip/occupancy operations and
synchronous batch completion need further GPU profiling and batching work.
The profile does not isolate their individual costs or justify a native
Metal rewrite by itself. The streaming overview avoids this full-atlas
initialization and is the dataset left open in the viewer.

## Open locally

```sh
./build/macos/render3d \
  --bricks cache/PHerc1218-volcomp/overview/manifest.json --pool 32 --warm 128
```

The complete download/repack command is documented in the README under
"Real volume download". Validate the importer separately with:

```sh
python3 tests/test_import_volcomp.py build/macos/volcomppack
```
