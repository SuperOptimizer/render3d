# Startup, GPU updates and memory — 2026-09-09

Measured on Apple M4 / 16 GiB, macOS 26.3, Release build through MoltenVK.
The real PHerc1218 data and baseline are described in
[the original real-data report](performance-macos-real-20260909.md).

| Measurement | Before | After |
|---|---:|---:|
| Overview navigation, median run mean | 4.423 ms/frame | 3.441 ms/frame |
| Overview navigation, median run p99 | 22.200 ms/frame | 6.269 ms/frame |
| Median peak memory footprint | 1450.1 MiB | 428.2 MiB |
| Initialize 237,568 full-resolution atlas blocks | 142.068 s | 3.273–3.878 s |

Navigation uses three repetitions at fixed full quality, 1920×1080,
`--tf 1 --pool 32 --warm 128 --warmup 400 --frames 1200 --headless --no-vsync`,
with `R3D_MV_EXERCISE=1 R3D_MV_FIT=512` and the overview manifest. No segment
is loaded. Source data is local and the OS file cache is warm; these are
rendering/streaming improvements, not download-speed measurements. Every
run had zero decode failures. Timings improve about 22% on average and 72%
at p99. Memory uses macOS `time -l` peak footprint, not just resident set size.

Standalone volumes now start empty and stream requested 16³ blocks, honoring
slice/slab bounds. A three-frame 1280×720 demand-startup run took 0.16 s and
decoded 128 blocks; this is responsiveness, not complete-volume residency.
LOD manifests retain their existing coarse fallback. Standalone shards
have no lower-resolution fallback source.

The equivalent bulk-load measurement explicitly sets `R3D_BRICKS_EAGER=1`
and uses `--pool 64 --frames 3 --size 1280 720 --tf 1 --depth 1` with the
full-resolution interior shard. Two runs initialized the same 237,568 blocks;
both output images were byte-identical to the baseline image. The interval
includes CPU decoding, uploads and GPU postprocessing. Batched mip generation
and occupancy updates remove duplicate mip builds and many small submissions.
Decoded residency remains 16³ and codec decoding remains on the CPU.

The flattened-surface allocation is now lazy and released on segment close,
removing about 1 GiB from the no-segment workload. Brick page-table updates
are device copies ordered with GPU work, with barriers before writes and
before sampling. This avoids host publication racing prior frames and
removes the eviction-path CPU drain.

## Verification and remaining work

Release build and 18 CPU/GPU tests passed. All 9 GPU tests passed again after
the final page-table barrier change. The two atlas/kernel integration tests
also passed with sanitizers. Coverage includes streaming startup, 128-block
eviction jobs, surface release/reopen, and exact mip/occupancy comparison
against an independent integer CPU reference, including decreasing maxima.

The [three-agent review](performance-review-20260909.md) tracks remaining
correctness and performance issues with acceptance checks. These benchmarks
do not cover loaded segments, large corpora, tracing or whole-scroll
full-resolution navigation. Decode counters currently include startup and
warmup, unlike frame timings; the review records that measurement limitation.
No evidence here calls for a native Metal rewrite.

[Raw benchmark results](benchmarks/macos-optimized-20260909.json) preserve
all three final runs. Local logs, test output and bulk-load screenshots are
under `build/bench-optimized/`.
