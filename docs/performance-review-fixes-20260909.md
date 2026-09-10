# Performance review fixes — 2026-09-09

Follow-up to the [three-agent review](performance-review-20260909.md).
Changes preserve native volume-compressor payloads, CPU-only codec decoding,
and 16³ decoded CPU/GPU residency. This report records mechanisms and tests;
a structural improvement is not a measured speedup unless a result appears below.

## CPU, storage and conversion

| Review IDs | Change | Evidence |
|---|---|---|
| C1 | `lodpack` writes its actual per-level quantizer. | End-to-end L0/L1 conversion checks footer values, native block decode, Zarr reconstruction and resume. |
| C2 | Prefetch checks source-chunk availability independently of decoded 16³ keys. | A cached block with colliding source coordinates cannot suppress the required fetch. |
| C4 | Reader-local initialization, immutable mapped payloads, per-source-cell download locks and thread-local HTTP handles replace the global I/O lock. | Delayed HTTP does not block an unrelated local read; concurrent requests deduplicate source work. |
| C5 | CPU compressed-file leases validate file identity and retain bounded encoded data; GPU jobs reuse a file within a batch. | 512 neighboring block decodes read one compressed file; replacement and noisy streams above 1 MiB are tested. |
| C6 | Headless ROI reads aligned 16-slice bands. | A 256×128×32 ROI with only eight decoded cache slots performs 256 block decodes, with byte-exact unaligned/negative-origin checks and transactional cancellation. |
| C7 | An evictable list replaces full-cache victim scans. | Existing multithreaded lease churn and all-pinned fallback tests, plus sanitizer checks. |
| C8 | Tracer distance-transform setup reads its regular footprint in block order, reusing existing scratch. | Original/new transforms match at interior, negative and edge coordinates, with repair enabled and disabled. |
| C9 | Source prefetch sorts/uniques cells; ink preparation uses a complete XYZ set and bounded batches. | Exact duplicate elimination, full 32-bit coordinates and flush behavior tested. |
| C10 | Hardware CRC32C with portable fallback. | Known vectors, unaligned lengths and real-shard checksum/throughput comparison. |
| C11 | Streaming R3F1 compression/decompression uses 64 KiB scratch and strided views, eliminating full-size intermediate copies. | Independent legacy files round-trip exact float bits, including NaN payloads and signed zero; custom allocators and malformed input preserve ownership/output. |
| C12 | Persistent workers process bounded conversion batches/tiles; completed native payloads are written and freed. Zarr conversion uses a rolling source-plane buffer. | Atomic failure/resume, 192³ source boundaries, memory-budget rejection, multilevel rounding and tile-boundary checks. |

The compressed CPU cache has 16 lazily allocated entries capped at 16 MiB each
(256 MiB maximum); this covers the native encoder's worst-case bound. Decode
cache residency remains 4 KiB per block. Hash collisions can serialize two
network fetches, but unrelated local reads do not take those locks. Source
mappings remain immutable for their open lifetime.

## Application and measurement

| Review IDs | Change | Evidence |
|---|---|---|
| C3 | Cached surface tile bounds cover the displayed slab, including normalized normals, both offset signs and thickness. | 1024² plane coverage and cancellation-prone normal interpolation tested against conservative bounds. |
| A1 | Trace slice candidates cache by content revision, basis and slice, then viewport-cull. | Exact oblique selection and unchanged-input cache-hit tests. |
| A2 | Hidden corpus overlays skip queries, load requests and tracing. | Actual hidden run: zero cached grids, hits and traces; visible fixture loads and draws. |
| A3 | Corpus grids are pinned briefly under the loader lock; intersections and drawing run outside it. | Worker progresses while another grid is pinned beyond a tiny cache budget; releasing pins permits safe eviction. |
| A4 | Corpus names sort once with `qsort`; ImGui clips the displayed list. | 10,000 names sorted correctly, fewer than 50 rows submitted for the test viewport. |
| M1 | Benchmark v2 separates startup, warmup, measured and final-flush counters; all timing summaries use the same measured sample set and named fields. | Counter partitioning and distinct phase-value tests, plus actual JSON smoke checks. |
| M2 | Sample storage grows with observed frames. | Growth beyond 4096 samples tested; short timed runs no longer reserve two million samples. |

GPU timestamp queries describe submissions two frames earlier; benchmark v2
explicitly reports that lag. CPU frame timings use the current measured
frames. GPU timestamps have a consistent observation window, not exact
submission alignment with CPU boundary counters.

## GPU residency and updates

| Review IDs | Change | Evidence |
|---|---|---|
| G1 | Batch eviction uses bounded clock traversal and protects pinned/currently requested blocks. | Atlas churn and slot-probe counters. |
| G2 | Sparse CPU/GPU block maps replace dense virtual-page arrays; source/warm indices use 128³ chunks. | Collision, deletion, reuse and known-air GPU tests; a 374,917,031-page manifest opens within a bounded metadata budget. |
| G3 | Page-table publication uses queue-ordered copies and sampling barriers. | Streaming and eviction integration tests. |
| G4 | Label/registration revisions skip idle scans; bounded cursors finish partial work. | Idle callback counts, stale-slot repair and retry tests. |
| G5 | Separate resource generations and conservative pane bounds limit redraws to affected views. | Three-pane test verifies selective redraw counts and pixel equivalence after forced redraw. |
| G6 | Transfer-function, mask and prediction updates use reusable staging and queued dirty rectangles. | Coalesced edits, restoration, texture resizing, mask preservation and clear/cancel image checks. |
| G7 | Streaming budgets adapt to completed work, with bounded moving/settled targets. | Real-data detail counters accompany timing; budgets remain capped at 256 blocks. |
| G8 | Compressed data uses immutable mappings by default; explicit warm caching uses ordinary CPU memory. | Default warm capacity is zero; matched explicit-cache and default-cache benchmarks below. |

Known-empty fine blocks retain a distinct page marker so they cannot reveal
coarse data beneath them. Pruning that bounded marker cache also forgets the
associated CPU negative entry, allowing the marker to be restored on demand.
GPU page hashes are sized to at least four times the atlas slot count.

An additional integration fix makes registration retry unavailable source
blocks without requiring a transform edit. Incomplete atlas blocks remain
zero until complete, preserving the cache-key invariant. Transform revisions
discard obsolete retries; stable completed registrations skip idle scans.

## Measurements

Apple M4, 16 GiB, macOS. These microbenchmarks use local data and do not
measure download speed. Historical frame-time improvements from the previous
pass are in [the earlier report](performance-optimized-20260909.md).

| Workload | Before | After |
|---|---:|---:|
| CRC32C, real 56.85 MB native shard | 566.6 MB/s | 7025.9 MB/s |
| 2048² surface encode/decode peak footprint | 195.7 MiB | 99.6 MiB |
| 2048² surface median encode | 54.59 ms | 54.18 ms |
| 2048² surface median decode | 9.84 ms | 9.87 ms |
| Dense 1024³ packing, wall time / maximum RSS | 2.55 s / 1237 MiB | 2.82 s / 58.6 MiB |
| Sparse 1024³ packing, wall time / maximum RSS | 0.51 s / 1064 MiB | 0.10 s / 38 MiB |
| Dense pyramid generation, wall time / maximum RSS | 1.59 s / 1190 MiB | 1.81 s / 48.3 MiB |
| Sparse pyramid generation, wall time / maximum RSS | 0.35 s / 1190 MiB | 0.20 s / 47 MiB |

Surface measurements use three encode/decode iterations per process and
identical coordinates. Both encoders produced 1,369,389 bytes. Memory is macOS
`time -l` peak footprint; this should not be confused with maximum RSS.

[Raw microbenchmark results](benchmarks/macos-review-fixes-20260909.json) and
local logs in `build/bench-review-fixes/` preserve measurements. Conversion
comparisons use original HEAD tool algorithms linked to the same current
codec libraries. They are single runs on a logical 1024³ fixture; dense
entries share one encoded pattern, testing compute rather than unique payload
disk bandwidth. Bounded memory trades approximately 11–14% runtime on these
dense cases. The JSON labels intermediate results separately.

### Real-volume renderer

Three repetitions per configuration, alternating cache settings, use the local
PHerc1218 overview, `R3D_MV_EXERCISE=1 R3D_MV_FIT=512`, fixed full quality,
1920×1080, `--tf 1 --pool 32 --warmup 400 --frames 1200 --headless --no-vsync`.
No segment is loaded. These are rendering/streaming measurements with warm OS
file caching, not download throughput.

| Median across three runs | Previous pass, warm128 | Current, warm128 | Current default, warm0 |
|---|---:|---:|---:|
| Mean CPU frame time | 3.441 ms | 3.239 ms | 3.250 ms |
| CPU frame p99 | 6.269 ms | 6.218 ms | 5.933 ms |
| Peak memory footprint | 428.2 MiB | 297.6 MiB | 291.6 MiB |

All six current runs had zero decode failures. The default measured about
5.5% lower mean frame time and 32% lower peak footprint than the previous
pass. The two current cache settings perform similarly on this warm local
workload; it does not establish their relative performance on cold storage.
Timed streaming does not guarantee identical intermediate residency: current
runs decoded 5,801–5,830 blocks across all intervals and ended with 7,300–7,305
resident blocks including startup seeds. JSON v2 preserves each interval and
the final residency, so measured-only counters should not be compared directly
to the previous v1 cumulative counters. Old CPU phase fields also had a field
indexing bug; the total CPU frame measurements used here are unaffected.

Two separate eager-load captures initialized the same 237,568 interior blocks
in 4.230 and 3.511 seconds, versus 3.273–3.878 seconds in the previous pass.
Both output images are byte-identical to the original real-data baseline.
This confirms equal-residency rendering but does not show an eager-load
speedup. Eager loading remains an opt-in diagnostic (`R3D_BRICKS_EAGER=1`);
normal startup requests visible 16³ blocks.

## Verification

The complete Release and ASAN/UBSAN suites each passed 31 tests: 18 CPU/app
and 13 GPU integration tests. The separate `shard` data test skipped in both
builds because the historical DCT `band/` fixture is absent. Synthetic DCT
conversion and native volume-compressor import tests passed, including the
standalone Python import identity/corruption/atomic-output test.
A one-second default-cache overview run retained exactly its 8,731 measured
frame samples, with zero decode failures, rather than reserving two million.

Repeated textured post-filter captures allow a maximum difference of one
8-bit channel step in at most 0.01% of components. Asynchronous loading can
change physical slots in a non-power-of-two atlas, slightly changing
normalized texture-coordinate interpolation rounding. The observed failing
exact comparison differed in three of 2,764,800 components by one step, with
identical decoded-block counts. Constant-volume filter identity remains exact.
