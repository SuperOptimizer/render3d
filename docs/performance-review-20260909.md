# Full-codebase performance review — 2026-09-09

Implementation follow-up: all 26 confirmed items below (C1–C12, G1–G8,
A1–A4, M1–M2) now have changes and focused verification recorded in
[the fixes report](performance-review-fixes-20260909.md). This ledger retains
the original findings and proposed acceptance checks. The measurement
hypotheses at the end remain hypotheses, not outstanding confirmed bugs.

Three independent agents reviewed CPU/data/codec/tooling, Vulkan/shaders,
and application/UI/workflows. This is a follow-up ledger for fixes after the
current measured GPU batching/startup/surface-allocation pass. The review was
read-only except for an independent GPU-kernel correctness test. Source
mechanisms below are confirmed; timing gains still need workload-specific
measurements. Hypotheses are listed separately rather than treated as proven
speedups. References use function names because the working tree is changing.

## Addressed in the current pass

- Eager standalone full-atlas loading: now opt-in (`R3D_BRICKS_EAGER=1`).
  Default standalone startup streams requested visible blocks; LOD trees
  retain their coarsest fallback. Standalone shards have no lower-resolution
  source to display as a fallback.
- Repeated per-block mip operations and duplicate eager-path mip builds:
  replaced with batched compute, once per upload.
- Occupancy: batch max-reduction and race-free halo dilation in the upload
  submission, including streamed standalone updates.
- Empty-segment 1 GiB flattened window: allocated on first valid segment
  use and released when the segment is closed; empty pane skips baking.
- Standalone streaming now honors the requested Z slice/slab bounds.
- Eviction job array: `job_evict[32]` was too small for 256-block batches.
  It now uses the shared batch constant; the test evicts 128 blocks/job.
- Vulkan core `shaderStorageImageExtendedFormats`: checked and enabled for
  R8 compute images. Parsed shader capabilities require it.
- Brick page tables now use queue-ordered device updates, with barriers
  before writes and before subsequent sampling. This removes eviction
  CPU drains and avoids host publication racing earlier frame reads.
- LOD seed uploads no longer read uninitialized world IDs for unused
  standalone-kernel metadata.

## Confirmed follow-up issues

| ID | Priority | Mechanism and location | Proposed fix / acceptance check |
|---|---|---|---|
| C1 | High correctness | `tools/lodpack/main.c`, shard writer creation passes q=0, rejected by `src/codec/shard.c`. | Use actual per-level q; small end-to-end conversion test. |
| C2 | High correctness | `src/core/cpuvol.c`, prefetch checks 128³ source coordinates against 16³ decoded/negative-cache keys. | Track/check source-chunk availability separately; cached block (1,0,0) must not suppress fetching source chunk (1,0,0). |
| C3 | High correctness | `src/main.c`, surface streaming revisits only ~384 sample points regardless of visible extent; displayed normal offset and thickness are not included. | Conservatively cover surface patches extruded through the visible slab at chosen LOD. Test a 1024² plane, offsets beyond16 voxels, both signs and thick slabs; eventual complete fine-block coverage. |
| C4 | High | `src/core/cpuvol.c`, `io_mu` spans HTTP retries, disk reads and CRC work. A slow cold fetch blocks unrelated local misses. | Narrow locking to reader publication; lease immutable mappings; deduplicate in-flight source cells and use independent HTTP handles. Delayed HTTP request must not stall local misses. |
| C5 | High | CPU `.volc` misses and Vulkan `ni_load_brick` repeatedly open/read/allocate the entire128³ compressed chunk for each16³ block. | Bounded leased compressed-file cache or per-batch deduplication. Verify file identity/replacement, count opens/read bytes; retain16³ decoded residency. Header-table cache already compares bytes, so allocation changes alone do not invalidate it. |
| C6 | High | `src/headless/headless.c`, ROI decoding calls `cpuvol_read_block` once per Z slice. An XY working set larger than the cache can decode each block16 times. | Block-order or aligned16-slice traversal, preserving cancellation/transactional output. Assert decoded-block counts with deliberately small cache and compare bytes. |
| G1 | High | `src/vk/vkbackend.c`, `bricks_pick_slot` scans pool per requested block: up to67.1M slot inspections for256 requests at64³ slots. | Free-slot stack plus efficient batch eviction/LRU. Protect pinned/currently wanted slots; benchmark full-pool churn and initial fill. |
| G2 | High | Vulkan metadata is dense per16³ virtual page: approximately42 bytes/page before network maps; permitted256M pages can cost~10.5 GiB. Warm metadata is only used at128³ chunk origins. | Chunk-index warm metadata and candidate storage sized to collected requests; eventually sparse/two-level pages. Test large manifests with no payload loading. |
| G3 | Addressed in current pass | Host page tables drained frames on eviction and could race publication against earlier shader reads. | Brick tables now always use device updates, with read→write and write→read barriers; GPU integration tests cover pool churn. |
| C7 | Medium | `cpuvol.c`, `cvc_victim` scans every slot under one mutex on each insertion, including free-cache fill. | Free list plus evictable LRU/clock with correct pin transitions. Preserve uncached scratch fallback when all slots are leased; multi-thread miss benchmarks. |
| C8 | Medium | `tracer.c`, distance-transform setup samples a regular96³ footprint voxel-by-voxel. | Bounded block read then threshold existing EDT scratch. Compare transforms/traces at edges and negative coordinates. |
| C9 | Medium | CPU prefetch source-cell deduplication is quadratic; ink preparation only deduplicates a short recent window. | Sort/unique or hash packed cells; compare exact requested-cell sets for large warped sheets. |
| C10 | Medium | `codec/shard.c`, serial byte-at-a-time CRC32C under cold-read I/O lock. | First benchmark cold real shards, then hardware CRC32C with portable fallback and known vectors. Warm verification is cached already. |
| G4 | Medium | Label/registration atlas synchronization repeatedly scans whole pools and invokes generation callbacks; partial batches restart from slot zero. | Dirty/reassigned queues, source generation notifications and persistent cursors. Measure idle CPU and paint latency; test stale-slot clearing. |
| G5 | Medium | Every residency publication invalidates all panes and surface layers, including unrelated views. | Separate resource generations, then conservative spatial dependency tracking. Measure panes redrawn/baked texels without suppressing needed refreshes. |
| G6 | Medium | Mask/prediction uploads allocate staging and wait synchronously; full masks reupload during painting. Transfer function changes idle the device. | Reuse staging, upload dirty rectangles, queue with frame work and defer destruction. Compare rapid-stroke final contents and p99. |
| G7 | Medium | Fixed64/128 CT budgets and8 label blocks/update can constrain detail arrival under other workloads. | Measure time-based budgets for each path; compare equal detail as well as frame time. Larger batches alone are not an unconditional improvement. |
| A1 | Medium | `main.c`, finished traces still scan the full trace grid for each visible2D pane every frame. | Cache generation/basis/slice candidate sets and viewport-cull. Compare visible trace points and large-trace CPU cost. |
| A2 | Medium | Hidden corpus lines (`mv_corpus_vis==0`) still query, queue loads, intersect and prepare draws. | Skip hidden pipeline; ensure hiding produces no new work and restoring resumes. |
| A3 | Medium | Corpus rendering holds the loader mutex through queries, intersections and ImGui tessellation. | Pin/snapshot immutable entries under a short lock; process outside it. Test eviction/activation and loader progress under navigation. |
| A4 | Medium | Expanded surface lists use insertion sort and emit every row while holding loader mutex. | Cached O(N log N) ordering and ImGuiListClipper; selection/frame-time test with10k entries. |
| M1 | Measurement | Timings exclude warmup but decode counters include warmup/startup/final flush. Frame stats keep4096 samples while phase stats retain a different window. | Report distinct interval deltas and consistent observation windows; deterministic counter-window test. |
| M2 | Measurement | `--seconds` preallocates2M profiling samples (~128 MB), even for short runs. | Grow on actual samples or consistent bounded statistics; short timed-run memory check. |
| G8 | Lower | Compressed warm tier duplicates immutable mmap-backed chunks in a Vulkan host-visible buffer; default reservation256 MiB–3 GiB. | Compare retained mmap pointers and ordinary bounded CPU compressed caching; measure memory, faults and decoder locality. |
| C11 | Lower | Headless surface encode/decode stages full interleaved/deinterleaved/little-endian copies. | Profile real large segments; representation-aware/streaming codec path preserving exact bits, allocators and transactional output. |
| C12 | Lower | Offline tools materialize whole1 GiB shard buffers and retain encoded chunks before writing. | Bounded producer/writer pipeline preserving ordering, downsample boundaries and source reuse; measure throughput and peak memory. |

## Measurement gaps and hypotheses

- Ray-march transcendental cost, gradient taps and surface-kernel divergence
  need GPU traces before algorithm/quality changes. There is no review
  evidence supporting a native Metal rewrite.
- Full candidate sorting may be wasteful compared with bounded top-k or
  buckets; profile after slot-allocation costs are removed.
- Compare macOS LTO/native build settings before adopting them.
- Expand decoder benchmarks beyond repeated one-chunk traversal: alternating
  chunks, representative real chunks, and concurrent workers.
- Add real loaded segments, growing/finished traces and large surface corpora
  to benchmarks. Overview-only navigation cannot validate all workflows.

Suggested sequence: correctness C1–C3; locality/serialization C4–C7 and G1;
metadata/page churn G2 (G3 is addressed); dirty work and UI G4–G7/A1–A4; then the measured
conversion, CRC and warm-buffer opportunities. M1/M2 should accompany new
benchmark claims. Each fix needs its focused acceptance check; "confirmed"
above refers to the code mechanism, not an already measured speedup.
