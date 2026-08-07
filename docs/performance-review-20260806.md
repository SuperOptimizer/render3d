# Performance implementation report — 2026-08-06

This is the implementation record for the 2026-08-06 renderer review. It
separates throughput, latency, correctness, and portability: a change which
only makes failure deterministic is not presented as a frame-time speedup.

## Test system and protocol

- NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB, driver 595.84, Vulkan 1.4.329
- x86-64, clang 21, release build, 1920x1080, vsync disabled
- 1024^3 synthetic gyroid for repeatable cube tests; real 3072^2 PHercParis4
  data for slab and image-quality tests
- Each steady-state scenario runs in one process: 400 warm-up frames, then
  1,200 measured frames. Warm-up samples are excluded.
- Reported GPU values are Vulkan timestamp-query results. JSON stores mean,
  p50, p95, p99, and maximum for each CPU/GPU zone.
- c5d revision `d09400711b5bc10dfd1075404e96edeb64c58893`, GPU ABI 20260806

The warmed protocol matters on this GPU: its clock rises from roughly 210 MHz
at idle to roughly 2055 MHz under load. The former process-per-scenario harness
included clock ramp in its average and could report 1.90, 4.70, or 5.98 ms for
the same scenario. `tools/perf.sh` now enforces the same-process warm-up.

## Final steady-state results

The table reports raycast time, not presentation or GUI time.

| Scenario | Mean | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Orbit | 1.455 ms | 1.393 | 1.811 | 1.846 | 1.884 |
| Zoom | 4.464 ms | 4.385 | 7.252 | 7.540 | 8.393 |
| Fly, interior | 3.295 ms | 3.832 | 6.585 | 7.252 | 7.983 |
| Orbit, low-cut 110 | 1.229 ms | 1.181 | 1.559 | 1.597 | 1.629 |
| Fly, low-cut 110 | 2.613 ms | 2.869 | 5.288 | 6.132 | 7.788 |
| Exterior | 1.198 ms | 1.192 | 1.208 | 1.400 | 1.659 |
| Dense interior | 6.147 ms | 6.144 | 6.168 | 6.245 | 6.310 |
| MIP worst case | 4.033 ms | 4.048 | 4.167 | 4.832 | 5.484 |
| Slab, one tile | 0.354 ms | 0.356 | 0.369 | 0.371 | 0.373 |
| Slab, real 3072^2 | 0.531 ms | 0.531 | 0.534 | 0.534 | 0.536 |

Against the pre-implementation warmed baseline, the default-quality shader is
effectively neutral: orbit 1.48 -> 1.46 ms, zoom 4.46 -> 4.46, exterior 1.22
-> 1.20, MIP 4.04 -> 4.03, dense 5.97 -> 6.15, and fly 3.19 -> 3.29. The two
roughly 3% increases are within the observed long-run thermal spread. Most of
this pass targeted stalls, unsafe assumptions, and large-data failure modes,
not a different default image.

Startup of the 1024^3 cube is 0.93 s wall time with 1,389,872 KiB peak RSS:
259 ms bulk upload (4.15 GB/s), 7 ms mip generation, and 82 ms occupancy build
and upload. The pre-pass values were 1.05 s, 263 ms, and 86 ms respectively.

## Implemented changes

### Reproducible codec and shader contract

CMake fetches an exact c5d commit by default. A local `R3D_C5D_DIR` checkout
must be clean and at that commit unless `R3D_ALLOW_UNPINNED_C5D=ON` is
explicitly selected. The revision and GPU ABI appear in `--probe` and every
benchmark JSON file. This closes the failure mode where a sibling c5d working
tree changed the private entropy/dequantization ABI and caused hangs or quiet
voxel corruption in render3d.

### Vulkan portability and memory ceilings

- The 232-byte frame block moved from push constants to a two-frame dynamic
  uniform-buffer ring, respecting `minUniformBufferOffsetAlignment`. This
  removes the accidental 256-byte device requirement; Vulkan only guarantees
  128 bytes of push constants.
- `--probe` now reports image, descriptor, allocation, heap-budget, format,
  subgroup, timestamp, and upload capabilities. `--gpu-mem MB` can lower the
  renderer's automatically derived device-local allocation ceiling.
- All renderer-owned `VkDeviceMemory` allocations are counted and rejected
  before crossing the allocation-count, per-allocation, or device-local budget
  limit. Non-device-local staging memory is not charged to the GPU heap.
- Slab/vslab images use a device-memory arena instead of one allocation per
  tile. The arena grows in 64 MiB blocks (or the image size for larger images),
  saving 192 MiB of needless commitment for small slabs compared with the
  original 256 MiB quantum while retaining low allocation counts.
- Fixed-size nonuniform descriptor arrays are enabled only when sampler,
  sampled-image, and total stage-resource limits all fit the 1,024 tiles plus
  the renderer's other bindings. Cube and brick pipelines remain usable on
  limited devices; tiled modes fail with a direct capability message.
- R8 sampled linear filtering is checked at renderer startup. A monolithic
  volume falls back to one mip when linear blits are unavailable. Brick mode,
  which cannot operate without storage and blit support, reports that exact
  requirement rather than failing later during image creation.

On the measured system, `--probe` reports an 8.00 GiB device-local heap, a
7.61 GiB driver budget, a 6.61 GiB renderer ceiling, 1,048,576 combined-image
descriptors, 64-byte UBO alignment, host image copy, and all required R8
features.

### Upload and synchronization paths

- Bulk cube upload keeps the measured-fast staged path, reusing a buffer no
  larger than 128 MiB. `R3D_STAGING=0` still forces host image copy for A/B
  tests.
- Foreground slab and clip streaming prefer host image copy but now have a
  reusable staged fallback. `R3D_NO_HOST_COPY=1` forces it for conformance
  testing. A real 3072^2 slab successfully streamed through this path. The
  background virtual-slab worker always uses queue-ordered staging: host image
  copy cannot be synchronized against newer render submissions sampling the
  same image without blocking the render thread.
- One-shot initialization/upload work waits on a fence rather than idling the
  entire queue. Every submit/present/queue-idle call is externally synchronized
  through the shared queue wrapper because render and streaming workers use the
  same Vulkan queue.
- Device-wide waits remain only in infrequent lifecycle operations such as
  resize, transfer-function replacement, and screenshots.

The portable fallback is correct but not yet fast: a six-slice 3072^2 whole-
window jump measured roughly 47-52 ms. It is a survival path, not the preferred
high-frequency streaming path.

### Asynchronous c5d brick streaming

The render thread now performs only visibility/LRU selection and page-table
handoff. A persistent worker waits for safe timeline points, GPU-decodes a
batch, builds its slot mips and occupancy, then makes it publishable. Evicted
and completed page entries are updated only after draining frames which could
read the mapped buffer. Read-to-write and write-to-read image barriers cover
the atlas and occupancy halo. Vulkan queue use is mutex-serialized.

With a generated 384^3 / 27-brick shard and a 2^3-slot hot atlas, a release run
decoded eight bricks in two jobs averaging 89.43 ms/job. The render loop's CPU
frame maximum was 4.42 ms, so the decode-sized hitch is gone. JSON and the GUI
expose decoded bricks, job count/time, failures, warm bytes, hot residency, and
in-flight work.

### Shard and remote-data safety

- Shard fields are decoded explicitly as little-endian values. Payload bounds
  use overflow-safe arithmetic and exclude the footer. Region dimensions and
  multiplications are checked.
- The 4 MiB index footer is CRC32C-validated. Results are cached by device,
  inode, size, mtime, and ctime so 16 region workers do not hash the same index
  16 times. Atomic download rename naturally invalidates the cache key.
- Missing shards remain sparse zero-fill; corrupt indexes, entries, or payloads
  are distinct hard errors and are never silently converted into missing data.
- Remote virtual-slab fetch no longer builds a shell command. Six libcurl
  workers use bounded timeouts and retries, validate the completed shard, and
  atomically rename it. Only authoritative HTTP 404/410 responses create a
  persistent `.missing` marker. Transient failures remain retryable.
- A remote window can span any safely allocatable number of shards; the old
  silent 64-shard truncation is removed.

The new safety test covers valid all-missing indexes, bad footer CRC, an
overflowing entry, corrupt-region error propagation, and destination-size
overflow.

### Benchmarking and optional quality tradeoff

`--warmup`, `--bench-json`, and `--bench-name` make the benchmark protocol
machine-readable and reproducible. The JSON schema stores the c5d revision,
quality policy, retained sample count, streaming state, and all timing zones.
The in-memory statistics ring grew to 4,096 samples and reports actual p50,
p95, p99, and maximum rather than labelling a maximum as p99.

Three quality policies are explicit:

- `full`: six-tap central gradient, full resolution at all times.
- `interactive` (default): the same six-tap output when still, half-resolution
  while moving.
- `fast`: four-tap tetrahedral gradient and half-resolution movement.

At a dense static view, `fast` changes raycast mean from 6.141 to 5.539 ms
(-9.8%). It is opt-in because real papyrus is noisy: at 640x360, full-vs-fast
real-slab difference is 3.248 LSB mean, p99 56, maximum 178, with 13.25% of
subpixels over 4 LSB and 6.94% over 16. Synthetic data is much closer (0.996
LSB mean). The default image therefore remains unchanged.

## Validation completed

- Warnings-as-errors dev and release builds
- Clean configure using the default fetched/pinned c5d dependency
- ASan+UBSan: CPU, shard-safety, GPU-reference, and c5d GPU conformance tests
- GPU conformance: full render against CPU reference passes
- c5d GPU conformance: exact/one-LSB tolerance passes
- TSan: CPU and shard-safety suites pass; a full SDL desktop run is not usable
  as a TSan oracle on this system because unsanitized GLib/GIO/glycin desktop
  workers emit many third-party races and eventually crash under TSan
- `R3D_NO_HOST_COPY=1` real-slab fallback run passes
- Queue-ordered virtual-slab worker run passes with remote fetch disabled
- Benchmark JSON parses and retains all 1,200 measured samples per scenario
- Vulkan validation was requested, but the Khronos validation layer is not
  installed on this host; GPU conformance ran without that layer

The data-dependent shard integration test is skipped when the optional local
`band/` corpus is absent; the synthetic safety suite still runs.

## Residual bottlenecks and next implementation order

These are not hidden under the completed work. They are the next measured
targets, in priority order.

1. **Asynchronous slab upload and batching.** A real 3072^2 one-slice scroll is
   about 11.05 ms average / 14.1 ms p95 on the CPU. The 1,200-frame z-sweep has
   a 254 ms CPU maximum at a whole-window refill even though GPU raycast p99 is
   only 0.534 ms. Move slice assembly and upload to a persistent producer,
   batch all tile copies into one command buffer, and publish ring layers by a
   timeline value. Acceptance: one-slice p95 below 4 ms and no refill over one
   16.7 ms frame on the measured 3072^2 corpus.
2. **Persistent shard decode pool and decoded-chunk cache.** Region calls still
   create up to 16 pthreads and can decode the same L0 chunks again for adjacent
   slices. Keep workers alive and cache a bounded number of decoded 16^3
   chunks keyed by shard inode/chunk coordinate. Acceptance: at least 2x lower
   p95 slice-production time with a fixed memory ceiling and identical bytes.
3. **Coarse brick fallback for hot-set overflow.** The hot atlas correctly
   avoids thrashing when every slot is wanted, but missing fine bricks render
   empty until decoded. Add a coarse always-resident atlas or hierarchical page
   entry. Acceptance: no empty holes during a camera jump and bounded uploads
   independent of visible fine-brick count.
4. **Queue-side or double-buffered page publication.** The current synchronized
   mapped page table removes the 89 ms decode hitch but can wait up to the
   normal frames-in-flight latency at eviction/publication. Publish via an
   ordered GPU copy or per-frame page buffers. Acceptance: brick handoff adds
   less than 0.25 ms to CPU p99 at 1080p.
5. **Remote request lifecycle.** Add cancellation/prioritization when the camera
   moves away, a bounded disk-cache policy, and a test HTTP server covering
   retry/404/corrupt/oversized responses. Current fetches are safe but cannot
   cancel an obsolete six-request batch.
6. **Cross-device qualification.** Re-run the JSON suite and forced fallback on
   Adreno/Turnip and on a device below the 1,024-descriptor threshold. The
   capability paths are implemented but only the RTX path was available here.

Run the standard benchmark with:

```sh
cmake --preset release && cmake --build --preset release
RESULT_DIR=bench-results/final tools/perf.sh ./build/release/render3d volume.u8 1024
```

Use `R3D_NO_HOST_COPY=1` for the portable streaming fallback and
`R3D_STAGING=0` for the bulk host-image-copy A/B.
