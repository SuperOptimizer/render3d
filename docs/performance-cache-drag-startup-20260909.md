# Cache batching, drag responsiveness, and warm startup — 2026-09-09

## Retained changes

Display deblocking collects its up to 26 halo rectangles before reading them.
The CPU volume cache copies resident rectangles under one mutex acquisition;
missing regions fall back to the existing status-aware decode path after the
lock is released. Only 3904 halo bytes are requested for an interior block.
Decoded residency remains 16³ / 4 KiB per block. There is no extra decoded cache
or speculative voxel decode; the worker uses approximately 6 KiB more bounded
stack scratch to collect the rectangles. Cache hits preserve eviction safety
and update the unpinned LRU.

Streaming now prioritizes desired detail ahead of intermediate parents. The
coarsest fallback remains pinned. A block requested by multiple panes inherits
the highest requested priority and nearest distance.

After a complete multiview collect, the renderer discards obsolete queued CT
requests and signals obsolete active transfers through worker-owned atomic
cancellation flags. The visibility check examines all candidates, including
those beyond the decode budget. It does not cancel on frames where a busy
decode worker prevented collection. Overlay producers keep their independent
lifetimes. Canceled transfers do not enter the 30-second transport-failure
backoff; returning to a previous view can request them again.

## Cold startup

Real PHerc1667 compressed files already on local disk; fresh temporary dataset,
no seed cache or network source. Deblocking enabled, pool 32, 64×64, one frame.
Three alternating before/after pairs against the prior grouped-decode binary:

| Run | Before | After |
|---|---:|---:|
| 1 | 2.722 s | 2.442 s |
| 2 | 2.690 s | 2.389 s |
| 3 | 2.663 s | 2.512 s |
| Median | **2.690 s** | **2.442 s** |

The median improved **9.2%**. An earlier noisier set measured 2.903 → 2.440 s;
the steadier repeat above is the reported result. Every generated seed in both
sets had SHA-256
`2683a4f8307150af6117cb1c5c2ac020d50965e63cfd9e62e1005b0fa1ed7dc2`.
This is CPU decode/filtering plus atlas initialization, not a download test.

## Time to sharp detail

The native renderer regression uses one fetch worker and a server that holds
the old view's payload until explicitly released. Another old-view chunk is
queued behind it. After dragging, the new view must display its fine value
(100, versus the coarse fallback's 40) before the server releases the old body.
The test also verifies that the queued obsolete payload was never requested.
Returning to the original view must recover without failure backoff.

An initial measured run reached sharp pixels in 15 ms and recovered the old
view in 14 ms. A full-suite run exercised curl's idle callback cadence: new
sharp detail took 1022 ms and returning took 20 ms. In both runs, the new view
became sharp before the old response was released. These are controlled local-server
measurements, not WAN latency predictions; the regression allows up to two
seconds to accommodate idle callbacks and scheduling. The separate LOD regression
uses one upload of budget to verify fine pixels arrive before the intermediate
parent, with coarse and fine values deliberately different.

## Warm startup profile

Five alternating pairs reopen PHerc1667 with its existing filtered seed, in
fresh processes and isolated dataset directories with no network source.
The seed contains 517,734,400 voxel bytes. Median total startup:

- Before: 1006.878 ms.
- After: 1006.312 ms.

There is no material warm-startup change from cache batching: this path skips
decoding. New `R3D_TRACE_STARTUP=1` instrumentation separates renderer creation,
dataset setup, seed uploads/waits, seed reads/metadata, and total startup.
Across the five after runs, median component times were approximately:

- Renderer creation: 63 ms.
- Seed upload and waiting for staging reuse: 839 ms.
- Seed file reads and metadata processing: 96 ms.
- Dataset setup including the seed: 940 ms.

The next substantial warm-startup target is the serialized upload loop: it
uploads at most 256 blocks and waits before refilling the same staging buffer.
Double buffering or larger upload batches should be investigated and measured;
neither was implemented in this pass. The upload measurement includes host
submission work and waiting, so it is not a claim of 839 ms of GPU execution.

## Real-data navigation and frame waits

Repeated the controlled PHerc1667 HTTP workload: 1920×1080, full quality,
pool 32, fit 128, scripted pan/zoom/slice movement, 1600 frames per run.
Two alternating pairs each use unrestricted transport and transport with
75 ms response latency plus 512 KiB/s per connection. Available source coverage
is the cached real-data region; unknown chunks fail instead of becoming air.

| Transport | Mean before / after | p99 before / after |
|---|---:|---:|
| Unrestricted local HTTP | 5.080 / 5.095 ms | 7.245 / 7.177 ms |
| Throttled local HTTP | 5.151 / 5.178 ms | 7.026 / 7.084 ms |

Average frame times are effectively unchanged. Runs decoded 4341–4513 blocks
with zero decode failures. Each downloaded the same 18 L0, 14 L1, and six L2
nonempty chunks, totaling 1,674,639 compressed bytes.

All eight runs enabled `R3D_TRACE_WAITS=1`. No wait exceeded its 20 ms threshold;
the worst after-frame was 12.299 ms. The earlier isolated 50 ms stall did not
recur, so its cause remains unestablished. The tracing still separates semaphore
wait and query retrieval if it returns. These tests cover offscreen rendering,
not windowed presentation.

## Validation and artifacts

Extended CPU tests compare batched regions with full upstream decode under
concurrent eviction from eight cache slots, cross-block/chunk reads, batches
larger than 32, fully resident reads without decoding, and missing/air status.
Existing deblock parity covers chunk seams, block seams, edges, corners,
partial volume boundaries, and late neighbors. Renderer tests cover filtered
seed reload, desired LOD priority, canceled downloads, and streaming teardown.

Artifacts: `build/cache-batch/` contains the before binary, cold/streaming/warm
harnesses, per-run JSON and logs, and validation logs. `cold.log` is the final
cold comparison; `cold-initial.log` retains the earlier set. `warm-results.json`
contains the startup phase traces. `stream-results.json` contains all eight
navigation runs.

Final Release and AddressSanitizer runs each passed **43 tests**, with the
fixture-dependent shard test skipped (44 total). The three-level test helper
now computes each level's correct ceiling-divided dimensions. Final drag
measurements were 14/14 ms (Release) and 13/14 ms (sanitizer), consistent with
the faster callback case; the slower 1022 ms observation remains relevant.
Logs: `release-tests.log` and `asan-tests.log`. `git diff --check` passed.
