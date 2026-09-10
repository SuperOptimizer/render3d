# Decode scheduling and frame-wait tracing — 2026-09-09

## Retained optimization

Large decode batches now assign short consecutive runs to each worker. This
keeps neighboring requests on the same thread long enough to reuse entropy
restart state and reduces contention on the shared job cursor. Each requested
block is still decoded and cached independently at 16³ granularity.

The claim size is eight blocks when the batch has at least eight per actual
worker, four when it has at least four per worker, and one otherwise. The
caller counts as a worker. This preserves parallelism for small batches and
respects configured worker counts. It adds no cache memory, performs no extra
voxel decoding, and does not alter filtering or rendering quality.

## Cold-load results

Three alternating before/after pairs, real PHerc1667 on the M4, local compressed
shards, no seed cache or network source, pool 32, deblocking enabled, 64×64,
one headless frame. The before binary includes the previous entropy restart
optimization.

| Startup | Before | After |
|---|---:|---:|
| Run 1 | 3.932 s | 2.723 s |
| Run 2 | 3.759 s | 2.776 s |
| Run 3 | 3.782 s | 2.772 s |
| Median | 3.782 s | 2.772 s |

Median cold startup improved **27%**, about 1.01 seconds. Every generated seed
file was byte-identical, SHA-256
`2683a4f8307150af6117cb1c5c2ac020d50965e63cfd9e62e1005b0fa1ed7dc2`.
This measures rebuilding the decoded fallback from compressed files already
on disk, not WAN downloads or a warm seed-cache reopen.

An initial four-block grouping improved startup substantially; increasing the
large-batch grouping to eight provided a smaller additional gain. The final
policy also scales group size with the actual worker count.

## Streaming and frame waits

Repeated the controlled PHerc1667 fine-arrival workload from
`performance-entropy-streaming-20260909.md`: 1920×1080, full quality, pool 32,
fit 128, scripted pan/zoom/slice movement, 1,600 measured frames, fresh download
cache, two alternating pairs per transport mode. The throttled server adds
75 ms per response and limits each connection to 512 KiB/s. Source coverage is
the locally cached real-data region; unavailable chunks fail rather than
becoming false air.

Each run downloaded 18 L0 chunks, 14 L1 chunks, and six L2 chunks, then decoded
4,324–4,456 requested 16³ blocks during navigation. All eight runs had zero
decode failures. Medians across runs:

| Transport | Mean before / after | p99 before / after |
|---|---:|---:|
| Local unrestricted | 5.238 / 5.235 ms | 7.084 / 6.738 ms |
| Throttled | 5.541 / 5.401 ms | 7.707 / 7.617 ms |

There was no material average-frame regression. The isolated 50 ms wait from
the previous pass did not recur; the worst after-frame was 11.389 ms. This does
not establish that the earlier intermittent stall is fixed.

`R3D_TRACE_WAITS=1` now logs frame waits above 20 ms. It separates the timeline
semaphore wait from timestamp-query retrieval and reports frame, slot,
timeline value, and the completed slot's measured GPU duration. No after-run
in this pass triggered it. This will distinguish the two previously combined
host-side intervals if another long wait appears. The GPU duration belongs to
the completed slot, not the frame being submitted next.

## Rejected experiment

The updated CPU profile still showed entropy-table construction as a cost.
Retaining two parsed headers per decoder thread was tested first. It increased
median cold startup from 3.551 to 3.659 seconds and added memory, so it was
removed. The existing single-header cache and four bounded restart buffers
remain unchanged.

## Validation and artifacts

The streaming shutdown regression now exercises non-divisible batch sizes:
31, 33, 63, 65, 71, 73, 127, and 129. One set settles and checks that every
requested block becomes resident without decode failures. A second set
destroys the renderer while work is pending. Additional runs configure one
and sixteen background decode workers.

Artifacts live under `build/cache-next/`: `sample.txt`, `cold.log` (rejected
header-cache experiment), `group-cold.log`, `group8.log`, `final-cold.log`,
`cold.py`, `streaming.py`, `stream-results.json`, and per-run logs.

Release and AddressSanitizer each passed **43 tests**, with one fixture-dependent
shard test skipped (44 total). The additional one- and sixteen-worker runs
passed. Logs: `release-tests.log`, `asan-tests.log`, `threads1.log`, and
`threads16.log`. `git diff --check` passed.
