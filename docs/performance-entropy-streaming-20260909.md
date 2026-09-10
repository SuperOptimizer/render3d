# Entropy restarts and fine-detail streaming — 2026-09-09

## Decoder changes

The pinned volume-compressor stores groups of sixteen 16³ blocks in an entropy
substream. Its individual-block API walks earlier tokens again when another
block from the same substream is requested. It also dequantizes coefficients
for blocks whose voxel output will be discarded.

The render3d adapter now saves entropy restart positions at block boundaries
and only dequantizes/inverse-transforms the requested block. Each decoder
thread retains four compressed substreams, up to 64 KiB each, plus their
restart state (about 263 KiB of additional state per thread). Larger substreams
use the same selective decoder without saved checkpoints. The decoded voxel
cache remains 16³ and keeps its existing budget.

Checkpoints own their compressed bytes. Header and substream byte comparisons
invalidate state after source changes, even when allocations or mappings reuse
an address. No pointer into a temporary download or caller-owned mapping is
retained by the checkpoints. The adapter follows the pinned codec's token
validation and uses its inverse transform. Its derived code includes the
upstream MIT notice.

The regression compares output and error status with the public block decoder
for constant, smooth, and noisy chunks; forward, reverse, and random requests;
concurrent readers; mutations inside a warmed substream; truncation; and an
oversized substream that must not enter the bounded checkpoint storage.
Existing full-chunk/block and real-data seed comparisons also pass.

## Cold startup

Three alternating before/after pairs on the M4, real PHerc1667, local compressed
shards, no seed cache, no network source, pool 32, deblocking enabled, 64×64,
one headless frame:

| Startup | Before | After |
|---|---:|---:|
| Run 1 | 6.701 s | 3.656 s |
| Run 2 | 5.489 s | 3.778 s |
| Run 3 | 5.568 s | 3.786 s |
| Median | 5.568 s | 3.778 s |

Median startup fell **32%**, about 1.79 seconds, relative to the previous
optimization pass. Every generated seed file was byte-identical, SHA-256
`2683a4f8307150af6117cb1c5c2ac020d50965e63cfd9e62e1005b0fa1ed7dc2`.

A single 16,384-block sequential microbenchmark on a real coarse chunk measured
4.474 µs/block before and 1.946 µs/block after, with matching checksums. This
illustrates locality benefits; random requests and full application workloads
will have different gains.

## Real fine-detail arrivals

The earlier large-volume benchmark had no arriving fine data. This pass served
existing real PHerc1667 compressed chunks through a local indexed-Zarr HTTP
server, starting each run with a fresh download cache and the same resident
coarse seed. The server represents only the locally available source region.
Unknown chunks deliberately fail bounds validation rather than becoming false
air. These are controlled local transport measurements, not public-server WAN
measurements or a complete-volume bandwidth test.

Each run downloaded 18 nonempty L0 chunks, 14 L1 chunks, and six L2 chunks,
about 1.67 MB of compressed payload in total. Each decoded 4,317–4,425 requested
16³ blocks during navigation. All eight runs had zero decode failures.

1920×1080, full quality, pool 32, scripted pan/zoom/slice movement, fit 128,
1,600 measured frames including initial arrivals, two alternating pairs per
transport condition. Slow transport adds 75 ms per HTTP response and limits
each connection to 512 KiB/s. Values are medians across runs:

| Transport | Mean before / after | p99 before / after |
|---|---:|---:|
| Local unrestricted | 5.253 / 5.197 ms | 8.033 / 7.279 ms |
| Added latency and bandwidth limit | 5.346 / 5.175 ms | 8.189 / 7.043 ms |

These are modest navigation differences; the strongest measured gain remains
cold decoding. More requests can finish within a run as worker timing changes,
so this is not an exact fixed-work CPU decode comparison.

One unrestricted after-run contained a **49.8 ms frame**, dominated by a
49.5 ms GPU-completion wait. Its maximum main-thread streaming phase was only
2.6 ms. The other after-runs had maxima between 8.0 and 8.6 ms. The isolated
wait remains unresolved; these results do not establish that all navigation
stalls are gone. The GUI viewer was not stopped for benchmarking; other application activity
was not controlled.

## Cancellation correctness

The curl cancellation callback read the renderer's network shutdown flag
without the mutex used by its writer. That flag is now atomic.

The network regression waits until an HTTP server has begun a payload and then
stalls that payload. It shuts down the renderer while the transfer remains
blocked, requires shutdown within two seconds, and verifies that no downloaded
cache file was published. Observed shutdown was about 1.0 second, consistent
with curl's progress-callback cadence. Existing active-decode shutdown and
successful/rejected dataset-switch tests remain in the full suite.

## Artifacts

`build/entropy/` contains `cold.py`, `cold.log`, `micro-before.json`,
`micro-after.json`, `streaming.py`, `stream-results.json`, and per-run logs.
The streaming harness constructs indexed shards from already-cached real
payloads and removes its temporary dataset caches after each run.

Release and AddressSanitizer full suites each passed **43 tests**, with one
fixture-dependent shard test skipped (44 total). The final oversized-substream
regression also passed separately in both builds. Logs: `release-tests.log`,
`asan-tests.log`, `final-restart.log`, and `final-restart-asan.log`.
`git diff --check` passed.
