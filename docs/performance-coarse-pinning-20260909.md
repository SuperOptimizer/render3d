# Coarse residency and background CPU use — 2026-09-09

Sampling the current PHerc1667 viewer identified `bricks_touch_coarsest` as the
largest active CPU hotspot (3,103 leaf samples in the six-second capture).
Every view collection walked all 146,080 coarse virtual blocks, probing the
resident hash and updating slot stamps simply to protect coarse fallback.

The eviction clock now rejects resident coarsest-level blocks directly by
64-bit logical ID. Empty slots remain reusable, and fine blocks retain their
current-view protection. The repeated coarse-grid walks are removed. This
changes neither decoding, deblocking, mip selection, nor shader quality.

Hidden, minimized and occluded GUI windows also delay 33 ms per loop. On macOS,
these windows can lose presentation backpressure and otherwise consume a CPU
core while invisible. Event handling and background ingest remain active;
visible windows and headless benchmarks are unaffected by this delay.

## Validation

Release and AddressSanitizer: 36 passed, one fixture-dependent shard test
skipped. A new GPU regression streams more fine blocks than the atlas can hold
and verifies that an untouched patch still renders through its coarse fallback.
Existing deblocking, seed reload, dataset switch and GPU update tests pass.

The final isolated benchmark alternated the saved before binary and new binary
for three runs each. PHerc1667, 1280×720, interactive quality, deblocking on,
five seconds per cached static-view run:

| Median across three runs | Before | After |
|---|---:|---:|
| Mean CPU frame time | 1.258 ms | 0.133 ms |
| p99 CPU frame time | 2.408 ms | 0.493 ms |
| Resident blocks at end | 90,231 | 90,231 |
| Decoded/uploaded blocks | 2,413 | 2,413 |
| Decode failures | 0 | 0 |

This is an 89% reduction in cached-view CPU frame time. It is not a claim of
9× faster GPU rendering. A separate 1920×1080 full-quality forced-redraw pair
(400 warmup frames, 1,200 measured frames, pane cache disabled) remained
GPU-bound: mean CPU frame times were 12.44 ms before and 11.75 ms after, with
zero decode failures. Those single-run values are a smoke check, not a robust
GPU speedup measurement. Final redraw images differed in 181 of 6,220,800
pixel-channel bytes; asynchronous seam repair counts also differed by six
blocks. Cached-run screenshots retain frame-dependent ray jitter and are not
byte-comparable across different frame counts.

Artifacts: `build/perf-next/profile.txt`, `compare-final.log`, `results.json`,
`redraw-{before,after}.json`, and the Release/ASan test logs. Earlier comparison
logs include background-viewer contention; the final comparison ran after that
viewer had exited.
