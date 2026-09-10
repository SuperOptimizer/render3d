# Loading, shading, and recovery — 2026-09-09

## Cold loading

A five-second macOS `sample` profile of PHerc1667 startup, with local compressed
shards and no decoded seed file, was dominated by the codec's `vf_decode_sub`
entropy decoding. Deblocking and GPU upload were much smaller contributors.

Block centers now use the existing bounded raw CPU cache used for deblocking
neighbors. This avoids decoding a center again when a previous block's halo
already needed it. Cache entries remain 16³ and the cache stays at 32 MiB;
filtered pixels never enter the raw cache. Partial edge blocks keep the direct
decode path to preserve their encoded padding. Missing cache data falls back
to the supplied compressed block.

Three alternating before/after pairs on the M4, PHerc1667, 64×64, one headless
frame, pool 32, deblocking enabled, isolated manifests linked to existing local
shards, no seed and no network source:

| Startup | Before | After |
|---|---:|---:|
| Run 1 | 7.023 s | 5.539 s |
| Run 2 | 7.025 s | 5.560 s |
| Run 3 | 7.021 s | 5.482 s |
| Median | 7.023 s | 5.539 s |

Startup decreased **21%**, about 1.48 seconds. Every generated seed file was
byte-identical, SHA-256
`2683a4f8307150af6117cb1c5c2ac020d50965e63cfd9e62e1005b0fa1ed7dc2`.
This measures rebuilding a decoded seed from local compressed data, not download
speed or reopening an existing seed. Both binaries used the corrected shaders.

Full-suite testing caught an interaction with alternate manifest filenames:
the renderer used the selected file but the CPU deblocking reader opened the
directory's `manifest.json`. The CPU reader now accepts an explicit manifest
path and the renderer supplies it. Compact JSON format declarations are also
accepted. The existing far-block test with over five billion virtual blocks
caught the regression and passes after this correction.

## Perspective shading

A lookup-cache hit previously retained the gradient radius from the first
sample in a requested cell. Perspective footprint changes continuously with
depth even when integer LOD and cell coordinates stay unchanged. Cache hits
now refresh that radius.

A GPU regression compares perspective shading against shaders that perform
an uncached page lookup at every sample, for both full and fast quality.
Corrected renders are byte-identical. An isolated old shader fails the
regression: its full-quality render differs by 1,984 total channel levels.
The uncached variants are reference shaders and are never selected by the app.

## Navigation and streaming

1920×1080 full quality, scripted pan/zoom/slice movement, 150 warmup frames,
1,600 measured frames, two alternating pairs per case. Values below are
medians across runs; maximum is the worst frame observed across both runs.

| Case | Mean before / after | p99 before / after | Maximum before / after |
|---|---:|---:|---:|
| PHerc1218 overview, pool 32 | 2.561 / 2.556 ms | 5.292 / 5.289 ms | 11.118 / 12.453 ms |
| PHerc1218 overview, pool 8, closer zoom | 2.911 / 2.911 ms | 5.943 / 5.755 ms | 12.014 / 8.585 ms |
| PHerc1667 cached fallback | 5.979 / 5.979 ms | 7.702 / 7.659 ms | 9.008 / 8.300 ms |

All twelve runs had zero decode failures. The overview workloads decoded
thousands of arriving blocks while navigating; pool 8 deliberately introduces
residency pressure. No material average frame-time improvement or regression
was measured. Maxima vary, so these results do not establish a reliable
worst-frame improvement.

The large-volume case used an isolated local source with the coarse fallback
resident; its requested finer data was unavailable and no blocks were decoded
during measurement. It verifies navigation with huge logical dimensions, not
fine-level streaming throughput or slow-network responsiveness.

## Failure recovery

Native downloads now validate the codec header, entropy tables, and substream
bounds before publication, without decoding a whole chunk. Existing malformed
or truncated native downloads can be replaced. CPU decode failures evict the
matching downloaded file and cached compressed bytes, allowing normal demand
retry; immutable source shards are not removed. The renderer validates native
cache files before publishing their availability and clears stale availability
hints on rejected reads. Cached reopen remains distinct from a new transfer in
network counters.

Regression coverage includes:

- Disconnect halfway through a payload, then retry into the same directory.
- Malformed payloads, invalid range responses, CRC/index errors, and no partial
  cache artifact left after rejection.
- Damaged local native downloads repaired by CPU demand loading and by the
  renderer with deblocking both off and on; the renderer verifies the recovered
  fine-block intensity against the coarse fallback.
- A seed truncated halfway through its payload, forcing re-decode and an exact
  screenshot match with the original load.
- Rejected dataset selection preserving the current image, and a successful
  switch matching a fresh open of the destination dataset.
- Eight destroy/reopen cycles immediately after submitting decode jobs,
  exercising shutdown's joins and cache/staging lifetimes.

Header validation is not a checksum of every decoded voxel. It rejects malformed
structure and truncation; arbitrary bit corruption that remains a valid stream
is not guaranteed to be detected.

## Artifacts and validation

Harnesses and results are in `build/followup/`: `cold-sample.txt`,
`cold-final.py`, `cold-final.log`, `navigation.py`, `navigation-results.json`,
`gradient-pass.log`, and `gradient-negative.log`. Earlier exploratory cold
results overlap compilation and are not the reported startup numbers.
Final full-suite logs are `release-final.log` and `asan-final.log`.

Final Release and AddressSanitizer suites each passed **42 tests**, with one
fixture-dependent shard test skipped (43 total). `git diff --check` passed.
