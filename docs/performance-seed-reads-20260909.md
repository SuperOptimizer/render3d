# Batched seed reads — 2026-09-09

The cached loader now reads up to 256 consecutive seed blocks with one `fread`,
instead of reading each 4 KiB block separately. This reduces the PHerc1667 seed
load from 126,400 stdio calls to 494. It uses the existing bounded scratch and
staging storage, with no additional allocation, decoded residency, or file
format change. Upload waits and atlas row packing retain their previous behavior.

## Measurement

Nine alternating before/after pairs on the M4, real PHerc1667, fresh processes,
existing filtered seed, isolated temporary dataset directories, no network
source, pool 32, 64×64, one headless frame. Both paths read the same 517,734,400
voxel bytes. The baseline includes packed atlas-row uploads from the prior pass.

| Median | Before | After |
|---|---:|---:|
| Total cached startup | 347.659 ms | 324.651 ms |
| Seed reads and metadata | 84.193 ms | 38.050 ms |
| Packing/submission/waits | 205.571 ms | 213.593 ms |

Total median startup improved **6.6%**. Seven of nine alternating pairs improved.
Two after runs had large timing spikes, including a 579.760 ms total startup;
this is not evidence of improved tail latency. These measurements are cached
reopening, not cold decoding or network download throughput. Component medians
are calculated independently and need not sum to the median total.

## Experiments removed

Reading the next scratch batch while the GPU uploaded staging did not produce
a consistent overall improvement. Changing row-packing loop order also failed
to improve the median. Both experiments were removed. The retained change is
only batched file reading; staging is still drained before it is reused.

## Validation

The existing seed regression exercises partial volume edges, a 31-slot atlas
width, a partial final 256-block batch, and recovery after truncating a cache
that has already supplied multiple upload batches. It compares cold, cached,
and recovered screenshots byte for byte.

Artifacts are under `build/seed-overlap/`. `batch-only-results.json` and
`batch-only.log` contain the retained comparison; `overlap-results.json`,
`row-results.json`, and `batch-overlap-results.json` retain the rejected
experiments. The before binary and benchmark harnesses are saved alongside
per-run traces and validation logs.

Release and AddressSanitizer each passed **43 tests**, with one fixture-dependent
shard test skipped (44 total). Real PHerc1667 screenshots at fit 512 and 2048
were byte-identical to the before binary, retaining the hashes reported in
`performance-seed-upload-20260909.md`. Logs: `release-tests.log`,
`asan-tests.log`, and `pixels.log`. `git diff --check` passed.
