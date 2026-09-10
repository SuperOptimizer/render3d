# Packed seed uploads — 2026-09-09

Cached PHerc1667 startup improved from **971 ms to 348 ms (64%)** in five
alternating before/after pairs on the M4. This measures fresh processes
reopening local compressed data with an existing decoded seed, not downloading
or rebuilding that seed.

## Change

The warm loader previously submitted one image-copy region per 16³ seed block.
It now packs consecutive blocks into atlas rows and submits one region per row.
For this dataset's 126,400 blocks and 53-slot atlas width, this reduces copy
regions from 126,400 to 2,869. The 256-block batch size and staging-reuse waits
remain unchanged. Reducing per-region driver work was enough to produce a
large gain without adding concurrent upload buffers.

Temporary scratch is bounded at **1 MiB**, freed when the seed load finishes.
If allocation fails, the loader uses the existing upload path. Post-filtered
uploads retain their original processing path. Decoding, 16³ block residency,
seed file format, and normal demand streaming are unchanged.

## Measurements

Real PHerc1667; isolated temporary dataset directories, symlinked compressed
files and existing filtered seed, no network source, 64×64, pool 32, one
headless frame, deblocking enabled. Five alternating pairs with
`R3D_TRACE_STARTUP=1`:

| Median | Before | After |
|---|---:|---:|
| Total startup | 971.402 ms | 347.577 ms |
| Seed packing/submission/wait interval | 814.034 ms | 203.679 ms |

The second row includes host work and waiting; it is not GPU execution time.
Both paths upload the same 517,734,400 voxel bytes. This improves cached
startup, not cold decoding or frame rate.

## Correctness

Before/after 640×480 screenshots of the real dataset were byte-identical at
two view scales. SHA-256:

- Fit 512: `1ab117019a1ab411250fdc23a50107fbdc1e37679e4c2d625cabdb55f9319e31`.
- Fit 2048: `47a79f7512e15dae9d3c2ffad287b80e6fa416e1d15fdbdd69f9a7b700d69021`.

The seed reload regression now uses a 31-slot atlas width and 31×32×25
coarsest blocks, with partial volume edges and a partial final upload batch.
It compares screenshots from cold decoding, cached reopening, and recovery
from a truncated seed. This exercises row boundaries that do not align with
256-block batches and confirms staging storage is not reused prematurely.

Artifacts are under `build/seed-upload/`: before binary, warm-startup harness,
per-run logs, `warm-results.json`, screenshot comparisons, and test logs.

Release and AddressSanitizer each passed **43 tests**, with the fixture-dependent
shard test skipped (44 total). Logs: `release-tests.log` and `asan-tests.log`.
`git diff --check` passed.
