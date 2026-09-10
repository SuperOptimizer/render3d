# macOS performance measurements — 2026-09-09

The 16³ CPU decoder benefits from reusing parsed entropy tables, and the
renderer was admitting far too few blocks per streaming update. Both are
improved in the working tree. These measurements do not justify a native
Metal rewrite: the common views run comfortably within a 16.7 ms frame,
although heavy views still have slower tail frames.

## Method

Apple M4, 16 GiB unified memory, macOS 26.3, Homebrew clang 22.1.8,
MoltenVK. Baseline: commit `3fc0623`. volume-compressor remains pinned to
`70320c58c922c6c3bcd62f9c209eeb3fa3a332b8`.

No real dataset was available locally. The fixture is a deterministic 512³
textured volume (waves and blobs), with 512³ and 256³ LODs, encoded at q=2.
It has 32,768 fine blocks and 4,096 fallback blocks. All compressed input is
local; network latency and transcoding are excluded. Repeated renderer
launches reuse the on-disk fallback seed and OS file cache. This is a cold
GPU residency test, not a cold disk test.

Renderer runs: 1920×1080, headless, uncapped, `--quality full --tf 1`,
400 warmup frames followed by 1,200 measured frames. `--pool 36 --warm 64`;
the renderer caps this dataset's atlas to 34³ slots. Three repeats per
variant, alternating variant order. The table uses the median of run means
and median of run p99s; memory is the maximum process footprint reported by
macOS `/usr/bin/time -l`, including GPU allocations.

Multiview uses `R3D_MV_EXERCISE=1 R3D_MV_FIT=512` and no loaded segment:
three volume planes and an empty flattened pane. It is not a benchmark of a
real traced surface, ink inference, or registration.

## Results

| Workload | Mean frame before → after | p99 before → after | Fine blocks decoded before → after |
|---|---:|---:|---:|
| Static volume | 2.126 → 2.095 ms | 2.750 → 2.665 ms | 9,138 → 32,768 |
| Fly-through | 5.867 → 5.933 ms | 18.375 → 20.750 ms | 5,844 → 30,044 |
| Moving multiview | 5.632 → 5.314 ms | 23.419 → 23.351 ms | 2,934 → 6,690 |

Streaming counters include warmup, while frame timings exclude it. More
resident fine data changes what the renderer samples, so these are not
identical-work FPS comparisons. The important improvement is faster detail
arrival: the static test now loads the entire fine level in the run, and
motion loads 2.3–5.1× as many fine blocks. No decode failures occurred.
Fly-through mean time is about 1% higher and its p99 is 13% higher with the
larger batches and substantially more fine data. This tradeoff is explicit;
the change is not a blanket frame-rate speedup.

Peak footprint was approximately 433 MiB for volume views and 1,457 MiB
for multiview, essentially unchanged by the changes. The multiview log
identifies a 2048×2048×128 RG8 flattened-surface window: **1 GiB**, allocated
even with an empty segment. This is separate from block-cache residency.

### Decoder

A repeated traversal of the 512 blocks in one textured 128³ chunk measured:

- Original public block decode / old wrapper: **24.5 µs per 16³ block**.
- Parsed-header reuse: **6.0 µs per block**, approximately **4.1× faster**.
- Parsing alone: approximately 21 µs per call in the isolated diagnostic.
- Public whole-chunk decode: approximately 1.65 ms for 128³.

Microbenchmarks were separate from renderer runs. Single-block random access
still has substream entropy traversal cost; reconstructing all 512 blocks
separately is not as efficient as whole-chunk decode. The renderer retains
sparse reconstruction rather than spending RAM on unused voxels.

The cache retains one header and its entropy tables per thread (~43 KiB),
compares actual header bytes and encoded length, and rebinds the payload on
every call. It never retains caller-owned buffers or decoded chunk voxels.
Alternating unrelated headers can miss this one-entry cache. This adapter
uses internal functions of the pinned volume-compressor header; codec pin
updates must run the public-decoder equivalence tests.

Tables alone did not materially improve frame times with the old tiny
batches: GPU work and per-update overhead dominated. The streaming budgets
are now 64 blocks while moving and 128 when settled (256/512 KiB of CT
uploads), replacing 2–8 blocks per update. Existing atlas capacity and
wanted-block checks still bound residency and prevent unwanted evictions.

## Remaining priorities

1. Make the flattened-surface window allocate on demand and size it to the
   useful viewport/workload. The current empty-segment allocation is the
   clearest memory waste observed.
2. Profile ray marching and surface-window updates on real CT and a real
   segment before changing quality policy. Heavy full-quality views still
   exceed 16.7 ms at p99; faster CPU decoding does not remove that GPU cost.
3. Keep MoltenVK for now. A renderer rewrite would need measured evidence of
   translation overhead beyond the shader and cache work already visible.

## Reproduce and validate

```sh
cmake --build --preset macos
python3 tools/bench_macos.py
# Optional comparison against a saved previous binary:
python3 tools/bench_macos.py --baseline /path/to/previous/render3d
ctest --preset macos -L quick
ctest --preset macos -L gpu
```

The benchmark tool creates a fresh synthetic tree, records binary SHA-256s,
runs public-vs-cached block decoding, and saves per-run JSON and process
memory logs below `build/bench-<timestamp>`. A baseline binary must retain
access to its compiled shader directory. Summary measurements from this
session are in [benchmarks/macos-20260909.json](benchmarks/macos-20260909.json);
raw logs and the three tested binaries are in `build/bench-macos/` locally.

Correctness validation: all nine CPU tests and all eight rendering tests
pass. The block test compares all 512 blocks against the public full decoder
and checks copied buffers, mutated headers/payloads, truncation, and cache
bounds. Sanitizer validation covers blocks, persistence and headless APIs.
