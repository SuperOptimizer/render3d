# Large-volume support — 2026-09-09

PHerc1667 previously failed during a browser switch because its 5,289,749,386
virtual 16³ blocks exceeded the renderer's 32-bit address space. Logical IDs,
LOD offsets, resident maps, worker jobs, seed records, and CPU decoded-cache
keys now use 64 bits. Shaders calculate and hash paired 32-bit words; MoltenVK
does not need shaderInt64. Physical atlas slots remain 32-bit and bounded.

Source availability is now a bounded hint cache (at most 4 MiB per source),
with persistent compressed files authoritative after eviction. Fetch queue
coordinates use linear cell IDs instead of three truncated 16-bit axes.
CPU decoding and GPU residency remain 16³-granular. Seed-cache format version 3
invalidates records with the old ID layout.

Browser selections are validated before replacing the current dataset. An
invalid selection reports an error while preserving the current image and
navigation. A later initialization failure attempts to reopen the prior dataset.

## Validation

- Release: 33 passed, one fixture-dependent shard test skipped.
- AddressSanitizer: 33 passed, the same fixture-dependent test skipped.
- GPU hash tests cover high words, low-word aliases, coordinate multiplication,
  carries, deletion and reuse. A sparse PHerc1667-shaped fixture renders an
  actual L0 block whose logical ID exceeds 32 bits.
- CPU regression samples distinct blocks beyond the old 20-bit axis boundary.
- Failed-switch regression compares the output image against an unchanged run.
- Real PHerc1218 → PHerc1667 headless browser-switch path: successful, zero
  decode failures. The new seed cache reloads successfully.
- Real PHerc1667, 20276×22122×42209, six pyramid levels: five-second cached
  1280×720 static-view run completed with zero decode failures. Startup was
  1.47 seconds, 2,413 blocks decoded during measurement, 90,230 resident blocks
  out of 148,877 slots. Mean CPU frame time was 1.48 ms. This is a cached static
  view smoke test, not a navigation or cold-download benchmark; ASan tests were
  running concurrently. Logs and JSON are in build/wide-*.

## Remaining bounds

This removes the 32-bit total-block limit, not all resource limits. Axes are
32-bit, reader metadata is capped at four million shard entries, and the
coarsest pyramid must be small enough for practical startup and residency.
Coarsest levels above UINT32_MAX blocks are rejected explicitly. Total logical
counts are checked for 64-bit overflow before allocation. Very large datasets
still need suitable pyramid levels; the full source is never decoded into RAM.
