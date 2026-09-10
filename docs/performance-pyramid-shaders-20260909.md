# Pyramid shaders and CPU halo copies — 2026-09-09

Pyramid volumes now use dedicated full-quality and fast-gradient shaders. The
compiler knows that these draws use the pyramid page table and can remove the
standalone-shard sampling branches. Sampling distances, gradient taps and
quality settings are unchanged. Other volume layouts retain their existing
pipelines. `R3D_NO_LOD_PIPELINE=1` forces the generic pipeline for comparisons.

Smaller workgroups were tested first and rejected: median raycasting times
were 10.245 ms at 16×8, 10.249 ms at 8×8, and 10.968 ms at 8×4. Those experimental
variants are not included in the source build configuration.

## GPU results

Alternating generic/specialized runs on the M4, PHerc1667, 1920×1080, full
quality, pane cache disabled, 150 warmup frames and 400 measured frames.
Both pipelines include the fine-block cache correction below. Medians across
three runs of each pipeline:

| Measurement | Generic | Specialized |
|---|---:|---:|
| CPU frame time | 11.092 ms | 9.091 ms |
| GPU raycast time | 9.350 ms | 7.481 ms |

End-to-end frame time decreased 18%. GPU timestamp intervals can vary with
query completion and scheduling, so the CPU frame measurement is the more
conservative overall result. All runs had zero decode failures. Final images
differed in 205–311 of 6,220,800 channel bytes, with a maximum difference of two
intensity levels and mean absolute differences below 0.00006. No sampling or
quality reduction was introduced.

A separate scripted multiview navigation workload used the real PHerc1218
cached overview at 1920×1080, full quality, the same frame counts, and two runs
per pipeline. Median CPU frame time fell from 2.798 to 2.390 ms (15%), and GPU
raycast time from 2.703 to 2.291 ms. All runs had zero decode failures.

## CPU deblocking

The halo callback now requests only the needed rectangular slices of adjacent
16³ blocks. Interior blocks copy 3,904 neighbor bytes instead of 106,496.
Decoded cache entries remain 16³. CPU region reads combine contiguous rows
into plane/block copies, and the deblocking passes stop updating halo values
that later axes cannot use. The filter still matches upstream at faces,
corners, missing neighbors and partial volume edges.

A real PHerc1667 CPU-cache benchmark repeatedly filtered 64 blocks at level 5
(6,400 filters per measured pass). Median warm time across five passes fell
from 158.182 to 67.176 ms, a 58% reduction. Checksums were identical. This is a
warm neighborhood benchmark, not total volume-load time.

Two alternating complete coarsest-level loads, using isolated manifests and
symlinks to the existing real shards with no seed cache or network source,
reduced median startup from 7.415 to 7.061 seconds (about 5%). The entire
filtered seed file was byte-identical in every run:
`2683a4f8307150af6117cb1c5c2ac020d50965e63cfd9e62e1005b0fa1ed7dc2`.

The region-copy work also fixed a 32-bit overflow in block-grid clipping near
UINT32_MAX. Bounds now use the checked grid dimensions from volume opening.

## Fine-block cache correctness

The ray and flattened-surface lookup caches now expire when a sample crosses
the requested 16³ cell, even if the previous lookup selected a larger coarse
fallback. Previously that fallback could hide an already-resident fine block
in the next cell. Empty-space jumps are likewise limited to the requested
cell so they cannot jump over fine data that has not been probed.

New regressions place a fine block with intensity 200 behind a coarse fallback
with intensity 40, and repeat with an empty coarse level. They cover ordinary
rays, threshold skipping and flattened-surface baking. The old ray cache
returned 40; an independent old-surface-shader negative control also returned
40. Corrected renders return 200 in all three paths.

## Checks and artifacts

Release and AddressSanitizer: 39 passed, one fixture-dependent shard test
skipped. New tests compare generic and specialized full/fast renders and
exercise region reads at the maximum axis boundary. Existing renderer,
streaming, deblocking, partial-edge and dataset-switch tests pass.

Raw reports and benchmark harnesses are under `build/gpu-next/` and
`build/halo-next/`; the final CPU load report is `cold-final.log` and the final
warm halo report is `after-final.log`. Final GPU reports are
`build/gpu-next/cache-fixed/lod-results.json` and `navigation-results.json`
in the same directory; image comparisons are in `image-parity.log`. Earlier
reports in the parent directory predate the fine-block cache correction and
are not the final performance numbers.
