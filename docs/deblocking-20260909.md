# CPU display deblocking — 2026-09-09

CT blocks are filtered before atlas upload by default. `R3D_DEBLOCK=0` disables
it for raw display comparisons. No shader decoding was added.

The implementation applies volume-compressor's quantizer-gated four-tap filter
in X/Y/Z order. A 20³ scratch region contains the requested 16³ block and a
two-voxel halo. Only adjacent 16³ blocks are decoded, through a bounded 32 MiB
raw-neighbor cache. Filtering the halo as well preserves upstream's corner and
edge results. The quantizer comes from the actual compressed stream header.
Byte parity with upstream assumes the level uses a consistent quantizer.

The neighborhood loader reads existing shards and local `.volc` files. It
never initiates network requests. Unavailable neighbors leave their seams
unfiltered; resident blocks with incomplete neighborhoods are retried from raw
source data, up to 32 blocks per second after new visible requests. Thus late
arrivals cannot cause repeated smoothing of previously filtered voxels.
Fully filtered coarsest levels persist separately in `seed-deblock.raw`;
incomplete neighborhoods are not written into that persistent cache. Mips and
occupancy are generated from the filtered uploads.

This changes CT display values. Raw CPU sampling, registration inputs, class
labels and prediction overlays retain their original values. Disable the
filter when comparing raw registration inputs against CT display values.

## Validation

- Release and AddressSanitizer: 35 passed, one fixture-dependent shard test skipped.
- Upstream byte comparison at five quantizers, smooth discontinuities, strong
  edges and noisy patterns; includes every block in a 160×48×48 region, outer
  faces, shared corners, and the 128-voxel chunk seam.
- Missing-neighbor test followed by repair from fresh raw data.
- Renderer test confirms changed seam pixels with filtering enabled, and exact
  image equality between first filtered decode and filtered seed-cache reload.
- Raw registration conformance runs explicitly with display deblocking disabled.

PHerc1667 (5.29 billion virtual blocks), 1280×720 cached static view, five-second
runs on this Mac, no viewer or test suite running during the on/off comparison:

| Measurement | Off | On |
|---|---:|---:|
| Cached startup | 983 ms | 985 ms |
| Mean CPU frame | 1.43 ms | 1.38 ms |
| Streaming worker time | 24 ms | 118 ms |
| Decoded/uploaded blocks, including repairs | 2,413 | 2,519 |
| Decode failures | 0 | 0 |

The similar frame times are a smoke-test result, not evidence of a speedup.
Filtering consumes additional worker time while streaming. The first filtered
startup took 9.7 seconds while other tests were running; subsequent launches
reuse the filtered seed cache. Network acquisition and navigation workloads
are not measured by this static comparison. Raw logs/JSON: `build/deblock-*`.
