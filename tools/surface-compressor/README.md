Source snapshot of https://github.com/SuperOptimizer/surface-compressor, version
1.0.0. `snapshot.json` records the upstream commit and per-file SHA256 hashes.
`R3D_SURFCOMP_DIR` can override it for codec development. The default build uses
these committed sources and requires no network fetch.

Files are MIT licensed; see LICENSE. Numeric transforms are floating point.
The renderer uses `-ffast-math -fno-finite-math-only` for the codec, retaining
numeric optimizations while preserving NaN/Inf validation. `src/core/surface.c`
provides a bounded, thread-safe 64x64 XYZ cache and transactional geometry-window
reads. The codec also offers its own generic bounded LRU region-access API.

This snapshot writes container version 4 and reads versions 1 through 4.
Scalar streams share entropy tables across blocks; joint XYZ packets share
inline tables across three planes. No neighboring patch needs decoding.
The codec is synchronous, starts no threads, and exposes no parallel API.
Concurrent calls from an external scheduler may share a reader. New blocks use
a float32 inverse; legacy blocks retain float64 inverse accumulation.
