# dct3d

A tiny, dependency-free lossy codec for **16×16×16 voxel chunks**. A 3D float
DCT-II, a dead-zone quantizer, and an adaptive binary range coder, packaged as a
single `.c` / `.h` pair.

Extracted and generalized from the block codec in
[SuperOptimizer/matter-compressor](https://github.com/SuperOptimizer/matter-compressor)
(the Vesuvius Challenge scroll compressor).

- **One dependency:** the C standard library. Nothing else.
- **Seven dtypes:** `u8 u16 u32 s8 s16 s32 f32`.
- **Auto-normalized quality:** one `quality` number means comparable *relative*
  fidelity for any dtype and any data scale — the codec measures each chunk's
  own value range and quantizes relative to it.
- **Two error bounds:** a *relative* `max_error` and an *absolute* `tau`, either
  of which enables a sparse exact-correction pass.
- **Float-only core.** Every value/transform/quant/normalization computation is
  `float`. Integers appear only where they must: inside the range coder (a
  bit-exact arithmetic coder), byte I/O, and loop indices.
- **Reentrant, lock-free, thread-free.** No shared mutable state on the hot path,
  no `pthread`, no globals to guard. Call it from as many of *your own* threads
  as you like; the library starts none.
- **C23, pure scalar, no SIMD** — written to auto-vectorize under `-O2/-O3
  -ffast-math`.

## Build

```sh
cc -std=c23 -O3 -ffast-math -c dct3d.c
```

Just drop `dct3d.c` and `dct3d.h` into your project. No build system, no config.

## API

One encode/decode pair per dtype:

```c
size_t dct3d_encode_u16(const uint16_t *chunk, float quality,
                        float max_error, float tau, uint8_t *out);
int    dct3d_decode_u16(const uint8_t *blob, size_t len, uint16_t *chunk);
```

…and likewise `_u8 _u32 _s8 _s16 _s32 _f32`.

| arg | meaning |
|-----|---------|
| `chunk` | `DCT3D_N3` (=4096) voxels, z-major: index `(z*16 + y)*16 + x` |
| `quality` | quantizer coarseness, `> 0`. Smaller = higher fidelity, bigger blob. Auto-normalized. |
| `max_error` | optional **relative** bound in `[0,1)`: caps per-voxel error at `max_error × chunk_value_range`. `0` disables. |
| `tau` | optional **absolute** bound in raw input units (`tau=2` → within ±2). `0` disables. |
| `out` | output buffer, capacity `>= DCT3D_MAX_BYTES` |

If both `max_error` and `tau` are set, the tighter one wins. `encode` returns the
blob length; `decode` returns `1` on success, `0` on a malformed/truncated blob
(it never reads or writes out of bounds, and rejects a blob decoded with the
wrong-typed function).

## Example

```c
#include "dct3d.h"

uint16_t chunk[DCT3D_N3];              // your 16^3 data
// ... fill chunk ...

uint8_t blob[DCT3D_MAX_BYTES];
size_t  n = dct3d_encode_u16(chunk, /*quality*/ 2.0f,
                             /*max_error*/ 0.0f, /*tau*/ 4.0f, blob);
// n bytes in `blob`; every reconstructed voxel will be within ±4 of the input.

uint16_t back[DCT3D_N3];
if (!dct3d_decode_u16(blob, n, back)) { /* corrupt blob */ }
```

Chunk your own volume however you like (tile a 128³ into 8³ blocks of 16³, etc.)
and call the codec per block — that stays embarrassingly parallel across your
threads.

## Real-world results

Measured on **dense-center scroll chunks** from the Vesuvius Challenge
`PHercParis4` 2.4 µm masked volume
(`20260411134726-2.400um-0.2m-78keV-masked.zarr`), the same data this codec was
tuned on. Six 128³ chunks near the scroll core were pulled from S3, each tiled
into 512 blocks of 16³ (u8). All six are **100 % material** (no air), mean
intensity 60–100, spanning the full 0–255 range.

Averaged over all six chunks (3M voxels), quality swept `1…64`, each alone and
with `tau = 2·quality`:

```
setting        ratio   b/vox   PSNR    MAE    SSIM     err p90/p95/p99/max
------------------------------------------------------------------------
q1              6.4x   0.162  50.14  0.515  0.9997     1 /  2 /  2 /  7
q1  tau=2       5.4x   0.189  50.67  0.483  0.9997     1 /  1 /  2 /  2
q2             10.2x   0.101  47.00  0.814  0.9994     2 /  2 /  3 /  9
q2  tau=4       9.8x   0.105  47.11  0.806  0.9994     2 /  2 /  3 /  4
q4             16.6x   0.061  43.93  1.208  0.9988     3 /  3 /  4 / 14
q4  tau=8      16.5x   0.062  43.95  1.207  0.9988     3 /  3 /  4 /  8
q8             27.4x   0.037  40.84  1.746  0.9975     4 /  5 /  7 / 22
q8  tau=16     27.3x   0.037  40.84  1.746  0.9975     4 /  5 /  7 / 16
q16            45.0x   0.022  37.77  2.486  0.9948     5 /  7 / 10 / 35
q16 tau=32     45.0x   0.022  37.77  2.486  0.9948     5 /  7 / 10 / 30
q32            72.3x   0.014  34.81  3.478  0.9894     7 / 10 / 14 / 50
q32 tau=64     72.3x   0.014  34.81  3.478  0.9894     7 / 10 / 14 / 50
q64           108.5x   0.009  32.00  4.763  0.9793    10 / 14 / 20 / 76
q64 tau=128   108.5x   0.009  32.00  4.763  0.9793    10 / 14 / 20 / 76
```

- **ratio** = raw ÷ compressed; **b/vox** = bytes per voxel (raw is 1.0).
- **PSNR** in dB, **MAE** mean absolute error, **SSIM** global structural
  similarity, and the **p90/p95/p99/max** columns are percentiles of the
  per-voxel absolute error.

### Reading the numbers

- **q1 ≈ visually lossless:** ~6× smaller, 50 dB, SSIM 0.9997, 99 % of voxels
  off by ≤2. **q8 ≈ 27×** still holds 41 dB / 0.997 SSIM.
- **Graceful high end:** q64 reaches ~108× at 32 dB and SSIM still 0.98 — a
  usable preview at ~0.009 bytes/voxel.
- **`tau` does what it says.** The `max` error column collapses to exactly `tau`
  wherever the bound bites (`q1 tau=2` → max 2, `q2 tau=4` → max 4, …). Because
  the correction residuals are coded as small integer deltas (not raw floats),
  the bound is nearly free: `q2 tau=4` costs ~5 % over plain `q2`, and by q≥16
  the natural error is already under `2q`, so `tau=2q` is a no-op (identical
  rows).
- **Two lossless coder wins** (identical PSNR/MAE/SSIM/max): the block header
  (mean + value range) is entropy-coded into the range stream rather than kept
  as a raw 18-byte struct, and the range-coder contexts are seeded with static
  priors trained on this scroll data instead of starting at 50/50. Together they
  add roughly +5–15 % ratio, most at high quality where the header used to
  dominate and the per-block contexts never had time to adapt.

### Cross-dtype sanity

Auto-normalization means the same `quality` gives the same *relative* fidelity
regardless of dtype. Synthetic blocks at `q=1.0`, RMSE as a fraction of the
value range lands around a few tenths of a percent for every type. Absolute
`tau` is honored per-voxel for all dtypes: the correction quantum shrinks with
tau, so e.g. a u16 block spanning ~8500 at `tau=1` reconstructs within ±1, and
f32 at `tau=0.001` within 0.001 (see **Notes & caveats** for the u32/s32 and
f32-precision limits).

## Performance

Single-threaded throughput on an **Apple M4** (Homebrew clang 22,
`-O3 -ffast-math -march=native`), measured over the dense-center scroll blocks
above (u8, one 16³ block = 4096 voxels = 4 KB raw):

```
  q      encode              decode
         MB/s   ns/block      MB/s   ns/block
  ----------------------------------------------
  q1     117     35000        121     33800
  q4     179     22900        252     16300
  q16    229     17900        457      9000
  q64    256     16000        657      6200
```

Throughput rises with `quality` because there are fewer significant coefficients
to range-code. Decode outruns encode at high quality once the sparse-coefficient
path dominates. The per-coefficient step table and the frequency scan order are
pure functions of the geometry, so they are computed once and cached rather than
rebuilt per block — that alone is ~1.8× encode and up to ~3× decode at high q.
Seeding the coder contexts with trained priors also nudges throughput up (fewer
bits coded), so it helps ratio and speed at once.

Because the codec is **thread-free and every block is independent**, it scales
out trivially — the same M4 across its 10 cores (OpenMP, one block per work
item) exceeds **1 GB/s**. There are no locks or shared writable state to contend
on; the only shared reads are the static cosine/step/scan tables.

## Deblocking

At aggressive quality the independent per-block quantization leaves visible
value steps across the 16-voxel block grid. Two complementary tools address it
(both optional; the bitstream never changes):

- **Decode-side post-filter** ([`deblock.h`](deblock.h)) — for data that is
  *already encoded*. A signal-adaptive boundary filter in the H.26x family:
  thresholds scale with the quantizer step, so it closes coding seams while
  leaving genuine edges alone. Needs decoded neighbors, so it is a post-pass,
  not part of `dct3d_decode_*`: either hand it one chunk + its six decoded
  face-neighbors (`dct3d_deblock_chunk_*`, order-independent, parallel-safe) or
  an assembled region in place (`dct3d_deblock_*`). Pass the encode `quality`
  as `step`. Measured on real 2.4 µm scroll CT: at q32 it removes ~70 % of the
  seam excess and *improves* PSNR by ~0.4 dB; at q64, ~98 % and +0.6 dB.

- **Encode-side compensation** ([`predeblock.h`](predeblock.h)) — for *future*
  encodes, when readers should get low-seam data from a plain lone-block
  decode with no post-pass. Blocking is visible because the two sides of a
  seam err independently; `dct3d_encode_deblock_*` injects the neighbor's
  known reconstruction error into this block's boundary layers (tapered
  inward) before the DCT, so both sides err alike and the step vanishes while
  the true gradient is preserved. Encode in checkerboard order (evens plain,
  odds compensated) and every seam is fixed once. ~30 % seam-excess cut alone
  at rate parity; decode stays pure and block-independent.

The two compose (enc-side at write, dec-side at read), but when combined,
lower the decode-side `strength` — each already closes part of the same seam.

**Lapped transforms: evaluated, negative result (for now).**
[`lapped.h/.c`](lapped.h) implements a TDLT-style boundary pre/post filter pair
(Tran/Tu/Liang butterfly, diagonal antisymmetric scaling, separable 3D; perfect
reconstruction verified). On real scroll CT it *loses* to plain coding at every
scale strength tried: the post-filter's 1/s synthesis gain amplifies
quantization noise in the boundary bands (q32: seam energy ×2.2, MSE +54 % at
the classic scales; still worse at near-identity scales), and the expected rate
win never appears because the per-block value-range normalization and trained
coder priors already absorb what the pre-filter decorrelates. A proper
orthogonal LOT (rotation-based, quantizer-matched) might behave differently,
but the bar it must clear — the decode-side filter's −98 % seam excess *plus*
PSNR gain at q64 — is above typical lapped blocking gains. The code stays as a
reproducible experiment (`eval_real.c` has the harness); it is not part of the
format.

## Notes & caveats

- **Lossy.** Reconstruction is close, not bit-exact.
- **Same-build codec.** f32 + fast-math is not bit-reproducible across ISAs / opt
  levels / FMA contraction, so an encoder and decoder compiled together agree
  exactly (and the `tau` bound holds), but a decoder built differently may drift
  by a small delta. If you need a hard cross-machine bound, compile both sides
  the same way (and pin `-ffp-contract=off`).
- **u32 / s32 magnitudes above 2²⁴ lose their low bits.** The pipeline is
  float32 (24-bit mantissa), so a value like 3·10⁹ can't be represented exactly
  *before* any lossy step — a bare `(uint32_t)(float)v` round-trip already
  rounds it. u32/s32 are fully faithful up to ~16.7M in magnitude (which covers
  the usual voxel/label ranges); beyond that, expect an error on the order of
  `magnitude · 2⁻²⁴` regardless of `quality`/`tau`. u8/u16/s8/s16/f32 are
  unaffected. If you need exact large 32-bit integers, offset/scale them into
  range before encoding.
- **`tau` is met to the correction quantum.** The correction pass rounds
  residuals to a quantum that shrinks with `tau`, so the honored per-voxel bound
  is `tau` (not the older `max(tau, 0.5·vspan/255)`) for every dtype — verified
  for u8/u16/u32/f32. For f32, "met" is still subject to the same-build float
  caveat above.
- **Fixed 16³ block.** The transform, scan order, and contexts are specialized to
  `DCT3D_N = 16`. Tile larger volumes yourself.
- **Non-finite f32 input is sanitized, not preserved.** A NaN/Inf voxel is
  replaced with the block's finite minimum at encode (so every *finite* voxel
  still round-trips and the blob is always decodable); the non-finite value
  itself is not recovered.
- **Robust to corruption.** Truncated or bit-flipped blobs are rejected or safely
  bounded — fuzzed with truncations and random bit flips, never crashes.

## Ecosystem

This repo is the C codec plus everything needed to store scroll volumes with it
and read them back through the standard tools:

| Piece | What it is | Where |
|-------|------------|-------|
| **`dct3d.c` / `dct3d.h`** | the codec (this README) | repo root |
| **`dct3d-zarr`** | a **Zarr v3 codec** wrapping dct3d — `pip install` it and stock zarr reads/writes dct3d arrays | [`dct3d_zarr/`](dct3d_zarr/) · [README-zarr.md](README-zarr.md) |
| **`dct3d_export`** | multithreaded S3 → dct3d-Zarr-v3 → SFTP scroll exporter | [`tools/export/`](tools/export/) |
| **`examples/read_slice.py`** | minimal reader: pull a slice through the zarr-python API | [`examples/`](examples/) |

### Install the Python codec

Use a **stable CPython (3.11–3.13)** in a virtualenv (not free-threaded `3.14t` —
numcodecs has no wheels there and will fail to build):

```sh
python3.13 -m venv .venv && source .venv/bin/activate
pip install "zarr>=3" numpy cffi          # cffi compiles the dct3d extension
pip install .                             # installs dct3d-zarr from this repo
```

Prebuilt wheels are produced by CI (`.github/workflows/wheels.yml`); until they
are published to PyPI, install from source as above. Once installed, the `dct3d`
codec auto-registers with zarr via an entry point — a stock `zarr.open(...)` on a
dct3d array just works, no import needed. Details + API in
[README-zarr.md](README-zarr.md).

### Read an exported volume

```sh
pip install "zarr>=3" dct3d-zarr numpy pillow requests "fsspec[http]"
python examples/read_slice.py             # fetch a shard, decode a slice via zarr
```

## License

Derived from matter-compressor; see that project for upstream licensing.
