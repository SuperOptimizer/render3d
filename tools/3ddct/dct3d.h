// dct3d.h — standalone lossy codec for a single 16x16x16 voxel chunk.
//
// A 3D DCT-II float transform, dead-zone quantizer, and adaptive binary range
// coder, packaged as a dependency-free (C standard library only) single
// .c/.h pair. Extracted and generalized from SuperOptimizer/matter-compressor.
//
// - Fixed block size: 16^3 = 4096 voxels per chunk.
// - Seven input dtypes: u8, u16, u32, s8, s16, s32, f32.
// - Auto-normalized quality: `quality` is a relative fidelity knob (~1.0 =
//   high quality) that adapts to each chunk's own value range, so the same
//   number means comparable relative error for any dtype.
// - Lossy. Reconstruction is close but not bit-exact; f32 + fast-math is not
//   bit-reproducible across ISAs, so treat it as a same-build codec.
// - Reentrant and lock-free: every function is a pure transform of its
//   arguments with no shared mutable state. Call it from as many of your own
//   threads as you like; the library starts none.
// - Pure scalar C23, no SIMD intrinsics, written to auto-vectorize under
//   -O2/-O3 -ffast-math.
//
// Usage:
//     #include "dct3d.h"
//     uint16_t chunk[16*16*16];        // your data, z-major (z*256 + y*16 + x)
//     uint8_t  out[DCT3D_MAX_BYTES];   // worst-case output buffer
//     size_t   n = dct3d_encode_u16(chunk, /*quality*/1.0f, /*max_error*/0.0f,
//                                   /*tau*/0.0f, out);
//     uint16_t back[16*16*16];
//     dct3d_decode_u16(out, n, back);
//
// The `quality` argument is the quantizer coarseness (smaller = higher
// fidelity, larger file). `max_error` (relative, in [0,1)) and `tau` (absolute,
// in raw input units) each enable a sparse correction pass bounding the
// per-voxel error; pass 0 to disable either.

#ifndef DCT3D_H
#define DCT3D_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Block geometry (fixed).
#define DCT3D_N      16
#define DCT3D_N3     (DCT3D_N * DCT3D_N * DCT3D_N)   // 4096 voxels

// Worst-case encoded size in bytes. The range coder never expands incompressible
// input by more than a small constant, and the optional correction pass stores
// at most one exact residual per voxel; this bound covers both with margin.
#define DCT3D_MAX_BYTES (DCT3D_N3 * 8 + 256)

// One encode/decode pair per supported dtype. `chunk` points to DCT3D_N3
// contiguous voxels in z-major order: index (z*DCT3D_N + y)*DCT3D_N + x.
//
// encode: writes the compressed blob to `out` (capacity >= DCT3D_MAX_BYTES) and
//   returns its length in bytes. Knobs:
//     quality   > 0  quantizer coarseness (smaller = higher fidelity, bigger
//                    file). Auto-normalized: the same value gives comparable
//                    *relative* fidelity for any dtype and data scale.
//     max_error [0,1) optional *relative* error bound: caps per-voxel error at
//                    max_error * chunk_value_range. 0 disables.
//     tau       >= 0  optional *absolute* error bound in raw input units (e.g.
//                    tau=2 => reconstructed values within +-2 of the original,
//                    honored for every dtype). 0 disables.
//   If both max_error and tau are set, the tighter (smaller) bound wins. Either
//   one enables a sparse correction pass; both off = pure DCT.
// decode: reconstructs DCT3D_N3 voxels into `chunk` from a `len`-byte blob.
//   Returns 1 on success, 0 on a malformed/truncated blob. Robust to corrupt
//   input: it never reads or writes out of bounds.
//
// The blob records its own dtype; decoding with the wrong-typed function
// returns 0 rather than producing garbage.
//
// Caveats (see README for detail): the internal pipeline is float32, so u32/s32
// values above 2^24 in magnitude lose low bits before any lossy step (u8/u16/
// s8/s16 and small 32-bit values are unaffected); non-finite f32 voxels are
// sanitized to the block's finite min at encode; and f32 tau-honoring is subject
// to the same-build float-reproducibility caveat.
#define DCT3D_DECL(T, name)                                                    \
    size_t dct3d_encode_##name(const T *chunk, float quality, float max_error, \
                               float tau, uint8_t *out);                       \
    int    dct3d_decode_##name(const uint8_t *blob, size_t len, T *chunk);

DCT3D_DECL(uint8_t,  u8)
DCT3D_DECL(uint16_t, u16)
DCT3D_DECL(uint32_t, u32)
DCT3D_DECL(int8_t,   s8)
DCT3D_DECL(int16_t,  s16)
DCT3D_DECL(int32_t,  s32)
DCT3D_DECL(float,    f32)

#undef DCT3D_DECL

#ifdef __cplusplus
}
#endif

#endif // DCT3D_H
