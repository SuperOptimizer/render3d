// predeblock.h — ENCODE-side deblocking pre-conditioner for dct3d.
//
// The decode-side filter (deblock.h) needs decoded neighbors at read time.
// This module instead removes the seam at WRITE time, so a plain lone-chunk
// dct3d_decode_* reconstructs with far less blocking and readers need no
// post-pass. The bitstream format is untouched: output is an ordinary dct3d
// blob.
//
// Why blocking happens: each block quantizes independently, so its
// reconstruction error is discontinuous at the seam — the signal step across
// the boundary is reproduced with two unrelated errors, and the difference
// reads as a visible 16-voxel grid. The fix here: when encoding a block whose
// neighbor is already encoded, inject the neighbor's KNOWN reconstruction
// error into this block's boundary layers (tapered a few voxels deep) before
// the forward DCT. Both sides of the seam then err (approximately) IDENTICALLY,
// so the reconstructed step matches the true step and the seam disappears —
// while the actual signal gradient across the boundary is preserved exactly.
// No iteration, no rate model, one extra add per boundary voxel.
//
// (Survey note: lapped transforms — LOT/LBT/Tran-Tu-Liang/Daala TDLT — solve
// this more elegantly but all require a boundary-straddling inverse post-filter
// at DECODE, i.e. neighbor access on the read path, which defeats the point of
// fixing it at encode. Quantization-bin-constrained POCS solves are decode-pure
// but iterative; with a fixed dequantization midpoint like ours they reduce to
// choosing different levels anyway. Error injection gets most of the benefit at
// ~zero cost.)
//
// Intended encode order: checkerboard. Encode all even-parity (bz+by+bx even)
// blocks plainly, decode their boundary planes, then encode odd-parity blocks
// with the measured neighbor errors. Every seam is even|odd, so every seam gets
// compensated exactly once, and all six neighbors of an odd block are ready.
// A raster/causal order (3 of 6 faces) also works at half the effect.
//
// Face error planes are N*N floats: (neighbor reconstructed) - (neighbor
// original) on the neighbor's plane touching this chunk, indexed by the two
// in-plane axes in ZYX order with the normal axis dropped:
//   x faces: idx = z*N + y      y faces: idx = z*N + x      z faces: idx = y*N + x
// Pass NULL for a face with no neighbor (volume boundary) or an uncompensated
// face (e.g. the neighbor is odd-parity too).
//
// tau note: with a correction bound set, corrections are enforced against the
// SHIFTED source, so the guarantee vs the true original loosens to tau + |shift|
// on the tapered boundary layers (shift <= the neighbor's own error, itself
// tau-bounded). Interior voxels keep the plain tau guarantee.

#ifndef DCT3D_PREDEBLOCK_H
#define DCT3D_PREDEBLOCK_H

#include <stddef.h>
#include <stdint.h>

#include "dct3d.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encode one 16^3 chunk with encode-side seam compensation. Signature matches
// dct3d_encode_<dtype> plus the six optional face-error planes and a strength
// scale (1 = inject the full neighbor error, 0 = plain encode). Returns the
// blob length; decode with the ordinary dct3d_decode_<dtype>.
#define DCT3D_PREDEBLOCK_DECL(T, name)                                        \
    size_t dct3d_encode_deblock_##name(const T *chunk,                        \
        const float *err_xlo, const float *err_xhi,                           \
        const float *err_ylo, const float *err_yhi,                           \
        const float *err_zlo, const float *err_zhi,                           \
        float quality, float max_error, float tau,                            \
        float strength, uint8_t *out);

DCT3D_PREDEBLOCK_DECL(uint8_t,  u8)
DCT3D_PREDEBLOCK_DECL(uint16_t, u16)
DCT3D_PREDEBLOCK_DECL(float,    f32)

#undef DCT3D_PREDEBLOCK_DECL

#ifdef __cplusplus
}
#endif

#endif // DCT3D_PREDEBLOCK_H
