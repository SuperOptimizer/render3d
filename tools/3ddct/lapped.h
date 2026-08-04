// lapped.h — time-domain lapped pre/post filtering for dct3d.
//
// The reference-quality answer to blocking (JPEG XR's LBT, Daala's TDLT,
// Tran/Tu/Liang 2003): an invertible pre-filter P applied across each block
// boundary BEFORE the per-block DCT, and its exact inverse P^-1 applied across
// the same boundaries AFTER the per-block inverse DCT. The pre-filter compacts
// cross-boundary smoothness into the block interiors, so the block DCT codes
// fewer high-frequency coefficients (coding gain) and quantization error is
// spread smoothly across the seam by the post-filter (no visible blocking).
//
// Contract change vs plain dct3d: each block still RANGE-DECODES independently,
// but correct reconstruction of lapped content requires the post-filter, which
// straddles boundaries — the same center+neighbors access pattern as the
// heuristic deblock post-pass (deblock.h), except this one is the exact inverse
// of what the encoder did, not a threshold heuristic. Skipping the post-filter
// yields a usable but soft-boundaried image.
//
// Filter form (per boundary, per line crossing it, overlap m=DCT3D_LAP_M on
// each side): fold into symmetric/antisymmetric halves, scale the
// antisymmetric half by s_i < 1 (pre) or 1/s_i (post), unfold. In matrix
// terms P = F * diag(I, V) * F with F the butterfly and V = diag(s_i) — the
// Tran/Tu/Liang factorization with V diagonal, which is where most of the
// lapped coding gain lives while staying trivially invertible. Applied
// separably along x, y, z. Perfect reconstruction in exact arithmetic; here
// float (the codec is tolerance-only anyway) plus integer rounding when the
// working dtype is integral.
//
// APIs are VOLUME-form only (dense ZYX array, every internal 16-aligned
// plane, in place): the three axis passes are separable and commute, but a
// per-chunk post-filter fed only face neighbors would be inexact at chunk
// edges/corners (the cross terms need diagonal neighbors). The exact reader
// pattern is region + halo: decode the region plus a DCT3D_LAP_M-voxel halo,
// post-filter the assembly, crop. Under the recommended deployment rule —
// lap only faces INTERIOR to a shard, never across shard boundaries — a
// whole-shard read post-filters exactly with no halo at all, the shard stays
// fully self-contained, and only the shard-boundary seams remain (which the
// heuristic deblock post-pass covers).
//
// A face with no neighbor (volume/shard edge) is never filtered. Pre and post
// must agree on which planes were lapped — that bookkeeping (e.g. a per-blob
// face mask in the container) is the caller's until it moves into the dct3d
// bitstream.

#ifndef DCT3D_LAPPED_H
#define DCT3D_LAPPED_H

#include <stddef.h>
#include <stdint.h>

#include "dct3d.h"

#ifdef __cplusplus
extern "C" {
#endif

// Overlap half-width: samples taken from EACH side of a boundary.
#define DCT3D_LAP_M 4

// In place over every internal plane at a multiple of DCT3D_N. prefilter
// before tiling+encoding; postfilter after decoding+assembly. For integer
// dtypes the filtered values are rounded back to the dtype, so a lossless
// pre->post round trip reconstructs within ~1 unit (documented caveat; the
// float path is exact to fp precision).
#define DCT3D_LAP_DECL(T, name)                                              \
    void dct3d_lap_prefilter_##name(T *vol, const size_t dims[3]);           \
    void dct3d_lap_postfilter_##name(T *vol, const size_t dims[3]);

DCT3D_LAP_DECL(uint8_t,  u8)
DCT3D_LAP_DECL(uint16_t, u16)
DCT3D_LAP_DECL(float,    f32)

#undef DCT3D_LAP_DECL

#ifdef __cplusplus
}
#endif

#endif // DCT3D_LAPPED_H
