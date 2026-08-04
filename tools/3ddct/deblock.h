// deblock.h — decode-side deblocking post-filter for dct3d.
//
// dct3d decodes each 16^3 chunk independently. At high compression the
// per-block quantization leaves a small step in value across the shared face
// between two adjacent blocks; assembled into a volume these steps read as a
// visible 16-voxel grid. A lone-chunk decode cannot remove them — the seam is a
// relationship BETWEEN two blocks, so the filter must see both sides. This
// module is therefore a POST-PROCESS: decode the chunk and its face neighbors
// first (with the normal, pure dct3d_decode_*), then hand them all here. The
// bitstream and the per-chunk decode contract are untouched.
//
// The filter is signal-adaptive, modeled on the JPEG/H.26x boundary filters:
// it only softens a seam that looks like a quantization step (both sides
// locally flat, the step small relative to the data range) and leaves genuine
// edges that happen to fall on a block boundary alone.
//
// Two granularities:
//
//   1. Chunk API (the primary form): given one decoded 16^3 chunk and its six
//      decoded face-neighbor chunks (NULL for any missing/volume-boundary
//      face), produce the deblocked chunk. Out-of-place and pure — pass
//      pristine decoded inputs and the result is order-independent, so chunks
//      can be filtered in parallel. Each call fixes only the CENTER chunk's
//      side of each seam; filtering every chunk (each seeing the same pristine
//      neighbors) heals both sides consistently.
//
//          uint8_t c[4096], nb[6][4096], out[4096];
//          dct3d_deblock_chunk_u8(c, nb[0],nb[1],nb[2],nb[3],nb[4],nb[5],
//                                 1.0f, out);
//
//   2. Volume API: same filter over a dense assembled ZYX array in place,
//      treating every internal 16-aligned plane as a seam. Convenient when a
//      whole region is already assembled (e.g. a decoded shard).
//
// strength scales the correction (0 = no-op, 1 = calibrated default, up to ~2
// = stronger). Thresholds scale with the local data range so the same strength
// behaves comparably across content.

#ifndef DCT3D_DEBLOCK_H
#define DCT3D_DEBLOCK_H

#include <stddef.h>
#include <stdint.h>

#include "dct3d.h"   // DCT3D_N (block size)

#ifdef __cplusplus
extern "C" {
#endif

// `step` is the codec quantizer scale in RAW value units — pass the `quality`
// the data was encoded with (for u8 sources they coincide; the encoder's
// normalized qstep maps back through ~vspan/255 ≈ 1 for full-range blocks).
// Quantization seams scale with the quantizer, real edges do not, so this is
// what separates them (the H.26x QP-threshold model). Pass 0 to fall back to
// thresholds derived from the local data range — that mode only works on
// smooth synthetic content; on textured CT it degenerates to a near-no-op.
//
// Chunk API. `chunk` and each non-NULL neighbor are DCT3D_N3 voxels, z-major.
// Neighbor naming is by the face of `chunk` it touches: xlo is the neighbor at
// lower x (its x=N-1 plane meets chunk's x=0 plane), xhi at higher x, etc.
// Writes the filtered chunk to `out` (may not alias any input). Only voxels in
// the two layers nearest each neighbored face can change.
#define DCT3D_DEBLOCK_CHUNK_DECL(T, name)                                     \
    void dct3d_deblock_chunk_##name(const T *chunk,                           \
                                    const T *xlo, const T *xhi,               \
                                    const T *ylo, const T *yhi,               \
                                    const T *zlo, const T *zhi,               \
                                    float step, float strength, T *out);

// Volume API. dims are ZYX voxel extents; deblocks in place across every
// internal plane at a multiple of DCT3D_N.
#define DCT3D_DEBLOCK_VOL_DECL(T, name)                                       \
    void dct3d_deblock_##name(T *vol, const size_t dims[3], float step,       \
                              float strength);

#define DCT3D_DEBLOCK_DECL(T, name) \
    DCT3D_DEBLOCK_CHUNK_DECL(T, name) \
    DCT3D_DEBLOCK_VOL_DECL(T, name)

DCT3D_DEBLOCK_DECL(uint8_t,  u8)
DCT3D_DEBLOCK_DECL(uint16_t, u16)
DCT3D_DEBLOCK_DECL(uint32_t, u32)
DCT3D_DEBLOCK_DECL(int8_t,   s8)
DCT3D_DEBLOCK_DECL(int16_t,  s16)
DCT3D_DEBLOCK_DECL(int32_t,  s32)
DCT3D_DEBLOCK_DECL(float,    f32)

#undef DCT3D_DEBLOCK_DECL
#undef DCT3D_DEBLOCK_CHUNK_DECL
#undef DCT3D_DEBLOCK_VOL_DECL

#ifdef __cplusplus
}
#endif

#endif // DCT3D_DEBLOCK_H
