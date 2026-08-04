// downsample.h — gradient-magnitude-weighted 2x2x2 downsample, computed one
// output Z-slab at a time from up to 4 source octants (a Z-slab of an output
// shard only ever needs the 4 level-(N-1) octants sharing its dz half; the
// other 4 belong to the other half of the shard's Z extent).
//
// Plain box/mean pooling blurs the sharp intensity steps that mark sheet
// boundaries and ink signal in CT data. For each output voxel we instead weight
// its 8 source voxels by a local gradient-magnitude estimate before averaging,
// so high-contrast voxels (edges/boundaries) pull the result toward themselves
// instead of being diluted by their smoother neighbors. Degenerates to a plain
// mean in flat regions (all weights equal), so it never diverges from the
// obvious answer where there's nothing to preserve.
//
// Weight for source voxel v (one of the 8 in a 2x2x2 cell): the magnitude of
// the central difference at v. w(v) = 1 + gain * |grad(v)|, gain fixed so a
// full 0..255 step dominates a flat region by roughly an order of magnitude.
//
// STREAMING DESIGN: an output shard's 2x2x2x(SHARD_VOX^3) parent region used
// to be materialized as one dense (2*SHARD_VOX)^3 cube (~8 GiB) before running
// the reduction below — and a naive "decode one octant fully, fold into a
// SHARD_VOX^3 accumulator" redesign doesn't actually help, since the f32
// accumulator pair is itself ~8 GiB (same order as the cube it replaces).
// The real fix streams by Z-SLAB: the output shard is processed in thin Z
// slabs (see DOWNSAMPLE_SLAB_OUT below); each slab needs only the two
// source-Z inner-chunk rows per contributing octant (a small
// slab_src x SHARD_VOX x SHARD_VOX buffer, decoded straight from the
// already-in-memory compressed blob via shard_decode_u8_zrows — no
// full-shard dense decode ever happens), and only the 4 (not 8) octants
// sharing that slab's dz half. Peak residency: 4 decode scratch buffers
// (slab_src * SHARD_VOX^2 bytes each, ~128 MiB at slab_out=64) + one
// slab-sized accumulator pair (out_edge^2 * slab_out f32 x2, ~256 MiB) + the
// finalized SHARD_VOX^3 u8 output shard (~1 GiB, needed anyway for encoding)
// — on the order of 1.5-2 GiB total, not 8.
//
// Y/X seams: within a slab, the 4 live octants still differ in dy/dx, i.e.
// still meet at Y/X faces not resident in the same buffer. Gradient there
// falls back to a one-sided difference (see grad_mag_seam in the .c) exactly
// as in the original single-fold design: a deliberate, documented
// approximation, justified because the gradient here is only a heuristic
// edge-preservation weight (not a reconstructed value), one-sided and central
// differences agree in sign/order-of-magnitude for any real edge, and both
// are exactly zero in flat regions (so flat-degeneracy is unaffected) — and
// it now affects a strictly smaller sliver of the volume than before (2 of
// SHARD_VOX faces per slab vs. per whole octant, same relative fraction).
// Z-slab boundaries are NOT a seam: each octant's decode range includes a
// 1-source-voxel halo from its own neighboring Z-rows (still the same shard,
// same compressed blob) whenever available, so the Z direction keeps an
// exact central difference everywhere except the true outer edge of the
// whole level-(N-1) volume (handled by the existing clamp).

#ifndef DCT3D_DOWNSCALE_DOWNSAMPLE_H
#define DCT3D_DOWNSCALE_DOWNSAMPLE_H

#include <stddef.h>
#include <stdint.h>

// Output-Z-voxels processed per slab. Must divide SHARD_VOX/2 (== 512) evenly
// so every slab maps to a whole number of inner-chunk-Z-rows on each octant
// side. 64 -> 128 source voxels/octant/slab -> 8 inner-chunk Z-rows/slab.
#define DOWNSAMPLE_SLAB_OUT 64

typedef struct {
    float *wsum;  // out_edge^2 * slab_out — sum of weights
    float *vsum;  // out_edge^2 * slab_out — sum of weight*value
    size_t out_edge;   // full output shard edge (SHARD_VOX)
    size_t slab_out;   // this slab's Z thickness in output voxels
} downsample_acc;

void downsample_acc_reset(downsample_acc *acc);

// Fold one decoded octant's Z-slab contribution into `acc`.
//
// `oct_slab` is dense oct_zdim x out_edge x out_edge u8 (z-major), where
// out_edge == SHARD_VOX is this octant's own (source) Y/X edge, and
// oct_zdim == halo_lo + 2*acc->slab_out + halo_hi: the source-Z range for
// this output slab (2*slab_out voxels) plus up to 1 true-halo voxel on each
// Z side (halo_lo/halo_hi in {0,1}; 0 only at the outer edge of the whole
// level-(N-1) volume, where the caller instead duplicates the boundary row
// so the clamped-gradient behavior still applies — see srcfetch.c).
// `z0` is the row within `oct_slab` where the real (non-halo) slab begins
// (== halo_lo).
//
// (dy, dx) in {0,1} select which of the 4 octants sharing this Z-half this
// is (matching srcfetch's shard-coord offsets); there is no `dz` parameter
// because the caller already selected the Z-half by choosing which octant
// pair to decode a slab range from — this function always places output at
// rows [0, slab_out) of `acc` (the current slab).
void downsample_fold_octant_slab(downsample_acc *acc, const uint8_t *oct_slab,
                                 size_t oct_zdim, size_t z0, int dy, int dx);

// Finalize `acc` into `out_slab` (out_edge * out_edge * acc->slab_out u8,
// z-major): out = round(vsum/wsum), or 0 where a cell got no contribution.
void downsample_finalize_slab(const downsample_acc *acc, uint8_t *out_slab);

#endif  // DCT3D_DOWNSCALE_DOWNSAMPLE_H
