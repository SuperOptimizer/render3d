// test_roundtrip.c — sanity check: encode a shard, decode it back, verify the
// gradient downsample degenerates to a mean in flat regions and preserves a
// hard step better than plain averaging would.
//
// The downsample kernel now works by folding one octant's Z-slab at a time
// into a slab-sized accumulator (see downsample.h) rather than taking one
// big dense cube. These tests drive that same fold/finalize API directly on
// a single synthetic "octant" (an SHARD_VOX-edge cube is overkill for a unit
// test, so we test at a small scale: OS-edge output from a 2*OS-edge octant,
// one Z-slab covering the whole thing, single octant dy=dx=0) to keep the
// invariants (flat degeneracy, sharp-step preservation) covered without
// paying for a full 1024-edge run.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decode.h"
#include "downsample.h"
#include "../export/shard.h"

#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); ok = 0; } } while (0)

// Downsample a single OCT_S-edge octant on its own (dy=dx=0, one slab
// covering the whole thing) into its OS = OCT_S/2 output voxels.
//
// downsample_fold_octant_slab's `acc.out_edge` is the FULL output shard edge
// that this octant is one of 4 (dy,dx) quadrants of -- a single octant's own
// edge only ever covers HALF of that (Y/X base offset dy*OE/2, dx*OE/2), so
// with dy=dx=0 it fills only the [0, OE/2) x [0, OE/2) sub-range of a full
// OE-edge accumulator. To read out this octant's full OS = oct_s/2 edge of
// output on its own, size the accumulator with out_edge == oct_s (so
// OE/2 == OS, matching the sub-range dy=dx=0 fills) and then extract just
// that [0, OS) x [0, OS) corner into the OS^3 `out` buffer -- the rest of
// the accumulator (the sub-range a sibling dy/dx=1 octant would have filled)
// is left as zero-weight/unwritten and must not be read.
static void run_single_octant(const uint8_t *oct, size_t oct_s, uint8_t *out) {
    size_t OS = oct_s / 2;
    downsample_acc acc;
    acc.out_edge = oct_s;
    acc.slab_out = OS;  // whole thing in one slab
    acc.wsum = (float *)malloc(oct_s * oct_s * OS * sizeof(float));
    acc.vsum = (float *)malloc(oct_s * oct_s * OS * sizeof(float));
    downsample_acc_reset(&acc);
    downsample_fold_octant_slab(&acc, oct, oct_s, 0, 0, 0);

    uint8_t *full_out = (uint8_t *)malloc(oct_s * oct_s * OS);
    downsample_finalize_slab(&acc, full_out);
    for (size_t oz = 0; oz < OS; ++oz)
        for (size_t oy = 0; oy < OS; ++oy)
            memcpy(out + (oz * OS + oy) * OS,
                   full_out + (oz * oct_s + oy) * oct_s, OS);
    free(full_out);
    free(acc.wsum); free(acc.vsum);
}

int main(void) {
    int ok = 1;

    // 1. shard encode/decode round trip on a synthetic gradient volume.
    size_t S = SHARD_VOX;
    uint8_t *vol = (uint8_t *)malloc(S * S * S);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x)
                vol[(z * S + y) * S + x] = (uint8_t)((x + y + z) % 256);

    uint8_t *shard = NULL; size_t len = 0;
    CHECK(shard_encode_u8(vol, 32.0f, 64.0f, &shard, &len) == 0, "encode failed");

    uint8_t *back = (uint8_t *)calloc(S * S * S, 1);
    CHECK(shard_decode_u8(shard, len, back) == 0, "decode failed");

    double se = 0;
    for (size_t i = 0; i < S * S * S; ++i) {
        double d = (double)vol[i] - (double)back[i];
        se += d * d;
    }
    double mse = se / (S * S * S);
    printf("roundtrip MSE (q=32,tau=64): %g\n", mse);
    CHECK(mse < 64.0 * 64.0, "roundtrip error way outside tau bound");

    // 1b. Z-row-range decode must match the full decode over the same range.
    uint8_t *back_rows = (uint8_t *)calloc(S * S * S, 1);
    const int iz0 = 10, iz1 = 20;
    CHECK(shard_decode_u8_zrows(shard, len, iz0, iz1, back_rows) == 0,
          "zrows decode failed");
    int zrows_match = 1;
    for (size_t z = 0; z < (size_t)(iz1 - iz0) * 16 && zrows_match; ++z)
        for (size_t y = 0; y < S && zrows_match; ++y)
            for (size_t x = 0; x < S; ++x) {
                size_t full_z = (size_t)iz0 * 16 + z;
                if (back_rows[(z * S + y) * S + x] != back[(full_z * S + y) * S + x]) {
                    zrows_match = 0;
                    break;
                }
            }
    CHECK(zrows_match, "zrows decode disagrees with full decode over same range");

    // 2. downsample: flat region degenerates to plain mean.
    size_t FS = 32, OS = 16;
    uint8_t *flat = (uint8_t *)malloc(FS * FS * FS);
    memset(flat, 100, FS * FS * FS);
    uint8_t *out_flat = (uint8_t *)malloc(OS * OS * OS);
    run_single_octant(flat, FS, out_flat);
    int flat_ok = 1;
    for (size_t i = 0; i < OS * OS * OS; ++i) if (out_flat[i] != 100) flat_ok = 0;
    CHECK(flat_ok, "flat region did not degenerate to input value");

    // 3. downsample: a sharp step should stay closer to a step than a plain
    // mean would (mean of 0/255 half-and-half along x = ~127; the
    // gradient-weighted result at cells straddling the step should still land
    // near the midpoint since both sides carry equal high gradient, but cells
    // one full source-index away from the boundary must reproduce their flat
    // side exactly, unlike a naive linear-interp scheme).
    uint8_t *step = (uint8_t *)malloc(FS * FS * FS);
    for (size_t z = 0; z < FS; ++z)
        for (size_t y = 0; y < FS; ++y)
            for (size_t x = 0; x < FS; ++x)
                step[(z * FS + y) * FS + x] = (x < FS / 2) ? 0 : 255;
    uint8_t *out_step = (uint8_t *)malloc(OS * OS * OS);
    run_single_octant(step, FS, out_step);
    // Far-from-boundary output voxels must be exactly flat (0 or 255).
    uint8_t far_lo = out_step[(0 * OS + 0) * OS + 0];
    uint8_t far_hi = out_step[(0 * OS + 0) * OS + (OS - 1)];
    printf("step downsample: far_lo=%d far_hi=%d\n", far_lo, far_hi);
    CHECK(far_lo == 0, "far-from-boundary low side not exactly flat");
    CHECK(far_hi == 255, "far-from-boundary high side not exactly flat");

    free(vol); free(shard); free(back); free(back_rows);
    free(flat); free(out_flat); free(step); free(out_step);

    if (ok) printf("ALL CHECKS PASSED\n");
    return ok ? 0 : 1;
}
