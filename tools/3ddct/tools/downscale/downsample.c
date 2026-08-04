// downsample.c — see downsample.h.

#include "downsample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Gain on the gradient-magnitude weight. Chosen so a full-range step
// (|grad| ~ 255) gets ~10x the pull of a flat-region voxel (|grad| ~ 0),
// while flat regions (the common case) still reduce to a plain mean.
#define GRAD_GAIN (9.0f / 255.0f)

static inline size_t clampi(long v, long lo, long hi) {
    return (size_t)(v < lo ? lo : (v > hi ? hi : v));
}

// Central-difference gradient magnitude at (z,y,x) in `oct_slab`
// (Zd x Wd x Wd, z-major), z clamped to the buffer's own Z extent (used when
// there's no true halo row available, i.e. at the outer edge of the whole
// level-(N-1) volume — matches the old design's edge-clamp behavior), y/x
// falling back to a one-sided difference when on a Y/X seam face shared with
// a sibling octant not resident in this buffer (see the seam parameters).
static inline float grad_mag(const uint8_t *oct_slab, size_t Zd, size_t Wd,
                            size_t z, size_t y, size_t x,
                            int seam_y, int seam_x) {
    const size_t W2 = Wd * Wd;
    long zl = (long)z, zhi = (long)Zd - 1;
    size_t z0 = clampi(zl - 1, 0, zhi), z1 = clampi(zl + 1, 0, zhi);
    float gz = (float)oct_slab[z1 * W2 + y * Wd + x] - (float)oct_slab[z0 * W2 + y * Wd + x];

    float gy;
    if (seam_y >= 0 && (long)y == seam_y) {
        long yn = (seam_y == 0) ? (long)y + 1 : (long)y - 1;
        float dv = (float)oct_slab[z * W2 + (size_t)yn * Wd + x] -
                   (float)oct_slab[z * W2 + y * Wd + x];
        gy = (seam_y == 0) ? dv : -dv;
    } else {
        long yl = (long)y, yhi = (long)Wd - 1;
        size_t y0 = clampi(yl - 1, 0, yhi), y1 = clampi(yl + 1, 0, yhi);
        gy = (float)oct_slab[z * W2 + y1 * Wd + x] - (float)oct_slab[z * W2 + y0 * Wd + x];
    }

    float gx;
    if (seam_x >= 0 && (long)x == seam_x) {
        long xn = (seam_x == 0) ? (long)x + 1 : (long)x - 1;
        float dv = (float)oct_slab[z * W2 + y * Wd + (size_t)xn] -
                   (float)oct_slab[z * W2 + y * Wd + x];
        gx = (seam_x == 0) ? dv : -dv;
    } else {
        long xl = (long)x, xhi = (long)Wd - 1;
        size_t x0 = clampi(xl - 1, 0, xhi), x1 = clampi(xl + 1, 0, xhi);
        gx = (float)oct_slab[z * W2 + y * Wd + x1] - (float)oct_slab[z * W2 + y * Wd + x0];
    }
    return sqrtf(gz * gz + gy * gy + gx * gx);
}

void downsample_acc_reset(downsample_acc *acc) {
    size_t n = acc->out_edge * acc->out_edge * acc->slab_out;
    memset(acc->wsum, 0, n * sizeof(float));
    memset(acc->vsum, 0, n * sizeof(float));
}

void downsample_fold_octant_slab(downsample_acc *acc, const uint8_t *oct_slab,
                                 size_t oct_zdim, size_t z0, int dy, int dx) {
    const size_t OE = acc->out_edge;
    const size_t W = OE;   // this octant's own SHARD_VOX == OE edge in Y/X (one octant == one parent shard)

    // Seam faces (see downsample.h): this octant covers HALF the output
    // shard's Y/X extent (its own SHARD_VOX == OE source voxels downsample to
    // OE/2 output cells); dy==0/dx==0 -> seam is far face (W-1), dy==1/dx==1
    // -> seam is near face (0).
    int seam_y = (dy == 0) ? (int)W - 1 : 0;
    int seam_x = (dx == 0) ? (int)W - 1 : 0;

    const size_t half = OE / 2;
    size_t base_oy = (size_t)dy * half, base_ox = (size_t)dx * half;

    for (size_t sz = 0; sz < 2 * acc->slab_out; ++sz) {
        size_t z = z0 + sz;                   // row within oct_slab
        size_t oz = sz / 2;                   // output row within this slab
        for (size_t y = 0; y < W; ++y) {
            size_t oy = base_oy + y / 2;
            const uint8_t *row = oct_slab + (z * W + y) * W;
            float *wrow = acc->wsum + (oz * OE + oy) * OE;
            float *vrow = acc->vsum + (oz * OE + oy) * OE;
            for (size_t x = 0; x < W; ++x) {
                size_t ox = base_ox + x / 2;
                float g = grad_mag(oct_slab, oct_zdim, W, z, y, x, seam_y, seam_x);
                float v = (float)row[x];
                float w = 1.0f + GRAD_GAIN * g;
                wrow[ox] += w;
                vrow[ox] += w * v;
            }
        }
    }
}

void downsample_finalize_slab(const downsample_acc *acc, uint8_t *out_slab) {
    size_t n = acc->out_edge * acc->out_edge * acc->slab_out;
    for (size_t i = 0; i < n; ++i) {
        float w = acc->wsum[i];
        float r = (w > 0.0f) ? (acc->vsum[i] / w + 0.5f) : 0.0f;
        out_slab[i] = (uint8_t)(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
    }
}
