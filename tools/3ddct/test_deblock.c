// test_deblock.c — measure decode-side deblocking on assembled dct3d blocks.
//
// Build a smooth multi-block u8 volume, encode/decode each 16^3 block with the
// real codec at aggressive compression (which introduces seams), assemble, then
// compare a seam-discontinuity metric and the true error vs the original before
// and after dct3d_deblock_u8. Also a genuine-edge guard: a sharp step that lands
// ON a block boundary must NOT be smoothed away.

#include "dct3d.h"
#include "deblock.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N DCT3D_N
#define GB 4                       // blocks per axis
#define D  (GB * N)                // volume extent per axis (64)
#define VOX ((size_t)D * D * D)

static int fails = 0;
#define CHECK(c, m, ...) do{ if(!(c)){ printf("FAIL: " m "\n", ##__VA_ARGS__); fails++; } }while(0)

// Mean squared jump across the planes normal to axis `ax` at multiples of N
// (the internal seams), skipping the volume boundary. A pure measure of the
// block-grid discontinuity.
static double seam_energy(const uint8_t *v, int ax) {
    double se = 0; long cnt = 0;
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < D; ++y)
    for (int x = 0; x < D; ++x) {
        int c = ax == 0 ? z : ax == 1 ? y : x;
        if (c == 0 || c % N != 0) continue;             // only interior seams
        size_t i  = ((size_t)z * D + y) * D + x;
        size_t im = ax == 0 ? i - (size_t)D * D : ax == 1 ? i - D : i - 1;
        double d = (double)v[i] - (double)v[im];
        se += d * d; cnt++;
    }
    return cnt ? se / cnt : 0.0;
}

static double mse(const uint8_t *a, const uint8_t *b) {
    double se = 0;
    for (size_t i = 0; i < VOX; ++i) { double d = (double)a[i] - (double)b[i]; se += d * d; }
    return se / VOX;
}

// Encode+decode every 16^3 block of `src` into `dst` at the given quality.
static void codec_roundtrip(const uint8_t *src, uint8_t *dst, float q) {
    uint8_t blob[DCT3D_MAX_BYTES];
    for (int bz = 0; bz < GB; ++bz)
    for (int by = 0; by < GB; ++by)
    for (int bx = 0; bx < GB; ++bx) {
        uint8_t blk[DCT3D_N3], back[DCT3D_N3];
        for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            size_t vi = ((size_t)(bz*N+z) * D + (by*N+y)) * D + (bx*N+x);
            blk[(z*N + y)*N + x] = src[vi];
        }
        size_t n = dct3d_encode_u8(blk, q, 0.0f, 0.0f, blob);
        dct3d_decode_u8(blob, n, back);
        for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            size_t vi = ((size_t)(bz*N+z) * D + (by*N+y)) * D + (bx*N+x);
            dst[vi] = back[(z*N + y)*N + x];
        }
    }
}

int main(void) {
    const size_t dims[3] = { D, D, D };

    // --- Case 1: smooth gradient + low-freq ripple (a codec seam generator) ---
    uint8_t *orig = malloc(VOX), *dec = malloc(VOX), *deb = malloc(VOX);
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < D; ++y)
    for (int x = 0; x < D; ++x) {
        double f = 128.0
                 + 60.0 * sin(x * 0.11 + 0.5) * cos(y * 0.09)
                 + 40.0 * sin(z * 0.07 + y * 0.03);
        f = f < 0 ? 0 : f > 255 ? 255 : f;
        orig[((size_t)z*D + y)*D + x] = (uint8_t)(f + 0.5);
    }

    codec_roundtrip(orig, dec, /*q=*/24.0f);   // aggressive -> visible seams
    memcpy(deb, dec, VOX);
    dct3d_deblock_u8(deb, dims, 0.0f, 1.0f);

    double s_pre  = seam_energy(dec, 0) + seam_energy(dec, 1) + seam_energy(dec, 2);
    double s_post = seam_energy(deb, 0) + seam_energy(deb, 1) + seam_energy(deb, 2);
    double e_pre  = mse(orig, dec);
    double e_post = mse(orig, deb);

    printf("smooth q=24:  seam_energy  pre=%.3f  post=%.3f  (%.0f%% reduction)\n",
           s_pre, s_post, 100.0 * (s_pre - s_post) / (s_pre + 1e-9));
    printf("              MSE-vs-orig  pre=%.3f  post=%.3f\n", e_pre, e_post);

    // Deblock must REDUCE seam energy...
    CHECK(s_post < s_pre * 0.8, "deblock did not reduce seam energy enough");
    // ...and must not INCREASE true error (it should hold or improve it).
    CHECK(e_post <= e_pre * 1.02, "deblock increased MSE vs original");

    // --- Case 2: genuine sharp edge landing on a block boundary at x=2N ---
    // A hard step of ~100 across x=32. Deblock must NOT flatten it.
    for (size_t i = 0; i < VOX; ++i) {
        int x = (int)(i % D);
        orig[i] = x < 2*N ? 40 : 200;
    }
    // no codec here — test the filter's edge discrimination directly.
    memcpy(deb, orig, VOX);
    dct3d_deblock_u8(deb, dims, 0.0f, 1.0f);
    // The step across x=2N in the deblocked array must stay close to 160.
    double edge_pre = 200.0 - 40.0;
    double edge_min = 1e9;
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < D; ++y) {
        size_t i = ((size_t)z*D + y)*D + (2*N);
        double step = (double)deb[i] - (double)deb[i-1];
        if (step < edge_min) edge_min = step;
    }
    printf("edge guard:   real step pre=%.0f  min post=%.0f (must stay ~%.0f)\n",
           edge_pre, edge_min, edge_pre);
    CHECK(edge_min > edge_pre * 0.9, "deblock smoothed a genuine edge (%.0f < %.0f)",
          edge_min, edge_pre * 0.9);

    free(orig); free(dec); free(deb);
    printf(fails ? "\n%d CHECK(s) FAILED\n" : "\nall deblock tests passed\n", fails);
    return fails ? 1 : 0;
}
