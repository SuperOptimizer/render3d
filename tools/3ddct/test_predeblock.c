// test_predeblock.c — measure encode-side seam compensation, alone and
// combined with the decode-side chunk filter.
//
// Checkerboard experiment on a smooth multi-block u8 volume: encode even-parity
// blocks plainly, odd-parity blocks with the even neighbors' measured
// reconstruction errors injected (predeblock). Compare seam energy / MSE /
// encoded bytes across: plain | encode-side | decode-side | both.

#include "dct3d.h"
#include "deblock.h"
#include "predeblock.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N DCT3D_N
#define GB 4
#define D  (GB * N)
#define VOX ((size_t)D * D * D)

static int fails = 0;
#define CHECK(c, m, ...) do{ if(!(c)){ printf("FAIL: " m "\n", ##__VA_ARGS__); fails++; } }while(0)

static double seam_energy(const uint8_t *v) {
    double se = 0; long cnt = 0;
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < D; ++y)
    for (int x = 0; x < D; ++x) {
        size_t i = ((size_t)z * D + y) * D + x;
        if (z && z % N == 0) { double d = (double)v[i] - v[i - (size_t)D*D]; se += d*d; cnt++; }
        if (y && y % N == 0) { double d = (double)v[i] - v[i - D];           se += d*d; cnt++; }
        if (x && x % N == 0) { double d = (double)v[i] - v[i - 1];           se += d*d; cnt++; }
    }
    return cnt ? se / cnt : 0.0;
}
static double mse(const uint8_t *a, const uint8_t *b) {
    double se = 0;
    for (size_t i = 0; i < VOX; ++i) { double d = (double)a[i] - b[i]; se += d*d; }
    return se / VOX;
}

static void get_block(const uint8_t *v, int bz, int by, int bx, uint8_t *blk) {
    for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x)
        blk[(z*N + y)*N + x] = v[((size_t)(bz*N+z)*D + (by*N+y))*D + (bx*N+x)];
}
static void put_block(uint8_t *v, int bz, int by, int bx, const uint8_t *blk) {
    for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x)
        v[((size_t)(bz*N+z)*D + (by*N+y))*D + (bx*N+x)] = blk[(z*N + y)*N + x];
}

// Face-error plane of the neighbor touching block (bz,by,bx) on face `f`
// (0..5 = xlo,xhi,ylo,yhi,zlo,zhi): recon - orig on the neighbor's plane.
// Returns NULL (and leaves buf alone) when there is no neighbor.
static const float *face_err(const uint8_t *recon, const uint8_t *orig,
                             int bz, int by, int bx, int f, float *buf) {
    int gz0 = bz*N, gy0 = by*N, gx0 = bx*N;
    int px, py, pz;             // global coords of the neighbor plane
    switch (f) {
        case 0: if (bx == 0)      return NULL; px = gx0 - 1;  break;
        case 1: if (bx == GB - 1) return NULL; px = gx0 + N;  break;
        case 2: if (by == 0)      return NULL; py = gy0 - 1;  break;
        case 3: if (by == GB - 1) return NULL; py = gy0 + N;  break;
        case 4: if (bz == 0)      return NULL; pz = gz0 - 1;  break;
        default:if (bz == GB - 1) return NULL; pz = gz0 + N;  break;
    }
    for (int a = 0; a < N; ++a) for (int b = 0; b < N; ++b) {
        size_t i;
        if (f < 2)      i = ((size_t)(gz0+a)*D + (gy0+b))*D + px;   // idx z*N+y
        else if (f < 4) i = ((size_t)(gz0+a)*D + py)*D + (gx0+b);   // idx z*N+x
        else            i = ((size_t)pz*D + (gy0+a))*D + (gx0+b);   // idx y*N+x
        buf[a*N + b] = (float)recon[i] - (float)orig[i];
    }
    return buf;
}

// Decode-side chunk filter over a whole volume: pristine copy as neighbor
// source, filtered result written to dst.
static void deblock_chunks(const uint8_t *src, uint8_t *dst, float strength) {
    uint8_t c[DCT3D_N3], nb[6][DCT3D_N3], o[DCT3D_N3];
    for (int bz = 0; bz < GB; ++bz)
    for (int by = 0; by < GB; ++by)
    for (int bx = 0; bx < GB; ++bx) {
        get_block(src, bz, by, bx, c);
        const uint8_t *p[6] = {0,0,0,0,0,0};
        if (bx > 0)      { get_block(src, bz, by, bx-1, nb[0]); p[0] = nb[0]; }
        if (bx < GB-1)   { get_block(src, bz, by, bx+1, nb[1]); p[1] = nb[1]; }
        if (by > 0)      { get_block(src, bz, by-1, bx, nb[2]); p[2] = nb[2]; }
        if (by < GB-1)   { get_block(src, bz, by+1, bx, nb[3]); p[3] = nb[3]; }
        if (bz > 0)      { get_block(src, bz-1, by, bx, nb[4]); p[4] = nb[4]; }
        if (bz < GB-1)   { get_block(src, bz+1, by, bx, nb[5]); p[5] = nb[5]; }
        dct3d_deblock_chunk_u8(c, p[0], p[1], p[2], p[3], p[4], p[5], 0.0f, strength, o);
        put_block(dst, bz, by, bx, o);
    }
}

int main(void) {
    uint8_t *orig = malloc(VOX), *volP = malloc(VOX), *volE = malloc(VOX);
    uint8_t *volD = malloc(VOX), *volB = malloc(VOX);
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < D; ++y)
    for (int x = 0; x < D; ++x) {
        double f = 128.0 + 60.0*sin(x*0.11 + 0.5)*cos(y*0.09) + 40.0*sin(z*0.07 + y*0.03);
        f = f < 0 ? 0 : f > 255 ? 255 : f;
        orig[((size_t)z*D + y)*D + x] = (uint8_t)(f + 0.5);
    }

    printf("q    variant      seam_E   MSE     bytes\n");
    float qs[] = {16, 24, 40, 64};
    for (int qi = 0; qi < 4; ++qi) {
        float q = qs[qi];
        uint8_t blob[DCT3D_MAX_BYTES], blk[DCT3D_N3], back[DCT3D_N3];
        size_t bytesP = 0, bytesE = 0;

        // --- plain pass ---
        for (int bz = 0; bz < GB; ++bz) for (int by = 0; by < GB; ++by) for (int bx = 0; bx < GB; ++bx) {
            get_block(orig, bz, by, bx, blk);
            size_t n = dct3d_encode_u8(blk, q, 0, 0, blob); bytesP += n;
            dct3d_decode_u8(blob, n, back);
            put_block(volP, bz, by, bx, back);
        }

        // --- encode-side checkerboard pass ---
        // even parity: identical to plain (copy). odd parity: re-encode with
        // even neighbors' errors injected.
        memcpy(volE, volP, VOX);
        for (int bz = 0; bz < GB; ++bz) for (int by = 0; by < GB; ++by) for (int bx = 0; bx < GB; ++bx) {
            int par = (bz + by + bx) & 1;
            if (!par) { get_block(orig, bz, by, bx, blk); bytesE += dct3d_encode_u8(blk, q, 0, 0, blob); continue; }
            float eb[6][N*N]; const float *ep[6];
            for (int f = 0; f < 6; ++f) ep[f] = face_err(volP, orig, bz, by, bx, f, eb[f]);
            get_block(orig, bz, by, bx, blk);
            size_t n = dct3d_encode_deblock_u8(blk, ep[0], ep[1], ep[2], ep[3], ep[4], ep[5],
                                               q, 0, 0, 1.0f, blob);
            bytesE += n;
            dct3d_decode_u8(blob, n, back);
            put_block(volE, bz, by, bx, back);
        }

        // --- decode-side on plain; both = decode-side on encode-side ---
        deblock_chunks(volP, volD, 1.0f);
        deblock_chunks(volE, volB, 1.0f);

        double sP = seam_energy(volP), sE = seam_energy(volE);
        double sD = seam_energy(volD), sB = seam_energy(volB);
        double eP = mse(orig, volP), eE = mse(orig, volE);
        double eD = mse(orig, volD), eB = mse(orig, volB);
        printf("%-4.0f plain       %7.2f  %6.3f  %6zu\n", q, sP, eP, bytesP);
        printf("     enc-side    %7.2f  %6.3f  %6zu   (%.0f%% seam cut)\n", sE, eE, bytesE, 100*(sP-sE)/sP);
        printf("     dec-side    %7.2f  %6.3f     -     (%.0f%% seam cut)\n", sD, eD, 100*(sP-sD)/sP);
        printf("     both        %7.2f  %6.3f     -     (%.0f%% seam cut)\n", sB, eB, 100*(sP-sB)/sP);

        CHECK(sE < sP, "q=%.0f: enc-side did not reduce seam energy", q);
        CHECK(sB <= sD && sB <= sE, "q=%.0f: combined not best", q);
        // Encode-side trades a bounded MSE increase (deliberately shifted
        // boundary layers) for seam continuity; the dial is `strength`.
        CHECK(eE < eP * 1.30, "q=%.0f: enc-side MSE beyond tradeoff bound (%.3f vs %.3f)", q, eE, eP);
        CHECK(bytesE < bytesP * 1.10, "q=%.0f: enc-side rate blew up", q);
    }

    // --- chunk-API edge guard: a genuine step on a block boundary survives ---
    for (size_t i = 0; i < VOX; ++i) orig[i] = (int)(i % D) < 2*N ? 40 : 200;
    deblock_chunks(orig, volD, 1.0f);
    double edge_min = 1e9;
    for (int z = 0; z < D; ++z) for (int y = 0; y < D; ++y) {
        size_t i = ((size_t)z*D + y)*D + 2*N;
        double step = (double)volD[i] - volD[i-1];
        if (step < edge_min) edge_min = step;
    }
    printf("chunk edge guard: min step %.0f (true 160)\n", edge_min);
    CHECK(edge_min > 144, "chunk filter smoothed a genuine edge");

    free(orig); free(volP); free(volE); free(volD); free(volB);
    printf(fails ? "\n%d CHECK(s) FAILED\n" : "\nall predeblock tests passed\n", fails);
    return fails ? 1 : 0;
}
