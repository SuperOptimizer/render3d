// eval_real.c — deblocking evaluation on a real CT chunk (128^3 raw u8).
//
// Usage: eval_real <raw128.file> <outdir>
// Runs the export-ladder settings on real scroll data across the four
// variants (plain | enc-side | dec-side | both), prints metrics, and writes
// mid-Z slice PGMs into <outdir> for visual judgment.
//
// Get a real 128^3 chunk (raw u8, 2 MiB — one uncompressed zarr chunk of the
// PHerc0332 2.4um open-data volume, mid-scroll so it is dense papyrus):
//   curl -o real128.raw https://vesuvius-challenge-open-data.s3.amazonaws.com/\
//     PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr/0/131/61/61

#include "dct3d.h"
#include "deblock.h"
#include "lapped.h"
#include "predeblock.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N DCT3D_N
#define D 128
#define GB (D / N)          // 8 blocks per axis
#define VOX ((size_t)D * D * D)

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
// Baseline discontinuity of the same planes in the ORIGINAL — real data has
// texture, so report seam energy as excess over this.
static double mse(const uint8_t *a, const uint8_t *b) {
    double se = 0;
    for (size_t i = 0; i < VOX; ++i) { double d = (double)a[i] - b[i]; se += d*d; }
    return se / VOX;
}
static double psnr(double m) { return m > 0 ? 10.0 * log10(255.0*255.0 / m) : 99.0; }

static void get_block(const uint8_t *v, int bz, int by, int bx, uint8_t *blk) {
    for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x)
        blk[(z*N + y)*N + x] = v[((size_t)(bz*N+z)*D + (by*N+y))*D + (bx*N+x)];
}
static void put_block(uint8_t *v, int bz, int by, int bx, const uint8_t *blk) {
    for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x)
        v[((size_t)(bz*N+z)*D + (by*N+y))*D + (bx*N+x)] = blk[(z*N + y)*N + x];
}

static const float *face_err(const uint8_t *recon, const uint8_t *orig,
                             int bz, int by, int bx, int f, float *buf) {
    int gz0 = bz*N, gy0 = by*N, gx0 = bx*N;
    int px = 0, py = 0, pz = 0;
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
        if (f < 2)      i = ((size_t)(gz0+a)*D + (gy0+b))*D + px;
        else if (f < 4) i = ((size_t)(gz0+a)*D + py)*D + (gx0+b);
        else            i = ((size_t)pz*D + (gy0+a))*D + (gx0+b);
        buf[a*N + b] = (float)recon[i] - (float)orig[i];
    }
    return buf;
}

static void deblock_chunks(const uint8_t *src, uint8_t *dst, float step, float strength) {
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
        dct3d_deblock_chunk_u8(c, p[0], p[1], p[2], p[3], p[4], p[5], step, strength, o);
        put_block(dst, bz, by, bx, o);
    }
}

static void write_pgm(const char *dir, const char *name, const uint8_t *v, int zslice) {
    char path[512]; snprintf(path, sizeof path, "%s/%s.pgm", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fprintf(f, "P5\n%d %d\n255\n", D, D);
    fwrite(v + (size_t)zslice * D * D, 1, (size_t)D * D, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <raw 128^3 u8> <outdir>\n", argv[0]); return 2; }
    uint8_t *orig = malloc(VOX);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(orig, 1, VOX, f) != VOX) { fprintf(stderr, "bad input\n"); return 2; }
    fclose(f);
    const char *outdir = argv[2];

    // data stats
    {
        long hist_nz = 0; double sum = 0; int mn = 255, mx = 0;
        for (size_t i = 0; i < VOX; ++i) {
            int v = orig[i]; sum += v; hist_nz += v > 0;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        printf("data: min=%d max=%d mean=%.1f nonzero=%.1f%%  orig seam-plane E=%.2f\n",
               mn, mx, sum / VOX, 100.0 * hist_nz / VOX, seam_energy(orig));
    }

    uint8_t *volP = malloc(VOX), *volE = malloc(VOX), *volD = malloc(VOX), *volB = malloc(VOX);
    uint8_t *volL = malloc(VOX), *pref = malloc(VOX);
    uint8_t blob[DCT3D_MAX_BYTES], blk[DCT3D_N3], back[DCT3D_N3];
    const size_t vdims[3] = { D, D, D };

    // export ladder points + a harsher one, and an enc-strength sweep at q=32.
    struct { float q, tau, encs, decs; const char *tag; } runs[] = {
        { 32, 64, 1.0f, 1.0f, "q32t64" },       // the 2.4um export setting
        { 64, 128, 1.0f, 1.0f, "q64t128" },     // the 1.1um export setting
        { 64, 0, 1.0f, 1.0f, "q64pure" },       // no correction pass: worst blocking
        { 32, 64, 0.5f, 1.0f, "q32s05" },       // enc strength 0.5
        { 32, 64, 0.75f, 1.0f, "q32s075" },     // enc strength 0.75
    };
    printf("\nrun       variant      seamE   excess   MSE     PSNR   bytes\n");
    for (size_t r = 0; r < sizeof runs / sizeof runs[0]; ++r) {
        float q = runs[r].q, tau = runs[r].tau, es = runs[r].encs, ds = runs[r].decs;
        size_t bytesP = 0, bytesE = 0, bytesPodd = 0;
        for (int bz = 0; bz < GB; ++bz) for (int by = 0; by < GB; ++by) for (int bx = 0; bx < GB; ++bx) {
            get_block(orig, bz, by, bx, blk);
            size_t n = dct3d_encode_u8(blk, q, 0, tau, blob); bytesP += n;
            if ((bz + by + bx) & 1) bytesPodd += n;
            dct3d_decode_u8(blob, n, back);
            put_block(volP, bz, by, bx, back);
        }
        memcpy(volE, volP, VOX);
        for (int bz = 0; bz < GB; ++bz) for (int by = 0; by < GB; ++by) for (int bx = 0; bx < GB; ++bx) {
            if (!((bz + by + bx) & 1)) continue;
            float eb[6][N*N]; const float *ep[6];
            for (int fc = 0; fc < 6; ++fc) ep[fc] = face_err(volP, orig, bz, by, bx, fc, eb[fc]);
            get_block(orig, bz, by, bx, blk);
            size_t n = dct3d_encode_deblock_u8(blk, ep[0], ep[1], ep[2], ep[3], ep[4], ep[5],
                                               q, 0, tau, es, blob);
            bytesE += n;
            dct3d_decode_u8(blob, n, back);
            put_block(volE, bz, by, bx, back);
        }
        deblock_chunks(volP, volD, q, ds);
        deblock_chunks(volE, volB, q, ds);

        // lapped: prefilter source -> plain per-block codec -> postfilter
        size_t bytesL = 0;
        memcpy(pref, orig, VOX);
        dct3d_lap_prefilter_u8(pref, vdims);
        for (int bz = 0; bz < GB; ++bz) for (int by = 0; by < GB; ++by) for (int bx = 0; bx < GB; ++bx) {
            get_block(pref, bz, by, bx, blk);
            size_t n = dct3d_encode_u8(blk, q, 0, tau, blob); bytesL += n;
            dct3d_decode_u8(blob, n, back);
            put_block(volL, bz, by, bx, back);
        }
        dct3d_lap_postfilter_u8(volL, vdims);

        double s0 = seam_energy(orig);
        double sP = seam_energy(volP), sE = seam_energy(volE), sD = seam_energy(volD), sB = seam_energy(volB);
        double sL = seam_energy(volL);
        double eP = mse(orig, volP), eE = mse(orig, volE), eD = mse(orig, volD), eB = mse(orig, volB);
        double eL = mse(orig, volL);
        printf("%-9s plain       %6.2f  %6.2f  %6.2f  %5.2f  %7zu\n", runs[r].tag, sP, sP - s0, eP, psnr(eP), bytesP);
        printf("          enc-side    %6.2f  %6.2f  %6.2f  %5.2f  %7zu (odd: %zu vs plain %zu)\n", sE, sE - s0, eE, psnr(eE), bytesP - bytesPodd + bytesE, bytesE, bytesPodd);
        printf("          dec-side    %6.2f  %6.2f  %6.2f  %5.2f\n", sD, sD - s0, eD, psnr(eD));
        printf("          both        %6.2f  %6.2f  %6.2f  %5.2f\n", sB, sB - s0, eB, psnr(eB));
        printf("          lapped      %6.2f  %6.2f  %6.2f  %5.2f  %7zu\n", sL, sL - s0, eL, psnr(eL), bytesL);

        if (r < 3) {   // slice dumps for the ladder points
            char nm[64]; int zs = 64;
            snprintf(nm, sizeof nm, "%s_orig", runs[r].tag);  write_pgm(outdir, nm, orig, zs);
            snprintf(nm, sizeof nm, "%s_plain", runs[r].tag); write_pgm(outdir, nm, volP, zs);
            snprintf(nm, sizeof nm, "%s_enc", runs[r].tag);   write_pgm(outdir, nm, volE, zs);
            snprintf(nm, sizeof nm, "%s_dec", runs[r].tag);   write_pgm(outdir, nm, volD, zs);
            snprintf(nm, sizeof nm, "%s_both", runs[r].tag);  write_pgm(outdir, nm, volB, zs);
            snprintf(nm, sizeof nm, "%s_lap", runs[r].tag);   write_pgm(outdir, nm, volL, zs);
        }
    }
    free(orig); free(volP); free(volE); free(volD); free(volB); free(volL); free(pref);
    return 0;
}
