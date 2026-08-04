// deblock.c — implementation. See deblock.h.
//
// Signal-adaptive boundary deblocking. Chunk API (center + 6 decoded neighbors
// -> new center) and volume API (dense assembled ZYX array, in place). The math
// runs in float; only the two voxel layers nearest each seam are touched, in a
// JPEG/H.26x "weak filter" shape.

#include "deblock.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#define N DCT3D_N   // block size / seam spacing (16)
#define IDX(z, y, x) ((((size_t)(z)) * N + (size_t)(y)) * N + (size_t)(x))

// ---------------------------------------------------------------------------
// One boundary sample. Given the four straddling values p1,p0 | q0,q1 (p on the
// low-coordinate side of the seam), decide whether the p0|q0 step is a coding
// seam and, if so, return the signed nudge to apply to p0 (q0 gets the opposite;
// p1/q1 get *d1, also opposite-signed across the seam). Real edges (large step,
// or activity on either side) yield 0.
//
//   alpha  max |p0-q0| still considered a seam (scaled by range*strength)
//   beta   max on-side activity |p1-p0|,|q1-q0| for a side to count as "flat"
static inline float filt_delta(float p1, float p0, float q0, float q1,
                               float alpha, float beta, float *d1) {
    *d1 = 0.0f;
    float d = q0 - p0;
    float ad = fabsf(d);
    if (ad >= alpha || ad == 0.0f) return 0.0f;         // real edge / no step
    if (fabsf(p1 - p0) >= beta) return 0.0f;            // activity on p side
    if (fabsf(q1 - q0) >= beta) return 0.0f;            // activity on q side
    // Move p0/q0 toward the seam midpoint and the outer pair a fraction of
    // that, giving a monotone four-tap ramp p1<=p0<=mid<=q0<=q1.
    float d0 = 0.40f * d;
    // Taper the correction to zero as the step approaches alpha, so there is no
    // discontinuity in behavior at the edge/seam decision boundary.
    float w = 1.0f - ad / alpha;   // in (0,1]
    d0 *= w;
    *d1 = 0.4f * d0;
    return d0;
}

// Threshold model. Preferred: proportional to the quantizer step (`step` > 0)
// — quantization seams scale with the quantizer while real edges do not, so
// step-relative thresholds separate them on textured data (the H.26x
// QP-threshold model). KA bounds the seam step, KB the on-side activity;
// overridable at compile time for tuning sweeps. Fallback (`step` <= 0):
// fractions of the data range — adequate only for smooth content.
#ifndef DCT3D_DEBLOCK_KA
#define DCT3D_DEBLOCK_KA 2.0f
#endif
#ifndef DCT3D_DEBLOCK_KB
#define DCT3D_DEBLOCK_KB 1.2f
#endif
static inline void thresholds(float step, float range, float strength,
                              float *alpha, float *beta) {
    if (step > 0.0f) {
        *alpha = DCT3D_DEBLOCK_KA * step * strength;
        *beta  = DCT3D_DEBLOCK_KB * step * strength;
    } else {
        *alpha = 0.06f * range * strength;
        *beta  = 0.04f * range * strength;
    }
}

// ============================================================================
// Chunk API. Center-side-only application: for each neighbored face, compute
// the same filt_delta both chunks would agree on (pristine values, union
// range) and apply only the center's half of the correction. Filtering every
// chunk against pristine neighbors therefore heals both sides of every seam
// consistently, in any order, in parallel.
// ============================================================================
#define MINMAX_BODY(T)                                                          \
    static inline void minmax_##T(const T *c, float *mn, float *mx) {           \
        float lo = (float)c[0], hi = lo;                                        \
        for (size_t i = 1; i < (size_t)DCT3D_N3; ++i) {                         \
            float v = (float)c[i];                                              \
            if (v < lo) lo = v; if (v > hi) hi = v;                             \
        }                                                                       \
        *mn = lo; *mx = hi;                                                     \
    }
MINMAX_BODY(uint8_t)  MINMAX_BODY(uint16_t) MINMAX_BODY(uint32_t)
MINMAX_BODY(int8_t)   MINMAX_BODY(int16_t)  MINMAX_BODY(int32_t)
MINMAX_BODY(float)
#undef MINMAX_BODY

// Per-face loop. Reads the neighbor pristine and the center side from `out`
// (running values, so corrections at chunk edges compose like the volume
// pass), writes only the center's two layers.
//
// CLO/CHI: center voxel indices at depth 0/1 from the face.
// NLO/NHI: neighbor voxel indices at depth 0/1 from the face (its far side).
// SIGN: +1 when the center is the q side (lo faces), -1 when the p side.
#define FACE_LOOP(T, NBR, CD0, CD1, ND0, ND1, SIGN, LO, HI, ROUND)              \
    do {                                                                        \
        float range = 1.0f;                                                     \
        if (!(step > 0.0f)) {   /* range only feeds the fallback thresholds */  \
            float nmn, nmx; minmax_##T(NBR, &nmn, &nmx);                        \
            float mn = cmn < nmn ? cmn : nmn, mx = cmx > nmx ? cmx : nmx;       \
            range = mx - mn;                                                    \
        }                                                                       \
        if (range > 0.0f) {                                                     \
            float alpha, beta; thresholds(step, range, strength, &alpha, &beta); \
            for (int a = 0; a < N; ++a)                                         \
            for (int b = 0; b < N; ++b) {                                       \
                float n0 = (float)NBR[ND0], n1 = (float)NBR[ND1];               \
                float c0 = (float)out[CD0], c1 = (float)out[CD1];               \
                /* orient as p (low side) | q (high side) */                    \
                float p1, p0, q0, q1;                                           \
                if ((SIGN) > 0) { p1 = n1; p0 = n0; q0 = c0; q1 = c1; }         \
                else            { p1 = c1; p0 = c0; q0 = n0; q1 = n1; }         \
                float d1w; float d0w = filt_delta(p1, p0, q0, q1, alpha, beta, &d1w); \
                if (d0w == 0.0f) continue;                                      \
                float v0, v1;                                                   \
                if ((SIGN) > 0) { v0 = c0 - d0w; v1 = c1 - d1w; }               \
                else            { v0 = c0 + d0w; v1 = c1 + d1w; }               \
                out[CD0] = (T)ROUND(v0 < (float)(LO) ? (float)(LO) : v0 > (float)(HI) ? (float)(HI) : v0); \
                out[CD1] = (T)ROUND(v1 < (float)(LO) ? (float)(LO) : v1 > (float)(HI) ? (float)(HI) : v1); \
            }                                                                   \
        }                                                                       \
    } while (0)

#define DEBLOCK_CHUNK_BODY(T, LO, HI, ROUND)                                    \
    do {                                                                        \
        memcpy(out, chunk, sizeof(T) * (size_t)DCT3D_N3);                       \
        if (!(strength > 0.0f)) return;                                         \
        float cmn = 0.0f, cmx = 0.0f;                                           \
        if (!(step > 0.0f)) minmax_##T(chunk, &cmn, &cmx);                      \
        /* a,b loop variables map to the two in-plane axes of each face */      \
        if (xlo) FACE_LOOP(T, xlo, IDX(a,b,0),   IDX(a,b,1),   IDX(a,b,N-1), IDX(a,b,N-2), +1, LO, HI, ROUND); \
        if (xhi) FACE_LOOP(T, xhi, IDX(a,b,N-1), IDX(a,b,N-2), IDX(a,b,0),   IDX(a,b,1),   -1, LO, HI, ROUND); \
        if (ylo) FACE_LOOP(T, ylo, IDX(a,0,b),   IDX(a,1,b),   IDX(a,N-1,b), IDX(a,N-2,b), +1, LO, HI, ROUND); \
        if (yhi) FACE_LOOP(T, yhi, IDX(a,N-1,b), IDX(a,N-2,b), IDX(a,0,b),   IDX(a,1,b),   -1, LO, HI, ROUND); \
        if (zlo) FACE_LOOP(T, zlo, IDX(0,a,b),   IDX(1,a,b),   IDX(N-1,a,b), IDX(N-2,a,b), +1, LO, HI, ROUND); \
        if (zhi) FACE_LOOP(T, zhi, IDX(N-1,a,b), IDX(N-2,a,b), IDX(0,a,b),   IDX(1,a,b),   -1, LO, HI, ROUND); \
    } while (0)

// ============================================================================
// Volume API. Same filter over every internal 16-aligned plane, in place.
// ============================================================================
#define DEBLOCK_VOL_BODY(T, LO, HI, ROUND)                                      \
    do {                                                                        \
        size_t nz = dims[0], ny = dims[1], nx = dims[2];                        \
        if (nz == 0 || ny == 0 || nx == 0) return;                             \
        if (!(strength > 0.0f)) return;                                        \
                                                                                \
        /* Fallback mode only: data range over a coarse subsample. */          \
        float range = 1.0f;                                                     \
        if (!(step > 0.0f)) {                                                   \
            float vmin = (float)vol[0], vmax = (float)vol[0];                   \
            size_t total = nz * ny * nx;                                        \
            size_t stepc = total / 4096; if (stepc == 0) stepc = 1;             \
            for (size_t i = 0; i < total; i += stepc) {                         \
                float v = (float)vol[i];                                        \
                if (v < vmin) vmin = v; if (v > vmax) vmax = v;                 \
            }                                                                   \
            range = vmax - vmin; if (!(range > 0.0f)) return;                   \
        }                                                                       \
        float alpha, beta; thresholds(step, range, strength, &alpha, &beta);    \
        const float lo = (float)(LO), hi = (float)(HI);                        \
                                                                                \
        /* --- x-normal seams: planes at x = N, 2N, ... --- */                 \
        for (size_t z = 0; z < nz; ++z)                                        \
        for (size_t y = 0; y < ny; ++y) {                                      \
            T *row = vol + (z * ny + y) * nx;                                   \
            for (size_t x = N; x + 1 < nx; x += N) {                           \
                if (x < 2) continue;                                          \
                float p1 = (float)row[x - 2], p0 = (float)row[x - 1];          \
                float q0 = (float)row[x],     q1 = (float)row[x + 1];          \
                float d1; float d0 = filt_delta(p1, p0, q0, q1, alpha, beta, &d1); \
                if (d0 == 0.0f) continue;                                     \
                float np0 = p0 + d0, nq0 = q0 - d0;                            \
                float np1 = p1 + d1, nq1 = q1 - d1;                            \
                row[x - 1] = (T)ROUND(np0 < lo ? lo : np0 > hi ? hi : np0);    \
                row[x]     = (T)ROUND(nq0 < lo ? lo : nq0 > hi ? hi : nq0);    \
                row[x - 2] = (T)ROUND(np1 < lo ? lo : np1 > hi ? hi : np1);    \
                row[x + 1] = (T)ROUND(nq1 < lo ? lo : nq1 > hi ? hi : nq1);    \
            }                                                                   \
        }                                                                       \
        /* --- y-normal seams: planes at y = N, 2N, ... --- */                 \
        for (size_t z = 0; z < nz; ++z)                                        \
        for (size_t y = N; y + 1 < ny; y += N) {                              \
            if (y < 2) continue;                                              \
            for (size_t x = 0; x < nx; ++x) {                                  \
                T *col = vol + (z * ny) * nx + x;                              \
                float p1 = (float)col[(y - 2) * nx], p0 = (float)col[(y - 1) * nx]; \
                float q0 = (float)col[y * nx],       q1 = (float)col[(y + 1) * nx]; \
                float d1; float d0 = filt_delta(p1, p0, q0, q1, alpha, beta, &d1); \
                if (d0 == 0.0f) continue;                                     \
                float np0 = p0 + d0, nq0 = q0 - d0;                            \
                float np1 = p1 + d1, nq1 = q1 - d1;                            \
                col[(y - 1) * nx] = (T)ROUND(np0 < lo ? lo : np0 > hi ? hi : np0); \
                col[y * nx]       = (T)ROUND(nq0 < lo ? lo : nq0 > hi ? hi : nq0); \
                col[(y - 2) * nx] = (T)ROUND(np1 < lo ? lo : np1 > hi ? hi : np1); \
                col[(y + 1) * nx] = (T)ROUND(nq1 < lo ? lo : nq1 > hi ? hi : nq1); \
            }                                                                   \
        }                                                                       \
        /* --- z-normal seams: planes at z = N, 2N, ... --- */                 \
        for (size_t z = N; z + 1 < nz; z += N) {                              \
            if (z < 2) continue;                                              \
            for (size_t y = 0; y < ny; ++y)                                    \
            for (size_t x = 0; x < nx; ++x) {                                  \
                T *col = vol + (y * nx) + x;                                   \
                size_t pz = ny * nx;                                          \
                float p1 = (float)col[(z - 2) * pz], p0 = (float)col[(z - 1) * pz]; \
                float q0 = (float)col[z * pz],       q1 = (float)col[(z + 1) * pz]; \
                float d1; float d0 = filt_delta(p1, p0, q0, q1, alpha, beta, &d1); \
                if (d0 == 0.0f) continue;                                     \
                float np0 = p0 + d0, nq0 = q0 - d0;                            \
                float np1 = p1 + d1, nq1 = q1 - d1;                            \
                col[(z - 1) * pz] = (T)ROUND(np0 < lo ? lo : np0 > hi ? hi : np0); \
                col[z * pz]       = (T)ROUND(nq0 < lo ? lo : nq0 > hi ? hi : nq0); \
                col[(z - 2) * pz] = (T)ROUND(np1 < lo ? lo : np1 > hi ? hi : np1); \
                col[(z + 1) * pz] = (T)ROUND(nq1 < lo ? lo : nq1 > hi ? hi : nq1); \
            }                                                                   \
        }                                                                       \
    } while (0)

#define IDENTITY(x) (x)

#define DEFINE_DEBLOCK(T, name, LO, HI, ROUND)                                  \
    void dct3d_deblock_chunk_##name(const T *chunk,                             \
                                    const T *xlo, const T *xhi,                 \
                                    const T *ylo, const T *yhi,                 \
                                    const T *zlo, const T *zhi,                 \
                                    float step, float strength, T *out) {      \
        DEBLOCK_CHUNK_BODY(T, LO, HI, ROUND);                                   \
    }                                                                           \
    void dct3d_deblock_##name(T *vol, const size_t dims[3], float step,         \
                              float strength) {                                 \
        DEBLOCK_VOL_BODY(T, LO, HI, ROUND);                                     \
    }

DEFINE_DEBLOCK(uint8_t,  u8,  0, 255, roundf)
DEFINE_DEBLOCK(uint16_t, u16, 0, 65535, roundf)
DEFINE_DEBLOCK(uint32_t, u32, 0, 4294967040.0, roundf)
DEFINE_DEBLOCK(int8_t,   s8,  -128, 127, roundf)
DEFINE_DEBLOCK(int16_t,  s16, -32768, 32767, roundf)
DEFINE_DEBLOCK(int32_t,  s32, -2147483648.0, 2147483520.0, roundf)
DEFINE_DEBLOCK(float,    f32, -FLT_MAX, FLT_MAX, IDENTITY)
