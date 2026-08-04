// lapped.c — implementation. See lapped.h.
//
// Per boundary plane, per line crossing it, the symmetric pair at distance i
// from the seam (i = 0 nearest) folds into sum/difference; the difference is
// scaled by s_i (pre) or 1/s_i (post):
//     a = lo + hi;  b = (lo - hi) * s;  lo' = (a+b)/2;  hi' = (a-b)/2
// For 0 < s < 1 both outputs are convex combinations of the inputs, so the
// PRE-filter is range-preserving (no clamp needed even for u8); the POST
// filter amplifies the difference and can locally overshoot on quantization
// noise, so integer paths clamp on store. Pairs of adjacent boundaries are
// disjoint (2*DCT3D_LAP_M <= DCT3D_N), and the three axis passes act on
// different tensor indices, so they commute — the inverse works in any axis
// order and in place.

#include "lapped.h"

#include <math.h>
#include <stddef.h>

#define N DCT3D_N
#define M DCT3D_LAP_M

// Antisymmetric-half scales, boundary-nearest first. s < 1 compresses the
// cross-boundary difference (that is both the coding gain and the deblocking);
// values in the LBT/TDLT ballpark, overridable for tuning sweeps.
#ifndef DCT3D_LAP_SCALES
#define DCT3D_LAP_SCALES { 0.55f, 0.75f, 0.90f, 0.97f }
#endif
static const float k_s[M] = DCT3D_LAP_SCALES;

// One pair, one direction. dir=+1 pre (multiply by s), dir=-1 post (divide).
static inline void lap_pair(float *lo, float *hi, float s, int dir) {
    float a = *lo + *hi;
    float b = *lo - *hi;
    b = dir > 0 ? b * s : b / s;
    *lo = 0.5f * (a + b);
    *hi = 0.5f * (a - b);
}

// The three axis passes share one body; only the stride between the two sides
// of a seam and the iteration bounds differ. ROUND/LO/HI handle the dtype.
#define LAP_BODY(T, LO, HI, ROUND, DIR)                                        \
    do {                                                                       \
        size_t nz = dims[0], ny = dims[1], nx = dims[2];                       \
        if (nz == 0 || ny == 0 || nx == 0) return;                            \
        const float lo_c = (float)(LO), hi_c = (float)(HI);                    \
        /* x-normal planes */                                                  \
        for (size_t z = 0; z < nz; ++z)                                        \
        for (size_t y = 0; y < ny; ++y) {                                      \
            T *row = vol + (z * ny + y) * nx;                                  \
            for (size_t c = N; c < nx; c += N) {                               \
                size_t depth = c < nx - c ? c : nx - c;                        \
                size_t m = depth < M ? depth : M;                              \
                for (size_t i = 0; i < m; ++i) {                               \
                    float a = (float)row[c - 1 - i], b = (float)row[c + i];    \
                    lap_pair(&a, &b, k_s[i], DIR);                             \
                    a = ROUND(a); b = ROUND(b);                                \
                    row[c - 1 - i] = (T)(a < lo_c ? lo_c : a > hi_c ? hi_c : a); \
                    row[c + i]     = (T)(b < lo_c ? lo_c : b > hi_c ? hi_c : b); \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* y-normal planes */                                                  \
        for (size_t z = 0; z < nz; ++z)                                        \
        for (size_t c = N; c < ny; c += N) {                                   \
            size_t depth = c < ny - c ? c : ny - c;                            \
            size_t m = depth < M ? depth : M;                                  \
            for (size_t x = 0; x < nx; ++x) {                                  \
                T *base = vol + (z * ny) * nx + x;                             \
                for (size_t i = 0; i < m; ++i) {                               \
                    float a = (float)base[(c - 1 - i) * nx];                   \
                    float b = (float)base[(c + i) * nx];                       \
                    lap_pair(&a, &b, k_s[i], DIR);                             \
                    a = ROUND(a); b = ROUND(b);                                \
                    base[(c - 1 - i) * nx] = (T)(a < lo_c ? lo_c : a > hi_c ? hi_c : a); \
                    base[(c + i) * nx]     = (T)(b < lo_c ? lo_c : b > hi_c ? hi_c : b); \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* z-normal planes */                                                  \
        for (size_t c = N; c < nz; c += N) {                                   \
            size_t depth = c < nz - c ? c : nz - c;                            \
            size_t m = depth < M ? depth : M;                                  \
            size_t pz = ny * nx;                                               \
            for (size_t y = 0; y < ny; ++y)                                    \
            for (size_t x = 0; x < nx; ++x) {                                  \
                T *base = vol + y * nx + x;                                    \
                for (size_t i = 0; i < m; ++i) {                               \
                    float a = (float)base[(c - 1 - i) * pz];                   \
                    float b = (float)base[(c + i) * pz];                       \
                    lap_pair(&a, &b, k_s[i], DIR);                             \
                    a = ROUND(a); b = ROUND(b);                                \
                    base[(c - 1 - i) * pz] = (T)(a < lo_c ? lo_c : a > hi_c ? hi_c : a); \
                    base[(c + i) * pz]     = (T)(b < lo_c ? lo_c : b > hi_c ? hi_c : b); \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

#define IDENTITY(x) (x)

#define DEFINE_LAP(T, name, LO, HI, ROUND)                                     \
    void dct3d_lap_prefilter_##name(T *vol, const size_t dims[3]) {            \
        LAP_BODY(T, LO, HI, ROUND, +1);                                        \
    }                                                                          \
    void dct3d_lap_postfilter_##name(T *vol, const size_t dims[3]) {           \
        LAP_BODY(T, LO, HI, ROUND, -1);                                        \
    }

DEFINE_LAP(uint8_t,  u8,  0, 255, roundf)
DEFINE_LAP(uint16_t, u16, 0, 65535, roundf)
DEFINE_LAP(float,    f32, -3.4e38f, 3.4e38f, IDENTITY)
