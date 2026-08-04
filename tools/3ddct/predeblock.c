// predeblock.c — implementation. See predeblock.h.
//
// Pure source pre-conditioning wrapped around the ordinary encoder: add the
// neighbor's reconstruction error to this chunk's boundary layers with an
// inward taper, then hand the shifted chunk to dct3d_encode_<dtype>. The taper
// spreads the injected shift across a few layers so its energy sits in low
// frequencies that survive coarse quantization (a 1-layer spike would mostly
// die in the dead zone at exactly the qualities where blocking is worst).

#include "predeblock.h"

#include <math.h>
#include <string.h>

#define N  DCT3D_N
#define N3 DCT3D_N3
#define IDX(z, y, x) ((((size_t)(z)) * N + (size_t)(y)) * N + (size_t)(x))

// Inward taper: weight of the injected face error at depth d from the face.
#define TAPER_DEPTH 4
static const float k_taper[TAPER_DEPTH] = { 1.0f, 0.70f, 0.40f, 0.18f };

// Accumulate one face's tapered error field into the float working chunk.
// CD(d): center voxel index at depth d from the face, for in-plane coords a,b.
// EIDX: index into the face error plane for a,b.
#define INJECT_FACE(ERR, CD, EIDX)                                            \
    do {                                                                      \
        for (int d = 0; d < TAPER_DEPTH; ++d) {                               \
            float wd = k_taper[d] * strength;                                 \
            for (int a = 0; a < N; ++a)                                       \
            for (int b = 0; b < N; ++b)                                       \
                mod[CD] += wd * (ERR)[EIDX];                                  \
        }                                                                     \
    } while (0)

#define PREDEBLOCK_BODY(T, name, LO, HI, ROUND)                               \
    do {                                                                      \
        int any = (err_xlo || err_xhi || err_ylo || err_yhi ||                \
                   err_zlo || err_zhi);                                       \
        if (!any || !(strength > 0.0f))                                       \
            return dct3d_encode_##name(chunk, quality, max_error, tau, out);  \
        float mod[N3];                                                        \
        for (int i = 0; i < N3; ++i) mod[i] = (float)chunk[i];                \
        if (err_xlo) INJECT_FACE(err_xlo, IDX(a, b, d),         a * N + b);   \
        if (err_xhi) INJECT_FACE(err_xhi, IDX(a, b, N - 1 - d), a * N + b);   \
        if (err_ylo) INJECT_FACE(err_ylo, IDX(a, d, b),         a * N + b);   \
        if (err_yhi) INJECT_FACE(err_yhi, IDX(a, N - 1 - d, b), a * N + b);   \
        if (err_zlo) INJECT_FACE(err_zlo, IDX(d, a, b),         a * N + b);   \
        if (err_zhi) INJECT_FACE(err_zhi, IDX(N - 1 - d, a, b), a * N + b);   \
        T shifted[N3];                                                        \
        for (int i = 0; i < N3; ++i) {                                        \
            float v = ROUND(mod[i]);                                          \
            v = v < (float)(LO) ? (float)(LO) : v > (float)(HI) ? (float)(HI) : v; \
            shifted[i] = (T)v;                                                \
        }                                                                     \
        return dct3d_encode_##name(shifted, quality, max_error, tau, out);    \
    } while (0)

#define IDENTITY(x) (x)

size_t dct3d_encode_deblock_u8(const uint8_t *chunk,
    const float *err_xlo, const float *err_xhi,
    const float *err_ylo, const float *err_yhi,
    const float *err_zlo, const float *err_zhi,
    float quality, float max_error, float tau,
    float strength, uint8_t *out) {
    PREDEBLOCK_BODY(uint8_t, u8, 0, 255, roundf);
}

size_t dct3d_encode_deblock_u16(const uint16_t *chunk,
    const float *err_xlo, const float *err_xhi,
    const float *err_ylo, const float *err_yhi,
    const float *err_zlo, const float *err_zhi,
    float quality, float max_error, float tau,
    float strength, uint8_t *out) {
    PREDEBLOCK_BODY(uint16_t, u16, 0, 65535, roundf);
}

size_t dct3d_encode_deblock_f32(const float *chunk,
    const float *err_xlo, const float *err_xhi,
    const float *err_ylo, const float *err_yhi,
    const float *err_zlo, const float *err_zhi,
    float quality, float max_error, float tau,
    float strength, uint8_t *out) {
    PREDEBLOCK_BODY(float, f32, -3.4e38f, 3.4e38f, IDENTITY);
}
