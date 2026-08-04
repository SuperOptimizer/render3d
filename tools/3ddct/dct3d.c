// dct3d.c — implementation. See dct3d.h.
//
// Pipeline (encode):  load->float | normalize | subtract mean | 3D DCT |
//                     dead-zone quant | range-code levels | optional corrections.
// Pipeline (decode):  range-decode levels | dequant | 3D inverse DCT |
//                     add mean | denormalize | round/clamp->dtype | corrections.
//
// All value/transform/quant/normalization math is float. Integers appear only
// where they must: the binary range coder (a bit-exact arithmetic coder whose
// low/range renormalization is integer by definition), byte serialization, and
// loop/array indices.

#include "dct3d.h"

#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <string.h>

#define N        DCT3D_N
#define N3       DCT3D_N3
#define DZ_FRAC  0.80f     // dead-zone width as a fraction of the step
#define HF_EXP   0.65f     // high-frequency quantizer power-law exponent

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Orthonormal separable DCT-II cosine matrix, built on first use.
//   cm[k][n] = ck * cos(pi*(2n+1)*k / 2N),  ck = sqrt(1/N) (k=0) else sqrt(2/N)
// Orthonormal, so the inverse (DCT-III) is the transpose. Concurrent first
// callers race only to write identical values, but the ready flag needs a
// release store / acquire load pair: on weakly-ordered ISAs (ARM) a plain flag
// could be observed set before the table stores land, letting a reader see
// zeros. Still lock-free — no mutex, just publish ordering.
// ============================================================================
static float g_cm[N][N];
static atomic_int g_cm_ready = 0;

static void dct_init(void) {
    if (atomic_load_explicit(&g_cm_ready, memory_order_acquire)) return;
    for (int k = 0; k < N; ++k) {
        double ck = (k == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
        for (int n = 0; n < N; ++n)
            g_cm[k][n] = (float)(ck * cos(M_PI * (2.0 * n + 1.0) * k / (2.0 * N)));
    }
    atomic_store_explicit(&g_cm_ready, 1, memory_order_release);
}

// 1D forward DCT-II of one length-16 line, even/odd butterfly (half the MACs).
static inline void dct1d_fwd(const float *restrict in, float *restrict out) {
    const int S = N, H = S / 2;
    float s[N / 2], d[N / 2];
    for (int n = 0; n < H; ++n) { s[n] = in[n] + in[S - 1 - n]; d[n] = in[n] - in[S - 1 - n]; }
    for (int k = 0; k < S; k += 2) { float a = 0.0f; for (int n = 0; n < H; ++n) a += g_cm[k][n] * s[n]; out[k] = a; }
    for (int k = 1; k < S; k += 2) { float a = 0.0f; for (int n = 0; n < H; ++n) a += g_cm[k][n] * d[n]; out[k] = a; }
}

// 1D inverse (DCT-III), sparse-aware: post-dequant lines are mostly zero.
static inline void dct1d_inv(const float *restrict in, float *restrict out) {
    const int S = N, H = S / 2;
    float e[N / 2], o[N / 2];
    for (int n = 0; n < H; ++n) { e[n] = 0.0f; o[n] = 0.0f; }
    for (int k = 0; k < S; k += 2) { float v = in[k]; if (v != 0.0f) for (int n = 0; n < H; ++n) e[n] += g_cm[k][n] * v; }
    for (int k = 1; k < S; k += 2) { float v = in[k]; if (v != 0.0f) for (int n = 0; n < H; ++n) o[n] += g_cm[k][n] * v; }
    for (int n = 0; n < H; ++n) { out[n] = e[n] + o[n]; out[S - 1 - n] = e[n] - o[n]; }
}

// Transform every contiguous length-16 line along the last axis, skipping
// all-zero lines. Out-of-place.
static inline void lines_fwd(const float *restrict src, float *restrict dst) {
    const int S = N; float ol[N];
    for (int off = 0; off < S * S; ++off) {
        const float *v = src + (size_t)off * S; float *o = dst + (size_t)off * S;
        int nz = 0; for (int i = 0; i < S; ++i) if (v[i] != 0.0f) { nz = 1; break; }
        if (!nz) { for (int i = 0; i < S; ++i) o[i] = 0.0f; continue; }
        dct1d_fwd(v, ol); for (int i = 0; i < S; ++i) o[i] = ol[i];
    }
}
static inline void lines_inv(const float *restrict src, float *restrict dst) {
    const int S = N; float ol[N];
    for (int off = 0; off < S * S; ++off) {
        const float *v = src + (size_t)off * S; float *o = dst + (size_t)off * S;
        int nz = 0; for (int i = 0; i < S; ++i) if (v[i] != 0.0f) { nz = 1; break; }
        if (!nz) { for (int i = 0; i < S; ++i) o[i] = 0.0f; continue; }
        dct1d_inv(v, ol); for (int i = 0; i < S; ++i) o[i] = ol[i];
    }
}

// Cache-blocked axis rotate (z,y,x) -> (x,z,y). Three rotates restore the
// original axis order after three line passes.
#define ROT_TILE 8
static inline void rot(const float *restrict src, float *restrict dst) {
    const int S = N;
    for (int zt = 0; zt < S; zt += ROT_TILE)
    for (int xt = 0; xt < S; xt += ROT_TILE)
        for (int z = zt; z < zt + ROT_TILE; ++z)
        for (int x = xt; x < xt + ROT_TILE; ++x) {
            const float *sp = src + ((size_t)z * S) * S + x;
            float *dp = dst + ((size_t)x * S + z) * S;
            for (int y = 0; y < S; ++y) dp[y] = sp[(size_t)y * S];
        }
}

// 3D forward/inverse DCT on a 16^3 block. `a` and `b` are caller scratch.
static void dct3_fwd(const float *restrict blk, float *restrict coef,
                     float *restrict a, float *restrict b) {
    lines_fwd(blk, a); rot(a, b);
    lines_fwd(b, a);   rot(a, b);
    lines_fwd(b, a);   rot(a, coef);
}
static void dct3_inv(const float *restrict coef, float *restrict blk,
                     float *restrict a, float *restrict b) {
    lines_inv(coef, a); rot(a, b);
    lines_inv(b, a);    rot(a, b);
    lines_inv(b, a);    rot(a, blk);
}

// ============================================================================
// Dead-zone quantizer (pure float).
//   step(k) = qstep * (1 + L1freq)^HF_EXP   per coefficient
//   level   = sign(c) * floor(|c|/step + (1 - DZ_FRAC))  for |c| >= dz else 0
//   recon   = sign(l) * step * (|l| - 1 + DZ_FRAC + 0.40)
// ============================================================================
// The per-coefficient step is qstep * (1 + L1freq)^HF_EXP, and the weight
// depends only on L1freq = cz+cy+cx in [0, 3*(N-1)] (46 distinct values for
// N=16). Precompute those 46 powf results once (idempotent write, same
// lock-free pattern as the cosine table) so step_build is a scalar multiply
// per coefficient instead of a powf. Bit-identical to the old triple-powf loop.
static float g_hfw[3 * (N - 1) + 1];
static atomic_int g_hfw_ready = 0;
static void hfw_init(void) {
    if (atomic_load_explicit(&g_hfw_ready, memory_order_acquire)) return;
    for (int f = 0; f <= 3 * (N - 1); ++f) g_hfw[f] = powf(1.0f + (float)f, HF_EXP);
    atomic_store_explicit(&g_hfw_ready, 1, memory_order_release);
}
static void step_build(float qstep, float step_tab[N3]) {
    for (int cz = 0; cz < N; ++cz) for (int cy = 0; cy < N; ++cy) for (int cx = 0; cx < N; ++cx) {
        int i = (cz * N + cy) * N + cx;
        step_tab[i] = qstep * g_hfw[cz + cy + cx];
    }
}
static inline float quant_one(float c, float step) {
    float dz = DZ_FRAC * step, a = fabsf(c); float lv = 0.0f;
    if (a >= dz) lv = floorf((a - dz) / step + 1.0f);
    return c < 0.0f ? -lv : lv;
}
static inline float deq_one(float lv, float step) {
    if (lv == 0.0f) return 0.0f;
    float a = fabsf(lv);
    float r = (a - 1.0f) * step + DZ_FRAC * step + 0.40f * step;
    return lv < 0.0f ? -r : r;
}

// ============================================================================
// Binary range coder (CABAC-style). Bit-exact integer arithmetic coder: its
// low/range renormalization is integer by definition (float would not round-trip
// bit-for-bit), so this is the one place integers are load-bearing.
// ============================================================================
typedef uint8_t  rc_u8;
typedef uint32_t rc_u32;
typedef uint64_t rc_u64;
typedef struct { rc_u8 *buf; size_t cap, len; rc_u64 low; rc_u32 range; rc_u8 cache; rc_u64 cache_size; } rc_enc;
typedef struct { const rc_u8 *buf; size_t len, pos; rc_u32 code, range; } rc_dec;
typedef struct { uint16_t p0; } ctx_t;   // adaptive P(bit==0) in 1/4096 units
#define RC_TOP (1u << 24)

static inline void ctx_init(ctx_t *c) { c->p0 = 1u << 11; }
static inline void ctx_init_p(ctx_t *c, uint16_t p0) { c->p0 = p0 ? p0 : (1u << 11); }

// Optional prior training. Compiled in with -DDCT3D_TRAIN: each context-coded
// bit is tallied by (class, slot) so a training run can dump the stationary
// P(bit==0) per context. Off by default (zero cost).
#ifdef DCT3D_TRAIN
enum { TR_SIG=0, TR_MAG=1, TR_EOB=2, TR_NCLS=3 };
#define TR_SLOTS 32
long g_tr_n[TR_NCLS][TR_SLOTS], g_tr_z[TR_NCLS][TR_SLOTS];
#define TRAIN(cls,slot,bit) do{ if((slot)<TR_SLOTS){ g_tr_n[cls][slot]++; g_tr_z[cls][slot]+=((bit)==0); } }while(0)
#else
#define TRAIN(cls,slot,bit) ((void)0)
#endif

static void enc_init(rc_enc *e, rc_u8 *buf, size_t cap) {
    e->buf = buf; e->cap = cap; e->len = 0; e->low = 0; e->range = 0xFFFFFFFFu; e->cache = 0; e->cache_size = 1;
}
static void enc_putbyte(rc_enc *e, rc_u8 b) { if (e->len < e->cap) e->buf[e->len++] = b; else e->len++; }
static void enc_shift_low(rc_enc *e) {
    if ((rc_u32)(e->low >> 32) != 0 || e->low < 0xFF000000ull) {
        rc_u8 carry = (rc_u8)(e->low >> 32);
        do { enc_putbyte(e, (rc_u8)(e->cache + carry)); e->cache = 0xFF; } while (--e->cache_size);
        e->cache = (rc_u8)(e->low >> 24);
    }
    e->cache_size++; e->low = (e->low << 8) & 0xFFFFFFFFull;
}
static void enc_bit(rc_enc *e, ctx_t *c, int bit) {
    rc_u32 r0 = (e->range >> 12) * c->p0;
    if (bit == 0) { e->range = r0;        c->p0 = (uint16_t)(c->p0 + ((4096 - c->p0) >> 4)); }
    else          { e->low += r0; e->range -= r0; c->p0 = (uint16_t)(c->p0 - (c->p0 >> 4)); }
    while (e->range < RC_TOP) { enc_shift_low(e); e->range <<= 8; }
}
static void enc_bypass(rc_enc *e, int bit) {
    e->range >>= 1; if (bit) e->low += e->range;
    while (e->range < RC_TOP) { enc_shift_low(e); e->range <<= 8; }
}
static void enc_bypass_n(rc_enc *e, rc_u32 v, int k) {
    while (k > 16) { enc_bypass_n(e, (v >> (k - 16)) & 0xFFFF, 16); k -= 16; }
    if (!k) return;
    e->range >>= k; e->low += (rc_u64)(v & ((1u << k) - 1)) * e->range;
    while (e->range < RC_TOP) { enc_shift_low(e); e->range <<= 8; }
}
static void enc_flush(rc_enc *e) { for (int i = 0; i < 5; ++i) enc_shift_low(e); }

static void dec_init(rc_dec *d, const rc_u8 *buf, size_t len) {
    d->buf = buf; d->len = len; d->pos = 0; d->code = 0; d->range = 0xFFFFFFFFu;
    for (int i = 0; i < 5; ++i) { rc_u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0; d->code = (d->code << 8) | b; }
}
static int dec_bit(rc_dec *d, ctx_t *c) {
    rc_u32 r0 = (d->range >> 12) * c->p0; int bit;
    if (d->code < r0) { d->range = r0; bit = 0;            c->p0 = (uint16_t)(c->p0 + ((4096 - c->p0) >> 4)); }
    else              { d->code -= r0; d->range -= r0; bit = 1; c->p0 = (uint16_t)(c->p0 - (c->p0 >> 4)); }
    while (d->range < RC_TOP) { rc_u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0; d->code = (d->code << 8) | b; d->range <<= 8; }
    return bit;
}
static int dec_bypass(rc_dec *d) {
    d->range >>= 1; int bit = (d->code >= d->range); if (bit) d->code -= d->range;
    while (d->range < RC_TOP) { rc_u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0; d->code = (d->code << 8) | b; d->range <<= 8; }
    return bit;
}
static rc_u32 dec_bypass_n(rc_dec *d, int k) {
    rc_u32 v = 0;
    while (k > 16) { v = (v << 16) | dec_bypass_n(d, 16); k -= 16; }
    if (!k) return v;
    d->range >>= k;
    rc_u32 q = d->code / d->range, m = (1u << k) - 1; if (q > m) q = m;
    d->code -= q * d->range;
    while (d->range < RC_TOP) { rc_u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0; d->code = (d->code << 8) | b; d->range <<= 8; }
    return (v << k) | q;
}

// Exp-Golomb (order 0), v >= 0, coded in bypass bits.
static void enc_eg(rc_enc *e, rc_u32 v) {
    rc_u32 nb = 0, t = v + 1; while (t > 1) { t >>= 1; nb++; }
    for (rc_u32 i = 0; i < nb; ++i) enc_bypass(e, 1); enc_bypass(e, 0);
    if (nb) enc_bypass_n(e, (v + 1) & ((1u << nb) - 1), (int)nb);
}
static rc_u32 dec_eg(rc_dec *d) {
    rc_u32 nb = 0; while (dec_bypass(d)) { if (++nb > 31) return 0; }
    if (!nb) return 0;
    return ((1u << nb) | dec_bypass_n(d, (int)nb)) - 1;
}

// Signed Exp-Golomb (zig-zag mapped), for small signed integers.
static void enc_seg(rc_enc *e, long v) {
    rc_u32 z = v < 0 ? (rc_u32)(-2 * v - 1) : (rc_u32)(2 * v);
    enc_eg(e, z);
}
static long dec_seg(rc_dec *d) {
    rc_u32 z = dec_eg(d);
    return (z & 1) ? -(long)((z + 1) >> 1) : (long)(z >> 1);
}

// Store a float header field compactly: one flag bit, then either an integer
// (signed Exp-Golomb) when the value is an exact integer in +-2^24 (the common
// case for integer-dtype vmin/vspan, and cheap), or raw 32 bits otherwise
// (f32 data). Lossless in both branches.
static void enc_fval(rc_enc *e, float f) {
    float r = truncf(f);
    if (r == f && fabsf(f) < 16777216.0f) { enc_bypass(e, 0); enc_seg(e, (long)r); }
    else { enc_bypass(e, 1); uint32_t b; memcpy(&b, &f, 4); enc_bypass_n(e, b, 32); }
}
static float dec_fval(rc_dec *d) {
    if (!dec_bypass(d)) return (float)dec_seg(d);
    uint32_t b = dec_bypass_n(d, 32); float f; memcpy(&f, &b, 4); return f;
}

// ============================================================================
// Coefficient context coder. Ascending-frequency scan with an EOB, per-band
// significance contexts conditioned on recent significance density, and an
// adaptive-unary + Exp-Golomb magnitude ladder.
// ============================================================================
#define NB_BANDS 8
#define MAGCTX   12
#define EOB_CTX  14
typedef struct {
    ctx_t sig[NB_BANDS * 4];
    ctx_t mag[MAGCTX];
    ctx_t eob[EOB_CTX];
} coef_ctx;

// Static priors: P(bit==0) in 1/4096 units per context, trained on PHercParis4
// 2.4um dense scroll blocks across q in {1..64} (tools: DCT3D_TRAIN build). Per-
// block contexts reset every 4096-voxel block, so without priors each adaptive
// bin spends its first ~32 bits no better than a coin flip; seeding from the
// corpus stationary distribution recovers that. 2048 == untrained/neutral slot.
static const uint16_t PRI_SIG[NB_BANDS * 4] = {
    1980, 575, 846, 743, 3916,3707,3515,1463, 3969,3753,3527,2313, 4002,3826,3635,2819,
    4039,3851,3613,2974, 3985,3840,3703,3475, 4021,3991,3951,3969, 4065,4024,2048,2048,
};
static const uint16_t PRI_MAG[MAGCTX] = {
    1893,1217,908,728,610,527,465,416,377,346,320,297,
};
static const uint16_t PRI_EOB[EOB_CTX] = {
    74,1,1,1,1,5,109,504,772,1063,1458,2326,4095,2048,
};

static void coef_ctx_init(coef_ctx *a) {
    for (size_t i = 0; i < sizeof a->sig / sizeof a->sig[0]; ++i) ctx_init_p(&a->sig[i], PRI_SIG[i]);
    for (int i = 0; i < MAGCTX;  ++i) ctx_init_p(&a->mag[i], PRI_MAG[i]);
    for (int i = 0; i < EOB_CTX; ++i) ctx_init_p(&a->eob[i], PRI_EOB[i]);
}

static void enc_magnitude(rc_enc *e, coef_ctx *ac, rc_u32 m) {
    ctx_t *mag = ac->mag; rc_u32 v = m - 1, k = 0;
    while (k < (rc_u32)(MAGCTX - 1) && v > 0) { TRAIN(TR_MAG,k,1); enc_bit(e, &mag[k], 1); v -= 1; k++; if (v == 0) { TRAIN(TR_MAG,k,0); enc_bit(e, &mag[k], 0); return; } }
    if (v == 0) { TRAIN(TR_MAG,k,0); enc_bit(e, &mag[k], 0); return; }
    TRAIN(TR_MAG,k,1); enc_bit(e, &mag[k], 1);
    rc_u32 x = v, nbits = 0, tt = x + 1; while (tt > 1) { tt >>= 1; nbits++; }
    for (rc_u32 i = 0; i < nbits; ++i) enc_bypass(e, 1); enc_bypass(e, 0);
    if (nbits) enc_bypass_n(e, (x + 1) & ((1u << nbits) - 1), (int)nbits);
}
static rc_u32 dec_magnitude(rc_dec *d, coef_ctx *ac) {
    ctx_t *mag = ac->mag; rc_u32 v = 0, k = 0;
    while (k < (rc_u32)(MAGCTX - 1)) { if (dec_bit(d, &mag[k])) { v += 1; k++; } else return v + 1; }
    if (!dec_bit(d, &mag[k])) return v + 1;
    rc_u32 nbits = 0; while (dec_bypass(d)) { if (++nbits > 31) break; }
    rc_u32 x = nbits ? ((1u << nbits) | dec_bypass_n(d, (int)nbits)) - 1 : 0;
    return v + x + 1;
}

// Ascending-L1-frequency scan order over the 16^3 coefficient cube. It is a
// pure function of the fixed geometry (same table every call), so it is built
// once by counting sort into a shared table rather than rebuilt per block.
// Idempotent write, same release/acquire publish pattern as the cosine table.
static uint16_t g_scan[N3];
static atomic_int g_scan_ready = 0;
static void scan_init(void) {
    if (atomic_load_explicit(&g_scan_ready, memory_order_acquire)) return;
    enum { FMAX = 3 * (N - 1) };   // max L1 frequency = 45
    int hist[FMAX + 2];
    for (int i = 0; i < FMAX + 2; ++i) hist[i] = 0;
    for (int idx = 0; idx < N3; ++idx) {
        int cz = idx / (N * N), cy = (idx / N) % N, cx = idx % N;
        hist[cz + cy + cx + 1]++;
    }
    for (int i = 1; i < FMAX + 2; ++i) hist[i] += hist[i - 1];
    for (int idx = 0; idx < N3; ++idx) {
        int cz = idx / (N * N), cy = (idx / N) % N, cx = idx % N;
        g_scan[hist[cz + cy + cx]++] = (uint16_t)idx;
    }
    atomic_store_explicit(&g_scan_ready, 1, memory_order_release);
}
static inline int band_of(rc_u32 idx) {
    rc_u32 cz = idx / (N * N), cy = (idx / N) % N, cx = idx % N, freq = cz + cy + cx;
    int b = (int)(freq * NB_BANDS / (3u * N)); if (b >= NB_BANDS) b = NB_BANDS - 1; return b;
}

static void enc_eob(rc_enc *e, coef_ctx *c, rc_u32 v) {
    int kmax = 0; while ((1u << kmax) <= (rc_u32)N3) kmax++;
    int k = 0; while ((1u << k) <= v) k++;
    for (int i = 0; i < k; ++i) { TRAIN(TR_EOB,i,1); enc_bit(e, &c->eob[i], 1); }
    if (k < kmax) { TRAIN(TR_EOB,k,0); enc_bit(e, &c->eob[k], 0); }
    if (k > 1) enc_bypass_n(e, v & ((1u << (k - 1)) - 1), k - 1);
}
static rc_u32 dec_eob(rc_dec *d, coef_ctx *c) {
    int kmax = 0; while ((1u << kmax) <= (rc_u32)N3) kmax++;
    int k = 0; while (k < kmax && dec_bit(d, &c->eob[k])) k++;
    if (k == 0) return 0;
    if (k == 1) return 1;
    return (1u << (k - 1)) | dec_bypass_n(d, k - 1);
}

// Encode/decode quantized levels (float-valued integers) in raster order.
static void enc_coefs(rc_enc *e, const float *lvl, const uint16_t *scan) {
    coef_ctx ac; coef_ctx_init(&ac);
    rc_u32 eob = 0; for (rc_u32 p = N3; p-- > 0; ) { if (lvl[scan[p]] != 0.0f) { eob = p + 1; break; } }
    enc_eob(e, &ac, eob);
    rc_u32 hist = 0;
    for (rc_u32 p = 0; p < eob; ++p) {
        rc_u32 idx = scan[p]; int b = band_of(idx); float v = lvl[idx];
        int dens = __builtin_popcount(hist & 0xFFFFu); dens = dens < 3 ? dens : 3;
        if (p != eob - 1) { TRAIN(TR_SIG, b * 4 + dens, v != 0.0f); enc_bit(e, &ac.sig[b * 4 + dens], v != 0.0f); }
        hist = (hist << 1) | (v != 0.0f);
        if (v == 0.0f) continue;
        rc_u32 m = (rc_u32)fabsf(v);
        enc_magnitude(e, &ac, m);
        enc_bypass(e, v < 0.0f ? 1 : 0);
    }
}
static void dec_coefs(rc_dec *d, float *lvl, const uint16_t *scan) {
    coef_ctx ac; coef_ctx_init(&ac);
    for (int i = 0; i < N3; ++i) lvl[i] = 0.0f;
    rc_u32 eob = dec_eob(d, &ac); if (eob > (rc_u32)N3) eob = N3;
    rc_u32 hist = 0;
    for (rc_u32 p = 0; p < eob; ++p) {
        rc_u32 idx = scan[p]; int b = band_of(idx);
        int dens = __builtin_popcount(hist & 0xFFFFu); dens = dens < 3 ? dens : 3;
        int sig = (p == eob - 1) ? 1 : dec_bit(d, &ac.sig[b * 4 + dens]);
        hist = (hist << 1) | sig;
        if (!sig) continue;
        rc_u32 m = dec_magnitude(d, &ac);
        int neg = dec_bypass(d);
        lvl[idx] = neg ? -(float)m : (float)m;
    }
}

// ============================================================================
// Raw preamble (de)serialization. Everything else rides in the range stream.
// ============================================================================
static void     put_u8(uint8_t **p, uint32_t v)  { *(*p)++ = (uint8_t)v; }
static uint32_t get_u8(const uint8_t **p)         { return *(*p)++; }

// ============================================================================
// dtype ids and blob framing.
//
// Blob layout: a 2-byte raw preamble [u8 MAGIC][u8 dtype] (so a wrong-typed or
// non-dct3d blob is rejected before touching the coder), then ONE range-coded
// stream carrying everything else:
//   enc_fval(vmin) enc_fval(vspan) [f32 qstep raw] seg(dcq)
//   | has-corr flag | coefficient levels | corrections
//
// vmin/vspan describe the block's raw value range; the working domain is
// (value - vmin)/vspan * 255 so the fixed quality calibration behaves uniformly
// across dtypes. vmin/vspan are coded as integers when exact (the integer-dtype
// common case), else raw. qstep is raw f32 (must reproduce the step table bit-
// for-bit). dcq is the block mean fine-quantized to 1/DC_Q units, coded as a
// small signed integer. This replaces the old fixed 18-byte raw header, whose
// blob share reached ~35% at high quality.
// ============================================================================
enum { DT_U8=1, DT_U16=2, DT_U32=3, DT_S8=4, DT_S16=5, DT_S32=6, DT_F32=7 };
#define MAGIC 0xD3u
#define HDR_BYTES 2            // raw preamble: MAGIC + dtype
#define DC_Q 16                // dc fine-quantization: 1/16 normalized unit

// Clamp to [lo,hi], mapping NaN to lo. NaN-safety matters on the decode path:
// a corrupt blob can decode vmin/vspan to values that make a voxel non-finite,
// and casting a non-finite (or out-of-range) float to an integer dtype is UB —
// this clamp is what makes the store well-defined for any input.
static inline float clampf(float v, float lo, float hi) {
    if (!(v >= lo)) return lo;      // false when v < lo OR v is NaN
    return v > hi ? hi : v;
}

// Shared float-domain encode. `blk_in` holds the block loaded to float; vmin
// and vspan (vspan > 0) describe its value range. Returns encoded length.
static size_t encode_float(const float *blk_in, float vmin, float vspan,
                           int dtype, float quality, float max_error, float tau,
                           uint8_t *out) {
    dct_init(); hfw_init(); scan_init();

    const float scale = 255.0f / vspan;              // raw -> normalized(0..255)
    float qstep = quality; if (!(qstep > 0.0f)) qstep = 1.0f;

    // normalize, take the mean, center. The mean (dc) is fine-quantized to
    // 1/DC_Q normalized units and coded as a small integer instead of a raw
    // f32 (its own slot in the transform, coef[0], is forced to ~0 by centering
    // and would otherwise waste 32 bits). Encoder and decoder both use the
    // quantized dcq so the split is exact; the sub-1/DC_Q remainder is absorbed
    // by coef[0] and coded losslessly with everything else.
    float norm[N3], blk[N3];
    float sum = 0.0f;
    for (int i = 0; i < N3; ++i) { float w = (blk_in[i] - vmin) * scale; norm[i] = w; sum += w; }
    float dc = sum / (float)N3;
    long  dcq_i = (long)lroundf(dc * (float)DC_Q);
    float dcq = (float)dcq_i / (float)DC_Q;
    for (int i = 0; i < N3; ++i) blk[i] = norm[i] - dcq;

    float a[N3], b[N3], coef[N3];
    dct3_fwd(blk, coef, a, b);

    float step[N3]; step_build(qstep, step);
    float lvl[N3];
    for (int i = 0; i < N3; ++i) lvl[i] = quant_one(coef[i], step[i]);

    // Effective correction tolerance in the normalized domain. max_error is a
    // fraction of the value range (=> * 255). tau is absolute in raw units
    // (=> * scale). If both are set, take the tighter bound.
    float tnorm = 0.0f;
    if (max_error > 0.0f) tnorm = max_error * 255.0f;
    if (tau > 0.0f) { float t2 = tau * scale; tnorm = (tnorm == 0.0f) ? t2 : (t2 < tnorm ? t2 : tnorm); }

    // Correction quantum (normalized units). Corrections are stored as integer
    // multiples of cq, so a corrected voxel lands within 0.5*cq of exact. To
    // honor tau we need 0.5*cq*(vspan/255) <= tau, i.e. cq <= 2*tnorm; we use
    // cq = min(1, tnorm) which satisfies that with margin and stays coarse (1
    // normalized unit) whenever the request is looser than one unit — so u8 and
    // other narrow-range data keep the cheap ~integer deltas, while a tight tau
    // on a wide-range dtype (u16/u32/f32) shrinks the quantum enough to meet it.
    float cq = tnorm < 1.0f ? tnorm : 1.0f;
    uint16_t cpos[N3]; float cdel[N3]; int ncorr = 0;
    if (tnorm > 0.0f) {
        float rc[N3], rb[N3];
        for (int i = 0; i < N3; ++i) rc[i] = deq_one(lvl[i], step[i]);
        dct3_inv(rc, rb, a, b);
        for (int i = 0; i < N3; ++i) {
            float err = norm[i] - (rb[i] + dcq);
            if (err > tnorm || err < -tnorm) { cpos[ncorr] = (uint16_t)i; cdel[ncorr] = err; ncorr++; }
        }
    }

    uint8_t *p = out;
    put_u8(&p, MAGIC);
    put_u8(&p, (uint32_t)dtype);

    rc_enc e; enc_init(&e, p, DCT3D_MAX_BYTES - HDR_BYTES);
    enc_fval(&e, vmin);
    enc_fval(&e, vspan);
    // enc_fval is lossless in both branches, so qstep still reproduces the
    // step table bit-for-bit; typical qualities (1,2,4..64) hit the integer
    // path at ~5-13 bits instead of 32 raw (+3-5% ratio at high quality).
    enc_fval(&e, qstep);
    enc_seg(&e, dcq_i);
    ctx_t cflag; ctx_init(&cflag);
    enc_bit(&e, &cflag, ncorr > 0);

    enc_coefs(&e, lvl, g_scan);

    if (ncorr > 0) {
        // Correction stream: [eg count][fval cq][ {eg gap, sign, eg(|delta|-1)} ]*.
        // delta is err/cq rounded to the nearest nonzero integer, so the applied
        // correction is delta*cq and the residual is < 0.5*cq normalized (=>
        // < 0.5*cq*vspan/255 raw), meeting tau by construction. cq rides in the
        // stream (corrections-only) since the decoder does not know tau; the
        // common case cq==1.0 hits enc_fval's integer path at ~3 bits.
        enc_eg(&e, (rc_u32)(ncorr - 1));
        enc_fval(&e, cq);
        rc_u32 prev = 0;
        for (int c = 0; c < ncorr; ++c) {
            enc_eg(&e, (rc_u32)cpos[c] - prev); prev = cpos[c];
            long di = (long)lroundf(cdel[c] / cq);
            if (di == 0) di = cdel[c] < 0.0f ? -1 : 1;   // corrected => nonzero
            enc_bypass(&e, di < 0 ? 1 : 0);
            enc_eg(&e, (rc_u32)((di < 0 ? -di : di) - 1));
        }
    }
    enc_flush(&e);
    return HDR_BYTES + e.len;
}

// Shared float-domain decode. Fills `norm_out` with reconstructed raw values.
// Returns 1 on success, 0 on a malformed blob or dtype mismatch.
static int decode_float(const uint8_t *blob, size_t len, int want_dtype, float *out_val) {
    if (len < HDR_BYTES) return 0;
    const uint8_t *p = blob;
    if (get_u8(&p) != MAGIC) return 0;
    if ((int)get_u8(&p) != want_dtype) return 0;

    dct_init(); hfw_init(); scan_init();

    rc_dec d; dec_init(&d, p, len - HDR_BYTES);
    float vmin  = dec_fval(&d);
    float vspan = dec_fval(&d);
    float qstep = dec_fval(&d);
    float dc    = (float)dec_seg(&d) / (float)DC_Q;
    if (!(vspan > 0.0f) || !(qstep > 0.0f) ||
        !isfinite(vmin) || !isfinite(dc)) return 0;

    float step[N3]; step_build(qstep, step);

    ctx_t cflag; ctx_init(&cflag);
    int has_corr = dec_bit(&d, &cflag);

    float lvl[N3]; dec_coefs(&d, lvl, g_scan);

    float coef[N3], a[N3], b[N3], rb[N3];
    for (int i = 0; i < N3; ++i) coef[i] = deq_one(lvl[i], step[i]);
    dct3_inv(coef, rb, a, b);

    float norm[N3];
    for (int i = 0; i < N3; ++i) norm[i] = rb[i] + dc;

    if (has_corr) {
        // ncorr and positions are attacker-controlled on a corrupt blob; clamp
        // both so decoding can never read or write out of bounds.
        rc_u32 ncorr = dec_eg(&d) + 1, pos = 0;
        if (ncorr > (rc_u32)N3) ncorr = N3;
        float cq = dec_fval(&d);
        if (!(cq > 0.0f) || !isfinite(cq)) cq = 1.0f;   // corrupt guard
        for (rc_u32 c = 0; c < ncorr; ++c) {
            pos += dec_eg(&d);
            int neg = dec_bypass(&d);
            rc_u32 mag = dec_eg(&d) + 1;
            if (pos >= (rc_u32)N3) break;
            float delta = (float)mag * cq;
            norm[pos] += neg ? -delta : delta;
        }
    }

    // denormalize back to raw value space.
    const float inv = vspan / 255.0f;
    for (int i = 0; i < N3; ++i) out_val[i] = norm[i] * inv + vmin;
    return 1;
}

// ============================================================================
// Typed public wrappers. Each measures the block's value range (for f32,
// vspan is derived from min/max; for integer types it is likewise the observed
// span, so a flat block never divides by zero), encodes in the float domain,
// and on decode rounds+clamps back to the target type (f32 stores verbatim).
// ============================================================================
#define ENC_BODY(T, DT, ROUND, LO, HI)                                        \
    do {                                                                      \
        float fb[N3];                                                         \
        /* min/max over FINITE voxels only. Integer dtypes are always finite  \
         * (the isfinite test folds away); for f32 this keeps a stray NaN/Inf \
         * from poisoning the header and losing the whole block. Non-finite   \
         * voxels are substituted with the finite min so every finite voxel   \
         * still round-trips and the blob is always decodable. */             \
        float vmin = 0, vmax = 0; int seen = 0;                               \
        for (int i = 0; i < N3; ++i) {                                        \
            float v = (float)chunk[i];                                        \
            if (!isfinite(v)) continue;                                       \
            if (!seen) { vmin = vmax = v; seen = 1; }                         \
            else { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }          \
        }                                                                     \
        if (!seen) { vmin = vmax = 0; }   /* all non-finite: encode as flat */\
        for (int i = 0; i < N3; ++i) {                                        \
            float v = (float)chunk[i]; fb[i] = isfinite(v) ? v : vmin;        \
        }                                                                     \
        float vspan = vmax - vmin; if (!(vspan > 0.0f)) vspan = 1.0f;         \
        return encode_float(fb, vmin, vspan, (DT), quality, max_error, tau,   \
                            out);                                             \
    } while (0)

#define DEC_BODY(T, DT, ROUND, LO, HI)                                        \
    do {                                                                      \
        float fb[N3];                                                         \
        if (!decode_float(blob, len, (DT), fb)) return 0;                     \
        for (int i = 0; i < N3; ++i) {                                        \
            float v = ROUND(fb[i]);                                           \
            v = clampf(v, (float)(LO), (float)(HI));                          \
            chunk[i] = (T)v;                                                  \
        }                                                                     \
        return 1;                                                             \
    } while (0)

// Integer types: round to nearest, clamp to the type's representable range.
size_t dct3d_encode_u8 (const uint8_t  *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(uint8_t,  DT_U8,  roundf, 0, 255); }
int    dct3d_decode_u8 (const uint8_t  *blob, size_t len, uint8_t  *chunk) { DEC_BODY(uint8_t,  DT_U8,  roundf, 0, 255); }
size_t dct3d_encode_u16(const uint16_t *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(uint16_t, DT_U16, roundf, 0, 65535); }
int    dct3d_decode_u16(const uint8_t  *blob, size_t len, uint16_t *chunk) { DEC_BODY(uint16_t, DT_U16, roundf, 0, 65535); }
size_t dct3d_encode_u32(const uint32_t *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(uint32_t, DT_U32, roundf, 0, 4294967295.0); }
// Decode HI is the largest float <= UINT32_MAX (4294967295 is not representable
// as a float; (float)that rounds up to 2^32 and casting it back is UB). Values
// this large already carry the 2^24 float-precision loss, so the clamp is moot.
int    dct3d_decode_u32(const uint8_t  *blob, size_t len, uint32_t *chunk) { DEC_BODY(uint32_t, DT_U32, roundf, 0, 4294967040.0); }
size_t dct3d_encode_s8 (const int8_t   *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(int8_t,   DT_S8,  roundf, -128, 127); }
int    dct3d_decode_s8 (const uint8_t  *blob, size_t len, int8_t   *chunk) { DEC_BODY(int8_t,   DT_S8,  roundf, -128, 127); }
size_t dct3d_encode_s16(const int16_t  *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(int16_t,  DT_S16, roundf, -32768, 32767); }
int    dct3d_decode_s16(const uint8_t  *blob, size_t len, int16_t  *chunk) { DEC_BODY(int16_t,  DT_S16, roundf, -32768, 32767); }
size_t dct3d_encode_s32(const int32_t  *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(int32_t,  DT_S32, roundf, -2147483648.0, 2147483647.0); }
// Decode HI is the largest float < 2^31 (2147483647 rounds up to 2^31 as a
// float, which overflows int32 on cast). Same 2^24 precision caveat applies.
int    dct3d_decode_s32(const uint8_t  *blob, size_t len, int32_t  *chunk) { DEC_BODY(int32_t,  DT_S32, roundf, -2147483648.0, 2147483520.0); }

// f32: store verbatim (no rounding, no clamp). ROUND is identity; range is
// the full float line, so clampf is a no-op.
#define IDENTITY(x) (x)
size_t dct3d_encode_f32(const float *chunk, float quality, float max_error, float tau, uint8_t *out) { ENC_BODY(float, DT_F32, IDENTITY, -FLT_MAX, FLT_MAX); }
int    dct3d_decode_f32(const uint8_t *blob, size_t len, float *chunk) { DEC_BODY(float, DT_F32, IDENTITY, -FLT_MAX, FLT_MAX); }
