// Round-trip + bound tests for dct3d.
#include "dct3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng(uint32_t *s){ uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; *s=x; return x; }

static int fails = 0;
#define CHECK(cond, msg, ...) do{ if(!(cond)){ printf("FAIL: " msg "\n", ##__VA_ARGS__); fails++; } }while(0)

// stats helper: max abs err and PSNR-ish over a scale
static void stats(const double *o, const double *r, int n, double *maxe, double *rmse){
    double m=0, se=0;
    for(int i=0;i<n;++i){ double e=fabs(o[i]-r[i]); if(e>m)m=e; se+=e*e; }
    *maxe=m; *rmse=sqrt(se/n);
}

#define GEN(TY, name, MAKE) \
typedef TY T_##name; \
static void test_##name(void){ \
    typedef TY T; \
    T in[DCT3D_N3], back[DCT3D_N3]; \
    uint8_t blob[DCT3D_MAX_BYTES]; \
    uint32_t s=0xC0FFEE ^ (uint32_t)sizeof(T); \
    for(int i=0;i<DCT3D_N3;++i){ MAKE; } \
    /* lossy, no correction */ \
    size_t n = dct3d_encode_##name(in, 1.0f, 0.0f, 0.0f, blob); \
    CHECK(n>0 && n<=DCT3D_MAX_BYTES, #name " enc size %zu", n); \
    int ok = dct3d_decode_##name(blob, n, back); \
    CHECK(ok, #name " decode failed"); \
    double o[DCT3D_N3], r[DCT3D_N3]; \
    for(int i=0;i<DCT3D_N3;++i){ o[i]=(double)in[i]; r[i]=(double)back[i]; } \
    double maxe, rmse; stats(o,r,DCT3D_N3,&maxe,&rmse); \
    printf("%-4s q=1.0        : bytes=%4zu  maxerr=%.3g  rmse=%.3g\n", #name, n, maxe, rmse); \
    /* wrong-dtype decode must be rejected */ \
    T dummy[DCT3D_N3]; (void)dummy; \
    /* absolute tau bound */ \
    double span=0; { double lo=o[0],hi=o[0]; for(int i=0;i<DCT3D_N3;++i){ if(o[i]<lo)lo=o[i]; if(o[i]>hi)hi=o[i]; } span=hi-lo; } \
    float tau = (float)(span>0? span*0.02 : 1.0); \
    n = dct3d_encode_##name(in, 4.0f, 0.0f, tau, blob); \
    ok = dct3d_decode_##name(blob, n, back); \
    CHECK(ok, #name " decode(tau) failed"); \
    for(int i=0;i<DCT3D_N3;++i) r[i]=(double)back[i]; \
    stats(o,r,DCT3D_N3,&maxe,&rmse); \
    /* allow +-1 quantization on integer store on top of tau */ \
    double slack = (sizeof(T)==4 && (T)0.5!=0)? 0 : 1.0; \
    CHECK(maxe <= (double)tau + slack + 1e-3, "%s tau bound: maxerr=%.4g tau=%.4g", #name, maxe, (double)tau); \
    printf("%-4s q=4.0 tau=%.3g: bytes=%4zu  maxerr=%.3g  rmse=%.3g\n", #name, (double)tau, n, maxe, rmse); \
}

// generators per type
GEN(uint8_t,  u8,  in[i]=(T)(90 + (i%16)*4 + (rng(&s)%24)))
GEN(uint16_t, u16, in[i]=(T)(20000 + (i%16)*300 + (rng(&s)%600)))
GEN(uint32_t, u32, in[i]=(T)(1000000u + (i%16)*8000u + (rng(&s)%12000u)))
GEN(int8_t,   s8,  in[i]=(T)(-20 + (i%16)*3 + (int)(rng(&s)%10)))
GEN(int16_t,  s16, in[i]=(T)(-5000 + (i%16)*200 + (int)(rng(&s)%400)))
GEN(int32_t,  s32, in[i]=(T)(-500000 + (i%16)*4000 + (int)(rng(&s)%8000)))
GEN(float,    f32, in[i]=(T)(1.5f + 0.03f*(i%16) + 0.001f*(rng(&s)%50)))

static void test_reject_wrong_dtype(void){
    uint8_t in[DCT3D_N3]; for(int i=0;i<DCT3D_N3;++i) in[i]=(uint8_t)i;
    uint8_t blob[DCT3D_MAX_BYTES];
    size_t n = dct3d_encode_u8(in, 1.0f, 0, 0, blob);
    uint16_t back[DCT3D_N3];
    int ok = dct3d_decode_u16(blob, n, back);   // wrong type
    CHECK(!ok, "wrong-dtype decode should be rejected");
}

static void test_flat_block(void){
    // constant block must not divide by zero and must round-trip near-exactly.
    uint16_t in[DCT3D_N3]; for(int i=0;i<DCT3D_N3;++i) in[i]=4242;
    uint8_t blob[DCT3D_MAX_BYTES];
    size_t n = dct3d_encode_u16(in, 1.0f, 0, 0, blob);
    uint16_t back[DCT3D_N3];
    int ok = dct3d_decode_u16(blob, n, back);
    CHECK(ok, "flat block decode failed");
    int bad=0; for(int i=0;i<DCT3D_N3;++i) if(abs((int)back[i]-4242)>1) bad++;
    CHECK(bad==0, "flat block not preserved (%d off)", bad);
    printf("flat  const=4242  : bytes=%4zu  (should be tiny)\n", n);
}

static void test_corrupt(void){
    // truncated / garbage blobs must be rejected or bounded, never crash.
    uint8_t back8[DCT3D_N3];
    uint8_t garbage[64]; for(int i=0;i<64;++i) garbage[i]=(uint8_t)(i*7);
    (void)dct3d_decode_u8(garbage, 64, back8);      // must not crash
    (void)dct3d_decode_u8(garbage, 2, back8);       // too short
    uint8_t in[DCT3D_N3]; uint32_t s=1;
    for(int i=0;i<DCT3D_N3;++i) in[i]=(uint8_t)(rng(&s));
    uint8_t blob[DCT3D_MAX_BYTES];
    size_t n = dct3d_encode_u8(in,2.0f,0,4.0f,blob);
    for(size_t cut=0; cut<n; cut+=7) (void)dct3d_decode_u8(blob, cut, back8); // truncations
    for(int t=0;t<200;++t){ // random bit flips
        uint8_t tmp[DCT3D_MAX_BYTES]; memcpy(tmp,blob,n);
        tmp[rng(&s)%n] ^= (uint8_t)(1u<<(rng(&s)&7));
        (void)dct3d_decode_u8(tmp, n, back8);
    }
    printf("corrupt/truncate  : survived (no crash)\n");
}

// --- regression tests for the three correctness findings from code review ----

// tau on wide-range dtypes must be honored per-voxel (was floored at ~vspan/510).
// NOTE: the f32 case needs real (non-fast-math) float; build this file without
// -ffast-math to exercise it meaningfully.
static void test_tau_wide(void){
    uint8_t blob[DCT3D_MAX_BYTES];
    uint16_t u[DCT3D_N3], ub[DCT3D_N3]; uint32_t s=7;
    for(int i=0;i<DCT3D_N3;++i) u[i]=(uint16_t)(30000 + (i%16)*500 + rng(&s)%1000);
    for(float tau=1; tau<=8; tau*=2){
        size_t n=dct3d_encode_u16(u,8.0f,0,tau,blob); dct3d_decode_u16(blob,n,ub);
        int mx=0; for(int i=0;i<DCT3D_N3;++i){int e=abs((int)u[i]-ub[i]); if(e>mx)mx=e;}
        CHECK(mx<=(int)tau, "u16 wide tau=%.0f maxerr=%d", tau, mx);
    }
    float f[DCT3D_N3], fb[DCT3D_N3];
    for(int i=0;i<DCT3D_N3;++i) f[i]=100.0f + 0.5f*(i%16) + 0.01f*(rng(&s)%50);
    for(float tau=0.001f; tau<=0.1f; tau*=10){
        size_t n=dct3d_encode_f32(f,8.0f,0,tau,blob); dct3d_decode_f32(blob,n,fb);
        float mx=0; for(int i=0;i<DCT3D_N3;++i){float e=fabsf(f[i]-fb[i]); if(e>mx)mx=e;}
        CHECK(mx<=tau*1.001f, "f32 wide tau=%.3f maxerr=%.5f", tau, mx);
    }
}

// NaN/Inf f32 input must yield a decodable blob preserving finite voxels.
static void test_nonfinite_f32(void){
    uint8_t blob[DCT3D_MAX_BYTES];
    float f[DCT3D_N3], fb[DCT3D_N3]; uint32_t s=3;
    for(int i=0;i<DCT3D_N3;++i) f[i]=10.0f + 0.1f*(rng(&s)%40);
    f[100]=NAN; f[2000]=INFINITY; f[3000]=-INFINITY;
    for(int i=0;i<DCT3D_N3;++i) fb[i]=-999.0f;
    size_t n=dct3d_encode_f32(f,2.0f,0,0.5f,blob);
    CHECK(dct3d_decode_f32(blob,n,fb), "f32 NaN/Inf decodable");
    int bad=0; for(int i=0;i<DCT3D_N3;++i){ if(i==100||i==2000||i==3000) continue;
        if(fabsf(f[i]-fb[i])>0.51f) bad++; }
    CHECK(bad==0, "f32 NaN/Inf finite voxels preserved (%d bad)", bad);
    CHECK(isfinite(fb[100]), "f32 NaN voxel -> finite");
}

// small-magnitude u32 must round-trip losslessly within tau.
static void test_u32_small(void){
    uint8_t blob[DCT3D_MAX_BYTES];
    uint32_t u[DCT3D_N3], ub[DCT3D_N3]; uint32_t s=9;
    for(int i=0;i<DCT3D_N3;++i) u[i]=100 + (i%16)*3 + rng(&s)%20;
    size_t n=dct3d_encode_u32(u,1.0f,0,1.0f,blob); dct3d_decode_u32(blob,n,ub);
    int mx=0; for(int i=0;i<DCT3D_N3;++i){int e=abs((int)u[i]-(int)ub[i]); if(e>mx)mx=e;}
    CHECK(mx<=1, "u32 small-magnitude tau=1 maxerr=%d", mx);
}

int main(void){
    test_u8(); test_u16(); test_u32();
    test_s8(); test_s16(); test_s32();
    test_f32();
    test_reject_wrong_dtype();
    test_flat_block();
    test_corrupt();
    test_tau_wide();
    test_nonfinite_f32();
    test_u32_small();
    printf(fails? "\n%d CHECK(S) FAILED\n" : "\nALL CHECKS PASSED\n", fails);
    return fails?1:0;
}
