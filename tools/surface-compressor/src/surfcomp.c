#include "surfcomp.h"
#include "dct64.h"
#include "entropy.h"
#include "scan64.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
/* Finite-only assumptions also annotate public scalar arguments in LLVM;
 * source FP pragmas cannot reliably undo them. Keep all other fast-math
 * optimizations, but preserve nonfinite classification at API boundaries. */
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "Compile surfcomp with -fno-finite-math-only after any -ffast-math flags"
#endif
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128 &&
               sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,
               "surfcomp requires IEEE binary32/binary64 floating-point types");
const char *sfc_version(void) { return SFC_VERSION_STRING; }
uint32_t sfc_format_version(void) { return SFC_FORMAT_VERSION; }
static uint32_t u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static uint64_t u64(const uint8_t *p) {
  return u32(p) | (uint64_t)u32(p + 4) << 32;
}
static void p32(uint8_t *p, uint32_t x) {
  for (int i = 0; i < 4; i++)
    p[i] = (uint8_t)(x >> (8 * i));
}
static void p64(uint8_t *p, uint64_t x) {
  p32(p, (uint32_t)x);
  p32(p + 4, (uint32_t)(x >> 32));
}
static void pd(uint8_t *p, double d) {
  uint64_t x;
  memcpy(&x, &d, 8);
  p64(p, x);
}
static double gd(const uint8_t *p) {
  uint64_t x = u64(p);
  double d;
  memcpy(&d, &x, 8);
  return d;
}
size_t sfc_sample_size(sfc_dtype t) {
  return t == SFC_U8 ? 1 : t == SFC_U16 ? 2 : t == SFC_F32 ? 4 : 0;
}
#include "crc32.h"
static uint32_t crc_update(uint32_t c, const void *data, size_t n) {
  const uint8_t *p = data;
  while (n >= 8) {
    uint32_t a = u32(p) ^ c, b = u32(p + 4);
    c = sfc_crc_table[7][a & 255] ^ sfc_crc_table[6][(a >> 8) & 255] ^
        sfc_crc_table[5][(a >> 16) & 255] ^ sfc_crc_table[4][a >> 24] ^
        sfc_crc_table[3][b & 255] ^ sfc_crc_table[2][(b >> 8) & 255] ^
        sfc_crc_table[1][(b >> 16) & 255] ^ sfc_crc_table[0][b >> 24];
    p += 8;
    n -= 8;
  }
  while (n--)
    c = sfc_crc_table[0][(c ^ *p++) & 255] ^ (c >> 8);
  return c;
}
uint32_t sfc_crc32(const void *data, size_t n) {
  return ~crc_update(~0u, data, n);
}
static uint32_t header_crc(const uint8_t h[64]) {
  uint32_t c = crc_update(~0u, h, 48);
  if (u32(h + 4) >= 3)
    c = crc_update(c, h + 52, 12);
  return ~c;
}
static uint32_t index_crc(uint32_t channel, uint64_t block,
                          const uint8_t e[32]) {
  uint8_t identity[12];
  p32(identity, channel);
  p64(identity + 4, block);
  return ~crc_update(crc_update(~0u, identity, 12), e, 28);
}
static uint32_t bits_get(const void *p, sfc_dtype t) {
  uint32_t v = 0;
  if (t == SFC_U16) {
    uint16_t x;
    memcpy(&x, p, 2);
    v = x;
  } else if (t == SFC_U8)
    v = *(const uint8_t *)p;
  else
    memcpy(&v, p, 4);
  return v;
}
static void bits_put(void *p, sfc_dtype t, uint32_t v) {
  if (t == SFC_U16) {
    uint16_t x = (uint16_t)v;
    memcpy(p, &x, 2);
  } else if (t == SFC_U8)
    *(uint8_t *)p = (uint8_t)v;
  else
    memcpy(p, &v, 4);
}
static double value(uint32_t v, sfc_dtype t) {
  if (t != SFC_F32)
    return v;
  float f;
  memcpy(&f, &v, 4);
  return f;
}
static uint32_t cast_value(double d, sfc_dtype t) {
  if (t == SFC_U8)
    return (uint32_t)fmax(0, fmin(255, floor(d + .5)));
  if (t == SFC_U16)
    return (uint32_t)fmax(0, fmin(65535, floor(d + .5)));
  float f = (float)d;
  uint32_t b;
  memcpy(&b, &f, 4);
  return b;
}
/* The required compiler policy preserves this check across FP promotion. */
static int finite_number(double v) {
  uint64_t bits;
  memcpy(&bits, &v, 8);
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
/* Float32 transform inputs and coefficients; widened inverse accumulation
 * prevents large smooth blocks from spending their error budget on summation.
 * Neither direction uses integer transform arithmetic. */
static void dct(const float *restrict in, float *restrict out, int inverse) {
  /* The stored float32 basis has exact even/odd mirror symmetry. Pairing
   * mirrored samples halves products without substituting an approximate
   * cosine basis. Widened inverse sums preserve the reconstruction contract. */
  if (inverse == 2) {
    float tmp[4096];
    unsigned active[64], count = 0;
    for (unsigned y = 0; y < 64; y++) {
      float halves[2][32] = {{0}};
      int used = 0;
      for (unsigned j = 0; j < 64; j++) {
        float v = in[y * 64 + j];
        if (v == 0)
          continue;
        used = 1;
        float *row = halves[j & 1];
        for (unsigned k = 0; k < 32; k++)
          row[k] += v * sfc_basis[j][k];
      }
      for (unsigned k = 0; k < 32; k++) {
        tmp[y * 64 + k] = halves[0][k] + halves[1][k];
        tmp[y * 64 + 63 - k] = halves[0][k] - halves[1][k];
      }
      if (used)
        active[count++] = y;
    }
    for (unsigned k = 0; k < 32; k++) {
      float halves[2][64] = {{0}};
      for (unsigned a = 0; a < count; a++) {
        unsigned j = active[a];
        float v = sfc_basis[j][k];
        float *row = halves[j & 1];
        for (unsigned x = 0; x < 64; x++)
          row[x] += tmp[j * 64 + x] * v;
      }
      for (unsigned x = 0; x < 64; x++) {
        out[k * 64 + x] = (float)(halves[0][x] + halves[1][x]);
        out[(63 - k) * 64 + x] = (float)(halves[0][x] - halves[1][x]);
      }
    }
    return;
  }
  if (inverse) {
    double tmp[4096];
    unsigned active[64], count = 0;
    for (unsigned y = 0; y < 64; y++) {
      double halves[2][32] = {{0}};
      int used = 0;
      for (unsigned j = 0; j < 64; j++) {
        double v = (double)in[y * 64 + j];
        if (v == 0)
          continue;
        used = 1;
        double *row = halves[j & 1];
        for (unsigned k = 0; k < 32; k++)
          row[k] += v * (double)sfc_basis[j][k];
      }
      for (unsigned k = 0; k < 32; k++) {
        tmp[y * 64 + k] = halves[0][k] + halves[1][k];
        tmp[y * 64 + 63 - k] = halves[0][k] - halves[1][k];
      }
      if (used)
        active[count++] = y;
    }
    for (unsigned k = 0; k < 32; k++) {
      double halves[2][64] = {{0}};
      for (unsigned a = 0; a < count; a++) {
        unsigned j = active[a];
        double v = (double)sfc_basis[j][k];
        double *row = halves[j & 1];
        for (unsigned x = 0; x < 64; x++)
          row[x] += tmp[j * 64 + x] * v;
      }
      for (unsigned x = 0; x < 64; x++) {
        out[k * 64 + x] = (float)(halves[0][x] + halves[1][x]);
        out[(63 - k) * 64 + x] = (float)(halves[0][x] - halves[1][x]);
      }
    }
    return;
  }
  float tmp[4096];
  for (unsigned y = 0; y < 64; y++) {
    float pairs[2][32];
    for (unsigned j = 0; j < 32; j++) {
      pairs[0][j] = in[y * 64 + j] + in[y * 64 + 63 - j];
      pairs[1][j] = in[y * 64 + j] - in[y * 64 + 63 - j];
    }
    for (unsigned k = 0; k < 64; k++) {
      float v = 0;
      for (unsigned j = 0; j < 32; j++)
        v += pairs[k & 1][j] * sfc_basis[k][j];
      tmp[y * 64 + k] = v;
    }
  }
  for (unsigned j = 0; j < 32; j++)
    for (unsigned x = 0; x < 64; x++) {
      float a = tmp[j * 64 + x], b = tmp[(63 - j) * 64 + x];
      tmp[j * 64 + x] = a + b;
      tmp[(63 - j) * 64 + x] = a - b;
    }
  for (unsigned k = 0; k < 64; k++) {
    float row[64] = {0};
    for (unsigned j = 0; j < 32; j++) {
      float v = sfc_basis[k][j];
      const float *pair = tmp + ((k & 1) ? 63 - j : j) * 64;
      for (unsigned x = 0; x < 64; x++)
        row[x] += pair[x] * v;
    }
    memcpy(out + k * 64, row, sizeof row);
  }
}
static double float_ulp(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  unsigned exponent = (bits >> 23) & 255;
  if (exponent == 255)
    return DBL_MAX;
  /* Spacing toward increasing magnitude, including subnormals and FLT_MAX. */
  bits = exponent >= 24 ? (exponent - 23) << 23
                        : 1u << (exponent ? exponent - 1 : 0);
  float spacing;
  memcpy(&spacing, &bits, 4);
  return (double)spacing;
}
static size_t varput(uint8_t *p, uint64_t v) {
  size_t n = 0;
  do {
    p[n] = (uint8_t)(v & 127);
    v >>= 7;
    if (v)
      p[n] |= 128;
    n++;
  } while (v);
  return n;
}
static int varget(const uint8_t *p, size_t n, size_t *at, uint64_t *v) {
  *v = 0;
  for (unsigned s = 0; s < 64; s += 7) {
    if (*at >= n)
      return -1;
    uint8_t b = p[(*at)++];
    if (s == 63 && b > 1)
      return -1;
    *v |= (uint64_t)(b & 127) << s;
    if (!(b & 128))
      return 0;
  }
  return -1;
}
// clang-format off: block_entropy uses the variable-length mask parser.
#include "mask.h"
#include "block_entropy.h"
// clang-format on

static void sample_le(uint8_t *p, uint32_t v, size_t z) {
  for (size_t j = 0; j < z; j++)
    p[j] = (uint8_t)(v >> (8 * j));
}
static uint32_t sample_read(const uint8_t *p, size_t z) {
  uint32_t v = 0;
  for (size_t j = 0; j < z; j++)
    v |= (uint32_t)p[j] << (8 * j);
  return v;
}
static int strides(size_t p, size_t r, size_t z, size_t w, size_t h) {
  return z && p >= z && w && h && (w - 1) <= (SIZE_MAX - z) / p &&
         r >= (w - 1) * p + z && (h - 1) <= (SIZE_MAX - ((w - 1) * p + z)) / r;
}
#include "inpaint.h"

typedef struct {
  uint8_t buffers[2][SFC_BLOCK_BOUND];
} sfc_encode_scratch;
typedef struct {
  uint8_t *bytes;
  size_t size;
  double square;
} sfc_choice;
typedef struct {
  sfc_choice values[4];
  unsigned count, preferred;
} sfc_candidates;
static int save_choice(sfc_candidates *all, const uint8_t *b, size_t n,
                       double square) {
  if (!all)
    return 0;
  if (all->count >= 4)
    return SFC_LIMIT;
  sfc_choice *c = all->values + all->count;
  if (!c->bytes)
    c->bytes = malloc(SFC_BLOCK_BOUND);
  if (!c->bytes)
    return SFC_NOMEM;
  memcpy(c->bytes, b, n);
  c->size = pack_mask(c->bytes, n);
  c->square = square;
  all->count++;
  return 0;
}
static void free_choices(sfc_candidates *all) {
  for (unsigned i = 0; i < 4; i++)
    free(all->values[i].bytes);
}
static int encode_candidates(const sfc_channel *c, const void *src, size_t ps,
                             size_t rs, const uint8_t *mask, uint8_t **enc,
                             size_t *size, sfc_encode_scratch *scratch,
                             sfc_candidates *all) {
  if (!enc || !size)
    return SFC_INVALID;
  *enc = NULL;
  *size = 0;
  size_t z = c ? sfc_sample_size(c->dtype) : 0;
  if (!c || !src || !strides(ps, rs, z, 64, 64) || c->flags & ~3u ||
      !finite_number(c->tolerance) || c->tolerance < 0 ||
      (!(c->flags & SFC_EXACT) && c->tolerance <= 0))
    return SFC_INVALID;
  uint8_t *b = scratch->buffers[0], *raw = scratch->buffers[1];
  if (all) {
    all->count = 0;
    all->preferred = 0;
  }
  /* Headers/masks need zeroing; all emitted sample bytes are overwritten.
   * Worst temporary varint stream: <=6 bytes per coefficient including runs.
   * Escapes stop before exceeding the current raw-size incumbent. */
  _Static_assert(SFC_BLOCK_BOUND >= 544 + 4096 * 6, "block scratch bound");
  memset(b, 0, 544);
  uint32_t values[4096];
  unsigned valid_count = 0;
  uint8_t flags = (c->flags & SFC_COORDINATE) ? 4 : 0;
  for (unsigned i = 0; i < 4096; i++) {
    int valid = !mask || mask[i];
    valid_count += (unsigned)valid;
    values[i] =
        valid ? bits_get((const uint8_t *)src + (i / 64) * rs + (i % 64) * ps,
                         c->dtype)
              : cast_value(flags & 4 ? -1 : 0, c->dtype);
  }
  if (!valid_count)
    flags |= 2;
  else if (valid_count != 4096)
    flags |= 1;
  memcpy(b, "B641", 4);
  b[4] = (uint8_t)c->dtype;
  b[6] = flags;
  size_t begin = 32;
  if (flags & 1) {
    for (unsigned i = 0; i < 4096; i++)
      if (mask[i])
        b[32 + i / 8] |= (uint8_t)(1u << (i % 8));
    begin += 512;
  }
  memcpy(raw, b, begin);
  size_t rn = begin;
  raw[5] = 2;
  if (valid_count)
    for (unsigned i = 0; i < 4096;) {
      unsigned j = i + 1;
      while (j < 4096 && values[j] == values[i])
        j++;
      rn += varput(raw + rn, j - i);
      sample_le(raw + rn, values[i], z);
      rn += z;
      i = j;
    }
  if (rn > begin + 4096 * z) {
    raw[5] = 0;
    rn = begin + 4096 * z;
    for (unsigned i = 0; i < 4096; i++)
      sample_le(raw + begin + i * z, values[i], z);
  }
  int status = save_choice(all, raw, rn, 0);
  if (status)
    return status;
  if (!valid_count || c->flags & SFC_EXACT)
    goto use_raw;
  double base = 0;
  for (unsigned i = 0; i < 4096; i++)
    if ((!mask || mask[i]) && finite_number(value(values[i], c->dtype))) {
      base = value(values[i], c->dtype);
      break;
    }
  float input[4096], transformed[4096], recon[4096], last = 0;
  uint8_t known[4096];
  int32_t coef[4096];
  for (unsigned i = 0; i < 4096; i++) {
    double v = value(values[i], c->dtype);
    known[i] = (!mask || mask[i]) && finite_number(v);
    if (known[i]) {
      double d = v - base;
      if (fabs(d) > 1e30)
        goto use_raw;
      last = (float)d;
    }
    input[i] = last;
  }
  if (flags & 1)
    fill_invalid(input, known);
  /* Center after extending holes, so the DCT's DC and numerical allowance
   * depend on local variation, not the arbitrary first sample. */
  double mean = 0;
  for (unsigned i = 0; i < 4096; i++)
    mean += (double)input[i];
  mean /= 4096;
  base += mean;
  for (unsigned i = 0; i < 4096; i++)
    input[i] = (float)((double)input[i] - mean);
  dct(input, transformed, 0);
  /* One forward DCT, three quantization candidates. Choosing by final byte
   * count avoids paying for extra exceptions just to zero more coefficients.
   * The original step remains a candidate, so this cannot enlarge a block. */
  const int depth_channel =
      (c->flags & SFC_COORDINATE) && c->name[0] == 'z' && !c->name[1];
  /* Probe the usually smaller coarse representation first, so its byte bar
   * prunes finer candidates before inverse transforms. Keep a lower-distortion
   * result within 2% of the smallest completed candidate, never above raw. */
  const size_t raw_size = rn;
  size_t smallest = rn;
  double best_square = 0;
  static const float factors[] = {2.0f, 1.0f, .5f};
  for (unsigned candidate = 0; candidate < sizeof factors / sizeof factors[0];
       candidate++) {
    size_t bar = smallest + smallest / 50;
    if (bar > raw_size)
      bar = raw_size;
    if (all)
      bar = SFC_BLOCK_BOUND; /* Final entropy size, not varint size, decides. */
    float step = (float)(c->tolerance * (double)factors[candidate]);
    if (!finite_number(step) || step < 1e-30f || step > 1e30f)
      goto next_candidate;
    float reciprocal = 1.0f / step;
    double l1 = 0;
    for (unsigned i = 0; i < 4096; i++) {
      float q = roundf(transformed[i] * reciprocal);
      if (!finite_number(q) || fabsf(q) > 4194304)
        goto next_candidate;
      coef[i] = (int32_t)q;
      recon[i] = (float)coef[i] * step;
      l1 += fabs((double)recon[i]);
    }
    b[5] = 1;
    pd(b + 8, base);
    uint32_t step_bits;
    memcpy(&step_bits, &step, 4);
    p32(b + 16, step_bits);
    p32(b + 20, 1); /* float32 inverse, explicit bounded-error contract */
    size_t n = begin;
    unsigned pos = 0;
    while (pos < 4096) {
      unsigned run = 0;
      while (pos + run < 4096 && !coef[pos + run])
        run++;
      n += varput(b + n, run);
      pos += run;
      if (pos == 4096)
        break;
      int64_t v = coef[pos++];
      n += varput(b + n, v < 0 ? (uint64_t)(-v) * 2 - 1 : (uint64_t)v * 2);
    }
    p32(b + 24, (uint32_t)(n - begin));
    if (n >= bar)
      goto next_candidate;
    /* Reconstruction checks, including rounding margin, enforce the same
     * tolerance for every candidate. Larger steps may need more escapes. */
    dct(recon, input, 2);
    double square = 0;
    unsigned exceptions = 0, prev = 0;
    for (unsigned i = 0; i < 4096; i++) {
      if (mask && !mask[i])
        continue;
      double orig = value(values[i], c->dtype);
      double estimate = base + (double)input[i];
      if (!finite_number(estimate) ||
          (c->dtype == SFC_F32 && fabs(estimate) > FLT_MAX))
        goto next_candidate;
      uint32_t v = cast_value(estimate, c->dtype);
      /* Both encoder and decoder may deviate from the exact basis sum. */
      double drift = float_ulp(input[i]) + 8e-7 * l1 + 2e-34;
      double error = fabs(value(v, c->dtype) - orig);
      if (c->dtype == SFC_F32)
        error += drift + float_ulp((float)value(v, c->dtype));
      else {
        /* Integer output rounding is discontinuous at half-integers. Check
         * both ends of the numerical interval before accepting the sample. */
        double low = value(cast_value(estimate - drift, c->dtype), c->dtype);
        double high = value(cast_value(estimate + drift, c->dtype), c->dtype);
        error = fmax(fabs(low - orig), fabs(high - orig));
      }
      /* tifxyz uses nonpositive Z as invalid. A lossy sample must not cross
       * that semantic boundary even when its absolute error would be allowed.
       */
      int invalid_depth = depth_channel && orig > 0 && value(v, c->dtype) <= 0;
      if (!finite_number(orig) || !finite_number(value(v, c->dtype)) ||
          invalid_depth || error > c->tolerance) {
        if (n + 16 >= bar)
          goto next_candidate;
        n += varput(b + n, i - prev);
        prev = i;
        sample_le(b + n, values[i], z);
        n += z;
        exceptions++;
      } else {
        double d = value(v, c->dtype) - orig;
        square += d * d;
      }
    }
    p32(b + 28, exceptions);
    if (n >= bar)
      goto next_candidate;
    status = save_choice(all, b, n, square);
    if (status)
      return status;
    if (n < smallest)
      smallest = n;
    if (rn <= smallest + smallest / 50 && square >= best_square)
      goto next_candidate;
    best_square = square;
    if (all)
      all->preferred = all->count - 1;
    /* Swap two bounded buffers; prefer lower distortion near equal sizes. */
    uint8_t *previous = raw;
    raw = b;
    b = previous;
    rn = n;
    memcpy(b, raw, begin);
  next_candidate:;
  }
use_raw:
  *enc = raw;
  *size = pack_mask(raw, rn);
  return 0;
}
int sfc_encode_block(const sfc_channel *c, const void *src, size_t ps,
                     size_t rs, const uint8_t *mask, uint8_t **encoded,
                     size_t *size) {
  if (!encoded || !size)
    return SFC_INVALID;
  *encoded = NULL;
  *size = 0;
  sfc_encode_scratch *scratch = malloc(sizeof *scratch);
  if (!scratch)
    return SFC_NOMEM;
  uint8_t *borrowed = NULL;
  size_t n = 0;
  int rc =
      encode_candidates(c, src, ps, rs, mask, &borrowed, &n, scratch, NULL);
  if (!rc) {
    *encoded = malloc(n);
    if (!*encoded)
      rc = SFC_NOMEM;
    else {
      memcpy(*encoded, borrowed, n);
      *size = n;
    }
  }
  free(scratch);
  return rc;
}
#include "select_entropy.h"

static int decode_block(const uint8_t *b, size_t n, sfc_dtype t, void *dst,
                        size_t ps, size_t rs, uint8_t *mask,
                        const se_model *models) {
  size_t z = sfc_sample_size(t);
  if (!b || !dst || !strides(ps, rs, z, 64, 64) || n < 32 ||
      n > SFC_BLOCK_BOUND || memcmp(b, "B641", 4) || b[4] != t || b[5] > 3 ||
      (b[5] == 3 && !models) || b[6] & ~15u || b[7] || u32(b + 20) > 1 ||
      ((b[6] & 3) == 3))
    return SFC_INVALID;
  if (b[5] != 1 && b[5] != 3)
    for (unsigned j = 8; j < 32; j++)
      if (b[j])
        return SFC_INVALID;
  unsigned flags = b[6];
  size_t at = 32;
  uint8_t valid[4096];
  uint32_t vals[4096];
  if (block_prefix(b, n, &at, valid))
    return SFC_INVALID;
  if (flags & 2) {
    if (n != 32 || b[5] != 2)
      return SFC_INVALID;
    memset(vals, 0, sizeof vals);
  } else if (b[5] == 0) {
    if (n - at != 4096 * z)
      return SFC_INVALID;
    for (unsigned i = 0; i < 4096; i++)
      vals[i] = sample_read(b + at + i * z, z);
  } else if (b[5] == 2) {
    unsigned i = 0;
    while (i < 4096) {
      uint64_t run;
      if (varget(b, n, &at, &run) || !run || run > 4096 - i || n - at < z)
        return SFC_INVALID;
      uint32_t v = sample_read(b + at, z);
      at += z;
      while (run--)
        vals[i++] = v;
    }
    if (at != n)
      return SFC_INVALID;
  } else {
    double base = gd(b + 8);
    uint32_t step_bits = u32(b + 16);
    float step;
    memcpy(&step, &step_bits, 4);
    uint32_t cn = u32(b + 24), ex = u32(b + 28);
    if (!finite_number(base) || !finite_number(step) || step < 1e-30f ||
        step > 1e30f || cn > n - at || ex > 4096)
      return SFC_INVALID;
    size_t end = at + cn;
    float coef[4096] = {0}, out[4096];
    unsigned i = 0;
    if (b[5] == 3) {
      if (entropy_decode(b + at, cn, models, step, coef))
        return SFC_INVALID;
      at = end;
    } else {
      while (i < 4096) {
        uint64_t run, v;
        if (varget(b, end, &at, &run) || run > 4096 - i)
          return SFC_INVALID;
        i += (unsigned)run;
        if (i == 4096)
          break;
        if (varget(b, end, &at, &v) || v > 8388608)
          return SFC_INVALID;
        int64_t signed_v = v & 1 ? -(int64_t)(v / 2) - 1 : (int64_t)(v / 2);
        coef[i++] = (float)signed_v * step;
      }
      if (at != end)
        return SFC_INVALID;
    }
    dct(coef, out, u32(b + 20) ? 2 : 1);
    for (i = 0; i < 4096; i++) {
      double v = base + (double)out[i];
      if (!finite_number(v) ||
          (t == SFC_F32 && fabs(v) > 3.4028234663852886e38))
        return SFC_INVALID;
      vals[i] = cast_value(v, t);
    }
    unsigned prev = 0;
    for (i = 0; i < ex; i++) {
      uint64_t delta;
      if (varget(b, n, &at, &delta) || delta > 4095 - prev || (i && !delta) ||
          n - at < z)
        return SFC_INVALID;
      prev += (unsigned)delta;
      vals[prev] = sample_read(b + at, z);
      at += z;
    }
    if (at != n)
      return SFC_INVALID;
  }
  for (unsigned i = 0; i < 4096; i++) {
    if (!valid[i])
      vals[i] = cast_value(flags & 4 ? -1 : 0, t);
    bits_put((uint8_t *)dst + (i / 64) * rs + (i % 64) * ps, t, vals[i]);
  }
  if (mask)
    memcpy(mask, valid, 4096);
  return 0;
}

int sfc_decode_block(const uint8_t *b, size_t n, sfc_dtype t, void *dst,
                     size_t ps, size_t rs, uint8_t *mask) {
  return decode_block(b, n, t, dst, ps, rs, mask, NULL);
}

#include "xyz.h"

#define HEADER 64u
#define DESC 128u
#define ENTRY 32u
#define META_LIMIT (64u << 20)
#define SFC_TABLE_CACHE 16u
#define SFC_READ_SCRATCH 8u
typedef struct {
  uint64_t offset;
  uint32_t size, crc;
  se_model *models;
  unsigned refs;
} sfc_table_cache;
typedef struct {
  uint8_t bytes[SFC_XYZ_BOUND];
  sfc_xyz_decode xyz;
} sfc_read_work;
struct sfc_reader {
  sfc_read_at read;
  void *user;
  FILE *owned;
  uint64_t size, meta_off, meta_len, data_off;
  uint32_t count, version;
  atomic_flag table_lock;
  unsigned next_table;
  sfc_table_cache table_cache[SFC_TABLE_CACHE];
  struct {
    atomic_flag used;
    sfc_read_work *work;
  } scratch[SFC_READ_SCRATCH];
  sfc_channel *channels;
  uint64_t *indices;
};
typedef struct {
  uint8_t *packet;
  sfc_candidates choices;
  size_t size;
  uint64_t index, block;
  uint32_t channel;
  uint8_t entry[32];
} sfc_pending;
struct sfc_writer {
  FILE *f;
  char *path;
  sfc_channel *channels;
  uint64_t *indices;
  uint32_t count, current;
  uint64_t block, data_off;
  uint8_t header[64];
  sfc_encode_scratch *encode_scratch;
  sfc_entropy_work *entropy_work;
  sfc_xyz_work *xyz_work;
  unsigned pending_count;
  int failed;
  sfc_pending pending[SFC_GROUP];
};
static uint64_t tiles(uint64_t n) { return n / 64 + (n % 64 != 0); }
static int channel_ok(const sfc_channel *c) {
  return memchr(c->name, 0, 64) && c->name[0] && sfc_sample_size(c->dtype) &&
         c->width && c->height && c->width <= INT64_MAX &&
         c->height <= INT64_MAX &&
         tiles(c->width) <= UINT64_MAX / tiles(c->height) &&
         !(c->flags & ~7u) &&
         (!(c->flags & SFC_XYZ) ||
          (c->flags == (SFC_XYZ | SFC_COORDINATE) && c->dtype == SFC_F32 &&
           c->components == 3)) &&
         finite_number(c->tolerance) && c->components <= SFC_MAX_CHANNELS &&
         c->component < (c->components ? c->components : 1) &&
         c->tolerance >= 0 && ((c->flags & SFC_EXACT) || c->tolerance > 0);
}
/* Multichannel images use contiguous, explicitly numbered descriptors. */
static int groups_ok(const sfc_channel *channels, uint32_t count) {
  for (uint32_t i = 0; i < count;) {
    const sfc_channel *c = channels + i;
    uint32_t n = c->components ? c->components : 1;
    if (n > count - i || c->component)
      return 0;
    for (uint32_t k = 1; k < n; k++) {
      const sfc_channel *p = c + k;
      if (p->components != n || p->component != k || p->width != c->width ||
          p->height != c->height || p->dtype != c->dtype ||
          ((p->flags ^ c->flags) & SFC_XYZ) ||
          ((c->flags & SFC_XYZ) && p->tolerance != c->tolerance))
        return 0;
    }
    i += n;
  }
  return 1;
}
static int file_read(void *u, uint64_t off, void *dst, size_t n) {
  FILE *f = u;
  uint8_t *p = dst;
  if (off > INT64_MAX || n > (uint64_t)INT64_MAX - off)
    return -1;
  while (n) {
    ssize_t got = pread(fileno(f), p, n, (off_t)off);
    if (got <= 0)
      return -1;
    p += got;
    n -= (size_t)got;
    off += (uint64_t)got;
  }
  return 0;
}
static int read_bytes(sfc_reader *r, uint64_t off, void *p, size_t n) {
  return off > r->size || n > r->size - off ? SFC_INVALID
         : r->read(r->user, off, p, n)      ? SFC_IO
                                            : 0;
}
int sfc_open(sfc_read_at read, void *user, uint64_t size, sfc_reader **out) {
  if (!out)
    return SFC_INVALID;
  *out = NULL;
  if (!read || size < HEADER || size > INT64_MAX)
    return SFC_INVALID;
  uint8_t h[64];
  if (read(user, 0, h, 64))
    return SFC_IO;
  uint32_t count = u32(h + 8);
  uint64_t mo = u64(h + 16), ml = u64(h + 24), data = u64(h + 32);
  if (memcmp(h, "SFC1", 4) || (u32(h + 4) < 1 || u32(h + 4) > 4) || !count ||
      count > SFC_MAX_CHANNELS || u32(h + 12) ||
      mo != HEADER + (uint64_t)count * DESC || ml > META_LIMIT || mo > size ||
      ml > size - mo || data < mo + ml || data > size || u64(h + 40) != size ||
      u32(h + 48) != header_crc(h))
    return SFC_INVALID;
  for (unsigned i = u32(h + 4) >= 3 ? 56 : 52; i < 64; i++)
    if (h[i])
      return SFC_INVALID;
  sfc_reader *r = calloc(1, sizeof *r);
  if (!r)
    return SFC_NOMEM;
  atomic_flag_clear(&r->table_lock);
  for (unsigned i = 0; i < SFC_READ_SCRATCH; i++)
    atomic_flag_clear(&r->scratch[i].used);
  r->version = u32(h + 4);
  r->read = read;
  r->user = user;
  r->size = size;
  r->count = count;
  r->meta_off = mo;
  r->meta_len = ml;
  r->data_off = data;
  r->channels = calloc(count, sizeof *r->channels);
  r->indices = calloc(count, sizeof *r->indices);
  if (!r->channels || !r->indices) {
    sfc_close(r);
    return SFC_NOMEM;
  }
  uint32_t description_crc = ~0u;
  uint64_t expected = mo + ml;
  int rc = SFC_INVALID;
  for (uint32_t i = 0; i < count; i++) {
    uint8_t d[128];
    if (read_bytes(r, HEADER + (uint64_t)i * DESC, d, 128)) {
      rc = SFC_IO;
      goto fail;
    }
    description_crc = crc_update(description_crc, d, 128);
    sfc_channel *c = r->channels + i;
    memcpy(c->name, d, 64);
    c->width = u64(d + 64);
    c->height = u64(d + 72);
    c->dtype = (sfc_dtype)u32(d + 80);
    c->flags = u32(d + 84);
    c->tolerance = gd(d + 88);
    c->components = u32(d + 112);
    c->component = u32(d + 116);
    if (!channel_ok(c) || (r->version < 4 && (c->flags & SFC_XYZ)))
      goto fail;
    for (uint32_t j = 0; j < i; j++)
      if (!strcmp(c->name, r->channels[j].name))
        goto fail;
    uint64_t n = tiles(c->width) * tiles(c->height);
    if (n > (data - expected) / ENTRY || u64(d + 96) != expected ||
        u64(d + 104) != n)
      goto fail;
    r->indices[i] = expected;
    expected += n * ENTRY;
    for (unsigned j = 120; j < 128; j++)
      if (d[j])
        goto fail;
  }
  if (expected != data || !groups_ok(r->channels, count))
    goto fail;
  if (r->version >= 3) {
    uint8_t chunk[16384];
    for (uint64_t off = 0; off < ml;) {
      size_t n = ml - off > sizeof chunk ? sizeof chunk : (size_t)(ml - off);
      if (read_bytes(r, mo + off, chunk, n)) {
        rc = SFC_IO;
        goto fail;
      }
      description_crc = crc_update(description_crc, chunk, n);
      off += n;
    }
    if (~description_crc != u32(h + 52))
      goto fail;
  }
  *out = r;
  return 0;
fail:
  sfc_close(r);
  return rc;
}
int sfc_open_file(const char *path, sfc_reader **out) {
  if (!path || !out)
    return SFC_INVALID;
  *out = NULL;
  FILE *f = fopen(path, "rb");
  if (!f)
    return SFC_IO;
  if (fseeko(f, 0, SEEK_END)) {
    fclose(f);
    return SFC_IO;
  }
  off_t n = ftello(f);
  if (n < 0) {
    fclose(f);
    return SFC_IO;
  }
  int rc = sfc_open(file_read, f, (uint64_t)n, out);
  if (rc)
    fclose(f);
  else
    (*out)->owned = f;
  return rc;
}
void sfc_close(sfc_reader *r) {
  if (!r)
    return;
  if (r->owned)
    fclose(r->owned);
  for (unsigned i = 0; i < SFC_TABLE_CACHE; i++)
    free(r->table_cache[i].models);
  for (unsigned i = 0; i < SFC_READ_SCRATCH; i++)
    free(r->scratch[i].work);
  free(r->channels);
  free(r->indices);
  free(r);
}
uint32_t sfc_channel_count(const sfc_reader *r) { return r ? r->count : 0; }
const sfc_channel *sfc_channel_info(const sfc_reader *r, uint32_t c) {
  return r && c < r->count ? r->channels + c : NULL;
}
int sfc_find_channel(const sfc_reader *r, const char *name) {
  if (r && name)
    for (uint32_t i = 0; i < r->count; i++)
      if (!strcmp(r->channels[i].name, name))
        return (int)i;
  return -1;
}
uint64_t sfc_metadata_size(const sfc_reader *r) { return r ? r->meta_len : 0; }
int sfc_read_metadata(sfc_reader *r, void *dst, size_t n) {
  return !r || (!dst && n) || n != r->meta_len
             ? SFC_INVALID
             : read_bytes(r, r->meta_off, dst, n);
}
static int validate_entry(sfc_reader *r, uint32_t ch, uint64_t block,
                          const uint8_t d[32]) {
  uint64_t off = u64(d);
  uint32_t n = u32(d + 8);
  if (off < r->data_off || off > r->size || n < 32 ||
      n > ((r->channels[ch].flags & SFC_XYZ) ? SFC_XYZ_BOUND
                                             : SFC_BLOCK_BOUND) ||
      n > r->size - off || u32(d + 24) > 1 ||
      (r->version >= 3 ? u32(d + 28) != index_crc(ch, block, d)
                       : u32(d + 28) != 0))
    return SFC_INVALID;
  return 0;
}
static int entry(sfc_reader *r, uint32_t ch, uint64_t bx, uint64_t by,
                 uint8_t d[32]) {
  const sfc_channel *c = sfc_channel_info(r, ch);
  if (!c || bx >= tiles(c->width) || by >= tiles(c->height))
    return SFC_INVALID;
  int rc = read_bytes(r, r->indices[ch] + (by * tiles(c->width) + bx) * ENTRY,
                      d, 32);
  if (rc)
    return rc;
  return validate_entry(r, ch, by * tiles(c->width) + bx, d);
}
int sfc_block_range(sfc_reader *r, uint32_t ch, uint64_t bx, uint64_t by,
                    float *lo, float *hi) {
  if (!lo || !hi)
    return SFC_INVALID;
  uint8_t e[32];
  int rc = entry(r, ch, bx, by, e);
  if (rc)
    return rc;
  uint32_t a = u32(e + 16), b = u32(e + 20);
  memcpy(lo, &a, 4);
  memcpy(hi, &b, 4);
  if (u32(e + 24)) {
    *lo = 1;
    *hi = -1;
  } else if (!finite_number(*lo) || !finite_number(*hi) || *lo > *hi)
    return SFC_INVALID;
  return 0;
}
static void table_lock(sfc_reader *r) {
  while (
      atomic_flag_test_and_set_explicit(&r->table_lock, memory_order_acquire)) {
  }
}
static void table_unlock(sfc_reader *r) {
  atomic_flag_clear_explicit(&r->table_lock, memory_order_release);
}
/* Borrow immutable cache entries. The lock protects only pointer/refcount
 * changes; I/O, allocation, table building and decompression happen unlocked.
 */
static int read_tables(sfc_reader *r, uint64_t block_offset, const uint8_t *b,
                       size_t n, se_model **out, int *slot) {
  *out = NULL;
  *slot = -1;
  if (!b || n < 32)
    return SFC_INVALID;
  size_t at;
  if (block_prefix(b, n, &at, NULL))
    return SFC_INVALID;
  if (r->version < 2 || at > n || n - at < 24 || u32(b + 24) < 24 ||
      u32(b + 24) > n - at)
    return SFC_INVALID;
  const uint8_t *p = b + at;
  uint64_t offset = u64(p);
  uint32_t size = u32(p + 8), crc = u32(p + 12);
  if (size < 60 || size > SE_TABLES_MAX_BYTES || offset < r->data_off ||
      offset > block_offset || size > block_offset - offset)
    return SFC_INVALID;
  table_lock(r);
  for (unsigned i = 0; i < SFC_TABLE_CACHE; i++) {
    sfc_table_cache *c = r->table_cache + i;
    if (c->models && c->offset == offset && c->size == size && c->crc == crc) {
      c->refs++;
      *out = c->models;
      *slot = (int)i;
      table_unlock(r);
      return 0;
    }
  }
  table_unlock(r);
  uint8_t bytes[SE_TABLES_MAX_BYTES];
  int rc = read_bytes(r, offset, bytes, size);
  if (rc)
    return rc;
  if (sfc_crc32(bytes, size) != crc)
    return SFC_INVALID;
  se_model *models = malloc(sizeof(se_model) * SE_NMODELS);
  if (!models)
    return SFC_NOMEM;
  se_cur cursor = {bytes, bytes + size};
  if (!se_tables_read(&cursor, models) || cursor.p != cursor.end) {
    free(models);
    return SFC_INVALID;
  }
  table_lock(r);
  /* A concurrent miss may already have installed the same table. */
  for (unsigned i = 0; i < SFC_TABLE_CACHE; i++) {
    sfc_table_cache *c = r->table_cache + i;
    if (c->models && c->offset == offset && c->size == size && c->crc == crc) {
      c->refs++;
      *out = c->models;
      *slot = (int)i;
      table_unlock(r);
      free(models);
      return 0;
    }
  }
  se_model *old = NULL;
  for (unsigned i = 0; i < SFC_TABLE_CACHE; i++) {
    unsigned k = (r->next_table + i) % SFC_TABLE_CACHE;
    sfc_table_cache *c = r->table_cache + k;
    if (c->refs)
      continue;
    old = c->models;
    c->offset = offset;
    c->size = size;
    c->crc = crc;
    c->models = models;
    c->refs = 1;
    r->next_table = (k + 1) % SFC_TABLE_CACHE;
    *slot = (int)k;
    break;
  }
  table_unlock(r);
  free(old);
  *out = models; /* If all entries were borrowed, caller owns this transient. */
  return 0;
}
static void release_tables(sfc_reader *r, se_model *models, int slot) {
  if (slot < 0) {
    free(models);
    return;
  }
  table_lock(r);
  r->table_cache[slot].refs--;
  table_unlock(r);
}
static sfc_read_work *borrow_read_work(sfc_reader *r, int *slot) {
  *slot = -1;
  for (unsigned i = 0; i < SFC_READ_SCRATCH; i++)
    if (!atomic_flag_test_and_set_explicit(&r->scratch[i].used,
                                           memory_order_acquire)) {
      *slot = (int)i;
      if (!r->scratch[i].work)
        r->scratch[i].work = malloc(sizeof(sfc_read_work));
      return r->scratch[i].work;
    }
  return malloc(sizeof(sfc_read_work));
}
static void release_read_work(sfc_reader *r, sfc_read_work *w, int slot) {
  if (slot < 0)
    free(w);
  else
    atomic_flag_clear_explicit(&r->scratch[slot].used, memory_order_release);
}
static int read_block_entry(sfc_reader *r, uint32_t ch, const uint8_t e[32],
                            void *dst, size_t ps, size_t rs, uint8_t *mask) {
  uint32_t n = u32(e + 8);
  int scratch;
  sfc_read_work *work = borrow_read_work(r, &scratch);
  uint8_t *b = work ? work->bytes : NULL;
  int rc = b ? read_bytes(r, u64(e), b, n) : SFC_NOMEM;
  if (!rc &&
      (sfc_crc32(b, n) != u32(e + 12) || (r->version < 3 && u32(b + 20)) ||
       (r->version < 4 && (b[6] & 8))))
    rc = SFC_INVALID;
  if (!rc && (r->channels[ch].flags & SFC_XYZ)) {
    if (!dst || !strides(ps, rs, 4, 64, 64) || n < 128 ||
        gd(b + 8) != r->channels[ch].tolerance)
      rc = SFC_INVALID;
    else
      rc = xyz_unpack(b, n, &work->xyz);
    if (!rc) {
      unsigned component = r->channels[ch].component;
      for (unsigned i = 0; i < 4096; i++)
        memcpy((uint8_t *)dst + (i / 64) * rs + (i % 64) * ps,
               work->xyz.original[i] + component, 4);
      if (mask)
        memcpy(mask, work->xyz.valid, 4096);
    }
  } else if (!rc && b[5] == 3) {
    se_model *models;
    int slot;
    rc = read_tables(r, u64(e), b, n, &models, &slot);
    if (!rc) {
      rc = decode_block(b, n, r->channels[ch].dtype, dst, ps, rs, mask, models);
      release_tables(r, models, slot);
    }
  } else if (!rc)
    rc = sfc_decode_block(b, n, r->channels[ch].dtype, dst, ps, rs, mask);
  release_read_work(r, work, scratch);
  return rc;
}
int sfc_read_block(sfc_reader *r, uint32_t ch, uint64_t bx, uint64_t by,
                   void *dst, size_t ps, size_t rs, uint8_t *mask) {
  uint8_t e[32];
  int rc = entry(r, ch, bx, by, e);
  return rc ? rc : read_block_entry(r, ch, e, dst, ps, rs, mask);
}
int sfc_read_xyz(sfc_reader *r, uint32_t ch, uint64_t bx, uint64_t by,
                 void *dst, size_t ps, size_t rs, uint8_t *mask) {
  const sfc_channel *c = sfc_channel_info(r, ch);
  if (!c || ch + 2 >= r->count || !dst || !strides(ps, rs, 12, 64, 64))
    return SFC_INVALID;
  for (unsigned k = 0; k < 3; k++) {
    const sfc_channel *p = c + k;
    if (p->dtype != SFC_F32 || !(p->flags & SFC_COORDINATE) ||
        p->width != c->width || p->height != c->height)
      return SFC_INVALID;
  }
  if (c->flags & SFC_XYZ) {
    if (c->component)
      return SFC_INVALID;
    uint8_t e[32];
    int rc = entry(r, ch, bx, by, e);
    if (rc)
      return rc;
    /* Each component has its own position-bound index checksum. */
    for (unsigned k = 1; k < 3; k++) {
      uint8_t other[32];
      rc = entry(r, ch + k, bx, by, other);
      if (rc)
        return rc;
      if (memcmp(e, other, 16))
        return SFC_INVALID;
    }
    size_t n = u32(e + 8);
    int slot;
    sfc_read_work *work = borrow_read_work(r, &slot);
    if (!work) {
      release_read_work(r, work, slot);
      return SFC_NOMEM;
    }
    uint8_t *b = work->bytes;
    rc = read_bytes(r, u64(e), b, n);
    if (!rc && (sfc_crc32(b, n) != u32(e + 12) || n < 128 ||
                gd(b + 8) != c->tolerance))
      rc = SFC_INVALID;
    if (!rc)
      rc = xyz_unpack(b, n, &work->xyz);
    if (!rc) {
      for (unsigned i = 0; i < 4096; i++)
        memcpy((uint8_t *)dst + (i / 64) * rs + (i % 64) * ps,
               work->xyz.original[i], 12);
      if (mask)
        memcpy(mask, work->xyz.valid, 4096);
    }
    release_read_work(r, work, slot);
    return rc;
  }
  float (*tmp)[3] = malloc(4096 * 12);
  if (!tmp)
    return SFC_NOMEM;
  uint8_t valid[4096], other[4096];
  int rc = 0;
  for (unsigned k = 0; !rc && k < 3; k++) {
    rc = sfc_read_block(r, ch + k, bx, by, tmp[0] + k, 12, 768,
                        k ? other : valid);
    if (!rc && k)
      for (unsigned i = 0; i < 4096; i++)
        valid[i] &= other[i];
  }
  if (!rc) {
    for (unsigned i = 0; i < 4096; i++) {
      if (!valid[i])
        tmp[i][0] = tmp[i][1] = tmp[i][2] = -1;
      memcpy((uint8_t *)dst + (i / 64) * rs + (i % 64) * ps, tmp[i], 12);
    }
    if (mask)
      memcpy(mask, valid, 4096);
  }
  free(tmp);
  return rc;
}
int sfc_read_region(sfc_reader *r, uint32_t ch, uint64_t x, uint64_t y,
                    uint32_t w, uint32_t h, void *dst, size_t ps, size_t rs,
                    uint8_t *valid) {
  const sfc_channel *c = sfc_channel_info(r, ch);
  size_t z = c ? sfc_sample_size(c->dtype) : 0;
  if (!c || !dst || !strides(ps, rs, z, w, h) || x > c->width ||
      y > c->height || w > c->width - x || h > c->height - y ||
      (uint64_t)w * h > SIZE_MAX / (z + 1))
    return SFC_INVALID;
  size_t pixels = (size_t)w * h;
  uint8_t *tmp = malloc(pixels * z), *mask = malloc(pixels), block[16384],
          vm[4096];
  if (!tmp || !mask) {
    free(tmp);
    free(mask);
    return SFC_NOMEM;
  }
  int rc = 0;
  uint8_t entries[128][32];
  for (uint64_t by = y / 64; !rc && by <= (y + h - 1) / 64; by++) {
    uint64_t stop = (x + w - 1) / 64 + 1;
    for (uint64_t start = x / 64; !rc && start < stop;) {
      unsigned count = stop - start > 128 ? 128 : (unsigned)(stop - start);
      uint64_t first = by * tiles(c->width) + start;
      rc =
          read_bytes(r, r->indices[ch] + first * ENTRY, entries, count * ENTRY);
      if (rc)
        break;
      for (unsigned k = 0; !rc && k < count; k++) {
        uint64_t bx = start + k;
        rc = validate_entry(r, ch, first + k, entries[k]);
        if (!rc)
          rc = read_block_entry(r, ch, entries[k], block, z, 64 * z, vm);
        if (rc)
          break;
        uint64_t x0 = x > bx * 64 ? x : bx * 64, y0 = y > by * 64 ? y : by * 64;
        uint64_t x1 = x + w < (bx + 1) * 64 ? x + w : (bx + 1) * 64;
        uint64_t y1 = y + h < (by + 1) * 64 ? y + h : (by + 1) * 64;
        for (uint64_t yy = y0; yy < y1; yy++) {
          size_t to = (size_t)(yy - y) * w + (size_t)(x0 - x);
          size_t from = (size_t)(yy - by * 64) * 64 + (size_t)(x0 - bx * 64);
          size_t width = (size_t)(x1 - x0);
          memcpy(tmp + to * z, block + from * z, width * z);
          memcpy(mask + to, vm + from, width);
        }
      }
      start += count;
    }
  }
  if (!rc) {
    for (uint32_t yy = 0; yy < h; yy++) {
      if (ps == z)
        memcpy((uint8_t *)dst + (size_t)yy * rs, tmp + (size_t)yy * w * z,
               (size_t)w * z);
      else
        for (uint32_t xx = 0; xx < w; xx++)
          memcpy((uint8_t *)dst + (size_t)yy * rs + (size_t)xx * ps,
                 tmp + ((size_t)yy * w + xx) * z, z);
    }
    if (valid)
      memcpy(valid, mask, pixels);
  }
  free(tmp);
  free(mask);
  return rc;
}
int sfc_create(const char *path, const sfc_channel *channels, uint32_t count,
               const void *meta, size_t ml, sfc_writer **out) {
  if (!out)
    return SFC_INVALID;
  *out = NULL;
  if (!path || !channels || !count || count > SFC_MAX_CHANNELS ||
      ml > META_LIMIT || (!meta && ml))
    return SFC_INVALID;
  uint64_t pos = HEADER + (uint64_t)count * DESC + ml;
  for (uint32_t i = 0; i < count; i++) {
    const sfc_channel *c = channels + i;
    if (!channel_ok(c))
      return SFC_INVALID;
    for (uint32_t j = 0; j < i; j++)
      if (!strcmp(c->name, channels[j].name))
        return SFC_INVALID;
    uint64_t n = tiles(c->width) * tiles(c->height);
    if (n > ((uint64_t)INT64_MAX - pos) / ENTRY)
      return SFC_LIMIT;
    pos += n * ENTRY;
  }
  if (!groups_ok(channels, count))
    return SFC_INVALID;
  sfc_writer *w = calloc(1, sizeof *w);
  if (!w)
    return SFC_NOMEM;
  w->path = strdup(path);
  w->channels = malloc(count * sizeof *channels);
  w->indices = calloc(count, sizeof *w->indices);
  if (!w->path || !w->channels || !w->indices) {
    sfc_cancel(w);
    return SFC_NOMEM;
  }
  memcpy(w->channels, channels, count * sizeof *channels);
  w->count = count;
  w->data_off = pos;
  w->f = fopen(path, "w+bx");
  if (!w->f) {
    sfc_cancel(w);
    return SFC_IO;
  }
  uint8_t *hd = w->header;
  memcpy(hd, "SFC1", 4);
  p32(hd + 4, 4);
  p32(hd + 8, count);
  p64(hd + 16, HEADER + (uint64_t)count * DESC);
  p64(hd + 24, ml);
  p64(hd + 32, pos);
  if (fwrite(hd, 1, HEADER, w->f) != HEADER)
    goto io;
  uint32_t description_crc = ~0u;
  uint64_t index = HEADER + (uint64_t)count * DESC + ml;
  for (uint32_t i = 0; i < count; i++) {
    uint8_t d[128] = {0};
    const sfc_channel *c = channels + i;
    memcpy(d, c->name, strlen(c->name));
    p64(d + 64, c->width);
    p64(d + 72, c->height);
    p32(d + 80, c->dtype);
    p32(d + 84, c->flags);
    pd(d + 88, c->tolerance);
    p32(d + 112, c->components);
    p32(d + 116, c->component);
    uint64_t n = tiles(c->width) * tiles(c->height);
    p64(d + 96, index);
    p64(d + 104, n);
    w->indices[i] = index;
    index += n * ENTRY;
    description_crc = crc_update(description_crc, d, DESC);
    if (fwrite(d, 1, DESC, w->f) != DESC)
      goto io;
  }
  if (ml && fwrite(meta, 1, ml, w->f) != ml)
    goto io;
  description_crc = crc_update(description_crc, meta, ml);
  p32(hd + 52, ~description_crc);
  if (fseeko(w->f, (off_t)pos, SEEK_SET))
    goto io;
  *out = w;
  return 0;
io:
  sfc_cancel(w);
  return SFC_IO;
}
/* Bounded encoder staging: at most 16 source packets and their token streams.
 * Table cost is charged to the group; retain original packets if not smaller.
 */
static int flush_blocks(sfc_writer *w) {
  unsigned count = w->pending_count;
  if (!count)
    return 0;
  if (!w->entropy_work)
    w->entropy_work = malloc(sizeof *w->entropy_work);
  if (!w->entropy_work)
    return SFC_NOMEM;
  sfc_candidates *choices[SFC_GROUP];
  for (unsigned i = 0; i < count; i++)
    choices[i] = &w->pending[i].choices;
  sfc_entropy_work *s = w->entropy_work;
  int rc = select_entropy(choices, count, s);
  if (rc)
    return rc;
  off_t start = ftello(w->f);
  if (start < 0)
    return SFC_IO;
  if (s->table_size &&
      fwrite(s->tables, 1, s->table_size, w->f) != s->table_size)
    return SFC_IO;
  uint32_t table_crc = sfc_crc32(s->tables, s->table_size);
  for (unsigned i = 0; i < count; i++) {
    sfc_pending *p = w->pending + i;
    uint8_t *b = s->best[i];
    size_t n = s->sizes[i];
    if (b[5] == 3) {
      size_t at;
      if (block_prefix(b, n, &at, NULL))
        return SFC_INVALID;
      p64(b + at, (uint64_t)start);
      p32(b + at + 8, (uint32_t)s->table_size);
      p32(b + at + 12, table_crc);
    }
    off_t offset = ftello(w->f);
    if (offset < 0 || (uint64_t)offset > (uint64_t)INT64_MAX - n ||
        fwrite(b, 1, n, w->f) != n)
      return SFC_IO;
    p64(p->entry, (uint64_t)offset);
    p32(p->entry + 8, (uint32_t)n);
    p32(p->entry + 12, sfc_crc32(b, n));
    p32(p->entry + 28, index_crc(p->channel, p->block, p->entry));
    if (fseeko(w->f, (off_t)p->index, SEEK_SET) ||
        fwrite(p->entry, 1, ENTRY, w->f) != ENTRY ||
        fseeko(w->f, offset + (off_t)n, SEEK_SET))
      return SFC_IO;
    p->packet = NULL;
  }
  w->pending_count = 0;
  return 0;
}
int sfc_write_block(sfc_writer *w, const void *src, size_t ps, size_t rs,
                    const uint8_t *mask) {
  if (!w || w->failed || w->current >= w->count ||
      w->pending_count >= SFC_GROUP)
    return SFC_INVALID;
  const sfc_channel *c = w->channels + w->current;
  if (c->flags & SFC_XYZ)
    return SFC_INVALID;
  uint8_t effective[4096];
  uint64_t bx = w->block % tiles(c->width), by = w->block / tiles(c->width);
  for (unsigned i = 0; i < 4096; i++)
    effective[i] = (!mask || mask[i]) && bx * 64 + i % 64 < c->width &&
                   by * 64 + i / 64 < c->height;
  size_t z = sfc_sample_size(c->dtype);
  if (!src || !strides(ps, rs, z, 64, 64))
    return SFC_INVALID;
  sfc_pending *pending = w->pending + w->pending_count;
  int rc = 0;
  if (!w->encode_scratch)
    w->encode_scratch = malloc(sizeof *w->encode_scratch);
  if (!w->encode_scratch)
    return SFC_NOMEM;
  uint8_t *borrowed;
  size_t n;
  rc = encode_candidates(c, src, ps, rs, effective, &borrowed, &n,
                         w->encode_scratch, &pending->choices);
  if (rc)
    return rc;
  sfc_choice *preferred = pending->choices.values + pending->choices.preferred;
  pending->packet = preferred->bytes;
  pending->size = preferred->size;
  w->pending_count++;
  pending->index = w->indices[w->current] + w->block * ENTRY;
  pending->channel = w->current;
  pending->block = w->block;
  uint8_t *e = pending->entry;
  memset(e, 0, ENTRY);
  double lo = DBL_MAX, hi = -DBL_MAX;
  for (unsigned i = 0; i < 4096; i++)
    if (effective[i]) {
      double v =
          value(bits_get((const uint8_t *)src + (i / 64) * rs + (i % 64) * ps,
                         c->dtype),
                c->dtype);
      if (finite_number(v)) {
        if (v < lo)
          lo = v;
        if (v > hi)
          hi = v;
      }
    }
  float flo = 0, fhi = 0;
  if (lo > hi)
    p32(e + 24, 1);
  else {
    flo = nextafterf((float)fmax(-FLT_MAX, lo - c->tolerance), -FLT_MAX);
    fhi = nextafterf((float)fmin(FLT_MAX, hi + c->tolerance), FLT_MAX);
    if (!finite_number(flo))
      flo = -FLT_MAX;
    if (!finite_number(fhi))
      fhi = FLT_MAX;
  }
  uint32_t bits;
  memcpy(&bits, &flo, 4);
  p32(e + 16, bits);
  memcpy(&bits, &fhi, 4);
  p32(e + 20, bits);
  if (w->pending_count == SFC_GROUP ||
      w->block + 1 == tiles(c->width) * tiles(c->height)) {
    rc = flush_blocks(w);
    if (rc) {
      w->failed = 1;
      return rc;
    }
  }
  w->block++;
  if (w->block == tiles(c->width) * tiles(c->height)) {
    w->current++;
    w->block = 0;
  }
  return 0;
}
int sfc_write_xyz_block(sfc_writer *w, const void *src, size_t ps, size_t rs,
                        const uint8_t *mask) {
  if (!w || w->failed || w->current >= w->count || !src ||
      !strides(ps, rs, 12, 64, 64))
    return SFC_INVALID;
  const sfc_channel *c = w->channels + w->current;
  if (!(c->flags & SFC_XYZ) || c->component || w->pending_count)
    return SFC_INVALID;
  if (!w->xyz_work)
    w->xyz_work = calloc(1, sizeof *w->xyz_work);
  if (!w->xyz_work)
    return SFC_NOMEM;
  uint64_t bx = w->block % tiles(c->width), by = w->block / tiles(c->width);
  uint8_t valid[4096];
  for (unsigned i = 0; i < 4096; i++)
    valid[i] = (!mask || mask[i]) && bx * 64 + i % 64 < c->width &&
               by * 64 + i / 64 < c->height;
  size_t n;
  int rc = xyz_encode(src, ps, rs, valid, c->tolerance, w->xyz_work, &n);
  if (rc)
    return rc;
  uint8_t *b = w->xyz_work->best;
  off_t off = ftello(w->f);
  if (off < 0 || (uint64_t)off > (uint64_t)INT64_MAX - n ||
      fwrite(b, 1, n, w->f) != n)
    goto io;
  uint32_t crc = sfc_crc32(b, n);
  for (unsigned k = 0; k < 3; k++) {
    uint8_t e[32] = {0};
    p64(e, (uint64_t)off);
    p32(e + 8, (uint32_t)n);
    p32(e + 12, crc);
    double lo = DBL_MAX, hi = -DBL_MAX;
    for (unsigned i = 0; i < 4096; i++)
      if (valid[i]) {
        float v;
        memcpy(&v, (const uint8_t *)src + (i / 64) * rs + (i % 64) * ps + 4 * k,
               4);
        lo = fmin(lo, v);
        hi = fmax(hi, v);
      }
    if (lo > hi)
      p32(e + 24, 1);
    else {
      pf(e + 16,
         nextafterf((float)fmax(-FLT_MAX, lo - c->tolerance), -FLT_MAX));
      pf(e + 20, nextafterf((float)fmin(FLT_MAX, hi + c->tolerance), FLT_MAX));
    }
    p32(e + 28, index_crc(w->current + k, w->block, e));
    if (fseeko(w->f, (off_t)(w->indices[w->current + k] + w->block * ENTRY),
               SEEK_SET) ||
        fwrite(e, 1, 32, w->f) != 32)
      goto io;
  }
  if (fseeko(w->f, off + (off_t)n, SEEK_SET))
    goto io;
  if (++w->block == tiles(c->width) * tiles(c->height)) {
    w->block = 0;
    w->current += 3;
  }
  return 0;
io:
  w->failed = 1;
  return SFC_IO;
}
void sfc_cancel(sfc_writer *w) {
  if (!w)
    return;
  if (w->f) {
    fclose(w->f);
    if (w->path)
      unlink(w->path);
  }
  for (unsigned i = 0; i < SFC_GROUP; i++)
    free_choices(&w->pending[i].choices);
  free(w->encode_scratch);
  free(w->entropy_work);
  xyz_free(w->xyz_work);
  free(w->path);
  free(w->channels);
  free(w->indices);
  free(w);
}
int sfc_finish(sfc_writer *w) {
  if (!w)
    return SFC_INVALID;
  if (w->failed || w->pending_count || w->current != w->count) {
    sfc_cancel(w);
    return SFC_INVALID;
  }
  off_t end = ftello(w->f);
  int rc = 0;
  if (end < 0)
    rc = SFC_IO;
  else {
    p64(w->header + 40, (uint64_t)end);
    p32(w->header + 48, header_crc(w->header));
    if (fseeko(w->f, 0, SEEK_SET) || fwrite(w->header, 1, 64, w->f) != 64 ||
        fflush(w->f))
      rc = SFC_IO;
  }
  if (rc) {
    sfc_cancel(w);
    return rc;
  }
  if (fclose(w->f))
    rc = SFC_IO;
  w->f = NULL;
  if (rc)
    unlink(w->path);
  sfc_cancel(w);
  return rc;
}

#include "cache.h"
