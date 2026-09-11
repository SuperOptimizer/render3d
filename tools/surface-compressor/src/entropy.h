/* Entropy primitives adapted from volume-compressor.
MIT License

Copyright (c) 2026 SuperOpt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <stdbool.h>
#define SE_NTOK 32u
#define SE_TOK_EOB 31u
#define SE_NMODELS 10u
#define SE_DC_CTX 9u
#define SE_PROB_BITS 10u
#define SE_PROB_SCALE (1u << SE_PROB_BITS)
#define SE_TABLES_MAX_BYTES (SE_NMODELS * (4u + SE_NTOK * 2u))
static inline void se_wr_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
static inline uint32_t se_rd_u16(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}
static inline uint32_t se_rd_u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
typedef struct se_cur {
  const uint8_t *p, *end;
} se_cur;
static inline bool se_cur_has(const se_cur *c, size_t n) {
  return (size_t)(c->end - c->p) >= n;
}
static inline size_t se_leb_put(uint8_t *p, uint32_t v) {
  size_t n = 0;
  while (v >= 0x80u) {
    p[n++] = (uint8_t)(v | 0x80u);
    v >>= 7;
  }
  p[n++] = (uint8_t)v;
  return n;
}
/* canonical: no trailing zero continuation, <= 5 bytes, fits 32 bits */
static inline bool se_leb_get(se_cur *c, uint32_t *out) {
  uint32_t v = 0, sh = 0;
  for (int i = 0; i < 5; i++) {
    if (!se_cur_has(c, 1))
      return false;
    uint32_t b = *c->p++;
    if (i == 4 && (b & 0xf0u))
      return false;
    v |= (b & 0x7fu) << sh;
    if (!(b & 0x80u)) {
      if (i > 0 && (b & 0x7fu) == 0)
        return false;
      *out = v;
      return true;
    }
    sh += 7;
  }
  return false;
}
static inline uint32_t se_zigzag(int32_t v) {
  return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31);
}
static inline int32_t se_unzigzag(uint32_t u) {
  return (int32_t)(u >> 1) ^ -(int32_t)(u & 1u);
}

/* ---- LSB-first bypass bit writer / reader (64-bit accumulators) ---- */
typedef struct se_bitw {
  uint8_t *buf;
  size_t cap, pos;
  uint64_t acc;
  uint32_t nbits;
} se_bitw;
static inline void se_bw_init(se_bitw *w, uint8_t *buf, size_t cap) {
  w->buf = buf;
  w->cap = cap;
  w->pos = 0;
  w->acc = 0;
  w->nbits = 0;
}
__attribute__((always_inline)) static inline bool
se_bw_put(se_bitw *w, uint32_t v, uint32_t n) { /* n <= 32, v < 2^n */
  w->acc |= (uint64_t)v << w->nbits;
  w->nbits += n;
  while (w->nbits >= 8) {
    if (w->pos >= w->cap)
      return false;
    w->buf[w->pos++] = (uint8_t)w->acc;
    w->acc >>= 8;
    w->nbits -= 8;
  }
  return true;
}
static inline bool se_bw_flush(se_bitw *w, size_t *n_out) {
  if (w->nbits > 0) {
    if (w->pos >= w->cap)
      return false;
    w->buf[w->pos++] = (uint8_t)w->acc;
    w->acc = 0;
    w->nbits = 0;
  }
  *n_out = w->pos;
  return true;
}
typedef struct se_bitr {
  const uint8_t *buf;
  size_t n, pos;
  uint64_t acc;
  uint32_t nbits;
} se_bitr;
static inline void se_br_init(se_bitr *r, const uint8_t *buf, size_t n) {
  r->buf = buf;
  r->n = n;
  r->pos = 0;
  r->acc = 0;
  r->nbits = 0;
}
__attribute__((always_inline)) static inline bool
se_br_get(se_bitr *r, uint32_t n, uint32_t *out) { /* n <= 32 */
  if (r->nbits < n) {
    if (r->pos + 8 <=
        r->n) { /* bulk refill: whole bytes that fit in the accumulator */
      uint64_t w;
      w = (uint64_t)se_rd_u32(r->buf + r->pos) |
          (uint64_t)se_rd_u32(r->buf + r->pos + 4) << 32;
      uint32_t take = (63u - r->nbits) >> 3; /* 4..7 bytes when nbits < 32 */
      r->acc |= (w & ((1ull << (take * 8u)) - 1ull)) << r->nbits;
      r->nbits += take * 8u;
      r->pos += take;
    } else {
      while (r->nbits < n) {
        if (r->pos >= r->n)
          return false;
        r->acc |= (uint64_t)r->buf[r->pos++] << r->nbits;
        r->nbits += 8;
      }
    }
  }
  *out =
      (uint32_t)(r->acc & ((n == 32u) ? 0xffffffffull : ((1ull << n) - 1ull)));
  r->acc >>= n;
  r->nbits -= n;
  return true;
}
/* exact consumption: all bytes read, remaining padding bits zero */
static inline bool se_br_finished(const se_bitr *r) {
  return r->pos == r->n && r->acc == 0 && r->nbits < 8;
}

/* ---- HybridUint: tokens 0..3 literal; else tok = 2 + msb(u), low msb bits
 * bypassed ---- */
__attribute__((always_inline)) static inline bool
se_hyb_emit(se_bitw *w, uint32_t u, uint32_t *tok) {
  if (u < 4u) {
    *tok = u;
    return true;
  }
  uint32_t k = 31u - (uint32_t)__builtin_clz(u);
  *tok = 2u + k;
  return se_bw_put(w, u & ((1u << k) - 1u), k);
}
__attribute__((always_inline)) static inline bool
se_hyb_read(se_bitr *r, uint32_t tok, uint32_t *u) { /* tok pre-validated */
  if (tok < 4u) {
    *u = tok;
    return true;
  }
  uint32_t k = tok - 2u, extra;
  if (!se_br_get(r, k, &extra))
    return false;
  *u = (1u << k) | extra;
  return true;
}

/* ---- context models: 0..2 runs+EOB by band, 3..8 levels by band x (run==0), 9
 * DC ---- */
__attribute__((always_inline)) static inline uint32_t se_band_of(uint32_t pos) {
  return pos < 128u ? 0u : (pos < 1024u ? 1u : 2u);
}
__attribute__((always_inline)) static inline uint32_t se_run_ctx(uint32_t pos) {
  return se_band_of(pos);
}
__attribute__((always_inline)) static inline uint32_t
se_level_ctx(uint32_t pos, uint32_t run) {
  return 3u + se_band_of(pos) * 2u + (run == 0u ? 1u : 0u);
}

/* ---- entropy coder: tANS (FSE), 10-bit tables, 2 interleaved lanes ---- */
/* tANS (FSE-style) tables. Decoder: state in [0,SCALE); entry gives symbol,
 * bit count and the new-state base. */
typedef struct se_dentry {
  uint16_t ns;
  uint8_t sym, nb;
} se_dentry;
typedef struct se_model {
  uint16_t freq[SE_NTOK];
  uint16_t cum[SE_NTOK + 1];
  se_dentry dt[SE_PROB_SCALE];
} se_model;

static inline uint32_t se_highbit(uint32_t v) {
  return 31u - (uint32_t)__builtin_clz(v);
}
static void se_model_fill(se_model *m) {
  m->cum[0] = 0;
  for (uint32_t s = 0; s < SE_NTOK; s++)
    m->cum[s + 1] = (uint16_t)(m->cum[s] + m->freq[s]);
  /* spread symbols over the table (FSE step), then assign next states */
  const uint32_t size = SE_PROB_SCALE, mask = size - 1u,
                 step = (size >> 1) + (size >> 3) + 3u;
  uint32_t pos = 0;
  for (uint32_t s = 0; s < SE_NTOK; s++)
    for (uint32_t i = 0; i < m->freq[s]; i++) {
      m->dt[pos].sym = (uint8_t)s;
      pos = (pos + step) & mask;
    }
  uint16_t next[SE_NTOK];
  for (uint32_t s = 0; s < SE_NTOK; s++)
    next[s] = m->freq[s];
  for (uint32_t u = 0; u < size; u++) {
    uint32_t s = m->dt[u].sym;
    uint32_t nx = next[s]++;
    uint32_t nb = SE_PROB_BITS - se_highbit(nx);
    m->dt[u].nb = (uint8_t)nb;
    m->dt[u].ns = (uint16_t)((nx << nb) - size);
  }
}
/* encoder: normalise counts to sum exactly SE_PROB_SCALE (nonzero counts keep
 * freq >= 1) */
static bool se_model_build(se_model *m, const uint32_t counts[SE_NTOK]) {
  uint64_t total = 0;
  for (uint32_t s = 0; s < SE_NTOK; s++)
    total += counts[s];
  if (total == 0)
    return false;
  uint32_t assigned = 0;
  for (uint32_t s = 0; s < SE_NTOK; s++) {
    if (counts[s] == 0) {
      m->freq[s] = 0;
      continue;
    }
    uint64_t f = (uint64_t)counts[s] * SE_PROB_SCALE / total;
    if (f == 0)
      f = 1;
    m->freq[s] = (uint16_t)f;
    assigned += (uint32_t)f;
  }
  if (assigned != SE_PROB_SCALE) {
    uint32_t big = 0;
    for (uint32_t s = 1; s < SE_NTOK; s++)
      if (m->freq[s] > m->freq[big])
        big = s;
    int32_t nf =
        (int32_t)m->freq[big] + ((int32_t)SE_PROB_SCALE - (int32_t)assigned);
    if (nf < 1)
      return false;
    m->freq[big] = (uint16_t)nf;
  }
  se_model_fill(m);
  return true;
}
/* decoder: adopt transmitted frequencies verbatim; never renormalises */
static bool se_model_from_freqs(se_model *m, const uint32_t freqs[SE_NTOK]) {
  uint32_t sum = 0;
  for (uint32_t s = 0; s < SE_NTOK; s++) {
    if (freqs[s] > SE_PROB_SCALE)
      return false;
    sum += freqs[s];
    m->freq[s] = (uint16_t)freqs[s];
  }
  if (sum != SE_PROB_SCALE)
    return false;
  se_model_fill(m);
  return true;
}
/* compact tables: per model u32 LE presence bitmap + LEB128 freq per set bit,
 * ascending */
static size_t se_tables_write(const se_model m[SE_NMODELS], uint8_t *out) {
  size_t n = 0;
  for (uint32_t i = 0; i < SE_NMODELS; i++) {
    uint32_t bm = 0;
    for (uint32_t s = 0; s < SE_NTOK; s++)
      if (m[i].freq[s])
        bm |= 1u << s;
    se_wr_u32(out + n, bm);
    n += 4;
    for (uint32_t s = 0; s < SE_NTOK; s++)
      if (m[i].freq[s])
        n += se_leb_put(out + n, m[i].freq[s]);
  }
  return n;
}
static bool se_tables_read(se_cur *c, se_model m[SE_NMODELS]) {
  for (uint32_t i = 0; i < SE_NMODELS; i++) {
    if (!se_cur_has(c, 4))
      return false;
    uint32_t bm = se_rd_u32(c->p);
    c->p += 4;
    if (bm == 0)
      return false;
    uint32_t f[SE_NTOK] = {0}, sum = 0;
    for (uint32_t s = 0; s < SE_NTOK; s++) {
      if (!(bm >> s & 1u))
        continue;
      uint32_t v;
      if (!se_leb_get(c, &v) || v == 0 || v > SE_PROB_SCALE)
        return false;
      sum += v;
      if (sum > SE_PROB_SCALE)
        return false;
      f[s] = v;
    }
    if (!se_model_from_freqs(&m[i], f))
      return false;
  }
  return true;
}
/* tANS encoder tables (FSE_buildCTable): per model a state table and per
 * symbol (deltaNbBits, deltaFindState). Encoding runs backwards over the
 * symbols; the emitted bit fields are buffered and written in decode order
 * so the decoder reads a plain forward LSB-first bit stream. */
typedef struct se_esym {
  uint32_t delta_nb;
  int32_t delta_fs;
} se_esym;
typedef struct se_etab {
  uint16_t state[SE_PROB_SCALE];
  se_esym sym[SE_NTOK];
} se_etab;
static void se_etabs_init(const se_model m[SE_NMODELS],
                          se_etab et[SE_NMODELS]) {
  const uint32_t size = SE_PROB_SCALE, mask = size - 1u,
                 step = (size >> 1) + (size >> 3) + 3u;
  for (uint32_t c = 0; c < SE_NMODELS; c++) {
    uint8_t tsym[SE_PROB_SCALE];
    uint32_t pos = 0;
    for (uint32_t s = 0; s < SE_NTOK; s++)
      for (uint32_t i = 0; i < m[c].freq[s]; i++) {
        tsym[pos] = (uint8_t)s;
        pos = (pos + step) & mask;
      }
    uint32_t cum[SE_NTOK + 1];
    for (uint32_t s = 0; s <= SE_NTOK; s++)
      cum[s] = m[c].cum[s];
    for (uint32_t u = 0; u < size; u++)
      et[c].state[cum[tsym[u]]++] = (uint16_t)(size + u);
    uint32_t total = 0;
    for (uint32_t s = 0; s < SE_NTOK; s++) {
      uint32_t f = m[c].freq[s];
      if (f == 0) {
        et[c].sym[s].delta_nb = 0xffffffffu; /* marker: unusable */
        et[c].sym[s].delta_fs = 0;
        continue;
      }
      uint32_t maxbits =
          f == 1 ? SE_PROB_BITS : SE_PROB_BITS - se_highbit(f - 1u);
      et[c].sym[s].delta_nb = (maxbits << 16) - (f << maxbits);
      et[c].sym[s].delta_fs = (int32_t)total - (int32_t)(f == 1 ? 1u : f);
      total += f;
    }
  }
}
/* fields: value | nbits<<27, built in reverse symbol order */
static size_t se_tans_encode2(const se_etab et[SE_NMODELS],
                              const uint16_t *syms, size_t n, uint32_t *fields,
                              uint8_t *out, size_t out_cap) {
  uint32_t x[2] = {SE_PROB_SCALE, SE_PROB_SCALE};
  size_t nf = 0;
  for (size_t i = n; i-- > 0;) {
    const se_esym *e = &et[syms[i] >> 8].sym[syms[i] & 0xffu];
    if (e->delta_nb == 0xffffffffu)
      return 0;
    uint32_t l = (uint32_t)i & 1u;
    uint32_t nb = (x[l] + e->delta_nb) >> 16;
    fields[nf++] = (x[l] & ((1u << nb) - 1u)) | nb << 27;
    x[l] = et[syms[i] >> 8].state[(x[l] >> nb) + (uint32_t)e->delta_fs];
  }
  /* initial decoder states (lane 1 pushed first so lane 0 is read first) */
  fields[nf++] = (x[1] - SE_PROB_SCALE) | SE_PROB_BITS << 27;
  fields[nf++] = (x[0] - SE_PROB_SCALE) | SE_PROB_BITS << 27;
  se_bitw w;
  se_bw_init(&w, out, out_cap);
  for (size_t i = nf; i-- > 0;)
    if (!se_bw_put(&w, fields[i] & 0x7ffffffu, fields[i] >> 27))
      return 0;
  size_t used;
  if (!se_bw_flush(&w, &used))
    return 0;
  return used;
}
typedef struct se_rdec {
  uint32_t x[2];
  uint32_t k;
  se_bitr br;
} se_rdec;
static inline bool se_rdec_init(se_rdec *d, const uint8_t *in, size_t in_n) {
  se_br_init(&d->br, in, in_n);
  if (!se_br_get(&d->br, SE_PROB_BITS, &d->x[0]) ||
      !se_br_get(&d->br, SE_PROB_BITS, &d->x[1]))
    return false;
  d->k = 0;
  return true;
}
__attribute__((always_inline)) static inline int
se_rdec_get(se_rdec *d, const se_model *m) {
  uint32_t k = d->k;
  const se_dentry *e = &m->dt[d->x[k]];
  uint32_t bits;
  if (!se_br_get(&d->br, e->nb, &bits))
    return -1;
  d->x[k] = e->ns + bits;
  d->k = k ^ 1u;
  return (int)e->sym;
}
/* exact consumption: all bytes read, zero padding, both lanes back at state 0
 */
static inline bool se_rdec_finished(const se_rdec *d) {
  return se_br_finished(&d->br) && d->x[0] == 0 && d->x[1] == 0;
}
