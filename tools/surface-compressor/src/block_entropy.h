/* Internal coefficient coding. Floating-point quantization and reconstruction
 * are unchanged; only the integer transport representation differs. */
#define SFC_GROUP 16u
/* One DC + two symbols for each of 4095 AC values + one EOB. Sign bits
 * are bypass bits, not tokens. Dense blocks exactly fit this bound. */
#define SFC_TOKENS 8192u
_Static_assert(SFC_TOKENS >= 1 + 2 * (SFC_SAMPLES - 1) + 1, "token bound");
#define SFC_BYPASS 20000u
#define SFC_ANS 11000u

typedef struct {
  uint16_t symbols[SFC_TOKENS];
  uint8_t bypass[SFC_BYPASS];
  size_t ns, nb;
} sfc_tokens;

static int entropy_emit(sfc_tokens *t, se_bitw *bw,
                        uint32_t counts[SE_NMODELS][SE_NTOK], unsigned ctx,
                        uint32_t value) {
  uint32_t tok;
  if (t->ns >= SFC_TOKENS || !se_hyb_emit(bw, value, &tok) || tok >= SE_NTOK)
    return SFC_INVALID;
  t->symbols[t->ns++] = (uint16_t)((ctx << 8) | tok);
  counts[ctx][tok]++;
  return 0;
}
static int entropy_tokenize(const uint8_t *b, size_t n, sfc_tokens *t,
                            uint32_t counts[SE_NMODELS][SE_NTOK]) {
  if (!b || !t || !counts || n < 32 || b[5] != 1)
    return SFC_INVALID;
  size_t at, cn = u32(b + 24);
  if (block_prefix(b,n,&at,NULL) || cn > n - at)
    return SFC_INVALID;
  size_t end = at + cn;
  int32_t coeff[4096] = {0};
  unsigned i = 0;
  while (i < 4096) {
    uint64_t run, v;
    if (varget(b, end, &at, &run) || run > 4096 - i)
      return SFC_INVALID;
    i += (unsigned)run;
    if (i == 4096)
      break;
    if (varget(b, end, &at, &v) || v > 8388608)
      return SFC_INVALID;
    coeff[i++] = se_unzigzag((uint32_t)v);
  }
  if (at != end)
    return SFC_INVALID;
  t->ns = 0;
  se_bitw bw;
  se_bw_init(&bw, t->bypass, sizeof t->bypass);
  if (entropy_emit(t, &bw, counts, SE_DC_CTX, se_zigzag(coeff[0])))
    return SFC_INVALID;
  unsigned prev = 0;
  for (unsigned k = 1; k < 4096; k++) {
    int32_t v = coeff[sfc_scan[k]];
    if (!v)
      continue;
    unsigned run = k - prev - 1;
    if (entropy_emit(t, &bw, counts, se_run_ctx(prev + 1), run) ||
        entropy_emit(t, &bw, counts, se_level_ctx(k, run),
                     (uint32_t)(v < 0 ? -v : v) - 1) ||
        !se_bw_put(&bw, v < 0, 1))
      return SFC_INVALID;
    prev = k;
  }
  unsigned ctx = se_run_ctx(prev + 1);
  if (t->ns >= SFC_TOKENS)
    return SFC_INVALID;
  t->symbols[t->ns++] = (uint16_t)((ctx << 8) | SE_TOK_EOB);
  counts[ctx][SE_TOK_EOB]++;
  return se_bw_flush(&bw, &t->nb) ? 0 : SFC_INVALID;
}

static int entropy_decode(const uint8_t *p, size_t n, const se_model *models,
                          float step, float coef[4096]) {
  if (!models || n < 24)
    return SFC_INVALID;
  uint32_t an = u32(p + 16), bn = u32(p + 20);
  if (an < 3 || an > SFC_ANS || bn > SFC_BYPASS || an > n - 24 ||
      bn != n - 24 - an)
    return SFC_INVALID;
  se_rdec d;
  se_bitr br;
  if (!se_rdec_init(&d, p + 24, an))
    return SFC_INVALID;
  se_br_init(&br, p + 24 + an, bn);
  int tok = se_rdec_get(&d, models + SE_DC_CTX);
  uint32_t v;
  if (tok < 0 || tok > 25 || !se_hyb_read(&br, (uint32_t)tok, &v) ||
      v > 8388608)
    return SFC_INVALID;
  coef[0] = (float)se_unzigzag(v) * step;
  unsigned prev = 0;
  for (;;) {
    tok = se_rdec_get(&d, models + se_run_ctx(prev + 1));
    if (tok == SE_TOK_EOB)
      break;
    uint32_t run, sign;
    if (tok < 0 || tok > 13 || !se_hyb_read(&br, (uint32_t)tok, &run) ||
        run >= 4095 - prev)
      return SFC_INVALID;
    unsigned k = prev + 1 + run;
    tok = se_rdec_get(&d, models + se_level_ctx(k, run));
    if (tok < 0 || tok > 23 || !se_hyb_read(&br, (uint32_t)tok, &v) ||
        v >= 4194304 || !se_br_get(&br, 1, &sign))
      return SFC_INVALID;
    int32_t signed_v = sign ? -(int32_t)(v + 1) : (int32_t)(v + 1);
    coef[sfc_scan[k]] = (float)signed_v * step;
    prev = k;
  }
  return se_rdec_finished(&d) && se_br_finished(&br) ? 0 : SFC_INVALID;
}
