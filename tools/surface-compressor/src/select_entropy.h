/* Shared-model selection measures actual encoded bytes, including mask,
 * escapes, per-block stream descriptors, and the group's tables. */
typedef struct {
  se_model models[SE_NMODELS];
  se_etab etabs[SE_NMODELS];
  uint32_t counts[SE_NMODELS][SE_NTOK], dummy[SE_NMODELS][SE_NTOK];
  sfc_tokens tokens;
  uint32_t fields[SFC_TOKENS + 2];
  uint8_t ans[SFC_ANS], packet[SFC_BLOCK_BOUND];
  uint8_t trial[SFC_GROUP][SFC_BLOCK_BOUND], best[SFC_GROUP][SFC_BLOCK_BOUND];
  size_t sizes[SFC_GROUP], table_size, total;
  uint8_t tables[SE_TABLES_MAX_BYTES];
} sfc_entropy_work;
static int entropy_packet(sfc_entropy_work *w, const sfc_choice *choice,
                          const uint8_t **packet, size_t *size) {
  *packet = choice->bytes;
  *size = choice->size;
  if (choice->bytes[5] != 1)
    return 0;
  memset(w->dummy, 0, sizeof w->dummy);
  int rc = entropy_tokenize(choice->bytes, choice->size, &w->tokens, w->dummy);
  if (rc)
    return rc;
  size_t an = se_tans_encode2(w->etabs, w->tokens.symbols, w->tokens.ns,
                              w->fields, w->ans, sizeof w->ans);
  if (!an)
    return 0; /* This model may not contain a competing candidate's symbols. */
  size_t at;
  if (block_prefix(choice->bytes, choice->size, &at, NULL))
    return SFC_INVALID;
  size_t cn = u32(choice->bytes + 24),
         nn = choice->size - cn + 24 + an + w->tokens.nb;
  if (nn >= choice->size || nn > SFC_BLOCK_BOUND)
    return 0;
  uint8_t *b = w->packet;
  memcpy(b, choice->bytes, at);
  b[5] = 3;
  p32(b + 24, (uint32_t)(24 + an + w->tokens.nb));
  memset(b + at, 0, 16);
  p32(b + at + 16, (uint32_t)an);
  p32(b + at + 20, (uint32_t)w->tokens.nb);
  memcpy(b + at + 24, w->ans, an);
  memcpy(b + at + 24 + an, w->tokens.bypass, w->tokens.nb);
  memcpy(b + at + 24 + an + w->tokens.nb, choice->bytes + at + cn,
         choice->size - at - cn);
  *packet = b;
  *size = nn;
  return 0;
}
static int select_entropy(sfc_candidates *const choices[], unsigned count,
                          sfc_entropy_work *w) {
  if (!count || count > SFC_GROUP)
    return SFC_INVALID;
  unsigned selected[SFC_GROUP];
  int alternatives = 0;
  w->total = 0;
  w->table_size = 0;
  for (unsigned i = 0; i < count; i++) {
    if (!choices[i]->count || choices[i]->preferred >= choices[i]->count)
      return SFC_INVALID;
    selected[i] = choices[i]->preferred;
    alternatives |= choices[i]->count > 1;
    sfc_choice *c = choices[i]->values + selected[i];
    w->sizes[i] = c->size;
    w->total += c->size;
    memcpy(w->best[i], c->bytes, c->size);
  }
  for (unsigned pass = 0; pass < (alternatives ? 3u : 1u); pass++) {
    memset(w->counts, 0, sizeof w->counts);
    for (unsigned i = 0; i < count; i++) {
      unsigned n = pass == 1 ? choices[i]->count : 1;
      for (unsigned j = 0; j < n; j++) {
        const sfc_choice *c =
            choices[i]->values + (pass == 1 ? j : selected[i]);
        if (c->bytes[5] == 1) {
          int rc = entropy_tokenize(c->bytes, c->size, &w->tokens, w->counts);
          if (rc)
            return rc;
        }
      }
    }
    for (unsigned m = 0; m < SE_NMODELS; m++) {
      uint32_t total = 0;
      for (unsigned t = 0; t < SE_NTOK; t++)
        total += w->counts[m][t];
      if (!total)
        w->counts[m][0] = 1;
      if (!se_model_build(w->models + m, w->counts[m]))
        return SFC_INVALID;
    }
    uint8_t tables[SE_TABLES_MAX_BYTES];
    size_t table_size = se_tables_write(w->models, tables);
    se_etabs_init(w->models, w->etabs);
    size_t total = 0, sizes[SFC_GROUP];
    int uses_tables = 0;
    for (unsigned i = 0; i < count; i++) {
      size_t minimum = SIZE_MAX, best_size = SIZE_MAX;
      double error = DBL_MAX;
      unsigned winner = selected[i];
      unsigned n = pass ? choices[i]->count : 1;
      for (unsigned j = 0; j < n; j++) {
        unsigned k = pass ? j : selected[i];
        const sfc_choice *c = choices[i]->values + k;
        const uint8_t *packet;
        size_t size;
        int rc = entropy_packet(w, c, &packet, &size);
        if (rc)
          return rc;
        if (size < minimum)
          minimum = size;
        size_t limit = minimum + minimum / 50;
        if (best_size > limit ||
            (size <= limit &&
             (c->square < error || (c->square == error && size < best_size)))) {
          memcpy(w->trial[i], packet, size);
          best_size = size;
          error = c->square;
          winner = k;
        }
      }
      selected[i] = winner;
      sizes[i] = best_size;
      total += best_size;
      uses_tables |= w->trial[i][5] == 3;
    }
    if (!uses_tables)
      table_size = 0;
    total += table_size;
    if (total < w->total) {
      w->total = total;
      w->table_size = table_size;
      memcpy(w->tables, tables, table_size);
      for (unsigned i = 0; i < count; i++) {
        w->sizes[i] = sizes[i];
        memcpy(w->best[i], w->trial[i], sizes[i]);
      }
    }
  }
  return 0;
}
