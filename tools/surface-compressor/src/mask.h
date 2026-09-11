/* Mixed masks may use alternating positive varint runs. The two-byte length
 * includes one initial-bit byte. Keep the bitmap unless runs save bytes. */
static int block_prefix(const uint8_t *b, size_t n, size_t *at,
                        uint8_t *valid) {
  if (!b || n < 32 || ((b[6] & 8) && !(b[6] & 1)))
    return SFC_INVALID;
  *at = 32;
  if (!(b[6] & 1)) {
    if (valid)
      memset(valid, (b[6] & 2) ? 0 : 1, 4096);
    return 0;
  }
  if (!(b[6] & 8)) {
    if (n < 544)
      return SFC_INVALID;
    if (valid)
      for (unsigned i = 0; i < 4096; i++)
        valid[i] = (b[32 + i / 8] >> (i % 8)) & 1;
    *at = 544;
    return 0;
  }
  if (n < 35)
    return SFC_INVALID;
  size_t size = (size_t)b[32] | (size_t)b[33] << 8;
  if (size < 3 || size > 509 || size > n - 34 || b[34] > 1)
    return SFC_INVALID;
  size_t p = 35, end = 34 + size;
  unsigned pos = 0, bit = b[34];
  while (pos < 4096) {
    uint64_t run;
    if (varget(b, end, &p, &run) || !run || run > 4096 - pos)
      return SFC_INVALID;
    if (valid)
      memset(valid + pos, (int)bit, (size_t)run);
    pos += (unsigned)run;
    bit ^= 1;
  }
  if (p != end)
    return SFC_INVALID;
  *at = end;
  return 0;
}
static size_t pack_mask(uint8_t *b, size_t n) {
  if (n < 544 || !(b[6] & 1) || (b[6] & 8))
    return n;
  uint8_t runs[515];
  size_t size = 1;
  unsigned bit = b[32] & 1, pos = 0;
  runs[0] = (uint8_t)bit;
  while (pos < 4096) {
    unsigned start = pos;
    do {
      pos++;
    } while (pos < 4096 && ((b[32 + pos / 8] >> (pos % 8)) & 1) == bit);
    size += varput(runs + size, pos - start);
    bit ^= 1;
    if (size + 2 >= 512)
      return n;
  }
  memmove(b + 34 + size, b + 544, n - 544);
  b[6] |= 8;
  b[32] = (uint8_t)size;
  b[33] = (uint8_t)(size >> 8);
  memcpy(b + 34, runs, size);
  return n - 512 + 2 + size;
}
