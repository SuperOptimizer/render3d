/* Private implementation; included by surfcomp.c. */
#define SFC_CACHE_BUCKETS 256u
typedef struct sfc_cache_tile {
  struct sfc_cache_tile *prev, *next, *hash_next;
  uint64_t bx, by;
  uint32_t channel;
  unsigned xyz;
  size_t sample_bytes, bytes;
  uint8_t data[]; /* packed samples followed by 4096 validity bytes */
} sfc_cache_tile;
struct sfc_cache {
  sfc_reader *reader;
  atomic_bool busy;
  sfc_cache_stats stats;
  sfc_cache_tile *head, *tail, *buckets[SFC_CACHE_BUCKETS];
};
static void cache_lock(sfc_cache *c) {
  while (atomic_exchange_explicit(&c->busy, 1, memory_order_acquire)) {
    while (atomic_load_explicit(&c->busy, memory_order_relaxed)) {
    }
  }
}
static void cache_unlock(sfc_cache *c) {
  atomic_store_explicit(&c->busy, 0, memory_order_release);
}
static unsigned cache_hash(uint32_t ch, uint64_t bx, uint64_t by,
                           unsigned xyz) {
  uint64_t h = bx * UINT64_C(0x9e3779b97f4a7c15) ^
               by * UINT64_C(0xbf58476d1ce4e5b9) ^ ((uint64_t)ch << 1) ^ xyz;
  h ^= h >> 32;
  h ^= h >> 16;
  return (unsigned)h & (SFC_CACHE_BUCKETS - 1);
}
static void cache_unlink(sfc_cache *c, sfc_cache_tile *t) {
  if (t->prev)
    t->prev->next = t->next;
  else
    c->head = t->next;
  if (t->next)
    t->next->prev = t->prev;
  else
    c->tail = t->prev;
}
static void cache_front(sfc_cache *c, sfc_cache_tile *t) {
  t->prev = NULL;
  t->next = c->head;
  if (c->head)
    c->head->prev = t;
  else
    c->tail = t;
  c->head = t;
}
static void cache_remove_tail(sfc_cache *c) {
  sfc_cache_tile *t = c->tail;
  unsigned h = cache_hash(t->channel, t->bx, t->by, t->xyz);
  sfc_cache_tile **p = &c->buckets[h];
  while (*p != t)
    p = &(*p)->hash_next;
  *p = t->hash_next;
  cache_unlink(c, t);
  c->stats.resident_bytes -= t->bytes;
  c->stats.resident_tiles--;
  free(t);
}
int sfc_cache_create(sfc_reader *r, size_t budget, sfc_cache **out) {
  if (!out)
    return SFC_INVALID;
  *out = NULL;
  if (!r)
    return SFC_INVALID;
  sfc_cache *c = calloc(1, sizeof(*c));
  if (!c)
    return SFC_NOMEM;
  c->reader = r;
  c->stats.byte_budget = budget;
  atomic_init(&c->busy, 0);
  *out = c;
  return SFC_OK;
}
void sfc_cache_clear(sfc_cache *c) {
  if (!c)
    return;
  cache_lock(c);
  while (c->tail)
    cache_remove_tail(c);
  cache_unlock(c);
}
void sfc_cache_destroy(sfc_cache *c) {
  if (!c)
    return;
  sfc_cache_clear(c);
  free(c);
}
int sfc_cache_get_stats(sfc_cache *c, sfc_cache_stats *out) {
  if (!c || !out)
    return SFC_INVALID;
  cache_lock(c);
  *out = c->stats;
  cache_unlock(c);
  return SFC_OK;
}
/* Called with cache locked. Failed decodes are never admitted. */
static int cache_tile(sfc_cache *c, uint32_t ch, unsigned xyz, uint64_t bx,
                      uint64_t by, sfc_cache_tile **out) {
  unsigned h = cache_hash(ch, bx, by, xyz);
  for (sfc_cache_tile *t = c->buckets[h]; t; t = t->hash_next) {
    if (t->channel == ch && t->xyz == xyz && t->bx == bx && t->by == by) {
      c->stats.hits++;
      cache_unlink(c, t);
      cache_front(c, t);
      *out = t;
      return SFC_OK;
    }
  }
  c->stats.misses++;
  size_t z = xyz ? 12 : sfc_sample_size(sfc_channel_info(c->reader, ch)->dtype);
  size_t bytes = sizeof(sfc_cache_tile) + SFC_SAMPLES * (z + 1);
  if (bytes > c->stats.byte_budget)
    return SFC_LIMIT;
  while (c->stats.resident_bytes > c->stats.byte_budget - bytes) {
    cache_remove_tail(c);
    c->stats.evictions++;
  }
  sfc_cache_tile *t = malloc(bytes);
  if (!t)
    return SFC_NOMEM;
  int rc = xyz ? sfc_read_xyz(c->reader, ch, bx, by, t->data, z, 64 * z,
                              t->data + SFC_SAMPLES * z)
               : sfc_read_block(c->reader, ch, bx, by, t->data, z, 64 * z,
                                t->data + SFC_SAMPLES * z);
  if (rc) {
    free(t);
    return rc;
  }
  t->channel = ch;
  t->xyz = xyz;
  t->bx = bx;
  t->by = by;
  t->sample_bytes = z;
  t->bytes = bytes;
  t->hash_next = c->buckets[h];
  c->buckets[h] = t;
  cache_front(c, t);
  c->stats.resident_bytes += bytes;
  c->stats.resident_tiles++;
  *out = t;
  return SFC_OK;
}
static int cache_region(sfc_cache *c, uint32_t ch, unsigned xyz, uint64_t x,
                        uint64_t y, uint32_t w, uint32_t h, void *dst,
                        size_t ps, size_t rs, uint8_t *valid) {
  const sfc_channel *info = c ? sfc_channel_info(c->reader, ch) : NULL;
  size_t z = info ? (xyz ? 12 : sfc_sample_size(info->dtype)) : 0;
  if (!info || !dst || !strides(ps, rs, z, w, h) || x > info->width ||
      y > info->height || w > info->width - x || h > info->height - y ||
      (uint64_t)w * h > SIZE_MAX)
    return SFC_INVALID;
  unsigned offset = 0;
  if (xyz) {
    if (sfc_channel_count(c->reader) - ch < 3)
      return SFC_INVALID;
    for (unsigned k = 0; k < 3; k++) {
      const sfc_channel *p = sfc_channel_info(c->reader, ch + k);
      if (p->dtype != SFC_F32 || !(p->flags & SFC_COORDINATE) ||
          p->width != info->width || p->height != info->height ||
          ((p->flags & SFC_XYZ) && p->component != k))
        return SFC_INVALID;
    }
  } else if (info->flags & SFC_XYZ) {
    offset = info->component * 4;
    ch -= info->component;
    xyz = 1;
  }
  for (uint64_t yy = y; yy < y + h;) {
    size_t nh = 64 - yy % 64;
    if (nh > y + h - yy)
      nh = (size_t)(y + h - yy);
    for (uint64_t xx = x; xx < x + w;) {
      size_t nw = 64 - xx % 64;
      if (nw > x + w - xx)
        nw = (size_t)(x + w - xx);
      cache_lock(c);
      sfc_cache_tile *t;
      int rc = cache_tile(c, ch, xyz, xx / 64, yy / 64, &t);
      if (!rc) {
        for (size_t j = 0; j < nh; j++) {
          size_t src = (yy % 64 + j) * 64 + xx % 64;
          uint8_t *to = (uint8_t *)dst + (size_t)(yy - y + j) * rs +
                        (size_t)(xx - x) * ps;
          if (ps == z && t->sample_bytes == z)
            memcpy(to, t->data + src * z, nw * z);
          else
            for (size_t i = 0; i < nw; i++)
              memcpy(to + i * ps,
                     t->data + (src + i) * t->sample_bytes + offset, z);
          if (valid)
            memcpy(valid + (size_t)(yy - y + j) * w + (size_t)(xx - x),
                   t->data + SFC_SAMPLES * t->sample_bytes + src, nw);
        }
      }
      cache_unlock(c);
      if (rc)
        return rc;
      xx += nw;
    }
    yy += nh;
  }
  return SFC_OK;
}
int sfc_cache_read_region(sfc_cache *c, uint32_t ch, uint64_t x, uint64_t y,
                          uint32_t w, uint32_t h, void *dst, size_t ps,
                          size_t rs, uint8_t *valid) {
  return cache_region(c, ch, 0, x, y, w, h, dst, ps, rs, valid);
}
int sfc_cache_read_xyz_region(sfc_cache *c, uint32_t ch, uint64_t x, uint64_t y,
                              uint32_t w, uint32_t h, void *dst, size_t ps,
                              size_t rs, uint8_t *valid) {
  return cache_region(c, ch, 1, x, y, w, h, dst, ps, rs, valid);
}
