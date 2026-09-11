#include "core/surface.h"
#include "surfcomp.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TILE_FLOATS (4096u * 3u)
#define TILE_BYTES (TILE_FLOATS * sizeof(float))
#define ROI_LIMIT (256u << 20)
struct cached_tile {
  uint64_t key, age;
  float *data;
};
struct r3d_surface_reader {
  sfc_reader *file;
  pthread_mutex_t mu;
  r3d_surface_info info;
  int channels[3];
  struct cached_tile *slots;
  uint32_t *hash;
  uint32_t cap, hash_cap, used;
  uint64_t clock;
  _Atomic uint64_t hits, misses;
  _Atomic uint32_t resident;
};
static uint32_t hash_key(uint64_t x) {
  x ^= x >> 30;
  x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27;
  x *= UINT64_C(0x94d049bb133111eb);
  return (uint32_t)(x ^ (x >> 31));
}
static uint32_t locate(r3d_surface_reader *r, uint64_t key) {
  uint32_t h = hash_key(key) & (r->hash_cap - 1);
  while (r->hash[h]) {
    uint32_t slot = r->hash[h] - 1;
    if (r->slots[slot].key == key)
      return h;
    h = (h + 1) & (r->hash_cap - 1);
  }
  return h;
}
static void erase(r3d_surface_reader *r, uint64_t key) {
  uint32_t h = locate(r, key);
  r->hash[h] = 0;
  for (uint32_t j = (h + 1) & (r->hash_cap - 1); r->hash[j];
       j = (j + 1) & (r->hash_cap - 1)) {
    uint32_t slot = r->hash[j] - 1;
    r->hash[j] = 0;
    r->hash[locate(r, r->slots[slot].key)] = slot + 1;
  }
}
int r3d_surface_open(const char *path, size_t budget,
                     r3d_surface_reader **out) {
  if (!out)
    return -1;
  *out = NULL;
  r3d_surface_reader *r = calloc(1, sizeof *r);
  if (!r)
    return -1;
  atomic_init(&r->hits, 0);
  atomic_init(&r->misses, 0);
  atomic_init(&r->resident, 0);
  if (pthread_mutex_init(&r->mu, NULL)) {
    free(r);
    return -1;
  }
  if (sfc_open_file(path, &r->file))
    goto fail;
  for (int a = 0; a < 3; a++) {
    char name[2] = {"xyz"[a], 0};
    r->channels[a] = sfc_find_channel(r->file, name);
    if (r->channels[a] < 0)
      goto fail;
    const sfc_channel *c = sfc_channel_info(r->file, (uint32_t)r->channels[a]);
    if (c->dtype != SFC_F32)
      goto fail;
    if (!a) {
      r->info.width = c->width;
      r->info.height = c->height;
    } else if (c->width != r->info.width || c->height != r->info.height)
      goto fail;
  }
  size_t n = (size_t)sfc_metadata_size(r->file);
  char *meta = malloc(n + 1);
  if (!meta)
    goto fail;
  if (sfc_read_metadata(r->file, meta, n)) {
    free(meta);
    goto fail;
  }
  meta[n] = 0;
  const char *p = strstr(meta, "\"scale\"");
  float sx = 0, sy = 0;
  if (p)
    p = strchr(p, '[');
  int ok = p && sscanf(p, "[ %f , %f", &sx, &sy) == 2 && isfinite(sx) &&
           isfinite(sy) && sx > 0 && sy > 0;
  free(meta);
  if (!ok)
    goto fail;
  r->info.sx = sx;
  r->info.sy = sy;
  if (!budget)
    budget = 256u << 20;
  size_t max = budget / TILE_BYTES;
  if (!max)
    goto fail;
  if (max > 65536)
    max = 65536;
  r->cap = (uint32_t)max;
  r->hash_cap = 2;
  while (r->hash_cap < r->cap * 2)
    r->hash_cap *= 2;
  r->slots = calloc(r->cap, sizeof *r->slots);
  r->hash = calloc(r->hash_cap, sizeof *r->hash);
  if (!r->slots || !r->hash)
    goto fail;
  *out = r;
  return 0;
fail:
  r3d_surface_close(r);
  return -1;
}
void r3d_surface_close(r3d_surface_reader *r) {
  if (!r)
    return;
  for (uint32_t i = 0; i < r->used; i++)
    free(r->slots[i].data);
  free(r->slots);
  free(r->hash);
  sfc_close(r->file);
  pthread_mutex_destroy(&r->mu);
  free(r);
}
void r3d_surface_get_info(const r3d_surface_reader *r, r3d_surface_info *out) {
  if (r && out)
    *out = r->info;
}
/* Mutex covers misses as well as hits: concurrent reads share one decode,
 * and a slot cannot be evicted until the requesting reader has copied it. */
static float *tile(r3d_surface_reader *r, uint64_t bx, uint64_t by) {
  uint64_t nx = r->info.width / 64 + (r->info.width % 64 != 0),
           key = by * nx + bx;
  uint32_t h = locate(r, key);
  if (r->hash[h]) {
    struct cached_tile *s = r->slots + r->hash[h] - 1;
    s->age = ++r->clock;
    r->hits++;
    return s->data;
  }
  r->misses++;
  float *data = malloc(TILE_BYTES);
  if (!data)
    return NULL;
  uint8_t valid[4096], mask[4096];
  memset(valid, 1, sizeof valid);
  if (r->channels[1] == r->channels[0] + 1 &&
      r->channels[2] == r->channels[0] + 2) {
    if (sfc_read_xyz(r->file, (uint32_t)r->channels[0], bx, by, data, 12, 768,
                     valid)) {
      free(data);
      return NULL;
    }
  } else
    for (int a = 0; a < 3; a++) {
      if (sfc_read_block(r->file, (uint32_t)r->channels[a], bx, by, data + a,
                         12, 768, mask)) {
        free(data);
        return NULL;
      }
      for (unsigned k = 0; k < 4096; k++)
        valid[k] &= mask[k];
    }
  for (unsigned k = 0; k < 4096; k++)
    if (!valid[k] || !(data[k * 3 + 2] > 0) || !isfinite(data[k * 3]) ||
        !isfinite(data[k * 3 + 1]) || !isfinite(data[k * 3 + 2]))
      data[k * 3] = data[k * 3 + 1] = data[k * 3 + 2] = -1;
  uint32_t slot = r->used;
  if (slot < r->cap)
    r->used++;
  else {
    uint64_t oldest = UINT64_MAX;
    slot = 0;
    for (uint32_t i = 0; i < r->used; i++)
      if (r->slots[i].age < oldest) {
        oldest = r->slots[i].age;
        slot = i;
      }
    erase(r, r->slots[slot].key);
    free(r->slots[slot].data);
  }
  r->slots[slot] = (struct cached_tile){key, ++r->clock, data};
  atomic_store_explicit(&r->resident, r->used, memory_order_relaxed);
  r->hash[locate(r, key)] = slot + 1;
  return data;
}
int r3d_surface_read(r3d_surface_reader *r, uint64_t x, uint64_t y, uint32_t w,
                     uint32_t h, float *dst, size_t stride) {
  if (!r || !dst || !w || !h || x > r->info.width || y > r->info.height ||
      w > r->info.width - x || h > r->info.height - y ||
      stride < (size_t)w * 3 || (uint64_t)w * h > ROI_LIMIT / 12 ||
      (size_t)(h - 1) > (SIZE_MAX / sizeof(float) - (size_t)w * 3) / stride)
    return -1;
  float *tmp = malloc((size_t)w * h * 12);
  if (!tmp)
    return -1;
  int rc = 0;
  pthread_mutex_lock(&r->mu);
  for (uint64_t by = y / 64; !rc && by <= (y + h - 1) / 64; by++)
    for (uint64_t bx = x / 64; !rc && bx <= (x + w - 1) / 64; bx++) {
      float *data = tile(r, bx, by);
      if (!data) {
        rc = -1;
        break;
      }
      uint64_t x0 = x > bx * 64 ? x : bx * 64, y0 = y > by * 64 ? y : by * 64,
               x1 = x + w < (bx + 1) * 64 ? x + w : (bx + 1) * 64,
               y1 = y + h < (by + 1) * 64 ? y + h : (by + 1) * 64;
      for (uint64_t yy = y0; yy < y1; yy++)
        memcpy(tmp + ((size_t)(yy - y) * w + (size_t)(x0 - x)) * 3,
               data +
                   ((size_t)(yy - by * 64) * 64 + (size_t)(x0 - bx * 64)) * 3,
               (size_t)(x1 - x0) * 12);
    }
  pthread_mutex_unlock(&r->mu);
  if (!rc)
    for (uint32_t j = 0; j < h; j++)
      memcpy(dst + (size_t)j * stride, tmp + (size_t)j * w * 3, (size_t)w * 12);
  free(tmp);
  return rc;
}
int r3d_surface_point(r3d_surface_reader *r, uint64_t x, uint64_t y,
                      float xyz[3]) {
  if (!r || !xyz || x >= r->info.width || y >= r->info.height)
    return -1;
  pthread_mutex_lock(&r->mu);
  float *data = tile(r, x / 64, y / 64);
  if (data)
    memcpy(xyz, data + ((y % 64) * 64 + x % 64) * 3, 3 * sizeof(float));
  pthread_mutex_unlock(&r->mu);
  return data ? 0 : -1;
}
int r3d_surface_block_bounds(r3d_surface_reader *r, uint64_t bx, uint64_t by,
                             float lo[3], float hi[3]) {
  if (!r || !lo || !hi)
    return -1;
  float a[3], b[3];
  for (unsigned k = 0; k < 3; k++)
    if (sfc_block_range(r->file, (uint32_t)r->channels[k], bx, by, a + k,
                        b + k))
      return -1;
  memcpy(lo, a, sizeof a);
  memcpy(hi, b, sizeof b);
  return 0;
}
void r3d_surface_get_stats(r3d_surface_reader *r, r3d_surface_stats *s) {
  if (!r || !s)
    return;
  *s = (r3d_surface_stats){
      atomic_load_explicit(&r->hits, memory_order_relaxed),
      atomic_load_explicit(&r->misses, memory_order_relaxed),
      (uint64_t)atomic_load_explicit(&r->resident, memory_order_relaxed) *
          TILE_BYTES,
      (uint64_t)r->cap * TILE_BYTES};
}

/* Materialize only an explicitly requested window for existing geometry code.
 * The caller owns the result; global origin remains with the window request. */
int r3d_surface_window(r3d_surface_reader *r, uint64_t x, uint64_t y,
                       uint32_t w, uint32_t h, r3d_tifxyz *out) {
  if (!r || !out || !w || !h || (uint64_t)w * h > ROI_LIMIT / 12)
    return -1;
  r3d_tifxyz s = {.w = w, .h = h, .sx = r->info.sx, .sy = r->info.sy};
  s.xyz = malloc((size_t)w * h * 12);
  if (!s.xyz)
    return -1;
  if (r3d_surface_read(r, x, y, w, h, s.xyz, (size_t)w * 3)) {
    free(s.xyz);
    return -1;
  }
  for (uint64_t k = 0; k < (uint64_t)w * h; k++) {
    const float *p = s.xyz + k * 3;
    if (!r3d_tifxyz_valid(p))
      continue;
    for (unsigned a = 0; a < 3; a++) {
      if (!s.nvalid || p[a] < s.bbox[0][a])
        s.bbox[0][a] = p[a];
      if (!s.nvalid || p[a] > s.bbox[1][a])
        s.bbox[1][a] = p[a];
    }
    s.nvalid++;
  }
  *out = s;
  return 0;
}

int r3d_surface_axis_window(uint64_t full, uint32_t extent, uint64_t origin,
                            double local, const uint64_t *jump, uint64_t *next,
                            double *camera) {
  if (!full || !extent || extent > full || origin > full - extent || !next ||
      !camera || !isfinite(local))
    return -1;
  uint64_t center;
  double fraction = 0;
  if (jump)
    center = *jump < full ? *jump : full - 1;
  else {
    double whole = floor(local);
    fraction = local - whole;
    if (whole >= 0) {
      uint64_t room = full - 1 - origin;
      if (whole >= (double)room) {
        center = full - 1;
        fraction = 0;
      } else
        center = origin + (uint64_t)whole;
    } else if (-whole > (double)origin) {
      center = 0;
      fraction = 0;
    } else
      center = origin - (uint64_t)(-whole);
  }
  double relative =
      center >= origin ? (double)(center - origin) : -(double)(origin - center);
  relative += fraction;
  *next = origin;
  if (jump || relative < (double)extent * .25 ||
      relative > (double)extent * .75) {
    *next = center > extent / 2 ? (center - extent / 2) / 64 * 64 : 0;
    if (*next > full - extent)
      *next = full - extent;
  }
  *camera =
      (center >= *next ? (double)(center - *next) : -(double)(*next - center)) +
      fraction;
  return 0;
}
