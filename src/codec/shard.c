#include "shard.h"
#include "bytes.h"
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
struct volcomp_shard_writer {
  FILE *f;
  uint8_t footer[32], *index;
  uint32_t count;
  uint64_t cursor;
  int failed;
};
static uint32_t crc_table[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;
static void crc_init(void) {
  for (uint32_t j = 0; j < 256; j++) {
    uint32_t c = j;
    for (unsigned i = 0; i < 8; i++)
      c = (c >> 1) ^ (0x82f63b78u & (0u - (c & 1u)));
    crc_table[j] = c;
  }
}
uint32_t volcomp_crc32c(const void *data, size_t n) {
  pthread_once(&crc_once, crc_init);
  const uint8_t *p = data;
  uint32_t c = UINT32_MAX;
  while (n--)
    c = (c >> 8) ^ crc_table[(c ^ *p++) & 255u];
  return ~c;
}
static uint32_t brick_count(uint32_t sd, uint32_t bd) {
  if (bd != 128 || sd < bd || sd % bd)
    return 0;
  uint64_t d = sd / bd;
  return d <= 64 ? (uint32_t)(d * d * d) : 0;
}
volcomp_shard_writer *volcomp_shard_create(const char *path, uint32_t sd,
                                           uint32_t bd, uint32_t lod, float q) {
  uint32_t nb = brick_count(sd, bd), bits;
  if (!path || !nb || !isfinite(q) || q < 1 || q > 255)
    return NULL;
  volcomp_shard_writer *w = calloc(1, sizeof *w);
  if (!w)
    return NULL;
  w->count = nb;
  w->index = calloc(nb, 16);
  if (!w->index) {
    free(w);
    return NULL;
  }
  w->f = fopen(path, "wb");
  if (!w->f) {
    free(w->index);
    free(w);
    return NULL;
  }
  for (uint32_t i = 0; i < nb; i++)
    r3c_put64(w->index + 16 * i, VOLCOMP_SHARD_OFFSET_MISSING);
  r3c_put32(w->footer, VOLCOMP_SHARD_MAGIC);
  r3c_put32(w->footer + 4, 1);
  r3c_put32(w->footer + 8, sd);
  r3c_put32(w->footer + 12, bd);
  r3c_put32(w->footer + 16, nb);
  r3c_put32(w->footer + 20, lod);
  memcpy(&bits, &q, 4);
  r3c_put32(w->footer + 24, bits);
  return w;
}
int volcomp_shard_put(volcomp_shard_writer *w, uint32_t i, const uint8_t *data,
                      size_t n) {
  if (!w || i >= w->count || !data || !n || n > UINT32_MAX || w->failed)
    return -1;
  if (fwrite(data, 1, n, w->f) != n) {
    w->failed = 1;
    return -1;
  }
  uint8_t *e = w->index + 16 * i;
  r3c_put64(e, w->cursor);
  r3c_put32(e + 8, (uint32_t)n);
  r3c_put32(e + 12, volcomp_crc32c(data, n));
  w->cursor += n;
  return 0;
}
int volcomp_shard_put_zero(volcomp_shard_writer *w, uint32_t i) {
  if (!w || i >= w->count || w->failed)
    return -1;
  memset(w->index + 16 * i, 0, 16);
  r3c_put64(w->index + 16 * i, VOLCOMP_SHARD_OFFSET_ZERO);
  return 0;
}
int volcomp_shard_close(volcomp_shard_writer *w) {
  if (!w)
    return -1;
  size_t n = (size_t)w->count * 16;
  int rc = w->failed || fwrite(w->index, 1, n, w->f) != n ||
           fwrite(w->footer, 1, 32, w->f) != 32;
  if (fflush(w->f) != 0 || fsync(fileno(w->f)) != 0)
    rc = 1;
  if (fclose(w->f) != 0)
    rc = 1;
  free(w->index);
  free(w);
  return rc ? -1 : 0;
}
int volcomp_shard_open(const char *path, volcomp_shard_reader *r) {
  if (!r || !path)
    return -1;
  memset(r, 0, sizeof *r);
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    return -1;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 32 ||
      (uint64_t)st.st_size > SIZE_MAX) {
    close(fd);
    return -1;
  }
  size_t n = (size_t)st.st_size;
  void *map = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED)
    return -1;
  const uint8_t *f = (const uint8_t *)map + n - 32;
  uint32_t nb = r3c_u32(f + 16), sd = r3c_u32(f + 8), bd = r3c_u32(f + 12),
           bits = r3c_u32(f + 24);
  float q;
  memcpy(&q, &bits, 4);
  if (r3c_u32(f) != VOLCOMP_SHARD_MAGIC || r3c_u32(f + 4) != 1 || !nb ||
      nb != brick_count(sd, bd) || (size_t)nb * 16 > n - 32 || !isfinite(q) ||
      q < 1 || q > 255) {
    munmap(map, n);
    return -1;
  }
  r->verified = calloc(nb, sizeof *r->verified);
  if (!r->verified) {
    munmap(map, n);
    return -1;
  }
  r->map = map;
  r->map_n = n;
  r->index_off = n - 32 - (size_t)nb * 16;
  r->foot = (volcomp_shard_footer){VOLCOMP_SHARD_MAGIC, 1, sd, bd, nb,
                                   r3c_u32(f + 20),     q, 0};
  return 0;
}
const uint8_t *volcomp_shard_brick(const volcomp_shard_reader *r, uint32_t i,
                                   size_t *n) {
  if (n)
    *n = 0;
  if (!r || !r->map || i >= r->foot.nbricks)
    return NULL;
  const uint8_t *e = r->map + r->index_off + (size_t)i * 16;
  uint64_t off = r3c_u64(e);
  uint32_t len = r3c_u32(e + 8);
  if (!len || off > r->index_off || len > r->index_off - off)
    return NULL;
  const uint8_t *p = r->map + (size_t)off;
  uint8_t checked = atomic_load(&r->verified[i]);
  if (!checked) {
    checked = volcomp_crc32c(p, len) == r3c_u32(e + 12) ? 1u : 2u;
    atomic_store(&r->verified[i], checked);
  }
  if (checked != 1u)
    return NULL;
  if (n)
    *n = len;
  return p;
}
int volcomp_shard_brick_is_zero(const volcomp_shard_reader *r, uint32_t i) {
  if (!r || !r->map || i >= r->foot.nbricks)
    return 0;
  const uint8_t *e = r->map + r->index_off + (size_t)i * 16;
  return r3c_u64(e) == VOLCOMP_SHARD_OFFSET_ZERO && !r3c_u64(e + 8);
}
void volcomp_shard_close_reader(volcomp_shard_reader *r) {
  if (!r)
    return;
  if (r->map)
    munmap((void *)r->map, r->map_n);
  free(r->verified);
  memset(r, 0, sizeof *r);
}
