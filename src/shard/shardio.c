#include "shard/shardio.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "dct3d.h"

/* Zarr's index trailer uses CRC32C (Castagnoli, reflected). pthread_once keeps
 * the lookup table race-free when several decode workers open shards at once. */
static uint32_t crc_tab[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;

/* The index is 4 MiB, and a region decode can have 16 workers open the same
 * shard. Cache CRC results by immutable file identity so integrity checking is
 * paid once per inode, not once per worker/chunk run. Atomic download rename
 * naturally produces a new inode and therefore a new validation. */
#define CRC_CACHE_N 256u
typedef struct crc_cache_entry {
  dev_t dev;
  ino_t ino;
  off_t size;
  struct timespec mtim, ctim;
  int result; /* 1 valid, -1 invalid, 0 unused */
} crc_cache_entry;

static crc_cache_entry crc_cache[CRC_CACHE_N];
static uint32_t crc_cache_next;
static pthread_mutex_t crc_cache_mu = PTHREAD_MUTEX_INITIALIZER;

static void crc_init(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (uint32_t)-(int)(c & 1));
    crc_tab[i] = c;
  }
}

static uint32_t crc32c(const uint8_t *p, size_t n) {
  pthread_once(&crc_once, crc_init);
  uint32_t c = UINT32_MAX;
  while (n--) c = crc_tab[(c ^ *p++) & 0xffu] ^ (c >> 8);
  return c ^ UINT32_MAX;
}

static uint32_t get_u32le(const uint8_t *p);

static int same_file(const crc_cache_entry *e, const struct stat *st) {
  return e->result && e->dev == st->st_dev && e->ino == st->st_ino && e->size == st->st_size &&
         e->mtim.tv_sec == st->st_mtim.tv_sec && e->mtim.tv_nsec == st->st_mtim.tv_nsec &&
         e->ctim.tv_sec == st->st_ctim.tv_sec && e->ctim.tv_nsec == st->st_ctim.tv_nsec;
}

/* The lock deliberately covers the first CRC calculation. That makes other
 * workers wait for and reuse it rather than all hashing the same 4 MiB index.
 * Subsequent opens only take the short lookup path. */
static int validate_index_once(const struct stat *st, const uint8_t *index) {
  pthread_mutex_lock(&crc_cache_mu);
  for (uint32_t i = 0; i < CRC_CACHE_N; i++) {
    if (same_file(&crc_cache[i], st)) {
      int result = crc_cache[i].result;
      pthread_mutex_unlock(&crc_cache_mu);
      return result;
    }
  }
  uint32_t want = get_u32le(index + R3D_SHARD_NCHUNKS * 16u);
  int result = want == crc32c(index, R3D_SHARD_NCHUNKS * 16u) ? 1 : -1;
  crc_cache_entry *e = &crc_cache[crc_cache_next++ % CRC_CACHE_N];
  *e = (crc_cache_entry){.dev = st->st_dev,
                         .ino = st->st_ino,
                         .size = st->st_size,
                         .mtim = st->st_mtim,
                         .ctim = st->st_ctim,
                         .result = result};
  pthread_mutex_unlock(&crc_cache_mu);
  return result;
}

static uint32_t get_u32le(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

static uint64_t get_u64le(const uint8_t *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8u * i);
  return v;
}

int r3d_shard_store_init(r3d_shard_store *s, const char *dir, uint64_t nz, uint64_t ny,
                         uint64_t nx) {
  if (!s || !dir || strlen(dir) >= sizeof s->dir) return -1;
  strcpy(s->dir, dir);
  s->nz = nz;
  s->ny = ny;
  s->nx = nx;
  return 0;
}

int r3d_shard_open_path(const char *path, r3d_shard *sh) {
  if (!path || !sh) return R3D_SHARD_CORRUPT;
  memset(sh, 0, sizeof *sh);
  int fd = open(path, O_RDONLY);
  if (fd < 0) return errno == ENOENT ? R3D_SHARD_ABSENT : R3D_SHARD_CORRUPT;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 0 ||
      (uint64_t)st.st_size < R3D_SHARD_INDEX_BYTES ||
      (uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
    close(fd);
    return R3D_SHARD_CORRUPT;
  }
  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return R3D_SHARD_CORRUPT;
  sh->map = map;
  sh->n = (size_t)st.st_size;
  sh->index = sh->map + sh->n - R3D_SHARD_INDEX_BYTES;
  if (validate_index_once(&st, sh->index) < 0) {
    r3d_shard_close(sh);
    return R3D_SHARD_CORRUPT;
  }
  return R3D_SHARD_OK;
}

int r3d_shard_open(const r3d_shard_store *s, uint32_t sz, uint32_t sy, uint32_t sx,
                   r3d_shard *sh) {
  if (!s || !sh) return R3D_SHARD_CORRUPT;
  char path[600];
  int pn = snprintf(path, sizeof path, "%s/%u_%u_%u.shard", s->dir, sz, sy, sx);
  if (pn < 0 || (size_t)pn >= sizeof path) return R3D_SHARD_CORRUPT;
  return r3d_shard_open_path(path, sh);
}

void r3d_shard_close(r3d_shard *sh) {
  if (sh->map) munmap((void *)sh->map, sh->n);
  memset(sh, 0, sizeof *sh);
}

/* 0 = present, 1 = missing fill-value, -1 = invalid/corrupt. */
static int chunk_location(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                          const uint8_t **blob, size_t *nbytes) {
  if (!sh || !sh->map || !blob || !nbytes || cz >= R3D_SHARD_GRID || cy >= R3D_SHARD_GRID ||
      cx >= R3D_SHARD_GRID)
    return -1;
  size_t ci = ((size_t)cz * R3D_SHARD_GRID + cy) * R3D_SHARD_GRID + cx;
  uint64_t off = get_u64le(sh->index + ci * 16);
  uint64_t n = get_u64le(sh->index + ci * 16 + 8);
  if (off == UINT64_MAX && n == UINT64_MAX) return 1;
  if (off == UINT64_MAX || n == UINT64_MAX || n > SIZE_MAX) return -1;
  size_t payload_n = sh->n - R3D_SHARD_INDEX_BYTES;
  if (off > payload_n || n > (uint64_t)payload_n - off) return -1;
  *blob = sh->map + (size_t)off;
  *nbytes = (size_t)n;
  return 0;
}

const uint8_t *r3d_shard_chunk_blob(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                                    size_t *nbytes) {
  const uint8_t *blob = NULL;
  return chunk_location(sh, cz, cy, cx, &blob, nbytes) == 0 ? blob : NULL;
}

int r3d_shard_chunk_decode(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                           uint8_t out[DCT3D_N3]) {
  size_t n = 0;
  const uint8_t *blob = NULL;
  int loc = chunk_location(sh, cz, cy, cx, &blob, &n);
  if (loc > 0) {
    memset(out, 0, DCT3D_N3);
    return 0; /* missing = masked = zero */
  }
  if (loc < 0) {
    memset(out, 0, DCT3D_N3);
    return -1;
  }
  /* dct3d_decode_u8 returns nonzero on success (0 = corrupt input) */
  if (!dct3d_decode_u8(blob, n, out)) {
    memset(out, 0, DCT3D_N3);
    return -1;
  }
  return 0;
}

/* --- threaded region decode --- */

typedef struct region_job {
  const r3d_shard_store *s;
  uint64_t z0, y0, x0; /* region origin (dst voxel 0,0,0) */
  uint64_t z1, y1, x1; /* clamped exclusive bounds of data to write */
  uint32_t sdy, sdx;   /* dst strides: slice = sdx*sdy, row = sdx (caller-sized) */
  uint8_t *dst;
  /* work list: all (shard, chunk) pairs intersecting the region, flattened
   * into chunk coords in the global 16^3-chunk grid */
  uint64_t c0z, c0y, c0x, c1z, c1y, c1x; /* inclusive chunk bounds */
  _Atomic uint64_t next;                 /* linear chunk cursor */
  _Atomic int failed;
  uint64_t total;
} region_job;

static void *region_worker(void *arg) {
  region_job *j = arg;
  const uint32_t C = R3D_SHARD_CHUNK, G = R3D_SHARD_GRID;
  uint64_t sy_n = j->c1y - j->c0y + 1, sx_n = j->c1x - j->c0x + 1;
  r3d_shard cur = {0};
  uint64_t cur_sz = UINT64_MAX, cur_sy = UINT64_MAX, cur_sx = UINT64_MAX;
  uint8_t chunk[DCT3D_N3];

  for (;;) {
    /* batches of 64 chunks: consecutive x-run stays inside one shard */
    uint64_t i0 = atomic_fetch_add_explicit(&j->next, 64, memory_order_relaxed);
    if (i0 >= j->total) break;
    uint64_t i_end = i0 + 64 < j->total ? i0 + 64 : j->total;
    for (uint64_t i = i0; i < i_end; i++) {
    uint64_t cz = j->c0z + i / (sy_n * sx_n);
    uint64_t cy = j->c0y + (i / sx_n) % sy_n;
    uint64_t cx = j->c0x + i % sx_n;

    uint64_t sz = cz / G, sy = cy / G, sx = cx / G;
    if (sz != cur_sz || sy != cur_sy || sx != cur_sx) {
      /* Chunk order is x-fastest, so shard reuse is high. Keep coordinates
       * separately: packing them into fixed-width bit fields aliases very
       * large but otherwise valid stores. */
      r3d_shard_close(&cur);
      cur_sz = sz;
      cur_sy = sy;
      cur_sx = sx;
      int open_rc = r3d_shard_open(j->s, (uint32_t)sz, (uint32_t)sy, (uint32_t)sx, &cur);
      if (open_rc != R3D_SHARD_OK) {
        cur.map = NULL; /* absent shard: zeros; corrupt shard: error + zeros */
        if (open_rc == R3D_SHARD_CORRUPT)
          atomic_store_explicit(&j->failed, 1, memory_order_relaxed);
      }
    }
    if (cur.map) {
      if (r3d_shard_chunk_decode(&cur, (uint32_t)(cz % G), (uint32_t)(cy % G),
                                 (uint32_t)(cx % G), chunk) != 0) {
        atomic_store_explicit(&j->failed, 1, memory_order_relaxed);
        memset(chunk, 0, sizeof chunk);
      }
    } else {
      memset(chunk, 0, sizeof chunk);
    }

    /* copy the intersection of this chunk with the (clamped) region */
    uint64_t wz0 = cz * C, wy0 = cy * C, wx0 = cx * C;
    uint64_t iz0 = wz0 > j->z0 ? wz0 : j->z0;
    uint64_t iy0 = wy0 > j->y0 ? wy0 : j->y0;
    uint64_t ix0 = wx0 > j->x0 ? wx0 : j->x0;
    uint64_t iz1 = wz0 + C < j->z1 ? wz0 + C : j->z1;
    uint64_t iy1 = wy0 + C < j->y1 ? wy0 + C : j->y1;
    uint64_t ix1 = wx0 + C < j->x1 ? wx0 + C : j->x1;
    for (uint64_t z = iz0; z < iz1; z++)
      for (uint64_t y = iy0; y < iy1; y++)
        memcpy(j->dst + ((z - j->z0) * j->sdy + (y - j->y0)) * j->sdx + (ix0 - j->x0),
               chunk + ((z - wz0) * C + (y - wy0)) * C + (ix0 - wx0), ix1 - ix0);
    }
  }
  r3d_shard_close(&cur);
  return NULL;
}

int r3d_shard_decode_region(const r3d_shard_store *s, uint64_t z0, uint64_t y0, uint64_t x0,
                            uint32_t dz, uint32_t dy, uint32_t dx, uint8_t *dst, int nthreads) {
  if (!s || !dst || dz == 0 || dy == 0 || dx == 0 ||
      (size_t)dy > SIZE_MAX / (size_t)dx ||
      (size_t)dz > SIZE_MAX / ((size_t)dy * (size_t)dx))
    return -1;
  memset(dst, 0, (size_t)dz * dy * dx); /* out-of-volume / missing = 0 */

  /* clamp region to volume */
  if (z0 >= s->nz || y0 >= s->ny || x0 >= s->nx) return 0;
  uint64_t z1 = dz < s->nz - z0 ? z0 + dz : s->nz;
  uint64_t y1 = dy < s->ny - y0 ? y0 + dy : s->ny;
  uint64_t x1 = dx < s->nx - x0 ? x0 + dx : s->nx;

  region_job j = {
      .s = s, .z0 = z0, .y0 = y0, .x0 = x0, .z1 = z1, .y1 = y1, .x1 = x1,
      .sdy = dy, .sdx = dx, .dst = dst,
      .c0z = z0 / R3D_SHARD_CHUNK, .c0y = y0 / R3D_SHARD_CHUNK, .c0x = x0 / R3D_SHARD_CHUNK,
      .c1z = (z1 - 1) / R3D_SHARD_CHUNK, .c1y = (y1 - 1) / R3D_SHARD_CHUNK,
      .c1x = (x1 - 1) / R3D_SHARD_CHUNK,
  };
  j.total = (j.c1z - j.c0z + 1) * (j.c1y - j.c0y + 1) * (j.c1x - j.c0x + 1);
  atomic_store(&j.next, 0);
  atomic_store(&j.failed, 0);

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  uint32_t nt = nthreads > 0 ? (uint32_t)nthreads : (ncpu > 1 ? (uint32_t)ncpu : 1);
  if (nt > 16) nt = 16;
  if ((uint64_t)nt > j.total) nt = (uint32_t)j.total;

  pthread_t tids[16];
  uint32_t made = 0;
  for (; made < nt; made++)
    if (pthread_create(&tids[made], NULL, region_worker, &j) != 0) break;
  if (made == 0) region_worker(&j);
  for (uint32_t t = 0; t < made; t++) pthread_join(tids[t], NULL);
  return atomic_load_explicit(&j.failed, memory_order_relaxed) ? -1 : 0;
}
