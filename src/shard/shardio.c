#include "shard/shardio.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "dct3d.h"

int r3d_shard_store_init(r3d_shard_store *s, const char *dir, uint64_t nz, uint64_t ny,
                         uint64_t nx) {
  if (strlen(dir) >= sizeof s->dir) return -1;
  strcpy(s->dir, dir);
  s->nz = nz;
  s->ny = ny;
  s->nx = nx;
  return 0;
}

int r3d_shard_open(const r3d_shard_store *s, uint32_t sz, uint32_t sy, uint32_t sx,
                   r3d_shard *sh) {
  memset(sh, 0, sizeof *sh);
  char path[600];
  snprintf(path, sizeof path, "%s/%u_%u_%u.shard", s->dir, sz, sy, sx);
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st) != 0 || (size_t)st.st_size < R3D_SHARD_INDEX_BYTES) {
    close(fd);
    return -1;
  }
  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return -1;
  sh->map = map;
  sh->n = (size_t)st.st_size;
  sh->index = sh->map + sh->n - R3D_SHARD_INDEX_BYTES;
  return 0;
}

void r3d_shard_close(r3d_shard *sh) {
  if (sh->map) munmap((void *)sh->map, sh->n);
  memset(sh, 0, sizeof *sh);
}

const uint8_t *r3d_shard_chunk_blob(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                                    size_t *nbytes) {
  if (cz >= R3D_SHARD_GRID || cy >= R3D_SHARD_GRID || cx >= R3D_SHARD_GRID) return NULL;
  size_t ci = ((size_t)cz * R3D_SHARD_GRID + cy) * R3D_SHARD_GRID + cx;
  uint64_t off, n;
  memcpy(&off, sh->index + ci * 16, 8);
  memcpy(&n, sh->index + ci * 16 + 8, 8);
  if (off == UINT64_MAX || n == UINT64_MAX) return NULL; /* missing */
  if (off + n > sh->n) return NULL;                      /* corrupt */
  *nbytes = (size_t)n;
  return sh->map + off;
}

int r3d_shard_chunk_decode(const r3d_shard *sh, uint32_t cz, uint32_t cy, uint32_t cx,
                           uint8_t out[DCT3D_N3]) {
  size_t n = 0;
  const uint8_t *blob = r3d_shard_chunk_blob(sh, cz, cy, cx, &n);
  if (!blob) {
    memset(out, 0, DCT3D_N3);
    return 0; /* missing = masked = zero */
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
  uint64_t total;
} region_job;

static void *region_worker(void *arg) {
  region_job *j = arg;
  const uint32_t C = R3D_SHARD_CHUNK, G = R3D_SHARD_GRID;
  uint64_t sy_n = j->c1y - j->c0y + 1, sx_n = j->c1x - j->c0x + 1;
  r3d_shard cur = {0};
  uint64_t cur_key = UINT64_MAX;
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
    uint64_t key = (sz << 40) | (sy << 20) | sx;
    if (key != cur_key) { /* chunk order is x-fastest: shard reuse is high */
      r3d_shard_close(&cur);
      cur_key = key;
      if (r3d_shard_open(j->s, (uint32_t)sz, (uint32_t)sy, (uint32_t)sx, &cur) != 0)
        cur.map = NULL; /* absent shard: zeros */
    }
    if (cur.map)
      r3d_shard_chunk_decode(&cur, (uint32_t)(cz % G), (uint32_t)(cy % G), (uint32_t)(cx % G),
                             chunk);
    else
      memset(chunk, 0, sizeof chunk);

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
  if (dz == 0 || dy == 0 || dx == 0) return -1;
  memset(dst, 0, (size_t)dz * dy * dx); /* out-of-volume / missing = 0 */

  /* clamp region to volume */
  if (z0 >= s->nz || y0 >= s->ny || x0 >= s->nx) return 0;
  uint64_t z1 = z0 + dz < s->nz ? z0 + dz : s->nz;
  uint64_t y1 = y0 + dy < s->ny ? y0 + dy : s->ny;
  uint64_t x1 = x0 + dx < s->nx ? x0 + dx : s->nx;

  region_job j = {
      .s = s, .z0 = z0, .y0 = y0, .x0 = x0, .z1 = z1, .y1 = y1, .x1 = x1,
      .sdy = dy, .sdx = dx, .dst = dst,
      .c0z = z0 / R3D_SHARD_CHUNK, .c0y = y0 / R3D_SHARD_CHUNK, .c0x = x0 / R3D_SHARD_CHUNK,
      .c1z = (z1 - 1) / R3D_SHARD_CHUNK, .c1y = (y1 - 1) / R3D_SHARD_CHUNK,
      .c1x = (x1 - 1) / R3D_SHARD_CHUNK,
  };
  j.total = (j.c1z - j.c0z + 1) * (j.c1y - j.c0y + 1) * (j.c1x - j.c0x + 1);
  atomic_store(&j.next, 0);

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  uint32_t nt = nthreads > 0 ? (uint32_t)nthreads : (ncpu > 1 ? (uint32_t)ncpu : 1);
  if (nt > 16) nt = 16;
  if ((uint64_t)nt > j.total) nt = (uint32_t)j.total;

  pthread_t tids[16];
  for (uint32_t t = 0; t < nt; t++) pthread_create(&tids[t], NULL, region_worker, &j);
  for (uint32_t t = 0; t < nt; t++) pthread_join(tids[t], NULL);
  return 0;
}
