/* volcomppack — encode volumes into volcomp .vcs shards for the CPU-decode renderer.
 *
 *   volcomppack raw  <in.u8> <nx> <ny> <nz> <out.vcs> [q=2] [threads=0]
 *   volcomppack band <band_dir> <Z> <Y> <X> <out.vcs> [q=2] [threads=0]
 *
 * raw:  dims must be multiples of 128 and equal (cubic shard); one L0 shard.
 * band: decodes one 1024^3 shard's worth of voxels from the 3ddct band store
 *       (tools/fetch_band.py layout) and encodes it as one volcomp shard.
 * q is the volume-compressor quantizer (1..255). */
#include <pthread.h>
#include "../common/workers.h"
#include <stdbool.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include "brick.h"
#include "bytes.h"
#include "shard.h"
#include "shard/shardio.h"

#define BD 128u

/* Zarr v3 sharding_indexed, 1024^3 shard / 128^3 chunks, little-endian
 * offset/length index with trailing CRC32C. Preserve native payload bytes. */
static int import_zarr_shard(const char *in, const char *out, uint32_t lod, float q) {
  int fd = open(in, O_RDONLY);
  struct stat st;
  if (fd < 0) return -1;
  if (fstat(fd, &st) != 0 || st.st_size < 8196 || (uint64_t)st.st_size > SIZE_MAX) {
    close(fd);
    return -1;
  }
  size_t n = (size_t)st.st_size, index_off = n - 8196;
  const uint8_t *map = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return -1;
  int rc = -1;
  if (volcomp_crc32c(map + index_off, 8192) != r3c_u32(map + n - 4)) goto done;
  char *tmp = malloc(strlen(out) + 16);
  if (!tmp) goto done;
  snprintf(tmp, strlen(out) + 16u, "%s.tmp.XXXXXX", out);
  fd = mkstemp(tmp);
  if (fd < 0) { free(tmp); goto done; }
  close(fd);
  volcomp_shard_writer *w = volcomp_shard_create(tmp, 1024, 128, lod, q);
  rc = w ? 0 : -1;
  uint32_t present = 0;
  for (uint32_t b = 0; b < 512 && rc == 0; b++) {
    const uint8_t *e = map + index_off + b * 16;
    uint64_t off = r3c_u64(e), len = r3c_u64(e + 8);
    if (off == UINT64_MAX && len == UINT64_MAX) {
      rc = volcomp_shard_put_zero(w, b); /* Zarr fill_value is zero */
    } else if (off > index_off || len > index_off - off || len < 8 ||
               len > UINT32_MAX || memcmp(map + off, "VOLC", 4) != 0) {
      rc = -1;
    } else {
      rc = volcomp_shard_put(w, b, map + off, (size_t)len);
      present++;
    }
  }
  if (w && volcomp_shard_close(w) != 0) rc = -1;
  if (rc == 0 && rename(tmp, out) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  else printf("imported %s: %u native chunks, %zu bytes, no re-encoding\n", out, present, n);
  free(tmp);
done:
  munmap((void *)map, n);
  return rc;
}

typedef struct enc_job {
  const uint8_t *vol;
  const r3d_shard *source;
  uint32_t dim, bpa, first, end;
  uint8_t *prepared; /* bounded band-decode batch, indexed relative to first */
  float q;
  uint8_t **blobs;
  size_t *ns;
  _Atomic uint32_t next;
  _Atomic int fail;
} enc_job;

static void *prepare_worker(void *arg) {
  enc_job *j = arg;
  uint8_t chunk[4096];
  uint32_t total = (j->end - j->first) * 512u;
  for (;;) {
    /*64 DCT chunks per claim, matching regiondecode's fine scheduling.
     * This balances performance/efficiency cores without enlarging raw RAM. */
    uint32_t first = atomic_fetch_add_explicit(&j->next, 64u, memory_order_relaxed);
    if (first >= total || atomic_load(&j->fail)) break;
    uint32_t b = j->first + first / 512u, z = first % 512u / 64u;
    uint32_t bz = b / (j->bpa * j->bpa), by = (b / j->bpa) % j->bpa, bx = b % j->bpa;
    uint8_t *cube = j->prepared + (size_t)(b - j->first) * BD * BD * BD;
    for (uint32_t y = 0; y < 8; y++)
      for (uint32_t x = 0; x < 8; x++) {
        if (r3d_shard_chunk_decode(j->source, bz * 8u + z, by * 8u + y,
                                   bx * 8u + x, chunk) != 0) {
          atomic_store(&j->fail, 1);
          return NULL;
        }
        for (uint32_t zz = 0; zz < 16; zz++)
          for (uint32_t yy = 0; yy < 16; yy++)
            memcpy(cube + (((size_t)z * 16u + zz) * BD + y * 16u + yy) * BD + x * 16u,
                   chunk + (zz * 16u + yy) * 16u, 16u);
      }
  }
  return NULL;
}
static void *enc_worker(void *arg) {
  enc_job *j = arg;
  uint8_t *scratch = j->prepared ? NULL : malloc((size_t)BD * BD * BD);
  if (!j->prepared && !scratch) {
    atomic_store(&j->fail, 1);
    return NULL;
  }
  volcomp_brick_params p = volcomp_brick_defaults(j->q);
  for (;;) {
    uint32_t b = atomic_fetch_add(&j->next, 1);
    if (b >= j->end || atomic_load(&j->fail)) break;
    uint8_t *cube = j->prepared ? j->prepared + (size_t)(b - j->first) * BD * BD * BD : scratch;
    if (!j->prepared) {
      uint32_t bz = b / (j->bpa * j->bpa), by = (b / j->bpa) % j->bpa, bx = b % j->bpa;
      for (uint32_t z = 0; z < BD; z++)
        for (uint32_t y = 0; y < BD; y++)
          memcpy(cube + ((size_t)z * BD + y) * BD,
                 j->vol + (((size_t)(bz * BD + z) * j->dim + by * BD + y) * j->dim + bx * BD), BD);
    }
    bool zero = true;
    for (size_t i = 0; i < (size_t)BD * BD * BD; i++)
      if (cube[i]) { zero = false; break; }
    if (zero) continue;
    if (volcomp_brick_encode(&p, cube, BD, &j->blobs[b], &j->ns[b]) != 0) atomic_store(&j->fail, 1);
  }
  free(scratch);
  return NULL;
}

static int pack_volume(const uint8_t *vol, const r3d_shard *source, uint32_t dim, const char *out, float q,
                       unsigned nthreads) {
  uint32_t bpa = dim / BD, nb = bpa * bpa * bpa;
  uint8_t **blobs = calloc(nb, sizeof *blobs);
  size_t *ns = calloc(nb, sizeof *ns);
  if (!blobs || !ns) { free(blobs); free(ns); return -1; }
  enc_job j = {.vol = vol, .source = source, .dim = dim, .bpa = bpa, .q = q, .blobs = blobs, .ns = ns};
  atomic_store(&j.next, 0);
  atomic_store(&j.fail, 0);
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  uint32_t nt = nthreads ? nthreads : (ncpu > 1 ? (uint32_t)ncpu : 1);
  if (nt > 16) nt = 16;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  char *tmp = malloc(strlen(out) + 16u);
  if (!tmp) { free(blobs); free(ns); return -1; }
  snprintf(tmp, strlen(out) + 16u, "%s.tmp.XXXXXX", out);
  int fd = mkstemp(tmp);
  if (fd < 0) { free(tmp); free(blobs); free(ns); return -1; }
  close(fd);
  volcomp_shard_writer *w = volcomp_shard_create(tmp, dim, BD, 0, q);
  int rc = w ? 0 : -1;
  r3d_tool_workers pool;
  bool pool_ready = rc == 0 && r3d_tool_workers_init(&pool, nt) == 0;
  if (!pool_ready) rc = -1;
  /* Preserve original automatic CPU parallelism for band decoding, even
   * when the caller requests fewer encoder threads. Only16 raw chunks
   * (32MiB) are prepared, then the bounded encoder phase drains them. */
  uint32_t batch = source ? 16u : nt;
  r3d_tool_workers decode_pool;
  bool decode_ready = false;
  if (source && rc == 0) {
    uint32_t decode_threads = ncpu > 1 ? (uint32_t)ncpu : 1u;
    if (decode_threads > 16u) decode_threads = 16u;
    j.prepared = malloc((size_t)batch * BD * BD * BD);
    if (j.prepared) decode_ready = r3d_tool_workers_init(&decode_pool, decode_threads) == 0;
    if (!decode_ready) rc = -1;
  }
  size_t total = 0;
  for (uint32_t first = 0; first < nb && rc == 0; first += batch) {
    j.first = first;
    j.end = first + batch < nb ? first + batch : nb;
    if (source) {
      atomic_store(&j.next, 0);
      r3d_tool_workers_run(&decode_pool, prepare_worker, &j);
      if (atomic_load(&j.fail)) { rc = -1; break; }
    }
    atomic_store(&j.next, first);
    r3d_tool_workers_run(&pool, enc_worker, &j);
    if (atomic_load(&j.fail)) { rc = -1; break; }
    for (uint32_t b = first; b < j.end && rc == 0; b++) {
      rc = ns[b] ? volcomp_shard_put(w, b, blobs[b], ns[b]) : volcomp_shard_put_zero(w, b);
      total += ns[b];
      free(blobs[b]);
      blobs[b] = NULL;
    }
  }
  if (decode_ready) r3d_tool_workers_destroy(&decode_pool);
  if (pool_ready) r3d_tool_workers_destroy(&pool);
  free(j.prepared);
  if (w && volcomp_shard_close(w) != 0) rc = -1;
  if (rc == 0 && rename(tmp, out) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  free(tmp);
  for (uint32_t b = 0; b < nb; b++) free(blobs[b]);
  free(blobs);
  free(ns);
  if (rc != 0) { fprintf(stderr, "volcomppack: conversion failed\n"); return -1; }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
  double raw = (double)dim * dim * dim;
  printf("volcomppack: %s: %u bricks, %.1f MB (%.1fx, q=%.2f), encode %.1fs (%.0f MB/s)\n", out, nb,
         (double)total / 1e6, raw / (double)total, (double)q, secs, raw / 1e6 / secs);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) goto usage;
  float q = 2.0f;
  unsigned nthreads = 0;

  if (strcmp(argv[1], "zarr-shard") == 0 && argc == 6) {
    unsigned long level = strtoul(argv[4], NULL, 10);
    float quality = strtof(argv[5], NULL);
    if (level > 31 || !(quality >= 1.0f && quality <= 255.0f)) return EXIT_FAILURE;
    int rc = import_zarr_shard(argv[2], argv[3], (uint32_t)level, quality);
    if (rc != 0) fprintf(stderr, "volcomppack: invalid or unreadable Zarr shard %s\n", argv[2]);
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (strcmp(argv[1], "raw") == 0 && argc >= 7) {
    uint32_t nx = (uint32_t)atoi(argv[3]), ny = (uint32_t)atoi(argv[4]),
             nz = (uint32_t)atoi(argv[5]);
    if (argc > 7) q = (float)atof(argv[7]);
    if (argc > 8) nthreads = (unsigned)atoi(argv[8]);
    if (!nx || nx > 4096u || nx != ny || ny != nz || nx % BD != 0 ||
        !(q >= 1.0f && q <= 255.0f)) {
      fprintf(stderr, "volcomppack raw: dims must be equal multiples of 128\n");
      return EXIT_FAILURE;
    }
    int fd = open(argv[2], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || (size_t)st.st_size != (size_t)nx * ny * nz) {
      fprintf(stderr, "volcomppack: bad input file/size\n");
      return EXIT_FAILURE;
    }
    const uint8_t *vol = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (vol == MAP_FAILED) return EXIT_FAILURE;
    int rc = pack_volume(vol, NULL, nx, argv[6], q, nthreads);
    munmap((void *)vol, (size_t)st.st_size);
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (strcmp(argv[1], "band") == 0 && argc >= 7) {
    uint32_t Z = (uint32_t)atoi(argv[3]), Y = (uint32_t)atoi(argv[4]),
             X = (uint32_t)atoi(argv[5]);
    if (argc > 7) q = (float)atof(argv[7]);
    if (argc > 8) nthreads = (unsigned)atoi(argv[8]);
    r3d_shard_store store;
    if (r3d_shard_store_init(&store, argv[2], 68608, 43008, 43008) != 0) return EXIT_FAILURE;
    r3d_shard source;
    if (r3d_shard_open(&store, Z, Y, X, &source) != 0) return EXIT_FAILURE;
    int rc = pack_volume(NULL, &source, 1024, argv[6], q, nthreads);
    r3d_shard_close(&source);
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

usage:
  fprintf(stderr,
          "usage: volcomppack raw  <in.u8> <n> <n> <n> <out.vcs> [q] [threads]\n"
          "       volcomppack zarr-shard <input> <out.vcs> <lod> <q>\n"
          "       volcomppack band <band_dir> <Z> <Y> <X> <out.vcs> [q] [threads]\n");
  return EXIT_FAILURE;
}
