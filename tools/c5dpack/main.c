/* c5dpack — encode volumes into c5d .c5s shards for the GPU-decode renderer.
 *
 *   c5dpack raw  <in.u8> <nx> <ny> <nz> <out.c5s> [q=2] [threads=0]
 *   c5dpack band <band_dir> <Z> <Y> <X> <out.c5s> [q=2] [threads=0]
 *
 * raw:  dims must be multiples of 128 and equal (cubic shard); one L0 shard.
 * band: decodes one 1024^3 shard's worth of voxels from the 3ddct band store
 *       (tools/fetch_band.py layout) and encodes it as one c5d shard.
 * lossless/tau stay off (the GPU decode path rejects them). */
#include <pthread.h>
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
#include "shard.h"
#include "shard/shardio.h"

#define BD 128u

typedef struct enc_job {
  const uint8_t *vol;
  uint32_t dim, bpa;
  float q;
  uint8_t **blobs;
  size_t *ns;
  _Atomic uint32_t next;
  _Atomic int fail;
} enc_job;

static void *enc_worker(void *arg) {
  enc_job *j = arg;
  uint8_t *cube = malloc((size_t)BD * BD * BD);
  if (!cube) {
    atomic_store(&j->fail, 1);
    return NULL;
  }
  c5d_brick_params p = c5d_brick_defaults(j->q);
  uint32_t nb = j->bpa * j->bpa * j->bpa;
  for (;;) {
    uint32_t b = atomic_fetch_add(&j->next, 1);
    if (b >= nb || atomic_load(&j->fail)) break;
    uint32_t bz = b / (j->bpa * j->bpa), by = (b / j->bpa) % j->bpa, bx = b % j->bpa;
    for (uint32_t z = 0; z < BD; z++)
      for (uint32_t y = 0; y < BD; y++)
        memcpy(cube + ((size_t)z * BD + y) * BD,
               j->vol + (((size_t)(bz * BD + z) * j->dim + by * BD + y) * j->dim + bx * BD), BD);
    if (c5d_brick_encode(&p, cube, BD, &j->blobs[b], &j->ns[b]) != 0) atomic_store(&j->fail, 1);
  }
  free(cube);
  return NULL;
}

static int pack_volume(const uint8_t *vol, uint32_t dim, const char *out, float q,
                       unsigned nthreads) {
  uint32_t bpa = dim / BD, nb = bpa * bpa * bpa;
  uint8_t **blobs = calloc(nb, sizeof *blobs);
  size_t *ns = calloc(nb, sizeof *ns);
  if (!blobs || !ns) return -1;
  enc_job j = {.vol = vol, .dim = dim, .bpa = bpa, .q = q, .blobs = blobs, .ns = ns};
  atomic_store(&j.next, 0);
  atomic_store(&j.fail, 0);
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  uint32_t nt = nthreads ? nthreads : (ncpu > 1 ? (uint32_t)ncpu : 1);
  if (nt > 16) nt = 16;
  pthread_t tids[16];
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (uint32_t t = 0; t < nt; t++) pthread_create(&tids[t], NULL, enc_worker, &j);
  for (uint32_t t = 0; t < nt; t++) pthread_join(tids[t], NULL);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (atomic_load(&j.fail)) {
    fprintf(stderr, "c5dpack: encode failed\n");
    return -1;
  }

  c5d_shard_writer *w = c5d_shard_create(out, dim, BD, 0, q);
  if (!w) {
    fprintf(stderr, "c5dpack: cannot create %s\n", out);
    return -1;
  }
  size_t total = 0;
  for (uint32_t b = 0; b < nb; b++) {
    if (c5d_shard_put(w, b, blobs[b], ns[b]) != 0) return -1;
    total += ns[b];
    free(blobs[b]);
  }
  if (c5d_shard_close(w) != 0) return -1;
  double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
  double raw = (double)dim * dim * dim;
  printf("c5dpack: %s: %u bricks, %.1f MB (%.1fx, q=%.2f), encode %.1fs (%.0f MB/s)\n", out, nb,
         (double)total / 1e6, raw / (double)total, (double)q, secs, raw / 1e6 / secs);
  free(blobs);
  free(ns);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) goto usage;
  float q = 2.0f;
  unsigned nthreads = 0;

  if (strcmp(argv[1], "raw") == 0 && argc >= 7) {
    uint32_t nx = (uint32_t)atoi(argv[3]), ny = (uint32_t)atoi(argv[4]),
             nz = (uint32_t)atoi(argv[5]);
    if (argc > 7) q = (float)atof(argv[7]);
    if (argc > 8) nthreads = (unsigned)atoi(argv[8]);
    if (nx != ny || ny != nz || nx % BD != 0) {
      fprintf(stderr, "c5dpack raw: dims must be equal multiples of 128\n");
      return EXIT_FAILURE;
    }
    int fd = open(argv[2], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || (size_t)st.st_size != (size_t)nx * ny * nz) {
      fprintf(stderr, "c5dpack: bad input file/size\n");
      return EXIT_FAILURE;
    }
    const uint8_t *vol = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (vol == MAP_FAILED) return EXIT_FAILURE;
    return pack_volume(vol, nx, argv[6], q, nthreads) == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (strcmp(argv[1], "band") == 0 && argc >= 7) {
    uint32_t Z = (uint32_t)atoi(argv[3]), Y = (uint32_t)atoi(argv[4]),
             X = (uint32_t)atoi(argv[5]);
    if (argc > 7) q = (float)atof(argv[7]);
    if (argc > 8) nthreads = (unsigned)atoi(argv[8]);
    r3d_shard_store store;
    if (r3d_shard_store_init(&store, argv[2], 68608, 43008, 43008) != 0) return EXIT_FAILURE;
    uint8_t *vol = malloc((size_t)1024 * 1024 * 1024);
    if (!vol) return EXIT_FAILURE;
    printf("c5dpack: decoding 3ddct shard %u/%u/%u...\n", Z, Y, X);
    if (r3d_shard_decode_region(&store, (uint64_t)Z * 1024, (uint64_t)Y * 1024,
                                (uint64_t)X * 1024, 1024, 1024, 1024, vol, 0) != 0)
      return EXIT_FAILURE;
    int rc = pack_volume(vol, 1024, argv[6], q, nthreads);
    free(vol);
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

usage:
  fprintf(stderr,
          "usage: c5dpack raw  <in.u8> <n> <n> <n> <out.c5s> [q] [threads]\n"
          "       c5dpack band <band_dir> <Z> <Y> <X> <out.c5s> [q] [threads]\n");
  return EXIT_FAILURE;
}
