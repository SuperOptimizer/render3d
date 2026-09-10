/* Reproducible synthetic input and public-vs-cached CPU block decode timing. */
#include "synthtree.h"
#include <time.h>
#include <volcomp.h>

static double seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--make-tree") == 0) {
    if (mkdir(argv[2], 0755) != 0) { perror(argv[2]); return 1; }
    uint32_t dim[3] = {512, 512, 512};
    return st_make_tree(argv[2], dim, 2, 0) != 0;
  }
  if (argc != 2) {
    fprintf(stderr, "usage: bench_blocks <shard.vcs> | --make-tree <new-directory>\n");
    return 2;
  }
  volcomp_shard_reader r;
  if (volcomp_shard_open(argv[1], &r) != 0) return 1;
  size_t n = 0;
  const uint8_t *blob = volcomp_shard_brick(&r, 0, &n);
  uint8_t *out = malloc(VOLCOMP_CHUNK_VOXELS);
  if (!blob || !out) {
    free(out);
    volcomp_shard_close_reader(&r);
    return 1;
  }
  const uint32_t iterations = 16384;
  uint64_t checksum = 0;
  double us[2];
  int rc = 0;
  for (uint32_t mode = 0; mode < 2 && !rc; mode++) {
    double start = seconds();
    for (uint32_t i = 0; i < iterations; i++) {
      uint32_t b = i % 512;
      bool ok = mode ? r3d_decode_block(blob, n, b / 64, b / 8 % 8, b % 8, out) == 0
                     : volcomp_decode_block(blob, n, b / 64, b / 8 % 8, b % 8,
                                            out, 4096) == VOLCOMP_OK;
      if (!ok) { rc = 1; break; }
      checksum += out[i % 4096];
    }
    us[mode] = (seconds() - start) * 1e6 / iterations;
  }
  double start = seconds();
  for (uint32_t i = 0; i < 32 && !rc; i++) {
    if (volcomp_decode(blob, n, out, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK) rc = 1;
    checksum += out[i];
  }
  double full_ms = (seconds() - start) * 1e3 / 32.0;
  if (!rc)
    printf("{\"chunk_bytes\":%zu,\"iterations\":%u,\"public_block_us\":%.3f,"
           "\"cached_block_us\":%.3f,\"full_chunk_ms\":%.3f,\"checksum\":%llu}\n",
           n, iterations, us[0], us[1], full_ms, (unsigned long long)checksum);
  free(out);
  volcomp_shard_close_reader(&r);
  return rc;
}
