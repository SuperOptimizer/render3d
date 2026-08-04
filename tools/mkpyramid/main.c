/* mkpyramid — build downsampled LOD levels L2..L5 (÷4..÷32, isotropic box
 * filter) of one shard band as flat .u8 files the clipmap renderer mmaps.
 *
 *   mkpyramid <band_dir> <band_Z> <out_dir>
 *
 * Level ℓ file: dims (43008>>ℓ)² xy × (1024>>ℓ) z, z relative to the band
 * start (band_Z*1024). Resumable: finished shards are recorded in
 * <out_dir>/.done and skipped on rerun. Missing shards stay zero. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "shard/shardio.h"

#define NX 43008ull
#define NY 43008ull
#define NZ 68608ull
#define SD 1024u
#define L_FIRST 2
#define L_LAST 5

static uint64_t now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

typedef struct level_file {
  uint8_t *map;
  size_t n;
  uint64_t lx, ly, lz;
} level_file;

static int level_open(level_file *lf, const char *dir, int l) {
  lf->lx = NX >> l;
  lf->ly = NY >> l;
  lf->lz = (uint64_t)SD >> l;
  lf->n = (size_t)(lf->lx * lf->ly * lf->lz);
  char path[600];
  snprintf(path, sizeof path, "%s/L%d.u8", dir, l);
  int fd = open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0 || ftruncate(fd, (off_t)lf->n) != 0) return -1;
  lf->map = mmap(NULL, lf->n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return lf->map == MAP_FAILED ? -1 : 0;
}

/* 2x downsample (2^3 box) src(d)^3 -> dst(d/2)^3, both tight cubes */
static void down2(const uint8_t *src, uint32_t d, uint8_t *dst) {
  uint32_t h = d / 2;
  for (uint32_t z = 0; z < h; z++)
    for (uint32_t y = 0; y < h; y++) {
      const uint8_t *s0 = src + ((size_t)(2 * z) * d + 2 * y) * d;
      const uint8_t *s1 = s0 + d;
      const uint8_t *s2 = src + ((size_t)(2 * z + 1) * d + 2 * y) * d;
      const uint8_t *s3 = s2 + d;
      uint8_t *o = dst + ((size_t)z * h + y) * h;
      for (uint32_t x = 0; x < h; x++) {
        uint32_t v = s0[2 * x] + s0[2 * x + 1] + s1[2 * x] + s1[2 * x + 1] + s2[2 * x] +
                     s2[2 * x + 1] + s3[2 * x] + s3[2 * x + 1];
        o[x] = (uint8_t)(v / 8);
      }
    }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: mkpyramid <band_dir> <band_Z> <out_dir>\n");
    return EXIT_FAILURE;
  }
  const char *band = argv[1], *out = argv[3];
  uint32_t bz = (uint32_t)atoi(argv[2]);
  mkdir(out, 0755);

  r3d_shard_store store;
  if (r3d_shard_store_init(&store, band, NZ, NY, NX) != 0) return EXIT_FAILURE;

  level_file lf[L_LAST + 1];
  for (int l = L_FIRST; l <= L_LAST; l++)
    if (level_open(&lf[l], out, l) != 0) {
      fprintf(stderr, "mkpyramid: cannot map level %d\n", l);
      return EXIT_FAILURE;
    }

  /* resumability */
  char done_path[600];
  snprintf(done_path, sizeof done_path, "%s/.done", out);
  char done[42][42];
  memset(done, 0, sizeof done);
  FILE *df = fopen(done_path, "r");
  if (df) {
    uint32_t y, x;
    while (fscanf(df, "%u %u", &y, &x) == 2)
      if (y < 42 && x < 42) done[y][x] = 1;
    fclose(df);
  }
  df = fopen(done_path, "a");

  uint8_t *full = malloc((size_t)SD * SD * SD); /* 1 GiB decode buffer */
  uint8_t *half = malloc((size_t)512 * 512 * 512);
  uint8_t *q[8];
  for (int l = 2; l <= 5; l++) q[l] = malloc((size_t)(SD >> l) * (SD >> l) * (SD >> l));
  if (!full || !half) return EXIT_FAILURE;

  uint32_t todo = 0, processed = 0, absent = 0;
  for (uint32_t y = 0; y < 42; y++)
    for (uint32_t x = 0; x < 42; x++)
      if (!done[y][x]) todo++;
  uint64_t t_start = now_ms();

  for (uint32_t sy = 0; sy < 42; sy++) {
    for (uint32_t sx = 0; sx < 42; sx++) {
      if (done[sy][sx]) continue;
      r3d_shard sh;
      if (r3d_shard_open(&store, bz, sy, sx, &sh) != 0) {
        absent++;
        processed++;
        continue; /* not downloaded (yet) or masked-out: leave zeros, no .done */
      }
      r3d_shard_close(&sh);

      r3d_shard_decode_region(&store, (uint64_t)bz * SD, (uint64_t)sy * SD, (uint64_t)sx * SD,
                              SD, SD, SD, full, 0);
      down2(full, SD, half);
      down2(half, 512, q[2]);
      down2(q[2], 256, q[3]);
      down2(q[3], 128, q[4]);
      down2(q[4], 64, q[5]);

      for (int l = L_FIRST; l <= L_LAST; l++) {
        uint32_t d = SD >> l;
        uint64_t Y0 = ((uint64_t)sy * SD) >> l, X0 = ((uint64_t)sx * SD) >> l;
        for (uint32_t z = 0; z < d; z++)
          for (uint32_t yy = 0; yy < d; yy++)
            memcpy(lf[l].map + ((uint64_t)z * lf[l].ly + Y0 + yy) * lf[l].lx + X0,
                   q[l] + ((size_t)z * d + yy) * d, d);
      }
      fprintf(df, "%u %u\n", sy, sx);
      fflush(df);
      processed++;
      if (processed % 10 == 0) {
        double el = (double)(now_ms() - t_start) / 1000.0;
        printf("%u/%u shards (%.0fs elapsed, %.1fs/shard, %u absent)\n", processed, todo, el,
               el / (double)processed, absent);
        fflush(stdout);
      }
    }
  }
  fclose(df);
  for (int l = L_FIRST; l <= L_LAST; l++) {
    msync(lf[l].map, lf[l].n, MS_SYNC);
    munmap(lf[l].map, lf[l].n);
  }
  printf("mkpyramid: done (%u processed, %u absent -> rerun after fetch to fill)\n", processed,
         absent);
  /* sidecar */
  char jp[600];
  snprintf(jp, sizeof jp, "%s/pyramid.json", out);
  FILE *jf = fopen(jp, "w");
  if (jf) {
    fprintf(jf,
            "{\"band_z\":%u,\"nx\":%llu,\"ny\":%llu,\"levels\":[2,3,4,5],"
            "\"band_depth\":%u}\n",
            bz, (unsigned long long)NX, (unsigned long long)NY, SD);
    fclose(jf);
  }
  return EXIT_SUCCESS;
}
