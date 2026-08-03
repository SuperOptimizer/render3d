/* assemble — concatenate a contiguous grid of 128^3 raw u8 bricks (c5d corpus
 * naming: <sample>_z<Z>_y<Y>_x<X>.u8) into one flat volume.u8 (spec/volume.md).
 *
 *   assemble <corpus_dir> <out.u8> [sample]
 *
 * Scans the directory, takes the bounding box of all matching brick coords,
 * requires every brick in the box to be present and exactly 128^3 bytes. */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define BD 128u /* brick dim */
#define BRICK_BYTES ((size_t)BD * BD * BD)

typedef struct coord {
  int z, y, x;
} coord;

static int parse_name(const char *name, const char *sample, coord *c) {
  size_t plen = strlen(sample);
  if (strncmp(name, sample, plen) != 0) return -1;
  int z, y, x;
  char tail[8];
  /* "%4s" guards against e.g. ".json": require exactly ".u8" */
  if (sscanf(name + plen, "_z%d_y%d_x%d%4s", &z, &y, &x, tail) != 4) return -1;
  if (strcmp(tail, ".u8") != 0) return -1;
  c->z = z;
  c->y = y;
  c->x = x;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "usage: assemble <corpus_dir> <out.u8> [sample=PHercParis4]\n");
    return EXIT_FAILURE;
  }
  const char *dirpath = argv[1], *outpath = argv[2];
  const char *sample = argc == 4 ? argv[3] : "PHercParis4";

  DIR *dir = opendir(dirpath);
  if (!dir) {
    fprintf(stderr, "assemble: cannot open %s\n", dirpath);
    return EXIT_FAILURE;
  }
  coord *coords = NULL;
  size_t ncoords = 0, cap = 0;
  coord lo = {INT_MAX, INT_MAX, INT_MAX}, hi = {INT_MIN, INT_MIN, INT_MIN};
  struct dirent *de;
  while ((de = readdir(dir))) {
    coord c;
    if (parse_name(de->d_name, sample, &c) != 0) continue;
    if (ncoords == cap) {
      cap = cap ? cap * 2 : 64;
      coords = realloc(coords, cap * sizeof *coords);
      if (!coords) return EXIT_FAILURE;
    }
    coords[ncoords++] = c;
    if (c.z < lo.z) lo.z = c.z;
    if (c.y < lo.y) lo.y = c.y;
    if (c.x < lo.x) lo.x = c.x;
    if (c.z > hi.z) hi.z = c.z;
    if (c.y > hi.y) hi.y = c.y;
    if (c.x > hi.x) hi.x = c.x;
  }
  closedir(dir);
  if (ncoords == 0) {
    fprintf(stderr, "assemble: no %s_z*_y*_x*.u8 bricks in %s\n", sample, dirpath);
    return EXIT_FAILURE;
  }
  uint32_t gz = (uint32_t)(hi.z - lo.z + 1), gy = (uint32_t)(hi.y - lo.y + 1),
           gx = (uint32_t)(hi.x - lo.x + 1);
  size_t expect = (size_t)gz * gy * gx;
  if (ncoords != expect) {
    fprintf(stderr, "assemble: bounding box %ux%ux%u needs %zu bricks, found %zu\n", gz, gy, gx,
            expect, ncoords);
    return EXIT_FAILURE;
  }
  uint32_t nx = gx * BD, ny = gy * BD, nz = gz * BD;
  size_t total = (size_t)nx * ny * nz;
  printf("assemble: %zu bricks -> %ux%ux%u (%.2f GiB) from %s (z%d-%d y%d-%d x%d-%d)\n", ncoords,
         nx, ny, nz, (double)total / (1u << 30), sample, lo.z, hi.z, lo.y, hi.y, lo.x, hi.x);
  free(coords);

  int ofd = open(outpath, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (ofd < 0 || ftruncate(ofd, (off_t)total) != 0) {
    fprintf(stderr, "assemble: cannot create %s: %s\n", outpath, strerror(errno));
    return EXIT_FAILURE;
  }
  uint8_t *out = mmap(NULL, total, PROT_WRITE, MAP_SHARED, ofd, 0);
  if (out == MAP_FAILED) {
    fprintf(stderr, "assemble: mmap out failed\n");
    return EXIT_FAILURE;
  }

  uint8_t *brick = malloc(BRICK_BYTES);
  if (!brick) return EXIT_FAILURE;
  size_t done = 0;
  for (int bz = lo.z; bz <= hi.z; bz++) {
    for (int by = lo.y; by <= hi.y; by++) {
      for (int bx = lo.x; bx <= hi.x; bx++) {
        char path[PATH_MAX];
        snprintf(path, sizeof path, "%s/%s_z%d_y%d_x%d.u8", dirpath, sample, bz, by, bx);
        int fd = open(path, O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0 || (size_t)st.st_size != BRICK_BYTES) {
          fprintf(stderr, "assemble: %s missing or not %zu bytes\n", path, BRICK_BYTES);
          return EXIT_FAILURE;
        }
        for (size_t off = 0; off < BRICK_BYTES;) {
          ssize_t r = read(fd, brick + off, BRICK_BYTES - off);
          if (r <= 0) {
            fprintf(stderr, "assemble: short read on %s\n", path);
            return EXIT_FAILURE;
          }
          off += (size_t)r;
        }
        close(fd);
        size_t ox = (size_t)(bx - lo.x) * BD, oy = (size_t)(by - lo.y) * BD,
               oz = (size_t)(bz - lo.z) * BD;
        for (uint32_t z = 0; z < BD; z++)
          for (uint32_t y = 0; y < BD; y++)
            memcpy(out + ((oz + z) * ny + (oy + y)) * nx + ox, brick + ((size_t)z * BD + y) * BD,
                   BD);
        done++;
        if (done % 64 == 0) {
          printf("  %zu/%zu bricks\r", done, expect);
          fflush(stdout);
        }
      }
    }
  }
  free(brick);
  if (msync(out, total, MS_SYNC) != 0 || munmap(out, total) != 0 || close(ofd) != 0) {
    fprintf(stderr, "assemble: flush failed: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }
  printf("assemble: wrote %s (%zu bytes)\n", outpath, total);
  return EXIT_SUCCESS;
}
