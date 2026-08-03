#include "core/volume.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int r3d_volume_open(r3d_volume *v, const char *path, uint32_t nx, uint32_t ny, uint32_t nz) {
  memset(v, 0, sizeof *v);
  if (!path || nx == 0 || ny == 0 || nz == 0) return -1;

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "volume: cannot open %s\n", path);
    return -1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return -1;
  }
  size_t want = (size_t)nx * ny * nz;
  if ((size_t)st.st_size != want) {
    fprintf(stderr, "volume: %s is %lld bytes, expected %zu (%ux%ux%u)\n", path,
            (long long)st.st_size, want, nx, ny, nz);
    close(fd);
    return -1;
  }
  void *map = mmap(NULL, want, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd); /* mapping keeps its own reference */
  if (map == MAP_FAILED) {
    fprintf(stderr, "volume: mmap failed for %s\n", path);
    return -1;
  }
  madvise(map, want, MADV_WILLNEED);
  v->voxels = map;
  v->nx = nx;
  v->ny = ny;
  v->nz = nz;
  v->nbytes = want;
  return 0;
}

void r3d_volume_close(r3d_volume *v) {
  if (v->voxels) munmap((void *)v->voxels, v->nbytes);
  memset(v, 0, sizeof *v);
}
