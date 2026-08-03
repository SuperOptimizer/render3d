/* Raw u8 volume access: mmap'd flat file, x-fastest ([z][y][x], spec/volume.md).
 * Carries the c5d spatial-hierarchy metadata from day 1 so later brick/LOD
 * streaming does not change this interface. */
#ifndef R3D_VOLUME_H
#define R3D_VOLUME_H

#include <stddef.h>
#include <stdint.h>

#define R3D_CHUNK_DIM 16u
#define R3D_BRICK_DIM 128u
#define R3D_SHARD_DIM 1024u

typedef struct r3d_volume {
  const uint8_t *voxels; /* mmap'd, read-only, nx*ny*nz bytes */
  uint32_t nx, ny, nz;
  size_t nbytes;
  float voxel_um; /* informational; 0 if unknown */
} r3d_volume;

/* mmap `path` and validate its size against dims. Returns 0 on success. */
int r3d_volume_open(r3d_volume *v, const char *path, uint32_t nx, uint32_t ny, uint32_t nz);
void r3d_volume_close(r3d_volume *v);

static inline uint8_t r3d_volume_at(const r3d_volume *v, uint32_t x, uint32_t y, uint32_t z) {
  return v->voxels[((size_t)z * v->ny + y) * v->nx + x];
}

#endif /* R3D_VOLUME_H */
