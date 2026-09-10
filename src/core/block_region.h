#ifndef R3D_BLOCK_REGION_H
#define R3D_BLOCK_REGION_H
#include <stdbool.h>
#include <stdint.h>
/* Packed X-fastest raw voxel rectangle; available distinguishes missing data
 * from known air. The caller owns the output storage. */
typedef struct r3d_block_region {
  uint32_t x, y, z, nx, ny, nz;
  uint8_t *out;
  bool available;
} r3d_block_region;
#endif
