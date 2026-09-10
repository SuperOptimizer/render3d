#ifndef R3D_DEBLOCK_H
#define R3D_DEBLOCK_H
#include "block_region.h"
#include <stdbool.h>
#include <stdint.h>
/* Reads a rectangular part of one raw 16^3 block, packed X-fastest.
 * False means unavailable, not air. Never apply display filtering here. */
typedef bool (*r3d_deblock_read)(void *, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint8_t *);
/* Filter a decoded block using a two-voxel halo. Returns false if any needed
 * neighbor is unavailable; available seams are still filtered. No network or
 * allocation is performed here. dims and origin are level-local voxels. */
bool r3d_deblock16(uint8_t block[4096], float q, const uint32_t origin[3],
                   const uint32_t dims[3], r3d_deblock_read read, void *ctx);
/* Optional batched reader receives at most 26 halo rectangles (3904 bytes).
 * It must set available on every region, including unavailable neighbors. */
typedef void (*r3d_deblock_read_batch)(void *, r3d_block_region *, uint32_t);
bool r3d_deblock16_batch(uint8_t block[4096], float q, const uint32_t origin[3],
                         const uint32_t dims[3], r3d_deblock_read_batch read, void *ctx);
#endif
