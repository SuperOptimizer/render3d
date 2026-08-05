/* Cut a flat .u8 sub-volume out of a local dct3d shard band (threaded chunk
 * decode; missing data zero-fills). Feeds the tiled-slab renderer with real
 * cross sections wider than one shard.
 *
 * Usage: bandcut <band_dir> <out.u8> <z0> <nz> <y0> <ny> <x0> <nx>
 * (world voxel coords of the 43008^3 PHercParis3 export; z-blocks of 16) */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "shard/shardio.h"

int main(int argc, char **argv) {
  if (argc != 9) {
    fprintf(stderr, "usage: %s <band_dir> <out.u8> <z0> <nz> <y0> <ny> <x0> <nx>\n", argv[0]);
    return 2;
  }
  uint64_t z0 = strtoull(argv[3], NULL, 0), y0 = strtoull(argv[5], NULL, 0),
           x0 = strtoull(argv[7], NULL, 0);
  uint32_t nz = (uint32_t)strtoul(argv[4], NULL, 0), ny = (uint32_t)strtoul(argv[6], NULL, 0),
           nx = (uint32_t)strtoul(argv[8], NULL, 0);
  r3d_shard_store st;
  if (r3d_shard_store_init(&st, argv[1], 43008, 43008, 43008) != 0) {
    fprintf(stderr, "bandcut: bad band dir %s\n", argv[1]);
    return 1;
  }
  FILE *f = fopen(argv[2], "wb");
  if (!f) {
    fprintf(stderr, "bandcut: cannot write %s\n", argv[2]);
    return 1;
  }
  const uint32_t ZB = 16; /* chunk-aligned z batches bound the buffer */
  uint8_t *buf = malloc((size_t)nx * ny * ZB);
  if (!buf) return 1;
  for (uint32_t z = 0; z < nz; z += ZB) {
    uint32_t dz = z + ZB > nz ? nz - z : ZB;
    if (r3d_shard_decode_region(&st, z0 + z, y0, x0, dz, ny, nx, buf, 0) != 0) {
      fprintf(stderr, "bandcut: decode failed at z=%" PRIu64 "\n", z0 + z);
      return 1;
    }
    if (fwrite(buf, 1, (size_t)nx * ny * dz, f) != (size_t)nx * ny * dz) return 1;
    printf("bandcut: %u/%u slices\n", z + dz, nz);
    fflush(stdout);
  }
  free(buf);
  fclose(f);
  printf("bandcut: wrote %s (%ux%ux%u)\n", argv[2], nx, ny, nz);
  return 0;
}
