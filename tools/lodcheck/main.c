/* Compare c5d bricks against the decoded Zarr shard they were transcoded
 * from. This validates coordinate mapping, both container CRCs, zero sentinels
 * and codec fidelity without requiring the renderer or a window. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "brick.h"
#include "shard.h"
#include "shard/shardio.h"

#define BD 128u
#define CD 16u
#define BPA 8u
#define CPA 8u

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int decode_zarr_brick(const r3d_shard *zs, uint32_t bi, uint8_t *out) {
  uint32_t bz = bi / (BPA * BPA), by = (bi / BPA) % BPA, bx = bi % BPA;
  uint8_t chunk[CD * CD * CD];
  for (uint32_t cz = 0; cz < CPA; cz++)
    for (uint32_t cy = 0; cy < CPA; cy++)
      for (uint32_t cx = 0; cx < CPA; cx++) {
        if (r3d_shard_chunk_decode(zs, bz * CPA + cz, by * CPA + cy, bx * CPA + cx,
                                   chunk) != 0)
          return -1;
        for (uint32_t z = 0; z < CD; z++)
          for (uint32_t y = 0; y < CD; y++)
            memcpy(out + (((size_t)(cz * CD + z) * BD + cy * CD + y) * BD + cx * CD),
                   chunk + (z * CD + y) * CD, CD);
      }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "usage: lodcheck <zarr-shard> <c5d-shard> [brick-stride]\n");
    return 2;
  }
  uint32_t stride = argc == 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 1u;
  if (!stride) return 2;
  r3d_shard zs;
  c5d_shard_reader cs;
  if (r3d_shard_open_path(argv[1], &zs) != R3D_SHARD_OK) {
    fprintf(stderr, "lodcheck: invalid Zarr shard %s\n", argv[1]);
    return 1;
  }
  if (c5d_shard_open(argv[2], &cs) != 0 || cs.foot.shard_dim != 1024u ||
      cs.foot.brick_dim != BD || cs.foot.nbricks != BPA * BPA * BPA) {
    fprintf(stderr, "lodcheck: invalid c5d shard %s\n", argv[2]);
    r3d_shard_close(&zs);
    return 1;
  }
  uint8_t *ref = malloc((size_t)BD * BD * BD);
  uint8_t *got = malloc((size_t)BD * BD * BD);
  if (!ref || !got) return 1;
  uint64_t nvox = 0, abs_sum = 0, sq_sum = 0;
  uint32_t checked = 0, maxerr = 0;
  double decode_seconds = 0.0;
  int failed = 0;
  for (uint32_t b = 0; b < cs.foot.nbricks; b += stride) {
    if (decode_zarr_brick(&zs, b, ref) != 0) {
      failed = 1;
      break;
    }
    size_t bn = 0;
    const uint8_t *blob = c5d_shard_brick(&cs, b, &bn);
    if (blob) {
      double started = now_seconds();
      int decode_rc = c5d_brick_decode_par(blob, bn, got, BD, 4);
      decode_seconds += now_seconds() - started;
      if (decode_rc != 0) {
        failed = 1;
        break;
      }
    } else if (c5d_shard_brick_is_zero(&cs, b)) {
      memset(got, 0, (size_t)BD * BD * BD);
    } else {
      fprintf(stderr, "lodcheck: brick %u is missing/corrupt\n", b);
      failed = 1;
      break;
    }
    for (size_t i = 0; i < (size_t)BD * BD * BD; i++) {
      uint32_t e = ref[i] > got[i] ? ref[i] - got[i] : got[i] - ref[i];
      abs_sum += e;
      sq_sum += (uint64_t)e * e;
      if (e > maxerr) maxerr = e;
    }
    nvox += (size_t)BD * BD * BD;
    checked++;
  }
  double mae = nvox ? (double)abs_sum / (double)nvox : 0.0;
  double mse = nvox ? (double)sq_sum / (double)nvox : 0.0;
  double psnr = mse > 0.0 ? 10.0 * log10(255.0 * 255.0 / mse) : HUGE_VAL;
  printf("lodcheck: L%u %u bricks, %llu voxels: MAE %.3f, PSNR %.2f dB, max %u\n",
         cs.foot.lod_level, checked, (unsigned long long)nvox, mae, psnr, maxerr);
  printf("lodcheck: c5d CPU decode %.2f ms/checked brick (4 threads)\n",
         checked ? decode_seconds * 1000.0 / (double)checked : 0.0);
  free(ref);
  free(got);
  c5d_shard_close_reader(&cs);
  r3d_shard_close(&zs);
  if (failed || !checked || psnr < 30.0) {
    fprintf(stderr, "lodcheck: FAIL\n");
    return 1;
  }
  return 0;
}
