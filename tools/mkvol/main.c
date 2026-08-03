/* mkvol — synthetic test volumes for renderer development/tests.
 *   mkvol <out.u8> <n> <sphere|shells|gyroid>
 * Cube volumes n^3, u8, x-fastest (spec/volume.md). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t gen(const char *kind, float x, float y, float z) {
  /* inputs in [-1,1] */
  float r = sqrtf(x * x + y * y + z * z);
  if (strcmp(kind, "sphere") == 0) {
    return r < 0.8f ? (uint8_t)(255.0f * (1.0f - r / 0.8f)) : 0;
  }
  if (strcmp(kind, "shells") == 0) {
    /* nested spherical shells, scroll-wrap-ish density */
    float s = 0.5f + 0.5f * sinf(r * 40.0f);
    float fade = r < 0.9f ? 1.0f : 0.0f;
    return (uint8_t)(255.0f * s * s * fade);
  }
  if (strcmp(kind, "gyroid") == 0) {
    float k = 9.42477796f; /* 3 periods over [-1,1] */
    float g = sinf(k * x) * cosf(k * y) + sinf(k * y) * cosf(k * z) + sinf(k * z) * cosf(k * x);
    float d = 1.0f - fabsf(g); /* dense near the gyroid surface */
    return d > 0.7f ? (uint8_t)(255.0f * (d - 0.7f) / 0.3f) : 0;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: mkvol <out.u8> <n> <sphere|shells|gyroid>\n");
    return EXIT_FAILURE;
  }
  const char *outpath = argv[1], *kind = argv[3];
  long n = strtol(argv[2], NULL, 10);
  if (n < 8 || n > 4096) {
    fprintf(stderr, "mkvol: n out of range\n");
    return EXIT_FAILURE;
  }
  FILE *f = fopen(outpath, "wb");
  if (!f) {
    fprintf(stderr, "mkvol: cannot open %s\n", outpath);
    return EXIT_FAILURE;
  }
  uint8_t *row = malloc((size_t)n);
  if (!row) return EXIT_FAILURE;
  float inv = 2.0f / (float)(n - 1);
  for (long z = 0; z < n; z++) {
    float fz = (float)z * inv - 1.0f;
    for (long y = 0; y < n; y++) {
      float fy = (float)y * inv - 1.0f;
      for (long x = 0; x < n; x++) row[x] = gen(kind, (float)x * inv - 1.0f, fy, fz);
      if (fwrite(row, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "mkvol: write failed\n");
        return EXIT_FAILURE;
      }
    }
  }
  free(row);
  if (fclose(f) != 0) return EXIT_FAILURE;
  printf("mkvol: wrote %s (%ld^3 %s)\n", outpath, n, kind);
  return EXIT_SUCCESS;
}
