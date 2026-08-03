#include "core/screenshot.h"

#include <stdio.h>
#include <stdlib.h>

int r3d_screenshot_ppm(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  fprintf(f, "P6\n%u %u\n255\n", w, h);
  uint8_t *row = malloc((size_t)w * 3);
  if (!row) {
    fclose(f);
    return -1;
  }
  int rc = 0;
  for (uint32_t y = 0; y < h && rc == 0; y++) {
    const uint8_t *src = rgba + (size_t)y * w * 4;
    for (uint32_t x = 0; x < w; x++) {
      row[x * 3 + 0] = src[x * 4 + 0];
      row[x * 3 + 1] = src[x * 4 + 1];
      row[x * 3 + 2] = src[x * 4 + 2];
    }
    if (fwrite(row, 1, (size_t)w * 3, f) != (size_t)w * 3) rc = -1;
  }
  free(row);
  if (fclose(f) != 0) rc = -1;
  return rc;
}
