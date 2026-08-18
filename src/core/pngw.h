/* Minimal 8-bit grayscale PNG writer (zlib deflate) — shared by rendseg
 * and the viewer's ink-map export. */
#pragma once
#include <stdint.h>

int r3d_png_write_gray(const char *path, const uint8_t *img, uint32_t w,
                       uint32_t h);
