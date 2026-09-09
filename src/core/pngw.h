/* Minimal 8-bit PNG writer (zlib deflate) — grayscale for rendseg and the
 * viewer's ink-map export, RGBA for --view-shot's composited slices. */
#pragma once
#include <stdint.h>

int r3d_png_write_gray(const char *path, const uint8_t *img, uint32_t w,
                       uint32_t h);

/* 8-bit RGBA (colour type 6); `img` is w*h*4, row-major, x fastest. */
int r3d_png_write_rgba(const char *path, const uint8_t *img, uint32_t w,
                       uint32_t h);
