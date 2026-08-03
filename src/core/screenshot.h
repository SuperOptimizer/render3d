/* Zero-dependency PPM (P6) screenshot writer. */
#ifndef R3D_SCREENSHOT_H
#define R3D_SCREENSHOT_H

#include <stdint.h>

/* Writes RGBA8 pixels (alpha dropped) as binary PPM. Returns 0 on success. */
int r3d_screenshot_ppm(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h);

#endif /* R3D_SCREENSHOT_H */
