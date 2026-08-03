/* Transfer function: piecewise-linear control points -> 256-entry RGBA8 LUT.
 * RGB = emission color, A = extinction. */
#ifndef R3D_TRANSFER_H
#define R3D_TRANSFER_H

#include <stdint.h>

#define R3D_TF_MAX_POINTS 16

typedef struct r3d_tf_point {
  uint8_t value;      /* input voxel value 0..255 */
  uint8_t rgba[4];
} r3d_tf_point;

typedef struct r3d_tf {
  r3d_tf_point pts[R3D_TF_MAX_POINTS]; /* sorted by value; first at 0, last at 255 */
  uint32_t npts;
} r3d_tf;

void r3d_tf_build(const r3d_tf *tf, uint8_t lut[256][4]);

/* Presets: 0 = grayscale ramp, 1 = "scroll" (papyrus emphasis), 2 = bone/ink
 * high-pass. Returns number of presets if idx out of range. */
uint32_t r3d_tf_preset(uint32_t idx, r3d_tf *out);

#endif /* R3D_TRANSFER_H */
