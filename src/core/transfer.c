#include "core/transfer.h"

#include <string.h>

void r3d_tf_build(const r3d_tf *tf, uint8_t lut[256][4]) {
  for (uint32_t i = 0; i < 256; i++) {
    /* find surrounding control points */
    const r3d_tf_point *lo = &tf->pts[0], *hi = &tf->pts[tf->npts - 1];
    for (uint32_t k = 0; k + 1 < tf->npts; k++) {
      if (tf->pts[k].value <= i && i <= tf->pts[k + 1].value) {
        lo = &tf->pts[k];
        hi = &tf->pts[k + 1];
        break;
      }
    }
    uint32_t span = (uint32_t)(hi->value - lo->value);
    uint32_t t_num = span ? i - lo->value : 0, t_den = span ? span : 1;
    for (int c = 0; c < 4; c++) {
      uint32_t a = lo->rgba[c], b = hi->rgba[c];
      lut[i][c] = (uint8_t)((a * (t_den - t_num) + b * t_num) / t_den);
    }
  }
}

uint32_t r3d_tf_preset(uint32_t idx, r3d_tf *out) {
  switch (idx) {
  case 0: /* grayscale identity ramp */
    out->npts = 2;
    out->pts[0] = (r3d_tf_point){0, {0, 0, 0, 0}};
    out->pts[1] = (r3d_tf_point){255, {255, 255, 255, 255}};
    return 0;
  case 1: /* scroll: air transparent, papyrus warm, dense ink/mineral bright */
    out->npts = 5;
    out->pts[0] = (r3d_tf_point){0, {0, 0, 0, 0}};
    out->pts[1] = (r3d_tf_point){40, {0, 0, 0, 0}};
    out->pts[2] = (r3d_tf_point){90, {150, 110, 70, 40}};
    out->pts[3] = (r3d_tf_point){160, {230, 190, 130, 160}};
    out->pts[4] = (r3d_tf_point){255, {255, 250, 240, 255}};
    return 1;
  case 2: /* high-pass: only dense structures */
    out->npts = 4;
    out->pts[0] = (r3d_tf_point){0, {0, 0, 0, 0}};
    out->pts[1] = (r3d_tf_point){120, {0, 0, 0, 0}};
    out->pts[2] = (r3d_tf_point){180, {200, 200, 255, 120}};
    out->pts[3] = (r3d_tf_point){255, {255, 255, 255, 255}};
    return 2;
  default:
    return 3; /* preset count */
  }
}
