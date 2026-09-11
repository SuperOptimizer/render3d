/* Encoder-only extension across holes. Fit a plane to valid samples, then
 * interpolate its residual along populated rows and between populated rows.
 * Valid input samples are never changed. No fill values are stored as geometry.
 */
static void fill_invalid(float input[4096], const uint8_t known[4096]) {
  double sx = 0, sy = 0, sz = 0, sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
  unsigned count = 0;
  for (unsigned i = 0; i < 4096; i++) {
    if (!known[i])
      continue;
    double x = (double)(i % 64), y = (double)(i / 64), z = input[i];
    count++;
    sx += x;
    sy += y;
    sz += z;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
    sxz += x * z;
    syz += y * z;
  }
  if (!count || count == 4096)
    return;
  double mx = sx / count, my = sy / count, mean = sz / count;
  double xx = sxx - sx * mx, xy = sxy - sx * my, yy = syy - sy * my;
  double xz = sxz - sx * mean, yz = syz - sy * mean, ax = 0, ay = 0;
  double det = xx * yy - xy * xy;
  if (det > 1e-10 * xx * yy && xx > 0 && yy > 0) {
    ax = (xz * yy - yz * xy) / det;
    ay = (yz * xx - xz * xy) / det;
  } else if (xx >= yy && xx > 0)
    ax = xz / xx;
  else if (yy > 0)
    ay = yz / yy;
  float residual[4096];
  unsigned rows[64], nr = 0;
  for (unsigned y = 0; y < 64; y++) {
    int prev = -1;
    for (unsigned x = 0; x < 64; x++) {
      unsigned i = y * 64 + x;
      if (!known[i])
        continue;
      float v =
          (float)((double)input[i] - (mean + ax * (x - mx) + ay * (y - my)));
      residual[i] = v;
      if (prev < 0) {
        rows[nr++] = y;
        for (unsigned k = 0; k < x; k++)
          residual[y * 64 + k] = v;
      } else {
        float a = residual[y * 64 + (unsigned)prev];
        for (unsigned k = (unsigned)prev + 1; k < x; k++)
          residual[y * 64 + k] = a + (v - a) * ((float)(k - (unsigned)prev) /
                                                (float)(x - (unsigned)prev));
      }
      prev = (int)x;
    }
    if (prev >= 0)
      for (unsigned k = (unsigned)prev + 1; k < 64; k++)
        residual[y * 64 + k] = residual[y * 64 + (unsigned)prev];
  }
  for (unsigned y = 0; y < rows[0]; y++)
    memcpy(residual + y * 64, residual + rows[0] * 64, 64 * sizeof(float));
  for (unsigned r = 1; r < nr; r++) {
    unsigned a = rows[r - 1], b = rows[r];
    for (unsigned y = a + 1; y < b; y++) {
      float t = (float)(y - a) / (float)(b - a);
      for (unsigned x = 0; x < 64; x++)
        residual[y * 64 + x] =
            residual[a * 64 + x] +
            (residual[b * 64 + x] - residual[a * 64 + x]) * t;
    }
  }
  for (unsigned y = rows[nr - 1] + 1; y < 64; y++)
    memcpy(residual + y * 64, residual + rows[nr - 1] * 64, 64 * sizeof(float));
  for (unsigned i = 0; i < 4096; i++)
    if (!known[i]) {
      double v = mean + ax * ((double)(i % 64) - mx) +
                 ay * ((double)(i / 64) - my) + residual[i];
      if (finite_number(v) && fabs(v) <= 1e30)
        input[i] = (float)v;
    }
}
