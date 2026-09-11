/* Joint XYZ packets contain exactly one spatial patch. Predictor and rotation
 * parameters are stored float32; fitting and final verification use double.
 * No neighboring patch, mutable global state, or scheduling is involved. */
_Static_assert(SFC_XYZ_BOUND >=
                   128 + SE_TABLES_MAX_BYTES + 3 * SFC_BLOCK_BOUND + 4096 * 14,
               "joint packet bound");
static void pf(uint8_t *p, float f) {
  uint32_t v;
  memcpy(&v, &f, 4);
  p32(p, v);
}
static float gf(const uint8_t *p) {
  uint32_t v = u32(p);
  float f;
  memcpy(&f, &v, 4);
  return f;
}
typedef struct {
  float plane[3][4096], decoded[3][4096], original[4096][3];
  uint8_t valid[4096], other[4096], packet[SFC_XYZ_BOUND], best[SFC_XYZ_BOUND];
  sfc_candidates choices[3];
  sfc_encode_scratch encode;
  sfc_entropy_work entropy;
} sfc_xyz_work;
static void xyz_free(sfc_xyz_work *w) {
  if (w) {
    for (unsigned k = 0; k < 3; k++)
      free_choices(w->choices + k);
    free(w);
  }
}
static double xyz_predict(const float p[9], unsigned k, unsigned i) {
  return (double)p[k * 3] + (double)p[k * 3 + 1] * ((double)(i % 64) - 31.5) +
         (double)p[k * 3 + 2] * ((double)(i / 64) - 31.5);
}
static void xyz_fit(sfc_xyz_work *w, float p[9], float rot[9]) {
  double sx = 0, sy = 0, xx = 0, xy = 0, yy = 0, sz[3] = {0}, xz[3] = {0},
         yz[3] = {0};
  unsigned n = 0;
  for (unsigned i = 0; i < 4096; i++)
    if (w->valid[i]) {
      double x = (double)(i % 64) - 31.5, y = (double)(i / 64) - 31.5;
      n++;
      sx += x;
      sy += y;
      xx += x * x;
      xy += x * y;
      yy += y * y;
      for (unsigned k = 0; k < 3; k++) {
        double z = w->original[i][k];
        sz[k] += z;
        xz[k] += x * z;
        yz[k] += y * z;
      }
    }
  memset(p, 0, 9 * sizeof(float));
  memset(rot, 0, 9 * sizeof(float));
  for (unsigned k = 0; k < 3; k++)
    rot[k * 3 + k] = 1;
  if (!n)
    return;
  double mx = sx / n, my = sy / n;
  xx -= sx * mx;
  xy -= sx * my;
  yy -= sy * my;
  double det = xx * yy - xy * xy;
  for (unsigned k = 0; k < 3; k++) {
    double mean = sz[k] / n, a = 0, b = 0, u = xz[k] - sx * mean,
           v = yz[k] - sy * mean;
    if (xx > 0 && yy > 0 && det > 1e-10 * xx * yy) {
      a = (u * yy - v * xy) / det;
      b = (v * xx - u * xy) / det;
    } else if (xx >= yy && xx > 0)
      a = u / xx;
    else if (yy > 0)
      b = v / yy;
    p[k * 3] = (float)(mean - a * mx - b * my);
    p[k * 3 + 1] = (float)a;
    p[k * 3 + 2] = (float)b;
  }
  double cov[3][3] = {{0}}, r[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (unsigned i = 0; i < 4096; i++)
    if (w->valid[i]) {
      double d[3];
      for (unsigned k = 0; k < 3; k++)
        d[k] = (double)w->original[i][k] - xyz_predict(p, k, i);
      for (unsigned k = 0; k < 3; k++)
        for (unsigned j = 0; j < 3; j++)
          cov[k][j] += d[k] * d[j];
    }
  /* Bounded Jacobi eigensolve. Sorting/sign conventions need not be bit exact:
   * the selected matrix is stored in the packet and verified after rounding. */
  for (unsigned iter = 0; iter < 12; iter++) {
    unsigned a = 0, b = 1;
    if (fabs(cov[0][2]) > fabs(cov[a][b])) {
      a = 0;
      b = 2;
    }
    if (fabs(cov[1][2]) > fabs(cov[a][b])) {
      a = 1;
      b = 2;
    }
    if (fabs(cov[a][b]) < 1e-12 * (fabs(cov[a][a]) + fabs(cov[b][b]) + 1))
      break;
    double theta = .5 * atan2(2 * cov[a][b], cov[b][b] - cov[a][a]);
    double c = cos(theta), s = sin(theta), aa = cov[a][a], bb = cov[b][b],
           ab = cov[a][b];
    for (unsigned k = 0; k < 3; k++)
      if (k != a && k != b) {
        double u = cov[k][a], v = cov[k][b];
        cov[k][a] = cov[a][k] = c * u - s * v;
        cov[k][b] = cov[b][k] = s * u + c * v;
      }
    cov[a][a] = c * c * aa - 2 * c * s * ab + s * s * bb;
    cov[b][b] = s * s * aa + 2 * c * s * ab + c * c * bb;
    cov[a][b] = cov[b][a] = 0;
    for (unsigned k = 0; k < 3; k++) {
      double u = r[k][a], v = r[k][b];
      r[k][a] = c * u - s * v;
      r[k][b] = s * u + c * v;
    }
  }
  for (unsigned k = 0; k < 3; k++)
    for (unsigned j = 0; j < 3; j++)
      rot[k * 3 + j] = (float)r[k][j];
}
/* Maximum change between conforming reconstructions of a child packet. Raw
 * values/escapes are exact; treating escapes as transform output is
 * conservative. */
static int xyz_child_guard(const uint8_t *b, size_t n, const se_model *models,
                           const float out[4096], double *guard) {
  *guard = 0;
  if (b[5] != 1 && b[5] != 3)
    return 0;
  size_t at;
  if (block_prefix(b, n, &at, NULL))
    return SFC_INVALID;
  size_t end = at + u32(b + 24);
  if (end > n)
    return SFC_INVALID;
  float step = gf(b + 16), coef[4096] = {0};
  if (b[5] == 3) {
    if (entropy_decode(b + at, end - at, models, step, coef))
      return SFC_INVALID;
  } else {
    unsigned i = 0;
    while (i < 4096) {
      uint64_t run, v;
      if (varget(b, end, &at, &run) || run > 4096 - i)
        return SFC_INVALID;
      i += (unsigned)run;
      if (i == 4096)
        break;
      if (varget(b, end, &at, &v) || v > 8388608)
        return SFC_INVALID;
      coef[i++] = (float)se_unzigzag((uint32_t)v) * step;
    }
    if (at != end)
      return SFC_INVALID;
  }
  double l1 = 0, ulp = 0;
  for (unsigned i = 0; i < 4096; i++) {
    l1 += fabs((double)coef[i]);
    ulp = fmax(ulp, float_ulp(out[i]));
  }
  /* Residual rounding plus base/output rounding. The decoded magnitude alone
   * need not bound the pre-base residual; sum of coefficients does. */
  *guard = (u32(b + 20) ? 8e-7 : 2e-13) * l1 +
           2 * float_ulp((float)fmin(l1, FLT_MAX)) + 2 * ulp + 2e-34;
  return 0;
}
typedef struct {
  float decoded[3][4096], original[4096][3];
  uint8_t valid[4096], other[4096];
  se_model models[SE_NMODELS];
} sfc_xyz_decode;
static int xyz_unpack(const uint8_t *b, size_t n, sfc_xyz_decode *w) {
  if (!b || n < 224 || n > SFC_XYZ_BOUND || memcmp(b, "J641", 4) || b[4] > 1)
    return SFC_INVALID;
  for (unsigned i = 5; i < 8; i++)
    if (b[i])
      return SFC_INVALID;
  for (unsigned i = 16; i < 20; i++)
    if (b[i])
      return SFC_INVALID;
  for (unsigned i = 104; i < 112; i++)
    if (b[i])
      return SFC_INVALID;
  if (u32(b + 124))
    return SFC_INVALID;
  double e = gd(b + 8);
  if (!finite_number(e) || e <= 0)
    return SFC_INVALID;
  unsigned tn = u32(b + 20), ex = u32(b + 24);
  if (tn > SE_TABLES_MAX_BYTES || ex > 4096 || tn > n - 128)
    return SFC_INVALID;
  if (sfc_crc32(b + 128, tn) != u32(b + 28))
    return SFC_INVALID;
  se_model *models = w->models;
  se_cur cursor = {b + 128, b + 128 + tn};
  if (tn && (!se_tables_read(&cursor, models) || cursor.p != cursor.end))
    return SFC_INVALID;
  float p[9], r[9];
  for (unsigned k = 0; k < 9; k++) {
    p[k] = gf(b + 32 + 4 * k);
    r[k] = gf(b + 68 + 4 * k);
    if (!finite_number(p[k]) || !finite_number(r[k]))
      return SFC_INVALID;
    if (!b[4] && (p[k] != 0 || r[k] != 0))
      return SFC_INVALID;
  }
  if (b[4])
    for (unsigned k = 0; k < 3; k++)
      for (unsigned j = 0; j < 3; j++) {
        double dot = 0;
        for (unsigned a = 0; a < 3; a++)
          dot += (double)r[a * 3 + k] * r[a * 3 + j];
        if (fabs(dot - (k == j)) > 1e-5)
          return SFC_INVALID;
      }
  size_t at = 128 + tn;
  int used = 0;
  for (unsigned k = 0; k < 3; k++) {
    size_t len = u32(b + 112 + 4 * k);
    if (len < 32 || len > SFC_BLOCK_BOUND || len > n - at)
      return SFC_INVALID;
    const uint8_t *child = b + at;
    if (child[5] == 3) {
      size_t prefix;
      if (!tn || block_prefix(child, len, &prefix, NULL) || len - prefix < 24)
        return SFC_INVALID;
      for (unsigned j = 0; j < 16; j++)
        if (child[prefix + j])
          return SFC_INVALID;
      used = 1;
    }
    if (decode_block(child, len, SFC_F32, w->decoded[k], 4, 256,
                     k ? w->other : w->valid, tn ? models : NULL))
      return SFC_INVALID;
    if (k && memcmp(w->valid, w->other, 4096))
      return SFC_INVALID;
    at += len;
  }
  if ((tn != 0) != used)
    return SFC_INVALID;
  for (unsigned i = 0; i < 4096; i++)
    for (unsigned k = 0; k < 3; k++) {
      double v = w->decoded[k][i];
      if (b[4]) {
        v = xyz_predict(p, k, i);
        for (unsigned j = 0; j < 3; j++)
          v += (double)r[k * 3 + j] * w->decoded[j][i];
      }
      if (w->valid[i] && (!finite_number(v) || fabs(v) > FLT_MAX))
        return SFC_INVALID;
      w->original[i][k] = w->valid[i] ? (float)v : -1;
    }
  unsigned prev = 0;
  for (unsigned j = 0; j < ex; j++) {
    uint64_t delta;
    if (varget(b, n, &at, &delta) || delta > 4095 - prev || (j && !delta) ||
        n - at < 12)
      return SFC_INVALID;
    prev += (unsigned)delta;
    if (!w->valid[prev])
      return SFC_INVALID;
    for (unsigned k = 0; k < 3; k++)
      w->original[prev][k] = gf(b + at + 4 * k);
    at += 12;
  }
  if (at != n)
    return SFC_INVALID;
  for (unsigned i = 0; i < 4096; i++)
    if (w->valid[i])
      for (unsigned k = 0; k < 3; k++)
        if (!finite_number(w->original[i][k]) ||
            (k == 2 && w->original[i][k] <= 0))
          return SFC_INVALID;
  return 0;
}
static int xyz_encode(const void *src, size_t ps, size_t rs,
                      const uint8_t *mask, double error, sfc_xyz_work *w,
                      size_t *size) {
  if (!src || !strides(ps, rs, 12, 64, 64) || !finite_number(error) ||
      error <= 0)
    return SFC_INVALID;
  for (unsigned i = 0; i < 4096; i++) {
    w->valid[i] = !mask || mask[i];
    memcpy(w->original[i], (const uint8_t *)src + (i / 64) * rs + (i % 64) * ps,
           12);
    if (w->valid[i])
      for (unsigned k = 0; k < 3; k++)
        if (!finite_number(w->original[i][k]) ||
            (k == 2 && w->original[i][k] <= 0))
          return SFC_INVALID;
  }
  float predictor[9], rotation[9];
  xyz_fit(w, predictor, rotation);
  size_t best = SIZE_MAX;
  double best_square = DBL_MAX;
  for (unsigned mode = 0; mode < 2; mode++) {
    int usable = 1;
    for (unsigned i = 0; i < 4096; i++)
      for (unsigned k = 0; k < 3; k++) {
        double v = w->original[i][k];
        if (mode) {
          v = 0;
          for (unsigned a = 0; a < 3; a++)
            v += (double)rotation[a * 3 + k] *
                 ((double)w->original[i][a] - xyz_predict(predictor, a, i));
        }
        if (!w->valid[i])
          v = 0;
        if (!finite_number(v) || fabs(v) > 1e30)
          usable = 0;
        w->plane[k][i] = (float)fmax(-FLT_MAX, fmin(FLT_MAX, v));
      }
    if (mode && !usable)
      continue;
    sfc_candidates *choices[3];
    for (unsigned k = 0; k < 3; k++) {
      sfc_channel c = {0};
      c.dtype = SFC_F32;
      c.flags = SFC_COORDINATE;
      c.tolerance = mode ? error : error / sqrt(3.0);
      if (!mode && k == 2)
        strcpy(c.name, "z");
      uint8_t *packet;
      size_t n;
      int rc = encode_candidates(&c, w->plane[k], 4, 256, w->valid, &packet, &n,
                                 &w->encode, mode ? w->choices + k : NULL);
      if (rc)
        return rc;
      if (!mode) {
        w->choices[k].count = 0;
        w->choices[k].preferred = 0;
        rc = save_choice(w->choices + k, packet, n, 0);
        if (rc)
          return rc;
      }
      choices[k] = w->choices + k;
    }
    int rc = select_entropy(choices, 3, &w->entropy);
    if (rc)
      return rc;
    sfc_entropy_work *ent = &w->entropy;
    se_model *models = ent->models;
    se_cur cursor = {ent->tables, ent->tables + ent->table_size};
    if (ent->table_size &&
        (!se_tables_read(&cursor, models) || cursor.p != cursor.end))
      return SFC_INVALID;
    uint8_t *b = w->packet;
    memset(b, 0, 128);
    memcpy(b, "J641", 4);
    b[4] = (uint8_t)mode;
    pd(b + 8, error);
    p32(b + 20, (uint32_t)ent->table_size);
    p32(b + 28, sfc_crc32(ent->tables, ent->table_size));
    if (mode)
      for (unsigned k = 0; k < 9; k++) {
        pf(b + 32 + 4 * k, predictor[k]);
        pf(b + 68 + 4 * k, rotation[k]);
      }
    size_t at = 128;
    memcpy(b + at, ent->tables, ent->table_size);
    at += ent->table_size;
    double guards[3];
    for (unsigned k = 0; k < 3; k++) {
      uint8_t *child = ent->best[k];
      size_t len = ent->sizes[k];
      if (decode_block(child, len, SFC_F32, w->decoded[k], 4, 256, NULL,
                       models) ||
          xyz_child_guard(child, len, models, w->decoded[k], guards + k))
        return SFC_INVALID;
      p32(b + 112 + 4 * k, (uint32_t)len);
      memcpy(b + at, child, len);
      at += len;
    }
    unsigned prev = 0, ex = 0;
    double square = 0;
    for (unsigned i = 0; i < 4096; i++)
      if (w->valid[i]) {
        double err2 = 0, guard2 = 0;
        int escape = 0;
        for (unsigned k = 0; k < 3; k++) {
          double v = w->decoded[k][i], g = guards[k];
          if (mode) {
            v = xyz_predict(predictor, k, i);
            g = 0;
            double magnitude = fabs((double)predictor[k * 3]) +
                               31.5 * (fabs((double)predictor[k * 3 + 1]) +
                                       fabs((double)predictor[k * 3 + 2]));
            for (unsigned j = 0; j < 3; j++) {
              v += (double)rotation[k * 3 + j] * w->decoded[j][i];
              g += fabs((double)rotation[k * 3 + j]) * guards[j];
              magnitude += fabs((double)rotation[k * 3 + j] * w->decoded[j][i]);
            }
            g += 2 * float_ulp((float)fmax(-FLT_MAX, fmin(FLT_MAX, v))) +
                 1e-12 * (magnitude + 1);
          }
          if (!finite_number(v) || fabs(v) > FLT_MAX) {
            escape = 1;
            continue;
          }
          float out = (float)v;
          double d = (double)out - w->original[i][k];
          err2 += d * d;
          guard2 += g * g;
          if (k == 2 && out <= 0)
            escape = 1;
        }
        if (escape || sqrt(err2) + sqrt(guard2) > error) {
          if (at + 14 > SFC_XYZ_BOUND)
            return SFC_LIMIT;
          at += varput(b + at, i - prev);
          prev = i;
          for (unsigned k = 0; k < 3; k++)
            pf(b + at + 4 * k, w->original[i][k]);
          at += 12;
          ex++;
        } else
          square += err2;
      }
    p32(b + 24, ex);
    if (at < best || (at == best && square < best_square)) {
      best = at;
      best_square = square;
      memcpy(w->best, b, at);
    }
  }
  *size = best;
  return best == SIZE_MAX ? SFC_INVALID : 0;
}
int sfc_encode_xyz_block(const void *src, size_t ps, size_t rs,
                         const uint8_t *mask, double error, uint8_t **encoded,
                         size_t *size) {
  if (!encoded || !size)
    return SFC_INVALID;
  *encoded = NULL;
  *size = 0;
  sfc_xyz_work *w = calloc(1, sizeof *w);
  if (!w)
    return SFC_NOMEM;
  size_t n;
  int rc = xyz_encode(src, ps, rs, mask, error, w, &n);
  if (!rc) {
    *encoded = malloc(n);
    if (!*encoded)
      rc = SFC_NOMEM;
    else {
      memcpy(*encoded, w->best, n);
      *size = n;
    }
  }
  xyz_free(w);
  return rc;
}
int sfc_decode_xyz_block(const uint8_t *b, size_t n, void *dst, size_t ps,
                         size_t rs, uint8_t *mask) {
  if (!dst || !strides(ps, rs, 12, 64, 64))
    return SFC_INVALID;
  sfc_xyz_decode *w = malloc(sizeof *w);
  if (!w)
    return SFC_NOMEM;
  int rc = xyz_unpack(b, n, w);
  if (!rc) {
    for (unsigned i = 0; i < 4096; i++)
      memcpy((uint8_t *)dst + (i / 64) * rs + (i % 64) * ps, w->original[i],
             12);
    if (mask)
      memcpy(mask, w->valid, 4096);
  }
  free(w);
  return rc;
}
