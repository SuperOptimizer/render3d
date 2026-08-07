#include "core/umbilicus.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void r3d_umbilicus_init(r3d_umbilicus *u) { memset(u, 0, sizeof *u); }

void r3d_umbilicus_free(r3d_umbilicus *u) {
  free(u->points);
  memset(u, 0, sizeof *u);
}

static size_t lower_bound(const r3d_umbilicus *u, double z) {
  size_t lo = 0, hi = u->count;
  while (lo < hi) {
    size_t m = lo + (hi - lo) / 2;
    if (u->points[m].z < z) lo = m + 1;
    else hi = m;
  }
  return lo;
}

const r3d_umbilicus_point *r3d_umbilicus_find(const r3d_umbilicus *u, double z) {
  if (!u) return NULL;
  size_t i = lower_bound(u, z);
  return i < u->count && u->points[i].z == z ? &u->points[i] : NULL;
}

int r3d_umbilicus_set(r3d_umbilicus *u, double x, double y, double z) {
  if (!u || !isfinite(x) || !isfinite(y) || !isfinite(z)) return -1;
  size_t i = lower_bound(u, z);
  if (i < u->count && u->points[i].z == z) {
    u->points[i] = (r3d_umbilicus_point){x, y, z};
    u->dirty = true;
    return 0;
  }
  if (u->count == u->capacity) {
    size_t cap = u->capacity ? u->capacity * 2 : 32;
    if (cap < u->capacity || cap > SIZE_MAX / sizeof *u->points) return -1;
    void *p = realloc(u->points, cap * sizeof *u->points);
    if (!p) return -1;
    u->points = p;
    u->capacity = cap;
  }
  memmove(&u->points[i + 1], &u->points[i], (u->count - i) * sizeof *u->points);
  u->points[i] = (r3d_umbilicus_point){x, y, z};
  u->count++;
  u->dirty = true;
  return 0;
}

bool r3d_umbilicus_remove(r3d_umbilicus *u, double z) {
  if (!u) return false;
  size_t i = lower_bound(u, z);
  if (i >= u->count || u->points[i].z != z) return false;
  memmove(&u->points[i], &u->points[i + 1], (u->count - i - 1) * sizeof *u->points);
  u->count--;
  u->dirty = true;
  return true;
}

static const char *skip_ws(const char *p, const char *end) {
  while (p < end && isspace((unsigned char)*p)) p++;
  return p;
}

/* Locate the closing delimiter while respecting JSON strings. */
static const char *matching(const char *p, const char *end, char open, char close) {
  int depth = 0;
  bool string = false, escape = false;
  for (; p < end; p++) {
    if (string) {
      if (escape) escape = false;
      else if (*p == '\\') escape = true;
      else if (*p == '"') string = false;
      continue;
    }
    if (*p == '"') string = true;
    else if (*p == open) depth++;
    else if (*p == close && --depth == 0) return p;
  }
  return NULL;
}

static const char *key_value(const char *begin, const char *end, const char *key) {
  size_t n = strlen(key);
  for (const char *p = begin; p + n + 2 <= end; p++) {
    if (*p != '"' || (size_t)(end - p) < n + 2 || memcmp(p + 1, key, n) != 0 ||
        p[n + 1] != '"')
      continue;
    p = skip_ws(p + n + 2, end);
    if (p >= end || *p != ':') return NULL;
    return skip_ws(p + 1, end);
  }
  return NULL;
}

static int object_number(const char *begin, const char *end, const char *key, double *out) {
  const char *p = key_value(begin, end, key);
  if (!p) return -1;
  errno = 0;
  char *q = NULL;
  double v = strtod(p, &q);
  if (q == p || q > end || errno == ERANGE || !isfinite(v)) return -1;
  *out = v;
  return 0;
}

static int parse_point(r3d_umbilicus *u, const char **cursor, const char *end) {
  const char *p = skip_ws(*cursor, end);
  double x, y, z;
  if (p < end && *p == '{') {
    const char *q = matching(p, end, '{', '}');
    if (!q || object_number(p + 1, q, "x", &x) != 0 ||
        object_number(p + 1, q, "y", &y) != 0 ||
        object_number(p + 1, q, "z", &z) != 0)
      return -1;
    *cursor = q + 1;
  } else if (p < end && *p == '[') {
    /* Volume Cartographer's positional form is [z,y,x]. */
    const char *q = matching(p, end, '[', ']');
    if (!q) return -1;
    char *n = NULL;
    errno = 0;
    z = strtod(skip_ws(p + 1, q), &n);
    if (!n || n <= p + 1 || errno == ERANGE) return -1;
    p = skip_ws(n, q);
    if (p >= q || *p++ != ',') return -1;
    y = strtod(skip_ws(p, q), &n);
    if (!n || n <= p || errno == ERANGE) return -1;
    p = skip_ws(n, q);
    if (p >= q || *p++ != ',') return -1;
    x = strtod(skip_ws(p, q), &n);
    if (!n || n <= p || errno == ERANGE || !isfinite(x) || !isfinite(y) || !isfinite(z))
      return -1;
    *cursor = q + 1;
  } else {
    return -1;
  }
  return r3d_umbilicus_set(u, x, y, z);
}

int r3d_umbilicus_load(r3d_umbilicus *u, const char *path) {
  if (!u || !path) return -1;
  FILE *f = fopen(path, "rb");
  if (!f) return errno == ENOENT ? 1 : -1;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
  long ln = ftell(f);
  if (ln < 0 || (unsigned long)ln > 64ul * 1024 * 1024 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }
  size_t n = (size_t)ln;
  char *json = malloc(n + 1);
  if (!json) { fclose(f); return -1; }
  bool ok = fread(json, 1, n, f) == n;
  if (fclose(f) != 0) ok = false;
  if (!ok) { free(json); return -1; }
  json[n] = 0;

  const char *end = json + n;
  const char *a = key_value(json, end, "control_points");
  if (!a) a = key_value(json, end, "points");
  if (!a) {
    const char *root = skip_ws(json, end);
    if (root < end && *root == '[') a = root;
  }
  if (!a || *a != '[') { free(json); return -1; }
  const char *ae = matching(a, end, '[', ']');
  if (!ae) { free(json); return -1; }

  r3d_umbilicus tmp;
  r3d_umbilicus_init(&tmp);
  const char *p = skip_ws(a + 1, ae);
  while (p < ae) {
    if (parse_point(&tmp, &p, ae) != 0) {
      r3d_umbilicus_free(&tmp);
      free(json);
      return -1;
    }
    p = skip_ws(p, ae);
    if (p < ae && *p == ',') p = skip_ws(p + 1, ae);
    else if (p < ae) {
      r3d_umbilicus_free(&tmp);
      free(json);
      return -1;
    }
  }
  tmp.dirty = false;
  r3d_umbilicus_free(u);
  *u = tmp;
  free(json);
  return 0;
}

static void json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (; s && *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
    else if (c == '\n') fputs("\\n", f);
    else if (c >= 0x20) fputc(c, f);
  }
  fputc('"', f);
}

static void write_points(FILE *f, const r3d_umbilicus *u, const char *key, bool comma) {
  fprintf(f, "  \"%s\": [", key);
  for (size_t i = 0; i < u->count; i++) {
    const r3d_umbilicus_point *p = &u->points[i];
    fprintf(f, "%s\n    {\"x\": %.17g, \"y\": %.17g, \"z\": %.17g}",
            i ? "," : "", p->x, p->y, p->z);
  }
  fprintf(f, "%s\n  ]%s\n", u->count ? "" : "", comma ? "," : "");
}

int r3d_umbilicus_save(r3d_umbilicus *u, const char *path, const char *source,
                       uint32_t nz, uint32_t ny, uint32_t nx) {
  if (!u || !path || !*path || strlen(path) > 900) return -1;
  char tmp[1024];
  int tn = snprintf(tmp, sizeof tmp, "%s.tmp.XXXXXX", path);
  if (tn < 0 || (size_t)tn >= sizeof tmp) return -1;
  int fd = mkstemp(tmp);
  if (fd < 0) return -1;
  FILE *f = fdopen(fd, "w");
  if (!f) { close(fd); unlink(tmp); return -1; }
  fputs("{\n  \"type\": \"render3d_umbilicus\",\n  \"version\": 1,\n"
        "  \"coordinate_order\": \"x,y,z\",\n  \"source\": ", f);
  json_string(f, source ? source : "");
  fprintf(f, ",\n  \"volume_shape_zyx\": [%u, %u, %u],\n", nz, ny, nx);
  write_points(f, u, "control_points", true);
  write_points(f, u, "points", false);
  fputs("}\n", f);
  bool failed = ferror(f) != 0;
  if (fflush(f) != 0) failed = true;
  if (fsync(fd) != 0) failed = true;
  if (fclose(f) != 0) failed = true;
  if (!failed && rename(tmp, path) == 0) {
    u->dirty = false;
    return 0;
  }
  unlink(tmp);
  return -1;
}
