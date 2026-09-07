#include "core/annot.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char *const r3d_annot_class_name[R3D_ANNOT_NCLASS] = {
    "untouched", "in-face", "out-face", "ignore", "erase"};

const char *const r3d_annot_layer_file[R3D_ANNOT_NLAYER] = {
    "ct", "faces_in", "faces_out", "ignore", "source", "rv_class", "pred_in", "pred_out"};

/* ===================== minimal JSON scanning ============================= */

static const char *skip_ws(const char *p, const char *end) {
  while (p < end && isspace((unsigned char)*p)) p++;
  return p;
}

/* Closing delimiter for the bracket at p, respecting strings. */
static const char *matching(const char *p, const char *end, char open, char close) {
  int depth = 0;
  bool in_string = false, escape = false;
  for (; p < end; p++) {
    if (in_string) {
      if (escape) escape = false;
      else if (*p == '\\') escape = true;
      else if (*p == '"') in_string = false;
      continue;
    }
    if (*p == '"') in_string = true;
    else if (*p == open) depth++;
    else if (*p == close && --depth == 0) return p;
  }
  return NULL;
}

/* End (one past) of the JSON value starting at p. */
static const char *value_end(const char *p, const char *end) {
  if (p >= end) return NULL;
  if (*p == '{') { const char *q = matching(p, end, '{', '}'); return q ? q + 1 : NULL; }
  if (*p == '[') { const char *q = matching(p, end, '[', ']'); return q ? q + 1 : NULL; }
  if (*p == '"') {
    bool escape = false;
    for (const char *q = p + 1; q < end; q++) {
      if (escape) escape = false;
      else if (*q == '\\') escape = true;
      else if (*q == '"') return q + 1;
    }
    return NULL;
  }
  const char *q = p;
  while (q < end && *q != ',' && *q != ']' && *q != '}' && !isspace((unsigned char)*q)) q++;
  return q > p ? q : NULL;
}

/* Value of `key` inside the object body [begin,end). Only keys at this
 * object's top level are considered. */
static const char *key_value(const char *begin, const char *end, const char *key) {
  size_t n = strlen(key);
  const char *p = skip_ws(begin, end);
  while (p < end) {
    if (*p != '"') { p++; continue; }
    const char *ks = p + 1;
    const char *ke = value_end(p, end); /* one past the closing quote */
    if (!ke) return NULL;
    const char *c = skip_ws(ke, end);
    if (c >= end || *c != ':') { p = ke; continue; }
    const char *v = skip_ws(c + 1, end);
    const char *ve = value_end(v, end);
    if (!ve) return NULL;
    if ((size_t)(ke - 1 - ks) == n && memcmp(ks, key, n) == 0) return v;
    p = ve;
  }
  return NULL;
}

static int json_number(const char *v, const char *end, double *out) {
  if (!v || v >= end) return -1;
  errno = 0;
  char *q = NULL;
  double d = strtod(v, &q);
  if (q == v || q > end || errno == ERANGE || !isfinite(d)) return -1;
  *out = d;
  return 0;
}

/* Fixed-length integer array, e.g. "dims_zyx": [Z,Y,X]. */
static int obj_ints(const char *b, const char *e, const char *key, int64_t *out, int want) {
  const char *v = key_value(b, e, key);
  if (!v || v >= e || *v != '[') return -1;
  const char *ae = matching(v, e, '[', ']');
  if (!ae) return -1;
  const char *p = skip_ws(v + 1, ae);
  for (int i = 0; i < want; i++) {
    double d;
    if (json_number(p, ae, &d) != 0) return -1;
    if (d < -9.0e15 || d > 9.0e15) return -1;
    out[i] = (int64_t)llround(d);
    const char *ve = value_end(p, ae);
    if (!ve) return -1;
    p = skip_ws(ve, ae);
    if (i + 1 < want) {
      if (p >= ae || *p != ',') return -1;
      p = skip_ws(p + 1, ae);
    }
  }
  return 0;
}

static int obj_string(const char *b, const char *e, const char *key, char *out, size_t cap) {
  const char *v = key_value(b, e, key);
  if (!v || v >= e || *v != '"') return -1;
  const char *ve = value_end(v, e);
  if (!ve) return -1;
  size_t n = (size_t)(ve - v) - 2u;
  if (n >= cap) return -1;
  memcpy(out, v + 1, n); /* no escape handling: packet paths are plain */
  out[n] = 0;
  return 0;
}

static bool obj_bool(const char *b, const char *e, const char *key, bool dflt) {
  const char *v = key_value(b, e, key);
  if (!v || v >= e) return dflt;
  if ((size_t)(e - v) >= 4u && memcmp(v, "true", 4) == 0) return true;
  if ((size_t)(e - v) >= 5u && memcmp(v, "false", 5) == 0) return false;
  return dflt;
}

static char *read_text(const char *path, size_t *out_size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long n = ftell(f);
  if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
  char *s = malloc((size_t)n + 1u);
  if (!s) { fclose(f); return NULL; }
  bool ok = fread(s, 1, (size_t)n, f) == (size_t)n;
  fclose(f);
  if (!ok) { free(s); return NULL; }
  s[n] = 0;
  if (out_size) *out_size = (size_t)n;
  return s;
}

/* ===================== atomic publish ==================================== */

static int publish(const char *dir, const char *name, const void *data, size_t size) {
  char tmp[1200], dst[1200];
  if (snprintf(tmp, sizeof tmp, "%s/.%s.tmp.%ld", dir, name, (long)getpid()) >= (int)sizeof tmp)
    return -1;
  if (snprintf(dst, sizeof dst, "%s/%s", dir, name) >= (int)sizeof dst) return -1;
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  bool ok = size == 0 || fwrite(data, 1, size, f) == size;
  if (ok) ok = fflush(f) == 0;
  if (ok) ok = fsync(fileno(f)) == 0;
  if (fclose(f) != 0) ok = false;
  if (!ok || rename(tmp, dst) != 0) {
    unlink(tmp);
    return -1;
  }
  return 0;
}

/* ===================== manifest ========================================== */

static bool is_file(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}
static bool is_dir(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static void dirname_of(const char *path, char *out, size_t cap) {
  const char *slash = strrchr(path, '/');
  if (!slash) { snprintf(out, cap, "."); return; }
  size_t n = (size_t)(slash - path);
  if (n == 0) n = 1; /* "/foo" -> "/" */
  if (n >= cap) n = cap - 1;
  memcpy(out, path, n);
  out[n] = 0;
}

static int manifest_single(r3d_annot_manifest *m, const char *dir) {
  m->ent = calloc(1, sizeof *m->ent);
  if (!m->ent) return -1;
  m->count = 1;
  snprintf(m->ent[0].path, sizeof m->ent[0].path, "%s", dir);
  const char *base = strrchr(dir, '/');
  snprintf(m->ent[0].name, sizeof m->ent[0].name, "%s", base ? base + 1 : dir);
  return 0;
}

int r3d_annot_manifest_load(r3d_annot_manifest *m, const char *path) {
  if (!m || !path) return -1;
  memset(m, 0, sizeof *m);

  char json_path[1024];
  if (is_dir(path)) {
    if (snprintf(json_path, sizeof json_path, "%s/packet.json", path) >= (int)sizeof json_path)
      return -1;
    if (!is_file(json_path)) {
      char meta[1100];
      if (snprintf(meta, sizeof meta, "%s/meta.json", path) >= (int)sizeof meta) return -1;
      if (!is_file(meta)) return -1;
      return manifest_single(m, path);
    }
  } else {
    if (snprintf(json_path, sizeof json_path, "%s", path) >= (int)sizeof json_path) return -1;
    if (!is_file(json_path)) return -1;
  }

  size_t n = 0;
  char *text = read_text(json_path, &n);
  if (!text) return -1;
  const char *end = text + n;
  const char *arr = key_value(text, end, "packets");
  if (!arr || *arr != '[') { free(text); return -1; }
  const char *ae = matching(arr, end, '[', ']');
  if (!ae) { free(text); return -1; }

  char root[1024];
  dirname_of(json_path, root, sizeof root);
  snprintf(m->path, sizeof m->path, "%s", json_path);

  uint32_t cap = 16;
  m->ent = calloc(cap, sizeof *m->ent);
  if (!m->ent) { free(text); return -1; }

  const char *p = skip_ws(arr + 1, ae);
  while (p < ae && *p == '{') {
    const char *oe = matching(p, ae, '{', '}');
    if (!oe) break;
    char rel[600];
    if (obj_string(p + 1, oe, "path", rel, sizeof rel) == 0) {
      if (m->count == cap) {
        uint32_t nc = cap * 2u;
        void *q = realloc(m->ent, (size_t)nc * sizeof *m->ent);
        if (!q) break;
        m->ent = q;
        memset((r3d_annot_entry *)q + cap, 0, (size_t)cap * sizeof *m->ent);
        cap = nc;
      }
      r3d_annot_entry *e = &m->ent[m->count];
      snprintf(e->name, sizeof e->name, "%s", rel);
      if (rel[0] == '/')
        snprintf(e->path, sizeof e->path, "%s", rel);
      else
        snprintf(e->path, sizeof e->path, "%s/%s", root, rel);
      int64_t v[3];
      if (obj_ints(p + 1, oe, "origin_zyx", v, 3) == 0)
        for (int k = 0; k < 3; k++) e->origin[k] = v[k];
      if (obj_ints(p + 1, oe, "dims_zyx", v, 3) == 0)
        for (int k = 0; k < 3; k++)
          e->dims[k] = v[k] > 0 && v[k] < 0x7fffffff ? (uint32_t)v[k] : 0u;
      m->count++;
    }
    p = skip_ws(oe + 1, ae);
    if (p < ae && *p == ',') p = skip_ws(p + 1, ae);
  }
  free(text);
  if (m->count == 0) { free(m->ent); m->ent = NULL; return -1; }
  return 0;
}

void r3d_annot_manifest_free(r3d_annot_manifest *m) {
  if (!m) return;
  free(m->ent);
  memset(m, 0, sizeof *m);
}

/* ===================== packet I/O ======================================== */

static uint8_t *read_volume(const char *dir, const char *base, size_t want, bool *missing) {
  char path[1300];
  if (snprintf(path, sizeof path, "%s/%s.u8", dir, base) >= (int)sizeof path) return NULL;
  if (missing) *missing = false;
  struct stat st;
  if (stat(path, &st) != 0) {
    if (missing) *missing = true;
    return NULL;
  }
  if ((size_t)st.st_size != want) {
    fprintf(stderr, "annot: %s is %lld bytes, expected %zu\n", path, (long long)st.st_size, want);
    return NULL;
  }
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  uint8_t *buf = malloc(want);
  if (!buf) { fclose(f); return NULL; }
  bool ok = fread(buf, 1, want, f) == want;
  fclose(f);
  if (!ok) { free(buf); return NULL; }
  return buf;
}

void r3d_annot_recount(r3d_annot_packet *p) {
  memset(p->counts, 0, sizeof p->counts);
  size_t n = r3d_annot_voxels(p);
  for (size_t i = 0; i < n; i++) {
    uint8_t c = p->correction[i];
    p->counts[c < R3D_ANNOT_NCLASS ? c : 0u]++;
  }
}

int r3d_annot_packet_load(r3d_annot_packet *p, const char *dir) {
  if (!p || !dir) return -1;
  memset(p, 0, sizeof *p);
  if (snprintf(p->dir, sizeof p->dir, "%s", dir) >= (int)sizeof p->dir) return -1;

  char meta_path[1100];
  if (snprintf(meta_path, sizeof meta_path, "%s/meta.json", dir) >= (int)sizeof meta_path)
    return -1;
  size_t mn = 0;
  p->meta_json = read_text(meta_path, &mn);
  if (!p->meta_json) {
    fprintf(stderr, "annot: cannot read %s\n", meta_path);
    return -1;
  }
  const char *mb = p->meta_json, *me = p->meta_json + mn;
  int64_t dims[3];
  if (obj_ints(mb, me, "dims_zyx", dims, 3) != 0 || dims[0] <= 0 || dims[1] <= 0 ||
      dims[2] <= 0 || dims[0] > 4096 || dims[1] > 4096 || dims[2] > 4096) {
    fprintf(stderr, "annot: %s has no usable dims_zyx\n", meta_path);
    free(p->meta_json);
    p->meta_json = NULL;
    return -1;
  }
  p->nz = (uint32_t)dims[0];
  p->ny = (uint32_t)dims[1];
  p->nx = (uint32_t)dims[2];
  if (obj_ints(mb, me, "origin_zyx", p->origin, 3) != 0)
    p->origin[0] = p->origin[1] = p->origin[2] = 0;
  if (json_number(key_value(mb, me, "voxel_um"), me, &p->voxel_um) != 0 || p->voxel_um <= 0.0)
    p->voxel_um = 0.0;
  p->done = obj_bool(mb, me, "done", false);

  size_t want = r3d_annot_voxels(p);
  for (int i = 0; i < R3D_ANNOT_NLAYER; i++) {
    bool missing = false;
    p->layer[i] = read_volume(dir, r3d_annot_layer_file[i], want, &missing);
    if (!p->layer[i] && !missing) { r3d_annot_packet_free(p); return -1; }
    if (!p->layer[i] && i == R3D_ANNOT_L_CT) {
      fprintf(stderr, "annot: %s/ct.u8 is required\n", dir);
      r3d_annot_packet_free(p);
      return -1;
    }
  }

  bool missing = false;
  p->correction = read_volume(dir, "correction", want, &missing);
  if (!p->correction) {
    if (!missing) { r3d_annot_packet_free(p); return -1; }
    p->correction = calloc(want, 1);
    if (!p->correction) { r3d_annot_packet_free(p); return -1; }
  } else {
    for (size_t i = 0; i < want; i++)
      if (p->correction[i] >= R3D_ANNOT_NCLASS) p->correction[i] = 0;
  }
  r3d_annot_recount(p);
  return 0;
}

void r3d_annot_packet_free(r3d_annot_packet *p) {
  if (!p) return;
  for (int i = 0; i < R3D_ANNOT_NLAYER; i++) free(p->layer[i]);
  free(p->correction);
  free(p->meta_json);
  memset(p, 0, sizeof *p);
}

int r3d_annot_correction_save(r3d_annot_packet *p) {
  if (!p || !p->correction) return -1;
  if (publish(p->dir, "correction.u8", p->correction, r3d_annot_voxels(p)) != 0) return -1;
  p->dirty = false;
  return 0;
}

int r3d_annot_set_done(r3d_annot_packet *p, bool done) {
  if (!p || !p->meta_json) return -1;
  const char *src = p->meta_json;
  size_t n = strlen(src);
  const char *v = key_value(src, src + n, "done");
  char *out = NULL;
  const char *lit = done ? "true" : "false";
  size_t lit_n = strlen(lit);
  if (v) { /* rewrite the value token in place */
    const char *ve = value_end(v, src + n);
    if (!ve) return -1;
    size_t head = (size_t)(v - src), tail = n - (size_t)(ve - src);
    out = malloc(head + lit_n + tail + 1u);
    if (!out) return -1;
    memcpy(out, src, head);
    memcpy(out + head, lit, lit_n);
    memcpy(out + head + lit_n, ve, tail);
    out[head + lit_n + tail] = 0;
  } else { /* insert right after the opening brace */
    const char *brace = memchr(src, '{', n);
    if (!brace) return -1;
    size_t head = (size_t)(brace - src) + 1u, tail = n - head;
    const char *rest = skip_ws(brace + 1, src + n);
    bool empty = rest >= src + n || *rest == '}';
    size_t ins_n = lit_n + (empty ? 8u : 9u); /* "done": X  (+ comma) */
    out = malloc(head + ins_n + tail + 1u);
    if (!out) return -1;
    memcpy(out, src, head);
    int w = snprintf(out + head, ins_n + 1u, "\"done\": %s%s", lit, empty ? "" : ",");
    if (w < 0) { free(out); return -1; }
    memcpy(out + head + (size_t)w, src + head, tail);
    out[head + (size_t)w + tail] = 0;
  }
  int rc = publish(p->dir, "meta.json", out, strlen(out));
  if (rc == 0) {
    free(p->meta_json);
    p->meta_json = out;
    p->done = done;
  } else {
    free(out);
  }
  return rc;
}

/* ===================== undo journal ====================================== */

typedef struct undo_entry {
  uint32_t *idx;
  uint8_t *old;
  uint32_t count, cap;
} undo_entry;

struct r3d_annot_undo {
  undo_entry *e;
  uint32_t levels, count, head; /* ring: newest is (head-1) mod levels */
  bool grouping;                /* an interactive drag is open */
};

r3d_annot_undo *r3d_annot_undo_create(uint32_t levels) {
  if (levels == 0) levels = 1;
  r3d_annot_undo *u = calloc(1, sizeof *u);
  if (!u) return NULL;
  u->e = calloc(levels, sizeof *u->e);
  if (!u->e) { free(u); return NULL; }
  u->levels = levels;
  return u;
}

void r3d_annot_undo_destroy(r3d_annot_undo *u) {
  if (!u) return;
  for (uint32_t i = 0; i < u->levels; i++) {
    free(u->e[i].idx);
    free(u->e[i].old);
  }
  free(u->e);
  free(u);
}

uint32_t r3d_annot_undo_depth(const r3d_annot_undo *u) { return u ? u->count : 0u; }

static undo_entry *undo_begin(r3d_annot_undo *u) {
  if (!u) return NULL;
  undo_entry *e = &u->e[u->head];
  if (!u->grouping) e->count = 0;
  return e;
}

static void undo_commit(r3d_annot_undo *u, const undo_entry *e) {
  if (!u || !e || u->grouping) return;
  if (e->count == 0) return; /* a no-op stroke never consumes a level */
  u->head = (u->head + 1u) % u->levels;
  if (u->count < u->levels) u->count++;
}

void r3d_annot_undo_begin(r3d_annot_undo *u) {
  if (!u || u->grouping) return;
  u->e[u->head].count = 0;
  u->grouping = true;
}

void r3d_annot_undo_end(r3d_annot_undo *u) {
  if (!u || !u->grouping) return;
  u->grouping = false;
  undo_commit(u, &u->e[u->head]);
}

static bool undo_record(undo_entry *e, uint32_t idx, uint8_t old) {
  if (!e) return true;
  if (e->count == e->cap) {
    uint32_t nc = e->cap ? e->cap * 2u : 4096u;
    uint32_t *ni = realloc(e->idx, (size_t)nc * sizeof *ni);
    if (!ni) return false;
    e->idx = ni;
    uint8_t *no = realloc(e->old, nc);
    if (!no) return false;
    e->old = no;
    e->cap = nc;
  }
  e->idx[e->count] = idx;
  e->old[e->count] = old;
  e->count++;
  return true;
}

int64_t r3d_annot_undo_pop(r3d_annot_packet *p, r3d_annot_undo *u) {
  if (!p || !u) return 0;
  if (u->grouping) r3d_annot_undo_end(u); /* an interrupted drag still undoes */
  if (u->count == 0) return 0;
  u->head = (u->head + u->levels - 1u) % u->levels;
  u->count--;
  undo_entry *e = &u->e[u->head];
  size_t n = r3d_annot_voxels(p);
  int64_t restored = 0;
  for (uint32_t i = e->count; i-- > 0;) {
    if (e->idx[i] >= n) continue;
    uint8_t cur = p->correction[e->idx[i]];
    if (cur == e->old[i]) continue;
    p->counts[cur]--;
    p->counts[e->old[i]]++;
    p->correction[e->idx[i]] = e->old[i];
    restored++;
  }
  e->count = 0;
  if (restored) p->dirty = true;
  return restored;
}

/* ===================== stroke application ================================ */

static bool paint_voxel(r3d_annot_packet *p, undo_entry *e, int64_t x, int64_t y, int64_t z,
                        uint8_t cls, int64_t *changed) {
  if (x < 0 || y < 0 || z < 0 || x >= (int64_t)p->nx || y >= (int64_t)p->ny ||
      z >= (int64_t)p->nz)
    return true;
  size_t idx = ((size_t)z * p->ny + (size_t)y) * p->nx + (size_t)x;
  uint8_t old = p->correction[idx];
  if (old == cls) return true;
  if (!undo_record(e, (uint32_t)idx, old)) return false;
  p->counts[old]--;
  p->counts[cls]++;
  p->correction[idx] = cls;
  (*changed)++;
  return true;
}

/* Squared distance from (px,py) to the segment (ax,ay)-(bx,by). */
static double seg_dist2(double px, double py, double ax, double ay, double bx, double by) {
  double dx = bx - ax, dy = by - ay;
  double len2 = dx * dx + dy * dy;
  double t = 0.0;
  if (len2 > 0.0) {
    t = ((px - ax) * dx + (py - ay) * dy) / len2;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
  }
  double qx = ax + t * dx - px, qy = ay + t * dy - py;
  return qx * qx + qy * qy;
}

/* Capsule stamp: every voxel within `radius` of the segment, on every slice of
 * the depth window (a cylinder through z, not a sphere). */
static bool stamp_capsule(r3d_annot_packet *p, undo_entry *e, int32_t ax, int32_t ay,
                          int32_t bx, int32_t by, int32_t radius, int32_t z0, int32_t z1,
                          uint8_t cls, int64_t *changed) {
  double r = radius < 0 ? 0.0 : (double)radius;
  int64_t lo_x = (int64_t)(ax < bx ? ax : bx) - radius;
  int64_t hi_x = (int64_t)(ax > bx ? ax : bx) + radius;
  int64_t lo_y = (int64_t)(ay < by ? ay : by) - radius;
  int64_t hi_y = (int64_t)(ay > by ? ay : by) + radius;
  if (lo_x < 0) lo_x = 0;
  if (lo_y < 0) lo_y = 0;
  if (hi_x >= (int64_t)p->nx) hi_x = (int64_t)p->nx - 1;
  if (hi_y >= (int64_t)p->ny) hi_y = (int64_t)p->ny - 1;
  double r2 = r * r + 1e-9;
  for (int64_t y = lo_y; y <= hi_y; y++)
    for (int64_t x = lo_x; x <= hi_x; x++) {
      if (seg_dist2((double)x, (double)y, (double)ax, (double)ay, (double)bx, (double)by) > r2)
        continue;
      for (int64_t z = z0; z <= z1; z++)
        if (!paint_voxel(p, e, x, y, z, cls, changed)) return false;
    }
  return true;
}

static bool stamp_line(r3d_annot_packet *p, undo_entry *e, int32_t ax, int32_t ay, int32_t bx,
                       int32_t by, int32_t z0, int32_t z1, uint8_t cls, int64_t *changed) {
  int64_t x = ax, y = ay;
  int64_t dx = (int64_t)bx - ax, dy = (int64_t)by - ay;
  int64_t sx = dx < 0 ? -1 : 1, sy = dy < 0 ? -1 : 1;
  if (dx < 0) dx = -dx;
  if (dy < 0) dy = -dy;
  int64_t err = dx - dy;
  for (;;) {
    for (int64_t z = z0; z <= z1; z++)
      if (!paint_voxel(p, e, x, y, z, cls, changed)) return false;
    if (x == bx && y == by) break;
    int64_t e2 = err * 2;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 < dx) { err += dx; y += sy; }
  }
  return true;
}

int64_t r3d_annot_apply(r3d_annot_packet *p, const r3d_annot_stroke *s, r3d_annot_undo *undo) {
  if (!p || !p->correction || !s || s->cls >= R3D_ANNOT_NCLASS) return -1;
  if (s->op != R3D_ANNOT_OP_BRUSH && s->op != R3D_ANNOT_OP_LINE) return -1;
  if (s->npts == 0 || !s->pts) return -1;
  int32_t depth = s->depth < 0 ? -s->depth : s->depth;
  int64_t z0 = (int64_t)s->z - depth, z1 = (int64_t)s->z + depth;
  if (z1 < 0 || z0 >= (int64_t)p->nz) return 0;
  if (z0 < 0) z0 = 0;
  if (z1 >= (int64_t)p->nz) z1 = (int64_t)p->nz - 1;

  undo_entry *e = undo_begin(undo);
  int64_t changed = 0;
  bool ok = true;
  uint32_t segs = s->npts == 1u ? 1u : s->npts - 1u;
  for (uint32_t i = 0; i < segs && ok; i++) {
    int32_t ax = s->pts[2u * i], ay = s->pts[2u * i + 1u];
    int32_t bx = s->npts == 1u ? ax : s->pts[2u * (i + 1u)];
    int32_t by = s->npts == 1u ? ay : s->pts[2u * (i + 1u) + 1u];
    if (s->op == R3D_ANNOT_OP_BRUSH)
      ok = stamp_capsule(p, e, ax, ay, bx, by, s->radius < 0 ? 0 : s->radius, (int32_t)z0,
                         (int32_t)z1, s->cls, &changed);
    else
      ok = stamp_line(p, e, ax, ay, bx, by, (int32_t)z0, (int32_t)z1, s->cls, &changed);
  }
  undo_commit(undo, e);
  if (changed) p->dirty = true;
  return ok ? changed : -1;
}

int r3d_annot_run_strokes(r3d_annot_packet *p, const r3d_annot_stroke *s, uint32_t count,
                          r3d_annot_undo *undo) {
  if (!p || (!s && count)) return -1;
  for (uint32_t i = 0; i < count; i++) {
    if (s[i].op == R3D_ANNOT_OP_UNDO) {
      r3d_annot_undo_pop(p, undo);
      continue;
    }
    if (r3d_annot_apply(p, &s[i], undo) < 0) return -1;
  }
  return 0;
}

/* ===================== stroke scripts ==================================== */

void r3d_annot_strokes_free(r3d_annot_stroke *s, uint32_t count) {
  if (!s) return;
  for (uint32_t i = 0; i < count; i++) free((void *)s[i].pts);
  free(s);
}

static int parse_points(const char *b, const char *e, int32_t **out, uint32_t *n) {
  const char *v = key_value(b, e, "points");
  *out = NULL;
  *n = 0;
  if (!v || *v != '[') return -1;
  const char *ae = matching(v, e, '[', ']');
  if (!ae) return -1;
  uint32_t cap = 16, count = 0;
  int32_t *pts = malloc((size_t)cap * 2u * sizeof *pts);
  if (!pts) return -1;
  const char *p = skip_ws(v + 1, ae);
  while (p < ae && *p == '[') {
    const char *pe = matching(p, ae, '[', ']');
    if (!pe) break;
    const char *q = skip_ws(p + 1, pe);
    double dx, dy;
    const char *qe = value_end(q, pe);
    if (!qe || json_number(q, pe, &dx) != 0) break;
    q = skip_ws(qe, pe);
    if (q >= pe || *q != ',') break;
    q = skip_ws(q + 1, pe);
    if (json_number(q, pe, &dy) != 0) break;
    int64_t xy[2] = {(int64_t)llround(dx), (int64_t)llround(dy)};
    if (count == cap) {
      uint32_t nc = cap * 2u;
      int32_t *np = realloc(pts, (size_t)nc * 2u * sizeof *np);
      if (!np) break;
      pts = np;
      cap = nc;
    }
    pts[2u * count] = (int32_t)xy[0];
    pts[2u * count + 1u] = (int32_t)xy[1];
    count++;
    p = skip_ws(pe + 1, ae);
    if (p < ae && *p == ',') p = skip_ws(p + 1, ae);
  }
  if (count == 0) { free(pts); return -1; }
  *out = pts;
  *n = count;
  return 0;
}

int r3d_annot_strokes_load(const char *path, r3d_annot_stroke **out, uint32_t *count) {
  if (!path || !out || !count) return -1;
  *out = NULL;
  *count = 0;
  size_t n = 0;
  char *text = read_text(path, &n);
  if (!text) return -1;
  const char *end = text + n;
  const char *arr = key_value(text, end, "strokes");
  if (!arr) arr = skip_ws(text, end); /* a bare top-level array is accepted */
  if (!arr || *arr != '[') { free(text); return -1; }
  const char *ae = matching(arr, end, '[', ']');
  if (!ae) { free(text); return -1; }

  uint32_t cap = 16, cn = 0;
  r3d_annot_stroke *st = calloc(cap, sizeof *st);
  if (!st) { free(text); return -1; }
  const char *p = skip_ws(arr + 1, ae);
  while (p < ae && *p == '{') {
    const char *oe = matching(p, ae, '{', '}');
    if (!oe) break;
    char op[32] = "brush";
    (void)obj_string(p + 1, oe, "op", op, sizeof op);
    if (cn == cap) {
      uint32_t nc = cap * 2u;
      void *q = realloc(st, (size_t)nc * sizeof *st);
      if (!q) break;
      st = q;
      memset(st + cap, 0, (size_t)cap * sizeof *st);
      cap = nc;
    }
    r3d_annot_stroke *s = &st[cn];
    double d;
    if (strcmp(op, "undo") == 0) {
      s->op = R3D_ANNOT_OP_UNDO;
      cn++;
    } else {
      s->op = strcmp(op, "line") == 0 ? R3D_ANNOT_OP_LINE : R3D_ANNOT_OP_BRUSH;
      s->cls = 0;
      if (json_number(key_value(p + 1, oe, "class"), oe, &d) == 0 && d >= 0.0 &&
          d < (double)R3D_ANNOT_NCLASS)
        s->cls = (uint8_t)llround(d);
      s->radius = json_number(key_value(p + 1, oe, "radius"), oe, &d) == 0 ? (int32_t)llround(d) : 0;
      s->depth = json_number(key_value(p + 1, oe, "depth"), oe, &d) == 0 ? (int32_t)llround(d) : 0;
      s->z = json_number(key_value(p + 1, oe, "z"), oe, &d) == 0 ? (int32_t)llround(d) : 0;
      int32_t *pts = NULL;
      uint32_t np = 0;
      if (parse_points(p + 1, oe, &pts, &np) != 0) {
        fprintf(stderr, "annot: stroke %u has no usable points\n", cn);
        free(text);
        r3d_annot_strokes_free(st, cn);
        return -1;
      }
      s->pts = pts;
      s->npts = np;
      cn++;
    }
    p = skip_ws(oe + 1, ae);
    if (p < ae && *p == ',') p = skip_ws(p + 1, ae);
  }
  free(text);
  *out = st;
  *count = cn;
  return 0;
}

/* ===================== slice compositing ================================= */

void r3d_annot_view_default(r3d_annot_view *v) {
  v->show = R3D_ANNOT_SHOW_FACES | R3D_ANNOT_SHOW_IGNORE | R3D_ANNOT_SHOW_CORRECTION;
  v->ct_gain = 1.0f;
  v->label_alpha = 0.45f;
  v->corr_alpha = 0.85f;
  v->lut = NULL;
}

static const uint8_t k_rv_rgb[4][3] = {
    {0, 0, 0}, {200, 90, 90}, {90, 110, 210}, {220, 200, 60}};
/* Saturated correction palette: bright red / bright blue / neutral grey /
 * green ("nothing here"), so a correction always wins the eye over the
 * exporter's dimmer label bands. */
static const uint8_t k_corr_rgb[R3D_ANNOT_NCLASS][3] = {
    {0, 0, 0}, {255, 48, 48}, {56, 128, 255}, {200, 200, 200}, {0, 230, 120}};

static void blend(float *dst, const uint8_t rgb[3], float a) {
  for (int k = 0; k < 3; k++) dst[k] += ((float)rgb[k] - dst[k]) * a;
}

void r3d_annot_composite(const r3d_annot_packet *p, uint32_t z, const r3d_annot_view *v,
                         uint8_t *rgba) {
  if (!p || !rgba || !v || z >= p->nz) return;
  size_t plane = (size_t)p->ny * p->nx;
  size_t base = (size_t)z * plane;
  const uint8_t *ct = p->layer[R3D_ANNOT_L_CT] + base;
  const uint8_t *fin = p->layer[R3D_ANNOT_L_FACES_IN] ? p->layer[R3D_ANNOT_L_FACES_IN] + base : NULL;
  const uint8_t *fout =
      p->layer[R3D_ANNOT_L_FACES_OUT] ? p->layer[R3D_ANNOT_L_FACES_OUT] + base : NULL;
  const uint8_t *ign = p->layer[R3D_ANNOT_L_IGNORE] ? p->layer[R3D_ANNOT_L_IGNORE] + base : NULL;
  const uint8_t *rv = p->layer[R3D_ANNOT_L_RV_CLASS] ? p->layer[R3D_ANNOT_L_RV_CLASS] + base : NULL;
  const uint8_t *pin = p->layer[R3D_ANNOT_L_PRED_IN] ? p->layer[R3D_ANNOT_L_PRED_IN] + base : NULL;
  const uint8_t *pout =
      p->layer[R3D_ANNOT_L_PRED_OUT] ? p->layer[R3D_ANNOT_L_PRED_OUT] + base : NULL;
  const uint8_t *corr = p->correction + base;
  float la = v->label_alpha, ca = v->corr_alpha;
  if (la < 0.0f) la = 0.0f;
  if (la > 1.0f) la = 1.0f;
  if (ca < 0.0f) ca = 0.0f;
  if (ca > 1.0f) ca = 1.0f;

  for (uint32_t y = 0; y < p->ny; y++)
    for (uint32_t x = 0; x < p->nx; x++) {
      size_t i = (size_t)y * p->nx + x;
      float c[3];
      uint8_t s = ct[i];
      if (v->lut) {
        for (int k = 0; k < 3; k++) c[k] = (float)v->lut[s][k] * v->ct_gain;
      } else {
        float g = (float)s * v->ct_gain;
        c[0] = c[1] = c[2] = g;
      }
      for (int k = 0; k < 3; k++) {
        if (c[k] < 0.0f) c[k] = 0.0f;
        if (c[k] > 255.0f) c[k] = 255.0f;
      }
      if ((v->show & R3D_ANNOT_SHOW_RV) && rv && rv[i] > 0 && rv[i] < 4)
        blend(c, k_rv_rgb[rv[i]], la * 0.5f);
      if ((v->show & R3D_ANNOT_SHOW_PRED) && ((x + y) & 3u) < 2u) {
        if (pin && pin[i]) blend(c, k_corr_rgb[R3D_ANNOT_IN], la * 0.7f);
        else if (pout && pout[i]) blend(c, k_corr_rgb[R3D_ANNOT_OUT], la * 0.7f);
      }
      if (v->show & R3D_ANNOT_SHOW_FACES) {
        if (fin && fin[i]) blend(c, k_corr_rgb[R3D_ANNOT_IN], la);
        else if (fout && fout[i]) blend(c, k_corr_rgb[R3D_ANNOT_OUT], la);
      }
      if ((v->show & R3D_ANNOT_SHOW_IGNORE) && ign && ign[i]) {
        static const uint8_t grey[3] = {110, 110, 110};
        blend(c, grey, 0.55f);
      }
      if ((v->show & R3D_ANNOT_SHOW_CORRECTION) && corr[i] > 0 && corr[i] < R3D_ANNOT_NCLASS)
        blend(c, k_corr_rgb[corr[i]], ca);
      uint8_t *o = rgba + i * 4u;
      for (int k = 0; k < 3; k++) o[k] = (uint8_t)(c[k] < 0.0f ? 0.0f : c[k] > 255.0f ? 255.0f : c[k]);
      o[3] = 255u;
    }
}
