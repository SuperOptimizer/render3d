#include "core/view.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/pngw.h"

const char *const r3d_view_kind_name[R3D_VIEW_NKIND] = {
    "ct", "sdf", "prob", "signed", "density", "count", "class"};

int r3d_view_kind_parse(const char *s) {
  if (!s) return -1;
  for (int i = 0; i < R3D_VIEW_NKIND; i++)
    if (strcmp(s, r3d_view_kind_name[i]) == 0) return i;
  return -1;
}

/* ===================== minimal JSON scanning =============================
 * Same shape as core/annot.c's reader (spec/annot.md): enough to walk an
 * object and pull typed values out of it, with no allocation and no schema.
 * Unknown keys are simply never asked for, which is what the spec wants. */

static const char *vj_ws(const char *p, const char *end) {
  while (p < end && isspace((unsigned char)*p)) p++;
  return p;
}

static const char *vj_match(const char *p, const char *end, char open, char close) {
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

static const char *vj_value_end(const char *p, const char *end) {
  if (p >= end) return NULL;
  if (*p == '{') { const char *q = vj_match(p, end, '{', '}'); return q ? q + 1 : NULL; }
  if (*p == '[') { const char *q = vj_match(p, end, '[', ']'); return q ? q + 1 : NULL; }
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

/* Value of `key` at this object's top level, or NULL. */
static const char *vj_key(const char *begin, const char *end, const char *key) {
  size_t n = strlen(key);
  const char *p = vj_ws(begin, end);
  while (p < end) {
    if (*p != '"') { p++; continue; }
    const char *ks = p + 1;
    const char *ke = vj_value_end(p, end);
    if (!ke) return NULL;
    const char *c = vj_ws(ke, end);
    if (c >= end || *c != ':') { p = ke; continue; }
    const char *v = vj_ws(c + 1, end);
    const char *ve = vj_value_end(v, end);
    if (!ve) return NULL;
    if ((size_t)(ke - 1 - ks) == n && memcmp(ks, key, n) == 0) return v;
    p = ve;
  }
  return NULL;
}

static int vj_number(const char *v, const char *end, double *out) {
  if (!v || v >= end) return -1;
  errno = 0;
  char *q = NULL;
  double d = strtod(v, &q);
  if (q == v || q > end || errno == ERANGE || !isfinite(d)) return -1;
  *out = d;
  return 0;
}

static int vj_num_key(const char *b, const char *e, const char *key, double *out) {
  return vj_number(vj_key(b, e, key), e, out);
}

static int vj_ints(const char *b, const char *e, const char *key, int64_t *out, int want) {
  const char *v = vj_key(b, e, key);
  if (!v || v >= e || *v != '[') return -1;
  const char *ae = vj_match(v, e, '[', ']');
  if (!ae) return -1;
  const char *p = vj_ws(v + 1, ae);
  for (int i = 0; i < want; i++) {
    double d;
    if (vj_number(p, ae, &d) != 0) return -1;
    if (d < -9.0e15 || d > 9.0e15) return -1;
    out[i] = (int64_t)llround(d);
    const char *ve = vj_value_end(p, ae);
    if (!ve) return -1;
    p = vj_ws(ve, ae);
    if (i + 1 < want) {
      if (p >= ae || *p != ',') return -1;
      p = vj_ws(p + 1, ae);
    }
  }
  return 0;
}

static int vj_string(const char *b, const char *e, const char *key, char *out, size_t cap) {
  const char *v = vj_key(b, e, key);
  if (!v || v >= e || *v != '"') return -1;
  const char *ve = vj_value_end(v, e);
  if (!ve) return -1;
  size_t n = (size_t)(ve - v) - 2u;
  if (n >= cap) return -1;
  memcpy(out, v + 1, n); /* packet strings are plain: no escape handling */
  out[n] = 0;
  return 0;
}

/* Copy the raw text of `key`'s value (used for free-form "provenance"). */
static void vj_raw(const char *b, const char *e, const char *key, char *out, size_t cap) {
  out[0] = 0;
  const char *v = vj_key(b, e, key);
  if (!v) return;
  const char *ve = vj_value_end(v, e);
  if (!ve) return;
  size_t n = (size_t)(ve - v);
  if (n >= cap) n = cap - 1u;
  memcpy(out, v, n);
  out[n] = 0;
}

static char *vj_read_text(const char *path, size_t *out_size) {
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

static bool vj_is_file(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}
static bool vj_is_dir(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* ===================== layer defaults ==================================== */

/* Ten well-separated hues; layer N gets hue N so a freshly opened packet is
 * legible without touching a single colour picker. */
static const float k_layer_col[10][3] = {
    {1.00f, 0.30f, 0.30f}, {0.30f, 0.70f, 1.00f}, {0.40f, 0.95f, 0.45f},
    {1.00f, 0.80f, 0.25f}, {0.85f, 0.45f, 1.00f}, {0.25f, 0.95f, 0.90f},
    {1.00f, 0.55f, 0.20f}, {0.65f, 0.85f, 0.30f}, {1.00f, 0.45f, 0.70f},
    {0.55f, 0.60f, 1.00f}};

/* class colours; class 0 is transparent and never looked up */
static const float k_class_col[8][3] = {
    {0.00f, 0.00f, 0.00f}, {0.95f, 0.35f, 0.35f}, {0.35f, 0.55f, 0.95f},
    {0.95f, 0.85f, 0.25f}, {0.35f, 0.90f, 0.50f}, {0.85f, 0.40f, 0.95f},
    {0.30f, 0.90f, 0.90f}, {0.95f, 0.60f, 0.25f}};

void r3d_view_layer_defaults(r3d_view_layer *l, uint32_t ordinal) {
  l->show = l->kind != R3D_VIEW_CT;
  l->solo = false;
  memcpy(l->color, k_layer_col[ordinal % 10u], sizeof l->color);
  l->opacity = 0.7f;
  l->threshold = l->kind == R3D_VIEW_PROB ? 0.5f : 0.0f;
  l->tol = 1.5f;
  l->heatmap = false;
}

/* ===================== packet parsing ==================================== */

static int layer_push(r3d_view_packet *p, const r3d_view_layer *src) {
  if (p->count == p->cap) {
    uint32_t nc = p->cap ? p->cap * 2u : 16u;
    void *q = realloc(p->layer, (size_t)nc * sizeof *p->layer);
    if (!q) return -1;
    p->layer = q;
    memset(p->layer + p->cap, 0, (size_t)(nc - p->cap) * sizeof *p->layer);
    p->cap = nc;
  }
  p->layer[p->count++] = *src;
  return 0;
}

static uint8_t *read_layer_file(const char *dir, const char *file, size_t want) {
  char path[1400];
  if (snprintf(path, sizeof path, "%s/%s", dir, file) >= (int)sizeof path) {
    fprintf(stderr, "view: layer path too long (%s/%s)\n", dir, file);
    return NULL;
  }
  struct stat st;
  if (stat(path, &st) != 0) {
    fprintf(stderr, "view: %s is missing\n", path);
    return NULL;
  }
  if ((size_t)st.st_size != want) {
    fprintf(stderr, "view: %s is %lld bytes, expected %zu\n", path, (long long)st.st_size, want);
    return NULL;
  }
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "view: cannot open %s\n", path);
    return NULL;
  }
  uint8_t *buf = malloc(want ? want : 1u);
  if (!buf) { fclose(f); return NULL; }
  bool ok = want == 0 || fread(buf, 1, want, f) == want;
  fclose(f);
  if (!ok) {
    fprintf(stderr, "view: short read on %s\n", path);
    free(buf);
    return NULL;
  }
  return buf;
}

int r3d_view_packet_parse(r3d_view_packet *p, const char *json, size_t n, const char *dir) {
  if (!p || !json) return -1;
  memset(p, 0, sizeof *p);
  p->ct = -1;
  if (dir) snprintf(p->dir, sizeof p->dir, "%s", dir);
  const char *b = json, *e = json + n;

  char fmt[64] = "";
  if (vj_string(b, e, "format", fmt, sizeof fmt) == 0 && strcmp(fmt, "tsm.view.v1") != 0) {
    fprintf(stderr, "view: format \"%s\" is not tsm.view.v1\n", fmt);
    return -1;
  }
  int64_t dims[3];
  if (vj_ints(b, e, "dims_zyx", dims, 3) != 0) {
    fprintf(stderr, "view: meta has no usable dims_zyx\n");
    return -1;
  }
  for (int k = 0; k < 3; k++)
    if (dims[k] <= 0 || dims[k] > 4096) {
      fprintf(stderr, "view: dims_zyx[%d] = %lld out of range (1..4096)\n", k,
              (long long)dims[k]);
      return -1;
    }
  p->nz = (uint32_t)dims[0];
  p->ny = (uint32_t)dims[1];
  p->nx = (uint32_t)dims[2];
  if (vj_ints(b, e, "origin_zyx", p->origin, 3) != 0)
    p->origin[0] = p->origin[1] = p->origin[2] = 0;
  if (vj_num_key(b, e, "voxel_um", &p->voxel_um) != 0 || p->voxel_um <= 0.0) p->voxel_um = 0.0;
  (void)vj_string(b, e, "scroll", p->scroll, sizeof p->scroll);
  vj_raw(b, e, "provenance", p->provenance, sizeof p->provenance);

  const char *arr = vj_key(b, e, "layers");
  if (!arr || *arr != '[') {
    fprintf(stderr, "view: meta has no layers[]\n");
    return -1;
  }
  const char *ae = vj_match(arr, e, '[', ']');
  if (!ae) { fprintf(stderr, "view: layers[] is malformed\n"); return -1; }

  size_t want = r3d_view_voxels(p);
  const char *q = vj_ws(arr + 1, ae);
  while (q < ae && *q == '{') {
    const char *oe = vj_match(q, ae, '{', '}');
    if (!oe) break;
    const char *lb = q + 1;
    r3d_view_layer l = {0};
    l.axis = -1;
    char kind[32] = "";
    if (vj_string(lb, oe, "file", l.file, sizeof l.file) != 0 ||
        vj_string(lb, oe, "name", l.name, sizeof l.name) != 0 ||
        vj_string(lb, oe, "group", l.group, sizeof l.group) != 0 ||
        vj_string(lb, oe, "kind", kind, sizeof kind) != 0) {
      fprintf(stderr, "view: layer %u needs file/name/group/kind\n", p->count);
      goto fail;
    }
    if (strchr(l.file, '/') != NULL || l.file[0] == '.') {
      fprintf(stderr, "view: layer file \"%s\" must be a plain name in the packet dir\n", l.file);
      goto fail;
    }
    int kd = r3d_view_kind_parse(kind);
    if (kd < 0) {
      fprintf(stderr, "view: layer %s.%s has unknown kind \"%s\"\n", l.group, l.name, kind);
      goto fail;
    }
    l.kind = (r3d_view_kind)kd;
    if (l.kind == R3D_VIEW_SDF) {
      if (vj_num_key(lb, oe, "clip", &l.clip) != 0 || !(l.clip > 0.0)) {
        fprintf(stderr, "view: sdf layer %s.%s needs a positive clip\n", l.group, l.name);
        goto fail;
      }
    }
    if (l.kind == R3D_VIEW_CLASS) {
      const char *pal = vj_key(lb, oe, "palette");
      if (pal && *pal == '[') {
        const char *pe = vj_match(pal, oe, '[', ']');
        const char *pp = pe ? vj_ws(pal + 1, pe) : NULL;
        while (pp && pp < pe && *pp == '"' && l.npalette < R3D_VIEW_MAX_CLASS) {
          const char *se = vj_value_end(pp, pe);
          if (!se) break;
          size_t sn = (size_t)(se - pp) - 2u;
          if (sn >= sizeof l.palette[0]) sn = sizeof l.palette[0] - 1u;
          memcpy(l.palette[l.npalette], pp + 1, sn);
          l.palette[l.npalette][sn] = 0;
          l.npalette++;
          pp = vj_ws(se, pe);
          if (pp < pe && *pp == ',') pp = vj_ws(pp + 1, pe);
        }
      }
    }
    if (l.kind == R3D_VIEW_SIGNED) {
      if (vj_string(lb, oe, "vec", l.vec, sizeof l.vec) == 0) {
        double ax;
        if (vj_num_key(lb, oe, "axis", &ax) != 0 || ax < 0.0 || ax > 2.0) {
          fprintf(stderr, "view: signed layer %s.%s has vec but no axis 0/1/2\n", l.group,
                  l.name);
          goto fail;
        }
        l.axis = (int)llround(ax);
      }
    }
    for (uint32_t i = 0; i < p->count; i++)
      if (strcmp(p->layer[i].group, l.group) == 0 && strcmp(p->layer[i].name, l.name) == 0) {
        fprintf(stderr, "view: duplicate layer %s.%s\n", l.group, l.name);
        goto fail;
      }
    if (l.kind == R3D_VIEW_CT) {
      if (p->ct >= 0) {
        fprintf(stderr, "view: more than one kind-ct layer\n");
        goto fail;
      }
      p->ct = (int)p->count;
    }
    if (p->count >= R3D_VIEW_MAX_LAYERS) {
      fprintf(stderr, "view: more than %u layers\n", R3D_VIEW_MAX_LAYERS);
      goto fail;
    }
    r3d_view_layer_defaults(&l, p->count);
    if (dir) {
      l.data = read_layer_file(dir, l.file, want);
      if (!l.data) goto fail;
    }
    if (layer_push(p, &l) != 0) {
      free(l.data);
      goto fail;
    }
    q = vj_ws(oe + 1, ae);
    if (q < ae && *q == ',') q = vj_ws(q + 1, ae);
  }

  if (p->ct < 0 && dir) {
    fprintf(stderr, "view: no layer of kind ct\n");
    goto fail;
  }
  return 0;
fail:
  r3d_view_packet_free(p);
  return -1;
}

int r3d_view_packet_load(r3d_view_packet *p, const char *dir) {
  if (!p || !dir) return -1;
  char meta[1100];
  if (snprintf(meta, sizeof meta, "%s/meta.json", dir) >= (int)sizeof meta) return -1;
  size_t n = 0;
  char *text = vj_read_text(meta, &n);
  if (!text) {
    fprintf(stderr, "view: cannot read %s\n", meta);
    return -1;
  }
  int rc = r3d_view_packet_parse(p, text, n, dir);
  free(text);
  return rc;
}

void r3d_view_packet_free(r3d_view_packet *p) {
  if (!p) return;
  for (uint32_t i = 0; i < p->count; i++) free(p->layer[i].data);
  free(p->layer);
  memset(p, 0, sizeof *p);
  p->ct = -1;
}

int r3d_view_find(const r3d_view_packet *p, const char *spec) {
  if (!p || !spec || !*spec) return -1;
  const char *dot = strchr(spec, '.');
  if (dot) {
    size_t gn = (size_t)(dot - spec);
    for (uint32_t i = 0; i < p->count; i++)
      if (strlen(p->layer[i].group) == gn && memcmp(p->layer[i].group, spec, gn) == 0 &&
          strcmp(p->layer[i].name, dot + 1) == 0)
        return (int)i;
    return -1;
  }
  int hit = -1;
  for (uint32_t i = 0; i < p->count; i++)
    if (strcmp(p->layer[i].name, spec) == 0) {
      if (hit >= 0) return -1; /* ambiguous */
      hit = (int)i;
    }
  return hit;
}

int r3d_view_merge(r3d_view_packet *dst, r3d_view_packet *src) {
  if (!dst || !src) return -1;
  if (dst->nz != src->nz || dst->ny != src->ny || dst->nx != src->nx ||
      memcmp(dst->origin, src->origin, sizeof dst->origin) != 0) {
    r3d_view_packet_free(dst);
    *dst = *src;
    memset(src, 0, sizeof *src);
    src->ct = -1;
    return 0;
  }
  for (uint32_t i = 0; i < src->count; i++) {
    r3d_view_layer *s = &src->layer[i];
    int at = -1;
    for (uint32_t k = 0; k < dst->count; k++)
      if (strcmp(dst->layer[k].group, s->group) == 0 && strcmp(dst->layer[k].name, s->name) == 0)
        at = (int)k;
    if (at >= 0) {
      r3d_view_layer *d = &dst->layer[at];
      /* keep the user's display state; take the new bytes and decoding */
      bool show = d->show, solo = d->solo, heat = d->heatmap;
      float col[3], op = d->opacity, th = d->threshold, tol = d->tol;
      memcpy(col, d->color, sizeof col);
      free(d->data);
      *d = *s;
      d->show = show;
      d->solo = solo;
      d->heatmap = heat;
      d->opacity = op;
      d->threshold = th;
      d->tol = tol;
      memcpy(d->color, col, sizeof col);
    } else {
      r3d_view_layer nl = *s;
      r3d_view_layer_defaults(&nl, dst->count);
      if (layer_push(dst, &nl) != 0) return -1;
      if (nl.kind == R3D_VIEW_CT && dst->ct < 0) dst->ct = (int)dst->count - 1;
    }
    s->data = NULL;
  }
  if (src->provenance[0]) snprintf(dst->provenance, sizeof dst->provenance, "%s", src->provenance);
  r3d_view_packet_free(src);
  return 0;
}

/* ===================== manifest ========================================== */

static void dirname_of(const char *path, char *out, size_t cap) {
  const char *slash = strrchr(path, '/');
  if (!slash) { snprintf(out, cap, "."); return; }
  size_t n = (size_t)(slash - path);
  if (n == 0) n = 1;
  if (n >= cap) n = cap - 1;
  memcpy(out, path, n);
  out[n] = 0;
}

int r3d_view_manifest_load(r3d_view_manifest *m, const char *path) {
  if (!m || !path) return -1;
  memset(m, 0, sizeof *m);

  char json_path[1024];
  if (vj_is_dir(path)) {
    if (snprintf(json_path, sizeof json_path, "%s/view.json", path) >= (int)sizeof json_path)
      return -1;
    if (!vj_is_file(json_path)) {
      char meta[1100];
      if (snprintf(meta, sizeof meta, "%s/meta.json", path) >= (int)sizeof meta) return -1;
      if (!vj_is_file(meta)) return -1;
      m->ent = calloc(1, sizeof *m->ent);
      if (!m->ent) return -1;
      m->count = 1;
      snprintf(m->ent[0].path, sizeof m->ent[0].path, "%s", path);
      const char *base = strrchr(path, '/');
      snprintf(m->ent[0].name, sizeof m->ent[0].name, "%s", base ? base + 1 : path);
      return 0;
    }
  } else {
    if (snprintf(json_path, sizeof json_path, "%s", path) >= (int)sizeof json_path) return -1;
    if (!vj_is_file(json_path)) return -1;
  }

  size_t n = 0;
  char *text = vj_read_text(json_path, &n);
  if (!text) return -1;
  const char *end = text + n;
  const char *arr = vj_key(text, end, "packets");
  const char *ae = arr && *arr == '[' ? vj_match(arr, end, '[', ']') : NULL;
  if (!ae) { free(text); return -1; }

  char root[1024];
  dirname_of(json_path, root, sizeof root);
  snprintf(m->path, sizeof m->path, "%s", json_path);

  uint32_t cap = 16;
  m->ent = calloc(cap, sizeof *m->ent);
  if (!m->ent) { free(text); return -1; }

  const char *p = vj_ws(arr + 1, ae);
  while (p < ae && *p == '{') {
    const char *oe = vj_match(p, ae, '{', '}');
    if (!oe) break;
    char rel[600];
    if (vj_string(p + 1, oe, "path", rel, sizeof rel) == 0) {
      if (m->count == cap) {
        uint32_t nc = cap * 2u;
        void *q = realloc(m->ent, (size_t)nc * sizeof *m->ent);
        if (!q) break;
        m->ent = q;
        memset((r3d_view_entry *)q + cap, 0, (size_t)cap * sizeof *m->ent);
        cap = nc;
      }
      r3d_view_entry *ent = &m->ent[m->count];
      snprintf(ent->name, sizeof ent->name, "%s", rel);
      if (rel[0] == '/') snprintf(ent->path, sizeof ent->path, "%s", rel);
      else snprintf(ent->path, sizeof ent->path, "%s/%s", root, rel);
      int64_t v[3];
      if (vj_ints(p + 1, oe, "origin_zyx", v, 3) == 0)
        for (int k = 0; k < 3; k++) ent->origin[k] = v[k];
      if (vj_ints(p + 1, oe, "dims_zyx", v, 3) == 0)
        for (int k = 0; k < 3; k++)
          ent->dims[k] = v[k] > 0 && v[k] < 0x7fffffff ? (uint32_t)v[k] : 0u;
      m->count++;
    }
    p = vj_ws(oe + 1, ae);
    if (p < ae && *p == ',') p = vj_ws(p + 1, ae);
  }
  free(text);
  if (m->count == 0) { free(m->ent); m->ent = NULL; return -1; }
  return 0;
}

void r3d_view_manifest_free(r3d_view_manifest *m) {
  if (!m) return;
  free(m->ent);
  memset(m, 0, sizeof *m);
}

/* ===================== decoding ========================================== */

double r3d_view_decode(const r3d_view_layer *l, uint8_t u, bool *no_data) {
  if (no_data) *no_data = false;
  switch (l->kind) {
    case R3D_VIEW_CT: return (double)u / 255.0;
    case R3D_VIEW_SDF:
      if (u == 0) {
        if (no_data) *no_data = true;
        return 0.0;
      }
      return ((double)u - 128.0) * l->clip / 127.0;
    case R3D_VIEW_PROB: return (double)u / 255.0;
    case R3D_VIEW_SIGNED: return ((double)u - 127.5) / 127.5;
    case R3D_VIEW_DENSITY: return (double)u / 1000.0;
    case R3D_VIEW_COUNT: return (double)u;
    case R3D_VIEW_CLASS: return (double)u;
    case R3D_VIEW_NKIND: break;
  }
  return (double)u;
}

int r3d_view_value_text(const r3d_view_layer *l, uint8_t u, char *out, size_t cap) {
  bool nd = false;
  double d = r3d_view_decode(l, u, &nd);
  if (nd) return snprintf(out, cap, "-- (no data)");
  switch (l->kind) {
    case R3D_VIEW_SDF: return snprintf(out, cap, "%+.2f vox", d);
    case R3D_VIEW_COUNT: return snprintf(out, cap, "%d vox", (int)d);
    case R3D_VIEW_CLASS:
      if (u < l->npalette) return snprintf(out, cap, "%u %s", u, l->palette[u]);
      return snprintf(out, cap, "%u", u);
    case R3D_VIEW_DENSITY: return snprintf(out, cap, "%.3f", d);
    default: return snprintf(out, cap, "%.3f", d);
  }
}

/* ===================== slice geometry ==================================== */

void r3d_view_slice_dims(const r3d_view_packet *p, int axis, uint32_t *w, uint32_t *h) {
  switch (axis) {
    case R3D_VIEW_AXIS_Y: *w = p->nx; *h = p->nz; return;
    case R3D_VIEW_AXIS_X: *w = p->ny; *h = p->nz; return;
    default: *w = p->nx; *h = p->ny; return;
  }
}

uint32_t r3d_view_slice_count(const r3d_view_packet *p, int axis) {
  switch (axis) {
    case R3D_VIEW_AXIS_Y: return p->ny;
    case R3D_VIEW_AXIS_X: return p->nx;
    default: return p->nz;
  }
}

void r3d_view_voxel(const r3d_view_packet *p, int axis, uint32_t s, uint32_t u, uint32_t v,
                    uint32_t zyx[3]) {
  (void)p;
  switch (axis) {
    case R3D_VIEW_AXIS_Y: zyx[0] = v; zyx[1] = s; zyx[2] = u; return;
    case R3D_VIEW_AXIS_X: zyx[0] = v; zyx[1] = u; zyx[2] = s; return;
    default: zyx[0] = s; zyx[1] = v; zyx[2] = u; return;
  }
}

static inline size_t slice_index(const r3d_view_packet *p, int axis, uint32_t s, uint32_t u,
                                 uint32_t v) {
  switch (axis) {
    case R3D_VIEW_AXIS_Y: return ((size_t)v * p->ny + s) * p->nx + u;
    case R3D_VIEW_AXIS_X: return ((size_t)v * p->ny + u) * p->nx + s;
    default: return ((size_t)s * p->ny + v) * p->nx + u;
  }
}

uint8_t r3d_view_sample(const r3d_view_packet *p, const r3d_view_layer *l, int axis, uint32_t s,
                        uint32_t u, uint32_t v) {
  if (!l->data) return 0;
  return l->data[slice_index(p, axis, s, u, v)];
}

/* ===================== compositing ======================================= */

void r3d_view_opts_default(r3d_view_opts *o) {
  memset(o, 0, sizeof *o);
  o->axis = R3D_VIEW_AXIS_Z;
  o->slice = 0;
  o->ct_gain = 1.0f;
  o->cmp_a = o->cmp_b = -1;
  o->cmp_class = -1;
  const float agree[3] = {0.25f, 0.95f, 0.35f};
  const float aonly[3] = {0.95f, 0.25f, 0.25f};
  const float bonly[3] = {0.30f, 0.50f, 1.00f};
  memcpy(o->cmp_col[0], agree, sizeof agree);
  memcpy(o->cmp_col[1], aonly, sizeof aonly);
  memcpy(o->cmp_col[2], bonly, sizeof bonly);
}

/* Is this layer "present" at byte u, under its own thresholds? spec item 3. */
static bool layer_present(const r3d_view_layer *l, uint8_t u, int cmp_class) {
  switch (l->kind) {
    case R3D_VIEW_SDF: {
      if (u == 0) return false;
      double d = ((double)u - 128.0) * l->clip / 127.0;
      return fabs(d) <= (double)l->tol;
    }
    case R3D_VIEW_PROB: return (double)u / 255.0 >= (double)l->threshold;
    case R3D_VIEW_CLASS: return cmp_class < 0 ? u != 0u : (int)u == cmp_class;
    case R3D_VIEW_COUNT: return u != 0u && (double)u >= (double)l->threshold;
    case R3D_VIEW_DENSITY: return (double)u / 1000.0 > (double)l->threshold;
    case R3D_VIEW_SIGNED: return fabs(((double)u - 127.5) / 127.5) >= (double)l->threshold;
    case R3D_VIEW_CT:
    case R3D_VIEW_NKIND: break;
  }
  return u != 0u;
}

static void blend(float *dst, const float rgb[3], float a) {
  if (a <= 0.0f) return;
  if (a > 1.0f) a = 1.0f;
  for (int k = 0; k < 3; k++) dst[k] += (rgb[k] - dst[k]) * a;
}

/* Sequential ramp for density/count: black -> red -> yellow -> white. */
static void heat_seq(float t, float rgb[3]) {
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  rgb[0] = t < 0.4f ? t / 0.4f : 1.0f;
  rgb[1] = t < 0.4f ? 0.0f : (t < 0.8f ? (t - 0.4f) / 0.4f : 1.0f);
  rgb[2] = t < 0.8f ? 0.0f : (t - 0.8f) / 0.2f;
}

/* Diverging ramp for signed/sdf: blue (negative) -> grey -> red (positive). */
static void heat_div(float t, float rgb[3]) {
  if (t < -1.0f) t = -1.0f;
  if (t > 1.0f) t = 1.0f;
  if (t < 0.0f) {
    rgb[0] = 0.5f + 0.5f * t;
    rgb[1] = 0.5f + 0.5f * t;
    rgb[2] = 1.0f;
  } else {
    rgb[0] = 1.0f;
    rgb[1] = 0.5f - 0.5f * t;
    rgb[2] = 0.5f - 0.5f * t;
  }
}

/* Which layers actually draw: solo wins over show. */
static bool layer_visible(const r3d_view_packet *p, uint32_t i) {
  bool any_solo = false;
  for (uint32_t k = 0; k < p->count; k++) any_solo = any_solo || p->layer[k].solo;
  const r3d_view_layer *l = &p->layer[i];
  if (l->kind == R3D_VIEW_CT) return false; /* the CT is the base image */
  return any_solo ? l->solo : l->show;
}

/* The two siblings that complete a `vec` triple, or -1. */
static int vec_sibling(const r3d_view_packet *p, const r3d_view_layer *l, int axis) {
  for (uint32_t i = 0; i < p->count; i++) {
    const r3d_view_layer *c = &p->layer[i];
    if (c->kind == R3D_VIEW_SIGNED && c->axis == axis && c->vec[0] &&
        strcmp(c->vec, l->vec) == 0 && strcmp(c->group, l->group) == 0)
      return (int)i;
  }
  return -1;
}

/* Auto range for the sequential kinds: the slice's own maximum. */
static double slice_max(const r3d_view_packet *p, const r3d_view_layer *l, int axis,
                        uint32_t s, uint32_t w, uint32_t h) {
  uint8_t m = 0;
  for (uint32_t v = 0; v < h; v++)
    for (uint32_t u = 0; u < w; u++) {
      uint8_t b = l->data[slice_index(p, axis, s, u, v)];
      if (b > m) m = b;
    }
  bool nd;
  double d = r3d_view_decode(l, m, &nd);
  return d > 0.0 ? d : 1.0;
}

/* Per-layer draw state, resolved once per slice so the pixel loop below does
 * no searching: the composite must stay allocation-free and linear in pixels
 * even for a 25-layer packet. */
typedef struct draw_state {
  bool draw;     /* this layer paints */
  bool rgb_head; /* the axis-0 member of a complete, visible `vec` triple */
  int sib[3];    /* the triple's three layer indices when rgb_head */
  float inv_range; /* density/count: 1 / the slice maximum */
} draw_state;

void r3d_view_composite(const r3d_view_packet *p, const r3d_view_opts *o, uint8_t *rgba) {
  if (!p || !o || !rgba || p->ct < 0 || p->count > R3D_VIEW_MAX_LAYERS) return;
  uint32_t w, h;
  r3d_view_slice_dims(p, o->axis, &w, &h);
  if (o->slice >= r3d_view_slice_count(p, o->axis)) return;
  const r3d_view_layer *ct = &p->layer[p->ct];
  bool compare = o->cmp_a >= 0 && o->cmp_b >= 0 && (uint32_t)o->cmp_a < p->count &&
                 (uint32_t)o->cmp_b < p->count &&
                 p->layer[o->cmp_a].kind == p->layer[o->cmp_b].kind;

  draw_state st[R3D_VIEW_MAX_LAYERS];
  for (uint32_t i = 0; i < p->count; i++) {
    const r3d_view_layer *l = &p->layer[i];
    st[i] = (draw_state){.sib = {-1, -1, -1}, .inv_range = 1.0f};
    st[i].draw = l->data && layer_visible(p, i) &&
                 !(compare && ((int)i == o->cmp_a || (int)i == o->cmp_b));
    if (!st[i].draw) continue;
    if (l->kind == R3D_VIEW_DENSITY || l->kind == R3D_VIEW_COUNT) {
      double mx = slice_max(p, l, o->axis, o->slice, w, h);
      st[i].inv_range = (float)(1.0 / mx);
    }
    if (l->kind == R3D_VIEW_SIGNED && l->vec[0] && l->axis >= 0 && l->axis <= 2) {
      int s0 = vec_sibling(p, l, 0), s1 = vec_sibling(p, l, 1), s2 = vec_sibling(p, l, 2);
      bool whole = s0 >= 0 && s1 >= 0 && s2 >= 0 && p->layer[s0].data && p->layer[s1].data &&
                   p->layer[s2].data && layer_visible(p, (uint32_t)s0);
      if (whole) {
        if (l->axis == 0) {
          st[i].rgb_head = true;
          st[i].sib[0] = s0;
          st[i].sib[1] = s1;
          st[i].sib[2] = s2;
        } else {
          st[i].draw = false; /* painted by the axis-0 member */
        }
      }
    }
  }

  for (uint32_t v = 0; v < h; v++)
    for (uint32_t u = 0; u < w; u++) {
      size_t si = slice_index(p, o->axis, o->slice, u, v);
      float c[3];
      float g = ct->data ? (float)ct->data[si] / 255.0f * o->ct_gain : 0.0f;
      if (g < 0.0f) g = 0.0f;
      if (g > 1.0f) g = 1.0f;
      c[0] = c[1] = c[2] = g;

      for (uint32_t i = 0; i < p->count; i++) {
        if (!st[i].draw) continue;
        const r3d_view_layer *l = &p->layer[i];
        uint8_t b = l->data[si];
        float col[3];
        switch (l->kind) {
          case R3D_VIEW_PROB: {
            float pr = (float)b / 255.0f;
            if (pr < l->threshold) break;
            blend(c, l->color, pr * l->opacity);
            break;
          }
          case R3D_VIEW_SDF: {
            if (b == 0) break; /* no data */
            double d = ((double)b - 128.0) * l->clip / 127.0;
            if (l->heatmap) {
              heat_div((float)(d / l->clip), col);
              blend(c, col, l->opacity * 0.55f);
            }
            if (fabs(d) <= (double)l->tol) blend(c, l->color, l->opacity);
            break;
          }
          case R3D_VIEW_SIGNED: {
            if (st[i].rgb_head) { /* the triple as one RGB normal image */
              for (int k = 0; k < 3; k++)
                col[k] = 0.5f + 0.5f * (((float)p->layer[st[i].sib[k]].data[si] - 127.5f) /
                                        127.5f);
              blend(c, col, l->opacity);
              break;
            }
            float sv = ((float)b - 127.5f) / 127.5f;
            if (fabsf(sv) < l->threshold) break;
            heat_div(sv, col);
            blend(c, col, l->opacity * fabsf(sv));
            break;
          }
          case R3D_VIEW_DENSITY: {
            float d = (float)b / 1000.0f;
            if (d <= l->threshold) break;
            heat_seq(d * st[i].inv_range, col);
            blend(c, col, l->opacity);
            break;
          }
          case R3D_VIEW_COUNT: {
            if (b == 0 || (float)b < l->threshold) break;
            heat_seq((float)b * st[i].inv_range, col);
            blend(c, col, l->opacity);
            break;
          }
          case R3D_VIEW_CLASS: {
            if (b == 0) break;
            memcpy(col, k_class_col[b % 8u], sizeof col);
            blend(c, col, l->opacity);
            break;
          }
          case R3D_VIEW_CT:
          case R3D_VIEW_NKIND: break;
        }
      }

      if (compare) {
        const r3d_view_layer *la = &p->layer[o->cmp_a], *lb = &p->layer[o->cmp_b];
        bool pa = la->data && layer_present(la, la->data[si], o->cmp_class);
        bool pb = lb->data && layer_present(lb, lb->data[si], o->cmp_class);
        float alpha = la->opacity > 0.0f ? la->opacity : 0.8f;
        if (pa && pb) blend(c, o->cmp_col[0], alpha);
        else if (pa) blend(c, o->cmp_col[1], alpha);
        else if (pb) blend(c, o->cmp_col[2], alpha);
      }

      uint8_t *out = rgba + ((size_t)v * w + u) * 4u;
      for (int k = 0; k < 3; k++) {
        float f = c[k] * 255.0f;
        out[k] = (uint8_t)(f < 0.0f ? 0.0f : f > 255.0f ? 255.0f : f);
      }
      out[3] = 255u;
    }
}

int r3d_view_compare(const r3d_view_packet *p, const r3d_view_opts *o, r3d_view_cmp *out) {
  if (!p || !o || !out) return -1;
  memset(out, 0, sizeof *out);
  if (o->cmp_a < 0 || o->cmp_b < 0 || (uint32_t)o->cmp_a >= p->count ||
      (uint32_t)o->cmp_b >= p->count)
    return -1;
  const r3d_view_layer *la = &p->layer[o->cmp_a], *lb = &p->layer[o->cmp_b];
  if (la->kind != lb->kind || !la->data || !lb->data) return -1;
  uint32_t w, h;
  r3d_view_slice_dims(p, o->axis, &w, &h);
  if (o->slice >= r3d_view_slice_count(p, o->axis)) return -1;
  for (uint32_t v = 0; v < h; v++)
    for (uint32_t u = 0; u < w; u++) {
      size_t si = slice_index(p, o->axis, o->slice, u, v);
      bool pa = layer_present(la, la->data[si], o->cmp_class);
      bool pb = layer_present(lb, lb->data[si], o->cmp_class);
      if (pa && pb) out->agree++;
      else if (pa) out->a_only++;
      else if (pb) out->b_only++;
    }
  double denom = 2.0 * (double)out->agree + (double)out->a_only + (double)out->b_only;
  out->dice = denom > 0.0 ? 2.0 * (double)out->agree / denom : 0.0;
  return 0;
}

uint32_t r3d_view_hover(const r3d_view_packet *p, const r3d_view_opts *o, uint32_t u, uint32_t v,
                        char *out, size_t cap) {
  if (!p || !o || !out || cap == 0) return 0;
  out[0] = 0;
  uint32_t w, h;
  r3d_view_slice_dims(p, o->axis, &w, &h);
  if (u >= w || v >= h || o->slice >= r3d_view_slice_count(p, o->axis)) return 0;
  uint32_t zyx[3];
  r3d_view_voxel(p, o->axis, o->slice, u, v, zyx);
  size_t si = slice_index(p, o->axis, o->slice, u, v);
  size_t at = 0;
  uint32_t lines = 0;
  int n = snprintf(out, cap, "voxel z%u y%u x%u   scroll z%lld y%lld x%lld", zyx[0], zyx[1],
                   zyx[2], (long long)(p->origin[0] + (int64_t)zyx[0]),
                   (long long)(p->origin[1] + (int64_t)zyx[1]),
                   (long long)(p->origin[2] + (int64_t)zyx[2]));
  if (n < 0) return 0;
  at = (size_t)n < cap ? (size_t)n : cap - 1u;
  lines++;
  for (uint32_t i = 0; i < p->count && at + 1u < cap; i++) {
    const r3d_view_layer *l = &p->layer[i];
    if (!l->data) continue;
    if (l->kind != R3D_VIEW_CT && !layer_visible(p, i)) continue;
    char val[64];
    (void)r3d_view_value_text(l, l->data[si], val, sizeof val);
    int k = snprintf(out + at, cap - at, "\n%s.%s  %s", l->group, l->name, val);
    if (k < 0) break;
    at += (size_t)k < cap - at ? (size_t)k : cap - at - 1u;
    lines++;
  }
  return lines;
}

/* ===================== headless shot ===================================== */

/* Apply a "group.name,group.name,..." show list: everything else is hidden. */
static int apply_show(r3d_view_packet *p, const char *list) {
  for (uint32_t i = 0; i < p->count; i++) p->layer[i].show = false;
  const char *s = list;
  while (*s) {
    const char *comma = strchr(s, ',');
    size_t n = comma ? (size_t)(comma - s) : strlen(s);
    char spec[192];
    if (n >= sizeof spec) n = sizeof spec - 1u;
    memcpy(spec, s, n);
    spec[n] = 0;
    if (n) {
      int at = r3d_view_find(p, spec);
      if (at < 0) {
        fprintf(stderr, "view: --show names no layer \"%s\"\n", spec);
        return -1;
      }
      p->layer[at].show = true;
    }
    if (!comma) break;
    s = comma + 1;
  }
  return 0;
}

int r3d_view_select(r3d_view_packet *p, r3d_view_opts *o, const char *show,
                    const char *compare) {
  if (!p || !o) return -1;
  if (compare) {
    const char *comma = strchr(compare, ',');
    if (!comma) {
      fprintf(stderr, "view: --compare needs group.name,group.name\n");
      return -1;
    }
    char a[192], b[192];
    size_t an = (size_t)(comma - compare);
    if (an >= sizeof a) an = sizeof a - 1u;
    memcpy(a, compare, an);
    a[an] = 0;
    snprintf(b, sizeof b, "%s", comma + 1);
    o->cmp_a = r3d_view_find(p, a);
    o->cmp_b = r3d_view_find(p, b);
    if (o->cmp_a < 0 || o->cmp_b < 0) {
      fprintf(stderr, "view: --compare names no layer (%s / %s)\n", a, b);
      return -1;
    }
    if (p->layer[o->cmp_a].kind != p->layer[o->cmp_b].kind) {
      fprintf(stderr, "view: --compare layers are of different kinds\n");
      return -1;
    }
  }
  if (show) return apply_show(p, show);
  /* compare on its own is a mask over the CT, not a mask over every other
   * layer's own rendering: hide the rest unless the caller asked for it */
  if (compare)
    for (uint32_t i = 0; i < p->count; i++) p->layer[i].show = false;
  return 0;
}

int r3d_view_shot(const char *packet, const char *out_png, int axis, int64_t slice,
                  const char *show, const char *compare) {
  if (!packet || !out_png) return -1;
  r3d_view_manifest m = {0};
  if (r3d_view_manifest_load(&m, packet) != 0) {
    fprintf(stderr, "view: %s is not a packet or a view.json manifest\n", packet);
    return -1;
  }
  char dir[900];
  snprintf(dir, sizeof dir, "%s", m.ent[0].path);
  r3d_view_manifest_free(&m);

  r3d_view_packet p = {0};
  if (r3d_view_packet_load(&p, dir) != 0) return -1;
  r3d_view_opts o;
  r3d_view_opts_default(&o);
  o.axis = axis < 0 || axis > 2 ? R3D_VIEW_AXIS_Z : axis;
  uint32_t ns = r3d_view_slice_count(&p, o.axis);
  o.slice = slice < 0 ? ns / 2u : (uint32_t)(slice >= (int64_t)ns ? ns - 1u : slice);

  int rc = r3d_view_select(&p, &o, show, compare);

  uint32_t w = 0, h = 0;
  r3d_view_slice_dims(&p, o.axis, &w, &h);
  uint8_t *rgba = rc == 0 ? malloc((size_t)w * h * 4u) : NULL;
  if (rc == 0 && !rgba) rc = -1;
  if (rc == 0) {
    r3d_view_composite(&p, &o, rgba);
    rc = r3d_png_write_rgba(out_png, rgba, w, h);
    if (rc != 0) fprintf(stderr, "view: cannot write %s\n", out_png);
  }
  if (rc == 0) {
    r3d_view_cmp cm;
    printf("view: %s axis %c slice %u -> %s (%ux%u)\n", dir, "zyx"[o.axis], o.slice, out_png, w,
           h);
    if (r3d_view_compare(&p, &o, &cm) == 0)
      printf("view: compare agree=%llu a_only=%llu b_only=%llu dice=%.4f\n",
             (unsigned long long)cm.agree, (unsigned long long)cm.a_only,
             (unsigned long long)cm.b_only, cm.dice);
  }
  free(rgba);
  r3d_view_packet_free(&p);
  return rc;
}

/* ===================== live protocol client ==============================
 * TCP, little-endian, one request at a time, 4-byte magics (spec/view.md):
 *   request  'TSV1' u32 hdr_len  JSON
 *   response 'TSVR' u32 hdr_len  JSON  then the layer bytes, in order
 *   error    'TSVE' u32 msg_len  UTF-8 text
 * Mirrors tools/surf/surfserver.py's framing (see core/surfpred.c). */

static int lv_connect(const char *host, int port) {
  char portstr[16];
  snprintf(portstr, sizeof portstr, "%d", port);
  struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM};
  struct addrinfo *res = NULL;
  if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) return -1;
  int fd = -1;
  for (struct addrinfo *a = res; a; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) return -1;
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  struct timeval rto = {.tv_sec = 300}, sto = {.tv_sec = 60};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof rto);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sto, sizeof sto);
  return fd;
}

static int lv_io(int fd, void *buf, size_t n, bool wr) {
  uint8_t *p = buf;
  while (n) {
    ssize_t k = wr ? send(fd, p, n, MSG_NOSIGNAL) : read(fd, p, n);
    if (k <= 0) return -1;
    p += (size_t)k;
    n -= (size_t)k;
  }
  return 0;
}

static void lv_put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
static uint32_t lv_get32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

#define LV_MAX_HDR (1u << 22) /* 4 MiB of JSON is far more than any manifest */

static void lv_err(char *err, size_t cap, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void lv_err(char *err, size_t cap, const char *fmt, ...) {
  if (!err || cap == 0) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, cap, fmt, ap);
  va_end(ap);
}

int r3d_view_fetch(const char *host, int port, const int64_t origin[3], const int64_t dims[3],
                   const char *want, const char *tta, r3d_view_packet *out,
                   r3d_view_hello *hello, char *err, size_t errcap) {
  if (err && errcap) err[0] = 0;
  if (out) { memset(out, 0, sizeof *out); out->ct = -1; }
  bool is_hello = origin == NULL || dims == NULL;

  char req[512];
  int rn;
  if (is_hello) {
    rn = snprintf(req, sizeof req, "{\"hello\": true}");
  } else {
    /* the want list arrives as "student,ct" and goes out as a JSON array */
    char arr[256];
    size_t at = 0;
    arr[at++] = '[';
    const char *s = want && *want ? want : "student";
    while (*s && at + 4u < sizeof arr) {
      const char *comma = strchr(s, ',');
      size_t n = comma ? (size_t)(comma - s) : strlen(s);
      if (n && at + n + 4u < sizeof arr) {
        if (at > 1u) arr[at++] = ',';
        arr[at++] = '"';
        memcpy(arr + at, s, n);
        at += n;
        arr[at++] = '"';
      }
      if (!comma) break;
      s = comma + 1;
    }
    arr[at++] = ']';
    arr[at] = 0;
    rn = snprintf(req, sizeof req,
                  "{\"origin_zyx\": [%lld, %lld, %lld], \"dims_zyx\": [%lld, %lld, %lld], "
                  "\"want\": %s, \"tta\": \"%s\"}",
                  (long long)origin[0], (long long)origin[1], (long long)origin[2],
                  (long long)dims[0], (long long)dims[1], (long long)dims[2], arr,
                  tta && *tta ? tta : "none");
  }
  if (rn < 0 || rn >= (int)sizeof req) {
    lv_err(err, errcap, "request too long");
    return -1;
  }

  int fd = lv_connect(host, port);
  if (fd < 0) {
    lv_err(err, errcap, "cannot connect to %s:%d", host, port);
    return -1;
  }
  uint8_t hdr[8];
  memcpy(hdr, "TSV1", 4);
  lv_put32(hdr + 4, (uint32_t)rn);
  int rc = -1;
  char *json = NULL;
  if (lv_io(fd, hdr, 8, true) != 0 || lv_io(fd, req, (size_t)rn, true) != 0) {
    lv_err(err, errcap, "send failed");
    goto done;
  }
  if (lv_io(fd, hdr, 8, false) != 0) {
    lv_err(err, errcap, "no reply from %s:%d", host, port);
    goto done;
  }
  uint32_t len = lv_get32(hdr + 4);
  if (len > LV_MAX_HDR) {
    lv_err(err, errcap, "reply header of %u bytes is implausible", len);
    goto done;
  }
  json = malloc((size_t)len + 1u);
  if (!json) { lv_err(err, errcap, "out of memory"); goto done; }
  if (lv_io(fd, json, len, false) != 0) {
    lv_err(err, errcap, "short reply header");
    goto done;
  }
  json[len] = 0;
  if (memcmp(hdr, "TSVE", 4) == 0) {
    lv_err(err, errcap, "server: %s", json);
    goto done;
  }
  if (memcmp(hdr, "TSVR", 4) != 0) {
    lv_err(err, errcap, "bad reply magic %.4s", (const char *)hdr);
    goto done;
  }
  if (hello) {
    memset(hello, 0, sizeof *hello);
    (void)vj_string(json, json + len, "groups", hello->groups, sizeof hello->groups);
    if (hello->groups[0] == 0) { /* a JSON array: keep it verbatim */
      vj_raw(json, json + len, "groups", hello->groups, sizeof hello->groups);
    }
    (void)vj_string(json, json + len, "scroll", hello->scroll, sizeof hello->scroll);
    (void)vj_string(json, json + len, "checkpoint", hello->checkpoint, sizeof hello->checkpoint);
    if (vj_ints(json, json + len, "max_dims_zyx", hello->max_dims, 3) != 0)
      hello->max_dims[0] = hello->max_dims[1] = hello->max_dims[2] = 0;
    hello->valid = true;
  }
  if (out) {
    if (r3d_view_packet_parse(out, json, len, NULL) != 0) {
      lv_err(err, errcap, "reply meta.json is not a usable tsm.view.v1 manifest");
      goto done;
    }
    size_t want_bytes = r3d_view_voxels(out);
    for (uint32_t i = 0; i < out->count; i++) {
      out->layer[i].data = malloc(want_bytes ? want_bytes : 1u);
      if (!out->layer[i].data || lv_io(fd, out->layer[i].data, want_bytes, false) != 0) {
        lv_err(err, errcap, "short layer payload for %s.%s", out->layer[i].group,
               out->layer[i].name);
        r3d_view_packet_free(out);
        goto done;
      }
    }
  }
  rc = 0;
done:
  free(json);
  close(fd);
  return rc;
}

static void *lv_worker(void *arg) {
  r3d_view_live *l = arg;
  pthread_mutex_lock(&l->mu);
  for (;;) {
    while (!l->quit && !l->req_pending) pthread_cond_wait(&l->cv, &l->mu);
    if (l->quit) break;
    bool is_hello = l->req_hello;
    int64_t origin[3], dims[3];
    char want[128], tta[32];
    memcpy(origin, l->req_origin, sizeof origin);
    memcpy(dims, l->req_dims, sizeof dims);
    snprintf(want, sizeof want, "%s", l->req_want);
    snprintf(tta, sizeof tta, "%s", l->req_tta);
    uint32_t gen = l->req_gen;
    l->req_pending = false;
    l->busy = true;
    snprintf(l->status, sizeof l->status, is_hello ? "hello %s:%d..." : "predicting on %s:%d...",
             l->host, l->port);
    pthread_mutex_unlock(&l->mu);

    r3d_view_packet got = {0};
    r3d_view_hello hello = {0};
    char err[256] = "";
    int rc = r3d_view_fetch(l->host, l->port, is_hello ? NULL : origin, is_hello ? NULL : dims,
                            want, tta, is_hello ? NULL : &got, &hello, err, sizeof err);

    pthread_mutex_lock(&l->mu);
    l->busy = false;
    if (rc != 0) {
      snprintf(l->status, sizeof l->status, "%s", err[0] ? err : "request failed");
      r3d_view_packet_free(&got);
    } else if (is_hello) {
      l->hello = hello;
      snprintf(l->status, sizeof l->status, "server %s:%d groups %s", l->host, l->port,
               hello.groups[0] ? hello.groups : "?");
    } else {
      r3d_view_packet_free(&l->res);
      l->res = got;
      memset(&got, 0, sizeof got);
      l->res_fresh = true;
      l->res_gen = gen;
      snprintf(l->status, sizeof l->status, "%u layer%s from %s:%d", l->res.count,
               l->res.count == 1u ? "" : "s", l->host, l->port);
    }
  }
  pthread_mutex_unlock(&l->mu);
  return NULL;
}

int r3d_view_live_start(r3d_view_live *l, const char *hostport) {
  if (!l) return -1;
  memset(l, 0, sizeof *l);
  l->res.ct = -1;
  if (!hostport || !*hostport) return -1;
  const char *colon = strrchr(hostport, ':');
  if (!colon || colon == hostport) {
    fprintf(stderr, "view: --view-serve needs host:port\n");
    return -1;
  }
  size_t hn = (size_t)(colon - hostport);
  if (hn >= sizeof l->host) return -1;
  memcpy(l->host, hostport, hn);
  l->host[hn] = 0;
  l->port = atoi(colon + 1);
  if (l->port <= 0 || l->port > 65535) {
    fprintf(stderr, "view: --view-serve port %s is not a port\n", colon + 1);
    return -1;
  }
  pthread_mutex_init(&l->mu, NULL);
  pthread_cond_init(&l->cv, NULL);
  if (pthread_create(&l->th, NULL, lv_worker, l) != 0) {
    pthread_cond_destroy(&l->cv);
    pthread_mutex_destroy(&l->mu);
    return -1;
  }
  l->th_up = true;
  pthread_mutex_lock(&l->mu);
  l->req_hello = true;
  l->req_pending = true;
  l->req_gen++;
  pthread_cond_signal(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return 0;
}

void r3d_view_live_stop(r3d_view_live *l) {
  if (!l || !l->th_up) return;
  pthread_mutex_lock(&l->mu);
  l->quit = true;
  pthread_cond_signal(&l->cv);
  pthread_mutex_unlock(&l->mu);
  pthread_join(l->th, NULL);
  l->th_up = false;
  r3d_view_packet_free(&l->res);
  pthread_cond_destroy(&l->cv);
  pthread_mutex_destroy(&l->mu);
}

void r3d_view_live_request(r3d_view_live *l, const int64_t origin[3], const int64_t dims[3],
                           const char *want, const char *tta) {
  if (!l || !l->th_up) return;
  pthread_mutex_lock(&l->mu);
  memcpy(l->req_origin, origin, sizeof l->req_origin);
  memcpy(l->req_dims, dims, sizeof l->req_dims);
  snprintf(l->req_want, sizeof l->req_want, "%s", want && *want ? want : "student");
  snprintf(l->req_tta, sizeof l->req_tta, "%s", tta && *tta ? tta : "none");
  l->req_hello = false;
  l->req_pending = true;
  l->req_gen++;
  snprintf(l->status, sizeof l->status, "queued");
  pthread_cond_signal(&l->cv);
  pthread_mutex_unlock(&l->mu);
}

bool r3d_view_live_poll(r3d_view_live *l, r3d_view_packet *out) {
  if (!l || !l->th_up || !out) return false;
  bool got = false;
  pthread_mutex_lock(&l->mu);
  if (l->res_fresh) {
    *out = l->res;
    memset(&l->res, 0, sizeof l->res);
    l->res.ct = -1;
    l->res_fresh = false;
    got = true;
  }
  pthread_mutex_unlock(&l->mu);
  return got;
}

void r3d_view_live_hello(r3d_view_live *l, r3d_view_hello *out) {
  if (!out) return;
  memset(out, 0, sizeof *out);
  if (!l || !l->th_up) return;
  pthread_mutex_lock(&l->mu);
  *out = l->hello;
  pthread_mutex_unlock(&l->mu);
}

void r3d_view_live_status(r3d_view_live *l, char *out, size_t cap, bool *busy) {
  if (!out || cap == 0) return;
  out[0] = 0;
  if (!l || !l->th_up) {
    if (busy) *busy = false;
    return;
  }
  pthread_mutex_lock(&l->mu);
  snprintf(out, cap, "%s", l->status);
  if (busy) *busy = l->busy || l->req_pending;
  pthread_mutex_unlock(&l->mu);
}
