/* TSM model-view packets (spec/view.md).
 *
 * A view packet is one CT crop plus any number of named uint8 layers — student
 * heads, teacher outputs, label stores, human bands — each tagged with a *kind*
 * that says how to decode and draw it.  Unlike an annotation packet
 * (core/annot.h) the layer list is data-driven: meta.json's `layers[]` is the
 * authority and the viewer builds its panel from it before touching a file.
 *
 * Everything here is pure CPU state: no GPU, no SDL, no ImGui.  src/viewui.c
 * is a thin ImGui shell over it and `--view-shot` drives exactly the same
 * compositor without a window. */
#ifndef R3D_VIEW_H
#define R3D_VIEW_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- kinds ------------------------------------------------------------- */

typedef enum r3d_view_kind {
  R3D_VIEW_CT = 0,   /* u/255, grayscale base image */
  R3D_VIEW_SDF,      /* (u-128)*clip/127 voxels; u == 0 means "no data" */
  R3D_VIEW_PROB,     /* u/255 */
  R3D_VIEW_SIGNED,   /* (u-127.5)/127.5 */
  R3D_VIEW_DENSITY,  /* u/1000 sheets per voxel */
  R3D_VIEW_COUNT,    /* u, an integer voxel count */
  R3D_VIEW_CLASS,    /* u, names from the layer's palette */
  R3D_VIEW_NKIND
} r3d_view_kind;

extern const char *const r3d_view_kind_name[R3D_VIEW_NKIND];
/* Returns the kind, or -1 when the string names no kind in the spec. */
int r3d_view_kind_parse(const char *s);

/* Slice axis: the packet's own z/y/x. */
enum { R3D_VIEW_AXIS_Z = 0, R3D_VIEW_AXIS_Y = 1, R3D_VIEW_AXIS_X = 2 };

#define R3D_VIEW_MAX_CLASS 16u  /* palette entries kept per class layer */
#define R3D_VIEW_MAX_LAYERS 128u /* a packet with more is rejected; the compositor
                                  * keeps its per-layer scratch on the stack so it
                                  * can stay allocation-free */

/* ---- layers ------------------------------------------------------------ */

typedef struct r3d_view_layer {
  char file[256];  /* as written in meta.json, relative to the packet dir */
  char name[64];
  char group[64];
  r3d_view_kind kind;
  double clip;                            /* sdf: required, voxels */
  char palette[R3D_VIEW_MAX_CLASS][32];   /* class: value names */
  uint32_t npalette;
  char vec[32];                           /* signed: triple name, "" = none */
  int axis;                               /* signed vec: 0/1/2 = x/y/z -> R/G/B */
  uint8_t *data;                          /* Z*Y*X, owned */

  /* display state, seeded by r3d_view_layer_defaults */
  bool show;
  bool solo;
  float color[3];   /* 0..1 */
  float opacity;    /* 0..1 */
  float threshold;  /* prob (probability), density (sheets/vox), count (int) */
  float tol;        /* sdf zero-crossing half-width, voxels */
  bool heatmap;     /* sdf: also draw the signed blue/red heatmap */
} r3d_view_layer;

/* ---- packets ----------------------------------------------------------- */

typedef struct r3d_view_packet {
  char dir[1024];   /* "" for a packet that arrived over the wire */
  uint32_t nz, ny, nx;
  int64_t origin[3]; /* origin_zyx: packet voxel (0,0,0) in scroll space */
  double voxel_um;
  char scroll[128];
  char provenance[2048]; /* verbatim JSON text of "provenance", or "" */
  r3d_view_layer *layer;
  uint32_t count, cap;
  int ct;               /* index of the kind-ct layer */
} r3d_view_packet;

static inline size_t r3d_view_voxels(const r3d_view_packet *p) {
  return (size_t)p->nz * (size_t)p->ny * (size_t)p->nx;
}

void r3d_view_layer_defaults(r3d_view_layer *l, uint32_t ordinal);

/* Load <dir>/meta.json and every layer file it lists.  Hard errors: a bad or
 * absent meta.json, a missing layer file, a file whose size is not Z*Y*X, no
 * kind-ct layer, a duplicate group.name, an sdf layer without `clip`.
 * Unknown keys anywhere are ignored. */
int r3d_view_packet_load(r3d_view_packet *p, const char *dir);

/* Parse a meta.json body already in memory.  `dir` may be NULL, in which case
 * no layer file is read and every layer's `data` is left NULL for the caller
 * to fill (this is how the live protocol's TSVR header is consumed). */
int r3d_view_packet_parse(r3d_view_packet *p, const char *json, size_t n, const char *dir);

void r3d_view_packet_free(r3d_view_packet *p);

/* Index of the layer named "<group>.<name>" (or a bare "<name>" when it is
 * unambiguous), or -1. */
int r3d_view_find(const r3d_view_packet *p, const char *spec);

/* Move every layer of `src` into `dst`, replacing same group.name entries and
 * appending the rest; `src` is emptied (its layer data is now owned by dst).
 * When the dims differ, `dst` is freed and wholly replaced by `src`.
 * Returns 0 on success. */
int r3d_view_merge(r3d_view_packet *dst, r3d_view_packet *src);

/* ---- manifest ---------------------------------------------------------- */

typedef struct r3d_view_entry {
  char path[768];
  char name[256];
  int64_t origin[3];
  uint32_t dims[3];
} r3d_view_entry;

typedef struct r3d_view_manifest {
  char path[1024]; /* the view.json actually read, or "" for a bare dir */
  r3d_view_entry *ent;
  uint32_t count;
} r3d_view_manifest;

/* `path` may be a view.json, a directory containing one, or a single packet
 * directory (one holding meta.json), which yields one entry. */
int r3d_view_manifest_load(r3d_view_manifest *m, const char *path);
void r3d_view_manifest_free(r3d_view_manifest *m);

/* ---- decoding ---------------------------------------------------------- */

/* The spec's decode column.  `no_data` is set for an sdf byte of 0. */
double r3d_view_decode(const r3d_view_layer *l, uint8_t u, bool *no_data);
/* Human-readable decoded value ("0.42", "3 vox", "recto", "-- (no data)"). */
int r3d_view_value_text(const r3d_view_layer *l, uint8_t u, char *out, size_t cap);

/* ---- slices ------------------------------------------------------------ */

/* Slice geometry for `axis`: w/h of the image and the number of slices. */
void r3d_view_slice_dims(const r3d_view_packet *p, int axis, uint32_t *w, uint32_t *h);
uint32_t r3d_view_slice_count(const r3d_view_packet *p, int axis);
/* Packet voxel under image pixel (u,v) of slice `s`. */
void r3d_view_voxel(const r3d_view_packet *p, int axis, uint32_t s, uint32_t u, uint32_t v,
                    uint32_t zyx[3]);
/* One layer byte at image pixel (u,v) of slice `s`. */
uint8_t r3d_view_sample(const r3d_view_packet *p, const r3d_view_layer *l, int axis,
                        uint32_t s, uint32_t u, uint32_t v);

/* ---- compositing ------------------------------------------------------- */

typedef struct r3d_view_opts {
  int axis;
  uint32_t slice;
  float ct_gain;
  /* compare mode: two layer indices of the same kind, or -1 */
  int cmp_a, cmp_b;
  int cmp_class; /* class compare: the class that counts as "present", or -1
                  * for "any non-zero" */
  float cmp_col[3][3]; /* agree / A-only / B-only, 0..1 */
} r3d_view_opts;

void r3d_view_opts_default(r3d_view_opts *o);

/* Composite the slice into `rgba` (w*h*4, row-major, u fastest).  Pure CPU and
 * allocation-free — the interactive mode uploads the result as an ImGui
 * texture, `--view-shot` writes it as a PNG and the tests read it directly. */
void r3d_view_composite(const r3d_view_packet *p, const r3d_view_opts *o, uint8_t *rgba);

typedef struct r3d_view_cmp {
  uint64_t agree, a_only, b_only;
  double dice; /* 2*agree / (2*agree + a_only + b_only); 0 when both are empty */
} r3d_view_cmp;

/* Per-slice agreement counts for the compare pair.  Returns -1 when the pair
 * is unset or the two layers are of different kinds. */
int r3d_view_compare(const r3d_view_packet *p, const r3d_view_opts *o, r3d_view_cmp *out);

/* Hover readout: one "group.name  value" line per visible layer, plus the
 * scroll-space coordinate.  Returns the number of lines written. */
uint32_t r3d_view_hover(const r3d_view_packet *p, const r3d_view_opts *o, uint32_t u,
                        uint32_t v, char *out, size_t cap);

/* ---- headless shot ----------------------------------------------------- */

/* `--view-shot`: load, composite, write an RGBA PNG.  No window, no GPU.
 * `show` is a comma-separated group.name list (NULL = the packet defaults);
 * `compare` is "group.name,group.name". */
int r3d_view_shot(const char *packet, const char *out_png, int axis, int64_t slice,
                  const char *show, const char *compare);

/* ---- live protocol client (tsm serve) ---------------------------------- */

/* Blocking one-shot: connect to host:port, send one 'TSV1' request and decode
 * the 'TSVR' response into `out` (or the 'TSVE' text into `err`).  Returns 0
 * on success.  `hello` sends {"hello": true} and yields a layer-less packet
 * whose server facts land in the r3d_view_hello fields. */
typedef struct r3d_view_hello {
  char groups[256];
  char scroll[128];
  char checkpoint[512];
  int64_t max_dims[3];
  bool valid;
} r3d_view_hello;

int r3d_view_fetch(const char *host, int port, const int64_t origin[3],
                   const int64_t dims[3], const char *want, const char *tta,
                   r3d_view_packet *out, r3d_view_hello *hello, char *err, size_t errcap);

typedef struct r3d_view_live {
  char host[256];
  int port;

  pthread_t th;
  bool th_up;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  bool quit;

  /* request (caller under mu; last write wins) */
  bool req_pending, req_hello;
  int64_t req_origin[3], req_dims[3];
  char req_want[128], req_tta[32];
  uint32_t req_gen;

  /* result (worker under mu; taken by r3d_view_live_poll) */
  bool res_fresh;
  uint32_t res_gen;
  r3d_view_packet res;
  bool busy;
  char status[256];
  r3d_view_hello hello;
} r3d_view_live;

/* `hostport` is "host:port" ("" or NULL disables).  Starts the worker and
 * queues the hello handshake.  Returns 0 on success. */
int r3d_view_live_start(r3d_view_live *l, const char *hostport);
void r3d_view_live_stop(r3d_view_live *l);
void r3d_view_live_request(r3d_view_live *l, const int64_t origin[3], const int64_t dims[3],
                           const char *want, const char *tta);
/* True once per fresh result; the packet is moved into *out (caller frees). */
bool r3d_view_live_poll(r3d_view_live *l, r3d_view_packet *out);
/* Snapshot of the worker's status line (safe from the UI thread). */
void r3d_view_live_status(r3d_view_live *l, char *out, size_t cap, bool *busy);
/* Snapshot of the server's hello facts; `valid` is false until it lands. */
void r3d_view_live_hello(r3d_view_live *l, r3d_view_hello *out);

#endif /* R3D_VIEW_H */
