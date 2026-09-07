/* TSM two-face annotation packets.
 *
 * A packet is a directory holding raw uint8 volumes in (z,y,x) C order (x
 * fastest), all with the identical dims given by meta.json.  render3d paints
 * a `correction.u8` of the same shape next to them; the TSM trainer reads it
 * back as a per-voxel override of the exported labels.  See spec/annot.md for
 * the on-disk contract the exporter must match.
 *
 * Everything in this header is pure CPU state: no GPU, no SDL, no ImGui.  The
 * interactive mode (src/annotui.c) is a thin shell over it, and the
 * `--annot-apply` CLI drives exactly the same code the mouse does. */
#ifndef R3D_ANNOT_H
#define R3D_ANNOT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* correction.u8 classes. 0 means "the exporter's label stands"; 4 is a
 * positive assertion that neither face is here (distinct from 0, which is
 * merely the absence of an opinion). */
enum {
  R3D_ANNOT_UNTOUCHED = 0,
  R3D_ANNOT_IN = 1,
  R3D_ANNOT_OUT = 2,
  R3D_ANNOT_IGNORE = 3,
  R3D_ANNOT_ERASE = 4,
  R3D_ANNOT_NCLASS = 5
};
extern const char *const r3d_annot_class_name[R3D_ANNOT_NCLASS];

/* Packet layers, in load order. CT is mandatory; the rest are optional and
 * left NULL when the file is absent. */
enum {
  R3D_ANNOT_L_CT = 0,
  R3D_ANNOT_L_FACES_IN,
  R3D_ANNOT_L_FACES_OUT,
  R3D_ANNOT_L_IGNORE,
  R3D_ANNOT_L_SOURCE,
  R3D_ANNOT_L_RV_CLASS,
  R3D_ANNOT_L_PRED_IN,
  R3D_ANNOT_L_PRED_OUT,
  R3D_ANNOT_NLAYER
};
/* File basenames without the ".u8" suffix, indexed by the enum above. */
extern const char *const r3d_annot_layer_file[R3D_ANNOT_NLAYER];

typedef struct r3d_annot_packet {
  char dir[1024];
  uint32_t nz, ny, nx;
  int64_t origin[3]; /* origin_zyx: packet voxel (0,0,0) in scroll space */
  double voxel_um;
  uint8_t *layer[R3D_ANNOT_NLAYER]; /* NULL = absent */
  uint8_t *correction;              /* always allocated (zeros when new) */
  char *meta_json;                  /* verbatim meta.json text (for done) */
  bool done;
  bool dirty;                       /* correction differs from disk */
  uint64_t counts[R3D_ANNOT_NCLASS];
} r3d_annot_packet;

static inline size_t r3d_annot_voxels(const r3d_annot_packet *p) {
  return (size_t)p->nz * (size_t)p->ny * (size_t)p->nx;
}

/* ---- manifest ---------------------------------------------------------- */

typedef struct r3d_annot_entry {
  char path[768];   /* absolute packet directory */
  char name[256];   /* display label (the manifest's relative path) */
  int64_t origin[3];
  uint32_t dims[3]; /* z,y,x; 0 when the manifest omits them */
} r3d_annot_entry;

typedef struct r3d_annot_manifest {
  char path[1024]; /* the packet.json actually read, or "" for a bare dir */
  r3d_annot_entry *ent;
  uint32_t count;
} r3d_annot_manifest;

/* `path` may be a packet.json, a directory containing one, or a single packet
 * directory (a directory holding meta.json) — the last yields one entry.
 * Returns 0 on success. */
int r3d_annot_manifest_load(r3d_annot_manifest *m, const char *path);
void r3d_annot_manifest_free(r3d_annot_manifest *m);

/* ---- packet I/O -------------------------------------------------------- */

int r3d_annot_packet_load(r3d_annot_packet *p, const char *dir);
void r3d_annot_packet_free(r3d_annot_packet *p);

/* correction.u8, written to a temp file in the same directory and renamed —
 * a killed run never leaves a half-written correction. Clears p->dirty. */
int r3d_annot_correction_save(r3d_annot_packet *p);

/* Set meta.json's "done" flag, preserving every other key verbatim (the value
 * token is rewritten in place, or the key inserted). Atomic like the above. */
int r3d_annot_set_done(r3d_annot_packet *p, bool done);

void r3d_annot_recount(r3d_annot_packet *p);

/* ---- strokes ----------------------------------------------------------- */

typedef enum r3d_annot_op {
  R3D_ANNOT_OP_BRUSH = 0, /* capsule sweep of `radius` through the points */
  R3D_ANNOT_OP_LINE = 1,  /* 1-voxel polyline through the points */
  R3D_ANNOT_OP_UNDO = 2   /* only meaningful in a stroke script */
} r3d_annot_op;

typedef struct r3d_annot_stroke {
  r3d_annot_op op;
  uint8_t cls;     /* 0..4; 0 erases back to "untouched" */
  int32_t radius;  /* brush radius in voxels (ignored by LINE) */
  int32_t depth;   /* paints z-depth .. z+depth (0 = this slice only) */
  int32_t z;
  const int32_t *pts; /* npts interleaved x,y pairs */
  uint32_t npts;
} r3d_annot_stroke;

/* Per-stroke undo journal: each entry stores only the voxels a stroke
 * actually changed, so 20+ levels over a 512^3 packet cost nothing. */
typedef struct r3d_annot_undo r3d_annot_undo;
r3d_annot_undo *r3d_annot_undo_create(uint32_t levels);
void r3d_annot_undo_destroy(r3d_annot_undo *u);
uint32_t r3d_annot_undo_depth(const r3d_annot_undo *u);
/* Group several r3d_annot_apply calls into ONE undo level — an interactive
 * drag paints incrementally as the mouse moves but must undo as one stroke.
 * Outside a begin/end pair every apply is its own level. */
void r3d_annot_undo_begin(r3d_annot_undo *u);
void r3d_annot_undo_end(r3d_annot_undo *u);

/* Apply one stroke. `undo` may be NULL. Returns the number of voxels changed
 * (0 when the stroke was a no-op), or -1 on a bad argument. */
int64_t r3d_annot_apply(r3d_annot_packet *p, const r3d_annot_stroke *s,
                        r3d_annot_undo *undo);
/* Revert the newest journal entry. Returns voxels restored, 0 when empty. */
int64_t r3d_annot_undo_pop(r3d_annot_packet *p, r3d_annot_undo *u);

/* Parse a stroke script: {"strokes":[{"op":"brush","class":1,"z":4,
 * "radius":2,"depth":0,"points":[[x,y],...]}, {"op":"undo"}, ...]}.
 * Point arrays are owned by *out and freed by r3d_annot_strokes_free. */
int r3d_annot_strokes_load(const char *path, r3d_annot_stroke **out, uint32_t *count);
void r3d_annot_strokes_free(r3d_annot_stroke *s, uint32_t count);

/* Run a whole script (honouring "undo" entries) against a packet. */
int r3d_annot_run_strokes(r3d_annot_packet *p, const r3d_annot_stroke *s,
                          uint32_t count, r3d_annot_undo *undo);

/* ---- slice compositing ------------------------------------------------- */

/* Layer visibility bits for r3d_annot_view.show. */
enum {
  R3D_ANNOT_SHOW_FACES = 1u << 0,
  R3D_ANNOT_SHOW_IGNORE = 1u << 1,
  R3D_ANNOT_SHOW_RV = 1u << 2,
  R3D_ANNOT_SHOW_PRED = 1u << 3,
  R3D_ANNOT_SHOW_CORRECTION = 1u << 4,
  R3D_ANNOT_SHOW_ALL = 0x1fu
};

typedef struct r3d_annot_view {
  uint32_t show;      /* R3D_ANNOT_SHOW_* bits */
  float ct_gain;      /* CT intensity multiplier (1 = as stored) */
  float label_alpha;  /* exporter-label blend weight */
  float corr_alpha;   /* correction blend weight */
  const uint8_t (*lut)[4]; /* optional 256-entry transfer LUT for the CT */
} r3d_annot_view;

void r3d_annot_view_default(r3d_annot_view *v);

/* Composite slice z into `rgba` (p->nx * p->ny * 4, row-major, x fastest).
 * Pure CPU and allocation-free: the interactive mode hands the result to
 * ImGui as a texture, and the tests check it directly. */
void r3d_annot_composite(const r3d_annot_packet *p, uint32_t z,
                         const r3d_annot_view *v, uint8_t *rgba);

#endif /* R3D_ANNOT_H */
