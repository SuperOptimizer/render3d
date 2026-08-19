/* 3D voxel labelling: a sparse per-brick class-id volume the user paints in
 * the GUI (papyrus / ink / not ink / recto / verso / ...), persisted
 * losslessly as one C5L1 label brick per 128^3 block (c5d label codec).
 * Class 0 is "unlabelled"; painting class 0 erases. */
#ifndef R3D_LABELVOL_H
#define R3D_LABELVOL_H

#include <stdbool.h>
#include <stdint.h>

#define R3D_LBL_BRICK 128u
#define R3D_LBL_NCLASS 9u   /* incl. class 0 = unlabelled */
#define R3D_LBL_MAXLEV 12u  /* LOD gen pyramid depth (covers any brick grid) */

extern const char *const r3d_lbl_class_name[R3D_LBL_NCLASS];
extern const float r3d_lbl_class_rgb[R3D_LBL_NCLASS][3];

typedef struct r3d_labelvol {
  uint32_t dim[3]; /* volume voxels (x, y, z) */
  uint32_t nlev;   /* gen-pyramid levels (level 0 = the paint grid) */
  uint32_t lnb[R3D_LBL_MAXLEV][3]; /* per-level brick grid */
  uint8_t **data;  /* level-0 bricks (128^3 u8 class ids), NULL = all zero */
  uint32_t *gens[R3D_LBL_MAXLEV]; /* per-level edit counters (0 = untouched) */
  uint32_t *saved; /* level-0 gen at the last save */
  uint64_t nvox[R3D_LBL_NCLASS]; /* labelled voxel count per class */
  uint64_t edits;  /* total paint ops that changed something */
} r3d_labelvol;

int r3d_labelvol_init(r3d_labelvol *lv, const uint32_t dim[3]);
void r3d_labelvol_free(r3d_labelvol *lv);

/* Sphere brush: set class cls (0 erases) within `radius` voxels of p (world
 * voxel coordinates). Returns the number of voxels that changed. */
uint64_t r3d_labelvol_paint(r3d_labelvol *lv, const double p[3], double radius, uint8_t cls);

/* Edit counter for one LOD brick: level-0 is the paint grid; level L covers
 * 2^L^3 level-0 voxels per voxel (renderer LOD bricks). 0 = region has no
 * labels. Cheap (array lookup) — safe to poll every frame. */
uint32_t r3d_labelvol_gen(const r3d_labelvol *lv, uint32_t level, uint32_t bx, uint32_t by,
                          uint32_t bz);

/* Fill out[128^3] with the class ids of one LOD brick (level > 0 is
 * stride-sampled from the paint grid). */
void r3d_labelvol_fetch(const r3d_labelvol *lv, uint32_t level, uint32_t bx, uint32_t by,
                        uint32_t bz, uint8_t *out);

uint32_t r3d_labelvol_dirty(const r3d_labelvol *lv); /* bricks with unsaved edits */

/* dir holds manifest.json + b_<bx>_<by>_<bz>.c5l label bricks. Save writes
 * only bricks edited since the last save (and unlinks emptied ones) to
 * unique temp files, then publishes manifest.json last; a brick or the
 * manifest that fails to publish stays dirty and is retried on the next
 * save (return -1, but whatever did land is durable). Load requires
 * manifest.json to name this exact volume's dims (refuses otherwise),
 * decodes each brick into scratch storage and validates it before ever
 * touching the live brick, and returns 0 on a valid dir with zero bricks
 * (nothing painted yet). Either call leaves the live volume fully intact
 * on failure. */
int r3d_labelvol_save(r3d_labelvol *lv, const char *dir);
int r3d_labelvol_load(r3d_labelvol *lv, const char *dir);

#endif /* R3D_LABELVOL_H */
