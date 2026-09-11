/* Surface files. The surface-compressor container (.sfc) is render3d's
 * default surface type; tifxyz directories (Volume Cartographer interchange:
 * x/y/z.tif + meta.json) remain readable everywhere and are re-encoded to
 * .sfc on first use. Conversion streams TIFF bands through bounded memory,
 * keeps meta.json verbatim as the container metadata, stores every other
 * single-image TIFF in the directory as an exact auxiliary channel, and
 * publishes the output atomically (temporary sibling + rename). */
#ifndef R3D_SURFCONV_H
#define R3D_SURFCONV_H
#include "core/tifxyz.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
  R3D_SURF_NONE = 0, /* neither a .sfc file nor a tifxyz directory */
  R3D_SURF_SFC,      /* surface-compressor container */
  R3D_SURF_TIFXYZ    /* directory holding meta.json (+ x/y/z.tif) */
} r3d_surf_kind;
r3d_surf_kind r3d_surf_kind_of(const char *path);

/* Maximum Euclidean XYZ error (voxels) for new encodes: R3D_SFC_ERROR or 0.1,
 * the surface-compressor tifxyz default. */
double r3d_surf_default_error(void);

/* Called between block rows; return nonzero to cancel (encode fails, no
 * output is left behind). May be NULL. */
typedef int (*r3d_surf_progress)(void *ud, uint64_t done, uint64_t total);

/* tifxyz dir -> .sfc. Returns 0, -1 on a read/encode failure, -2 when the
 * output cannot be created (unwritable location). Never overwrites a
 * finished file except through the final rename. */
int r3d_surf_encode(const char *tifxyz_dir, const char *sfc_path, double error,
                    r3d_surf_progress cb, void *ud);
/* .sfc -> tifxyz dir (x/y/z.tif float32 LZW, meta.json, auxiliary channels
 * as <name>.tif). Refuses an existing destination. */
int r3d_surf_decode(const char *sfc_path, const char *tifxyz_dir,
                    r3d_surf_progress cb, void *ud);

/* Whole-surface loads into the in-memory grid every consumer uses.
 * r3d_surf_load accepts either kind without converting anything. */
int r3d_surf_load_sfc(const char *sfc_path, r3d_tifxyz *out);
int r3d_surf_load(const char *path, r3d_tifxyz *out);

/* The .sfc a tifxyz directory converts to: <dir>.sfc, with a trailing
 * ".tifxyz" replaced (foo.tifxyz -> foo.sfc). A .sfc path maps to itself. */
int r3d_surf_sibling(const char *path, char *out, size_t n);
/* Resolve any surface path to a .sfc, encoding a tifxyz directory when its
 * sibling .sfc is missing or older than the TIFF planes / meta.json. Falls
 * back to cache/sfc/<basename>.sfc when the sibling location is read-only.
 * Returns 0 with out set, -1 otherwise. */
int r3d_surf_resolve(const char *path, double error, r3d_surf_progress cb,
                     void *ud, char *out, size_t n);
#endif
