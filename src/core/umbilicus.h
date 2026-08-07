/* Umbilicus control-point storage. Coordinates are full-resolution voxels in
 * x,y,z order in memory and in the JSON object fields. The writer emits both
 * Villa's `control_points` key and Volume Cartographer's `points` key so one
 * annotation can be consumed by either workflow. */
#ifndef R3D_UMBILICUS_H
#define R3D_UMBILICUS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct r3d_umbilicus_point {
  double x, y, z;
} r3d_umbilicus_point;

typedef struct r3d_umbilicus {
  r3d_umbilicus_point *points;
  size_t count, capacity;
  bool dirty;
} r3d_umbilicus;

void r3d_umbilicus_init(r3d_umbilicus *u);
void r3d_umbilicus_free(r3d_umbilicus *u);

/* Exact-slice lookup. Points are maintained in ascending z order. */
const r3d_umbilicus_point *r3d_umbilicus_find(const r3d_umbilicus *u, double z);

/* Insert or replace the point at z. */
int r3d_umbilicus_set(r3d_umbilicus *u, double x, double y, double z);
bool r3d_umbilicus_remove(r3d_umbilicus *u, double z);

/* Load current Villa JSON ({"control_points":[{x,y,z},...]}) or the Volume
 * Cartographer `points`/root-array forms. Array points [z,y,x] are accepted.
 * load returns 1 when path does not exist, 0 on success, and -1 on error. */
int r3d_umbilicus_load(r3d_umbilicus *u, const char *path);

/* Atomic JSON save (temporary file + rename). */
int r3d_umbilicus_save(r3d_umbilicus *u, const char *path, const char *source,
                       uint32_t nz, uint32_t ny, uint32_t nx);

#endif /* R3D_UMBILICUS_H */
