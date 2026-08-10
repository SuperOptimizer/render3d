/* tifxyz segment surfaces (Volume Cartographer / vc3d interchange format).
 * A directory holds x.tif, y.tif, z.tif — three single-channel float32 TIFFs
 * of identical W×H; pixel (i,j) of the triplet is the volume-space position
 * (in full-resolution voxels) of surface grid point (i,j) — plus meta.json
 * whose required "scale" [sx,sy] maps grid indices to voxel arc length:
 * grid_index = nominal_voxels * scale (0.05 → one grid step ≈ 20 voxels).
 * Invalid grid points are (-1,-1,-1); per vc3d, any loaded point with z <= 0
 * is forced invalid (files commonly store 0s at unmapped points). */
#ifndef R3D_TIFXYZ_H
#define R3D_TIFXYZ_H

#include <stdint.h>

typedef struct r3d_tifxyz {
  uint32_t w, h;   /* grid dims (x.tif width/height) */
  float sx, sy;    /* meta.json scale */
  float *xyz;      /* w*h*3 floats, x,y,z interleaved; invalid = -1,-1,-1 */
  float bbox[2][3]; /* min/max over valid points (x,y,z); 0s if none */
  uint64_t nvalid;
} r3d_tifxyz;

/* Load <dir>/{x,y,z}.tif + <dir>/meta.json. Returns 0 on success, -1 on
 * error (message on stderr). Accepts classic and BigTIFF, striped or tiled,
 * any libtiff-supported compression; samples must be 32-bit float. */
int r3d_tifxyz_load(r3d_tifxyz *s, const char *dir);
void r3d_tifxyz_free(r3d_tifxyz *s);

static inline const float *r3d_tifxyz_at(const r3d_tifxyz *s, uint32_t i, uint32_t j) {
  return s->xyz + ((uint64_t)j * s->w + i) * 3;
}
static inline int r3d_tifxyz_valid(const float *p) { return p[0] != -1.0f; }

#endif /* R3D_TIFXYZ_H */
