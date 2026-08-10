/* 2x2 multi-view layout + ortho plane-view math (pure, header-only —
 * unit-tested in test_quick). vc3d-style viewer: top-left = flattened
 * segment, top-right = XY, bottom-left = XZ, bottom-right = YZ.
 *
 * A plane view is an axis-aligned orthographic camera over two world axes
 * (u,v) at a slice along the third (the view normal). Screen x maps to +u,
 * screen y (down) maps to +v — the CT image convention vc3d uses. All world
 * coordinates are full-resolution voxels; the renderer's normalized-box
 * conversion (divide by max dim) happens only when building FrameParams. */
#ifndef R3D_MVIEW_H
#define R3D_MVIEW_H

#include <stdint.h>

enum { R3D_MV_SEG = 0, R3D_MV_XY = 1, R3D_MV_XZ = 2, R3D_MV_YZ = 3 };

/* world axis indices (x=0,y=1,z=2) per plane view: u (screen right),
 * v (screen down), n (view normal). SEG uses grid coords, not these. */
static const uint8_t r3d_mv_axes[4][3] = {
    {0, 1, 2}, /* SEG placeholder renders as an XY overview */
    {0, 1, 2}, /* XY: u=x v=y n=z */
    {0, 2, 1}, /* XZ: u=x v=z n=y */
    {1, 2, 0}, /* YZ: u=y v=z n=x */
};

typedef struct r3d_mview {
  double cu, cv;  /* view center (world voxels along u,v) */
  double zoom;    /* screen pixels per world voxel */
  double slice;   /* world voxel coordinate along the normal */
  int32_t px, py; /* quadrant origin in window pixels */
  int32_t pw, ph; /* quadrant size in window pixels */
} r3d_mview;

/* Fixed equal quadrants covering the whole w x h drawable (right/bottom
 * absorb odd pixels). Order: SEG, XY, XZ, YZ. */
static inline void r3d_mv_layout(r3d_mview v[4], int32_t w, int32_t h) {
  int32_t w0 = w / 2, h0 = h / 2;
  int32_t rect[4][4] = {{0, 0, w0, h0},
                        {w0, 0, w - w0, h0},
                        {0, h0, w0, h - h0},
                        {w0, h0, w - w0, h - h0}};
  for (int i = 0; i < 4; i++) {
    v[i].px = rect[i][0];
    v[i].py = rect[i][1];
    v[i].pw = rect[i][2];
    v[i].ph = rect[i][3];
  }
}

/* view containing window pixel (x,y); -1 if outside every quadrant */
static inline int r3d_mv_hit(const r3d_mview v[4], float x, float y) {
  for (int i = 0; i < 4; i++)
    if (x >= (float)v[i].px && x < (float)(v[i].px + v[i].pw) && y >= (float)v[i].py &&
        y < (float)(v[i].py + v[i].ph))
      return i;
  return -1;
}

/* window pixel -> world (u,v) in this view */
static inline void r3d_mv_unproject(const r3d_mview *v, float x, float y, double *u,
                                    double *w) {
  *u = v->cu + ((double)x - ((double)v->px + (double)v->pw * 0.5)) / v->zoom;
  *w = v->cv + ((double)y - ((double)v->py + (double)v->ph * 0.5)) / v->zoom;
}

/* world (u,v) -> window pixel */
static inline void r3d_mv_project(const r3d_mview *v, double u, double w, float *x,
                                  float *y) {
  *x = (float)(((double)v->px + (double)v->pw * 0.5) + (u - v->cu) * v->zoom);
  *y = (float)(((double)v->py + (double)v->ph * 0.5) + (w - v->cv) * v->zoom);
}

/* zoom about a window-pixel anchor: the world point under the cursor stays
 * put. factor > 1 zooms in. */
static inline void r3d_mv_zoom(r3d_mview *v, float x, float y, double factor,
                               double zmin, double zmax) {
  double u, w;
  r3d_mv_unproject(v, x, y, &u, &w);
  double nz = v->zoom * factor;
  if (nz < zmin) nz = zmin;
  if (nz > zmax) nz = zmax;
  v->cu = u - ((double)x - ((double)v->px + (double)v->pw * 0.5)) / nz;
  v->cv = w - ((double)y - ((double)v->py + (double)v->ph * 0.5)) / nz;
  v->zoom = nz;
}

#endif /* R3D_MVIEW_H */
