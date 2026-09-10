/* Conservative CT ray footprints for pane-cache invalidation. This follows
 * raycast.slang's volume transform, crop box and slab clipping; it does not
 * use camera/frustum guesses from the streaming request heuristic. */
#ifndef R3D_VIEWBOUNDS_H
#define R3D_VIEWBOUNDS_H
#include "render/render_types.h"
#include <math.h>
static inline void r3d_view_dir(const r3d_frame_params *p, const double in[3],
                                double out[3]) {
  for (int a = 0; a < 3; a++)
    out[a] = (double)p->vol_r0[a] * in[0] + (double)p->vol_r1[a] * in[1] +
             (double)p->vol_r2[a] * in[2];
}
/* extent and result are normalized volume coordinates; maxdim converts the
 * voxel-valued slab/crop fields. False means unknown: invalidate
 * conservatively. Perspective views retain the whole clipped volume box.
 * Orthographic views additionally intersect their ray-origin interval and
 * direction with it. Brick gradient/filter samples reuse their selected slot,
 * so they do not introduce reads of a different virtual block beyond this ray
 * footprint. */
static inline bool r3d_view_bounds(const r3d_frame_params *p,
                                   const float extent[3], float maxdim,
                                   float lo[3], float hi[3]) {
  if (!p->brick_mode || (p->view_flags & R3D_VIEW_SURF) || !(maxdim > 0) ||
      !isfinite(maxdim))
    return false;
  const double md = (double)maxdim;
  const double ext[3] = {(double)extent[0], (double)extent[1],
                         (double)extent[2]};
  double boxlo[3] = {0, 0, 0}, boxhi[3];
  for (int a = 0; a < 3; a++) {
    if (!(ext[a] > 0) || !isfinite(ext[a]))
      return false;
    boxhi[a] = ext[a];
  }
  bool lod = (p->brick_mode & 0x20000u) != 0;
  bool oblique = lod && !(p->view_flags & R3D_VIEW_CROP) && p->slab_depth &&
                 (p->view_flags & R3D_VIEW_OBLIQUE);
  if (lod && (p->view_flags & R3D_VIEW_CROP)) {
    const float org[3] = {p->slab_x0, p->slab_y0, p->slab_z0};
    const float edge[3] = {p->slab_px, p->slab_py, p->slab_nx};
    for (int a = 0; a < 3; a++) {
      if (!isfinite(org[a]) || !isfinite(edge[a]) || edge[a] < 0)
        return false;
      boxlo[a] = fmax(0, fmin(ext[a], (double)org[a] / md));
      boxhi[a] =
          fmax(boxlo[a], fmin(ext[a], ((double)org[a] + (double)edge[a]) / md));
    }
  } else if (lod && p->slab_depth && !oblique) {
    uint32_t axis = (p->view_flags >> 8) & 3u;
    int a = axis == 1 ? 0 : (axis == 2 ? 1 : 2);
    if (!isfinite(p->slab_z0))
      return false;
    boxlo[a] = fmax(0, fmin(ext[a], (double)p->slab_z0 / md));
    boxhi[a] =
        fmax(boxlo[a], fmin(ext[a], ((double)p->slab_z0 + p->slab_depth) / md));
  }
  if (p->view_flags & R3D_VIEW_ORTHO) {
    double v[3], origin[3], right[3], up[3], direction[3], originlo[3],
        originhi[3];
    const float tr[3] = {p->vol_tx, p->vol_ty, p->vol_tz};
    for (int a = 0; a < 3; a++)
      v[a] = (double)p->cam_origin[a] - (double)tr[a] - (double)ext[a] * 0.5;
    r3d_view_dir(p, v, origin);
    for (int a = 0; a < 3; a++)
      v[a] = (double)p->cam_right[a];
    r3d_view_dir(p, v, right);
    for (int a = 0; a < 3; a++)
      v[a] = (double)p->cam_up[a];
    r3d_view_dir(p, v, up);
    double len = 0;
    for (int a = 0; a < 3; a++)
      len += (double)p->cam_forward[a] * (double)p->cam_forward[a];
    len = sqrt(len);
    if (!(len > 0) || !isfinite(len))
      return false;
    for (int a = 0; a < 3; a++)
      v[a] = (double)p->cam_forward[a] / len;
    r3d_view_dir(p, v, direction);
    double tlo = 0, thi = HUGE_VAL;
    if (oblique) {
      if (!isfinite(p->slab_z0))
        return false;
      tlo = (double)p->slab_z0 / md;
      thi = ((double)p->slab_z0 + p->slab_depth) / md;
    }
    for (int a = 0; a < 3; a++) {
      double span = fabs(right[a]) + fabs(up[a]);
      origin[a] += (double)ext[a] * 0.5;
      originlo[a] = origin[a] - span;
      originhi[a] = origin[a] + span;
      if (!isfinite(originlo[a]) || !isfinite(originhi[a]) ||
          !isfinite(direction[a]))
        return false;
      if (direction[a] > 0) {
        tlo = fmax(tlo, (boxlo[a] - originhi[a]) / direction[a]);
        thi = fmin(thi, (boxhi[a] - originlo[a]) / direction[a]);
      } else if (direction[a] < 0) {
        tlo = fmax(tlo, (boxhi[a] - originlo[a]) / direction[a]);
        thi = fmin(thi, (boxlo[a] - originhi[a]) / direction[a]);
      } else if (originhi[a] < boxlo[a] || originlo[a] > boxhi[a])
        return false;
    }
    if (!isfinite(tlo) || !isfinite(thi) || tlo > thi)
      return false;
    for (int a = 0; a < 3; a++) {
      double d0 = direction[a] * tlo, d1 = direction[a] * thi;
      boxlo[a] = fmax(boxlo[a], originlo[a] + fmin(d0, d1));
      boxhi[a] = fmin(boxhi[a], originhi[a] + fmax(d0, d1));
    }
  }
  /* Cover float rounding in the shader's matrix products and slab divides. */
  for (int a = 0; a < 3; a++) {
    if (!isfinite(boxlo[a]) || !isfinite(boxhi[a]) || boxlo[a] > boxhi[a])
      return false;
    lo[a] = (float)fmax(0, boxlo[a] - 1e-5);
    hi[a] = (float)fmin(ext[a], boxhi[a] + 1e-5);
  }
  return true;
}
#endif
