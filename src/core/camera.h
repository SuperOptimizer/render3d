/* Free-fly camera. World space = volume-normalized: the volume occupies
 * [0,1]^3 (scaled by aspect ratios later if non-cubic). The raycaster gets a
 * pre-scaled basis (right *= tan(fov/2)*aspect, up *= tan(fov/2)) so the shader
 * does zero trig and no matrix inverses. */
#ifndef R3D_CAMERA_H
#define R3D_CAMERA_H

#include "core/mathx.h"

typedef struct r3d_camera {
  r3d_v3 pos;
  float yaw;   /* radians, 0 = +Z */
  float pitch; /* radians, clamped to ±(π/2 - ε) */
  float fov_y; /* radians */
  /* orbit state (valid while orbit functions are used) */
  r3d_v3 target;
  float dist;
} r3d_camera;

void r3d_camera_init(r3d_camera *c, r3d_v3 pos);
/* move: local axes (right, up, forward), scaled by dist */
void r3d_camera_move(r3d_camera *c, r3d_v3 local, float dist);
void r3d_camera_look(r3d_camera *c, float dyaw, float dpitch);

/* Turntable orbit around `target`. Keeps current yaw/pitch; derives pos. */
void r3d_camera_orbit_set(r3d_camera *c, r3d_v3 target, float dist);
void r3d_camera_orbit_drag(r3d_camera *c, float dyaw, float dpitch);
void r3d_camera_orbit_zoom(r3d_camera *c, float factor); /* dist *= factor */
void r3d_camera_orbit_pan(r3d_camera *c, r3d_v3 local, float dist); /* moves target */
/* out basis is fov/aspect pre-scaled, ready for r3d_frame_params */
void r3d_camera_basis(const r3d_camera *c, float aspect, r3d_v3 *right, r3d_v3 *up,
                      r3d_v3 *forward);

#endif /* R3D_CAMERA_H */
