#include "core/camera.h"

#define PITCH_LIMIT 1.55334303f /* π/2 - 0.0175 (1°) */

void r3d_camera_init(r3d_camera *c, r3d_v3 pos) {
  c->pos = pos;
  c->yaw = 0.0f;
  c->pitch = 0.0f;
  c->fov_y = 1.04719755f; /* 60° */
}

static r3d_v3 forward_of(const r3d_camera *c) {
  float cp = cosf(c->pitch);
  return v3(sinf(c->yaw) * cp, sinf(c->pitch), cosf(c->yaw) * cp);
}

void r3d_camera_move(r3d_camera *c, r3d_v3 local, float dist) {
  r3d_v3 fwd = forward_of(c);
  r3d_v3 right = v3_norm(v3_cross(fwd, v3(0, 1, 0)));
  if (v3_len(right) == 0.0f) right = v3(1, 0, 0); /* looking straight up/down */
  r3d_v3 up = v3(0, 1, 0);                        /* world-up flying */
  r3d_v3 d = v3_add(v3_add(v3_scale(right, local.x), v3_scale(up, local.y)),
                    v3_scale(fwd, local.z));
  c->pos = v3_add(c->pos, v3_scale(d, dist));
}

void r3d_camera_look(r3d_camera *c, float dyaw, float dpitch) {
  c->yaw += dyaw;
  c->pitch = fclampf(c->pitch + dpitch, -PITCH_LIMIT, PITCH_LIMIT);
}

void r3d_camera_basis(const r3d_camera *c, float aspect, r3d_v3 *right, r3d_v3 *up,
                      r3d_v3 *forward) {
  r3d_v3 fwd = forward_of(c);
  r3d_v3 r = v3_norm(v3_cross(fwd, v3(0, 1, 0)));
  if (v3_len(r) == 0.0f) r = v3(1, 0, 0);
  r3d_v3 u = v3_cross(r, fwd);
  float t = tanf(c->fov_y * 0.5f);
  *right = v3_scale(r, t * aspect);
  *up = v3_scale(u, t);
  *forward = fwd;
}
