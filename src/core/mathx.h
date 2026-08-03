/* Header-only float vector math for camera/ray work. Deliberately minimal:
 * only what the renderer uses. Column-major conventions where matrices appear
 * (none yet — the raycaster works from a camera basis, not matrix inverses). */
#ifndef R3D_MATHX_H
#define R3D_MATHX_H

#include <math.h>

typedef struct r3d_v3 {
  float x, y, z;
} r3d_v3;

static inline r3d_v3 v3(float x, float y, float z) { return (r3d_v3){x, y, z}; }
static inline r3d_v3 v3_add(r3d_v3 a, r3d_v3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline r3d_v3 v3_sub(r3d_v3 a, r3d_v3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline r3d_v3 v3_scale(r3d_v3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline float v3_dot(r3d_v3 a, r3d_v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline r3d_v3 v3_cross(r3d_v3 a, r3d_v3 b) {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float v3_len(r3d_v3 a) { return sqrtf(v3_dot(a, a)); }
static inline r3d_v3 v3_norm(r3d_v3 a) {
  float l = v3_len(a);
  return l > 0.0f ? v3_scale(a, 1.0f / l) : v3(0.0f, 0.0f, 0.0f);
}

static inline float fclampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
static inline float flerpf(float a, float b, float t) { return a + (b - a) * t; }

#endif /* R3D_MATHX_H */
