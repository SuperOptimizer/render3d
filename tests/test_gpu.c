/* GPU conformance test (ctest label: gpu — needs the real GPU + a display).
 * Renders a 64^3 synthetic volume via the actual binary (--frames 1 --shot,
 * MODE_FLAT: compositing without shading) and compares against a CPU
 * reference raymarcher that replicates the shader math bit-for-bit modulo
 * filtering precision. Usage: test_gpu <path-to-render3d> */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core/camera.h"
#include "core/mathx.h"

#define N 64
static uint8_t vol[N][N][N]; /* [z][y][x] */

static uint8_t gen_shells(float x, float y, float z) {
  float r = sqrtf(x * x + y * y + z * z);
  float s = 0.5f + 0.5f * sinf(r * 40.0f);
  float fade = r < 0.9f ? 1.0f : 0.0f;
  return (uint8_t)(255.0f * s * s * fade);
}

/* --- replicate shader helpers --- */
static float hash12(uint32_t px, uint32_t py, uint32_t frame) {
  uint32_t h = px * 374761393u + py * 668265263u + frame * 2246822519u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return (float)(h >> 8) * (1.0f / 16777216.0f);
}

static float sample_volume(float px, float py, float pz) {
  /* GPU trilinear, clamp-to-edge, normalized coords, R8_UNORM */
  float tx = fclampf(px * N - 0.5f, 0.0f, N - 1.0f);
  float ty = fclampf(py * N - 0.5f, 0.0f, N - 1.0f);
  float tz = fclampf(pz * N - 0.5f, 0.0f, N - 1.0f);
  int x0 = (int)tx, y0 = (int)ty, z0 = (int)tz;
  int x1 = x0 + 1 < N ? x0 + 1 : N - 1, y1 = y0 + 1 < N ? y0 + 1 : N - 1,
      z1 = z0 + 1 < N ? z0 + 1 : N - 1;
  float fx = tx - (float)x0, fy = ty - (float)y0, fz = tz - (float)z0;
#define V(Z, Y, X) ((float)vol[Z][Y][X] / 255.0f)
  float c00 = flerpf(V(z0, y0, x0), V(z0, y0, x1), fx);
  float c10 = flerpf(V(z0, y1, x0), V(z0, y1, x1), fx);
  float c01 = flerpf(V(z1, y0, x0), V(z1, y0, x1), fx);
  float c11 = flerpf(V(z1, y1, x0), V(z1, y1, x1), fx);
#undef V
  return flerpf(flerpf(c00, c10, fy), flerpf(c01, c11, fy), fz);
}

static float sample_tf_alpha(float v) {
  /* default TF: identity ramp in all channels; 256-texel 1D linear sample */
  float t = fclampf(v * 256.0f - 0.5f, 0.0f, 255.0f);
  int i0 = (int)t, i1 = i0 + 1 < 256 ? i0 + 1 : 255;
  return flerpf((float)i0 / 255.0f, (float)i1 / 255.0f, t - (float)i0);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_gpu <render3d-binary>\n");
    return 77; /* skip */
  }

  for (int z = 0; z < N; z++)
    for (int y = 0; y < N; y++)
      for (int x = 0; x < N; x++) {
        float inv = 2.0f / (N - 1);
        vol[z][y][x] = gen_shells((float)x * inv - 1, (float)y * inv - 1, (float)z * inv - 1);
      }

  char volpath[] = "/tmp/r3d_gpu_vol_XXXXXX", ppmpath[64];
  int fd = mkstemp(volpath);
  if (fd < 0 || write(fd, vol, sizeof vol) != (ssize_t)sizeof vol) return 1;
  close(fd);
  snprintf(ppmpath, sizeof ppmpath, "%s.ppm", volpath);

  char cmd[512];
  snprintf(cmd, sizeof cmd, "%s %s %d %d %d --frames 1 --mode 5 --shot %s >/dev/null 2>&1",
           argv[1], volpath, N, N, N, ppmpath);
  if (system(cmd) != 0) {
    fprintf(stderr, "test_gpu: render command failed\n");
    return 1;
  }

  FILE *f = fopen(ppmpath, "rb");
  if (!f) {
    fprintf(stderr, "test_gpu: no screenshot\n");
    return 1;
  }
  unsigned w = 0, h = 0, maxv = 0;
  if (fscanf(f, "P6 %u %u %u", &w, &h, &maxv) != 3 || maxv != 255) return 1;
  fgetc(f); /* single whitespace after header */
  uint8_t *img = malloc((size_t)w * h * 3);
  if (!img || fread(img, 1, (size_t)w * h * 3, f) != (size_t)w * h * 3) return 1;
  fclose(f);

  /* CPU reference march (MODE_FLAT: no shading) */
  r3d_camera cam;
  r3d_camera_init(&cam, v3(0.5f, 0.5f, -1.5f));
  r3d_v3 right, up, fwd;
  r3d_camera_basis(&cam, (float)w / (float)h, &right, &up, &fwd);

  double sum_err = 0, max_err = 0;
  uint64_t npix = 0;
  for (uint32_t py = 0; py < h; py += 3) {   /* sample grid: every 3rd pixel */
    for (uint32_t px = 0; px < w; px += 3) {
      float u = ((float)px + 0.5f) / (float)w, vv = ((float)py + 0.5f) / (float)h;
      float nx = u * 2 - 1, ny = 1 - vv * 2;
      r3d_v3 dir = v3_norm(
          v3_add(fwd, v3_add(v3_scale(right, nx), v3_scale(up, ny))));
      r3d_v3 o = cam.pos;
      /* slab test [0,1]^3 */
      float tn = -1e30f, tf_ = 1e30f;
      const float od[3] = {o.x, o.y, o.z}, dd[3] = {dir.x, dir.y, dir.z};
      for (int a = 0; a < 3; a++) {
        float t0 = (0.0f - od[a]) / dd[a], t1 = (1.0f - od[a]) / dd[a];
        float lo = t0 < t1 ? t0 : t1, hi = t0 < t1 ? t1 : t0;
        if (lo > tn) tn = lo;
        if (hi < tf_) tf_ = hi;
      }
      float bg[3] = {0.02f, 0.02f, 0.03f};
      float accr = 0, acca = 0;
      if (tn < 0) tn = 0;
      if (tn < tf_) {
        float voxel = 1.0f / N, dt = voxel;
        float t = tn + hash12(px, py, 0) * dt;
        for (int steps = 0; t < tf_ && steps < 4096; steps++, t += dt) {
          float v = sample_volume(o.x + t * dir.x, o.y + t * dir.y, o.z + t * dir.z);
          float a = sample_tf_alpha(v); /* pow(1-a, 1) == 1-a */
          if (a > 0.003f) {
            float col = sample_tf_alpha(v); /* identity ramp: rgb == alpha curve */
            accr += (1 - acca) * a * col;
            acca += (1 - acca) * a;
            if (acca >= 0.98f) break;
          }
        }
      }
      float ref = accr + (1 - acca) * bg[0];
      float got = (float)img[((size_t)py * w + px) * 3] / 255.0f;
      double err = fabs((double)ref - (double)got) * 255.0;
      sum_err += err;
      if (err > max_err) max_err = err;
      npix++;
    }
  }
  free(img);
  unlink(volpath);
  unlink(ppmpath);

  double mean = sum_err / (double)npix;
  printf("test_gpu: %llu pixels, mean err %.3f LSB, max err %.1f LSB\n",
         (unsigned long long)npix, mean, max_err);
  /* trilinear subtexel precision + fast-math divergence budget */
  if (mean > 2.0 || max_err > 64.0) {
    fprintf(stderr, "test_gpu: FAIL tolerance (mean<=2.0, max<=64)\n");
    return 1;
  }
  printf("test_gpu: ok\n");
  return 0;
}
