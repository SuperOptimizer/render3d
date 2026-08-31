#include "render3d/headless.h"

#include "core/cpuvol.h"
#include "core/flatten.h"

#include <tifxyz.h> /* c5d TFX1/tifxyz codec, not core/tifxyz.h */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct r3d_headless_volume {
  r3d_cpuvol core;
  r3d_headless_allocator allocator;
};

struct r3d_headless_uv_rasterizer {
  const float *xyz;
  const float *uv;
  const uint8_t *confidence;
  const uint8_t *provenance;
  float *normal;
  uint32_t grid_width, grid_height;
  uint32_t raster_width, raster_height;
  double pixel_size, minimum_u, minimum_v;
  r3d_headless_allocator allocator;
};

static void *hl_malloc_default(void *user, size_t size) {
  (void)user;
  return malloc(size);
}

static void hl_free_default(void *user, void *pointer) {
  (void)user;
  free(pointer);
}

static bool hl_allocator(const r3d_headless_allocator *in,
                         r3d_headless_allocator *out) {
  if (in == NULL) {
    *out = (r3d_headless_allocator){NULL, hl_malloc_default, hl_free_default};
    return true;
  }
  if (in->allocate == NULL || in->release == NULL) return false;
  *out = *in;
  return true;
}

static bool hl_cancelled(const r3d_headless_callbacks *cb) {
  return cb != NULL && cb->cancelled != NULL && cb->cancelled(cb->user) != 0;
}

static void hl_progress(const r3d_headless_callbacks *cb, const char *phase,
                        uint64_t done, uint64_t total) {
  if (cb != NULL && cb->progress != NULL) cb->progress(cb->user, phase, done, total);
}

static bool hl_mul(size_t a, size_t b, size_t *out) {
  if (b != 0u && a > SIZE_MAX / b) return false;
  *out = a * b;
  return true;
}

static bool hl_surface_size(uint32_t w, uint32_t h, size_t components,
                            size_t item_size, size_t *out) {
  size_t n = 0u;
  return w != 0u && h != 0u && hl_mul((size_t)w, (size_t)h, &n) &&
         hl_mul(n, components, &n) && hl_mul(n, item_size, out);
}

static bool hl_valid_xyz(const float *p) {
  return p[0] >= 0.0F && isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]);
}

uint32_t r3d_headless_abi_version(void) { return R3D_HEADLESS_ABI_VERSION; }

const char *r3d_headless_status_string(r3d_headless_status status) {
  switch (status) {
    case R3D_HEADLESS_OK: return "ok";
    case R3D_HEADLESS_E_INVALID_ARGUMENT: return "invalid argument";
    case R3D_HEADLESS_E_OUT_OF_MEMORY: return "out of memory";
    case R3D_HEADLESS_E_IO: return "I/O error";
    case R3D_HEADLESS_E_FORMAT: return "invalid or unsupported format";
    case R3D_HEADLESS_E_CANCELLED: return "cancelled";
    case R3D_HEADLESS_E_NUMERIC: return "numeric error";
    case R3D_HEADLESS_E_EXISTS: return "destination already exists";
    case R3D_HEADLESS_E_INTERNAL: return "internal error";
    case R3D_HEADLESS_E_RESOURCE_LIMIT: return "resource limit";
    default: return "unknown status";
  }
}

void r3d_headless_bytes_release_v1(r3d_headless_bytes *bytes,
                                   const r3d_headless_allocator *allocator) {
  r3d_headless_allocator a;
  if (bytes == NULL || !hl_allocator(allocator, &a)) return;
  a.release(a.user, bytes->data);
  *bytes = (r3d_headless_bytes){0};
}

void r3d_headless_surface_release_v1(r3d_headless_surface *surface,
                                     const r3d_headless_allocator *allocator) {
  r3d_headless_allocator a;
  if (surface == NULL || !hl_allocator(allocator, &a)) return;
  a.release(a.user, surface->xyz);
  a.release(a.user, surface->metadata);
  *surface = (r3d_headless_surface){0};
}

static r3d_headless_status hl_surface_from_c5d(const c5d_tifxyz *source,
                                               const r3d_headless_allocator *allocator,
                                               r3d_headless_surface *out) {
  r3d_headless_allocator a;
  size_t xyz_size = 0u;
  if (!hl_allocator(allocator, &a) || out == NULL || source == NULL ||
      !hl_surface_size(source->w, source->h, 3u, sizeof(float), &xyz_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  float *xyz = a.allocate(a.user, xyz_size);
  uint8_t *metadata = source->meta_len == 0u ? NULL : a.allocate(a.user, source->meta_len);
  if (xyz == NULL || (source->meta_len != 0u && metadata == NULL)) {
    a.release(a.user, xyz);
    a.release(a.user, metadata);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  const size_t n = (size_t)source->w * source->h;
  for (size_t k = 0u; k < n; ++k) {
    xyz[k * 3u] = source->plane[0][k];
    xyz[k * 3u + 1u] = source->plane[1][k];
    xyz[k * 3u + 2u] = source->plane[2][k];
  }
  if (source->meta_len != 0u) memcpy(metadata, source->meta, source->meta_len);
  *out = (r3d_headless_surface){source->w, source->h, xyz, metadata, source->meta_len};
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_tfx1_encode_v1(
    uint32_t width, uint32_t height, const float *xyz, const uint8_t *metadata,
    size_t metadata_size, int32_t log2_quantization,
    const r3d_headless_allocator *allocator, const r3d_headless_callbacks *callbacks,
    r3d_headless_bytes *out_bytes) {
  r3d_headless_allocator a;
  size_t xyz_size = 0u;
  if (!hl_allocator(allocator, &a) || out_bytes == NULL || xyz == NULL ||
      (metadata_size != 0u && metadata == NULL) ||
      !hl_surface_size(width, height, 3u, sizeof(float), &xyz_size) ||
      log2_quantization < -1 || log2_quantization > 24)
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  (void)xyz_size;
  if (hl_cancelled(callbacks)) return R3D_HEADLESS_E_CANCELLED;
  const size_t n = (size_t)width * height;
  size_t plane_bytes = 0u;
  if (!hl_mul(n, sizeof(float), &plane_bytes)) return R3D_HEADLESS_E_INVALID_ARGUMENT;
  float *planes = malloc(plane_bytes * 3u);
  if (planes == NULL) return R3D_HEADLESS_E_OUT_OF_MEMORY;
  c5d_tifxyz source = {.w = width, .h = height,
                       .meta = (uint8_t *)metadata, .meta_len = metadata_size};
  for (size_t c = 0u; c < 3u; ++c) source.plane[c] = planes + c * n;
  for (size_t k = 0u; k < n; ++k)
    for (size_t c = 0u; c < 3u; ++c) source.plane[c][k] = xyz[k * 3u + c];
  uint8_t *encoded = NULL;
  size_t encoded_size = 0u;
  hl_progress(callbacks, "tfx1-encode", 0u, 1u);
  const int rc = c5d_tifxyz_encode(&source, (int)log2_quantization, &encoded, &encoded_size);
  free(planes);
  if (rc != 0 || encoded == NULL) {
    free(encoded);
    return R3D_HEADLESS_E_FORMAT;
  }
  if (hl_cancelled(callbacks)) {
    free(encoded);
    return R3D_HEADLESS_E_CANCELLED;
  }
  uint8_t *copy = encoded;
  if (allocator != NULL) {
    copy = a.allocate(a.user, encoded_size);
    if (copy == NULL) {
      free(encoded);
      return R3D_HEADLESS_E_OUT_OF_MEMORY;
    }
    memcpy(copy, encoded, encoded_size);
    free(encoded);
  }
  *out_bytes = (r3d_headless_bytes){copy, encoded_size};
  hl_progress(callbacks, "tfx1-encode", 1u, 1u);
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_tfx1_encode_file_v1(
    const char *path, uint32_t width, uint32_t height, const float *xyz,
    const uint8_t *metadata, size_t metadata_size, int32_t log2_quantization,
    const r3d_headless_callbacks *callbacks) {
  struct stat st;
  if (path == NULL || path[0] == '\0') return R3D_HEADLESS_E_INVALID_ARGUMENT;
  if (lstat(path, &st) == 0) return R3D_HEADLESS_E_EXISTS;
  if (errno != ENOENT) return R3D_HEADLESS_E_IO;
  r3d_headless_bytes encoded = {0};
  r3d_headless_status status = r3d_headless_tfx1_encode_v1(
      width, height, xyz, metadata, metadata_size, log2_quantization,
      NULL, callbacks, &encoded);
  if (status != R3D_HEADLESS_OK) return status;
  const size_t path_size = strlen(path);
  if (path_size > SIZE_MAX - 16u) {
    r3d_headless_bytes_release_v1(&encoded, NULL);
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  }
  char *temporary = malloc(path_size + 16u);
  if (temporary == NULL) {
    r3d_headless_bytes_release_v1(&encoded, NULL);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  (void)snprintf(temporary, path_size + 16u, "%s.tmp.XXXXXX", path);
  const int descriptor = mkstemp(temporary);
  if (descriptor < 0) {
    free(temporary);
    r3d_headless_bytes_release_v1(&encoded, NULL);
    return R3D_HEADLESS_E_IO;
  }
  size_t written = 0u;
  while (written < encoded.size) {
    const ssize_t result = write(
        descriptor, encoded.data + written, encoded.size - written);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) {
      status = R3D_HEADLESS_E_IO;
      break;
    }
    written += (size_t)result;
  }
  if (status == R3D_HEADLESS_OK &&
      (fsync(descriptor) != 0 || hl_cancelled(callbacks))) {
    status = hl_cancelled(callbacks) ? R3D_HEADLESS_E_CANCELLED
                                     : R3D_HEADLESS_E_IO;
  }
  if (close(descriptor) != 0 && status == R3D_HEADLESS_OK)
    status = R3D_HEADLESS_E_IO;
  if (status == R3D_HEADLESS_OK && link(temporary, path) != 0)
    status = errno == EEXIST ? R3D_HEADLESS_E_EXISTS : R3D_HEADLESS_E_IO;
  (void)unlink(temporary);
  free(temporary);
  r3d_headless_bytes_release_v1(&encoded, NULL);
  return status;
}

r3d_headless_status r3d_headless_tfx1_decode_v1(
    const uint8_t *bytes, size_t size, const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_surface *out_surface) {
  if (bytes == NULL || size == 0u || out_surface == NULL)
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  if (hl_cancelled(callbacks)) return R3D_HEADLESS_E_CANCELLED;
  c5d_tifxyz decoded = {0};
  hl_progress(callbacks, "tfx1-decode", 0u, 1u);
  if (c5d_tifxyz_decode(bytes, size, &decoded) != 0) return R3D_HEADLESS_E_FORMAT;
  r3d_headless_status status = hl_cancelled(callbacks)
      ? R3D_HEADLESS_E_CANCELLED
      : hl_surface_from_c5d(&decoded, allocator, out_surface);
  c5d_tifxyz_free(&decoded);
  if (status == R3D_HEADLESS_OK) hl_progress(callbacks, "tfx1-decode", 1u, 1u);
  return status;
}

r3d_headless_status r3d_headless_tifxyz_load_v1(
    const char *directory, const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_surface *out_surface) {
  if (directory == NULL || directory[0] == '\0' || out_surface == NULL)
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  if (hl_cancelled(callbacks)) return R3D_HEADLESS_E_CANCELLED;
  c5d_tifxyz loaded = {0};
  hl_progress(callbacks, "tifxyz-load", 0u, 1u);
  if (c5d_tifxyz_load_dir(directory, &loaded) != 0) return R3D_HEADLESS_E_IO;
  r3d_headless_status status = hl_cancelled(callbacks)
      ? R3D_HEADLESS_E_CANCELLED
      : hl_surface_from_c5d(&loaded, allocator, out_surface);
  c5d_tifxyz_free(&loaded);
  if (status == R3D_HEADLESS_OK) hl_progress(callbacks, "tifxyz-load", 1u, 1u);
  return status;
}

static void hl_remove_stage(const char *path) {
  char child[4096];
  static const char *names[4] = {"x.tif", "y.tif", "z.tif", "meta.json"};
  for (size_t i = 0u; i < 4u; ++i) {
    const int n = snprintf(child, sizeof child, "%s/%s", path, names[i]);
    if (n > 0 && (size_t)n < sizeof child) (void)unlink(child);
  }
  (void)rmdir(path);
}

r3d_headless_status r3d_headless_tifxyz_save_v1(
    const char *directory, const r3d_headless_surface *surface,
    const r3d_headless_callbacks *callbacks) {
  size_t xyz_size = 0u;
  struct stat st;
  if (directory == NULL || directory[0] == '\0' || surface == NULL || surface->xyz == NULL ||
      (surface->metadata_size != 0u && surface->metadata == NULL) ||
      !hl_surface_size(surface->width, surface->height, 3u, sizeof(float), &xyz_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  (void)xyz_size;
  if (lstat(directory, &st) == 0) return R3D_HEADLESS_E_EXISTS;
  if (errno != ENOENT) return R3D_HEADLESS_E_IO;
  if (hl_cancelled(callbacks)) return R3D_HEADLESS_E_CANCELLED;
  char stage[4096];
  const int sn = snprintf(stage, sizeof stage, "%s.tmp.%ld", directory, (long)getpid());
  if (sn < 0 || (size_t)sn >= sizeof stage) return R3D_HEADLESS_E_INVALID_ARGUMENT;
  if (lstat(stage, &st) == 0) return R3D_HEADLESS_E_EXISTS;
  if (errno != ENOENT || mkdir(stage, 0755) != 0) return R3D_HEADLESS_E_IO;
  const size_t n = (size_t)surface->width * surface->height;
  size_t plane_bytes = 0u;
  if (!hl_mul(n, sizeof(float), &plane_bytes)) {
    hl_remove_stage(stage);
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  }
  float *planes = malloc(plane_bytes * 3u);
  if (planes == NULL) {
    hl_remove_stage(stage);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  c5d_tifxyz target = {.w = surface->width, .h = surface->height,
                       .meta = surface->metadata, .meta_len = surface->metadata_size};
  for (size_t c = 0u; c < 3u; ++c) target.plane[c] = planes + c * n;
  for (size_t k = 0u; k < n; ++k)
    for (size_t c = 0u; c < 3u; ++c) target.plane[c][k] = surface->xyz[k * 3u + c];
  hl_progress(callbacks, "tifxyz-save", 0u, 1u);
  int rc = c5d_tifxyz_save_dir(stage, &target);
  free(planes);
  if (rc == 0 && hl_cancelled(callbacks)) {
    hl_remove_stage(stage);
    return R3D_HEADLESS_E_CANCELLED;
  }
  if (rc == 0) rc = rename(stage, directory);
  if (rc != 0) {
    hl_remove_stage(stage);
    return errno == EEXIST || errno == ENOTEMPTY ? R3D_HEADLESS_E_EXISTS
                                                 : R3D_HEADLESS_E_IO;
  }
  hl_progress(callbacks, "tifxyz-save", 1u, 1u);
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_volume_open_v1(
    const char *root, uint32_t cache_bricks, const r3d_headless_allocator *allocator,
    r3d_headless_volume **out_volume) {
  r3d_headless_allocator a;
  if (root == NULL || root[0] == '\0' || cache_bricks == 0u || out_volume == NULL ||
      !hl_allocator(allocator, &a))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  r3d_headless_volume *volume = a.allocate(a.user, sizeof *volume);
  if (volume == NULL) return R3D_HEADLESS_E_OUT_OF_MEMORY;
  memset(volume, 0, sizeof *volume);
  volume->allocator = a;
  if (r3d_cpuvol_open(&volume->core, root, cache_bricks) != 0) {
    a.release(a.user, volume);
    return R3D_HEADLESS_E_IO;
  }
  *out_volume = volume;
  return R3D_HEADLESS_OK;
}

void r3d_headless_volume_close_v1(r3d_headless_volume *volume) {
  if (volume == NULL) return;
  const r3d_headless_allocator a = volume->allocator;
  r3d_cpuvol_close(&volume->core);
  a.release(a.user, volume);
}

r3d_headless_status r3d_headless_volume_read_roi_v1(
    r3d_headless_volume *volume, uint32_t level, int64_t x0, int64_t y0,
    int64_t z0, uint32_t nx, uint32_t ny, uint32_t nz,
    const r3d_headless_callbacks *callbacks, uint8_t *out_zyx) {
  size_t plane = 0u, total = 0u;
  if (volume == NULL || out_zyx == NULL || nx == 0u || ny == 0u || nz == 0u ||
      level >= volume->core.nlev || !hl_mul((size_t)nx, ny, &plane) ||
      !hl_mul(plane, nz, &total))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  uint8_t *temporary = malloc(total);
  if (temporary == NULL) return R3D_HEADLESS_E_OUT_OF_MEMORY;
  for (uint32_t z = 0u; z < nz; ++z) {
    if (hl_cancelled(callbacks)) {
      free(temporary);
      return R3D_HEADLESS_E_CANCELLED;
    }
    r3d_cpuvol_read_block(&volume->core, level, x0, y0, z0 + (int64_t)z,
                          nx, ny, 1u, temporary + (size_t)z * plane);
    hl_progress(callbacks, "volume-read-roi", (uint64_t)z + 1u, nz);
  }
  memcpy(out_zyx, temporary, total);
  free(temporary);
  return R3D_HEADLESS_OK;
}

static bool hl_normalize3(double v[3]) {
  const double length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (!(length > 1e-12) || !isfinite(length)) return false;
  for (size_t c = 0u; c < 3u; ++c) v[c] /= length;
  return true;
}

r3d_headless_status r3d_headless_trace_grid_v1(
    const r3d_headless_trace_config *config, r3d_headless_sample_fn sample,
    void *sample_user, const r3d_headless_callbacks *callbacks,
    float *out_xyz, uint8_t *out_confidence) {
  size_t xyz_size = 0u, confidence_size = 0u;
  if (config == NULL || sample == NULL || out_xyz == NULL || out_confidence == NULL ||
      config->width == 0u || config->height == 0u || !(config->grid_step > 0.0) ||
      !(config->normal_search_radius >= 0.0) || !(config->normal_search_step > 0.0) ||
      !isfinite(config->minimum_support) ||
      !hl_surface_size(config->width, config->height, 3u, sizeof(float), &xyz_size) ||
      !hl_surface_size(config->width, config->height, 1u, sizeof(uint8_t), &confidence_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  double u[3], v[3], normal[3];
  memcpy(u, config->tangent_u_xyz, sizeof u);
  memcpy(v, config->tangent_v_xyz, sizeof v);
  if (!hl_normalize3(u)) return R3D_HEADLESS_E_INVALID_ARGUMENT;
  const double projection = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  for (size_t c = 0u; c < 3u; ++c) v[c] -= projection * u[c];
  if (!hl_normalize3(v)) return R3D_HEADLESS_E_INVALID_ARGUMENT;
  normal[0] = u[1] * v[2] - u[2] * v[1];
  normal[1] = u[2] * v[0] - u[0] * v[2];
  normal[2] = u[0] * v[1] - u[1] * v[0];
  float *xyz = malloc(xyz_size);
  uint8_t *confidence = malloc(confidence_size);
  if (xyz == NULL || confidence == NULL) {
    free(xyz);
    free(confidence);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  const uint64_t total = (uint64_t)config->width * config->height;
  const double cu = ((double)config->width - 1.0) * 0.5;
  const double cv = ((double)config->height - 1.0) * 0.5;
  for (uint64_t index = 0u; index < total; ++index) {
    if ((index & 255u) == 0u && hl_cancelled(callbacks)) {
      free(xyz);
      free(confidence);
      return R3D_HEADLESS_E_CANCELLED;
    }
    const uint32_t i = (uint32_t)(index % config->width);
    const uint32_t j = (uint32_t)(index / config->width);
    double base[3];
    for (size_t c = 0u; c < 3u; ++c)
      base[c] = config->seed_xyz[c] + ((double)i - cu) * config->grid_step * u[c] +
                ((double)j - cv) * config->grid_step * v[c];
    double best = -HUGE_VAL, best_offset = 0.0;
    const uint64_t count = (uint64_t)floor(2.0 * config->normal_search_radius /
                                           config->normal_search_step + 1e-12) + 1u;
    for (uint64_t s = 0u; s < count; ++s) {
      const double offset = -config->normal_search_radius +
                            (double)s * config->normal_search_step;
      double p[3] = {base[0] + offset * normal[0], base[1] + offset * normal[1],
                     base[2] + offset * normal[2]};
      double value = 0.0;
      if (sample(sample_user, p, &value) != 0 || !isfinite(value)) {
        free(xyz);
        free(confidence);
        return R3D_HEADLESS_E_INTERNAL;
      }
      if (value > best) {
        best = value;
        best_offset = offset;
      }
    }
    if (best < config->minimum_support) {
      xyz[index * 3u] = xyz[index * 3u + 1u] = xyz[index * 3u + 2u] = -1.0F;
      confidence[index] = 0u;
    } else {
      for (size_t c = 0u; c < 3u; ++c)
        xyz[index * 3u + c] = (float)(base[c] + best_offset * normal[c]);
      const double clipped = best < 0.0 ? 0.0 : best > 255.0 ? 255.0 : best;
      confidence[index] = (uint8_t)lround(clipped);
    }
    if ((index & 255u) == 255u || index + 1u == total)
      hl_progress(callbacks, "trace-grid", index + 1u, total);
  }
  memcpy(out_xyz, xyz, xyz_size);
  memcpy(out_confidence, confidence, confidence_size);
  free(xyz);
  free(confidence);
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_trace_points_v1(
    const double *base_xyz, const double *normal_xyz, uint64_t point_count,
    double normal_search_radius, double normal_search_step,
    double minimum_support, r3d_headless_sample_fn sample, void *sample_user,
    const r3d_headless_callbacks *callbacks, float *out_xyz,
    uint8_t *out_confidence) {
  size_t point_size = 0u, xyz_size = 0u;
  if (base_xyz == NULL || normal_xyz == NULL || point_count == 0u ||
      point_count > SIZE_MAX || sample == NULL || out_xyz == NULL ||
      out_confidence == NULL || !(normal_search_radius >= 0.0) ||
      !(normal_search_step > 0.0) || !isfinite(normal_search_radius) ||
      !isfinite(normal_search_step) || !isfinite(minimum_support) ||
      !hl_mul((size_t)point_count, sizeof(uint8_t), &point_size) ||
      !hl_mul((size_t)point_count, 3u * sizeof(float), &xyz_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  float *xyz = malloc(xyz_size);
  uint8_t *confidence = malloc(point_size);
  if (xyz == NULL || confidence == NULL) {
    free(xyz);
    free(confidence);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  const double steps_exact = 2.0 * normal_search_radius / normal_search_step;
  if (!isfinite(steps_exact) || steps_exact > (double)(UINT64_MAX - 1u)) {
    free(xyz);
    free(confidence);
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  }
  const uint64_t search_count = (uint64_t)floor(steps_exact + 1e-12) + 1u;
  for (uint64_t index = 0u; index < point_count; ++index) {
    if ((index & 255u) == 0u && hl_cancelled(callbacks)) {
      free(xyz);
      free(confidence);
      return R3D_HEADLESS_E_CANCELLED;
    }
    const double *base = base_xyz + index * 3u;
    double normal[3] = {normal_xyz[index * 3u], normal_xyz[index * 3u + 1u],
                        normal_xyz[index * 3u + 2u]};
    if (!isfinite(base[0]) || !isfinite(base[1]) || !isfinite(base[2]) ||
        !hl_normalize3(normal)) {
      free(xyz);
      free(confidence);
      return R3D_HEADLESS_E_INVALID_ARGUMENT;
    }
    double best = -HUGE_VAL, best_offset = 0.0;
    for (uint64_t search = 0u; search < search_count; ++search) {
      const double offset = -normal_search_radius +
                            (double)search * normal_search_step;
      const double point[3] = {base[0] + offset * normal[0],
                               base[1] + offset * normal[1],
                               base[2] + offset * normal[2]};
      double value = 0.0;
      if (sample(sample_user, point, &value) != 0 || !isfinite(value)) {
        free(xyz);
        free(confidence);
        return R3D_HEADLESS_E_INTERNAL;
      }
      /* On a plateau of equal support keep the sample nearest the seed so a
       * thick or neighbouring sheet inside the window cannot pull the trace
       * to the far end of the search interval. */
      if (value > best ||
          (value == best && fabs(offset) < fabs(best_offset))) {
        best = value;
        best_offset = offset;
      }
    }
    if (best < minimum_support) {
      xyz[index * 3u] = xyz[index * 3u + 1u] = xyz[index * 3u + 2u] = -1.0F;
      confidence[index] = 0u;
    } else {
      for (size_t component = 0u; component < 3u; ++component)
        xyz[index * 3u + component] =
            (float)(base[component] + best_offset * normal[component]);
      const double clipped = best < 0.0 ? 0.0 : best > 255.0 ? 255.0 : best;
      confidence[index] = (uint8_t)lround(clipped);
    }
    if ((index & 255u) == 255u || index + 1u == point_count)
      hl_progress(callbacks, "trace-points", index + 1u, point_count);
  }
  memcpy(out_xyz, xyz, xyz_size);
  memcpy(out_confidence, confidence, point_size);
  free(xyz);
  free(confidence);
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_flatten_v1(
    const float *xyz, uint32_t width, uint32_t height, double target_pitch,
    uint32_t max_iterations, const r3d_headless_callbacks *callbacks,
    float *out_uv, r3d_headless_flatten_stats *out_stats) {
  size_t uv_size = 0u;
  if (xyz == NULL || out_uv == NULL || out_stats == NULL || max_iterations == 0u ||
      !(target_pitch > 0.0) || !isfinite(target_pitch) ||
      !hl_surface_size(width, height, 2u, sizeof(float), &uv_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  if (hl_cancelled(callbacks)) return R3D_HEADLESS_E_CANCELLED;
  float *uv = malloc(uv_size);
  if (uv == NULL) return R3D_HEADLESS_E_OUT_OF_MEMORY;
  r3d_flatten_stats core_stats = {0};
  hl_progress(callbacks, "flatten", 0u, 1u);
  const int rc = r3d_flatten_slim(xyz, width, height, target_pitch,
                                  max_iterations, uv, &core_stats);
  if (rc != 0) {
    free(uv);
    return R3D_HEADLESS_E_NUMERIC;
  }
  if (hl_cancelled(callbacks)) {
    free(uv);
    return R3D_HEADLESS_E_CANCELLED;
  }
  const r3d_headless_flatten_stats stats = {
      core_stats.iters, core_stats.nvert, core_stats.ntri, core_stats.e0,
      core_stats.e1, core_stats.stretch0, core_stats.stretch1};
  memcpy(out_uv, uv, uv_size);
  *out_stats = stats;
  free(uv);
  hl_progress(callbacks, "flatten", 1u, 1u);
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_flatten_resample_v1(
    const float *xyz, const float *uv, uint32_t width, uint32_t height,
    double target_pitch, const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_surface *out_surface) {
  r3d_headless_allocator a;
  size_t input_size = 0u;
  if (xyz == NULL || uv == NULL || out_surface == NULL || !(target_pitch > 0.0) ||
      !hl_allocator(allocator, &a) ||
      !hl_surface_size(width, height, 3u, sizeof(float), &input_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  (void)input_size;
  if (hl_cancelled(callbacks)) return R3D_HEADLESS_E_CANCELLED;
  float *core_xyz = NULL;
  uint32_t ow = 0u, oh = 0u;
  hl_progress(callbacks, "flatten-resample", 0u, 1u);
  if (r3d_flatten_resample(xyz, uv, width, height, target_pitch,
                           &core_xyz, &ow, &oh) != 0)
    return R3D_HEADLESS_E_NUMERIC;
  size_t output_size = 0u;
  if (hl_cancelled(callbacks) ||
      !hl_surface_size(ow, oh, 3u, sizeof(float), &output_size)) {
    free(core_xyz);
    return hl_cancelled(callbacks) ? R3D_HEADLESS_E_CANCELLED
                                   : R3D_HEADLESS_E_NUMERIC;
  }
  float *copy = a.allocate(a.user, output_size);
  if (copy == NULL) {
    free(core_xyz);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  memcpy(copy, core_xyz, output_size);
  free(core_xyz);
  *out_surface = (r3d_headless_surface){ow, oh, copy, NULL, 0u};
  hl_progress(callbacks, "flatten-resample", 1u, 1u);
  return R3D_HEADLESS_OK;
}

static bool hl_grid_normal(const float *xyz, uint32_t w, uint32_t h,
                           uint32_t i, uint32_t j, double normal[3]) {
  const uint32_t il = i == 0u ? i : i - 1u;
  const uint32_t ir = i + 1u < w ? i + 1u : i;
  const uint32_t jt = j == 0u ? j : j - 1u;
  const uint32_t jb = j + 1u < h ? j + 1u : j;
  const float *left = xyz + ((size_t)j * w + il) * 3u;
  const float *right = xyz + ((size_t)j * w + ir) * 3u;
  const float *top = xyz + ((size_t)jt * w + i) * 3u;
  const float *bottom = xyz + ((size_t)jb * w + i) * 3u;
  if (!hl_valid_xyz(left) || !hl_valid_xyz(right) || !hl_valid_xyz(top) ||
      !hl_valid_xyz(bottom)) return false;
  double du[3], dv[3];
  for (size_t c = 0u; c < 3u; ++c) {
    du[c] = (double)right[c] - (double)left[c];
    dv[c] = (double)bottom[c] - (double)top[c];
  }
  normal[0] = du[1] * dv[2] - du[2] * dv[1];
  normal[1] = du[2] * dv[0] - du[0] * dv[2];
  normal[2] = du[0] * dv[1] - du[1] * dv[0];
  return hl_normalize3(normal);
}

r3d_headless_status r3d_headless_uv_rasterizer_open_v1(
    const float *xyz, const float *uv, const uint8_t *confidence,
    const uint8_t *provenance, uint32_t grid_width, uint32_t grid_height,
    double pixel_size, double minimum_u, double minimum_v,
    uint32_t raster_width, uint32_t raster_height,
    const r3d_headless_allocator *allocator,
    r3d_headless_uv_rasterizer **out_rasterizer) {
  r3d_headless_allocator a;
  size_t normal_size = 0u, uv_size = 0u;
  if (xyz == NULL || uv == NULL || confidence == NULL || provenance == NULL ||
      out_rasterizer == NULL || !(pixel_size > 0.0) || !isfinite(pixel_size) ||
      !isfinite(minimum_u) || !isfinite(minimum_v) ||
      grid_width < 2u || grid_height < 2u || raster_width == 0u ||
      raster_height == 0u || !hl_allocator(allocator, &a) ||
      !hl_surface_size(grid_width, grid_height, 3u, sizeof(float), &normal_size) ||
      !hl_surface_size(grid_width, grid_height, 2u, sizeof(float), &uv_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  (void)uv_size;
  r3d_headless_uv_rasterizer *rasterizer =
      a.allocate(a.user, sizeof *rasterizer);
  float *normal = a.allocate(a.user, normal_size);
  if (rasterizer == NULL || normal == NULL) {
    a.release(a.user, rasterizer);
    a.release(a.user, normal);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  const uint64_t count = (uint64_t)grid_width * grid_height;
  for (uint64_t index = 0u; index < count; ++index) {
    double value[3] = {0.0, 0.0, 0.0};
    const uint32_t i = (uint32_t)(index % grid_width);
    const uint32_t j = (uint32_t)(index / grid_width);
    if (hl_grid_normal(xyz, grid_width, grid_height, i, j, value)) {
      for (size_t axis = 0u; axis < 3u; ++axis)
        normal[index * 3u + axis] = (float)value[axis];
    } else {
      normal[index * 3u] = normal[index * 3u + 1u] =
          normal[index * 3u + 2u] = 0.0F;
    }
  }
  *rasterizer = (r3d_headless_uv_rasterizer){
      xyz, uv, confidence, provenance, normal, grid_width, grid_height,
      raster_width, raster_height, pixel_size, minimum_u, minimum_v, a};
  *out_rasterizer = rasterizer;
  return R3D_HEADLESS_OK;
}

void r3d_headless_uv_rasterizer_close_v1(
    r3d_headless_uv_rasterizer *rasterizer) {
  if (rasterizer == NULL) return;
  const r3d_headless_allocator a = rasterizer->allocator;
  a.release(a.user, rasterizer->normal);
  a.release(a.user, rasterizer);
}

static void hl_uv_triangle(
    const r3d_headless_uv_rasterizer *rasterizer, const uint32_t ids[3],
    uint32_t band_y0, uint32_t band_height, uint64_t maximum_pixel_tests,
    uint64_t *pixel_tests, float *xyz, float *normal, uint8_t *confidence,
    uint8_t *provenance, uint8_t *valid, bool *limited) {
  const float *a = rasterizer->uv + (size_t)ids[0] * 2u;
  const float *b = rasterizer->uv + (size_t)ids[1] * 2u;
  const float *c = rasterizer->uv + (size_t)ids[2] * 2u;
  if (!isfinite(a[0]) || !isfinite(a[1]) || !isfinite(b[0]) ||
      !isfinite(b[1]) || !isfinite(c[0]) || !isfinite(c[1])) return;
  const double au = (double)a[0], av = (double)a[1];
  const double bu = (double)b[0], bv = (double)b[1];
  const double cu = (double)c[0], cv = (double)c[1];
  const double determinant = (bu - au) * (cv - av) -
                             (cu - au) * (bv - av);
  if (!(determinant > 1e-12)) return;
  const double eps = 1e-9;
  int64_t x0 = (int64_t)ceil((fmin(au, fmin(bu, cu)) -
                              rasterizer->minimum_u) / rasterizer->pixel_size - eps);
  int64_t x1 = (int64_t)floor((fmax(au, fmax(bu, cu)) -
                               rasterizer->minimum_u) / rasterizer->pixel_size + eps);
  int64_t y0 = (int64_t)ceil((fmin(av, fmin(bv, cv)) -
                              rasterizer->minimum_v) / rasterizer->pixel_size - eps);
  int64_t y1 = (int64_t)floor((fmax(av, fmax(bv, cv)) -
                               rasterizer->minimum_v) / rasterizer->pixel_size + eps);
  if (x0 < 0) x0 = 0;
  if (y0 < (int64_t)band_y0) y0 = band_y0;
  if (x1 >= (int64_t)rasterizer->raster_width) x1 = rasterizer->raster_width - 1u;
  const uint32_t band_y1 = band_y0 + band_height;
  if (y1 >= (int64_t)band_y1) y1 = band_y1 - 1u;
  if (x0 > x1 || y0 > y1) return;
  const uint64_t candidates = (uint64_t)(x1 - x0 + 1) * (uint64_t)(y1 - y0 + 1);
  if (candidates > maximum_pixel_tests - *pixel_tests) {
    *limited = true;
    return;
  }
  *pixel_tests += candidates;
  for (int64_t y = y0; y <= y1; ++y) {
    const double v = rasterizer->minimum_v + (double)y * rasterizer->pixel_size;
    for (int64_t x = x0; x <= x1; ++x) {
      const uint64_t index = (uint64_t)(y - band_y0) *
                                 rasterizer->raster_width + (uint64_t)x;
      const uint8_t bit = (uint8_t)(1u << (index & 7u));
      if ((valid[index >> 3u] & bit) != 0u) continue;
      const double u = rasterizer->minimum_u + (double)x * rasterizer->pixel_size;
      const double w1 = ((u - au) * (cv - av) -
                         (cu - au) * (v - av)) / determinant;
      const double w2 = ((bu - au) * (v - av) -
                         (u - au) * (bv - av)) / determinant;
      const double weights[3] = {1.0 - w1 - w2, w1, w2};
      if (weights[0] < -1e-7 || weights[1] < -1e-7 || weights[2] < -1e-7)
        continue;
      double length_squared = 0.0;
      for (size_t axis = 0u; axis < 3u; ++axis) {
        double p = 0.0, n = 0.0;
        for (size_t corner = 0u; corner < 3u; ++corner) {
          p += weights[corner] *
               (double)rasterizer->xyz[(size_t)ids[corner] * 3u + axis];
          n += weights[corner] *
               (double)rasterizer->normal[(size_t)ids[corner] * 3u + axis];
        }
        xyz[index * 3u + axis] = (float)p;
        normal[index * 3u + axis] = (float)n;
        length_squared += n * n;
      }
      const double length = sqrt(length_squared);
      if (length > 1e-12)
        for (size_t axis = 0u; axis < 3u; ++axis)
          normal[index * 3u + axis] =
              (float)((double)normal[index * 3u + axis] / length);
      else
        normal[index * 3u] = normal[index * 3u + 1u] = normal[index * 3u + 2u] = 0.0F;
      double confidence_value = 0.0;
      size_t provenance_corner = 0u;
      for (size_t corner = 0u; corner < 3u; ++corner) {
        confidence_value += weights[corner] *
                            (double)rasterizer->confidence[ids[corner]];
        if (corner != 0u && weights[corner] > weights[provenance_corner])
          provenance_corner = corner;
      }
      long rounded = lround(confidence_value);
      if (rounded < 0) rounded = 0;
      if (rounded > 255) rounded = 255;
      confidence[index] = (uint8_t)rounded;
      provenance[index] = rasterizer->provenance[ids[provenance_corner]];
      valid[index >> 3u] |= bit;
    }
  }
}

r3d_headless_status r3d_headless_uv_rasterize_band_v1(
    r3d_headless_uv_rasterizer *rasterizer, uint32_t band_y0,
    uint32_t band_height, uint64_t maximum_pixel_tests,
    const r3d_headless_callbacks *callbacks, float *out_xyz,
    float *out_normal, uint8_t *out_confidence, uint8_t *out_provenance,
    uint8_t *out_valid_bits, size_t out_valid_size) {
  if (rasterizer == NULL || out_xyz == NULL || out_normal == NULL ||
      out_confidence == NULL || out_provenance == NULL || out_valid_bits == NULL ||
      band_height == 0u || band_y0 >= rasterizer->raster_height ||
      band_height > rasterizer->raster_height - band_y0 ||
      maximum_pixel_tests == 0u) return R3D_HEADLESS_E_INVALID_ARGUMENT;
  size_t xyz_size = 0u, scalar_size = 0u;
  if (!hl_surface_size(rasterizer->raster_width, band_height, 3u,
                       sizeof(float), &xyz_size) ||
      !hl_surface_size(rasterizer->raster_width, band_height, 1u,
                       sizeof(uint8_t), &scalar_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  const size_t valid_size = scalar_size / 8u + (scalar_size % 8u != 0u ? 1u : 0u);
  if (out_valid_size != valid_size) return R3D_HEADLESS_E_INVALID_ARGUMENT;
  const r3d_headless_allocator a = rasterizer->allocator;
  float *xyz = a.allocate(a.user, xyz_size);
  float *normal = a.allocate(a.user, xyz_size);
  uint8_t *confidence = a.allocate(a.user, scalar_size);
  uint8_t *provenance = a.allocate(a.user, scalar_size);
  uint8_t *valid = a.allocate(a.user, valid_size);
  if (xyz == NULL || normal == NULL || confidence == NULL ||
      provenance == NULL || valid == NULL) {
    a.release(a.user, xyz); a.release(a.user, normal);
    a.release(a.user, confidence); a.release(a.user, provenance);
    a.release(a.user, valid);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  memset(xyz, 0, xyz_size); memset(normal, 0, xyz_size);
  memset(confidence, 0, scalar_size); memset(provenance, 0, scalar_size);
  memset(valid, 0, valid_size);
  uint64_t tests = 0u, triangle = 0u;
  bool limited = false;
  for (uint32_t row = 0u; row + 1u < rasterizer->grid_height; ++row) {
    for (uint32_t column = 0u; column + 1u < rasterizer->grid_width; ++column) {
      if ((triangle & 1023u) == 0u && hl_cancelled(callbacks)) {
        a.release(a.user, xyz); a.release(a.user, normal);
        a.release(a.user, confidence); a.release(a.user, provenance);
        a.release(a.user, valid);
        return R3D_HEADLESS_E_CANCELLED;
      }
      const uint32_t a_index = row * rasterizer->grid_width + column;
      const uint32_t ids[2][3] = {
          {a_index, a_index + 1u, a_index + rasterizer->grid_width},
          {a_index + 1u, a_index + rasterizer->grid_width + 1u,
           a_index + rasterizer->grid_width}};
      for (size_t half = 0u; half < 2u; ++half) {
        hl_uv_triangle(rasterizer, ids[half], band_y0, band_height,
                       maximum_pixel_tests, &tests, xyz, normal, confidence,
                       provenance, valid, &limited);
        ++triangle;
        if (limited) break;
      }
      if (limited) break;
    }
    if (limited) break;
  }
  if (limited || hl_cancelled(callbacks)) {
    a.release(a.user, xyz); a.release(a.user, normal);
    a.release(a.user, confidence); a.release(a.user, provenance);
    a.release(a.user, valid);
    return limited ? R3D_HEADLESS_E_RESOURCE_LIMIT : R3D_HEADLESS_E_CANCELLED;
  }
  memcpy(out_xyz, xyz, xyz_size); memcpy(out_normal, normal, xyz_size);
  memcpy(out_confidence, confidence, scalar_size);
  memcpy(out_provenance, provenance, scalar_size);
  memcpy(out_valid_bits, valid, valid_size);
  a.release(a.user, xyz); a.release(a.user, normal);
  a.release(a.user, confidence); a.release(a.user, provenance);
  a.release(a.user, valid);
  hl_progress(callbacks, "uv-raster-band", band_height, band_height);
  return R3D_HEADLESS_OK;
}

r3d_headless_status r3d_headless_rasterize_grid_v1(
    const float *xyz, uint32_t width, uint32_t height, double span,
    double sample_step, r3d_headless_raster_mode mode,
    r3d_headless_sample_fn sample, void *sample_user,
    const r3d_headless_callbacks *callbacks, float *out_values,
    uint8_t *out_valid) {
  size_t values_size = 0u, valid_size = 0u;
  if (xyz == NULL || sample == NULL || out_values == NULL || out_valid == NULL ||
      !(span >= 0.0) || !(sample_step > 0.0) ||
      (mode != R3D_HEADLESS_RASTER_MEAN && mode != R3D_HEADLESS_RASTER_MAX) ||
      !hl_surface_size(width, height, 1u, sizeof(float), &values_size) ||
      !hl_surface_size(width, height, 1u, sizeof(uint8_t), &valid_size))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  float *values = malloc(values_size);
  uint8_t *valid = malloc(valid_size);
  if (values == NULL || valid == NULL) {
    free(values);
    free(valid);
    return R3D_HEADLESS_E_OUT_OF_MEMORY;
  }
  const uint64_t total = (uint64_t)width * height;
  const uint64_t samples = (uint64_t)floor(2.0 * span / sample_step + 1e-12) + 1u;
  for (uint64_t index = 0u; index < total; ++index) {
    if ((index & 255u) == 0u && hl_cancelled(callbacks)) {
      free(values);
      free(valid);
      return R3D_HEADLESS_E_CANCELLED;
    }
    const uint32_t i = (uint32_t)(index % width);
    const uint32_t j = (uint32_t)(index / width);
    const float *p = xyz + index * 3u;
    double normal[3];
    if (!hl_valid_xyz(p) || !hl_grid_normal(xyz, width, height, i, j, normal)) {
      values[index] = 0.0F;
      valid[index] = 0u;
      continue;
    }
    double aggregate = mode == R3D_HEADLESS_RASTER_MAX ? -HUGE_VAL : 0.0;
    for (uint64_t s = 0u; s < samples; ++s) {
      const double offset = -span + (double)s * sample_step;
      const double q[3] = {(double)p[0] + offset * normal[0],
                           (double)p[1] + offset * normal[1],
                           (double)p[2] + offset * normal[2]};
      double value = 0.0;
      if (sample(sample_user, q, &value) != 0 || !isfinite(value)) {
        free(values);
        free(valid);
        return R3D_HEADLESS_E_INTERNAL;
      }
      if (mode == R3D_HEADLESS_RASTER_MAX) {
        if (value > aggregate) aggregate = value;
      } else {
        aggregate += value;
      }
    }
    values[index] = (float)(mode == R3D_HEADLESS_RASTER_MEAN
                                ? aggregate / (double)samples
                                : aggregate);
    valid[index] = 255u;
    if ((index & 255u) == 255u || index + 1u == total)
      hl_progress(callbacks, "rasterize-grid", index + 1u, total);
  }
  memcpy(out_values, values, values_size);
  memcpy(out_valid, valid, valid_size);
  free(values);
  free(valid);
  return R3D_HEADLESS_OK;
}

static double hl_triangle_area2(const float *a, const float *b, const float *c) {
  const double ux = (double)b[0] - (double)a[0],
               uy = (double)b[1] - (double)a[1],
               uz = (double)b[2] - (double)a[2];
  const double vx = (double)c[0] - (double)a[0],
               vy = (double)c[1] - (double)a[1],
               vz = (double)c[2] - (double)a[2];
  const double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz,
               cz = ux * vy - uy * vx;
  return sqrt(cx * cx + cy * cy + cz * cz);
}

r3d_headless_status r3d_headless_grid_audit_v1(
    const float *xyz, const float *uv, uint32_t width, uint32_t height,
    double degenerate_area_epsilon, const r3d_headless_callbacks *callbacks,
    r3d_headless_grid_audit *out_audit) {
  size_t ignored = 0u;
  if (xyz == NULL || out_audit == NULL || !(degenerate_area_epsilon >= 0.0) ||
      !isfinite(degenerate_area_epsilon) ||
      !hl_surface_size(width, height, 3u, sizeof(float), &ignored))
    return R3D_HEADLESS_E_INVALID_ARGUMENT;
  r3d_headless_grid_audit audit = {0};
  const size_t n = (size_t)width * height;
  for (size_t k = 0u; k < n; ++k) {
    if (hl_valid_xyz(xyz + k * 3u)) ++audit.valid_vertices;
    else ++audit.invalid_vertices;
  }
  const uint64_t total = (uint64_t)(width - 1u) * (height - 1u);
  uint64_t qindex = 0u;
  for (uint32_t j = 0u; j + 1u < height; ++j)
    for (uint32_t i = 0u; i + 1u < width; ++i, ++qindex) {
      if ((qindex & 1023u) == 0u && hl_cancelled(callbacks))
        return R3D_HEADLESS_E_CANCELLED;
      const size_t k[4] = {(size_t)j * width + i, (size_t)j * width + i + 1u,
                           (size_t)(j + 1u) * width + i,
                           (size_t)(j + 1u) * width + i + 1u};
      if (!hl_valid_xyz(xyz + k[0] * 3u) || !hl_valid_xyz(xyz + k[1] * 3u) ||
          !hl_valid_xyz(xyz + k[2] * 3u) || !hl_valid_xyz(xyz + k[3] * 3u)) {
        ++audit.unsupported_quads;
        continue;
      }
      static const uint32_t tri[2][3] = {{0u, 1u, 2u}, {1u, 3u, 2u}};
      for (size_t t = 0u; t < 2u; ++t) {
        const size_t ka = k[tri[t][0]], kb = k[tri[t][1]], kc = k[tri[t][2]];
        const double area2 = hl_triangle_area2(xyz + ka * 3u, xyz + kb * 3u,
                                               xyz + kc * 3u);
        if (!isfinite(area2) || area2 <= 2.0 * degenerate_area_epsilon)
          ++audit.degenerate_triangles;
        else
          ++audit.valid_triangles;
        if (uv != NULL) {
          const float *a = uv + ka * 2u, *b = uv + kb * 2u, *c = uv + kc * 2u;
          const double signed_area2 = ((double)b[0] - (double)a[0]) *
                                          ((double)c[1] - (double)a[1]) -
                                      ((double)b[1] - (double)a[1]) *
                                          ((double)c[0] - (double)a[0]);
          if (!(signed_area2 > 0.0)) ++audit.flipped_uv_triangles;
        }
      }
      if ((qindex & 1023u) == 1023u || qindex + 1u == total)
        hl_progress(callbacks, "grid-audit", qindex + 1u, total);
    }
  *out_audit = audit;
  return R3D_HEADLESS_OK;
}
