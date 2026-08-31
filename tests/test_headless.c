#include "render3d/headless.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int sheet_sample(void *user, const double xyz[3], double *value) {
  (void)user;
  *value = 255.0 * exp(-xyz[2] * xyz[2]);
  return 0;
}

static int cancel_now(void *user) {
  (void)user;
  return 1;
}

int main(void) {
  assert(r3d_headless_abi_version() == R3D_HEADLESS_ABI_VERSION);
  float xyz[4u * 4u * 3u];
  for (uint32_t j = 0; j < 4; ++j)
    for (uint32_t i = 0; i < 4; ++i) {
      size_t k = ((size_t)j * 4u + i) * 3u;
      xyz[k] = (float)i * 2.0F;
      xyz[k + 1u] = (float)j * 3.0F;
      xyz[k + 2u] = 5.0F;
    }

  const uint8_t metadata[] = "{\"format\":\"tifxyz\",\"scale\":[0.5,0.333333333]}\n";
  r3d_headless_bytes encoded = {0};
  assert(r3d_headless_tfx1_encode_v1(4, 4, xyz, metadata, sizeof metadata - 1u,
                                      -1, NULL, NULL, &encoded) == R3D_HEADLESS_OK);
  assert(encoded.size > 28u);
  r3d_headless_surface decoded = {0};
  assert(r3d_headless_tfx1_decode_v1(encoded.data, encoded.size, NULL, NULL,
                                     &decoded) == R3D_HEADLESS_OK);
  assert(decoded.width == 4u && decoded.height == 4u);
  assert(memcmp(decoded.xyz, xyz, sizeof xyz) == 0);
  assert(decoded.metadata_size == sizeof metadata - 1u);

  char temp[] = "/tmp/render3d-headless-XXXXXX";
  char *root = mkdtemp(temp);
  assert(root != NULL);
  assert(rmdir(root) == 0);
  assert(r3d_headless_tifxyz_save_v1(root, &decoded, NULL) == R3D_HEADLESS_OK);
  assert(r3d_headless_tifxyz_save_v1(root, &decoded, NULL) == R3D_HEADLESS_E_EXISTS);
  r3d_headless_surface loaded = {0};
  assert(r3d_headless_tifxyz_load_v1(root, NULL, NULL, &loaded) == R3D_HEADLESS_OK);
  assert(memcmp(loaded.xyz, xyz, sizeof xyz) == 0);

  float uv[4u * 4u * 2u];
  r3d_headless_flatten_stats fs = {0};
  assert(r3d_headless_flatten_v1(xyz, 4, 4, 2.5, 8, NULL, uv, &fs) ==
         R3D_HEADLESS_OK);
  assert(fs.vertex_count == 16u && fs.triangle_count == 18u);
  assert(fs.final_energy <= fs.initial_energy + 1e-9);
  r3d_headless_grid_audit audit = {0};
  assert(r3d_headless_grid_audit_v1(xyz, uv, 4, 4, 1e-9, NULL, &audit) ==
         R3D_HEADLESS_OK);
  assert(audit.valid_triangles == 18u && audit.flipped_uv_triangles == 0u);

  float render[16];
  uint8_t valid[16];
  assert(r3d_headless_rasterize_grid_v1(xyz, 4, 4, 6.0, 1.0,
                                         R3D_HEADLESS_RASTER_MAX, sheet_sample,
                                         NULL, NULL, render, valid) == R3D_HEADLESS_OK);
  assert(valid[0] == 255u && render[0] > 254.0F);

  uint8_t mesh_confidence[16], mesh_provenance[16];
  for (size_t i = 0u; i < 16u; ++i) {
    mesh_confidence[i] = (uint8_t)(100u + i);
    mesh_provenance[i] = (uint8_t)(1u + i);
  }
  r3d_headless_uv_rasterizer *uv_rasterizer = NULL;
  assert(r3d_headless_uv_rasterizer_open_v1(
      xyz, uv, mesh_confidence, mesh_provenance, 4u, 4u, 1.0,
      (double)uv[0], (double)uv[1], 7u, 10u, NULL, &uv_rasterizer) ==
      R3D_HEADLESS_OK);
  float band_xyz[7u * 2u * 3u], band_normal[7u * 2u * 3u];
  uint8_t band_confidence[14], band_provenance[14], band_valid[2] = {0u};
  assert(r3d_headless_uv_rasterize_band_v1(
      uv_rasterizer, 0u, 2u, 1000u, NULL, band_xyz, band_normal,
      band_confidence, band_provenance, band_valid, sizeof band_valid) ==
      R3D_HEADLESS_OK);
  assert((band_valid[0] & 1u) != 0u && band_confidence[0] == 100u &&
         band_provenance[0] == 1u && fabsf(band_xyz[2] - 5.0F) < 1e-6F);
  memset(band_confidence, 9, sizeof band_confidence);
  assert(r3d_headless_uv_rasterize_band_v1(
      uv_rasterizer, 0u, 2u, 1u, NULL, band_xyz, band_normal,
      band_confidence, band_provenance, band_valid, sizeof band_valid) ==
      R3D_HEADLESS_E_RESOURCE_LIMIT);
  assert(band_confidence[0] == 9u);
  r3d_headless_uv_rasterizer_close_v1(uv_rasterizer);

  r3d_headless_trace_config tc = {
      .seed_xyz = {0.0, 0.0, 1.8}, .tangent_u_xyz = {1.0, 0.0, 0.0},
      .tangent_v_xyz = {0.0, 1.0, 0.0}, .grid_step = 1.0,
      .normal_search_radius = 3.0, .normal_search_step = 0.1,
      .minimum_support = 100.0, .width = 3, .height = 3};
  float traced[27];
  uint8_t confidence[9];
  assert(r3d_headless_trace_grid_v1(&tc, sheet_sample, NULL, NULL, traced,
                                     confidence) == R3D_HEADLESS_OK);
  assert(fabsf(traced[2]) < 0.051F && confidence[0] > 250u);
  const r3d_headless_callbacks cancelled = {NULL, cancel_now, NULL};
  float untouched[27];
  uint8_t untouched_conf[9];
  for (size_t i = 0; i < 27; ++i) untouched[i] = 7.0F;
  memset(untouched_conf, 7, sizeof untouched_conf);
  assert(r3d_headless_trace_grid_v1(&tc, sheet_sample, NULL, &cancelled,
                                     untouched, untouched_conf) == R3D_HEADLESS_E_CANCELLED);
  assert(untouched[0] == 7.0F && untouched_conf[0] == 7u);

  const double bases[9] = {1.0, 2.0, 1.8, 3.0, 4.0, -1.2, 5.0, 6.0, 2.4};
  const double normals[9] = {0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, -3.0};
  float traced_points[9];
  uint8_t point_confidence[3];
  assert(r3d_headless_trace_points_v1(
             bases, normals, 3u, 3.0, 0.1, 100.0, sheet_sample, NULL, NULL,
             traced_points, point_confidence) == R3D_HEADLESS_OK);
  assert(fabsf(traced_points[2]) < 0.051F);
  assert(fabsf(traced_points[5]) < 0.051F);
  assert(fabsf(traced_points[8]) < 0.051F);
  assert(point_confidence[0] > 250u && point_confidence[2] > 250u);
  for (size_t i = 0u; i < 9u; ++i) traced_points[i] = 9.0F;
  memset(point_confidence, 9, sizeof point_confidence);
  assert(r3d_headless_trace_points_v1(
             bases, normals, 3u, 3.0, 0.1, 100.0, sheet_sample, NULL,
             &cancelled, traced_points, point_confidence) ==
         R3D_HEADLESS_E_CANCELLED);
  assert(traced_points[0] == 9.0F && point_confidence[0] == 9u);

  r3d_headless_surface_release_v1(&loaded, NULL);
  r3d_headless_surface_release_v1(&decoded, NULL);
  r3d_headless_bytes_release_v1(&encoded, NULL);
  char path[4096];
  const char *names[] = {"x.tif", "y.tif", "z.tif", "meta.json"};
  for (size_t i = 0; i < 4; ++i) {
    const int n = snprintf(path, sizeof path, "%s/%s", root, names[i]);
    assert(n > 0 && (size_t)n < sizeof path);
    assert(unlink(path) == 0);
  }
  assert(rmdir(root) == 0);
  return 0;
}
