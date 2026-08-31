/* Stable, append-only headless ABI for render3d.
 *
 * Coordinates passed through this interface are always XYZ voxel coordinates.
 * Dense buffers are C-order ZYX: x is the fastest-varying dimension.  Every
 * function returns a status and leaves caller-visible outputs untouched when
 * it fails or is cancelled.  New entry points are added with a new suffix;
 * existing v1 declarations will not change.
 */
#ifndef RENDER3D_HEADLESS_H
#define RENDER3D_HEADLESS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define R3D_HEADLESS_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define R3D_HEADLESS_API __attribute__((visibility("default")))
#else
#  define R3D_HEADLESS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define R3D_HEADLESS_ABI_VERSION 1u

typedef enum r3d_headless_status {
  R3D_HEADLESS_OK = 0,
  R3D_HEADLESS_E_INVALID_ARGUMENT = 1,
  R3D_HEADLESS_E_OUT_OF_MEMORY = 2,
  R3D_HEADLESS_E_IO = 3,
  R3D_HEADLESS_E_FORMAT = 4,
  R3D_HEADLESS_E_CANCELLED = 5,
  R3D_HEADLESS_E_NUMERIC = 6,
  R3D_HEADLESS_E_EXISTS = 7,
  R3D_HEADLESS_E_INTERNAL = 8,
  R3D_HEADLESS_E_RESOURCE_LIMIT = 9
} r3d_headless_status;

typedef void *(*r3d_headless_malloc_fn)(void *user, size_t size);
typedef void (*r3d_headless_free_fn)(void *user, void *pointer);
typedef struct r3d_headless_allocator {
  void *user;
  r3d_headless_malloc_fn allocate;
  r3d_headless_free_fn release;
} r3d_headless_allocator;

typedef int (*r3d_headless_cancel_fn)(void *user);
typedef void (*r3d_headless_progress_fn)(void *user, const char *phase,
                                         uint64_t completed, uint64_t total);
typedef struct r3d_headless_callbacks {
  void *user;
  r3d_headless_cancel_fn cancelled;
  r3d_headless_progress_fn progress;
} r3d_headless_callbacks;

typedef struct r3d_headless_bytes {
  uint8_t *data;
  size_t size;
} r3d_headless_bytes;

/* Interleaved XYZ surface grid. Invalid vertices are (-1,-1,-1). */
typedef struct r3d_headless_surface {
  uint32_t width;
  uint32_t height;
  float *xyz;
  uint8_t *metadata;
  size_t metadata_size;
} r3d_headless_surface;

typedef struct r3d_headless_flatten_stats {
  uint32_t iterations;
  uint32_t vertex_count;
  uint32_t triangle_count;
  double initial_energy;
  double final_energy;
  double initial_stretch;
  double final_stretch;
} r3d_headless_flatten_stats;

typedef struct r3d_headless_grid_audit {
  uint64_t valid_vertices;
  uint64_t invalid_vertices;
  uint64_t valid_triangles;
  uint64_t degenerate_triangles;
  uint64_t flipped_uv_triangles;
  uint64_t unsupported_quads;
} r3d_headless_grid_audit;

typedef struct r3d_headless_volume r3d_headless_volume;
typedef struct r3d_headless_uv_rasterizer r3d_headless_uv_rasterizer;

/* sample() returns zero on success and writes value (normally [0,255]). */
typedef int (*r3d_headless_sample_fn)(void *user, const double xyz[3],
                                      double *value);

typedef struct r3d_headless_trace_config {
  double seed_xyz[3];
  double tangent_u_xyz[3];
  double tangent_v_xyz[3];
  double grid_step;
  double normal_search_radius;
  double normal_search_step;
  double minimum_support;
  uint32_t width;
  uint32_t height;
} r3d_headless_trace_config;

typedef enum r3d_headless_raster_mode {
  R3D_HEADLESS_RASTER_MEAN = 0,
  R3D_HEADLESS_RASTER_MAX = 1
} r3d_headless_raster_mode;

R3D_HEADLESS_API uint32_t r3d_headless_abi_version(void);
R3D_HEADLESS_API const char *r3d_headless_status_string(r3d_headless_status status);

/* The release functions require the same allocator used to create output. */
R3D_HEADLESS_API void r3d_headless_bytes_release_v1(
    r3d_headless_bytes *bytes, const r3d_headless_allocator *allocator);
R3D_HEADLESS_API void r3d_headless_surface_release_v1(
    r3d_headless_surface *surface, const r3d_headless_allocator *allocator);

/* TFX1 is c5d's compact tifxyz stream. metadata is carried verbatim. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_tfx1_encode_v1(
    uint32_t width, uint32_t height, const float *xyz, const uint8_t *metadata,
    size_t metadata_size, int32_t log2_quantization,
    const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_bytes *out_bytes);
/* Encode and atomically publish a TFX1 file without materializing a second
 * caller-language copy. Existing destinations are never overwritten. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_tfx1_encode_file_v1(
    const char *path, uint32_t width, uint32_t height, const float *xyz,
    const uint8_t *metadata, size_t metadata_size, int32_t log2_quantization,
    const r3d_headless_callbacks *callbacks);
R3D_HEADLESS_API r3d_headless_status r3d_headless_tfx1_decode_v1(
    const uint8_t *bytes, size_t size, const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_surface *out_surface);

/* Directory I/O uses x.tif/y.tif/z.tif/meta.json. Save is append-only and
 * publishes by one directory rename; an existing destination is never touched. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_tifxyz_load_v1(
    const char *directory, const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_surface *out_surface);
R3D_HEADLESS_API r3d_headless_status r3d_headless_tifxyz_save_v1(
    const char *directory, const r3d_headless_surface *surface,
    const r3d_headless_callbacks *callbacks);

/* Open a render3d volume manifest. Its source.json may name an upstream local
 * or HTTPS Zarr v2 source; misses are fetched and transcoded lazily by core.
 * read_roi writes C-order [nz,ny,nx] uint8 values. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_volume_open_v1(
    const char *root, uint32_t cache_bricks,
    const r3d_headless_allocator *allocator, r3d_headless_volume **out_volume);
R3D_HEADLESS_API void r3d_headless_volume_close_v1(r3d_headless_volume *volume);
R3D_HEADLESS_API r3d_headless_status r3d_headless_volume_read_roi_v1(
    r3d_headless_volume *volume, uint32_t level, int64_t x0, int64_t y0,
    int64_t z0, uint32_t nx, uint32_t ny, uint32_t nz,
    const r3d_headless_callbacks *callbacks, uint8_t *out_zyx);

/* Deterministic evidence trace on a structured tangent lattice. Each vertex
 * searches the supplied scalar field along the lattice normal. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_trace_grid_v1(
    const r3d_headless_trace_config *config, r3d_headless_sample_fn sample,
    void *sample_user, const r3d_headless_callbacks *callbacks,
    float *out_xyz, uint8_t *out_confidence);

/* Deterministic evidence tracing for arbitrary seed points and search
 * directions. base_xyz and normal_xyz contain point_count interleaved XYZ
 * doubles. Each normal is normalized independently before searching. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_trace_points_v1(
    const double *base_xyz, const double *normal_xyz, uint64_t point_count,
    double normal_search_radius, double normal_search_step,
    double minimum_support, r3d_headless_sample_fn sample, void *sample_user,
    const r3d_headless_callbacks *callbacks, float *out_xyz,
    uint8_t *out_confidence);

/* Cumulative physical edge-length initialization followed by render3d's
 * flip-safe symmetric-Dirichlet relaxation. UV is width*height*2. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_flatten_v1(
    const float *xyz, uint32_t width, uint32_t height, double target_pitch,
    uint32_t max_iterations, const r3d_headless_callbacks *callbacks,
    float *out_uv, r3d_headless_flatten_stats *out_stats);
R3D_HEADLESS_API r3d_headless_status r3d_headless_flatten_resample_v1(
    const float *xyz, const float *uv, uint32_t width, uint32_t height,
    double target_pitch, const r3d_headless_allocator *allocator,
    const r3d_headless_callbacks *callbacks, r3d_headless_surface *out_surface);

/* Sample at each grid vertex along its mesh-derived normal over [-span,+span]. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_rasterize_grid_v1(
    const float *xyz, uint32_t width, uint32_t height, double span,
    double sample_step, r3d_headless_raster_mode mode,
    r3d_headless_sample_fn sample, void *sample_user,
    const r3d_headless_callbacks *callbacks, float *out_values,
    uint8_t *out_valid);

/* Prepare deterministic UV triangle ownership for a structured surface.
 * Input arrays remain caller-owned and must stay live until close.  Validity
 * uses one LSB-first bit per C-order raster pixel. */
R3D_HEADLESS_API r3d_headless_status r3d_headless_uv_rasterizer_open_v1(
    const float *xyz, const float *uv, const uint8_t *confidence,
    const uint8_t *provenance, uint32_t grid_width, uint32_t grid_height,
    double pixel_size, double minimum_u, double minimum_v,
    uint32_t raster_width, uint32_t raster_height,
    const r3d_headless_allocator *allocator,
    r3d_headless_uv_rasterizer **out_rasterizer);
R3D_HEADLESS_API void r3d_headless_uv_rasterizer_close_v1(
    r3d_headless_uv_rasterizer *rasterizer);
R3D_HEADLESS_API r3d_headless_status r3d_headless_uv_rasterize_band_v1(
    r3d_headless_uv_rasterizer *rasterizer, uint32_t band_y0,
    uint32_t band_height, uint64_t maximum_pixel_tests,
    const r3d_headless_callbacks *callbacks, float *out_xyz,
    float *out_normal, uint8_t *out_confidence, uint8_t *out_provenance,
    uint8_t *out_valid_bits, size_t out_valid_size);

R3D_HEADLESS_API r3d_headless_status r3d_headless_grid_audit_v1(
    const float *xyz, const float *uv, uint32_t width, uint32_t height,
    double degenerate_area_epsilon, const r3d_headless_callbacks *callbacks,
    r3d_headless_grid_audit *out_audit);

#ifdef __cplusplus
}
#endif

#endif /* RENDER3D_HEADLESS_H */
