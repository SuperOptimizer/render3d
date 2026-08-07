/* The render-backend interface — the only rendering header main/core include.
 * One implementation per backend (Vulkan now, GL 4.6 later); no backend types
 * cross this boundary. SDL_Window* is the sole platform type. */
#ifndef R3D_RENDER_H
#define R3D_RENDER_H

#include <stdint.h>

#include "core/volume.h"
#include "render/render_types.h"

typedef struct SDL_Window SDL_Window;
typedef union SDL_Event SDL_Event;
typedef struct r3d_renderer r3d_renderer; /* opaque, backend-owned */

/* All functions return 0 on success, nonzero on failure (errors to stderr). */
int r3d_create(SDL_Window *win, const r3d_config *cfg, r3d_renderer **out);
void r3d_destroy(r3d_renderer *r);

int r3d_upload_volume(r3d_renderer *r, const r3d_volume_desc *d, const uint8_t *voxels);

/* Tiled-slab mode: a thin, wide z-window over a large source volume, rendered
 * from up to 2x2 tile textures with a ring-buffered z axis (see core/slab.h).
 * init creates the tiles; window uploads/scrolls to start slice z0 (clamped),
 * incrementally when the move is < ring depth. The source volume must stay
 * open across window calls. Fills p->slab_* fields via r3d_slab_params. */
typedef struct r3d_slab_desc {
  uint32_t nx, ny, nz; /* source dims */
  uint32_t wz;         /* ring depth (window slices), e.g. 32 */
} r3d_slab_desc;
int r3d_slab_init(r3d_renderer *r, const r3d_slab_desc *d);
int r3d_slab_window(r3d_renderer *r, const r3d_volume *src, uint32_t z0);
void r3d_slab_params(const r3d_renderer *r, r3d_frame_params *p); /* set slab_* fields */

/* Clipmap mode: 6 nested octaves over a dct3d shard band + pyramid files
 * (43k^2 cross sections; see core/clip.h + src/vk/vkclip.h). Call
 * r3d_clip_frame once per frame BEFORE r3d_frame: it recenters on the focus
 * (world voxels), pumps async fills, and sets the clip_* params fields. */
int r3d_clip_begin(r3d_renderer *r, const char *band_dir, const char *pyramid_dir,
                   uint32_t band_z, uint32_t depth_max);
int r3d_clip_frame(r3d_renderer *r, double fx, double fy, uint64_t z0, r3d_frame_params *p);

/* Bricks mode: a c5d .c5s shard decoded ON the GPU (entropy/IDCT/deblock
 * compute kernels) into an R8 atlas image — compressed bytes are all that
 * crosses to the GPU. Two-tier GPU cache: a WARM buffer of compressed bricks
 * (warm_mb MB, LRU) feeds budgeted decodes into the HOT atlas (pool_bpa slots
 * per axis, LRU, page-table indirected). pool_bpa >= the volume's bricks per
 * axis (or 0 when the volume fits the default pool) selects full identity
 * residency up front — the streaming pump is then a no-op. */
int r3d_bricks_begin(r3d_renderer *r, const char *c5s_path, uint32_t pool_bpa, uint32_t warm_mb);
void r3d_bricks_params(const r3d_renderer *r, r3d_frame_params *p);

/* Streaming pump: call once per frame BEFORE r3d_frame. eye/fwd = camera
 * position/forward in VOLUME space (normalized [0,1] cube coords, model
 * transform already inverted); half_tan = tan(fov_y/2) * frustum diagonal
 * factor; pixel_cone = world-space ray-cone growth per unit distance (two
 * adjacent vertical pixel rays), used to choose the c5d resolution whose
 * voxel pitch matches apparent magnification; gate = minimum visible voxel
 * value [0,1] (TF-aware, empty bricks below it never occupy slots); budget =
 * max bricks decoded this call. */
void r3d_bricks_stream(r3d_renderer *r, const float eye[3], const float fwd[3], float half_tan,
                       float pixel_cone, uint32_t slice_z0, uint32_t slice_depth, float gate,
                       uint32_t budget);
/* Volume box in renderer coordinates. Legacy cubic shards return 1,1,1;
 * manifests preserve rectangular physical proportions using max(shape). */
void r3d_bricks_extent(const r3d_renderer *r, float extent[3]);
/* True base-level voxel shape in renderer x,y,z order. */
void r3d_bricks_shape(const r3d_renderer *r, uint32_t shape[3]);

typedef struct r3d_bricks_stats {
  uint32_t nb, hot, hot_cap;    /* volume bricks; resident slots; pool slots */
  uint32_t warm_bricks;         /* compressed bricks in the warm cache */
  uint64_t warm_bytes, warm_cap;
  uint32_t inflight;            /* requests decoded this frame */
  uint64_t decoded, jobs, stream_ns; /* cumulative worker throughput/latency */
  uint32_t failures;
  uint32_t nlevels, lod_wanted[8]; /* latest projected-footprint desired set */
  uint64_t lod_requests[8];        /* cumulative visible desired bricks */
} r3d_bricks_stats;
void r3d_bricks_get_stats(r3d_renderer *r, r3d_bricks_stats *st);
/* Wait for the currently queued streaming batch (benchmark/shutdown boundary). */
void r3d_bricks_flush(r3d_renderer *r);
/* Virtual slab: a W x H x D window positioned anywhere in a sharded Zarr
 * export, streamed from a local cache with remote fetch on miss
 * (R3D_VSLAB_NOFETCH=1 disables). shard_url is the array's `/c` URL; NULL
 * retains the PHercParis3 default. The cache directory is created as needed. */
typedef struct r3d_vslab_desc {
  uint32_t nx, ny, nz; /* full export dimensions */
  uint32_t W, H, D;    /* visible window dimensions */
  const char *cache_dir;
  const char *shard_url;
  uint32_t z_prefetch; /* GPU-resident slices before/after visible D */
  /* Optional CPU-decoded full-XY window cache. Intended for thin whole-plane
   * annotation views; threads bounds background decode CPU use. */
  uint32_t prefetch_slots;
  uint32_t prefetch_threads;
} r3d_vslab_desc;
#define R3D_VSLAB_PREFETCH_MAX 6
int r3d_vslab_begin(r3d_renderer *r, const r3d_vslab_desc *d);
void r3d_vslab_frame(r3d_renderer *r, double fx, double fy, int64_t z0, uint32_t budget,
                     r3d_frame_params *p);
void r3d_vslab_get(const r3d_renderer *r, int64_t o[3], uint32_t *pending);
uint32_t r3d_vslab_resident_pending(const r3d_renderer *r);
/* Replace the ordered nearby-z prefetch targets. Earlier entries have higher
 * fill priority; invalid and duplicate entries are ignored. */
void r3d_vslab_prefetch(r3d_renderer *r, const int64_t *z0, uint32_t count);
typedef struct r3d_vslab_prefetch_stats {
  uint32_t ready, filling, capacity;
  uint64_t hits, misses;
  int64_t wanted[R3D_VSLAB_PREFETCH_MAX];
  double last_decode_ms;
} r3d_vslab_prefetch_stats;
void r3d_vslab_prefetch_get(r3d_renderer *r, r3d_vslab_prefetch_stats *st);

int r3d_set_transfer(r3d_renderer *r, const uint8_t rgba[256][4]);
/* Select the full six-tap or fast four-tap shading pipeline. */
void r3d_set_quality(r3d_renderer *r, uint32_t quality);

/* Acquire -> raycast -> blit -> present. Fills st (may be zero). */
int r3d_frame(r3d_renderer *r, const r3d_frame_params *p, r3d_frame_stats *st);
int r3d_resize(r3d_renderer *r);

/* GUI (Dear ImGui via cimgui). Call r3d_gui_event for every SDL event, then
 * r3d_gui_begin once per frame BEFORE building widgets with cimgui.h calls;
 * r3d_frame draws the accumulated UI over the volume image. */
int r3d_gui_begin(r3d_renderer *r);
void r3d_gui_event(r3d_renderer *r, const SDL_Event *ev);

/* Copy the last rendered frame to caller-provided RGBA8 buffer (screenshot).
 * Pass rgba=NULL to query size via *w,*h first; buffer must hold (*w)*(*h)*4. */
int r3d_read_frame(r3d_renderer *r, uint8_t *rgba, uint32_t *w, uint32_t *h);

#endif /* R3D_RENDER_H */
