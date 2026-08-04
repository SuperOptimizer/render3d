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
int r3d_set_transfer(r3d_renderer *r, const uint8_t rgba[256][4]);

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
