/* Vulkan implementation of render.h. Frame graph (M1):
 *   raycast.comp (storage image, GENERAL) -> blit -> swapchain -> present
 * 2 frames in flight, timeline semaphore for CPU pacing, binary semaphores for
 * WSI, timestamp queries around the dispatch. */
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <blosc.h>
#include <curl/curl.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/clip.h"
#include "core/lod.h"
#include "core/slab.h"
#include "core/vslab.h"
#include "render/render.h"
#include "brick.h"
#include "shard.h" /* c5d .c5s reader */
#include "shard/shardio.h"
#include "vk/vkc5d.h"
#include "vk/vkclip.h"
#include "vk/vkctx.h"
#include "vk/vkgui.h"
#include "vk/vkres.h"
#include "vk/vkswap.h"

#define FRAMES_IN_FLIGHT 2
#define BR_LOD_MAX 8u
#define BR_PAGE_HEADER 64u

typedef struct r3d_brlod_level {
  uint32_t scale;
  uint32_t nx, ny, nz;       /* true voxels at this level (x/y/z) */
  uint32_t bx, by, bz;       /* true brick grid */
  uint32_t sx, sy, sz;       /* c5d shard grid */
  uint32_t page_off;         /* logical-brick offset (header excluded) */
  uint32_t shard_off;
} r3d_brlod_level;

typedef struct r3d_brlod_reader {
  c5d_shard_reader sr;
  bool open;
  bool failed; /* open() failed: never retry (net-streamed trees have no local
                * shards below the coarsest level -- previously every candidate
                * cost an ENOENT syscall on the render thread every frame) */
} r3d_brlod_reader;

struct r3d_renderer {
  SDL_Window *win;
  r3d_config cfg;
  r3d_vkctx vk;
  r3d_vkswap swap;

  r3d_vkimage offscreen; /* RGBA8 storage image at drawable size */
  /* pane cache: the offscreen persists across frames, so a view whose
   * inputs (params + GPU-visible scene generation) are unchanged keeps last
   * frame's pixels and skips its raycast dispatch entirely */
  VkImageLayout os_layout;         /* current offscreen layout (UNDEFINED at start) */
  uint64_t os_layout_key;          /* nviews/origins/viewports of the last clear */
  uint64_t pane_key[R3D_MAX_VIEWS]; /* per-view input hash of the last render */
  uint64_t scene_gen;              /* bumped whenever GPU-visible data changes */
  bool pane_cache_off;             /* R3D_NO_PANE_CACHE=1 */
  /* presenter thread: vkQueuePresentKHR blocks on WSLg's software WSI (it
   * fence-waits the frame and memcpys the 8 MB image) — moving it off the
   * render thread lets the next frame's poll/GUI/record overlap that wait.
   * Presents are strictly serialized (one outstanding). */
  struct {
    pthread_t th;
    bool up, quit;
    pthread_mutex_t mu;
    pthread_cond_t cv, done_cv;
    bool pending, busy;
    uint32_t img;
    VkResult last;      /* result of the most recent present */
    bool need_resize;   /* OUT_OF_DATE/SUBOPTIMAL seen */
    bool failed;        /* other error */
  } pres;
  r3d_vkimage volume;    /* R8 3D + full mip chain (dummy 2^3 until upload) */
  r3d_vkimage tf;        /* 256x1 RGBA8 transfer function */
  r3d_vkimage occ;       /* per-8^3-block max (dilated), for empty-space skip */
  r3d_vkimage surf_coords, surf_normals; /* tifxyz grid (RGBA32F; w=valid), surf mode */
  bool surf_active;
  /* flattened surface volume: R8 3D window over (u, v, layer) resampled from
   * the bricks cache by the surfvol kernel; rebuilt in-frame when dirty */
  struct {
    bool active, dirty;
    r3d_vkimage vol;
    r3d_vkcomp comp;
    r3d_vkimage pred;  /* live 2.5D ink prediction (R32F 2D; 1x1 until fed) */
    bool pred_on;
    float pred_g0u, pred_g0v, pred_ppg; /* grid-space mapping of pred texels */
    uint32_t W, H, L, nback;
    float u0, v0, step, zoff0;
    float sx, sy;      /* tifxyz grid scale (kernel push) */
    uint32_t prog_row; /* residency-arrival re-bake cursor (UINT32_MAX idle) */
    bool rebake_again; /* residency landed mid-pass: run one more full pass */
    /* per-layer staleness: a progressive pass only bakes the visible layer
     * band (+-8), so 128-layer windows stop paying 1 GB of writes per pass;
     * layers scrubbed into view later bake lazily. bit l set = layer l may
     * hold pre-arrival content */
    uint64_t stale[4];
    uint32_t pass_z0, pass_z1; /* layer band of the running pass */
    bool baked;        /* window content valid (a full bake has been recorded) */
    bool shift_pending; /* integer-texel window move: shift + bake exposed bands */
    int32_t sh_u, sh_v, sh_z;
    uint32_t vx0, vy0, vz0, vx1, vy1, vz1; /* view-visible sub-box (texels) */
  } sv;
  VkSampler samp_vol;    /* trilinear + mip linear, clamp */
  VkSampler samp_tf;     /* linear, clamp */
  VkSampler samp_near;   /* nearest, clamp (occupancy) */
  VkSampler samp_slab;   /* trilinear, clamp UV + REPEAT W (ring z) */

  /* slab mode */
  bool slab_mode;
  r3d_slab_layout slab;
  r3d_vkimage tiles[R3D_SLAB_TILES]; /* gy-major (element j*MAX_GRID+i); unused stay null */
  r3d_vkarena tile_arena; /* shared device-memory blocks avoid one allocation per tile */
  int64_t slab_z0;       /* current window start; -1 = nothing uploaded */
  uint8_t *slice_buf;    /* CPU assembly buffer, tile_w * tile_h bytes */
  r3d_vkbuf stream_stage; /* reusable portable upload buffer */
  /* overview pyramid: prefiltered levels (scale 4<<lev, full ring z) living
   * in tiles[] after the base grid — see r3d_slab_ov_layout */
  r3d_slab_layout ovl[R3D_SLAB_OV_MAX];
  uint32_t ov_nlev, ov_base[R3D_SLAB_OV_MAX];
  uint8_t *ov_bufs[R3D_SLAB_OV_MAX]; /* downsampled composite slice per level */

  r3d_vkclip *clipm;     /* clipmap mode (NULL unless r3d_clip_begin) */

  /* bricks mode (c5d GPU-decoded atlas) */
  r3d_vkc5d *c5d;
  r3d_vkimage brick_atlas;
  VkImageView brick_atlas_mip0; /* single-mip storage view for the pack kernel */
  r3d_vkimage brick_occ;        /* 8^3-block occupancy reduced from the atlas */
  r3d_vkbuf page_buf;
  uint32_t bricks_bpa, bricks_abpa, bricks_amips;
  bool bricks_identity; /* atlas layout == world layout: direct sampling */
  bool bricks_lod;
  uint32_t bricks_nlev, bricks_nx, bricks_ny, bricks_nz, bricks_maxdim;
  char bricks_root[1024];
  /* overlay volume (e.g. 3D ink predictions): a second c5d LOD tree with
   * IDENTICAL geometry — its bricks ride the same page table and slot
   * assignment, decoded into a parallel atlas whenever a CT brick lands */
  char ink_root[1024];
  r3d_brlod_reader *ink_readers;
  r3d_vkimage ink_atlas;
  bool ink_active;
  /* net ingest: on a brick miss, fetch the owning RAW zarr chunk from
   * source.json's URL, transcode its bricks to c5d and cache them under
   * <root>/bricks/L<l>/<z>_<y>_<x>.c5b (empty file = absent upstream) —
   * each chunk is downloaded exactly once, then everything is local */
  struct {
    bool active, quit;
    char url[1024];
    uint32_t chsz[BR_LOD_MAX];
    bool raw[BR_LOD_MAX];
    float q0;
    pthread_t th[64];
    uint32_t nth;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t queue[256];   /* chunk ids: level<<48 | z<<32 | y<<16 | x */
    uint32_t qn;
    uint32_t qins; /* insertion cursor: reset each pump pass so the newest
                    * view's chunks jump ahead of the older backlog while
                    * keeping the pass's own nearest-first order */
    uint64_t inflight[64]; /* chunks being fetched right now (>= max fetchers) */
    uint32_t nin;
    _Atomic uint64_t fetched, absent_chunks, encoded;
    /* per-brick cache state, written by workers only: 0 unknown,
     * 1 = .c5b on disk, 2 = definitively absent/air. The render thread
     * never touches the filesystem — it consults this map. */
    _Atomic uint8_t *have;
    /* second source: the overlay tree (url2[0] != 0 = active). Same id
     * space (geometry-identical), own chunk layout, cache root and map;
     * queue entries carry the source in bit 63. */
    char url2[1024], root2[1280];
    uint32_t chsz2[BR_LOD_MAX];
    bool raw2[BR_LOD_MAX];
    float q02;
    _Atomic uint8_t *have2;
    _Atomic uint64_t fetched2;
  } ni;
  r3d_brlod_level bricks_lev[BR_LOD_MAX];
  r3d_brlod_reader *bricks_readers;
  uint32_t bricks_nreaders;

  /* bricks streaming (hot atlas smaller than the volume): two-tier GPU cache.
   * WARM = compressed blobs in a host-visible device buffer (LRU, first-fit
   * allocator); HOT = atlas slots (LRU, page-table indirected). The per-frame
   * pump (r3d_bricks_stream) turns frustum-prioritized requests into budgeted
   * warm->hot GPU decodes plus incremental per-slot mips and occupancy. */
  struct {
    bool active;
    bool cpu_decode; /* keep streaming codec work off the render queue */
    c5d_shard_reader sr; /* stays open: streaming reads blobs on demand */
    bool sr_open;
    uint32_t nb, nslots, frame, last_inflight, hot_cached;
    uint32_t lod_wanted[BR_LOD_MAX];
    uint64_t lod_requests[BR_LOD_MAX];
    uint64_t decoded, jobs, stream_ns;
    uint32_t failures;
    uint32_t *slot_brick, *slot_use; /* per slot: brick idx / last-wanted frame */
    uint32_t *brick_slot;            /* per brick: slot or UINT32_MAX */
    uint32_t *brick_want;            /* frame stamp: candidate de-duplication */
    int16_t *brick_maxk;             /* decoded max, -1 unknown (never re-request empties) */
    struct bcand { float d2; uint32_t b, priority; } *cands;
    uint32_t ncand_pending; /* begin/collect/submit multi-view pump state */
    bool stream_open;
    r3d_c5d_src *srcs;
    uint32_t *sel_b, *sel_slot;
    uint8_t *maxes;
    r3d_vkimage occraw;         /* world-indexed raw occupancy (pre-dilate) */
    r3d_vkcomp omax, odil;      /* region-form occupancy kernels */
    bool comp_ready;
    r3d_vkbuf warm;
    r3d_vkbuf raw_stage; /* upload staging (write-combined; never read back) */
    uint8_t *raw_host;   /* decode target + seed-cache source (heap: decode
                            passes and cache fwrite both READ it; reading the
                            mapped staging buffer ran at WC speeds) */
    uint64_t warm_cap, warm_bytes;
    uint32_t warm_bricks;
    uint32_t *warm_off, *warm_len, *warm_use; /* per brick; off UINT32_MAX = absent */
    uint32_t *warm_list, warm_list_cap;
    uint8_t *ink_missing;     /* per-slot: overlay zero-filled, awaiting net */
    uint64_t ink_fetch_seen;  /* ni.fetched2 at the last repair pass */ /* brick ids currently warm (LRU scan set) */
    struct wfnode { uint32_t off, len; } *wfree; /* offset-sorted free list */
    uint32_t nwf, cwf;
    VkCommandPool upload_pool;
    pthread_t worker;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    bool worker_up, quit;
    uint32_t job_state; /* 0 idle, 1 ready, 2 running, 3 complete */
    uint32_t job_kind;  /* 0 decode batch, 1 overlay repair (no page writes) */
    uint32_t job_n, job_nevict, job_evict[32]; /* matches BR_MAX_BATCH */
    uint64_t job_timeline;
    int job_rc;
  } bs;

  /* virtual slab: windowed toroidal streaming over the whole export.
   * Fills run on ONE worker thread (fetch + decode + host_image_copy); the
   * render thread only enqueues jobs and applies completed ones to the cell
   * ledger + validity table. Safety: a fresh cell's page key is cleared at
   * enqueue and the worker drains then-in-flight frames (timeline) before
   * writing, so the GPU never samples a tile mid-write; z-strip refills only
   * touch layers outside the published [za,zb) validity range. */
  struct {
    bool active, fetch;
    bool band; /* small-xy window: fills are window-wide 16-slice z bands
                * (per-cell decode would re-decode whole 1024^2 dct3d chunks
                * per cell — up to 60x amplification on narrow cells) */
    r3d_vslab v;
    r3d_shard_store store;
    struct vscell { int64_t cx, cy, za, zb; } *cells; /* base phys grid */
    uint8_t *box;   /* decode staging: tex^2 * wz (worker-owned) */
    uint8_t *bbox;  /* band decode staging: window-wide * 16 (worker-owned) */
    uint8_t *ds[4]; /* per-level downsample strips (one layer, worker-owned) */
    char band_dir[512];
    char shard_url[1200];
    uint32_t pending;          /* cells missing visible z data */
    uint32_t resident_pending; /* cells missing any z-margin data */
    pthread_t worker;
    VkCommandPool upload_pool; /* worker-owned staged-upload pool (fallback path) */
    r3d_vkbuf upload_stage;    /* never shared with the foreground slab path */
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_mutex_t fetch_mu; /* current/prefetch workers publish downloads atomically */
    bool quit;
    struct vsjob { /* st: 0 free, 1 ready, 2 running, 3 done */
      int64_t cx, cy, cx1, cy1, zs0; /* cell box (cx..cx1, cy..cy1); per-cell
                                        jobs have cx1 == cx, cy1 == cy */
      uint32_t nz, st;
      uint64_t tl; /* timeline value to drain before writing tiles */
    } q[4];
    /* Whole-XY decoded-window cache. A separate CPU worker fills nearby z
     * windows while the upload worker keeps the current plane responsive. */
    struct {
      bool up, quit, failed;
      uint32_t nslots, nthreads, window_nz;
      pthread_t worker;
      pthread_mutex_t mu;
      pthread_cond_t cv;
      int64_t wanted[R3D_VSLAB_PREFETCH_MAX];
      uint64_t clock, hits, misses;
      double last_decode_ms;
      struct vspc_entry {
        uint8_t *data;
        int64_t z0;
        uint32_t nz, state, refs; /* state: 0 empty, 1 filling, 2 ready */
        uint64_t used;
      } slot[R3D_VSLAB_PREFETCH_MAX];
    } pc;
  } vsl;

  PFN_vkTransitionImageLayoutEXT fp_transition;
  PFN_vkCopyMemoryToImageEXT fp_copy_mem;

  VkDescriptorSetLayout dsl;
  VkPipelineLayout pipe_layout;
  VkPipeline raycast[R3D_QUALITY_COUNT][6]; /* quality x sampling architecture */
  VkPipeline raycast_cube_8x8; /* X1-85 reduced-resolution divergence path */
  uint32_t quality;
  VkDescriptorPool dpool;
  VkDescriptorSet dset;
  r3d_vkbuf frame_ubo; /* FRAMES_IN_FLIGHT aligned r3d_frame_params records */
  VkDeviceSize frame_ubo_stride;
  uint32_t tile_descriptors; /* 1 on limited devices; full pool when indexing is native */
  bool tiled_modes;

  uint32_t wg_x, wg_y; /* raycast workgroup size (R3D_WG=8x8|16x8|16x16) */
  bool adaptive_wg; /* measured X1-85 16x8 static / 8x8 reduced-resolution split */

  VkCommandPool pool;
  VkCommandBuffer cmd[FRAMES_IN_FLIGHT];
  VkSemaphore acquire[FRAMES_IN_FLIGHT];
  VkSemaphore timeline;
  uint64_t timeline_value;
  uint64_t slot_value[FRAMES_IN_FLIGHT];
  uint64_t slot_gpu[FRAMES_IN_FLIGHT][4]; /* total, raycast, blit, gui (ns) */
  VkQueryPool query;
  bool slot_has_query[FRAMES_IN_FLIGHT];
  uint32_t slot;

  r3d_vkbuf readback; /* lazily sized for screenshots */

  bool gui_up;   /* cimgui initialized */
  bool gui_open; /* NewFrame issued, awaiting render/discard */
};

static int load_spv(const char *dir, const char *name, uint32_t **words, size_t *nbytes) {
  char path[1024];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "vk: cannot open shader %s\n", path);
    return -1;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0 || (n % 4) != 0) {
    fclose(f);
    return -1;
  }
  uint32_t *buf = malloc((size_t)n);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(buf);
    return -1;
  }
  fclose(f);
  *words = buf;
  *nbytes = (size_t)n;
  return 0;
}

static int create_compute_pipeline(r3d_renderer *r, const char *name, VkPipeline *out) {
  uint32_t *spv = NULL;
  size_t spv_n = 0;
  if (load_spv(r->cfg.spv_dir, name, &spv, &spv_n) != 0) return -1;
  VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_n,
      .pCode = spv,
  };
  VkShaderModule mod;
  VkResult res = vkCreateShaderModule(r->vk.dev, &smci, NULL, &mod);
  free(spv);
  if (res != VK_SUCCESS) return -1;
  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main"},
      .layout = r->pipe_layout,
  };
  res = vkCreateComputePipelines(r->vk.dev, VK_NULL_HANDLE, 1, &cpci, NULL, out);
  vkDestroyShaderModule(r->vk.dev, mod, NULL);
  return res == VK_SUCCESS ? 0 : -1;
}

static void pres_start(r3d_renderer *r);
static void pres_stop(r3d_renderer *r);
static void pres_drain(r3d_renderer *r);

static int create_offscreen(r3d_renderer *r) {
  VkExtent3D e = {r->swap.extent.width, r->swap.extent.height, 1};
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8G8B8A8_UNORM, e, 1,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->offscreen) != 0)
    return -1;
  r->os_layout = VK_IMAGE_LAYOUT_UNDEFINED; /* fresh image: contents undefined */
  r->pane_cache_off = getenv("R3D_NO_PANE_CACHE") && *getenv("R3D_NO_PANE_CACHE") == '1';
  r->os_layout_key = 0;
  memset(r->pane_key, 0, sizeof r->pane_key);
  VkDescriptorImageInfo ii = {.imageView = r->offscreen.view,
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 2,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
  return 0;
}

static int create_pipeline(r3d_renderer *r) {
  r->tiled_modes = r->vk.caps.descriptor_indexing &&
                   r->vk.caps.max_sampled_images >= R3D_SLAB_TILES + 3u &&
                   r->vk.caps.max_stage_resources >= R3D_SLAB_TILES + 6u;
  r->tile_descriptors = r->tiled_modes ? R3D_SLAB_TILES : 1u;
  if (!r->tiled_modes)
    fprintf(stderr,
            "vk: non-uniform descriptor array unavailable (%u sampled images); "
            "cube and bricks modes remain available, slab/clip/vslab disabled\n",
            r->vk.caps.max_sampled_images);
  VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, /* volume: array of 1024 (slab tiles; cube uses [0]) */
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = r->tile_descriptors,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 3,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 4, /* bricks mode: c5d atlas */
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 5, /* bricks mode: page table */
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 6, /* per-frame parameters; dynamic offset selects frame slot */
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 7, /* surf mode: tifxyz coords grid (RGBA32F, w = valid) */
       .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 8, /* surf mode: per-vertex surface normals */
       .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 9, /* surf mode: flattened surface volume window (RG8 3D) */
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 10, /* overlay (ink) atlas: slot-parallel to the brick atlas */
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
  };
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 11,
      .pBindings = bindings,
  };
  if (vkCreateDescriptorSetLayout(r->vk.dev, &dslci, NULL, &r->dsl) != VK_SUCCESS) return -1;

  VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &r->dsl,
  };
  if (vkCreatePipelineLayout(r->vk.dev, &plci, NULL, &r->pipe_layout) != VK_SUCCESS) return -1;

  /* One pipeline per sampling mode. R3D_WG forces a fixed cube workgroup;
   * otherwise the measured X1-85 path additionally keeps an 8x8 pipeline for
   * reduced-resolution interaction while full resolution remains 16x8. */
  const char *wg = getenv("R3D_WG");
  const char *names[R3D_QUALITY_COUNT][6] = {
      {"raycast_cube.spv", "raycast_slab.spv", "raycast_clip.spv", "raycast_bricks.spv",
       "raycast_vslab.spv", "raycast_surf.spv"},
      {"raycast_fast_cube.spv", "raycast_fast_slab.spv", "raycast_fast_clip.spv",
       "raycast_fast_bricks.spv", "raycast_fast_vslab.spv", "raycast_surf.spv"}};
  r->wg_x = 16;
  r->wg_y = 8;
  if (wg && strcmp(wg, "8x8") == 0) {
    names[0][0] = "raycast_8x8.spv";
    r->wg_x = r->wg_y = 8;
  } else if (wg && strcmp(wg, "16x16") == 0) {
    names[0][0] = "raycast_16x16.spv";
    r->wg_x = r->wg_y = 16;
  }
  r->adaptive_wg = !wg && r->vk.caps.vendor_id == 0x5143u &&
                   r->vk.caps.device_id == 0x43050c01u && r->vk.caps.subgroup_size == 128u;
  for (uint32_t q = 0; q < R3D_QUALITY_COUNT; q++)
  for (uint32_t m = 0; m < 6; m++) {
    if (!r->tiled_modes && m != 0 && m != 3) continue;
    if (create_compute_pipeline(r, names[q][m], &r->raycast[q][m]) != 0) return -1;
  }
  if (r->adaptive_wg &&
      create_compute_pipeline(r, "raycast_8x8.spv", &r->raycast_cube_8x8) != 0)
    return -1;

  VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->tile_descriptors + 5},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2},
  };
  VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 5,
      .pPoolSizes = sizes,
  };
  if (vkCreateDescriptorPool(r->vk.dev, &dpci, NULL, &r->dpool) != VK_SUCCESS) return -1;
  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = r->dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &r->dsl,
  };
  if (vkAllocateDescriptorSets(r->vk.dev, &dsai, &r->dset) != VK_SUCCESS) return -1;
  VkDeviceSize a = r->vk.caps.min_ubo_alignment;
  if (a == 0) a = 1;
  r->frame_ubo_stride = (sizeof(r3d_frame_params) + a - 1) & ~(a - 1);
  if (r3d_vkbuf_create_host(&r->vk, r->frame_ubo_stride * FRAMES_IN_FLIGHT * R3D_MAX_VIEWS,
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &r->frame_ubo) != 0)
    return -1;
  VkDescriptorBufferInfo ubi = {
      .buffer = r->frame_ubo.buf, .offset = 0, .range = sizeof(r3d_frame_params)};
  VkWriteDescriptorSet uw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = r->dset,
                             .dstBinding = 6,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                             .pBufferInfo = &ubi};
  vkUpdateDescriptorSets(r->vk.dev, 1, &uw, 0, NULL);
  return 0;
}

/* --- occupancy build: per-8^3-block max, then 26-neighbor max dilation so
 * trilinear fine samples near block borders can never read past a "skipped"
 * verdict. Parallel over z-blocks. --- */
#define OCC_BLOCK 8u

typedef struct occ_job {
  const uint8_t *vox;
  uint8_t *out;
  uint32_t nx, ny, nz, ox, oy, oz, z0, z1;
} occ_job;

static void *occ_worker(void *arg) {
  occ_job *j = arg;
  for (uint32_t bz = j->z0; bz < j->z1; bz++) {
    uint32_t zlo = bz * OCC_BLOCK, zhi = zlo + OCC_BLOCK > j->nz ? j->nz : zlo + OCC_BLOCK;
    for (uint32_t by = 0; by < j->oy; by++) {
      uint32_t ylo = by * OCC_BLOCK, yhi = ylo + OCC_BLOCK > j->ny ? j->ny : ylo + OCC_BLOCK;
      for (uint32_t bx = 0; bx < j->ox; bx++) {
        uint32_t xlo = bx * OCC_BLOCK, xhi = xlo + OCC_BLOCK > j->nx ? j->nx : xlo + OCC_BLOCK;
        uint8_t m = 0;
        for (uint32_t z = zlo; z < zhi; z++)
          for (uint32_t y = ylo; y < yhi; y++) {
            const uint8_t *row = j->vox + ((size_t)z * j->ny + y) * j->nx;
            for (uint32_t x = xlo; x < xhi; x++)
              if (row[x] > m) m = row[x];
          }
        j->out[((size_t)bz * j->oy + by) * j->ox + bx] = m;
      }
    }
  }
  return NULL;
}

static uint8_t *build_occupancy(const uint8_t *vox, uint32_t nx, uint32_t ny, uint32_t nz,
                                uint32_t *ox_out, uint32_t *oy_out, uint32_t *oz_out) {
  uint32_t ox = (nx + OCC_BLOCK - 1) / OCC_BLOCK, oy = (ny + OCC_BLOCK - 1) / OCC_BLOCK,
           oz = (nz + OCC_BLOCK - 1) / OCC_BLOCK;
  size_t n = (size_t)ox * oy * oz;
  uint8_t *raw = malloc(n), *dil = malloc(n);
  if (!raw || !dil) {
    free(raw);
    free(dil);
    return NULL;
  }

  uint32_t nthreads = oz < 12 ? oz : 12;
  pthread_t tids[12];
  occ_job jobs[12];
  for (uint32_t i = 0; i < nthreads; i++) {
    jobs[i] = (occ_job){.vox = vox, .out = raw, .nx = nx, .ny = ny, .nz = nz,
                        .ox = ox, .oy = oy, .oz = oz,
                        .z0 = oz * i / nthreads, .z1 = oz * (i + 1) / nthreads};
    pthread_create(&tids[i], NULL, occ_worker, &jobs[i]);
  }
  for (uint32_t i = 0; i < nthreads; i++) pthread_join(tids[i], NULL);

  for (uint32_t z = 0; z < oz; z++)
    for (uint32_t y = 0; y < oy; y++)
      for (uint32_t x = 0; x < ox; x++) {
        uint8_t m = 0;
        for (int dz = -1; dz <= 1; dz++)
          for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
              int zz = (int)z + dz, yy = (int)y + dy, xx = (int)x + dx;
              if (zz < 0 || yy < 0 || xx < 0 || zz >= (int)oz || yy >= (int)oy || xx >= (int)ox)
                continue;
              uint8_t v = raw[((size_t)zz * oy + (size_t)yy) * ox + (size_t)xx];
              if (v > m) m = v;
            }
        dil[((size_t)z * oy + y) * ox + x] = m;
      }
  free(raw);
  *ox_out = ox;
  *oy_out = oy;
  *oz_out = oz;
  return dil;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Write binding 0 (the vol[] tile array) — views may repeat (cube: same view). */
static void write_volume_dset(r3d_renderer *r, VkImageView views[R3D_SLAB_TILES],
                              VkSampler sampler) {
  VkDescriptorImageInfo ii[R3D_SLAB_TILES];
  for (uint32_t i = 0; i < R3D_SLAB_TILES; i++)
    ii[i] = (VkDescriptorImageInfo){.sampler = sampler,
                                    .imageView = views[i],
                                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = r->tile_descriptors,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
}

static void write_image_dset(r3d_renderer *r, uint32_t binding, VkDescriptorType type,
                             VkImageView view, VkSampler sampler, VkImageLayout layout) {
  r->scene_gen++;
  VkDescriptorImageInfo ii = {.sampler = sampler, .imageView = view, .imageLayout = layout};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = binding,
      .descriptorCount = 1,
      .descriptorType = type,
      .pImageInfo = &ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
}

/* Staging upload for small images (TF, dummy volume): buffer -> mip0, then
 * transition the whole image to SHADER_READ_ONLY. */
static int upload_small_image(r3d_renderer *r, r3d_vkimage *im, const void *data,
                              VkDeviceSize nbytes) {
  r3d_vkbuf stage;
  if (r3d_vkbuf_create_host(&r->vk, nbytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stage) != 0)
    return -1;
  memcpy(stage.mapped, data, nbytes);
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  if (!cmd) return -1;
  r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0,
                       im->mips);
  VkBufferImageCopy region = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = im->extent,
  };
  vkCmdCopyBufferToImage(cmd, stage.buf, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &region);
  r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, im->mips);
  int rc = r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
  r3d_vkbuf_destroy(&r->vk, &stage);
  return rc;
}

/* mip0 is in TRANSFER_SRC; blit-minify the chain, then everything to
 * SHADER_READ_ONLY. Spec allows sloppy 3D linear blits — compute fallback
 * (mipdown.slang) is planned if test_gpu flags this driver (docs/measured.md). */
static int gen_mips(r3d_renderer *r, r3d_vkimage *im) {
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  if (!cmd) return -1;
  VkExtent3D e = im->extent;
  for (uint32_t m = 1; m < im->mips; m++) {
    VkExtent3D ne = {e.width > 1 ? e.width / 2 : 1, e.height > 1 ? e.height / 2 : 1,
                     e.depth > 1 ? e.depth / 2 : 1};
    r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, 0,
                         VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, m, 1);
    VkImageBlit2 blit = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 0, 1},
        .srcOffsets = {{0, 0, 0}, {(int32_t)e.width, (int32_t)e.height, (int32_t)e.depth}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1},
        .dstOffsets = {{0, 0, 0}, {(int32_t)ne.width, (int32_t)ne.height, (int32_t)ne.depth}},
    };
    VkBlitImageInfo2 bi = {
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage = im->img,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = im->img,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &blit,
        .filter = VK_FILTER_LINEAR,
    };
    vkCmdBlitImage2(cmd, &bi);
    r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, m, 1);
    e = ne;
  }
  r3d_vk_image_barrier(cmd, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, im->mips);
  return r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
}

int r3d_create(SDL_Window *win, const r3d_config *cfg, r3d_renderer **out) {
  *out = NULL;
  r3d_renderer *r = calloc(1, sizeof *r);
  if (!r) return -1;
  r->win = win;
  r->cfg = *cfg;

  uint32_t next = 0;
  const char *const *exts = cfg->headless ? NULL : SDL_Vulkan_GetInstanceExtensions(&next);
  if (r3d_vkctx_create(&r->vk, exts, next, cfg->validate) != 0) goto fail;
  r3d_vkctx_set_budget(&r->vk, cfg->gpu_budget_bytes);
  VkFormatFeatureFlags r8 = r->vk.caps.r8_optimal;
  VkFormatFeatureFlags base_r8 =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  if ((r8 & base_r8) != base_r8) {
    fprintf(stderr, "vk: R8_UNORM optimal images lack sampled linear filtering\n");
    goto fail;
  }
  /* Tiles are numerous but individually small. A 64 MiB growth quantum keeps
   * allocation counts low without reserving hundreds of MiB for a tiny slab. */
  r3d_vkarena_init(&r->tile_arena, (VkDeviceSize)64 << 20);
  if (cfg->headless) { /* no surface/swapchain: the offscreen is the target */
    memset(&r->swap, 0, sizeof r->swap);
    r->swap.extent.width = cfg->headless_w ? cfg->headless_w : 1280u;
    r->swap.extent.height = cfg->headless_h ? cfg->headless_h : 720u;
    r->swap.format = VK_FORMAT_R8G8B8A8_UNORM;
    r->swap.nimages = 2;
    r3d_vkgui_set_display(r->swap.extent.width, r->swap.extent.height);
  } else if (r3d_vkswap_create(&r->vk, win, cfg->vsync, &r->swap) != 0) {
    goto fail;
  }
  if (create_pipeline(r) != 0) goto fail;
  if (create_offscreen(r) != 0) goto fail;
  if (!cfg->headless && getenv("R3D_PRESENT_THREAD")) pres_start(r);
  r->fp_transition = (PFN_vkTransitionImageLayoutEXT)(void (*)(void))vkGetDeviceProcAddr(
      r->vk.dev, "vkTransitionImageLayoutEXT");
  r->fp_copy_mem = (PFN_vkCopyMemoryToImageEXT)(void (*)(void))vkGetDeviceProcAddr(
      r->vk.dev, "vkCopyMemoryToImageEXT");

  VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = r->vk.qfam,
  };
  if (vkCreateCommandPool(r->vk.dev, &cpi, NULL, &r->pool) != VK_SUCCESS) goto fail;
  VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = r->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = FRAMES_IN_FLIGHT,
  };
  if (vkAllocateCommandBuffers(r->vk.dev, &cai, r->cmd) != VK_SUCCESS) goto fail;

  for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (vkCreateSemaphore(r->vk.dev, &sci, NULL, &r->acquire[i]) != VK_SUCCESS) goto fail;
  }
  VkSemaphoreTypeCreateInfo tsi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
  };
  VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tsi};
  if (vkCreateSemaphore(r->vk.dev, &sci, NULL, &r->timeline) != VK_SUCCESS) goto fail;

  if (r->vk.caps.timestamps) {
    VkQueryPoolCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = FRAMES_IN_FLIGHT * 4,
    };
    if (vkCreateQueryPool(r->vk.dev, &qci, NULL, &r->query) != VK_SUCCESS) goto fail;
  }

  /* samplers */
  VkSamplerCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = VK_LOD_CLAMP_NONE,
  };
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_vol) != VK_SUCCESS) goto fail;
  smci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  smci.maxLod = 0.0f;
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_tf) != VK_SUCCESS) goto fail;
  smci.magFilter = VK_FILTER_NEAREST;
  smci.minFilter = VK_FILTER_NEAREST;
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_near) != VK_SUCCESS) goto fail;

  /* dummy 2^3 volume so the descriptor set is valid before the real upload
   * (2, not 1: vkres dimensionality heuristic needs depth>1 for a 3D image) */
  {
    uint8_t zeros[8] = {0};
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){2, 2, 2}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->volume) != 0)
      goto fail;
    if (upload_small_image(r, &r->volume, zeros, sizeof zeros) != 0) goto fail;
    VkImageView vv[R3D_SLAB_TILES];
    for (uint32_t e = 0; e < R3D_SLAB_TILES; e++) vv[e] = r->volume.view;
    write_volume_dset(r, vv, r->samp_vol);
    /* dummy occupancy: fully occupied so nothing is skipped before upload */
    uint8_t full[8];
    memset(full, 0xff, sizeof full);
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){2, 2, 2}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->occ) != 0)
      goto fail;
    if (upload_small_image(r, &r->occ, full, sizeof full) != 0) goto fail;
    write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->occ.view, r->samp_near,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  /* default transfer function: grayscale ramp, linear alpha */
  {
    uint8_t ramp[256][4];
    for (uint32_t i = 0; i < 256; i++) {
      ramp[i][0] = ramp[i][1] = ramp[i][2] = (uint8_t)i;
      ramp[i][3] = (uint8_t)i;
    }
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8G8B8A8_UNORM, (VkExtent3D){256, 1, 1}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->tf) != 0)
      goto fail;
    if (upload_small_image(r, &r->tf, ramp, sizeof ramp) != 0) goto fail;
    write_image_dset(r, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->tf.view, r->samp_tf,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  /* bricks-mode dummies: atlas = dummy volume view; page = 4-byte buffer */
  {
    if (r3d_vkbuf_create_host(&r->vk, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &r->page_buf) != 0)
      goto fail;
    ((uint32_t *)r->page_buf.mapped)[0] = 0xFFFFFFFFu;
    write_image_dset(r, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->volume.view,
                     r->samp_vol, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkDescriptorBufferInfo pbi = {.buffer = r->page_buf.buf, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet pw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                               .dstSet = r->dset,
                               .dstBinding = 5,
                               .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               .pBufferInfo = &pbi};
    vkUpdateDescriptorSets(r->vk.dev, 1, &pw, 0, NULL);
  }

  if (r3d_vkgui_init(&r->vk, cfg->headless ? NULL : win, r->swap.format, r->swap.nimages) != 0)
    goto fail;
  r->gui_up = true;

  *out = r;
  return 0;
fail:
  r3d_destroy(r);
  return -1;
}

int r3d_gui_begin(r3d_renderer *r) {
  if (!r->gui_up) return -1;
  if (!r->gui_open) {
    r3d_vkgui_new_frame();
    r->gui_open = true;
  }
  return 0;
}

void r3d_gui_event(r3d_renderer *r, const SDL_Event *ev) {
  if (r->gui_up) r3d_vkgui_event(ev);
}

/* Tear down everything DATASET-scoped in the bricks family (streamer, both
 * atlases, page table, net ingest, surf textures, surface volume) so another
 * r3d_bricks_begin can follow — the dataset is ordinary mutable state. Core
 * renderer objects (descriptor set, pipelines, samplers) are untouched. */
static void bricks_teardown(r3d_renderer *r) {
  if (r->bs.worker_up) {
    pthread_mutex_lock(&r->bs.mu);
    r->bs.quit = true;
    pthread_cond_broadcast(&r->bs.cv);
    pthread_mutex_unlock(&r->bs.mu);
    pthread_join(r->bs.worker, NULL);
    pthread_mutex_destroy(&r->bs.mu);
    pthread_cond_destroy(&r->bs.cv);
    r->bs.worker_up = false;
  }
  if (r->ni.active) {
    pthread_mutex_lock(&r->ni.mu);
    r->ni.quit = true;
    pthread_cond_broadcast(&r->ni.cv);
    pthread_mutex_unlock(&r->ni.mu);
    for (uint32_t t = 0; t < r->ni.nth; t++) pthread_join(r->ni.th[t], NULL);
    pthread_mutex_destroy(&r->ni.mu);
    pthread_cond_destroy(&r->ni.cv);
  }
  free((void *)r->ni.have);
  free((void *)r->ni.have2);
  if (r->vk.dev) r3d_vkctx_device_wait_idle(&r->vk);
  if (r->bs.upload_pool) vkDestroyCommandPool(r->vk.dev, r->bs.upload_pool, NULL);
  if (r->c5d) r3d_vkc5d_destroy(r->c5d);
  r->c5d = NULL;
  if (r->brick_atlas_mip0) vkDestroyImageView(r->vk.dev, r->brick_atlas_mip0, NULL);
  r->brick_atlas_mip0 = VK_NULL_HANDLE;
  r3d_vkimage_destroy(&r->vk, &r->brick_atlas);
  r3d_vkimage_destroy(&r->vk, &r->brick_occ);
  r3d_vkbuf_destroy(&r->vk, &r->page_buf);
  r3d_vkbuf_destroy(&r->vk, &r->bs.raw_stage);
  if (r->bs.comp_ready) {
    r3d_vkcomp_destroy(&r->vk, &r->bs.omax);
    r3d_vkcomp_destroy(&r->vk, &r->bs.odil);
  }
  r3d_vkimage_destroy(&r->vk, &r->bs.occraw);
  r3d_vkbuf_destroy(&r->vk, &r->bs.warm);
  if (r->bs.sr_open) c5d_shard_close_reader(&r->bs.sr);
  for (uint32_t i = 0; i < r->bricks_nreaders; i++)
    if (r->bricks_readers[i].open) c5d_shard_close_reader(&r->bricks_readers[i].sr);
  free(r->bricks_readers);
  r->bricks_readers = NULL;
  if (r->ink_readers)
    for (uint32_t i = 0; i < r->bricks_nreaders; i++)
      if (r->ink_readers[i].open) c5d_shard_close_reader(&r->ink_readers[i].sr);
  free(r->ink_readers);
  r->ink_readers = NULL;
  r3d_vkimage_destroy(&r->vk, &r->ink_atlas);
  r->ink_active = false;
  free(r->bs.slot_brick);
  free(r->bs.slot_use);
  free(r->bs.brick_slot);
  free(r->bs.brick_want);
  free(r->bs.brick_maxk);
  free(r->bs.cands);
  free(r->bs.srcs);
  free(r->bs.sel_b);
  free(r->bs.sel_slot);
  free(r->bs.maxes);
  free(r->bs.raw_host);
  free(r->bs.warm_off);
  free(r->bs.warm_len);
  free(r->bs.warm_use);
  free(r->bs.warm_list);
  free(r->bs.wfree);
  free(r->bs.ink_missing);
  memset(&r->bs, 0, sizeof r->bs);
  memset(&r->ni, 0, sizeof r->ni);
  r3d_vkimage_destroy(&r->vk, &r->surf_coords);
  r3d_vkimage_destroy(&r->vk, &r->surf_normals);
  r->surf_active = false;
  r3d_vkimage_destroy(&r->vk, &r->sv.vol);
  r3d_vkimage_destroy(&r->vk, &r->sv.pred);
  r3d_vkcomp_destroy(&r->vk, &r->sv.comp);
  memset(&r->sv, 0, sizeof r->sv);
  r->bricks_lod = false;
  r->bricks_nlev = 0;
  r->bricks_nreaders = 0;
  r->bricks_bpa = r->bricks_abpa = 0;
  r->bricks_nx = r->bricks_ny = r->bricks_nz = r->bricks_maxdim = 0;
  r->bricks_root[0] = r->ink_root[0] = 0;
}

int r3d_bricks_end(r3d_renderer *r) {
  if (!r->bs.active && !r->bricks_lod && !r->bricks_bpa) return 0;
  bricks_teardown(r);
  /* leave dataset bindings pointing at safe placeholders (occupancy image),
   * matching the pre-begin state well enough for non-bricks pipelines */
  write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->occ.view, r->samp_near,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return 0;
}

void r3d_destroy(r3d_renderer *r) {
  if (!r) return;
  pres_stop(r);
  if (r->bs.worker_up) {
    pthread_mutex_lock(&r->bs.mu);
    r->bs.quit = true;
    pthread_cond_broadcast(&r->bs.cv);
    pthread_mutex_unlock(&r->bs.mu);
    pthread_join(r->bs.worker, NULL);
    pthread_mutex_destroy(&r->bs.mu);
    pthread_cond_destroy(&r->bs.cv);
    r->bs.worker_up = false;
  }
  if (r->vsl.active) {
    pthread_mutex_lock(&r->vsl.mu);
    r->vsl.quit = true;
    pthread_cond_broadcast(&r->vsl.cv);
    pthread_mutex_unlock(&r->vsl.mu);
    if (r->vsl.pc.up) {
      pthread_mutex_lock(&r->vsl.pc.mu);
      r->vsl.pc.quit = true;
      pthread_cond_broadcast(&r->vsl.pc.cv);
      pthread_mutex_unlock(&r->vsl.pc.mu);
    }
    pthread_join(r->vsl.worker, NULL);
    if (r->vsl.pc.up) {
      pthread_join(r->vsl.pc.worker, NULL);
      pthread_mutex_destroy(&r->vsl.pc.mu);
      pthread_cond_destroy(&r->vsl.pc.cv);
    }
    pthread_mutex_destroy(&r->vsl.mu);
    pthread_cond_destroy(&r->vsl.cv);
    pthread_mutex_destroy(&r->vsl.fetch_mu);
  }
  if (r->vk.dev) r3d_vkctx_device_wait_idle(&r->vk);
  if (r->bs.upload_pool) vkDestroyCommandPool(r->vk.dev, r->bs.upload_pool, NULL);
  if (r->vsl.upload_pool) vkDestroyCommandPool(r->vk.dev, r->vsl.upload_pool, NULL);
  r3d_vkbuf_destroy(&r->vk, &r->vsl.upload_stage);
  if (r->gui_up) {
    if (r->gui_open) r3d_vkgui_discard();
    r3d_vkgui_shutdown();
  }
  r3d_vkbuf_destroy(&r->vk, &r->readback);
  r3d_vkbuf_destroy(&r->vk, &r->frame_ubo);
  r3d_vkbuf_destroy(&r->vk, &r->stream_stage);
  if (r->query) vkDestroyQueryPool(r->vk.dev, r->query, NULL);
  if (r->timeline) vkDestroySemaphore(r->vk.dev, r->timeline, NULL);
  for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    if (r->acquire[i]) vkDestroySemaphore(r->vk.dev, r->acquire[i], NULL);
  if (r->pool) vkDestroyCommandPool(r->vk.dev, r->pool, NULL);
  if (r->dpool) vkDestroyDescriptorPool(r->vk.dev, r->dpool, NULL);
  for (uint32_t q = 0; q < R3D_QUALITY_COUNT; q++)
    for (uint32_t m = 0; m < 6; m++)
      if (r->raycast[q][m]) vkDestroyPipeline(r->vk.dev, r->raycast[q][m], NULL);
  if (r->raycast_cube_8x8) vkDestroyPipeline(r->vk.dev, r->raycast_cube_8x8, NULL);
  if (r->pipe_layout) vkDestroyPipelineLayout(r->vk.dev, r->pipe_layout, NULL);
  if (r->dsl) vkDestroyDescriptorSetLayout(r->vk.dev, r->dsl, NULL);
  r3d_vkclip_destroy(r->clipm);
  if (r->c5d) r3d_vkc5d_destroy(r->c5d);
  if (r->brick_atlas_mip0) vkDestroyImageView(r->vk.dev, r->brick_atlas_mip0, NULL);
  r3d_vkimage_destroy(&r->vk, &r->brick_atlas);
  r3d_vkimage_destroy(&r->vk, &r->brick_occ);
  r3d_vkbuf_destroy(&r->vk, &r->page_buf);
  r3d_vkbuf_destroy(&r->vk, &r->bs.raw_stage);
  if (r->bs.comp_ready) {
    r3d_vkcomp_destroy(&r->vk, &r->bs.omax);
    r3d_vkcomp_destroy(&r->vk, &r->bs.odil);
  }
  r3d_vkimage_destroy(&r->vk, &r->bs.occraw);
  r3d_vkbuf_destroy(&r->vk, &r->bs.warm);
  if (r->bs.sr_open) c5d_shard_close_reader(&r->bs.sr);
  for (uint32_t i = 0; i < r->bricks_nreaders; i++)
    if (r->bricks_readers[i].open) c5d_shard_close_reader(&r->bricks_readers[i].sr);
  free(r->bricks_readers);
  if (r->ink_readers)
    for (uint32_t i = 0; i < r->bricks_nreaders; i++)
      if (r->ink_readers[i].open) c5d_shard_close_reader(&r->ink_readers[i].sr);
  free(r->ink_readers);
  r3d_vkimage_destroy(&r->vk, &r->ink_atlas);
  if (r->ni.active) {
    pthread_mutex_lock(&r->ni.mu);
    r->ni.quit = true;
    pthread_cond_broadcast(&r->ni.cv);
    pthread_mutex_unlock(&r->ni.mu);
    for (uint32_t t = 0; t < r->ni.nth; t++) pthread_join(r->ni.th[t], NULL);
    pthread_mutex_destroy(&r->ni.mu);
    pthread_cond_destroy(&r->ni.cv);
  }
  free((void *)r->ni.have);
  free(r->bs.slot_brick);
  free(r->bs.slot_use);
  free(r->bs.brick_slot);
  free(r->bs.brick_want);
  free(r->bs.brick_maxk);
  free(r->bs.cands);
  free(r->bs.srcs);
  free(r->bs.sel_b);
  free(r->bs.sel_slot);
  free(r->bs.maxes);
  free(r->bs.raw_host);
  free(r->bs.warm_off);
  free(r->bs.warm_len);
  free(r->bs.warm_use);
  free(r->bs.warm_list);
  free(r->bs.wfree);
  if (r->samp_vol) vkDestroySampler(r->vk.dev, r->samp_vol, NULL);
  if (r->samp_tf) vkDestroySampler(r->vk.dev, r->samp_tf, NULL);
  if (r->samp_near) vkDestroySampler(r->vk.dev, r->samp_near, NULL);
  if (r->samp_slab) vkDestroySampler(r->vk.dev, r->samp_slab, NULL);
  for (uint32_t i = 0; i < R3D_SLAB_TILES; i++) r3d_vkimage_destroy(&r->vk, &r->tiles[i]);
  r3d_vkarena_destroy(&r->vk, &r->tile_arena);
  free(r->slice_buf);
  for (uint32_t i = 0; i < R3D_SLAB_OV_MAX; i++) free(r->ov_bufs[i]);
  free(r->vsl.cells);
  free(r->vsl.box);
  free(r->vsl.bbox);
  for (uint32_t i = 0; i < 4; i++) free(r->vsl.ds[i]);
  for (uint32_t i = 0; i < R3D_VSLAB_PREFETCH_MAX; i++) free(r->vsl.pc.slot[i].data);
  r3d_vkimage_destroy(&r->vk, &r->volume);
  r3d_vkimage_destroy(&r->vk, &r->tf);
  r3d_vkimage_destroy(&r->vk, &r->occ);
  r3d_vkimage_destroy(&r->vk, &r->surf_coords);
  r3d_vkimage_destroy(&r->vk, &r->surf_normals);
  r3d_vkimage_destroy(&r->vk, &r->sv.vol);
  r3d_vkimage_destroy(&r->vk, &r->sv.pred);
  r3d_vkcomp_destroy(&r->vk, &r->sv.comp);
  r3d_vkimage_destroy(&r->vk, &r->offscreen);
  r3d_vkswap_destroy(&r->vk, &r->swap);
  r3d_vkctx_destroy(&r->vk);
  free(r);
}

static void *pres_thread(void *arg) {
  r3d_renderer *r = arg;
  for (;;) {
    pthread_mutex_lock(&r->pres.mu);
    while (!r->pres.quit && !r->pres.pending) pthread_cond_wait(&r->pres.cv, &r->pres.mu);
    if (r->pres.quit) {
      pthread_mutex_unlock(&r->pres.mu);
      return NULL;
    }
    uint32_t img = r->pres.img;
    r->pres.pending = false;
    r->pres.busy = true;
    pthread_mutex_unlock(&r->pres.mu);
    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &r->swap.render_done[img],
        .swapchainCount = 1,
        .pSwapchains = &r->swap.swapchain,
        .pImageIndices = &img,
    };
    VkResult pr = r3d_vkctx_queue_present(&r->vk, &pi);
    pthread_mutex_lock(&r->pres.mu);
    r->pres.last = pr;
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) r->pres.need_resize = true;
    else if (pr != VK_SUCCESS) r->pres.failed = true;
    r->pres.busy = false;
    pthread_cond_broadcast(&r->pres.done_cv);
    pthread_mutex_unlock(&r->pres.mu);
  }
}

/* wait until no present is queued or in flight (swapchain about to change,
 * shutdown, or the next present is about to be queued) */
static void pres_drain(r3d_renderer *r) {
  if (!r->pres.up) return;
  pthread_mutex_lock(&r->pres.mu);
  while (r->pres.pending || r->pres.busy) pthread_cond_wait(&r->pres.done_cv, &r->pres.mu);
  pthread_mutex_unlock(&r->pres.mu);
}

static void pres_start(r3d_renderer *r) {
  if (r->pres.up || getenv("R3D_NO_PRESENT_THREAD")) return;
  pthread_mutex_init(&r->pres.mu, NULL);
  pthread_cond_init(&r->pres.cv, NULL);
  pthread_cond_init(&r->pres.done_cv, NULL);
  if (pthread_create(&r->pres.th, NULL, pres_thread, r) == 0) r->pres.up = true;
}

static void pres_stop(r3d_renderer *r) {
  if (!r->pres.up) return;
  pres_drain(r);
  pthread_mutex_lock(&r->pres.mu);
  r->pres.quit = true;
  pthread_cond_broadcast(&r->pres.cv);
  pthread_mutex_unlock(&r->pres.mu);
  pthread_join(r->pres.th, NULL);
  pthread_mutex_destroy(&r->pres.mu);
  pthread_cond_destroy(&r->pres.cv);
  pthread_cond_destroy(&r->pres.done_cv);
  r->pres.up = false;
}

int r3d_resize(r3d_renderer *r) {
  if (r->cfg.headless) return 0; /* fixed offscreen, nothing to recreate */
  pres_drain(r); /* no present may reference the old swapchain */
  int rc = r3d_vkswap_recreate(&r->vk, r->win, r->cfg.vsync, &r->swap);
  if (rc != 0) return rc; /* rc==1: minimized, keep old resources */
  r3d_vkimage_destroy(&r->vk, &r->offscreen);
  return create_offscreen(r);
}

int r3d_upload_volume(r3d_renderer *r, const r3d_volume_desc *d, const uint8_t *voxels) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  uint32_t maxdim = d->nx > d->ny ? d->nx : d->ny;
  if (d->nz > maxdim) maxdim = d->nz;
  if (maxdim > r->vk.caps.max_dim_3d) {
    fprintf(stderr, "vk: volume %ux%ux%u exceeds maxImageDimension3D=%u (use bricks)\n", d->nx,
            d->ny, d->nz, r->vk.caps.max_dim_3d);
    return -1;
  }
  uint32_t mips = 1;
  while ((maxdim >> mips) >= 1 && mips < 16) mips++;
  VkFormatFeatureFlags r8 = r->vk.caps.r8_optimal;
  VkFormatFeatureFlags mip_features = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                      VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  if ((r8 & mip_features) != mip_features) {
    fprintf(stderr, "vk: R8_UNORM linear blits unavailable; using one volume mip\n");
    mips = 1;
  }

  /* Staging beats host-image-copy for the bulk upload on every GPU measured so
   * far (Turnip 251 vs 339 ms/GiB, RTX 4060 294 vs 547) — the transient cost is
   * one reusable <=128 MiB buffer, so it is the default. R3D_STAGING=0 forces
   * the host-image-copy path back on. Streaming modes prefer host image copy
   * but have a reusable-buffer staged fallback on devices without it. */
  const char *stg = getenv("R3D_STAGING");
  bool use_hic = r->vk.caps.host_image_copy && r->fp_transition && r->fp_copy_mem && stg &&
                 *stg == '0';
  VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            (mips > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u) |
                            (use_hic ? VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT : 0u);

  r3d_vkimage im;
  VkExtent3D extent = {d->nx, d->ny, d->nz};
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, extent, mips, usage, &im) != 0) return -1;

  size_t total = (size_t)d->nx * d->ny * d->nz;
  uint64_t t0 = now_ns();
  if (use_hic) {
    VkHostImageLayoutTransitionInfoEXT tr = {
        .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
        .image = im.img,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if (r->fp_transition(r->vk.dev, 1, &tr) != VK_SUCCESS) goto fail;
    VkMemoryToImageCopyEXT region = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
        .pHostPointer = voxels,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = extent,
    };
    VkCopyMemoryToImageInfoEXT ci = {
        .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
        .dstImage = im.img,
        .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .regionCount = 1,
        .pRegions = &region,
    };
    if (r->fp_copy_mem(r->vk.dev, &ci) != VK_SUCCESS) goto fail;
    /* Host-written GENERAL -> mip source, or directly to shader-read when
     * this format cannot be linearly blitted. */
    VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    if (!cmd) goto fail;
    r3d_vk_image_barrier(
        cmd, im.img, VK_IMAGE_LAYOUT_GENERAL,
        mips > 1 ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        mips > 1 ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        mips > 1 ? VK_ACCESS_2_TRANSFER_READ_BIT : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
    if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
  } else {
    /* reusable staging slab: whole slices, <=128 MiB per copy */
    size_t slice = (size_t)d->nx * d->ny;
    uint32_t zslab = (uint32_t)((128u << 20) / slice);
    if (zslab == 0) zslab = 1;
    if (zslab > d->nz) zslab = d->nz;
    r3d_vkbuf stage;
    if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)(slice * zslab),
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stage) != 0)
      goto fail;
    VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    r3d_vk_image_barrier(cmd, im.img, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
    if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
    for (uint32_t z = 0; z < d->nz; z += zslab) {
      uint32_t nz = z + zslab > d->nz ? d->nz - z : zslab;
      memcpy(stage.mapped, voxels + (size_t)z * slice, slice * nz);
      cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
      VkBufferImageCopy region = {
          .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .imageOffset = {0, 0, (int32_t)z},
          .imageExtent = {d->nx, d->ny, nz},
      };
      vkCmdCopyBufferToImage(cmd, stage.buf, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                             &region);
      if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
    }
    r3d_vkbuf_destroy(&r->vk, &stage);
    cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    r3d_vk_image_barrier(
        cmd, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mips > 1 ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        mips > 1 ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        mips > 1 ? VK_ACCESS_2_TRANSFER_READ_BIT : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
    if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) goto fail;
  }
  double up_ms = (double)(now_ns() - t0) / 1e6;

  t0 = now_ns();
  if (mips > 1 && gen_mips(r, &im) != 0) goto fail;
  double mip_ms = (double)(now_ns() - t0) / 1e6;
  printf("volume: %ux%ux%u (%zu MiB, %u mips) upload[%s] %.0f ms (%.2f GB/s), mips %.0f ms\n",
         d->nx, d->ny, d->nz, total >> 20, mips, use_hic ? "host-image-copy" : "staging", up_ms,
         (double)total / up_ms / 1e6, mip_ms);

  /* occupancy pyramid for empty-space skipping */
  t0 = now_ns();
  uint32_t ox, oy, oz;
  uint8_t *occ = build_occupancy(voxels, d->nx, d->ny, d->nz, &ox, &oy, &oz);
  if (!occ) goto fail;
  r3d_vkimage occ_im;
  if (oz < 2) { /* keep the image 3D (vkres heuristic): duplicate the layer */
    uint8_t *grown = realloc(occ, (size_t)ox * oy * 2);
    if (!grown) {
      free(occ);
      goto fail;
    }
    occ = grown;
    memcpy(occ + (size_t)ox * oy, occ, (size_t)ox * oy);
    oz = 2;
  }
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){ox, oy, oz}, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &occ_im) != 0) {
    free(occ);
    goto fail;
  }
  int occ_rc = upload_small_image(r, &occ_im, occ, (VkDeviceSize)ox * oy * oz);
  free(occ);
  if (occ_rc != 0) {
    r3d_vkimage_destroy(&r->vk, &occ_im);
    goto fail;
  }
  printf("volume: occupancy %ux%ux%u built+uploaded in %.0f ms\n", ox, oy, oz,
         (double)(now_ns() - t0) / 1e6);

  r3d_vkctx_device_wait_idle(&r->vk);
  r3d_vkimage_destroy(&r->vk, &r->volume);
  r3d_vkimage_destroy(&r->vk, &r->occ);
  r->volume = im;
  r->occ = occ_im;
  VkImageView vv[R3D_SLAB_TILES];
    for (uint32_t e = 0; e < R3D_SLAB_TILES; e++) vv[e] = r->volume.view;
  write_volume_dset(r, vv, r->samp_vol);
  write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->occ.view, r->samp_near,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return 0;
fail:
  r3d_vkimage_destroy(&r->vk, &im);
  return -1;
}

int r3d_surf_begin(r3d_renderer *r, uint32_t w, uint32_t h, const float *coords_rgba,
                   const float *normals_rgba) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!w || !h || !coords_rgba || !normals_rgba || r->surf_active) return -1;
  VkExtent3D e = {w, h, 1};
  VkDeviceSize n = (VkDeviceSize)w * h * 4 * sizeof(float);
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R32G32B32A32_SFLOAT, e, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->surf_coords) != 0 ||
      r3d_vkimage_create(&r->vk, VK_FORMAT_R32G32B32A32_SFLOAT, e, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->surf_normals) != 0)
    return -1;
  if (upload_small_image(r, &r->surf_coords, coords_rgba, n) != 0 ||
      upload_small_image(r, &r->surf_normals, normals_rgba, n) != 0)
    return -1;
  write_image_dset(r, 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, r->surf_coords.view, VK_NULL_HANDLE,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image_dset(r, 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, r->surf_normals.view, VK_NULL_HANDLE,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  r->surf_active = true;
  return 0;
}

int r3d_surf_swap(r3d_renderer *r, uint32_t w, uint32_t h, const float *coords_rgba,
                  const float *normals_rgba, float sx, float sy) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!r->surf_active || !w || !h || !coords_rgba || !normals_rgba) return -1;
  /* rare, user-triggered: idle the device so the old grid images retire */
  pres_drain(r); /* queue ops need external sync */
  vkDeviceWaitIdle(r->vk.dev);
  r3d_vkimage_destroy(&r->vk, &r->surf_coords);
  r3d_vkimage_destroy(&r->vk, &r->surf_normals);
  VkExtent3D e = {w, h, 1};
  VkDeviceSize n = (VkDeviceSize)w * h * 4 * sizeof(float);
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R32G32B32A32_SFLOAT, e, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->surf_coords) != 0 ||
      r3d_vkimage_create(&r->vk, VK_FORMAT_R32G32B32A32_SFLOAT, e, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->surf_normals) != 0)
    return -1;
  if (upload_small_image(r, &r->surf_coords, coords_rgba, n) != 0 ||
      upload_small_image(r, &r->surf_normals, normals_rgba, n) != 0)
    return -1;
  write_image_dset(r, 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, r->surf_coords.view, VK_NULL_HANDLE,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image_dset(r, 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, r->surf_normals.view, VK_NULL_HANDLE,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  if (r->sv.active) { /* window texture survives; rebind the grid taps and
                       * force the next _window call to rebake everything */
    r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->surf_coords.view, r->samp_near,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->surf_normals.view, r->samp_near,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    r->sv.sx = sx;
    r->sv.sy = sy;
    r->sv.step = 0.0f;
    r->sv.dirty = false;
    r->sv.prog_row = UINT32_MAX;
    r->sv.baked = false;
    r->sv.shift_pending = false;
  }
  return 0;
}

/* rows per progressive surfvol re-bake dispatch (2048x128x96 ~= 5 ms GPU) */
#define SV_PROG_ROWS 128u

/* push-constant mirror of surfvol.comp */
typedef struct sv_push {
  float u0, v0, step, zoff0;
  float sx, sy;
  uint32_t W, H, L, nback;
  uint32_t abpa, lod;
  float lstep;
  uint32_t use_ink;
  uint32_t x0, y0, z0;
  uint32_t use_pred;
  float pg0u, pg0v, pppg;
} sv_push;

int r3d_surfvol_begin(r3d_renderer *r, uint32_t w, uint32_t h, uint32_t layers,
                      uint32_t nback, float sx, float sy) {
  if (!r->surf_active || !r->bricks_lod || r->sv.active) return -1;
  uint32_t cap = r->vk.caps.max_dim_3d;
  if (w > cap) w = cap;
  if (h > cap) h = cap;
  if (layers > cap) layers = cap;
  if (nback >= layers) nback = layers / 2;
  /* fit the single allocation limit and the remaining device budget (never
   * request an image the driver cannot allocate; shrink depth first, then
   * the xy extent, keeping the window usable) */
  {
    uint64_t acap = r->vk.caps.max_alloc_bytes;
    uint64_t avail = r3d_vkctx_budget_available(&r->vk);
    if (avail > (512ull << 20)) avail -= 256ull << 20; /* keep slack for staging */
    if (acap > avail && avail) acap = avail;
    if (!acap) acap = 1ull << 30;
    while ((uint64_t)w * h * layers * 2u > acap) {
      if (layers > 48) layers = layers * 3 / 4;
      else if (w >= h && w > 512) w /= 2;
      else if (h > 512) h /= 2;
      else break;
    }
    if (nback >= layers) nback = layers / 2;
  }
  printf("surfvol: %ux%ux%u window (%.0f MB, %u layers behind)\n", w, h, layers,
         (double)w * h * layers * 2.0 / 1048576.0, nback);
  r->sv.W = w;
  r->sv.H = h;
  r->sv.L = layers;
  r->sv.nback = nback;
  r->sv.sx = sx;
  r->sv.sy = sy;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8G8_UNORM, (VkExtent3D){w, h, layers}, 1,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         &r->sv.vol) != 0)
    return -1;
  if (r3d_vk_image_to_general(&r->vk, r->pool, &r->sv.vol) != 0) return -1;
  static const VkDescriptorType tt[7] = {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
  char sp[1024];
  snprintf(sp, sizeof sp, "%s/surfvol.spv", r->cfg.spv_dir);
  if (r3d_vkcomp_create(&r->vk, sp, tt, 7, sizeof(sv_push), &r->sv.comp) != 0) return -1;
  r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        r->sv.vol.view, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
  r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        r->surf_coords.view, r->samp_near,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        r->surf_normals.view, r->samp_near,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  r3d_vkcomp_bind_buffer(&r->vk, &r->sv.comp, 3, r->page_buf.buf, 0, VK_WHOLE_SIZE);
  r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        r->brick_atlas.view, r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  /* overlay atlas when active, else the CT atlas as an inert placeholder */
  r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        r->ink_active ? r->ink_atlas.view : r->brick_atlas.view, r->samp_vol,
                        VK_IMAGE_LAYOUT_GENERAL);
  /* live 2.5D ink prediction: 1x1 placeholder until r3d_surfvol_inkpred */
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R32_SFLOAT, (VkExtent3D){1, 1, 1}, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->sv.pred) != 0)
    return -1;
  if (r3d_vk_image_to_general(&r->vk, r->pool, &r->sv.pred) != 0) return -1;
  r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        r->sv.pred.view, r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  r->sv.pred_on = false;
  write_image_dset(r, 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->sv.vol.view,
                   r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  r->sv.step = 0.0f; /* no window yet */
  r->sv.active = true;
  r->sv.dirty = false;
  r->sv.prog_row = UINT32_MAX;
  r->sv.baked = false;
  r->sv.shift_pending = false;
  return 0;
}

void r3d_surfvol_window(r3d_renderer *r, double u0, double v0, float step, float zoff0) {
  if (!r->sv.active) return;
  if ((float)u0 == r->sv.u0 && (float)v0 == r->sv.v0 && step == r->sv.step &&
      zoff0 == r->sv.zoff0)
    return;
  /* Same pitch and an integer-texel move (the window origin snaps in texel
   * multiples, zoff in whole layers): shift the surviving content in place
   * and re-bake only the exposed bands, instead of the full ~75 ms window
   * bake. uv moves and zoff moves are handled separately; a combined move
   * (rare: it needs both snaps in one frame) falls through to a full bake. */
  if (r->sv.baked && !r->sv.dirty && !r->sv.shift_pending && step == r->sv.step &&
      step > 0.0f) {
    double du = (u0 - (double)r->sv.u0) / (double)step;
    double dv = (v0 - (double)r->sv.v0) / (double)step;
    double dz = (double)zoff0 - (double)r->sv.zoff0; /* layer pitch lstep = 1 */
    long idu = lround(du), idv = lround(dv), idz = lround(dz);
    bool integral = fabs(du - (double)idu) < 1e-3 && fabs(dv - (double)idv) < 1e-3 &&
                    fabs(dz - (double)idz) < 1e-3;
    bool fits = labs(idu) < (long)r->sv.W && labs(idv) < (long)r->sv.H &&
                labs(idz) < (long)r->sv.L;
    bool uv_only = idz == 0 && (idu != 0 || idv != 0);
    bool z_only = idz != 0 && idu == 0 && idv == 0;
    if (integral && fits && (uv_only || z_only)) {
      r->sv.sh_u = (int32_t)idu;
      r->sv.sh_v = (int32_t)idv;
      r->sv.sh_z = (int32_t)idz;
      r->sv.shift_pending = true;
      r->sv.u0 = (float)u0;
      r->sv.v0 = (float)v0;
      r->sv.zoff0 = zoff0;
      /* an in-flight progressive pass restarts when the shift executes */
      return;
    }
  }
  r->sv.u0 = (float)u0;
  r->sv.v0 = (float)v0;
  r->sv.step = step;
  r->sv.zoff0 = zoff0;
  r->sv.dirty = true;
  r->sv.shift_pending = false;
  r->sv.prog_row = UINT32_MAX; /* mapping changed: the full rebuild supersedes */
}

uint32_t r3d_max_dim3d(const r3d_renderer *r) { return r->vk.caps.max_dim_3d; }

void r3d_surfvol_visible(r3d_renderer *r, uint32_t x0, uint32_t y0, uint32_t z0, uint32_t x1,
                         uint32_t y1, uint32_t z1) {
  if (!r->sv.active) return;
  r->sv.vx0 = x0 < r->sv.W ? x0 : r->sv.W;
  r->sv.vy0 = y0 < r->sv.H ? y0 : r->sv.H;
  r->sv.vz0 = z0 < r->sv.L ? z0 : r->sv.L;
  r->sv.vx1 = x1 < r->sv.W ? x1 : r->sv.W;
  r->sv.vy1 = y1 < r->sv.H ? y1 : r->sv.H;
  r->sv.vz1 = z1 < r->sv.L ? z1 : r->sv.L;
}

int r3d_surfvol_inkpred(r3d_renderer *r, const float *pred, uint32_t w, uint32_t h,
                        float g0u, float g0v, float px_per_grid) {
  if (!r->sv.active || !w || !h) return -1;
  /* size change: recreate the texture and rebind (device-idle is acceptable
   * here — predictions arrive at most every few seconds) */
  if (r->sv.pred.extent.width != w || r->sv.pred.extent.height != h) {
    pres_drain(r); /* queue ops need external sync */
  vkDeviceWaitIdle(r->vk.dev);
    r3d_vkimage_destroy(&r->vk, &r->sv.pred);
    if (r3d_vkimage_create(&r->vk, VK_FORMAT_R32_SFLOAT, (VkExtent3D){w, h, 1}, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &r->sv.pred) != 0)
      return -1;
    if (r3d_vk_image_to_general(&r->vk, r->pool, &r->sv.pred) != 0) return -1;
    r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->sv.pred.view, r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  }
  { /* tightly packed R32F upload (the staged helper assumes 1-byte texels) */
    r3d_vkbuf stage = {0};
    VkDeviceSize bytes = (VkDeviceSize)w * h * 4u;
    if (r3d_vkbuf_create_host(&r->vk, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stage) != 0)
      return -1;
    memcpy(stage.mapped, pred, bytes);
    VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
    if (!cmd) {
      r3d_vkbuf_destroy(&r->vk, &stage);
      return -1;
    }
    r3d_vk_image_barrier(cmd, r->sv.pred.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
                         VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
    VkBufferImageCopy reg = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {w, h, 1},
    };
    vkCmdCopyBufferToImage(cmd, stage.buf, r->sv.pred.img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    r3d_vk_image_barrier(cmd, r->sv.pred.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
    int urc = r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
    r3d_vkbuf_destroy(&r->vk, &stage);
    if (urc != 0) return -1;
  }
  r->sv.pred_g0u = g0u;
  r->sv.pred_g0v = g0v;
  r->sv.pred_ppg = px_per_grid;
  r->sv.pred_on = true;
  r3d_surfvol_mark(r); /* repaint the window with the new probabilities */
  r->scene_gen++;
  return 0;
}

void r3d_surfvol_mark(r3d_renderer *r) {
  /* Residency arrival with an unchanged window mapping: rewriting any texel
   * subset in place is exactly correct, so re-bake progressively (a row band
   * per frame) instead of hitching one frame with the full-window dispatch.
   * If a pass is already running, let it finish and queue one more full pass
   * (restarting on every arrival would starve the bottom rows under heavy
   * streaming). */
  if (!r->sv.active || r->sv.step <= 0.0f || r->sv.dirty) return;
  memset(r->sv.stale, 0xff, sizeof r->sv.stale); /* every layer may be stale */
  if (r->sv.prog_row != UINT32_MAX)
    r->sv.rebake_again = true; /* the running pass's band must be redone too */
  /* else: the frame loop starts a pass over the visible band next frame */
}

static bool sv_layers_stale(const r3d_renderer *r, uint32_t z0, uint32_t z1) {
  for (uint32_t z = z0; z < z1 && z < 256u; z++)
    if (r->sv.stale[z >> 6] & (1ull << (z & 63u))) return true;
  return false;
}
static void sv_layers_clear(r3d_renderer *r, uint32_t z0, uint32_t z1) {
  for (uint32_t z = z0; z < z1 && z < 256u; z++) r->sv.stale[z >> 6] &= ~(1ull << (z & 63u));
}

void r3d_surfvol_params(const r3d_renderer *r, r3d_frame_params *p) {
  p->slab_x0 = r->sv.u0;
  p->slab_y0 = r->sv.v0;
  p->slab_px = r->sv.step;
  p->slab_nx = (float)r->sv.W;
  p->slab_ny = (float)r->sv.H;
  p->slab_wz = r->sv.L;
  p->slab_grid = r->sv.nback;
  p->max_mip = r->sv.zoff0;
  /* surf views ignore the model transform, so vol_tx/ty carry the tifxyz
   * grid scale (cells per voxel) for the stretch heatmap's grid taps */
  p->vol_tx = r->sv.sx;
  p->vol_ty = r->sv.sy;
}

int r3d_set_transfer(r3d_renderer *r, const uint8_t rgba[256][4]) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  r3d_vkctx_device_wait_idle(&r->vk);
  return upload_small_image(r, &r->tf, rgba, 256 * 4);
}

void r3d_set_quality(r3d_renderer *r, uint32_t quality) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  r->quality = quality < R3D_QUALITY_COUNT ? quality : R3D_QUALITY_FULL;
}

static bool stream_host_copy(const r3d_renderer *r) {
  return r->vk.caps.host_image_copy && r->fp_transition && r->fp_copy_mem;
}

static VkImageUsageFlags stream_image_usage(const r3d_renderer *r) {
  VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (stream_host_copy(r)) u |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
  return u;
}

static int stream_image_to_general(r3d_renderer *r, VkCommandPool pool, r3d_vkimage *img) {
  if (!stream_host_copy(r)) return r3d_vk_image_to_general(&r->vk, pool, img);
  VkHostImageLayoutTransitionInfoEXT tr = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
      .image = img->img,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  return r->fp_transition(r->vk.dev, 1, &tr) == VK_SUCCESS ? 0 : -1;
}

static int stream_copy(r3d_renderer *r, VkCommandPool pool, const uint8_t *host,
                       uint32_t stride, uint32_t dx, uint32_t dy, uint32_t layer,
                       uint32_t w, uint32_t h, r3d_vkimage *img) {
  if (!stream_host_copy(r))
    return r3d_vk_upload_image_staged_buf(
        &r->vk, pool, &r->stream_stage, img, host, stride,
        (VkOffset3D){(int32_t)dx, (int32_t)dy, (int32_t)layer}, (VkExtent3D){w, h, 1});
  VkMemoryToImageCopyEXT region = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
      .pHostPointer = host,
      .memoryRowLength = stride,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {(int32_t)dx, (int32_t)dy, (int32_t)layer},
      .imageExtent = {w, h, 1},
  };
  VkCopyMemoryToImageInfoEXT ci = {
      .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
      .dstImage = img->img,
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &region,
  };
  return r->fp_copy_mem(r->vk.dev, &ci) == VK_SUCCESS ? 0 : -1;
}

/* NxN box-downsample of a flat slice (edge boxes clamp) */
static void slab_downsample(const uint8_t *src, uint32_t nx, uint32_t ny, uint32_t n,
                            uint8_t *dst) {
  uint32_t ox = (nx + n - 1) / n, oy = (ny + n - 1) / n;
  for (uint32_t y = 0; y < oy; y++) {
    uint32_t y0 = y * n, y1 = y0 + n > ny ? ny : y0 + n;
    uint8_t *drow = dst + (size_t)y * ox;
    for (uint32_t x = 0; x < ox; x++) {
      uint32_t x0 = x * n, x1 = x0 + n > nx ? nx : x0 + n;
      uint32_t sum = 0;
      for (uint32_t yy = y0; yy < y1; yy++) {
        const uint8_t *sr = src + (size_t)yy * nx;
        for (uint32_t xx = x0; xx < x1; xx++) sum += sr[xx];
      }
      drow[x] = (uint8_t)(sum / ((y1 - y0) * (x1 - x0)));
    }
  }
}

/* assemble one tile slice (payload + clamped 1-texel apron) from a flat
 * composite slice (row stride l->nx) and host-copy it into ring `layer` */
static int slab_upload_tile_slice(r3d_renderer *r, const r3d_slab_layout *l,
                                  const uint8_t *sbase, uint32_t i, uint32_t j, uint32_t layer,
                                  r3d_vkimage *img) {
  uint32_t tw = r3d_slab_tile_w(l), th = r3d_slab_tile_h(l);
  for (uint32_t dr = 0; dr < th; dr++) {
    const uint8_t *srow = sbase + (size_t)r3d_slab_src_row(l, j, dr) * l->nx;
    uint8_t *drow = r->slice_buf + (size_t)dr * tw;
    int64_t w0 = (int64_t)i * l->px - 1; /* world col of dst col 0 */
    uint32_t lo = w0 < 0 ? (uint32_t)(-w0) : 0;
    int64_t hi64 = (int64_t)l->nx - 1 - w0; /* last dst col with src */
    uint32_t hi = hi64 >= (int64_t)tw - 1 ? tw - 1 : (uint32_t)hi64;
    memcpy(drow + lo, srow + (w0 + lo), hi - lo + 1);
    for (uint32_t c = 0; c < lo; c++) drow[c] = srow[0];
    for (uint32_t c = hi + 1; c < tw; c++) drow[c] = srow[l->nx - 1];
  }
  return stream_copy(r, r->pool, r->slice_buf, tw, 0, 0, layer, tw, th, img);
}

/* Per-tile allocation budget. Total slab payload is nx*ny*wz no matter how it
 * is tiled, so this does not bound memory — it bounds how much of it lands in
 * ONE image. Concentrating 4.8 GB into a single allocation is what fails on an
 * 8 GB card, not the total. */
#define R3D_SLAB_TILE_BUDGET ((VkDeviceSize)1 << 30)

int r3d_slab_init(r3d_renderer *r, const r3d_slab_desc *d) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!r->tiled_modes) {
    fprintf(stderr, "slab: device lacks native non-uniform descriptor indexing\n");
    return -1;
  }
  /* Base tiles may use the device's real 3D image limit (16384 on desktop
   * parts) instead of the 2048 the original Adreno target allowed: fewer,
   * larger tiles mean fewer descriptors, fewer per-tile fill loops and fewer
   * duplicated aprons. Grow the edge from the old default so this can only
   * ever match or improve on previous behaviour. */
  VkDeviceSize budget = r->vk.caps.max_alloc_bytes;
  if (budget > R3D_SLAB_TILE_BUDGET) budget = R3D_SLAB_TILE_BUDGET;
  uint64_t max_texels = (uint64_t)budget / (d->wz ? d->wz : 1u);
  uint32_t cap = r->vk.caps.max_dim_3d < R3D_SLAB_MAX_TILE ? r->vk.caps.max_dim_3d
                                                           : R3D_SLAB_MAX_TILE;
  while (cap < r->vk.caps.max_dim_3d && (uint64_t)(cap + 1) * (cap + 1) <= max_texels) cap++;
  if (r3d_slab_layout_init_cap(&r->slab, d->nx, d->ny, d->nz, d->wz, cap) != 0) {
    fprintf(stderr, "slab: unsupported layout %ux%ux%u wz=%u (max %u per tiled axis)\n", d->nx,
            d->ny, d->nz, d->wz, R3D_SLAB_MAX_GRID * (cap - 2));
    return -1;
  }
  printf("slab: tile cap %u (device %u) -> %ux%u grid, payload %ux%u, %.0f MiB/tile\n", cap,
         r->vk.caps.max_dim_3d, r->slab.gx, r->slab.gy, r->slab.px, r->slab.py,
         (double)((uint64_t)r3d_slab_tile_w(&r->slab) * r3d_slab_tile_h(&r->slab) * r->slab.wz) /
             1048576.0);
  printf("slab: streaming upload path %s\n", stream_host_copy(r) ? "host-image-copy" : "staging");
  uint32_t tw = r3d_slab_tile_w(&r->slab), th = r3d_slab_tile_h(&r->slab);
  r->slice_buf = malloc((size_t)tw * th);
  if (!r->slice_buf) return -1;

  VkSamplerCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT, /* ring z wraps */
  };
  if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_slab) != VK_SUCCESS) return -1;

  for (uint32_t j = 0; j < r->slab.gy; j++)
    for (uint32_t i = 0; i < r->slab.gx; i++) {
      r3d_vkimage *t = &r->tiles[j * r->slab.gx + i];
      if (r3d_vkimage_create_arena(&r->vk, &r->tile_arena, VK_FORMAT_R8_UNORM,
                                   (VkExtent3D){tw, th, r->slab.wz}, 1,
                                   stream_image_usage(r), t) != 0)
        return -1;
      if (stream_image_to_general(r, r->pool, t) != 0) return -1;
    }

  /* overview pyramid: prefiltered levels at scale 4<<lev up to ovs, each a
   * tiled virtual slab over the downsampled composite (full ring z). Appended
   * to tiles[] right after the base grid; the shader picks the level whose
   * prefilter matches the ray-cone footprint (mip-style). */
  r->ov_nlev = r3d_slab_ov_levels(&r->slab);
  uint32_t nt = r->slab.gx * r->slab.gy;
  for (uint32_t lev = 0; lev < r->ov_nlev; lev++) {
    r3d_slab_ov_layout(&r->slab, lev, &r->ovl[lev]);
    const r3d_slab_layout *ol = &r->ovl[lev];
    r->ov_base[lev] = nt;
    nt += ol->gx * ol->gy;
    if (nt > R3D_SLAB_TILES) {
      fprintf(stderr, "slab: tile pool exhausted (%u)\n", nt);
      return -1;
    }
    r->ov_bufs[lev] = malloc((size_t)ol->nx * ol->ny);
    if (!r->ov_bufs[lev]) return -1;
    uint32_t otw = r3d_slab_tile_w(ol), oth = r3d_slab_tile_h(ol);
    for (uint32_t j = 0; j < ol->gy; j++)
      for (uint32_t i = 0; i < ol->gx; i++) {
        r3d_vkimage *t = &r->tiles[r->ov_base[lev] + j * ol->gx + i];
        if (r3d_vkimage_create_arena(&r->vk, &r->tile_arena, VK_FORMAT_R8_UNORM,
                                     (VkExtent3D){otw, oth, r->slab.wz}, 1,
                                     stream_image_usage(r), t) != 0)
          return -1;
        if (stream_image_to_general(r, r->pool, t) != 0) return -1;
      }
  }

  VkImageView views[R3D_SLAB_TILES];
  for (uint32_t e = 0; e < R3D_SLAB_TILES; e++)
    views[e] = r->tiles[e].view ? r->tiles[e].view : r->tiles[0].view;
  /* NOTE: descriptors must use GENERAL for these images */
  VkDescriptorImageInfo ii[R3D_SLAB_TILES];
  for (uint32_t e = 0; e < R3D_SLAB_TILES; e++)
    ii[e] = (VkDescriptorImageInfo){
        .sampler = r->samp_slab, .imageView = views[e], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = r->tile_descriptors,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);

  r->slab_mode = true;
  r->slab_z0 = -1;
  return 0;
}

int r3d_slab_window(r3d_renderer *r, const r3d_volume *src, uint32_t z0) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!r->slab_mode) return -1;
  if (z0 > r3d_slab_z0_max(&r->slab)) z0 = r3d_slab_z0_max(&r->slab);
  if ((int64_t)z0 == r->slab_z0) return 0;

  uint32_t s0, s1;
  int full;
  r3d_slab_scroll_range(&r->slab, r->slab_z0, (int64_t)z0, &s0, &s1, &full);
  uint64_t t0 = now_ns();

  /* in-flight frames may sample layers we are about to overwrite */
  if (r->timeline_value) {
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &r->timeline,
        .pValues = &r->timeline_value,
    };
    if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return -1;
  }

  const r3d_slab_layout *l = &r->slab;
  for (uint32_t s = s0; s < s1; s++) {
    uint32_t layer = r3d_slab_ring_layer(l, s);
    const uint8_t *sbase = src->voxels + (size_t)s * src->ny * src->nx;
    for (uint32_t j = 0; j < l->gy; j++)
      for (uint32_t i = 0; i < l->gx; i++)
        if (slab_upload_tile_slice(r, l, sbase, i, j, layer, &r->tiles[j * l->gx + i]) != 0)
          return -1;

    /* overview pyramid: 4x box down from the source slice, then 2x chains;
     * each level uploads through the same tile-assembly path */
    for (uint32_t lev = 0; lev < r->ov_nlev; lev++) {
      const r3d_slab_layout *ol = &r->ovl[lev];
      if (lev == 0)
        slab_downsample(sbase, l->nx, l->ny, 4, r->ov_bufs[0]);
      else
        slab_downsample(r->ov_bufs[lev - 1], r->ovl[lev - 1].nx, r->ovl[lev - 1].ny, 2,
                        r->ov_bufs[lev]);
      for (uint32_t j = 0; j < ol->gy; j++)
        for (uint32_t i = 0; i < ol->gx; i++)
          if (slab_upload_tile_slice(r, ol, r->ov_bufs[lev], i, j, layer,
                                     &r->tiles[r->ov_base[lev] + j * ol->gx + i]) != 0)
            return -1;
    }
  }
  double ms = (double)(now_ns() - t0) / 1e6;
  if (full || ms > 5.0)
    printf("slab: %s %u slice(s) -> z0=%u in %.1f ms\n", full ? "window" : "scrolled", s1 - s0,
           z0, ms);
  r->slab_z0 = (int64_t)z0;
  return 0;
}

void r3d_slab_params(const r3d_renderer *r, r3d_frame_params *p) {
  p->slab_grid = r->slab.gx | (r->slab.gy << 8) | (r->slab.ovs << 16);
  p->slab_wz = r->slab.wz;
  p->slab_z0 = (float)r->slab_z0;
  p->slab_nx = (float)r->slab.nx;
  p->slab_ny = (float)r->slab.ny;
  p->slab_px = (float)r->slab.px;
  p->slab_py = (float)r->slab.py;
  p->slab_depth = r->slab.wz - 2; /* max; caller may lower it per frame */
}

/* ---------- bricks: two-tier GPU cache (warm compressed / hot atlas) ------ */

#define BR_INVALID 0xFFFFFFFFu
#define BR_SLOT_DIM 128u
#define BR_SHARD_BPA 8u
#define BR_AMIPS 4u
#define BR_NOISE_FLOOR 5 /* decoded-air codec noise ceiling (LSBs) */
#define BR_MAX_BATCH 32u
#define BR_RAW_BYTES ((size_t)BR_SLOT_DIM * BR_SLOT_DIM * BR_SLOT_DIM)

static uint32_t bricks_page_index(const r3d_renderer *r, uint32_t b) {
  return (r->bricks_lod ? BR_PAGE_HEADER : 0u) + b;
}

static int parse_u64_triplet(const char *p, uint64_t out[3]) {
  const char *b = strchr(p, '[');
  if (!b) return -1;
  unsigned long long a = 0, c = 0, d = 0;
  if (sscanf(b, "[ %llu , %llu , %llu", &a, &c, &d) != 3) return -1;
  out[0] = (uint64_t)a;
  out[1] = (uint64_t)c;
  out[2] = (uint64_t)d;
  return 0;
}

/* Read lodpack's deliberately small, self-describing manifest.  This parser
 * only accepts the render3d.c5d-lod.v1 schema; it is not a general JSON
 * parser, so malformed/external JSON fails closed instead of being guessed. */
static int bricks_manifest_open(r3d_renderer *r, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long ln = ftell(f);
  if (ln <= 0 || ln > (1 << 20) || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }
  char *json = malloc((size_t)ln + 1u);
  if (!json) {
    fclose(f);
    return -1;
  }
  int rc = -1;
  if (fread(json, 1, (size_t)ln, f) != (size_t)ln) goto done;
  json[ln] = 0;
  if (!strstr(json, "\"format\": \"render3d.c5d-lod.v1\"")) goto done;
  const char *shape = strstr(json, "\"shape\"");
  uint64_t base[3]; /* manifest order z,y,x */
  if (!shape || parse_u64_triplet(shape, base) != 0 || base[0] > UINT32_MAX ||
      base[1] > UINT32_MAX || base[2] > UINT32_MAX)
    goto done;

  const char *levels = strstr(json, "\"levels\"");
  if (!levels) goto done;
  uint32_t nlev = 0, pages = 0, shards = 0;
  const char *p = levels;
  while (nlev < BR_LOD_MAX && (p = strstr(p, "\"level\""))) {
    unsigned lev = UINT32_MAX, scale = 0;
    const char *colon = strchr(p, ':');
    const char *sp = strstr(p, "\"scale\"");
    const char *shp = strstr(p, "\"shape\"");
    const char *shards_p = strstr(p, "\"shards\"");
    if (!colon || !sp || !shp || !shards_p || sscanf(colon + 1, " %u", &lev) != 1 ||
        !(colon = strchr(sp, ':')) || sscanf(colon + 1, " %u", &scale) != 1 || lev != nlev ||
        scale != (1u << lev))
      goto done;
    uint64_t vd[3], sd[3];
    if (parse_u64_triplet(shp, vd) != 0 || parse_u64_triplet(shards_p, sd) != 0) goto done;
    r3d_brlod_level *l = &r->bricks_lev[nlev];
    if (vd[0] > UINT32_MAX || vd[1] > UINT32_MAX || vd[2] > UINT32_MAX ||
        sd[0] > UINT32_MAX || sd[1] > UINT32_MAX || sd[2] > UINT32_MAX)
      goto done;
    *l = (r3d_brlod_level){.scale = scale,
                           .nx = (uint32_t)vd[2],
                           .ny = (uint32_t)vd[1],
                           .nz = (uint32_t)vd[0],
                           .bx = ((uint32_t)vd[2] + 127u) / 128u,
                           .by = ((uint32_t)vd[1] + 127u) / 128u,
                           .bz = ((uint32_t)vd[0] + 127u) / 128u,
                           .sx = (uint32_t)sd[2],
                           .sy = (uint32_t)sd[1],
                           .sz = (uint32_t)sd[0],
                           .page_off = pages,
                           .shard_off = shards};
    if (l->bx > 1023u || l->by > 1023u || l->bz > 1023u || !l->bx || !l->by || !l->bz ||
        !l->sx || !l->sy || !l->sz)
      goto done;
    uint64_t np = (uint64_t)l->bx * l->by * l->bz;
    uint64_t ns = (uint64_t)l->sx * l->sy * l->sz;
    if (np > UINT32_MAX - pages || ns > UINT32_MAX - shards) goto done;
    pages += (uint32_t)np;
    shards += (uint32_t)ns;
    nlev++;
    p = shards_p + 8;
  }
  if (!nlev || r->bricks_lev[0].nx != base[2] || r->bricks_lev[0].ny != base[1] ||
      r->bricks_lev[0].nz != base[0])
    goto done;
  r->bricks_readers = calloc(shards, sizeof *r->bricks_readers);
  if (!r->bricks_readers) goto done;
  const char *slash = strrchr(path, '/');
  size_t root_n = slash ? (size_t)(slash - path) : 1u;
  if (root_n >= sizeof r->bricks_root) goto done;
  if (slash)
    memcpy(r->bricks_root, path, root_n);
  else
    r->bricks_root[0] = '.';
  r->bricks_root[root_n] = 0;
  r->bricks_lod = true;
  r->bricks_nlev = nlev;
  r->bricks_nreaders = shards;
  r->bricks_nx = (uint32_t)base[2];
  r->bricks_ny = (uint32_t)base[1];
  r->bricks_nz = (uint32_t)base[0];
  r->bricks_maxdim = r->bricks_nx;
  if (r->bricks_ny > r->bricks_maxdim) r->bricks_maxdim = r->bricks_ny;
  if (r->bricks_nz > r->bricks_maxdim) r->bricks_maxdim = r->bricks_nz;
  r->bs.nb = pages;
  rc = 0;
done:
  if (rc != 0) {
    free(r->bricks_readers);
    r->bricks_readers = NULL;
  }
  free(json);
  fclose(f);
  return rc;
}

static const uint8_t *brlod_blob(r3d_renderer *r, const char *root,
                                 r3d_brlod_reader *readers, uint32_t b, size_t *n) {
  uint32_t li = 0;
  while (li + 1u < r->bricks_nlev && b >= r->bricks_lev[li + 1u].page_off) li++;
  const r3d_brlod_level *l = &r->bricks_lev[li];
  uint32_t local = b - l->page_off;
  uint32_t bx = local % l->bx, by = (local / l->bx) % l->by,
           bz = local / (l->bx * l->by);
  uint32_t sx = bx / BR_SHARD_BPA, sy = by / BR_SHARD_BPA, sz = bz / BR_SHARD_BPA;
  uint32_t ri = l->shard_off + (sz * l->sy + sy) * l->sx + sx;
  if (ri >= r->bricks_nreaders) return NULL;
  r3d_brlod_reader *rd = &readers[ri];
  if (rd->failed) return NULL;
  if (!rd->open) {
    char path[1400];
    int pn = snprintf(path, sizeof path, "%s/c5d/L%u/%u_%u_%u.c5s", root, li, sz, sy, sx);
    if (pn < 0 || (size_t)pn >= sizeof path || c5d_shard_open(path, &rd->sr) != 0) {
      rd->failed = true;
      return NULL;
    }
    if (rd->sr.foot.brick_dim != BR_SLOT_DIM || rd->sr.foot.shard_dim != 1024u ||
        rd->sr.foot.lod_level != li) {
      c5d_shard_close_reader(&rd->sr);
      rd->failed = true;
      return NULL;
    }
    rd->open = true;
  }
  uint32_t lbx = bx % BR_SHARD_BPA, lby = by % BR_SHARD_BPA, lbz = bz % BR_SHARD_BPA;
  uint32_t bi = (lbz * BR_SHARD_BPA + lby) * BR_SHARD_BPA + lbx;
  return c5d_shard_brick(&rd->sr, bi, n);
}

/* global brick index -> (level, brick coords) */
static void brlod_locate(r3d_renderer *r, uint32_t b, uint32_t *li, uint32_t *bx,
                         uint32_t *by, uint32_t *bz) {
  uint32_t l = 0;
  while (l + 1u < r->bricks_nlev && b >= r->bricks_lev[l + 1u].page_off) l++;
  const r3d_brlod_level *lv = &r->bricks_lev[l];
  uint32_t local = b - lv->page_off;
  *li = l;
  *bx = local % lv->bx;
  *by = (local / lv->bx) % lv->by;
  *bz = local / (lv->bx * lv->by);
}

static void ni_brick_path(r3d_renderer *r, char path[1400], int nsrc, uint32_t li,
                          uint32_t bz, uint32_t by, uint32_t bx) {
  snprintf(path, 1400, "%s/bricks/L%u/%u_%u_%u.c5b", nsrc ? r->ni.root2 : r->bricks_root,
           li, bz, by, bx);
}

static const uint8_t *bricks_source_blob(r3d_renderer *r, uint32_t b, size_t *n) {
  if (!r->bricks_lod) return c5d_shard_brick(&r->bs.sr, b, n);
  return brlod_blob(r, r->bricks_root, r->bricks_readers, b, n);
}

/* Load a net-ingested brick blob from its cache file (WORKER THREAD ONLY —
 * the render thread never performs file IO; it selects these bricks via the
 * ni.have map and the worker resolves them here). malloc'd result. */
static uint8_t *ni_load_brick(r3d_renderer *r, uint32_t b, size_t *n, int nsrc) {
  uint32_t li, bx, by, bz;
  brlod_locate(r, b, &li, &bx, &by, &bz);
  char path[1400];
  ni_brick_path(r, path, nsrc, li, bz, by, bx);
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long fn = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (fn <= 0) {
    fclose(f);
    return NULL;
  }
  uint8_t *buf = malloc((size_t)fn);
  size_t got = buf ? fread(buf, 1, (size_t)fn, f) : 0;
  fclose(f);
  if (got != (size_t)fn) {
    free(buf);
    return NULL;
  }
  *n = got;
  return buf;
}

/* Ingest unit: a CELL of lcm(chunk, brick) voxels per axis — one chunk
 * when the chunk edge is a brick multiple (128/256/512), a 2x2x2 chunk
 * group of 3x3x3 bricks for 192^3 trees (the eligible-scroll surface
 * predictions), where bricks straddle chunk boundaries. */
static uint32_t ni_cell_dim(uint32_t chsz) {
  uint32_t a = chsz, b = BR_SLOT_DIM;
  while (b) {
    uint32_t t = a % b;
    a = b;
    b = t;
  }
  return chsz / a * BR_SLOT_DIM; /* lcm */
}

/* Enqueue the cell that owns brick b (dedup against queue+inflight).
 * Returns true while the brick may still arrive (queued/inflight/back-off),
 * false when the cache says the brick is definitively absent upstream. */
static bool bricks_net_request(r3d_renderer *r, uint32_t b, int nsrc) {
  if (!r->ni.active || (nsrc && !r->ni.url2[0])) return false;
  _Atomic uint8_t *hv = nsrc ? r->ni.have2 : r->ni.have;
  if (atomic_load(&hv[b]) == 2u) return false; /* definitively absent */
  uint32_t li, bx, by, bz;
  brlod_locate(r, b, &li, &bx, &by, &bz);
  uint32_t cb = ni_cell_dim(nsrc ? r->ni.chsz2[li] : r->ni.chsz[li]) / BR_SLOT_DIM;
  uint64_t id = ((uint64_t)nsrc << 63) | ((uint64_t)li << 48) |
                ((uint64_t)(bz / cb) << 32) | ((uint64_t)(by / cb) << 16) |
                (uint64_t)(bx / cb);
  pthread_mutex_lock(&r->ni.mu);
  int fq = -1;
  for (uint32_t i = 0; i < r->ni.qn && fq < 0; i++)
    if (r->ni.queue[i] == id) fq = (int)i;
  bool known = fq >= 0;
  for (uint32_t i = 0; i < r->ni.nin && !known; i++) known = r->ni.inflight[i] == id;
  if (fq >= (int)r->ni.qins) { /* backlog entry re-requested: promote it into
                                * the current pass's head block */
    memmove(r->ni.queue + r->ni.qins + 1, r->ni.queue + r->ni.qins,
            ((uint32_t)fq - r->ni.qins) * sizeof id);
    r->ni.queue[r->ni.qins++] = id;
  }
  if (!known) {
    if (r->ni.qn >= 256u) r->ni.qn = 255u; /* full: the stalest tail drops */
    if (r->ni.qins > r->ni.qn) r->ni.qins = r->ni.qn;
    memmove(r->ni.queue + r->ni.qins + 1, r->ni.queue + r->ni.qins,
            (r->ni.qn - r->ni.qins) * sizeof id);
    r->ni.queue[r->ni.qins++] = id;
    r->ni.qn++;
    pthread_cond_signal(&r->ni.cv);
  }
  pthread_mutex_unlock(&r->ni.mu);
  return true;
}

static size_t ni_curl_write(const void *data, size_t sz, size_t nm, void *ud) {
  struct ni_buf { uint8_t *p; size_t n, cap; } *bf = ud;
  size_t n = sz * nm;
  if (bf->n + n > bf->cap) {
    size_t nc = bf->cap ? bf->cap * 2 : (4u << 20);
    while (nc < bf->n + n) nc *= 2;
    uint8_t *np = realloc(bf->p, nc);
    if (!np) return 0;
    bf->p = np;
    bf->cap = nc;
  }
  memcpy(bf->p + bf->n, data, n);
  bf->n += n;
  return n;
}

static int ni_write_file(const char *path, const void *data, size_t n) {
  char tmp[1460];
  int pn = snprintf(tmp, sizeof tmp, "%s.tmp.%ld.%lx", path, (long)getpid(),
                    (unsigned long)pthread_self());
  if (pn < 0 || (size_t)pn >= sizeof tmp) return -1;
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  int rc = (n == 0 || fwrite(data, 1, n, f) == n) && fflush(f) == 0 ? 0 : -1;
  if (fclose(f) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

static int ni_abort_cb(void *ud, curl_off_t dt, curl_off_t dn, curl_off_t ut,
                       curl_off_t un) {
  (void)dt;
  (void)dn;
  (void)ut;
  (void)un;
  const r3d_renderer *r = ud;
  return r->ni.quit ? 1 : 0; /* nonzero aborts the transfer promptly */
}

static void *ni_worker(void *arg) {
  r3d_renderer *r = arg;
  CURL *curl = curl_easy_init();
  struct { uint8_t *p; size_t n, cap; } buf = {0};
  uint8_t *chunk = NULL, *cellbuf = NULL,
          *raw = malloc((size_t)BR_SLOT_DIM * BR_SLOT_DIM * BR_SLOT_DIM);
  size_t chunk_cap = 0, cell_cap = 0;
  if (!curl || !raw) goto out;
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ni_curl_write);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ni_abort_cb);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, r);
  for (;;) {
    pthread_mutex_lock(&r->ni.mu);
    while (!r->ni.quit && r->ni.qn == 0) pthread_cond_wait(&r->ni.cv, &r->ni.mu);
    if (r->ni.quit) {
      pthread_mutex_unlock(&r->ni.mu);
      break;
    }
    uint64_t id = r->ni.queue[0]; /* newest pass first, nearest-first within */
    memmove(r->ni.queue, r->ni.queue + 1, --r->ni.qn * sizeof id);
    if (r->ni.qins) r->ni.qins--;
    if (r->ni.nin < 64u) r->ni.inflight[r->ni.nin++] = id;
    pthread_mutex_unlock(&r->ni.mu);

    int nsrc = (int)(id >> 63);
    uint32_t li = (uint32_t)(id >> 48) & 0x3fffu, cz = (uint32_t)(id >> 32) & 0xffffu,
             cy = (uint32_t)(id >> 16) & 0xffffu, cx = (uint32_t)id & 0xffffu;
    _Atomic uint8_t *hvm = nsrc ? r->ni.have2 : r->ni.have;
    uint32_t chsz = nsrc ? r->ni.chsz2[li] : r->ni.chsz[li];
    uint32_t cell = ni_cell_dim(chsz), cb = cell / BR_SLOT_DIM, cc = cell / chsz;
    size_t chunk_bytes = (size_t)chsz * chsz * chsz;
    size_t cell_bytes = (size_t)cell * cell * cell;
    { /* cache files from an earlier session? publish them without fetching */
      const r3d_brlod_level *lv = &r->bricks_lev[li];
      bool all_known = true;
      for (uint32_t sz_ = 0; sz_ < cb && all_known; sz_++)
        for (uint32_t sy = 0; sy < cb && all_known; sy++)
          for (uint32_t sx = 0; sx < cb; sx++) {
            uint32_t bz = cz * cb + sz_, by = cy * cb + sy, bx = cx * cb + sx;
            if (bx >= lv->bx || by >= lv->by || bz >= lv->bz) continue;
            char path[1400];
            ni_brick_path(r, path, nsrc, li, bz, by, bx);
            struct stat st;
            if (stat(path, &st) != 0) {
              all_known = false;
              break;
            }
            atomic_store(&hvm[lv->page_off + (bz * lv->by + by) * lv->bx + bx],
                         st.st_size ? 1u : 2u);
          }
      if (all_known) {
        pthread_mutex_lock(&r->ni.mu);
        for (uint32_t i = 0; i < r->ni.nin; i++)
          if (r->ni.inflight[i] == id) {
            r->ni.inflight[i] = r->ni.inflight[--r->ni.nin];
            break;
          }
        pthread_mutex_unlock(&r->ni.mu);
        continue;
      }
    }
    bool netfail = false, quitting = false, have = false;
    if (cell_cap < cell_bytes) {
      uint8_t *ncb = realloc(cellbuf, cell_bytes);
      if (ncb) {
        cellbuf = ncb;
        cell_cap = cell_bytes;
      } else
        netfail = true;
    }
    if (!netfail) memset(cellbuf, 0, cell_bytes);
    for (uint32_t icz = 0; icz < cc && !netfail && !quitting; icz++)
      for (uint32_t icy = 0; icy < cc && !netfail && !quitting; icy++)
        for (uint32_t icx = 0; icx < cc; icx++) {
          char url[1400];
          snprintf(url, sizeof url, "%s/%u/%u/%u/%u", nsrc ? r->ni.url2 : r->ni.url, li,
                   cz * cc + icz, cy * cc + icy, cx * cc + icx);
          long code = 0;
          CURLcode crc = CURLE_OK;
          for (int attempt = 0; attempt < 4; attempt++) {
            buf.n = 0;
            curl_easy_setopt(curl, CURLOPT_URL, url);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
            crc = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
            if (crc == CURLE_OK && (code == 200 || code == 404)) break;
            if (r->ni.quit) break;
            for (int w = 0; w < (1 << attempt) && !r->ni.quit; w++) sleep(1);
          }
          if (r->ni.quit) { /* teardown: leave the cell unfinished */
            quitting = true;
            break;
          }
          if (!(crc == CURLE_OK && (code == 200 || code == 404))) {
            netfail = true;
            break;
          }
          if (code == 404) {
            atomic_fetch_add(&r->ni.absent_chunks, 1);
            continue; /* absent = air; cell is pre-zeroed */
          }
          atomic_fetch_add(nsrc ? &r->ni.fetched2 : &r->ni.fetched, 1);
          bool ok = true;
          if (chunk_cap < chunk_bytes) {
            uint8_t *nc = realloc(chunk, chunk_bytes);
            if (nc) {
              chunk = nc;
              chunk_cap = chunk_bytes;
            } else
              ok = false;
          }
          if (ok) {
            if (nsrc ? r->ni.raw2[li] : r->ni.raw[li]) {
              ok = buf.n == chunk_bytes;
              if (ok) memcpy(chunk, buf.p, chunk_bytes);
            } else {
              size_t nbytes = 0, cbytes = 0, blocksize = 0;
              blosc_cbuffer_sizes(buf.p, &nbytes, &cbytes, &blocksize);
              ok = nbytes == chunk_bytes && cbytes <= buf.n &&
                   blosc_decompress_ctx(buf.p, chunk, chunk_bytes, 1) == (int)chunk_bytes;
            }
          }
          if (!ok) { /* bad payload reads as air */
            fprintf(stderr, "bricks: bad chunk payload %s (%zu bytes)\n", url, buf.n);
            continue;
          }
          for (uint32_t zz = 0; zz < chsz; zz++) /* blit into the cell */
            for (uint32_t yy = 0; yy < chsz; yy++)
              memcpy(cellbuf + (((size_t)icz * chsz + zz) * cell +
                                ((size_t)icy * chsz + yy)) *
                                   cell +
                         (size_t)icx * chsz,
                     chunk + ((size_t)zz * chsz + yy) * chsz, chsz);
          have = true;
        }
    if (quitting) {
      pthread_mutex_lock(&r->ni.mu);
      for (uint32_t i = 0; i < r->ni.nin; i++)
        if (r->ni.inflight[i] == id) {
          r->ni.inflight[i] = r->ni.inflight[--r->ni.nin];
          break;
        }
      pthread_mutex_unlock(&r->ni.mu);
      break;
    }
    if (!netfail) {
      const r3d_brlod_level *lv = &r->bricks_lev[li];
      const char *rt = nsrc ? r->ni.root2 : r->bricks_root;
      char dir[1360];
      snprintf(dir, sizeof dir, "%s/bricks", rt);
      mkdir(dir, 0755);
      snprintf(dir, sizeof dir, "%s/bricks/L%u", rt, li);
      mkdir(dir, 0755);
      float q = (nsrc ? r->ni.q02 : r->ni.q0) / (float)(1u << (li < 3u ? li : 3u));
      if (q < 0.25f) q = 0.25f;
      for (uint32_t sz_ = 0; sz_ < cb; sz_++)
        for (uint32_t sy = 0; sy < cb; sy++)
          for (uint32_t sx = 0; sx < cb; sx++) {
            uint32_t bz = cz * cb + sz_, by = cy * cb + sy, bx = cx * cb + sx;
            if (bx >= lv->bx || by >= lv->by || bz >= lv->bz) continue;
            char path[1400];
            ni_brick_path(r, path, nsrc, li, bz, by, bx);
            struct stat st;
            if (stat(path, &st) == 0) { /* already cached by an earlier run */
              atomic_store(&hvm[lv->page_off + (bz * lv->by + by) * lv->bx + bx],
                           st.st_size ? 1u : 2u);
              continue;
            }
            bool zero = !have;
            if (have) {
              for (uint32_t rr = 0; rr < BR_SLOT_DIM; rr++)
                for (uint32_t qq = 0; qq < BR_SLOT_DIM; qq++)
                  memcpy(raw + ((size_t)rr * BR_SLOT_DIM + qq) * BR_SLOT_DIM,
                         cellbuf + (((size_t)(sz_ * BR_SLOT_DIM + rr) * cell +
                                     sy * BR_SLOT_DIM + qq) *
                                        cell +
                                    sx * BR_SLOT_DIM),
                         BR_SLOT_DIM);
              zero = true;
              for (size_t v = 0; v < (size_t)BR_SLOT_DIM * BR_SLOT_DIM * BR_SLOT_DIM; v++)
                if (raw[v]) {
                  zero = false;
                  break;
                }
            }
            uint32_t gb = lv->page_off + (bz * lv->by + by) * lv->bx + bx;
            if (zero) {
              ni_write_file(path, NULL, 0); /* empty marker = air/absent */
              atomic_store(&hvm[gb], 2u);
              continue;
            }
            c5d_brick_params bp = c5d_brick_defaults(1.0f);
            bp.q = q;
            uint8_t *enc = NULL;
            size_t en = 0;
            if (c5d_brick_encode(&bp, raw, BR_SLOT_DIM, &enc, &en) == 0) {
              ni_write_file(path, enc, en);
              atomic_store(&hvm[gb], 1u);
              atomic_fetch_add(&r->ni.encoded, 1);
              free(enc);
            }
          }
    }
    pthread_mutex_lock(&r->ni.mu);
    for (uint32_t i = 0; i < r->ni.nin; i++)
      if (r->ni.inflight[i] == id) {
        r->ni.inflight[i] = r->ni.inflight[--r->ni.nin];
        break;
      }
    pthread_mutex_unlock(&r->ni.mu);
  }
out:
  if (curl) curl_easy_cleanup(curl);
  free(buf.p);
  free(chunk);
  free(cellbuf);
  free(raw);
  return NULL;
}

static int bcand_cmp(const void *a, const void *b) {
  const struct bcand *ca = a, *cb = b;
  if (ca->priority != cb->priority) return ca->priority < cb->priority ? -1 : 1;
  float d = ca->d2 - cb->d2;
  return d < 0.0f ? -1 : (d > 0.0f ? 1 : 0);
}

static void bricks_candidate(r3d_renderer *r, uint32_t b, float d2, uint32_t priority, int gate,
                             uint32_t *ncand) {
  if (r->bs.brick_maxk[b] >= 0 && r->bs.brick_maxk[b] < gate) return;
  uint32_t slot = r->bs.brick_slot[b];
  if (slot != BR_INVALID) {
    r->bs.slot_use[slot] = r->bs.frame;
    if (r->bs.warm_off[b] != BR_INVALID) r->bs.warm_use[b] = r->bs.frame;
    return;
  }
  if (r->bs.brick_want[b] == r->bs.frame) return;
  r->bs.brick_want[b] = r->bs.frame;
  r->bs.cands[(*ncand)++] = (struct bcand){d2, b, priority};
}

static void bricks_axis_bounds(float eye, float radius, float edge, uint32_t n, uint32_t *lo,
                               uint32_t *hi) {
  int a = (int)ceilf((eye - radius) / edge - 0.5f);
  int b = (int)floorf((eye + radius) / edge - 0.5f) + 1;
  if (a < 0) a = 0;
  if (b < 0) b = 0;
  if (a > (int)n) a = (int)n;
  if (b > (int)n) b = (int)n;
  *lo = (uint32_t)a;
  *hi = (uint32_t)b;
}

/* warm-tier allocator: offset-sorted first-fit free list, 64-byte granules */
static uint32_t warm_align(uint32_t n) { return (n + 63u) & ~63u; }

static uint32_t warm_alloc(r3d_renderer *r, uint32_t len) {
  len = warm_align(len);
  for (uint32_t i = 0; i < r->bs.nwf; i++) {
    struct wfnode *f = &r->bs.wfree[i];
    if (f->len < len) continue;
    uint32_t off = f->off;
    f->off += len;
    f->len -= len;
    if (f->len == 0) memmove(f, f + 1, (size_t)(--r->bs.nwf - i) * sizeof *f);
    return off;
  }
  return BR_INVALID;
}

static void warm_release(r3d_renderer *r, uint32_t off, uint32_t rawlen) {
  uint32_t len = warm_align(rawlen);
  struct wfnode *fl = r->bs.wfree;
  uint32_t i = 0;
  while (i < r->bs.nwf && fl[i].off < off) i++;
  if (i > 0 && fl[i - 1].off + fl[i - 1].len == off) { /* merge into predecessor */
    fl[i - 1].len += len;
    if (i < r->bs.nwf && fl[i - 1].off + fl[i - 1].len == fl[i].off) {
      fl[i - 1].len += fl[i].len;
      memmove(&fl[i], &fl[i + 1], (size_t)(--r->bs.nwf - i) * sizeof *fl);
    }
    return;
  }
  if (i < r->bs.nwf && off + len == fl[i].off) { /* merge into successor */
    fl[i].off = off;
    fl[i].len += len;
    return;
  }
  if (r->bs.nwf == r->bs.cwf) {
    r->bs.cwf = r->bs.cwf ? r->bs.cwf * 2 : 64;
    r->bs.wfree = fl = realloc(fl, (size_t)r->bs.cwf * sizeof *fl);
  }
  memmove(&fl[i + 1], &fl[i], (size_t)(r->bs.nwf++ - i) * sizeof *fl);
  fl[i] = (struct wfnode){off, len};
}

static bool warm_evict_one(r3d_renderer *r) {
  /* LRU over the compact warm-resident set — never the full virtual-brick
   * range (44M bricks for a large LOD tree; scanning that per eviction once
   * cost ~10% of the render thread on PHercParis4). */
  uint32_t best = BR_INVALID, besti = 0, bu = UINT32_MAX;
  for (uint32_t i = 0; i < r->bs.warm_bricks; i++) {
    uint32_t b = r->bs.warm_list[i];
    if (r->bs.warm_use[b] != r->bs.frame && r->bs.warm_use[b] < bu) {
      bu = r->bs.warm_use[b];
      best = b;
      besti = i;
    }
  }
  if (best == BR_INVALID) return false;
  warm_release(r, r->bs.warm_off[best], r->bs.warm_len[best]);
  r->bs.warm_bytes -= r->bs.warm_len[best];
  r->bs.warm_off[best] = BR_INVALID;
  r->bs.warm_list[besti] = r->bs.warm_list[--r->bs.warm_bricks];
  return true;
}

/* compressed blob for brick b, resident in the warm tier when it fits (LRU
 * evictions as needed); falls back to the mmap'd shard when the tier thrashes.
 * Current-frame entries are never evicted, so batch blob pointers stay valid. */
static const uint8_t *warm_get(r3d_renderer *r, uint32_t b, size_t *n) {
  if (r->bs.warm_off[b] != BR_INVALID) {
    r->bs.warm_use[b] = r->bs.frame;
    *n = r->bs.warm_len[b];
    return (const uint8_t *)r->bs.warm.mapped + r->bs.warm_off[b];
  }
  size_t sz = 0;
  const uint8_t *blob = bricks_source_blob(r, b, &sz);
  if (!blob) return NULL;
  *n = sz;
  if ((uint64_t)warm_align((uint32_t)sz) > r->bs.warm_cap) return blob;
  uint32_t off;
  while ((off = warm_alloc(r, (uint32_t)sz)) == BR_INVALID)
    if (!warm_evict_one(r)) return blob;
  if (r->bs.warm_bricks == r->bs.warm_list_cap) {
    uint32_t nc = r->bs.warm_list_cap ? r->bs.warm_list_cap * 2u : 1024u;
    uint32_t *nl = realloc(r->bs.warm_list, (size_t)nc * sizeof *nl);
    if (!nl) {
      warm_release(r, off, (uint32_t)sz);
      return blob;
    }
    r->bs.warm_list = nl;
    r->bs.warm_list_cap = nc;
  }
  memcpy((uint8_t *)r->bs.warm.mapped + off, blob, sz);
  r->bs.warm_off[b] = off;
  r->bs.warm_len[b] = (uint32_t)sz;
  r->bs.warm_use[b] = r->bs.frame;
  r->bs.warm_list[r->bs.warm_bricks] = b;
  r->bs.warm_bricks++;
  r->bs.warm_bytes += sz;
  return (const uint8_t *)r->bs.warm.mapped + off;
}

static uint32_t bricks_pick_slot(const r3d_renderer *r) {
  uint32_t best = BR_INVALID, bu = UINT32_MAX;
  for (uint32_t s = 0; s < r->bs.nslots; s++) {
    if (r->bs.slot_brick[s] == BR_INVALID) return s;
    if (r->bs.slot_use[s] != r->bs.frame && r->bs.slot_use[s] < bu) {
      bu = r->bs.slot_use[s];
      best = s;
    }
  }
  return best;
}

/* after a decode batch: per-slot mip blits + incremental world-indexed
 * occupancy (region max-reduce, then re-dilate each brick plus a 1-block halo
 * so neighbor borders self-heal as fill order interleaves). One submission;
 * atlas and occupancy images live in GENERAL for their whole lifetime. */
static int bricks_post_fill(r3d_renderer *r, const uint32_t *sel_slot, const uint32_t *sel_b,
                            uint32_t n) {
  uint32_t abpa = r->bricks_abpa, bpa = r->bricks_bpa, odim = bpa * 16u;
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->bs.upload_pool);
  if (!cmd) return -1;
  /* A prior render submission can still be sampling a neighbouring resident
   * brick while dilation updates this batch's one-block halo. Since all work
   * uses one queue, these barriers order earlier sampled reads before writes;
   * the barriers at the end order the writes before later render submissions. */
  r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT, 0,
                       r->bricks_amips);
  if (!r->bricks_lod) {
    r3d_vk_image_barrier(cmd, r->bs.occraw.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, 0, 1);
    r3d_vk_image_barrier(cmd, r->brick_occ.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, 0, 1);
  }
  for (uint32_t m = 1; m < r->bricks_amips; m++) {
    for (uint32_t i = 0; i < n; i++) {
      uint32_t s = sel_slot[i];
      int32_t sx = (int32_t)(s % abpa), sy = (int32_t)((s / abpa) % abpa),
              sz = (int32_t)(s / (abpa * abpa));
      int32_t d0 = (int32_t)(BR_SLOT_DIM >> (m - 1)), d1 = d0 / 2;
      VkImageBlit2 blit = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
          .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 0, 1},
          .srcOffsets = {{sx * d0, sy * d0, sz * d0},
                         {(sx + 1) * d0, (sy + 1) * d0, (sz + 1) * d0}},
          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1},
          .dstOffsets = {{sx * d1, sy * d1, sz * d1},
                         {(sx + 1) * d1, (sy + 1) * d1, (sz + 1) * d1}},
      };
      VkBlitImageInfo2 bi2 = {.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                              .srcImage = r->brick_atlas.img,
                              .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                              .dstImage = r->brick_atlas.img,
                              .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                              .regionCount = 1,
                              .pRegions = &blit,
                              .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2(cmd, &bi2);
    }
    r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, m, 1);
  }
  if (!r->bricks_lod) {
    for (uint32_t i = 0; i < n; i++) {
      uint32_t s = sel_slot[i], b = sel_b[i];
      uint32_t pc[6] = {(s % abpa) * BR_SLOT_DIM,          ((s / abpa) % abpa) * BR_SLOT_DIM,
                        (s / (abpa * abpa)) * BR_SLOT_DIM, (b % bpa) * 16u,
                        ((b / bpa) % bpa) * 16u,           (b / (bpa * bpa)) * 16u};
      r3d_vkcomp_dispatch(cmd, &r->bs.omax, pc, sizeof pc, 64, 1, 1);
    }
    r3d_vk_image_barrier(cmd, r->bs.occraw.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
    for (uint32_t i = 0; i < n; i++) {
      uint32_t b = sel_b[i];
      uint32_t bx = (b % bpa) * 16u, by = ((b / bpa) % bpa) * 16u,
               bz = (b / (bpa * bpa)) * 16u;
      uint32_t pc[5] = {bx ? bx - 1 : 0, by ? by - 1 : 0, bz ? bz - 1 : 0, 18u, odim};
      r3d_vkcomp_dispatch(cmd, &r->bs.odil, pc, sizeof pc, (18 * 18 * 18 + 63) / 64, 1, 1);
    }
  }
  r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, r->bricks_amips);
  if (!r->bricks_lod)
    r3d_vk_image_barrier(cmd, r->brick_occ.img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
  return r3d_vk_oneshot_end(&r->vk, r->bs.upload_pool, cmd);
}

/* Upload CPU-decoded bricks in one transfer submission.  Streaming decode on
 * the render queue caused multi-hundred-millisecond stalls on unified GPUs;
 * the worker now performs entropy+IDCT on CPU and leaves the queue only this
 * compact copy plus the per-slot mip blits above. */
static int bricks_upload_raw(r3d_renderer *r, r3d_vkimage *atlas, const uint32_t *sel_slot,
                             uint32_t n) {
  VkBufferImageCopy reg[BR_MAX_BATCH];
  uint32_t abpa = r->bricks_abpa;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t s = sel_slot[i];
    reg[i] = (VkBufferImageCopy){
        .bufferOffset = (VkDeviceSize)i * BR_RAW_BYTES,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {(int32_t)((s % abpa) * BR_SLOT_DIM),
                        (int32_t)(((s / abpa) % abpa) * BR_SLOT_DIM),
                        (int32_t)((s / (abpa * abpa)) * BR_SLOT_DIM)},
        .imageExtent = {BR_SLOT_DIM, BR_SLOT_DIM, BR_SLOT_DIM},
    };
  }
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->bs.upload_pool);
  if (!cmd) return -1;
  r3d_vk_image_barrier(cmd, atlas->img, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
  vkCmdCopyBufferToImage(cmd, r->bs.raw_stage.buf, atlas->img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, n, reg);
  r3d_vk_image_barrier(cmd, atlas->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT, 0, 1);
  for (uint32_t m = 1; m < r->bricks_amips; m++) {
    for (uint32_t i = 0; i < n; i++) {
      uint32_t s = sel_slot[i];
      int32_t sx = (int32_t)(s % abpa), sy = (int32_t)((s / abpa) % abpa),
              sz = (int32_t)(s / (abpa * abpa));
      int32_t d0 = (int32_t)(BR_SLOT_DIM >> (m - 1u)), d1 = d0 / 2;
      VkImageBlit2 blit = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
          .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1u, 0, 1},
          .srcOffsets = {{sx * d0, sy * d0, sz * d0},
                         {(sx + 1) * d0, (sy + 1) * d0, (sz + 1) * d0}},
          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1},
          .dstOffsets = {{sx * d1, sy * d1, sz * d1},
                         {(sx + 1) * d1, (sy + 1) * d1, (sz + 1) * d1}},
      };
      VkBlitImageInfo2 bi = {.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                             .srcImage = atlas->img,
                             .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                             .dstImage = atlas->img,
                             .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                             .regionCount = 1,
                             .pRegions = &blit,
                             .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2(cmd, &bi);
    }
    r3d_vk_image_barrier(cmd, atlas->img, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT, m, 1);
  }
  r3d_vk_image_barrier(cmd, atlas->img, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, r->bricks_amips);
  return r3d_vk_oneshot_end(&r->vk, r->bs.upload_pool, cmd);
}

/* Parallel CPU brick decode: one single-threaded c5d_brick_decode per brick,
 * bricks distributed across all cores (an atomic cursor; the caller
 * participates). Beats the old sequential-bricks x 4-lane-within-brick shape
 * ~3x on 32-brick batches. Blobs are resolved by the caller beforehand
 * (brlod_blob lazily opens shard readers and is not thread-safe); a NULL
 * blob with ni_fallback resolves through the net-ingest brick cache (file
 * IO on the decode threads, never the render thread). */
struct brdec_item {
  const uint8_t *blob;
  size_t bn;
  uint32_t b;
};
struct brdec {
  r3d_renderer *r;
  const struct brdec_item *it;
  uint8_t *raw;    /* n consecutive BR_RAW_BYTES slabs */
  uint8_t *maxes;  /* optional per-brick max out */
  uint8_t *loaded; /* optional per-brick out: 1 = decoded from a real source */
  unsigned par;    /* c5d threads per brick (set by brdec_run) */
  bool zero_on_fail; /* overlay semantics: absent/failed brick = zeros */
  bool ni_fallback;
  int ni_src; /* net-ingest cache to fall back on (0 = CT, 1 = overlay) */
  uint32_t n;
  _Atomic uint32_t next;
  _Atomic int rc;
};

static void *brdec_worker(void *arg) {
  struct brdec *j = arg;
  for (;;) {
    uint32_t i = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed);
    if (i >= j->n) return NULL;
    uint8_t *dst = j->raw + (size_t)i * BR_RAW_BYTES;
    const uint8_t *blob = j->it[i].blob;
    size_t bn = j->it[i].bn;
    uint8_t *owned = NULL;
    if (!blob && j->ni_fallback) {
      owned = ni_load_brick(j->r, j->it[i].b, &bn, j->ni_src);
      blob = owned;
    }
    /* small batches leave most of the pool idle when each brick decodes on
     * one thread (8-brick jobs on 22 threads measured ~75 ms/job): split the
     * brick itself across c5d's own persistent pool */
    int rc = blob ? c5d_brick_decode_par(blob, bn, dst, BR_SLOT_DIM, j->par) : -1;
    free(owned);
    if (j->loaded) j->loaded[i] = rc == 0;
    if (rc != 0) {
      if (j->zero_on_fail) {
        memset(dst, 0, BR_RAW_BYTES);
        continue;
      }
      fprintf(stderr, "bricks: decode failed b=%u n=%zu%s\n", j->it[i].b, bn,
              j->it[i].blob ? "" : " (cache tier)");
      atomic_store(&j->rc, -1);
      continue;
    }
    if (j->maxes) {
      uint8_t mx = 0;
      for (size_t v = 0; v < BR_RAW_BYTES; v++)
        if (dst[v] > mx) mx = dst[v];
      j->maxes[i] = mx;
    }
  }
}

/* Persistent decode pool: brdec_run used to spawn+join up to ncpu pthreads
 * per batch (2-3 batches per job) -- ~1000 thread creations in an 11 s
 * session, each paying stack mmap + first-touch zeroing (clear_page_erms in
 * the profile). Workers now park on a condvar and pick up jobs by
 * generation; the caller participates and waits for stragglers. */
static struct {
  pthread_mutex_t mu;
  pthread_cond_t cv, done_cv;
  pthread_t th[64];
  uint32_t nth;
  bool up, quit;
  struct brdec *job;
  uint64_t gen;      /* bumped per job */
  uint32_t busy;     /* workers still inside the current job */
} g_dpool = {.mu = PTHREAD_MUTEX_INITIALIZER,
             .cv = PTHREAD_COND_INITIALIZER,
             .done_cv = PTHREAD_COND_INITIALIZER};
static pthread_mutex_t g_dpool_run_mu = PTHREAD_MUTEX_INITIALIZER; /* one job at a time */

static void *dpool_thread(void *arg) {
  (void)arg;
  uint64_t seen = 0;
  for (;;) {
    pthread_mutex_lock(&g_dpool.mu);
    while (!g_dpool.quit && g_dpool.gen == seen) pthread_cond_wait(&g_dpool.cv, &g_dpool.mu);
    if (g_dpool.quit) {
      pthread_mutex_unlock(&g_dpool.mu);
      return NULL;
    }
    seen = g_dpool.gen;
    struct brdec *j = g_dpool.job;
    g_dpool.busy++;
    pthread_mutex_unlock(&g_dpool.mu);
    brdec_worker(j);
    pthread_mutex_lock(&g_dpool.mu);
    if (--g_dpool.busy == 0) pthread_cond_broadcast(&g_dpool.done_cv);
    pthread_mutex_unlock(&g_dpool.mu);
  }
}

static void dpool_ensure(void) {
  if (g_dpool.up) return;
  long nc = sysconf(_SC_NPROCESSORS_ONLN);
  if (nc < 2) nc = 2;
  /* leave two hardware threads for the render thread + fetchers */
  uint32_t want = (uint32_t)(nc > 3 ? nc - 2 : 1);
  if (want > 64) want = 64;
  const char *ev = getenv("R3D_DECODE_THREADS");
  if (ev && atoi(ev) > 0) want = atoi(ev) > 64 ? 64u : (uint32_t)atoi(ev);
  for (uint32_t t = 0; t < want; t++)
    if (pthread_create(&g_dpool.th[g_dpool.nth], NULL, dpool_thread, NULL) == 0) g_dpool.nth++;
  g_dpool.up = true;
}

static int brdec_run(struct brdec *j) {
  pthread_mutex_lock(&g_dpool_run_mu);
  atomic_store(&j->next, 0);
  atomic_store(&j->rc, 0);
  pthread_mutex_lock(&g_dpool.mu);
  dpool_ensure();
  { /* threads per brick: fill the pool, cap at 4 (diminishing returns) */
    unsigned per = j->n ? (g_dpool.nth + 1u + j->n - 1u) / j->n : 1u;
    j->par = per < 1u ? 1u : per > 4u ? 4u : per;
  }
  g_dpool.job = j;
  g_dpool.gen++;
  pthread_cond_broadcast(&g_dpool.cv);
  pthread_mutex_unlock(&g_dpool.mu);
  brdec_worker(j); /* the caller works too */
  /* wait until every worker that entered this job has left it; workers that
   * never woke in time see next==n and return immediately */
  pthread_mutex_lock(&g_dpool.mu);
  while (g_dpool.busy) pthread_cond_wait(&g_dpool.done_cv, &g_dpool.mu);
  g_dpool.job = NULL;
  pthread_mutex_unlock(&g_dpool.mu);
  pthread_mutex_unlock(&g_dpool_run_mu);
  return atomic_load(&j->rc);
}

static int bricks_decode_batch(r3d_renderer *r, uint32_t n) {
  if (!r->bs.cpu_decode) {
    if (r3d_vkc5d_decode(r->c5d, r->bs.srcs, n, r->brick_atlas_mip0, r->bs.maxes) != 0)
      return -1;
    return bricks_post_fill(r, r->bs.sel_slot, r->bs.sel_b, n);
  }
  uint8_t *raw = r->bs.raw_host;
  struct brdec_item items[BR_MAX_BATCH];
  for (uint32_t i = 0; i < n; i++)
    items[i] = (struct brdec_item){r->bs.srcs[i].blob, r->bs.srcs[i].n, r->bs.sel_b[i]};
  struct brdec job = {
      .r = r, .it = items, .raw = raw, .maxes = r->bs.maxes, .ni_fallback = true, .n = n};
  if (brdec_run(&job) != 0) return -1;
  memcpy(r->bs.raw_stage.mapped, raw, (size_t)n * BR_RAW_BYTES);
  if (bricks_upload_raw(r, &r->brick_atlas, r->bs.sel_slot, n) != 0) return -1;
  if (r->ink_active) {
    /* same bricks, same slots, the overlay tree's data (absent brick = 0) */
    for (uint32_t i = 0; i < n; i++) {
      size_t bn = 0;
      items[i].blob = brlod_blob(r, r->ink_root, r->ink_readers, r->bs.sel_b[i], &bn);
      items[i].bn = bn;
      items[i].b = r->bs.sel_b[i];
    }
    uint8_t loaded[BR_MAX_BATCH] = {0};
    struct brdec ijob = {.r = r, .it = items, .raw = raw, .loaded = loaded,
                         .zero_on_fail = true, .ni_fallback = r->ni.url2[0] != 0,
                         .ni_src = 1, .n = n};
    brdec_run(&ijob);
    /* Only bricks that neither the local tree nor the net cache could supply
     * are flagged for repair + demand-fetched. Previously every non-local
     * brick was flagged even when the .c5b cache decoded it fine, so the
     * repair pass re-decoded correct bricks on the render thread (~1/3 of
     * all decode CPU in a warm session, and 100+ ms frame hitches). */
    for (uint32_t i = 0; i < n; i++)
      if (!loaded[i] && r->ni.url2[0]) {
        bricks_net_request(r, r->bs.sel_b[i], 1);
        if (r->bs.ink_missing) r->bs.ink_missing[r->bs.sel_slot[i]] = 1;
      }
    memcpy(r->bs.raw_stage.mapped, raw, (size_t)n * BR_RAW_BYTES);
    if (bricks_upload_raw(r, &r->ink_atlas, r->bs.sel_slot, n) != 0) return -1;
  }
  return 0;
}

/* Decode and post-process one streaming batch away from the render thread.
 * The render thread drains and invalidates evicted page entries before the job
 * starts. It also publishes completed entries after another timeline drain;
 * the worker never writes mapped memory while a shader may read it. */
static int bricks_ink_repair_exec(r3d_renderer *r, uint32_t n);

static void *bricks_worker(void *arg) {
  r3d_renderer *r = arg;
  for (;;) {
    pthread_mutex_lock(&r->bs.mu);
    while (!r->bs.quit && r->bs.job_state != 1) pthread_cond_wait(&r->bs.cv, &r->bs.mu);
    if (r->bs.quit) {
      pthread_mutex_unlock(&r->bs.mu);
      return NULL;
    }
    uint32_t n = r->bs.job_n;
    uint32_t kind = r->bs.job_kind;
    uint64_t timeline = r->bs.job_timeline;
    r->bs.job_state = 2;
    pthread_mutex_unlock(&r->bs.mu);

    uint64_t started = now_ns();
    int rc = 0;
    if (kind == 1) { /* overlay repair: upload only, slots stay as they are */
      rc = bricks_ink_repair_exec(r, n);
      pthread_mutex_lock(&r->bs.mu);
      r->bs.jobs++;
      r->bs.stream_ns += now_ns() - started;
      if (rc != 0) r->bs.failures++;
      r->bs.job_rc = rc;
      r->bs.job_state = 3;
      pthread_cond_broadcast(&r->bs.cv);
      pthread_mutex_unlock(&r->bs.mu);
      continue;
    }
    if (timeline) {
      VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                .semaphoreCount = 1,
                                .pSemaphores = &r->timeline,
                                .pValues = &timeline};
      if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) rc = -1;
    }
    if (rc == 0 && bricks_decode_batch(r, n) != 0)
      rc = -1;

    if (rc != 0) {
      fprintf(stderr, "bricks: stream decode failed (batch of %u)\n", n);
      for (uint32_t i = 0; i < n; i++) {
        r->bs.slot_brick[r->bs.sel_slot[i]] = BR_INVALID;
        r->bs.brick_slot[r->bs.sel_b[i]] = BR_INVALID;
      }
    } else {
      for (uint32_t i = 0; i < n; i++) {
        uint32_t b = r->bs.sel_b[i], s = r->bs.sel_slot[i];
        uint8_t m = r->bs.maxes[i];
        r->bs.brick_maxk[b] = m;
        if (m < BR_NOISE_FLOOR) {
          r->bs.slot_brick[s] = BR_INVALID;
          r->bs.brick_slot[b] = BR_INVALID;
          continue;
        }
      }
    }

    pthread_mutex_lock(&r->bs.mu);
    r->bs.jobs++;
    r->bs.stream_ns += now_ns() - started;
    if (rc == 0) r->bs.decoded += n;
    else r->bs.failures++;
    uint32_t hot = 0;
    for (uint32_t s = 0; s < r->bs.nslots; s++)
      if (r->bs.slot_brick[s] != BR_INVALID) hot++;
    r->bs.hot_cached = hot;
    r->bs.job_rc = rc;
    r->bs.job_state = 3;
    pthread_cond_broadcast(&r->bs.cv);
    pthread_mutex_unlock(&r->bs.mu);
  }
}

static int img_general_clear(r3d_renderer *r, r3d_vkimage *img) {
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
  VkClearColorValue z = {{0}};
  VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(cmd, img->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rng);
  r3d_vk_image_barrier(cmd, img->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0, 1);
  return r3d_vk_oneshot_end(&r->vk, r->pool, cmd);
}

/* Decoded-seed cache: the pinned coarsest level decodes to the same ~1 GB of
 * raw 128^3 bricks on every launch (~9 s of entropy work for PHercParis4's
 * 555 dense bricks even across all cores). First launch writes the decoded
 * slabs to <root>/seed.raw; later launches stream that file into the atlas
 * instead (~1 s). Layout: header, a table sized for the whole level, then
 * one BR_RAW_BYTES slab per decoded brick in table order. Guarded by the
 * manifest's size+mtime (trees are write-once; a re-ingest rewrites it). */
#define SEED_CACHE_MAGIC "r3dseed1"
struct seed_hdr {
  char magic[8];
  uint32_t dim, level, count, nres;
  uint64_t man_size, man_mtime;
};
struct seed_ent {
  uint32_t gid;
  uint8_t max;
  uint8_t pad[3];
};

static void seed_manifest_stat(const char *root, uint64_t *size, uint64_t *mtime) {
  char mp[1400];
  snprintf(mp, sizeof mp, "%s/manifest.json", root);
  struct stat st;
  *size = *mtime = 0;
  if (stat(mp, &st) == 0) {
    *size = (uint64_t)st.st_size;
    *mtime = (uint64_t)st.st_mtime;
  }
}

/* open + validate a seed cache; returns entries (malloc'd) or NULL */
static struct seed_ent *seed_cache_open(const char *root, uint32_t level, uint32_t count,
                                        FILE **out_f, uint32_t *out_nres) {
  char path[1400];
  snprintf(path, sizeof path, "%s/seed.raw", root);
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  struct seed_hdr h;
  uint64_t msz, mmt;
  seed_manifest_stat(root, &msz, &mmt);
  struct seed_ent *ents = NULL;
  if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, SEED_CACHE_MAGIC, 8) != 0 ||
      h.dim != BR_SLOT_DIM || h.level != level || h.count != count || h.nres > count ||
      h.man_size != msz || h.man_mtime != mmt)
    goto fail;
  ents = malloc((size_t)h.nres * sizeof *ents);
  if (!ents || fread(ents, sizeof *ents, h.nres, f) != h.nres) goto fail;
  if (fseek(f, (long)(sizeof h + (size_t)count * sizeof *ents), SEEK_SET) != 0) goto fail;
  *out_f = f;
  *out_nres = h.nres;
  return ents;
fail:
  free(ents);
  fclose(f);
  return NULL;
}

int r3d_bricks_begin(r3d_renderer *r, const char *c5s_path, uint32_t pool_bpa,
                     uint32_t warm_mb) {
  VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                              VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                              VK_FORMAT_FEATURE_BLIT_DST_BIT;
  if ((r->vk.caps.r8_optimal & need) != need) {
    fprintf(stderr, "bricks: R8_UNORM storage and blit support are required on this GPU\n");
    return -1;
  }
  uint32_t bpa = 0, nb = 0;
  if (c5d_shard_open(c5s_path, &r->bs.sr) == 0) {
    r->bs.sr_open = true;
    if (r->bs.sr.foot.brick_dim != BR_SLOT_DIM) {
      fprintf(stderr, "bricks: brick_dim %u unsupported\n", r->bs.sr.foot.brick_dim);
      return -1;
    }
    bpa = r->bs.sr.foot.shard_dim / BR_SLOT_DIM;
    nb = r->bs.sr.foot.nbricks;
    r->bs.nb = nb;
    r->bricks_nlev = 1;
    r->bricks_nx = r->bricks_ny = r->bricks_nz = r->bs.sr.foot.shard_dim;
    r->bricks_maxdim = r->bs.sr.foot.shard_dim;
  } else if (bricks_manifest_open(r, c5s_path) == 0) {
    nb = r->bs.nb;
    printf("bricks: LOD manifest %u levels, %ux%ux%u voxels, %u virtual bricks\n",
           r->bricks_nlev, r->bricks_nx, r->bricks_ny, r->bricks_nz, nb);
    /* source.json next to the manifest enables on-demand net ingest: brick
     * misses fetch their raw zarr chunk, transcode, and cache to disk */
    char sp[1360];
    snprintf(sp, sizeof sp, "%s/source.json", r->bricks_root);
    FILE *sf = fopen(sp, "r");
    if (sf) {
      char sj[8192] = {0};
      size_t sn = fread(sj, 1, sizeof sj - 1, sf);
      fclose(sf);
      (void)sn;
      const char *u = strstr(sj, "\"url\": \"");
      const char *q = strstr(sj, "\"quality\": ");
      bool ok = u && q && strstr(sj, "render3d.c5d-source.v1");
      if (ok) {
        u += 8;
        const char *ue = strchr(u, '"');
        ok = ue && (size_t)(ue - u) < sizeof r->ni.url;
        if (ok) {
          memcpy(r->ni.url, u, (size_t)(ue - u));
          r->ni.url[ue - u] = 0;
          r->ni.q0 = strtof(q + 11, NULL);
          const char *lp = sj;
          for (uint32_t l = 0; l < r->bricks_nlev && ok; l++) {
            lp = strstr(lp, "\"chunk\": ");
            ok = lp != NULL;
            if (!ok) break;
            r->ni.chsz[l] = (uint32_t)strtoul(lp + 9, NULL, 10);
            const char *rp = strstr(lp, "\"raw\": ");
            r->ni.raw[l] = rp && strncmp(rp + 7, "true", 4) == 0;
            ok = r->ni.chsz[l] >= 32 && r->ni.chsz[l] <= 1024; /* any cubic:
                   * non-brick-multiples (192) ingest as lcm cells */
            lp += 9;
          }
        }
      }
      if (ok) {
        r->ni.have = calloc(r->bs.nb, sizeof *r->ni.have);
        ok = r->ni.have != NULL;
      }
      if (ok) {
        pthread_mutex_init(&r->ni.mu, NULL);
        pthread_cond_init(&r->ni.cv, NULL);
        /* remote chunk fetch is latency-bound: one fetcher per hardware
         * thread; R3D_FETCHERS=N overrides (1..64) */
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        r->ni.nth = ncpu > 0 ? (ncpu > 16 ? 16u : (uint32_t)ncpu) : 8u;
        const char *fe = getenv("R3D_FETCHERS");
        if (fe && atoi(fe) > 0) r->ni.nth = atoi(fe) > 64 ? 64u : (uint32_t)atoi(fe);
        for (uint32_t t = 0; t < r->ni.nth; t++)
          if (pthread_create(&r->ni.th[t], NULL, ni_worker, r) != 0) {
            r->ni.nth = t;
            break;
          }
        r->ni.active = r->ni.nth > 0;
        if (r->ni.active)
          printf("bricks: net ingest active (%s, %u fetchers, cache %s/bricks)\n",
                 r->ni.url, r->ni.nth, r->bricks_root);
      } else {
        fprintf(stderr, "bricks: malformed %s ignored\n", sp);
      }
    }
  } else {
    fprintf(stderr, "bricks: cannot open c5d shard or LOD manifest %s\n", c5s_path);
    return -1;
  }
  VkCommandPoolCreateInfo upci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                                           VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                  .queueFamilyIndex = r->vk.qfam};
  if (vkCreateCommandPool(r->vk.dev, &upci, NULL, &r->bs.upload_pool) != VK_SUCCESS) return -1;
  /* hot pool: identity (slot == brick, direct sampling) when the whole volume
   * fits; otherwise a smaller LRU atlas fed by the streaming pump */
  uint32_t abpa = pool_bpa ? pool_bpa : (r->bricks_lod ? 8u : (bpa < 8u ? bpa : 8u));
  if (!r->bricks_lod && abpa > bpa) abpa = bpa;
  /* Warm-tier size, decided up front because the atlas ceiling budgets around
   * it: explicit --warm wins, else 1/6 of the memory budget in [256 MB, 3 GiB]
   * (the allocator's u32 offsets cap it at 3 GiB). */
  uint64_t warm_want = warm_mb ? (uint64_t)warm_mb << 20 : 0;
  if (!warm_want) {
    warm_want = r3d_vkctx_budget_available(&r->vk) / 6;
    if (warm_want < (256ull << 20)) warm_want = 256ull << 20;
  }
  if (warm_want > (3ull << 30)) warm_want = 3ull << 30;
  /* Atlas ceiling from the device, not a constant: image dimension limit,
   * single-allocation limit (this Adreno: 2048 / ~4 GiB -> 12^3 = 3.6 GiB),
   * and the memory budget — assuming a second identical atlas may join for
   * an overlay in LOD mode, after the warm tier, a surfvol window, and
   * slack. An 8 GiB card lands around 9^3 instead of failing; bigger cards
   * ride the per-allocation limit. */
  uint32_t max_abpa = r->vk.caps.max_dim_3d / BR_SLOT_DIM;
  while (max_abpa > 4u &&
         (uint64_t)max_abpa * max_abpa * max_abpa * BR_RAW_BYTES > r->vk.caps.max_alloc_bytes)
    max_abpa--;
  uint64_t avail = r3d_vkctx_budget_available(&r->vk);
  uint64_t reserve = warm_want + (2ull << 30); /* surfvol window + slack */
  uint64_t for_atlas =
      (avail > reserve ? avail - reserve : avail / 4) / (r->bricks_lod ? 2u : 1u);
  while (max_abpa > 4u &&
         (uint64_t)max_abpa * max_abpa * max_abpa * BR_RAW_BYTES > for_atlas)
    max_abpa--;
  if (r->bricks_lod && !pool_bpa) {
    /* no explicit --pool: spend the headroom. The atlas is the streaming
     * working set (a bigger pool means fewer evictions/re-decodes when
     * zoomed in), so default to the budget-derived ceiling rather than 8^3:
     * a 16 GB card lands at 12^3 = 3.4 GiB, an 8 GB card around 9^3. */
    abpa = max_abpa;
  }
  if (r->bricks_lod) {
    /* the coarsest level is permanently pinned in the atlas; grow the pool so
     * streaming still has headroom (a 6-level 81 TB volume pins 1216 slots,
     * which starves an 8^3 pool into never decoding anything) */
    const r3d_brlod_level *coarse_lv = &r->bricks_lev[r->bricks_nlev - 1u];
    uint32_t pinned = coarse_lv->bx * coarse_lv->by * coarse_lv->bz + 384u;
    while (abpa < max_abpa && (uint64_t)abpa * abpa * abpa < pinned) abpa++;
    if ((uint64_t)abpa * abpa * abpa < pinned)
      fprintf(stderr,
              "bricks: coarsest level (%u bricks) nearly fills the %u^3 slot pool; "
              "streaming will be limited\n",
              pinned - 384u, abpa);
  }
  if (abpa > max_abpa) abpa = max_abpa;
  if (!abpa) return -1;
  bool streaming = r->bricks_lod || abpa < bpa;
  r->bs.cpu_decode = streaming && r->bricks_lod && !getenv("R3D_BRICKS_GPU_DECODE");
  const uint32_t SLOT = BR_SLOT_DIM, APRON = 0;
  uint32_t adim = abpa * SLOT;
  const uint32_t amips = r->bricks_lod ? 1u : BR_AMIPS;
  r->bricks_amips = amips;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){adim, adim, adim}, amips,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         &r->brick_atlas) != 0)
    return -1;
  VkImageViewCreateInfo m0v = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = r->brick_atlas.img,
      .viewType = VK_IMAGE_VIEW_TYPE_3D,
      .format = VK_FORMAT_R8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  if (vkCreateImageView(r->vk.dev, &m0v, NULL, &r->brick_atlas_mip0) != VK_SUCCESS) return -1;
  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  r3d_vk_image_barrier(cmd, r->brick_atlas.img, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0, amips);
  if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) return -1;

  if (!r->bs.cpu_decode && r3d_vkc5d_create(&r->c5d, &r->vk, r->cfg.spv_dir, 8) != 0)
    return -1;

  /* world-indexed occupancy images (cleared: absent bricks read as empty) +
   * the persistent region-form occupancy kernels */
  uint32_t odim = r->bricks_lod ? 1u : bpa * 16u;
  VkImageUsageFlags ou =
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){odim, odim, odim}, 1, ou,
                         &r->bs.occraw) != 0 ||
      r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, (VkExtent3D){odim, odim, odim}, 1, ou,
                         &r->brick_occ) != 0)
    return -1;
  if (img_general_clear(r, &r->bs.occraw) != 0 || img_general_clear(r, &r->brick_occ) != 0)
    return -1;
  if (!r->bricks_lod) {
    VkDescriptorType tt[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    char sp[1024];
    snprintf(sp, sizeof sp, "%s/occmax.spv", r->cfg.spv_dir);
    if (r3d_vkcomp_create(&r->vk, sp, tt, 2, 24, &r->bs.omax) != 0) return -1;
    snprintf(sp, sizeof sp, "%s/occdilate.spv", r->cfg.spv_dir);
    if (r3d_vkcomp_create(&r->vk, sp, tt, 2, 20, &r->bs.odil) != 0) return -1;
    r->bs.comp_ready = true;
    r3d_vkcomp_bind_image(&r->vk, &r->bs.omax, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->brick_atlas.view, r->samp_near, VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&r->vk, &r->bs.omax, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                          r->bs.occraw.view, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&r->vk, &r->bs.odil, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->bs.occraw.view, r->samp_near, VK_IMAGE_LAYOUT_GENERAL);
    r3d_vkcomp_bind_image(&r->vk, &r->bs.odil, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                          r->brick_occ.view, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
  }

  /* page table + CPU residency state */
  r3d_vkbuf_destroy(&r->vk, &r->page_buf);
  uint32_t page_words = nb + (r->bricks_lod ? BR_PAGE_HEADER : 0u);
  if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)page_words * 4,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            &r->page_buf) != 0)
    return -1;
  uint32_t *page = r->page_buf.mapped;
  for (uint32_t b = 0; b < page_words; b++) page[b] = BR_INVALID;
  if (r->bricks_lod) {
    memset(page, 0, BR_PAGE_HEADER * sizeof *page);
    page[0] = r->bricks_nlev;
    page[1] = r->bricks_nx;
    page[2] = r->bricks_ny;
    page[3] = r->bricks_nz;
    for (uint32_t l = 0; l < r->bricks_nlev; l++) {
      const r3d_brlod_level *bl = &r->bricks_lev[l];
      page[4u + l * 4u] = BR_PAGE_HEADER + bl->page_off;
      page[5u + l * 4u] = bl->bx | (bl->by << 10u) | (bl->bz << 20u);
      page[6u + l * 4u] = bl->scale;
    }
  }
  uint32_t nslots = abpa * abpa * abpa;
  r->bs.nb = nb;
  r->bs.nslots = nslots;
  r->bs.slot_brick = malloc((size_t)nslots * 4);
  r->bs.slot_use = calloc(nslots, 4);
  r->bs.brick_slot = malloc((size_t)nb * 4);
  r->bs.brick_want = calloc(nb, 4);
  r->bs.brick_maxk = malloc((size_t)nb * 2);
  r->bs.cands = malloc((size_t)nb * sizeof(struct bcand));
  uint32_t scap = streaming ? BR_MAX_BATCH : nb;
  r->bs.srcs = malloc((size_t)scap * sizeof(r3d_c5d_src));
  r->bs.sel_b = malloc((size_t)scap * 4);
  r->bs.sel_slot = malloc((size_t)scap * 4);
  r->bs.maxes = malloc(scap);
  r->bs.warm_off = malloc((size_t)nb * 4);
  r->bs.warm_len = calloc(nb, 4);
  r->bs.warm_use = calloc(nb, 4);
  if (!r->bs.slot_brick || !r->bs.slot_use || !r->bs.brick_slot || !r->bs.brick_want ||
      !r->bs.brick_maxk ||
      !r->bs.cands || !r->bs.srcs || !r->bs.sel_b || !r->bs.sel_slot || !r->bs.maxes ||
      !r->bs.warm_off || !r->bs.warm_len || !r->bs.warm_use)
    return -1;
  memset(r->bs.slot_brick, 0xFF, (size_t)nslots * 4);
  memset(r->bs.brick_slot, 0xFF, (size_t)nb * 4);
  memset(r->bs.brick_maxk, 0xFF, (size_t)nb * 2); /* -1 = unknown */
  memset(r->bs.warm_off, 0xFF, (size_t)nb * 4);
  r->bricks_bpa = bpa;
  r->bricks_abpa = abpa;

  if (!streaming) {
    /* identity residency: decode the whole shard up front, slot == brick */
    uint32_t np = 0;
    for (uint32_t b = 0; b < nb; b++) {
      size_t n = 0;
      const uint8_t *blob = c5d_shard_brick(&r->bs.sr, b, &n);
      if (!blob) { /* missing brick: page stays invalid, never requested */
        r->bs.brick_maxk[b] = 0;
        continue;
      }
      uint32_t bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
      r->bs.srcs[np] = (r3d_c5d_src){.blob = blob, .n = n,
                                     .sx = bx * SLOT + APRON, .sy = by * SLOT + APRON,
                                     .sz = bz * SLOT + APRON};
      r->bs.sel_b[np] = b;
      r->bs.sel_slot[np] = b;
      np++;
    }
    printf("bricks: decoding %u/%u bricks on GPU (%s)...\n", np, nb,
           getenv("R3D_C5D_HYBRID") ? "hybrid" : "full-GPU");
    uint64_t t0 = now_ns();
    int rc = r3d_vkc5d_decode(r->c5d, r->bs.srcs, np, r->brick_atlas_mip0, r->bs.maxes);
    double ms = (double)(now_ns() - t0) / 1e6;
    if (rc != 0) {
      fprintf(stderr, "bricks: GPU decode failed\n");
      return -1;
    }
    printf("bricks: decoded %u bricks in %.0f ms (%.0f bricks/s, %.2f GB/s raw)\n", np, ms,
           (double)np * 1000.0 / ms, (double)np * 2.097152 / ms);
    if (bricks_post_fill(r, r->bs.sel_slot, r->bs.sel_b, np) != 0) return -1;
    for (uint32_t k = 0; k < np; k++) {
      uint32_t b = r->bs.sel_b[k];
      page[bricks_page_index(r, b)] = b | ((uint32_t)r->bs.maxes[k] << 24);
      r->bs.slot_brick[b] = b;
      r->bs.brick_slot[b] = b;
      r->bs.brick_maxk[b] = r->bs.maxes[k];
    }
    r->bricks_identity = np == nb;
  } else {
    r->bs.warm_cap = warm_want;
    if (r3d_vkbuf_create_host(&r->vk, r->bs.warm_cap,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &r->bs.warm) != 0)
      return -1;
    if (r->bs.cpu_decode) {
      if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)BR_MAX_BATCH * BR_RAW_BYTES,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &r->bs.raw_stage) != 0)
        return -1;
      r->bs.raw_host = malloc((size_t)BR_MAX_BATCH * BR_RAW_BYTES);
      if (!r->bs.raw_host) return -1;
    }
    warm_release(r, 0, (uint32_t)r->bs.warm_cap); /* one node spanning the tier */
    /* A complete coarsest level is tiny (PHerc1218: three bricks).  Seed it
     * synchronously so every fine request has a resident fallback from the
     * first rendered frame; later levels replace it sample-by-sample. */
    if (r->bricks_lod) {
      const r3d_brlod_level *cl = &r->bricks_lev[r->bricks_nlev - 1u];
      uint32_t first = cl->page_off, count = cl->bx * cl->by * cl->bz;
      uint32_t level = r->bricks_nlev - 1u;
      uint32_t cursor = 0, next_slot = 0;
      r->bs.frame = 1;
      uint64_t st0 = now_ns();
      if (r->bs.cpu_decode) { /* fast path: stream previously decoded slabs */
        FILE *scf = NULL;
        uint32_t snres = 0;
        struct seed_ent *ents = seed_cache_open(r->bricks_root, level, count, &scf, &snres);
        if (ents) {
          uint8_t *raw = r->bs.raw_stage.mapped;
          uint32_t np = 0;
          bool ok = snres <= nslots;
          for (uint32_t i = 0; i < count; i++) r->bs.brick_maxk[first + i] = 0;
          for (uint32_t e = 0; ok && e < snres; e++) {
            uint32_t b = ents[e].gid;
            uint8_t m = ents[e].max;
            if (b < first || b >= first + count ||
                fread(raw + (size_t)np * BR_RAW_BYTES, 1, BR_RAW_BYTES, scf) !=
                    BR_RAW_BYTES) {
              ok = false;
              break;
            }
            uint32_t s = next_slot++;
            r->bs.sel_slot[np++] = s;
            r->bs.brick_maxk[b] = m;
            if (m >= BR_NOISE_FLOOR) {
              page[bricks_page_index(r, b)] = s | ((uint32_t)m << 24u);
              r->bs.slot_brick[s] = b;
              r->bs.slot_use[s] = r->bs.frame;
              r->bs.brick_slot[b] = s;
            }
            if (np == BR_MAX_BATCH || e + 1u == snres) {
              SDL_PumpEvents();
              if (bricks_upload_raw(r, &r->brick_atlas, r->bs.sel_slot, np) != 0) ok = false;
              np = 0;
            }
          }
          fclose(scf);
          free(ents);
          if (ok) {
            cursor = count;
            printf("bricks: seeded L%u fallback from seed.raw (%u bricks, %.0f ms)\n", level,
                   snres, (double)(now_ns() - st0) / 1e6);
          } else { /* unusable cache: reset the level's state, decode below */
            fprintf(stderr, "bricks: seed.raw unusable, re-decoding\n");
            for (uint32_t i = 0; i < count; i++) {
              uint32_t b = first + i;
              uint32_t s = r->bs.brick_slot[b];
              if (s != BR_INVALID) r->bs.slot_brick[s] = BR_INVALID;
              r->bs.brick_slot[b] = BR_INVALID;
              r->bs.brick_maxk[b] = -1;
              page[bricks_page_index(r, b)] = BR_INVALID;
            }
            next_slot = 0;
          }
        }
      }
      /* decode path; writes seed.raw as a side effect so the entropy work
       * only ever happens once per tree */
      FILE *wf = NULL;
      struct seed_ent *wents = NULL;
      uint32_t wn = 0;
      char wtmp[1408] = "", wfin[1400] = "";
      if (cursor < count && r->bs.cpu_decode) {
        snprintf(wfin, sizeof wfin, "%s/seed.raw", r->bricks_root);
        snprintf(wtmp, sizeof wtmp, "%s.tmp", wfin);
        wf = fopen(wtmp, "wb");
        wents = wf ? malloc((size_t)count * sizeof *wents) : NULL;
        long roff = (long)(sizeof(struct seed_hdr) + (size_t)count * sizeof(struct seed_ent));
        if (wf && (!wents || fseek(wf, roff, SEEK_SET) != 0)) {
          fclose(wf);
          wf = NULL;
          unlink(wtmp);
        }
      }
      while (cursor < count) {
        uint32_t np = 0;
        while (cursor < count && np < BR_MAX_BATCH) {
          uint32_t b = first + cursor++;
          size_t n = 0;
          const uint8_t *blob = warm_get(r, b, &n);
          if (!blob) {
            r->bs.brick_maxk[b] = 0;
            continue;
          }
          if (next_slot >= nslots) break;
          uint32_t s = next_slot++;
          r->bs.srcs[np] = (r3d_c5d_src){.blob = blob,
                                         .n = n,
                                         .sx = (s % abpa) * BR_SLOT_DIM,
                                         .sy = ((s / abpa) % abpa) * BR_SLOT_DIM,
                                         .sz = (s / (abpa * abpa)) * BR_SLOT_DIM};
          r->bs.sel_b[np] = b;
          r->bs.sel_slot[np] = s;
          np++;
        }
        if (!np) continue;
        SDL_PumpEvents(); /* multi-second synchronous phase: stay responsive */
        if (bricks_decode_batch(r, np) != 0)
          return -1;
        if (wf) {
          if (fwrite(r->bs.raw_host, BR_RAW_BYTES, np, wf) != np) {
            fclose(wf);
            wf = NULL;
            unlink(wtmp);
          } else {
            for (uint32_t i = 0; i < np; i++)
              wents[wn++] = (struct seed_ent){.gid = r->bs.sel_b[i], .max = r->bs.maxes[i]};
          }
        }
        for (uint32_t i = 0; i < np; i++) {
          uint32_t b = r->bs.sel_b[i], s = r->bs.sel_slot[i];
          uint8_t m = r->bs.maxes[i];
          r->bs.brick_maxk[b] = m;
          if (m < BR_NOISE_FLOOR) continue;
          page[bricks_page_index(r, b)] = s | ((uint32_t)m << 24u);
          r->bs.slot_brick[s] = b;
          r->bs.slot_use[s] = r->bs.frame;
          r->bs.brick_slot[b] = s;
        }
      }
      if (wents) { /* the decode path ran: finish (or discard) the cache */
        if (wf) {
          struct seed_hdr h = {.dim = BR_SLOT_DIM, .level = level, .count = count, .nres = wn};
          memcpy(h.magic, SEED_CACHE_MAGIC, 8);
          seed_manifest_stat(r->bricks_root, &h.man_size, &h.man_mtime);
          bool wok = fseek(wf, 0, SEEK_SET) == 0 && fwrite(&h, sizeof h, 1, wf) == 1 &&
                     fwrite(wents, sizeof *wents, wn, wf) == wn && fclose(wf) == 0;
          if (!wok || rename(wtmp, wfin) != 0) unlink(wtmp);
        }
        free(wents);
        printf("bricks: seeded L%u fallback (%u bricks, %.0f ms%s)\n", level, count,
               (double)(now_ns() - st0) / 1e6, wf ? "; seed.raw cached" : "");
      } else if (!r->bs.cpu_decode) {
        printf("bricks: seeded L%u fallback (%u bricks)\n", level, count);
      }
    }
    if (pthread_mutex_init(&r->bs.mu, NULL) != 0) return -1;
    if (pthread_cond_init(&r->bs.cv, NULL) != 0) {
      pthread_mutex_destroy(&r->bs.mu);
      return -1;
    }
    r->bs.worker_up = true;
    if (pthread_create(&r->bs.worker, NULL, bricks_worker, r) != 0) {
      r->bs.worker_up = false;
      pthread_cond_destroy(&r->bs.cv);
      pthread_mutex_destroy(&r->bs.mu);
      return -1;
    }
    r->bs.active = true;
    r->bricks_identity = false;
    if (r->bricks_lod)
      printf("bricks: streaming %u LOD bricks through a %u^3-slot hot atlas "
             "(%llu MB warm tier, %s decode)\n",
             nb, abpa, (unsigned long long)(r->bs.warm_cap >> 20),
             r->bs.cpu_decode ? "CPU-async" : "GPU");
    else
      printf("bricks: streaming %u^3 bricks through a %u^3-slot hot atlas "
             "(%llu MB warm tier)\n",
             bpa, abpa, (unsigned long long)(r->bs.warm_cap >> 20));
  }

  if (amips > 1u && getenv("R3D_DUMP_MIP1")) { /* debug: write one z-slice of mip1 as PGM */
    uint32_t d1 = adim / 2;
    r3d_vkbuf rb;
    if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)d1 * d1, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &rb) == 0) {
      VkCommandBuffer dc = r3d_vk_oneshot_begin(&r->vk, r->pool);
      r3d_vk_image_barrier(dc, r->brick_atlas.img, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, 1, 1);
      VkBufferImageCopy reg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1},
                               .imageOffset = {0, 0, (int32_t)(d1 / 2)},
                               .imageExtent = {d1, d1, 1}};
      vkCmdCopyImageToBuffer(dc, r->brick_atlas.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb.buf, 1, &reg);
      r3d_vk_image_barrier(dc, r->brick_atlas.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, 1, 1);
      r3d_vk_oneshot_end(&r->vk, r->pool, dc);
      FILE *f = fopen("mip1_slice.pgm", "wb");
      if (f) {
        fprintf(f, "P5\n%u %u\n255\n", d1, d1);
        fwrite(rb.mapped, 1, (size_t)d1 * d1, f);
        fclose(f);
        printf("bricks: dumped mip1_slice.pgm (%ux%u)\n", d1, d1);
      }
      r3d_vkbuf_destroy(&r->vk, &rb);
    }
  }

  r3d_vkctx_device_wait_idle(&r->vk);
  /* binding 3 (cube-mode occupancy slot) now serves bricks mode too */
  write_image_dset(r, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->brick_occ.view,
                   r->samp_near, VK_IMAGE_LAYOUT_GENERAL);
  write_image_dset(r, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->brick_atlas.view,
                   r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  VkDescriptorBufferInfo pbi = {.buffer = r->page_buf.buf, .range = VK_WHOLE_SIZE};
  VkWriteDescriptorSet pw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = r->dset,
                             .dstBinding = 5,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &pbi};
  vkUpdateDescriptorSets(r->vk.dev, 1, &pw, 0, NULL);
  return 0;
}

void r3d_bricks_params(const r3d_renderer *r, r3d_frame_params *p) {
  if (r->bricks_lod)
    p->brick_mode = r->bricks_abpa | (r->bricks_nlev << 8u) | 0x20000u;
  else
    p->brick_mode = r->bricks_bpa | (r->bricks_abpa << 8u) |
                    (r->bricks_identity ? 0x10000u : 0u);
}

/* Arm the overlay tree as the second net-ingest source when it carries a
 * source.json (same parse as the CT tree's). Requires the fetcher pool to
 * already be up (CT source present). */
static void ni_overlay_source(r3d_renderer *r) {
  r->ni.url2[0] = 0;
  if (!r->ni.active) return;
  char sp[1360];
  snprintf(sp, sizeof sp, "%s/source.json", r->ink_root);
  FILE *sf = fopen(sp, "r");
  if (!sf) return;
  char sj[8192] = {0};
  size_t sn = fread(sj, 1, sizeof sj - 1, sf);
  fclose(sf);
  (void)sn;
  const char *u = strstr(sj, "\"url\": \"");
  const char *q = strstr(sj, "\"quality\": ");
  bool ok = u && q && strstr(sj, "render3d.c5d-source.v1");
  if (!ok) return;
  u += 8;
  const char *ue = strchr(u, '"');
  if (!ue || (size_t)(ue - u) >= sizeof r->ni.url2) return;
  memcpy(r->ni.url2, u, (size_t)(ue - u));
  r->ni.url2[ue - u] = 0;
  r->ni.q02 = strtof(q + 11, NULL);
  const char *lp = sj;
  for (uint32_t l = 0; l < r->bricks_nlev; l++) {
    lp = strstr(lp, "\"chunk\": ");
    if (!lp) {
      r->ni.url2[0] = 0;
      return;
    }
    r->ni.chsz2[l] = (uint32_t)strtoul(lp + 9, NULL, 10);
    const char *rp = strstr(lp, "\"raw\": ");
    r->ni.raw2[l] = rp && strncmp(rp + 7, "true", 4) == 0;
    if (r->ni.chsz2[l] < 32 || r->ni.chsz2[l] > 1024) {
      r->ni.url2[0] = 0;
      return;
    }
    lp += 9;
  }
  snprintf(r->ni.root2, sizeof r->ni.root2, "%s", r->ink_root);
  if (!r->ni.have2) r->ni.have2 = calloc(r->bs.nb, 1);
  else memset((void *)r->ni.have2, 0, r->bs.nb);
  if (!r->bs.ink_missing) r->bs.ink_missing = calloc(r->bs.nslots, 1);
  else memset(r->bs.ink_missing, 0, r->bs.nslots);
  r->bs.ink_fetch_seen = 0;
  if (!r->ni.have2) r->ni.url2[0] = 0;
  printf("bricks: overlay net ingest active (%s)\n", r->ni.url2);
}

int r3d_bricks_overlay_switch(r3d_renderer *r, const char *lod_root) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!r->ink_active) return r3d_bricks_overlay(r, lod_root);
  if (strcmp(r->ink_root, lod_root) == 0) return 0;
  /* drain the async decode job — its worker reads the overlay readers */
  pthread_mutex_lock(&r->bs.mu);
  while (r->bs.job_state == 1 || r->bs.job_state == 2)
    pthread_cond_wait(&r->bs.cv, &r->bs.mu);
  pthread_mutex_unlock(&r->bs.mu);
  if (r->ni.active) { /* retire the old tree's fetches: purge queued overlay
       * chunks and let in-flight ones finish (they write the old cache) */
    r->ni.url2[0] = 0;
    for (;;) {
      pthread_mutex_lock(&r->ni.mu);
      uint32_t w = 0;
      for (uint32_t i = 0; i < r->ni.qn; i++)
        if (!(r->ni.queue[i] >> 63)) r->ni.queue[w++] = r->ni.queue[i];
      if (r->ni.qins > w) r->ni.qins = w;
      r->ni.qn = w;
      bool busy = false;
      for (uint32_t i = 0; i < r->ni.nin; i++) busy = busy || (r->ni.inflight[i] >> 63);
      pthread_mutex_unlock(&r->ni.mu);
      if (!busy) break;
      struct timespec ts = {0, 20000000};
      nanosleep(&ts, NULL);
    }
  }
  for (uint32_t i = 0; i < r->bricks_nreaders; i++)
    if (r->ink_readers[i].open) c5d_shard_close_reader(&r->ink_readers[i].sr);
  free(r->ink_readers);
  r->ink_readers = NULL;
  pres_drain(r); /* queue ops need external sync */
  vkDeviceWaitIdle(r->vk.dev); /* the atlas is bound to in-flight frames */
  r3d_vkimage_destroy(&r->vk, &r->ink_atlas);
  r->ink_active = false;
  r->ink_root[0] = 0;
  int rc = r3d_bricks_overlay(r, lod_root);
  if (rc == 0 && r->sv.active) { /* the surfvol taps the overlay atlas too:
                                  * rebind and force a full window re-bake */
    r3d_vkcomp_bind_image(&r->vk, &r->sv.comp, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          r->ink_atlas.view, r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
    r->sv.step = 0.0f;
    r->sv.dirty = false;
    r->sv.prog_row = UINT32_MAX;
    r->sv.baked = false;
    r->sv.shift_pending = false;
  }
  return rc;
}

int r3d_bricks_overlay(r3d_renderer *r, const char *lod_root) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!r->bricks_lod || !r->bs.cpu_decode || r->ink_active) {
    fprintf(stderr, "bricks: overlay needs an active CPU-decode LOD manifest\n");
    return -1;
  }
  size_t rn = strlen(lod_root);
  if (rn >= sizeof r->ink_root) return -1;
  memcpy(r->ink_root, lod_root, rn + 1);
  char mp[1280];
  snprintf(mp, sizeof mp, "%s/manifest.json", lod_root);
  FILE *mf = fopen(mp, "r");
  if (!mf) {
    fprintf(stderr, "bricks: overlay manifest %s missing\n", mp);
    return -1;
  }
  char head[512] = {0};
  size_t hn = fread(head, 1, sizeof head - 1, mf);
  fclose(mf);
  (void)hn;
  char want[128];
  snprintf(want, sizeof want, "\"shape\": [%u, %u, %u]", r->bricks_nz, r->bricks_ny,
           r->bricks_nx);
  if (!strstr(head, want)) {
    fprintf(stderr, "bricks: overlay shape mismatch (need %s)\n", want);
    return -1;
  }
  r->ink_readers = calloc(r->bricks_nreaders, sizeof *r->ink_readers);
  if (!r->ink_readers) return -1;
  if (r3d_vkimage_create(&r->vk, VK_FORMAT_R8_UNORM, r->brick_atlas.extent, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         &r->ink_atlas) != 0)
    return -1;
  if (img_general_clear(r, &r->ink_atlas) != 0) return -1;
  write_image_dset(r, 10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, r->ink_atlas.view,
                   r->samp_vol, VK_IMAGE_LAYOUT_GENERAL);
  r->ink_active = true;
  /* backfill the already-resident bricks (incl. the pinned coarsest level);
   * like the CT seed, the decoded slabs are cached in <ink_root>/seed.raw so
   * the overlay's entropy work also only happens once per tree */
  const r3d_brlod_level *cl = &r->bricks_lev[r->bricks_nlev - 1u];
  uint32_t level = r->bricks_nlev - 1u, lcount = cl->bx * cl->by * cl->bz;
  uint8_t *raw = r->bs.raw_stage.mapped;
  uint32_t sel[BR_MAX_BATCH], nb_ = 0, filled = 0;
  uint64_t t0 = now_ns();
  {
    FILE *scf = NULL;
    uint32_t snres = 0;
    struct seed_ent *ents = seed_cache_open(r->ink_root, level, lcount, &scf, &snres);
    if (ents) {
      long roff = (long)(sizeof(struct seed_hdr) + (size_t)lcount * sizeof(struct seed_ent));
      bool ok = true;
      for (uint32_t s = 0; ok && s < r->bs.nslots; s++) {
        uint32_t b = r->bs.slot_brick[s];
        if (b == BR_INVALID) continue;
        uint32_t e = 0;
        while (e < snres && ents[e].gid != b) e++;
        if (e == snres) { /* resident set changed: the cache can't serve it */
          ok = false;
          break;
        }
        if (fseek(scf, roff + (long)((size_t)e * BR_RAW_BYTES), SEEK_SET) != 0 ||
            fread(raw + (size_t)nb_ * BR_RAW_BYTES, 1, BR_RAW_BYTES, scf) != BR_RAW_BYTES) {
          ok = false;
          break;
        }
        sel[nb_++] = s;
        filled++;
        if (nb_ == BR_MAX_BATCH) {
          SDL_PumpEvents();
          if (bricks_upload_raw(r, &r->ink_atlas, sel, nb_) != 0) ok = false;
          nb_ = 0;
        }
      }
      if (ok && nb_ && bricks_upload_raw(r, &r->ink_atlas, sel, nb_) != 0) ok = false;
      fclose(scf);
      free(ents);
      if (ok) {
        ni_overlay_source(r);
        printf("bricks: overlay %s active (%u bricks from seed.raw, %.0f ms)\n", lod_root,
               filled, (double)(now_ns() - t0) / 1e6);
        return 0;
      }
      /* unusable cache: the atlas may hold partial rows — the decode path
       * below rewrites every resident slot (absent bricks become zeros) */
      fprintf(stderr, "bricks: ink seed.raw unusable, re-decoding\n");
      nb_ = 0;
      filled = 0;
    }
  }
  FILE *wf = NULL;
  struct seed_ent *wents = malloc((size_t)lcount * sizeof *wents);
  uint32_t wn = 0;
  char wtmp[1408] = "", wfin[1400] = "";
  snprintf(wfin, sizeof wfin, "%s/seed.raw", r->ink_root);
  snprintf(wtmp, sizeof wtmp, "%s.tmp", wfin);
  if (wents) {
    wf = fopen(wtmp, "wb");
    long roff = (long)(sizeof(struct seed_hdr) + (size_t)lcount * sizeof(struct seed_ent));
    if (wf && fseek(wf, roff, SEEK_SET) != 0) {
      fclose(wf);
      wf = NULL;
      unlink(wtmp);
    }
  }
  struct brdec_item items[BR_MAX_BATCH];
  uint32_t selb[BR_MAX_BATCH];
  for (uint32_t s = 0; s < r->bs.nslots; s++) {
    uint32_t b = r->bs.slot_brick[s];
    if (b == BR_INVALID) continue;
    size_t bn = 0;
    items[nb_].blob = brlod_blob(r, r->ink_root, r->ink_readers, b, &bn);
    items[nb_].bn = bn;
    items[nb_].b = b;
    selb[nb_] = b;
    sel[nb_++] = s;
    filled++;
    if (nb_ == BR_MAX_BATCH) {
      SDL_PumpEvents();
      struct brdec job = {.r = r, .it = items, .raw = r->bs.raw_host, .zero_on_fail = true,
                          .n = nb_};
      brdec_run(&job);
      if (wf && wn + nb_ <= lcount) {
        if (fwrite(r->bs.raw_host, BR_RAW_BYTES, nb_, wf) != nb_) {
          fclose(wf);
          wf = NULL;
          unlink(wtmp);
        } else {
          for (uint32_t i = 0; i < nb_; i++) wents[wn++] = (struct seed_ent){.gid = selb[i]};
        }
      }
      memcpy(raw, r->bs.raw_host, (size_t)nb_ * BR_RAW_BYTES);
      if (bricks_upload_raw(r, &r->ink_atlas, sel, nb_) != 0) return -1;
      nb_ = 0;
    }
  }
  if (nb_) {
    struct brdec job = {.r = r, .it = items, .raw = r->bs.raw_host, .zero_on_fail = true,
                        .n = nb_};
    brdec_run(&job);
    if (wf && wn + nb_ <= lcount) {
      if (fwrite(r->bs.raw_host, BR_RAW_BYTES, nb_, wf) != nb_) {
        fclose(wf);
        wf = NULL;
        unlink(wtmp);
      } else {
        for (uint32_t i = 0; i < nb_; i++) wents[wn++] = (struct seed_ent){.gid = selb[i]};
      }
    }
    memcpy(raw, r->bs.raw_host, (size_t)nb_ * BR_RAW_BYTES);
    if (bricks_upload_raw(r, &r->ink_atlas, sel, nb_) != 0) return -1;
  }
  if (wf) {
    struct seed_hdr h = {.dim = BR_SLOT_DIM, .level = level, .count = lcount, .nres = wn};
    memcpy(h.magic, SEED_CACHE_MAGIC, 8);
    seed_manifest_stat(r->ink_root, &h.man_size, &h.man_mtime);
    bool wok = fseek(wf, 0, SEEK_SET) == 0 && fwrite(&h, sizeof h, 1, wf) == 1 &&
               fwrite(wents, sizeof *wents, wn, wf) == wn && fclose(wf) == 0;
    if (!wok || rename(wtmp, wfin) != 0) unlink(wtmp);
  }
  free(wents);
  ni_overlay_source(r);
  printf("bricks: overlay %s active (%u resident bricks backfilled, %.0f ms)\n", lod_root,
         filled, (double)(now_ns() - t0) / 1e6);
  return 0;
}

void r3d_bricks_extent(const r3d_renderer *r, float extent[3]) {
  float d = r->bricks_maxdim ? (float)r->bricks_maxdim : 1.0f;
  extent[0] = (float)r->bricks_nx / d;
  extent[1] = (float)r->bricks_ny / d;
  extent[2] = (float)r->bricks_nz / d;
}

void r3d_bricks_shape(const r3d_renderer *r, uint32_t shape[3]) {
  shape[0] = r->bricks_nx;
  shape[1] = r->bricks_ny;
  shape[2] = r->bricks_nz;
}

/* Fill resident slots whose overlay decode zero-filled while the chunk was
 * still upstream; runs on the render thread with the decode job idle, a
 * batch per fetch event. */
/* Render-thread half of the overlay repair: scan the waiting slots whose
 * chunk has landed and post them as a worker job (kind 1). Decoding used to
 * run right here — spawning 24 threads from the render thread and then
 * fence-waiting a 64 MB upload behind the in-flight frame: the 100+ ms
 * "stream" hitches in the profile. Caller holds no lock; the worker must be
 * idle (job_state == 0). Returns true when a job was posted. */
static bool bricks_ink_repair_post(r3d_renderer *r) {
  uint32_t nb_ = 0;
  for (uint32_t s = 0; s < r->bs.nslots && nb_ < BR_MAX_BATCH; s++) {
    if (!r->bs.ink_missing[s]) continue;
    uint32_t b = r->bs.slot_brick[s];
    if (b == BR_INVALID) {
      r->bs.ink_missing[s] = 0;
      continue;
    }
    uint8_t hv = atomic_load(&r->ni.have2[b]);
    if (hv == 0u) continue; /* still fetching */
    r->bs.ink_missing[s] = 0;
    if (hv == 2u) continue; /* definitively absent: zeros are correct */
    r->bs.sel_b[nb_] = b;
    r->bs.sel_slot[nb_] = s;
    nb_++;
  }
  if (!nb_) return false;
  pthread_mutex_lock(&r->bs.mu);
  r->bs.job_kind = 1;
  r->bs.job_n = nb_;
  r->bs.job_nevict = 0;
  r->bs.job_timeline = 0;
  r->bs.job_rc = 0;
  r->bs.job_state = 1;
  pthread_cond_signal(&r->bs.cv);
  pthread_mutex_unlock(&r->bs.mu);
  return true;
}

/* Worker half: decode the posted overlay bricks from the net cache and
 * upload them into their (already CT-resident) slots. No page-table entry
 * changes, so the render thread's publish step only re-bakes/invalidates. */
static int bricks_ink_repair_exec(r3d_renderer *r, uint32_t n) {
  struct brdec_item items[BR_MAX_BATCH];
  for (uint32_t i = 0; i < n; i++)
    items[i] = (struct brdec_item){NULL, 0, r->bs.sel_b[i]};
  struct brdec job = {.r = r, .it = items, .raw = r->bs.raw_host, .zero_on_fail = true,
                      .ni_fallback = true, .ni_src = 1, .n = n};
  brdec_run(&job);
  memcpy(r->bs.raw_stage.mapped, r->bs.raw_host, (size_t)n * BR_RAW_BYTES);
  return bricks_upload_raw(r, &r->ink_atlas, r->bs.sel_slot, n);
}

bool r3d_bricks_stream_begin(r3d_renderer *r) {
  r->bs.last_inflight = 0;
  r->bs.ncand_pending = 0;
  r->bs.stream_open = false;
  if (!r->bs.active) return false;
  if (r->ni.active) { /* this pass's fetches go to the queue head; in-flight
                       * transfers always run to completion */
    pthread_mutex_lock(&r->ni.mu);
    r->ni.qins = 0;
    pthread_mutex_unlock(&r->ni.mu);
  }
  pthread_mutex_lock(&r->bs.mu);
  if (r->bs.job_state == 3) {
    uint64_t timeline = r->timeline_value;
    pthread_mutex_unlock(&r->bs.mu);
    if (timeline) {
      VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                .semaphoreCount = 1,
                                .pSemaphores = &r->timeline,
                                .pValues = &timeline};
      if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return false;
    }
    /* No previously submitted shader can still read these host-coherent page
     * entries. Publish the finished batch before this frame is submitted. */
    pthread_mutex_lock(&r->bs.mu);
    if (r->bs.job_state == 3 && r->bs.job_rc == 0) {
      uint32_t *page = r->page_buf.mapped;
      uint32_t published = 0;
      for (uint32_t i = 0; i < r->bs.job_n; i++) {
        uint32_t b = r->bs.sel_b[i], s = r->bs.sel_slot[i];
        uint8_t m = r->bs.maxes[i];
        if (m >= BR_NOISE_FLOOR) {
          page[bricks_page_index(r, b)] = s | ((uint32_t)m << 24);
          published++;
        }
      }
      /* residency changed under the views: the surfvol re-bakes (only now
       * are the entries visible to its kernel) and every pane re-renders */
      if (published || r->bs.job_kind == 1) {
        r3d_surfvol_mark(r);
        r->scene_gen++;
      }
    }
    r->bs.job_state = 0;
    r->bs.job_kind = 0;
  }
  if (r->bs.job_state != 0) {
    r->bs.last_inflight = r->bs.job_n;
    pthread_mutex_unlock(&r->bs.mu);
    return false;
  }
  pthread_mutex_unlock(&r->bs.mu);
  if (r->ink_active && r->bs.ink_missing && r->ni.url2[0]) {
    uint64_t f2 = atomic_load(&r->ni.fetched2);
    if (f2 != r->bs.ink_fetch_seen) {
      r->bs.ink_fetch_seen = f2;
      /* overlay chunks landed: hand the waiting slots to the worker; this
       * frame's collect is skipped (the worker owns sel_b/sel_slot now) */
      if (bricks_ink_repair_post(r)) {
        r->bs.last_inflight = r->bs.job_n;
        return false;
      }
    }
  }
  r->bs.frame++;
  memset(r->bs.lod_wanted, 0, sizeof r->bs.lod_wanted);
  r->bs.stream_open = true;
  return true;
}

/* Pin the coarsest level's LRU stamps (it is fully resident by construction)
 * — shared by every collect flavor, cheap enough to run once per open. */
static void bricks_touch_coarsest(r3d_renderer *r) {
  const r3d_brlod_level *coarse = &r->bricks_lev[r->bricks_nlev - 1u];
  uint32_t coarse_n = coarse->bx * coarse->by * coarse->bz;
  for (uint32_t i = 0; i < coarse_n; i++) {
    uint32_t b = coarse->page_off + i, slot = r->bs.brick_slot[b];
    if (slot != BR_INVALID) r->bs.slot_use[slot] = r->bs.frame;
    if (r->bs.warm_off[b] != BR_INVALID) r->bs.warm_use[b] = r->bs.frame;
  }
}

/* AABB collect for an ortho/plane view: request the level whose voxel pitch
 * matches pixel_cone across the given volume-space box (auto-coarsened until
 * the walk is tractable), parent-first like the cone pump. */
void r3d_bricks_stream_box(r3d_renderer *r, const float lo[3], const float hi[3],
                           float pixel_cone, float gate) {
  if (!r->bs.stream_open || !r->bricks_lod) return;
  int g8 = (int)(gate * 255.0f + 0.5f);
  if (g8 < BR_NOISE_FLOOR) g8 = BR_NOISE_FLOOR;
  float maxdim = (float)r->bricks_maxdim;
  float vpp = fmaxf(pixel_cone * maxdim, 1.0f); /* base voxels per pixel */
  uint32_t desired = (uint32_t)floorf(log2f(vpp));
  if (desired >= r->bricks_nlev) desired = r->bricks_nlev - 1u;
  bricks_touch_coarsest(r);
  uint32_t rng[3][2];
  const r3d_brlod_level *l;
  float edge;
  for (;;) { /* coarsen until the box holds a sane brick count */
    l = &r->bricks_lev[desired];
    edge = (float)(BR_SLOT_DIM * l->scale) / maxdim;
    const uint32_t bd[3] = {l->bx, l->by, l->bz};
    uint64_t count = 1;
    for (int a = 0; a < 3; a++) {
      int64_t a0 = (int64_t)floorf(lo[a] / edge), a1 = (int64_t)floorf(hi[a] / edge) + 1;
      if (a0 < 0) a0 = 0;
      if (a1 > (int64_t)bd[a]) a1 = (int64_t)bd[a];
      if (a1 < a0) a1 = a0;
      rng[a][0] = (uint32_t)a0;
      rng[a][1] = (uint32_t)a1;
      count *= (uint64_t)(a1 - a0);
    }
    if (count <= 2048u || desired + 1u >= r->bricks_nlev) break;
    desired++;
  }
  float cx0 = (lo[0] + hi[0]) * 0.5f, cy0 = (lo[1] + hi[1]) * 0.5f,
        cz0 = (lo[2] + hi[2]) * 0.5f;
  uint32_t ncand = r->bs.ncand_pending;
  const r3d_brlod_level *pl = desired + 1u < r->bricks_nlev ? &r->bricks_lev[desired + 1u] : NULL;
  for (uint32_t bz = rng[2][0]; bz < rng[2][1]; bz++)
    for (uint32_t by = rng[1][0]; by < rng[1][1]; by++)
      for (uint32_t bx = rng[0][0]; bx < rng[0][1]; bx++) {
        float dx = ((float)bx + 0.5f) * edge - cx0;
        float dy = ((float)by + 0.5f) * edge - cy0;
        float dz = ((float)bz + 0.5f) * edge - cz0;
        float d2 = dx * dx + dy * dy + dz * dz;
        r->bs.lod_wanted[desired]++;
        r->bs.lod_requests[desired]++;
        uint32_t b = l->page_off + (bz * l->by + by) * l->bx + bx;
        if (pl)
          bricks_candidate(r,
                           pl->page_off + ((bz >> 1u) * pl->by + (by >> 1u)) * pl->bx +
                               (bx >> 1u),
                           d2, 0u, g8, &ncand);
        bricks_candidate(r, b, d2, 1u, g8, &ncand);
      }
  r->bs.ncand_pending = ncand;
}

void r3d_bricks_stream_point(r3d_renderer *r, const float p[3], uint32_t level, float gate) {
  if (!r->bs.stream_open || !r->bricks_lod) return;
  int g8 = (int)(gate * 255.0f + 0.5f);
  if (g8 < BR_NOISE_FLOOR) g8 = BR_NOISE_FLOOR;
  if (level >= r->bricks_nlev) level = r->bricks_nlev - 1u;
  float maxdim = (float)r->bricks_maxdim;
  const r3d_brlod_level *l = &r->bricks_lev[level];
  float edge = (float)(BR_SLOT_DIM * l->scale) / maxdim;
  uint32_t ncand = r->bs.ncand_pending;
  uint32_t bc[3];
  const uint32_t bd[3] = {l->bx, l->by, l->bz};
  for (int a = 0; a < 3; a++) {
    float v = p[a] / edge;
    int64_t i = (int64_t)v;
    if (i < 0) i = 0;
    if (i >= (int64_t)bd[a]) i = (int64_t)bd[a] - 1;
    bc[a] = (uint32_t)i;
  }
  if (level + 1u < r->bricks_nlev) {
    const r3d_brlod_level *pl = &r->bricks_lev[level + 1u];
    bricks_candidate(r,
                     pl->page_off + ((bc[2] >> 1u) * pl->by + (bc[1] >> 1u)) * pl->bx +
                         (bc[0] >> 1u),
                     0.0f, 0u, g8, &ncand);
  }
  r->bs.lod_wanted[level]++;
  r->bs.lod_requests[level]++;
  bricks_candidate(r, l->page_off + (bc[2] * l->by + bc[1]) * l->bx + bc[0], 0.0f, 1u, g8,
                   &ncand);
  r->bs.ncand_pending = ncand;
}

void r3d_bricks_stream(r3d_renderer *r, const float eye[3], const float fwd[3], float half_tan,
                       float pixel_cone, uint32_t slice_z0, uint32_t slice_depth, float gate,
                       uint32_t budget) {
  if (!r3d_bricks_stream_begin(r)) return;
  if (budget == 0) {
    r->bs.stream_open = false;
    return;
  }
  uint32_t bpa = r->bricks_bpa, abpa = r->bricks_abpa, nb = r->bs.nb;
  (void)abpa;
  float tanw = half_tan * 1.15f + 1e-3f;
  int g8 = (int)(gate * 255.0f + 0.5f);
  if (g8 < BR_NOISE_FLOOR) g8 = BR_NOISE_FLOOR;

  /* desired set: bricks inside the (slightly widened) view cone or hugging the
   * camera. Resident ones get their LRU stamps; the rest become requests. */
  uint32_t ncand = 0;
  if (r->bricks_lod) {
    bricks_touch_coarsest(r);
    float maxdim = (float)r->bricks_maxdim;
    float lod_factor = fmaxf(pixel_cone * maxdim, 1e-6f);
    /* Level li is desired only inside its distance shell.  Restrict the grid
     * walk to that shell's AABB instead of rescanning all 861k virtual bricks
     * every frame.  The coarsest level is already complete and pinned. */
    for (uint32_t li = 0; li + 1u < r->bricks_nlev; li++) {
      const r3d_brlod_level *l = &r->bricks_lev[li];
      float edge = (float)(BR_SLOT_DIM * l->scale) / maxdim;
      float brad = 0.8660254f * edge;
      float outer = exp2f((float)li + 1.0f) / lod_factor + brad;
      uint32_t x0, x1, y0, y1, z0, z1;
      bricks_axis_bounds(eye[0], outer, edge, l->bx, &x0, &x1);
      bricks_axis_bounds(eye[1], outer, edge, l->by, &y0, &y1);
      bricks_axis_bounds(eye[2], outer, edge, l->bz, &z0, &z1);
      if (slice_depth) {
        uint64_t zlast = (uint64_t)slice_z0 + slice_depth;
        if (zlast > r->bricks_nz) zlast = r->bricks_nz;
        uint32_t zbrick = BR_SLOT_DIM * l->scale;
        uint32_t sz0 = slice_z0 / zbrick;
        uint32_t sz1 = (uint32_t)((zlast + zbrick - 1u) / zbrick);
        if (z0 < sz0) z0 = sz0;
        if (z1 > sz1) z1 = sz1;
      }
      for (uint32_t bz = z0; bz < z1; bz++)
        for (uint32_t by = y0; by < y1; by++)
          for (uint32_t bx = x0; bx < x1; bx++) {
        uint32_t local = (bz * l->by + by) * l->bx + bx;
        float cx = ((float)bx + 0.5f) * edge - eye[0];
        float cy = ((float)by + 0.5f) * edge - eye[1];
        float cz = ((float)bz + 0.5f) * edge - eye[2];
        float d2 = cx * cx + cy * cy + cz * cz;
        float along = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
        bool vis = d2 < (edge + brad) * (edge + brad);
        if (!vis && along > 0.0f) {
          float perp2 = d2 - along * along;
          float rad = along * tanw + brad;
          vis = perp2 < rad * rad;
        }
        if (!vis) continue;
        uint32_t desired = r3d_lod_pick(sqrtf(d2), pixel_cone, maxdim, r->bricks_nlev);
        if (li != desired) continue;
        r->bs.lod_wanted[desired]++;
        r->bs.lod_requests[desired]++;
        uint32_t b = l->page_off + local;
        /* Keep the immediate parent resident before refining children.  This
         * avoids cold regions jumping straight from L7 to fine data and gives
         * the shader a stable, spatially matching fallback during motion. */
        const r3d_brlod_level *pl = &r->bricks_lev[li + 1u];
        uint32_t pb = pl->page_off + ((bz >> 1u) * pl->by + (by >> 1u)) * pl->bx +
                      (bx >> 1u);
        bricks_candidate(r, pb, d2, 0u, g8, &ncand);
        bricks_candidate(r, b, d2, 1u, g8, &ncand);
      }
    }
  } else {
    float inv = 1.0f / (float)bpa;
    float brad = 0.8660254f * inv;
    for (uint32_t b = 0; b < nb; b++) {
      if (r->bs.brick_maxk[b] >= 0 && r->bs.brick_maxk[b] < g8) continue;
      float cx = ((float)(b % bpa) + 0.5f) * inv - eye[0];
      float cy = ((float)((b / bpa) % bpa) + 0.5f) * inv - eye[1];
      float cz = ((float)(b / (bpa * bpa)) + 0.5f) * inv - eye[2];
      float d2 = cx * cx + cy * cy + cz * cz;
      float along = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
      bool vis = d2 < (inv + brad) * (inv + brad);
      if (!vis && along > 0.0f) {
        float perp2 = d2 - along * along;
        float rad = along * tanw + brad;
        vis = perp2 < rad * rad;
      }
      if (!vis) continue;
      uint32_t slot = r->bs.brick_slot[b];
      if (slot != BR_INVALID) {
        r->bs.slot_use[slot] = r->bs.frame;
        if (r->bs.warm_off[b] != BR_INVALID) r->bs.warm_use[b] = r->bs.frame;
        continue;
      }
      r->bs.cands[ncand++] = (struct bcand){d2, b, 0u};
    }
  }
  r->bs.ncand_pending = ncand;
  r3d_bricks_stream_submit(r, budget);
}

void r3d_bricks_stream_submit(r3d_renderer *r, uint32_t budget) {
  if (!r->bs.stream_open) return;
  r->bs.stream_open = false;
  uint32_t ncand = r->bs.ncand_pending;
  r->bs.ncand_pending = 0;
  if (!ncand || !budget) return;
  if (budget > BR_MAX_BATCH) budget = BR_MAX_BATCH;
  uint32_t abpa = r->bricks_abpa;
  qsort(r->bs.cands, ncand, sizeof(struct bcand), bcand_cmp);

  /* nearest-first: warm-tier blob + hot slot per request, up to the budget */
  uint32_t n = 0, nevict = 0;
  uint32_t evict[BR_MAX_BATCH];
  for (uint32_t k = 0; k < ncand && n < budget; k++) {
    uint32_t b = r->bs.cands[k].b;
    if (r->bs.brick_slot[b] != BR_INVALID) continue; /* duped across collects */
    size_t bn = 0;
    const uint8_t *blob = warm_get(r, b, &bn);
    bool from_cache = false;
    if (!blob && r->ni.active && atomic_load(&r->ni.have[b]) == 1u)
      from_cache = true; /* on disk; the decode worker reads it (no IO here) */
    if (!blob && !from_cache) {
      /* not in any local tier: net-ingest it (stays a candidate until the
       * fetch pool caches it), or mark definitively absent */
      if (!bricks_net_request(r, b, 0)) r->bs.brick_maxk[b] = 0;
      continue;
    }
    uint32_t s = bricks_pick_slot(r);
    if (s == BR_INVALID) break; /* every slot wanted this frame: don't thrash */
    uint32_t old = r->bs.slot_brick[s];
    if (old != BR_INVALID) { /* LRU eviction; page invalidated after the drain */
      r->bs.brick_slot[old] = BR_INVALID;
      evict[nevict++] = old;
    }
    r->bs.slot_brick[s] = b;
    r->bs.slot_use[s] = r->bs.frame;
    r->bs.brick_slot[b] = s;
    r->bs.srcs[n] = (r3d_c5d_src){.blob = blob,
                                  .n = bn,
                                  .sx = (s % abpa) * BR_SLOT_DIM,
                                  .sy = ((s / abpa) % abpa) * BR_SLOT_DIM,
                                  .sz = (s / (abpa * abpa)) * BR_SLOT_DIM};
    r->bs.sel_b[n] = b;
    r->bs.sel_slot[n] = s;
    n++;
  }
  if (!n) return;

  /* Make evicted mappings unavailable before asynchronous overwrite. Host
   * writes to a mapped descriptor buffer must not overlap shader reads, so
   * first drain the frames that could still observe the old entries. This
   * waits at most the normal frames-in-flight latency, never the decode. */
  uint64_t timeline = r->timeline_value;
  if (timeline && nevict) { /* only evictions need the readers drained: a
                             * batch into free slots touches no live entry */
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                              .semaphoreCount = 1,
                              .pSemaphores = &r->timeline,
                              .pValues = &timeline};
    if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return;
  }
  uint32_t *page = r->page_buf.mapped;
  for (uint32_t i = 0; i < nevict; i++)
    page[bricks_page_index(r, evict[i])] = BR_INVALID;
  if (nevict) r->scene_gen++;
  pthread_mutex_lock(&r->bs.mu);
  memcpy(r->bs.job_evict, evict, (size_t)nevict * sizeof *evict);
  r->bs.job_nevict = nevict;
  r->bs.job_n = n;
  r->bs.job_timeline = 0; /* the render-thread drain above already completed */
  r->bs.job_rc = 0;
  r->bs.job_state = 1;
  r->bs.last_inflight = n;
  pthread_cond_signal(&r->bs.cv);
  pthread_mutex_unlock(&r->bs.mu);
}

void r3d_bricks_get_stats(r3d_renderer *r, r3d_bricks_stats *st) {
  memset(st, 0, sizeof *st);
  if (r->bs.worker_up) pthread_mutex_lock(&r->bs.mu);
  st->nb = r->bs.nb;
  st->hot_cap = r->bs.nslots;
  if (r->bs.worker_up) st->hot = r->bs.hot_cached;
  else
    for (uint32_t s = 0; s < r->bs.nslots; s++)
      if (r->bs.slot_brick[s] != BR_INVALID) st->hot++;
  st->warm_bricks = r->bs.warm_bricks;
  st->warm_bytes = r->bs.warm_bytes;
  st->warm_cap = r->bs.warm_cap;
  st->inflight = r->bs.last_inflight;
  st->decoded = r->bs.decoded;
  st->jobs = r->bs.jobs;
  st->stream_ns = r->bs.stream_ns;
  st->failures = r->bs.failures;
  st->nlevels = r->bricks_lod ? r->bricks_nlev : 1u;
  memcpy(st->lod_wanted, r->bs.lod_wanted, sizeof st->lod_wanted);
  memcpy(st->lod_requests, r->bs.lod_requests, sizeof st->lod_requests);
  if (r->bs.worker_up) pthread_mutex_unlock(&r->bs.mu);
  if (r->ni.active) {
    pthread_mutex_lock(&r->ni.mu);
    st->net_pending = r->ni.qn + r->ni.nin;
    pthread_mutex_unlock(&r->ni.mu);
    st->net_fetched = atomic_load(&r->ni.fetched);
    st->net_encoded = atomic_load(&r->ni.encoded);
  }
}

void r3d_bricks_flush(r3d_renderer *r) {
  if (!r->bs.worker_up) return;
  pthread_mutex_lock(&r->bs.mu);
  while (r->bs.job_state == 1 || r->bs.job_state == 2)
    pthread_cond_wait(&r->bs.cv, &r->bs.mu);
  bool complete = r->bs.job_state == 3;
  pthread_mutex_unlock(&r->bs.mu);
  if (!complete) return;

  /* The worker has finished the atlas upload, but the page entries are
   * intentionally published by the render thread only after all earlier
   * shader reads have drained.  Shutdown/benchmark flushes must perform the
   * same publication as the normal next-frame pump; otherwise the last good
   * batch is silently discarded. */
  uint64_t timeline = r->timeline_value;
  if (timeline) {
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                              .semaphoreCount = 1,
                              .pSemaphores = &r->timeline,
                              .pValues = &timeline};
    if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return;
  }
  pthread_mutex_lock(&r->bs.mu);
  if (r->bs.job_state == 3 && r->bs.job_rc == 0) {
    uint32_t *page = r->page_buf.mapped;
    uint32_t published = 0;
    for (uint32_t i = 0; i < r->bs.job_n; i++) {
      uint32_t b = r->bs.sel_b[i], s = r->bs.sel_slot[i];
      uint8_t m = r->bs.maxes[i];
      if (m >= BR_NOISE_FLOOR) {
        page[bricks_page_index(r, b)] = s | ((uint32_t)m << 24);
        published++;
      }
    }
    /* re-bake the flattened surface volume only once the new entries are
     * actually visible to its kernel (the decode-counter trigger in the app
     * loop can fire a frame early and then miss the publication) */
    if (published) {
      r3d_surfvol_mark(r);
      r->scene_gen++; /* plane views sample the newly resident bricks */
    }
  }
  r->bs.job_state = 0;
  pthread_mutex_unlock(&r->bs.mu);
}


/* ---------- virtual slab: toroidal streaming window over the export ------- */

#define VS_URL \
  "https://dl.ash2txt.org/community-uploads/forrest/exports/PHercParis3/" \
  "20260427095331-2.400um-0.2m-78keV-masked.zarr/0/c"

/* strided NxN box downsample (w,h multiples of N) */
static void vs_down(const uint8_t *src, uint32_t stride, uint32_t w, uint32_t h, uint32_t N,
                    uint8_t *dst) {
  uint32_t ow = w / N, oh = h / N;
  for (uint32_t y = 0; y < oh; y++) {
    uint8_t *drow = dst + (size_t)y * ow;
    for (uint32_t x = 0; x < ow; x++) {
      uint32_t sum = 0;
      for (uint32_t yy = 0; yy < N; yy++) {
        const uint8_t *sr = src + (size_t)(y * N + yy) * stride + (size_t)x * N;
        for (uint32_t xx = 0; xx < N; xx++) sum += sr[xx];
      }
      drow[x] = (uint8_t)(sum / (N * N));
    }
  }
}

typedef struct vs_fetch_req {
  char url[1200], path[1024], part[1088];
  int result; /* 0 downloaded, 1 permanently missing, -1 transient failure */
  long http;
  CURLcode curl_rc;
} vs_fetch_req;

typedef struct vs_coord {
  int64_t z, y, x;
} vs_coord;

static pthread_once_t curl_once = PTHREAD_ONCE_INIT;
static CURLcode curl_init_rc = CURLE_FAILED_INIT;
static void curl_init_once(void) { curl_init_rc = curl_global_init(CURL_GLOBAL_DEFAULT); }

static void *vs_fetch_one(void *arg) {
  vs_fetch_req *q = arg;
  q->result = -1;
  q->http = 0;
  q->curl_rc = CURLE_FAILED_INIT;
  pthread_once(&curl_once, curl_init_once);
  if (curl_init_rc != CURLE_OK) return NULL;
  for (unsigned attempt = 0; attempt < 3; attempt++) {
    FILE *f = fopen(q->part, "wb");
    if (!f) return NULL;
    CURL *curl = curl_easy_init();
    if (!curl) {
      fclose(f);
      return NULL;
    }
    curl_easy_setopt(curl, CURLOPT_URL, q->url);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "render3d/0.0.1");
    q->curl_rc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &q->http);
    curl_easy_cleanup(curl);
    if (fclose(f) != 0 && q->curl_rc == CURLE_OK) q->curl_rc = CURLE_WRITE_ERROR;

    bool http_ok = q->http == 0 || (q->http >= 200 && q->http < 300);
    if (q->curl_rc == CURLE_OK && http_ok) {
      r3d_shard sh;
      if (r3d_shard_open_path(q->part, &sh) == R3D_SHARD_OK) {
        r3d_shard_close(&sh);
        if (rename(q->part, q->path) == 0) {
          q->result = 0;
          return NULL;
        }
      }
    }
    unlink(q->part);
    if (q->http == 404 || q->http == 410) {
      char marker[1100];
      int n = snprintf(marker, sizeof marker, "%s.missing", q->path);
      if (n > 0 && (size_t)n < sizeof marker) {
        int fd = open(marker, O_CREAT | O_WRONLY | O_CLOEXEC, 0644);
        if (fd >= 0) close(fd);
      }
      q->result = 1;
      return NULL;
    }
    if (attempt + 1 < 3) {
      struct timespec delay = {.tv_sec = 0, .tv_nsec = (long)(attempt + 1) * 250000000L};
      nanosleep(&delay, NULL);
    }
  }
  return NULL;
}

/* Make sure shards covering a world box exist locally. Downloads are direct
 * libcurl requests (no shell), concurrent in batches of six, retried on
 * transport/5xx failures, CRC-validated, and atomically renamed. Only an
 * authoritative 404/410 creates a persistent sparse-data marker. */
static int vs_fetch(r3d_renderer *r, int64_t wx0, int64_t wy0, int64_t zs0, uint32_t nx,
                    uint32_t ny, uint32_t nz) {
  if (!r->vsl.fetch) return 0;
  if (!nx || !ny || !nz || wx0 < 0 || wy0 < 0 || zs0 < 0 ||
      (uint64_t)wx0 + nx - 1 > INT64_MAX || (uint64_t)wy0 + ny - 1 > INT64_MAX ||
      (uint64_t)zs0 + nz - 1 > INT64_MAX)
    return -1;
  /* Apron requests at the positive volume faces intentionally extend by one
   * texel. Clamp them here so they do not become pointless HTTP requests for
   * shard x/y==grid-size (or z just beyond the last slice). */
  if ((uint64_t)wx0 >= r->vsl.store.nx || (uint64_t)wy0 >= r->vsl.store.ny ||
      (uint64_t)zs0 >= r->vsl.store.nz)
    return 0;
  if (nx > r->vsl.store.nx - (uint64_t)wx0) nx = (uint32_t)(r->vsl.store.nx - (uint64_t)wx0);
  if (ny > r->vsl.store.ny - (uint64_t)wy0) ny = (uint32_t)(r->vsl.store.ny - (uint64_t)wy0);
  if (nz > r->vsl.store.nz - (uint64_t)zs0) nz = (uint32_t)(r->vsl.store.nz - (uint64_t)zs0);
  if (!nx || !ny || !nz) return 0;
  int64_t sx0 = wx0 / 1024, sx1 = (wx0 + nx - 1) / 1024;
  int64_t sy0 = wy0 / 1024, sy1 = (wy0 + ny - 1) / 1024;
  int64_t sz0 = zs0 / 1024, sz1 = (zs0 + nz - 1) / 1024;
  uint64_t nsx = (uint64_t)(sx1 - sx0) + 1, nsy = (uint64_t)(sy1 - sy0) + 1;
  uint64_t nsz = (uint64_t)(sz1 - sz0) + 1;
  if (nsx > UINT64_MAX / nsy || nsx * nsy > UINT64_MAX / nsz ||
      nsx * nsy * nsz > SIZE_MAX / sizeof(vs_coord))
    return -1;
  size_t cap = (size_t)(nsx * nsy * nsz);
  vs_coord *miss = calloc(cap, sizeof *miss);
  if (!miss) return -1;
  size_t nm = 0;
  for (int64_t sz = sz0; sz <= sz1; sz++)
    for (int64_t sy = sy0; sy <= sy1; sy++)
      for (int64_t sx = sx0; sx <= sx1; sx++) {
        char path[1024];
        int pn = snprintf(path, sizeof path, "%s/%lld_%lld_%lld.shard", r->vsl.band_dir,
                          (long long)sz, (long long)sy, (long long)sx);
        if (pn < 0 || (size_t)pn >= sizeof path) {
          free(miss);
          return -1;
        }
        if (access(path, F_OK) == 0) continue;
        char mk[1088];
        int mn = snprintf(mk, sizeof mk, "%s.missing", path);
        if (mn < 0 || (size_t)mn >= sizeof mk) {
          free(miss);
          return -1;
        }
        if (access(mk, F_OK) == 0) continue;
        miss[nm].z = sz;
        miss[nm].y = sy;
        miss[nm++].x = sx;
      }
  bool failed = false;
  const char *base = getenv("R3D_SHARD_URL");
  if (!base || !*base) base = r->vsl.shard_url[0] ? r->vsl.shard_url : VS_URL;
  for (size_t b = 0; b < nm; b += 6) {
    size_t be = b + 6 < nm ? b + 6 : nm;
    vs_fetch_req req[6];
    pthread_t threads[6];
    bool threaded[6] = {false};
    printf("vslab: fetching %zu shard(s)...\n", be - b);
    fflush(stdout);
    for (size_t i = b; i < be; i++) {
      vs_fetch_req *q = &req[i - b];
      memset(q, 0, sizeof *q);
      q->result = -1;
      q->curl_rc = CURLE_URL_MALFORMAT;
      int pn = snprintf(q->path, sizeof q->path, "%s/%lld_%lld_%lld.shard", r->vsl.band_dir,
                        (long long)miss[i].z, (long long)miss[i].y,
                        (long long)miss[i].x);
      int tn = pn < 0 || (size_t)pn >= sizeof q->path
                   ? -1
                   : snprintf(q->part, sizeof q->part, "%s.part", q->path);
      int un = snprintf(q->url, sizeof q->url, "%s/%lld/%lld/%lld", base,
                        (long long)miss[i].z, (long long)miss[i].y,
                        (long long)miss[i].x);
      if (pn < 0 || (size_t)pn >= sizeof q->path || tn < 0 ||
          (size_t)tn >= sizeof q->part || un < 0 || (size_t)un >= sizeof q->url)
        continue;
      if (pthread_create(&threads[i - b], NULL, vs_fetch_one, q) == 0)
        threaded[i - b] = true;
      else
        vs_fetch_one(q);
    }
    for (size_t i = 0; i < be - b; i++)
      if (threaded[i]) pthread_join(threads[i], NULL);
    for (size_t i = b; i < be; i++) {
      const vs_fetch_req *q = &req[i - b];
      if (q->result == 0) continue;
      if (q->result == 1) {
        printf("vslab: shard %lld_%lld_%lld absent (HTTP %ld)\n", (long long)miss[i].z,
               (long long)miss[i].y, (long long)miss[i].x, q->http);
      } else {
        fprintf(stderr, "vslab: shard %lld_%lld_%lld fetch failed (curl %d, HTTP %ld)\n",
                (long long)miss[i].z, (long long)miss[i].y, (long long)miss[i].x,
                (int)q->curl_rc, q->http);
        failed = true;
      }
    }
  }
  free(miss);
  return failed ? -1 : 0;
}

static int vs_fetch_serial(r3d_renderer *r, int64_t wx0, int64_t wy0, int64_t zs0,
                           uint32_t nx, uint32_t ny, uint32_t nz) {
  pthread_mutex_lock(&r->vsl.fetch_mu);
  int rc = vs_fetch(r, wx0, wy0, zs0, nx, ny, nz);
  pthread_mutex_unlock(&r->vsl.fetch_mu);
  return rc;
}

static bool vs_pc_contains(const struct vspc_entry *e, int64_t z0, uint32_t nz) {
  return e->state == 2 && e->z0 <= z0 && z0 + (int64_t)nz <= e->z0 + (int64_t)e->nz;
}

static void *vs_prefetch_worker(void *arg) {
  r3d_renderer *r = arg;
  r3d_vslab *v = &r->vsl.v;
  for (;;) {
    pthread_mutex_lock(&r->vsl.pc.mu);
    int victim = -1;
    int64_t target = -1;
    for (;;) {
      if (r->vsl.pc.quit || r->vsl.pc.failed) {
        pthread_mutex_unlock(&r->vsl.pc.mu);
        return NULL;
      }
      for (uint32_t w = 0; w < R3D_VSLAB_PREFETCH_MAX && target < 0; w++) {
        int64_t z0 = r->vsl.pc.wanted[w];
        if (z0 < 0) continue;
        bool have = false;
        for (uint32_t i = 0; i < r->vsl.pc.nslots; i++) {
          const struct vspc_entry *e = &r->vsl.pc.slot[i];
          if ((e->state == 1 || e->state == 2) && e->z0 == z0 &&
              e->nz == r->vsl.pc.window_nz) {
            have = true;
            break;
          }
        }
        if (!have) target = z0;
      }
      if (target >= 0) {
        for (uint32_t i = 0; i < r->vsl.pc.nslots; i++)
          if (r->vsl.pc.slot[i].state == 0) {
            victim = (int)i;
            break;
          }
        /* Preserve the two current targets when possible; evict the oldest
         * unpinned non-target entry first. */
        for (uint32_t pass = 0; pass < 2 && victim < 0; pass++) {
          uint64_t oldest = UINT64_MAX;
          for (uint32_t i = 0; i < r->vsl.pc.nslots; i++) {
            const struct vspc_entry *e = &r->vsl.pc.slot[i];
            bool wanted = false;
            for (uint32_t w = 0; w < R3D_VSLAB_PREFETCH_MAX; w++)
              wanted |= e->z0 == r->vsl.pc.wanted[w];
            if (e->state != 2 || e->refs || (!pass && wanted) || e->used >= oldest) continue;
            oldest = e->used;
            victim = (int)i;
          }
        }
        if (victim >= 0) break;
      }
      target = -1;
      pthread_cond_wait(&r->vsl.pc.cv, &r->vsl.pc.mu);
    }
    struct vspc_entry *e = &r->vsl.pc.slot[victim];
    e->state = 1;
    e->z0 = target;
    e->nz = r->vsl.pc.window_nz;
    e->used = ++r->vsl.pc.clock;
    pthread_mutex_unlock(&r->vsl.pc.mu);

    size_t plane = (size_t)v->nx * (size_t)v->ny;
    size_t bytes = plane * r->vsl.pc.window_nz;
    if (!e->data) e->data = malloc(bytes);
    uint64_t started = now_ns();
    uint32_t fetch_nz = r->vsl.pc.window_nz;
    if ((uint64_t)target + fetch_nz > v->nz) fetch_nz = (uint32_t)(v->nz - (uint64_t)target);
    int rc = !e->data ||
                     vs_fetch_serial(r, 0, 0, target, (uint32_t)v->nx, (uint32_t)v->ny,
                                     fetch_nz) != 0 ||
                     r3d_shard_decode_region(&r->vsl.store, (uint64_t)target, 0, 0,
                                             r->vsl.pc.window_nz,
                                             (uint32_t)v->ny, (uint32_t)v->nx, e->data,
                                             (int)r->vsl.pc.nthreads) != 0
                 ? -1
                 : 0;
    double ms = (double)(now_ns() - started) / 1e6;
    pthread_mutex_lock(&r->vsl.pc.mu);
    e->state = rc == 0 ? 2u : 0u;
    e->used = ++r->vsl.pc.clock;
    r->vsl.pc.last_decode_ms = ms;
    if (rc != 0) r->vsl.pc.failed = true;
    pthread_cond_broadcast(&r->vsl.pc.cv);
    pthread_mutex_unlock(&r->vsl.pc.mu);
    if (rc == 0)
      printf("vslab: prefetched decoded z%lld+%u (%.0f ms, %.1f MiB)\n",
             (long long)target, r->vsl.pc.window_nz, ms, (double)bytes / 1048576.0);
    else
      fprintf(stderr, "vslab: decoded prefetch failed at z%lld; disabling cache\n",
              (long long)target);
  }
}

/* Copy a whole-plane decoded cache hit into the upload worker's apron tile. */
static int vs_pc_copy_cell(r3d_renderer *r, int64_t cx, int64_t cy, int64_t zs0,
                           uint32_t nz) {
  if (!r->vsl.pc.up) return 0;
  r3d_vslab *v = &r->vsl.v;
  pthread_mutex_lock(&r->vsl.pc.mu);
  struct vspc_entry *e = NULL;
  for (uint32_t i = 0; i < r->vsl.pc.nslots; i++)
    if (vs_pc_contains(&r->vsl.pc.slot[i], zs0, nz)) {
      e = &r->vsl.pc.slot[i];
      break;
    }
  if (!e) {
    r->vsl.pc.misses++;
    pthread_mutex_unlock(&r->vsl.pc.mu);
    return 0;
  }
  e->refs++;
  e->used = ++r->vsl.pc.clock;
  uint8_t *cached = e->data;
  int64_t cached_z0 = e->z0;
  pthread_mutex_unlock(&r->vsl.pc.mu);

  uint32_t P = v->px, tex = P + 2;
  int64_t wx0 = cx * (int64_t)P - 1, wy0 = cy * (int64_t)P - 1;
  size_t plane = (size_t)v->nx * (size_t)v->ny;
  memset(r->vsl.box, 0, (size_t)nz * tex * tex);
  for (uint32_t k = 0; k < nz; k++) {
    const uint8_t *slice = cached + (size_t)(zs0 + k - cached_z0) * plane;
    uint8_t *dst = r->vsl.box + (size_t)k * tex * tex;
    for (uint32_t ty = 0; ty < tex; ty++) {
      int64_t wy = wy0 + ty;
      if (wy < 0) wy = 0;
      if (wy >= (int64_t)v->ny) continue;
      uint32_t lead = wx0 < 0 ? (uint32_t)(-wx0) : 0;
      int64_t sx = wx0 < 0 ? 0 : wx0;
      if (sx >= (int64_t)v->nx || lead >= tex) continue;
      size_t n = tex - lead;
      if (n > v->nx - (uint64_t)sx) n = (size_t)(v->nx - (uint64_t)sx);
      const uint8_t *src = slice + (size_t)wy * (size_t)v->nx + (size_t)sx;
      memcpy(dst + (size_t)ty * tex + lead, src, n);
      if (lead) memset(dst + (size_t)ty * tex, src[0], lead);
    }
  }

  pthread_mutex_lock(&r->vsl.pc.mu);
  e->refs--;
  r->vsl.pc.hits++;
  pthread_cond_broadcast(&r->vsl.pc.cv);
  pthread_mutex_unlock(&r->vsl.pc.mu);
  return 1;
}

/* Upload consecutive world slices in at most two submissions (the z ring may
 * wrap once). host_image_height preserves a larger source-plane pitch when
 * the upload is a sub-rectangle of a decoded 3-D box. */
static int vs_copy_depth(r3d_renderer *r, const uint8_t *host, uint32_t row_length,
                         uint32_t host_image_height, uint32_t dx, uint32_t dy, int64_t zs0,
                         uint32_t nz, uint32_t w, uint32_t h, r3d_vkimage *img) {
  uint32_t done = 0;
  while (done < nz) {
    uint32_t layer = r3d_vs_layer(&r->vsl.v, zs0 + done);
    uint32_t run = nz - done;
    if (run > r->vsl.v.wz - layer) run = r->vsl.v.wz - layer;
    size_t plane = (size_t)row_length * host_image_height;
    if (r3d_vk_upload_image_staged_buf_pitch(
            &r->vk, r->vsl.upload_pool, &r->vsl.upload_stage, img,
            host + (size_t)done * plane, row_length, host_image_height,
            (VkOffset3D){(int32_t)dx, (int32_t)dy, (int32_t)layer},
            (VkExtent3D){w, h, run}) != 0)
      return -1;
    done += run;
  }
  return 0;
}

/* Scatter a z stack of level strips (n x n texels at world texel origin t0)
 * into the up to four overlapped pyramid tiles, aprons included. */
static int vs_scatter_depth(r3d_renderer *r, uint32_t l, int64_t cx, int64_t cy,
                            const uint8_t *strip, int64_t zs0, uint32_t nz) {
  const r3d_vs_level *lv = &r->vsl.v.lv[l];
  int64_t P = r->vsl.v.px;
  uint32_t n = (uint32_t)P / lv->s;
  int64_t t0x = cx * (int64_t)n, t0y = cy * (int64_t)n;
  int64_t p0x = (t0x - 1) / P, p1x = (t0x + n) / P;
  int64_t p0y = (t0y - 1) / P, p1y = (t0y + n) / P;
  if (p0x < 0) p0x = 0;
  if (p0y < 0) p0y = 0;
  for (int64_t py = p0y; py <= p1y; py++)
    for (int64_t px = p0x; px <= p1x; px++) {
      int64_t ax = t0x > px * P - 1 ? t0x : px * P - 1;
      int64_t bx = t0x + n < px * P + P + 1 ? t0x + n : px * P + P + 1;
      int64_t ay = t0y > py * P - 1 ? t0y : py * P - 1;
      int64_t by = t0y + n < py * P + P + 1 ? t0y + n : py * P + P + 1;
      if (ax >= bx || ay >= by) continue;
      uint32_t ti = lv->base + r3d_vs_phys(py, lv->gy) * lv->gx + r3d_vs_phys(px, lv->gx);
      if (vs_copy_depth(r, strip + (size_t)(ay - t0y) * n + (size_t)(ax - t0x), n, n,
                        (uint32_t)(ax - px * P + 1), (uint32_t)(ay - py * P + 1), zs0, nz,
                        (uint32_t)(bx - ax), (uint32_t)(by - ay), &r->tiles[ti]) != 0)
        return -1;
    }
  return 0;
}

/* fill world z slices [zs0, zs0+nz) of base cell (cx,cy): decode (apron
 * included, volume edges duplicated), upload the base tile layers, and
 * derive + scatter every pyramid level */
static int vs_fill(r3d_renderer *r, int64_t cx, int64_t cy, int64_t zs0, uint32_t nz) {
  r3d_vslab *v = &r->vsl.v;
  uint32_t P = v->px, tex = P + 2;
  int64_t wx0 = cx * (int64_t)P - 1, wy0 = cy * (int64_t)P - 1;
  int64_t ax = wx0 < 0 ? 0 : wx0, ay = wy0 < 0 ? 0 : wy0;
  uint32_t lx = (uint32_t)(ax - wx0), ly = (uint32_t)(ay - wy0); /* leading dup */
  if (!vs_pc_copy_cell(r, cx, cy, zs0, nz)) {
    if (vs_fetch_serial(r, ax, ay, zs0, tex, tex, nz) != 0) return -1;
    if (r3d_shard_decode_region(&r->vsl.store, (uint64_t)zs0, (uint64_t)ay, (uint64_t)ax, nz,
                                tex - ly, tex - lx, r->vsl.box, 0) != 0)
      return -1;
    /* un-shift in place when the apron clamped at world 0 (edge cells): the
     * decode produced packed (tex-lx)x(tex-ly) slices; expand descending so
     * dst addresses stay >= src, duplicating the clamped border */
    if (lx || ly) {
      for (int64_t z = nz - 1; z >= 0; z--) {
        const uint8_t *sp = r->vsl.box + (size_t)z * (tex - ly) * (tex - lx);
        uint8_t *dp = r->vsl.box + (size_t)z * tex * tex;
        for (int64_t y = tex - 1; y >= 0; y--) {
          int64_t sy = y - ly < 0 ? 0 : y - ly;
          memmove(dp + (size_t)y * tex + lx, sp + (size_t)sy * (tex - lx), tex - lx);
          if (lx) dp[(size_t)y * tex] = dp[(size_t)y * tex + 1];
        }
      }
    }
  }
  uint32_t lmax = 0; /* pyramid levels are a prefix of the table */
  for (uint32_t l = 1; l < R3D_VS_LEVELS && v->lv[l].gx; l++) lmax = l;
  uint32_t slot = r3d_vs_phys(cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(cx, v->lv[0].gx);
  if (vs_copy_depth(r, r->vsl.box, tex, tex, 0, 0, zs0, nz, tex, tex,
                    &r->tiles[v->lv[0].base + slot]) != 0)
    return -1;
  if (!lmax) return 0;
  for (uint32_t k = 0; k < nz; k++) {
    const uint8_t *slice = r->vsl.box + (size_t)k * tex * tex;
    /* pyramid: 4x from the payload, then 2x chain; scatter each present level */
    vs_down(slice + tex + 1, tex, P, P, 4,
            r->vsl.ds[0] + (size_t)k * (P / 4) * (P / 4));
    for (uint32_t l = 2; l <= lmax; l++)
      vs_down(r->vsl.ds[l - 2] +
                  (size_t)k * (P / r3d_vs_scale[l - 1]) * (P / r3d_vs_scale[l - 1]),
              P / r3d_vs_scale[l - 1], P / r3d_vs_scale[l - 1],
              P / r3d_vs_scale[l - 1], 2,
              r->vsl.ds[l - 1] +
                  (size_t)k * (P / r3d_vs_scale[l]) * (P / r3d_vs_scale[l]));
  }
  for (uint32_t l = 1; l <= lmax; l++)
    if (vs_scatter_depth(r, l, cx, cy, r->vsl.ds[l - 1], zs0, nz) != 0) return -1;
  return 0;
}

/* window-wide band fill: decode ONE (cell-box x nz) slab and scatter it into
 * every cell tile. Band mode (small xy windows — exactly the no-pyramid case)
 * exists because per-cell decode re-decodes whole 1024^2 dct3d chunks per
 * cell: up to ~60x amplification on narrow cells vs ~2x window-wide. Cells
 * touching the volume boundary fall back to the per-cell path (apron dup). */
static int vs_fill_band(r3d_renderer *r, int64_t cx0, int64_t cy0, int64_t cx1, int64_t cy1,
                        int64_t zs0, uint32_t nz) {
  r3d_vslab *v = &r->vsl.v;
  uint32_t P = v->px, tex = P + 2;
  int64_t ax = cx0 * (int64_t)P - 1, ay = cy0 * (int64_t)P - 1;
  int64_t bx = (cx1 + 1) * (int64_t)P + 1, by = (cy1 + 1) * (int64_t)P + 1;
  if (ax < 0 || ay < 0 || bx > (int64_t)v->nx || by > (int64_t)v->ny) {
    for (int64_t cy = cy0; cy <= cy1; cy++)
      for (int64_t cx = cx0; cx <= cx1; cx++)
        if (vs_fill(r, cx, cy, zs0, nz) != 0) return -1;
    return 0;
  }
  uint32_t bw = (uint32_t)(bx - ax), bh = (uint32_t)(by - ay);
  if (vs_fetch_serial(r, ax, ay, zs0, bw, bh, nz) != 0) return -1;
  if (r3d_shard_decode_region(&r->vsl.store, (uint64_t)zs0, (uint64_t)ay, (uint64_t)ax, nz, bh,
                              bw, r->vsl.bbox, 0) != 0)
    return -1;
  for (int64_t cy = cy0; cy <= cy1; cy++)
    for (int64_t cx = cx0; cx <= cx1; cx++) {
        uint32_t slot =
            r3d_vs_phys(cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(cx, v->lv[0].gx);
        const uint8_t *src = r->vsl.bbox + (size_t)(cy * (int64_t)P - 1 - ay) * bw +
                             (size_t)(cx * (int64_t)P - 1 - ax);
        if (vs_copy_depth(r, src, bw, bh, 0, 0, zs0, nz, tex, tex,
                          &r->tiles[v->lv[0].base + slot]) != 0)
          return -1;
    }
  return 0;
}

/* fill worker: drains ready jobs; each = drain in-flight frames from enqueue
 * time, then fetch + decode + upload one cell box (or z strip) */
static void *vs_worker(void *arg) {
  r3d_renderer *r = arg;
  for (;;) {
    pthread_mutex_lock(&r->vsl.mu);
    int ji = -1;
    for (;;) {
      if (r->vsl.quit) {
        pthread_mutex_unlock(&r->vsl.mu);
        return NULL;
      }
      for (uint32_t i = 0; i < 4 && ji < 0; i++)
        if (r->vsl.q[i].st == 1) ji = (int)i;
      if (ji >= 0) break;
      pthread_cond_wait(&r->vsl.cv, &r->vsl.mu);
    }
    struct vsjob j = r->vsl.q[ji];
    r->vsl.q[ji].st = 2;
    pthread_mutex_unlock(&r->vsl.mu);
    if (j.tl) {
      VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                .semaphoreCount = 1,
                                .pSemaphores = &r->timeline,
                                .pValues = &j.tl};
      vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX);
    }
    int rc = r->vsl.band ? vs_fill_band(r, j.cx, j.cy, j.cx1, j.cy1, j.zs0, j.nz)
                         : vs_fill(r, j.cx, j.cy, j.zs0, j.nz);
    if (getenv("R3D_VSLAB_DEBUG"))
      fprintf(stderr, "vsjob (%lld,%lld)-(%lld,%lld) z%lld+%u -> %d\n", (long long)j.cx,
              (long long)j.cy, (long long)j.cx1, (long long)j.cy1, (long long)j.zs0, j.nz, rc);
    if (rc != 0)
      fprintf(stderr, "vslab: fill (%lld,%lld) failed\n", (long long)j.cx, (long long)j.cy);
    pthread_mutex_lock(&r->vsl.mu);
    r->vsl.q[ji].st = rc == 0 ? 3 : 0;
    pthread_mutex_unlock(&r->vsl.mu);
  }
}

static int vs_mkdirs(const char *path) {
  if (!path || !*path || strlen(path) >= 512) return -1;
  char buf[512];
  snprintf(buf, sizeof buf, "%s", path);
  for (char *p = buf + 1; *p; p++) {
    if (*p != '/') continue;
    *p = 0;
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
    *p = '/';
  }
  if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
  struct stat st;
  return stat(buf, &st) == 0 && S_ISDIR(st.st_mode) ? 0 : -1;
}

int r3d_vslab_begin(r3d_renderer *r, const r3d_vslab_desc *d) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!d || !d->cache_dir || !d->nx || !d->ny || !d->nz || !d->W || !d->H || !d->D ||
      strlen(d->cache_dir) >= sizeof r->vsl.band_dir ||
      (d->shard_url && strlen(d->shard_url) >= sizeof r->vsl.shard_url)) {
    fprintf(stderr, "vslab: invalid descriptor\n");
    return -1;
  }
  if (!r->tiled_modes) {
    fprintf(stderr, "vslab: device lacks native non-uniform descriptor indexing\n");
    return -1;
  }
  if (r3d_vslab_init_margin(&r->vsl.v, d->nx, d->ny, d->nz, d->W, d->H, d->D,
                            d->z_prefetch) != 0) {
    fprintf(stderr, "vslab: bad %ux%ux%u window for %ux%ux%u volume\n", d->W, d->H, d->D,
            d->nx, d->ny, d->nz);
    return -1;
  }
  r3d_vslab *v = &r->vsl.v;
  if (v->ntiles > R3D_SLAB_TILES) {
    fprintf(stderr, "vslab: window needs %u tiles (> %u)\n", v->ntiles, R3D_SLAB_TILES);
    return -1;
  }
  uint32_t tex = v->px + 2;
  double gb = (double)v->ntiles * tex * tex * v->wz / 1e9;
  uint64_t tile_bytes = (uint64_t)v->ntiles * tex * tex * v->wz;
  if (tile_bytes > r3d_vkctx_budget_available(&r->vk)) {
    fprintf(stderr,
            "vslab: %ux%ux%u needs %.1f GB of tiles, but renderer budget has %.1f GB free\n",
            d->W, d->H, d->D, gb, (double)r3d_vkctx_budget_available(&r->vk) / 1e9);
    return -1;
  }
  if (vs_mkdirs(d->cache_dir) != 0) {
    fprintf(stderr, "vslab: cannot create cache directory %s: %s\n", d->cache_dir,
            strerror(errno));
    return -1;
  }
  if (r3d_shard_store_init(&r->vsl.store, d->cache_dir, d->nz, d->ny, d->nx) != 0)
    return -1;
  snprintf(r->vsl.band_dir, sizeof r->vsl.band_dir, "%s", d->cache_dir);
  if (d->shard_url)
    snprintf(r->vsl.shard_url, sizeof r->vsl.shard_url, "%s", d->shard_url);
  if (!r->samp_slab) {
    VkSamplerCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT, /* ring z wraps */
    };
    if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_slab) != VK_SUCCESS) return -1;
  }
  VkCommandPoolCreateInfo upci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = r->vk.qfam,
  };
  if (vkCreateCommandPool(r->vk.dev, &upci, NULL, &r->vsl.upload_pool) != VK_SUCCESS) return -1;
  for (uint32_t t = 0; t < v->ntiles; t++) {
    if (r3d_vkimage_create_arena(&r->vk, &r->tile_arena, VK_FORMAT_R8_UNORM,
                                 (VkExtent3D){tex, tex, v->wz}, 1,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 &r->tiles[t]) != 0)
      return -1;
    if (r3d_vk_image_to_general(&r->vk, r->vsl.upload_pool, &r->tiles[t]) != 0) return -1;
  }
  VkImageView views[R3D_SLAB_TILES];
  for (uint32_t e = 0; e < R3D_SLAB_TILES; e++)
    views[e] = r->tiles[e].view ? r->tiles[e].view : r->tiles[0].view;
  VkDescriptorImageInfo ii[R3D_SLAB_TILES];
  for (uint32_t e = 0; e < R3D_SLAB_TILES; e++)
    ii[e] = (VkDescriptorImageInfo){
        .sampler = r->samp_slab, .imageView = views[e], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = r->tile_descriptors,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
  /* validity table on binding 5 (bricks and vslab are exclusive): 4 words per
   * base slot = {key, za, zb, pad} — the z range lets partial fills render
   * and gates in-progress strip writes out of the sampled set */
  uint32_t nslot = v->lv[0].gx * v->lv[0].gy;
  r3d_vkbuf_destroy(&r->vk, &r->page_buf);
  if (r3d_vkbuf_create_host(&r->vk, (VkDeviceSize)nslot * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            &r->page_buf) != 0)
    return -1;
  memset(r->page_buf.mapped, 0, (size_t)nslot * 16);
  VkDescriptorBufferInfo pbi = {.buffer = r->page_buf.buf, .range = VK_WHOLE_SIZE};
  VkWriteDescriptorSet pw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = r->dset,
                             .dstBinding = 5,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &pbi};
  vkUpdateDescriptorSets(r->vk.dev, 1, &pw, 0, NULL);
  r->vsl.cells = malloc((size_t)nslot * sizeof(struct vscell));
  r->vsl.box = malloc((size_t)tex * tex * v->wz);
  for (uint32_t l = 0; l < 4; l++) {
    size_t n = (size_t)v->px / r3d_vs_scale[l + 1];
    r->vsl.ds[l] = malloc(n * n * v->wz);
  }
  if (!r->vsl.cells || !r->vsl.box || !r->vsl.ds[3]) return -1;
  r->vsl.band = (d->W > d->H ? d->W : d->H) < 4096; /* == the no-pyramid case */
  if (r->vsl.band) {
    size_t bw = (size_t)v->lv[0].gx * v->px + 2;
    r->vsl.bbox = malloc(bw * bw * 16);
    if (!r->vsl.bbox) return -1;
  }
  for (uint32_t i = 0; i < nslot; i++)
    r->vsl.cells[i] = (struct vscell){INT64_MIN, INT64_MIN, 0, 0};
  r->vsl.fetch = !getenv("R3D_VSLAB_NOFETCH");
  pthread_mutex_init(&r->vsl.mu, NULL);
  pthread_cond_init(&r->vsl.cv, NULL);
  pthread_mutex_init(&r->vsl.fetch_mu, NULL);
  memset(r->vsl.q, 0, sizeof r->vsl.q);
  r->vsl.quit = false;
  if (pthread_create(&r->vsl.worker, NULL, vs_worker, r) != 0) return -1;
  r->vsl.active = true;
  if (d->prefetch_slots && !getenv("R3D_VSLAB_NOPC") && d->W == d->nx && d->H == d->ny) {
    r->vsl.pc.nslots = d->prefetch_slots > R3D_VSLAB_PREFETCH_MAX
                           ? R3D_VSLAB_PREFETCH_MAX
                           : d->prefetch_slots;
    r->vsl.pc.nthreads = d->prefetch_threads ? d->prefetch_threads : 4;
    r->vsl.pc.window_nz = d->D + 2;
    if (r->vsl.pc.nthreads > 16) r->vsl.pc.nthreads = 16;
    for (uint32_t i = 0; i < R3D_VSLAB_PREFETCH_MAX; i++) r->vsl.pc.wanted[i] = -1;
    pthread_mutex_init(&r->vsl.pc.mu, NULL);
    pthread_cond_init(&r->vsl.pc.cv, NULL);
    r->vsl.pc.quit = false;
    if (pthread_create(&r->vsl.pc.worker, NULL, vs_prefetch_worker, r) == 0) {
      r->vsl.pc.up = true;
      double mib = (double)d->nx * d->ny * r->vsl.pc.window_nz / 1048576.0;
      printf("vslab: decoded prefetch cache %u x %.1f MiB (%u threads)\n",
             r->vsl.pc.nslots, mib, r->vsl.pc.nthreads);
    } else {
      pthread_mutex_destroy(&r->vsl.pc.mu);
      pthread_cond_destroy(&r->vsl.pc.cv);
      fprintf(stderr, "vslab: cannot start decoded prefetch worker; continuing without it\n");
    }
  } else if (d->prefetch_slots && !getenv("R3D_VSLAB_NOPC")) {
    fprintf(stderr, "vslab: decoded prefetch requires a whole-XY window; disabling it\n");
  }
  v->x0 = v->y0 = -1; /* set by the first frame */
  v->z0 = -1;
  printf("vslab: %ux%ux%u window in %ux%ux%u, z ring %u (+/-%u), payload %u, "
         "%u tiles (%.1f GB), fetch %s\n",
         d->W, d->H, d->D, d->nx, d->ny, d->nz, v->wz, v->z_margin, v->px, v->ntiles,
         gb, r->vsl.fetch ? "on" : "off");
  printf("vslab: streaming upload path staging (queue-ordered background worker)\n");
  return 0;
}

void r3d_vslab_frame(r3d_renderer *r, double fx, double fy, int64_t z0, uint32_t budget,
                     r3d_frame_params *p) {
  r3d_vslab *v = &r->vsl.v;
  if (!r->vsl.active) return;
  int64_t px0 = v->x0, py0 = v->y0; /* last frame's origin -> motion direction */
  int64_t pz0 = v->z0;
  /* window follows the focus (cell-anchored tiles: origin moves freely) */
  int64_t x0 = (int64_t)fx - v->W / 2, y0 = (int64_t)fy - v->H / 2;
  x0 = x0 < 0 ? 0 : x0 & ~15ll;
  y0 = y0 < 0 ? 0 : y0 & ~15ll;
  if (x0 > r3d_vs_max0(v->nx, v->W)) x0 = r3d_vs_max0(v->nx, v->W) & ~15ll;
  if (y0 > r3d_vs_max0(v->ny, v->H)) y0 = r3d_vs_max0(v->ny, v->H) & ~15ll;
  if (z0 < 0) z0 = 0;
  /* D is the visible span. The larger ring keeps a contiguous neighborhood
   * GPU-resident for fine z scrolling; its unavailable margin shifts across
   * at volume faces. */
  if (z0 > r3d_vs_max0(v->nz, v->D)) z0 = r3d_vs_max0(v->nz, v->D);
  int64_t za_want, zb_want;
  r3d_vs_zrange(v, z0, &za_want, &zb_want);
  int64_t visible_end = z0 + (int64_t)v->D + 2;
  if (visible_end > (int64_t)v->nz) visible_end = (int64_t)v->nz;
  int zdir = pz0 < 0 || z0 >= pz0 ? 1 : -1;
  v->x0 = x0;
  v->y0 = y0;
  v->z0 = z0;

  uint32_t *page = r->page_buf.mapped;
  pthread_mutex_lock(&r->vsl.mu);
  /* apply completed fills: fold into the cell ledger, publish validity (key +
   * z range; partial z coverage renders immediately, the rest samples 0) */
  for (uint32_t i = 0; i < 4; i++) {
    struct vsjob *j = &r->vsl.q[i];
    if (j->st != 3) continue;
    for (int64_t cy = j->cy; cy <= j->cy1; cy++)
      for (int64_t cx = j->cx; cx <= j->cx1; cx++) {
        uint32_t slot = r3d_vs_phys(cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(cx, v->lv[0].gx);
        struct vscell *c = &r->vsl.cells[slot];
        if (c->cx != cx || c->cy != cy) { c->za = j->zs0; c->zb = j->zs0 + j->nz; }
        else {
          if (j->zs0 < c->za) c->za = j->zs0;
          if (j->zs0 + j->nz > c->zb) c->zb = j->zs0 + j->nz;
        }
        c->cx = cx;
        c->cy = cy;
        /* clamp the tracked range to the ring capacity around the window */
        if (c->za < za_want) c->za = za_want;
        if (c->zb > zb_want) c->zb = zb_want;
        if (c->zb < c->za) c->zb = c->za;
        page[slot * 4 + 1] = (uint32_t)c->za;
        page[slot * 4 + 2] = (uint32_t)c->zb;
        page[slot * 4] = r3d_vs_key(cx, cy);
      }
    j->st = 0;
  }
  /* enumerate needed base cells, nearest-first */
  int64_t cx0, cx1, cy0, cy1;
  r3d_vs_range(v, 0, x0, v->nx, v->W, &cx0, &cx1);
  r3d_vs_range(v, 0, y0, v->ny, v->H, &cy0, &cy1);
  struct vjob {
    int64_t cx, cy, zs0;
    uint32_t nz, prio; /* lower fills first; visible is always priority 0 */
    uint64_t d;
  } jobs[64];
  uint32_t nj = 0, nvisible = 0;
  int64_t ccx = (x0 + v->W / 2) / (int64_t)v->px;
  int64_t ccy = (y0 + v->H / 2) / (int64_t)v->px;
  for (int64_t cy = cy0; cy <= cy1 && nj < 64; cy++)
    for (int64_t cx = cx0; cx <= cx1 && nj < 64; cx++) {
      uint32_t slot = r3d_vs_phys(cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(cx, v->lv[0].gx);
      struct vscell *c = &r->vsl.cells[slot];
      struct vjob j = {cx, cy, z0, (uint32_t)(visible_end - z0), 0, 0};
      bool visible = c->cx == cx && c->cy == cy && c->za <= z0 && c->zb >= visible_end;
      nvisible += !visible;
      if (c->cx == cx && c->cy == cy) {
        if (c->za <= za_want && c->zb >= zb_want) continue; /* fully resident */
        int64_t s0 = z0, s1 = visible_end;
        if (!visible && c->za < visible_end && c->zb > z0) {
          /* Complete the visible interval first without rewriting a range the
           * renderer may currently sample. */
          if (c->za > z0) s1 = c->za < visible_end ? c->za : visible_end;
          else {
            s0 = c->zb > z0 ? c->zb : z0;
            s1 = visible_end;
          }
        } else if (visible) {
          /* Grow the margin in the direction of travel first, in bounded
           * publishes so the opposite side is not starved for long. */
          uint32_t margin_chunks = (v->z_margin + 15) / 16;
          if (zdir >= 0 && c->zb < zb_want) {
            s0 = c->zb;
            s1 = s0 + 16 < zb_want ? s0 + 16 : zb_want;
            int64_t filled = c->zb - visible_end;
            j.prio = 1 + (uint32_t)(filled > 0 ? filled / 16 : 0);
          } else if (zdir < 0 && c->za > za_want) {
            s1 = c->za;
            s0 = s1 - 16 > za_want ? s1 - 16 : za_want;
            int64_t filled = z0 - c->za;
            j.prio = 1 + (uint32_t)(filled > 0 ? filled / 16 : 0);
          } else if (c->za > za_want) {
            s1 = c->za;
            s0 = s1 - 16 > za_want ? s1 - 16 : za_want;
            int64_t filled = z0 - c->za;
            j.prio = 1 + margin_chunks + (uint32_t)(filled > 0 ? filled / 16 : 0);
          } else {
            s0 = c->zb;
            s1 = s0 + 16 < zb_want ? s0 + 16 : zb_want;
            int64_t filled = c->zb - visible_end;
            j.prio = 1 + margin_chunks + (uint32_t)(filled > 0 ? filled / 16 : 0);
          }
        }
        j.zs0 = s0;
        j.nz = (uint32_t)(s1 - s0);
      }
      int64_t dx = cx - ccx, dy = cy - ccy;
      j.d = (uint64_t)(dx * dx + dy * dy);
      jobs[nj++] = j;
    }
  /* selection sort is fine at <= 64 entries */
  for (uint32_t a = 0; a + 1 < nj; a++)
    for (uint32_t b = a + 1; b < nj; b++)
      if (jobs[b].prio < jobs[a].prio ||
          (jobs[b].prio == jobs[a].prio && jobs[b].d < jobs[a].d)) {
        struct vjob t = jobs[a];
        jobs[a] = jobs[b];
        jobs[b] = t;
      }
  /* hand the nearest jobs to the worker (skip cells it already holds) */
  uint32_t enq = 0;
  if (r->vsl.band) {
    /* group per-cell jobs sharing a z range into one window-wide band job,
     * capped at 16 slices (one dct3d chunk row) so publishes stay frequent */
    for (uint32_t k = 0; k < nj && enq < budget; k++) {
      if (!jobs[k].nz) continue; /* consumed by an earlier group */
      int64_t bx0 = jobs[k].cx, bx1 = jobs[k].cx, by0 = jobs[k].cy, by1 = jobs[k].cy;
      for (uint32_t m = k + 1; m < nj; m++)
        if (jobs[m].nz == jobs[k].nz && jobs[m].zs0 == jobs[k].zs0) {
          if (jobs[m].cx < bx0) bx0 = jobs[m].cx;
          if (jobs[m].cx > bx1) bx1 = jobs[m].cx;
          if (jobs[m].cy < by0) by0 = jobs[m].cy;
          if (jobs[m].cy > by1) by1 = jobs[m].cy;
          jobs[m].nz = 0;
        }
      uint32_t nz = jobs[k].nz > 16 ? 16 : jobs[k].nz;
      int fi = -1;
      bool dup = false;
      for (uint32_t i = 0; i < 4; i++) {
        if (r->vsl.q[i].st == 0) { if (fi < 0) fi = (int)i; }
        else if (r->vsl.q[i].zs0 == jobs[k].zs0 && r->vsl.q[i].cx <= bx1 &&
                 r->vsl.q[i].cx1 >= bx0 && r->vsl.q[i].cy <= by1 && r->vsl.q[i].cy1 >= by0)
          dup = true;
      }
      if (dup) continue;
      if (fi < 0) break;
      for (int64_t cy = by0; cy <= by1; cy++)
        for (int64_t cx = bx0; cx <= bx1; cx++) {
          uint32_t slot = r3d_vs_phys(cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(cx, v->lv[0].gx);
          struct vscell *c = &r->vsl.cells[slot];
          if (c->cx != cx || c->cy != cy)
            page[slot * 4] = 0; /* fresh tile: invalidate BEFORE the worker writes */
        }
      r->vsl.q[fi] = (struct vsjob){.cx = bx0,
                                    .cy = by0,
                                    .cx1 = bx1,
                                    .cy1 = by1,
                                    .zs0 = jobs[k].zs0,
                                    .nz = nz,
                                    .st = 1,
                                    .tl = r->timeline_value};
      enq++;
    }
  } else
  for (uint32_t k = 0; k < nj && enq < budget; k++) {
    int fi = -1;
    bool dup = false;
    for (uint32_t i = 0; i < 4; i++) {
      if (r->vsl.q[i].st == 0) { if (fi < 0) fi = (int)i; }
      else if (r->vsl.q[i].cx == jobs[k].cx && r->vsl.q[i].cy == jobs[k].cy) dup = true;
    }
    if (dup) continue;
    if (fi < 0) break;
    uint32_t slot =
        r3d_vs_phys(jobs[k].cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(jobs[k].cx, v->lv[0].gx);
    struct vscell *c = &r->vsl.cells[slot];
    if (c->cx != jobs[k].cx || c->cy != jobs[k].cy)
      page[slot * 4] = 0; /* fresh tile: invalidate BEFORE the worker writes */
    r->vsl.q[fi] = (struct vsjob){.cx = jobs[k].cx,
                                  .cy = jobs[k].cy,
                                  .cx1 = jobs[k].cx,
                                  .cy1 = jobs[k].cy,
                                  .zs0 = jobs[k].zs0,
                                  .nz = jobs[k].nz,
                                  .st = 1,
                                  .tl = r->timeline_value};
    enq++;
  }
  /* prefetch margin: with the window's own jobs all queued and capacity to
   * spare, pull in the cell ring one step beyond the window edge in the
   * direction of motion — pans then land on already-resident data. The
   * straddle column (+1 in each grid axis) hosts it; a span check keeps the
   * toroidal mapping from evicting a wanted in-window cell. */
  if (px0 >= 0 && (x0 != px0 || y0 != py0) && !r->vsl.band && !getenv("R3D_VSLAB_NOPREF")) {
    struct { int64_t cx, cy; } pf[48];
    uint32_t npf = 0;
    int64_t mxc = ((int64_t)v->nx - 1) / v->px, myc = ((int64_t)v->ny - 1) / v->px;
    if (x0 != px0) {
      int64_t cxc = x0 > px0 ? cx1 + 1 : cx0 - 1;
      if (cxc >= 0 && cxc <= mxc && cx1 - cx0 + 2 <= (int64_t)v->lv[0].gx)
        for (int64_t cy = cy0; cy <= cy1 && npf < 48; cy++)
          pf[npf++] = (typeof(pf[0])){cxc, cy};
    }
    if (y0 != py0) {
      int64_t cyc = y0 > py0 ? cy1 + 1 : cy0 - 1;
      if (cyc >= 0 && cyc <= myc && cy1 - cy0 + 2 <= (int64_t)v->lv[0].gy)
        for (int64_t cx = cx0; cx <= cx1 && npf < 48; cx++)
          pf[npf++] = (typeof(pf[0])){cx, cyc};
    }
    for (uint32_t k = 0; k < npf; k++) {
      uint32_t slot =
          r3d_vs_phys(pf[k].cy, v->lv[0].gy) * v->lv[0].gx + r3d_vs_phys(pf[k].cx, v->lv[0].gx);
      struct vscell *c = &r->vsl.cells[slot];
      if (c->cx == pf[k].cx && c->cy == pf[k].cy && c->za <= za_want &&
          c->zb >= zb_want)
        continue; /* already resident */
      int fi = -1;
      bool dup = false;
      for (uint32_t i = 0; i < 4; i++) {
        if (r->vsl.q[i].st == 0) { if (fi < 0) fi = (int)i; }
        else if (r->vsl.q[i].cx == pf[k].cx && r->vsl.q[i].cy == pf[k].cy) dup = true;
      }
      if (dup) continue;
      if (fi < 0) break;
      if (c->cx != pf[k].cx || c->cy != pf[k].cy) page[slot * 4] = 0;
      r->vsl.q[fi] = (struct vsjob){.cx = pf[k].cx,
                                    .cy = pf[k].cy,
                                    .cx1 = pf[k].cx,
                                    .cy1 = pf[k].cy,
                                    .zs0 = za_want,
                                    .nz = (uint32_t)(zb_want - za_want),
                                    .st = 1,
                                    .tl = r->timeline_value};
      enq++;
    }
  }
  if (enq) pthread_cond_broadcast(&r->vsl.cv);
  r->vsl.pending = nvisible;
  r->vsl.resident_pending = nj;
  pthread_mutex_unlock(&r->vsl.mu);

  /* params */
  p->slab_grid = v->lv[0].gx | (v->lv[0].gy << 6) | (1u << 24);
  uint32_t bm = 0;
  for (uint32_t l = 1; l < R3D_VS_LEVELS; l++)
    bm |= (v->lv[l].gx | (v->lv[l].gy << 4)) << ((l - 1) * 8);
  p->brick_mode = bm;
  p->slab_wz = v->wz;
  p->slab_z0 = (float)z0;
  p->slab_nx = (float)v->W;
  p->slab_ny = (float)v->H;
  p->slab_px = (float)v->px;
  p->slab_py = (float)v->px;
  p->slab_depth = v->D;
  p->slab_x0 = (float)x0;
  p->slab_y0 = (float)y0;
}

void r3d_vslab_get(const r3d_renderer *r, int64_t o[3], uint32_t *pending) {
  o[0] = r->vsl.v.x0;
  o[1] = r->vsl.v.y0;
  o[2] = r->vsl.v.z0;
  if (pending) *pending = r->vsl.pending;
}

uint32_t r3d_vslab_resident_pending(const r3d_renderer *r) {
  return r ? r->vsl.resident_pending : 0;
}

void r3d_vslab_prefetch(r3d_renderer *r, const int64_t *z0, uint32_t count) {
  if (!r || !r->vsl.pc.up) return;
  int64_t max = r3d_vs_max0(r->vsl.v.nz, r->vsl.v.D);
  int64_t wanted[R3D_VSLAB_PREFETCH_MAX];
  for (uint32_t i = 0; i < R3D_VSLAB_PREFETCH_MAX; i++) wanted[i] = -1;
  uint32_t n = 0;
  for (uint32_t i = 0; z0 && i < count && n < R3D_VSLAB_PREFETCH_MAX; i++) {
    if (z0[i] < 0 || z0[i] > max) continue;
    bool duplicate = false;
    for (uint32_t j = 0; j < n; j++) duplicate |= wanted[j] == z0[i];
    if (!duplicate) wanted[n++] = z0[i];
  }
  pthread_mutex_lock(&r->vsl.pc.mu);
  bool changed = false;
  for (uint32_t i = 0; i < R3D_VSLAB_PREFETCH_MAX; i++) {
    changed |= r->vsl.pc.wanted[i] != wanted[i];
    r->vsl.pc.wanted[i] = wanted[i];
  }
  if (changed) pthread_cond_broadcast(&r->vsl.pc.cv);
  pthread_mutex_unlock(&r->vsl.pc.mu);
}

void r3d_vslab_prefetch_get(r3d_renderer *r, r3d_vslab_prefetch_stats *st) {
  if (!st) return;
  memset(st, 0, sizeof *st);
  for (uint32_t i = 0; i < R3D_VSLAB_PREFETCH_MAX; i++) st->wanted[i] = -1;
  if (!r || !r->vsl.pc.up) return;
  pthread_mutex_lock(&r->vsl.pc.mu);
  st->capacity = r->vsl.pc.nslots;
  for (uint32_t i = 0; i < R3D_VSLAB_PREFETCH_MAX; i++)
    st->wanted[i] = r->vsl.pc.wanted[i];
  st->hits = r->vsl.pc.hits;
  st->misses = r->vsl.pc.misses;
  st->last_decode_ms = r->vsl.pc.last_decode_ms;
  for (uint32_t i = 0; i < r->vsl.pc.nslots; i++) {
    st->ready += r->vsl.pc.slot[i].state == 2;
    st->filling += r->vsl.pc.slot[i].state == 1;
  }
  pthread_mutex_unlock(&r->vsl.pc.mu);
}

int r3d_clip_begin(r3d_renderer *r, const char *band_dir, const char *pyramid_dir,
                   uint32_t band_z, uint32_t depth_max) {
  r->scene_gen++; /* GPU-visible data changes: pane cache must miss */
  if (!r->tiled_modes) {
    fprintf(stderr, "clip: device lacks native non-uniform descriptor indexing\n");
    return -1;
  }
  if (!r->samp_slab) { /* REPEAT-W ring sampler (shared with slab mode) */
    VkSamplerCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
    if (vkCreateSampler(r->vk.dev, &smci, NULL, &r->samp_slab) != VK_SUCCESS) return -1;
  }
  uint64_t z0 = (uint64_t)band_z * 1024 + 512 - depth_max / 2;
  if (r3d_vkclip_create(&r->clipm, &r->vk, r->fp_transition, r->fp_copy_mem, band_dir,
                        pyramid_dir, band_z, depth_max, 21504, 21504, z0) != 0)
    return -1;

  VkDescriptorImageInfo ii[R3D_SLAB_TILES];
  for (uint32_t e = 0; e < R3D_SLAB_TILES; e++) {
    uint32_t l = e < R3D_CLIP_LEVELS ? e : R3D_CLIP_LEVELS - 1;
    ii[e] = (VkDescriptorImageInfo){.sampler = r->samp_slab,
                                    .imageView = r3d_vkclip_view(r->clipm, l),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  }
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = r->dset,
      .dstBinding = 0,
      .descriptorCount = r->tile_descriptors,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = ii,
  };
  vkUpdateDescriptorSets(r->vk.dev, 1, &w, 0, NULL);
  return 0;
}

int r3d_clip_frame(r3d_renderer *r, double fx, double fy, uint64_t z0, r3d_frame_params *p) {
  if (!r->clipm) return -1;
  r3d_vkclip_update(r->clipm, (int64_t)fx, (int64_t)fy, z0);
  if (r3d_vkclip_pump(r->clipm, r->timeline, r->timeline_value) != 0) return -1;
  r3d_vkclip_params(r->clipm, p);
  return 0;
}

int r3d_frame(r3d_renderer *r, const r3d_frame_params *p, r3d_frame_stats *st) {
  return r3d_frame_views(r, p, 1, st);
}

/* Shift the surface-volume window's surviving content in place after an
 * integer-texel origin move: same-image strip copies along the moving axis
 * (strip length = |shift|, so each copy's src/dst are disjoint), ordered so
 * every strip is read before a later copy overwrites it (ascending for a
 * positive shift, descending for negative) with transfer-transfer barriers
 * between strips. new[p] = old[p + shift]. */
static void sv_shift_copies(r3d_renderer *r, VkCommandBuffer cmd) {
  int32_t su = r->sv.sh_u, sv2 = r->sv.sh_v, sz = r->sv.sh_z;
  uint32_t W = r->sv.W, H = r->sv.H, L = r->sv.L;
  uint32_t dx0 = su < 0 ? (uint32_t)-su : 0, dx1 = su > 0 ? W - (uint32_t)su : W;
  uint32_t dy0 = sv2 < 0 ? (uint32_t)-sv2 : 0, dy1 = sv2 > 0 ? H - (uint32_t)sv2 : H;
  uint32_t dz0 = sz < 0 ? (uint32_t)-sz : 0, dz1 = sz > 0 ? L - (uint32_t)sz : L;
  int axis = su ? 0 : sv2 ? 1 : 2;
  int32_t s = axis == 0 ? su : axis == 1 ? sv2 : sz;
  uint32_t d = (uint32_t)(s > 0 ? s : -s);
  uint32_t a0 = axis == 0 ? dx0 : axis == 1 ? dy0 : dz0;
  uint32_t a1 = axis == 0 ? dx1 : axis == 1 ? dy1 : dz1;
  if (a1 <= a0) return;
  uint32_t nstrips = (a1 - a0 + d - 1) / d;
  VkImageSubresourceLayers sub = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  for (uint32_t k = 0; k < nstrips; k++) {
    uint32_t idx = s > 0 ? k : nstrips - 1 - k;
    uint32_t b0 = a0 + idx * d;
    uint32_t bl = b0 + d > a1 ? a1 - b0 : d;
    int32_t dst[3] = {(int32_t)dx0, (int32_t)dy0, (int32_t)dz0};
    uint32_t ext[3] = {dx1 - dx0, dy1 - dy0, dz1 - dz0};
    dst[axis] = (int32_t)b0;
    ext[axis] = bl;
    VkImageCopy c = {
        .srcSubresource = sub,
        .srcOffset = {dst[0] + su, dst[1] + sv2, dst[2] + sz},
        .dstSubresource = sub,
        .dstOffset = {dst[0], dst[1], dst[2]},
        .extent = {ext[0], ext[1], ext[2]},
    };
    if (k)
      r3d_vk_image_barrier(cmd, r->sv.vol.img, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT, 0,
                           1);
    vkCmdCopyImage(cmd, r->sv.vol.img, VK_IMAGE_LAYOUT_GENERAL, r->sv.vol.img,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &c);
  }
}

int r3d_frame_views(r3d_renderer *r, const r3d_frame_params *views, uint32_t nviews,
                    r3d_frame_stats *st) {
  if (!nviews || nviews > R3D_MAX_VIEWS) return -1;
  const r3d_frame_params *p = &views[0];
  if (st) memset(st, 0, sizeof *st);
  uint32_t slot = r->slot;
  uint64_t tp = now_ns();

  /* pace: wait for this slot's previous submission, then read its timestamps */
  if (r->slot_value[slot]) {
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &r->timeline,
        .pValues = &r->slot_value[slot],
    };
    if (vkWaitSemaphores(r->vk.dev, &wi, UINT64_MAX) != VK_SUCCESS) return -1;
    if (r->slot_has_query[slot]) {
      uint64_t ts[4] = {0};
      if (vkGetQueryPoolResults(r->vk.dev, r->query, slot * 4, 4, sizeof ts, ts, sizeof ts[0],
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        double period = r->vk.caps.ts_period_ns;
        r->slot_gpu[slot][0] = (uint64_t)((double)(ts[3] - ts[0]) * period);
        r->slot_gpu[slot][1] = (uint64_t)((double)(ts[1] - ts[0]) * period);
        r->slot_gpu[slot][2] = (uint64_t)((double)(ts[2] - ts[1]) * period);
        r->slot_gpu[slot][3] = (uint64_t)((double)(ts[3] - ts[2]) * period);
      }
    }
  }
  if (st) {
    st->gpu_ns = r->slot_gpu[slot][0];
    st->gpu_raycast_ns = r->slot_gpu[slot][1];
    st->gpu_blit_ns = r->slot_gpu[slot][2];
    st->gpu_gui_ns = r->slot_gpu[slot][3];
    st->cpu_wait_ns = now_ns() - tp;
  }

  tp = now_ns();
  if (r->pres.up) { /* a deferred present may have asked for a resize */
    pthread_mutex_lock(&r->pres.mu);
    bool nr = r->pres.need_resize, failed = r->pres.failed;
    r->pres.need_resize = false;
    pthread_mutex_unlock(&r->pres.mu);
    if (failed) return -1;
    if (nr) {
      if (r->gui_open) {
        r3d_vkgui_discard();
        r->gui_open = false;
      }
      int rc = r3d_resize(r);
      return rc == 0 ? 1 : rc;
    }
  }
  uint32_t img = 0;
  bool headless = r->cfg.headless;
  VkResult ar = headless ? VK_SUCCESS
                         : vkAcquireNextImageKHR(r->vk.dev, r->swap.swapchain, UINT64_MAX,
                                                 r->acquire[slot], VK_NULL_HANDLE, &img);
  if (st) st->cpu_acquire_ns = now_ns() - tp;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    if (r->gui_open) {
      r3d_vkgui_discard();
      r->gui_open = false;
    }
    int rc = r3d_resize(r);
    return rc == 0 ? 1 : rc; /* 1 = frame skipped, retry next loop */
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) return -1;

  tp = now_ns();
  VkCommandBuffer cmd = r->cmd[slot];
  for (uint32_t v = 0; v < nviews; v++)
    memcpy((uint8_t *)r->frame_ubo.mapped +
               (size_t)(slot * R3D_MAX_VIEWS + v) * r->frame_ubo_stride,
           &views[v], sizeof views[v]);
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cmd, &bi);

  if (r->query) {
    vkCmdResetQueryPool(cmd, r->query, slot * 4, 4);
    /* ts0 here (not after the bake) so "raycast" GPU time includes the
     * surfvol bake bands - previously they were invisible to the profile */
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, r->query, slot * 4);
  }
  bool sv_baked_now = false;

  if (r->sv.active && r->sv.step > 0.0f && !r->sv.dirty && !r->sv.shift_pending &&
      r->sv.prog_row == UINT32_MAX && r->sv.baked) {
    /* lazy layer refresh: only bake the band the view can currently see */
    uint32_t vz0 = r->sv.vz0 > 8u ? r->sv.vz0 - 8u : 0;
    uint32_t vz1 = r->sv.vz1 + 8u < r->sv.L ? r->sv.vz1 + 8u : r->sv.L;
    if (vz1 <= vz0) {
      vz0 = 0;
      vz1 = r->sv.L;
    }
    if (sv_layers_stale(r, vz0, vz1)) {
      r->sv.pass_z0 = vz0;
      r->sv.pass_z1 = vz1;
      r->sv.prog_row = 0;
    }
  }

  if (r->sv.active && r->sv.step > 0.0f &&
      (r->sv.dirty || r->sv.prog_row != UINT32_MAX || r->sv.shift_pending)) {
    /* update the flattened surface volume before this frame samples it:
     * full window when the mapping changed shape (dirty); shift-in-place +
     * exposed-band bakes for integer-texel window moves; else the next row
     * band of a progressive residency-arrival re-bake (~SV_PROG_ROWS
     * rows/frame keeps the extra GPU work under a vsync interval; the full
     * 2048^2x96 dispatch was a ~75 ms frame hitch). Serialize against the
     * previous frame's sampling reads, do the work, then make the writes
     * visible to this frame's raycast. */
    struct svband { uint32_t x0, y0, z0, w, h, l; } bands[2];
    uint32_t nband = 0;
    bool shifted = false;
    if (r->sv.dirty) {
      r->sv.dirty = false;
      r->sv.baked = true;
      /* Bake the view-visible sub-box now (with a margin) and refresh the
       * rest progressively — a zoom's pitch change stops costing the
       * whole-window dispatch. Outside the box the content keeps the OLD
       * mapping until the progressive pass reaches it; that's only visible
       * if the user pans within the ~quarter second the pass takes. */
      uint32_t m = 64;
      uint32_t x0 = r->sv.vx0 > m ? r->sv.vx0 - m : 0;
      uint32_t y0 = r->sv.vy0 > m ? r->sv.vy0 - m : 0;
      uint32_t z0 = r->sv.vz0 > 8u ? r->sv.vz0 - 8u : 0;
      uint32_t x1 = r->sv.vx1 + m < r->sv.W ? r->sv.vx1 + m : r->sv.W;
      uint32_t y1 = r->sv.vy1 + m < r->sv.H ? r->sv.vy1 + m : r->sv.H;
      uint32_t z1 = r->sv.vz1 + 8u < r->sv.L ? r->sv.vz1 + 8u : r->sv.L;
      if (x1 > x0 && y1 > y0 && z1 > z0 &&
          ((size_t)(x1 - x0) * (y1 - y0) * (z1 - z0)) * 2u <
              (size_t)r->sv.W * r->sv.H * r->sv.L) {
        bands[nband++] = (struct svband){x0, y0, z0, x1 - x0, y1 - y0, z1 - z0};
        /* the rest of the window still holds the old mapping: every layer is
         * stale; the lazy pass bakes the visible band first, others on scrub */
        memset(r->sv.stale, 0xff, sizeof r->sv.stale);
        r->sv.prog_row = UINT32_MAX;
      } else { /* no useful visibility hint: the whole window in one go */
        bands[nband++] = (struct svband){0, 0, 0, r->sv.W, r->sv.H, r->sv.L};
        memset(r->sv.stale, 0, sizeof r->sv.stale);
        r->sv.prog_row = UINT32_MAX;
      }
    } else if (r->sv.shift_pending) {
      r->sv.shift_pending = false;
      shifted = true;
      if (r->sv.prog_row != UINT32_MAX)
        r->sv.prog_row = 0; /* restart: rows moved under the cursor */
      int32_t su = r->sv.sh_u, sv2 = r->sv.sh_v, sz = r->sv.sh_z;
      if (su)
        bands[nband++] = (struct svband){su > 0 ? r->sv.W - (uint32_t)su : 0, 0, 0,
                                         (uint32_t)labs(su), r->sv.H, r->sv.L};
      if (sv2)
        bands[nband++] = (struct svband){0, sv2 > 0 ? r->sv.H - (uint32_t)sv2 : 0, 0,
                                         r->sv.W, (uint32_t)labs(sv2), r->sv.L};
      if (sz)
        bands[nband++] = (struct svband){0, 0, sz > 0 ? r->sv.L - (uint32_t)sz : 0,
                                         r->sv.W, r->sv.H, (uint32_t)labs(sz)};
    } else {
      uint32_t y0 = r->sv.prog_row, rows = r->sv.H - y0;
      /* keep the per-frame texel budget of the old full-depth band */
      uint32_t lz = r->sv.pass_z1 > r->sv.pass_z0 ? r->sv.pass_z1 - r->sv.pass_z0 : r->sv.L;
      uint32_t budget_rows = SV_PROG_ROWS * (r->sv.L / (lz ? lz : 1u));
      if (budget_rows < SV_PROG_ROWS) budget_rows = SV_PROG_ROWS;
      if (rows > budget_rows) rows = budget_rows;
      r->sv.prog_row = y0 + rows >= r->sv.H ? UINT32_MAX : y0 + rows;
      if (r->sv.prog_row == UINT32_MAX) {
        if (r->sv.rebake_again) {
          r->sv.rebake_again = false; /* residency landed mid-pass: the band is
                                       * still stale; the frame loop restarts */
        } else {
          sv_layers_clear(r, r->sv.pass_z0, r->sv.pass_z1);
        }
      }
      bands[nband++] = (struct svband){0, y0, r->sv.pass_z0, r->sv.W, rows, lz};
    }
    r3d_vk_image_barrier(cmd, r->sv.vol.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                             VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         0, 1);
    if (shifted) sv_shift_copies(r, cmd);
    uint32_t lodq = 0;
    for (float s = r->sv.step; s >= 2.0f && lodq + 1u < r->bricks_nlev; s *= 0.5f) lodq++;
    sv_push push = {.u0 = r->sv.u0,
                    .v0 = r->sv.v0,
                    .step = r->sv.step,
                    .zoff0 = r->sv.zoff0,
                    .sx = r->sv.sx,
                    .sy = r->sv.sy,
                    .W = r->sv.W,
                    .H = r->sv.H,
                    .L = r->sv.L,
                    .nback = r->sv.nback,
                    .abpa = r->bricks_abpa,
                    .lod = lodq,
                    .lstep = 1.0f, /* depth stays native-res regardless of xy zoom */
                    .use_ink = r->ink_active ? 1u : 0u,
                    .use_pred = r->sv.pred_on ? 1u : 0u,
                    .pg0u = r->sv.pred_g0u,
                    .pg0v = r->sv.pred_g0v,
                    .pppg = r->sv.pred_ppg};
    for (uint32_t b = 0; b < nband; b++) {
      push.x0 = bands[b].x0;
      push.y0 = bands[b].y0;
      push.z0 = bands[b].z0;
      r3d_vkcomp_dispatch(cmd, &r->sv.comp, &push, sizeof push, (bands[b].w + 7) / 8,
                          (bands[b].h + 7) / 8, (bands[b].l + 3) / 4);
    }
    sv_baked_now = nband > 0 || shifted;
    r3d_vk_image_barrier(cmd, r->sv.vol.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, 1);
  }

  /* offscreen -> GENERAL for compute write. Multi-view layouts may leave
   * uncovered regions (side panel strip), so clear the image first there;
   * single view overwrites every pixel. */
  /* A single view anchored at the origin is the classic path: render into
   * the top-left region, blit upscales (adaptive resolution). Any view with
   * a nonzero origin (multiview — including a SOLO pane beside the panel)
   * must render in place and blit the whole offscreen 1:1, or the blit
   * would stretch the pane across the window and shear it off the ImGui
   * overlays. */
  bool region_upscale = nviews == 1 && views[0].view_org == 0;
  /* Pane cache is only meaningful for in-place multiview rendering; the
   * single-view upscale path renders every pixel every frame anyway. The
   * offscreen keeps its contents between frames (layout tracked in
   * os_layout, never discarded via UNDEFINED once written). */
  bool cache_panes = !region_upscale && !r->pane_cache_off;
  uint64_t lay_key = 1469598103934665603ull ^ nviews;
  for (uint32_t v = 0; v < nviews; v++) {
    lay_key = (lay_key ^ views[v].view_org) * 1099511628211ull;
    lay_key = (lay_key ^ views[v].viewport[0]) * 1099511628211ull;
    lay_key = (lay_key ^ views[v].viewport[1]) * 1099511628211ull;
  }
  lay_key = (lay_key ^ r->offscreen.extent.width) * 1099511628211ull;
  lay_key = (lay_key ^ r->offscreen.extent.height) * 1099511628211ull;
  bool need_clear = !region_upscale && (lay_key != r->os_layout_key || !cache_panes);
  if (need_clear) {
    /* uncovered regions (panel strip, gutters) only change with the layout:
     * clear once per layout change instead of every frame (on Dozen a full
     * clear was a CPU fill of the whole 8 MB image into a fresh D3D12
     * upload resource - ~2-3 ms of render-thread time per frame) */
    r3d_vk_image_barrier(cmd, r->offscreen.img, r->os_layout,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                         VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);
    VkClearColorValue cc = {.float32 = {0.05f, 0.05f, 0.06f, 1.0f}};
    VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, r->offscreen.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1,
                         &rr);
    r3d_vk_image_barrier(cmd, r->offscreen.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, 0, 1);
    r->os_layout_key = lay_key;
    memset(r->pane_key, 0, sizeof r->pane_key); /* every pane re-renders */
  } else {
    /* single view: contents fully overwritten (discard is fine); multiview
     * cached: keep the previous pixels, just move to GENERAL for writes */
    r3d_vk_image_barrier(cmd, r->offscreen.img,
                         cache_panes ? r->os_layout : VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, 0, 1);
  }
  r->os_layout = VK_IMAGE_LAYOUT_GENERAL;
  if (sv_baked_now) r->scene_gen++; /* the surf view samples fresh texels */
  uint32_t panes_drawn = 0;
  /* p->viewport may be smaller than the drawable (adaptive resolution while
   * the camera moves, single-view only): render into the top-left region,
   * blit upscales. Multi-view renders each view into its view_org rect and
   * blits the whole offscreen 1:1. */
  uint32_t rw = p->viewport[0] ? p->viewport[0] : r->swap.extent.width;
  uint32_t rh = p->viewport[1] ? p->viewport[1] : r->swap.extent.height;
  if (rw > r->offscreen.extent.width) rw = r->offscreen.extent.width;
  if (rh > r->offscreen.extent.height) rh = r->offscreen.extent.height;
  VkPipeline bound = VK_NULL_HANDLE;
  for (uint32_t v = 0; v < nviews; v++) {
    const r3d_frame_params *vp = &views[v];
    uint32_t rmode =
        vp->view_flags & R3D_VIEW_SURF
            ? 5u
            : (vp->slab_grid & (1u << 24)
                   ? 4u
                   : (vp->clip_valid ? 2u : (vp->brick_mode ? 3u : (vp->slab_grid ? 1u : 0u))));
    uint32_t vw = vp->viewport[0] ? vp->viewport[0] : r->swap.extent.width;
    uint32_t vh = vp->viewport[1] ? vp->viewport[1] : r->swap.extent.height;
    uint32_t ox = vp->view_org & 0xffffu, oy = vp->view_org >> 16;
    if (ox >= r->offscreen.extent.width || oy >= r->offscreen.extent.height) continue;
    if (vw > r->offscreen.extent.width - ox) vw = r->offscreen.extent.width - ox;
    if (vh > r->offscreen.extent.height - oy) vh = r->offscreen.extent.height - oy;
    /* R3D_WG only builds full-quality cube variants; fast quality keeps the
     * shader's default 16x8 local size. On the X1-85, 8x8 is 18.7% faster for
     * the divergent reduced-resolution orbit path, but loses on dense static
     * views, so use it only while adaptive resolution is actually reduced. */
    VkPipeline pipeline = r->raycast[r->quality][rmode];
    uint32_t wgx = rmode == 0 && r->quality == R3D_QUALITY_FULL ? r->wg_x : 16u;
    uint32_t wgy = rmode == 0 && r->quality == R3D_QUALITY_FULL ? r->wg_y : 8u;
    if (nviews == 1 && rmode == 0 && r->quality == R3D_QUALITY_FULL && r->adaptive_wg &&
        (vw < r->swap.extent.width || vh < r->swap.extent.height)) {
      pipeline = r->raycast_cube_8x8;
      wgx = wgy = 8;
    }
    if (cache_panes) {
      /* input hash: every param byte except frame_index (per-frame jitter
       * seed - a cached pane keeps its last jitter, which is exactly a
       * static image), plus the GPU-visible scene generation and pipeline */
      r3d_frame_params hp = *vp;
      hp.frame_index = 0;
      const uint8_t *hb = (const uint8_t *)&hp;
      uint64_t k = 1469598103934665603ull;
      for (size_t i = 0; i < sizeof hp; i++) k = (k ^ hb[i]) * 1099511628211ull;
      k = (k ^ r->scene_gen) * 1099511628211ull;
      k = (k ^ (uint64_t)(uintptr_t)pipeline) * 1099511628211ull;
      k = (k ^ ((uint64_t)wgx << 32 | wgy)) * 1099511628211ull;
      if (k == 0) k = 1;
      if (r->pane_key[v] == k) continue; /* unchanged: keep the pixels */
      r->pane_key[v] = k;
    }
    panes_drawn++;
    if (pipeline != bound) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      bound = pipeline;
    }
    uint32_t ubo_offset = (uint32_t)((slot * R3D_MAX_VIEWS + v) * r->frame_ubo_stride);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe_layout, 0, 1, &r->dset,
                            1, &ubo_offset);
    vkCmdDispatch(cmd, (vw + wgx - 1) / wgx, (vh + wgy - 1) / wgy, 1);
  }
  if (st) st->panes_drawn = panes_drawn;
  if (!region_upscale) { /* views render in place; blit is identity */
    rw = r->offscreen.extent.width;
    rh = r->offscreen.extent.height;
  }
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, r->query, slot * 4 + 1);
  r->slot_has_query[slot] = r->query != VK_NULL_HANDLE;

  /* offscreen -> blit src; swapchain image -> blit dst */
  r3d_vk_image_barrier(cmd, r->offscreen.img, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT, 0, 1);
  r->os_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; /* where the frame leaves it */
  if (headless) {
    /* no swapchain: the frame ends here. ImGui's frame is closed without a
     * draw so the panel logic still ran; timestamps stay well-formed. */
    if (r->query) {
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, r->query, slot * 4 + 2);
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r->query, slot * 4 + 3);
    }
    if (r->gui_open) {
      r3d_vkgui_discard();
      r->gui_open = false;
    }
    vkEndCommandBuffer(cmd);
    if (st) st->cpu_record_ns = now_ns() - tp;
    tp = now_ns();
    uint64_t hv = ++r->timeline_value;
    VkSemaphoreSubmitInfo hsig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                  .semaphore = r->timeline,
                                  .value = hv,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo hcsi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                      .commandBuffer = cmd};
    VkSubmitInfo2 hsi = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                         .commandBufferInfoCount = 1,
                         .pCommandBufferInfos = &hcsi,
                         .signalSemaphoreInfoCount = 1,
                         .pSignalSemaphoreInfos = &hsig};
    if (r3d_vkctx_queue_submit2(&r->vk, 1, &hsi, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
    r->slot_value[slot] = hv;
    if (st) st->cpu_submit_ns = now_ns() - tp;
    r->slot = (slot + 1) % FRAMES_IN_FLIGHT;
    return 0;
  }
  r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 1);

  VkImageBlit2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {(int32_t)rw, (int32_t)rh, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0},
                     {(int32_t)r->swap.extent.width, (int32_t)r->swap.extent.height, 1}},
  };
  VkBlitImageInfo2 blit = {
      .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage = r->offscreen.img,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage = r->swap.images[img],
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &region,
      /* LINEAR: identity when rendering at full size, smooth when upscaling */
      .filter = VK_FILTER_LINEAR,
  };
  vkCmdBlitImage2(cmd, &blit);
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, r->query, slot * 4 + 2);

  if (r->gui_open) {
    r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         0, 1);
    r3d_vkgui_render(cmd, r->swap.views[img], r->swap.extent);
    r->gui_open = false;
    r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, 0, 1);
  } else {
    r3d_vk_image_barrier(cmd, r->swap.images[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                         0, 1);
  }
  if (r->query)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r->query, slot * 4 + 3);
  vkEndCommandBuffer(cmd);
  if (st) st->cpu_record_ns = now_ns() - tp;

  tp = now_ns();
  uint64_t signal_value = ++r->timeline_value;
  VkSemaphoreSubmitInfo waits[] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = r->acquire[slot],
       .stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT},
  };
  VkSemaphoreSubmitInfo signals[] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = r->swap.render_done[img],
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = r->timeline,
       .value = signal_value,
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
  };
  VkCommandBufferSubmitInfo csi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                   .commandBuffer = cmd};
  VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = waits,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &csi,
      .signalSemaphoreInfoCount = 2,
      .pSignalSemaphoreInfos = signals,
  };
  if (r3d_vkctx_queue_submit2(&r->vk, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
  r->slot_value[slot] = signal_value;

  if (r->pres.up) {
    /* strictly one present outstanding: wait for the previous one (usually
     * long done — it overlapped this frame's poll/GUI/record), then queue */
    pthread_mutex_lock(&r->pres.mu);
    while (r->pres.pending || r->pres.busy) pthread_cond_wait(&r->pres.done_cv, &r->pres.mu);
    bool failed = r->pres.failed;
    r->pres.img = img;
    r->pres.pending = true;
    pthread_cond_broadcast(&r->pres.cv);
    pthread_mutex_unlock(&r->pres.mu);
    if (failed) return -1;
  } else {
    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &r->swap.render_done[img],
        .swapchainCount = 1,
        .pSwapchains = &r->swap.swapchain,
        .pImageIndices = &img,
    };
    VkResult pr = r3d_vkctx_queue_present(&r->vk, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
      if (r3d_resize(r) != 0) return -1;
    } else if (pr != VK_SUCCESS) {
      return -1;
    }
  }

  if (st) st->cpu_submit_ns = now_ns() - tp;
  r->slot = (slot + 1) % FRAMES_IN_FLIGHT;
  return 0;
}

int r3d_read_frame(r3d_renderer *r, uint8_t *rgba, uint32_t *w, uint32_t *h) {
  *w = r->offscreen.extent.width;
  *h = r->offscreen.extent.height;
  if (!rgba) return 0;

  VkDeviceSize need = (VkDeviceSize)*w * *h * 4;
  if (r->readback.size < need) {
    r3d_vkbuf_destroy(&r->vk, &r->readback);
    if (r3d_vkbuf_create_host(&r->vk, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &r->readback) != 0)
      return -1;
  }
  r3d_vkctx_device_wait_idle(&r->vk); /* screenshot path; simplicity over speed */

  VkCommandBuffer cmd = r3d_vk_oneshot_begin(&r->vk, r->pool);
  if (!cmd) return -1;
  /* offscreen was left in TRANSFER_SRC by the last frame */
  VkBufferImageCopy2 region = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {*w, *h, 1},
  };
  VkCopyImageToBufferInfo2 ci = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
      .srcImage = r->offscreen.img,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstBuffer = r->readback.buf,
      .regionCount = 1,
      .pRegions = &region,
  };
  vkCmdCopyImageToBuffer2(cmd, &ci);
  if (r3d_vk_oneshot_end(&r->vk, r->pool, cmd) != 0) return -1;
  memcpy(rgba, r->readback.mapped, need);
  return 0;
}
