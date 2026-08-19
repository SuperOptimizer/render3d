#include "vk/vkclip.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/clip.h"
#include "shard/shardio.h"
#include "vk/vkres.h"

#define QCAP 512u
#define TEX R3D_CLIP_TEX /* 2048 */
#define SLOTS 64u        /* fixed per-level ring storage; wzl must fit */
#define BAND_SLICES 1024u
#define VOL_NX 43008ull
#define VOL_NY 43008ull
#define VOL_NZ 68608ull

typedef struct clip_job {
  uint32_t level;
  uint64_t ls;   /* level slice (absolute: world_z / 2^level) */
  uint32_t gen;  /* level generation at enqueue; stale jobs are dropped */
  uint32_t slot; /* ring slot the job owns while producing */
} clip_job;

/* Staging ownership. Exactly one party may touch a region's bytes:
 *   FREE      nobody; main may hand it to the worker
 *   QUEUED    a job exists; the worker has not claimed the region yet
 *   PRODUCING the worker owns the bytes (main must not retarget the region)
 *   READY     a complete payload waits for the pump
 *   CONSUMING the pump owns the bytes (the worker must not overwrite them)
 * Every transition except PRODUCING->READY happens on the main thread; all of
 * them happen under r3d_vkclip::mu. `gpu`/`gpu_gen` are main-thread only. */
typedef enum clip_slot_state {
  CLIP_FREE = 0,
  CLIP_QUEUED,
  CLIP_PRODUCING,
  CLIP_READY,
  CLIP_CONSUMING,
} clip_slot_state;

typedef struct clip_slot {
  int64_t req;       /* ls requested into this slot, -1 none */
  int64_t done;      /* ls staged in this slot, -1 none */
  int64_t gpu;       /* ls in the texture layer, -1 none */
  uint32_t req_gen;  /* generation the request belongs to */
  uint32_t done_gen; /* generation the staged bytes belong to */
  uint32_t gpu_gen;  /* generation the texture layer belongs to */
  clip_slot_state state;
} clip_slot;

typedef struct clip_level_state {
  uint8_t *stage;         /* wzl regions x TEX^2 (immutable after create) */
  clip_slot slot[SLOTS];
  uint32_t gen;           /* bumped on XY recenter and on a Z ring discontinuity */
  uint64_t ls0;           /* last requested window start, level slices (main only) */
  bool ls0_set;
} clip_level_state;

struct r3d_vkclip {
  r3d_vkctx *vk;
  PFN_vkTransitionImageLayoutEXT fp_transition;
  PFN_vkCopyMemoryToImageEXT fp_copy_mem;
  VkCommandPool upload_pool; /* portable staged-upload fallback */
  r3d_vkbuf upload_stage;    /* worker-independent reusable fallback buffer */

  r3d_clip clip;
  r3d_vkimage tex[R3D_CLIP_LEVELS];
  clip_level_state st[R3D_CLIP_LEVELS];

  r3d_shard_store store;
  uint32_t band_z; /* shard row; band world z = band_z*1024 .. +1024 */
  struct {
    uint8_t *map;
    size_t n;
    uint64_t lx, ly, lz;
  } pyr[R3D_CLIP_LEVELS]; /* levels 2..5 */

  uint64_t z0; /* current window start (world slices) */

  /* job queue */
  clip_job q[QCAP];
  uint32_t q_head, q_tail; /* main pushes at tail, worker pops at head */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  bool mu_up;
  bool quit;
  bool worker_up;
  bool trace;
  pthread_t worker;
  uint8_t *scratch; /* L1 source region: 2*TEX x 2*TEX x 2 voxels */
};

/* ---------- worker ---------- */

static void produce_from_pyramid(r3d_vkclip *cl, uint32_t l, uint64_t ls, uint8_t *dst) {
  const r3d_clip_level *lv = &cl->clip.lv[l];
  uint64_t lx = cl->pyr[l].lx, ly = cl->pyr[l].ly, lz = cl->pyr[l].lz;
  int64_t zp = (int64_t)ls - (int64_t)(((uint64_t)cl->band_z * BAND_SLICES) >> l);
  if (!cl->pyr[l].map || !lx || !ly || zp < 0 || zp >= (int64_t)lz) {
    memset(dst, 0, (size_t)TEX * TEX);
    return;
  }
  int64_t vx0 = lv->ox / lv->s - 1, vy0 = lv->oy / lv->s - 1; /* texel 0 in level voxels */
  const uint8_t *zbase = cl->pyr[l].map + (size_t)zp * ly * lx;
  for (uint32_t t = 0; t < TEX; t++) {
    int64_t vy = vy0 + t;
    if (vy < 0) vy = 0;
    if (vy >= (int64_t)ly) vy = (int64_t)ly - 1;
    const uint8_t *srow = zbase + (size_t)vy * lx;
    uint8_t *drow = dst + (size_t)t * TEX;
    /* lo/hi bound the in-volume span of this row; both are clamped into
     * [0,TEX]/[-1,TEX-1] so a fully out-of-volume row degenerates to pure
     * edge duplication instead of walking off either end of drow */
    int64_t lo = vx0 < 0 ? -vx0 : 0;
    if (lo > (int64_t)TEX) lo = (int64_t)TEX;
    int64_t hi = (int64_t)TEX - 1;
    if (vx0 + hi >= (int64_t)lx) hi = (int64_t)lx - 1 - vx0;
    if (hi >= (int64_t)TEX) hi = (int64_t)TEX - 1;
    if (hi < lo - 1) hi = lo - 1;
    if (hi >= lo) memcpy(drow + lo, srow + vx0 + lo, (size_t)(hi - lo + 1));
    uint8_t left = lo < (int64_t)TEX && hi >= lo ? drow[lo] : srow[0];
    uint8_t right = hi >= lo ? drow[hi] : srow[lx - 1];
    for (int64_t c = 0; c < lo; c++) drow[c] = left;
    for (int64_t c = hi + 1; c < (int64_t)TEX; c++) drow[c] = right;
  }
}

/* Decode a region whose origin may sit outside the volume. The read is
 * intersected with [0,nx)x[0,ny) and the clamped border texel is duplicated
 * outward (same apron rule as the pyramid path) — casting a negative origin to
 * uint64_t instead would land the whole read out of range and silently return
 * a zero-filled, "valid" slice. dst holds dz*h*w bytes; the clamped read lands
 * at the front of dst and is expanded in place: every destination offset is
 * >= its source offset and successive destinations are >= w apart, so a
 * descending slice/row walk never clobbers unread data. */
static int decode_region_clamped(r3d_vkclip *cl, uint64_t z0, uint32_t dz, int64_t sy,
                                 int64_t sx, uint32_t h, uint32_t w, uint8_t *dst) {
  const int64_t nx = (int64_t)cl->clip.nx, ny = (int64_t)cl->clip.ny;
  int64_t dx0 = sx < 0 ? -sx : 0, dy0 = sy < 0 ? -sy : 0;
  if (dx0 > (int64_t)w) dx0 = (int64_t)w;
  if (dy0 > (int64_t)h) dy0 = (int64_t)h;
  int64_t rx = sx + dx0, ry = sy + dy0;
  int64_t rw = (int64_t)w - dx0, rh = (int64_t)h - dy0;
  if (rx >= nx) rw = 0;
  else if (rx + rw > nx) rw = nx - rx;
  if (ry >= ny) rh = 0;
  else if (ry + rh > ny) rh = ny - ry;
  if (rw <= 0 || rh <= 0) { /* nothing of the region is inside the volume */
    memset(dst, 0, (size_t)dz * h * w);
    return 0;
  }
  if (r3d_shard_decode_region(&cl->store, z0, (uint64_t)ry, (uint64_t)rx, dz, (uint32_t)rh,
                              (uint32_t)rw, dst, 0) != 0)
    return -1;
  for (int64_t k = (int64_t)dz - 1; k >= 0; k--) {
    uint8_t *dsl = dst + (size_t)k * h * w;
    const uint8_t *ssl = dst + (size_t)k * (size_t)rh * (size_t)rw;
    for (int64_t r = rh - 1; r >= 0; r--)
      memmove(dsl + (size_t)(dy0 + r) * w + (size_t)dx0, ssl + (size_t)r * (size_t)rw,
              (size_t)rw);
    for (int64_t r = dy0; r < dy0 + rh; r++) { /* left/right apron */
      uint8_t *row = dsl + (size_t)r * w;
      memset(row, row[dx0], (size_t)dx0);
      memset(row + dx0 + rw, row[dx0 + rw - 1], (size_t)((int64_t)w - dx0 - rw));
    }
    const uint8_t *top = dsl + (size_t)dy0 * w;
    for (int64_t r = 0; r < dy0; r++) memcpy(dsl + (size_t)r * w, top, w);
    const uint8_t *bot = dsl + (size_t)(dy0 + rh - 1) * w;
    for (int64_t r = dy0 + rh; r < (int64_t)h; r++) memcpy(dsl + (size_t)r * w, bot, w);
  }
  return 0;
}

static int produce_slice(r3d_vkclip *cl, uint32_t l, uint64_t ls, uint8_t *dst) {
  const r3d_clip_level *lv = &cl->clip.lv[l];
  if (l == 0) {
    return decode_region_clamped(cl, ls, 1, lv->oy - 1, lv->ox - 1, TEX, TEX, dst);
  } else if (l == 1) {
    /* 2^3 box from full-res decode of the covered region */
    uint32_t W = 2 * TEX;
    if (decode_region_clamped(cl, ls * 2, 2, lv->oy - 2, lv->ox - 2, W, W, cl->scratch) != 0)
      return -1;
    for (uint32_t y = 0; y < TEX; y++) {
      const uint8_t *r0 = cl->scratch + (size_t)(2 * y) * W;
      const uint8_t *r1 = r0 + W;
      const uint8_t *r2 = cl->scratch + (size_t)W * W + (size_t)(2 * y) * W;
      const uint8_t *r3 = r2 + W;
      uint8_t *o = dst + (size_t)y * TEX;
      for (uint32_t x = 0; x < TEX; x++) {
        uint32_t v = r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1] + r2[2 * x] +
                     r2[2 * x + 1] + r3[2 * x] + r3[2 * x + 1];
        o[x] = (uint8_t)(v / 8);
      }
    }
  } else {
    produce_from_pyramid(cl, l, ls, dst);
  }
  return 0;
}

static void *clip_worker(void *arg) {
  r3d_vkclip *cl = arg;
  for (;;) {
    pthread_mutex_lock(&cl->mu);
    while (cl->q_head == cl->q_tail && !cl->quit) pthread_cond_wait(&cl->cv, &cl->mu);
    if (cl->quit) {
      pthread_mutex_unlock(&cl->mu);
      return NULL;
    }
    clip_job j = cl->q[cl->q_head % QCAP];
    cl->q_head++;
    clip_level_state *st = &cl->st[j.level];
    clip_slot *sl = &st->slot[j.slot];
    /* claim the staging region: the slot must still be waiting on exactly this
     * request. Anything else (recentered, retargeted, already owned by the
     * pump) means the job is stale and its bytes must not be written. */
    if (st->gen != j.gen || sl->state != CLIP_QUEUED || sl->req != (int64_t)j.ls ||
        sl->req_gen != j.gen) {
      pthread_mutex_unlock(&cl->mu);
      continue;
    }
    sl->state = CLIP_PRODUCING;
    pthread_mutex_unlock(&cl->mu);

    int rc = produce_slice(cl, j.level, j.ls, st->stage + (size_t)j.slot * TEX * TEX);
    if (rc != 0)
      fprintf(stderr, "clip: corrupt shard data while filling L%u slice %llu\n", j.level,
              (unsigned long long)j.ls);

    pthread_mutex_lock(&cl->mu);
    if (rc == 0 && st->gen == j.gen) {
      sl->done = (int64_t)j.ls;
      sl->done_gen = j.gen;
      sl->state = CLIP_READY;
    } else if (rc != 0 && st->gen == j.gen) {
      /* decode failed: release the region but keep req/req_gen so the slice is
       * not re-attempted every frame (a new generation clears it) */
      sl->done = -1;
      sl->state = CLIP_FREE;
    } else { /* superseded mid-production: drop the payload */
      sl->req = -1;
      sl->done = -1;
      sl->state = CLIP_FREE;
    }
    pthread_mutex_unlock(&cl->mu);
    if (cl->trace)
      fprintf(stderr, "clip-fill L%u ls=%llu slot=%u rc=%d\n", j.level,
              (unsigned long long)j.ls, j.slot, rc);
  }
}

/* ---------- main-thread API ---------- */

static void enqueue(r3d_vkclip *cl, uint32_t l, uint64_t ls) {
  clip_level_state *st = &cl->st[l];
  uint32_t k = r3d_clip_ring_layer(&cl->clip, l, ls);
  if (k >= SLOTS) return; /* create() rejects such configs; belt and braces */
  pthread_mutex_lock(&cl->mu);
  clip_slot *sl = &st->slot[k];
  uint32_t g = st->gen;
  /* owned: the worker or the pump holds the staging bytes, so the region
   * cannot be retargeted now — the next frame re-requests this slice. */
  bool owned = sl->state == CLIP_PRODUCING || sl->state == CLIP_CONSUMING;
  bool pending = sl->state == CLIP_QUEUED && sl->req == (int64_t)ls && sl->req_gen == g;
  bool staged = sl->state == CLIP_READY && sl->done == (int64_t)ls && sl->done_gen == g;
  bool resident = sl->gpu == (int64_t)ls && sl->gpu_gen == g;
  /* attempted: a decode for this exact slice/generation already failed; a new
   * generation is what clears it, not another identical request */
  bool attempted = sl->state == CLIP_FREE && sl->req == (int64_t)ls && sl->req_gen == g;
  if (!owned && !pending && !staged && !attempted) {
    /* whatever the region holds belongs to another slice or generation:
     * release it (a queued job for it drops itself when the worker claims
     * the slot and sees the request identity no longer matches) */
    sl->req = -1;
    sl->done = -1;
    sl->state = CLIP_FREE;
    if (!resident && cl->q_tail - cl->q_head < QCAP) {
      cl->q[cl->q_tail % QCAP] = (clip_job){.level = l, .ls = ls, .gen = g, .slot = k};
      cl->q_tail++;
      sl->req = (int64_t)ls;
      sl->req_gen = g;
      sl->state = CLIP_QUEUED;
      pthread_cond_signal(&cl->cv);
    }
  }
  pthread_mutex_unlock(&cl->mu);
}

/* New generation for level l: staged/queued payloads and resident layers keyed
 * to the old window are void. Regions owned by the worker or the pump keep
 * their owner; the generation check at handoff discards their result. */
static void invalidate_level(r3d_vkclip *cl, uint32_t l) {
  clip_level_state *st = &cl->st[l];
  pthread_mutex_lock(&cl->mu);
  st->gen++;
  for (uint32_t k = 0; k < SLOTS; k++) {
    clip_slot *sl = &st->slot[k];
    sl->req = -1;
    sl->gpu = -1;
    sl->gpu_gen = st->gen;
    if (sl->state == CLIP_QUEUED || sl->state == CLIP_READY) {
      sl->done = -1;
      sl->state = CLIP_FREE;
    }
  }
  pthread_mutex_unlock(&cl->mu);
}

static void request_level_window(r3d_vkclip *cl, uint32_t l) {
  uint64_t ls0 = cl->z0 >> l;
  for (uint32_t k = 0; k < cl->clip.lv[l].wzl; k++) enqueue(cl, l, ls0 + k);
}

void r3d_vkclip_update(r3d_vkclip *cl, int64_t fx, int64_t fy, uint64_t z0) {
  cl->z0 = z0;
  /* coarse-to-fine: L5 fills in milliseconds and covers everything, so the
   * screen is never empty while L1/L0 grind through shard decodes */
  for (uint32_t i = 0; i < R3D_CLIP_LEVELS; i++) {
    uint32_t l = R3D_CLIP_LEVELS - 1 - i;
    clip_level_state *st = &cl->st[l];
    uint64_t ls0 = z0 >> l;
    bool bump = false;
    if (r3d_clip_need_recenter(&cl->clip, l, fx, fy)) {
      r3d_clip_recenter(&cl->clip, l, fx, fy);
      bump = true; /* XY remap: every slot's content meaning changed */
    } else if (st->ls0_set) {
      uint64_t prev = st->ls0;
      uint64_t d = ls0 > prev ? ls0 - prev : prev - ls0;
      /* Z discontinuity: the window jumped further than the ring is deep, so
       * every slot aliases to a different level slice at once. A smaller Z
       * step needs no new generation — a slot's payload is keyed by its level
       * slice and its owner state already blocks aliased overwrites. */
      if (d >= cl->clip.lv[l].wzl) bump = true;
    }
    st->ls0 = ls0;
    st->ls0_set = true;
    if (bump) invalidate_level(cl, l);
    request_level_window(cl, l);
  }
}

int r3d_vkclip_pump(r3d_vkclip *cl, VkSemaphore timeline, uint64_t tv) {
  /* collect finished slices; bound uploads per frame to limit hitches */
  uint32_t budget = 24;
  bool waited = false;
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS && budget; l++) {
    clip_level_state *st = &cl->st[l];
    for (uint32_t k = 0; k < cl->clip.lv[l].wzl && budget; k++) {
      clip_slot *sl = &st->slot[k];
      pthread_mutex_lock(&cl->mu);
      if (sl->state != CLIP_READY) {
        pthread_mutex_unlock(&cl->mu);
        continue;
      }
      if (sl->done < 0 || sl->done_gen != st->gen) { /* stale completion */
        sl->done = -1;
        sl->req = -1;
        sl->state = CLIP_FREE;
        pthread_mutex_unlock(&cl->mu);
        continue;
      }
      int64_t done = sl->done;
      uint32_t dgen = sl->done_gen;
      sl->state = CLIP_CONSUMING; /* the worker must not touch the region now */
      pthread_mutex_unlock(&cl->mu);

      int rc = 0;
      if (!waited && tv) { /* frames in flight may sample these layers */
        VkSemaphoreWaitInfo wi = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &timeline,
            .pValues = &tv,
        };
        if (vkWaitSemaphores(cl->vk->dev, &wi, UINT64_MAX) != VK_SUCCESS) rc = -1;
        else waited = true;
      }
      const uint8_t *src = st->stage + (size_t)k * TEX * TEX;
      if (rc == 0 && cl->fp_copy_mem) {
        VkMemoryToImageCopyEXT region = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
            .pHostPointer = src,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, (int32_t)k},
            .imageExtent = {TEX, TEX, 1},
        };
        VkCopyMemoryToImageInfoEXT ci = {
            .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
            .dstImage = cl->tex[l].img,
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount = 1,
            .pRegions = &region,
        };
        if (cl->fp_copy_mem(cl->vk->dev, &ci) != VK_SUCCESS) rc = -1;
      } else if (rc == 0 &&
                 r3d_vk_upload_image_staged_buf(cl->vk, cl->upload_pool, &cl->upload_stage,
                                                &cl->tex[l], src, TEX,
                                                (VkOffset3D){0, 0, (int32_t)k},
                                                (VkExtent3D){TEX, TEX, 1}) != 0) {
        rc = -1;
      }

      pthread_mutex_lock(&cl->mu);
      if (rc != 0) { /* upload failed: hand the payload back, keep it stageable */
        sl->state = CLIP_READY;
        pthread_mutex_unlock(&cl->mu);
        return -1;
      }
      bool fresh = st->gen == dgen;
      sl->gpu = fresh ? done : -1;
      sl->gpu_gen = st->gen;
      sl->done = -1;
      sl->req = -1;
      sl->state = CLIP_FREE;
      pthread_mutex_unlock(&cl->mu);
      budget--;
    }
  }
  return 0;
}

void r3d_vkclip_params(const r3d_vkclip *cl, r3d_frame_params *p) {
  p->slab_nx = (float)cl->clip.nx;
  p->slab_ny = (float)cl->clip.ny;
  p->slab_z0 = (float)cl->z0;
  p->slab_wz = cl->clip.depth_max;
  uint32_t mask = 0;
  /* gpu/gpu_gen/gen are written only on the main thread (update/pump), so this
   * main-thread read needs no lock. */
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS; l++) {
    const clip_level_state *st = &cl->st[l];
    uint64_t ls0 = cl->z0 >> l;
    bool ok = true;
    for (uint32_t k = 0; k < cl->clip.lv[l].wzl; k++) {
      uint32_t slot = r3d_clip_ring_layer(&cl->clip, l, ls0 + k);
      if (st->slot[slot].gpu != (int64_t)(ls0 + k) || st->slot[slot].gpu_gen != st->gen) {
        ok = false;
        break;
      }
    }
    if (ok) mask |= 1u << l;
    p->clip_orig[l][0] = (float)cl->clip.lv[l].ox;
    p->clip_orig[l][1] = (float)cl->clip.lv[l].oy;
  }
  /* never render with nothing: L5 covers all and fills in one frame */
  p->clip_valid = mask;
}

VkImageView r3d_vkclip_view(const r3d_vkclip *cl, uint32_t level) {
  return cl->tex[level].view;
}

uint32_t r3d_vkclip_band_z(const r3d_vkclip *cl) { return cl->band_z; }

int r3d_vkclip_create(r3d_vkclip **out, r3d_vkctx *c,
                      PFN_vkTransitionImageLayoutEXT fp_transition,
                      PFN_vkCopyMemoryToImageEXT fp_copy_mem, const char *band_dir,
                      const char *pyramid_dir, uint32_t band_z, uint32_t depth_max, int64_t fx,
                      int64_t fy, uint64_t z0) {
  *out = NULL;
  if (!c || !band_dir || !pyramid_dir) return -1;
  /* ring depth must fit the fixed slot arrays: wzl = depth_max/2^l + 3 peaks at
   * L0, so depth_max is capped at SLOTS-3 (61). Reject rather than let the
   * derived loops in enqueue/pump/params run past the storage. */
  if (depth_max < 2 || depth_max > SLOTS - 3) {
    fprintf(stderr, "clip: depth_max %u out of range (2..%u)\n", depth_max, SLOTS - 3);
    return -1;
  }
  if ((uint64_t)band_z * BAND_SLICES + BAND_SLICES > VOL_NZ) {
    fprintf(stderr, "clip: band_z %u past the volume (%llu slices)\n", band_z,
            (unsigned long long)VOL_NZ);
    return -1;
  }
  r3d_vkclip *cl = calloc(1, sizeof *cl);
  if (!cl) return -1;
  cl->vk = c;
  cl->fp_transition = fp_transition;
  cl->fp_copy_mem = fp_copy_mem;
  cl->band_z = band_z;
  cl->z0 = z0;
  cl->trace = getenv("R3D_CLIP_TRACE") != NULL;
  r3d_clip_init(&cl->clip, VOL_NX, VOL_NY, VOL_NZ, depth_max, fx, fy);
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS; l++) {
    if (cl->clip.lv[l].wzl > SLOTS) { /* r3d_clip_init changed its sizing rule */
      fprintf(stderr, "clip: L%u ring depth %u exceeds %u slots\n", l, cl->clip.lv[l].wzl,
              SLOTS);
      goto fail;
    }
  }
  if (r3d_shard_store_init(&cl->store, band_dir, VOL_NZ, VOL_NY, VOL_NX) != 0) goto fail;

  VkCommandPoolCreateInfo upci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = c->qfam,
  };
  if (vkCreateCommandPool(c->dev, &upci, NULL, &cl->upload_pool) != VK_SUCCESS) goto fail;

  for (uint32_t l = 2; l < R3D_CLIP_LEVELS; l++) {
    char path[600];
    snprintf(path, sizeof path, "%s/L%u.u8", pyramid_dir, l);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
      fprintf(stderr, "clip: missing %s (run mkpyramid)\n", path);
      goto fail;
    }
    struct stat stt;
    if (fstat(fd, &stt) != 0 || stt.st_size < 0) {
      fprintf(stderr, "clip: cannot stat %s\n", path);
      close(fd);
      goto fail;
    }
    cl->pyr[l].lx = VOL_NX >> l;
    cl->pyr[l].ly = VOL_NY >> l;
    cl->pyr[l].lz = (uint64_t)BAND_SLICES >> l;
    /* every byte produce_from_pyramid can address must be backed, or a short
     * file turns into SIGBUS / out-of-bounds reads at the far edge */
    size_t need = (size_t)(cl->pyr[l].lx * cl->pyr[l].ly * cl->pyr[l].lz);
    if ((uint64_t)stt.st_size < (uint64_t)need) {
      fprintf(stderr, "clip: %s is %llu bytes, needs %llu (rerun mkpyramid)\n", path,
              (unsigned long long)stt.st_size, (unsigned long long)need);
      close(fd);
      goto fail;
    }
    cl->pyr[l].n = need; /* map exactly what the indexed dims cover */
    cl->pyr[l].map = mmap(NULL, cl->pyr[l].n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (cl->pyr[l].map == MAP_FAILED) {
      cl->pyr[l].map = NULL;
      cl->pyr[l].n = 0;
      fprintf(stderr, "clip: cannot map %s\n", path);
      goto fail;
    }
  }

  cl->scratch = malloc((size_t)(2 * TEX) * (2 * TEX) * 2);
  if (!cl->scratch) goto fail;
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS; l++) {
    uint32_t wzl = cl->clip.lv[l].wzl;
    cl->st[l].stage = malloc((size_t)wzl * TEX * TEX);
    if (!cl->st[l].stage) goto fail;
    for (uint32_t k = 0; k < SLOTS; k++)
      cl->st[l].slot[k] = (clip_slot){.req = -1, .done = -1, .gpu = -1, .state = CLIP_FREE};
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (cl->fp_transition && cl->fp_copy_mem) usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
    /* explicitly 3D: the ring is sampled as a texture array-in-depth and wzl
     * may legally be small, so extent-based dimensionality inference must not
     * decide this one */
    if (r3d_vkimage_create_typed(c, VK_IMAGE_TYPE_3D, VK_FORMAT_R8_UNORM,
                                 (VkExtent3D){TEX, TEX, wzl}, 1, usage, &cl->tex[l]) != 0)
      goto fail;
    if (cl->fp_transition && cl->fp_copy_mem) {
      VkHostImageLayoutTransitionInfoEXT tr = {
          .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
          .image = cl->tex[l].img,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      if (cl->fp_transition(c->dev, 1, &tr) != VK_SUCCESS) goto fail;
    } else if (r3d_vk_image_to_general(c, cl->upload_pool, &cl->tex[l]) != 0) {
      goto fail;
    }
  }

  pthread_mutex_init(&cl->mu, NULL);
  pthread_cond_init(&cl->cv, NULL);
  cl->mu_up = true;
  if (pthread_create(&cl->worker, NULL, clip_worker, cl) != 0) goto fail;
  cl->worker_up = true;
  *out = cl;
  return 0;
fail:
  r3d_vkclip_destroy(cl);
  return -1;
}

void r3d_vkclip_destroy(r3d_vkclip *cl) {
  if (!cl) return;
  if (cl->worker_up) {
    pthread_mutex_lock(&cl->mu);
    cl->quit = true;
    pthread_cond_broadcast(&cl->cv);
    pthread_mutex_unlock(&cl->mu);
    pthread_join(cl->worker, NULL);
  }
  if (cl->mu_up) {
    pthread_mutex_destroy(&cl->mu);
    pthread_cond_destroy(&cl->cv);
  }
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS; l++) {
    r3d_vkimage_destroy(cl->vk, &cl->tex[l]);
    free(cl->st[l].stage);
    if (cl->pyr[l].map) munmap(cl->pyr[l].map, cl->pyr[l].n);
  }
  if (cl->upload_pool) vkDestroyCommandPool(cl->vk->dev, cl->upload_pool, NULL);
  r3d_vkbuf_destroy(cl->vk, &cl->upload_stage);
  free(cl->scratch);
  free(cl);
}
