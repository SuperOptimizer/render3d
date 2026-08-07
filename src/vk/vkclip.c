#include "vk/vkclip.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
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

typedef struct clip_job {
  uint32_t level;
  uint64_t ls;  /* level slice (absolute: world_z / 2^level) */
  uint32_t gen; /* level generation at enqueue; stale jobs are dropped */
} clip_job;

typedef struct clip_level_state {
  uint8_t *stage;                 /* wzl slots x TEX^2 (worker writes) */
  int64_t slot_req[64];           /* main: ls requested into slot, -1 none */
  _Atomic int64_t slot_done[64];  /* worker: ls staged in slot */
  int64_t slot_gpu[64];           /* main: ls currently in the texture */
  _Atomic uint32_t gen;
} clip_level_state;

struct r3d_vkclip {
  r3d_vkctx *vk;
  PFN_vkTransitionImageLayoutEXT fp_transition;
  PFN_vkCopyMemoryToImageEXT fp_copy_mem;
  VkCommandPool upload_pool; /* portable staged-upload fallback */
  r3d_vkbuf upload_stage;     /* worker-independent reusable fallback buffer */

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
  bool quit;
  pthread_t worker;
  uint8_t *scratch; /* L1 source region: 2*TEX x 2*TEX x 2 voxels */
};

/* ---------- worker ---------- */

static void produce_from_pyramid(r3d_vkclip *cl, uint32_t l, uint64_t ls, uint8_t *dst) {
  const r3d_clip_level *lv = &cl->clip.lv[l];
  uint64_t lx = cl->pyr[l].lx, ly = cl->pyr[l].ly, lz = cl->pyr[l].lz;
  int64_t zp = (int64_t)ls - (int64_t)(((uint64_t)cl->band_z * 1024) >> l);
  if (!cl->pyr[l].map || zp < 0 || zp >= (int64_t)lz) {
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
    int64_t lo = vx0 < 0 ? -vx0 : 0;
    int64_t hi = (int64_t)TEX - 1;
    if (vx0 + hi >= (int64_t)lx) hi = (int64_t)lx - 1 - vx0;
    if (hi >= lo) memcpy(drow + lo, srow + vx0 + lo, (size_t)(hi - lo + 1));
    for (int64_t c = 0; c < lo; c++) drow[c] = srow[0];
    for (int64_t c = hi + 1; c < TEX; c++) drow[c] = srow[lx - 1];
  }
}

static int produce_slice(r3d_vkclip *cl, uint32_t l, uint64_t ls, uint8_t *dst) {
  const r3d_clip_level *lv = &cl->clip.lv[l];
  if (l == 0) {
    return r3d_shard_decode_region(&cl->store, ls, (uint64_t)(lv->oy - 1),
                                   (uint64_t)(lv->ox - 1), 1, TEX, TEX, dst, 0);
  } else if (l == 1) {
    /* 2^3 box from full-res decode of the covered region */
    uint32_t W = 2 * TEX;
    if (r3d_shard_decode_region(&cl->store, ls * 2, (uint64_t)(lv->oy - 2),
                                (uint64_t)(lv->ox - 2), 2, W, W, cl->scratch, 0) != 0)
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
    pthread_mutex_unlock(&cl->mu);

    clip_level_state *st = &cl->st[j.level];
    if (atomic_load(&st->gen) != j.gen) continue; /* recentered since enqueue */
    uint32_t slot = r3d_clip_ring_layer(&cl->clip, j.level, j.ls);
    int rc = produce_slice(cl, j.level, j.ls, st->stage + (size_t)slot * TEX * TEX);
    if (rc != 0)
      fprintf(stderr, "clip: corrupt shard data while filling L%u slice %llu\n", j.level,
              (unsigned long long)j.ls);
    if (rc == 0 && atomic_load(&st->gen) == j.gen)
      atomic_store(&st->slot_done[slot], (int64_t)j.ls);
    if (getenv("R3D_CLIP_TRACE"))
      fprintf(stderr, "clip-fill L%u ls=%llu slot=%u\n", j.level, (unsigned long long)j.ls,
              slot);
  }
}

/* ---------- main-thread API ---------- */

static void enqueue(r3d_vkclip *cl, uint32_t l, uint64_t ls) {
  clip_level_state *st = &cl->st[l];
  uint32_t slot = r3d_clip_ring_layer(&cl->clip, l, ls);
  if (st->slot_req[slot] == (int64_t)ls) return; /* already requested */
  pthread_mutex_lock(&cl->mu);
  if (cl->q_tail - cl->q_head < QCAP) {
    cl->q[cl->q_tail % QCAP] =
        (clip_job){.level = l, .ls = ls, .gen = atomic_load(&cl->st[l].gen)};
    cl->q_tail++;
    st->slot_req[slot] = (int64_t)ls;
    pthread_cond_signal(&cl->cv);
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
    if (r3d_clip_need_recenter(&cl->clip, l, fx, fy)) {
      r3d_clip_recenter(&cl->clip, l, fx, fy);
      atomic_fetch_add(&cl->st[l].gen, 1);
      for (uint32_t s = 0; s < 64; s++) {
        cl->st[l].slot_req[s] = -1;
        cl->st[l].slot_gpu[s] = -1;
        atomic_store(&cl->st[l].slot_done[s], -1);
      }
    }
    request_level_window(cl, l);
  }
}

int r3d_vkclip_pump(r3d_vkclip *cl, VkSemaphore timeline, uint64_t tv) {
  /* collect finished slices; bound uploads per frame to limit hitches */
  uint32_t budget = 24;
  bool waited = false;
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS && budget; l++) {
    clip_level_state *st = &cl->st[l];
    for (uint32_t slot = 0; slot < cl->clip.lv[l].wzl && budget; slot++) {
      int64_t done = atomic_load(&st->slot_done[slot]);
      if (done < 0 || st->slot_gpu[slot] == done) continue;
      if (!waited && tv) { /* frames in flight may sample these layers */
        VkSemaphoreWaitInfo wi = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &timeline,
            .pValues = &tv,
        };
        if (vkWaitSemaphores(cl->vk->dev, &wi, UINT64_MAX) != VK_SUCCESS) return -1;
        waited = true;
      }
      const uint8_t *src = st->stage + (size_t)slot * TEX * TEX;
      if (cl->fp_copy_mem) {
        VkMemoryToImageCopyEXT region = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
            .pHostPointer = src,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, (int32_t)slot},
            .imageExtent = {TEX, TEX, 1},
        };
        VkCopyMemoryToImageInfoEXT ci = {
            .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
            .dstImage = cl->tex[l].img,
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount = 1,
            .pRegions = &region,
        };
        if (cl->fp_copy_mem(cl->vk->dev, &ci) != VK_SUCCESS) return -1;
      } else if (r3d_vk_upload_image_staged_buf(cl->vk, cl->upload_pool, &cl->upload_stage,
                                                &cl->tex[l], src, TEX,
                                                (VkOffset3D){0, 0, (int32_t)slot},
                                                (VkExtent3D){TEX, TEX, 1}) != 0) {
        return -1;
      }
      st->slot_gpu[slot] = done;
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
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS; l++) {
    const clip_level_state *st = &cl->st[l];
    uint64_t ls0 = cl->z0 >> l;
    bool ok = true;
    for (uint32_t k = 0; k < cl->clip.lv[l].wzl; k++) {
      uint32_t slot = r3d_clip_ring_layer(&cl->clip, l, ls0 + k);
      if (st->slot_gpu[slot] != (int64_t)(ls0 + k)) {
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
  r3d_vkclip *cl = calloc(1, sizeof *cl);
  if (!cl) return -1;
  cl->vk = c;
  cl->fp_transition = fp_transition;
  cl->fp_copy_mem = fp_copy_mem;
  cl->band_z = band_z;
  cl->z0 = z0;
  r3d_clip_init(&cl->clip, 43008, 43008, 68608, depth_max, fx, fy);
  if (r3d_shard_store_init(&cl->store, band_dir, 68608, 43008, 43008) != 0) goto fail;

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
    fstat(fd, &stt);
    cl->pyr[l].lx = 43008u >> l;
    cl->pyr[l].ly = 43008u >> l;
    cl->pyr[l].lz = 1024u >> l;
    cl->pyr[l].n = (size_t)stt.st_size;
    cl->pyr[l].map = mmap(NULL, cl->pyr[l].n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (cl->pyr[l].map == MAP_FAILED) goto fail;
  }

  cl->scratch = malloc((size_t)(2 * TEX) * (2 * TEX) * 2);
  if (!cl->scratch) goto fail;
  for (uint32_t l = 0; l < R3D_CLIP_LEVELS; l++) {
    uint32_t wzl = cl->clip.lv[l].wzl;
    cl->st[l].stage = malloc((size_t)wzl * TEX * TEX);
    if (!cl->st[l].stage) goto fail;
    for (uint32_t s = 0; s < 64; s++) {
      cl->st[l].slot_req[s] = -1;
      cl->st[l].slot_gpu[s] = -1;
      atomic_store(&cl->st[l].slot_done[s], -1);
    }
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (cl->fp_transition && cl->fp_copy_mem) usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
    if (r3d_vkimage_create(c, VK_FORMAT_R8_UNORM, (VkExtent3D){TEX, TEX, wzl}, 1, usage,
                           &cl->tex[l]) != 0)
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
  if (pthread_create(&cl->worker, NULL, clip_worker, cl) != 0) goto fail;
  *out = cl;
  return 0;
fail:
  r3d_vkclip_destroy(cl);
  return -1;
}

void r3d_vkclip_destroy(r3d_vkclip *cl) {
  if (!cl) return;
  if (cl->worker) {
    pthread_mutex_lock(&cl->mu);
    cl->quit = true;
    pthread_cond_broadcast(&cl->cv);
    pthread_mutex_unlock(&cl->mu);
    pthread_join(cl->worker, NULL);
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
