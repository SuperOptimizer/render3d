#include "core/inklive.h"

#include <arpa/inet.h>
#include <math.h>
#include <stdatomic.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static double il_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int il_connect(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (connect(fd, (struct sockaddr *)&a, sizeof a) != 0) {
    close(fd);
    return -1;
  }
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  /* never hang forever on a wedged server: inference of a max-size region
   * is ~10-30 s on a laptop GPU, and a full TTA+ensemble pass multiplies
   * that ~16x, so allow generous but finite waits */
  struct timeval rto = {.tv_sec = 900}, sto = {.tv_sec = 60};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof rto);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sto, sizeof sto);
  return fd;
}

static int il_io(int fd, void *buf, size_t n, bool write_side) {
  uint8_t *p = buf;
  while (n) {
    ssize_t k = write_side ? write(fd, p, n) : read(fd, p, n);
    if (k <= 0) return -1;
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

/* rendseg's bilinear surface sampling over one rect: 21 layers, 1-voxel
 * pitch along the local (grid-derived) normal, level-0 CT. Rows are striped
 * across a thread fan-out (cpuvol is thread-safe; each thread keeps its own
 * brick memo, and a column's 21 taps mostly share one brick). */
struct il_job {
  r3d_inklive *il;
  const float *xyz;
  uint8_t *img;
  uint32_t rw, rh, up, W, H, gen, tid, nth, nl;
  bool flip; /* verso: reverse the layer order (z-flip of the slab) */
  _Atomic uint32_t *abort;
};

static void *il_sample_rows(void *ud) {
  struct il_job *jb = ud;
  r3d_inklive *il = jb->il;
  const float *xyz = jb->xyz;
  uint8_t *img = jb->img;
  uint32_t W = jb->W, H = jb->H, nl = jb->nl, up = jb->up, gen = jb->gen;
  uint32_t cw = jb->rw + 1; /* corner grid pitch */
  for (uint32_t oy = jb->tid; oy < H; oy += jb->nth) {
    if ((oy & 15u) == 0 && atomic_load_explicit(&il->req_gen, memory_order_relaxed) != gen) {
      atomic_store(jb->abort, 1); /* superseded: a newer rect is waiting */
      return NULL;
    }
    if (atomic_load_explicit(jb->abort, memory_order_relaxed)) return NULL;
    double gv = (double)oy / up;
    uint32_t j = (uint32_t)gv;
    double fv = gv - j;
    for (uint32_t ox = 0; ox < W; ox++) {
      double gu = (double)ox / up;
      uint32_t i = (uint32_t)gu;
      double fu = gu - i;
      const float *p00 = xyz + ((size_t)j * cw + i) * 3;
      const float *p10 = p00 + 3, *p01 = p00 + (size_t)cw * 3, *p11 = p01 + 3;
      if (p00[0] < 0 || p10[0] < 0 || p01[0] < 0 || p11[0] < 0) continue;
      double p[3], du[3], dv[3];
      for (int a = 0; a < 3; a++) {
        double q0 = (double)p00[a] * (1 - fu) + (double)p10[a] * fu;
        double q1 = (double)p01[a] * (1 - fu) + (double)p11[a] * fu;
        p[a] = q0 * (1 - fv) + q1 * fv;
        du[a] = ((double)p10[a] - (double)p00[a]) * (1 - fv) +
                ((double)p11[a] - (double)p01[a]) * fv;
        dv[a] = q1 - q0;
      }
      double n[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2],
                     du[0] * dv[1] - du[1] * dv[0]};
      double nrm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (nrm > 1e-9)
        for (int a = 0; a < 3; a++) n[a] /= nrm;
      else
        continue;
      for (uint32_t l = 0; l < nl; l++) {
        double o = (double)l - (double)(nl - 1) / 2.0;
        double q[3] = {p[0] + o * n[0], p[1] + o * n[1], p[2] + o * n[2]};
        double val = r3d_cpuvol_tri(&il->vol, 0, q, NULL);
        /* verso = the same symmetric sample set with the layer order
         * reversed (equivalent to negating the normal): the model reads
         * the slab from the other face of the sheet */
        uint32_t lw = jb->flip ? nl - 1u - l : l;
        img[(size_t)lw * W * H + (size_t)oy * W + ox] =
            (uint8_t)(val < 0 ? 0 : val > 255 ? 255 : val);
      }
    }
  }
  return NULL;
}

static uint8_t *il_sample(r3d_inklive *il, const float *xyz, uint32_t rw, uint32_t rh,
                          uint32_t up, uint32_t gen, uint32_t nl, bool flip, uint32_t *ow,
                          uint32_t *oh) {
  uint32_t W = rw * up, H = rh * up;
  uint8_t *img = calloc((size_t)W * H * nl, 1);
  if (!img) return NULL;
  long nc = sysconf(_SC_NPROCESSORS_ONLN);
  uint32_t nth = nc > 2 ? (uint32_t)(nc - 2) : 1u; /* leave the render thread room */
  if (nth > 16) nth = 16;
  if (nth > H) nth = H ? H : 1;
  _Atomic uint32_t abort_flag = 0;
  struct il_job jobs[16];
  pthread_t th[16];
  uint32_t spawned = 0;
  for (uint32_t t = 0; t < nth; t++) {
    jobs[t] = (struct il_job){.il = il, .xyz = xyz, .img = img, .rw = rw, .rh = rh, .up = up,
                              .W = W, .H = H, .gen = gen, .tid = t, .nth = nth,
                              .nl = nl, .flip = flip, .abort = &abort_flag};
    if (t + 1 < nth) {
      if (pthread_create(&th[spawned], NULL, il_sample_rows, &jobs[t]) == 0) spawned++;
      else il_sample_rows(&jobs[t]); /* no thread: do this stripe inline */
    }
  }
  il_sample_rows(&jobs[nth - 1]);
  for (uint32_t t = 0; t < spawned; t++) pthread_join(th[t], NULL);
  if (atomic_load(&abort_flag)) {
    free(img);
    return NULL;
  }
  *ow = W;
  *oh = H;
  return img;
}

/* Cold regions: the sampler used to demand-fetch one cell at a time under
 * cpuvol's IO lock, stalling every sampler thread. Enumerate the level-0
 * bricks the surface can touch (corner points, +-11 voxels for the layer
 * span, so a point near a brick face also pulls its neighbour) and fetch
 * the missing cells in parallel first. */
static void il_prefetch(r3d_inklive *il, const float *xyz, uint32_t rw, uint32_t rh,
                        uint32_t nl) {
  if (!il->vol.url[0]) return;
  uint32_t cw = rw + 1, ch = rh + 1;
  size_t cap = (size_t)cw * ch * 8u;
  if (cap > 4000000u) cap = 4000000u;
  uint32_t *list = malloc(cap * 3u * sizeof *list);
  if (!list) return;
  uint32_t n = 0;
  const double span = (double)(nl - 1) / 2.0 + 1.0; /* slab half-depth */
  for (uint32_t j = 0; j < ch; j++)
    for (uint32_t i = 0; i < cw; i++) {
      const float *p = xyz + ((size_t)j * cw + i) * 3;
      if (p[0] < 0) continue;
      /* local normal from grid neighbours (one-sided at the rect edge) so
       * only bricks along the 21-layer line are pulled, not a whole cube */
      const float *pu0 = i > 0 ? p - 3 : p, *pu1 = i + 1 < cw ? p + 3 : p;
      const float *pv0 = j > 0 ? p - (size_t)cw * 3 : p, *pv1 = j + 1 < ch ? p + (size_t)cw * 3 : p;
      double nrm[3] = {0, 0, 0};
      if (pu0[0] >= 0 && pu1[0] >= 0 && pv0[0] >= 0 && pv1[0] >= 0) {
        double du[3], dv[3];
        for (int a = 0; a < 3; a++) {
          du[a] = (double)pu1[a] - (double)pu0[a];
          dv[a] = (double)pv1[a] - (double)pv0[a];
        }
        nrm[0] = du[1] * dv[2] - du[2] * dv[1];
        nrm[1] = du[2] * dv[0] - du[0] * dv[2];
        nrm[2] = du[0] * dv[1] - du[1] * dv[0];
        double l = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        if (l > 1e-9)
          for (int a = 0; a < 3; a++) nrm[a] /= l;
        else
          nrm[0] = nrm[1] = nrm[2] = 0;
      }
      bool have_n = nrm[0] != 0 || nrm[1] != 0 || nrm[2] != 0;
      int64_t lo[3], hi[3];
      for (int a = 0; a < 3; a++) {
        /* with a normal: the segment p +- span*n; without: a full cube */
        double e = have_n ? span * fabs(nrm[a]) + 0.5 : span;
        double v0 = (double)p[a] - e, v1 = (double)p[a] + e;
        lo[a] = v0 < 0 ? 0 : (int64_t)(v0 / 128.0);
        hi[a] = v1 < 0 ? 0 : (int64_t)(v1 / 128.0);
      }
      for (int64_t bz = lo[2]; bz <= hi[2]; bz++)
        for (int64_t by = lo[1]; by <= hi[1]; by++)
          for (int64_t bx = lo[0]; bx <= hi[0]; bx++) {
            bool dup = false; /* neighbours repeat: check the recent tail
                               * (the prefetch dedupes by cell anyway) */
            for (uint32_t k = n > 64 ? n - 64 : 0; k < n; k++)
              if (list[k * 3] == (uint32_t)bx && list[k * 3 + 1] == (uint32_t)by &&
                  list[k * 3 + 2] == (uint32_t)bz) {
                dup = true;
                break;
              }
            if (dup || n >= cap) continue;
            list[n * 3] = (uint32_t)bx;
            list[n * 3 + 1] = (uint32_t)by;
            list[n * 3 + 2] = (uint32_t)bz;
            n++;
          }
    }
  if (n) {
    pthread_mutex_lock(&il->mu);
    snprintf(il->status, sizeof il->status, "prefetching %u bricks...", n);
    pthread_mutex_unlock(&il->mu);
    r3d_cpuvol_prefetch(&il->vol, 0, list, n, 12);
  }
  free(list);
}

static void *il_worker(void *ud) {
  r3d_inklive *il = ud;
  uint32_t done_gen = 0;
  for (;;) {
    pthread_mutex_lock(&il->mu);
    while (!il->quit && atomic_load(&il->req_gen) == done_gen)
      pthread_cond_wait(&il->cv, &il->mu);
    if (il->quit) {
      pthread_mutex_unlock(&il->mu);
      return NULL;
    }
    uint32_t gen = atomic_load(&il->req_gen);
    float *xyz = il->req_xyz;
    il->req_xyz = NULL;
    uint32_t rw = il->rw, rh = il->rh, i0 = il->ri0, j0 = il->rj0, up = il->up;
    uint32_t tta = il->req_tta;
    bool flip = il->req_flip;
    /* TTA depth-shift range S (bits 8..11) beyond the built-in 2-layer
     * slack needs a deeper sampled slab: model depth 17 + 2*S layers */
    uint32_t dsr = (tta >> 8) & 0xfu;
    uint32_t nl = R3D_INKLIVE_LAYERS;
    if ((tta & 2u) && dsr > 2) nl = 17u + 2u * dsr;
    snprintf(il->status, sizeof il->status, "sampling %ux%u px...", rw * up, rh * up);
    pthread_mutex_unlock(&il->mu);

    double t0 = il_now_ms();
    if (xyz) il_prefetch(il, xyz, rw, rh, nl);
    uint32_t W = 0, H = 0;
    uint8_t *stack = xyz ? il_sample(il, xyz, rw, rh, up, gen, nl, flip, &W, &H) : NULL;
    double t_sampled = il_now_ms() - t0;
    free(xyz);
    float *pred = NULL;
    if (stack) {
      pthread_mutex_lock(&il->mu);
      snprintf(il->status, sizeof il->status, "inferring %ux%u px...", W, H);
      pthread_mutex_unlock(&il->mu);
      if (il->fd < 0) il->fd = il_connect(il->port);
      for (int attempt = 0; attempt < 2 && il->fd >= 0; attempt++) {
        /* INK2 carries the TTA/ensemble bitmask; plain INK1 keeps old
         * servers working for the fast path */
        uint8_t hdr[20] = {'I', 'N', 'K', tta ? '2' : '1'};
        uint32_t v[4] = {nl, H, W, tta};
        size_t hlen = tta ? 20 : 16;
        memcpy(hdr + 4, v, hlen - 4);
        uint8_t rhdr[12];
        if (il_io(il->fd, hdr, hlen, true) == 0 &&
            il_io(il->fd, stack, (size_t)nl * W * H, true) == 0 &&
            il_io(il->fd, rhdr, sizeof rhdr, false) == 0 &&
            memcmp(rhdr, "INKR", 4) == 0) {
          uint32_t ph, pw;
          memcpy(&ph, rhdr + 4, 4);
          memcpy(&pw, rhdr + 8, 4);
          if (ph == H && pw == W) {
            pred = malloc((size_t)W * H * sizeof *pred);
            if (pred && il_io(il->fd, pred, (size_t)W * H * sizeof *pred, false) == 0)
              break;
            free(pred);
            pred = NULL;
          }
        }
        close(il->fd); /* stale/broken connection: reconnect once */
        il->fd = il_connect(il->port);
      }
    }
    free(stack);

    float pmax = 0.0f;
    double psum = 0.0;
    size_t pn = 0;
    if (pred) { /* scrollprize display rescale: confident no-ink sits at 0.25 */
      for (size_t k = 0; k < (size_t)W * H; k++) {
        float v = (pred[k] - 0.25f) * 2.0f;
        v = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
        pred[k] = v;
        if (v > pmax) pmax = v;
        psum += (double)v;
        pn++;
      }
    }
    pthread_mutex_lock(&il->mu);
    if (pred) {
      free(il->res);
      il->res = pred;
      il->res_w = W;
      il->res_h = H;
      il->res_i0 = i0;
      il->res_j0 = j0;
      il->res_up = up;
      il->res_gen = gen;
      il->res_fresh = true;
      il->res_ms = il_now_ms() - t0;
      snprintf(il->status, sizeof il->status, "%ux%u in %.1fs", W, H, il->res_ms / 1e3);
      printf("inklive: %ux%u prediction in %.1f s (sample %.1f s + infer), "
             "ink max %.2f mean %.3f\n",
             W, H, il->res_ms / 1e3, t_sampled / 1e3, (double)pmax,
             pn ? psum / (double)pn : 0.0);
    } else {
      snprintf(il->status, sizeof il->status,
               il->fd < 0 ? "server unreachable (port %d)" : "request failed (port %d)",
               il->port);
    }
    done_gen = gen;
    pthread_mutex_unlock(&il->mu);
  }
}

int r3d_inklive_start(r3d_inklive *il, const char *vol_root, int port) {
  memset(il, 0, sizeof *il);
  il->port = port;
  il->fd = -1;
  if (r3d_cpuvol_open(&il->vol, vol_root, 512) != 0) return -1;
  il->vol_ok = true;
  pthread_mutex_init(&il->mu, NULL);
  pthread_cond_init(&il->cv, NULL);
  snprintf(il->status, sizeof il->status, "idle");
  if (pthread_create(&il->th, NULL, il_worker, il) != 0) {
    r3d_cpuvol_close(&il->vol);
    il->vol_ok = false;
    return -1;
  }
  il->th_up = true;
  return 0;
}

void r3d_inklive_stop(r3d_inklive *il) {
  if (il->th_up) {
    pthread_mutex_lock(&il->mu);
    il->quit = true;
    pthread_cond_broadcast(&il->cv);
    pthread_mutex_unlock(&il->mu);
    pthread_join(il->th, NULL);
  }
  if (il->fd >= 0) close(il->fd);
  free(il->req_xyz);
  free(il->res);
  if (il->vol_ok) r3d_cpuvol_close(&il->vol);
  if (il->th_up) {
    pthread_mutex_destroy(&il->mu);
    pthread_cond_destroy(&il->cv);
  }
  memset(il, 0, sizeof *il);
  il->fd = -1;
}

void r3d_inklive_request(r3d_inklive *il, float *xyz, uint32_t i0, uint32_t j0,
                         uint32_t w, uint32_t h, uint32_t up, uint32_t tta, bool flip) {
  if (!il->th_up) {
    free(xyz);
    return;
  }
  pthread_mutex_lock(&il->mu);
  free(il->req_xyz); /* supersede an unclaimed request */
  il->req_xyz = xyz;
  il->ri0 = i0;
  il->rj0 = j0;
  il->rw = w;
  il->rh = h;
  il->up = up;
  il->req_tta = tta;
  il->req_flip = flip;
  atomic_fetch_add(&il->req_gen, 1);
  pthread_cond_broadcast(&il->cv);
  pthread_mutex_unlock(&il->mu);
}

bool r3d_inklive_poll(r3d_inklive *il, const float **pred, uint32_t *w, uint32_t *h,
                      uint32_t *i0, uint32_t *j0, uint32_t *up) {
  if (!il->th_up) return false;
  pthread_mutex_lock(&il->mu);
  bool fresh = il->res_fresh;
  if (fresh) {
    il->res_fresh = false;
    *pred = il->res;
    *w = il->res_w;
    *h = il->res_h;
    *i0 = il->res_i0;
    *j0 = il->res_j0;
    *up = il->res_up;
  }
  pthread_mutex_unlock(&il->mu);
  return fresh;
}
