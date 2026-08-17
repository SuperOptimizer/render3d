#include "core/surfpred.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <brick.h> /* c5d brick encode */

#define SP_BRICK 128u
#define SP_CELL 256u /* 2x2x2 bricks == one P+1 brick */
#define SP_RAW ((size_t)SP_BRICK * SP_BRICK * SP_BRICK)
#define SP_CELL_RAW ((size_t)SP_CELL * SP_CELL * SP_CELL)
#define SP_RING 8u

bool r3d_surfpred_url(const char *url) {
  return url && strncmp(url, "predict://", 10) == 0;
}

static int sp_connect(int port) {
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
  struct timeval rto = {.tv_sec = 300}, sto = {.tv_sec = 60};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof rto);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sto, sizeof sto);
  return fd;
}

static int sp_io(int fd, void *buf, size_t n, bool wr) {
  uint8_t *p = buf;
  while (n) {
    ssize_t k = wr ? write(fd, p, n) : read(fd, p, n);
    if (k <= 0) return -1;
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

static int sp_write_file(const char *path, const void *data, size_t n) {
  char tmp[1460];
  int pn = snprintf(tmp, sizeof tmp, "%s.tmp.%ld.%lx", path, (long)getpid(),
                    (unsigned long)pthread_self());
  if (pn < 0 || (size_t)pn >= sizeof tmp) return -1;
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  int rc = n && fwrite(data, 1, n, f) != n ? -1 : 0;
  if (fclose(f) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

int r3d_surfpred_open(r3d_surfpred *sp, const char *pred_root) {
  memset(sp, 0, sizeof *sp);
  sp->fd = -1;
  sp->margin = 32;
  sp->th = 0.2f;
  sp->q = 2.0f;
  snprintf(sp->root, sizeof sp->root, "%s", pred_root);
  char mp[1200];
  snprintf(mp, sizeof mp, "%s/source.json", pred_root);
  FILE *f = fopen(mp, "rb");
  if (!f) return -1;
  char sj[16384] = {0};
  size_t sn = fread(sj, 1, sizeof sj - 1, f);
  fclose(f);
  (void)sn;
  const char *up = strstr(sj, "\"url\": \"");
  if (!up || !r3d_surfpred_url(up + 8)) return -1;
  const char *pp = strchr(up + 8 + 10, ':'); /* predict://host:port */
  sp->port = pp ? atoi(pp + 1) : 9744;
  if (sp->port <= 0) sp->port = 9744;
  const char *cp = strstr(sj, "\"ct_root\": \"");
  if (!cp) {
    fprintf(stderr, "surfpred: %s has no ct_root\n", mp);
    return -1;
  }
  cp += 12;
  const char *ce = strchr(cp, '"');
  if (!ce || (size_t)(ce - cp) >= sizeof sp->ct_root) return -1;
  memcpy(sp->ct_root, cp, (size_t)(ce - cp));
  sp->ct_root[ce - cp] = 0;
  const char *tp = strstr(sj, "\"th\": ");
  if (tp) sp->th = strtof(tp + 6, NULL);
  const char *mg = strstr(sj, "\"margin\": ");
  if (mg) sp->margin = (uint32_t)strtoul(mg + 10, NULL, 10);
  const char *qp = strstr(sj, "\"quality\": ");
  if (qp) sp->q = strtof(qp + 11, NULL);
  const char *pl = strstr(sj, "\"pred_level\": ");
  sp->pred_level = pl ? (uint32_t)strtoul(pl + 14, NULL, 10) : 0u;
  if (sp->pred_level > 4) sp->pred_level = 4;
  if (sp->margin > 96) sp->margin = 96;
  for (uint32_t i = 0; i < SP_RING; i++) sp->ring.key[i] = UINT64_MAX;
  /* CT sampler: 3x3x3 bricks per prediction; a modest cache keeps the
   * neighbours of consecutive cells */
  if (r3d_cpuvol_open(&sp->ct, sp->ct_root, 96) != 0) {
    fprintf(stderr, "surfpred: cannot open CT tree %s\n", sp->ct_root);
    return -1;
  }
  if (sp->pred_level >= sp->ct.nlev) {
    fprintf(stderr, "surfpred: pred_level %u beyond the CT pyramid (%u levels)\n",
            sp->pred_level, sp->ct.nlev);
    r3d_cpuvol_close(&sp->ct);
    return -1;
  }
  sp->ct_ok = true;
  sp->self_ok = r3d_cpuvol_open_ex(&sp->self, sp->root, 16, false) == 0;
  pthread_mutex_init(&sp->mu, NULL);
  return 0;
}

/* all eight level-P bricks of the cell already on disk? then the cell can be
 * rebuilt from files instead of re-running the model */
static bool sp_cell_on_disk(r3d_surfpred *sp, uint32_t P, uint32_t cx, uint32_t cy, uint32_t cz) {
  const r3d_cpuvol_level *lp = &sp->ct.lev[P];
  for (uint32_t sz = 0; sz < 2; sz++)
    for (uint32_t sy = 0; sy < 2; sy++)
      for (uint32_t sx = 0; sx < 2; sx++) {
        uint32_t bx = cx * 2 + sx, by = cy * 2 + sy, bz = cz * 2 + sz;
        if (bx >= lp->bx || by >= lp->by || bz >= lp->bz) continue;
        char path[1500];
        snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", sp->root, P, bz, by, bx);
        struct stat st;
        if (stat(path, &st) != 0) return false;
      }
  return true;
}

/* --- ring of recent cells --- */
static const uint8_t *sp_ring_get(r3d_surfpred *sp, uint64_t key) {
  if (!sp->ring.data) return NULL;
  for (uint32_t i = 0; i < SP_RING; i++)
    if (sp->ring.key[i] == key) return sp->ring.data + (size_t)i * SP_CELL_RAW;
  return NULL;
}
static uint8_t *sp_ring_slot(r3d_surfpred *sp, uint64_t key) {
  if (!sp->ring.data) sp->ring.data = malloc((size_t)SP_RING * SP_CELL_RAW);
  if (!sp->ring.data) return NULL;
  uint32_t i = sp->ring.next;
  sp->ring.next = (i + 1) % SP_RING;
  sp->ring.key[i] = key;
  return sp->ring.data + (size_t)i * SP_CELL_RAW;
}

static void sp_write_brick(r3d_surfpred *sp, const c5d_brick_params *bp, uint32_t li,
                           uint32_t bx, uint32_t by, uint32_t bz, const uint8_t *raw,
                           r3d_cpuvol *cache) {
  char dir[1300], path[1500];
  snprintf(dir, sizeof dir, "%s/bricks", sp->root);
  mkdir(dir, 0755);
  snprintf(dir, sizeof dir, "%s/bricks/L%u", sp->root, li);
  mkdir(dir, 0755);
  snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", sp->root, li, bz, by, bx);
  bool zero = true;
  for (size_t i = 0; i < SP_RAW && zero; i++) zero = raw[i] == 0;
  struct stat st;
  if (stat(path, &st) != 0) {
    if (zero) {
      sp_write_file(path, NULL, 0);
    } else {
      uint8_t *enc = NULL;
      size_t en = 0;
      if (c5d_brick_encode(bp, raw, SP_BRICK, &enc, &en) == 0) {
        sp_write_file(path, enc, en);
        free(enc);
      }
    }
  }
  if (cache && !zero) r3d_cpuvol_cache_put(cache, li, bx, by, bz, raw);
}

void r3d_surfpred_close(r3d_surfpred *sp) {
  if (sp->fd >= 0) close(sp->fd);
  free(sp->ring.data);
  if (sp->self_ok) r3d_cpuvol_close(&sp->self);
  if (sp->ct_ok) {
    r3d_cpuvol_close(&sp->ct);
    pthread_mutex_destroy(&sp->mu);
  }
  memset(sp, 0, sizeof *sp);
  sp->fd = -1;
}

/* round-trip one block; reconnects once */
static int sp_predict(r3d_surfpred *sp, const uint8_t *in, uint8_t *out, uint32_t n) {
  for (int attempt = 0; attempt < 2; attempt++) {
    if (sp->fd < 0) sp->fd = sp_connect(sp->port);
    if (sp->fd < 0) return -1;
    uint8_t hdr[16] = {'S', 'R', 'F', '1'};
    uint32_t d[3] = {n, n, n};
    memcpy(hdr + 4, d, 12);
    uint8_t rh[16];
    if (sp_io(sp->fd, hdr, sizeof hdr, true) == 0 &&
        sp_io(sp->fd, (void *)in, (size_t)n * n * n, true) == 0 &&
        sp_io(sp->fd, rh, sizeof rh, false) == 0 && memcmp(rh, "SRFR", 4) == 0) {
      uint32_t r3[3];
      memcpy(r3, rh + 4, 12);
      if (r3[0] == n && r3[1] == n && r3[2] == n &&
          sp_io(sp->fd, out, (size_t)n * n * n, false) == 0)
        return 0;
    }
    close(sp->fd);
    sp->fd = -1;
  }
  return -1;
}

int r3d_surfpred_cell(r3d_surfpred *sp, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz,
                      uint8_t *out, r3d_cpuvol *cache) {
  uint32_t P = sp->pred_level;
  if (!sp->ct_ok || li > P + 1) return 0;
  /* the owning level-P cell (== level-P+1 brick) */
  uint32_t cx, cy, cz;
  if (li == P + 1) {
    cx = bx;
    cy = by;
    cz = bz;
  } else {
    uint32_t sh = P - li; /* fine brick -> level-P brick -> cell */
    cx = (bx >> sh) / 2;
    cy = (by >> sh) / 2;
    cz = (bz >> sh) / 2;
  }
  const r3d_cpuvol_level *lp = &sp->ct.lev[P];
  if (cx * SP_CELL >= lp->vx || cy * SP_CELL >= lp->vy || cz * SP_CELL >= lp->vz) return 0;
  uint64_t now = (uint64_t)time(NULL);
  if (sp->cool_until > now) return -1;
  uint64_t ckey = ((uint64_t)cz << 40) | ((uint64_t)cy << 20) | cx;

  pthread_mutex_lock(&sp->mu);
  c5d_brick_params bp = c5d_brick_defaults(1.0f);
  bp.q = sp->q;
  int rc = 0;
  const uint8_t *cellv = sp_ring_get(sp, ckey); /* thresholded 256^3 level-P */
  if (!cellv && sp->self_ok && sp_cell_on_disk(sp, P, cx, cy, cz)) {
    uint8_t *slot = sp_ring_slot(sp, ckey);
    if (slot) {
      r3d_cpuvol_read_block(&sp->self, P, (int64_t)cx * SP_CELL, (int64_t)cy * SP_CELL,
                            (int64_t)cz * SP_CELL, SP_CELL, SP_CELL, SP_CELL, slot);
      cellv = slot;
    }
  }
  if (!cellv) {
    uint32_t m = sp->margin, n = SP_CELL + 2 * m;
    size_t nn = (size_t)n * n * n;
    uint8_t *ct = malloc(nn), *pr = malloc(nn);
    if (ct && pr) {
      int64_t x0 = (int64_t)cx * SP_CELL - m, y0 = (int64_t)cy * SP_CELL - m,
              z0 = (int64_t)cz * SP_CELL - m;
      r3d_cpuvol_read_block(&sp->ct, P, x0, y0, z0, n, n, n, ct);
      bool any = false;
      for (size_t i = 0; i < nn && !any; i += 4096) any = ct[i] != 0;
      if (!any) {
        memset(pr, 0, nn);
        rc = 0;
      } else {
        rc = sp_predict(sp, ct, pr, n);
      }
      if (rc == 0) {
        uint8_t *slot = sp_ring_slot(sp, ckey);
        if (slot) {
          uint8_t thv = (uint8_t)(sp->th * 255.0f + 0.5f);
          for (uint32_t z = 0; z < SP_CELL; z++)
            for (uint32_t y = 0; y < SP_CELL; y++) {
              const uint8_t *src = pr + ((size_t)(m + z) * n + (m + y)) * n + m;
              uint8_t *dst = slot + ((size_t)z * SP_CELL + y) * SP_CELL;
              for (uint32_t x = 0; x < SP_CELL; x++) dst[x] = src[x] < thv ? 0 : src[x];
            }
          cellv = slot;
          /* emit the eight level-P bricks + the P+1 brick */
          uint8_t *raw = malloc(SP_RAW), *l1 = calloc(SP_RAW, 1);
          if (raw && l1) {
            for (uint32_t sz = 0; sz < 2; sz++)
              for (uint32_t sy = 0; sy < 2; sy++)
                for (uint32_t sx = 0; sx < 2; sx++) {
                  uint32_t obx = cx * 2 + sx, oby = cy * 2 + sy, obz = cz * 2 + sz;
                  if (obx >= lp->bx || oby >= lp->by || obz >= lp->bz) continue;
                  for (uint32_t z = 0; z < SP_BRICK; z++)
                    for (uint32_t y = 0; y < SP_BRICK; y++)
                      memcpy(raw + ((size_t)z * SP_BRICK + y) * SP_BRICK,
                             cellv + ((size_t)(sz * SP_BRICK + z) * SP_CELL + sy * SP_BRICK + y) *
                                         SP_CELL +
                                 sx * SP_BRICK,
                             SP_BRICK);
                  for (uint32_t z = 0; z < SP_BRICK / 2; z++)
                    for (uint32_t y = 0; y < SP_BRICK / 2; y++)
                      for (uint32_t x = 0; x < SP_BRICK / 2; x++) {
                        uint32_t s2 = 0;
                        for (uint32_t dz = 0; dz < 2; dz++)
                          for (uint32_t dy = 0; dy < 2; dy++)
                            for (uint32_t dx = 0; dx < 2; dx++)
                              s2 += raw[((size_t)(2 * z + dz) * SP_BRICK + 2 * y + dy) * SP_BRICK +
                                        2 * x + dx];
                        l1[((size_t)(sz * 64 + z) * SP_BRICK + sy * 64 + y) * SP_BRICK + sx * 64 +
                           x] = (uint8_t)((s2 + 4) / 8);
                      }
                  sp_write_brick(sp, &bp, P, obx, oby, obz, raw, cache);
                }
            if (P + 1 < sp->ct.nlev) {
              const r3d_cpuvol_level *l1v = &sp->ct.lev[P + 1];
              if (cx < l1v->bx && cy < l1v->by && cz < l1v->bz)
                sp_write_brick(sp, &bp, P + 1, cx, cy, cz, l1, cache);
              if (out && li == P + 1) memcpy(out, l1, SP_RAW);
            }
            sp->predicted_cells++;
          } else {
            rc = -1;
          }
          free(raw);
          free(l1);
        } else {
          rc = -1;
        }
      } else {
        sp->failed_cells++;
        sp->cool_until = now + 10; /* server down: retry in a bit, don't spin */
      }
    } else {
      rc = -1;
    }
    free(ct);
    free(pr);
  } else if (out && li == P + 1) {
    /* cell known but the P+1 brick requested: rebuild it from the ring */
    for (uint32_t z = 0; z < SP_BRICK; z++)
      for (uint32_t y = 0; y < SP_BRICK; y++)
        for (uint32_t x = 0; x < SP_BRICK; x++) {
          uint32_t s2 = 0;
          for (uint32_t dz = 0; dz < 2; dz++)
            for (uint32_t dy = 0; dy < 2; dy++)
              for (uint32_t dx = 0; dx < 2; dx++)
                s2 += cellv[((size_t)(2 * z + dz) * SP_CELL + 2 * y + dy) * SP_CELL + 2 * x + dx];
          out[((size_t)z * SP_BRICK + y) * SP_BRICK + x] = (uint8_t)((s2 + 4) / 8);
        }
  }
  if (rc == 0 && cellv) {
    if (li == P && out) memcpy(out, cellv, SP_CELL_RAW);
    if (li < P) {
      /* finer level: nearest-upsample the sub-block of the cell that this
       * brick covers (factor 2^(P-li)) and write just that brick */
      uint32_t sh = P - li, f = 1u << sh;
      uint32_t bxP = bx >> sh, byP = by >> sh, bzP = bz >> sh; /* level-P brick */
      /* offset of this fine brick inside the level-P brick, in level-P voxels */
      uint32_t ox = (bx & (f - 1)) * (SP_BRICK / f), oy = (by & (f - 1)) * (SP_BRICK / f),
               oz = (bz & (f - 1)) * (SP_BRICK / f);
      uint32_t cx0 = (bxP & 1u) * SP_BRICK + ox, cy0 = (byP & 1u) * SP_BRICK + oy,
               cz0 = (bzP & 1u) * SP_BRICK + oz; /* inside the 256^3 cell */
      uint8_t *raw = malloc(SP_RAW);
      if (raw) {
        for (uint32_t z = 0; z < SP_BRICK; z++)
          for (uint32_t y = 0; y < SP_BRICK; y++)
            for (uint32_t x = 0; x < SP_BRICK; x++)
              raw[((size_t)z * SP_BRICK + y) * SP_BRICK + x] =
                  cellv[((size_t)(cz0 + z / f) * SP_CELL + (cy0 + y / f)) * SP_CELL + cx0 + x / f];
        sp_write_brick(sp, &bp, li, bx, by, bz, raw, cache);
        if (out) memcpy(out, raw, SP_RAW);
        free(raw);
      } else {
        rc = -1;
      }
    }
  }
  pthread_mutex_unlock(&sp->mu);
  return rc == 0 ? 1 : -1;
}
