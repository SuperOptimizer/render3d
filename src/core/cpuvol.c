#include "core/cpuvol.h"
#include "core/surfpred.h"

#include <blosc.h>
#include <curl/curl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <brick.h> /* c5d brick decode */
#include <shard.h> /* c5d .c5s reader */

#define CV_BRICK 128u
#define CV_SHARD_BPA 8u /* 1024^3 shards = 8^3 bricks */
#define CV_RAW ((size_t)CV_BRICK * CV_BRICK * CV_BRICK)

/* Demand-fetch resource caps. A legal-but-pathological source chunk edge
 * must not be able to ask for gigabyte allocations or an unbounded HTTP
 * body: the fetch assembles an lcm(chunk,128)^3 cell and holds one source
 * chunk, one cell and one response at once. A 127-wide chunk would need a
 * 16256^3 cell; a 1024-wide chunk a 1 GiB one. Levels whose cell exceeds
 * the budget lose demand fetch, not the whole volume. */
#define CV_MAX_CELL_EDGE 512u
#define CV_MAX_CELL_BYTES ((size_t)256u << 20)
#define CV_MAX_CHUNK_BYTES ((size_t)256u << 20)
#define CV_MAX_BODY_BYTES ((size_t)512u << 20)
#define CV_BLOSC_HDR ((size_t)BLOSC_MIN_HEADER_LENGTH)

static uint32_t cv_cell_dim(uint32_t chsz) { /* lcm(chsz, brick) */
  uint32_t a = chsz, b = CV_BRICK;
  while (b) {
    uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a ? chsz / a * CV_BRICK : 0u;
}

/* true when the cell/chunk this edge implies fits the per-request budget */
static bool cv_cell_ok(uint32_t chsz) {
  uint32_t cell = cv_cell_dim(chsz);
  if (!chsz || !cell || cell > CV_MAX_CELL_EDGE) return false;
  size_t cellb = (size_t)cell * cell * cell, chb = (size_t)chsz * chsz * chsz;
  return cellb <= CV_MAX_CELL_BYTES && chb <= CV_MAX_CHUNK_BYTES;
}

/* Decode cache: a refcounted pool of 128^3 slabs shared by every sampler
 * on one volume. A reader holds a lease that pins its slot, so a pointer
 * handed out under the pool lock stays valid and immutable until that
 * thread leases a different brick; eviction only ever considers unpinned
 * slots. The pool is refcounted apart from the r3d_cpuvol so a lease that
 * outlives r3d_cpuvol_close keeps its bytes alive instead of dangling. */
typedef struct cv_cache {
  _Atomic uint32_t refs;
  pthread_mutex_t m;
  uint8_t *slabs;
  uint64_t *keys; /* key or UINT64_MAX */
  uint64_t *use;  /* LRU ticks */
  uint32_t *pin;  /* live leases per slot; eviction skips nonzero */
  uint32_t nslots;
  uint64_t tick;
  /* hash index over keys (open addressing, slot+1, 0 = empty) so a hit is
   * O(1) instead of a linear scan of every slot per non-memo lookup */
  uint32_t *hidx;
  uint32_t hmask;
} cv_cache;

static void cvc_unref(cv_cache *c) {
  if (!c) return;
  if (atomic_fetch_sub_explicit(&c->refs, 1u, memory_order_acq_rel) != 1u) return;
  pthread_mutex_destroy(&c->m);
  free(c->slabs);
  free(c->keys);
  free(c->use);
  free(c->pin);
  free(c->hidx);
  free(c);
}

static cv_cache *cvc_new(uint32_t nslots) {
  if (nslots < 8u) nslots = 8u;
  if (nslots > (1u << 20)) nslots = 1u << 20;
  cv_cache *c = calloc(1, sizeof *c);
  if (!c) return NULL;
  atomic_init(&c->refs, 1u);
  pthread_mutex_init(&c->m, NULL);
  c->nslots = nslots;
  c->slabs = malloc((size_t)nslots * CV_RAW);
  c->keys = malloc((size_t)nslots * sizeof *c->keys);
  c->use = calloc(nslots, sizeof *c->use);
  c->pin = calloc(nslots, sizeof *c->pin);
  uint32_t hs = 64;
  while (hs < nslots * 4u) hs *= 2u;
  c->hidx = calloc(hs, sizeof *c->hidx);
  c->hmask = hs - 1u;
  if (!c->slabs || !c->keys || !c->use || !c->pin || !c->hidx) {
    cvc_unref(c);
    return NULL;
  }
  for (uint32_t i = 0; i < nslots; i++) c->keys[i] = UINT64_MAX;
  return c;
}

/* one lease per thread: the brick a sampler is currently reading. The lease
 * holds a pool reference, so releasing it never touches the r3d_cpuvol and
 * is safe after close; a freed pool address can never be mistaken for a
 * live one because holding the lease is what keeps the pool allocated. */
typedef struct cv_lease {
  cv_cache *c;
  const uint8_t *ptr;
  uint64_t key;
  uint32_t slot;
} cv_lease;

static _Thread_local cv_lease cv_ls = {NULL, NULL, UINT64_MAX, UINT32_MAX};
static _Thread_local uint8_t *cv_scratch = NULL;
static _Thread_local bool cv_tls_hooked = false;
/* negative memo: (volume, open id, key) that resolved to air. Keyed on the
 * open id so a memo cannot survive close/reopen at the same address. */
static _Thread_local const r3d_cpuvol *cv_nvol = NULL;
static _Thread_local uint64_t cv_nid = 0, cv_nkey = UINT64_MAX;
static pthread_key_t cv_tls_key;
static pthread_once_t cv_tls_once = PTHREAD_ONCE_INIT;
static _Atomic uint64_t cv_next_id = 1;

static void cv_lease_drop(void) {
  cv_cache *c = cv_ls.c;
  if (!c) return;
  pthread_mutex_lock(&c->m);
  if (cv_ls.slot < c->nslots && c->pin[cv_ls.slot]) c->pin[cv_ls.slot]--;
  pthread_mutex_unlock(&c->m);
  cv_ls.c = NULL;
  cv_ls.ptr = NULL;
  cv_ls.key = UINT64_MAX;
  cv_ls.slot = UINT32_MAX;
  cvc_unref(c);
}

/* c->m held, and any lease this thread still holds is on c */
static void cv_lease_take(cv_cache *c, uint32_t slot, uint64_t key, const uint8_t *p) {
  if (cv_ls.c == c) {
    if (cv_ls.slot < c->nslots && c->pin[cv_ls.slot]) c->pin[cv_ls.slot]--;
  } else {
    atomic_fetch_add_explicit(&c->refs, 1u, memory_order_relaxed);
  }
  c->pin[slot]++;
  cv_ls.c = c;
  cv_ls.ptr = p;
  cv_ls.key = key;
  cv_ls.slot = slot;
  cv_nvol = NULL; /* moving on: do not let an air memo outlive its probe */
}

static void cv_tls_exit(void *unused) {
  (void)unused;
  cv_lease_drop();
  free(cv_scratch);
  cv_scratch = NULL;
  cv_tls_hooked = false;
}
static void cv_tls_init(void) { pthread_key_create(&cv_tls_key, cv_tls_exit); }
/* thread exit must release the lease and the decode scratch, or a detached
 * sampler would pin one slot of the pool forever */
static void cv_tls_hook(void) {
  if (cv_tls_hooked) return;
  pthread_once(&cv_tls_once, cv_tls_init);
  pthread_setspecific(cv_tls_key, (void *)1);
  cv_tls_hooked = true;
}

typedef struct cv_reader {
  c5d_shard_reader sr;
  bool open, failed;
} cv_reader;

static int cv_u64_triplet(const char *p, uint64_t out[3]) {
  const char *b = strchr(p, '[');
  if (!b) return -1;
  return sscanf(b + 1, " %llu , %llu , %llu", (unsigned long long *)&out[0],
                (unsigned long long *)&out[1], (unsigned long long *)&out[2]) == 3
             ? 0
             : -1;
}

int r3d_cpuvol_open(r3d_cpuvol *v, const char *root, uint32_t cache_bricks) {
  return r3d_cpuvol_open_ex(v, root, cache_bricks, true);
}

int r3d_cpuvol_open_ex(r3d_cpuvol *v, const char *root, uint32_t cache_bricks,
                       bool allow_predict) {
  memset(v, 0, sizeof *v);
  pthread_mutex_init(&v->mu, NULL);
  pthread_mutex_init(&v->io_mu, NULL);
  snprintf(v->root, sizeof v->root, "%s", root);
  char mp[1200];
  snprintf(mp, sizeof mp, "%s/manifest.json", root);
  FILE *f = fopen(mp, "rb");
  if (!f) return -1;
  char json[65536] = {0};
  size_t jn = fread(json, 1, sizeof json - 1, f);
  fclose(f);
  (void)jn;
  if (!strstr(json, "\"format\": \"render3d.c5d-lod.v1\"")) return -1;
  const char *shape = strstr(json, "\"shape\"");
  uint64_t base[3];
  if (!shape || cv_u64_triplet(shape, base) != 0) return -1;
  v->nz = base[0];
  v->ny = base[1];
  v->nx = base[2]; /* manifest order z,y,x */
  const char *p = strstr(json, "\"levels\"");
  if (!p) return -1;
  uint32_t nread = 0;
  while (v->nlev < R3D_CPUVOL_LEVELS && (p = strstr(p, "\"level\""))) {
    const char *shp = strstr(p, "\"shape\"");
    const char *shd = strstr(p, "\"shards\"");
    uint64_t vd[3], sd[3];
    if (!shp || !shd || cv_u64_triplet(shp, vd) != 0 || cv_u64_triplet(shd, sd) != 0)
      return -1;
    r3d_cpuvol_level *l = &v->lev[v->nlev];
    l->scale = 1u << v->nlev;
    l->vz = (uint32_t)vd[0];
    l->vy = (uint32_t)vd[1];
    l->vx = (uint32_t)vd[2];
    l->bx = (l->vx + CV_BRICK - 1) / CV_BRICK;
    l->by = (l->vy + CV_BRICK - 1) / CV_BRICK;
    l->bz = (l->vz + CV_BRICK - 1) / CV_BRICK;
    l->sz = (uint32_t)sd[0];
    l->sy = (uint32_t)sd[1];
    l->sx = (uint32_t)sd[2];
    l->shard_off = nread;
    nread += l->sx * l->sy * l->sz;
    v->nlev++;
    p += 7;
  }
  if (!v->nlev) return -1;
  v->nreaders = nread;
  v->readers = calloc(nread ? nread : 1, sizeof(cv_reader));
  v->cache = cvc_new(cache_bricks ? cache_bricks : 64);
  v->id = atomic_fetch_add_explicit(&cv_next_id, 1u, memory_order_relaxed);
  v->nneg = 8192;
  v->neg_key = malloc((size_t)v->nneg * sizeof *v->neg_key);
  v->neg_exp = calloc(v->nneg, sizeof *v->neg_exp);
  if (v->neg_key)
    for (uint32_t i = 0; i < v->nneg; i++) v->neg_key[i] = UINT64_MAX;
  if (!v->readers || !v->cache || !v->neg_key || !v->neg_exp) {
    r3d_cpuvol_close(v);
    return -1;
  }
  /* optional net source: chunk fetch config, same file the renderer uses */
  snprintf(mp, sizeof mp, "%s/source.json", root);
  f = fopen(mp, "rb");
  if (f) {
    char sj[16384] = {0};
    size_t sn = fread(sj, 1, sizeof sj - 1, f);
    fclose(f);
    (void)sn;
    const char *up = strstr(sj, "\"url\": \"");
    const char *qp = strstr(sj, "\"quality\": ");
    if (up) {
      up += 8;
      const char *ue = strchr(up, '"');
      if (ue && (size_t)(ue - up) < sizeof v->url) {
        memcpy(v->url, up, (size_t)(ue - up));
        v->url[ue - up] = 0;
      }
      v->q0 = qp ? strtof(qp + 11, NULL) : 2.0f;
      const char *lp = sj;
      for (uint32_t l = 0; l < v->nlev && (lp = strstr(lp, "\"chunk\": ")); l++) {
        v->chsz[l] = (uint32_t)strtoul(lp + 9, NULL, 10);
        const char *rp = strstr(lp, "\"raw\": ");
        v->raw[l] = rp && strncmp(rp + 7, "true", 4) == 0;
        if (v->chsz[l] < 32 || v->chsz[l] > 1024) {
          v->url[0] = 0;
          break;
        }
        if (!cv_cell_ok(v->chsz[l])) {
          fprintf(stderr,
                  "cpuvol: %s L%u chunk edge %u needs a %u^3 assembly cell (cap %u^3) "
                  "- demand fetch disabled for that level\n",
                  root, l, v->chsz[l], cv_cell_dim(v->chsz[l]), CV_MAX_CELL_EDGE);
          v->chsz[l] = 0;
        }
        lp += 9;
      }
      if (v->url[0] && r3d_surfpred_url(v->url) && !allow_predict) {
        v->url[0] = 0; /* plain file reads of a predict tree */
      } else if (v->url[0] && r3d_surfpred_url(v->url)) {
        v->sp = malloc(sizeof *v->sp);
        if (!v->sp || r3d_surfpred_open(v->sp, root) != 0) {
          fprintf(stderr, "cpuvol: predict source in %s unusable\n", root);
          free(v->sp);
          v->sp = NULL;
          v->url[0] = 0;
        } else {
          printf("cpuvol: %s predicts surfaces on demand (CT %s, port %d)\n", root,
                 v->sp->ct_root, v->sp->port);
        }
      }
    }
  }
  return 0;
}

void r3d_cpuvol_close(r3d_cpuvol *v) {
  cv_reader *rd = v->readers;
  if (rd)
    for (uint32_t i = 0; i < v->nreaders; i++)
      if (rd[i].open) c5d_shard_close_reader(&rd[i].sr);
  free(v->readers);
  /* a lease this thread still holds on this pool must go before the volume
   * does; leases held by other threads keep the pool alive on their own */
  if (cv_ls.c && cv_ls.c == v->cache) cv_lease_drop();
  if (cv_nvol == v) cv_nvol = NULL;
  cvc_unref(v->cache);
  v->cache = NULL;
  free(v->neg_key);
  free(v->neg_exp);
  if (v->sp) {
    r3d_surfpred_close(v->sp);
    free(v->sp);
    v->sp = NULL;
  }
  pthread_mutex_destroy(&v->mu);
  pthread_mutex_destroy(&v->io_mu);
  if (v->curl) curl_easy_cleanup(v->curl);
  memset(v, 0, sizeof *v);
}

/* decode brick (li,bx,by,bz) into a cache slot; NULL when absent on disk */
/* --- demand fetch: zarr chunks -> .c5b cache (mirrors ni_worker) --------- */

/* response accumulator: growth is checked and capped at buf.max, so a
 * proxy error page or an endless body aborts the transfer instead of
 * eating the heap. `over` distinguishes that from a transport failure. */
typedef struct cv_buf {
  uint8_t *p;
  size_t n, cap, max;
  bool over;
} cv_buf;

static size_t cv_curl_write(const void *data, size_t sz, size_t nm, void *ud) {
  cv_buf *b = ud;
  if (sz && nm > SIZE_MAX / sz) {
    b->over = true;
    return 0;
  }
  size_t n = sz * nm;
  if (n > b->max || b->n > b->max - n) {
    b->over = true;
    return 0;
  }
  if (n > b->cap - b->n) {
    size_t nc = b->cap ? b->cap : (size_t)(1u << 20);
    while (nc < b->n + n) {
      if (nc > b->max / 2u) {
        nc = b->max;
        break;
      }
      nc *= 2u;
    }
    if (nc < b->n + n) {
      b->over = true;
      return 0;
    }
    uint8_t *np = realloc(b->p, nc);
    if (!np) return 0;
    b->p = np;
    b->cap = nc;
  }
  memcpy(b->p + b->n, data, n);
  b->n += n;
  return n;
}

static int cv_write_file(const char *path, const void *data, size_t n) {
  char tmp[1460];
  int pn = snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
  if (pn < 0 || (size_t)pn >= sizeof tmp) return -1;
  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  int rc = n && fwrite(data, 1, n, f) != n ? -1 : 0;
  if (fclose(f) != 0) rc = -1;
  if (rc == 0 && rename(tmp, path) != 0) rc = -1;
  if (rc != 0) unlink(tmp);
  return rc;
}

/* Fetch the cell owning brick (bx,by,bz) at level li and write every brick
 * of the cell to <root>/bricks/L<li> (.c5b; empty file = absent/air). */
static inline uint32_t cv_hash(uint64_t key) {
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdull;
  key ^= key >> 33;
  return (uint32_t)key;
}

/* hash index maintenance (open addressing, linear probing, backward-shift
 * deletion so tombstones never accumulate) */
static int cvc_find(const cv_cache *c, uint64_t key) {
  uint32_t i = cv_hash(key) & c->hmask;
  for (;;) {
    uint32_t e = c->hidx[i];
    if (!e) return -1;
    if (c->keys[e - 1] == key) return (int)(e - 1);
    i = (i + 1) & c->hmask;
  }
}
static void cvc_hinsert(cv_cache *c, uint32_t slot) {
  uint32_t i = cv_hash(c->keys[slot]) & c->hmask;
  while (c->hidx[i]) i = (i + 1) & c->hmask;
  c->hidx[i] = slot + 1;
}
static void cvc_hremove(cv_cache *c, uint32_t slot) {
  uint32_t i = cv_hash(c->keys[slot]) & c->hmask;
  while (c->hidx[i] != slot + 1) {
    if (!c->hidx[i]) return;
    i = (i + 1) & c->hmask;
  }
  for (;;) { /* backward shift */
    uint32_t j = (i + 1) & c->hmask;
    c->hidx[i] = 0;
    for (;;) {
      uint32_t e = c->hidx[j];
      if (!e) return;
      uint32_t home = cv_hash(c->keys[e - 1]) & c->hmask;
      /* entry at j may move to i if its home is not in (i, j] cyclically */
      bool movable = (i <= j) ? (home <= i || home > j) : (home <= i && home > j);
      if (movable) {
        c->hidx[i] = e;
        i = j;
        break;
      }
      j = (j + 1) & c->hmask;
    }
  }
}
/* c->m held. Least-recently-used slot no reader is leasing, or -1 when
 * every slot is leased - the decode then simply is not cached rather than
 * yanking bytes out from under a sampler. */
static int cvc_victim(const cv_cache *c) {
  uint32_t best = UINT32_MAX;
  uint64_t oldest = UINT64_MAX;
  for (uint32_t s = 0; s < c->nslots; s++)
    if (!c->pin[s] && c->use[s] < oldest) {
      oldest = c->use[s];
      best = s;
    }
  return best == UINT32_MAX ? -1 : (int)best;
}
/* c->m held. Publish `raw` into a free/evictable slot; -1 when none. */
static int cvc_publish(cv_cache *c, uint64_t key, const uint8_t *raw) {
  int victim = cvc_victim(c);
  if (victim < 0) return -1;
  uint32_t vs = (uint32_t)victim;
  if (c->keys[vs] != UINT64_MAX) cvc_hremove(c, vs);
  c->keys[vs] = key;
  memcpy(c->slabs + (size_t)vs * CV_RAW, raw, CV_RAW);
  cvc_hinsert(c, vs);
  return victim;
}

static bool cv_neg_hit(const r3d_cpuvol *v, uint64_t key, uint64_t now_s) {
  uint32_t i = cv_hash(key ^ 0x9e3779b97f4a7c15ull) & (v->nneg - 1u);
  return v->neg_key[i] == key && v->neg_exp[i] > now_s;
}
static void cv_neg_put(r3d_cpuvol *v, uint64_t key, uint64_t exp_s) {
  uint32_t i = cv_hash(key ^ 0x9e3779b97f4a7c15ull) & (v->nneg - 1u);
  v->neg_key[i] = key;
  v->neg_exp[i] = exp_s;
}

#define CV_NEG_TTL_S 30u /* retry a failed fetch/decode after this long */

static void cv_cache_insert(r3d_cpuvol *v, uint64_t key, const uint8_t *raw);
static const uint8_t *cv_brick(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by,
                               uint32_t bz);

static CURL *cv_curl_new(void) {
  CURL *c = curl_easy_init();
  if (!c) return NULL;
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, cv_curl_write);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);
  curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L); /* worker threads */
  return c;
}

/* fetch the cell owning brick (bx,by,bz) with a caller-owned CURL handle;
 * bricks land in the decode cache directly (no encode->decode round trip
 * for the first use) and in the .c5b disk cache for later sessions */
static void cv_net_fetch_h(r3d_cpuvol *v, CURL *curl, uint32_t li, uint32_t bx, uint32_t by,
                           uint32_t bz) {
  if (v->sp) { /* predict source: produce the cell locally (writes the files
                * and inserts the bricks; the caller re-reads the file) */
    r3d_surfpred_cell(v->sp, li, bx, by, bz, NULL, v);
    return;
  }
  if (!v->url[0] || li >= v->nlev || !curl) return;
  if (!cv_cell_ok(v->chsz[li])) return; /* pathological edge: no fetch */
  uint64_t now = (uint64_t)time(NULL);
  pthread_mutex_lock(&v->mu);
  bool cooling = v->net_cool > now;
  pthread_mutex_unlock(&v->mu);
  if (cooling) return;
  uint32_t chsz = v->chsz[li], cell = cv_cell_dim(chsz), cb = cell / CV_BRICK,
           cc = cell / chsz;
  uint32_t cz = bz / cb, cy = by / cb, cx = bx / cb;
  size_t chunk_bytes = (size_t)chsz * chsz * chsz;
  size_t cell_bytes = (size_t)cell * cell * cell;
  /* one compressed chunk can only exceed its raw size by the blosc header */
  size_t body_cap = chunk_bytes + (chunk_bytes >> 3) + 4096u;
  if (body_cap > CV_MAX_BODY_BYTES) body_cap = CV_MAX_BODY_BYTES;
  uint8_t *cellbuf = calloc(1, cell_bytes);
  uint8_t *chunk = malloc(chunk_bytes);
  uint8_t *raw = malloc(CV_RAW);
  cv_buf buf = {NULL, 0, 0, body_cap, false};
  if (!cellbuf || !chunk || !raw) goto done;
  curl_easy_setopt(curl, CURLOPT_MAXFILESIZE_LARGE, (curl_off_t)body_cap);
  bool any = false;
  for (uint32_t icz = 0; icz < cc; icz++)
    for (uint32_t icy = 0; icy < cc; icy++)
      for (uint32_t icx = 0; icx < cc; icx++) {
        char url[1600];
        snprintf(url, sizeof url, "%s/%u/%u/%u/%u", v->url, li, cz * cc + icz,
                 cy * cc + icy, cx * cc + icx);
        long code = 0;
        CURLcode crc = CURLE_OK;
        for (int attempt = 0; attempt < 3; attempt++) {
          buf.n = 0;
          buf.over = false;
          curl_easy_setopt(curl, CURLOPT_URL, url);
          curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
          crc = curl_easy_perform(curl);
          curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
          if (crc == CURLE_FILESIZE_EXCEEDED) buf.over = true;
          if (buf.over) break; /* over the cap: retrying cannot help */
          if (crc == CURLE_OK && (code == 200 || code == 404)) break;
          sleep((unsigned)(1u << attempt));
        }
        if (buf.over) { /* no cache artifact: a sane server must be able to heal it */
          fprintf(stderr, "cpuvol: %s exceeds the %zu-byte response cap - cell abandoned\n",
                  url, body_cap);
          goto done;
        }
        if (!(crc == CURLE_OK && (code == 200 || code == 404))) {
          pthread_mutex_lock(&v->mu);
          v->net_cool = now + 30; /* network trouble: back off, stay air */
          pthread_mutex_unlock(&v->mu);
          goto done;
        }
        if (code == 404) continue; /* zarr: absent chunk is fill value */
        /* Validate before decode: prove a complete blosc header exists, that
         * its declared sizes agree with the body and with the exact voxel
         * count this chunk must hold, and that the decode produced all of
         * it. An HTML proxy page, a truncated frame or a bit-flipped size
         * field must fail the cell, not become durable false air. */
        bool ok;
        if (v->raw[li]) {
          ok = buf.n == chunk_bytes;
          if (ok) memcpy(chunk, buf.p, chunk_bytes);
        } else {
          size_t nb = 0;
          ok = buf.p && buf.n >= CV_BLOSC_HDR && buf.n <= (size_t)INT_MAX &&
               blosc_cbuffer_validate(buf.p, buf.n, &nb) == 0 && nb == chunk_bytes &&
               blosc_decompress_ctx(buf.p, chunk, chunk_bytes, 1) == (int)chunk_bytes;
        }
        if (!ok) { /* fail the whole cell: write no .c5b, no empty marker */
          fprintf(stderr, "cpuvol: %s: malformed chunk payload (%zu bytes) - cell abandoned\n",
                  url, buf.n);
          pthread_mutex_lock(&v->mu);
          if (v->net_cool < now + 5) v->net_cool = now + 5;
          pthread_mutex_unlock(&v->mu);
          goto done;
        }
        for (uint32_t zz = 0; zz < chsz; zz++)
          for (uint32_t yy = 0; yy < chsz; yy++)
            memcpy(cellbuf + (((size_t)icz * chsz + zz) * cell +
                              ((size_t)icy * chsz + yy)) *
                                 cell +
                       (size_t)icx * chsz,
                   chunk + ((size_t)zz * chsz + yy) * chsz, chsz);
        any = true;
      }
  {
    const r3d_cpuvol_level *l = &v->lev[li];
    char dir[1360];
    snprintf(dir, sizeof dir, "%s/bricks", v->root);
    mkdir(dir, 0755);
    snprintf(dir, sizeof dir, "%s/bricks/L%u", v->root, li);
    mkdir(dir, 0755);
    float q = v->q0 / (float)(1u << (li < 3u ? li : 3u));
    if (q < 0.25f) q = 0.25f;
    for (uint32_t sz2 = 0; sz2 < cb; sz2++)
      for (uint32_t sy2 = 0; sy2 < cb; sy2++)
        for (uint32_t sx2 = 0; sx2 < cb; sx2++) {
          uint32_t obz = cz * cb + sz2, oby = cy * cb + sy2, obx = cx * cb + sx2;
          if (obx >= l->bx || oby >= l->by || obz >= l->bz) continue;
          char path[1500];
          snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", v->root, li, obz,
                   oby, obx);
          struct stat st;
          if (stat(path, &st) == 0) continue; /* another fetcher won */
          bool zero = !any;
          if (any) {
            for (uint32_t rr = 0; rr < CV_BRICK; rr++)
              for (uint32_t qq = 0; qq < CV_BRICK; qq++)
                memcpy(raw + ((size_t)rr * CV_BRICK + qq) * CV_BRICK,
                       cellbuf + (((size_t)(sz2 * CV_BRICK + rr) * cell +
                                   sy2 * CV_BRICK + qq) *
                                      cell +
                                  sx2 * CV_BRICK),
                       CV_BRICK);
            zero = true;
            for (size_t k = 0; k < (size_t)CV_RAW; k++)
              if (raw[k]) {
                zero = false;
                break;
              }
          }
          if (zero) {
            cv_write_file(path, NULL, 0);
            continue;
          }
          cv_cache_insert(v, ((uint64_t)li << 60) | ((uint64_t)obz << 40) |
                                 ((uint64_t)oby << 20) | obx,
                          raw);
          c5d_brick_params bp = c5d_brick_defaults(1.0f);
          bp.q = q;
          uint8_t *enc = NULL;
          size_t en = 0;
          if (c5d_brick_encode(&bp, raw, CV_BRICK, &enc, &en) == 0) {
            cv_write_file(path, enc, en);
            free(enc);
          }
        }
  }
done:
  free(cellbuf);
  free(chunk);
  free(raw);
  free(buf.p);
}

static void cv_net_fetch(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz) {
  if (!v->curl) v->curl = cv_curl_new();
  cv_net_fetch_h(v, v->curl, li, bx, by, bz);
}

/* insert a decoded/raw brick into the LRU (thread-safe); no-op if present */
static void cv_cache_insert(r3d_cpuvol *v, uint64_t key, const uint8_t *raw) {
  cv_cache *c = v->cache;
  if (!c) return;
  pthread_mutex_lock(&c->m);
  if (cvc_find(c, key) < 0) {
    int slot = cvc_publish(c, key, raw); /* -1: every slot leased, skip */
    if (slot >= 0) c->use[slot] = ++c->tick;
  }
  pthread_mutex_unlock(&c->m);
  /* a positive result overrides any negative entry for this key */
  pthread_mutex_lock(&v->mu);
  uint32_t ni = cv_hash(key ^ 0x9e3779b97f4a7c15ull) & (v->nneg - 1u);
  if (v->neg_key[ni] == key) v->neg_key[ni] = UINT64_MAX;
  pthread_mutex_unlock(&v->mu);
}

void r3d_cpuvol_cache_put(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz,
                          const uint8_t *raw) {
  uint64_t key = ((uint64_t)li << 60) | ((uint64_t)bz << 40) | ((uint64_t)by << 20) | bx;
  cv_cache_insert(v, key, raw);
}

void r3d_cpuvol_read_block(r3d_cpuvol *v, uint32_t li, int64_t x0, int64_t y0, int64_t z0,
                           uint32_t nx, uint32_t ny, uint32_t nz, uint8_t *out) {
  memset(out, 0, (size_t)nx * ny * nz);
  if (li >= v->nlev) return;
  const r3d_cpuvol_level *l = &v->lev[li];
  int64_t bx0 = x0 < 0 ? 0 : x0 / CV_BRICK, bx1 = (x0 + nx - 1) / CV_BRICK;
  int64_t by0 = y0 < 0 ? 0 : y0 / CV_BRICK, by1 = (y0 + ny - 1) / CV_BRICK;
  int64_t bz0 = z0 < 0 ? 0 : z0 / CV_BRICK, bz1 = (z0 + nz - 1) / CV_BRICK;
  if (bx1 >= (int64_t)l->bx) bx1 = (int64_t)l->bx - 1;
  if (by1 >= (int64_t)l->by) by1 = (int64_t)l->by - 1;
  if (bz1 >= (int64_t)l->bz) bz1 = (int64_t)l->bz - 1;
  for (int64_t bz = bz0; bz <= bz1; bz++)
    for (int64_t by = by0; by <= by1; by++)
      for (int64_t bx = bx0; bx <= bx1; bx++) {
        const uint8_t *b = cv_brick(v, li, (uint32_t)bx, (uint32_t)by, (uint32_t)bz);
        if (!b) continue;
        /* overlap of this brick with the block, in block coordinates */
        int64_t ox0 = bx * CV_BRICK, oy0 = by * CV_BRICK, oz0 = bz * CV_BRICK;
        int64_t sx0 = ox0 > x0 ? ox0 : x0, sx1 = ox0 + CV_BRICK < x0 + nx ? ox0 + CV_BRICK : x0 + nx;
        int64_t sy0 = oy0 > y0 ? oy0 : y0, sy1 = oy0 + CV_BRICK < y0 + ny ? oy0 + CV_BRICK : y0 + ny;
        int64_t sz0 = oz0 > z0 ? oz0 : z0, sz1 = oz0 + CV_BRICK < z0 + nz ? oz0 + CV_BRICK : z0 + nz;
        if (sx1 <= sx0 || sy1 <= sy0 || sz1 <= sz0) continue;
        /* the volume's valid extent (partial edge bricks) */
        int64_t vx1 = (int64_t)l->vx, vy1 = (int64_t)l->vy, vz1 = (int64_t)l->vz;
        if (sx1 > vx1) sx1 = vx1;
        if (sy1 > vy1) sy1 = vy1;
        if (sz1 > vz1) sz1 = vz1;
        for (int64_t z = sz0; z < sz1; z++)
          for (int64_t y = sy0; y < sy1; y++)
            memcpy(out + ((size_t)(z - z0) * ny + (size_t)(y - y0)) * nx + (size_t)(sx0 - x0),
                   b + ((size_t)(z - oz0) * CV_BRICK + (size_t)(y - oy0)) * CV_BRICK +
                       (size_t)(sx0 - ox0),
                   (size_t)(sx1 - sx0));
      }
}

/* ---- parallel prefetch of an explicit brick list ---- */
struct cv_pf {
  r3d_cpuvol *v;
  uint32_t li;
  const uint64_t *cells; /* cz<<40 | cy<<20 | cx */
  uint32_t n;
  _Atomic uint32_t next;
  uint32_t cb;
};
static void *cv_pf_thread(void *ud) {
  struct cv_pf *j = ud;
  CURL *curl = cv_curl_new();
  for (;;) {
    uint32_t i = atomic_fetch_add(&j->next, 1);
    if (i >= j->n) break;
    uint64_t c = j->cells[i];
    uint32_t cx = (uint32_t)(c & 0xfffffu), cy = (uint32_t)((c >> 20) & 0xfffffu),
             cz = (uint32_t)(c >> 40);
    cv_net_fetch_h(j->v, curl, j->li, cx * j->cb, cy * j->cb, cz * j->cb);
  }
  if (curl) curl_easy_cleanup(curl);
  return NULL;
}

int r3d_cpuvol_prefetch(r3d_cpuvol *v, uint32_t li, const uint32_t *bxyz, uint32_t n,
                        uint32_t threads) {
  if (!v->url[0] || li >= v->nlev || !n || !v->cache) return 0;
  if (!cv_cell_ok(v->chsz[li])) return 0;
  uint32_t chsz = v->chsz[li], cell = cv_cell_dim(chsz), cb = cell / CV_BRICK;
  if (!cb) return 0;
  uint64_t *cells = malloc((size_t)n * sizeof *cells);
  if (!cells) return -1;
  uint32_t nc = 0;
  uint64_t now_s = (uint64_t)time(NULL);
  for (uint32_t i = 0; i < n; i++) {
    uint32_t bx = bxyz[i * 3], by = bxyz[i * 3 + 1], bz = bxyz[i * 3 + 2];
    uint64_t key = ((uint64_t)li << 60) | ((uint64_t)bz << 40) | ((uint64_t)by << 20) | bx;
    cv_cache *cch = v->cache;
    pthread_mutex_lock(&cch->m);
    bool have = cvc_find(cch, key) >= 0;
    pthread_mutex_unlock(&cch->m);
    if (!have) {
      pthread_mutex_lock(&v->mu);
      have = cv_neg_hit(v, key, now_s);
      pthread_mutex_unlock(&v->mu);
    }
    if (have) continue;
    char path[1400];
    snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", v->root, li, bz, by, bx);
    struct stat st;
    if (stat(path, &st) == 0) continue; /* on disk: the sampler decodes it */
    uint64_t c = ((uint64_t)(bz / cb) << 40) | ((uint64_t)(by / cb) << 20) | (bx / cb);
    bool dup = false;
    for (uint32_t k = 0; k < nc && !dup; k++) dup = cells[k] == c;
    if (!dup) cells[nc++] = c;
  }
  if (!nc) {
    free(cells);
    return 0;
  }
  struct cv_pf job = {.v = v, .li = li, .cells = cells, .n = nc, .cb = cb};
  atomic_store(&job.next, 0);
  if (threads < 1) threads = 1;
  if (threads > 16) threads = 16;
  if (threads > nc) threads = nc;
  pthread_t th[16];
  uint32_t spawned = 0;
  for (uint32_t t = 0; t + 1 < threads; t++)
    if (pthread_create(&th[spawned], NULL, cv_pf_thread, &job) == 0) spawned++;
  cv_pf_thread(&job);
  for (uint32_t t = 0; t < spawned; t++) pthread_join(th[t], NULL);
  free(cells);
  return (int)nc;
}


/* Thread-safe brick lookup returning a pinned, immutable brick. Hits: the
 * thread's own lease (no lock at all), else a hash probe under the pool
 * lock that leases the slot. Misses: blob IO under v->io_mu (shard readers
 * and the net fetch are not reentrant), decode OUTSIDE every lock into
 * thread-local scratch, then publish under the pool lock (re-checking:
 * another thread may have landed the same brick).
 *
 * The returned pointer stays valid and unmodified until this thread asks
 * for a different brick: the lease pins the slot and eviction only takes
 * unpinned slots. Consumers must therefore finish with one brick before
 * requesting the next, which every consumer here does. */
static void cv_memo_null(const r3d_cpuvol *v, uint64_t key) {
  cv_nvol = v;
  cv_nid = v->id;
  cv_nkey = key;
}

static const uint8_t *cv_brick(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by,
                               uint32_t bz) {
  uint64_t key = ((uint64_t)li << 60) | ((uint64_t)bz << 40) | ((uint64_t)by << 20) | bx;
  cv_cache *c = v->cache;
  if (!c) return NULL;
  if (cv_ls.c == c && cv_ls.key == key) return cv_ls.ptr; /* hot path: pinned */
  if (cv_nvol == v && cv_nid == v->id && cv_nkey == key) return NULL;
  if (cv_ls.c && cv_ls.c != c) cv_lease_drop(); /* lease only ever spans one pool */
  cv_tls_hook();

  pthread_mutex_lock(&c->m);
  int hs = cvc_find(c, key);
  if (hs >= 0) {
    c->use[hs] = ++c->tick;
    const uint8_t *hit = c->slabs + (size_t)hs * CV_RAW;
    cv_lease_take(c, (uint32_t)hs, key, hit);
    pthread_mutex_unlock(&c->m);
    return hit;
  }
  pthread_mutex_unlock(&c->m);

  uint64_t now_s = (uint64_t)time(NULL);
  pthread_mutex_lock(&v->mu);
  bool absent = cv_neg_hit(v, key, now_s);
  pthread_mutex_unlock(&v->mu);
  if (absent) { /* known absent: air */
    cv_memo_null(v, key);
    return NULL;
  }

  /* ---- miss: source the blob (serialized IO) ---- */
  const r3d_cpuvol_level *l = &v->lev[li];
  const uint8_t *blob = NULL;
  size_t bn = 0;
  uint8_t *owned = NULL;
  bool empty_file = false;
  pthread_mutex_lock(&v->io_mu);
  uint32_t sx = bx / CV_SHARD_BPA, sy = by / CV_SHARD_BPA, sz = bz / CV_SHARD_BPA;
  if (sx < l->sx && sy < l->sy && sz < l->sz) {
    cv_reader *rd =
        (cv_reader *)v->readers + l->shard_off + (sz * l->sy + sy) * l->sx + sx;
    if (!rd->open && !rd->failed) {
      char path[1400];
      snprintf(path, sizeof path, "%s/c5d/L%u/%u_%u_%u.c5s", v->root, li, sz, sy, sx);
      if (c5d_shard_open(path, &rd->sr) == 0 && rd->sr.foot.brick_dim == CV_BRICK &&
          rd->sr.foot.shard_dim == 1024u)
        rd->open = true;
      else
        rd->failed = true;
    }
    if (rd->open) {
      uint32_t bi = ((bz % CV_SHARD_BPA) * CV_SHARD_BPA + (by % CV_SHARD_BPA)) *
                        CV_SHARD_BPA +
                    (bx % CV_SHARD_BPA);
      const uint8_t *sb = c5d_shard_brick(&rd->sr, bi, &bn);
      if (sb && bn) { /* copy out: the reader's buffer is not ours to keep
                       * past the lock */
        owned = malloc(bn);
        if (owned) {
          memcpy(owned, sb, bn);
          blob = owned;
        }
      }
    }
  }
  if (!blob) { /* net-ingest cache file (empty = absent/air) */
    char path[1400];
    snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.c5b", v->root, li, bz, by, bx);
    FILE *bf = fopen(path, "rb");
    if (!bf && v->url[0]) { /* never fetched: pull the owning cell now */
      cv_net_fetch(v, li, bx, by, bz);
      bf = fopen(path, "rb");
    }
    if (bf) {
      fseek(bf, 0, SEEK_END);
      long fn = ftell(bf);
      fseek(bf, 0, SEEK_SET);
      if (fn > 0) {
        owned = malloc((size_t)fn);
        if (owned && fread(owned, 1, (size_t)fn, bf) == (size_t)fn) {
          blob = owned;
          bn = (size_t)fn;
        }
      } else {
        empty_file = true; /* the ingest wrote an empty marker: air/absent */
      }
      fclose(bf);
    }
  }
  pthread_mutex_unlock(&v->io_mu);
  if (!blob) {
    free(owned);
    pthread_mutex_lock(&v->mu);
    cv_neg_put(v, key, empty_file ? UINT64_MAX : now_s + CV_NEG_TTL_S);
    pthread_mutex_unlock(&v->mu);
    cv_memo_null(v, key);
    return NULL;
  }

  /* ---- decode outside the locks ---- */
  if (!cv_scratch) cv_scratch = malloc(CV_RAW);
  bool ok = cv_scratch && c5d_brick_decode(blob, bn, cv_scratch, CV_BRICK) == 0;
  free(owned);
  if (!ok) { /* remember the failure briefly; retry when the TTL lapses */
    pthread_mutex_lock(&v->mu);
    cv_neg_put(v, key, now_s + CV_NEG_TTL_S);
    pthread_mutex_unlock(&v->mu);
    return NULL;
  }
  pthread_mutex_lock(&c->m);
  hs = cvc_find(c, key); /* raced with another thread's insert? */
  if (hs < 0) {
    hs = cvc_publish(c, key, cv_scratch);
    if (hs < 0) { /* every slot leased: serve this decode uncached */
      pthread_mutex_unlock(&c->m);
      return cv_scratch;
    }
  }
  c->use[hs] = ++c->tick;
  const uint8_t *got = c->slabs + (size_t)hs * CV_RAW;
  cv_lease_take(c, (uint32_t)hs, key, got);
  pthread_mutex_unlock(&c->m);
  return got;
}

static double cv_vox(r3d_cpuvol *v, uint32_t li, int64_t lx, int64_t ly, int64_t lz) {
  const r3d_cpuvol_level *l = &v->lev[li];
  if (lx < 0 || ly < 0 || lz < 0 || lx >= (int64_t)l->vx || ly >= (int64_t)l->vy ||
      lz >= (int64_t)l->vz)
    return 0.0;
  const uint8_t *b =
      cv_brick(v, li, (uint32_t)lx / CV_BRICK, (uint32_t)ly / CV_BRICK, (uint32_t)lz / CV_BRICK);
  if (!b) return 0.0;
  uint32_t ox = (uint32_t)lx % CV_BRICK, oy = (uint32_t)ly % CV_BRICK,
           oz = (uint32_t)lz % CV_BRICK;
  return (double)b[((size_t)oz * CV_BRICK + oy) * CV_BRICK + ox];
}

double r3d_cpuvol_tri(r3d_cpuvol *v, uint32_t li, const double p[3], double grad[3]) {
  if (grad) grad[0] = grad[1] = grad[2] = 0.0;
  if (li >= v->nlev) return 0.0;
  const r3d_cpuvol_level *l = &v->lev[li];
  double lx = p[0] / l->scale, ly = p[1] / l->scale, lz = p[2] / l->scale;
  double fx = floor(lx), fy = floor(ly), fz = floor(lz);
  double tx = lx - fx, ty = ly - fy, tz = lz - fz;
  int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz;
  double c[2][2][2];
  for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
      for (int dx = 0; dx < 2; dx++)
        c[dz][dy][dx] = cv_vox(v, li, ix + dx, iy + dy, iz + dz);
  double c00 = c[0][0][0] * (1 - tx) + c[0][0][1] * tx;
  double c01 = c[0][1][0] * (1 - tx) + c[0][1][1] * tx;
  double c10 = c[1][0][0] * (1 - tx) + c[1][0][1] * tx;
  double c11 = c[1][1][0] * (1 - tx) + c[1][1][1] * tx;
  double c0 = c00 * (1 - ty) + c01 * ty;
  double c1 = c10 * (1 - ty) + c11 * ty;
  if (grad) {
    double gx0 = (c[0][0][1] - c[0][0][0]) * (1 - ty) + (c[0][1][1] - c[0][1][0]) * ty;
    double gx1 = (c[1][0][1] - c[1][0][0]) * (1 - ty) + (c[1][1][1] - c[1][1][0]) * ty;
    grad[0] = (gx0 * (1 - tz) + gx1 * tz) / l->scale;
    grad[1] = ((c01 - c00) * (1 - tz) + (c11 - c10) * tz) / l->scale;
    grad[2] = (c1 - c0) / l->scale;
  }
  return c0 * (1 - tz) + c1 * tz;
}

uint8_t r3d_cpuvol_at(r3d_cpuvol *v, uint32_t li, double x, double y, double z) {
  if (li >= v->nlev || x < 0.0 || y < 0.0 || z < 0.0) return 0;
  const r3d_cpuvol_level *l = &v->lev[li];
  uint32_t lx = (uint32_t)(x / l->scale), ly = (uint32_t)(y / l->scale),
           lz = (uint32_t)(z / l->scale);
  if (lx >= l->vx || ly >= l->vy || lz >= l->vz) return 0;
  const uint8_t *b = cv_brick(v, li, lx / CV_BRICK, ly / CV_BRICK, lz / CV_BRICK);
  if (!b) return 0;
  uint32_t ox = lx % CV_BRICK, oy = ly % CV_BRICK, oz = lz % CV_BRICK;
  return b[((size_t)oz * CV_BRICK + oy) * CV_BRICK + ox];
}
