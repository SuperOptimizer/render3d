#include "core/thread.h"
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

#include "native_source.h"
#include <brick.h> /* volcomp brick decode */
#include <shard.h> /* volcomp .vcs reader */

#define CV_BRICK 128u /* storage/fetch chunk */
#define CV_BLOCK 16u
#define CV_BLOCK_RAW 4096u
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

/* Decode cache: a refcounted pool of 16^3 blocks shared by every sampler
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
  uint32_t *pin;  /* live leases per slot; eviction skips nonzero */
  uint32_t nslots;
  uint32_t *prev, *next, head, tail; /* unpinned slots, oldest release first */
  _Atomic uint64_t decoded_blocks;
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
  free(c->pin);
  free(c->prev);
  free(c->next);
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
  c->slabs = malloc((size_t)nslots * CV_BLOCK_RAW);
  c->keys = malloc((size_t)nslots * sizeof *c->keys);
  c->pin = calloc(nslots, sizeof *c->pin);
  c->prev = malloc((size_t)nslots * sizeof *c->prev);
  c->next = malloc((size_t)nslots * sizeof *c->next);
  uint32_t hs = 64;
  while (hs < nslots * 4u) hs *= 2u;
  c->hidx = calloc(hs, sizeof *c->hidx);
  c->hmask = hs - 1u;
  if (!c->slabs || !c->keys || !c->pin || !c->hidx || !c->prev || !c->next) {
    cvc_unref(c);
    return NULL;
  }
  for (uint32_t i = 0; i < nslots; i++) {
    c->keys[i] = UINT64_MAX;
    c->prev[i] = i ? i - 1 : UINT32_MAX;
    c->next[i] = i + 1 < nslots ? i + 1 : UINT32_MAX;
  }
  c->head = 0; c->tail = nslots - 1;
  return c;
}

static void cvc_unlink(cv_cache *c, uint32_t s) {
  if (c->prev[s] != UINT32_MAX) c->next[c->prev[s]] = c->next[s];
  else c->head = c->next[s];
  if (c->next[s] != UINT32_MAX) c->prev[c->next[s]] = c->prev[s];
  else c->tail = c->prev[s];
}
static void cvc_append(cv_cache *c, uint32_t s) {
  c->prev[s] = c->tail; c->next[s] = UINT32_MAX;
  if (c->tail != UINT32_MAX) c->next[c->tail] = s;
  else c->head = s;
  c->tail = s;
}
static void cvc_unpin(cv_cache *c, uint32_t s) {
  if (s < c->nslots && c->pin[s] && --c->pin[s] == 0) cvc_append(c, s);
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
static _Thread_local CURL *cv_thread_curl = NULL;
static _Thread_local bool cv_tls_hooked = false;
/* negative memo: (volume, open id, key) that resolved to air. Keyed on the
 * open id so a memo cannot survive close/reopen at the same address. */
static _Thread_local const r3d_cpuvol *cv_nvol = NULL;
static _Thread_local uint64_t cv_nid = 0, cv_nkey = UINT64_MAX, cv_nexp = 0;
static pthread_key_t cv_tls_key;
static pthread_once_t cv_tls_once = PTHREAD_ONCE_INIT;
static _Atomic uint64_t cv_next_id = 1;

static void cv_lease_drop(void) {
  cv_cache *c = cv_ls.c;
  if (!c) return;
  pthread_mutex_lock(&c->m);
  cvc_unpin(c, cv_ls.slot);
  pthread_mutex_unlock(&c->m);
  cv_ls.c = NULL;
  cv_ls.ptr = NULL;
  cv_ls.key = UINT64_MAX;
  cv_ls.slot = UINT32_MAX;
  cvc_unref(c);
}

void r3d_cpuvol_release_thread(r3d_cpuvol *v) {
  if (v && cv_ls.c == v->cache) cv_lease_drop();
}

/* c->m held, and any lease this thread still holds is on c */
static void cv_lease_take(cv_cache *c, uint32_t slot, uint64_t key, const uint8_t *p) {
  if (cv_ls.c == c) {
    cvc_unpin(c, cv_ls.slot);
  } else {
    atomic_fetch_add_explicit(&c->refs, 1u, memory_order_relaxed);
  }
  if (!c->pin[slot]) cvc_unlink(c, slot);
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
  if (cv_thread_curl) curl_easy_cleanup(cv_thread_curl);
  cv_thread_curl = NULL;
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

#define CV_FILES 16u
#define CV_FETCHES 64u
/* Covers upstream VOLCOMP_ENCODE_BOUND (14,681,264 bytes), at most256MiB. */
#define CV_FILE_BYTES ((size_t)16u << 20)
typedef struct cv_file {
  pthread_mutex_t mu;
  uint64_t key;
  struct stat identity;
  uint8_t *data;
  size_t n;
} cv_file;
typedef struct cv_files {
  cv_file slot[CV_FILES];
  pthread_mutex_t fetch[CV_FETCHES];
  _Atomic uint64_t reads, bytes, hits;
} cv_files;

typedef struct cv_reader {
  pthread_mutex_t mu;
  volcomp_shard_reader sr;
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

int r3d_cpuvol_open(r3d_cpuvol *v, const char *root, uint32_t cache_blocks) {
  return r3d_cpuvol_open_ex(v, root, cache_blocks, true);
}

int r3d_cpuvol_open_ex(r3d_cpuvol *v, const char *root, uint32_t cache_blocks,
                       bool allow_predict) {
  memset(v, 0, sizeof *v);
  pthread_mutex_init(&v->mu, NULL);
  char mp[1200];
  struct stat input;
  if (stat(root,&input)==0 && S_ISREG(input.st_mode)) {
    if (strlen(root)>=sizeof mp) return -1;
    snprintf(mp,sizeof mp,"%s",root);
    const char *slash=strrchr(root,'/');
    size_t len=slash ? (size_t)(slash-root) : 0;
    if (len>=sizeof v->root) return -1;
    if (slash) { memcpy(v->root,root,len); if(!len)v->root[len++]='/';v->root[len]=0; }
    else snprintf(v->root,sizeof v->root,".");
  } else {
    snprintf(v->root, sizeof v->root, "%s", root);
    snprintf(mp, sizeof mp, "%s/manifest.json", root);
  }
  root=v->root;
  FILE *f = fopen(mp, "rb");
  if (!f) return -1;
  char json[65536] = {0};
  size_t jn = fread(json, 1, sizeof json - 1, f);
  fclose(f);
  (void)jn;
  const char *format=strstr(json,"\"format\"");
  format=format ? strchr(format,':') : NULL;
  if (!format) return -1;
  do { format++; } while (*format==' ' || *format=='\t' || *format=='\n' || *format=='\r');
  static const char expected_format[]="\"render3d.volcomp-lod.v1\"";
  if (strncmp(format,expected_format,sizeof expected_format-1)!=0) return -1;
  const char *shape = strstr(json, "\"shape\"");
  uint64_t base[3];
  if (!shape || cv_u64_triplet(shape, base) != 0) return -1;
  v->nz = base[0];
  v->ny = base[1];
  v->nx = base[2]; /* manifest order z,y,x */
  const char *p = strstr(json, "\"levels\"");
  if (!p) return -1;
  uint32_t nread = 0;
  uint64_t nblocks = 0;
  while (v->nlev < R3D_CPUVOL_LEVELS && (p = strstr(p, "\"level\""))) {
    const char *shp = strstr(p, "\"shape\"");
    const char *shd = strstr(p, "\"shards\"");
    uint64_t vd[3], sd[3];
    if (!shp || !shd || cv_u64_triplet(shp, vd) != 0 || cv_u64_triplet(shd, sd) != 0)
      return -1;
    r3d_cpuvol_level *l = &v->lev[v->nlev];
    if(vd[0]>UINT32_MAX || vd[1]>UINT32_MAX || vd[2]>UINT32_MAX ||
       sd[0]>UINT32_MAX || sd[1]>UINT32_MAX || sd[2]>UINT32_MAX)return -1;
    l->scale = 1u << v->nlev;
    l->vz = (uint32_t)vd[0];
    l->vy = (uint32_t)vd[1];
    l->vx = (uint32_t)vd[2];
    l->bx = (uint32_t)(((uint64_t)l->vx + CV_BRICK - 1) / CV_BRICK);
    l->gx = (uint32_t)(((uint64_t)l->vx + CV_BLOCK - 1) / CV_BLOCK);
    l->by = (uint32_t)(((uint64_t)l->vy + CV_BRICK - 1) / CV_BRICK);
    l->gy = (uint32_t)(((uint64_t)l->vy + CV_BLOCK - 1) / CV_BLOCK);
    l->bz = (uint32_t)(((uint64_t)l->vz + CV_BRICK - 1) / CV_BRICK);
    l->gz = (uint32_t)(((uint64_t)l->vz + CV_BLOCK - 1) / CV_BLOCK);
    l->sz = (uint32_t)sd[0];
    l->sy = (uint32_t)sd[1];
    l->sx = (uint32_t)sd[2];
    l->shard_off = nread;
    uint64_t plane=(uint64_t)l->gx*l->gy,shards=(uint64_t)l->sx*l->sy;
    if(!l->gx || !l->gy || !l->gz || !l->sx || !l->sy || !l->sz ||
       plane>(UINT64_MAX-1u-nblocks)/l->gz || shards>((4u<<20)-nread)/l->sz)return -1;
    l->block_off=nblocks;nblocks+=plane*l->gz;
    nread+=(uint32_t)(shards*l->sz);
    v->nlev++;
    p += 7;
  }
  if (!v->nlev) return -1;
  v->nreaders = nread;
  v->readers = calloc(nread ? nread : 1, sizeof(cv_reader));
  if (v->readers)
    for (uint32_t i = 0; i < nread; i++) pthread_mutex_init(&((cv_reader *)v->readers)[i].mu, NULL);
  cv_files *files = calloc(1, sizeof *files);
  v->files = files;
  if (files)
    for (uint32_t i = 0; i < CV_FILES; i++) {
      pthread_mutex_init(&files->slot[i].mu, NULL);
      files->slot[i].key = UINT64_MAX;
    }
  if (files) for (uint32_t i = 0; i < CV_FETCHES; i++) pthread_mutex_init(&files->fetch[i], NULL);
  v->cache = cvc_new(cache_blocks ? cache_blocks : 4096);
  v->id = atomic_fetch_add_explicit(&cv_next_id, 1u, memory_order_relaxed);
  v->nneg = 8192;
  v->neg_key = malloc((size_t)v->nneg * sizeof *v->neg_key);
  v->neg_exp = calloc(v->nneg, sizeof *v->neg_exp);
  if (v->neg_key)
    for (uint32_t i = 0; i < v->nneg; i++) v->neg_key[i] = UINT64_MAX;
  if (!v->readers || !v->files || !v->cache || !v->neg_key || !v->neg_exp) {
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
      v->native_source = strstr(sj, "\"native_volcomp\": true") != NULL;
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
    for (uint32_t i = 0; i < v->nreaders; i++) {
      if (rd[i].open) volcomp_shard_close_reader(&rd[i].sr);
      pthread_mutex_destroy(&rd[i].mu);
    }
  cv_files *files = v->files;
  if (files) {
    for (uint32_t i = 0; i < CV_FILES; i++) {
      free(files->slot[i].data);
      pthread_mutex_destroy(&files->slot[i].mu);
    }
    for (uint32_t i = 0; i < CV_FETCHES; i++) pthread_mutex_destroy(&files->fetch[i]);
    free(files);
  }
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
  memset(v, 0, sizeof *v);
}

/* decode brick (li,bx,by,bz) into a cache slot; NULL when absent on disk */
/* --- demand fetch: zarr chunks -> .volc cache (mirrors ni_worker) --------- */

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
 * of the cell to <root>/bricks/L<li> (.volc; empty file = absent/air). */
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
/* c->m held. Oldest released slot no reader is leasing, or -1 when
 * every slot is leased - the decode then simply is not cached rather than
 * yanking bytes out from under a sampler. */
static int cvc_victim(const cv_cache *c) {
  return c->head == UINT32_MAX ? -1 : (int)c->head;
}
/* c->m held. Publish `raw` into a free/evictable slot; -1 when none. */
static int cvc_publish(cv_cache *c, uint64_t key, const uint8_t *raw) {
  int victim = cvc_victim(c);
  if (victim < 0) return -1;
  uint32_t vs = (uint32_t)victim;
  cvc_unlink(c, vs);
  cvc_append(c, vs);
  if (c->keys[vs] != UINT64_MAX) cvc_hremove(c, vs);
  c->keys[vs] = key;
  memcpy(c->slabs + (size_t)vs * CV_BLOCK_RAW, raw, CV_BLOCK_RAW);
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
                               uint32_t bz, bool *complete);

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
 * persist compressed chunks for this consumer, renderer and future sessions. */
static void cv_net_fetch_h(r3d_cpuvol *v, CURL *curl, uint32_t li, uint32_t bx, uint32_t by,
                           uint32_t bz) {
  if (v->sp) { /* predict source: produce the cell locally (writes the files
                * and inserts the bricks; the caller re-reads the file) */
    r3d_surfpred_cell(v->sp, li, bx, by, bz, NULL, v);
    return;
  }
  if (!v->url[0] || li >= v->nlev || !curl) return;
  if (v->native_source) {
    (void)r3d_native_fetch(curl,v->url,v->root,li,bx,by,bz,NULL);
    return;
  }
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
        if (!ok) { /* fail the whole cell: write no .volc, no empty marker */
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
    if (q < 1.0f) q = 1.0f;
    for (uint32_t sz2 = 0; sz2 < cb; sz2++)
      for (uint32_t sy2 = 0; sy2 < cb; sy2++)
        for (uint32_t sx2 = 0; sx2 < cb; sx2++) {
          uint32_t obz = cz * cb + sz2, oby = cy * cb + sy2, obx = cx * cb + sx2;
          if (obx >= l->bx || oby >= l->by || obz >= l->bz) continue;
          char path[1500];
          snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.volc", v->root, li, obz,
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
          volcomp_brick_params bp = volcomp_brick_defaults(1.0f);
          bp.q = q;
          uint8_t *enc = NULL;
          size_t en = 0;
          if (volcomp_brick_encode(&bp, raw, CV_BRICK, &enc, &en) == 0) {
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
  cv_files *files = v->files;
  uint32_t cell = cv_cell_dim(v->chsz[li]), cb = cell / CV_BRICK;
  if (!cb) return;
  uint64_t key = ((uint64_t)li << 60) | ((uint64_t)(bz / cb) << 40) |
                 ((uint64_t)(by / cb) << 20) | (bx / cb);
  pthread_mutex_t *lock = &files->fetch[cv_hash(key) % CV_FETCHES];
  pthread_mutex_lock(lock);
  char path[1400];
  snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.volc", v->root, li, bz, by, bx);
  struct stat st;
  if (stat(path, &st) != 0) {
    cv_tls_hook();
    if (!cv_thread_curl) cv_thread_curl = cv_curl_new();
    cv_net_fetch_h(v, cv_thread_curl, li, bx, by, bz);
  }
  pthread_mutex_unlock(lock);
}

/* insert a decoded/raw brick into the LRU (thread-safe); no-op if present */
static void cv_cache_insert(r3d_cpuvol *v, uint64_t key, const uint8_t *raw) {
  cv_cache *c = v->cache;
  if (!c) return;
  pthread_mutex_lock(&c->m);
  if (cvc_find(c, key) < 0) {
    (void)cvc_publish(c, key, raw); /* -1: every slot leased, skip */
  }
  pthread_mutex_unlock(&c->m);
  /* a positive result overrides any negative entry for this key */
  pthread_mutex_lock(&v->mu);
  uint32_t ni = cv_hash(key ^ 0x9e3779b97f4a7c15ull) & (v->nneg - 1u);
  if (v->neg_key[ni] == key) v->neg_key[ni] = UINT64_MAX;
  pthread_mutex_unlock(&v->mu);
}

static uint64_t cv_block_id(const r3d_cpuvol *v,uint32_t li,uint32_t x,uint32_t y,uint32_t z) {
  const r3d_cpuvol_level *l=&v->lev[li];
  return l->block_off+((uint64_t)z*l->gy+y)*l->gx+x;
}
void r3d_cpuvol_cache_put(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by, uint32_t bz,
                          const uint8_t *raw) {
  if (!raw || li >= v->nlev) return;
  uint8_t block[CV_BLOCK_RAW];
  for (uint32_t z = 0; z < 8; z++)
    for (uint32_t y = 0; y < 8; y++)
      for (uint32_t x = 0; x < 8; x++) {
        for (uint32_t rz = 0; rz < 16; rz++)
          for (uint32_t ry = 0; ry < 16; ry++)
            memcpy(block + (rz * 16u + ry) * 16u,
                   raw + ((z * 16u + rz) * 128u + y * 16u + ry) * 128u + x * 16u, 16);
        if(bx*8u+x>=v->lev[li].gx || by*8u+y>=v->lev[li].gy || bz*8u+z>=v->lev[li].gz)continue;
        uint64_t key=cv_block_id(v,li,bx*8u+x,by*8u+y,bz*8u+z);
        cv_cache_insert(v, key, block);
      }
}

static bool cv_read_block(r3d_cpuvol *v, uint32_t li, int64_t x0, int64_t y0, int64_t z0,
                           uint32_t nx, uint32_t ny, uint32_t nz, uint8_t *out, bool status) {
  memset(out, 0, (size_t)nx * ny * nz);
  if (li >= v->nlev) return false;
  bool complete = true;
  const r3d_cpuvol_level *l = &v->lev[li];
  int64_t bx0 = x0 < 0 ? 0 : x0 / CV_BLOCK, bx1 = (x0 + nx - 1) / CV_BLOCK;
  int64_t by0 = y0 < 0 ? 0 : y0 / CV_BLOCK, by1 = (y0 + ny - 1) / CV_BLOCK;
  int64_t bz0 = z0 < 0 ? 0 : z0 / CV_BLOCK, bz1 = (z0 + nz - 1) / CV_BLOCK;
  if (bx1 >= (int64_t)l->gx) bx1 = (int64_t)l->gx - 1;
  if (by1 >= (int64_t)l->gy) by1 = (int64_t)l->gy - 1;
  if (bz1 >= (int64_t)l->gz) bz1 = (int64_t)l->gz - 1;
  for (int64_t bz = bz0; bz <= bz1; bz++)
    for (int64_t by = by0; by <= by1; by++)
      for (int64_t bx = bx0; bx <= bx1; bx++) {
        bool available = true;
        const uint8_t *b = cv_brick(v, li, (uint32_t)bx, (uint32_t)by, (uint32_t)bz, status ? &available : NULL);
        complete = complete && available;
        if (!b) continue;
        /* overlap of this brick with the block, in block coordinates */
        int64_t ox0 = bx * CV_BLOCK, oy0 = by * CV_BLOCK, oz0 = bz * CV_BLOCK;
        int64_t sx0 = ox0 > x0 ? ox0 : x0, sx1 = ox0 + CV_BLOCK < x0 + nx ? ox0 + CV_BLOCK : x0 + nx;
        int64_t sy0 = oy0 > y0 ? oy0 : y0, sy1 = oy0 + CV_BLOCK < y0 + ny ? oy0 + CV_BLOCK : y0 + ny;
        int64_t sz0 = oz0 > z0 ? oz0 : z0, sz1 = oz0 + CV_BLOCK < z0 + nz ? oz0 + CV_BLOCK : z0 + nz;
        if (sx1 <= sx0 || sy1 <= sy0 || sz1 <= sz0) continue;
        /* the volume's valid extent (partial edge bricks) */
        int64_t vx1 = (int64_t)l->vx, vy1 = (int64_t)l->vy, vz1 = (int64_t)l->vz;
        if (sx1 > vx1) sx1 = vx1;
        if (sy1 > vy1) sy1 = vy1;
        if (sz1 > vz1) sz1 = vz1;
        if (sx1 <= sx0 || sy1 <= sy0 || sz1 <= sz0) continue;
        size_t row_bytes=(size_t)(sx1-sx0);
        uint8_t *dst=out+((size_t)(sz0-z0)*ny+(size_t)(sy0-y0))*nx+(size_t)(sx0-x0);
        const uint8_t *src=b+((size_t)(sz0-oz0)*CV_BLOCK+(size_t)(sy0-oy0))*CV_BLOCK+(size_t)(sx0-ox0);
        /* Whole blocks and halo Z/Y faces are contiguous. Avoid hundreds of
         * tiny row copies when a plane or the entire overlap can be copied. */
        if (row_bytes==CV_BLOCK && nx==CV_BLOCK) {
          size_t plane_bytes=(size_t)(sy1-sy0)*CV_BLOCK;
          if (sy1-sy0==CV_BLOCK && ny==CV_BLOCK) {
            memcpy(dst,src,(size_t)(sz1-sz0)*plane_bytes);
          } else {
            for(int64_t z=sz0;z<sz1;z++) {
              memcpy(dst,src,plane_bytes);
              dst+=(size_t)nx*ny; src+=CV_BLOCK*CV_BLOCK;
            }
          }
        } else {
          for (int64_t z = sz0; z < sz1; z++)
            for (int64_t y = sy0; y < sy1; y++)
              memcpy(out + ((size_t)(z - z0) * ny + (size_t)(y - y0)) * nx + (size_t)(sx0 - x0),
                     b + ((size_t)(z - oz0) * CV_BLOCK + (size_t)(y - oy0)) * CV_BLOCK +
                         (size_t)(sx0 - ox0), row_bytes);
        }
      }
  return complete;
}
void r3d_cpuvol_read_regions(r3d_cpuvol *v, uint32_t li,
                            r3d_block_region *regions, uint32_t count) {
  cv_cache *c=v->cache;
  /* Keep each critical section bounded, including for callers with big lists.
   * Copy under the lock so no extra leases or decoded slabs are needed. */
  for(uint32_t base=0;base<count;) {
    uint32_t n=count-base; if(n>32u)n=32u;
    bool hit[32]={false};
    if(c && li<v->nlev) {
      const r3d_cpuvol_level *l=&v->lev[li];
      pthread_mutex_lock(&c->m);
      for(uint32_t i=0;i<n;i++) {
        r3d_block_region *q=&regions[base+i];
        uint32_t x=q->x%16u,y=q->y%16u,z=q->z%16u;
        if(!q->nx || !q->ny || !q->nz || q->nx>16u-x || q->ny>16u-y || q->nz>16u-z ||
           (uint64_t)q->x+q->nx>l->vx || (uint64_t)q->y+q->ny>l->vy ||
           (uint64_t)q->z+q->nz>l->vz)continue;
        int slot=cvc_find(c,cv_block_id(v,li,q->x/16u,q->y/16u,q->z/16u));
        if(slot<0)continue;
        const uint8_t *src=c->slabs+(size_t)slot*CV_BLOCK_RAW+(z*16u+y)*16u+x;
        for(uint32_t k=0;k<q->nz;k++)for(uint32_t j=0;j<q->ny;j++)
          memcpy(q->out+((size_t)k*q->ny+j)*q->nx,src+k*256u+j*16u,q->nx);
        if(!c->pin[slot]) { cvc_unlink(c,(uint32_t)slot); cvc_append(c,(uint32_t)slot); }
        q->available=hit[i]=true;
      }
      pthread_mutex_unlock(&c->m);
    }
    for(uint32_t i=0;i<n;i++)if(!hit[i]) {
      r3d_block_region *q=&regions[base+i];
      q->available=cv_read_block(v,li,q->x,q->y,q->z,q->nx,q->ny,q->nz,q->out,true);
    }
    base+=n;
  }
}

void r3d_cpuvol_read_block(r3d_cpuvol *v, uint32_t li, int64_t x0, int64_t y0, int64_t z0,
                           uint32_t nx, uint32_t ny, uint32_t nz, uint8_t *out) {
  (void)cv_read_block(v, li, x0, y0, z0, nx, ny, nz, out, false);
}
bool r3d_cpuvol_read_block_status(r3d_cpuvol *v, uint32_t li, int64_t x0, int64_t y0, int64_t z0,
                                  uint32_t nx, uint32_t ny, uint32_t nz, uint8_t *out) {
  return cv_read_block(v, li, x0, y0, z0, nx, ny, nz, out, true);

}

/* ---- parallel prefetch of an explicit brick list ---- */
struct cv_pf {
  r3d_cpuvol *v;
  uint32_t li;
  const uint64_t *cells; /* linear source-cell IDs */
  uint32_t n;
  _Atomic uint32_t next;
  uint32_t cb;
};
static void *cv_pf_thread(void *ud) {
  struct cv_pf *j = ud;
  for (;;) {
    uint32_t i = atomic_fetch_add(&j->next, 1);
    if (i >= j->n) break;
    uint64_t c = j->cells[i];
    const r3d_cpuvol_level *l=&j->v->lev[j->li];
    uint32_t nx=(l->bx+j->cb-1u)/j->cb,ny=(l->by+j->cb-1u)/j->cb;
    uint32_t cx=(uint32_t)(c%nx),cy=(uint32_t)((c/nx)%ny),cz=(uint32_t)(c/((uint64_t)nx*ny));
    cv_net_fetch(j->v, j->li, cx * j->cb, cy * j->cb, cz * j->cb);
  }
  return NULL;
}

/* Mappings live until close (which requires joining samplers). Only the
 * first open needs a lock; payload CRC/decode never holds this lock. */
static cv_reader *cv_source_reader(r3d_cpuvol *v, uint32_t li, uint32_t bx,
                                    uint32_t by, uint32_t bz, bool retry) {
  const r3d_cpuvol_level *l = &v->lev[li];
  uint32_t sx = bx / CV_SHARD_BPA, sy = by / CV_SHARD_BPA, sz = bz / CV_SHARD_BPA;
  if (sx >= l->sx || sy >= l->sy || sz >= l->sz) return NULL;
  cv_reader *rd = (cv_reader *)v->readers + l->shard_off + (sz * l->sy + sy) * l->sx + sx;
  pthread_mutex_lock(&rd->mu);
  if (!rd->open && (!rd->failed || retry)) {
    char path[1400];
    snprintf(path, sizeof path, "%s/volcomp/L%u/%u_%u_%u.vcs", v->root, li, sz, sy, sx);
    if (volcomp_shard_open(path, &rd->sr) == 0) {
      rd->open = rd->sr.foot.brick_dim == CV_BRICK && rd->sr.foot.shard_dim == 1024u;
      if (!rd->open) volcomp_shard_close_reader(&rd->sr);
    }
    rd->failed = !rd->open;
  }
  bool opened = rd->open;
  pthread_mutex_unlock(&rd->mu);
  return opened ? rd : NULL;
}
static uint32_t cv_source_index(uint32_t bx, uint32_t by, uint32_t bz) {
  return ((bz % CV_SHARD_BPA) * CV_SHARD_BPA + by % CV_SHARD_BPA) * CV_SHARD_BPA + bx % CV_SHARD_BPA;
}
static int cv_keycmp(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
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
  for (uint32_t i = 0; i < n; i++) {
    uint32_t bx = bxyz[i * 3], by = bxyz[i * 3 + 1], bz = bxyz[i * 3 + 2];
    if(bx>=v->lev[li].bx || by>=v->lev[li].by || bz>=v->lev[li].bz)continue;
    cv_reader *rd = cv_source_reader(v, li, bx, by, bz, true);
    if (rd) {
      uint32_t bi = cv_source_index(bx, by, bz);
      size_t encoded_n;
      if (volcomp_shard_brick_is_zero(&rd->sr, bi) ||
          volcomp_shard_brick(&rd->sr, bi, &encoded_n)) continue;
    }
    char path[1400];
    snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.volc", v->root, li, bz, by, bx);
    struct stat st;
    if (stat(path, &st) == 0) continue; /* on disk: the sampler decodes it */
    uint32_t nx=(v->lev[li].bx+cb-1u)/cb,ny=(v->lev[li].by+cb-1u)/cb;
    uint64_t c=((uint64_t)(bz/cb)*ny+by/cb)*nx+bx/cb;
    cells[nc++] = c;
  }
  qsort(cells, nc, sizeof *cells, cv_keycmp);
  uint32_t unique = 0;
  for (uint32_t i = 0; i < nc; i++)
    if (!unique || cells[i] != cells[unique - 1]) cells[unique++] = cells[i];
  nc = unique;
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
    if (r3d_thread_create(&th[spawned], NULL, cv_pf_thread, &job) == 0) spawned++;
  cv_pf_thread(&job);
  for (uint32_t t = 0; t < spawned; t++) pthread_join(th[t], NULL);
  free(cells);
  return (int)nc;
}


/* Thread-safe brick lookup returning a pinned, immutable brick. Hits: the
 * thread's own lease (no lock at all), else a hash probe under the pool
 * lock that leases the slot. Misses use immutable shard mappings or a
 * per-entry compressed-file lease, decode into thread-local scratch,
 * then publish under the pool lock (re-checking:
 * another thread may have landed the same brick).
 *
 * The returned pointer stays valid and unmodified until this thread asks
 * for a different brick: the lease pins the slot and eviction only takes
 * unpinned slots. Consumers must therefore finish with one brick before
 * requesting the next, which every consumer here does. */
static bool cv_identity_equal(const struct stat *a, const struct stat *b) {
#ifdef __APPLE__
#define CV_MTIME st_mtimespec
#define CV_CTIME st_ctimespec
#else
#define CV_MTIME st_mtim
#define CV_CTIME st_ctim
#endif
  return a->st_dev == b->st_dev && a->st_ino == b->st_ino && a->st_size == b->st_size &&
         a->CV_MTIME.tv_sec == b->CV_MTIME.tv_sec && a->CV_MTIME.tv_nsec == b->CV_MTIME.tv_nsec &&
         a->CV_CTIME.tv_sec == b->CV_CTIME.tv_sec && a->CV_CTIME.tv_nsec == b->CV_CTIME.tv_nsec;
#undef CV_MTIME
#undef CV_CTIME
}

static void cv_memo_null(const r3d_cpuvol *v, uint64_t key, uint64_t expiry) {
  cv_nvol = v;
  cv_nid = v->id;
  cv_nkey = key;
  cv_nexp = expiry;
}

static const uint8_t *cv_brick(r3d_cpuvol *v, uint32_t li, uint32_t bx, uint32_t by,
                               uint32_t bz, bool *complete) {
  if(li>=v->nlev || bx>=v->lev[li].gx || by>=v->lev[li].gy || bz>=v->lev[li].gz) {
    if(complete)*complete=true;return NULL;
  }
  uint64_t key=cv_block_id(v,li,bx,by,bz);
  cv_cache *c = v->cache;
  if (complete) *complete = true;
  if (!c) { if (complete) *complete = false; return NULL; }
  if (cv_ls.c == c && cv_ls.key == key) return cv_ls.ptr; /* hot path: pinned */
  if (cv_nvol == v && cv_nid == v->id && cv_nkey == key &&
      (cv_nexp == UINT64_MAX || (!complete && cv_nexp > (uint64_t)time(NULL)))) return NULL;
  if (cv_ls.c && cv_ls.c != c) cv_lease_drop(); /* lease only ever spans one pool */
  cv_tls_hook();

  pthread_mutex_lock(&c->m);
  int hs = cvc_find(c, key);
  if (hs >= 0) {
    const uint8_t *hit = c->slabs + (size_t)hs * CV_BLOCK_RAW;
    cv_lease_take(c, (uint32_t)hs, key, hit);
    pthread_mutex_unlock(&c->m);
    return hit;
  }
  pthread_mutex_unlock(&c->m);

  uint32_t ox = bx % 8u, oy = by % 8u, oz = bz % 8u;
  bx /= 8u; by /= 8u; bz /= 8u; /* source chunk containing this 16^3 block */
  uint64_t now_s = (uint64_t)time(NULL);
  pthread_mutex_lock(&v->mu);
  bool absent = cv_neg_hit(v, key, now_s);
  uint64_t expiry = absent ? v->neg_exp[cv_hash(key ^ 0x9e3779b97f4a7c15ull) & (v->nneg - 1u)] : 0;
  pthread_mutex_unlock(&v->mu);
  if (absent && (!complete || expiry == UINT64_MAX)) {
    if (complete) *complete = expiry == UINT64_MAX;
    cv_memo_null(v, key, expiry);
    return NULL;
  }

  /* ---- miss: source an immutable compressed blob ---- */
  const uint8_t *blob = NULL;
  size_t bn = 0;
  uint8_t *owned = NULL;
  bool empty_file = false;
  cv_file *leased = NULL;
  char cache_path[1400] = "";
  struct stat cache_identity;
  cv_reader *rd = cv_source_reader(v, li, bx, by, bz, complete != NULL);
  if (rd) {
    uint32_t bi = cv_source_index(bx, by, bz);
    blob = volcomp_shard_brick(&rd->sr, bi, &bn);
    empty_file = volcomp_shard_brick_is_zero(&rd->sr, bi) != 0;
  }
  if (!blob && !empty_file) { /* net-ingest cache file (empty = absent/air) */
    char path[1400];
    snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.volc", v->root, li, bz, by, bx);
    struct stat identity;
    if (stat(path, &identity) != 0 && v->url[0] && !absent) cv_net_fetch(v, li, bx, by, bz);
    uint64_t chunk_key = ((uint64_t)li << 60) | ((uint64_t)bz << 40) |
                         ((uint64_t)by << 20) | bx;
    cv_files *files = v->files;
    cv_file *entry = &files->slot[cv_hash(chunk_key) % CV_FILES];
    pthread_mutex_lock(&entry->mu);
    if (stat(path, &identity) == 0) {
      if (entry->key == chunk_key && entry->data && cv_identity_equal(&entry->identity, &identity)) {
        blob = entry->data; bn = entry->n;
        atomic_fetch_add(&files->hits, 1);
      } else {
        FILE *bf = fopen(path, "rb");
        if (bf) {
          int stat_rc = fstat(fileno(bf), &identity);
          if (stat_rc == 0 && identity.st_size > 0 &&
              (uint64_t)identity.st_size <= CV_MAX_BODY_BYTES) {
            size_t fn = (size_t)identity.st_size;
            owned = malloc(fn);
            if (owned && fread(owned, 1, fn, bf) == fn) {
              blob = owned; bn = fn;
              atomic_fetch_add(&files->reads, 1);
              atomic_fetch_add(&files->bytes, fn);
              if (fn <= CV_FILE_BYTES) {
                free(entry->data);
                entry->data = owned; entry->n = fn;
                entry->key = chunk_key; entry->identity = identity;
                owned = NULL;
              }
            }
          } else if (stat_rc == 0 && identity.st_size == 0) empty_file = true;
          fclose(bf);
        }
      }
    }
    if (blob && blob == entry->data) leased = entry;
    else pthread_mutex_unlock(&entry->mu);
    if (blob) {
      snprintf(cache_path,sizeof cache_path,"%s",path);
      cache_identity=identity;
    }
  }
  if (!blob) {
    free(owned);
    pthread_mutex_lock(&v->mu);
    expiry = empty_file ? UINT64_MAX : absent ? expiry : now_s + CV_NEG_TTL_S;
    cv_neg_put(v, key, expiry);
    pthread_mutex_unlock(&v->mu);
    cv_memo_null(v, key, expiry);
    if (complete) *complete = empty_file;
    return NULL;
  }

  /* ---- decode outside the locks ---- */
  if (!cv_scratch) cv_scratch = malloc(CV_BLOCK_RAW);
  bool ok = cv_scratch && r3d_decode_block(blob, bn, oz, oy, ox, cv_scratch) == 0;
  if (!ok && cv_scratch && v->native_source && cache_path[0]) {
    /* Evict only the downloaded file we actually decoded, allowing ordinary
     * demand retry to repair it. Never delete immutable source shards. */
    struct stat current;
    if (stat(cache_path,&current)==0 && cv_identity_equal(&current,&cache_identity))
      unlink(cache_path);
    if (leased) { free(leased->data); leased->data=NULL; leased->n=0; }
  }
  free(owned);
  if (leased) pthread_mutex_unlock(&leased->mu);
  if (ok) atomic_fetch_add(&c->decoded_blocks, 1);
  if (!ok) { /* remember the failure briefly; retry when the TTL lapses */
    pthread_mutex_lock(&v->mu);
    cv_neg_put(v, key, absent ? expiry : now_s + CV_NEG_TTL_S);
    pthread_mutex_unlock(&v->mu);
    if (complete) *complete = false;
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
  const uint8_t *got = c->slabs + (size_t)hs * CV_BLOCK_RAW;
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
      cv_brick(v, li, (uint32_t)lx / CV_BLOCK, (uint32_t)ly / CV_BLOCK, (uint32_t)lz / CV_BLOCK, NULL);
  if (!b) return 0.0;
  uint32_t ox = (uint32_t)lx % CV_BLOCK, oy = (uint32_t)ly % CV_BLOCK,
           oz = (uint32_t)lz % CV_BLOCK;
  return (double)b[((size_t)oz * CV_BLOCK + oy) * CV_BLOCK + ox];
}

double r3d_cpuvol_tri(r3d_cpuvol *v, uint32_t li, const double p[3], double grad[3]) {
  if (grad) grad[0] = grad[1] = grad[2] = 0.0;
  if (li >= v->nlev) return 0.0;
  const r3d_cpuvol_level *l = &v->lev[li];
  double lx = p[0] / l->scale, ly = p[1] / l->scale, lz = p[2] / l->scale;
  double fx = floor(lx), fy = floor(ly), fz = floor(lz);
  double tx = lx - fx, ty = ly - fy, tz = lz - fz;
  int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz;
  if (!grad && tx == 0 && ty == 0 && tz == 0) return cv_vox(v, li, ix, iy, iz);
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
  const uint8_t *b = cv_brick(v, li, lx / CV_BLOCK, ly / CV_BLOCK, lz / CV_BLOCK, NULL);
  if (!b) return 0;
  uint32_t ox = lx % CV_BLOCK, oy = ly % CV_BLOCK, oz = lz % CV_BLOCK;
  return b[((size_t)oz * CV_BLOCK + oy) * CV_BLOCK + ox];
}

void r3d_cpuvol_get_cache_stats(r3d_cpuvol *v, r3d_cpuvol_cache_stats *out) {
  memset(out,0,sizeof *out);
  cv_cache *c=v->cache;
  if (!c) return;
  pthread_mutex_lock(&c->m);
  out->capacity_blocks=c->nslots;
  for (uint32_t i=0;i<c->nslots;i++) out->resident_blocks+=c->keys[i]!=UINT64_MAX;
  out->capacity_bytes=(size_t)out->capacity_blocks*CV_BLOCK_RAW;
  out->resident_bytes=(size_t)out->resident_blocks*CV_BLOCK_RAW;
  out->decoded_blocks=atomic_load(&c->decoded_blocks);
  cv_files *files = v->files;
  out->compressed_reads = atomic_load(&files->reads);
  out->compressed_bytes = atomic_load(&files->bytes);
  out->compressed_hits = atomic_load(&files->hits);
  pthread_mutex_unlock(&c->m);
}
