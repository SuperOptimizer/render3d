// dct3d_downscale — build level N of an already-exported dct3d/zarr volume from
// its level N-1, via gradient-magnitude-weighted 2x2x2 downsampling (see
// downsample.h). Both source and destination live on the same SFTP server
// (dl.ash2txt.org community-uploads); this tool never touches the raw AWS
// open-data buckets, only our own compressed output.
//
// fetch pool -> [output-shard queue] -> encode pool -> [shard queue] -> upload
// pool. Same three-stage architecture as dct3d_export; fetch here means
// "download+decode+downsample up to 8 parent shards, streamed by Z-slab, into
// one output shard" (see srcfetch.c/downsample.h), not "download raw S3
// chunks" -- so unlike dct3d_export, downsampling happens INSIDE the fetch
// stage (it needs the decoded parent data, which is never otherwise
// materialized) and the encode stage only DCT-encodes the finished shard.
//
// (quality, tau) follow the project's fixed halving ladder (ladder.h): each
// level halves both from its parent, floor quality at 1 / tau at 2.
//
// Usage:
//   dct3d_downscale --scroll <PHercId> --vol <name.zarr> --um <voxel-um>
//                   --level <N>              # build level N from N-1 (N>=1)
//                   --threads N [--ram-budget-gb G]
//                   --sftp-host H --sftp-port P --sftp-user U --sftp-pass X
//                   [--known-hosts path] [--remote-root exports] [--resume]
//                   [--local-root DIR]       # sneakernet mode: write output
//                     shards + zarr.json under DIR/<scroll>/<vol>/<level>/...
//                     and read source shards local-first (SFTP fallback for
//                     levels that only exist on the server, e.g. L0); resume
//                     checks become local stats. SFTP args optional if all
//                     sources are already local.
//                   [--dl-threads N] [--ul-threads N] [--dry-run]

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../export/bqueue.h"
#include "../export/ladder.h"
#include "../export/shard.h"
#include "../export/upload.h"
#include "localio.h"
#include "sftpget.h"
#include "srcfetch.h"

// ---- source (level N-1) shape from its zarr.json ----------------------------

static int parse_shape(const char *json, int64_t shape[3]) {
    const char *s = strstr(json, "\"shape\"");
    if (!s) return -1;
    s = strchr(s, '[');
    if (!s) return -1;
    return sscanf(s, "[ %lld , %lld , %lld",
                  (long long *)&shape[0], (long long *)&shape[1],
                  (long long *)&shape[2]) == 3
               ? 0
               : -1;
}

static int fetch_remote_shape(const sftp_target *t, const char *remote_root,
                              const char *local_root, const char *scroll,
                              const char *vol, int level, int64_t shape[3]) {
    uint8_t *buf = NULL; size_t len = 0;
    if (local_root) {
        char lpath[2048];
        snprintf(lpath, sizeof(lpath), "%s/%s/%s/%d/zarr.json",
                 local_root, scroll, vol, level);
        if (local_read(lpath, &buf, &len) != 0) buf = NULL;
    }
    if (!buf) {
        if (!t->host) return -1;  // no remote configured, nothing local
        char remote[1536];
        snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/zarr.json",
                 remote_root, scroll, vol, level);
        if (sftp_download(t, remote, &buf, &len) != 0) return -1;
    }
    char *json = (char *)malloc(len + 1);
    memcpy(json, buf, len); json[len] = 0;
    free(buf);
    int rc = parse_shape(json, shape);
    free(json);
    return rc;
}

// ---- pipeline ---------------------------------------------------------------

typedef struct { int64_t sz, sy, sx; } shard_coord;

typedef struct {
    shard_coord sc;
    uint8_t *out;  // SHARD_VOX^3 dense, already-downsampled output shard (owned; encode consumes then releases)
} region_item;

typedef struct {
    shard_coord sc;
    uint8_t *shard;
    size_t len;
} shard_item;

typedef struct { bqueue free_slots; } buf_pool;

static void buf_pool_init(buf_pool *p, size_t n, size_t bytes) {
    bq_init(&p->free_slots, n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t *b = (uint8_t *)malloc(bytes);
        if (!b) { fprintf(stderr, "buf_pool: OOM slot %zu\n", i); exit(1); }
        bq_push(&p->free_slots, b);
    }
}
static uint8_t *buf_acquire(buf_pool *p) {
    void *b = NULL; bq_pop(&p->free_slots, &b); return (uint8_t *)b;
}
static void buf_release(buf_pool *p, uint8_t *b) { bq_push(&p->free_slots, b); }

typedef struct {
    src_ctx src;
    sftp_target sftp;
    const char *remote_root, *scroll, *vol;
    const char *local_root;  // NULL = SFTP output; else write shards locally
    int dst_level;
    float quality, tau;
    int dry_run, resume;

    shard_coord *items;
    size_t n_items;
    atomic_size_t next_item;

    bqueue region_q;   // region_item* ((2*SHARD_VOX)^3 buffers)
    bqueue shard_q;    // shard_item*
    buf_pool region_bufs;

    atomic_size_t done, failed, skipped;
} pipeline;

// Fetch pool: pull a shard coord, resume-check, assemble the parent region.
static void *fetch_main(void *arg) {
    pipeline *p = (pipeline *)arg;
    for (;;) {
        size_t idx = atomic_fetch_add(&p->next_item, 1);
        if (idx >= p->n_items) break;
        shard_coord sc = p->items[idx];

        if (p->resume && !p->dry_run) {
            long have = -1;
            if (p->local_root) {
                char lpath[2048];
                snprintf(lpath, sizeof(lpath), "%s/%s/%s/%d/c/%lld/%lld/%lld",
                         p->local_root, p->scroll, p->vol, p->dst_level,
                         (long long)sc.sz, (long long)sc.sy, (long long)sc.sx);
                have = local_size(lpath);
            } else {
                char remote[1536];
                snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/c/%lld/%lld/%lld",
                         p->remote_root, p->scroll, p->vol, p->dst_level,
                         (long long)sc.sz, (long long)sc.sy, (long long)sc.sx);
                have = sftp_stat_size(&p->sftp, remote);
            }
            if (have >= 0) {
                atomic_fetch_add(&p->skipped, 1);
                continue;
            }
        }

        uint8_t *out = buf_acquire(&p->region_bufs);
        if (srcfetch_region(&p->src, sc.sz, sc.sy, sc.sx, out) != 0) {
            fprintf(stderr, "fetch: failed %lld/%lld/%lld L%d\n",
                    (long long)sc.sz, (long long)sc.sy, (long long)sc.sx, p->dst_level);
            atomic_fetch_add(&p->failed, 1);
            buf_release(&p->region_bufs, out);
            continue;
        }

        region_item *ri = (region_item *)malloc(sizeof(region_item));
        ri->sc = sc; ri->out = out;
        if (!bq_push(&p->region_q, ri)) { buf_release(&p->region_bufs, out); free(ri); }
    }
    return NULL;
}

// Encode pool. Downsampling now happens inside srcfetch_region (streamed by
// Z-slab), so this stage just encodes the already-assembled output shard and
// returns its region buffer to the pool.
static void *encode_main(void *arg) {
    pipeline *p = (pipeline *)arg;
    void *item;
    while (bq_pop(&p->region_q, &item)) {
        region_item *ri = (region_item *)item;

        uint8_t *shard = NULL; size_t len = 0;
        int rc = shard_encode_u8(ri->out, p->quality, p->tau, &shard, &len);
        buf_release(&p->region_bufs, ri->out);
        if (rc != 0) {
            fprintf(stderr, "encode: OOM %lld/%lld/%lld\n",
                    (long long)ri->sc.sz, (long long)ri->sc.sy, (long long)ri->sc.sx);
            atomic_fetch_add(&p->failed, 1);
            free(ri);
            continue;
        }
        shard_item *si = (shard_item *)malloc(sizeof(shard_item));
        si->sc = ri->sc; si->shard = shard; si->len = len;
        free(ri);
        if (!bq_push(&p->shard_q, si)) { free(shard); free(si); }
    }
    return NULL;
}

// Upload pool.
static void *upload_main(void *arg) {
    pipeline *p = (pipeline *)arg;
    void *item;
    while (bq_pop(&p->shard_q, &item)) {
        shard_item *si = (shard_item *)item;
        int ok;
        char dst[2048];
        if (p->local_root) {
            snprintf(dst, sizeof(dst), "%s/%s/%s/%d/c/%lld/%lld/%lld",
                     p->local_root, p->scroll, p->vol, p->dst_level,
                     (long long)si->sc.sz, (long long)si->sc.sy, (long long)si->sc.sx);
            ok = p->dry_run ? 1 : (local_write_atomic(dst, si->shard, si->len) == 0);
        } else {
            snprintf(dst, sizeof(dst), "/%s/%s/%s/%d/c/%lld/%lld/%lld",
                     p->remote_root, p->scroll, p->vol, p->dst_level,
                     (long long)si->sc.sz, (long long)si->sc.sy, (long long)si->sc.sx);
            ok = p->dry_run ? 1 : (sftp_upload(&p->sftp, dst, si->shard, si->len) == 0);
        }
        if (ok) atomic_fetch_add(&p->done, 1);
        else {
            fprintf(stderr, "upload: failed %s\n", dst);
            atomic_fetch_add(&p->failed, 1);
        }
        free(si->shard);
        free(si);
    }
    return NULL;
}

static int upload_level_metadata(const pipeline *p, int64_t padded[3]) {
    char zj[2048];
    snprintf(zj, sizeof(zj),
             "{\n"
             "  \"zarr_format\": 3,\n"
             "  \"node_type\": \"array\",\n"
             "  \"shape\": [%lld, %lld, %lld],\n"
             "  \"data_type\": \"uint8\",\n"
             "  \"chunk_grid\": {\"name\": \"regular\", \"configuration\": "
             "{\"chunk_shape\": [%d, %d, %d]}},\n"
             "  \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": "
             "{\"separator\": \"/\"}},\n"
             "  \"fill_value\": 0,\n"
             "  \"codecs\": [{\"name\": \"sharding_indexed\", \"configuration\": {\n"
             "    \"chunk_shape\": [%d, %d, %d],\n"
             "    \"codecs\": [{\"name\": \"dct3d\", \"configuration\": "
             "{\"quality\": %g, \"max_error\": 0.0, \"tau\": %g}}],\n"
             "    \"index_codecs\": [{\"name\": \"bytes\", \"configuration\": "
             "{\"endian\": \"little\"}}, {\"name\": \"crc32c\"}],\n"
             "    \"index_location\": \"end\"\n"
             "  }}]\n"
             "}\n",
             (long long)padded[0], (long long)padded[1], (long long)padded[2],
             SHARD_VOX, SHARD_VOX, SHARD_VOX, INNER, INNER, INNER,
             (double)p->quality, (double)p->tau);

    char dst[2048];
    if (p->local_root)
        snprintf(dst, sizeof(dst), "%s/%s/%s/%d/zarr.json",
                 p->local_root, p->scroll, p->vol, p->dst_level);
    else
        snprintf(dst, sizeof(dst), "/%s/%s/%s/%d/zarr.json",
                 p->remote_root, p->scroll, p->vol, p->dst_level);
    if (p->dry_run) {
        printf("[dry-run] would write %s (%zu B)\n", dst, strlen(zj));
        return 0;
    }
    if (p->local_root)
        return local_write_atomic(dst, (const uint8_t *)zj, strlen(zj));
    return sftp_upload(&p->sftp, dst, (const uint8_t *)zj, strlen(zj));
}

// ---- CLI ---------------------------------------------------------------------

typedef struct {
    const char *scroll, *vol, *remote_root, *known_hosts, *local_root;
    const char *sftp_host, *sftp_user, *sftp_pass;
    int sftp_port, threads, level, dry_run, resume;
    double um;
    double ram_budget_gb;
    int dl_threads, ul_threads;
    long limit_shards;
    int64_t only[3];
} args_t;

static const char *argval(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[*i]); exit(2); }
    return argv[++(*i)];
}

int main(int argc, char **argv) {
    args_t a = {0};
    a.remote_root = "exports";
    a.threads = 6;
    a.level = 1;
    a.sftp_port = 22;
    a.ram_budget_gb = 10.0;
    a.limit_shards = -1;
    a.only[0] = -1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--scroll")) a.scroll = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--vol")) a.vol = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--um")) a.um = atof(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--level")) a.level = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--threads")) a.threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--remote-root")) a.remote_root = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--local-root")) a.local_root = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-host")) a.sftp_host = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-port")) a.sftp_port = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--sftp-user")) a.sftp_user = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-pass")) a.sftp_pass = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--known-hosts")) a.known_hosts = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--resume")) a.resume = 1;
        else if (!strcmp(argv[i], "--ram-budget-gb")) a.ram_budget_gb = atof(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--dl-threads")) a.dl_threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--ul-threads")) a.ul_threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--dry-run")) a.dry_run = 1;
        else if (!strcmp(argv[i], "--limit-shards")) a.limit_shards = atol(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--only-shard")) {
            const char *v = argval(argc, argv, &i);
            if (sscanf(v, "%lld,%lld,%lld", (long long *)&a.only[0],
                       (long long *)&a.only[1], (long long *)&a.only[2]) != 3) {
                fprintf(stderr, "--only-shard wants z,y,x\n"); return 2;
            }
        }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!a.scroll || !a.vol || a.um <= 0 || a.level < 1) {
        fprintf(stderr, "required: --scroll --vol --um --level (>=1)\n");
        return 2;
    }
    if (!a.dry_run && !a.local_root && (!a.sftp_host || !a.sftp_user || !a.sftp_pass)) {
        fprintf(stderr, "--sftp-host/user/pass required unless --dry-run or --local-root\n");
        return 2;
    }

    sftp_global_init();
    sftp_target sftp = {a.sftp_host, a.sftp_port, a.sftp_user, a.sftp_pass, a.known_hosts};

    int64_t src_shape[3];
    if (fetch_remote_shape(&sftp, a.remote_root, a.local_root, a.scroll, a.vol,
                           a.level - 1, src_shape) != 0) {
        fprintf(stderr, "L%d: source level L%d zarr.json not found\n", a.level, a.level - 1);
        return 1;
    }
    int64_t src_nshard[3], dst_shape[3], dst_nshard[3], dst_padded[3];
    for (int d = 0; d < 3; ++d) {
        src_nshard[d] = (src_shape[d] + SHARD_VOX - 1) / SHARD_VOX;
        dst_shape[d] = (src_shape[d] + 1) / 2;
        dst_nshard[d] = (dst_shape[d] + SHARD_VOX - 1) / SHARD_VOX;
        dst_padded[d] = dst_nshard[d] * SHARD_VOX;
    }

    float quality, tau;
    ladder_for_level(a.um, a.level, &quality, &tau);

    size_t total = (size_t)dst_nshard[0] * dst_nshard[1] * dst_nshard[2];
    printf("L%d <- L%d: src shape [%lld,%lld,%lld] -> dst shape [%lld,%lld,%lld] "
           "padded [%lld,%lld,%lld] shards %lldx%lldx%lld = %zu | q=%g tau=%g\n",
           a.level, a.level - 1,
           (long long)src_shape[0], (long long)src_shape[1], (long long)src_shape[2],
           (long long)dst_shape[0], (long long)dst_shape[1], (long long)dst_shape[2],
           (long long)dst_padded[0], (long long)dst_padded[1], (long long)dst_padded[2],
           (long long)dst_nshard[0], (long long)dst_nshard[1], (long long)dst_nshard[2],
           total, (double)quality, (double)tau);

    pipeline p = {0};
    p.src.sftp = sftp;
    p.src.remote_root = a.remote_root; p.src.scroll = a.scroll; p.src.vol = a.vol;
    p.src.local_root = a.local_root;
    p.src.src_level = a.level - 1;
    p.src.src_nshard[0] = src_nshard[0];
    p.src.src_nshard[1] = src_nshard[1];
    p.src.src_nshard[2] = src_nshard[2];
    p.sftp = sftp;
    p.remote_root = a.remote_root; p.scroll = a.scroll; p.vol = a.vol;
    p.local_root = a.local_root;
    p.dst_level = a.level;
    p.quality = quality; p.tau = tau;
    p.dry_run = a.dry_run; p.resume = a.resume;

    p.items = (shard_coord *)malloc(total * sizeof(shard_coord));
    size_t k = 0;
    if (a.only[0] >= 0) {
        p.items[k++] = (shard_coord){a.only[0], a.only[1], a.only[2]};
    } else {
        for (int64_t z = 0; z < dst_nshard[0]; ++z)
            for (int64_t y = 0; y < dst_nshard[1]; ++y)
                for (int64_t x = 0; x < dst_nshard[2]; ++x)
                    p.items[k++] = (shard_coord){z, y, x};
    }
    p.n_items = k;
    if (a.limit_shards >= 0 && (size_t)a.limit_shards < p.n_items)
        p.n_items = a.limit_shards;

    int enc_threads = a.threads > 0 ? a.threads : 3;
    int dl_threads = a.dl_threads > 0 ? a.dl_threads : enc_threads * 2;
    int ul_threads = a.ul_threads > 0 ? a.ul_threads : enc_threads * 2;
    if (dl_threads > 64) dl_threads = 64;
    if (ul_threads > 64) ul_threads = 64;

    // Region buffers now hold just the final SHARD_VOX^3 output shard
    // (~1 GiB) -- srcfetch_region streams the 8 parent shards in Z-slabs
    // internally instead of ever materializing the old (2*SHARD_VOX)^3 == 8x
    // bigger assembly cube (see srcfetch.c/downsample.c). Each in-flight
    // fetch also transiently holds ~1.1-1.2 GiB of its own (compressed
    // parent blobs + slab scratch + accumulator, freed when srcfetch_region
    // returns) that does NOT come from this pool -- that's accounted for
    // separately below via FETCH_TRANSIENT_GIB so the budget isn't
    // oversubscribed. NEVER floor n_bufs above what the budget actually
    // allows: a floor here silently blows past a small box's real RAM
    // (that's how the old 8 GiB design got OOM-killed the first time).
    size_t region_bytes = (size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX;
    double region_gib = (double)region_bytes / (1u << 30);
    // Per-fetch-thread transient peak (compressed blobs + 4 slab-decode
    // scratch buffers + slab accumulator pair) -- see srcfetch.c's memory
    // comment; ~0.65 GiB scratch+acc plus a generous ~0.5 GiB allowance for
    // the (compressed, usually much smaller) parent blobs.
    const double FETCH_TRANSIENT_GIB = 1.2;
    double budget_gib = a.ram_budget_gb * 0.7;
    // Solve for how many (region buffer + in-flight fetch transient) pairs
    // fit: each concurrently-active fetch thread needs one region buffer
    // (~1 GiB) plus its own ~1.2 GiB transient working set.
    size_t n_bufs = (size_t)(budget_gib / (region_gib + FETCH_TRANSIENT_GIB));
    if (n_bufs < 1) n_bufs = 1;
    // dl_threads is sized off enc_threads by default (see below), not capped
    // to n_bufs anymore as the sole gate -- but it still can't exceed the
    // number of region buffers, since every in-flight fetch needs one to
    // hand off to the encode stage.
    if ((size_t)dl_threads > n_bufs) dl_threads = (int)n_bufs;
    size_t shard_depth = dl_threads * 2; if (shard_depth < 4) shard_depth = 4;

    printf("L%d pools: dl=%d enc=%d ul=%d | %zu region buffers (~%.1f GiB each, "
           "~%.0f GiB in-flight), shard-queue depth %zu\n",
           a.level, dl_threads, enc_threads, ul_threads, n_bufs, region_gib,
           n_bufs * region_gib, shard_depth);

    buf_pool_init(&p.region_bufs, n_bufs, region_bytes);
    bq_init(&p.region_q, n_bufs);
    bq_init(&p.shard_q, shard_depth);

    if (upload_level_metadata(&p, dst_padded) != 0)
        fprintf(stderr, "L%d: metadata upload failed (continuing)\n", a.level);

    pthread_t *dl = malloc(dl_threads * sizeof(pthread_t));
    pthread_t *en = malloc(enc_threads * sizeof(pthread_t));
    pthread_t *ul = malloc(ul_threads * sizeof(pthread_t));
    for (int t = 0; t < ul_threads; ++t) pthread_create(&ul[t], NULL, upload_main, &p);
    for (int t = 0; t < enc_threads; ++t) pthread_create(&en[t], NULL, encode_main, &p);
    for (int t = 0; t < dl_threads; ++t) pthread_create(&dl[t], NULL, fetch_main, &p);

    for (int t = 0; t < dl_threads; ++t) pthread_join(dl[t], NULL);
    bq_close(&p.region_q);
    for (int t = 0; t < enc_threads; ++t) pthread_join(en[t], NULL);
    bq_close(&p.shard_q);
    for (int t = 0; t < ul_threads; ++t) pthread_join(ul[t], NULL);
    free(dl); free(en); free(ul);

    printf("L%d done: %zu ok, %zu failed, %zu resume-skipped of %zu\n",
           a.level, atomic_load(&p.done), atomic_load(&p.failed),
           atomic_load(&p.skipped), p.n_items);

    bq_close(&p.region_bufs.free_slots);
    void *b;
    while (bq_pop(&p.region_bufs.free_slots, &b)) free(b);
    bq_destroy(&p.region_bufs.free_slots);
    bq_destroy(&p.region_q);
    bq_destroy(&p.shard_q);
    free(p.items);

    return atomic_load(&p.failed) > 0 ? 1 : 0;
}
