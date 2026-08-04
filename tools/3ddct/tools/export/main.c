// dct3d_export — stream a source OME-zarr volume level to a dct3d-compressed,
// sharded Zarr v3 volume on an SFTP server, via a three-pool pipeline.
//
// download pool -> [dense-volume queue] -> compress pool -> [shard queue] ->
// upload pool. Each stage runs concurrently on its own resource (network in /
// CPU / network out) so compression never blocks on I/O; bounded queues + a
// reusable buffer pool bound in-flight RAM. Per shard: batched-fetch the source
// 128^3 chunks (libs3), assemble a dense SHARD_VOX^3 volume (zero-padded past
// the real extent), dct3d-encode its 16^3 inner chunks into a sharding_indexed
// shard, SFTP it to <root>/<scroll>/<vol>.zarr/<L>/<z>/<y>/<x>. Dims are padded
// up to a whole multiple of SHARD_VOX (full zero-filled edge shards).
//
// Default is level 0 only; pass --levels to export others.
//
// Usage:
//   dct3d_export --base <https-zarr-root> --scroll <PHercId> --vol <name.zarr>
//                --um <voxel-um> [--levels 0] --threads N [--ram-budget-gb G]
//                --sftp-host H --sftp-port P --sftp-user U --sftp-pass X
//                [--known-hosts path] [--remote-root exports] [--resume]
//                [--dl-threads N] [--ul-threads N]
//                [--limit-shards N] [--only-shard z,y,x] [--no-oracle] [--dry-run]

#include <curl/curl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bqueue.h"
#include "fetch.h"
#include "ladder.h"
#include "oracle.h"
#include "shard.h"
#include "upload.h"
#include "vendor/libs3.h"

// ---- source metadata (.zarray) ---------------------------------------------

// Minimal ".zarray" scrape: pull shape[3]. Source levels are known raw u8 128^3;
// we only need the shape to size the grid.
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

static int fetch_level_shape(s3_client *c, const char *base, int level,
                             int64_t shape[3]) {
    char url[1280];
    snprintf(url, sizeof(url), "%s/%d/.zarray", base, level);
    s3_response r = {0};
    s3_status st = s3_get(c, url, &r);
    int rc = -1;
    if ((st == S3_OK || st == S3_ERR_HTTP) && r.status == 200 && r.body)
        rc = parse_shape((const char *)r.body, shape);
    s3_response_free(&r);
    return rc;
}

// ---- three-pool pipeline ---------------------------------------------------
//
// download pool -> [dense-volume queue] -> compress pool -> [shard queue] ->
// upload pool. The stages run concurrently on their own resource (network in /
// CPU / network out); bounded queues bound in-flight RAM and apply backpressure.
// A dense 1024^3 volume and an encoded shard are each ~1 GiB, so queue depths
// are sized from a RAM budget.

typedef struct {
    int64_t sz, sy, sx;
} shard_coord;

// A dense-volume work item flowing download -> compress.
typedef struct {
    shard_coord sc;
    uint8_t *vol;   // SHARD_VOX^3 dense (owned; returned to buffer pool after encode)
    int all_air;    // known-empty (skip real encode work but still emit shard)
} vol_item;

// An encoded-shard item flowing compress -> upload.
typedef struct {
    shard_coord sc;
    uint8_t *shard;   // encoded bytes (malloc'd; freed after upload)
    size_t len;
} shard_item;

// A fixed pool of reusable SHARD_VOX^3 buffers (avoids per-shard 1 GiB
// malloc/free churn — perf showed millions of page faults). Blocking acquire.
typedef struct {
    bqueue free_slots;  // holds uint8_t* buffers
} buf_pool;

static void buf_pool_init(buf_pool *p, size_t n) {
    bq_init(&p->free_slots, n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t *b = (uint8_t *)malloc((size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX);
        if (!b) { fprintf(stderr, "buf_pool: OOM allocating slot %zu\n", i); exit(1); }
        bq_push(&p->free_slots, b);
    }
}
static uint8_t *buf_acquire(buf_pool *p) {
    void *b = NULL;
    bq_pop(&p->free_slots, &b);
    return (uint8_t *)b;
}
static void buf_release(buf_pool *p, uint8_t *b) { bq_push(&p->free_slots, b); }

// Shared pipeline state for one level.
typedef struct {
    // config
    s3_client *client;
    src_level lvl;
    float quality, tau;
    const char *remote_root, *scroll, *vol;
    sftp_target sftp;
    int dry_run, resume;
    int oracle_level;
    char oracle_base[1024];
    int64_t oracle_shape[3];
    oracle_map *omap;   // mmap'd whole-L4 oracle (NULL -> per-shard S3 fetch)

    // work source (shard coords)
    shard_coord *items;
    size_t n_items;
    atomic_size_t next_item;

    // stage queues + buffer pool
    bqueue vol_q;      // vol_item*
    bqueue shard_q;    // shard_item*
    buf_pool bufs;

    // counters
    atomic_size_t done, failed, skipped, air_shards;
} pipeline;

// Download pool: pull a shard coord, resume-check, oracle-check, fetch the dense
// volume from a pool buffer, push to vol_q. Empty shards ride through as all_air.
static void *download_main(void *arg) {
    pipeline *p = (pipeline *)arg;
    for (;;) {
        size_t idx = atomic_fetch_add(&p->next_item, 1);
        if (idx >= p->n_items) break;
        shard_coord sc = p->items[idx];
        int64_t oz = sc.sz * SHARD_VOX, oy = sc.sy * SHARD_VOX, ox = sc.sx * SHARD_VOX;

        if (p->resume && !p->dry_run) {
            char remote[1536];
            snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/c/%lld/%lld/%lld",
                     p->remote_root, p->scroll, p->vol, p->lvl.level,
                     (long long)sc.sz, (long long)sc.sy, (long long)sc.sx);
            if (sftp_size(&p->sftp, remote) >= 0) {
                atomic_fetch_add(&p->skipped, 1);
                continue;
            }
        }

        const int64_t *shp = p->lvl.shape;
        int fully_air = (oz >= shp[0] || oy >= shp[1] || ox >= shp[2]);
        oracle_region orc;
        orc.valid = 0;
        if (!fully_air && p->oracle_level >= 0) {
            if (p->omap && p->omap->valid) {
                // mmap'd whole-L4 oracle: pure mapped reads, no S3.
                oracle_region_from_map(p->omap, oz, oy, ox, &orc);
            } else if (oracle_fetch(p->client, p->oracle_base, p->oracle_level,
                                    p->oracle_shape, oz, oy, ox, &orc) != 0) {
                orc.valid = 0;
            }
            if (orc.valid && orc.all_zero) {
                fully_air = 1;
                atomic_fetch_add(&p->air_shards, 1);
            }
        }

        uint8_t *vol = buf_acquire(&p->bufs);
        if (fully_air) {
            memset(vol, 0, (size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX);
        } else if (fetch_shard_region(p->client, &p->lvl, oz, oy, ox, vol,
                                      orc.valid ? &orc : NULL) != 0) {
            fprintf(stderr, "download: fetch failed %lld/%lld/%lld L%d\n",
                    (long long)sc.sz, (long long)sc.sy, (long long)sc.sx, p->lvl.level);
            atomic_fetch_add(&p->failed, 1);
            buf_release(&p->bufs, vol);
            continue;
        }

        vol_item *vi = (vol_item *)malloc(sizeof(vol_item));
        vi->sc = sc; vi->vol = vol; vi->all_air = fully_air;
        if (!bq_push(&p->vol_q, vi)) { buf_release(&p->bufs, vol); free(vi); }
    }
    return NULL;
}

// Compress pool: pop a dense volume, encode it, return the buffer to the pool,
// push the encoded shard to shard_q.
static void *compress_main(void *arg) {
    pipeline *p = (pipeline *)arg;
    void *item;
    while (bq_pop(&p->vol_q, &item)) {
        vol_item *vi = (vol_item *)item;
        uint8_t *shard = NULL;
        size_t len = 0;
        int rc = shard_encode_u8(vi->vol, p->quality, p->tau, &shard, &len);
        buf_release(&p->bufs, vi->vol);  // volume no longer needed
        if (rc != 0) {
            fprintf(stderr, "compress: encode OOM %lld/%lld/%lld\n",
                    (long long)vi->sc.sz, (long long)vi->sc.sy, (long long)vi->sc.sx);
            atomic_fetch_add(&p->failed, 1);
            free(vi);
            continue;
        }
        shard_item *si = (shard_item *)malloc(sizeof(shard_item));
        si->sc = vi->sc; si->shard = shard; si->len = len;
        free(vi);
        if (!bq_push(&p->shard_q, si)) { free(shard); free(si); }
    }
    return NULL;
}

// Upload pool: pop an encoded shard, SFTP it, free it.
static void *upload_main(void *arg) {
    pipeline *p = (pipeline *)arg;
    void *item;
    while (bq_pop(&p->shard_q, &item)) {
        shard_item *si = (shard_item *)item;
        char remote[1536];
        snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/c/%lld/%lld/%lld",
                 p->remote_root, p->scroll, p->vol, p->lvl.level,
                 (long long)si->sc.sz, (long long)si->sc.sy, (long long)si->sc.sx);
        int ok = p->dry_run ? 1 : (sftp_upload(&p->sftp, remote, si->shard, si->len) == 0);
        if (ok) atomic_fetch_add(&p->done, 1);
        else {
            fprintf(stderr, "upload: failed %s\n", remote);
            atomic_fetch_add(&p->failed, 1);
        }
        free(si->shard);
        free(si);
    }
    return NULL;
}

// ---- zarr.json emission ----------------------------------------------------

// Emit the per-level array zarr.json (padded shape, dct3d codec) and upload it.
static int upload_level_metadata(const pipeline *base, int64_t padded[3],
                                 float quality, float tau) {
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
             (double)quality, (double)tau);

    char remote[1536];
    snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/zarr.json",
             base->remote_root, base->scroll, base->vol, base->lvl.level);
    if (base->dry_run) {
        printf("[dry-run] would write %s (%zu B)\n", remote, strlen(zj));
        return 0;
    }
    return sftp_upload(&base->sftp, remote, (const uint8_t *)zj, strlen(zj));
}

// ---- CLI -------------------------------------------------------------------

typedef struct {
    const char *base, *scroll, *vol, *remote_root, *known_hosts;
    const char *sftp_host, *sftp_user, *sftp_pass;
    int sftp_port, threads, lvl_lo, lvl_hi, dry_run;
    long limit_shards;
    double um;
    int no_oracle;
    int resume;        // skip shards already present on the server
    int64_t only[3];   // if only[0]>=0, process just this one shard coord
    double ram_budget_gb;  // bound on in-flight ~1 GiB buffers
    int dl_threads, ul_threads;  // 0 -> derived from --threads
    const char *oracle_cache_dir;  // if set, materialize+mmap the oracle level here
    int count_only;    // classify shards air/dense via oracle, print counts, exit
} args_t;

static void parse_levels(const char *s, int *lo, int *hi) {
    if (sscanf(s, "%d-%d", lo, hi) == 2) return;
    if (sscanf(s, "%d", lo) == 1) { *hi = *lo; return; }
    *lo = 0; *hi = 5;
}

static const char *argval(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[*i]); exit(2); }
    return argv[++(*i)];
}

int main(int argc, char **argv) {
    args_t a = {0};
    a.remote_root = "exports";
    a.threads = 6;
    a.lvl_lo = 0; a.lvl_hi = 0;  // level 0 only by default (override with --levels)
    a.sftp_port = 22;
    a.limit_shards = -1;
    a.only[0] = -1;
    a.ram_budget_gb = 10.0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--base")) a.base = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--scroll")) a.scroll = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--vol")) a.vol = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--um")) a.um = atof(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--levels")) parse_levels(argval(argc, argv, &i), &a.lvl_lo, &a.lvl_hi);
        else if (!strcmp(argv[i], "--threads")) a.threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--remote-root")) a.remote_root = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-host")) a.sftp_host = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-port")) a.sftp_port = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--sftp-user")) a.sftp_user = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-pass")) a.sftp_pass = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--known-hosts")) a.known_hosts = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--limit-shards")) a.limit_shards = atol(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--no-oracle")) a.no_oracle = 1;
        else if (!strcmp(argv[i], "--oracle-cache-dir")) a.oracle_cache_dir = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--count-only")) a.count_only = 1;
        else if (!strcmp(argv[i], "--resume")) a.resume = 1;
        else if (!strcmp(argv[i], "--ram-budget-gb")) a.ram_budget_gb = atof(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--dl-threads")) a.dl_threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--ul-threads")) a.ul_threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--only-shard")) {
            const char *v = argval(argc, argv, &i);
            if (sscanf(v, "%lld,%lld,%lld", (long long *)&a.only[0],
                       (long long *)&a.only[1], (long long *)&a.only[2]) != 3) {
                fprintf(stderr, "--only-shard wants z,y,x\n"); return 2;
            }
        }
        else if (!strcmp(argv[i], "--dry-run")) a.dry_run = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!a.base || !a.scroll || !a.vol || a.um <= 0) {
        fprintf(stderr, "required: --base --scroll --vol --um (and sftp opts unless --dry-run)\n");
        return 2;
    }
    if (!a.dry_run && !a.count_only && (!a.sftp_host || !a.sftp_user || !a.sftp_pass)) {
        fprintf(stderr, "--sftp-host/user/pass required unless --dry-run/--count-only\n");
        return 2;
    }

    sftp_global_init();
    s3_config cfg = {0};
    cfg.transfer_timeout_s = 60;
    cfg.max_retries = 4;
    s3_client *client = s3_client_new(&cfg);
    if (!client) { fprintf(stderr, "s3_client_new failed\n"); return 1; }

    sftp_target sftp = {a.sftp_host, a.sftp_port, a.sftp_user, a.sftp_pass, a.known_hosts};

    for (int level = a.lvl_lo; level <= a.lvl_hi; ++level) {
        int64_t shape[3];
        if (fetch_level_shape(client, a.base, level, shape) != 0) {
            fprintf(stderr, "L%d: no .zarray (stopping levels)\n", level);
            break;
        }
        // Pad each dim up to a whole multiple of SHARD_VOX.
        int64_t padded[3], nshard[3];
        for (int d = 0; d < 3; ++d) {
            nshard[d] = (shape[d] + SHARD_VOX - 1) / SHARD_VOX;
            padded[d] = nshard[d] * SHARD_VOX;
        }
        float quality, tau;
        ladder_for_level(a.um, level, &quality, &tau);

        size_t total = (size_t)nshard[0] * nshard[1] * nshard[2];
        printf("L%d shape [%lld,%lld,%lld] -> padded [%lld,%lld,%lld] "
               "shards %lldx%lldx%lld = %zu | q=%g tau=%g\n",
               level, (long long)shape[0], (long long)shape[1], (long long)shape[2],
               (long long)padded[0], (long long)padded[1], (long long)padded[2],
               (long long)nshard[0], (long long)nshard[1], (long long)nshard[2],
               total, (double)quality, (double)tau);

        // Resolve the air oracle level (L0<-L4, L1<-L5) if it exists.
        int oracle_level = a.no_oracle ? -1 : oracle_level_for(level);
        int64_t oracle_shape[3] = {0, 0, 0};
        if (oracle_level >= 0) {
            if (fetch_level_shape(client, a.base, oracle_level, oracle_shape) != 0) {
                fprintf(stderr, "L%d: oracle L%d .zarray missing — oracle disabled\n",
                        level, oracle_level);
                oracle_level = -1;
            } else {
                printf("L%d: using air oracle L%d [%lld,%lld,%lld]\n", level,
                       oracle_level, (long long)oracle_shape[0],
                       (long long)oracle_shape[1], (long long)oracle_shape[2]);
            }
        }

        // Materialize the oracle level to disk + mmap when a cache dir is given.
        // For big 2.4um volumes L4 is 20-30 GB — too big for RAM but fine on
        // disk, and the OS page-caches it, so per-shard lookups are mapped reads
        // with no S3 traffic. Without --oracle-cache-dir we fall back to the
        // per-shard 64^3 S3 fetch (fine for small volumes).
        oracle_map omap = {0};
        if (oracle_level >= 0 && a.oracle_cache_dir && !a.count_only) {
            char opath[2048];
            snprintf(opath, sizeof(opath), "%s/%s_L%d.oracle",
                     a.oracle_cache_dir, a.vol, oracle_level);
            if (oracle_map_build(client, a.base, oracle_level, oracle_shape, opath,
                                 &omap) != 0) {
                fprintf(stderr, "L%d: oracle mmap build failed — falling back to "
                        "per-shard S3 oracle\n", level);
                omap.valid = 0;
            }
        }

        pipeline p = {0};
        p.client = client;
        p.lvl.level = level;
        p.lvl.src_chunk = 128;
        snprintf(p.lvl.base, sizeof(p.lvl.base), "%s", a.base);
        p.lvl.shape[0] = shape[0]; p.lvl.shape[1] = shape[1]; p.lvl.shape[2] = shape[2];
        p.quality = quality; p.tau = tau;
        p.remote_root = a.remote_root; p.scroll = a.scroll; p.vol = a.vol;
        p.sftp = sftp; p.dry_run = a.dry_run; p.resume = a.resume;
        p.oracle_level = oracle_level;
        snprintf(p.oracle_base, sizeof(p.oracle_base), "%s", a.base);
        p.oracle_shape[0] = oracle_shape[0];
        p.oracle_shape[1] = oracle_shape[1];
        p.oracle_shape[2] = oracle_shape[2];
        p.omap = omap.valid ? &omap : NULL;

        // Build the shard-coord work list.
        p.items = (shard_coord *)malloc(total * sizeof(shard_coord));
        size_t k = 0;
        if (a.only[0] >= 0) {
            p.items[k++] = (shard_coord){a.only[0], a.only[1], a.only[2]};
        } else {
            for (int64_t z = 0; z < nshard[0]; ++z)
                for (int64_t y = 0; y < nshard[1]; ++y)
                    for (int64_t x = 0; x < nshard[2]; ++x)
                        p.items[k++] = (shard_coord){z, y, x};
        }
        p.n_items = k;
        if (a.limit_shards >= 0 && (size_t)a.limit_shards < p.n_items)
            p.n_items = a.limit_shards;

        // Count-only: classify every shard air/dense for a work estimate + ETA.
        // Uses a COARSER count level (L+5, i.e. L5 for an L0 export) — 8x smaller
        // than the L4 export oracle (2.5-4 GB vs 20-30 GB), so it downloads fast
        // and scans in one LINEAR sweep. One L5 voxel = a 32^3 L0 region; a shard
        // is 1024^3 = 32^3 L5 voxels. Mark a shard dense on the first nonzero L5
        // voxel inside it. Slightly coarser air detection than L4 but ample for
        // an ETA. Independent of the export's L4 oracle.
        if (a.count_only) {
            int clvl = level + 5;  // L0 -> L5
            int64_t cshape[3];
            oracle_map cmap = {0};
            if (fetch_level_shape(client, a.base, clvl, cshape) != 0) {
                printf("L%d COUNT: count level L%d missing; skipped\n", level, clvl);
            } else if (!a.oracle_cache_dir) {
                printf("L%d COUNT: needs --oracle-cache-dir; skipped\n", level);
            } else {
                char cpath[2048];
                snprintf(cpath, sizeof(cpath), "%s/%s_L%d.count",
                         a.oracle_cache_dir, a.vol, clvl);
                if (oracle_map_build(client, a.base, clvl, cshape, cpath, &cmap) != 0) {
                    printf("L%d COUNT: L%d build failed; skipped\n", level, clvl);
                } else {
                    const int64_t *osh = cmap.shape;
                    // shard = COUNT_EDGE count-voxels/axis. count level is L+5 =
                    // 32x downscale; shard 1024^3 -> 1024/32 = 32 count voxels.
                    const int64_t CE = SHARD_VOX / 32;  // 32
                    unsigned char *sd = (unsigned char *)calloc(
                        (size_t)nshard[0] * nshard[1] * nshard[2], 1);
                    for (int64_t vz = 0; vz < osh[0]; ++vz) {
                        int64_t sz = vz / CE; if (sz >= nshard[0]) continue;
                        for (int64_t vy = 0; vy < osh[1]; ++vy) {
                            int64_t sy = vy / CE; if (sy >= nshard[1]) continue;
                            const uint8_t *row = cmap.base + ((size_t)vz * osh[1] + vy) * osh[2];
                            int64_t srow = ((sz * nshard[1]) + sy) * nshard[2];
                            for (int64_t vx = 0; vx < osh[2]; ++vx)
                                if (row[vx]) { int64_t sx = vx / CE;
                                    if (sx < nshard[2]) sd[srow + sx] = 1; }
                        }
                    }
                    size_t dense = 0, tot = (size_t)nshard[0] * nshard[1] * nshard[2];
                    for (size_t i = 0; i < tot; ++i) dense += sd[i];
                    free(sd);
                    printf("L%d COUNT (via L%d): %zu total | %zu dense (work) | %zu air\n",
                           level, clvl, tot, dense, tot - dense);
                    oracle_map_close(&cmap);
                }
            }
            free(p.items);
            if (omap.valid) oracle_map_close(&omap);
            continue;
        }

        // Pool sizes: compress = --threads (CPU-bound); download/upload default
        // to 2x that (network-bound, mostly blocked). Buffer pool + queue depths
        // bound in-flight RAM: each 1 GiB dense volume slot and each queued
        // encoded shard is ~1 GiB, so total slots ~= budget / 1 GiB.
        int enc_threads = a.threads > 0 ? a.threads : 3;
        int dl_threads = a.dl_threads > 0 ? a.dl_threads : enc_threads * 2;
        int ul_threads = a.ul_threads > 0 ? a.ul_threads : enc_threads * 2;
        if (dl_threads > 32) dl_threads = 32;
        if (ul_threads > 32) ul_threads = 32;

        // Split the RAM budget: dense-volume buffers (pool) get ~60%, encoded
        // shard queue ~40%. Each ~1 GiB. Keep small minimums so tiny budgets work.
        size_t total_slots = (size_t)(a.ram_budget_gb);
        if (total_slots < 4) total_slots = 4;
        size_t n_bufs = total_slots * 3 / 5; if (n_bufs < 2) n_bufs = 2;
        size_t shard_depth = total_slots - n_bufs; if (shard_depth < 2) shard_depth = 2;

        printf("L%d pools: dl=%d enc=%d ul=%d | %zu vol buffers, shard-queue depth %zu "
               "(~%.0f GiB in-flight)\n",
               level, dl_threads, enc_threads, ul_threads, n_bufs, shard_depth,
               (double)(n_bufs + shard_depth));

        buf_pool_init(&p.bufs, n_bufs);
        bq_init(&p.vol_q, n_bufs);        // at most n_bufs volumes can be outstanding
        bq_init(&p.shard_q, shard_depth);

        // Write the level metadata first so a reader sees a valid array.
        if (upload_level_metadata(&p, padded, quality, tau) != 0)
            fprintf(stderr, "L%d: metadata upload failed (continuing)\n", level);

        pthread_t *dl = malloc(dl_threads * sizeof(pthread_t));
        pthread_t *en = malloc(enc_threads * sizeof(pthread_t));
        pthread_t *ul = malloc(ul_threads * sizeof(pthread_t));
        for (int t = 0; t < ul_threads; ++t) pthread_create(&ul[t], NULL, upload_main, &p);
        for (int t = 0; t < enc_threads; ++t) pthread_create(&en[t], NULL, compress_main, &p);
        for (int t = 0; t < dl_threads; ++t) pthread_create(&dl[t], NULL, download_main, &p);

        // Drain in stage order: downloads finish -> close vol_q -> compressors
        // finish -> close shard_q -> uploaders finish.
        for (int t = 0; t < dl_threads; ++t) pthread_join(dl[t], NULL);
        bq_close(&p.vol_q);
        for (int t = 0; t < enc_threads; ++t) pthread_join(en[t], NULL);
        bq_close(&p.shard_q);
        for (int t = 0; t < ul_threads; ++t) pthread_join(ul[t], NULL);
        free(dl); free(en); free(ul);

        printf("L%d done: %zu ok, %zu failed, %zu resume-skipped of %zu "
               "(%zu whole-shard air-skips)\n",
               level, atomic_load(&p.done), atomic_load(&p.failed),
               atomic_load(&p.skipped), p.n_items, atomic_load(&p.air_shards));

        // Free the buffer pool slots. All buffers are back in free_slots now
        // (compress released each after encoding); close so the drain terminates.
        bq_close(&p.bufs.free_slots);
        void *b;
        while (bq_pop(&p.bufs.free_slots, &b)) free(b);
        bq_destroy(&p.bufs.free_slots);
        bq_destroy(&p.vol_q);
        bq_destroy(&p.shard_q);
        free(p.items);
        if (omap.valid) oracle_map_close(&omap);
    }

    // Upload the OME group zarr.json once (references levels as multiscales).
    // Minimal group marker; a full multiscales block can be added later.
    if (!a.dry_run) {
        const char *group =
            "{\n  \"zarr_format\": 3,\n  \"node_type\": \"group\",\n"
            "  \"attributes\": {}\n}\n";
        char remote[1536];
        snprintf(remote, sizeof(remote), "/%s/%s/%s/zarr.json",
                 a.remote_root, a.scroll, a.vol);
        sftp_upload(&sftp, remote, (const uint8_t *)group, strlen(group));
    }

    s3_client_free(client);
    return 0;
}
