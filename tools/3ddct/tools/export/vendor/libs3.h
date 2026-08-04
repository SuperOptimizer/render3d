/*
 * libs3 -- a minimal C23 S3 client.
 *
 * A small replacement for the S3 subset of the AWS SDK: object GET/PUT/HEAD/
 * DELETE/COPY, byte-range reads, multipart upload, ListObjectsV2, and full AWS
 * credential resolution (env / INI / SSO / EC2 IMDSv2) with SigV4 signing.
 *
 * Only hard dependency: libcurl (>= 7.75 for built-in SigV4). JSON and XML are
 * parsed by tiny internal scrapers, not a general-purpose library.
 *
 * Pure C23 API. The implementation lives in libs3.c; include this header and
 * link against the compiled object/library -- curl does not leak into the
 * consumer's translation units.
 *
 * Threading: an s3_client is safe to share across threads. It keeps a
 * thread-local curl handle internally so concurrent calls do not contend.
 * The credential cache and the global abort flag are process-wide.
 *
 * Memory: every call that fills an out-parameter with owned memory has a
 * matching *_free. Calling *_free on a zero-initialised struct is safe.
 */
#ifndef LIBS3_H
#define LIBS3_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   /* FILE (s3_get_to_file) */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

/*
 * Return code for every libs3 function. S3_OK is zero; all errors are
 * negative so `if (s3_get(...) != S3_OK)` and `if (rc < 0)` both work.
 * The HTTP status of a completed request lives in s3_response.status,
 * separately from these transport-level codes.
 */
typedef enum s3_status {
    S3_OK = 0,
    S3_ERR_INVALID_ARG  = -1,  /* NULL/garbage argument or unparseable URL */
    S3_ERR_OOM          = -2,  /* allocation failed */
    S3_ERR_CURL         = -3,  /* libcurl init or transport failure */
    S3_ERR_HTTP         = -4,  /* request completed with a non-2xx status */
    S3_ERR_NO_CREDS     = -5,  /* credential resolution found nothing */
    S3_ERR_PARSE        = -6,  /* malformed JSON/XML response */
    S3_ERR_IO           = -7,  /* local file open/read failure */
    S3_ERR_ABORTED      = -8,  /* global abort flag was set */
    S3_ERR_BUSY         = -9,  /* async request not yet complete */
} s3_status;

/* Human-readable string for a status code (static storage, never NULL). */
const char *s3_status_str(s3_status st);

/* ------------------------------------------------------------------ */
/* S3 URL parsing                                                      */
/* ------------------------------------------------------------------ */

/*
 * Parsed s3:// URL. Recognises "s3://bucket/key", "S3://bucket/key", and the
 * region-qualified "s3+REGION://bucket/key" form (e.g. s3+us-west-2://...).
 * region is the empty string when the URL did not specify one.
 */
typedef struct s3_url {
    char *bucket;
    char *key;
    char *region;
} s3_url;

/* True if `url` looks like an s3:// / S3:// / s3+REGION:// URL. */
bool s3_url_is_s3(const char *url);

/*
 * Parse an S3 URL into `out` (caller owns; release with s3_url_free).
 * Returns S3_ERR_INVALID_ARG if `url` is not an S3 URL.
 */
s3_status s3_url_parse(const char *url, s3_url *out);

void s3_url_free(s3_url *u);

/*
 * Render the virtual-hosted HTTPS form of a parsed S3 URL into `buf`:
 *   https://<bucket>.s3.<region>.amazonaws.com/<key>
 * (region omitted from the host when empty). Returns S3_ERR_INVALID_ARG if
 * `buf` is too small; on success the result is NUL-terminated.
 */
s3_status s3_url_to_https(const s3_url *u, char *buf, size_t buflen);

/* ------------------------------------------------------------------ */
/* Credentials                                                         */
/* ------------------------------------------------------------------ */

/*
 * AWS SigV4 credentials. session_token is set only for temporary (STS)
 * credentials. region may be empty if not discoverable; pass it explicitly
 * via s3_config if so.
 */
typedef struct s3_credentials {
    char *access_key;
    char *secret_key;
    char *session_token;
    char *region;
} s3_credentials;

void s3_credentials_free(s3_credentials *c);

/*
 * Read credentials from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
 * AWS_SESSION_TOKEN / AWS_DEFAULT_REGION. Missing vars yield empty strings;
 * returns S3_ERR_NO_CREDS if access or secret key is absent.
 */
s3_status s3_credentials_from_env(s3_credentials *out);

/*
 * Full credential resolution, tried in order:
 *   1. explicit `profile` (or $AWS_PROFILE) via `aws configure export-credentials`
 *   2. EC2 instance role via IMDSv2 -- queried directly, cached in-process and
 *      refreshed shortly before expiry (no `aws` subprocess fork-storm)
 *   3. SSO profiles discovered in ~/.aws/config
 *   4. default `aws configure export-credentials`
 *   5. ~/.aws/credentials and ~/.aws/config INI files
 *   6. environment variables
 * Pass profile=NULL or "default" for the default profile. Thread-safe; the
 * IMDSv2 cache makes this cheap to call repeatedly on a long-running job.
 */
s3_status s3_credentials_load(const char *profile, s3_credentials *out);

/*
 * Optional per-request credential provider. Set s3_config.cred_provider to
 * have libs3 re-resolve credentials before every request instead of freezing
 * them at client construction -- required for multi-hour jobs on EC2 instance
 * -role (STS) credentials that rotate. The callback must fill `*out` with
 * freshly owned strings (libs3 frees them) and return S3_OK; backing it with
 * s3_credentials_load is fine since that is cache-served.
 */
typedef s3_status (*s3_cred_provider_fn)(void *userdata, s3_credentials *out);

/* ------------------------------------------------------------------ */
/* Client                                                              */
/* ------------------------------------------------------------------ */

typedef struct s3_client s3_client;

/*
 * Client configuration. Zero-initialise and override what you need; passing
 * NULL to s3_client_new uses all defaults with no credentials (anonymous,
 * suitable for public buckets).
 *
 * Credential precedence: cred_provider (if set) is called per request;
 * otherwise the static `creds` is used; if both are empty the request is
 * unsigned/anonymous. For non-S3 plain HTTP, bearer_token or basic
 * user/pass may be set instead.
 */
typedef struct s3_config {
    s3_credentials       creds;            /* static credentials (optional) */
    s3_cred_provider_fn  cred_provider;    /* per-request resolver (optional) */
    void                *cred_userdata;    /* passed to cred_provider */

    const char *region;                    /* fallback region when creds omit it */

    const char *bearer_token;              /* non-S3 HTTP: Authorization: Bearer */
    const char *basic_user;                /* non-S3 HTTP: basic auth user */
    const char *basic_pass;                /* non-S3 HTTP: basic auth pass */

    /*
     * Non-AWS S3 endpoint override (e.g. MinIO, localstack). When set, an
     * s3://bucket/key URL resolves to "<scheme>://<endpoint>/<bucket>/<key>"
     * (path-style) instead of the virtual-hosted amazonaws.com host. Leave
     * NULL for real AWS. `endpoint` is host[:port] only, no scheme.
     */
    const char *endpoint;
    bool  endpoint_insecure;               /* use http:// for endpoint (MinIO dev) */

    long  connect_timeout_s;               /* default 10 */
    long  transfer_timeout_s;              /* default 30 */
    int   max_retries;                     /* default 3 (5xx/401/403/network) */
    bool  follow_redirects;                /* default true */
    const char *user_agent;                /* default "libs3/1.0" */

    /*
     * s3_get_batch range coalescing: requests on the same URL whose ranges
     * are adjacent or separated by at most this many bytes are merged into
     * one transfer (gap bytes are downloaded and discarded). 0 -> default
     * (256 KiB); negative -> disabled. Merged spans are capped at 32 MiB.
     */
    int64_t coalesce_gap;
} s3_config;

/* Create a client. Returns NULL on allocation/curl-init failure. */
s3_client *s3_client_new(const s3_config *cfg);
void       s3_client_free(s3_client *c);

/* Last error message for the calling thread (static storage, never NULL). */
const char *s3_client_last_error(const s3_client *c);

/* ------------------------------------------------------------------ */
/* Process-wide fast abort                                             */
/* ------------------------------------------------------------------ */

/*
 * Flip a process-global flag that makes every in-flight transfer return
 * promptly (sub-millisecond on an active socket) and cancels pending
 * retries. Use on application shutdown so worker pools are not stuck inside
 * a multi-second S3 timeout. Affects all clients in the process.
 */
void s3_global_abort(void);
void s3_global_reset_abort(void);   /* clear the flag (e.g. between tests) */
bool s3_global_is_aborted(void);

/* ------------------------------------------------------------------ */
/* Responses                                                           */
/* ------------------------------------------------------------------ */

/*
 * Result of a single request. `status` is the HTTP status code (0 if the
 * transport never completed). `body`/`body_len` is the response payload
 * (heap-owned; NUL-terminated one past body_len for convenience but may
 * contain embedded NULs). Release with s3_response_free.
 */
typedef struct s3_response {
    long    status;
    uint8_t *body;
    size_t  body_len;
    char    *content_type;     /* may be NULL */
    uint64_t content_length;   /* from the Content-Length header, 0 if absent */
    char    *etag;             /* from the ETag header, NULL if absent */
    char    *last_modified;    /* from the Last-Modified header, NULL if absent */
} s3_response;

void s3_response_free(s3_response *r);

static inline bool s3_response_ok(const s3_response *r) {
    return r && r->status >= 200 && r->status < 300;
}
static inline bool s3_response_not_found(const s3_response *r) {
    return r && r->status == 404;
}

/* ------------------------------------------------------------------ */
/* Object operations                                                   */
/* ------------------------------------------------------------------ */
/*
 * `url` accepts either an s3:// form (auto-converted and SigV4-signed) or a
 * plain https:// URL. Each fills `*resp` (caller frees with
 * s3_response_free) and returns S3_OK when the transport completed -- check
 * resp->status for the HTTP result. A non-2xx status yields S3_ERR_HTTP but
 * still populates resp.
 */
s3_status s3_get(s3_client *c, const char *url, s3_response *resp);

/* Like s3_get but streams the body straight to `sink` (a writable FILE*) in
 * constant memory -- for large objects that shouldn't buffer fully in RAM.
 * resp->body stays NULL; resp->status and headers are still populated. */
s3_status s3_get_to_file(s3_client *c, const char *url, FILE *sink, s3_response *resp);

s3_status s3_get_range(s3_client *c, const char *url,
                       uint64_t offset, uint64_t length, s3_response *resp);

/*
 * Parallel single-object download. Fetches [offset, offset+length) of `url`
 * (length 0 = the whole object, sized via a HEAD) by splitting it into
 * ~`part_size`-byte ranges fetched concurrently (<= `max_concurrency` at once)
 * and assembling them in order into resp->body. One fat object, many sockets:
 * saturates bandwidth that a single TCP stream cannot. part_size 0 -> 8 MiB,
 * max_concurrency 0 -> 16. resp->status is 200/206 on success; any failed part
 * yields S3_ERR_HTTP (resp still freed-by-caller via s3_response_free).
 */
s3_status s3_get_parallel(s3_client *c, const char *url,
                          uint64_t offset, uint64_t length,
                          uint64_t part_size, size_t max_concurrency,
                          s3_response *resp);

s3_status s3_head(s3_client *c, const char *url, s3_response *resp);

s3_status s3_put(s3_client *c, const char *url,
                 const void *data, size_t len,
                 const char *content_type /* NULL -> octet-stream */,
                 s3_response *resp);

/* Streams `path` from disk in constant memory. */
s3_status s3_put_file(s3_client *c, const char *url,
                      const char *path,
                      const char *content_type /* NULL -> octet-stream */,
                      s3_response *resp);

/*
 * Optimistic-concurrency PUT. The write is applied only if the object's
 * current ETag matches `if_match` (typically an ETag from a prior GET);
 * if another writer changed it in between, S3 responds 412 Precondition
 * Failed and the call returns S3_ERR_HTTP with resp->status == 412 so
 * the caller can re-read and retry. Pass `if_match` == "*" to require
 * that the object already exists (any version). NULL/empty `if_match`
 * behaves like a plain s3_put. Lets collaborative volume edits do safe
 * read-modify-write on shared metadata without clobbering each other.
 */
s3_status s3_put_if_match(s3_client *c, const char *url,
                          const void *data, size_t len,
                          const char *content_type /* NULL ok */,
                          const char *if_match,
                          s3_response *resp);

s3_status s3_delete(s3_client *c, const char *url, s3_response *resp);

/* Server-side copy via x-amz-copy-source. src_url/dst_url are s3:// URLs. */
s3_status s3_copy(s3_client *c, const char *src_url, const char *dst_url,
                  s3_response *resp);

/*
 * Conditional GET. If `if_none_match` is a previously-seen ETag and the
 * object is unchanged, S3 responds 304 Not Modified with no body; the
 * call still returns S3_OK and resp->status == 304 so callers can keep
 * their cached copy. Pass NULL/empty if_none_match for an unconditional
 * GET. Lets VC's chunk cache revalidate cheaply instead of re-fetching.
 */
s3_status s3_get_conditional(s3_client *c, const char *url,
                             const char *if_none_match, s3_response *resp);

/* ------------------------------------------------------------------ */
/* Batched ranged GET (curl_multi)                                     */
/* ------------------------------------------------------------------ */
/*
 * Fetch many byte ranges concurrently over pooled connections -- the hot
 * path for chunk-heavy zarr rendering. `reqs` is an array of `n` range
 * requests; `out` must point at `n` zero-initialised s3_response slots
 * which the call fills (caller frees each with s3_response_free). At most
 * `max_concurrency` transfers run at once (0 -> a sane default). The
 * function returns S3_OK if every transfer completed at the transport
 * level; inspect each out[i].status for per-object HTTP results. A length
 * of 0 fetches the whole object.
 *
 * Adjacent/near-adjacent ranges on the same URL are coalesced into one
 * transfer (see s3_config.coalesce_gap) and split back transparently:
 * every out[i] still gets its own status, body and body_len. Responses
 * carved from a merged transfer share that transfer's content_type/etag/
 * last_modified (duplicated; s3_response_free as usual).
 */
typedef struct s3_range_req {
    const char *url;
    uint64_t    offset;
    uint64_t    length;   /* 0 -> whole object */
    /*
     * Optional caller-owned destination of `length` bytes (requires
     * length > 0). When set, the body is written straight into `dst`
     * (zero allocations, zero copies on the direct path); the response's
     * body stays NULL and body_len reports bytes written. Bodies larger
     * than `length` (e.g. a Range-ignoring server) are refused at the
     * transport level rather than overrunning the buffer.
     */
    uint8_t    *dst;
} s3_range_req;

s3_status s3_get_batch(s3_client *c,
                       const s3_range_req *reqs, size_t n,
                       size_t max_concurrency,
                       s3_response *out);

/*
 * Zero-allocation single ranged GET: the body is written straight into
 * `dst` (capacity = `length`). resp->body stays NULL; resp->body_len
 * reports bytes written (0 unless the HTTP status was 200/206 -- error
 * bodies are not delivered through `dst`). Same retry behaviour as
 * s3_get_range.
 */
s3_status s3_get_range_into(s3_client *c, const char *url,
                            uint64_t offset, uint64_t length,
                            uint8_t *dst, s3_response *resp);

/* ------------------------------------------------------------------ */
/* Asynchronous batched GET (submit / poll / cancel)                    */
/* ------------------------------------------------------------------ */
/*
 * Non-blocking variant of s3_get_batch for latency-sensitive callers
 * (render loops, frame-budgeted I/O). Submit returns immediately; the
 * transfers make progress only inside s3_batch_poll / s3_batch_wait,
 * called from the SAME THREAD that submitted (the batch borrows the
 * thread's persistent connection pool). Typical frame loop:
 *
 *   s3_batch *b;
 *   s3_batch_submit(c, reqs, n, 16, &b);
 *   ...per frame:
 *   s3_batch_poll(b, 2);                       // <= 2 ms of I/O pumping
 *   for (i ...) if (s3_batch_ready(b, i)) {
 *       s3_response r; s3_batch_take(b, i, &r); ... s3_response_free(&r);
 *   }
 *   ...view changed:
 *   s3_batch_cancel(b, i);                     // stale chunk: stop now
 *   ...teardown:
 *   s3_batch_free(b);                          // cancels whatever is left
 *
 * Requests are NOT range-coalesced (members of a merged transfer could
 * not be cancelled independently). reqs[].url / reqs[].dst buffers are
 * copied/borrowed at submit: urls are duplicated, dst buffers must stay
 * alive until the request is taken, cancelled, or the batch is freed.
 * A batch must be polled, taken from, cancelled, waited on and freed on
 * its submitting thread.
 */
typedef struct s3_batch s3_batch;

/* Start a batch: validates inputs, begins up to max_concurrency
 * transfers (0 -> a sane default), returns without blocking. */
s3_status s3_batch_submit(s3_client *c,
                          const s3_range_req *reqs, size_t n,
                          size_t max_concurrency,
                          s3_batch **out);

/*
 * Pump the batch's transfers for at most `budget_ms` milliseconds
 * (0 -> one non-blocking pump; negative -> until every request is
 * terminal). Returns the total number of terminal (done or cancelled)
 * requests, or -1 if `b` is NULL.
 */
int s3_batch_poll(s3_batch *b, long budget_ms);

/* True once request i has completed (its response is takeable). */
bool s3_batch_ready(const s3_batch *b, size_t i);

/*
 * Move request i's response out to the caller (caller frees with
 * s3_response_free). Returns S3_ERR_BUSY while in flight, S3_ERR_ABORTED
 * if it was cancelled, S3_ERR_INVALID_ARG if already taken / i out of
 * range. Inspect out->status for the HTTP result as with s3_get_batch.
 */
s3_status s3_batch_take(s3_batch *b, size_t i, s3_response *out);

/* Cancel request i: a pending request never starts; an in-flight one is
 * torn down immediately (its connection is dropped). No-op once the
 * request is terminal. */
void s3_batch_cancel(s3_batch *b, size_t i);

/* Block until every request is terminal. Returns S3_OK if every
 * completed transfer succeeded at the transport level (per-request HTTP
 * status still in each response), or the first error otherwise. */
s3_status s3_batch_wait(s3_batch *b);

/* Cancel anything still running, free untaken responses, release the
 * batch. NULL-safe. */
void s3_batch_free(s3_batch *b);

/*
 * Open `nconn` connections to `url`'s host from the calling thread's batch
 * pool (tiny 1-byte ranged GETs), so the first real s3_get_batch on this
 * thread doesn't pay TCP+TLS handshakes. Connections are per-thread; call
 * it on the thread that will issue the batches.
 */
s3_status s3_prewarm(s3_client *c, const char *url, size_t nconn);

/* ------------------------------------------------------------------ */
/* Multipart upload                                                    */
/* ------------------------------------------------------------------ */
/*
 * Lifecycle: s3_multipart_create -> N x s3_multipart_upload_part (or
 * _upload_part_file) -> s3_multipart_complete. On any failure call
 * s3_multipart_abort to release the staged parts server-side. Part numbers
 * are 1-based and must be >= 5 MiB except the last (S3 rule).
 */
typedef struct s3_multipart s3_multipart;

/* Begins an upload to `url`. Returns a handle (free via complete/abort). */
s3_status s3_multipart_create(s3_client *c, const char *url,
                              const char *content_type /* NULL ok */,
                              s3_multipart **out);

s3_status s3_multipart_upload_part(s3_multipart *m, int part_number,
                                   const void *data, size_t len);

s3_status s3_multipart_upload_part_file(s3_multipart *m, int part_number,
                                        const char *path);

/*
 * One part for a parallel upload. Set exactly one source: either
 * (`data`,`len`) for an in-memory part, or `path` for a file streamed
 * from disk. `part_number` is 1-based; per S3 every part except the
 * last must be >= 5 MiB.
 */
typedef struct s3_part_src {
    int         part_number;
    const void *data;    /* in-memory source (NULL if using path) */
    size_t      len;
    const char *path;    /* file source (NULL if using data) */
} s3_part_src;

/*
 * Upload `n` parts concurrently over pooled connections (curl_multi),
 * at most `max_concurrency` in flight (0 -> a sane default). This is
 * the fast path for writing large objects -- e.g. a multi-GB
 * recompressed zarr volume -- versus sequential
 * s3_multipart_upload_part calls. Each part's ETag is captured and
 * registered on `m` for s3_multipart_complete. Returns S3_OK only if
 * every part succeeded; on any failure the caller should
 * s3_multipart_abort. Parts may be supplied across multiple calls
 * (e.g. in waves); ordering at completion follows S3's part-number
 * sort, not call order.
 */
s3_status s3_multipart_upload_parts_parallel(s3_multipart *m,
                                             const s3_part_src *parts,
                                             size_t n,
                                             size_t max_concurrency);

/* Finalises the object; frees `m` regardless of outcome. */
s3_status s3_multipart_complete(s3_multipart *m, s3_response *resp);

/* Cancels the upload and frees `m`. Safe to call after a partial failure. */
s3_status s3_multipart_abort(s3_multipart *m);

/* ------------------------------------------------------------------ */
/* ListObjectsV2                                                       */
/* ------------------------------------------------------------------ */

typedef struct s3_object {
    char    *key;
    uint64_t size;
    char    *etag;   /* may be NULL */
} s3_object;

/*
 * One page of a ListObjectsV2 response. `prefixes` are the CommonPrefixes
 * (sub-"directories") when a delimiter was used; `objects` are the keys.
 * When `is_truncated` is true, pass `next_continuation_token` back into
 * s3_list to fetch the next page. Release with s3_list_result_free.
 */
typedef struct s3_list_result {
    char       **prefixes;
    size_t       prefix_count;
    s3_object   *objects;
    size_t       object_count;
    char        *next_continuation_token;  /* NULL when not truncated */
    bool         is_truncated;
} s3_list_result;

void s3_list_result_free(s3_list_result *r);

/*
 * One page of ListObjectsV2 against the bucket/prefix in `s3_url_prefix`
 * (an s3:// URL whose key part is treated as the prefix). `delimiter` is
 * usually "/" to list one level; pass NULL for a full recursive listing.
 * `continuation_token` is NULL for the first page.
 */
s3_status s3_list(s3_client *c, const char *s3_url_prefix,
                  const char *delimiter,
                  const char *continuation_token,
                  s3_list_result *out);

/*
 * Extended ListObjectsV2 with optional controls. Any field left
 * NULL/0 is omitted. `max_keys` caps objects per page (S3 max 1000;
 * smaller pages improve UI responsiveness on huge prefixes).
 * `start_after` resumes lexicographically after a given key (a cheap
 * alternative to a continuation token for the first page).
 */
typedef struct s3_list_params {
    const char *delimiter;            /* "/" for one level, NULL recursive */
    const char *continuation_token;   /* NULL for first page */
    const char *start_after;          /* NULL to start at the beginning */
    int         max_keys;             /* 0 -> server default (1000) */
} s3_list_params;

s3_status s3_list_ex(s3_client *c, const char *s3_url_prefix,
                     const s3_list_params *params,
                     s3_list_result *out);

/*
 * Callback invoked once per page by s3_list_all. Return true to continue
 * paginating, false to stop early. The s3_list_result is owned by the
 * iterator and freed after the callback returns -- copy what you keep.
 */
typedef bool (*s3_list_page_fn)(void *userdata, const s3_list_result *page);

/* Auto-paginates s3_list, invoking `cb` for every page. */
s3_status s3_list_all(s3_client *c, const char *s3_url_prefix,
                      const char *delimiter,
                      s3_list_page_fn cb, void *userdata);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBS3_H */
