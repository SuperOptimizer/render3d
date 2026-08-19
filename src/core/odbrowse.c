#include "core/odbrowse.h"

#include <curl/curl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct sbuf {
  char *p;
  size_t n, cap;
} sbuf;

/* One ListObjectsV2 page is at most 1000 keys; real bodies are KB-scale.
 * Cap growth defensively so a misbehaving/hostile endpoint cannot make one
 * page consume unbounded memory -- returning a short write count fails the
 * transfer through curl's normal error path. */
#define OD_MAX_BODY ((size_t)64 << 20)

static size_t sbuf_write(const void *data, size_t sz, size_t nm, void *ud) {
  sbuf *b = ud;
  size_t n = sz * nm;
  if (b->n + n + 1 > OD_MAX_BODY) return 0;
  if (b->n + n + 1 > b->cap) {
    size_t nc = b->cap ? b->cap * 2 : (256u << 10);
    while (nc < b->n + n + 1) nc *= 2;
    char *np = realloc(b->p, nc);
    if (!np) return 0;
    b->p = np;
    b->cap = nc;
  }
  memcpy(b->p + b->n, data, n);
  b->n += n;
  b->p[b->n] = 0;
  return n;
}

/* Defense in depth for the shell boundary further downstream (open-data
 * names are later interpolated into command strings by the caller): reject
 * any listing entry whose name is not built entirely from an ordinary
 * filename alphabet, and reject "." / ".." / a leading "-" which are legal
 * under that alphabet but dangerous as a path or CLI-argument component. */
static bool name_is_safe(const char *s, size_t len) {
  if (len == 0) return false;
  if (s[0] == '-') return false;
  if ((len == 1 && s[0] == '.') || (len == 2 && s[0] == '.' && s[1] == '.'))
    return false;
  for (size_t i = 0; i < len; i++) {
    char c = s[i];
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

static int push_str(char ***arr, uint32_t *n, const char *s, size_t len) {
  char **na = realloc(*arr, ((size_t)*n + 1) * sizeof **arr);
  if (!na) return -1;
  *arr = na;
  char *c = malloc(len + 1);
  if (!c) return -1;
  memcpy(c, s, len);
  c[len] = 0;
  na[(*n)++] = c;
  return 0;
}

/* extract the text of <tag>...</tag> starting at *p; advances *p past it */
static const char *xml_next(const char **p, const char *open, const char *close,
                            size_t *len) {
  const char *a = strstr(*p, open);
  if (!a) return NULL;
  a += strlen(open);
  const char *b = strstr(a, close);
  if (!b) return NULL;
  *p = b + strlen(close);
  *len = (size_t)(b - a);
  return a;
}

int r3d_odlist_fetch(const char *bucket_url, const char *prefix, r3d_odlist *out) {
  memset(out, 0, sizeof *out);
  CURL *curl = curl_easy_init();
  if (!curl) return -1;
  sbuf body = {0};
  int rc = -1;
  char token[1024] = "";
  for (int page = 0; page < 32; page++) { /* paginated (1000 keys/page) */
    char *pe = curl_easy_escape(curl, prefix, 0);
    char url[2048];
    if (token[0])
      snprintf(url, sizeof url,
               "%s/?list-type=2&delimiter=%%2F&prefix=%s&continuation-token=%s",
               bucket_url, pe ? pe : "", token);
    else
      snprintf(url, sizeof url, "%s/?list-type=2&delimiter=%%2F&prefix=%s", bucket_url,
               pe ? pe : "");
    curl_free(pe);
    body.n = 0;
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, sbuf_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    long code = 0;
    if (curl_easy_perform(curl) != CURLE_OK ||
        (curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code), code != 200) || !body.p)
      goto done;
    size_t plen = strlen(prefix);
    const char *p = body.p;
    size_t len;
    const char *s;
    while ((s = xml_next(&p, "<CommonPrefixes><Prefix>", "</Prefix>", &len)) != NULL) {
      if (len <= plen) continue;
      size_t l = len - plen;
      if (s[plen + l - 1] == '/') l--; /* drop trailing slash */
      if (!name_is_safe(s + plen, l)) {
        fprintf(stderr, "odbrowse: dropping unsafe directory name %.*s\n", (int)l, s + plen);
        continue;
      }
      if (push_str(&out->dirs, &out->ndirs, s + plen, l) != 0) goto done;
    }
    p = body.p;
    while ((s = xml_next(&p, "<Contents><Key>", "</Key>", &len)) != NULL) {
      if (len <= plen) continue;
      uint64_t sz = 0;
      const char *q = p;
      size_t sl;
      const char *ss = xml_next(&q, "<Size>", "</Size>", &sl);
      if (ss) sz = strtoull(ss, NULL, 10);
      size_t l = len - plen;
      if (!name_is_safe(s + plen, l)) {
        fprintf(stderr, "odbrowse: dropping unsafe file name %.*s\n", (int)l, s + plen);
        continue;
      }
      if (push_str(&out->files, &out->nfiles, s + plen, l) != 0) goto done;
      uint64_t *nf = realloc(out->file_sizes, out->nfiles * sizeof *nf);
      if (!nf) goto done;
      out->file_sizes = nf;
      out->file_sizes[out->nfiles - 1] = sz;
    }
    const char *tp = body.p;
    const char *tok = xml_next(&tp, "<NextContinuationToken>", "</NextContinuationToken>",
                               &len);
    if (!tok) { /* no continuation token: this really is the last page */
      rc = 0;
      goto done;
    }
    /* A token present but unusable (too long to store, or one we cannot
     * escape) means more of the bucket exists beyond what was collected.
     * Treat that the same as any other fetch failure -- fail closed rather
     * than return a partial listing dressed up as a complete one. */
    if (len >= sizeof token) goto done;
    /* token is URL-safe base64ish but may contain chars needing escaping */
    char rawtok[1024];
    memcpy(rawtok, tok, len);
    rawtok[len] = 0;
    char *te = curl_easy_escape(curl, rawtok, 0);
    if (!te) goto done;
    snprintf(token, sizeof token, "%s", te);
    curl_free(te);
    if (!token[0]) goto done;
    continue;
  }
  /* page budget exhausted with a continuation token still pending: the
   * bucket has more entries than were listed. Same fail-closed rule. */
done:
  if (rc != 0) r3d_odlist_free(out);
  free(body.p);
  curl_easy_cleanup(curl);
  return rc;
}

void r3d_odlist_free(r3d_odlist *l) {
  for (uint32_t i = 0; i < l->ndirs; i++) free(l->dirs[i]);
  for (uint32_t i = 0; i < l->nfiles; i++) free(l->files[i]);
  free(l->dirs);
  free(l->files);
  free(l->file_sizes);
  memset(l, 0, sizeof *l);
}
