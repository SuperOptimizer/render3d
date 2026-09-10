#include "native_source.h"
#include "shard.h"
#include "brick.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  unsigned char *p;
  size_t n, cap;
  uint64_t start, end, total;
} response;
static size_t body(void *p, size_t s, size_t n, void *u) {
  response *r = u;
  if (s && n > SIZE_MAX / s)
    return 0;
  size_t bytes = s * n;
  if (bytes > r->cap - r->n)
    return 0;
  memcpy(r->p + r->n, p, bytes);
  r->n += bytes;
  return bytes;
}
static size_t header(char *p, size_t s, size_t n, void *u) {
  response *r = u;
  size_t bytes = s * n;
  if (bytes > 14 && !strncasecmp(p, "Content-Range:", 14)) {
    char line[160];
    size_t k = bytes < sizeof line - 1 ? bytes : sizeof line - 1;
    memcpy(line, p, k);
    line[k] = 0;
    unsigned long long a, b, c;
    if (sscanf(line + 14, " bytes %llu-%llu/%llu", &a, &b, &c) == 3) {
      r->start = a;
      r->end = b;
      r->total = c;
    }
  }
  return bytes;
}
static long fetch(CURL *c, const char *url, const char *range, response *r) {
  r->n = 0;
  r->total = 0;
  curl_easy_setopt(c, CURLOPT_URL, url);
  curl_easy_setopt(c, CURLOPT_RANGE, range);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 15L);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, 45L);
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, body);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, r);
  curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, header);
  curl_easy_setopt(c, CURLOPT_HEADERDATA, r);
  long code = 0;
  CURLcode rc = curl_easy_perform(c);
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
  return code == 404 ? 404 : rc == CURLE_OK ? code : 0;
}
static uint64_t le64(const unsigned char *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; i++)
    v |= (uint64_t)p[i] << (8 * i);
  return v;
}
int r3d_native_fetch(CURL *c, const char *url, const char *root, uint32_t l,
                     uint32_t x, uint32_t y, uint32_t z, bool *downloaded) {
  if (downloaded) *downloaded=false;
  char remote[2048], path[1600], dir[1500], tmp[1640];
  if (!c ||
      snprintf(remote, sizeof remote, "%s/%u/c/%u/%u/%u", url, l, z / 8, y / 8,
               x / 8) >= (int)sizeof remote ||
      snprintf(path, sizeof path, "%s/bricks/L%u/%u_%u_%u.volc", root, l, z, y,
               x) >= (int)sizeof path)
    return 0;
  struct stat st;
  if (!stat(path, &st)) {
    if (!st.st_size) return 2;
    /* A damaged local download must not permanently suppress a retry. */
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    if (fstat(fileno(f),&st)) { fclose(f); return 0; }
    if (!st.st_size) { fclose(f); return 2; }
    size_t n = st.st_size > 0 && st.st_size <= (16u << 20) ? (size_t)st.st_size : 0;
    uint8_t *buf = n ? malloc(n) : NULL;
    if (n && !buf) { fclose(f); return 0; }
    bool valid = buf && fread(buf,1,n,f)==n && r3d_validate_brick(buf,n)==0;
    free(buf);fclose(f);
    if (valid) return 1;
    struct stat current;
    if (stat(path,&current) || current.st_dev!=st.st_dev || current.st_ino!=st.st_ino ||
        current.st_size!=st.st_size || current.st_mtime!=st.st_mtime) return 0;
    if (unlink(path)) return 0;
  }
  unsigned char index[8196];
  response r = {.p = index, .cap = sizeof index};
  long code = fetch(c, remote, "-8196", &r);
  unsigned char *payload = NULL;
  size_t len = 0;
  if (code != 404) {
    if (code != 206 || r.n != sizeof index || r.total < sizeof index ||
        r.start != r.total - sizeof index || r.end != r.total - 1)
      return 0;
    uint32_t crc = (uint32_t)index[8192] | ((uint32_t)index[8193] << 8) |
                   ((uint32_t)index[8194] << 16) |
                   ((uint32_t)index[8195] << 24);
    if (volcomp_crc32c(index, 8192) != crc)
      return 0;
    const unsigned char *e = index + (((z % 8) * 8 + y % 8) * 8 + x % 8) * 16;
    uint64_t off = le64(e), n = le64(e + 8), total = r.total;
    if (off != UINT64_MAX || n != UINT64_MAX) {
      if (off > total - 8196 || n > total - 8196 - off || n < 8 ||
          n > (16u << 20))
        return 0;
      len = (size_t)n;
      payload = malloc(len);
      if (!payload)
        return 0;
      char range[96];
      snprintf(range, sizeof range, "%llu-%llu", (unsigned long long)off,
               (unsigned long long)(off + n - 1));
      r = (response){.p = payload, .cap = len};
      code = fetch(c, remote, range, &r);
      if (code != 206 || r.n != len || r.total != total || r.start != off ||
          r.end != off + n - 1 || r3d_validate_brick(payload, len) != 0) {
        free(payload);
        return 0;
      }
    }
  }
  snprintf(dir, sizeof dir, "%s/bricks", root);
  mkdir(dir, 0755);
  snprintf(dir, sizeof dir, "%s/bricks/L%u", root, l);
  mkdir(dir, 0755);
  snprintf(tmp, sizeof tmp, "%s.tmp.XXXXXX", path);
  int fd = mkstemp(tmp);
  int ok = 0;
  if (fd >= 0) {
    FILE *f = fdopen(fd, "wb");
    if (f) {
      ok = !len || fwrite(payload, 1, len, f) == len;
      if (fclose(f))
        ok = 0;
    } else
      close(fd);
    if (ok && rename(tmp, path))
      ok = 0;
    if (!ok)
      unlink(tmp);
  }
  free(payload);
  if (downloaded) *downloaded=ok!=0;
  return ok ? (len ? 1 : 2) : 0;
}
