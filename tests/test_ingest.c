/* Net-ingest fault injection (ctest label: quick — offline, localhost only).
 * An in-process HTTP stub plays a zarr chunk server for a manifest with no
 * local shards, so every brick read demand-fetches. Asserts the fail-closed
 * ingest contract on the cpuvol path:
 *  - an HTML/proxy 200 or a truncated 200 publishes NO cache artifact and
 *    the region heals on a later healthy retry
 *  - a 404 is a legitimately absent chunk: a permanent empty marker
 *  - a healthy 200 lands the exact bytes, and the .c5b disk cache serves
 *    later sessions without the network
 * Each phase opens a fresh r3d_cpuvol: backoff and negative caches are
 * per-open state, while the disk artifacts under test persist. */
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/cpuvol.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

#define IB 128u /* brick == chunk edge (raw zarr chunks) */
#define IDIM 256u
#define CHUNK_BYTES ((size_t)IB * IB * IB)

static uint8_t ipat(uint32_t x, uint32_t y, uint32_t z) {
  return (uint8_t)(31u + ((x * 7u + y * 13u + z * 29u) & 127u)); /* never 0 */
}

/* ---- HTTP stub ----------------------------------------------------------- */

enum stub_mode { STUB_HTML, STUB_TRUNC, STUB_404, STUB_OK };
static _Atomic int g_mode = STUB_HTML;
static _Atomic bool g_stop = false;
static int g_listen = -1;

static void send_all(int fd, const void *p, size_t n) {
  const uint8_t *b = p;
  while (n) {
    ssize_t w = write(fd, b, n);
    if (w <= 0) return; /* client aborted: fine */
    b += w;
    n -= (size_t)w;
  }
}

static void serve_one(int fd) {
  char req[2048];
  size_t got = 0;
  while (got + 1 < sizeof req) {
    ssize_t r = read(fd, req + got, sizeof req - 1 - got);
    if (r <= 0) return;
    got += (size_t)r;
    req[got] = 0;
    if (strstr(req, "\r\n\r\n")) break;
  }
  uint32_t li = 0, cz = 0, cy = 0, cx = 0;
  if (sscanf(req, "GET /%u/%u/%u/%u ", &li, &cz, &cy, &cx) != 4) {
    send_all(fd, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n", 47);
    return;
  }
  int mode = g_mode;
  char hdr[256];
  if (mode == STUB_404) {
    send_all(fd, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
             65);
    return;
  }
  if (mode == STUB_HTML) {
    const char *body = "<html><body>captive portal</body></html>";
    snprintf(hdr, sizeof hdr,
             "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %zu\r\n"
             "Connection: close\r\n\r\n",
             strlen(body));
    send_all(fd, hdr, strlen(hdr));
    send_all(fd, body, strlen(body));
    return;
  }
  size_t n = mode == STUB_TRUNC ? CHUNK_BYTES / 2 : CHUNK_BYTES;
  snprintf(hdr, sizeof hdr,
           "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
           "Content-Length: %zu\r\nConnection: close\r\n\r\n",
           n);
  send_all(fd, hdr, strlen(hdr));
  uint8_t *body = malloc(CHUNK_BYTES);
  if (!body) return;
  for (uint32_t z = 0; z < IB; z++)
    for (uint32_t y = 0; y < IB; y++)
      for (uint32_t x = 0; x < IB; x++)
        body[((size_t)z * IB + y) * IB + x] = ipat(cx * IB + x, cy * IB + y, cz * IB + z);
  send_all(fd, body, n);
  free(body);
}

static void *stub_main(void *arg) {
  (void)arg;
  while (!g_stop) {
    int fd = accept(g_listen, NULL, NULL);
    if (fd < 0) {
      if (g_stop) break;
      continue;
    }
    serve_one(fd);
    close(fd);
  }
  return NULL;
}

static int stub_start(uint16_t *port) {
  g_listen = socket(AF_INET, SOCK_STREAM, 0);
  if (g_listen < 0) return -1;
  int one = 1;
  setsockopt(g_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = 0,
                          .sin_addr = {htonl(INADDR_LOOPBACK)}};
  if (bind(g_listen, (struct sockaddr *)&a, sizeof a) != 0 || listen(g_listen, 8) != 0)
    return -1;
  socklen_t al = sizeof a;
  if (getsockname(g_listen, (struct sockaddr *)&a, &al) != 0) return -1;
  *port = ntohs(a.sin_port);
  return 0;
}

/* ---- dataset: manifest + source.json, no local shards -------------------- */

static int make_dataset(const char *root, uint16_t port) {
  if (mkdir(root, 0755) != 0 && errno != EEXIST) return -1;
  char p[640];
  snprintf(p, sizeof p, "%s/manifest.json", root);
  FILE *f = fopen(p, "w");
  if (!f) return -1;
  fprintf(f,
          "{\n  \"format\": \"render3d.c5d-lod.v1\",\n"
          "  \"shape\": [%u, %u, %u],\n"
          "  \"shard_shape\": [1024, 1024, 1024],\n"
          "  \"brick_shape\": [128, 128, 128],\n  \"levels\": [\n"
          "    {\"level\": 0, \"scale\": 1, \"shape\": [%u, %u, %u], \"shards\": [1, 1, 1],"
          " \"c5d\": \"c5d/L0/{z}_{y}_{x}.c5s\"},\n"
          "    {\"level\": 1, \"scale\": 2, \"shape\": [%u, %u, %u], \"shards\": [1, 1, 1],"
          " \"c5d\": \"c5d/L1/{z}_{y}_{x}.c5s\"}\n  ]\n}\n",
          IDIM, IDIM, IDIM, IDIM, IDIM, IDIM, IDIM / 2, IDIM / 2, IDIM / 2);
  if (fclose(f) != 0) return -1;
  snprintf(p, sizeof p, "%s/source.json", root);
  f = fopen(p, "w");
  if (!f) return -1;
  fprintf(f,
          "{\n  \"format\": \"render3d.c5d-source.v1\",\n"
          "  \"url\": \"http://127.0.0.1:%u\",\n  \"quality\": 2,\n"
          "  \"levels\": [\n    {\"level\": 0, \"chunk\": 128, \"raw\": true},\n"
          "    {\"level\": 1, \"chunk\": 128, \"raw\": true}\n  ]\n}\n",
          port);
  return fclose(f) == 0 ? 0 : -1;
}

static bool brick_file_state(const char *root, uint32_t bx, uint32_t by, uint32_t bz,
                             long *size) {
  char p[700];
  snprintf(p, sizeof p, "%s/bricks/L0/%u_%u_%u.c5b", root, bz, by, bx);
  struct stat st;
  if (stat(p, &st) != 0) return false;
  *size = (long)st.st_size;
  return true;
}

/* one sample through a fresh cpuvol (fresh backoff/negative-cache state) */
static double sample_once(const char *root, double x, double y, double z) {
  r3d_cpuvol v;
  if (r3d_cpuvol_open(&v, root, 8) != 0) return -2.0;
  double p[3] = {x, y, z};
  double val = r3d_cpuvol_tri(&v, 0, p, NULL);
  r3d_cpuvol_close(&v);
  return val;
}

int main(void) {
  signal(SIGPIPE, SIG_IGN); /* aborted client writes must not kill the stub */
  char tmp[512];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_ingest_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[600];
  snprintf(root, sizeof root, "%s/vol", tmp);
  uint16_t port = 0;
  if (stub_start(&port) != 0) {
    fprintf(stderr, "stub bind failed\n");
    return 1;
  }
  pthread_t th;
  if (pthread_create(&th, NULL, stub_main, NULL) != 0) return 1;
  if (make_dataset(root, port) != 0) {
    fprintf(stderr, "dataset setup failed\n");
    return 1;
  }
  long fsz = 0;

  /* HTML 200: cell abandoned, nothing published */
  g_mode = STUB_HTML;
  CHECK(sample_once(root, 60, 60, 60) == 0.0);
  CHECK(!brick_file_state(root, 0, 0, 0, &fsz));

  /* truncated 200 on a different brick: same fail-closed outcome */
  g_mode = STUB_TRUNC;
  CHECK(sample_once(root, 200, 60, 60) == 0.0);
  CHECK(!brick_file_state(root, 1, 0, 0, &fsz));

  /* 404: legitimately absent chunk = permanent empty marker */
  g_mode = STUB_404;
  CHECK(sample_once(root, 60, 200, 60) == 0.0);
  CHECK(brick_file_state(root, 0, 1, 0, &fsz) && fsz == 0);

  /* healthy 200: exact bytes, and a durable .c5b for later sessions */
  g_mode = STUB_OK;
  double v = sample_once(root, 200.0, 200.0, 60.0);
  CHECK(fabs(v - (double)ipat(200, 200, 60)) < 0.01); /* raw cache hit: exact */
  CHECK(brick_file_state(root, 1, 1, 0, &fsz) && fsz > 0);
  /* later session, server unreachable: served from the disk cache (lossy) */
  g_mode = STUB_404;
  double v2 = sample_once(root, 200.0, 200.0, 60.0);
  CHECK(fabs(v2 - (double)ipat(200, 200, 60)) < 4.0);

  /* the HTML-poisoned brick HEALS once the server behaves */
  g_mode = STUB_OK;
  double v3 = sample_once(root, 60.0, 60.0, 60.0);
  CHECK(fabs(v3 - (double)ipat(60, 60, 60)) < 0.01);
  CHECK(brick_file_state(root, 0, 0, 0, &fsz) && fsz > 0);

  g_stop = true;
  shutdown(g_listen, SHUT_RDWR);
  close(g_listen);
  pthread_join(th, NULL);
  /* cleanup: fixed layout */
  char p[720];
  for (uint32_t bz = 0; bz < 2; bz++)
    for (uint32_t by = 0; by < 2; by++)
      for (uint32_t bx = 0; bx < 2; bx++) {
        snprintf(p, sizeof p, "%s/bricks/L0/%u_%u_%u.c5b", root, bz, by, bx);
        unlink(p);
      }
  snprintf(p, sizeof p, "%s/bricks/L0", root);
  rmdir(p);
  snprintf(p, sizeof p, "%s/bricks", root);
  rmdir(p);
  snprintf(p, sizeof p, "%s/manifest.json", root);
  unlink(p);
  snprintf(p, sizeof p, "%s/source.json", root);
  unlink(p);
  rmdir(root);
  rmdir(tmp);
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("net-ingest fault injection OK\n");
  return 0;
}
