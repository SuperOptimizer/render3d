/* Open-data listing parser (ctest label: quick — offline, localhost only).
 * An in-process HTTP stub plays S3 ListObjectsV2 and asserts the parser's
 * contract after the audit hardening:
 *  - a normal page yields dirs (CommonPrefixes) and files with sizes
 *  - keys with shell-hostile characters are dropped, never returned
 *  - an endlessly-truncated walk fails (bounded pages) instead of posing
 *    as a complete listing
 *  - HTTP errors and non-XML bodies never yield phantom entries */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "core/odbrowse.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

enum lmode { L_OK, L_TRUNC_FOREVER, L_500, L_HTML };
static _Atomic int g_mode = L_OK;
static _Atomic bool g_stop = false;
static int g_listen = -1;

static const char XML_OK[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<ListBucketResult><IsTruncated>false</IsTruncated>"
    "<CommonPrefixes><Prefix>PHercAAA/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>PHercBBB/</Prefix></CommonPrefixes>"
    "<Contents><Key>meta.json</Key><Size>123</Size></Contents>"
    "<Contents><Key>bad'name$(rm).zarr</Key><Size>5</Size></Contents>"
    "<Contents><Key>good-volume_1.129um.zarr</Key><Size>777</Size></Contents>"
    "</ListBucketResult>";

static const char XML_TRUNC[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<ListBucketResult><IsTruncated>true</IsTruncated>"
    "<NextContinuationToken>tokAAAA</NextContinuationToken>"
    "<Contents><Key>page-item.bin</Key><Size>1</Size></Contents>"
    "</ListBucketResult>";

static void send_all(int fd, const void *p, size_t n) {
  const uint8_t *b = p;
  while (n) {
    ssize_t w = write(fd, b, n);
    if (w <= 0) return;
    b += w;
    n -= (size_t)w;
  }
}

static void serve_one(int fd) {
  char req[4096];
  size_t got = 0;
  while (got + 1 < sizeof req) {
    ssize_t r = read(fd, req + got, sizeof req - 1 - got);
    if (r <= 0) return;
    got += (size_t)r;
    req[got] = 0;
    if (strstr(req, "\r\n\r\n")) break;
  }
  int mode = g_mode;
  const char *body;
  const char *status = "200 OK";
  const char *ctype = "application/xml";
  switch (mode) {
  case L_TRUNC_FOREVER: body = XML_TRUNC; break;
  case L_500:
    status = "500 Internal Server Error";
    body = "boom";
    ctype = "text/plain";
    break;
  case L_HTML:
    body = "<html><body>login required</body></html>";
    ctype = "text/html";
    break;
  default: body = XML_OK; break;
  }
  char hdr[256];
  snprintf(hdr, sizeof hdr,
           "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
           "Connection: close\r\n\r\n",
           status, ctype, strlen(body));
  send_all(fd, hdr, strlen(hdr));
  send_all(fd, body, strlen(body));
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

int main(void) {
  signal(SIGPIPE, SIG_IGN);
  g_listen = socket(AF_INET, SOCK_STREAM, 0);
  if (g_listen < 0) return 1;
  int one = 1;
  setsockopt(g_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = 0,
                          .sin_addr = {htonl(INADDR_LOOPBACK)}};
  if (bind(g_listen, (struct sockaddr *)&a, sizeof a) != 0 || listen(g_listen, 8) != 0)
    return 1;
  socklen_t al = sizeof a;
  if (getsockname(g_listen, (struct sockaddr *)&a, &al) != 0) return 1;
  pthread_t th;
  if (pthread_create(&th, NULL, stub_main, NULL) != 0) return 1;
  char url[64];
  snprintf(url, sizeof url, "http://127.0.0.1:%u", (unsigned)ntohs(a.sin_port));

  r3d_odlist l = {0};
  /* normal page: 2 dirs, safe files kept with sizes, hostile key dropped */
  g_mode = L_OK;
  CHECK(r3d_odlist_fetch(url, "", &l) == 0);
  CHECK(l.ndirs == 2);
  CHECK(l.ndirs == 2 && strcmp(l.dirs[0], "PHercAAA") == 0 &&
        strcmp(l.dirs[1], "PHercBBB") == 0);
  bool saw_meta = false, saw_good = false, saw_bad = false;
  for (uint32_t i = 0; i < l.nfiles; i++) {
    if (strcmp(l.files[i], "meta.json") == 0) saw_meta = l.file_sizes[i] == 123;
    if (strcmp(l.files[i], "good-volume_1.129um.zarr") == 0)
      saw_good = l.file_sizes[i] == 777;
    if (strstr(l.files[i], "bad")) saw_bad = true;
  }
  CHECK(saw_meta && saw_good && !saw_bad);
  r3d_odlist_free(&l);

  /* endlessly truncated: must fail closed, not return a partial listing */
  g_mode = L_TRUNC_FOREVER;
  memset(&l, 0, sizeof l);
  int rc = r3d_odlist_fetch(url, "", &l);
  CHECK(rc != 0);
  CHECK(l.ndirs == 0 && l.nfiles == 0);
  r3d_odlist_free(&l);

  /* HTTP 500 and an HTML body: errors or empty, never phantom entries */
  g_mode = L_500;
  memset(&l, 0, sizeof l);
  rc = r3d_odlist_fetch(url, "", &l);
  CHECK(rc != 0 || (l.ndirs == 0 && l.nfiles == 0));
  r3d_odlist_free(&l);
  g_mode = L_HTML;
  memset(&l, 0, sizeof l);
  rc = r3d_odlist_fetch(url, "", &l);
  CHECK(rc != 0 || (l.ndirs == 0 && l.nfiles == 0));
  r3d_odlist_free(&l);

  g_stop = true;
  shutdown(g_listen, SHUT_RDWR);
  close(g_listen);
  pthread_join(th, NULL);
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("od listing parser OK\n");
  return 0;
}
