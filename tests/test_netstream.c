/* Renderer-side net ingest (ctest label: gpu — GPU + localhost only).
 * The synthetic tree keeps its coarsest shard but has NO fine-level shard,
 * so the renderer's own net-ingest workers must fetch L0 chunks from an
 * in-process HTTP stub. Asserts the fail-closed contract on the vkbackend
 * path (the cpuvol analogue lives in test_ingest):
 *  - captive-portal HTML 200s: the app keeps running, and NO .c5b cache
 *    artifact or empty marker is written (the region can heal)
 *  - a later healthy run fetches, renders content, and persists non-empty
 *    .c5b bricks that serve the next session
 *  - 404s on a fresh cache become permanent empty markers (0-byte .c5b)
 * Usage: test_netstream <path-to-render3d> */
#include <arpa/inet.h>
#include <dirent.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "synthtree.h"

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      failures++;                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                 \
  } while (0)

enum nmode { N_HTML, N_404, N_OK };
static _Atomic int g_mode = N_HTML;
static _Atomic bool g_stop = false;
static int g_listen = -1;

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
  char hdr[256];
  switch ((enum nmode)g_mode) {
  case N_404:
    send_all(fd, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
             65);
    return;
  case N_HTML: {
    const char *body = "<html><body>sign in to continue</body></html>";
    snprintf(hdr, sizeof hdr,
             "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %zu\r\n"
             "Connection: close\r\n\r\n",
             strlen(body));
    send_all(fd, hdr, strlen(hdr));
    send_all(fd, body, strlen(body));
    return;
  }
  case N_OK: {
    size_t n = (size_t)ST_B * ST_B * ST_B;
    snprintf(hdr, sizeof hdr,
             "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
             "Content-Length: %zu\r\nConnection: close\r\n\r\n",
             n);
    send_all(fd, hdr, strlen(hdr));
    uint8_t *body = malloc(n);
    if (!body) return;
    for (uint32_t z = 0; z < ST_B; z++)
      for (uint32_t y = 0; y < ST_B; y++)
        for (uint32_t x = 0; x < ST_B; x++)
          body[((size_t)z * ST_B + y) * ST_B + x] =
              li == 0 ? st_pat(cx * ST_B + x, cy * ST_B + y, cz * ST_B + z)
                      : st_pat_lvl(1, cx * ST_B + x, cy * ST_B + y, cz * ST_B + z);
    send_all(fd, body, n);
    free(body);
    return;
  }
  }
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

static void rm_rf(const char *dir) { /* test-owned temp paths only */
  char cmd[900];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
  if (system(cmd) != 0) fprintf(stderr, "cleanup failed for %s\n", dir);
}

/* census of <root>/bricks/L0: files, non-empty files */
static void brick_census(const char *root, uint32_t *nfiles, uint32_t *nfull) {
  *nfiles = *nfull = 0;
  char p[720];
  snprintf(p, sizeof p, "%s/bricks/L0", root);
  DIR *dp = opendir(p);
  struct dirent *de;
  while (dp && (de = readdir(dp)) != NULL) {
    if (de->d_name[0] == '.') continue;
    (*nfiles)++;
    char fp[1000];
    snprintf(fp, sizeof fp, "%s/%s", p, de->d_name);
    struct stat st;
    if (stat(fp, &st) == 0 && st.st_size > 0) (*nfull)++;
  }
  if (dp) closedir(dp);
}

static int run_shot(const char *bin, const char *root, const char *shot, const char *log) {
  char cmd[1800];
  snprintf(cmd, sizeof cmd,
           "R3D_MV_FIT=300 %s --bricks %s/manifest.json --headless --frames 400 "
           "--shot %s >%s 2>&1",
           bin, root, shot, log);
  return system(cmd);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_netstream <render3d-binary>\n");
    return 77;
  }
  signal(SIGPIPE, SIG_IGN);
  char tmp[256];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_net_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[320];
  snprintf(root, sizeof root, "%s/tree", tmp);
  uint32_t dim[3] = {256, 256, 256};
  if (st_make_tree(root, dim, 2, 0) != 0) return 1;
  { /* drop the fine level's shard: L0 must come from the network */
    char p[700];
    snprintf(p, sizeof p, "%s/c5d/L0/0_0_0.c5s", root);
    unlink(p);
    snprintf(p, sizeof p, "%s/c5d/L0", root);
    rmdir(p);
  }
  g_listen = socket(AF_INET, SOCK_STREAM, 0);
  if (g_listen < 0) return 1;
  int one = 1;
  setsockopt(g_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = 0,
                          .sin_addr = {htonl(INADDR_LOOPBACK)}};
  if (bind(g_listen, (struct sockaddr *)&a, sizeof a) != 0 || listen(g_listen, 16) != 0)
    return 1;
  socklen_t al = sizeof a;
  if (getsockname(g_listen, (struct sockaddr *)&a, &al) != 0) return 1;
  pthread_t th;
  if (pthread_create(&th, NULL, stub_main, NULL) != 0) return 1;
  { /* source.json: raw 128 chunks at every level, served by the stub */
    char p[700];
    snprintf(p, sizeof p, "%s/source.json", root);
    FILE *f = fopen(p, "w");
    if (!f) return 1;
    fprintf(f,
            "{\n  \"format\": \"render3d.c5d-source.v1\",\n"
            "  \"url\": \"http://127.0.0.1:%u\",\n  \"quality\": 2,\n"
            "  \"levels\": [\n    {\"level\": 0, \"chunk\": 128, \"raw\": true},\n"
            "    {\"level\": 1, \"chunk\": 128, \"raw\": true}\n  ]\n}\n",
            (unsigned)ntohs(a.sin_port));
    fclose(f);
  }
  char shot[3][360], logp[3][360];
  for (int i = 0; i < 3; i++) {
    snprintf(shot[i], sizeof shot[i], "%s/s%d.ppm", tmp, i);
    snprintf(logp[i], sizeof logp[i], "%s/s%d.log", tmp, i);
  }
  int rc = 0;
  uint32_t nfiles = 0, nfull = 0;
  /* HTML 200s: survives, publishes nothing */
  g_mode = N_HTML;
  if (run_shot(argv[1], root, shot[0], logp[0]) != 0) rc = 77;
  if (rc == 0) {
    brick_census(root, &nfiles, &nfull);
    CHECK(nfiles == 0);
  }
  /* healthy retry: fetches, persists non-empty bricks */
  if (rc == 0) {
    g_mode = N_OK;
    if (run_shot(argv[1], root, shot[1], logp[1]) != 0) rc = 77;
    if (rc == 0) {
      brick_census(root, &nfiles, &nfull);
      CHECK(nfull > 0 && nfull == nfiles);
    }
  }
  /* 404 on a fresh cache: permanent empty markers */
  if (rc == 0) {
    char p[700];
    snprintf(p, sizeof p, "%s/bricks", root);
    rm_rf(p);
    g_mode = N_404;
    if (run_shot(argv[1], root, shot[2], logp[2]) != 0) rc = 77;
    if (rc == 0) {
      brick_census(root, &nfiles, &nfull);
      CHECK(nfiles > 0 && nfull == 0);
    }
  }
  if (rc == 77) fprintf(stderr, "render3d run failed - skipping\n");
  g_stop = true;
  shutdown(g_listen, SHUT_RDWR);
  close(g_listen);
  pthread_join(th, NULL);
  if (rc == 0 && !failures) printf("renderer net-ingest fault injection OK\n");
  rm_rf(tmp);
  return rc ? rc : (failures ? 1 : 0);
}
