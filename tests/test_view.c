/* Model-view packets: manifest + packet I/O, every kind's decoding, the slice
 * compositor (sdf band, prob threshold, class palette, vec triples, compare),
 * the hard-error cases the spec names, the --view-shot headless entry point
 * and the live 'TSV1'/'TSVR'/'TSVE' client against an in-test server. */
#include "core/view.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <arpa/inet.h>
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#define NZ 8u
#define NY 12u
#define NX 10u
#define NVOX ((size_t)NZ * NY * NX)

static void rmtree(const char *dir) {
  DIR *d = opendir(dir);
  if (!d) return;
  const struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
    char p[2048];
    snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
    rmtree(p);
    unlink(p);
  }
  closedir(d);
  rmdir(dir);
}

static int writef(const char *dir, const char *name, const void *p, size_t n) {
  char path[2048];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  int ok = n == 0 || fwrite(p, 1, n, f) == n;
  return fclose(f) == 0 && ok ? 0 : -1;
}

/* sdf byte for a signed distance of `d` voxels at clip 20 (spec: the inverse
 * of (u-128)*clip/127); 0 is reserved for "no data". */
static uint8_t sdf_byte(double d, double clip) {
  long u = 128 + lround(d * 127.0 / clip);
  if (u < 1) u = 1;
  if (u > 255) u = 255;
  return (uint8_t)u;
}

/* ---- a synthetic packet with one layer of every kind -------------------- */

static const char k_meta_fmt[] =
    "{\n"
    "  \"format\": \"tsm.view.v1\",\n"
    "  \"wibble\": [1, 2, 3],\n" /* unknown key: ignored, never an error */
    "  \"dims_zyx\": [%u, %u, %u],\n"
    "  \"origin_zyx\": [1000, 2000, 3000],\n"
    "  \"voxel_um\": 2.4,\n"
    "  \"scroll\": \"PHercTest\",\n"
    "  \"layers\": [\n"
    "    {\"file\": \"ct.u8\", \"name\": \"ct\", \"group\": \"ct\", \"kind\": \"ct\"},\n"
    "    {\"file\": \"student.sdf_in.u8\", \"name\": \"sdf_in\", \"group\": \"student\","
    " \"kind\": \"sdf\", \"clip\": 20.0, \"notes\": \"unknown keys are ignored\"},\n"
    "    {\"file\": \"student.ink.u8\", \"name\": \"ink\", \"group\": \"student\","
    " \"kind\": \"prob\"},\n"
    "    {\"file\": \"student.sin.u8\", \"name\": \"sin\", \"group\": \"student\","
    " \"kind\": \"signed\"},\n"
    "    {\"file\": \"student.nx.u8\", \"name\": \"nx\", \"group\": \"student\","
    " \"kind\": \"signed\", \"vec\": \"normal\", \"axis\": 0},\n"
    "    {\"file\": \"student.ny.u8\", \"name\": \"ny\", \"group\": \"student\","
    " \"kind\": \"signed\", \"vec\": \"normal\", \"axis\": 1},\n"
    "    {\"file\": \"student.nz.u8\", \"name\": \"nz\", \"group\": \"student\","
    " \"kind\": \"signed\", \"vec\": \"normal\", \"axis\": 2},\n"
    "    {\"file\": \"student.density.u8\", \"name\": \"density\", \"group\": \"student\","
    " \"kind\": \"density\"},\n"
    "    {\"file\": \"student.thickness.u8\", \"name\": \"thickness\", \"group\": \"student\","
    " \"kind\": \"count\"},\n"
    "    {\"file\": \"label.sdf_in.u8\", \"name\": \"sdf_in\", \"group\": \"label\","
    " \"kind\": \"sdf\", \"clip\": 20.0},\n"
    "    {\"file\": \"human.rv_class.u8\", \"name\": \"rv_class\", \"group\": \"human\","
    " \"kind\": \"class\", \"palette\": [\"bg\", \"recto\", \"verso\", \"contact\"]}\n"
    "  ],\n"
    "  \"provenance\": {\"checkpoint\": \"/tmp/latest.pt\", \"step\": 42}\n"
    "}\n";

/* CT is a flat 100 so every composite assertion below has an exact base grey;
 * the sheet lives at x = centre and every other layer varies only in x. */
static int synth_packet(const char *dir, double centre) {
  if (mkdir(dir, 0777) != 0 && errno != EEXIST) return -1;
  uint8_t *ct = malloc(NVOX), *sdf = malloc(NVOX), *lsdf = malloc(NVOX), *ink = malloc(NVOX),
          *sn = malloc(NVOX), *nx = malloc(NVOX), *ny = malloc(NVOX), *nzc = malloc(NVOX),
          *den = malloc(NVOX), *thk = malloc(NVOX), *cls = malloc(NVOX);
  if (!ct || !sdf || !lsdf || !ink || !sn || !nx || !ny || !nzc || !den || !thk || !cls) return -1;
  for (uint32_t z = 0; z < NZ; z++)
    for (uint32_t y = 0; y < NY; y++)
      for (uint32_t x = 0; x < NX; x++) {
        size_t i = ((size_t)z * NY + y) * NX + x;
        ct[i] = 100u;
        /* z == 0 is a deliberate "no data" slab for both sdf layers */
        sdf[i] = z == 0 ? 0u : sdf_byte((double)x - centre, 20.0);
        lsdf[i] = z == 0 ? 0u : sdf_byte((double)x - (centre + 1.0), 20.0);
        ink[i] = (uint8_t)(x * 25u);
        sn[i] = (uint8_t)(x * 25u);
        nx[i] = 255u;  /* +1 -> R = 1.0  */
        ny[i] = 0u;    /* -1 -> G = 0.0  */
        nzc[i] = 128u; /*  ~0 -> B = 0.5 */
        den[i] = (uint8_t)(x * 10u);
        thk[i] = (uint8_t)x;
        cls[i] = (uint8_t)(x % 4u);
      }
  int rc = 0;
  rc |= writef(dir, "ct.u8", ct, NVOX);
  rc |= writef(dir, "student.sdf_in.u8", sdf, NVOX);
  rc |= writef(dir, "label.sdf_in.u8", lsdf, NVOX);
  rc |= writef(dir, "student.ink.u8", ink, NVOX);
  rc |= writef(dir, "student.sin.u8", sn, NVOX);
  rc |= writef(dir, "student.nx.u8", nx, NVOX);
  rc |= writef(dir, "student.ny.u8", ny, NVOX);
  rc |= writef(dir, "student.nz.u8", nzc, NVOX);
  rc |= writef(dir, "student.density.u8", den, NVOX);
  rc |= writef(dir, "student.thickness.u8", thk, NVOX);
  rc |= writef(dir, "human.rv_class.u8", cls, NVOX);
  char meta[8192];
  int mn = snprintf(meta, sizeof meta, k_meta_fmt, NZ, NY, NX);
  rc |= mn <= 0 || mn >= (int)sizeof meta;
  rc |= writef(dir, "meta.json", meta, (size_t)mn);
  free(ct); free(sdf); free(lsdf); free(ink); free(sn); free(nx); free(ny); free(nzc);
  free(den); free(thk); free(cls);
  return rc;
}

static int synth_corpus(const char *root) {
  if (mkdir(root, 0777) != 0 && errno != EEXIST) return -1;
  char d[1024];
  snprintf(d, sizeof d, "%s/v000", root);
  if (synth_packet(d, 5.0) != 0) return -1;
  snprintf(d, sizeof d, "%s/v001", root);
  if (synth_packet(d, 4.0) != 0) return -1;
  const char *vj = "{\n  \"packets\": [\n"
                   "    {\"path\": \"v000\", \"origin_zyx\": [1000, 2000, 3000]},\n"
                   "    {\"path\": \"v001\", \"origin_zyx\": [1000, 2000, 3000]}\n"
                   "  ]\n}\n";
  return writef(root, "view.json", vj, strlen(vj));
}

/* ---- helpers ------------------------------------------------------------ */

static const uint8_t *px_at(const uint8_t *rgba, uint32_t w, uint32_t u, uint32_t v) {
  return rgba + ((size_t)v * w + u) * 4u;
}
static bool grey(const uint8_t *p, uint8_t g) {
  return p[0] == g && p[1] == g && p[2] == g && p[3] == 255u;
}
static int find(const r3d_view_packet *p, const char *spec) {
  int at = r3d_view_find(p, spec);
  assert(at >= 0);
  return at;
}
static void only(r3d_view_packet *p, const char *spec) {
  for (uint32_t i = 0; i < p->count; i++) p->layer[i].show = false;
  p->layer[find(p, spec)].show = true;
}

/* Decode one of our own PNGs (8-bit RGBA, filter "none" on every row) back to
 * pixels, so the --view-shot assertions can look at what was actually written
 * rather than only at the header. */
static uint8_t *png_read(const char *path, uint32_t *w, uint32_t *h) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  assert(fseek(f, 0, SEEK_END) == 0);
  long fn = ftell(f);
  assert(fn > 8 && fseek(f, 0, SEEK_SET) == 0);
  uint8_t *file = malloc((size_t)fn);
  assert(file && fread(file, 1, (size_t)fn, f) == (size_t)fn);
  assert(fclose(f) == 0);

  uint8_t *idat = NULL;
  size_t idat_n = 0;
  *w = *h = 0;
  for (size_t at = 8; at + 12 <= (size_t)fn;) {
    uint32_t len = ((uint32_t)file[at] << 24) | ((uint32_t)file[at + 1] << 16) |
                   ((uint32_t)file[at + 2] << 8) | file[at + 3];
    const char *type = (const char *)file + at + 4;
    const uint8_t *data = file + at + 8;
    if (memcmp(type, "IHDR", 4) == 0) {
      *w = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) | ((uint32_t)data[2] << 8) |
           data[3];
      *h = ((uint32_t)data[4] << 24) | ((uint32_t)data[5] << 16) | ((uint32_t)data[6] << 8) |
           data[7];
      assert(data[8] == 8 && data[9] == 6); /* 8-bit RGBA */
    } else if (memcmp(type, "IDAT", 4) == 0) {
      idat = realloc(idat, idat_n + len);
      assert(idat);
      memcpy(idat + idat_n, data, len);
      idat_n += len;
    }
    at += (size_t)len + 12u;
  }
  assert(*w && *h && idat);
  size_t rown = (size_t)*w * 4u + 1u;
  uLongf raw_n = (uLongf)(rown * *h);
  uint8_t *raw = malloc(raw_n);
  assert(raw && uncompress(raw, &raw_n, idat, (uLong)idat_n) == Z_OK && raw_n == rown * *h);
  uint8_t *rgba = malloc((size_t)*w * *h * 4u);
  assert(rgba);
  for (uint32_t j = 0; j < *h; j++) {
    assert(raw[(size_t)j * rown] == 0); /* filter: none */
    memcpy(rgba + (size_t)j * *w * 4u, raw + (size_t)j * rown + 1u, (size_t)*w * 4u);
  }
  free(raw);
  free(idat);
  free(file);
  return rgba;
}

/* ---- an in-test tsm serve ------------------------------------------------ */

typedef struct fake_server {
  int fd;
  int port;
  pthread_t th;
  _Atomic bool stop;
} fake_server;

static bool srv_io(int fd, void *buf, size_t n, bool wr) {
  uint8_t *p = buf;
  while (n) {
    ssize_t k = wr ? send(fd, p, n, MSG_NOSIGNAL) : read(fd, p, n);
    if (k <= 0) return false;
    p += (size_t)k;
    n -= (size_t)k;
  }
  return true;
}
static void srv_put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
static uint32_t srv_get32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void srv_frame(int fd, const char *magic, const void *body, uint32_t n) {
  uint8_t hdr[8];
  memcpy(hdr, magic, 4);
  srv_put32(hdr + 4, n);
  (void)srv_io(fd, hdr, 8, true);
  if (n) (void)srv_io(fd, (void *)(uintptr_t)body, n, true);
}

static const char k_reply_meta[] =
    "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [%u, %u, %u],"
    " \"origin_zyx\": [1000, 2000, 3000], \"layers\": ["
    "{\"file\": \"student.conf.u8\", \"name\": \"conf\", \"group\": \"student\","
    " \"kind\": \"prob\"},"
    "{\"file\": \"student.ink.u8\", \"name\": \"ink\", \"group\": \"student\","
    " \"kind\": \"prob\"}]}";

static void *srv_main(void *arg) {
  fake_server *s = arg;
  while (!s->stop) {
    struct pollfd listener = {.fd = s->fd, .events = POLLIN};
    if (poll(&listener, 1, 100) <= 0) continue;
    if (s->stop) break;
    int c = accept(s->fd, NULL, NULL);
    if (c < 0) continue;
    uint8_t hdr[8];
    if (srv_io(c, hdr, 8, false) && memcmp(hdr, "TSV1", 4) == 0) {
      uint32_t n = srv_get32(hdr + 4);
      char *req = calloc((size_t)n + 1u, 1);
      if (req && srv_io(c, req, n, false)) {
        if (strstr(req, "\"hello\"")) {
          static const char hello[] =
              "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [1, 1, 1], \"layers\": [],"
              " \"groups\": [\"student\", \"ct\"], \"max_dims_zyx\": [512, 512, 512],"
              " \"scroll\": \"PHercTest\", \"checkpoint\": \"/tmp/latest.pt\"}";
          srv_frame(c, "TSVR", hello, (uint32_t)strlen(hello));
        } else if (strstr(req, "\"tta\": \"flip8_rot4\"")) {
          static const char err[] = "tta flip8_rot4 refused";
          srv_frame(c, "TSVE", err, (uint32_t)strlen(err));
        } else if (strstr(req, "\"origin_zyx\"")) {
          char meta[1024];
          int mn = snprintf(meta, sizeof meta, k_reply_meta, NZ, NY, NX);
          srv_frame(c, "TSVR", meta, (uint32_t)mn);
          uint8_t *buf = malloc(NVOX);
          if (buf) {
            memset(buf, 200, NVOX); /* student.conf */
            (void)srv_io(c, buf, NVOX, true);
            memset(buf, 30, NVOX); /* student.ink, replacing the packet's own */
            (void)srv_io(c, buf, NVOX, true);
            free(buf);
          }
        } else {
          static const char err[] = "unrecognised request";
          srv_frame(c, "TSVE", err, (uint32_t)strlen(err));
        }
      }
      free(req);
    }
    close(c);
  }
  return NULL;
}

static int srv_start(fake_server *s) {
  memset(s, 0, sizeof *s);
  s->fd = socket(AF_INET, SOCK_STREAM, 0);
  if (s->fd < 0) return -1;
  int one = 1;
  setsockopt(s->fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = 0};
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(s->fd, (struct sockaddr *)&a, sizeof a) != 0 || listen(s->fd, 8) != 0) return -1;
  socklen_t al = sizeof a;
  if (getsockname(s->fd, (struct sockaddr *)&a, &al) != 0) return -1;
  s->port = ntohs(a.sin_port);
  struct timeval to = {.tv_sec = 1};
  setsockopt(s->fd, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);
  return pthread_create(&s->th, NULL, srv_main, s);
}

static void srv_stop(fake_server *s) {
  s->stop = true;
  /* poll() bounds the join on macOS too; SO_RCVTIMEO does not bound accept. */
  pthread_join(s->th, NULL);
  close(s->fd);
}

/* ---- main ---------------------------------------------------------------- */

int main(int argc, char **argv) {
  char root[] = "/tmp/render3d-view-XXXXXX";
  assert(mkdtemp(root) != NULL);
  assert(synth_corpus(root) == 0);

  /* "gui" mode: drive the real --view window path headlessly. Every widget
   * call, the compositor and the texture registration still run, and ImGui's
   * assert-on-error is live, so a malformed call fails the test. */
  if (argc > 2 && strcmp(argv[2], "gui") == 0) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd,
             "R3D_VIEW_CYCLE=3 %s --view %s --headless --frames 14 --size 640 480 "
             "--shot %s/f.ppm",
             argv[1], root, root);
    int rc = system(cmd);
    char shot[1200];
    snprintf(shot, sizeof shot, "%s/f.ppm", root);
    FILE *sf = fopen(shot, "rb");
    assert(rc == 0 && sf != NULL);
    assert(fclose(sf) == 0);
    rmtree(root);
    printf("view gui: ok\n");
    return 0;
  }

  /* ---- manifest -------------------------------------------------------- */
  r3d_view_manifest m = {0};
  assert(r3d_view_manifest_load(&m, root) == 0);
  assert(m.count == 2);
  assert(strcmp(m.ent[0].name, "v000") == 0 && strcmp(m.ent[1].name, "v001") == 0);
  assert(m.ent[0].origin[0] == 1000 && m.ent[0].origin[2] == 3000);
  {
    char vj[1200];
    snprintf(vj, sizeof vj, "%s/view.json", root);
    r3d_view_manifest m2 = {0};
    assert(r3d_view_manifest_load(&m2, vj) == 0 && m2.count == 2);
    r3d_view_manifest_free(&m2);
    r3d_view_manifest m3 = {0}; /* a lone packet dir opens too */
    assert(r3d_view_manifest_load(&m3, m.ent[0].path) == 0 && m3.count == 1);
    r3d_view_manifest_free(&m3);
    r3d_view_manifest m4 = {0};
    assert(r3d_view_manifest_load(&m4, "/nonexistent/nope") != 0);
  }

  /* ---- packet load ------------------------------------------------------ */
  r3d_view_packet p = {0};
  assert(r3d_view_packet_load(&p, m.ent[0].path) == 0);
  assert(p.nz == NZ && p.ny == NY && p.nx == NX);
  assert(p.origin[0] == 1000 && p.origin[1] == 2000 && p.origin[2] == 3000);
  assert(p.voxel_um > 2.39 && p.voxel_um < 2.41);
  assert(strcmp(p.scroll, "PHercTest") == 0);
  assert(strstr(p.provenance, "latest.pt") != NULL && strstr(p.provenance, "42") != NULL);
  assert(p.count == 11);
  assert(p.ct >= 0 && p.layer[p.ct].kind == R3D_VIEW_CT);
  /* the layer list is in manifest order and both sdf_in layers coexist */
  assert(strcmp(p.layer[1].group, "student") == 0 && strcmp(p.layer[1].name, "sdf_in") == 0);
  assert(r3d_view_find(&p, "student.sdf_in") == 1);
  assert(r3d_view_find(&p, "label.sdf_in") != 1 && r3d_view_find(&p, "label.sdf_in") > 0);
  assert(r3d_view_find(&p, "sdf_in") == -1); /* ambiguous bare name */
  assert(r3d_view_find(&p, "rv_class") >= 0);
  assert(r3d_view_find(&p, "student.nope") == -1);

  const int L_SDF = find(&p, "student.sdf_in"), L_LSDF = find(&p, "label.sdf_in");
  const int L_INK = find(&p, "student.ink"), L_SIN = find(&p, "student.sin");
  const int L_NX = find(&p, "student.nx"), L_DEN = find(&p, "student.density");
  const int L_THK = find(&p, "student.thickness"), L_CLS = find(&p, "human.rv_class");
  assert(p.layer[L_SDF].clip > 19.9 && p.layer[L_SDF].clip < 20.1);
  assert(p.layer[L_CLS].npalette == 4 && strcmp(p.layer[L_CLS].palette[1], "recto") == 0);
  assert(strcmp(p.layer[L_NX].vec, "normal") == 0 && p.layer[L_NX].axis == 0);
  assert(p.layer[L_SIN].vec[0] == 0 && p.layer[L_SIN].axis == -1);

  /* ---- decodings against the spec's formulas ---------------------------- */
  {
    bool nd = false;
    /* ct: u/255 */
    assert(fabs(r3d_view_decode(&p.layer[p.ct], 100u, &nd) - 100.0 / 255.0) < 1e-9 && !nd);
    /* sdf: (u-128)*clip/127, and u == 0 is "no data" */
    assert(fabs(r3d_view_decode(&p.layer[L_SDF], 128u, &nd)) < 1e-9 && !nd);
    assert(fabs(r3d_view_decode(&p.layer[L_SDF], 255u, &nd) - 127.0 * 20.0 / 127.0) < 1e-9);
    assert(fabs(r3d_view_decode(&p.layer[L_SDF], 1u, &nd) + 127.0 * 20.0 / 127.0) < 1e-9);
    (void)r3d_view_decode(&p.layer[L_SDF], 0u, &nd);
    assert(nd);
    /* prob: u/255 */
    assert(fabs(r3d_view_decode(&p.layer[L_INK], 200u, &nd) - 200.0 / 255.0) < 1e-9);
    /* signed: (u-127.5)/127.5 */
    assert(fabs(r3d_view_decode(&p.layer[L_SIN], 255u, &nd) - 1.0) < 1e-9);
    assert(fabs(r3d_view_decode(&p.layer[L_SIN], 0u, &nd) + 1.0) < 1e-9);
    /* density: u/1000; count: u; class: u */
    assert(fabs(r3d_view_decode(&p.layer[L_DEN], 90u, &nd) - 0.090) < 1e-9);
    assert(fabs(r3d_view_decode(&p.layer[L_THK], 7u, &nd) - 7.0) < 1e-9);
    assert(fabs(r3d_view_decode(&p.layer[L_CLS], 2u, &nd) - 2.0) < 1e-9);
    char t[64];
    (void)r3d_view_value_text(&p.layer[L_CLS], 2u, t, sizeof t);
    assert(strcmp(t, "2 verso") == 0);
    (void)r3d_view_value_text(&p.layer[L_SDF], 0u, t, sizeof t);
    assert(strstr(t, "no data") != NULL);
    (void)r3d_view_value_text(&p.layer[L_THK], 5u, t, sizeof t);
    assert(strcmp(t, "5 vox") == 0);
  }

  /* ---- slice geometry and sampling on all three axes -------------------- */
  {
    uint32_t w, h, zyx[3];
    r3d_view_slice_dims(&p, R3D_VIEW_AXIS_Z, &w, &h);
    assert(w == NX && h == NY && r3d_view_slice_count(&p, R3D_VIEW_AXIS_Z) == NZ);
    r3d_view_slice_dims(&p, R3D_VIEW_AXIS_Y, &w, &h);
    assert(w == NX && h == NZ && r3d_view_slice_count(&p, R3D_VIEW_AXIS_Y) == NY);
    r3d_view_slice_dims(&p, R3D_VIEW_AXIS_X, &w, &h);
    assert(w == NY && h == NZ && r3d_view_slice_count(&p, R3D_VIEW_AXIS_X) == NX);
    r3d_view_voxel(&p, R3D_VIEW_AXIS_Y, 3, 7, 2, zyx);
    assert(zyx[0] == 2 && zyx[1] == 3 && zyx[2] == 7);
    r3d_view_voxel(&p, R3D_VIEW_AXIS_X, 3, 7, 2, zyx);
    assert(zyx[0] == 2 && zyx[1] == 7 && zyx[2] == 3);
    /* thickness is x, so it reads back as the x coordinate on every axis */
    assert(r3d_view_sample(&p, &p.layer[L_THK], R3D_VIEW_AXIS_Z, 4, 6, 2) == 6);
    assert(r3d_view_sample(&p, &p.layer[L_THK], R3D_VIEW_AXIS_Y, 4, 6, 2) == 6);
    assert(r3d_view_sample(&p, &p.layer[L_THK], R3D_VIEW_AXIS_X, 6, 4, 2) == 6);
  }

  /* ---- compositing ------------------------------------------------------ */
  uint8_t *rgba = malloc((size_t)NY * NX * 4u);
  assert(rgba);
  r3d_view_opts o;
  r3d_view_opts_default(&o);
  o.slice = 4;

  /* sdf: the zero-crossing band shows at |d| <= tol and nowhere else. The
   * sheet sits at x = 5, so d(x) = x - 5 voxels and tol 1.5 covers x 4..6. */
  only(&p, "student.sdf_in");
  p.layer[L_SDF].color[0] = 1.0f;
  p.layer[L_SDF].color[1] = p.layer[L_SDF].color[2] = 0.0f;
  p.layer[L_SDF].opacity = 1.0f;
  p.layer[L_SDF].tol = 1.5f;
  r3d_view_composite(&p, &o, rgba);
  for (uint32_t x = 4; x <= 6; x++) {
    const uint8_t *q = px_at(rgba, NX, x, 3);
    assert(q[0] == 255 && q[1] == 0 && q[2] == 0);
  }
  assert(grey(px_at(rgba, NX, 3, 3), 100u));
  assert(grey(px_at(rgba, NX, 7, 3), 100u));
  /* a tighter tol keeps only the exact zero crossing */
  p.layer[L_SDF].tol = 0.5f;
  r3d_view_composite(&p, &o, rgba);
  assert(px_at(rgba, NX, 5, 3)[0] == 255);
  assert(grey(px_at(rgba, NX, 4, 3), 100u));
  assert(grey(px_at(rgba, NX, 6, 3), 100u));
  /* u == 0 is no data: slice z = 0 draws nothing at all */
  p.layer[L_SDF].tol = 1.5f;
  o.slice = 0;
  r3d_view_composite(&p, &o, rgba);
  for (uint32_t x = 0; x < NX; x++) assert(grey(px_at(rgba, NX, x, 3), 100u));
  o.slice = 4;

  /* prob: the threshold hides low probabilities, alpha tracks p */
  only(&p, "student.ink");
  p.layer[L_INK].color[0] = 0.0f;
  p.layer[L_INK].color[1] = 1.0f;
  p.layer[L_INK].color[2] = 0.0f;
  p.layer[L_INK].opacity = 1.0f;
  p.layer[L_INK].threshold = 0.5f;
  r3d_view_composite(&p, &o, rgba);
  assert(grey(px_at(rgba, NX, 2, 3), 100u));  /* p = 50/255 = 0.196 */
  assert(px_at(rgba, NX, 8, 3)[1] > 200);     /* p = 200/255 = 0.784 */
  assert(px_at(rgba, NX, 8, 3)[0] < 40);
  p.layer[L_INK].threshold = 0.1f;
  r3d_view_composite(&p, &o, rgba);
  assert(px_at(rgba, NX, 2, 3)[1] > 120); /* now blended in */

  /* class: class 0 is transparent, non-zero takes a palette colour */
  only(&p, "human.rv_class");
  p.layer[L_CLS].opacity = 1.0f;
  r3d_view_composite(&p, &o, rgba);
  assert(grey(px_at(rgba, NX, 0, 3), 100u)); /* x % 4 == 0 */
  assert(grey(px_at(rgba, NX, 4, 3), 100u));
  assert(!grey(px_at(rgba, NX, 1, 3), 100u));
  assert(memcmp(px_at(rgba, NX, 1, 3), px_at(rgba, NX, 5, 3), 3) == 0); /* same class */
  assert(memcmp(px_at(rgba, NX, 1, 3), px_at(rgba, NX, 2, 3), 3) != 0); /* different class */

  /* signed vec triple: nx/ny/nz become one RGB normal image, 0.5 + 0.5*v */
  for (uint32_t i = 0; i < p.count; i++) p.layer[i].show = false;
  p.layer[L_NX].show = p.layer[L_NX + 1].show = p.layer[L_NX + 2].show = true;
  p.layer[L_NX].opacity = 1.0f;
  r3d_view_composite(&p, &o, rgba);
  {
    const uint8_t *q = px_at(rgba, NX, 3, 3);
    assert(q[0] == 255);        /* nx = +1  -> 1.0 */
    assert(q[1] == 0);          /* ny = -1  -> 0.0 */
    assert(q[2] > 120 && q[2] < 135); /* nz ~ 0 -> 0.5 */
  }

  /* count: 0 is transparent, the ramp is auto-scaled to the slice */
  only(&p, "student.thickness");
  p.layer[L_THK].opacity = 1.0f;
  r3d_view_composite(&p, &o, rgba);
  assert(grey(px_at(rgba, NX, 0, 3), 100u)); /* thickness 0 */
  assert(!grey(px_at(rgba, NX, 9, 3), 100u));

  /* solo overrides show */
  only(&p, "student.thickness");
  r3d_view_composite(&p, &o, rgba);
  assert(!grey(px_at(rgba, NX, 8, 3), 100u)); /* thickness 8 paints... */
  p.layer[L_CLS].solo = true;
  r3d_view_composite(&p, &o, rgba);
  assert(grey(px_at(rgba, NX, 8, 3), 100u)); /* ...until another layer is soloed */
  assert(!grey(px_at(rgba, NX, 1, 3), 100u)); /* the soloed class layer draws */
  p.layer[L_CLS].solo = false;

  /* ---- compare mode ----------------------------------------------------- */
  for (uint32_t i = 0; i < p.count; i++) p.layer[i].show = false;
  o.cmp_a = L_SDF;
  o.cmp_b = L_LSDF;
  p.layer[L_SDF].tol = p.layer[L_LSDF].tol = 1.5f;
  p.layer[L_SDF].opacity = 1.0f;
  o.cmp_col[0][0] = 0.0f; o.cmp_col[0][1] = 1.0f; o.cmp_col[0][2] = 0.0f; /* agree  */
  o.cmp_col[1][0] = 1.0f; o.cmp_col[1][1] = 0.0f; o.cmp_col[1][2] = 0.0f; /* A only */
  o.cmp_col[2][0] = 0.0f; o.cmp_col[2][1] = 0.0f; o.cmp_col[2][2] = 1.0f; /* B only */
  r3d_view_composite(&p, &o, rgba);
  {
    /* A's band is x 4..6, B's (shifted one voxel) is x 5..7 */
    const uint8_t *ag = px_at(rgba, NX, 5, 3), *ao = px_at(rgba, NX, 4, 3),
                  *bo = px_at(rgba, NX, 7, 3);
    assert(ag[0] == 0 && ag[1] == 255 && ag[2] == 0);
    assert(ao[0] == 255 && ao[1] == 0 && ao[2] == 0);
    assert(bo[0] == 0 && bo[1] == 0 && bo[2] == 255);
    assert(grey(px_at(rgba, NX, 2, 3), 100u));
  }
  r3d_view_cmp cm;
  assert(r3d_view_compare(&p, &o, &cm) == 0);
  assert(cm.agree == 2u * NY && cm.a_only == NY && cm.b_only == NY);
  assert(fabs(cm.dice - 2.0 / 3.0) < 1e-9);
  { /* the no-data slab agrees on nothing */
    r3d_view_opts o0 = o;
    o0.slice = 0;
    r3d_view_cmp c0;
    assert(r3d_view_compare(&p, &o0, &c0) == 0);
    assert(c0.agree == 0 && c0.a_only == 0 && c0.b_only == 0 && c0.dice == 0.0);
  }
  { /* different kinds never compare */
    r3d_view_opts ok2 = o;
    ok2.cmp_b = L_INK;
    r3d_view_cmp c2;
    assert(r3d_view_compare(&p, &ok2, &c2) == -1);
  }
  o.cmp_a = o.cmp_b = -1;

  /* r3d_view_select: a --compare with no --show is a mask over the CT, not a
   * mask over every other layer's own rendering. Start from "everything on",
   * which is what a freshly loaded packet looks like. */
  {
    for (uint32_t i = 0; i < p.count; i++) p.layer[i].show = p.layer[i].kind != R3D_VIEW_CT;
    r3d_view_opts so;
    r3d_view_opts_default(&so);
    so.slice = 4;
    r3d_view_composite(&p, &so, rgba);
    assert(!grey(px_at(rgba, NX, 2, 3), 100u)); /* 10 layers cover the CT */
    assert(r3d_view_select(&p, &so, NULL, "student.sdf_in,label.sdf_in") == 0);
    assert(so.cmp_a == L_SDF && so.cmp_b == L_LSDF);
    for (uint32_t i = 0; i < p.count; i++) assert(!p.layer[i].show);
    r3d_view_composite(&p, &so, rgba);
    assert(grey(px_at(rgba, NX, 2, 3), 100u)); /* outside both bands: bare CT */
    assert(!grey(px_at(rgba, NX, 5, 3), 100u)); /* inside: the agree colour */
    /* an explicit --show still wins, alongside the compare pair */
    for (uint32_t i = 0; i < p.count; i++) p.layer[i].show = false;
    assert(r3d_view_select(&p, &so, "student.thickness",
                           "student.sdf_in,label.sdf_in") == 0);
    assert(p.layer[L_THK].show && !p.layer[L_CLS].show);
    /* and the pair itself is still refused when the kinds differ */
    assert(r3d_view_select(&p, &so, NULL, "student.sdf_in,student.ink") != 0);
    assert(r3d_view_select(&p, &so, NULL, "student.sdf_in") != 0);
  }

  /* ---- hover readout ----------------------------------------------------- */
  {
    only(&p, "human.rv_class");
    char text[2048];
    uint32_t lines = r3d_view_hover(&p, &o, 6, 3, text, sizeof text);
    assert(lines >= 2);
    assert(strstr(text, "scroll z1004 y2003 x3006") != NULL); /* origin + voxel */
    assert(strstr(text, "human.rv_class  2 verso") != NULL);  /* 6 % 4 == 2 */
    assert(strstr(text, "ct.ct") != NULL);                    /* the base image too */
    assert(strstr(text, "student.thickness") == NULL);        /* hidden layers stay out */
    assert(r3d_view_hover(&p, &o, NX, 3, text, sizeof text) == 0); /* out of range */
  }

  free(rgba);
  r3d_view_packet_free(&p);

  /* ---- hard errors the spec names ---------------------------------------- */
  {
    char bad[1200];
    r3d_view_packet q = {0};

    snprintf(bad, sizeof bad, "%s/bad_missing", root);
    assert(synth_packet(bad, 5.0) == 0);
    char rm[1400];
    snprintf(rm, sizeof rm, "%s/student.ink.u8", bad);
    assert(unlink(rm) == 0);
    assert(r3d_view_packet_load(&q, bad) != 0); /* a listed layer file is missing */

    snprintf(bad, sizeof bad, "%s/bad_size", root);
    assert(synth_packet(bad, 5.0) == 0);
    uint8_t *shortbuf = calloc(NVOX - 1u, 1);
    assert(shortbuf && writef(bad, "student.ink.u8", shortbuf, NVOX - 1u) == 0);
    free(shortbuf);
    assert(r3d_view_packet_load(&q, bad) != 0); /* size != Z*Y*X */

    snprintf(bad, sizeof bad, "%s/bad_meta", root);
    assert(synth_packet(bad, 5.0) == 0);
    static const char no_clip[] =
        "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [8, 12, 10], \"layers\": ["
        "{\"file\": \"ct.u8\", \"name\": \"ct\", \"group\": \"ct\", \"kind\": \"ct\"},"
        "{\"file\": \"student.sdf_in.u8\", \"name\": \"sdf_in\", \"group\": \"student\","
        " \"kind\": \"sdf\"}]}";
    assert(writef(bad, "meta.json", no_clip, strlen(no_clip)) == 0);
    assert(r3d_view_packet_load(&q, bad) != 0); /* sdf without clip */

    static const char dup[] =
        "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [8, 12, 10], \"layers\": ["
        "{\"file\": \"ct.u8\", \"name\": \"ct\", \"group\": \"ct\", \"kind\": \"ct\"},"
        "{\"file\": \"student.ink.u8\", \"name\": \"ink\", \"group\": \"student\","
        " \"kind\": \"prob\"},"
        "{\"file\": \"student.sin.u8\", \"name\": \"ink\", \"group\": \"student\","
        " \"kind\": \"prob\"}]}";
    assert(writef(bad, "meta.json", dup, strlen(dup)) == 0);
    assert(r3d_view_packet_load(&q, bad) != 0); /* duplicate group.name */

    static const char noct[] =
        "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [8, 12, 10], \"layers\": ["
        "{\"file\": \"student.ink.u8\", \"name\": \"ink\", \"group\": \"student\","
        " \"kind\": \"prob\"}]}";
    assert(writef(bad, "meta.json", noct, strlen(noct)) == 0);
    assert(r3d_view_packet_load(&q, bad) != 0); /* no kind-ct layer */

    static const char badfmt[] =
        "{\"format\": \"tsm.view.v2\", \"dims_zyx\": [8, 12, 10], \"layers\": ["
        "{\"file\": \"ct.u8\", \"name\": \"ct\", \"group\": \"ct\", \"kind\": \"ct\"}]}";
    assert(writef(bad, "meta.json", badfmt, strlen(badfmt)) == 0);
    assert(r3d_view_packet_load(&q, bad) != 0);

    static const char badkind[] =
        "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [8, 12, 10], \"layers\": ["
        "{\"file\": \"ct.u8\", \"name\": \"ct\", \"group\": \"ct\", \"kind\": \"ct\"},"
        "{\"file\": \"student.ink.u8\", \"name\": \"ink\", \"group\": \"student\","
        " \"kind\": \"wibble\"}]}";
    assert(writef(bad, "meta.json", badkind, strlen(badkind)) == 0);
    assert(r3d_view_packet_load(&q, bad) != 0);

    /* an unknown key beside every known one is fine, and only ct is required */
    static const char loose[] =
        "{\"format\": \"tsm.view.v1\", \"dims_zyx\": [8, 12, 10], \"nonsense\": {\"a\": 1},"
        " \"layers\": [{\"file\": \"ct.u8\", \"name\": \"ct\", \"group\": \"ct\","
        " \"kind\": \"ct\", \"who\": \"knows\"}]}";
    assert(writef(bad, "meta.json", loose, strlen(loose)) == 0);
    assert(r3d_view_packet_load(&q, bad) == 0 && q.count == 1);
    r3d_view_packet_free(&q);
  }

  /* ---- --view-shot end to end -------------------------------------------- */
  if (argc > 1) {
    char cmd[4096], png[1200];
    snprintf(png, sizeof png, "%s/shot.png", root);
    snprintf(cmd, sizeof cmd,
             "%s --view-shot %s %s --axis z --slice 4 "
             "--show student.sdf_in,human.rv_class",
             argv[1], m.ent[0].path, png);
    assert(system(cmd) == 0);
    FILE *f = fopen(png, "rb");
    assert(f);
    uint8_t hdr[33];
    assert(fread(hdr, 1, sizeof hdr, f) == sizeof hdr);
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    assert(memcmp(hdr, sig, 8) == 0);
    assert(memcmp(hdr + 12, "IHDR", 4) == 0);
    uint32_t pw = ((uint32_t)hdr[16] << 24) | ((uint32_t)hdr[17] << 16) |
                  ((uint32_t)hdr[18] << 8) | hdr[19];
    uint32_t ph = ((uint32_t)hdr[20] << 24) | ((uint32_t)hdr[21] << 16) |
                  ((uint32_t)hdr[22] << 8) | hdr[23];
    assert(pw == NX && ph == NY);
    assert(hdr[24] == 8 && hdr[25] == 6); /* 8-bit RGBA */
    assert(fseek(f, 0, SEEK_END) == 0 && ftell(f) > 100);
    assert(fclose(f) == 0);

    /* the y axis gives a differently shaped image */
    snprintf(cmd, sizeof cmd, "%s --view-shot %s %s --axis y --slice 5", argv[1],
             m.ent[0].path, png);
    assert(system(cmd) == 0);
    f = fopen(png, "rb");
    assert(f && fread(hdr, 1, sizeof hdr, f) == sizeof hdr);
    ph = ((uint32_t)hdr[20] << 24) | ((uint32_t)hdr[21] << 16) | ((uint32_t)hdr[22] << 8) |
         hdr[23];
    assert(ph == NZ);
    assert(fclose(f) == 0);

    /* with no --show at all every layer draws, so the CT is covered */
    snprintf(cmd, sizeof cmd, "%s --view-shot %s %s --axis z --slice 4", argv[1],
             m.ent[0].path, png);
    assert(system(cmd) == 0);
    {
      uint32_t pxw = 0, pxh = 0;
      uint8_t *img = png_read(png, &pxw, &pxh);
      assert(img && pxw == NX && pxh == NY);
      assert(!grey(px_at(img, NX, 2, 3), 100u));
      free(img);
    }

    /* compare mode: only the three-class mask, the CT visible everywhere else */
    snprintf(cmd, sizeof cmd,
             "%s --view-shot %s %s --slice 4 --compare student.sdf_in,label.sdf_in", argv[1],
             m.ent[0].path, png);
    assert(system(cmd) == 0);
    {
      uint32_t pxw = 0, pxh = 0;
      uint8_t *img = png_read(png, &pxw, &pxh);
      assert(img && pxw == NX && pxh == NY);
      /* outside both zero-crossing bands the composite is the bare CT grey */
      assert(grey(px_at(img, NX, 0, 3), 100u));
      assert(grey(px_at(img, NX, 2, 3), 100u));
      assert(grey(px_at(img, NX, 9, 3), 100u));
      /* A's band is x 4..6, B's x 5..7: agree green, A-only red, B-only blue */
      const uint8_t *ag = px_at(img, NX, 5, 3), *ao = px_at(img, NX, 4, 3),
                    *bo = px_at(img, NX, 7, 3);
      assert(ag[1] > ag[0] && ag[1] > ag[2]);
      assert(ao[0] > ao[1] && ao[0] > ao[2]);
      assert(bo[2] > bo[0] && bo[2] > bo[1]);
      free(img);
    }

    /* an explicit --show alongside --compare keeps that layer drawing */
    snprintf(cmd, sizeof cmd,
             "%s --view-shot %s %s --slice 4 --compare student.sdf_in,label.sdf_in "
             "--show human.rv_class",
             argv[1], m.ent[0].path, png);
    assert(system(cmd) == 0);
    {
      uint32_t pxw = 0, pxh = 0;
      uint8_t *img = png_read(png, &pxw, &pxh);
      assert(img && !grey(px_at(img, NX, 2, 3), 100u)); /* class 2 at x = 2 */
      assert(grey(px_at(img, NX, 0, 3), 100u));         /* class 0 stays clear */
      free(img);
    }
    snprintf(cmd, sizeof cmd, "%s --view-shot %s %s --show student.nope 2>/dev/null", argv[1],
             m.ent[0].path, png);
    assert(system(cmd) != 0);
    snprintf(cmd, sizeof cmd,
             "%s --view-shot %s %s --compare student.sdf_in,student.ink 2>/dev/null", argv[1],
             m.ent[0].path, png);
    assert(system(cmd) != 0); /* different kinds */
  }

  /* ---- live client -------------------------------------------------------- */
  {
    fake_server srv;
    assert(srv_start(&srv) == 0);
    char err[256];

    /* hello handshake, blocking */
    r3d_view_hello hello = {0};
    assert(r3d_view_fetch("127.0.0.1", srv.port, NULL, NULL, NULL, NULL, NULL, &hello, err,
                          sizeof err) == 0);
    assert(hello.valid && strstr(hello.groups, "student") != NULL);
    assert(hello.max_dims[0] == 512 && strcmp(hello.checkpoint, "/tmp/latest.pt") == 0);

    /* a box request, blocking: the reply is a packet like any on disk */
    const int64_t origin[3] = {1000, 2000, 3000};
    const int64_t dims[3] = {NZ, NY, NX};
    r3d_view_packet got = {0};
    assert(r3d_view_fetch("127.0.0.1", srv.port, origin, dims, "student,ct", "none", &got, NULL,
                          err, sizeof err) == 0);
    assert(got.count == 2 && got.nz == NZ && got.ny == NY && got.nx == NX);
    assert(got.layer[0].kind == R3D_VIEW_PROB && strcmp(got.layer[0].name, "conf") == 0);
    assert(got.layer[0].data[0] == 200 && got.layer[1].data[0] == 30);

    /* merging: "conf" is new, "ink" replaces the packet's own bytes */
    r3d_view_packet base = {0};
    assert(r3d_view_packet_load(&base, m.ent[0].path) == 0);
    uint32_t before = base.count;
    int ink_at = find(&base, "student.ink");
    base.layer[ink_at].threshold = 0.33f; /* display state must survive */
    assert(r3d_view_merge(&base, &got) == 0);
    assert(base.count == before + 1u);
    assert(base.layer[ink_at].data[0] == 30);
    assert(base.layer[ink_at].threshold > 0.32f && base.layer[ink_at].threshold < 0.34f);
    assert(find(&base, "student.conf") >= 0);
    assert(base.layer[find(&base, "student.conf")].data[0] == 200);
    r3d_view_packet_free(&base);

    /* the TSVE error path is reported, not swallowed */
    err[0] = 0;
    r3d_view_packet bad = {0};
    assert(r3d_view_fetch("127.0.0.1", srv.port, origin, dims, "student", "flip8_rot4", &bad,
                          NULL, err, sizeof err) != 0);
    assert(strstr(err, "flip8_rot4 refused") != NULL);

    /* a dead port fails cleanly rather than hanging */
    assert(r3d_view_fetch("127.0.0.1", 1, origin, dims, "student", "none", &bad, NULL, err,
                          sizeof err) != 0);

    /* the worker thread: start (queues hello), request, poll */
    char hostport[64];
    snprintf(hostport, sizeof hostport, "127.0.0.1:%d", srv.port);
    r3d_view_live live;
    assert(r3d_view_live_start(&live, hostport) == 0);
    bool saw_hello = false;
    for (int i = 0; i < 500 && !saw_hello; i++) {
      struct timespec ts = {.tv_nsec = 10 * 1000 * 1000};
      nanosleep(&ts, NULL);
      pthread_mutex_lock(&live.mu);
      saw_hello = live.hello.valid;
      pthread_mutex_unlock(&live.mu);
    }
    assert(saw_hello);
    r3d_view_live_request(&live, origin, dims, "student", "none");
    r3d_view_packet async = {0};
    bool got_async = false;
    for (int i = 0; i < 500 && !got_async; i++) {
      struct timespec ts = {.tv_nsec = 10 * 1000 * 1000};
      nanosleep(&ts, NULL);
      got_async = r3d_view_live_poll(&live, &async);
    }
    assert(got_async && async.count == 2);
    assert(async.layer[0].data[0] == 200);
    char st[256];
    bool busy = true;
    r3d_view_live_status(&live, st, sizeof st, &busy);
    assert(st[0] != 0);
    r3d_view_packet_free(&async);
    r3d_view_live_stop(&live);

    /* a malformed host:port is refused up front */
    r3d_view_live nope;
    assert(r3d_view_live_start(&nope, "nocolon") != 0);
    assert(r3d_view_live_start(&nope, "") != 0);
    srv_stop(&srv);
  }

  r3d_view_manifest_free(&m);
  rmtree(root);
  printf("view: ok\n");
  return 0;
}
