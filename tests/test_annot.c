/* Faces-annotation core: packet/manifest I/O, brush + line geometry, depth
 * mode, undo, atomic persistence, and the --annot-apply CLI. */
#include "core/annot.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "annotsynth.h"

#define NZ 16u
#define NY 24u
#define NX 20u

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

static uint32_t count_files(const char *dir, const char *prefix) {
  DIR *d = opendir(dir);
  if (!d) return 0;
  uint32_t n = 0;
  const struct dirent *e;
  while ((e = readdir(d)) != NULL)
    if (strncmp(e->d_name, prefix, strlen(prefix)) == 0) n++;
  closedir(d);
  return n;
}

static uint8_t corr_at(const r3d_annot_packet *p, uint32_t z, uint32_t y, uint32_t x) {
  return p->correction[((size_t)z * p->ny + y) * p->nx + x];
}

static uint64_t painted(const r3d_annot_packet *p) {
  uint64_t n = 0;
  for (int c = 1; c < R3D_ANNOT_NCLASS; c++) n += p->counts[c];
  return n;
}

/* Disc area used by the brush: voxels with squared distance <= r^2. */
static uint64_t disc_area(int r) {
  uint64_t n = 0;
  for (int y = -r; y <= r; y++)
    for (int x = -r; x <= r; x++)
      if (x * x + y * y <= r * r) n++;
  return n;
}

int main(int argc, char **argv) {
  char root[] = "/tmp/render3d-annot-XXXXXX";
  assert(mkdtemp(root) != NULL);
  assert(annotsynth_corpus(root, 3, NZ, NY, NX) == 0);

  /* ---- manifest ---------------------------------------------------- */
  r3d_annot_manifest m = {0};
  assert(r3d_annot_manifest_load(&m, root) == 0);
  assert(m.count == 3);
  assert(strcmp(m.ent[0].name, "p000") == 0);
  assert(m.ent[1].origin[0] == 100 && m.ent[1].origin[1] == 200 && m.ent[1].origin[2] == 300);
  assert(m.ent[2].dims[0] == NZ && m.ent[2].dims[1] == NY && m.ent[2].dims[2] == NX);
  { /* the packet.json path itself, and a bare packet directory */
    char pj[1024];
    snprintf(pj, sizeof pj, "%s/packet.json", root);
    r3d_annot_manifest m2 = {0};
    assert(r3d_annot_manifest_load(&m2, pj) == 0 && m2.count == 3);
    r3d_annot_manifest_free(&m2);
    r3d_annot_manifest m3 = {0};
    assert(r3d_annot_manifest_load(&m3, m.ent[0].path) == 0 && m3.count == 1);
    assert(strcmp(m3.ent[0].path, m.ent[0].path) == 0);
    r3d_annot_manifest_free(&m3);
    r3d_annot_manifest m4 = {0};
    assert(r3d_annot_manifest_load(&m4, "/nonexistent/nope") != 0);
  }

  /* ---- packet load -------------------------------------------------- */
  r3d_annot_packet p = {0};
  assert(r3d_annot_packet_load(&p, m.ent[0].path) == 0);
  assert(p.nz == NZ && p.ny == NY && p.nx == NX);
  assert(p.origin[0] == 0 && p.origin[1] == 200 && p.origin[2] == 300);
  assert(p.voxel_um > 2.39 && p.voxel_um < 2.41);
  assert(p.layer[R3D_ANNOT_L_CT] && p.layer[R3D_ANNOT_L_FACES_IN] &&
         p.layer[R3D_ANNOT_L_FACES_OUT] && p.layer[R3D_ANNOT_L_IGNORE] &&
         p.layer[R3D_ANNOT_L_SOURCE] && p.layer[R3D_ANNOT_L_RV_CLASS]);
  assert(p.layer[R3D_ANNOT_L_PRED_IN] && p.layer[R3D_ANNOT_L_PRED_OUT]);
  assert(!p.done && !p.dirty);
  assert(p.counts[R3D_ANNOT_UNTOUCHED] == (uint64_t)NZ * NY * NX && painted(&p) == 0);
  { /* p001 was generated without predictions */
    r3d_annot_packet q = {0};
    assert(r3d_annot_packet_load(&q, m.ent[1].path) == 0);
    assert(q.layer[R3D_ANNOT_L_PRED_IN] == NULL && q.layer[R3D_ANNOT_L_PRED_OUT] == NULL);
    r3d_annot_packet_free(&q);
  }

  r3d_annot_undo *u = r3d_annot_undo_create(24);
  assert(u && r3d_annot_undo_depth(u) == 0);

  /* ---- brush geometry ------------------------------------------------ */
  const int32_t centre[2] = {10, 12};
  r3d_annot_stroke s = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_IN, .radius = 3,
                        .depth = 0, .z = 8, .pts = centre, .npts = 1};
  int64_t n = r3d_annot_apply(&p, &s, u);
  assert(n == (int64_t)disc_area(3));
  assert(p.counts[R3D_ANNOT_IN] == disc_area(3));
  assert(corr_at(&p, 8, 12, 10) == R3D_ANNOT_IN);
  assert(corr_at(&p, 8, 12, 13) == R3D_ANNOT_IN);   /* exactly r away */
  assert(corr_at(&p, 8, 12, 14) == R3D_ANNOT_UNTOUCHED);
  assert(corr_at(&p, 8, 9, 10) == R3D_ANNOT_IN);
  assert(corr_at(&p, 7, 12, 10) == R3D_ANNOT_UNTOUCHED); /* neighbouring slice */
  assert(p.dirty);
  /* repainting the same class changes nothing and consumes no undo level */
  uint32_t depth_before = r3d_annot_undo_depth(u);
  assert(r3d_annot_apply(&p, &s, u) == 0);
  assert(r3d_annot_undo_depth(u) == depth_before);
  /* radius 0 is a single voxel */
  const int32_t one[2] = {2, 2};
  r3d_annot_stroke dot = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_OUT, .radius = 0,
                          .z = 8, .pts = one, .npts = 1};
  assert(r3d_annot_apply(&p, &dot, u) == 1);
  assert(corr_at(&p, 8, 2, 2) == R3D_ANNOT_OUT);

  /* capsule sweep: a fast drag leaves no gap between samples */
  const int32_t drag[4] = {2, 20, 17, 20};
  r3d_annot_stroke sweep = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_IGNORE, .radius = 1,
                            .z = 3, .pts = drag, .npts = 2};
  assert(r3d_annot_apply(&p, &sweep, u) > 0);
  for (int32_t x = 1; x <= 18; x++) assert(corr_at(&p, 3, 20, (uint32_t)x) == R3D_ANNOT_IGNORE);
  assert(corr_at(&p, 3, 20, 0) == R3D_ANNOT_UNTOUCHED);   /* radius-1 end cap */
  assert(corr_at(&p, 3, 19, 10) == R3D_ANNOT_IGNORE);     /* the capsule is 3 wide */
  assert(corr_at(&p, 3, 18, 10) == R3D_ANNOT_UNTOUCHED);

  /* clipping: a brush centred outside the volume paints only what is inside */
  const int32_t edge[2] = {-2, -2};
  r3d_annot_stroke off = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_ERASE, .radius = 3,
                          .z = 0, .pts = edge, .npts = 1};
  int64_t clipped = r3d_annot_apply(&p, &off, u);
  assert(clipped > 0 && clipped < (int64_t)disc_area(3));
  assert(corr_at(&p, 0, 0, 0) == R3D_ANNOT_ERASE);

  /* ---- depth mode ----------------------------------------------------- */
  const int32_t deep[2] = {6, 6};
  r3d_annot_stroke cyl = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_OUT, .radius = 2,
                          .depth = 2, .z = 8, .pts = deep, .npts = 1};
  assert(r3d_annot_apply(&p, &cyl, u) == (int64_t)disc_area(2) * 5);
  for (uint32_t z = 6; z <= 10; z++) assert(corr_at(&p, z, 6, 6) == R3D_ANNOT_OUT);
  assert(corr_at(&p, 5, 6, 6) == R3D_ANNOT_UNTOUCHED);
  assert(corr_at(&p, 11, 6, 6) == R3D_ANNOT_UNTOUCHED);
  { /* the depth window is clamped to the volume, not wrapped */
    const int32_t low[2] = {17, 2};
    r3d_annot_stroke c2 = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_IN, .radius = 0,
                           .depth = 4, .z = 1, .pts = low, .npts = 1};
    assert(r3d_annot_apply(&p, &c2, u) == 6); /* z 0..5 */
    assert(corr_at(&p, NZ - 1, 2, 17) == R3D_ANNOT_UNTOUCHED);
  }

  /* ---- fill line ------------------------------------------------------ */
  const int32_t diag[4] = {0, 0, 9, 9};
  r3d_annot_stroke line = {.op = R3D_ANNOT_OP_LINE, .cls = R3D_ANNOT_IN, .radius = 5,
                           .z = 12, .pts = diag, .npts = 2};
  assert(r3d_annot_apply(&p, &line, u) == 10); /* 1 voxel wide, radius ignored */
  for (uint32_t k = 0; k < 10; k++) assert(corr_at(&p, 12, k, k) == R3D_ANNOT_IN);
  assert(corr_at(&p, 12, 1, 0) == R3D_ANNOT_UNTOUCHED);

  /* ---- erase back to untouched ---------------------------------------- */
  uint64_t before = p.counts[R3D_ANNOT_IN];
  r3d_annot_stroke wipe = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_UNTOUCHED, .radius = 3,
                           .z = 8, .pts = centre, .npts = 1};
  assert(r3d_annot_apply(&p, &wipe, u) == (int64_t)disc_area(3));
  assert(p.counts[R3D_ANNOT_IN] == before - disc_area(3));
  assert(corr_at(&p, 8, 12, 10) == R3D_ANNOT_UNTOUCHED);

  /* ---- undo ------------------------------------------------------------ */
  assert(r3d_annot_undo_pop(&p, u) == (int64_t)disc_area(3));
  assert(corr_at(&p, 8, 12, 10) == R3D_ANNOT_IN);
  assert(p.counts[R3D_ANNOT_IN] == before);
  assert(r3d_annot_undo_pop(&p, u) == 10); /* the diagonal line */
  assert(corr_at(&p, 12, 5, 5) == R3D_ANNOT_UNTOUCHED);
  { /* >= 20 levels, popped in order, all the way back to empty */
    r3d_annot_packet q = {0};
    assert(r3d_annot_packet_load(&q, m.ent[2].path) == 0);
    r3d_annot_undo *uu = r3d_annot_undo_create(20);
    assert(uu);
    for (uint32_t k = 0; k < 20; k++) {
      int32_t pt[2] = {(int32_t)k, (int32_t)k};
      r3d_annot_stroke t = {.op = R3D_ANNOT_OP_BRUSH, .cls = R3D_ANNOT_IN, .radius = 0,
                            .z = 4, .pts = pt, .npts = 1};
      assert(r3d_annot_apply(&q, &t, uu) == 1);
    }
    assert(r3d_annot_undo_depth(uu) == 20);
    assert(painted(&q) == 20);
    for (uint32_t k = 0; k < 20; k++) assert(r3d_annot_undo_pop(&q, uu) == 1);
    assert(painted(&q) == 0);
    assert(r3d_annot_undo_pop(&q, uu) == 0); /* empty journal is a no-op */
    r3d_annot_undo_destroy(uu);
    r3d_annot_packet_free(&q);
  }

  /* ---- persistence ----------------------------------------------------- */
  assert(p.dirty);
  assert(r3d_annot_correction_save(&p) == 0);
  assert(!p.dirty);
  assert(count_files(p.dir, ".correction") == 0); /* no temp file left behind */
  uint64_t saved_counts[R3D_ANNOT_NCLASS];
  memcpy(saved_counts, p.counts, sizeof saved_counts);
  {
    r3d_annot_packet q = {0};
    assert(r3d_annot_packet_load(&q, p.dir) == 0);
    assert(memcmp(q.counts, saved_counts, sizeof saved_counts) == 0);
    assert(memcmp(q.correction, p.correction, (size_t)NZ * NY * NX) == 0);
    r3d_annot_packet_free(&q);
  }

  /* ---- done flag preserves the rest of meta.json ----------------------- */
  assert(r3d_annot_set_done(&p, true) == 0);
  {
    r3d_annot_packet q = {0};
    assert(r3d_annot_packet_load(&q, p.dir) == 0);
    assert(q.done);
    assert(q.nz == NZ && q.ny == NY && q.nx == NX); /* dims survived the rewrite */
    assert(q.origin[1] == 200 && q.voxel_um > 2.39);
    assert(strstr(q.meta_json, "\"layers\"") != NULL);
    r3d_annot_packet_free(&q);
  }
  assert(r3d_annot_set_done(&p, false) == 0); /* now the key already exists */
  {
    r3d_annot_packet q = {0};
    assert(r3d_annot_packet_load(&q, p.dir) == 0);
    assert(!q.done && q.origin[1] == 200);
    r3d_annot_packet_free(&q);
  }

  /* ---- composite -------------------------------------------------------- */
  {
    uint8_t *rgba = malloc((size_t)NY * NX * 4u);
    assert(rgba);
    r3d_annot_view v;
    r3d_annot_view_default(&v);
    memset(rgba, 7, (size_t)NY * NX * 4u);
    r3d_annot_composite(&p, NZ, &v, rgba); /* out of range: untouched */
    assert(rgba[0] == 7);
    r3d_annot_composite(&p, 8, &v, rgba);
    assert(rgba[3] == 255);
    const uint8_t *px = rgba + ((size_t)12 * NX + 10) * 4u; /* an in-face correction */
    assert(px[0] > px[2] && px[0] > 180);                   /* saturated red wins */
    v.show = 0;                                             /* CT only */
    r3d_annot_composite(&p, 8, &v, rgba);
    const uint8_t *g = rgba + ((size_t)12 * NX + 10) * 4u;
    assert(g[0] == g[1] && g[1] == g[2]);
    free(rgba);
  }

  r3d_annot_undo_destroy(u);
  r3d_annot_packet_free(&p);

  /* ---- --annot-apply CLI ------------------------------------------------ */
  if (argc > 1) {
    char spath[1024], cmd[4096];
    snprintf(spath, sizeof spath, "%s/strokes.json", root);
    FILE *f = fopen(spath, "wb");
    assert(f);
    fprintf(f,
            "{\"strokes\": [\n"
            "  {\"op\": \"brush\", \"class\": 1, \"z\": 5, \"radius\": 2, \"depth\": 1,"
            " \"points\": [[8, 8], [12, 8]]},\n"
            "  {\"op\": \"line\", \"class\": 2, \"z\": 5, \"points\": [[0, 15], [19, 15]]},\n"
            "  {\"op\": \"brush\", \"class\": 3, \"z\": 5, \"radius\": 4,"
            " \"points\": [[4, 4]]},\n"
            "  {\"op\": \"undo\"}\n]}\n");
    assert(fclose(f) == 0);
    snprintf(cmd, sizeof cmd, "%s --annot-apply %s %s", argv[1], m.ent[1].path, spath);
    assert(system(cmd) == 0);
    r3d_annot_packet q = {0};
    assert(r3d_annot_packet_load(&q, m.ent[1].path) == 0);
    assert(q.counts[R3D_ANNOT_IN] > 0);
    assert(q.counts[R3D_ANNOT_OUT] == 20); /* the full-width line */
    assert(q.counts[R3D_ANNOT_IGNORE] == 0); /* undone before the write */
    for (uint32_t z = 4; z <= 6; z++) assert(corr_at(&q, z, 8, 10) == R3D_ANNOT_IN);
    assert(corr_at(&q, 3, 8, 10) == R3D_ANNOT_UNTOUCHED);
    assert(corr_at(&q, 5, 15, 0) == R3D_ANNOT_OUT);
    r3d_annot_packet_free(&q);
    /* a manifest argument applies to its first packet */
    snprintf(cmd, sizeof cmd, "%s --annot-apply %s/nope %s", argv[1], root, spath);
    assert(system(cmd) != 0);
  }

  r3d_annot_manifest_free(&m);
  rmtree(root);
  printf("annot: ok\n");
  return 0;
}
