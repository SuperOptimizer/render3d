/* Seed-cache reload conformance (ctest label: gpu — needs the real GPU).
 * The pinned coarsest level is decoded once and cached in seed.raw; the
 * fast-load path re-uploads it in multi-brick staging batches. This renders
 * the same synthetic tree twice — first run decodes and writes seed.raw,
 * second run reloads it — and requires the two screenshots to be
 * byte-identical. Catches the staging-reuse race where reloaded batches
 * were overwritten mid-upload (block-scrambled coarse levels), and any
 * future divergence between the decode and reload paths. Odd atlas width
 * and a partial final batch exercise row repacking across batch boundaries.
 * Usage: test_seedreload <path-to-render3d> */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "synthtree.h"

static uint8_t *read_all(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long ln = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = ln > 0 ? malloc((size_t)ln) : NULL;
  size_t got = b ? fread(b, 1, (size_t)ln, f) : 0;
  fclose(f);
  if (b && got != (size_t)ln) {
    free(b);
    return NULL;
  }
  *n = got;
  return b;
}

static int grep_file(const char *path, const char *needle) {
  size_t n = 0;
  uint8_t *b = read_all(path, &n);
  if (!b) return 0;
  b[n ? n - 1 : 0] = 0;
  int hit = strstr((char *)b, needle) != NULL;
  free(b);
  return hit;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_seedreload <render3d-binary>\n");
    return 77; /* skip */
  }
  setenv("R3D_DEBLOCK","1",1);
  char tmp[256];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_seedrl_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[320];
  snprintf(root, sizeof root, "%s/tree", tmp);
  /* 31x32x25 coarsest blocks: neither atlas rows (31) nor the last batch
   * align with the 256-block staging capacity. Volume edges are partial. */
  uint32_t dim[3] = {962, 994, 770};
  if (st_make_tree(root, dim, 2, 1) != 0) {
    fprintf(stderr, "synthetic tree build failed\n");
    return 1;
  }
  char shot[3][360], logp[3][360];
  int rc = 0;
  for (int run = 0; run < 3 && rc == 0; run++) {
    snprintf(shot[run], sizeof shot[run], "%s/r%d.ppm", tmp, run);
    snprintf(logp[run], sizeof logp[run], "%s/r%d.log", tmp, run);
    if (run == 2) {
      char seed[400];struct stat st;
      snprintf(seed,sizeof seed,"%s/seed-deblock.raw",root);
      if (stat(seed,&st)!=0 || truncate(seed,st.st_size/2)!=0) return 1;
    }
    char cmd[1600];
    snprintf(cmd, sizeof cmd,
             "%s --bricks %s/manifest.json --headless --pool 31 --frames 300 --shot %s "
             ">%s 2>&1",
             argv[1], root, shot[run], logp[run]);
    if (system(cmd) != 0) {
      fprintf(stderr, "render3d run %d failed (see %s)\n", run, logp[run]);
      rc = 1;
    }
  }
  if (rc == 0 && !grep_file(logp[1], "fallback from seed")) {
    fprintf(stderr, "second run did not take the seed.raw reload path\n");
    rc = 1;
  }
  if (rc == 0) {
    size_t na = 0, nb = 0;
    uint8_t *a = read_all(shot[0], &na), *b = read_all(shot[1], &nb);
    if (!a || !b || na != nb || na == 0) {
      fprintf(stderr, "screenshot read failed (%zu vs %zu bytes)\n", na, nb);
      rc = 1;
    } else if (memcmp(a, b, na) != 0) {
      size_t diff = 0;
      for (size_t i = 0; i < na; i++) diff += a[i] != b[i];
      fprintf(stderr, "decode-seed and seed.raw-reload renders differ: %zu/%zu bytes\n",
              diff, na);
      rc = 1;
    } else
      printf("seed reload render identical (%zu bytes)\n", na);
    free(a);
    free(b);
  }
  if (rc == 0) {
    size_t na=0,nb=0;
    uint8_t *a=read_all(shot[0],&na),*b=read_all(shot[2],&nb);
    if (!a || !b || na!=nb || memcmp(a,b,na) ||
        !grep_file(logp[2],"seed.raw unusable, re-decoding")) rc=1;
    free(a);free(b);
  }
  for (int run = 0; run < 3; run++) {
    unlink(shot[run]);
    unlink(logp[run]);
  }
  st_rm_tree(root, 2);
  rmdir(tmp);
  return rc;
}
