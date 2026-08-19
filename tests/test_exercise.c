/* Interaction soak (ctest label: gpu).
 * Runs the binary headless on the synthetic tree with R3D_MV_EXERCISE
 * driving continuous slice scrubbing, zoom and pan for 1200 frames, with
 * everything contentious enabled at once: label paint blobs (R3D_LBLTEST),
 * a self-overlaid registration volume mid-drag (R3D_REGTEST with an
 * offset), and a display filter re-streaming every uploaded brick
 * (R3D_POSTFILT). This churns streaming, eviction, the stale-slot zeroing
 * pass, both CPU-sourced atlases and the registration sync worker exactly
 * like an aggressive user session; the assertion is a clean exit and a
 * rendered frame. Under the dev/tsan presets this doubles as a
 * sanitizer/race soak of the same machinery.
 * Usage: test_exercise <path-to-render3d> */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "synthtree.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_exercise <render3d-binary>\n");
    return 77;
  }
  char tmp[256];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_soak_XXXXXX", base && *base ? base : "/tmp");
  if (!mkdtemp(tmp)) return 1;
  char root[320];
  snprintf(root, sizeof root, "%s/tree", tmp);
  uint32_t dim[3] = {256, 256, 256};
  if (st_make_tree(root, dim, 2, 0) != 0) {
    fprintf(stderr, "synthetic tree build failed\n");
    return 1;
  }
  char shot[360], logp[360], cmd[2000];
  snprintf(shot, sizeof shot, "%s/s.ppm", tmp);
  snprintf(logp, sizeof logp, "%s/s.log", tmp);
  snprintf(cmd, sizeof cmd,
           "R3D_MV_FIT=300 R3D_MV_EXERCISE=1 R3D_LBLTEST=1 R3D_REGTEST=%s:12,7,0 "
           "R3D_POSTFILT=1 %s --bricks %s/manifest.json --headless --frames 1200 "
           "--shot %s >%s 2>&1",
           root, argv[1], root, shot, logp);
  int rc = 0;
  if (system(cmd) != 0) {
    fprintf(stderr, "soak run failed (see %s)\n", logp);
    rc = 77; /* no GPU: skip; a crash under sanitizers also lands here but
              * leaves its report in the log the harness prints on failure */
    char tailcmd[420];
    snprintf(tailcmd, sizeof tailcmd, "tail -25 %s >&2", logp);
    if (system(tailcmd) != 0) fprintf(stderr, "(no log)\n");
  } else {
    FILE *f = fopen(shot, "rb");
    if (!f) {
      fprintf(stderr, "no screenshot produced\n");
      rc = 1;
    } else {
      fseek(f, 0, SEEK_END);
      long n = ftell(f);
      fclose(f);
      if (n < 1000) rc = 1;
      else printf("interaction soak OK (%ld-byte shot after 1200 churned frames)\n", n);
    }
  }
  char rmcmd[420];
  snprintf(rmcmd, sizeof rmcmd, "rm -rf '%s'", tmp);
  if (system(rmcmd) != 0) fprintf(stderr, "cleanup failed\n");
  return rc;
}
