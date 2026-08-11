/* segpack: ingest tifxyz segment dirs into a segment store (c5d-compressed
 * .tfx grids + a binary manifest with per-tile AABBs for spatial queries).
 *
 *   segpack <store-dir> [-q <log2q>] [--force] <tifxyz-dir>...
 *
 * log2q < 0 (default 2 = 1/4-voxel max error) selects lossless. Existing
 * .tfx files are reused unless --force; the manifest is always rewritten.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "core/segstore.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  const char *store = NULL;
  const char *dirs[4096];
  uint32_t ndirs = 0;
  int log2q = 2;
  bool force = false, bench = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) log2q = atoi(argv[++i]);
    else if (strcmp(argv[i], "--force") == 0) force = true;
    else if (strcmp(argv[i], "--bench") == 0) bench = true;
    else if (!store) store = argv[i];
    else if (ndirs < sizeof dirs / sizeof *dirs) dirs[ndirs++] = argv[i];
  }
  if (!store || ndirs == 0) {
    fprintf(stderr, "usage: segpack <store-dir> [-q <log2q>] [--force] <tifxyz-dir>...\n");
    return EXIT_FAILURE;
  }
  mkdir(store, 0755); /* fine if it exists */
  int n = r3d_segstore_build(store, dirs, ndirs, log2q, force);
  if (n < 0) {
    fprintf(stderr, "segpack: build failed\n");
    return EXIT_FAILURE;
  }
  r3d_segstore st;
  if (r3d_segstore_open(&st, store) != 0) {
    fprintf(stderr, "segpack: store verification failed\n");
    return EXIT_FAILURE;
  }
  printf("segpack: %u segment(s), %llu index tiles in %s\n", st.n,
         (unsigned long long)st.ntiles, store);
  for (uint32_t i = 0; i < st.n; i++) {
    const r3d_segmeta *m = &st.segs[i];
    printf("  %-40s %ux%u  bbox (%.0f %.0f %.0f)-(%.0f %.0f %.0f)\n", m->name, m->w, m->h,
           (double)m->bbox[0][0], (double)m->bbox[0][1], (double)m->bbox[0][2],
           (double)m->bbox[1][0], (double)m->bbox[1][1], (double)m->bbox[1][2]);
  }
  if (bench && st.n) {
    double t0 = now_ms();
    const int NQ = 1000;
    uint32_t acc = 0;
    double zmid = ((double)st.segs[0].bbox[0][2] + (double)st.segs[0].bbox[1][2]) * 0.5;
    double bn[3] = {0, 0, 1};
    for (int q = 0; q < NQ; q++)
      acc += r3d_segstore_plane_query(&st, bn, zmid + q, 0.0, NULL, NULL, NULL, 0);
    double t1 = now_ms();
    r3d_tifxyz s;
    int rc = r3d_segstore_load(&st, 0, 1, &s);
    double t2 = now_ms();
    printf("bench: %d plane queries %.2f ms (%.1f us each, %.1f hits avg); "
           "load[0] full %.1f ms%s\n",
           NQ, t1 - t0, (t1 - t0) * 1e3 / NQ, (double)acc / NQ, t2 - t1,
           rc == 0 ? "" : " FAILED");
    if (rc == 0) r3d_tifxyz_free(&s);
    r3d_tifxyz s4;
    double t3 = now_ms();
    if (r3d_segstore_load(&st, 0, 4, &s4) == 0) {
      printf("bench: load[0] stride-4 %.1f ms (%ux%u)\n", now_ms() - t3, s4.w, s4.h);
      r3d_tifxyz_free(&s4);
    }
  }
  r3d_segstore_close(&st);
  return EXIT_SUCCESS;
}
