/* surfconv: convert between tifxyz directories and surface-compressor (.sfc)
 * surfaces, render3d's default surface type.
 *
 *   surfconv encode <tifxyz-dir> [out.sfc] [--error E]   tifxyz -> .sfc
 *   surfconv decode <in.sfc> <tifxyz-dir>                .sfc -> tifxyz
 *   surfconv resolve <surface> [--error E]               print the .sfc for
 *                                   a surface, converting a tifxyz dir when
 *                                   its .sfc is missing or stale
 *   surfconv info <surface>                              dimensions/scale
 *
 * Encoding streams TIFF bands, so memory stays bounded by the grid width;
 * the output is published atomically. Without out.sfc, encode writes
 * <dir>.sfc (a trailing ".tifxyz" is replaced). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "core/surfconv.h"

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int progress(void *ud, uint64_t done, uint64_t total) {
  int *last = ud;
  int pct = total ? (int)(done * 100 / total) : 100;
  if (pct / 10 != *last / 10) {
    fprintf(stderr, "surfconv: %d%%\n", pct);
    *last = pct;
  }
  return 0;
}

static int usage(void) {
  fprintf(stderr,
          "usage: surfconv encode <tifxyz-dir> [out.sfc] [--error E]\n"
          "       surfconv decode <in.sfc> <tifxyz-dir>\n"
          "       surfconv resolve <surface> [--error E]\n"
          "       surfconv info <surface>\n");
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  if (argc < 3) return usage();
  const char *cmd = argv[1], *pos[3] = {0};
  int npos = 0;
  double error = r3d_surf_default_error();
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--error") && i + 1 < argc) {
      error = strtod(argv[++i], NULL);
      if (!(error > 0) || !isfinite(error)) {
        fprintf(stderr, "surfconv: invalid --error\n");
        return EXIT_FAILURE;
      }
    } else if (argv[i][0] == '-' && argv[i][1]) {
      return usage();
    } else if (npos < 3) {
      pos[npos++] = argv[i];
    } else
      return usage();
  }
  int last = -1;
  double t0 = now_s();
  if (!strcmp(cmd, "encode")) {
    if (npos < 1) return usage();
    char out[2048];
    if (npos >= 2) snprintf(out, sizeof out, "%s", pos[1]);
    else if (r3d_surf_sibling(pos[0], out, sizeof out)) return EXIT_FAILURE;
    if (r3d_surf_kind_of(pos[0]) != R3D_SURF_TIFXYZ) {
      fprintf(stderr, "surfconv: %s is not a tifxyz directory\n", pos[0]);
      return EXIT_FAILURE;
    }
    if (r3d_surf_encode(pos[0], out, error, progress, &last)) {
      fprintf(stderr, "surfconv: encode failed\n");
      return EXIT_FAILURE;
    }
    struct stat st;
    printf("%s (%.1f MB, %.1f s)\n", out,
           stat(out, &st) == 0 ? (double)st.st_size / 1e6 : 0.0, now_s() - t0);
    return EXIT_SUCCESS;
  }
  if (!strcmp(cmd, "decode")) {
    if (npos != 2) return usage();
    if (r3d_surf_decode(pos[0], pos[1], progress, &last)) {
      fprintf(stderr, "surfconv: decode failed\n");
      return EXIT_FAILURE;
    }
    printf("%s (%.1f s)\n", pos[1], now_s() - t0);
    return EXIT_SUCCESS;
  }
  if (!strcmp(cmd, "resolve")) {
    if (npos != 1) return usage();
    char out[2048];
    if (r3d_surf_resolve(pos[0], error, progress, &last, out, sizeof out)) {
      fprintf(stderr, "surfconv: cannot resolve %s\n", pos[0]);
      return EXIT_FAILURE;
    }
    puts(out);
    return EXIT_SUCCESS;
  }
  if (!strcmp(cmd, "info")) {
    if (npos != 1) return usage();
    r3d_surf_kind k = r3d_surf_kind_of(pos[0]);
    r3d_tifxyz s;
    if (k == R3D_SURF_NONE || r3d_surf_load(pos[0], &s)) return EXIT_FAILURE;
    printf("%s: %s %ux%u scale %g %g  %llu valid points  bbox (%.1f %.1f %.1f)-(%.1f "
           "%.1f %.1f)\n",
           pos[0], k == R3D_SURF_SFC ? "sfc" : "tifxyz", s.w, s.h, (double)s.sx,
           (double)s.sy, (unsigned long long)s.nvalid, (double)s.bbox[0][0],
           (double)s.bbox[0][1], (double)s.bbox[0][2], (double)s.bbox[1][0],
           (double)s.bbox[1][1], (double)s.bbox[1][2]);
    r3d_tifxyz_free(&s);
    return EXIT_SUCCESS;
  }
  return usage();
}
