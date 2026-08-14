/* Headless tracer: grow a segment from a seed over a prediction LOD tree
 * (with demand streaming + normal grids) and save it as tifxyz — no GUI,
 * no Vulkan; exits when done. For fast iteration on the growth code.
 *
 *   tracecli <pred-root> [--seed x y z] [--step S] [--gens N] [--level L]
 *            [--cutoff C] [--out DIR] [--umbilicus FILE]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/cpuvol.h"
#include "core/tracer.h"
#include "core/umbilicus.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: tracecli <pred-root> [--seed x y z] [--step S] [--gens N] "
            "[--level L] [--cutoff C] [--out DIR] [--umbilicus FILE] [--nofill] "
            "[--nospiral] [--spiral-weight W] [--fuse DIR ...] [--ribbon ROWS] "
            "[--zspan S] [--wraps N] [--ct CT-ROOT]\n");
    return 2;
  }
  const char *root = argv[1], *out = NULL, *umbp = NULL;
  const char *fuse[16];
  uint32_t nfuse = 0;
  double zspan = 0.0;
  bool fill = true, spiral = true;
  r3d_tracer_cfg cfg = {.step = 20.0, .thresh = 0.35f, .max_ring = 60, .level = 1};
  bool have_seed = false;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--seed") == 0 && i + 3 < argc) {
      for (int a = 0; a < 3; a++) cfg.seed[a] = strtod(argv[++i], NULL);
      have_seed = true;
    } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc)
      cfg.step = strtod(argv[++i], NULL);
    else if (strcmp(argv[i], "--gens") == 0 && i + 1 < argc)
      cfg.max_ring = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc)
      cfg.level = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--cutoff") == 0 && i + 1 < argc)
      cfg.thresh = strtof(argv[++i], NULL);
    else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
      out = argv[++i];
    else if (strcmp(argv[i], "--umbilicus") == 0 && i + 1 < argc)
      umbp = argv[++i];
    else if (strcmp(argv[i], "--nofill") == 0)
      fill = false;
    else if (strcmp(argv[i], "--nospiral") == 0)
      spiral = false;
    else if (strcmp(argv[i], "--spiral-weight") == 0 && i + 1 < argc)
      cfg.wind_weight = strtod(argv[++i], NULL);
    else if (strcmp(argv[i], "--fuse") == 0 && i + 1 < argc && nfuse < 16)
      fuse[nfuse++] = argv[++i];
    else if (strcmp(argv[i], "--ribbon") == 0 && i + 1 < argc)
      cfg.rib_rows = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--wraps") == 0 && i + 1 < argc)
      cfg.rib_wraps = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--ct") == 0 && i + 1 < argc)
      snprintf(cfg.ct_root, sizeof cfg.ct_root, "%s", argv[++i]);
    else if (strcmp(argv[i], "--zspan") == 0 && i + 1 < argc)
      zspan = strtod(argv[++i], NULL);
    else {
      fprintf(stderr, "tracecli: bad arg %s\n", argv[i]);
      return 2;
    }
  }
  if (!have_seed) { /* default: volume center */
    r3d_cpuvol v;
    if (r3d_cpuvol_open(&v, root, 8) != 0) {
      fprintf(stderr, "tracecli: cannot open %s\n", root);
      return 1;
    }
    cfg.seed[0] = (double)v.nx * 0.5;
    cfg.seed[1] = (double)v.ny * 0.5;
    cfg.seed[2] = (double)v.nz * 0.5;
    r3d_cpuvol_close(&v);
  }
  r3d_umbilicus umb;
  r3d_umbilicus_init(&umb);
  if (umbp && r3d_umbilicus_load(&umb, umbp) != 0)
    fprintf(stderr, "tracecli: no umbilicus at %s (continuing)\n", umbp);
  if (zspan > 0) { /* thin-slab clamp about the seed (vc3d z_range) */
    cfg.z_min = cfg.seed[2] - zspan * 0.5;
    cfg.z_max = cfg.seed[2] + zspan * 0.5;
  }
  if (spiral && cfg.wind_weight == 0.0 && umb.count >= 2)
    cfg.wind_weight = 0.5; /* spiral prior on by default with an umbilicus */
  if (!spiral) cfg.wind_weight = 0.0;
  printf("tracecli: seed %.0f,%.0f,%.0f step %.0f gens %u level L%u spiral %s\n",
         cfg.seed[0], cfg.seed[1], cfg.seed[2], cfg.step, cfg.max_ring, cfg.level,
         cfg.wind_weight > 0 ? "on" : "off");
  r3d_tracer tr;
  if (r3d_tracer_start_fused(&tr, root, &cfg, &umb, fuse, nfuse) != 0) {
    fprintf(stderr, "tracecli: tracer start failed\n");
    return 1;
  }
  uint32_t last_ring = 0;
  bool done = false;
  while (!done) {
    usleep(500 * 1000);
    uint32_t ring = 0, nset = 0;
    r3d_tracer_snapshot(&tr, NULL, NULL, NULL, &ring, &nset, &done);
    if (ring != last_ring || done) {
      if (tr.sp_valid)
        printf("tracecli: generation %u/%u, %u points (omega %.1f rms %.1f)\n", ring,
               tr.cfg.max_ring, nset, tr.sp_omega, tr.sp_rms);
      else
        printf("tracecli: generation %u/%u, %u points\n", ring, tr.cfg.max_ring, nset);
      last_ring = ring;
    }
  }
  r3d_tracer_stop(&tr);
  int rc = 0;
  if (out && tr.nset > 8) {
    char mk[1400];
    snprintf(mk, sizeof mk, "mkdir -p '%s'", out);
    if (system(mk) != 0 || r3d_tracer_save(&tr, out, tr.cfg.thresh, fill) != 0) {
      fprintf(stderr, "tracecli: save failed (%s)\n", out);
      rc = 1;
    } else {
      printf("tracecli: saved %s (%ux%u, %u pts, cutoff %.2f)\n", out, tr.W, tr.H,
             tr.nset, (double)tr.cfg.thresh);
    }
  }
  r3d_tracer_free(&tr);
  r3d_umbilicus_free(&umb);
  return rc;
}
