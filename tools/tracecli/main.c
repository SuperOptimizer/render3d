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

/* seeds.json (written by the GUI "export seeds" button):
 * {"step": S, "gens": N, "level": L, "cutoff": C, "seeds": [[x,y,z], ...]}
 * Missing scalars keep the command-line/default values. */
#define SEEDS_JSON_MAX 256
static uint32_t seeds_json_load(const char *path, r3d_tracer_cfg *cfg,
                                double seeds[][3], uint32_t max) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "tracecli: cannot open %s\n", path);
    return 0;
  }
  static char buf[1 << 20];
  size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[n] = 0;
  const char *q;
  if ((q = strstr(buf, "\"step\"")) && (q = strchr(q, ':')))
    cfg->step = strtod(q + 1, NULL);
  if ((q = strstr(buf, "\"gens\"")) && (q = strchr(q, ':')))
    cfg->max_ring = (uint32_t)strtoul(q + 1, NULL, 10);
  if ((q = strstr(buf, "\"level\"")) && (q = strchr(q, ':')))
    cfg->level = (uint32_t)strtoul(q + 1, NULL, 10);
  if ((q = strstr(buf, "\"cutoff\"")) && (q = strchr(q, ':')))
    cfg->thresh = strtof(q + 1, NULL);
  const char *a = strstr(buf, "\"seeds\"");
  if (!a || !(a = strchr(a, '['))) return 0;
  a++; /* inside the outer array */
  uint32_t ns = 0;
  while (ns < max && (a = strchr(a, '['))) {
    char *e = NULL;
    double x = strtod(a + 1, &e);
    if (e == a + 1) break;
    while (*e == ',' || *e == ' ' || *e == '\n') e++;
    double y = strtod(e, &e);
    while (*e == ',' || *e == ' ' || *e == '\n') e++;
    double z = strtod(e, &e);
    seeds[ns][0] = x;
    seeds[ns][1] = y;
    seeds[ns][2] = z;
    ns++;
    a = e;
  }
  return ns;
}

static uint32_t g_subdiv = 0; /* --subdivide N: halve the step N times
                               * after growth, re-solving each level */

static void do_subdivides(r3d_tracer *tr) {
  for (uint32_t sd = 0; sd < g_subdiv; sd++) {
    printf("tracecli: subdivide %u/%u\n", sd + 1, g_subdiv);
    if (r3d_tracer_subdivide(tr) != 0) {
      fprintf(stderr, "tracecli: subdivide failed (grid too large?)\n");
      return;
    }
    bool sdone = false;
    while (!sdone) {
      usleep(500 * 1000);
      r3d_tracer_snapshot(tr, NULL, NULL, NULL, NULL, NULL, &sdone);
    }
    r3d_tracer_stop(tr);
  }
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* grow one seed to completion and save; returns 0 on success */
static int run_trace(const char *root, const r3d_tracer_cfg *cfg,
                     const r3d_umbilicus *umb, const char **fuse, uint32_t nfuse,
                     const char *out, bool fill) {
  r3d_tracer tr;
  if (r3d_tracer_start_fused(&tr, root, cfg, umb, fuse, nfuse) != 0) {
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
        printf("tracecli: generation %u/%u, %u points (omega %.1f rms %.1f)\n",
               ring, tr.cfg.max_ring, nset, tr.sp_omega, tr.sp_rms);
      else
        printf("tracecli: generation %u/%u, %u points\n", ring, tr.cfg.max_ring,
               nset);
      last_ring = ring;
    }
  }
  r3d_tracer_stop(&tr);
  do_subdivides(&tr);
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
  return rc;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: tracecli <pred-root> [--seed x y z] [--seeds FILE.json] [--step S] "
            "[--gens N] "
            "[--level L] [--cutoff C] [--out DIR] [--umbilicus FILE] [--nofill] "
            "[--nospiral] [--spiral-weight W] [--fuse DIR ...] [--ribbon ROWS] "
            "[--zspan S] [--wraps N] [--ct CT-ROOT] [--ct-min V] [--subdivide N]\n");
    return 2;
  }
  const char *root = argv[1], *out = NULL, *umbp = NULL, *seedsp = NULL;
  const char *fuse[16], *loadp = NULL;
  uint32_t nfuse = 0, rewind_gen = 0, regrow = 0;
  int derive_dir = 0;
  double zspan = 0.0;
  bool fill = true, spiral = true;
  r3d_tracer_cfg cfg = {.step = 20.0, .thresh = 0.35f, .max_ring = 60, .level = 1};
  bool have_seed = false;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--seed") == 0 && i + 3 < argc) {
      for (int a = 0; a < 3; a++) cfg.seed[a] = strtod(argv[++i], NULL);
      have_seed = true;
    } else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc)
      seedsp = argv[++i];
    else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc)
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
    else if (strcmp(argv[i], "--ct-min") == 0 && i + 1 < argc)
      cfg.ct_min = strtod(argv[++i], NULL);
    else if (strcmp(argv[i], "--zspan") == 0 && i + 1 < argc)
      zspan = strtod(argv[++i], NULL);
    else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc)
      loadp = argv[++i];
    else if (strcmp(argv[i], "--rewind") == 0 && i + 1 < argc)
      rewind_gen = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--regrow") == 0 && i + 1 < argc)
      regrow = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--subdivide") == 0 && i + 1 < argc)
      g_subdiv = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--derive") == 0 && i + 1 < argc)
      derive_dir = atoi(argv[++i]);
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
  if (seedsp) { /* batch: every seed from the json, sequentially */
    static double seeds[SEEDS_JSON_MAX][3];
    uint32_t ns = seeds_json_load(seedsp, &cfg, seeds, SEEDS_JSON_MAX);
    if (!ns) {
      fprintf(stderr, "tracecli: no seeds in %s\n", seedsp);
      return 1;
    }
    if (spiral && cfg.wind_weight == 0.0 && umb.count >= 2) cfg.wind_weight = 0.5;
    if (!spiral) cfg.wind_weight = 0.0;
    printf("tracecli: %u seed%s from %s (step %.0f gens %u level L%u spiral %s)\n",
           ns, ns == 1 ? "" : "s", seedsp, cfg.step, cfg.max_ring, cfg.level,
           cfg.wind_weight > 0 ? "on" : "off");
    int bad = 0;
    double tall = now_s();
    for (uint32_t si = 0; si < ns; si++) {
      r3d_tracer_cfg c2 = cfg;
      memcpy(c2.seed, seeds[si], sizeof c2.seed);
      char od[1300];
      snprintf(od, sizeof od, "%s/seed-%03u", out ? out : "traced-seeds", si);
      printf("tracecli: [%u/%u] seed %.0f,%.0f,%.0f -> %s\n", si + 1, ns,
             c2.seed[0], c2.seed[1], c2.seed[2], od);
      double t0 = now_s();
      bad += run_trace(root, &c2, &umb, fuse, nfuse, od, fill) != 0;
      printf("tracecli: [%u/%u] %.1fs\n", si + 1, ns, now_s() - t0);
    }
    printf("tracecli: batch done, %u/%u ok, %.1fs total\n", ns - (uint32_t)bad, ns,
           now_s() - tall);
    r3d_umbilicus_free(&umb);
    return bad ? 1 : 0;
  }
  printf("tracecli: seed %.0f,%.0f,%.0f step %.0f gens %u level L%u spiral %s\n",
         cfg.seed[0], cfg.seed[1], cfg.seed[2], cfg.step, cfg.max_ring, cfg.level,
         cfg.wind_weight > 0 ? "on" : "off");
  r3d_tracer tr;
  if (loadp && derive_dir) {
    /* unified-tracer stage 1: load a wrap, derive its neighbor */
    if (r3d_tracer_load(&tr, loadp, root) != 0) {
      fprintf(stderr, "tracecli: cannot load %s\n", loadp);
      return 1;
    }
    if (!out) {
      fprintf(stderr, "tracecli: --derive needs --out\n");
      return 2;
    }
    char mk[1300];
    snprintf(mk, sizeof mk, "mkdir -p '%s'", out);
    if (system(mk) != 0) return 1;
    int nh = r3d_tracer_derive(&tr, root, derive_dir, out);
    r3d_tracer_free(&tr);
    return nh > 0 ? 0 : 1;
  }
  if (loadp) {
    /* load -> optional rewind -> regrow (vc3d --resume/--rewind-gen) */
    if (r3d_tracer_load(&tr, loadp, root) != 0) {
      fprintf(stderr, "tracecli: cannot load %s\n", loadp);
      return 1;
    }
    if (rewind_gen && r3d_tracer_rewind(&tr, rewind_gen) != 0) {
      fprintf(stderr, "tracecli: rewind failed\n");
      return 1;
    }
    if (!regrow) regrow = 10;
    if (r3d_tracer_grow(&tr, regrow) != 0) {
      fprintf(stderr, "tracecli: regrow failed\n");
      return 1;
    }
  } else if (r3d_tracer_start_fused(&tr, root, &cfg, &umb, fuse, nfuse) != 0) {
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
  do_subdivides(&tr);
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
