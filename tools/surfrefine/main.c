/* surfrefine: automatic refinement of an existing segmentation surface.
 *
 *   surfrefine --pred ROOT --in SURF --out DIR [--ct-root DIR --ct-cut V]
 *              [--ct-reach R] [--level L] [--subdivide N] [--no-refine]
 *              [--no-ctsnap] [--qc-only] [--cutoff C] [--flag-conf F]
 *              [--report qc.json] [--threads N]
 *
 * SURF is a .sfc file or a tifxyz directory. The surface is loaded into the
 * tracer (tracer.json restores a faithful configuration; a published surface
 * without one is imported with assumed defaults), the mesh QC of the loaded
 * grid is recorded as the baseline, then: optional --subdivide (halve the
 * grid step N times), the solve-only refine pass toward the prediction
 * volume at ROOT (staged weights, tangential position memory), the optional
 * CT edge snap when --ct-root names the raw CT LOD tree, and the QC again.
 * The result is saved as a NEW tifxyz directory DIR (refused when DIR is
 * the source or inside it) plus DIR.sfc, and a JSON report with before/after
 * QC and the flagged grid tiles (16x16, the segment-store tile size) whose
 * trusted-cell fraction is below 0.5 or that hold a fold/kink. --qc-only
 * loads, measures and reports without solving or saving.
 *
 *   surfrefine --pred ROOT --group A,B,... --out DIR [--rounds N] [--level L] [--threads N]
 *
 * loads several neighbouring surfaces and solves them jointly
 * (r3d_tracer_group_refine: alternating per-sheet solves with cross-sheet
 * no-crossing and spacing terms), then saves each as DIR/<name> (+ .sfc). */
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "core/segstore.h"
#include "core/surfconv.h"
#include "core/tracer.h"

typedef struct qc_row {
  uint32_t folds, kinks;
  float twist, slant_p95, bend_p5, bend_med, fill, hole, wrap_frac, conf_mean;
  double area;
  uint32_t nset, ntrust;
} qc_row;

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static qc_row qc_take(r3d_tracer *t, float flag_conf) {
  r3d_tracer_qc(t);
  qc_row q = {0};
  q.folds = atomic_load(&t->qc_folds);
  q.kinks = atomic_load(&t->qc_kinks);
  q.twist = atomic_load(&t->qc_twist);
  q.slant_p95 = t->qc_slant_p95;
  q.bend_p5 = t->qc_bend_p5;
  q.bend_med = t->qc_bend_med;
  q.fill = t->qc_fill;
  q.hole = t->qc_hole;
  q.wrap_frac = t->qc_wrap_frac;
  q.area = t->qc_area_vx2;
  q.nset = t->nset;
  double cs = 0.0;
  uint64_t n = (uint64_t)t->W * t->H;
  for (uint64_t k = 0; k < n; k++) {
    if (t->state[k] != R3D_TR_SET) continue;
    cs += (double)t->conf[k];
    if (t->conf[k] >= flag_conf) q.ntrust++;
  }
  q.conf_mean = q.nset ? (float)(cs / q.nset) : 0.0f;
  return q;
}

static void qc_json(FILE *f, const char *key, const qc_row *q) {
  fprintf(f,
          "  \"%s\": {\"folds\": %u, \"kinks\": %u, \"twist\": %.4f, "
          "\"slant_p95\": %.4f, \"bend_p5\": %.2f, \"bend_med\": %.2f, "
          "\"fill\": %.4f, \"hole\": %.4f, \"wrap_frac\": %.4f, "
          "\"conf_mean\": %.4f, \"area_vx2\": %.1f, \"points\": %u, "
          "\"trusted\": %u}",
          key, q->folds, q->kinks, (double)q->twist, (double)q->slant_p95,
          (double)q->bend_p5, (double)q->bend_med, (double)q->fill,
          (double)q->hole, (double)q->wrap_frac, (double)q->conf_mean, q->area,
          q->nset, q->ntrust);
}

/* run the worker to completion; returns false if it never finished */
static bool wait_done(r3d_tracer *t, const char *what, bool verbose) {
  bool done = false;
  double t0 = now_s(), last = t0;
  while (!done) {
    usleep(100 * 1000);
    uint32_t ring = 0, nset = 0;
    r3d_tracer_snapshot(t, NULL, NULL, NULL, &ring, &nset, &done);
    if (verbose && now_s() - last > 5.0) {
      printf("surfrefine: %s running (%.0f s, %u points)\n", what, now_s() - t0, nset);
      fflush(stdout);
      last = now_s();
    }
  }
  r3d_tracer_stop(t);
  if (verbose) printf("surfrefine: %s done in %.1f s\n", what, now_s() - t0);
  return true;
}

/* true when `inner` equals `outer` or lies inside it (path prefix) */
/* canonical absolute path; a path that does not exist yet resolves through
 * its parent directory so "a//b/../c" and "a/c" compare equal */
static void canon(const char *p, char *out, size_t n) {
  if (realpath(p, out)) return;
  char parent[4096], base[1024];
  snprintf(parent, sizeof parent, "%s", p);
  size_t len = strlen(parent);
  while (len > 1 && parent[len - 1] == '/') parent[--len] = 0;
  char *sl = strrchr(parent, '/');
  if (sl && sl != parent) {
    snprintf(base, sizeof base, "%s", sl + 1);
    *sl = 0;
  } else {
    snprintf(base, sizeof base, "%s", sl ? sl + 1 : parent);
    snprintf(parent, sizeof parent, "%s", sl ? "/" : ".");
  }
  char pr[4096];
  if (!realpath(parent, pr)) snprintf(pr, sizeof pr, "%s", parent);
  snprintf(out, n, "%s/%s", pr, base);
}

static bool path_within(const char *inner, const char *outer) {
  char a[4096], b[4096];
  canon(inner, a, sizeof a);
  canon(outer, b, sizeof b);
  size_t lb = strlen(b);
  while (lb > 1 && b[lb - 1] == '/') b[--lb] = 0;
  return strncmp(a, b, lb) == 0 && (a[lb] == 0 || a[lb] == '/');
}

static int usage(void) {
  fprintf(stderr,
          "usage: surfrefine --pred ROOT --in SURF --out DIR [--ct-root DIR --ct-cut V]\n"
          "                  [--ct-reach R] [--level L] [--subdivide N] [--no-refine]\n"
          "                  [--no-ctsnap] [--qc-only] [--cutoff C] [--flag-conf F]\n"
          "                  [--report qc.json] [--threads N]\n"
          "       surfrefine --pred ROOT --group A,B,... --out DIR [--rounds N] [--level L]\n");
  return EXIT_FAILURE;
}

static const char *surf_base(const char *path, char *buf, size_t n) {
  size_t len = strlen(path);
  while (len > 1 && path[len - 1] == '/') len--;
  size_t start = len;
  while (start > 0 && path[start - 1] != '/') start--;
  size_t bl = len - start < n - 1 ? len - start : n - 1;
  memcpy(buf, path + start, bl);
  buf[bl] = 0;
  if (bl > 4 && !strcmp(buf + bl - 4, ".sfc")) buf[bl - 4] = 0;
  else if (bl > 7 && !strcmp(buf + bl - 7, ".tifxyz")) buf[bl - 7] = 0;
  return buf;
}

/* --group: joint refinement of N surfaces */
static int run_group(const char *pred, const char *group, const char *out, int rounds,
                     int level, int threads) {
  char list[4096];
  snprintf(list, sizeof list, "%s", group);
  r3d_tracer_group g = {0};
  static r3d_tracer tr[R3D_TR_GROUP_MAX];
  char scratch[R3D_TR_GROUP_MAX][1200], name[R3D_TR_GROUP_MAX][256];
  int rc = EXIT_FAILURE;
  uint32_t n = 0;
  for (char *tok = strtok(list, ","); tok && n < R3D_TR_GROUP_MAX; tok = strtok(NULL, ",")) {
    scratch[n][0] = 0;
    const char *src = tok;
    r3d_surf_kind kind = r3d_surf_kind_of(tok);
    if (kind == R3D_SURF_SFC) {
      snprintf(scratch[n], sizeof scratch[n], "%s/surfrefine-grp-%ld-%u",
               getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (long)getpid(), n);
      if (r3d_surf_decode(tok, scratch[n], NULL, NULL)) {
        fprintf(stderr, "surfrefine: cannot decode %s\n", tok);
        goto done;
      }
      src = scratch[n];
    } else if (kind != R3D_SURF_TIFXYZ) {
      fprintf(stderr, "surfrefine: %s is not a surface\n", tok);
      goto done;
    }
    if (path_within(out, tok)) {
      fprintf(stderr, "surfrefine: --out must not be inside a group member\n");
      goto done;
    }
    if (r3d_tracer_load(&tr[n], src, pred) != 0) {
      fprintf(stderr, "surfrefine: cannot load %s\n", tok);
      goto done;
    }
    if (level >= 0) tr[n].cfg.level = (uint32_t)level;
    if (threads > 0) tr[n].cfg.max_threads = (uint32_t)threads;
    surf_base(tok, name[n], sizeof name[n]);
    g.m[n] = &tr[n];
    n++;
  }
  g.n = n;
  if (n < 2) {
    fprintf(stderr, "surfrefine: --group needs at least two surfaces\n");
    goto done;
  }
  double t0 = now_s();
  int ran = r3d_tracer_group_refine(&g, rounds);
  if (ran < 0) {
    fprintf(stderr, "surfrefine: group refine failed\n");
    goto done;
  }
  printf("surfrefine: joint refine of %u surfaces, %d round%s in %.1f s\n", n, ran,
         ran == 1 ? "" : "s", now_s() - t0);
  if (mkdir(out, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "surfrefine: cannot create %s\n", out);
    goto done;
  }
  rc = EXIT_SUCCESS;
  for (uint32_t i = 0; i < n; i++) {
    char od[1400], sfc[1500];
    snprintf(od, sizeof od, "%s/%s", out, name[i]);
    if ((mkdir(od, 0755) != 0 && errno != EEXIST) ||
        r3d_tracer_save(&tr[i], od, tr[i].cfg.thresh, false) != 0) {
      fprintf(stderr, "surfrefine: save failed (%s)\n", od);
      rc = EXIT_FAILURE;
      continue;
    }
    if (r3d_surf_sibling(od, sfc, sizeof sfc) == 0) {
      unlink(sfc);
      if (r3d_surf_encode(od, sfc, r3d_surf_default_error(), NULL, NULL) != 0)
        fprintf(stderr, "surfrefine: warning: .sfc encode failed for %s\n", od);
    }
    r3d_tracer_qc(&tr[i]);
    printf("surfrefine: saved %s (%u points, folds %u kinks %u)\n", od, tr[i].nset,
           atomic_load(&tr[i].qc_folds), atomic_load(&tr[i].qc_kinks));
  }
done:
  for (uint32_t i = 0; i < n; i++) {
    r3d_tracer_stop(&tr[i]);
    r3d_tracer_free(&tr[i]);
  }
  for (uint32_t i = 0; i < R3D_TR_GROUP_MAX; i++)
    if (i < n && scratch[i][0]) {
      char cmd[1300];
      snprintf(cmd, sizeof cmd, "rm -rf '%s'", scratch[i]);
      (void)system(cmd);
    }
  return rc;
}

int main(int argc, char **argv) {
  const char *pred = NULL, *in = NULL, *out = NULL, *ct_root = NULL, *report = NULL;
  const char *group = NULL;
  int rounds = 3;
  double ct_cut = 128.0, ct_reach = 6.0, cutoff = -1.0;
  float flag_conf = 0.5f;
  int subdiv = 0, level = -1, threads = 0;
  bool do_refine = true, do_ctsnap = true, qc_only = false;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    bool more = i + 1 < argc;
    if (!strcmp(a, "--pred") && more) pred = argv[++i];
    else if (!strcmp(a, "--in") && more) in = argv[++i];
    else if (!strcmp(a, "--out") && more) out = argv[++i];
    else if (!strcmp(a, "--ct-root") && more) ct_root = argv[++i];
    else if (!strcmp(a, "--ct-cut") && more) ct_cut = atof(argv[++i]);
    else if (!strcmp(a, "--ct-reach") && more) ct_reach = atof(argv[++i]);
    else if (!strcmp(a, "--cutoff") && more) cutoff = atof(argv[++i]);
    else if (!strcmp(a, "--flag-conf") && more) flag_conf = (float)atof(argv[++i]);
    else if (!strcmp(a, "--subdivide") && more) subdiv = atoi(argv[++i]);
    else if (!strcmp(a, "--level") && more) level = atoi(argv[++i]);
    else if (!strcmp(a, "--threads") && more) threads = atoi(argv[++i]);
    else if (!strcmp(a, "--report") && more) report = argv[++i];
    else if (!strcmp(a, "--group") && more) group = argv[++i];
    else if (!strcmp(a, "--rounds") && more) rounds = atoi(argv[++i]);
    else if (!strcmp(a, "--no-refine")) do_refine = false;
    else if (!strcmp(a, "--no-ctsnap")) do_ctsnap = false;
    else if (!strcmp(a, "--qc-only")) qc_only = true;
    else return usage();
  }
  if (group) {
    if (!pred || !out) return usage();
    return run_group(pred, group, out, rounds, level, threads);
  }
  if (!in || (!qc_only && (!pred || !out))) return usage();
  if (subdiv < 0 || subdiv > 4) {
    fprintf(stderr, "surfrefine: --subdivide must be 0..4\n");
    return EXIT_FAILURE;
  }
  if (out && path_within(out, in)) {
    fprintf(stderr, "surfrefine: --out must not be the source or inside it\n");
    return EXIT_FAILURE;
  }
  /* the tracer reads tifxyz planes: decode an .sfc into a scratch dir */
  char scratch[1200] = "", src_dir[1200];
  r3d_surf_kind kind = r3d_surf_kind_of(in);
  if (kind == R3D_SURF_SFC) {
    snprintf(scratch, sizeof scratch, "%s/surfrefine-src-%ld", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp",
             (long)getpid());
    if (r3d_surf_decode(in, scratch, NULL, NULL)) {
      fprintf(stderr, "surfrefine: cannot decode %s\n", in);
      return EXIT_FAILURE;
    }
    snprintf(src_dir, sizeof src_dir, "%s", scratch);
  } else if (kind == R3D_SURF_TIFXYZ) {
    snprintf(src_dir, sizeof src_dir, "%s", in);
  } else {
    fprintf(stderr, "surfrefine: %s is neither a .sfc file nor a tifxyz directory\n", in);
    return EXIT_FAILURE;
  }
  int rc = EXIT_FAILURE;
  r3d_tracer t;
  double t0 = now_s();
  if (r3d_tracer_load(&t, src_dir, pred ? pred : "") != 0) {
    fprintf(stderr, "surfrefine: cannot load %s\n", in);
    goto cleanup_scratch;
  }
  if (level >= 0) t.cfg.level = (uint32_t)level;
  if (threads > 0) t.cfg.max_threads = (uint32_t)threads;
  if (cutoff >= 0.0) t.cfg.thresh = (float)cutoff;
  qc_row before = qc_take(&t, flag_conf);
  printf("surfrefine: loaded %s (%ux%u, %u points, step %.1f): folds %u kinks %u "
         "fill %.3f conf %.3f\n",
         in, t.W, t.H, t.nset, t.cfg.step, before.folds, before.kinks,
         (double)before.fill, (double)before.conf_mean);
  qc_row after = before;
  double t_sub = 0, t_ref = 0, t_snap = 0;
  if (!qc_only) {
    for (int s = 0; s < subdiv; s++) {
      double ts = now_s();
      if (r3d_tracer_subdivide(&t) != 0 || !wait_done(&t, "subdivide", true)) {
        fprintf(stderr, "surfrefine: subdivide failed\n");
        goto done;
      }
      t_sub += now_s() - ts;
    }
    if (do_refine) {
      double ts = now_s();
      if (r3d_tracer_refine(&t) != 0 || !wait_done(&t, "refine", true)) {
        fprintf(stderr, "surfrefine: refine failed\n");
        goto done;
      }
      t_ref = now_s() - ts;
    }
    if (do_ctsnap && ct_root && ct_root[0]) {
      double ts = now_s();
      if (r3d_tracer_ctsnap(&t, ct_root, ct_reach, ct_cut) != 0 ||
          !wait_done(&t, "ctsnap", true)) {
        fprintf(stderr, "surfrefine: CT snap failed\n");
        goto done;
      }
      t_snap = now_s() - ts;
    }
    after = qc_take(&t, flag_conf);
    printf("surfrefine: refined: folds %u kinks %u fill %.3f conf %.3f area %.0f -> %.0f\n",
           after.folds, after.kinks, (double)after.fill, (double)after.conf_mean,
           before.area, after.area);
    if (mkdir(out, 0755) != 0 && errno != EEXIST) {
      fprintf(stderr, "surfrefine: cannot create %s: %s\n", out, strerror(errno));
      goto done;
    }
    if (r3d_tracer_save(&t, out, t.cfg.thresh, false) != 0) {
      fprintf(stderr, "surfrefine: save failed (%s)\n", out);
      goto done;
    }
    char sfc[1300];
    if (r3d_surf_sibling(out, sfc, sizeof sfc) == 0) {
      unlink(sfc); /* a stale sibling from an earlier run */
      if (r3d_surf_encode(out, sfc, r3d_surf_default_error(), NULL, NULL) != 0)
        fprintf(stderr, "surfrefine: warning: .sfc encode failed for %s\n", out);
      else
        printf("surfrefine: saved %s and %s\n", out, sfc);
    }
  }
  /* ---- report: before/after QC + flagged tiles ---- */
  {
    char rp[1300];
    if (report) snprintf(rp, sizeof rp, "%s", report);
    else if (out) snprintf(rp, sizeof rp, "%s/refine_qc.json", out);
    else snprintf(rp, sizeof rp, "(stdout)");
    FILE *f = report || out ? fopen(rp, "w") : stdout;
    if (!f) {
      fprintf(stderr, "surfrefine: cannot write %s\n", rp);
      goto done;
    }
    fprintf(f, "{\n  \"source\": \"%s\",\n  \"output\": \"%s\",\n  \"grid\": [%u, %u],\n"
               "  \"step\": %.4f,\n  \"pred_root\": \"%s\",\n  \"ct_root\": \"%s\",\n"
               "  \"subdivide\": %d,\n  \"seconds\": {\"total\": %.1f, \"subdivide\": %.1f, "
               "\"refine\": %.1f, \"ctsnap\": %.1f},\n",
            in, out ? out : "", t.W, t.H, t.cfg.step, pred ? pred : "",
            ct_root ? ct_root : "", subdiv, now_s() - t0, t_sub, t_ref, t_snap);
    qc_json(f, "before", &before);
    fputs(",\n", f);
    qc_json(f, "after", &after);
    /* flagged tiles: low trusted fraction, or holding a fold/kink cell */
    const uint32_t T = R3D_SEGSTORE_TILE;
    uint32_t tw = (t.W + T - 1) / T, th = (t.H + T - 1) / T, nflag = 0;
    fputs(",\n  \"flag_conf\": ", f);
    fprintf(f, "%.3f,\n  \"tile\": %u,\n  \"flagged\": [", (double)flag_conf, T);
    for (uint32_t tj = 0; tj < th; tj++)
      for (uint32_t ti = 0; ti < tw; ti++) {
        uint32_t nset = 0, ntr = 0, nfk = 0;
        double c[3] = {0, 0, 0};
        for (uint32_t j = tj * T; j < (tj + 1) * T && j < t.H; j++)
          for (uint32_t i = ti * T; i < (ti + 1) * T && i < t.W; i++) {
            size_t k = (size_t)j * t.W + i;
            if (t.state[k] != R3D_TR_SET) continue;
            nset++;
            if (t.conf[k] >= flag_conf) ntr++;
            for (int a = 0; a < 3; a++) c[a] += t.pos[k * 3 + (size_t)a];
          }
        if (!nset) continue;
        for (uint32_t q = 0; q < t.qc_nfoldc && q < 16; q++)
          if (t.qc_fold_cell[q] / t.W / T == tj && t.qc_fold_cell[q] % t.W / T == ti) nfk++;
        for (uint32_t q = 0; q < t.qc_nkinkc && q < 16; q++)
          if (t.qc_kink_cell[q] / t.W / T == tj && t.qc_kink_cell[q] % t.W / T == ti) nfk++;
        double frac = (double)ntr / nset;
        if (frac >= 0.5 && !nfk) continue;
        fprintf(f, "%s\n    {\"tile\": [%u, %u], \"grid\": [%u, %u, %u, %u], "
                   "\"center\": [%.1f, %.1f, %.1f], \"trusted\": %.3f, \"points\": %u, "
                   "\"defects\": %u}",
                nflag ? "," : "", ti, tj, ti * T, tj * T,
                (ti + 1) * T < t.W ? (ti + 1) * T : t.W, (tj + 1) * T < t.H ? (tj + 1) * T : t.H,
                c[0] / nset, c[1] / nset, c[2] / nset, frac, nset, nfk);
        nflag++;
      }
    fprintf(f, "%s],\n  \"flagged_count\": %u\n}\n", nflag ? "\n  " : "", nflag);
    if (f != stdout && fclose(f) != 0) {
      fprintf(stderr, "surfrefine: writing %s failed\n", rp);
      goto done;
    }
    printf("surfrefine: %u flagged tile%s, report %s\n", nflag, nflag == 1 ? "" : "s", rp);
  }
  rc = EXIT_SUCCESS;
done:
  r3d_tracer_stop(&t);
  r3d_tracer_free(&t);
cleanup_scratch:
  if (scratch[0]) {
    char cmd[1300];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", scratch);
    (void)system(cmd);
  }
  return rc;
}
