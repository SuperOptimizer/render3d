/* surfrefine end to end on synthetic data: a spiral "prediction" LOD tree
 * (bright shells rho = 30 + 14 w about (128,128)), a tifxyz surface that
 * sits on one wrap but is perturbed off it, then the CLI refines it. The
 * refined surface must move back toward the wrap, QC must not get worse,
 * the report must exist with before/after blocks and the output must never
 * be written into the source. */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <unistd.h>

#include "core/surfconv.h"
#include "core/tifxyz.h"
#include "synthtree.h"

#undef assert
#define assert(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define TDIM 256u
#define SW 48u /* grid columns along the wrap (arc length) */
#define SH 20u /* grid rows along z */
#define STEP 4.0

static void write_plane(const char *path, const float *v, uint32_t w, uint32_t h) {
  TIFF *tf = TIFFOpen(path, "w");
  assert(tf);
  TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(tf, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, 64);
  for (uint32_t j = 0; j < h; j++)
    assert(TIFFWriteScanline(tf, (void *)(v + (size_t)j * w), j, 0) >= 0);
  TIFFClose(tf);
}

/* the wrap at radius r0 = 30 + 14*2 = 58, parameterised by arc length
 * (STEP voxels per column) and z (STEP per row); `off` pushes every point
 * radially off the sheet */
static void make_surface(const char *dir, double off) {
  assert(!mkdir(dir, 0755));
  float *x = malloc(SW * SH * sizeof *x), *y = malloc(SW * SH * sizeof *y),
        *z = malloc(SW * SH * sizeof *z);
  const double r0 = 58.0;
  for (uint32_t j = 0; j < SH; j++)
    for (uint32_t i = 0; i < SW; i++) {
      size_t k = (size_t)j * SW + i;
      double th = (double)i * STEP / r0 + 0.3;
      double r = r0 + off * sin((double)i * 0.37 + (double)j * 0.21);
      x[k] = (float)(128.0 + r * cos(th));
      y[k] = (float)(128.0 + r * sin(th));
      z[k] = (float)(90.0 + (double)j * STEP);
    }
  char p[700];
  snprintf(p, sizeof p, "%s/x.tif", dir);
  write_plane(p, x, SW, SH);
  snprintf(p, sizeof p, "%s/y.tif", dir);
  write_plane(p, y, SW, SH);
  snprintf(p, sizeof p, "%s/z.tif", dir);
  write_plane(p, z, SW, SH);
  snprintf(p, sizeof p, "%s/meta.json", dir);
  FILE *f = fopen(p, "w");
  assert(f);
  fprintf(f, "{\"format\":\"tifxyz\",\"scale\":[%.6f,%.6f]}\n", 1.0 / STEP, 1.0 / STEP);
  fclose(f);
  free(x);
  free(y);
  free(z);
}

/* mean |rho - nearest wrap radius| over valid points */
static double off_sheet(const r3d_tifxyz *s) {
  double sum = 0;
  uint64_t n = 0;
  for (uint32_t j = 0; j < s->h; j++)
    for (uint32_t i = 0; i < s->w; i++) {
      const float *p = r3d_tifxyz_at(s, i, j);
      if (!r3d_tifxyz_valid(p)) continue;
      double rho = hypot((double)p[0] - 128.0, (double)p[1] - 128.0);
      double m = fmod(rho - 30.0, 14.0);
      if (m < 0) m += 14.0;
      sum += m < 7.0 ? m : 14.0 - m;
      n++;
    }
  return n ? sum / (double)n : 1e9;
}

static char *slurp(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  char *buf = malloc(1 << 20);
  size_t n = fread(buf, 1, (1 << 20) - 1, f);
  fclose(f);
  buf[n] = 0;
  return buf;
}

static long jint(const char *json, const char *block, const char *key) {
  const char *b = strstr(json, block);
  assert(b);
  char k[64];
  snprintf(k, sizeof k, "\"%s\": ", key);
  const char *v = strstr(b, k);
  assert(v);
  return strtol(v + strlen(k), NULL, 10);
}

int main(int argc, char **argv) {
  assert(argc > 1);
  char cli[4096];
  assert(realpath(argv[1], cli));
  setenv("R3D_BEND_RMIN", "12", 1); /* tight synthetic spiral */
  char tmp[512];
  const char *base = getenv("TMPDIR");
  snprintf(tmp, sizeof tmp, "%s/r3d_surfrefine_XXXXXX", base && *base ? base : "/tmp");
  assert(mkdtemp(tmp));
  char root[600], src[600], out[600], out_sfc[700], rep[700], cmd[8192];
  snprintf(root, sizeof root, "%s/pred", tmp);
  uint32_t tdim[3] = {TDIM, TDIM, TDIM};
  st_spiral_mode = 1;
  assert(st_make_tree(root, tdim, 2, 0) == 0);
  snprintf(src, sizeof src, "%s/seg", tmp);
  make_surface(src, 3.0);
  r3d_tifxyz before;
  assert(!r3d_tifxyz_load(&before, src));
  double d0 = off_sheet(&before);
  assert(d0 > 1.5 && d0 < 3.5);

  /* --out inside the source is refused */
  snprintf(cmd, sizeof cmd, "'%s' --pred '%s' --in '%s' --out '%s/sub' >/dev/null 2>&1", cli, root,
           src, src);
  assert(system(cmd) != 0);
  /* --qc-only reports without writing */
  snprintf(rep, sizeof rep, "%s/qc-only.json", tmp);
  snprintf(cmd, sizeof cmd, "'%s' --pred '%s' --in '%s' --qc-only --report '%s' >/dev/null", cli,
           root, src, rep);
  assert(system(cmd) == 0);
  char *j = slurp(rep);
  assert(j && strstr(j, "\"before\"") && strstr(j, "\"after\"") && strstr(j, "\"flagged\""));
  free(j);

  /* the refinement itself: prediction level 0 (the 256^3 tree has two) */
  snprintf(out, sizeof out, "%s/refined", tmp);
  snprintf(cmd, sizeof cmd,
           "'%s' --pred '%s' --in '%s' --out '%s' --level 0 --no-ctsnap --threads 4", cli, root,
           src, out);
  int rc = system(cmd);
  assert(rc == 0);
  snprintf(rep, sizeof rep, "%s/refine_qc.json", out);
  j = slurp(rep);
  assert(j);
  long f0 = jint(j, "\"before\"", "folds"), f1 = jint(j, "\"after\"", "folds");
  long k0 = jint(j, "\"before\"", "kinks"), k1 = jint(j, "\"after\"", "kinks");
  long np = jint(j, "\"after\"", "points");
  assert(f1 <= f0 && k1 <= k0 + 2);
  assert(np >= (long)(SW * SH * 9 / 10));
  free(j);
  r3d_tifxyz after;
  assert(!r3d_tifxyz_load(&after, out));
  double d1 = off_sheet(&after);
  printf("off-sheet distance: %.2f -> %.2f vox (folds %ld -> %ld, kinks %ld -> %ld)\n", d0, d1, f0,
         f1, k0, k1);
  assert(d1 < 0.6 * d0);
  assert(after.nvalid >= before.nvalid * 9 / 10);
  /* the refined copy is a full surface artifact: .sfc sibling + sidecar */
  assert(!r3d_surf_sibling(out, out_sfc, sizeof out_sfc));
  assert(r3d_surf_kind_of(out_sfc) == R3D_SURF_SFC);
  struct stat st;
  snprintf(rep, sizeof rep, "%s/tracer.json", out);
  assert(!stat(rep, &st));
  /* the source is untouched */
  assert(!r3d_tifxyz_load(&before, src) && fabs(off_sheet(&before) - d0) < 1e-6);
  /* an .sfc input takes the same path */
  char sfc_in[700], out2[600];
  assert(!r3d_surf_resolve(src, 0.1, NULL, NULL, sfc_in, sizeof sfc_in));
  snprintf(out2, sizeof out2, "%s/refined2", tmp);
  snprintf(cmd, sizeof cmd,
           "'%s' --pred '%s' --in '%s' --out '%s' --level 0 --no-ctsnap --threads 4 >/dev/null",
           cli, root, sfc_in, out2);
  assert(system(cmd) == 0);
  r3d_tifxyz after2;
  assert(!r3d_tifxyz_load(&after2, out2));
  assert(off_sheet(&after2) < 0.6 * d0);
  r3d_tifxyz_free(&before);
  r3d_tifxyz_free(&after);
  r3d_tifxyz_free(&after2);
  if (argc > 2) { /* the viewer's editing path, headless: import the active
                   * surface, apply a scripted drag as an anchor + re-solve,
                   * then as a forced reopen-and-regrow */
    char app[4096];
    assert(realpath(argv[2], app));
    /* grid point (10,5) of the perturbed surface and its spot on the wrap */
    static const char *const drags[2] = {"158.1,174.0,110>159.8,176.5,110",
                                         "158.1,174.0,110>159.8,176.5,110!"};
    for (int m = 0; m < 2; m++) {
      snprintf(cmd, sizeof cmd,
               "R3D_BEND_RMIN=12 R3D_EDIT_TEST=20 R3D_DRAG_TEST='%s' '%s' --bricks "
               "'%s/manifest.json' --overlay '%s' --multiview '%s' --headless --frames %d 2>&1",
               drags[m], app, root, root, src, m ? 1200 : 400);
      FILE *pf = popen(cmd, "r");
      assert(pf);
      bool edited = false, applied = false, landed = false, reopened = false;
      char line[1024];
      while (fgets(line, sizeof line, pf)) {
        if (strstr(line, "tracer: editing ")) edited = true;
        if (strstr(line, "correction drag") &&
            strstr(line, m ? "reopen + regrow" : "anchor + re-solve"))
          applied = true;
        if (strstr(line, "tracer: reopened ")) reopened = true;
        double off = 0;
        const char *os = strstr(line, "anchor 0 off-sheet ");
        if (os && sscanf(os, "anchor 0 off-sheet %lf", &off) == 1 && off < 1.0) landed = true;
      }
      int prc = pclose(pf);
      printf("gui %s: edited %d applied %d reopened %d landed %d rc %d\n",
             m ? "regrow" : "re-solve", edited, applied, reopened, landed, prc);
      assert(prc == 0 && edited && applied && landed && (m == 0 || reopened));
    }
  }
  if (getenv("R3D_KEEP_FIXTURE")) { /* leave the tree + surface for manual runs */
    printf("fixture kept at %s\n", tmp);
  } else {
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", tmp);
    assert(system(cmd) == 0);
  }
  puts("surfrefine passed: qc-only, refine toward the prediction sheet, .sfc input");
  return 0;
}
