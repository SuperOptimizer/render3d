#include "core/labelvol.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <label.h> /* c5d C5L1 label-brick codec (angle include: c5d src dir) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define LB R3D_LBL_BRICK
#define LB3 ((size_t)LB * LB * LB)

/* on-disk brick files are trusted decoder input; cap the read so a
 * truncated/corrupt/hostile file can't drive an unbounded allocation */
#define LBL_MAX_BRICK_FILE (64u * 1024u * 1024u)
#define LBL_MANIFEST_MAX 4096u

const char *const r3d_lbl_class_name[R3D_LBL_NCLASS] = {
    "erase", "papyrus", "ink", "not ink", "recto", "verso", "background", "damage", "other"};
const float r3d_lbl_class_rgb[R3D_LBL_NCLASS][3] = {
    {0.50f, 0.50f, 0.50f}, /* 0 erase (swatch only; never drawn) */
    {0.82f, 0.66f, 0.43f}, /* 1 papyrus */
    {0.95f, 0.20f, 0.85f}, /* 2 ink */
    {0.20f, 0.80f, 0.80f}, /* 3 not ink */
    {0.30f, 0.90f, 0.30f}, /* 4 recto */
    {1.00f, 0.55f, 0.15f}, /* 5 verso */
    {0.35f, 0.45f, 0.90f}, /* 6 background */
    {0.95f, 0.90f, 0.20f}, /* 7 damage */
    {0.90f, 0.90f, 0.90f}, /* 8 other */
};

static size_t lbl_nbr(const r3d_labelvol *lv) {
  return (size_t)lv->lnb[0][0] * lv->lnb[0][1] * lv->lnb[0][2];
}

static size_t lbl_bidx(const r3d_labelvol *lv, uint32_t level, uint32_t bx, uint32_t by,
                       uint32_t bz) {
  return ((size_t)bz * lv->lnb[level][1] + by) * lv->lnb[level][0] + bx;
}

int r3d_labelvol_init(r3d_labelvol *lv, const uint32_t dim[3]) {
  memset(lv, 0, sizeof *lv);
  for (int a = 0; a < 3; a++) {
    if (!dim[a]) return -1;
    lv->dim[a] = dim[a];
  }
  /* level grids mirror the renderer's brick LODs: each level halves the
   * voxel resolution, bricks stay 128^3 */
  uint32_t d[3] = {dim[0], dim[1], dim[2]};
  for (uint32_t l = 0; l < R3D_LBL_MAXLEV; l++) {
    for (int a = 0; a < 3; a++) lv->lnb[l][a] = (d[a] + LB - 1u) / LB;
    lv->nlev = l + 1u;
    size_t n = (size_t)lv->lnb[l][0] * lv->lnb[l][1] * lv->lnb[l][2];
    lv->gens[l] = calloc(n, sizeof(uint32_t));
    if (!lv->gens[l]) goto fail;
    if (lv->lnb[l][0] == 1u && lv->lnb[l][1] == 1u && lv->lnb[l][2] == 1u) break;
    for (int a = 0; a < 3; a++) d[a] = (d[a] + 1u) / 2u;
  }
  size_t nb = lbl_nbr(lv);
  lv->data = calloc(nb, sizeof(uint8_t *));
  lv->saved = calloc(nb, sizeof(uint32_t));
  if (!lv->data || !lv->saved) goto fail;
  return 0;
fail:
  r3d_labelvol_free(lv);
  return -1;
}

void r3d_labelvol_free(r3d_labelvol *lv) {
  if (lv->data) {
    size_t nb = lbl_nbr(lv);
    for (size_t i = 0; i < nb; i++) free(lv->data[i]);
  }
  free(lv->data);
  free(lv->saved);
  for (uint32_t l = 0; l < R3D_LBL_MAXLEV; l++) free(lv->gens[l]);
  memset(lv, 0, sizeof *lv);
}

static void lbl_bump(r3d_labelvol *lv, uint32_t bx, uint32_t by, uint32_t bz) {
  for (uint32_t l = 0; l < lv->nlev; l++)
    lv->gens[l][lbl_bidx(lv, l, bx >> l, by >> l, bz >> l)]++;
}

uint64_t r3d_labelvol_paint(r3d_labelvol *lv, const double p[3], double radius, uint8_t cls) {
  if (!lv->data || cls >= R3D_LBL_NCLASS) return 0;
  if (radius < 0.0) radius = 0.0;
  double r2 = radius * radius;
  int64_t lo[3], hi[3];
  for (int a = 0; a < 3; a++) {
    lo[a] = (int64_t)ceil(p[a] - radius);
    hi[a] = (int64_t)floor(p[a] + radius);
    if (lo[a] < 0) lo[a] = 0;
    if (hi[a] > (int64_t)lv->dim[a] - 1) hi[a] = (int64_t)lv->dim[a] - 1;
    if (lo[a] > hi[a]) return 0;
  }
  uint64_t changed = 0;
  for (int64_t bz = lo[2] >> 7; bz <= hi[2] >> 7; bz++)
    for (int64_t by = lo[1] >> 7; by <= hi[1] >> 7; by++)
      for (int64_t bx = lo[0] >> 7; bx <= hi[0] >> 7; bx++) {
        size_t bi = lbl_bidx(lv, 0, (uint32_t)bx, (uint32_t)by, (uint32_t)bz);
        uint8_t *d = lv->data[bi];
        int64_t x0 = bx << 7, y0 = by << 7, z0 = bz << 7;
        int64_t zl = lo[2] > z0 ? lo[2] - z0 : 0, zh = hi[2] - z0 > 127 ? 127 : hi[2] - z0;
        int64_t yl = lo[1] > y0 ? lo[1] - y0 : 0, yh = hi[1] - y0 > 127 ? 127 : hi[1] - y0;
        int64_t xl = lo[0] > x0 ? lo[0] - x0 : 0, xh = hi[0] - x0 > 127 ? 127 : hi[0] - x0;
        bool touched = false;
        for (int64_t z = zl; z <= zh; z++) {
          double dz = (double)(z0 + z) - p[2];
          for (int64_t y = yl; y <= yh; y++) {
            double dy = (double)(y0 + y) - p[1];
            double dyz = dy * dy + dz * dz;
            if (dyz > r2) continue;
            for (int64_t x = xl; x <= xh; x++) {
              double dx = (double)(x0 + x) - p[0];
              if (dx * dx + dyz > r2) continue;
              size_t o = (size_t)((z * LB + y) * LB + x);
              uint8_t old = d ? d[o] : 0;
              if (old == cls) continue;
              if (!d) {
                d = calloc(1, LB3);
                if (!d) return changed;
                lv->data[bi] = d;
              }
              d[o] = cls;
              if (old) lv->nvox[old]--;
              if (cls) lv->nvox[cls]++;
              changed++;
              touched = true;
            }
          }
        }
        if (touched) lbl_bump(lv, (uint32_t)bx, (uint32_t)by, (uint32_t)bz);
      }
  if (changed) lv->edits++;
  return changed;
}

uint32_t r3d_labelvol_gen(const r3d_labelvol *lv, uint32_t level, uint32_t bx, uint32_t by,
                          uint32_t bz) {
  if (!lv->data) return 0;
  if (level >= lv->nlev) { /* coarser than the pyramid: fold into the top */
    uint32_t up = level - (lv->nlev - 1u);
    level = lv->nlev - 1u;
    bx >>= up;
    by >>= up;
    bz >>= up;
  }
  if (bx >= lv->lnb[level][0] || by >= lv->lnb[level][1] || bz >= lv->lnb[level][2]) return 0;
  return lv->gens[level][lbl_bidx(lv, level, bx, by, bz)];
}

void r3d_labelvol_fetch(const r3d_labelvol *lv, uint32_t level, uint32_t bx, uint32_t by,
                        uint32_t bz, uint8_t *out) {
  if (!lv->data) {
    memset(out, 0, LB3);
    return;
  }
  if (level == 0) {
    if (bx < lv->lnb[0][0] && by < lv->lnb[0][1] && bz < lv->lnb[0][2]) {
      const uint8_t *d = lv->data[lbl_bidx(lv, 0, bx, by, bz)];
      if (d) memcpy(out, d, LB3);
      else memset(out, 0, LB3);
    } else
      memset(out, 0, LB3);
    return;
  }
  /* coarse LOD brick: stride-sample the paint grid (thin painted sheets can
   * drop voxels at coarse zoom; labelling itself happens at level 0) */
  const uint8_t *cache = NULL;
  size_t cache_bi = SIZE_MAX;
  size_t o = 0;
  for (uint32_t oz = 0; oz < LB; oz++) {
    uint64_t wz = ((uint64_t)bz * LB + oz) << level;
    for (uint32_t oy = 0; oy < LB; oy++) {
      uint64_t wy = ((uint64_t)by * LB + oy) << level;
      for (uint32_t ox = 0; ox < LB; ox++, o++) {
        uint64_t wx = ((uint64_t)bx * LB + ox) << level;
        uint8_t v = 0;
        if (wx < lv->dim[0] && wy < lv->dim[1] && wz < lv->dim[2]) {
          size_t bi = lbl_bidx(lv, 0, (uint32_t)(wx >> 7), (uint32_t)(wy >> 7),
                               (uint32_t)(wz >> 7));
          if (bi != cache_bi) {
            cache = lv->data[bi];
            cache_bi = bi;
          }
          if (cache)
            v = cache[(size_t)(((wz & 127u) * LB + (wy & 127u)) * LB + (wx & 127u))];
        }
        out[o] = v;
      }
    }
  }
}

uint32_t r3d_labelvol_dirty(const r3d_labelvol *lv) {
  if (!lv->data) return 0;
  size_t nb = lbl_nbr(lv);
  uint32_t n = 0;
  for (size_t i = 0; i < nb; i++)
    if (lv->gens[0][i] != lv->saved[i]) n++;
  return n;
}

static void lbl_brick_path(char *out, size_t cap, const char *dir, uint32_t bx, uint32_t by,
                           uint32_t bz) {
  snprintf(out, cap, "%s/b_%u_%u_%u.c5l", dir, bx, by, bz);
}

/* Best-effort: persist a just-completed rename against a crash. Failure is
 * not fatal to the caller (the file content itself is already durable via
 * fclose; this only orders the directory entry after it). */
static void lbl_fsync_dir(const char *dir) {
  int fd = open(dir, O_RDONLY);
  if (fd < 0) return;
  (void)fsync(fd);
  (void)close(fd);
}

/* Parse manifest.json (fixed layout: this file is the only writer) and
 * confirm it describes `dim`. Refuses to load on any parse/version/size/
 * dimension mismatch so a load can never silently apply the wrong
 * dataset's bricks onto live storage. */
static int lbl_manifest_check(const char *dir, const uint32_t dim[3]) {
  char mp[1400];
  snprintf(mp, sizeof mp, "%s/manifest.json", dir);
  FILE *f = fopen(mp, "r");
  if (!f) return -1;
  char buf[LBL_MANIFEST_MAX];
  size_t n = fread(buf, 1, sizeof buf - 1u, f);
  bool clean_eof = feof(f) != 0; /* must fit the bound to be trustworthy */
  bool rd_err = ferror(f) != 0;
  fclose(f);
  if (rd_err || !clean_eof || n == 0) return -1;
  buf[n] = '\0';
  char magic[32] = {0};
  uint32_t version = 0, dx = 0, dy = 0, dz = 0, brick = 0;
  if (sscanf(buf,
             "{ \"magic\": \"%31[^\"]\", \"version\": %u, \"dim\": [%u, %u, %u], "
             "\"brick\": %u",
             magic, &version, &dx, &dy, &dz, &brick) != 6)
    return -1;
  if (strcmp(magic, "r3dlabels") != 0 || version != 1) return -1;
  if (brick != LB || dx != dim[0] || dy != dim[1] || dz != dim[2]) return -1;
  return 0;
}

int r3d_labelvol_save(r3d_labelvol *lv, const char *dir) {
  if (!lv->data) return -1;
  if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "labels: mkdir %s: %s\n", dir, strerror(errno));
    return -1;
  }
  c5d_label_params prm = c5d_label_defaults();
  prm.nthreads = 0; /* all cores: ~14 ms per brick */
  uint32_t wrote = 0, dropped = 0, failed = 0;
  int rc = 0;
  for (uint32_t bz = 0; bz < lv->lnb[0][2]; bz++)
    for (uint32_t by = 0; by < lv->lnb[0][1]; by++)
      for (uint32_t bx = 0; bx < lv->lnb[0][0]; bx++) {
        size_t bi = lbl_bidx(lv, 0, bx, by, bz);
        uint32_t gen_before = lv->gens[0][bi];
        if (gen_before == lv->saved[bi]) continue;
        uint8_t *d = lv->data[bi];
        bool any = false;
        if (d)
          for (size_t i = 0; i < LB3 && !any; i++) any = d[i] != 0;
        char bp[1408];
        lbl_brick_path(bp, sizeof bp, dir, bx, by, bz);
        if (!any) { /* erased to empty: drop the file */
          int ur = unlink(bp);
          if (ur != 0 && errno != ENOENT) {
            fprintf(stderr, "labels: unlink %s: %s\n", bp, strerror(errno));
            rc = -1;
            failed++;
            continue; /* leave dirty: retry on next save */
          }
          if (ur == 0) dropped++;
          /* only mark clean if nothing painted over this brick while we
           * were removing its file */
          if (gen_before == lv->gens[0][bi]) lv->saved[bi] = gen_before;
          continue;
        }
        c5d_label_channel ch = {C5D_LABEL_U8, C5D_LABEL_NO_MASK, d};
        uint8_t *buf = NULL;
        size_t bn = 0;
        if (c5d_label_encode(&prm, &ch, 1, LB, &buf, &bn) != 0) {
          fprintf(stderr, "labels: encode failed for brick %u,%u,%u\n", bx, by, bz);
          rc = -1;
          failed++;
          continue; /* leave dirty */
        }
        char tp[1456];
        snprintf(tp, sizeof tp, "%s.tmp.%ld", bp, (long)getpid());
        FILE *f = fopen(tp, "wb");
        bool ok = f && fwrite(buf, 1, bn, f) == bn;
        if (f) ok = (fclose(f) == 0) && ok;
        free(buf);
        if (!ok || rename(tp, bp) != 0) {
          fprintf(stderr, "labels: write %s failed: %s\n", bp, strerror(errno));
          unlink(tp);
          rc = -1;
          failed++;
          continue; /* leave dirty */
        }
        lbl_fsync_dir(dir);
        /* only mark clean if nothing painted over this brick while we were
         * encoding/writing it */
        if (gen_before == lv->gens[0][bi]) lv->saved[bi] = gen_before;
        wrote++;
      }
  /* manifest is published last: it names the dataset the just-written
   * bricks belong to, so it must never become visible before they do */
  char mp[1400];
  snprintf(mp, sizeof mp, "%s/manifest.json", dir);
  char mtp[1456];
  snprintf(mtp, sizeof mtp, "%s.tmp.%ld", mp, (long)getpid());
  FILE *mf = fopen(mtp, "w");
  bool mok = mf != NULL;
  if (mok)
    mok = fprintf(mf,
                  "{\n  \"magic\": \"r3dlabels\", \"version\": 1,\n"
                  "  \"dim\": [%u, %u, %u], \"brick\": %u,\n  \"classes\": [",
                  lv->dim[0], lv->dim[1], lv->dim[2], LB) >= 0;
  for (uint32_t c = 0; mok && c < R3D_LBL_NCLASS; c++)
    mok = fprintf(mf, "%s\"%s\"", c ? ", " : "", c ? r3d_lbl_class_name[c] : "") >= 0;
  if (mok) mok = fprintf(mf, "]\n}\n") >= 0;
  if (mf) mok = (fclose(mf) == 0) && mok;
  if (!mok || rename(mtp, mp) != 0) {
    fprintf(stderr, "labels: write %s failed: %s\n", mp, strerror(errno));
    unlink(mtp);
    rc = -1;
  } else {
    lbl_fsync_dir(dir);
  }
  printf("labels: saved %u brick(s) to %s (%u emptied, %u failed)\n", wrote, dir, dropped,
         failed);
  return rc;
}

int r3d_labelvol_load(r3d_labelvol *lv, const char *dir) {
  if (!lv->data) return -1;
  if (lbl_manifest_check(dir, lv->dim) != 0) {
    fprintf(stderr, "labels: %s/manifest.json missing or does not match this volume\n", dir);
    return -1;
  }
  DIR *dp = opendir(dir);
  if (!dp) return -1;
  uint8_t *tmp = malloc(LB3); /* decode target: never the live brick directly */
  if (!tmp) {
    closedir(dp);
    return -1;
  }
  uint32_t loaded = 0, failed = 0;
  struct dirent *de;
  while ((de = readdir(dp)) != NULL) {
    uint32_t bx, by, bz;
    char tail = 0;
    if (sscanf(de->d_name, "b_%u_%u_%u.c5%c", &bx, &by, &bz, &tail) != 4 || tail != 'l')
      continue;
    if (bx >= lv->lnb[0][0] || by >= lv->lnb[0][1] || bz >= lv->lnb[0][2]) continue;
    char bp[1408];
    lbl_brick_path(bp, sizeof bp, dir, bx, by, bz);
    FILE *f = fopen(bp, "rb");
    if (!f) {
      fprintf(stderr, "labels: failed to open %s\n", bp);
      failed++;
      continue;
    }
    bool sizeok = fseek(f, 0, SEEK_END) == 0;
    long fn = sizeok ? ftell(f) : -1;
    sizeok = sizeok && fn > 0 && (uint64_t)fn <= LBL_MAX_BRICK_FILE;
    if (sizeok) sizeok = fseek(f, 0, SEEK_SET) == 0;
    uint8_t *buf = sizeok ? malloc((size_t)fn) : NULL;
    bool ok = buf != NULL && fread(buf, 1, (size_t)fn, f) == (size_t)fn;
    fclose(f);
    if (!sizeok)
      fprintf(stderr, "labels: %s: bad size or exceeds %u byte cap\n", bp, LBL_MAX_BRICK_FILE);
    uint32_t ddim = 0, nchan = 0;
    c5d_label_type ty[C5D_LABEL_MAX_CHANNELS];
    uint32_t mk[C5D_LABEL_MAX_CHANNELS];
    if (ok)
      ok = c5d_label_info(buf, (size_t)fn, &ddim, &nchan, ty, mk) == 0 && ddim == LB &&
           nchan >= 1u && ty[0] == C5D_LABEL_U8;
    if (ok) {
      c5d_label_channel ch = {C5D_LABEL_U8, mk[0], tmp};
      ok = c5d_label_decode(buf, (size_t)fn, LB, &ch, 1, 0) == 0;
    }
    free(buf);
    if (ok) {
      for (size_t i = 0; i < LB3; i++)
        if (tmp[i] >= R3D_LBL_NCLASS) tmp[i] = 0; /* foreign class ids -> unlabelled */
      /* validated: publish the decoded brick into live storage. Only past
       * this point can the live volume change. */
      size_t bi = lbl_bidx(lv, 0, bx, by, bz);
      uint8_t *d = lv->data[bi];
      if (!d) {
        d = calloc(1, LB3);
        if (d) lv->data[bi] = d;
      }
      if (d) {
        for (size_t i = 0; i < LB3; i++)
          if (d[i]) lv->nvox[d[i]]--; /* retire the old content's counts */
        memcpy(d, tmp, LB3);
        for (size_t i = 0; i < LB3; i++)
          if (d[i]) lv->nvox[d[i]]++;
        lbl_bump(lv, bx, by, bz);
        lv->saved[bi] = lv->gens[0][bi]; /* just loaded = clean */
        loaded++;
      } else {
        ok = false;
      }
    }
    if (!ok) {
      fprintf(stderr, "labels: failed to load %s\n", bp);
      failed++;
    }
  }
  free(tmp);
  closedir(dp);
  if (loaded) lv->edits++;
  printf("labels: loaded %u brick(s) from %s (%u failed)\n", loaded, dir, failed);
  return failed ? -1 : 0; /* an empty-but-valid dir loads zero bricks and succeeds */
}
