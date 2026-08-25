"""Project 2.5D ink rasters into volumetric training labels.

Every traced segment on the open-data bucket has, per CT volume it was
registered to, a tifxyz grid (x/y/z.tif, one 3D position per grid cell) and
an ink-detection raster (2.5D model output on the flattened surface;
raster pixel (u,v) = grid cell (u*scale, v*scale) -- verified against a
rendseg re-render of a PHerc Paris 4 segment: identity orientation, zero
offset). Splatting the raster along the surface normal gives the
"surface-conditioned" volume labels the published 3D teacher was trained
on, for any scroll and any resolution.

Per canonical 128^3 block (see corpus.py grids F=2.4um / C=7.9um):
  * grid cells whose bbox (+band) meets the block are bilinearly sampled at
    ~1 canonical voxel spacing (positions, per-cell normal, raster prob);
  * ink band  |k| <= K  : voxel p + k*n <- prob, gaussian weight in k;
  * bg band   K < |k| <= B, prob < 0.1 : voxel <- 0 with a lower weight
    (tells the model "this side of the sheet is not ink");
  * blocks are chosen by raster ink coverage (plus a share of background)
    up to --max-blocks per (segment, volume, grid), so the on-disk cache
    stays within budget.
Output: <root>/<sample>/<grid>/<vol>_<bz>_<by>_<bx>.npz with
  ct u8 128^3, ink u8 (0..255 soft prob), weight u8 (0 = ignore),
  src u8 (1 = raster25d), meta json (sample, vol, grid, origin, segments).
Existing blocks are merged (weighted average of ink, max weight).

CLI:
  project_raster.py project --sample PHercParis4 --seg 20231106155351 --vol 20260411134726 --grid F
  project_raster.py project-all --sample PHercParis4 [--grid F,C] [--vols ...]
  project_raster.py stats [--sample X]
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

BLOCK = corpus.BLOCK
SRC_RASTER25D = 1
# bands in micrometres (converted to canonical voxels per grid)
# The published teacher's ink probability is spread over ~+-20 voxels (2.4um)
# around the traced surface (whole sheet thickness), so the ink band covers
# +-24um with a gaussian taper; a separate background band beyond it would
# reach the neighbouring wrap (~48um pitch), so negatives come only from
# raster < BG_THRESH inside the same band (same sheet, no ink).
INK_BAND_UM = 24.0     # +-K around the surface carries the raster value
BG_BAND_UM = 24.0      # K < |k| <= B, raster < BG_THRESH -> background (disabled: == INK)
BG_THRESH = 0.10
W_INK, W_BG = 200, 120


# ------------------------------------------------------------ downloads
def _get(url, dst, retries=4):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as o:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    o.write(b)
            os.replace(tmp, dst)
            return dst
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"project: retry {url}: {e}")
            time.sleep(2 * (i + 1))


def seg_cache_dir(root, seg, vid):
    return Path(root) / "segments" / seg["id"] / vid


def load_tifxyz(root, seg, vid):
    """(zyx float32 [h,w,3], scale) for segment `seg` registered on `vid`."""
    import tifffile
    d = seg_cache_dir(root, seg, vid)
    base = seg["per_vol"][vid]["tifxyz"]
    for n in ("x.tif", "y.tif", "z.tif", "meta.json"):
        _get(f"{base}/{n}", d / n)
    meta = json.load(open(d / "meta.json"))
    sc = meta["scale"]
    # stored as x/y/z; keep (z, y, x) to match volume indexing
    xyz = np.stack([tifffile.imread(d / n).astype(np.float32)
                    for n in ("z.tif", "y.tif", "x.tif")], axis=-1)
    # vc3d convention: unmapped = 0s -> invalid (as src/core/tifxyz.c)
    bad = ~np.isfinite(xyz).all(-1) | (xyz[..., 0] <= 0) | (xyz[..., 2] == -1)
    xyz[bad] = -1.0
    return xyz, (float(sc[0]), float(sc[1]))


def raster_ds8_url(url):
    d, n = url.rsplit("/", 1)
    return f"{d}/downsampled/{n[:-4]}-ds8.jpg"


def load_raster(root, seg, vid, full=False):
    """float32 [0,1] raster and its downsample factor vs full-res pixels."""
    d = seg_cache_dir(root, seg, vid)
    url = seg["per_vol"][vid]["raster"]
    if not full:
        try:
            f = _get(raster_ds8_url(url), d / "ink-ds8.jpg")
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            a = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
            return a, 8.0
        except Exception as e:  # noqa: BLE001
            print(f"project: ds8 raster unavailable ({e}), using full tif")
    import tifffile
    f = _get(url, d / "ink.tif")
    a = tifffile.imread(f)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        a = a / (a.max() if a.max() > 1 else 1.0)
    else:
        a = a.astype(np.float32) / 255.0
    return a, 1.0


# ------------------------------------------------------------ geometry
class SegGeom:
    """Grid cells of one segment in canonical-grid coordinates."""

    def __init__(self, xyz, scale, vol, grid):
        g = vol["grids"][grid]
        self.vox2can = 1.0 / (g["f"] * 2 ** g["level"])   # volume vox -> canonical
        self.xyz = xyz
        self.scale = scale
        self.h, self.w = xyz.shape[:2]
        v = xyz[..., 2] != -1
        self.cell_ok = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]
        c = xyz * self.vox2can
        p00, p10, p01, p11 = c[:-1, :-1], c[:-1, 1:], c[1:, :-1], c[1:, 1:]
        self.p00, self.p10, self.p01, self.p11 = p00, p10, p01, p11
        stack = np.stack([p00, p10, p01, p11])
        self.cmin = stack.min(0)
        self.cmax = stack.max(0)
        du = p10 - p00
        dv = p01 - p00
        n = np.cross(du, dv)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        self.n = n / np.maximum(nn, 1e-9)
        span = np.linalg.norm(du, axis=-1)[self.cell_ok]
        self.cell_span = float(np.median(span)) if span.size else 1.0
        self.up = max(2, int(math.ceil(self.cell_span)))  # samples per cell edge

    def cells_for_block(self, origin, pad):
        lo = np.asarray(origin, np.float32) - pad
        hi = lo + BLOCK + 2 * pad
        m = self.cell_ok & (self.cmax >= lo).all(-1) & (self.cmin <= hi).all(-1)
        return np.nonzero(m)

    def samples(self, jj, ii):
        """Bilinear samples for cells (jj, ii): positions [N,3] canonical,
        normals [N,3], raster coords (u,v) in full-res pixels [N]."""
        up = self.up
        t = (np.arange(up, dtype=np.float32) + 0.5) / up
        fu, fv = np.meshgrid(t, t, indexing="xy")         # [up, up]
        fu = fu.reshape(1, -1)
        fv = fv.reshape(1, -1)
        p00, p10 = self.p00[jj, ii][:, None], self.p10[jj, ii][:, None]
        p01, p11 = self.p01[jj, ii][:, None], self.p11[jj, ii][:, None]
        w00 = ((1 - fu) * (1 - fv))[..., None]
        w10 = (fu * (1 - fv))[..., None]
        w01 = ((1 - fu) * fv)[..., None]
        w11 = (fu * fv)[..., None]
        p = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11    # [C, up*up, 3]
        n = np.repeat(self.n[jj, ii][:, None], up * up, axis=1)
        u = (ii[:, None] + fu) / self.scale[0]
        v = (jj[:, None] + fv) / self.scale[1]
        return p.reshape(-1, 3), n.reshape(-1, 3), u.reshape(-1), v.reshape(-1)


def block_index_grid(geom, raster, rfac):
    """Blocks touched by the segment with a raster-ink score each:
    {(bz,by,bx): (ink_mean, n_cells)}."""
    jj, ii = np.nonzero(geom.cell_ok)
    # raster value at cell centres
    u = ((ii + 0.5) / geom.scale[0] / rfac).astype(np.int64)
    v = ((jj + 0.5) / geom.scale[1] / rfac).astype(np.int64)
    u = np.clip(u, 0, raster.shape[1] - 1)
    v = np.clip(v, 0, raster.shape[0] - 1)
    r = raster[v, u]
    ctr = 0.5 * (geom.cmin[jj, ii] + geom.cmax[jj, ii])
    b = np.floor(ctr / BLOCK).astype(np.int64)
    key = (b[:, 0] << 42) | (b[:, 1] << 21) | b[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    s = np.bincount(inv, weights=r) / cnt
    out = {}
    for k, sc, n in zip(uk.tolist(), s.tolist(), cnt.tolist()):
        out[(k >> 42, (k >> 21) & 0x1FFFFF, k & 0x1FFFFF)] = (sc, n)
    return out


def choose_blocks(blocks, max_blocks, rng, bg_share=0.35, min_cells=3):
    """Prefer blocks with MIXED ink/background (rank by s*(1-s), so a block
    entirely inside a letter stroke ranks like one with no ink), plus a
    random background share -- otherwise the surface band is ~80% ink and
    the model learns 'band = ink'."""
    keys = [k for k, (s, n) in blocks.items() if n >= min_cells]
    if max_blocks <= 0 or len(keys) <= max_blocks:
        return keys
    sc = np.array([blocks[k][0] for k in keys])
    mixed = sc * (1.0 - sc) + 0.02 * sc
    n_ink = int(max_blocks * (1 - bg_share))
    order = np.argsort(-mixed)
    ink_keys = [keys[i] for i in order[:n_ink]]
    rest = order[n_ink:]
    n_bg = min(max_blocks - n_ink, len(rest))
    bg_keys = [keys[i] for i in rng.choice(rest, n_bg, replace=False)] if n_bg else []
    return ink_keys + bg_keys


# ---------------------------------------------------------------- splat
def splat_block(geom, raster, rfac, origin, grid_um):
    """(ink u8, weight u8) for one block, or None if no surface samples."""
    K = max(1, int(round(INK_BAND_UM / grid_um)))
    B = max(K + 1, int(round(BG_BAND_UM / grid_um)))
    sig = max(0.75, K / 2.0)
    jj, ii = geom.cells_for_block(origin, pad=B + 1)
    if jj.size == 0:
        return None
    p, n, u, v = geom.samples(jj, ii)
    ui = np.clip((u / rfac).astype(np.int64), 0, raster.shape[1] - 1)
    vi = np.clip((v / rfac).astype(np.int64), 0, raster.shape[0] - 1)
    prob = raster[vi, ui]
    o = np.asarray(origin, np.float32)
    acc_w = np.zeros(BLOCK ** 3, np.float32)
    acc_wp = np.zeros(BLOCK ** 3, np.float32)
    acc_bg = np.zeros(BLOCK ** 3, np.float32)
    bgmask = prob < BG_THRESH
    for k in range(-B, B + 1):
        q = p + n * float(k) - o
        idx = np.floor(q + 0.5).astype(np.int64)
        inside = ((idx >= 0) & (idx < BLOCK)).all(-1)
        if abs(k) <= K:
            w = math.exp(-(k * k) / (2 * sig * sig))
            sel = inside
            flat = (idx[sel, 0] * BLOCK + idx[sel, 1]) * BLOCK + idx[sel, 2]
            acc_w += np.bincount(flat, minlength=BLOCK ** 3) * w
            acc_wp += np.bincount(flat, weights=prob[sel] * w, minlength=BLOCK ** 3)
        else:
            sel = inside & bgmask
            if not sel.any():
                continue
            flat = (idx[sel, 0] * BLOCK + idx[sel, 1]) * BLOCK + idx[sel, 2]
            acc_bg += np.bincount(flat, minlength=BLOCK ** 3)
    ink_hit = acc_w > 0.2
    if not ink_hit.any() and not (acc_bg > 0).any():
        return None
    ink = np.zeros(BLOCK ** 3, np.float32)
    ink[ink_hit] = acc_wp[ink_hit] / acc_w[ink_hit]
    weight = np.zeros(BLOCK ** 3, np.float32)
    weight[ink_hit] = W_INK * np.minimum(1.0, acc_w[ink_hit])
    bg_only = (~ink_hit) & (acc_bg > 0)
    weight[bg_only] = W_BG * np.minimum(1.0, acc_bg[bg_only])
    ink_u8 = np.clip(ink * 255 + 0.5, 0, 255).astype(np.uint8).reshape((BLOCK,) * 3)
    w_u8 = np.clip(weight + 0.5, 0, 255).astype(np.uint8).reshape((BLOCK,) * 3)
    return ink_u8, w_u8


# ------------------------------------------------------------- storage
def block_path(root, sample, grid, vid, b):
    return Path(root) / sample / grid / f"{vid}_{b[0]}_{b[1]}_{b[2]}.npz"


_SAVE_LOCK = threading.Lock()


def save_block(path, ct, ink, weight, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    with corpus.block_lock(path):
        if path.exists():
            old = np.load(path, allow_pickle=False)
            ow, oi = old["weight"].astype(np.float32), old["ink"].astype(np.float32)
            nw = weight.astype(np.float32)
            tot = ow + nw
            m = tot > 0
            ink = ink.astype(np.float32)
            ink[m] = (ow[m] * oi[m] + nw[m] * ink[m]) / tot[m]
            ink = np.clip(ink + 0.5, 0, 255).astype(np.uint8)
            weight = np.maximum(old["weight"], weight)
            om = json.loads(str(old["meta"]))
            meta["segments"] = sorted(set(om.get("segments", [])) | set(meta["segments"]))
            if old["ct"].max() > 0:
                ct = old["ct"]
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ct=ct, ink=ink, weight=weight,
                            src=np.uint8(SRC_RASTER25D), meta=json.dumps(meta))
        os.replace(tmp, path)


# ------------------------------------------------------------------ CLI
def project_pair(man, root, seg, vol, grid, max_blocks, workers, rng, full_raster=False,
                 dry=False):
    vid = vol["id"]
    t0 = time.time()
    xyz, scale = load_tifxyz(root, seg, vid)
    raster, rfac = load_raster(root, seg, vid, full=full_raster)
    geom = SegGeom(xyz, scale, vol, grid)
    blocks = block_index_grid(geom, raster, rfac)
    chosen = choose_blocks(blocks, max_blocks, rng)
    grid_um = corpus.GRIDS[grid]
    print(f"project: {seg['sample']}/{seg['id']} on {vid} grid {grid}: grid {geom.w}x{geom.h} "
          f"cells ok {int(geom.cell_ok.sum())} span {geom.cell_span:.1f} vox up {geom.up} "
          f"raster {raster.shape} blocks touched {len(blocks)} -> {len(chosen)} "
          f"(prep {time.time() - t0:.1f}s)", flush=True)
    if dry:
        return 0
    done = 0
    skipped = 0
    lock = threading.Lock()

    def work(b):
        nonlocal done, skipped
        path = block_path(root, seg["sample"], grid, vid, b)
        origin = tuple(int(x * BLOCK) for x in b)
        if path.exists():
            old = np.load(path, allow_pickle=False)
            om = json.loads(str(old["meta"]))
            if seg["id"] in om.get("segments", []):
                with lock:
                    skipped += 1
                return
        r = splat_block(geom, raster, rfac, origin, grid_um)
        if r is None:
            with lock:
                skipped += 1
            return
        ink, weight = r
        ct = corpus.read_block(vol, grid, origin)
        if ct.max() == 0:
            with lock:
                skipped += 1
            return
        weight[ct == 0] = 0
        meta = {"sample": seg["sample"], "vol": vid, "grid": grid, "origin": origin,
                "segments": [seg["id"]], "um": vol["um"], "keV": vol["keV"]}
        save_block(path, ct, ink, weight, meta)
        with lock:
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(chosen)} blocks ({time.time() - t0:.0f}s)", flush=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, chosen))
    print(f"project: wrote {done}, skipped {skipped} ({time.time() - t0:.0f}s)", flush=True)
    return done


def _pairs(man, args):
    segs = man["segments"]
    vols = man["volumes"]
    out = []
    for s in segs.values():
        if args.sample and s["sample"] != args.sample:
            continue
        if getattr(args, "seg", None) and s["id"] != args.seg:
            continue
        for vid, pv in s["per_vol"].items():
            if "tifxyz" not in pv or "raster" not in pv or vid not in vols:
                continue
            if getattr(args, "vol", None) and args.vol not in vid:
                continue
            if getattr(args, "vols", None) and not any(v in vid for v in args.vols.split(",")):
                continue
            v = vols[vid]
            if "shape" not in v:
                continue
            for g in args.grid.split(","):
                if g in v["grids"]:
                    out.append((s, v, g))
    return out


def cmd_project(args):
    man = corpus.load_manifest(args.root)
    pairs = _pairs(man, args)
    if not pairs:
        sys.exit("project: no (segment, volume, grid) matches")
    rng = np.random.default_rng(args.seed)
    for s, v, g in pairs:
        mb = args.max_blocks if g == "F" else args.max_blocks_c
        project_pair(man, args.root, s, v, g, mb, args.workers, rng,
                     full_raster=args.full_raster, dry=args.dry)


def cmd_stats(args):
    root = Path(args.root)
    tot = 0
    for sd in sorted(root.iterdir()):
        if not sd.is_dir() or sd.name in ("segments", "occ"):
            continue
        if args.sample and sd.name != args.sample:
            continue
        for gd in sorted(sd.iterdir()):
            fs = list(gd.glob("*.npz"))
            sz = sum(f.stat().st_size for f in fs)
            tot += sz
            print(f"{sd.name}/{gd.name}: {len(fs)} blocks, {sz / 1e9:.2f} GB")
    print(f"total {tot / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("project", "project-all"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
        p.add_argument("--sample", required=(name == "project-all"))
        if name == "project":
            p.add_argument("--seg", required=True)
            p.add_argument("--vol", required=True)
        else:
            p.add_argument("--vols", help="comma list of volume id substrings")
        p.add_argument("--grid", default="F,C")
        p.add_argument("--max-blocks", type=int, default=400, help="per pair on F")
        p.add_argument("--max-blocks-c", type=int, default=150, help="per pair on C")
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--full-raster", action="store_true")
        p.add_argument("--dry", action="store_true", help="only report block counts")
        p.set_defaults(fn=cmd_project)
    s = sub.add_parser("stats")
    s.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
    s.add_argument("--sample")
    s.set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
