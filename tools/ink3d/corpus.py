"""Scroll-corpus manifest and canonical-grid chunk access.

The Vesuvius open-data bucket publishes a catalog (metadata.json at the bucket
root) listing every sample, CT volume (OME-zarr, 6 levels, u8) and traced
segment (tifxyz meshes re-registered onto each volume, per-volume 2.5D ink
rasters, per-volume surface-volume layer stacks). This module turns that into
a flat manifest and gives resolution-independent access to CT data on two
canonical voxel grids:

    F  2.4 um   fine   (volumes scanned at <= 3.3 um)
    C  7.9 um   coarse (volumes scanned at <= 9.5 um; fine volumes via their
                       pyramid, so 2.4 um data also trains the coarse scale)

A block of the canonical grid is read from the pyramid level whose voxel size
is nearest and trilinearly rescaled by the residual factor (1.129 um -> L1 =
2.258 um -> x0.94, 3.24 um -> C via L1 = 6.48 um -> x0.82, ...).

CLI:
  corpus.py discover [--out DIR]           fetch catalog, write manifest.json
  corpus.py show     [--sample X]          summarize volumes/segments/grids
  corpus.py occupancy --sample X --vol V --grid F|C
                                           CT occupancy per 128^3 canonical block
  corpus.py read --sample X --vol V --grid F --origin z,y,x --out f.npy   smoke
"""
import argparse
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
CATALOG_URL = BUCKET + "/metadata.json"
DEFAULT_ROOT = Path(os.environ.get("R3D_INK_CORPUS",
                                   os.path.expanduser("~/r3d-data/ink-corpus")))
GRIDS = {"F": 2.4, "C": 7.9}
GRID_MAX_UM = {"F": 3.3, "C": 9.5}
BLOCK = 128          # canonical-grid sample block (voxels)
NLEVELS = 6


# ------------------------------------------------------------------ catalog
def fetch_catalog(root=DEFAULT_ROOT, refresh=False):
    root.mkdir(parents=True, exist_ok=True)
    f = root / "metadata.json"
    if refresh or not f.exists():
        print(f"corpus: fetching {CATALOG_URL}")
        req = urllib.request.Request(CATALOG_URL,
                                     headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=120) as r, open(f, "wb") as o:
            o.write(r.read())
    return json.load(open(f))


def _origin_path(d):
    for o in d.get("origins") or []:
        p = o.get("path")
        if p:
            return p
    return None


_VOL_IN_NAME = re.compile(r"-on-(\d{14})-")
_VOL_IN_STACK = re.compile(r"volume-(\d{14})")


def _zarr_json(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read())
    except Exception:
        return None


def build_manifest(cat, probe=True):
    """Flatten the catalog into {volumes: {vid: ...}, segments: {sid: ...}}."""
    vols, segs = {}, {}
    for sid, s in cat["samples"].items():
        for vid, v in (s.get("volumes") or {}).items():
            p = v.get("properties") or {}
            um = p.get("pixel_size_um")
            path = None
            preds = {}
            for d in v.get("data") or []:
                t = d.get("type")
                if t == "ome-zarr" and path is None:
                    path = _origin_path(d)
                elif t and t.endswith("-zarr"):
                    preds.setdefault(t, []).append(_origin_path(d))
            if not path or um is None:
                continue
            row = {"id": vid, "sample": sid, "um": float(um),
                   "keV": p.get("energy_keV"), "url": BUCKET + "/" + path.rstrip("/"),
                   "predictions": preds, "grids": {}}
            for g, gum in GRIDS.items():
                if row["um"] <= GRID_MAX_UM[g]:
                    L = max(0, min(NLEVELS - 1, round(math.log2(gum / row["um"]))))
                    row["grids"][g] = {"level": L, "f": gum / (row["um"] * 2 ** L)}
            if probe:
                za = _zarr_json(row["url"] + "/0/.zarray")
                if za:
                    row["shape"] = za["shape"]
                    row["chunks"] = za["chunks"]
            vols[vid] = row
        for gid, g in (s.get("segments") or {}).items():
            p = g.get("properties") or {}
            row = {"id": gid, "sample": sid, "orig_vol": g.get("original_volume_id"),
                   "width": p.get("width"), "height": p.get("height"),
                   "per_vol": {}}
            for d in g.get("data") or []:
                t, path = d.get("type"), _origin_path(d)
                if not path:
                    continue
                url = BUCKET + "/" + path.rstrip("/")
                if t == "tifxyz":
                    row["per_vol"].setdefault(row["orig_vol"], {})["tifxyz"] = url
                elif t == "tifxyz-transformed":
                    m = _VOL_IN_NAME.search(path)
                    if m:
                        row["per_vol"].setdefault(m[1], {})["tifxyz"] = url
                elif t in ("ink-detection", "layers-zarr"):
                    m = _VOL_IN_STACK.search(path)
                    if m:
                        key = "raster" if t == "ink-detection" else "layers"
                        row["per_vol"].setdefault(m[1], {})[key] = url
            segs[gid] = row
    return {"bucket": BUCKET, "grids": GRIDS, "volumes": vols, "segments": segs}


def load_manifest(root=DEFAULT_ROOT):
    f = Path(root) / "manifest.json"
    if not f.exists():
        sys.exit(f"corpus: {f} missing - run `corpus.py discover` first")
    return json.load(open(f))


# --------------------------------------------------------------- access
_LEVEL_CACHE = {}


def open_level(url, level):
    key = (url, level)
    a = _LEVEL_CACHE.get(key)
    if a is None:
        import zarr
        if "://" in url:
            import fsspec
            a = zarr.open(fsspec.get_mapper(f"{url}/{level}"), mode="r")
        else:
            a = zarr.open(str(Path(url) / str(level)), mode="r")
        _LEVEL_CACHE[key] = a
    return a


def canonical_shape(vol, grid):
    g = vol["grids"][grid]
    return tuple(int(s // 2 ** g["level"] / g["f"]) for s in vol["shape"])


def read_block(vol, grid, origin, size=BLOCK):
    """u8 block of the canonical grid `grid` at canonical-voxel `origin`
    (z,y,x), zero-padded at the volume edge. Trilinear rescale from the
    nearest pyramid level; no rescale when the level matches exactly."""
    g = vol["grids"][grid]
    L, f = g["level"], g["f"]
    a = open_level(vol["url"], L)
    out = np.zeros((size,) * 3, np.uint8)
    if abs(f - 1.0) < 1e-3:
        sl = tuple(slice(o, o + size) for o in origin)
        blk = a[sl]
        out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
        return out
    from scipy.ndimage import zoom
    # canonical c <-> level-L p = c * f; read with a 1-voxel guard band
    p0 = [max(0, int(math.floor(o * f)) - 1) for o in origin]
    p1 = [min(int(math.ceil((o + size) * f)) + 1, s) for o, s in zip(origin, a.shape)]
    if any(b <= a_ for a_, b in zip(p0, p1)):
        return out
    src = a[tuple(slice(a_, b) for a_, b in zip(p0, p1))]
    if src.max() == 0:
        return out
    res = zoom(src.astype(np.float32), 1.0 / f, order=1, prefilter=False)
    # canonical index of res[0] is p0 / f
    off = [int(round(o - p / f)) for o, p in zip(origin, p0)]
    z0, y0, x0 = off
    sub = res[max(z0, 0):z0 + size, max(y0, 0):y0 + size, max(x0, 0):x0 + size]
    dz, dy, dx = (max(-z0, 0), max(-y0, 0), max(-x0, 0))
    out[dz:dz + sub.shape[0], dy:dy + sub.shape[1], dx:dx + sub.shape[2]] = \
        np.clip(sub + 0.5, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------ occupancy
def occupancy_path(root, vol, grid):
    return Path(root) / "occ" / f"{vol['sample']}-{vol['id']}-{grid}.npy"


def occupancy(vol, grid, root=DEFAULT_ROOT, force=False):
    """Fraction of non-zero CT per canonical BLOCK^3 block, plus mean
    intensity of the non-zero voxels, from the coarsest level whose voxel
    is <= one block. Cached as float16 (2, Zb, Yb, Xb)."""
    f = occupancy_path(root, vol, grid)
    if f.exists() and not force:
        return np.load(f)
    g = vol["grids"][grid]
    L0, fac = g["level"], g["f"]
    # block edge in level-L0 voxels = BLOCK * fac; go up while >= 2 voxels
    L = L0
    edge = BLOCK * fac
    while L + 1 < NLEVELS and edge / 2 >= 2.0:
        L += 1
        edge /= 2
    a = open_level(vol["url"], L)
    Z, Y, X = a.shape
    ei = max(1, int(round(edge)))       # integer block edge at level L
    nb = [int(math.ceil(s / ei)) for s in (Z, Y, X)]
    occ = np.zeros((2,) + tuple(nb), np.float32)
    print(f"corpus: occupancy {vol['sample']}/{vol['id']} grid {grid}: level {L} "
          f"{a.shape} block edge {edge:.2f}->{ei} -> {nb}", flush=True)
    slab = max(ei, 256 - 256 % ei)
    for z0 in range(0, Z, slab):
        s = a[z0:z0 + slab]
        for bz in range(int(math.ceil(s.shape[0] / ei))):
            zi = z0 // ei + bz
            blk = s[bz * ei:(bz + 1) * ei]
            nz, ny, nx = blk.shape
            py, px = (-ny) % ei, (-nx) % ei
            if py or px:
                blk = np.pad(blk, ((0, 0), (0, py), (0, px)))
            v = blk.reshape(nz, nb[1], ei, nb[2], ei).astype(np.float32)
            nzc = (v > 0).sum(axis=(0, 2, 4))
            occ[0, zi] = nzc / float(nz * ei * ei)
            occ[1, zi] = v.sum(axis=(0, 2, 4)) / np.maximum(nzc, 1)
        print(f"  z {z0 + s.shape[0]}/{Z}", end="\r", flush=True)
    print()
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, occ.astype(np.float16))
    return occ.astype(np.float16)


# ------------------------------------------------------------------- CLI
def cmd_discover(args):
    root = Path(args.out)
    cat = fetch_catalog(root, refresh=args.refresh)
    man = build_manifest(cat, probe=not args.no_probe)
    json.dump(man, open(root / "manifest.json", "w"), indent=1)
    nv = len(man["volumes"])
    ns = len(man["segments"])
    nr = sum(1 for s in man["segments"].values()
             for pv in s["per_vol"].values() if "raster" in pv and "tifxyz" in pv)
    print(f"corpus: {nv} volumes, {ns} segments, {nr} (segment, volume) "
          f"pairs with tifxyz+ink raster -> {root / 'manifest.json'}")


def cmd_show(args):
    man = load_manifest(args.root)
    vols = man["volumes"]
    segs = man["segments"]
    by_sample = {}
    for v in vols.values():
        by_sample.setdefault(v["sample"], []).append(v)
    for sid in sorted(by_sample):
        if args.sample and sid != args.sample:
            continue
        ss = [s for s in segs.values() if s["sample"] == sid]
        print(f"{sid}: {len(ss)} segments")
        for v in sorted(by_sample[sid], key=lambda v: v["um"]):
            nr = sum(1 for s in ss if "raster" in s["per_vol"].get(v["id"], {}))
            nt = sum(1 for s in ss if "tifxyz" in s["per_vol"].get(v["id"], {}))
            gs = ", ".join(f"{g}:L{d['level']}x{d['f']:.2f}" for g, d in v["grids"].items())
            print(f"  {v['id']} {v['um']:6.3f}um {v['keV'] or 0:5.1f}keV "
                  f"shape {v.get('shape')} grids [{gs}] tifxyz {nt} rasters {nr} "
                  f"preds {sorted(v['predictions'])}")


def _vol_arg(man, args):
    v = man["volumes"].get(args.vol)
    if v is None:
        cands = [v for v in man["volumes"].values()
                 if v["sample"] == args.sample and args.vol in v["id"]]
        if len(cands) != 1:
            sys.exit(f"corpus: volume {args.vol!r} not found for {args.sample}")
        v = cands[0]
    return v


def cmd_occupancy(args):
    man = load_manifest(args.root)
    v = _vol_arg(man, args)
    for g in (args.grid.split(",") if args.grid else sorted(v["grids"])):
        if g not in v["grids"]:
            print(f"corpus: {v['id']} has no grid {g}")
            continue
        occ = occupancy(v, g, args.root, force=args.force)
        o = occ[0].astype(np.float32)
        print(f"  {g}: {o.shape} blocks, {(o > 0.05).sum()} occupied (>5%), "
              f"{(o > 0.5).sum()} dense (>50%)")


def cmd_read(args):
    man = load_manifest(args.root)
    v = _vol_arg(man, args)
    origin = tuple(int(x) for x in args.origin.split(","))
    import time
    t = time.time()
    b = read_block(v, args.grid, origin, args.size)
    print(f"corpus: read {v['id']} {args.grid} @ {origin} size {args.size}: "
          f"min {b.min()} max {b.max()} mean {b.mean():.1f} in {time.time() - t:.1f}s")
    if args.out:
        np.save(args.out, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--out", default=str(DEFAULT_ROOT))
    d.add_argument("--refresh", action="store_true", help="re-download the catalog")
    d.add_argument("--no-probe", action="store_true", help="skip .zarray shape probes")
    d.set_defaults(fn=cmd_discover)
    s = sub.add_parser("show")
    s.add_argument("--root", default=str(DEFAULT_ROOT))
    s.add_argument("--sample")
    s.set_defaults(fn=cmd_show)
    o = sub.add_parser("occupancy")
    o.add_argument("--root", default=str(DEFAULT_ROOT))
    o.add_argument("--sample", required=True)
    o.add_argument("--vol", required=True, help="volume id (or unique substring)")
    o.add_argument("--grid", help="F, C or F,C (default: all grids of the volume)")
    o.add_argument("--force", action="store_true")
    o.set_defaults(fn=cmd_occupancy)
    r = sub.add_parser("read")
    r.add_argument("--root", default=str(DEFAULT_ROOT))
    r.add_argument("--sample", required=True)
    r.add_argument("--vol", required=True)
    r.add_argument("--grid", default="F")
    r.add_argument("--origin", required=True, help="z,y,x on the canonical grid")
    r.add_argument("--size", type=int, default=BLOCK)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_read)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
