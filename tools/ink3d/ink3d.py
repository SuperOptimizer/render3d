#!/usr/bin/env python3
"""3D ink detection with a compute-once chunk cache.

Runs scrollprize/ink_3d_dino_guided (volumetric 3D ink segmentation, native
2.4um CT, 256^3 patches) over the chunks a traced surface passes through and
caches the probabilities into a local zarr laid out exactly like the published
prediction zarrs (v2, u1, 256^3 chunks, blosc-zstd, "/" separator).  A chunk
that exists in the cache is never recomputed -- all-zero results are written
too, so chunk existence IS the computed-once marker.

Run from villa's ink-detection env:

    ~/villa-ink/ink-detection/.venv/bin/python tools/ink3d/ink3d.py infer \
        --segment ~/r3d-data/paris4-segstore/gp-seg.tfx \
        --out ~/r3d-data/paris4-ink3d.zarr

    # then make it viewable as the red 3D-ink overlay:
    ... ink3d.py scene --zarr ~/r3d-data/paris4-ink3d.zarr \
        --out ~/r3d-data/paris4-ink3d-local
    ~/render3d-gpu --bricks paris4-lod/manifest.json --ink3d paris4-ink3d-local

Footprint sources (any mix): --segment tifxyz dirs (x/y/z.tif, level-0 voxel
coords -- both GUI-saved segments and the approved GT), --bbox z0:z1,y0:y1,x0:x1,
or --chunks-file with one "cz cy cx" triple per line.  --pad dilates the
surface footprint so ink sitting on the sheet is covered.

Quality: --halo 64 (default) infers each chunk from a 384^3 region as 8
overlapping 256^3 patches, gaussian-blended, so chunk borders see full context
(matches how the published predictions were tiled).  --halo 0 is ~8x faster
(one forward per chunk) at the cost of seams at chunk borders.  Normalization
is per-patch p1/p99 min-max, matching training.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "villa/vesuvius/src"))

DEFAULT_SOURCE = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/"
                  "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
DEFAULT_CKPT = str(Path.home() / "r3d-data/ink3d-ckpt/ckpt_78k_fullsup.pth")
CHUNK = 256
NLEVELS = 6


# ---------------------------------------------------------------- model

def load_model(ckpt_path, device="cuda"):
    import torch
    from vesuvius.models.run.inference import _normalize_train_py_model_config
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    torch.backends.cudnn.benchmark = True  # fixed 256^3 shapes: autotune convs
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = _normalize_train_py_model_config(ck)

    class Mgr:
        def __init__(self, mc):
            self.model_config = mc
            self.targets = mc.get("targets", {})
            self.train_patch_size = mc.get("train_patch_size", mc.get("patch_size", (256, 256, 256)))
            self.train_batch_size = mc.get("train_batch_size", mc.get("batch_size", 2))
            self.in_channels = mc.get("in_channels", 1)
            self.autoconfigure = mc.get("autoconfigure", False)
            self.enable_deep_supervision = bool(mc.get("enable_deep_supervision", False))
            self.model_name = mc.get("model_name", "Model")
            self.spacing = [1.0, 1.0, 1.0]

    net = NetworkFromConfig(Mgr(mc))
    state = ck.get("ema_model") or ck["model"]
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    step = ck.get("step")
    print(f"ink3d: model loaded ({sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
          f"step {step}, {'ema' if 'ema_model' in ck else 'raw'} weights)")
    return net


def normalize(patch):
    """Per-patch p1/p99 min-max to [0,1], matching training. Percentiles are
    taken on an 8x-strided subsample: indistinguishable on 16M-voxel patches
    and much cheaper than a full sort."""
    lo, hi = np.percentile(patch[::2, ::2, ::2], (1.0, 99.0))
    d = float(hi - lo)
    if d <= 1e-8:
        return np.zeros_like(patch, dtype=np.float32)
    return ((np.clip(patch, lo, hi) - lo) / d).astype(np.float32)


_gauss = None

def gaussian_map(ps):
    global _gauss
    if _gauss is None:
        sigma = ps / 8.0
        ax = np.arange(ps, dtype=np.float32) - (ps - 1) / 2.0
        g1 = np.exp(-(ax ** 2) / (2 * sigma ** 2)).astype(np.float32)
        g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
        _gauss = np.maximum(g, g.max() * 1e-4)
    return _gauss


def infer_region(net, region, halo, tta=False):
    """region: (256+2*halo)^3 u8 array. Returns center 256^3 prob f32."""
    import itertools
    import torch
    flips = (list(itertools.chain.from_iterable(
        itertools.combinations((2, 3, 4), r) for r in range(4))) if tta else [()])
    n = region.shape[0]
    # patch origins per axis: stride <= 128 coverage of [0, n-256]
    span = n - CHUNK
    if span == 0:
        origins = [0]
    else:
        k = max(1, int(np.ceil(span / 128)))
        origins = sorted({int(round(i * span / k)) for i in range(k + 1)})
    acc = np.zeros((n, n, n), dtype=np.float32)
    wac = np.zeros((n, n, n), dtype=np.float32)
    g = gaussian_map(CHUNK)
    for oz in origins:
        for oy in origins:
            for ox in origins:
                p = region[oz:oz + CHUNK, oy:oy + CHUNK, ox:ox + CHUNK]
                if p.max() == 0:
                    continue  # masked air: prob 0, weight via wac stays 0 -> 0
                x = torch.from_numpy(normalize(p)).pin_memory()
                x = x[None, None].to("cuda", non_blocking=True)
                pr = torch.zeros((CHUNK, CHUNK, CHUNK), device="cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for f in flips:
                        y = net(torch.flip(x, f) if f else x)["ink"]
                        y = torch.flip(y, f) if f else y
                        pr += torch.sigmoid(y.float())[0, 0]
                prob = (pr / len(flips)).cpu().numpy()
                acc[oz:oz + CHUNK, oy:oy + CHUNK, ox:ox + CHUNK] += prob * g
                wac[oz:oz + CHUNK, oy:oy + CHUNK, ox:ox + CHUNK] += g
    out = np.where(wac > 0, acc / np.maximum(wac, 1e-8), 0.0)
    c = halo
    return out[c:c + CHUNK, c:c + CHUNK, c:c + CHUNK]


# ---------------------------------------------------------------- footprint

def footprint_from_tifxyz(seg_dir, pad):
    import tifffile
    d = Path(seg_dir)
    xs = tifffile.imread(d / "x.tif").astype(np.float64)
    ys = tifffile.imread(d / "y.tif").astype(np.float64)
    zs = tifffile.imread(d / "z.tif").astype(np.float64)
    valid = (xs >= 0) & (ys >= 0) & (zs >= 0)
    pts = np.stack([zs[valid], ys[valid], xs[valid]], axis=1)
    chunks = set()
    for dz in (-pad, pad):
        for dy in (-pad, pad):
            for dx in (-pad, pad):
                q = np.floor((pts + [dz, dy, dx]) / CHUNK).astype(np.int64)
                q = q[(q >= 0).all(axis=1)]
                chunks.update(map(tuple, np.unique(q, axis=0)))
    print(f"ink3d: {seg_dir}: {valid.sum()} surface points -> {len(chunks)} chunks (pad {pad})")
    return chunks


# ---------------------------------------------------------------- zarr cache

def open_source(url):
    import zarr
    if "://" in url:
        import fsspec
        store = fsspec.get_mapper(url + "/0")
    else:
        store = str(Path(url) / "0")
    return zarr.open(store, mode="r")


def open_cache(path, shape):
    import zarr
    from numcodecs import Blosc
    root = zarr.open_group(str(path), mode="a")
    comp = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    for l in range(NLEVELS):
        s = [max(1, (d + (1 << l) - 1) >> l) for d in shape]
        if str(l) not in root:
            root.create_dataset(str(l), shape=s, chunks=(CHUNK, CHUNK, CHUNK),
                                dtype="u1", compressor=comp, fill_value=0,
                                dimension_separator="/")
    if "multiscales" not in root.attrs:
        root.attrs["multiscales"] = [{
            "axes": [{"name": a, "type": "space"} for a in "zyx"],
            "datasets": [{"path": str(l),
                          "coordinateTransformations": [
                              {"type": "scale", "scale": [float(1 << l)] * 3}]}
                         for l in range(NLEVELS)],
            "version": "0.4"}]
    return root


def chunk_exists(cache_path, level, cz, cy, cx):
    return (Path(cache_path) / str(level) / str(cz) / str(cy) / str(cx)).exists()


def read_region(src, z0, y0, x0, n):
    """Read n^3 at (z0,y0,x0) from source, zero-padded outside the volume."""
    Z, Y, X = src.shape
    out = np.zeros((n, n, n), dtype=np.uint8)
    az0, ay0, ax0 = max(z0, 0), max(y0, 0), max(x0, 0)
    az1, ay1, ax1 = min(z0 + n, Z), min(y0 + n, Y), min(x0 + n, X)
    if az0 >= az1 or ay0 >= ay1 or ax0 >= ax1:
        return out
    out[az0 - z0:az1 - z0, ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = \
        src[az0:az1, ay0:ay1, ax0:ax1]
    return out


def update_pyramid(root, l0_chunks):
    """Refresh coarse levels above the given set of freshly-written L0 chunks."""
    affected = set(l0_chunks)
    for l in range(1, NLEVELS):
        parents = {(cz // 2, cy // 2, cx // 2) for cz, cy, cx in affected}
        fine, coarse = root[str(l - 1)], root[str(l)]
        for cz, cy, cx in sorted(parents):
            z0, y0, x0 = cz * CHUNK * 2, cy * CHUNK * 2, cx * CHUNK * 2
            fz = min(CHUNK * 2, fine.shape[0] - z0)
            fy = min(CHUNK * 2, fine.shape[1] - y0)
            fx = min(CHUNK * 2, fine.shape[2] - x0)
            a = np.zeros((CHUNK * 2, CHUNK * 2, CHUNK * 2), dtype=np.uint8)
            a[:fz, :fy, :fx] = fine[z0:z0 + fz, y0:y0 + fy, x0:x0 + fx]
            pooled = a.reshape(CHUNK, 2, CHUNK, 2, CHUNK, 2).mean(axis=(1, 3, 5))
            oz, oy, ox = cz * CHUNK, cy * CHUNK, cx * CHUNK
            pz = min(CHUNK, coarse.shape[0] - oz)
            py = min(CHUNK, coarse.shape[1] - oy)
            px = min(CHUNK, coarse.shape[2] - ox)
            coarse[oz:oz + pz, oy:oy + py, ox:ox + px] = \
                pooled[:pz, :py, :px].astype(np.uint8)
        affected = parents
    print(f"ink3d: pyramid refreshed to L{NLEVELS - 1}")


# ---------------------------------------------------------------- commands

def cmd_infer(args):
    chunks = set()
    for seg in args.segment or []:
        chunks |= footprint_from_tifxyz(seg, args.pad)
    if args.bbox:
        try:
            rng = [tuple(map(int, p.split(":"))) for p in args.bbox.split(",")]
            (z0, z1), (y0, y1), (x0, x1) = rng
        except ValueError:
            sys.exit("ink3d: --bbox wants z0:z1,y0:y1,x0:x1 (voxels)")
        for cz in range(z0 // CHUNK, (z1 + CHUNK - 1) // CHUNK):
            for cy in range(y0 // CHUNK, (y1 + CHUNK - 1) // CHUNK):
                for cx in range(x0 // CHUNK, (x1 + CHUNK - 1) // CHUNK):
                    chunks.add((cz, cy, cx))
    if args.chunks_file:
        for ln in Path(args.chunks_file).read_text().split("\n"):
            if ln.strip():
                cz, cy, cx = map(int, ln.split())
                chunks.add((cz, cy, cx))
    if not chunks:
        sys.exit("ink3d: no chunks selected (need --segment / --bbox / --chunks-file)")

    src = open_source(args.source)
    Z, Y, X = src.shape
    print(f"ink3d: source {args.source} shape {src.shape}")
    grid = ((Z + CHUNK - 1) // CHUNK, (Y + CHUNK - 1) // CHUNK, (X + CHUNK - 1) // CHUNK)
    chunks = {(cz, cy, cx) for cz, cy, cx in chunks
              if cz < grid[0] and cy < grid[1] and cx < grid[2]}
    todo = sorted(c for c in chunks
                  if args.force or not chunk_exists(args.out, 0, *c))
    print(f"ink3d: {len(chunks)} chunks in footprint, {len(chunks) - len(todo)} cached, "
          f"{len(todo)} to compute")
    if not todo:
        return

    root = open_cache(args.out, src.shape)
    net = load_model(args.ckpt)
    out0 = root["0"]
    halo = args.halo

    # prefetch source regions ahead of the GPU: several fetcher threads so
    # the (many-small-request) S3 reads never gate inference
    fetched = {}
    next_i = [0]
    lock = threading.Lock()
    cv = threading.Condition(lock)
    NFETCH, DEPTH = 4, 8

    def fetcher():
        while True:
            with cv:
                while next_i[0] < len(todo) and len(fetched) >= DEPTH:
                    cv.wait()
                if next_i[0] >= len(todo):
                    return
                c = todo[next_i[0]]
                next_i[0] += 1
                fetched[c] = None  # claimed
            cz, cy, cx = c
            for attempt in range(4):
                try:
                    reg = read_region(src, cz * CHUNK - halo, cy * CHUNK - halo,
                                      cx * CHUNK - halo, CHUNK + 2 * halo)
                    break
                except Exception as e:
                    if attempt == 3:  # sys.exit only kills this thread
                        print(f"ink3d: source read failed for chunk {c}: {e}",
                              file=sys.stderr, flush=True)
                        os._exit(1)
                    time.sleep(2 << attempt)
            with cv:
                fetched[c] = reg
                cv.notify_all()

    for _ in range(NFETCH):
        threading.Thread(target=fetcher, daemon=True).start()

    written = []
    flushed = 0
    t_start = time.time()
    for i, c in enumerate(todo):
        cz, cy, cx = c
        with cv:
            while fetched.get(c) is None:
                cv.wait()
            region = fetched.pop(c)
            cv.notify_all()
        t0 = time.time()
        if region.max() == 0:
            prob8 = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)
            tag = "air"
        else:
            prob = infer_region(net, region, halo, args.tta)
            prob8 = np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)
            tag = f"ink {(prob8 >= 128).mean() * 100:.2f}%"
        z0, y0, x0 = cz * CHUNK, cy * CHUNK, cx * CHUNK
        out0[z0:min(z0 + CHUNK, Z), y0:min(y0 + CHUNK, Y), x0:min(x0 + CHUNK, X)] = \
            prob8[:min(CHUNK, Z - z0), :min(CHUNK, Y - y0), :min(CHUNK, X - x0)]
        written.append(c)
        done, left = i + 1, len(todo) - i - 1
        eta = (time.time() - t_start) / done * left
        print(f"ink3d: [{done}/{len(todo)}] chunk {cz},{cy},{cx}: {tag} "
              f"({time.time() - t0:.1f}s, eta {eta / 60:.0f}m)", flush=True)
        if done % 64 == 0:
            update_pyramid(root, written[flushed:])
            flushed = len(written)
    if written[flushed:]:
        update_pyramid(root, written[flushed:])
    print(f"ink3d: done, {len(written)} chunks in {(time.time() - t_start) / 60:.1f}m "
          f"-> {args.out}")


def cmd_scene(args):
    import zarr
    root = zarr.open_group(str(args.zarr), mode="r")
    shape = list(root["0"].shape)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = []
    for l in range(NLEVELS):
        s = [max(1, (d + (1 << l) - 1) >> l) for d in shape]
        sh = [(d + 1023) // 1024 for d in s]
        q = [2, 1, 0.5, 0.25, 0.25, 0.25][l]
        levels.append(
            f'    {{"level": {l}, "scale": {1 << l}, "shape": [{s[0]}, {s[1]}, {s[2]}], '
            f'"shards": [{sh[0]}, {sh[1]}, {sh[2]}], "zarr": "zarr/L{l}", '
            f'"c5d": "c5d/L{l}/{{z}}_{{y}}_{{x}}.c5s", "c5d_quality": {q}}}')
    (out / "manifest.json").write_text(
        '{\n  "format": "render3d.c5d-lod.v1",\n'
        f'  "shape": [{shape[0]}, {shape[1]}, {shape[2]}],\n'
        '  "shard_shape": [1024, 1024, 1024],\n'
        '  "brick_shape": [128, 128, 128],\n'
        '  "levels": [\n' + ",\n".join(levels) + "\n  ]\n}\n")
    url = "file://" + str(Path(args.zarr).resolve())
    lv = ",\n".join(f'    {{"level": {l}, "chunk": 256, "raw": false}}'
                    for l in range(NLEVELS))
    (out / "source.json").write_text(
        '{\n  "format": "render3d.c5d-source.v1",\n'
        f'  "url": "{url}",\n  "quality": 2,\n  "levels": [\n' + lv + "\n  ]\n}\n")
    print(f"ink3d: scene {out} -> {url}")
    print(f"ink3d: view with: ~/render3d-gpu --bricks <ct-lod>/manifest.json "
          f"--ink3d {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ai = sub.add_parser("infer", help="compute ink for chunks touched by a surface")
    ai.add_argument("--segment", action="append", help="tifxyz dir (repeatable)")
    ai.add_argument("--bbox", help="z0:z1,y0:y1,x0:x1 voxel box")
    ai.add_argument("--chunks-file", help="file with 'cz cy cx' lines")
    ai.add_argument("--source", default=DEFAULT_SOURCE, help="CT zarr url or path")
    ai.add_argument("--out", required=True, help="local cache zarr dir")
    ai.add_argument("--ckpt", default=DEFAULT_CKPT)
    ai.add_argument("--pad", type=int, default=64,
                    help="surface dilation in voxels (default 64 = 154um)")
    ai.add_argument("--halo", type=int, default=0,
                    help="context halo in voxels; 0 (default) = one forward per "
                         "chunk, fastest; 64 = 8 blended patches, no border seams")
    ai.add_argument("--tta", action="store_true",
                    help="8-flip mirroring TTA (~4.5x slower, closest to published)")
    ai.add_argument("--force", action="store_true", help="recompute cached chunks")
    ai.set_defaults(fn=cmd_infer)
    asc = sub.add_parser("scene", help="build a render3d overlay scene for a cache zarr")
    asc.add_argument("--zarr", required=True)
    asc.add_argument("--out", required=True)
    asc.set_defaults(fn=cmd_scene)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
