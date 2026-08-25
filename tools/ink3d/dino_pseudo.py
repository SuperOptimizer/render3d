"""DINO ink-likeness pseudo-labels.

Runs the scrollprize 3D DINOv2 backbone (dinovol v2, patch 8^3, 864-D
tokens) over canonical-grid blocks and scores every 8^3 patch by cosine
similarity to the published reference ink embedding (avg_ref_embedding.npy =
L2-normalised mean of 256 expert-clicked ink tokens; the ink_3d teacher was
trained with "DINO guidance" at threshold tau = 0.5 on the same quantity).
The map is upsampled x8 to voxels and stored next to the block's other
labels as

    dino  u8  cos rescaled: 127 + (cos - tau) / 0.3 * 127, clipped
    (so 127 = at threshold, >127 ink-like, <127 not)

Input normalisation follows the backbone's training ("robust": clip to
p1/p99, then (x - median) / (1.4826 * MAD)).

CLI:
  dino_pseudo.py check --sample PHercParis4 [--grid F] [--n 20]
        score already-labelled blocks and report agreement with the raster
        labels (and with the published teacher where available)
  dino_pseudo.py run --sample X --grid F [--halo 16] [--limit N]
        add a `dino` map to every labelled block that lacks one
  dino_pseudo.py new --sample X --vol V --grid F --n N
        create pseudo-label-only blocks (ct + dino) from CT occupancy
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

BLOCK = corpus.BLOCK
PATCH = 8
TAU = 0.5
CKPT_DIR = Path(os.environ.get("R3D_INK3D_CKPT_DIR", os.path.expanduser("~/r3d-data/ink3d-ckpt")))
DEFAULT_BACKBONE = CKPT_DIR / "dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt"
DEFAULT_REF = CKPT_DIR / "avg_ref_embedding.npy"
VILLA_SRC = os.path.expanduser("~/villa/vesuvius/src")


# ------------------------------------------------------------- backbone
def load_backbone(path=DEFAULT_BACKBONE, device="cuda"):
    import torch
    if VILLA_SRC not in sys.path:
        sys.path.insert(0, VILLA_SRC)
    from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import (
        build_dinovol_2_backbone)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]["model"]
    net = build_dinovol_2_backbone(cfg)
    sd = {k[len("backbone."):]: v for k, v in ck["teacher"].items() if k.startswith("backbone.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"dino: load: missing {len(missing)} unexpected {len(unexpected)}: "
              f"{missing[:3]} {unexpected[:3]}")
    net.eval().to(device)
    n = sum(p.numel() for p in net.parameters())
    print(f"dino: backbone step {ck.get('step')} {n / 1e6:.1f}M params, patch {cfg['patch_size']}")
    return net


def load_refs(ref=DEFAULT_REF):
    r = np.load(ref).astype(np.float32)
    r /= np.linalg.norm(r) + 1e-8
    recs = []
    for f in sorted(glob.glob(str(Path(ref).parent / "recorded_embeddings*.npy"))):
        a = np.load(f).astype(np.float32)
        a /= np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
        recs.append(a)
    recs = np.concatenate(recs) if recs else None
    return r, recs


def normalize_robust(a):
    x = a.astype(np.float32)
    v = x[x > 0] if (x > 0).any() else x.reshape(-1)
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    x = np.clip(x, lo, hi)
    v = np.clip(v, lo, hi)
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if not np.isfinite(mad) or mad < 1e-6:
        mad = max(float(np.std(v)), (hi - lo) / 2.0, 1e-6)
    return (x - med) / mad


class Scorer:
    def __init__(self, backbone, ref, recs=None, device="cuda", halo=0):
        import torch
        self.net = backbone
        self.dev = device
        self.ref = torch.from_numpy(ref).to(device)
        self.recs = torch.from_numpy(recs).to(device) if recs is not None else None
        assert halo % PATCH == 0
        self.halo = halo

    def cos_map(self, ct):
        """ct u8 [S,S,S] (S multiple of 8) -> cos map [S/8]^3 float32
        (and max-cos over recorded embeddings if available)."""
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(normalize_robust(ct))[None, None].to(self.dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.net.forward_features(x, masks=None, view_kind="global")
        tok = out["x_norm_patchtokens"][0].float()
        tok = F.normalize(tok, dim=-1)
        g = tuple(s // PATCH for s in ct.shape)
        cos = (tok @ self.ref).reshape(g)
        rec = (tok @ self.recs.T).max(dim=-1).values.reshape(g) if self.recs is not None else None
        return cos.cpu().numpy(), (rec.cpu().numpy() if rec is not None else None)

    def block_map(self, read_ct, origin):
        """cos map for a BLOCK^3 block at `origin` with halo context."""
        h = self.halo
        if h == 0:
            ct = read_ct(origin, BLOCK)
            cos, rec = self.cos_map(ct)
            return ct, cos, rec
        o = tuple(int(x - h) for x in origin)
        ct = read_ct(o, BLOCK + 2 * h)
        cos, rec = self.cos_map(ct)
        hp = h // PATCH
        n = BLOCK // PATCH
        sl = (slice(hp, hp + n),) * 3
        return ct[h:h + BLOCK, h:h + BLOCK, h:h + BLOCK], cos[sl], (rec[sl] if rec is not None else None)


def upsample(cos):
    """[n]^3 -> [8n]^3 trilinear."""
    from scipy.ndimage import zoom
    return zoom(cos, PATCH, order=1)


def to_u8(cos):
    return np.clip(127 + (cos - TAU) / 0.3 * 127 + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ IO
def _read_ct_fn(vol, grid):
    def f(origin, size):
        return corpus.read_block(vol, grid, origin, size)
    return f


def _read_ct_edgepad(vol, grid):
    """read_block that clamps negative origins (halo at the volume start)."""
    def f(origin, size):
        o = [max(0, x) for x in origin]
        blk = corpus.read_block(vol, grid, tuple(o), size)
        out = np.zeros((size,) * 3, np.uint8)
        d = [oo - x for oo, x in zip(o, origin)]
        out[d[0]:, d[1]:, d[2]:] = blk[:size - d[0], :size - d[1], :size - d[2]]
        return out
    return f


def add_dino(path, dino_u8, extra=None):
    d = dict(np.load(path, allow_pickle=False))
    d["dino"] = dino_u8
    if extra:
        d.update(extra)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


# ----------------------------------------------------------------- CLI
def _blocks(root, sample, grid):
    return sorted(Path(root, sample, grid).glob("*.npz"))


def cmd_check(args):
    man = corpus.load_manifest(args.root)
    files = _blocks(args.root, args.sample, args.grid)
    if args.vol:
        files = [f for f in files if f.name.startswith(args.vol)]
    if not files:
        sys.exit("dino: no labelled blocks")
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in rng.choice(len(files), min(args.n, len(files)), replace=False)]
    net = load_backbone(args.backbone)
    ref, recs = load_refs(args.ref)
    sc = Scorer(net, ref, recs, halo=args.halo)
    teacher = None
    if args.sample == "PHercParis4" and args.grid == "F":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import distill
        teacher = distill.open_level(distill.DEFAULT_TEACHER, 0)
    agg = {"n": 0, "ink_cos": [], "bg_cos": [], "dice_lab": [], "dice_t": [], "corr_t": []}
    t0 = time.time()
    for f in files:
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        vol = man["volumes"][meta["vol"]]
        origin = tuple(meta["origin"])
        ct, cos, rec = sc.block_map(_read_ct_edgepad(vol, args.grid), origin)
        up = upsample(cos)
        w, ink = d["weight"], d["ink"]
        lab_ink = (w >= 150) & (ink > 128)
        lab_bg = (w > 0) & (ink < 64)
        if lab_ink.sum() > 100:
            agg["ink_cos"].append(float(up[lab_ink].mean()))
        if lab_bg.sum() > 100:
            agg["bg_cos"].append(float(up[lab_bg].mean()))
        lab = w > 0
        if lab.sum() > 100:
            p, t = up[lab] > TAU, lab_ink[lab]
            agg["dice_lab"].append(2 * (p & t).sum() / max(1, p.sum() + t.sum()))
        if teacher is not None:
            z, y, x = origin
            tp = np.zeros((BLOCK,) * 3, np.uint8)
            a = teacher[z:z + BLOCK, y:y + BLOCK, x:x + BLOCK]
            tp[:a.shape[0], :a.shape[1], :a.shape[2]] = a
            tp[ct == 0] = 0
            t = tp > 128
            p = (up > TAU) & (ct > 0)
            if t.sum() > 100:
                agg["dice_t"].append(2 * (p & t).sum() / max(1, p.sum() + t.sum()))
                m = ct > 0
                agg["corr_t"].append(float(np.corrcoef(up[m], tp[m].astype(np.float32))[0, 1]))
        agg["n"] += 1
        print(f"  {f.name}: cos mean {cos.mean():.3f} max {cos.max():.3f} "
              f">tau {(cos > TAU).mean():.3f}"
              + (f" rec-max mean {rec.mean():.3f}" if rec is not None else ""), flush=True)
    dt = (time.time() - t0) / max(1, agg["n"])
    print(f"dino: {agg['n']} blocks, {dt:.2f}s/block (halo {args.halo})")
    for k in ("ink_cos", "bg_cos", "dice_lab", "dice_t", "corr_t"):
        v = agg[k]
        if v:
            print(f"  {k}: mean {np.mean(v):.3f}  (n={len(v)})")


def cmd_run(args):
    man = corpus.load_manifest(args.root)
    files = [f for f in _blocks(args.root, args.sample, args.grid)]
    if args.vol:
        files = [f for f in files if f.name.startswith(args.vol)]
    todo = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            if "dino" not in d.files or args.force:
                todo.append(f)
    if args.limit:
        todo = todo[:args.limit]
    print(f"dino: {len(todo)}/{len(files)} blocks to score")
    if not todo:
        return
    net = load_backbone(args.backbone)
    ref, recs = load_refs(args.ref)
    sc = Scorer(net, ref, recs, halo=args.halo)
    t0 = time.time()
    for i, f in enumerate(todo):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        vol = man["volumes"][meta["vol"]]
        origin = tuple(meta["origin"])
        if args.halo == 0:
            cos, rec = sc.cos_map(d["ct"])
        else:
            _, cos, rec = sc.block_map(_read_ct_edgepad(vol, args.grid), origin)
        add_dino(f, to_u8(upsample(cos)))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(todo)} ({(time.time() - t0) / (i + 1):.2f}s/block)", flush=True)
    print(f"dino: done {len(todo)} in {time.time() - t0:.0f}s")


def cmd_new(args):
    """Pseudo-label-only blocks: pick occupied CT blocks (no raster label)."""
    man = corpus.load_manifest(args.root)
    vol = None
    for v in man["volumes"].values():
        if v["sample"] == args.sample and args.vol in v["id"]:
            vol = v
    if vol is None:
        sys.exit("dino: volume not found")
    occ = corpus.occupancy(vol, args.grid, args.root)[0].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    cand = np.argwhere(occ > 0.5)
    have = {Path(f).name for f in _blocks(args.root, args.sample, args.grid)}
    picks = []
    for b in cand[rng.permutation(len(cand))]:
        name = f"{vol['id']}_{b[0]}_{b[1]}_{b[2]}.npz"
        if name in have:
            continue
        picks.append(tuple(int(x) for x in b))
        if len(picks) >= args.n:
            break
    print(f"dino: {len(picks)} new blocks from {len(cand)} occupied")
    net = load_backbone(args.backbone)
    ref, recs = load_refs(args.ref)
    sc = Scorer(net, ref, recs, halo=args.halo)
    out_dir = Path(args.root, args.sample, args.grid)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, b in enumerate(picks):
        origin = tuple(x * BLOCK for x in b)
        ct, cos, rec = sc.block_map(_read_ct_edgepad(vol, args.grid), origin)
        if ct.max() == 0:
            continue
        meta = {"sample": args.sample, "vol": vol["id"], "grid": args.grid, "origin": list(origin),
                "segments": [], "um": vol["um"], "keV": vol["keV"]}
        path = out_dir / f"{vol['id']}_{b[0]}_{b[1]}_{b[2]}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ct=ct, ink=np.zeros_like(ct), weight=np.zeros_like(ct),
                            src=np.uint8(3), dino=to_u8(upsample(cos)), meta=json.dumps(meta))
        os.replace(tmp, path)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(picks)} ({(time.time() - t0) / (i + 1):.2f}s/block)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
        p.add_argument("--sample", required=True)
        p.add_argument("--grid", default="F")
        p.add_argument("--backbone", default=str(DEFAULT_BACKBONE))
        p.add_argument("--ref", default=str(DEFAULT_REF))
        p.add_argument("--halo", type=int, default=16, help="context voxels (multiple of 8)")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--vol", default="", help="restrict to blocks of this volume id")

    c = sub.add_parser("check")
    common(c)
    c.add_argument("--n", type=int, default=20)
    c.set_defaults(fn=cmd_check)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    n = sub.add_parser("new")
    common(n)
    n.add_argument("--n", type=int, default=200)
    n.set_defaults(fn=cmd_new)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
