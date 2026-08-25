"""Train a small 3D ink model on the multi-scroll, multi-source corpus.

Data: 128^3 blocks written by project_raster.py / dino_pseudo.py under
<root>/<sample>/<grid>/ (ct, ink, weight, [dino], [teacher]) plus,
optionally, the 256^3 (ct, teacher pred) pairs harvested by distill.py.
Every source becomes a (target prob, weight) pair per voxel:

  raster25d  ink/255,                weight/255            x --w-raster
  teacher    teacher/255 (ct > 0),   0.6                   x --w-teacher
  dino       1 if dino > 127+m else 0, |dino-127|/127 (m=20 dead band) x --w-dino

loss = sum_src W_src * BCE(logit, T_src) / sum W  +  0.5 * weighted soft dice
against the W-merged target. Sampling is round-robin over
(sample, grid, kind) buckets so PHerc Paris 4 does not dominate. Held-out:
whole samples (--holdout, default PHerc0139) plus a deterministic block-hash
split inside every bucket for the fast validation table.

Augmentation: axis permutation + flips, random gamma/contrast/noise, and
+-25% isotropic scale jitter (applied on the GPU per batch).

CLI:
  train_corpus.py add-teacher --sample PHercParis4 [--grid F,C]  attach teacher preds
  train_corpus.py index [--root R]                                 list buckets
  train_corpus.py train --out ckpt.pth [--preset small] [--steps N] [--resume ckpt]
  train_corpus.py val --ckpt ckpt.pth                              per-bucket table
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import studentnet  # noqa: E402
from distill import DEFAULT_TEACHER, normalize, soft_dice_loss  # noqa: E402

BLOCK = corpus.BLOCK
DINO_DEAD = 20
TEACHER_W = 0.6


def is_val(b):
    return (b[0] * 73856093 ^ b[1] * 19349663 ^ b[2] * 83492791) % 8 == 0


# ------------------------------------------------------------ teacher
def cmd_add_teacher(args):
    """Attach the published teacher prediction to blocks of a sample whose
    volume has an ink-detection-3d zarr (PHerc Paris 4 2.4um)."""
    man = corpus.load_manifest(args.root)
    for g in args.grid.split(","):
        files = corpus.list_blocks(args.root, args.sample, g)
        todo = []
        for f in files:
            with np.load(f, allow_pickle=False) as d:
                if "teacher" not in d.files or args.force:
                    todo.append(f)
        print(f"add-teacher: {args.sample}/{g}: {len(todo)}/{len(files)} blocks")
        if not todo:
            continue
        levels = {}

        def work(f):
            with corpus.block_lock(f):
                d = dict(np.load(f, allow_pickle=False))
                meta = json.loads(str(d["meta"]))
                vol = man["volumes"][meta["vol"]]
                preds = vol["predictions"].get("ink-detection-3d-zarr")
                if not preds:
                    return 0
                url = corpus.BUCKET + "/" + preds[0].rstrip("/")
                gg = vol["grids"][g]
                L, fac = gg["level"], gg["f"]
                key = (url, L)
                with lock:
                    if key not in levels:
                        levels[key] = corpus.open_level(url, L)
                a = levels[key]
                origin = meta["origin"]
                vrow = dict(vol, url=url)
                t = corpus.read_block(vrow, g, tuple(origin))
                t[d["ct"] == 0] = 0
                d["teacher"] = t
                tmp = f.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **d)
                os.replace(tmp, f)
                return 1

        lock = threading.Lock()
        t0 = time.time()
        n = 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for i, r in enumerate(ex.map(work, todo)):
                n += r
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"add-teacher: {n} attached in {time.time() - t0:.0f}s")


# -------------------------------------------------------------- index
def build_index(root, pairs_dir=None, holdout=(), exclude_segments=()):
    """{bucket: [paths]} with bucket = (sample, grid, kind)."""
    root = Path(root)
    idx = {}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir() or sd.name in ("segments", "occ", "eval"):
            continue
        if sd.name in holdout:
            continue
        for gd in sorted(sd.iterdir()):
            if not gd.is_dir():
                continue
            files = [f for f in sorted(gd.glob("*.npz")) if ".tmp" not in f.name]
            if not files:
                continue
            idx.setdefault((sd.name, gd.name, "blocks"), []).extend(files)
    if pairs_dir:
        files = sorted(Path(pairs_dir).glob("*_*_*.npz"))
        files = [f for f in files if f.stem.count("_") == 2 and ".tmp" not in f.name]
        if files:
            idx[("PHercParis4", "F", "pairs256")] = files
    if exclude_segments:
        ex = set(exclude_segments)
        for k, files in list(idx.items()):
            if k[2] != "blocks":
                continue
            keep = []
            for f in files:
                try:
                    with np.load(f, allow_pickle=False) as d:
                        segs = json.loads(str(d["meta"])).get("segments", [])
                except Exception:  # noqa: BLE001
                    continue
                if not (ex & set(segs)):
                    keep.append(f)
            idx[k] = keep
    return idx


def split_index(idx):
    tr, va = {}, {}
    for k, files in idx.items():
        for f in files:
            parts = f.stem.split("_")
            try:
                b = tuple(int(x) for x in parts[-3:])
            except ValueError:
                continue
            (va if is_val(b) else tr).setdefault(k, []).append(f)
    return tr, va


def cmd_index(args):
    idx = build_index(args.root, args.pairs, holdout=args.holdout.split(","))
    tr, va = split_index(idx)
    tot = 0
    for k in sorted(idx):
        n = len(idx[k])
        tot += n
        print(f"{k[0]:14s} {k[1]} {k[2]:9s} {n:6d} blocks  (train {len(tr.get(k, []))}, val {len(va.get(k, []))})")
    print(f"total {tot}")


# ------------------------------------------------------------- loading
def load_sample(path, kind):
    """-> dict(ct f16 normalized, targets: list of (T f16, W f16)) on 128^3
    (pairs256 keep their 256^3 size; crops are taken later)."""
    d = np.load(path, allow_pickle=False)
    ct = d["ct"]
    out = {"ct": normalize(ct).astype(np.float16), "src": []}
    valid = ct > 0
    if kind == "pairs256":
        pr = d["pred"] if "pred" in d.files else d["pr"]
        out["src"].append(("teacher", (pr.astype(np.float16) / 255.0),
                           (valid.astype(np.float16) * TEACHER_W)))
        return out
    w = d["weight"]
    if w.max() > 0:
        out["src"].append(("raster", d["ink"].astype(np.float16) / 255.0,
                           w.astype(np.float16) / 255.0))
    if "teacher" in d.files:
        out["src"].append(("teacher", d["teacher"].astype(np.float16) / 255.0,
                           valid.astype(np.float16) * TEACHER_W))
    if "dino" in d.files:
        dn = d["dino"].astype(np.float32) - 127.0
        wd = np.clip((np.abs(dn) - DINO_DEAD) / (127.0 - DINO_DEAD), 0, 1) * valid
        out["src"].append(("dino", (dn > 0).astype(np.float16), wd.astype(np.float16)))
    return out


class Pool:
    """Replay pool: loader threads cycle blocks round-robin over buckets;
    training samples random crops (+ axis permutation and flips)."""

    def __init__(self, buckets, cap, rng, nthreads=4):
        self.buckets = [(k, list(v)) for k, v in buckets.items() if v]
        self.cap, self.rng = cap, rng
        self.buf = []
        self.loaded = 0
        self.lock = threading.Lock()
        self.quit = False
        self.bi = 0
        self.pos = {k: 0 for k, _ in self.buckets}
        for k, files in self.buckets:
            self.rng.shuffle(files)
        self.ths = [threading.Thread(target=self._loader, daemon=True) for _ in range(nthreads)]
        for t in self.ths:
            t.start()
        while True:
            with self.lock:
                if len(self.buf) >= min(16, self.cap):
                    break
            time.sleep(0.2)

    def _next_file(self):
        with self.lock:
            k, files = self.buckets[self.bi % len(self.buckets)]
            self.bi += 1
            i = self.pos[k]
            self.pos[k] = (i + 1) % len(files)
            if self.pos[k] == 0:
                self.rng.shuffle(files)
            return k, files[i]

    def _loader(self):
        while not self.quit:
            k, f = self._next_file()
            try:
                s = load_sample(f, k[2])
            except Exception as e:  # noqa: BLE001
                print(f"train: load error {f}: {e}", file=sys.stderr, flush=True)
                continue
            if not s["src"]:
                continue
            s["bucket"] = k
            with self.lock:
                if len(self.buf) >= self.cap:
                    self.buf.pop(self.rng.integers(len(self.buf)))
                self.buf.append(s)
                self.loaded += 1
            if len(self.buf) >= self.cap:
                time.sleep(0.05)

    def sample_crop(self, crop, srcw):
        """-> ct [crop]^3 f32, T [crop]^3, W [crop]^3 (merged), plus per-source
        lists for the BCE term."""
        with self.lock:
            s = self.buf[self.rng.integers(len(self.buf))]
        S = s["ct"].shape[0]
        z, y, x = (self.rng.integers(S - crop + 1) for _ in range(3))
        sl = (slice(z, z + crop), slice(y, y + crop), slice(x, x + crop))
        perm = list(self.rng.permutation(3))
        flips = [bool(self.rng.integers(2)) for _ in range(3)]

        def tf(a):
            a = a[sl].transpose(perm)
            for ax in range(3):
                if flips[ax]:
                    a = np.flip(a, ax)
            return np.ascontiguousarray(a, dtype=np.float32)

        ct = tf(s["ct"])
        srcs = [(name, tf(T), tf(W) * srcw.get(name, 1.0)) for name, T, W in s["src"]]
        return ct, srcs


# ------------------------------------------------------------- augment
def augment_batch(x, rng, torch):
    """Intensity aug on a normalized [B,1,...] batch (in place-ish)."""
    B = x.shape[0]
    g = torch.from_numpy(rng.uniform(0.7, 1.4, B).astype(np.float32)).to(x.device)
    c = torch.from_numpy(rng.uniform(0.75, 1.25, B).astype(np.float32)).to(x.device)
    o = torch.from_numpy(rng.uniform(-0.1, 0.1, B).astype(np.float32)).to(x.device)
    x = x.clamp(0, 1) ** g.view(B, 1, 1, 1, 1)
    x = (x - 0.5) * c.view(B, 1, 1, 1, 1) + 0.5 + o.view(B, 1, 1, 1, 1)
    if rng.integers(2):
        x = x + torch.randn_like(x) * float(rng.uniform(0.0, 0.05))
    return x


def scale_jitter(x, T, W, rng, torch, lo=0.75, hi=1.25):
    """Resample the batch by a random isotropic factor, then centre-crop /
    pad back to the original size."""
    import torch.nn.functional as F
    s = float(rng.uniform(lo, hi))
    if abs(s - 1.0) < 0.03:
        return x, T, W
    n = x.shape[-1]
    m = max(16, int(round(n * s)) // 8 * 8)
    x2 = F.interpolate(x, size=(m, m, m), mode="trilinear", align_corners=False)
    T2 = F.interpolate(T, size=(m, m, m), mode="trilinear", align_corners=False)
    W2 = F.interpolate(W, size=(m, m, m), mode="trilinear", align_corners=False)
    if m >= n:
        o = (m - n) // 2
        sl = (slice(None), slice(None), slice(o, o + n), slice(o, o + n), slice(o, o + n))
        return x2[sl], T2[sl], W2[sl]
    p = n - m
    pad = (p // 2, p - p // 2) * 3
    return (F.pad(x2, pad), F.pad(T2, pad), F.pad(W2, pad))


# --------------------------------------------------------------- train
def make_batch(pool, bs, crop, srcw, device, torch, rng, aug=True):
    """Merged targets: T = sum W_s T_s / sum W_s, W = sum W_s (clipped 1)
    plus stacked per-source (T_s, W_s) for the BCE term."""
    xs, Ts, Ws = [], [], []
    for _ in range(bs):
        ct, srcs = pool.sample_crop(crop, srcw)
        Wsum = np.zeros_like(ct)
        Tsum = np.zeros_like(ct)
        for _, T, W in srcs:
            Wsum += W
            Tsum += W * T
        T = np.where(Wsum > 0, Tsum / np.maximum(Wsum, 1e-6), 0.0).astype(np.float32)
        xs.append(ct)
        Ts.append(T)
        Ws.append(Wsum)
    x = torch.from_numpy(np.stack(xs))[:, None].to(device, non_blocking=True)
    T = torch.from_numpy(np.stack(Ts))[:, None].to(device, non_blocking=True)
    W = torch.from_numpy(np.stack(Ws))[:, None].to(device, non_blocking=True)
    if aug:
        x = augment_batch(x, rng, torch)
        x, T, W = scale_jitter(x, T, W, rng, torch)
    return x, T, W


def weighted_loss(logits, T, W, torch):
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits.float(), T, reduction="none")
    wsum = W.sum().clamp_min(1.0)
    l_bce = (bce * W).sum() / wsum
    p = torch.sigmoid(logits.float()) * W
    t = T * W
    num = 2.0 * (p * t).sum() + 1.0
    den = p.sum() + t.sum() + 1.0
    return l_bce + 0.5 * (1.0 - num / den), l_bce


def validate_buckets(net, va, device, torch, max_per=8, crop=BLOCK, srcw=None):
    """Per-bucket dice/MAE against the raster labels (weight>0) and, when
    present, against the teacher."""
    rows = []
    net.eval()
    for k in sorted(va):
        files = va[k][:max_per]
        st = {"raster": [0.0, 0.0, 0.0, 0], "teacher": [0.0, 0.0, 0.0, 0]}
        for f in files:
            s = load_sample(f, k[2])
            ct = s["ct"].astype(np.float32)
            S = ct.shape[0]
            o = (S - crop) // 2
            ct = ct[o:o + crop, o:o + crop, o:o + crop]
            x = torch.from_numpy(ct)[None, None].to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(net(x).float())[0, 0].cpu().numpy()
            for name, T, W in s["src"]:
                if name not in st:
                    continue
                T = T[o:o + crop, o:o + crop, o:o + crop].astype(np.float32)
                W = W[o:o + crop, o:o + crop, o:o + crop].astype(np.float32)
                m = W > 0
                if m.sum() < 100:
                    continue
                pm, tm = (p >= 0.5) & m, (T >= 0.5) & m
                st[name][0] += 2.0 * (pm & tm).sum()
                st[name][1] += pm.sum() + tm.sum()
                st[name][2] += float(np.abs(p[m] - T[m]).mean())
                st[name][3] += 1
        row = {"bucket": k, "n": len(files)}
        for name, (dn, dd, mae, n) in st.items():
            if n:
                row[name] = (dn / max(dd, 1.0), mae / n, n)
        rows.append(row)
    net.train()
    return rows


def print_val(rows, step):
    print(f"val@{step}:")
    for r in rows:
        k = r["bucket"]
        parts = []
        for name in ("raster", "teacher"):
            if name in r:
                d, m, n = r[name]
                parts.append(f"{name} dice {d:.3f} mae {m:.3f} (n={n})")
        print(f"  {k[0]:14s} {k[1]} {k[2]:9s} " + "  ".join(parts), flush=True)


def summary_dice(rows):
    ds = [r["raster"][0] for r in rows if "raster" in r]
    dt = [r["teacher"][0] for r in rows if "teacher" in r]
    return (float(np.mean(ds)) if ds else 0.0), (float(np.mean(dt)) if dt else 0.0)


def parse_srcw(s):
    out = {"raster": 1.0, "teacher": 1.0, "dino": 0.3}
    for kv in (s or "").split(","):
        if "=" in kv:
            k, v = kv.split("=")
            out[k.strip()] = float(v)
    return out


def cmd_train(args):
    import torch
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    holdout = [h for h in args.holdout.split(",") if h]
    excl = [e for e in args.exclude_segments.split(",") if e]
    idx = build_index(args.root, args.pairs, holdout=holdout, exclude_segments=excl)
    tr, va = split_index(idx)
    ntr = sum(len(v) for v in tr.values())
    print(f"train: {len(tr)} buckets, {ntr} train blocks, {sum(len(v) for v in va.values())} val "
          f"(holdout {holdout})")
    for k in sorted(tr):
        print(f"  {k[0]:14s} {k[1]} {k[2]:9s} {len(tr[k])}")
    srcw = parse_srcw(args.w_src)
    print(f"train: source weights {srcw}")

    if args.features:
        cfg = {"features": [int(x) for x in args.features.split(",")], "blocks": args.blocks,
               "in_channels": 1}
    else:
        cfg = dict(studentnet.PRESETS[args.preset], in_channels=1)
    net = studentnet.build(cfg).to(device)
    ema = studentnet.StudentNet(cfg).to(device)
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 250

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, args.steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    start, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck.get("student_config") != cfg:
            sys.exit(f"train: resume config {ck.get('student_config')} != {cfg}")
        net.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema_model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("step", 0)) + 1
        best = float(ck.get("val", {}).get("score", -1.0))
        for _ in range(start):
            sched.step()
        print(f"train: resumed from step {start - 1}, best {best:.3f}")

    pool = Pool(tr, args.cache_blocks, rng, nthreads=args.loaders)
    ema_d = 0.999
    t_last = time.time()
    loss_acc = n_acc = 0.0
    for step in range(start, args.steps):
        x, T, W = make_batch(pool, args.bs, args.crop, srcw, device, torch, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
        loss, l_bce = weighted_loss(logits, T, W, torch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if args.sleep > 0:
            torch.cuda.synchronize()
            time.sleep(args.sleep)
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1.0 - ema_d)
        loss_acc += loss.item()
        n_acc += 1
        if (step + 1) % 100 == 0:
            now = time.time()
            r = (now - t_last) / n_acc
            t_last = now
            print(f"train: step {step + 1}/{args.steps} loss {loss_acc / n_acc:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {r:.2f}s/step eta "
                  f"{(args.steps - step - 1) * r / 60:.0f}m ({pool.loaded} blocks seen)", flush=True)
            loss_acc = n_acc = 0.0
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            rows = validate_buckets(ema, va, device, torch, max_per=args.val_per, crop=args.crop)
            print_val(rows, step + 1)
            dr, dt = summary_dice(rows)
            score = dr if dr > 0 else dt
            tag = ""
            if score > best:
                best = score
                torch.save({"student_config": cfg, "model": net.state_dict(),
                            "ema_model": ema.state_dict(), "step": step,
                            "optimizer": opt.state_dict(),
                            "val": {"score": score, "raster_dice": dr, "teacher_dice": dt}},
                           args.out)
                tag = f" -> saved {args.out}"
            else:
                torch.save({"student_config": cfg, "model": net.state_dict(),
                            "ema_model": ema.state_dict(), "step": step,
                            "optimizer": opt.state_dict(),
                            "val": {"score": score, "raster_dice": dr, "teacher_dice": dt}},
                           str(args.out) + ".last")
            print(f"val@{step + 1}: mean raster dice {dr:.3f} teacher dice {dt:.3f}{tag}", flush=True)
    pool.quit = True


def cmd_val(args):
    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = studentnet.build(ck["student_config"]).cuda()
    net.load_state_dict(ck["ema_model"])
    idx = build_index(args.root, args.pairs, holdout=())
    tr, va = split_index(idx)
    rows = validate_buckets(net, va, "cuda", torch, max_per=args.val_per, crop=args.crop)
    print_val(rows, ck.get("step", -1) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-teacher")
    a.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
    a.add_argument("--sample", default="PHercParis4")
    a.add_argument("--grid", default="F,C")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_add_teacher)
    i = sub.add_parser("index")
    i.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
    i.add_argument("--pairs", default=os.path.expanduser("~/r3d-data/ink3d-distill"))
    i.add_argument("--holdout", default="PHerc0139")
    i.set_defaults(fn=cmd_index)
    t = sub.add_parser("train")
    t.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
    t.add_argument("--pairs", default=os.path.expanduser("~/r3d-data/ink3d-distill"),
                   help="distill.py 256^3 teacher pairs dir ('' to disable)")
    t.add_argument("--holdout", default="PHerc0139")
    t.add_argument("--exclude-segments", default="")
    t.add_argument("--out", required=True)
    t.add_argument("--preset", default="small")
    t.add_argument("--features")
    t.add_argument("--blocks", type=int, default=1)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--bs", type=int, default=2)
    t.add_argument("--crop", type=int, default=112)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-every", type=int, default=1000)
    t.add_argument("--val-per", type=int, default=8)
    t.add_argument("--cache-blocks", type=int, default=192)
    t.add_argument("--loaders", type=int, default=4)
    t.add_argument("--w-src", default="", help="e.g. raster=1,teacher=0.5,dino=0.3")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--sleep", type=float, default=0.0)
    t.add_argument("--resume")
    t.set_defaults(fn=cmd_train)
    v = sub.add_parser("val")
    v.add_argument("--root", default=str(corpus.DEFAULT_ROOT))
    v.add_argument("--pairs", default="")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--crop", type=int, default=112)
    v.add_argument("--val-per", type=int, default=16)
    v.set_defaults(fn=cmd_val)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
