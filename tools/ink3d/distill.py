#!/usr/bin/env python3
"""Distill the 3D ink teacher into small student models.

The published full-scroll ink-3d prediction zarr (and/or the local ink3d
cache zarr) provides teacher soft labels; students are scalable 3D residual
U-Nets from studentnet.py trained on (CT chunk -> teacher probability)
pairs with soft BCE + soft Dice. Run from villa's ink-detection env.

    # 1. sample training pairs (resumable; ~17MB/pair on disk)
    distill.py harvest --out ~/r3d-data/ink3d-distill [--n-ink 256 --n-bg 64]

    # 2. train a student (tiny ~1.3M params ~111x smaller, small ~15M ~9x)
    distill.py train --data ~/r3d-data/ink3d-distill \
        --out ~/r3d-data/ink3d-ckpt/student-tiny.pth --preset tiny
    #    ... or over the WHOLE scroll: chunks stream from S3 as training
    #    runs (~1 fresh chunk/s/loader, write-through cached up to a cap)
    distill.py train --stream --data ~/r3d-data/ink3d-distill \
        --out ~/r3d-data/ink3d-ckpt/student-small.pth --preset small --steps 100000

    # 3. metrics vs the teacher on the held-out split
    distill.py eval --data ~/r3d-data/ink3d-distill \
        --ckpt ~/r3d-data/ink3d-ckpt/student-tiny.pth

    # 4. use it everywhere ink3d.py is used (CLI --ckpt or GUI via env)
    ink3d.py infer --ckpt ~/r3d-data/ink3d-ckpt/student-tiny.pth ...
    R3D_INK3D_CKPT=~/r3d-data/ink3d-ckpt/student-tiny.pth ~/render3d-gpu ...

Chunks are scored for ink via the teacher zarr's level-5 pyramid (one 256^3
level-0 chunk is an 8^3 block there), so harvesting finds ink-dense regions
across the whole scroll without downloading full-resolution data first.
Background (papyrus-adjacent, ink-free) chunks are mixed in so students
learn silence too. The held-out split is deterministic (chunk-key hash).
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_TEACHER = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/"
                   "PHercParis4/representations/predictions/ink-3d/"
                   "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr")
DEFAULT_SOURCE = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/"
                  "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
CHUNK = 256


def open_level(url, level):
    import zarr
    if "://" in url:
        import fsspec
        return zarr.open(fsspec.get_mapper(f"{url}/{level}"), mode="r")
    return zarr.open(str(Path(url) / str(level)), mode="r")


def normalize(chunk):
    """Per-256^3-chunk p1/p99 min-max, matching ink3d.py inference."""
    lo, hi = np.percentile(chunk[::2, ::2, ::2], (1.0, 99.0))
    d = float(hi - lo)
    if d <= 1e-8:
        return np.zeros_like(chunk, dtype=np.float32)
    return ((np.clip(chunk, lo, hi) - lo) / d).astype(np.float32)


def is_val(cz, cy, cx):
    return (cz * 73856093 ^ cy * 19349663 ^ cx * 83492791) % 8 == 0


# ---------------------------------------------------------------- harvest

def score_chunks(teacher, out_dir):
    """Per-256^3-chunk mean teacher prediction over the whole volume, from
    the L5 pyramid (one L0 chunk = an 8^3 block). Cached as score.npy."""
    cache = Path(out_dir) / "score.npy"
    if cache.exists():
        return np.load(cache)
    t5 = open_level(teacher, 5)
    t0 = open_level(teacher, 0)
    Z, Y, X = t0.shape
    gz, gy, gx = (Z + CHUNK - 1) // CHUNK, (Y + CHUNK - 1) // CHUNK, (X + CHUNK - 1) // CHUNK
    print(f"distill: scoring {gz}x{gy}x{gx} chunks via teacher L5 {t5.shape}")
    score = np.zeros((gz, gy, gx), dtype=np.float32)
    for z0 in range(0, t5.shape[0], 256):  # one chunk-row slab at a time
        sl = t5[z0:z0 + 256].astype(np.float32)
        sz = sl.shape[0] - sl.shape[0] % 8
        if sz == 0:
            continue
        sl = sl[:sz, :t5.shape[1] // 8 * 8, :t5.shape[2] // 8 * 8]
        b = sl.reshape(sz // 8, 8, sl.shape[1] // 8, 8, sl.shape[2] // 8, 8)
        blk = b.mean(axis=(1, 3, 5))
        # the published pred zarr fills no-data space with ~127 (sigmoid(0),
        # smeared to 124..127 by its pyramid): a block whose MINIMUM is that
        # high is outside the scan mask -- real ink chunks always contain
        # near-zero background papyrus
        blk[b.min(axis=(1, 3, 5)) >= 120] = 0.0
        score[z0 // 8:z0 // 8 + blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
    np.save(cache, score)
    return score


def fetch_pair(s0, t0, c):
    """(CT u8, teacher u8) 256^3 pair for chunk c, or None if the CT is
    empty. Mask fill (~127) over empty CT is zeroed: it is not a label."""
    Z, Y, X = t0.shape
    cz, cy, cx = c
    z0, y0, x0 = cz * CHUNK, cy * CHUNK, cx * CHUNK
    ct = np.zeros((CHUNK, CHUNK, CHUNK), np.uint8)
    pr = np.zeros((CHUNK, CHUNK, CHUNK), np.uint8)
    a = s0[z0:min(z0 + CHUNK, Z), y0:min(y0 + CHUNK, Y), x0:min(x0 + CHUNK, X)]
    ct[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    if ct.max() == 0:
        return None
    b = t0[z0:min(z0 + CHUNK, Z), y0:min(y0 + CHUNK, Y), x0:min(x0 + CHUNK, X)]
    pr[:b.shape[0], :b.shape[1], :b.shape[2]] = b
    pr[ct == 0] = 0
    return ct, pr


def save_pair(f, ct, pr):
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ct=ct, pred=pr)
    tmp.rename(f)


def cmd_harvest(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = open_level(args.teacher, 0)
    s0 = open_level(args.source, 0)
    score = score_chunks(args.teacher, out)
    gz, gy, gx = score.shape
    ink_mask = score >= args.ink_thresh
    print(f"distill: {ink_mask.sum()} ink-bearing chunks "
          f"(score >= {args.ink_thresh}), max score {score.max():.1f}")
    ink_idx = np.argwhere(ink_mask)
    w = score[ink_mask] ** 0.5
    # oversample ~1.4x: mask-boundary chunks with empty CT get skipped at
    # download time, so extra candidates keep the yield near --n-ink
    n_ink = min(int(args.n_ink * 1.4), len(ink_idx))
    pick = rng.choice(len(ink_idx), size=n_ink, replace=False, p=w / w.sum())
    chosen = [tuple(map(int, ink_idx[i])) for i in pick]
    # background: papyrus-adjacent silent chunks (neighbours of ink chunks)
    bg = set()
    cset = set(chosen)
    tries = 0
    while len(bg) < args.n_bg and tries < args.n_bg * 50:
        tries += 1
        cz, cy, cx = chosen[rng.integers(n_ink)]
        d = rng.integers(-2, 3, size=3)
        c = (cz + int(d[0]), cy + int(d[1]), cx + int(d[2]))
        if (0 <= c[0] < gz and 0 <= c[1] < gy and 0 <= c[2] < gx
                and score[c] < args.ink_thresh / 4 and c not in cset):
            bg.add(c)
    chosen += sorted(bg)
    print(f"distill: harvesting {n_ink} ink + {len(bg)} background chunks")

    lock = threading.Lock()
    it = iter(chosen)
    stats = {"done": 0, "skip": 0, "empty": 0}

    def worker():
        while True:
            with lock:
                c = next(it, None)
            if c is None:
                return
            cz, cy, cx = c
            f = out / f"{cz}_{cy}_{cx}.npz"
            if f.exists():
                with lock:
                    stats["skip"] += 1
                continue
            pair = fetch_pair(s0, t0, c)
            if pair is None:
                with lock:
                    stats["empty"] += 1
                continue
            save_pair(f, *pair)
            with lock:
                stats["done"] += 1
                n = stats["done"]
            if n % 10 == 0:
                print(f"distill: {n} pairs fetched...", flush=True)

    ths = [threading.Thread(target=worker) for _ in range(6)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    (out / "manifest.json").write_text(json.dumps(
        {"teacher": args.teacher, "source": args.source,
         "ink_thresh": args.ink_thresh, "seed": args.seed}, indent=1))
    npairs = len(list(out.glob("*_*_*.npz")))
    print(f"distill: {stats['done']} new, {stats['skip']} existing, "
          f"{stats['empty']} empty-CT skipped -> {npairs} pairs in {out}")


# ---------------------------------------------------------------- data

class ChunkBuffer:
    """Replay buffer of normalized chunks; a loader thread cycles fresh
    chunks in from disk while training samples crops from what's loaded."""

    def __init__(self, load_one, cap, rng, nthreads=1):
        """load_one() -> (ct u8, pred u8) 256^3 or None (retry)."""
        self.load_one, self.rng = load_one, rng
        self.cap = cap
        self.buf = []  # list of (ct f16 norm, pred f16 prob)
        self.loaded = 0
        self.lock = threading.Lock()
        self.quit = False
        self.ths = [threading.Thread(target=self._loader, daemon=True)
                    for _ in range(nthreads)]
        for t in self.ths:
            t.start()
        while True:  # block until minimally warm
            with self.lock:
                if len(self.buf) >= min(8, self.cap):
                    break
            time.sleep(0.2)

    def _loader(self):
        while not self.quit:
            try:
                pair = self.load_one()
            except Exception as e:  # transient S3 hiccup: log, move on
                print(f"distill: loader error {e}", file=sys.stderr, flush=True)
                time.sleep(2)
                continue
            if pair is None:
                continue
            item = (normalize(pair[0]).astype(np.float16),
                    pair[1].astype(np.float16) / 255.0)
            with self.lock:
                if len(self.buf) >= self.cap:
                    self.buf.pop(0)
                self.buf.append(item)
                self.loaded += 1
            if len(self.buf) >= self.cap:
                time.sleep(0.5)  # steady state: ~2 fresh chunks/s per thread

    def sample_crop(self, crop):
        with self.lock:
            ct, pr = self.buf[self.rng.integers(len(self.buf))]
        z, y, x = (self.rng.integers(CHUNK - crop + 1) for _ in range(3))
        c = ct[z:z + crop, y:y + crop, x:x + crop]
        p = pr[z:z + crop, y:y + crop, x:x + crop]
        perm = list(self.rng.permutation(3))
        c, p = c.transpose(perm), p.transpose(perm)
        for ax in range(3):
            if self.rng.integers(2):
                c, p = np.flip(c, ax), np.flip(p, ax)
        return np.ascontiguousarray(c), np.ascontiguousarray(p)

    def stop(self):
        self.quit = True


def split_files(data_dir):
    files = sorted(Path(data_dir).glob("*_*_*.npz"))
    tr, va = [], []
    for f in files:
        cz, cy, cx = map(int, f.stem.split("_"))
        (va if is_val(cz, cy, cx) else tr).append(f)
    return tr, va


# ---------------------------------------------------------------- train

def soft_dice_loss(logits, target):
    import torch
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum() + 1.0
    den = p.sum() + target.sum() + 1.0
    return 1.0 - num / den


def validate(net, va_files, device, max_chunks=12):
    import torch
    mae = dice_n = dice_d = 0.0
    n = 0
    for f in va_files[:max_chunks]:
        d = np.load(f)
        x = torch.from_numpy(normalize(d["ct"]))[None, None].to(device)
        t = torch.from_numpy(d["pred"].astype(np.float32) / 255.0).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(net(x).float())[0, 0]
        mae += (p - t).abs().mean().item()
        pm, tm = p >= 0.5, t >= 0.5
        dice_n += 2.0 * (pm & tm).sum().item()
        dice_d += pm.sum().item() + tm.sum().item()
        n += 1
    return mae / max(n, 1), dice_n / max(dice_d, 1.0), n


def cmd_train(args):
    import torch
    import studentnet
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    tr, va = split_files(args.data)
    if not tr and not args.stream:
        sys.exit(f"distill: no training pairs in {args.data} (run harvest)")
    if not va:
        sys.exit(f"distill: no held-out pairs in {args.data} (run harvest first)")
    print(f"distill: {len(tr)} train / {len(va)} val chunks")
    if args.features:
        cfg = {"features": [int(v) for v in args.features.split(",")],
               "blocks": args.blocks, "in_channels": 1}
    else:
        cfg = dict(studentnet.PRESETS[args.preset], in_channels=1)
    net = studentnet.build(cfg).cuda()
    ema = studentnet.StudentNet(cfg).cuda()
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 250
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) *
        (0.5 * (1.0 + np.cos(np.pi * min(s, args.steps) / args.steps))))
    start, best_dice = 1, -1.0
    if args.resume:  # continue a killed run: weights, EMA, optimizer, step
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["student_config"] != cfg:
            sys.exit(f"distill: --resume config {ck['student_config']} != {cfg}")
        net.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema_model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("step", 0)) + 1
        best_dice = float(ck.get("val", {}).get("dice", -1.0))
        for _ in range(start - 1):
            sched.step()
        print(f"distill: resumed {args.resume} at step {start} "
              f"(best dice {best_dice:.3f})")
    if args.stream:
        # whole-scroll sampling: ink-weighted random chunks (+ ~20% silent
        # neighbours) from the teacher score grid, streamed from S3, with
        # write-through to the local shard dir up to --cache-limit-gb. The
        # held-out hash split is excluded so validation stays honest.
        data = Path(args.data)
        score = score_chunks(args.teacher, data)
        ink_idx = np.argwhere(score >= args.ink_thresh)
        w = score[tuple(ink_idx.T)] ** 0.5
        w /= w.sum()
        gz, gy, gx = score.shape
        s0, t0 = open_level(args.source, 0), open_level(args.teacher, 0)
        srng = np.random.default_rng(args.seed + 1)
        slock = threading.Lock()
        cache_bytes = [sum(f.stat().st_size for f in data.glob("*_*_*.npz"))]
        print(f"distill: streaming from {len(ink_idx)} ink-bearing chunks "
              f"(local cache {cache_bytes[0] / 1e9:.1f} GB, cap {args.cache_limit_gb} GB)")

        def load_stream():
            with slock:
                i = srng.choice(len(ink_idx), p=w)
                c = tuple(int(v) for v in ink_idx[i])
                if srng.random() < 0.2:  # silent neighbour
                    d = srng.integers(-2, 3, size=3)
                    c = (c[0] + int(d[0]), c[1] + int(d[1]), c[2] + int(d[2]))
            if not (0 <= c[0] < gz and 0 <= c[1] < gy and 0 <= c[2] < gx):
                return None
            if is_val(*c):
                return None
            f = data / f"{c[0]}_{c[1]}_{c[2]}.npz"
            if f.exists():
                d = np.load(f)
                return d["ct"], d["pred"]
            pair = fetch_pair(s0, t0, c)
            if pair is None:
                return None
            with slock:
                room = cache_bytes[0] < args.cache_limit_gb * 1e9
            if room:
                save_pair(f, *pair)
                with slock:
                    cache_bytes[0] += f.stat().st_size
            return pair

        buf = ChunkBuffer(load_stream, args.cache_chunks, rng, nthreads=args.loaders)
    else:
        def load_local():
            d = np.load(tr[rng.integers(len(tr))])
            return d["ct"], d["pred"]

        buf = ChunkBuffer(load_local, min(args.cache_chunks, len(tr)), rng)
    bce = torch.nn.BCEWithLogitsLoss()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ema_d = 0.999
    run_loss = None
    for step in range(start, args.steps + 1):
        xs, ts = zip(*(buf.sample_crop(args.crop) for _ in range(args.bs)))
        x = torch.from_numpy(np.stack(xs)).float()[:, None].cuda(non_blocking=True)
        t = torch.from_numpy(np.stack(ts)).float()[:, None].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x)
            loss = bce(y.float(), t) + 0.5 * soft_dice_loss(y.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1.0 - ema_d)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % 100 == 0:
            r = (time.time() - t_start) / (step - start + 1)
            print(f"distill: step {step}/{args.steps} loss {run_loss:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {r:.2f}s/step "
                  f"eta {(args.steps - step) * r / 60:.0f}m "
                  f"({buf.loaded} chunks seen)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            ema.eval()
            mae, dice, nv = validate(ema, va, "cuda")
            ema.train()
            tag = ""
            if dice > best_dice:
                best_dice = dice
                torch.save({"student_config": cfg, "model": net.state_dict(),
                            "ema_model": ema.state_dict(), "step": step,
                            "optimizer": opt.state_dict(),
                            "val": {"mae": mae, "dice": dice}}, out)
                tag = f" -> saved {out}"
            print(f"distill: val@{step}: dice {dice:.3f} mae {mae:.4f} "
                  f"({nv} chunks){tag}", flush=True)
    buf.stop()
    print(f"distill: done in {(time.time() - t_start) / 60:.1f}m, "
          f"best val dice {best_dice:.3f} -> {out}")


def cmd_eval(args):
    import torch
    import studentnet
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = studentnet.build(ck["student_config"]).cuda().eval()
    net.load_state_dict(ck.get("ema_model") or ck["model"])
    _, va = split_files(args.data)
    mae, dice, n = validate(net, va, "cuda", max_chunks=len(va))
    x = torch.rand(1, 1, 256, 256, 256, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        net(x)
        torch.cuda.synchronize()
        t0 = time.time()
        net(x)
        torch.cuda.synchronize()
    print(f"distill: {args.ckpt}: step {ck.get('step')} | val dice {dice:.3f} "
          f"mae {mae:.4f} ({n} chunks) | {time.time() - t0:.3f}s / 256^3 chunk")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="download (CT, teacher pred) chunk pairs")
    h.add_argument("--out", required=True)
    h.add_argument("--teacher", default=DEFAULT_TEACHER)
    h.add_argument("--source", default=DEFAULT_SOURCE)
    h.add_argument("--n-ink", type=int, default=256)
    h.add_argument("--n-bg", type=int, default=64)
    h.add_argument("--ink-thresh", type=float, default=4.0,
                   help="mean u8 pred over the chunk to count as ink-bearing")
    h.add_argument("--seed", type=int, default=7)
    h.set_defaults(fn=cmd_harvest)
    t = sub.add_parser("train", help="train a student on harvested pairs")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--preset", default="tiny", choices=sorted(
        __import__("studentnet").PRESETS))
    t.add_argument("--features", help="override: comma channel list")
    t.add_argument("--blocks", type=int, default=1)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--bs", type=int, default=4)
    t.add_argument("--crop", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-every", type=int, default=500)
    t.add_argument("--cache-chunks", type=int, default=64,
                   help="replay-buffer size in 256^3 chunks (~48MB each)")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--resume", help="checkpoint to continue from (same preset)")
    t.add_argument("--stream", action="store_true",
                   help="sample chunks from the WHOLE scroll via S3 instead of "
                        "only the harvested shards (write-through cached)")
    t.add_argument("--teacher", default=DEFAULT_TEACHER)
    t.add_argument("--source", default=DEFAULT_SOURCE)
    t.add_argument("--ink-thresh", type=float, default=4.0)
    t.add_argument("--cache-limit-gb", type=float, default=200.0,
                   help="stop growing the local shard cache past this size")
    t.add_argument("--loaders", type=int, default=4, help="stream fetch threads")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval", help="held-out metrics + runtime for a student")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", required=True)
    e.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
