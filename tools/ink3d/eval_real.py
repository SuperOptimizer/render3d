"""Honest evaluation against human ink labels (bruniss train_ink_eval).

The set is 100 crops of (65, 512, 512) surface-volume stacks from PHerc1667
2um segments (w013/w018/w023/w028), with a binary ink label and a
supervision mask on the centre layer (z = 32). A 3D ink model is run on the
whole stack (the stack is CT resampled along surface normals, so it is a
mildly warped volume) and its probability around the label layer is scored
against the label inside the mask: dice@0.5, precision/recall, and AUC.

Works for the published teacher and for students (loaded via ink3d.load_model).

CLI:
  eval_real.py --ckpt ~/r3d-data/ink3d-ckpt/student-small.pth [--n 100] [--zpad 2]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ink3d  # noqa: E402
from distill import normalize  # noqa: E402

DEFAULT_SET = Path(os.path.expanduser("~/r3d-data/ink-corpus/eval/train_ink_eval"))
LABEL_Z = 32


def auc_score(scores, labels):
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = labels > 0
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_stack(net, stack, device, pad_to=16, tile=None):
    """stack u8 (Z,H,W) -> prob f32 (Z,H,W)."""
    import torch
    Z, H, W = stack.shape
    Zp = (Z + pad_to - 1) // pad_to * pad_to
    x = np.zeros((Zp, H, W), np.float32)
    x[:Z] = normalize(stack)
    x[Z:] = x[Z - 1:Z]
    xt = torch.from_numpy(x)[None, None].to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            out = net(xt)
            p = torch.sigmoid(out["ink"].float())[0, 0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            p = torch.zeros((Zp, H, W), device=device)
            t = tile or 256
            for y in range(0, H, t):
                for xx in range(0, W, t):
                    o = net(xt[:, :, :, y:y + t, xx:xx + t])
                    p[:, y:y + t, xx:xx + t] = torch.sigmoid(o["ink"].float())[0, 0]
    return p[:Z].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default=str(DEFAULT_SET))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--zpad", type=int, default=2, help="average prob over z in [32-zpad, 32+zpad]")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--report", help="append a json line here")
    args = ap.parse_args()
    import tifffile
    import torch
    root = Path(args.set)
    man = json.load(open(root / "manifest.json"))
    crops = man["crops"][:args.n]
    net = ink3d.load_model(args.ckpt)
    device = "cuda"
    per_vol = {}
    tot = {"dn": 0.0, "dd": 0.0, "tp": 0, "fp": 0, "fn": 0}
    all_s, all_l = [], []
    t0 = time.time()
    for i, c in enumerate(crops):
        img = tifffile.imread(root / "images" / c["image_tif"])
        lab = tifffile.imread(root / "labels" / c["label_tif"])[LABEL_Z] > 0
        msk = tifffile.imread(root / "supervision_masks" / c["label_tif"])[LABEL_Z] > 0
        if msk.sum() == 0:
            continue
        p = run_stack(net, img, device)
        z0, z1 = LABEL_Z - args.zpad, LABEL_Z + args.zpad + 1
        s = p[z0:z1].mean(axis=0)
        sm, lm = s[msk], lab[msk]
        pm = sm >= args.thresh
        tp, fp, fn = (pm & lm).sum(), (pm & ~lm).sum(), (~pm & lm).sum()
        v = per_vol.setdefault(c["volume"], {"dn": 0.0, "dd": 0.0, "tp": 0, "fp": 0, "fn": 0, "n": 0,
                                             "s": [], "l": []})
        for d in (v, tot):
            d["dn"] += 2.0 * tp
            d["dd"] += pm.sum() + lm.sum()
            d["tp"] += int(tp)
            d["fp"] += int(fp)
            d["fn"] += int(fn)
        v["n"] += 1
        sub = np.random.default_rng(i).choice(len(sm), min(len(sm), 20000), replace=False)
        v["s"].append(sm[sub])
        v["l"].append(lm[sub])
        all_s.append(sm[sub])
        all_l.append(lm[sub])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(crops)} ({time.time() - t0:.0f}s)", flush=True)

    def fmt(d, s=None, l=None):
        dice = d["dn"] / max(d["dd"], 1.0)
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec = d["tp"] / max(d["tp"] + d["fn"], 1)
        auc = auc_score(np.concatenate(s), np.concatenate(l)) if s else float("nan")
        return dice, prec, rec, auc

    print(f"eval-real: {args.ckpt} ({len(crops)} crops, thresh {args.thresh}, z+-{args.zpad})")
    print(f"  {'volume':28s} {'n':>3s} {'dice':>6s} {'prec':>6s} {'rec':>6s} {'auc':>6s}")
    for vol, d in sorted(per_vol.items()):
        dice, prec, rec, auc = fmt(d, d["s"], d["l"])
        print(f"  {vol:28s} {d['n']:3d} {dice:6.3f} {prec:6.3f} {rec:6.3f} {auc:6.3f}")
    dice, prec, rec, auc = fmt(tot, all_s, all_l)
    print(f"  {'ALL':28s} {len(crops):3d} {dice:6.3f} {prec:6.3f} {rec:6.3f} {auc:6.3f}")
    if args.report:
        with open(args.report, "a") as f:
            f.write(json.dumps({"ckpt": args.ckpt, "n": len(crops), "dice": dice, "prec": prec,
                                "rec": rec, "auc": auc, "thresh": args.thresh}) + "\n")


if __name__ == "__main__":
    main()
