#!/usr/bin/env python3
"""Create a surface "predict tree" for a CT LOD tree that has no published
surface predictions: a c5d LOD tree with the CT's geometry and no data, whose
source.json points render3d's consumers (renderer overlay ingest, tracer /
cpuvol) at tools/surf/surfserver.py instead of an HTTP zarr. Bricks are
predicted on demand and cached under <out>/bricks/L{0,1}.

    tools/surf/mkpredtree.py <ct-lod-root> <out-root> [--port 9744] [--th 0.2] [--margin 32]

Then:  surfserver.py surf-m7 &   render3d --bricks <ct>/manifest.json --overlay <out> ...
       tracecli <out> ...  (predictions appear as the tracer explores)
"""
import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("ct_root")
ap.add_argument("out_root")
ap.add_argument("--port", type=int, default=9744)
ap.add_argument("--th", type=float, default=0.2)
ap.add_argument("--margin", type=int, default=32)
ap.add_argument("--quality", type=float, default=2.0)
ap.add_argument("--pred-level", type=int, default=-1,
                help="CT level fed to the model (0 for 8-9um scans, 2 for 2.4um); "
                     "default: inferred from the CT source url's '...um' tag")
a = ap.parse_args()

ct = os.path.abspath(a.ct_root)
out = os.path.abspath(a.out_root)
man = json.load(open(os.path.join(ct, "manifest.json")))
os.makedirs(out, exist_ok=True)
# byte-for-byte: the renderer's overlay check matches the manifest's shape
# text literally, so the copy must keep the CT manifest's formatting
shutil.copyfile(os.path.join(ct, "manifest.json"), os.path.join(out, "manifest.json"))
nlev = len(man["levels"])
P = a.pred_level
if P < 0:
    um = None
    try:
        import re
        src0 = json.load(open(os.path.join(ct, "source.json")))
        m = re.search(r"(\d+(?:\.\d+)?)um", src0.get("url", ""))
        um = float(m.group(1)) if m else None
    except Exception:
        pass
    # the model's native pitch is ~8-9um: pick the pyramid level closest to it
    P = 0 if um is None else max(0, min(nlev - 2, round(__import__("math").log2(8.6 / um)))) if um and um < 8.6 else 0
    print(f"inferred pred_level {P} (voxel {um} um)" if um else "no um tag: pred_level 0")
levels = [{"level": l, "chunk": 256 if l == P else 128, "raw": False} for l in range(nlev)]
src = {
    "format": "render3d.c5d-source.v1",
    "url": f"predict://127.0.0.1:{a.port}",
    "quality": a.quality,
    "levels": levels,
    "ct_root": ct,
    "pred_level": P,
    "th": a.th,
    "margin": a.margin,
    "note": "surface predictions produced on demand by tools/surf/surfserver.py",
}
with open(os.path.join(out, "source.json"), "w") as f:
    json.dump(src, f, indent=2)
print(f"predict tree {out}: {nlev} levels, CT {ct}, pred_level {P}, server port {a.port}, th {a.th}")
