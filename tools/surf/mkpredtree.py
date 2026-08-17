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
a = ap.parse_args()

ct = os.path.abspath(a.ct_root)
out = os.path.abspath(a.out_root)
man = json.load(open(os.path.join(ct, "manifest.json")))
os.makedirs(out, exist_ok=True)
# byte-for-byte: the renderer's overlay check matches the manifest's shape
# text literally, so the copy must keep the CT manifest's formatting
shutil.copyfile(os.path.join(ct, "manifest.json"), os.path.join(out, "manifest.json"))
nlev = len(man["levels"])
levels = [{"level": l, "chunk": 256 if l == 0 else 128, "raw": False} for l in range(nlev)]
src = {
    "format": "render3d.c5d-source.v1",
    "url": f"predict://127.0.0.1:{a.port}",
    "quality": a.quality,
    "levels": levels,
    "ct_root": ct,
    "th": a.th,
    "margin": a.margin,
    "note": "surface predictions produced on demand by tools/surf/surfserver.py",
}
with open(os.path.join(out, "source.json"), "w") as f:
    json.dump(src, f, indent=2)
print(f"predict tree {out}: {nlev} levels, CT {ct}, server port {a.port}, th {a.th}")
