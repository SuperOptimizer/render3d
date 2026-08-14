#!/usr/bin/env python3
"""Wrap a rendseg --layers raw stack as the zarr layout
koine_machines.inference.infer expects (matches prepare_9um_isotropic_input.py:
group with dataset "0", (Z,H,W) uint8, chunks (Z,128,128), zstd-5 bitshuffle)."""
import json
import sys

import numpy as np
import zarr
from numcodecs import Blosc

raw, out = sys.argv[1], sys.argv[2]
meta = json.load(open(raw + ".json"))
Z, H, W = meta["layers"], meta["height"], meta["width"]
a = np.fromfile(raw, dtype=np.uint8).reshape(Z, H, W)
g = zarr.open_group(out, mode="w")
g.attrs.update({"format": "render3d-rendseg-layers-v1", "surface_layer": meta["surface_layer"]})
d = g.create_dataset("0", shape=(Z, H, W), chunks=(Z, min(128, H), min(128, W)),
                     dtype=np.uint8,
                     compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
                     fill_value=0)
d[:] = a
print(f"wrote {out} ({Z}x{H}x{W})")
