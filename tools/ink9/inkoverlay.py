#!/usr/bin/env python3
"""Overlay an ink prediction tif on the segment's surface CT (layer 10 of the
rendseg stack). Applies the (p-0.25)/0.5 display rescale from the model card."""
import json
import sys

import numpy as np
import tifffile
from PIL import Image, ImageOps

raw, pred, out = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.load(open(raw + ".json"))
Z, H, W = meta["layers"], meta["height"], meta["width"]
a = np.fromfile(raw, dtype=np.uint8).reshape(Z, H, W)
base = np.asarray(ImageOps.autocontrast(Image.fromarray(a[meta["surface_layer"]]), cutoff=1),
                  dtype=np.float32)
p = tifffile.imread(pred).astype(np.float32) / 255.0
disp = np.clip((p - 0.25) / 0.5, 0.0, 1.0)
if disp.shape != base.shape:
    disp = np.asarray(Image.fromarray(disp).resize((W, H), Image.BILINEAR))
print(f"pred: raw mean {p.mean():.3f} p99 {np.percentile(p,99):.3f}; "
      f"display mean {disp.mean():.3f} >0.5 {(disp>0.5).mean()*100:.2f}%")
rgb = np.stack([base, base, base], axis=-1)
ink = np.zeros_like(rgb)
ink[..., 0] = 235.0  # amber-red ink tint
ink[..., 1] = 120.0
al = (disp * 0.85)[..., None]
outi = (rgb * (1 - al) + ink * al).astype(np.uint8)
Image.fromarray(outi).save(out)
# side-by-side ink-only map too
Image.fromarray((disp * 255).astype(np.uint8)).save(out.replace('.png', '-ink.png'))
print("wrote", out)
