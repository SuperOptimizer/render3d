#!/usr/bin/env python3
# Read one z-plane from a dct3d Zarr v3 export and save it as a PNG.
# Fetch the shard once; zarr-python + the dct3d codec decode it.
#
# Setup (from a clone of this repo):
#   pip install "zarr>=3" numpy cffi        # then, in the repo root:
#   pip install .                           # builds+installs dct3d-zarr
#   pip install pillow requests             # for this example
#
# Importing dct3d_zarr registers the "dct3d" codec with zarr (an entry point also
# auto-registers it, so a stock zarr.open resolves it without this import).
import io, json, os, tempfile, numpy as np, requests, zarr, dct3d_zarr  # noqa: F401
from PIL import Image

URL = ("https://dl.ash2txt.org/community-uploads/forrest/exports/PHerc0125/"
       "20250821151825-9.362um-1.2m-113keV-masked.zarr/0")
Z, SY, SX = 2560, 3, 4  # global z; shard column (y, x)

# One HTTP GET of the 1024^3 shard, wrapped as a single-shard local zarr store.
meta = requests.get(f"{URL}/zarr.json").json() | {"shape": [1024, 1024, 1024]}
d = tempfile.mkdtemp(); os.makedirs(f"{d}/c/0/0")
json.dump(meta, open(f"{d}/zarr.json", "w"))
open(f"{d}/c/0/0/0", "wb").write(requests.get(f"{URL}/c/{Z//1024}/{SY}/{SX}").content)

# zarr decodes the dct3d chunks transparently.
plane = zarr.open_array(d, mode="r")[Z % 1024, :, :]
Image.fromarray(plane, "L").save("slice.png")
print("wrote slice.png", plane.shape, "mean", round(float(plane.mean()), 1))
