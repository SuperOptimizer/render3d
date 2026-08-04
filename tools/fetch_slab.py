#!/usr/bin/env python3
"""Fetch a contiguous region of the PHercParis4 masked zarr into a flat .u8
volume (spec/volume.md order: x-fastest) + a .json sidecar with the dims.

Usage: tools/fetch_slab.py out.u8 Z0 NZ Y0 NY X0 NX
       (voxel offsets/extents at level 0; reads are chunk-parallel via zarr)

Resumable: skips if the output already exists with the right size.
"""
import json
import pathlib
import sys

import zarr  # zarr-python 3.x

URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/"
       "20260608103018-1.129um-0.2m-78keV-masked.zarr/0")


def main() -> int:
    if len(sys.argv) != 8:
        print(__doc__)
        return 1
    out = pathlib.Path(sys.argv[1])
    z0, nz, y0, ny, x0, nx = (int(v) for v in sys.argv[2:8])
    total = nx * ny * nz
    if out.exists() and out.stat().st_size == total:
        print(f"{out} already complete ({total} bytes)")
        return 0

    arr = zarr.open(URL, mode="r")
    print(f"source {arr.shape} chunks {arr.chunks}; fetching "
          f"z[{z0}:{z0+nz}) y[{y0}:{y0+ny}) x[{x0}:{x0+nx}) "
          f"= {total / 2**30:.2f} GiB raw")

    cz = arr.chunks[0]
    with open(out, "wb") as f:
        # stream in z-chunk-aligned slabs to bound memory
        z = z0
        while z < z0 + nz:
            zhi = min(((z // cz) + 1) * cz, z0 + nz)
            block = arr[z:zhi, y0:y0 + ny, x0:x0 + nx]
            f.write(block.tobytes())  # zarr returns C-order [z][y][x]
            done = zhi - z0
            print(f"  {done}/{nz} slices", flush=True)
            z = zhi
    json.dump({"nx": nx, "ny": ny, "nz": nz, "z0": z0, "y0": y0, "x0": x0,
               "url": URL, "voxel_um": 1.129}, open(out.with_suffix(".json"), "w"))
    print(f"wrote {out} ({total} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
