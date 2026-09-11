#!/usr/bin/env python3
"""Overwrite a TSM `fine.zarr` label store's `sdf` / `sdf_valid` channels from
refined render3d surfaces, copy-on-write.

    export_tsm_labels.py --store labels/fine.zarr --out labels/fine_surf.zarr \
        --samples a.bin [b.bin ...] [--clip 20] [--conf-min 0.25] \
        [--axis axis.json | --origin-sign outward] [--band 20]

Run it in the tsm project's environment (it imports `tsm.volume.BrickWriter`
and `tsm.labels`), e.g. from `build/tsm-source`:

    uv run python .../export_tsm_labels.py --store ... --out ... --samples ...

The samples are `surfsamples` output: a 16-byte header ("R3DS", uint32 count,
two reserved uint32) followed by `count` records of seven little-endian
float32 -- x, y, z, nx, ny, nz, conf -- with x/y/z in full-resolution voxels
of the same volume the store covers, and the normal in the "+v x +u" grid
convention (see surfsamples.c; which physical side that is depends on the
surface's grid handedness, so it is re-oriented here).

## What is written

`--out` is created as a fresh store with the source store's channels, shape,
chunking, origin, voxel size and scale (`BrickWriter` with those attrs -- the
same identity check `tsm labels` applies), then **every** 128^3 chunk of the
source is copied through unchanged, except that in each chunk touched by a
sample the `sdf` and `sdf_valid` channels are replaced inside the band:

* `sdf` = the signed distance in voxels to the nearest sample, clipped to
  +-`--clip` and encoded `u8 = round(128 + sdf*127/clip)` in 1..255
  (`tsm.labels.encode_sdf_u8`; 0 stays reserved for no-data);
* `sdf_valid` = 1 where the nearest sample's confidence is at least
  `--conf-min`, 0 otherwise;
* voxels farther than `--band` voxels from every sample keep the source
  store's `sdf` / `sdf_valid` bytes, and chunks with no sample within the
  band are byte-identical copies of the source.

## Sign convention (label_store.md)

Positive on the *outward* side, away from the scroll axis:
`sign = sign((p - s) . n_s)` with the sample normal `n_s` first flipped so
that `n_s . r_hat >= 0`, where `r_hat` is the outward radial direction at the
sample (the axis is vertical in z, so `r_hat` is the in-plane direction from
the interpolated axis (y, x) at the sample's z to the sample).

`--axis axis.json` is read exactly as `tsm.labels.load_axis` reads it
(control points in full-resolution xyz voxels).  Without it, `--origin-sign
outward` uses a single straight axis through the store's own `origin_zyx`
(the (y, x) of the store origin, extended over all z) as a crude fallback --
adequate only when the store's origin really is near the axis; a warning is
printed either way.

Refuses `--out` == `--store`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np
import zarr

from tsm.labels import encode_sdf_u8, load_axis
from tsm.volume import BrickWriter

SAMPLE_MAGIC = b"R3DS"
SAMPLE_HEADER = 16
SAMPLE_STRIDE = 28  # 7 float32


def read_samples(path: str) -> np.ndarray:
    """(N, 7) float64 array x,y,z,nx,ny,nz,conf from a surfsamples .bin/.csv."""
    if path.endswith(".csv"):
        rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        if rows.size and rows.shape[1] != 7:
            raise ValueError(f"{path}: expected 7 columns, got {rows.shape[1]}")
        return rows.reshape(-1, 7)
    with open(path, "rb") as fh:
        head = fh.read(SAMPLE_HEADER)
        if len(head) < SAMPLE_HEADER or head[:4] != SAMPLE_MAGIC:
            raise ValueError(f"{path}: not a surfsamples binary file")
        count = struct.unpack("<I", head[4:8])[0]
        body = fh.read(count * SAMPLE_STRIDE)
    if len(body) != count * SAMPLE_STRIDE:
        raise ValueError(f"{path}: truncated ({len(body)} bytes for {count} samples)")
    return np.frombuffer(body, dtype="<f4").reshape(count, 7).astype(np.float64)


def orient_normals(pts_xyz: np.ndarray, nrm_xyz: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Flip each normal so that n . r_hat >= 0 (label_store.md's outward rule).

    `axis` is (N, 3) in (z, y, x) as `tsm.labels.load_axis` returns it.
    """
    z = pts_xyz[:, 2]
    ay = np.interp(z, axis[:, 0], axis[:, 1])
    ax = np.interp(z, axis[:, 0], axis[:, 2])
    ry = pts_xyz[:, 1] - ay
    rx = pts_xyz[:, 0] - ax
    rn = np.sqrt(ry * ry + rx * rx)
    rn[rn == 0] = 1.0
    # r_hat has no z component, so the dot product is the in-plane part only
    dot = (nrm_xyz[:, 0] * rx + nrm_xyz[:, 1] * ry) / rn
    flip = np.where(dot < 0, -1.0, 1.0)[:, None]
    return nrm_xyz * flip


def fallback_axis(origin_zyx, shape_zyx) -> np.ndarray:
    """A straight vertical axis through the store origin's (y, x)."""
    z0 = float(origin_zyx[0])
    z1 = z0 + float(shape_zyx[0])
    y, x = float(origin_zyx[1]), float(origin_zyx[2])
    return np.array([[z0 - 1.0, y, x], [z1 + 1.0, y, x]], dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="source fine.zarr (never modified)")
    ap.add_argument("--out", required=True, help="destination store (created)")
    ap.add_argument("--samples", required=True, nargs="+", help="surfsamples .bin/.csv files")
    ap.add_argument("--clip", type=float, default=20.0, help="sdf clip in voxels (default 20)")
    ap.add_argument("--conf-min", type=float, default=0.25,
                    help="sdf_valid=1 only where the nearest sample's conf is at least this")
    ap.add_argument("--band", type=float, default=20.0,
                    help="only voxels within this many voxels of a sample are rewritten")
    ap.add_argument("--axis", help="umbilicus axis json (tsm.labels.load_axis format)")
    ap.add_argument("--origin-sign", choices=["outward"],
                    help="fallback: use a straight axis through the store origin")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args(argv)

    store = os.path.abspath(os.path.expanduser(args.store))
    out = os.path.abspath(os.path.expanduser(args.out))
    if store == out:
        print("export_tsm_labels: --out must differ from --store (copy-on-write only)",
              file=sys.stderr)
        return 2
    if args.clip <= 0 or args.band <= 0:
        print("export_tsm_labels: --clip and --band must be positive", file=sys.stderr)
        return 2
    if os.path.exists(out):
        if not args.force:
            print(f"export_tsm_labels: {out} exists (pass --force to replace)", file=sys.stderr)
            return 2
        shutil.rmtree(out)

    src = zarr.open_array(store=store, mode="r")
    attrs = dict(src.attrs)
    channels = list(attrs.get("channels", []))
    if "sdf" not in channels or "sdf_valid" not in channels:
        print(f"export_tsm_labels: {store} has no sdf/sdf_valid channels ({channels})",
              file=sys.stderr)
        return 2
    c_sdf, c_val = channels.index("sdf"), channels.index("sdf_valid")
    shape_zyx = tuple(int(v) for v in src.shape[1:])
    origin_zyx = tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0)))
    chunk = int(src.chunks[1])

    # --- samples -------------------------------------------------------- #
    rows = [read_samples(p) for p in args.samples]
    rows = [r for r in rows if r.size]
    if not rows:
        print("export_tsm_labels: no samples", file=sys.stderr)
        return 2
    samp = np.concatenate(rows, axis=0)
    pts = samp[:, 0:3]          # x y z, absolute level-0 voxels
    nrm = samp[:, 3:6]
    conf = samp[:, 6]

    if args.axis:
        axis = load_axis(args.axis)
    else:
        if args.origin_sign != "outward":
            print("export_tsm_labels: pass --axis or --origin-sign outward", file=sys.stderr)
            return 2
        axis = fallback_axis(origin_zyx, shape_zyx)
        print(f"export_tsm_labels: no --axis; signing against a straight axis at "
              f"(y,x) = ({axis[0][1]:.1f}, {axis[0][2]:.1f}) from the store origin",
              file=sys.stderr)
    nrm = orient_normals(pts, nrm, axis)

    # sample positions as store-local (z, y, x) float coordinates
    loc = np.stack([pts[:, 2] - origin_zyx[0],
                    pts[:, 1] - origin_zyx[1],
                    pts[:, 0] - origin_zyx[2]], axis=1)
    nloc = np.stack([nrm[:, 2], nrm[:, 1], nrm[:, 0]], axis=1)

    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is a tsm dependency
        print("export_tsm_labels: scipy is required (cKDTree)", file=sys.stderr)
        return 2
    tree = cKDTree(loc)

    # --- destination ----------------------------------------------------- #
    writer = BrickWriter(
        out,
        channels=channels,
        shape_zyx=shape_zyx,
        chunk=chunk,
        dtype=src.dtype,
        origin_zyx=origin_zyx,
        voxel_um=float(attrs.get("voxel_um", 2.4)),
        scale=float(attrs.get("scale", 1.0)),
    )
    dst = writer.array

    # chunks touched by any sample, grown by the band so a sample just
    # outside a chunk still reaches into it
    nz = [(shape_zyx[i] + chunk - 1) // chunk for i in range(3)]
    band = float(args.band)
    touched = set()
    lo = np.floor((loc - band) / chunk).astype(np.int64)
    hi = np.floor((loc + band) / chunk).astype(np.int64)
    for l, h in zip(lo, hi):
        for iz in range(max(l[0], 0), min(h[0], nz[0] - 1) + 1):
            for iy in range(max(l[1], 0), min(h[1], nz[1] - 1) + 1):
                for ix in range(max(l[2], 0), min(h[2], nz[2] - 1) + 1):
                    touched.add((iz, iy, ix))

    n_copy = 0
    n_edit = 0
    n_voxels = 0
    for iz in range(nz[0]):
        for iy in range(nz[1]):
            for ix in range(nz[2]):
                z0, y0, x0 = iz * chunk, iy * chunk, ix * chunk
                z1 = min(z0 + chunk, shape_zyx[0])
                y1 = min(y0 + chunk, shape_zyx[1])
                x1 = min(x0 + chunk, shape_zyx[2])
                block = np.asarray(src[:, z0:z1, y0:y1, x0:x1])
                if (iz, iy, ix) in touched:
                    zz, yy, xx = np.meshgrid(
                        np.arange(z0, z1, dtype=np.float64),
                        np.arange(y0, y1, dtype=np.float64),
                        np.arange(x0, x1, dtype=np.float64), indexing="ij")
                    q = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
                    d, idx = tree.query(q, distance_upper_bound=band)
                    inb = np.isfinite(d)
                    if inb.any():
                        j = idx[inb]
                        delta = q[inb] - loc[j]
                        sign = np.where(np.einsum("ij,ij->i", delta, nloc[j]) < 0, -1.0, 1.0)
                        sdf = np.zeros(q.shape[0], dtype=np.float64)
                        sdf[inb] = d[inb] * sign
                        val = np.zeros(q.shape[0], dtype=np.uint8)
                        val[inb] = (conf[j] >= args.conf_min).astype(np.uint8)
                        enc = np.zeros(q.shape[0], dtype=np.uint8)
                        enc[inb] = encode_sdf_u8(sdf[inb], args.clip)
                        m = inb.reshape(zz.shape)
                        block[c_sdf][m] = enc.reshape(zz.shape)[m]
                        block[c_val][m] = val.reshape(zz.shape)[m]
                        n_edit += 1
                        n_voxels += int(inb.sum())
                    else:
                        n_copy += 1
                else:
                    n_copy += 1
                dst[:, z0:z1, y0:y1, x0:x1] = block

    summary = {
        "store": store, "out": out, "samples": list(args.samples),
        "n_samples": int(samp.shape[0]), "clip": args.clip, "band": args.band,
        "conf_min": args.conf_min, "chunks_rewritten": n_edit,
        "chunks_copied": n_copy, "voxels_written": n_voxels,
        "axis": args.axis or "origin-fallback",
    }
    with open(os.path.join(out, "surface_labels.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"export_tsm_labels: {out}: {n_edit} chunks rewritten, {n_copy} copied, "
          f"{n_voxels} voxels from {samp.shape[0]} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
