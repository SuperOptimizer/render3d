"""export_tsm_labels.py: copy-on-write surface labels into a TSM fine store.

Run in the tsm project's environment, which supplies numpy / zarr / scipy and
the `tsm` package (the system python3 is free-threaded and has no numpy):

    cd build/tsm-source && uv run python -m pytest \
        ../../tests/test_export_tsm_labels.py -q

A synthetic `fine.zarr` (channels sdf, sdf_valid, ink, ink_valid; shape
(4,128,256,256), filled with a recognisable teacher pattern) and a synthetic
planar sample set (the plane x = 150, normals +x, i.e. away from an axis at
x = 0) are built in a tmp dir, the exporter is run on them, and the result is
checked for:

* the zero crossing of the decoded sdf lands within 1 voxel of the plane,
* the sign is positive outward (away from the axis) and negative inward,
* sdf_valid is 1 only inside the band and only where conf >= --conf-min,
* voxels outside the band keep the teacher bytes,
* chunks no sample reaches are byte-identical to the source,
* the other channels (ink, ink_valid) are byte-identical everywhere,
* `--out == --store` is refused.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.path.join(HERE, "..", "tools", "surfrefine", "export_tsm_labels.py")

CHUNK = 128
SHAPE = (128, 256, 256)  # z, y, x
CHANNELS = ["sdf", "sdf_valid", "ink", "ink_valid"]
PLANE_X = 150.0  # absolute voxel x of the synthetic sheet
CLIP = 20.0
BAND = 20.0


def decode_sdf(u: np.ndarray, clip: float = CLIP) -> np.ndarray:
    return (u.astype(np.float32) - 128.0) * (clip / 127.0)


def teacher_pattern(shape):
    """A deterministic, non-trivial byte pattern for every channel."""
    z, y, x = np.meshgrid(*[np.arange(s, dtype=np.int64) for s in shape], indexing="ij")
    out = np.empty((len(CHANNELS), *shape), dtype=np.uint8)
    out[0] = ((z * 7 + y * 3 + x * 11) % 255 + 1).astype(np.uint8)  # sdf, never 0
    out[1] = ((z + y + x) % 3).astype(np.uint8)                     # sdf_valid 0/1/2
    out[2] = ((x * 5 + y) % 256).astype(np.uint8)                   # ink
    out[3] = ((y // 16) % 2).astype(np.uint8)                       # ink_valid
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tsmlabels")
    store = str(tmp / "fine.zarr")
    src = zarr.create_array(
        store=store, shape=(len(CHANNELS), *SHAPE), chunks=(1, CHUNK, CHUNK, CHUNK),
        dtype=np.uint8, fill_value=0,
        attributes={"channels": CHANNELS, "voxel_um": 2.4,
                    "origin_zyx": [0, 0, 0], "scale": 1.0},
        overwrite=False)
    ref = teacher_pattern(SHAPE)
    src[:] = ref

    # planar sample set at x = PLANE_X, covering z 20..100, y 20..100 only
    # (so the chunk at x >= 128 that the plane does not reach, and the chunks
    #  far in z/y, stay untouched)
    zs = np.arange(20, 101, 0.5)
    ys = np.arange(20, 101, 0.5)
    zz, yy = np.meshgrid(zs, ys, indexing="ij")
    n = zz.size
    rec = np.zeros((n, 7), dtype="<f4")
    rec[:, 0] = PLANE_X          # x
    rec[:, 1] = yy.ravel()       # y
    rec[:, 2] = zz.ravel()       # z
    rec[:, 3] = 1.0              # nx (outward from an axis at x = 0)
    # confidence: below the threshold for y < 50 so the valid mask is testable
    rec[:, 6] = np.where(yy.ravel() < 50.0, 0.1, 0.9)
    sfile = tmp / "plane.bin"
    with open(sfile, "wb") as fh:
        fh.write(b"R3DS" + struct.pack("<III", n, 0, 0))
        fh.write(rec.tobytes())

    axis = tmp / "axis.json"
    axis.write_text(json.dumps({
        "coordinate_order": "xyz", "coordinate_space": "full_resolution",
        "control_points": [{"x": 0.0, "y": 60.0, "z": 0.0},
                           {"x": 0.0, "y": 60.0, "z": 200.0}]}))

    out = str(tmp / "out.zarr")
    r = subprocess.run(
        [sys.executable, EXPORT, "--store", store, "--out", out,
         "--samples", str(sfile), "--clip", str(CLIP), "--band", str(BAND),
         "--conf-min", "0.25", "--axis", str(axis)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return {"tmp": tmp, "store": store, "out": out, "ref": ref,
            "samples": str(sfile), "axis": str(axis), "stderr": r.stderr}


def test_attrs_and_shape(built):
    a = zarr.open_array(store=built["out"], mode="r")
    s = zarr.open_array(store=built["store"], mode="r")
    assert a.shape == s.shape and a.chunks == s.chunks and a.dtype == s.dtype
    assert list(a.attrs["channels"]) == CHANNELS
    assert list(a.attrs["origin_zyx"]) == [0, 0, 0]
    assert float(a.attrs["voxel_um"]) == 2.4 and float(a.attrs["scale"]) == 1.0


def test_zero_crossing_within_one_voxel(built):
    a = np.asarray(zarr.open_array(store=built["out"], mode="r")[0])
    sdf = decode_sdf(a)
    # a row well inside the sampled patch
    for z, y in [(60, 70), (40, 90), (90, 60)]:
        row = sdf[z, y, :]
        xs = np.arange(row.size)
        inb = np.abs(xs - PLANE_X) <= BAND - 1
        # crossing: last negative before the first positive
        neg = xs[inb][row[inb] < 0]
        pos = xs[inb][row[inb] > 0]
        assert neg.size and pos.size
        crossing = 0.5 * (neg.max() + pos.min())
        assert abs(crossing - PLANE_X) <= 1.0, (z, y, crossing)


def test_sign_is_outward(built):
    a = np.asarray(zarr.open_array(store=built["out"], mode="r")[0])
    sdf = decode_sdf(a)
    z, y = 60, 70
    assert sdf[z, y, int(PLANE_X) + 10] > 0   # away from the axis at x = 0
    assert sdf[z, y, int(PLANE_X) - 10] < 0   # toward it


def test_distance_is_accurate(built):
    a = np.asarray(zarr.open_array(store=built["out"], mode="r")[0])
    sdf = decode_sdf(a)
    z, y = 60, 70
    for dx in (-15, -5, -1, 1, 5, 15):
        x = int(PLANE_X) + dx
        # samples are 0.5 vox apart in z/y, so the nearest one is essentially
        # on the perpendicular foot; encoding granularity is clip/127 ~ 0.16
        assert abs(sdf[z, y, x] - dx) < 0.4, (dx, sdf[z, y, x])


def test_valid_only_in_band_and_above_conf(built):
    out = np.asarray(zarr.open_array(store=built["out"], mode="r")[1])
    ref = built["ref"][1]
    z = 60
    # high-confidence region (y >= 50), inside the band -> 1
    assert out[z, 70, int(PLANE_X) + 5] == 1
    assert out[z, 70, int(PLANE_X) - 5] == 1
    # low-confidence region (y < 50), inside the band -> 0 (written, not kept)
    assert out[z, 30, int(PLANE_X)] == 0
    # outside the band -> the teacher byte survives
    far_x = int(PLANE_X) + int(BAND) + 30
    assert out[z, 70, far_x] == ref[z, 70, far_x]
    # no voxel farther than BAND from the plane was rewritten
    xs = np.arange(SHAPE[2])
    outside = np.abs(xs - PLANE_X) > BAND
    assert np.array_equal(out[:, :, outside], ref[:, :, outside])


def test_sdf_outside_band_untouched(built):
    out = np.asarray(zarr.open_array(store=built["out"], mode="r")[0])
    ref = built["ref"][0]
    xs = np.arange(SHAPE[2])
    outside = np.abs(xs - PLANE_X) > BAND
    assert np.array_equal(out[:, :, outside], ref[:, :, outside])


def test_untouched_chunks_byte_identical(built):
    """A chunk no sample reaches must come through unchanged in every channel."""
    out = zarr.open_array(store=built["out"], mode="r")
    ref = built["ref"]
    # samples span x = 150 +- 20 -> only chunk ix = 1 in x; z 20..100 -> iz 0;
    # y 20..100 -> iy 0.  Chunk (0, 1, 0) (y 128..256, x 0..128) is untouched.
    for cz, cy, cx in [(0, 1, 0), (0, 1, 1)]:
        sl = (slice(None),
              slice(cz * CHUNK, (cz + 1) * CHUNK),
              slice(cy * CHUNK, (cy + 1) * CHUNK),
              slice(cx * CHUNK, (cx + 1) * CHUNK))
        assert np.array_equal(np.asarray(out[sl]), ref[sl]), (cz, cy, cx)


def test_other_channels_identical(built):
    out = zarr.open_array(store=built["out"], mode="r")
    ref = built["ref"]
    for c in (2, 3):  # ink, ink_valid
        assert np.array_equal(np.asarray(out[c]), ref[c])


def test_summary_written(built):
    with open(os.path.join(built["out"], "surface_labels.json")) as fh:
        s = json.load(fh)
    assert s["chunks_rewritten"] >= 1 and s["chunks_copied"] >= 1
    assert s["voxels_written"] > 0 and s["n_samples"] > 0


def test_refuses_in_place(built):
    r = subprocess.run(
        [sys.executable, EXPORT, "--store", built["store"], "--out", built["store"],
         "--samples", built["samples"], "--axis", built["axis"]],
        capture_output=True, text=True)
    assert r.returncode != 0 and "must differ" in r.stderr


def test_refuses_existing_out(built):
    r = subprocess.run(
        [sys.executable, EXPORT, "--store", built["store"], "--out", built["out"],
         "--samples", built["samples"], "--axis", built["axis"]],
        capture_output=True, text=True)
    assert r.returncode != 0 and "exists" in r.stderr


def test_source_store_unmodified(built):
    src = np.asarray(zarr.open_array(store=built["store"], mode="r")[:])
    assert np.array_equal(src, built["ref"])
