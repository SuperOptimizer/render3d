"""Round-trip and integration tests for the dct3d Zarr v3 codec."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import zarr
from zarr.codecs import ShardingCodec

import dct3d_zarr
from dct3d_zarr import Dct3dCodec
from dct3d_zarr import _lowlevel as ll

DTYPES = ["uint8", "uint16", "uint32", "int8", "int16", "int32", "float32"]


def _smooth(shape, dtype):
    g = np.mgrid[tuple(slice(0, s) for s in shape)].sum(0)
    if np.dtype(dtype).kind == "f":
        return (g / g.max()).astype(dtype)
    return (g % 200).astype(dtype)


# ---- low-level binding -------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_lowlevel_roundtrip(dtype):
    block = _smooth(ll.BLOCK_SHAPE, dtype)
    blob = ll.encode_block(block, quality=1.0, max_error=0.0, tau=0.0)
    assert 0 < len(blob) <= ll.MAX_BYTES
    back = ll.decode_block(blob, np.dtype(dtype))
    assert back.shape == ll.BLOCK_SHAPE
    assert back.dtype == np.dtype(dtype).newbyteorder("=")


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_lowlevel_tau_bound(dtype):
    rng = np.random.default_rng(1)
    block = rng.integers(0, 200, ll.N3).reshape(ll.BLOCK_SHAPE).astype(dtype)
    tau = 2.0
    blob = ll.encode_block(block, quality=8.0, max_error=0.0, tau=tau)
    back = ll.decode_block(blob, np.dtype(dtype))
    assert np.abs(block.astype(float) - back.astype(float)).max() <= tau + 1e-4


def test_lowlevel_rejects_bad_dtype():
    with pytest.raises(ll.Dct3dError):
        ll.encode_block(np.zeros(ll.BLOCK_SHAPE, dtype="float64"), 1.0, 0.0, 0.0)


def test_lowlevel_rejects_bad_shape():
    with pytest.raises(ll.Dct3dError):
        ll.encode_block(np.zeros((8, 8, 8), dtype="uint8"), 1.0, 0.0, 0.0)


def test_lowlevel_rejects_malformed_blob():
    with pytest.raises(ll.Dct3dError):
        ll.decode_block(b"\x00\x01not-a-real-blob", np.dtype("uint8"))


# ---- codec class metadata ----------------------------------------------------


def test_codec_to_from_dict_roundtrip():
    c = Dct3dCodec(quality=2.0, max_error=0.1, tau=3.0)
    d = c.to_dict()
    assert d["name"] == "dct3d"
    assert d["configuration"] == {"quality": 2.0, "max_error": 0.1, "tau": 3.0}
    assert Dct3dCodec.from_dict(d) == c


def test_codec_validates_params():
    with pytest.raises(ValueError):
        Dct3dCodec(quality=0.0)
    with pytest.raises(ValueError):
        Dct3dCodec(max_error=1.0)
    with pytest.raises(ValueError):
        Dct3dCodec(tau=-1.0)


# ---- full zarr integration ---------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_zarr_sharded_roundtrip(tmp_path, dtype):
    shape = (32, 32, 32)
    data = _smooth(shape, dtype)
    store = str(tmp_path / "vol.zarr")
    tau = 0.02 if np.dtype(dtype).kind == "f" else 2.0
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=(32, 32, 32),
        dtype=dtype,
        serializer=ShardingCodec(
            chunk_shape=(16, 16, 16), codecs=[Dct3dCodec(quality=1.0, tau=tau)]
        ),
        fill_value=0,
    )
    arr[:] = data
    back = zarr.open_array(store=store, mode="r")[:]
    assert back.shape == shape
    assert np.abs(data.astype(float) - back.astype(float)).max() <= tau + 1e-3


def test_zarr_compression_actually_happens(tmp_path):
    import os

    shape = (64, 64, 64)
    data = (np.mgrid[0:64, 0:64, 0:64].sum(0) % 256).astype(np.uint8)
    store = str(tmp_path / "vol.zarr")
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=(64, 64, 64),
        dtype="uint8",
        serializer=ShardingCodec(chunk_shape=(16, 16, 16), codecs=[Dct3dCodec(quality=1.0, tau=2.0)]),
        fill_value=0,
    )
    arr[:] = data
    on_disk = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(store)
        for f in fs
    )
    assert on_disk < data.nbytes / 10  # smooth gradient compresses hard


def test_partial_region_read(tmp_path):
    shape = (48, 48, 48)
    data = (np.mgrid[0:48, 0:48, 0:48].sum(0) % 256).astype(np.uint8)
    store = str(tmp_path / "vol.zarr")
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=(48, 48, 48),
        dtype="uint8",
        serializer=ShardingCodec(chunk_shape=(16, 16, 16), codecs=[Dct3dCodec(quality=1.0, tau=2.0)]),
        fill_value=0,
    )
    arr[:] = data
    z = zarr.open_array(store=store, mode="r")
    sub = z[3:20, 5:33, 0:16]
    assert np.abs(data[3:20, 5:33, 0:16].astype(int) - sub.astype(int)).max() <= 2


def test_entry_point_autodiscovery(tmp_path):
    """A fresh process that only imports zarr must resolve the dct3d codec."""
    shape = (16, 16, 16)
    data = (np.mgrid[0:16, 0:16, 0:16].sum(0) % 256).astype(np.uint8)
    store = str(tmp_path / "vol.zarr")
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=(16, 16, 16),
        dtype="uint8",
        serializer=Dct3dCodec(quality=1.0, tau=2.0),
        fill_value=0,
    )
    arr[:] = data
    reader = textwrap.dedent(f"""
        import sys, numpy as np, zarr
        assert "dct3d_zarr" not in sys.modules
        back = zarr.open_array(store={store!r}, mode="r")[:]
        assert "dct3d_zarr" in sys.modules, "entry point did not load codec"
        assert back.shape == (16, 16, 16)
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
