# dct3d-zarr — a Zarr v3 codec for dct3d

`dct3d-zarr` exposes the [dct3d](./README.md) 16³ lossy volume codec as a
**Zarr v3 codec** named `dct3d`. Install it and stock `zarr` reads and writes
dct3d-compressed arrays with no further wiring.

## Install

Use a **stable CPython (3.11–3.13)** in a virtualenv — `numpy`/`numcodecs`/`zarr`
ship prebuilt wheels there, so nothing compiles from source. (Free-threaded
`3.14t` and other bleeding-edge builds lack those wheels and will fail trying to
compile numcodecs' Blosc/Zstd extensions — that's a numcodecs/toolchain issue,
not dct3d.)

```sh
# from a clone of this repo:
python3.13 -m venv .venv           # any stable 3.11–3.13
source .venv/bin/activate

pip install "zarr>=3" numpy cffi   # cffi builds the dct3d extension (needs a C compiler)
pip install .                      # run in the repo root -> installs dct3d-zarr

# To READ a remote export over HTTP, also:
pip install pillow requests "fsspec[http]"
```

Prebuilt binary wheels for dct3d-zarr are built by CI
(`.github/workflows/wheels.yml`) for Linux/macOS/Windows; once published to PyPI
a plain `pip install dct3d-zarr` will need no compiler. The codec registers with
zarr through a `zarr.codecs` entry point on install, so **reading needs no
explicit import** — a stock `zarr.open(...)` on a dct3d array resolves the codec
automatically. See [`examples/read_slice.py`](examples/read_slice.py) for a
minimal reader.

```python
import numpy as np
import zarr
from zarr.codecs import ShardingCodec
from dct3d_zarr import Dct3dCodec

# One dct3d block == one 16³ zarr chunk. Aggregate many into a shard so a whole
# scroll isn't billions of tiny files — sharding is zarr-native; dct3d only ever
# sees a single 16³ block.
arr = zarr.create_array(
    store="scroll.zarr",
    shape=(1024, 1024, 1024),
    chunks=(512, 512, 512),                       # shard (the on-disk object)
    dtype="uint8",
    serializer=ShardingCodec(
        chunk_shape=(16, 16, 16),                 # inner chunk = one dct3d block
        codecs=[Dct3dCodec(quality=1.0, tau=2.0)],
    ),
    fill_value=0,
)
arr[:] = my_volume
```

Reading needs **no import at all** — the codec registers via a `zarr.codecs`
entry point, so any process that opens the array resolves `dct3d` automatically:

```python
import zarr
back = zarr.open_array(store="scroll.zarr", mode="r")[:]   # just works
```

## The design in one paragraph

The codec is deliberately geometry-fixed: it compresses exactly one **16³** block
per call — the atomic unit dct3d already operates on — and each encoded block is
fully self-contained (it carries its own value range and quantizer step), so it is
**independently decodable**. That makes it a clean Zarr v3 `array→bytes` codec.
Anything larger than 16³ is handled by zarr's native `sharding_indexed` codec,
which packs many 16³ inner chunks into one shard object with an index; dct3d and
sharding compose without either knowing about the other.

## Codec parameters

`Dct3dCodec(quality=1.0, max_error=0.0, tau=0.0)`

| param | meaning |
|-------|---------|
| `quality` | quantizer coarseness; smaller = higher fidelity, larger output. Auto-normalized per chunk so one value means comparable *relative* fidelity across dtypes. |
| `max_error` | optional *relative* per-voxel error bound in `[0, 1)` (fraction of the chunk's value range). `0` disables. |
| `tau` | optional *absolute* per-voxel error bound in raw input units. `0` disables. If both are set, the tighter wins. |

Supported dtypes: `uint8 uint16 uint32 int8 int16 int32 float32` (the seven dct3d
types). It is a **lossy** codec — reconstruction is within the requested bound, not
bit-exact, and (fast-math float) not bit-reproducible across builds.

## Deblocking decoded data

At aggressive quality, independent 16³-block quantization leaves visible steps
across the block grid. `dct3d_zarr.deblock` removes them after decode — a
signal-adaptive boundary filter that closes coding seams but leaves genuine
edges alone (thresholds scale with the quantizer, H.26x-style):

```python
import dct3d_zarr
sub = np.ascontiguousarray(arr[z0:z0+64, 0:512, 0:512])  # 16-aligned region
dct3d_zarr.deblock(sub, step=32)          # step = the encode quality; in place
```

Measured on real 2.4 µm scroll CT at q32: ~70 % of the seam excess removed and
PSNR *improves* ~0.4 dB (it removes quantization noise, not detail). Seams are
only fixed where both sides are inside the array you pass, so deblock an
assembled region, not a lone 16³ chunk.

## Requirements

- Python ≥ 3.10, `zarr` ≥ 3.0, `numpy`, `cffi`.
- Prebuilt wheels ship the compiled C extension (no toolchain needed). On a
  platform without a wheel, `pip` builds from the sdist, which needs a C23 compiler.

## Building from source

```sh
pip install -e ".[test]"   # editable install (compiles the cffi extension)
pytest -q                  # round-trip + entry-point discovery tests
```
