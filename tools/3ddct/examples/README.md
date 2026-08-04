# examples

## `read_slice.py`

Minimal reader for a dct3d Zarr v3 export: fetch one 1024³ shard over HTTP, mount
it as a single-shard local zarr store, and read a z-plane — **zarr-python + the
`dct3d` codec do the decode**, proving an exported volume is readable through the
stock zarr API.

```sh
# from a clone of this repo:
pip install "zarr>=3" numpy cffi && pip install .   # build+install dct3d-zarr
pip install pillow requests                          # for this example
python examples/read_slice.py                        # writes slice.png
```

Edit `URL` / `Z` / `SY` / `SX` at the top to point at another export or plane.
