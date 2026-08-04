"""dct3d-zarr — a Zarr v3 codec for the dct3d 16^3 lossy volume codec.

Importing this package registers the ``dct3d`` codec with zarr. That happens
automatically through the ``zarr.codecs`` entry point, so end users normally do
not import this at all — a stock ``zarr.open`` on an array whose metadata names
the ``dct3d`` codec just works after ``pip install dct3d-zarr``.
"""

from __future__ import annotations

from ._lowlevel import deblock
from .codec import CODEC_NAME, Dct3dCodec, register

__all__ = ["CODEC_NAME", "Dct3dCodec", "deblock", "register"]

# Register on import. The entry point (pyproject.toml) points zarr's registry at
# `dct3d_zarr:Dct3dCodec`; importing the package is idempotent-safe regardless.
register()
