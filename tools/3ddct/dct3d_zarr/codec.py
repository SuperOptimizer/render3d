"""`dct3d` — a Zarr v3 array->bytes codec backed by the dct3d C library.

One zarr chunk == one 16^3 dct3d block. The codec is deliberately geometry-fixed:
it requires the array's chunk shape to be exactly (16, 16, 16), which makes each
chunk a self-contained, independently-decodable dct3d blob. Aggregation into
larger objects is zarr's job — wrap this codec in `sharding_indexed` with an
inner chunk shape of [16, 16, 16] and any shard-sized outer chunk.

Registered under the codec name ``dct3d`` via the ``zarr.codecs`` entry point
(see pyproject.toml), so a stock ``pip install dct3d-zarr`` makes zarr read and
write these arrays with no further wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

from . import _lowlevel as ll

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec

CODEC_NAME = "dct3d"


@dataclass(frozen=True)
class Dct3dCodec(ArrayBytesCodec):
    """Lossy 3D-DCT array->bytes codec for 16^3 chunks.

    Parameters
    ----------
    quality:
        Quantizer coarseness; smaller = higher fidelity, larger blob. Auto-
        normalized per chunk so the same number means comparable *relative*
        fidelity across dtypes. Default 1.0.
    max_error:
        Optional *relative* per-voxel error bound in [0, 1) (fraction of the
        chunk's value range). 0 disables. If both this and ``tau`` are set, the
        tighter bound wins.
    tau:
        Optional *absolute* per-voxel error bound in raw input units. 0 disables.
    """

    # variable-sized output (it's a compressor)
    is_fixed_size = False

    quality: float = 1.0
    max_error: float = 0.0
    tau: float = 0.0

    def __init__(self, *, quality: float = 1.0, max_error: float = 0.0, tau: float = 0.0) -> None:
        q = float(quality)
        me = float(max_error)
        t = float(tau)
        if not (q > 0.0):
            raise ValueError(f"dct3d quality must be > 0, got {quality!r}")
        if not (0.0 <= me < 1.0):
            raise ValueError(f"dct3d max_error must be in [0, 1), got {max_error!r}")
        if not (t >= 0.0):
            raise ValueError(f"dct3d tau must be >= 0, got {tau!r}")
        object.__setattr__(self, "quality", q)
        object.__setattr__(self, "max_error", me)
        object.__setattr__(self, "tau", t)

    # ---- metadata (name + configuration) ----------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, CODEC_NAME, require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": CODEC_NAME,
            "configuration": {
                "quality": self.quality,
                "max_error": self.max_error,
                "tau": self.tau,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    # ---- validation -------------------------------------------------------

    def validate(self, *, shape, dtype, chunk_grid) -> None:  # noqa: ANN001
        native = dtype.to_native_dtype()
        if str(np.dtype(native).newbyteorder("=")) not in ll.SUPPORTED_DTYPES:
            raise ValueError(
                f"dct3d codec does not support dtype {native!r}; "
                f"supported: {ll.SUPPORTED_DTYPES}"
            )

    # ---- the actual transform --------------------------------------------

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        arr = chunk_array.as_ndarray_like()
        arr = np.ascontiguousarray(np.asarray(arr))
        if arr.shape != ll.BLOCK_SHAPE:
            raise ValueError(
                f"dct3d requires a {ll.BLOCK_SHAPE} chunk (use sharding with inner "
                f"chunk_shape [16,16,16]); got chunk shape {arr.shape}"
            )
        blob = ll.encode_block(arr, self.quality, self.max_error, self.tau)
        return chunk_spec.prototype.buffer.from_bytes(blob)

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        blob = chunk_bytes.to_bytes()
        native = chunk_spec.dtype.to_native_dtype()
        block = ll.decode_block(blob, np.dtype(native))
        # Present the array in the dtype the spec asked for (endianness included).
        want = np.dtype(native)
        if block.dtype != want:
            block = block.astype(want)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(block)

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        return self._decode_sync(chunk_bytes, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        # Lossy, variable-size output — zarr requires NotImplementedError here.
        raise NotImplementedError


def register() -> None:
    """Register the codec under the name ``dct3d`` with zarr's codec registry.

    Called automatically via the ``zarr.codecs`` entry point on import; exposed
    for explicit/manual registration too.
    """
    register_codec(CODEC_NAME, Dct3dCodec)
