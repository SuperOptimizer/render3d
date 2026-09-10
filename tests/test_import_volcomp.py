"""Native Zarr import: payload identity, missing chunks, corruption, atomic output.
Run: python3 tests/test_import_volcomp.py build/macos/volcomppack
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff


def main():
    pack = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="r3d-import-") as tmp:
        root = Path(tmp)
        raw = root / "raw.u8"
        raw.write_bytes(bytes(range(256)) * 8192)
        native = root / "native.vcs"
        subprocess.run([pack, "raw", str(raw), "128", "128", "128", str(native)], check=True)
        encoded = native.read_bytes()
        off, size = struct.unpack_from("<QI", encoded, len(encoded)-48)
        payload = encoded[off:off+size]
        index = bytearray(b"\xff" * 8192)
        struct.pack_into("<QQ", index, 37*16, 0, len(payload))
        source, output = root / "source.zarr", root / "out.vcs"

        def write_index(ix):
            source.write_bytes(payload + ix + struct.pack("<I", crc32c(ix)))

        def run():
            return subprocess.run([pack, "zarr-shard", str(source), str(output), "2", "2"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        write_index(index)
        assert run().returncode == 0
        good = output.read_bytes()
        for i in range(512):
            off, length = struct.unpack_from("<QI", good, len(good)-8224+i*16)
            if i == 37:
                assert good[off:off+length] == payload
            else:
                assert off == 2**64-2 and length == 0
        corrupt = bytearray(source.read_bytes())
        corrupt[-1] ^= 1
        source.write_bytes(corrupt)
        assert run().returncode != 0 and output.read_bytes() == good
        bad = bytearray(index)
        struct.pack_into("<QQ", bad, 37*16, len(payload)+1, len(payload))
        write_index(bad)
        assert run().returncode != 0 and output.read_bytes() == good
        write_index(index)
        corrupt = bytearray(source.read_bytes())
        corrupt[0] ^= 1
        source.write_bytes(corrupt)
        assert run().returncode != 0 and output.read_bytes() == good
        source.write_bytes(b"short")
        assert run().returncode != 0 and output.read_bytes() == good
    print("Zarr native import: identity, missing entries, corrupt index/bounds/magic, atomic output passed")


if __name__ == "__main__":
    main()
