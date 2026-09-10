#!/usr/bin/env python3
"""Download selected native volcomp Zarr levels and repackage them for render3d.

BASE is the URL of the .zarr group, not the parent scroll listing. Defaults
fetch complete levels 3–5 and one level-0 shard at z/y/x 11/3/3 (PHerc1218).
Native payloads are preserved; no quantization or voxel reconstruction occurs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import urllib.request


def metadata(base, level):
    with urllib.request.urlopen(f"{base}/{level}/zarr.json", timeout=30) as response:
        j = json.load(response)
    def require(condition):
        if not condition:
            raise ValueError(f"unsupported native Zarr layout at {base}/{level}")
    require(j["zarr_format"] == 3 and j["node_type"] == "array")
    require(j["data_type"] == "uint8" and j["fill_value"] == 0)
    require(len(j["shape"]) == 3 and all(isinstance(d, int) and 0 < d <= 2**32-1 for d in j["shape"]))
    require(j["chunk_grid"] == {"name": "regular", "configuration": {"chunk_shape": [1024]*3}})
    require(j["chunk_key_encoding"] == {"name": "default", "configuration": {"separator": "/"}})
    require(len(j["codecs"]) == 1 and j["codecs"][0]["name"] == "sharding_indexed")
    c = j["codecs"][0]["configuration"]
    require(c["chunk_shape"] == [128]*3 and c.get("index_location", "end") == "end")
    require(c["index_codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}},
                                  {"name": "crc32c"}])
    require(len(c["codecs"]) == 1 and c["codecs"][0]["name"] == "volcomp")
    q = c["codecs"][0]["configuration"]["q"]
    require(isinstance(q, (int, float)) and 1 <= q <= 255)
    return j, q


def verify(source, target):
    old, new = source.read_bytes(), target.read_bytes()
    for i in range(512):
        off, size = struct.unpack_from("<QQ", old, len(old) - 8196 + i*16)
        dest, length = struct.unpack_from("<QI", new, len(new) - 8224 + i*16)
        if off == 2**64-1:
            assert dest == 2**64-2 and length == 0
        else:
            assert size == length and old[off:off+size] == new[dest:dest+length]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("output", type=Path)
    p.add_argument("--first-level", type=int, default=3)
    p.add_argument("--last-level", type=int, default=5)
    p.add_argument("--interior", type=int, nargs=3, default=[11, 3, 3], metavar=("Z", "Y", "X"))
    p.add_argument("--packer", type=Path, default=Path("build/macos/volcomppack"))
    a = p.parse_args()
    if not 1 <= a.first_level <= a.last_level <= 7 or min(a.interior) < 0:
        p.error("require 1 <= first-level <= last-level <= 7 and nonnegative shard coordinates")
    base = a.base.rstrip("/")
    src = a.output / "source"
    src.mkdir(parents=True, exist_ok=True)
    levels = list(range(a.first_level, a.last_level + 1))
    meta = {l: metadata(base, l) for l in [0, *levels]}
    jobs = []
    for l, (j, q) in meta.items():
        (src / f"L{l}.json").write_text(json.dumps(j, indent=2) + "\n")
        shape = [(d+1023)//1024 for d in j["shape"]]
        if l == 0:
            coords = [tuple(a.interior)]
            if any(c >= d for c, d in zip(a.interior, shape)):
                p.error("interior shard lies outside the source volume")
        else:
            coords = [(z,y,x) for z in range(shape[0]) for y in range(shape[1]) for x in range(shape[2])]
        for z,y,x in coords:
            path = src / f"L{l}-{z}_{y}_{x}.zarr-shard"
            jobs.append((l, q, path, f"{base}/{l}/c/{z}/{y}/{x}"))
    provenance = {"url": base, "source_levels": levels,
                  "interior_origin_zyx": [i*1024 for i in a.interior],
                  "interior_shape_zyx": [1024]*3,
                  "payloads": "byte-identical native streams; container conversion only"}
    (src / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def fetch(job):
        _, _, path, url = job
        if not path.exists():
            tmp = path.with_suffix(".part")
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "20",
                            "--max-time", "600", "-o", str(tmp), url], check=True)
            tmp.replace(path)
        return job

    with ThreadPoolExecutor(max_workers=3) as ex:
        for l, q, source, _ in ex.map(fetch, jobs):
            rellevel = l-a.first_level
            name = source.name.split("-", 1)[1].split(".")[0]
            target = (a.output / "interior-1024.vcs" if l == 0 else
                      a.output / f"overview/volcomp/L{rellevel}/{name}.vcs")
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(a.packer.resolve()), "zarr-shard", str(source), str(target),
                            str(max(0, rellevel)), str(q)], check=True)
            verify(source, target)
            print(f"Verified {source.name}: {source.stat().st_size/2**20:.1f} MiB", flush=True)
    entries = [{"level": l-a.first_level, "scale": 2**(l-a.first_level),
                "shape": meta[l][0]["shape"],
                "shards": [(d+1023)//1024 for d in meta[l][0]["shape"]],
                "volcomp": f"volcomp/L{l-a.first_level}/{{z}}_{{y}}_{{x}}.vcs"} for l in levels]
    manifest = {"format": "render3d.volcomp-lod.v1", "shape": entries[0]["shape"],
                "shard_shape": [1024]*3, "brick_shape": [128]*3, "levels": entries}
    tree = a.output / "overview"
    (tree / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tree / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Open: build/macos/render3d --bricks {tree / 'manifest.json'}")


if __name__ == "__main__":
    main()
