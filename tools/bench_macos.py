#!/usr/bin/env python3
"""Local synthetic streaming benchmarks; requires a built macos preset.

python3 tools/bench_macos.py --baseline /path/to/previous/render3d
Results include uncapped, fixed-full-quality frame timings and decoded blocks.
Before/after rendering times alone are not equivalent-work comparisons when
one binary has loaded substantially more fine blocks during the same run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/macos/render3d"))
    parser.add_argument("--fixture-tool", type=Path, default=Path("build/macos/bench_blocks"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("build") / ("bench-" + time.strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=400)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    variants = {"current": args.binary.resolve()}
    if args.baseline:
        variants = {"baseline": args.baseline.resolve(), **variants}
    metadata = {"platform": platform.platform(), "machine": platform.machine(),
                "binaries": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                             for k, v in variants.items()},
                "frames": args.frames, "warmup": args.warmup,
                "repeats": args.repeats, "quality": "full", "size": [1920, 1080],
                "pool": 36, "warm_mib": 64, "synthetic_shape": [512, 512, 512]}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fixture = args.fixture_tool.resolve()
    subprocess.run([fixture, "--make-tree", out / "tree"], check=True)
    with (out / "decode.jsonl").open("w") as f:
        for _ in range(args.repeats):
            subprocess.run([fixture, out / "tree/volcomp/L0/0_0_0.vcs"], stdout=f, check=True)
    common = ["--bricks", str(out / "tree/manifest.json"), "--headless",
              "--size", "1920", "1080", "--no-vsync", "--tf", "1", "--warmup",
              str(args.warmup), "--frames", str(args.frames), "--pool", "36",
              "--warm", "64", "--quality", "full"]
    scenarios = [("static", ["--slab-view", "--depth", "0"], {}),
                 ("fly", ["--slab-view", "--depth", "0", "--bench", "fly"], {}),
                 ("exercise", [], {"R3D_MV_EXERCISE": "1", "R3D_MV_FIT": "512"})]
    rows = []
    for rep in range(args.repeats):
        order = list(variants) if rep % 2 == 0 else list(variants)[::-1]
        for name, extra, env in scenarios:
            for variant in order:
                stem = f"{name}-{variant}-{rep}"
                command = [str(variants[variant]), *common, *extra,
                           "--bench-json", str(out / (stem + ".json"))]
                if platform.system() == "Darwin":
                    command = ["/usr/bin/time", "-l", *command]
                start = time.monotonic()
                with (out / (stem + ".log")).open("w") as f:
                    subprocess.run(command, stdout=f, stderr=f, check=True,
                                   env={**os.environ, **env}, timeout=180)
                j = json.loads((out / (stem + ".json")).read_text())
                cpu = j["timings"]["cpu_frame"]
                row = {"scenario": name, "variant": variant, "repeat": rep,
                       "wall_s": time.monotonic() - start, "frame_ms": cpu["mean_ms"],
                       "p99_ms": cpu["p99_ms"], "gpu_ms": j["timings"]["gpu_frame"]["mean_ms"],
                       **j["brick_stream"]}
                rows.append(row)
                (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(f"{stem:24} frame {row['frame_ms']:6.2f} ms  p99 {row['p99_ms']:6.2f} ms"
                      f"  decoded {row['decoded']:6}  failures {row['failures']}", flush=True)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
