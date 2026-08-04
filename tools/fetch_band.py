#!/usr/bin/env python3
"""Download a full shard row (one Z of 1024-slice depth, all XY) of the
PHercParis3 dct3d zarr into band/<Z>_<Y>_<X>.shard. Resumable: existing
files with plausible sizes are skipped; partial downloads go to .part first.

Usage: tools/fetch_band.py [band_dir] [Z=33] [workers=8]
"""
import concurrent.futures as cf
import pathlib
import sys
import urllib.request

BASE = ("https://dl.ash2txt.org/community-uploads/forrest/exports/PHercParis3/"
        "20260427095331-2.400um-0.2m-78keV-masked.zarr/0/c")
GY, GX = 42, 42  # 43008 / 1024
IDX_BYTES = 64 * 64 * 64 * 16 + 4  # a shard is at least its footer index


def fetch(job):
    z, y, x, out = job
    if out.exists() and out.stat().st_size >= IDX_BYTES:
        return "skip"
    url = f"{BASE}/{z}/{y}/{x}"
    part = out.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            want = int(r.headers.get("Content-Length", 0))
            n = 0
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                n += len(buf)
        if want and n != want:
            part.unlink(missing_ok=True)
            return f"short {y},{x}"
        part.rename(out)
        return "ok"
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        if e.code == 404:
            out.with_suffix(".missing").touch()  # sparse region: no shard
            return "absent"
        return f"http {e.code} {y},{x}"
    except Exception as e:  # noqa: BLE001 — log and retry on next run
        part.unlink(missing_ok=True)
        return f"err {y},{x}: {e}"


def main() -> int:
    band = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "band")
    z = int(sys.argv[2]) if len(sys.argv) > 2 else 33
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    band.mkdir(exist_ok=True)
    jobs = []
    for y in range(GY):
        for x in range(GX):
            out = band / f"{z}_{y}_{x}.shard"
            if out.exists() or out.with_suffix(".missing").exists():
                continue
            jobs.append((z, y, x, out))
    print(f"band Z={z}: {GY*GX} shards, {len(jobs)} to fetch, {workers} workers")
    done = fail = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(fetch, jobs):
            done += 1
            if res not in ("ok", "skip", "absent"):
                fail += 1
                print(res, flush=True)
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} ({fail} errors)", flush=True)
    print(f"finished: {done} processed, {fail} errors "
          f"({'rerun to retry' if fail else 'complete'})")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
