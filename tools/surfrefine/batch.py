#!/usr/bin/env python3
"""Refine a corpus of surfaces with `surfrefine` and collect one report.

    batch.py --surfrefine build/macos/surfrefine --pred ROOT [--ct-root DIR]
             --out-dir DIR (--inputs a.sfc b/ ... | --glob 'cache/od/segments/*.sfc'
                             | --mirror community-uploads/forrest/surfcomp --sftp-config ~/ash2txt)
             [--workers 4] [--threads-per-job 4] [--surfrefine-args '--subdivide 1']
             [--upload surfcomp-refined --sftp-config ~/ash2txt]

For every input surface (.sfc file or tifxyz directory) the driver runs one
`surfrefine` subprocess writing <out-dir>/<name>/ (+ <name>.sfc and
refine_qc.json). Surfaces whose refine_qc.json already exists are skipped, so
an interrupted run resumes. --mirror lists the dl.ash2txt.org surfcomp mirror
over sftp (same config file as tools/export_surfaces.py), downloads each .sfc
into <out-dir>/src/, refines it, and with --upload puts <name>.sfc and
refine_qc.json under the given remote prefix in the same scroll/segment
layout. The aggregate corpus_report.json lists per-surface before/after QC,
flagged tile counts and failures; the review queue in the render3d GUI reads
that file."""
import argparse, concurrent.futures as cf, glob, json, os, subprocess, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def name_of(path):
    p = path.rstrip("/")
    base = os.path.basename(p)
    for suf in (".sfc", ".tifxyz"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfrefine", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ct-root")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--glob")
    ap.add_argument("--mirror", help="remote prefix under the sftp root, e.g. surfcomp")
    ap.add_argument("--scroll", action="append", default=[], help="restrict --mirror to scrolls")
    ap.add_argument("--upload", help="remote prefix for refined outputs")
    ap.add_argument("--sftp-config")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads-per-job", type=int, default=4)
    ap.add_argument("--surfrefine-args", default="")
    ap.add_argument("--log", default="refine.log")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    lock = threading.Lock()
    logf = open(a.log, "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with lock:
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

    items = []  # (name, local input path or None, remote path or None, remote dir)
    for p in a.inputs:
        items.append((name_of(p), p, None, None))
    if a.glob:
        for p in sorted(glob.glob(a.glob)):
            items.append((name_of(p), p, None, None))
    sftp = None
    if a.mirror or a.upload:
        if not a.sftp_config:
            sys.exit("--mirror/--upload need --sftp-config")
        from export_surfaces import Sftp, read_sftp_config
        host, port, user, password = read_sftp_config(a.sftp_config)
        local = threading.local()

        def sftp():
            if not hasattr(local, "s"):
                local.s = Sftp(host, port, user, password)
            return local.s
    if a.mirror:
        s = sftp()

        def walk(c, prefix):
            out = []
            for e in c.listdir_attr(prefix):
                p = f"{prefix}/{e.filename}"
                if e.longname.startswith("d"):
                    out += walk(c, p)
                elif e.filename.endswith(".sfc"):
                    out.append(p)
            return out
        scrolls = [e for e in s.retry(lambda c: c.listdir(a.mirror)) if not a.scroll or e in a.scroll]
        for sc in scrolls:
            for rp in s.retry(lambda c, sc=sc: walk(c, f"{a.mirror}/{sc}")):
                items.append((name_of(rp), None, rp, os.path.dirname(rp)[len(a.mirror) + 1:]))
    if not items:
        sys.exit("no inputs")
    log(f"{len(items)} surfaces")
    results = {}

    def one(item):
        name, lp, rp, rdir = item
        odir = os.path.join(a.out_dir, name)
        qc = os.path.join(odir, "refine_qc.json")
        try:
            if os.path.exists(qc):
                results[name] = json.load(open(qc)) | {"status": "skipped"}
                log(f"skip   {name}")
                return
            if lp is None:
                lp = os.path.join(a.out_dir, "src", name + ".sfc")
                os.makedirs(os.path.dirname(lp), exist_ok=True)
                if not os.path.exists(lp):
                    sftp().retry(lambda c: c.get(rp, lp + ".part"))
                    os.replace(lp + ".part", lp)
            cmd = [a.surfrefine, "--pred", a.pred, "--in", lp, "--out", odir,
                   "--threads", str(a.threads_per_job)]
            if a.ct_root:
                cmd += ["--ct-root", a.ct_root]
            cmd += a.surfrefine_args.split()
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"surfrefine rc={r.returncode}: {r.stderr[-600:]}")
            rep = json.load(open(qc)) | {"status": "ok", "seconds": time.time() - t0}
            if a.upload:
                base = f"{a.upload}/{rdir}" if rdir else a.upload
                sftp().mkdirs(base)
                sftp().put(odir + ".sfc", f"{base}/{name}.sfc")
                sftp().put(qc, f"{base}/{name}.refine_qc.json")
                rep["remote"] = f"{base}/{name}.sfc"
            results[name] = rep
            b, f = rep["before"], rep["after"]
            log(f"ok     {name} folds {b['folds']}->{f['folds']} kinks {b['kinks']}->{f['kinks']} "
                f"conf {b['conf_mean']:.3f}->{f['conf_mean']:.3f} flagged {rep['flagged_count']} "
                f"({time.time() - t0:.0f}s)")
        except Exception as e:
            results[name] = {"status": f"failed: {e}"}
            log(f"FAILED {name}: {e}")

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, items))
    report = {"pred_root": a.pred, "ct_root": a.ct_root, "out_dir": os.path.abspath(a.out_dir),
              "surfaces": results}
    rp = os.path.join(a.out_dir, "corpus_report.json")
    json.dump(report, open(rp, "w"), indent=1)
    nf = sum(1 for r in results.values() if r.get("status", "").startswith("failed"))
    log(f"done: {len(results) - nf} ok, {nf} failed; report {rp}")
    if nf:
        sys.exit(1)


if __name__ == "__main__":
    main()
