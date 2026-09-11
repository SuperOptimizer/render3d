#!/usr/bin/env python3
"""Mirror every published tifxyz surface of the Vesuvius Challenge open-data
bucket as a surface-compressor (.sfc) file on dl.ash2txt.org.

    export_surfaces.py --encoder /path/to/surface-compressor \
        --sftp-config ~/ash2txt [--remote surfcomp] \
        [--workers 4] [--error 0.1] [--scroll PHercParis4 ...] [--work /tmp/surfexport]

Layout mirrors the bucket: <remote>/<scroll>/segments/<seg>/mesh/<variant>.sfc
(where the bucket has <scroll>/segments/<seg>/mesh/<variant>.tifxyz/{x,y,z}.tif
+ meta.json). Per surface: download the four files with curl, encode with the
surface-compressor CLI (joint XYZ, Euclidean error budget in voxels), upload
over sftp, delete the local copy. Surfaces already present on the server
(non-empty file) are skipped, so the script resumes. A manifest.json and a
README.md are uploaded to <remote>/ at the end.

Needs python3-paramiko, curl and the surface-compressor CLI (libtiff build).
The sftp config is the four-line "host: / port: / user: / pass:" file."""
import argparse, concurrent.futures as cf, json, os, shutil, subprocess, sys
import threading, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PARTS = ("x.tif", "y.tif", "z.tif", "meta.json")


def s3_list(prefix, delim=True):
    dirs, files, token = [], [], None
    while True:
        u = f"{B}/?list-type=2&prefix={urllib.parse.quote(prefix)}" + ("&delimiter=/" if delim else "")
        if token:
            u += "&continuation-token=" + urllib.parse.quote(token)
        for attempt in range(5):
            try:
                x = ET.fromstring(urllib.request.urlopen(u, timeout=60).read())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        dirs += [c.find("s:Prefix", NS).text for c in x.findall("s:CommonPrefixes", NS)]
        files += [(c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)) for c in x.findall("s:Contents", NS)]
        t = x.find("s:NextContinuationToken", NS)
        if t is None:
            return dirs, files
        token = t.text


def discover(scrolls):
    """[(scroll, seg, variant, tifxyz_bytes)] for every complete published tifxyz."""
    out = []
    all_scrolls = [d.rstrip("/") for d in s3_list("")[0]]
    for sc in all_scrolls:
        if scrolls and sc not in scrolls:
            continue
        for sg in s3_list(f"{sc}/segments/")[0]:
            seg = sg.rstrip("/").split("/")[-1]
            for v in s3_list(sg + "mesh/")[0]:
                variant = v.rstrip("/").split("/")[-1]
                if not variant.endswith(".tifxyz"):
                    continue
                names = {k.split("/")[-1]: s for k, s in s3_list(v, delim=False)[1]}
                if all(n in names for n in PARTS):
                    out.append((sc, seg, variant, sum(names[n] for n in PARTS[:3])))
    return out


def read_sftp_config(path):
    cfg = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip().lower()] = v.strip()
    return cfg["host"], int(cfg.get("port", 22)), cfg["user"], cfg["pass"]


class Sftp:
    """One connection per worker; reconnects on failure."""

    def __init__(self, host, port, user, password):
        self.args = (host, port, user, password)
        self.t = self.c = None
        self.lock = threading.Lock()

    def connect(self):
        self.close()
        host, port, user, password = self.args
        self.t = paramiko.Transport((host, port))
        self.t.connect(username=user, password=password)
        self.c = paramiko.SFTPClient.from_transport(self.t)

    def close(self):
        for o in (self.c, self.t):
            try:
                if o:
                    o.close()
            except Exception:
                pass
        self.t = self.c = None

    def retry(self, fn):
        for attempt in range(4):
            try:
                if not self.c:
                    self.connect()
                return fn(self.c)
            except Exception as e:
                self.close()
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

    def size(self, path):
        def f(c):
            try:
                return c.stat(path).st_size
            except FileNotFoundError:
                return -1
        return self.retry(f)

    def mkdirs(self, path):
        def f(c):
            cur = ""
            for part in path.split("/"):
                if not part:
                    continue
                cur = f"{cur}/{part}" if cur else part
                try:
                    c.stat(cur)
                except FileNotFoundError:
                    c.mkdir(cur)
        return self.retry(f)

    def put(self, local, remote):
        def f(c):
            tmp = remote + ".part"
            c.put(local, tmp)
            try:
                c.remove(remote)
            except FileNotFoundError:
                pass
            c.rename(tmp, remote)
            return c.stat(remote).st_size
        return self.retry(f)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}... rc={r.returncode}: {r.stderr[-800:]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="surface-compressor CLI")
    ap.add_argument("--sftp-config", required=True)
    ap.add_argument("--remote", default="surfcomp",
                    help="path under the sftp root (dl.ash2txt.org logs forrest in at community-uploads/forrest/)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--error", type=float, default=0.1)
    ap.add_argument("--work", default="/tmp/surfexport")
    ap.add_argument("--scroll", action="append", default=[])
    ap.add_argument("--log", default="export.log")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-upload", action="store_true",
                    help="convert only; keep .sfc files under <work>/mirror/ in the remote layout")
    a = ap.parse_args()

    log_lock = threading.Lock()
    logf = open(a.log, "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with log_lock:
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

    log("discovering published surfaces...")
    surfaces = discover(set(a.scroll))
    total_bytes = sum(s[3] for s in surfaces)
    log(f"{len(surfaces)} surfaces, {total_bytes / 1e9:.1f} GB tifxyz")
    if a.dry_run:
        for s in surfaces:
            print(*s)
        return

    mirror = os.path.join(a.work, "mirror")
    os.makedirs(mirror, exist_ok=True)
    local = threading.local()
    if not a.no_upload:
        global paramiko
        import paramiko  # only uploads need it
        host, port, user, password = read_sftp_config(a.sftp_config)

    def sftp():
        if a.no_upload:
            return None
        if not hasattr(local, "s"):
            local.s = Sftp(host, port, user, password)
        return local.s

    done = {"n": 0, "skipped": 0, "failed": 0, "in": 0, "out": 0}
    results = []

    def one(item):
        sc, seg, variant, nbytes = item
        stem = variant[: -len(".tifxyz")]
        rdir = f"{a.remote}/{sc}/segments/{seg}/mesh"
        rpath = f"{rdir}/{stem}.sfc"
        label = f"{sc}/{seg}/{stem}"
        # a converted copy kept from a --no-upload run (or an interrupted
        # upload) lives in the local mirror; reuse it instead of re-encoding
        lpath = os.path.join(mirror, rpath[len(a.remote) + 1:])
        try:
            have = sftp().size(rpath) if not a.no_upload else \
                (os.path.getsize(lpath) if os.path.exists(lpath) else -1)
            if have > 0:
                done["skipped"] += 1
                results.append((sc, seg, stem, nbytes, have, "skipped"))
                log(f"skip   {label} ({have / 1e6:.1f} MB {'local' if a.no_upload else 'on server'})")
                return
            t0 = t1 = t2 = time.time()
            wd = None
            if os.path.exists(lpath) and os.path.getsize(lpath) > 0:
                out = lpath
            else:
                wd = os.path.join(a.work, f"{sc}-{seg}-{stem}")
                shutil.rmtree(wd, ignore_errors=True)
                src = os.path.join(wd, variant)
                os.makedirs(src)
                url = f"{B}/{sc}/segments/{seg}/mesh/{variant}"
                cmd = ["curl", "-fsS", "--retry", "5", "--retry-all-errors", "--parallel"]
                for p in PARTS:
                    cmd += ["-o", os.path.join(src, p), f"{url}/{p}"]
                run(cmd)
                t1 = time.time()
                out = os.path.join(wd, stem + ".sfc")
                run([a.encoder, "encode", src, out, "--error", str(a.error)])
                t2 = time.time()
                os.makedirs(os.path.dirname(lpath), exist_ok=True)
                os.replace(out, lpath)
                out = lpath
                shutil.rmtree(wd, ignore_errors=True)
            osz = os.path.getsize(out)
            if a.no_upload:
                t3 = time.time()
            else:
                sftp().mkdirs(rdir)
                rsz = sftp().put(out, rpath)
                t3 = time.time()
                if rsz != osz:
                    raise RuntimeError(f"uploaded size {rsz} != {osz}")
                os.remove(out)
            done["n"] += 1
            done["in"] += nbytes
            done["out"] += osz
            results.append((sc, seg, stem, nbytes, osz, "ok"))
            log(f"ok     {label} {nbytes / 1e6:.1f} -> {osz / 1e6:.1f} MB "
                f"(dl {t1 - t0:.0f}s enc {t2 - t1:.0f}s up {t3 - t2:.0f}s) "
                f"[{done['n'] + done['skipped'] + done['failed']}/{len(surfaces)}]")
        except Exception as e:
            done["failed"] += 1
            results.append((sc, seg, stem, nbytes, 0, f"failed: {e}"))
            log(f"FAILED {label}: {e}")

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, surfaces))
    log(f"done: {done['n']} converted ({done['in'] / 1e9:.1f} GB tifxyz -> "
        f"{done['out'] / 1e9:.1f} GB sfc), {done['skipped']} skipped, {done['failed']} failed")

    if a.no_upload:
        log("no-upload run: converted files are under " + mirror)
        if done["failed"]:
            sys.exit(1)
        return
    # manifest + README from the server's point of view
    s = sftp()
    manifest = {"format": "surface-compressor .sfc (container v4, joint XYZ)",
                "error_voxels": a.error, "source": B, "surfaces": []}
    for sc, seg, stem, nbytes, osz, status in sorted(results):
        rpath = f"{a.remote}/{sc}/segments/{seg}/mesh/{stem}.sfc"
        size = s.size(rpath)
        if size > 0:
            manifest["surfaces"].append({"scroll": sc, "segment": seg, "name": stem,
                                         "path": rpath[len(a.remote) + 1:],
                                         "bytes": size, "tifxyz_bytes": nbytes})
    mpath = os.path.join(a.work, "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    s.put(mpath, f"{a.remote}/manifest.json")
    readme = os.path.join(a.work, "README.md")
    open(readme, "w").write(README.format(n=len(manifest["surfaces"]), error=a.error))
    s.put(readme, f"{a.remote}/README.md")
    log(f"manifest: {len(manifest['surfaces'])} surfaces listed")
    if done["failed"]:
        sys.exit(1)


README = """# Published segment surfaces as surface-compressor (.sfc) files

Every tifxyz surface published under `<scroll>/segments/<segment>/mesh/` in the
`vesuvius-challenge-open-data` bucket, re-encoded with
[surface-compressor](https://github.com/SuperOptimizer/surface-compressor)
(container version 4, joint XYZ coding, maximum Euclidean error {error} voxel
in the target volume's full-resolution voxel units). Paths mirror the bucket
with `.tifxyz` replaced by `.sfc`; `meta.json` travels inside the container
as its metadata. {n} surfaces are listed in `manifest.json`.

* Open directly in [render3d](https://github.com/SuperOptimizer/render3d):
  `render3d --bricks <volume manifest> --multiview <name>.sfc`
* Restore the original tifxyz layout: `surface-compressor decode <name>.sfc <dir>`
  (or render3d's `surfconv decode`). Readers decode any 64x64 block on its own,
  so the C API (`sfc_read_xyz`, `sfc_cache_read_xyz_region`) serves partial
  regions without loading the whole surface.

Volumes for these surfaces are mirrored in the sibling `volcomp/` directory.
"""

if __name__ == "__main__":
    main()
