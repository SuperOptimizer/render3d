#!/usr/bin/env bash
# Fully automated, windowless perf suite for the multiview/streaming path.
# No display, no input, no swapchain: every scenario runs --headless for a
# fixed wall time (or frame count), writes render3d's benchmark JSON, and the
# script prints a compact table and (optionally) a diff against a baseline
# results directory. Made for unattended agent loops: build -> run -> compare.
#
#   tools/perf_headless.sh <bricks-manifest> <tifxyz-dir> [overlay-root] [segstore]
#
# Env: BIN (default build/release/render3d), SECONDS_PER (default 20),
#      WARM (default 200 frames), SIZE (default "1920 1080"),
#      RESULT_DIR (default bench-results/headless-<stamp>), BASELINE (dir),
#      COLD_TREE=1 (also run a cold-cache scenario on a scratch copy of the
#      tree; needs network), EXTRA (extra args, e.g. "--inklive").
set -eu

MANIFEST="${1:?bricks manifest.json}"
SEG="${2:?tifxyz dir}"
OVERLAY="${3:-}"
SEGSTORE="${4:-}"
BIN="${BIN:-./build/release/render3d}"
SECONDS_PER="${SECONDS_PER:-20}"
WARM="${WARM:-200}"
SIZE="${SIZE:-1920 1080}"
RESULT_DIR="${RESULT_DIR:-bench-results/headless-$(date +%Y%m%d-%H%M%S)}"
BASELINE="${BASELINE:-}"
EXTRA="${EXTRA:-}"
mkdir -p "$RESULT_DIR"

COMMON="--bricks $MANIFEST --multiview $SEG --tf 1 --size $SIZE --headless --warmup $WARM $EXTRA"
[ -n "$OVERLAY" ] && COMMON="$COMMON --overlay $OVERLAY"
[ -n "$SEGSTORE" ] && COMMON="$COMMON --segments $SEGSTORE"

run() { # slug, description, env assignments..., -- extra args
  slug="$1"; desc="$2"; shift 2
  envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  log="$RESULT_DIR/$slug.log"
  json="$RESULT_DIR/$slug.json"
  if env "${envs[@]}" timeout $((SECONDS_PER * 6 + 120)) "$BIN" $COMMON --seconds "$SECONDS_PER" \
       --bench-json "$json" --bench-name "$slug" "$@" > "$log" 2>&1; then
    line=$(grep -E "^frame " "$log" | tail -1 | sed 's/^frame *[0-9]* |//')
    printf "%-22s %s\n" "$slug" "$line"
  else
    printf "%-22s FAILED (rc=%s) — see %s\n" "$slug" "$?" "$log"
  fi
}

echo "render3d headless perf — $(date) — $RESULT_DIR"
echo "bin: $BIN   $SECONDS_PER s per scenario, warmup $WARM frames, $SIZE"
echo "----------------------------------------------------------------------"
run static      "idle multiview, pane cache on"      --
run static-nocache "idle multiview, all panes redrawn" R3D_NO_PANE_CACHE=1 --
run exercise    "scripted scrub/zoom/pan"             R3D_MV_EXERCISE=1 --
run exercise-nocache "scripted, all panes redrawn"    R3D_MV_EXERCISE=1 R3D_NO_PANE_CACHE=1 --
if [ "${COLD_TREE:-0}" = "1" ]; then
  root=$(dirname "$MANIFEST")
  scratch=$(mktemp -d)
  cp -r "$root" "$scratch/tree" && rm -rf "$scratch/tree/bricks"
  cold="--bricks $scratch/tree/manifest.json"
  # overlay stays warm; only the CT tree is cold
  run cold-static "cold CT cache, idle" -- $cold
  rm -rf "$scratch"
fi

# ---- summary table from the JSON (mean/p95/p99 cpu + gpu, fps, decode) ----
python3 - "$RESULT_DIR" "$BASELINE" <<'PY'
import json, os, sys, glob
rd, base = sys.argv[1], sys.argv[2]
def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        t = j.get("timings", {})
        cpu = t.get("cpu_frame", {}); gpu = t.get("gpu_frame", {})
        bs = j.get("brick_stream", {})
        m = cpu.get("mean_ms", 0.0)
        out[os.path.basename(f)[:-5]] = dict(
            fps=1000.0 / m if m else 0.0, cpu=m, cpu95=cpu.get("p95_ms", 0.0),
            cpu99=cpu.get("p99_ms", 0.0), gpu=gpu.get("mean_ms", 0.0),
            gpu99=gpu.get("p99_ms", 0.0), dec=bs.get("decoded", 0),
            job=bs.get("mean_job_ms", 0.0), frames=j.get("measured_frames", 0))
    return out
cur = load(rd)
print()
print(f"{'scenario':22} {'fps':>7} {'cpu ms':>7} {'p95':>6} {'p99':>6} {'gpu ms':>7} {'gpu99':>6} {'decoded':>8} {'ms/job':>7} {'frames':>7}")
for k, v in cur.items():
    print(f"{k:22} {v['fps']:7.1f} {v['cpu']:7.2f} {v['cpu95']:6.2f} {v['cpu99']:6.2f} {v['gpu']:7.2f} {v['gpu99']:6.2f} {v['dec']:8d} {v['job']:7.1f} {v['frames']:7d}")
if base and os.path.isdir(base):
    b = load(base)
    print(f"\nvs baseline {base}:")
    for k, v in cur.items():
        if k not in b: continue
        o = b[k]
        d = lambda a, c: (f"{100*(c-a)/a:+.1f}%" if a else "n/a")
        print(f"{k:22} fps {d(o['fps'], v['fps']):>8}  cpu {d(o['cpu'], v['cpu']):>8}  gpu {d(o['gpu'], v['gpu']):>8}  p99 {d(o['cpu99'], v['cpu99']):>8}")
PY
echo "results: $RESULT_DIR"
