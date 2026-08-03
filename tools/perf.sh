#!/usr/bin/env sh
# Automated perf suite: scripted camera paths (orbit / zoom / fly) over the
# real volume, plus static worst-case views, at 1080p uncapped. Prints one
# "profile avg" line per scenario; paste results into docs/measured.md.
#   tools/perf.sh [binary] [volume.u8] [n]
set -eu

BIN="${1:-./build/release/render3d}"
VOL="${2:-volume.u8}"
N="${3:-1024}"
COMMON="--size 1920 1080 --no-vsync --tf 1 --frames 300"

run() {
  desc="$1"; shift
  line=$(timeout 120 "$BIN" "$VOL" "$N" "$N" "$N" $COMMON "$@" 2>/dev/null | grep 'profile avg' || echo 'FAILED')
  printf '%-28s %s\n' "$desc" "$line"
}

echo "== render3d perf suite ($(date -I), $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo '?')) =="
run "bench orbit"              --bench orbit
run "bench zoom"               --bench zoom
run "bench fly (interior)"     --bench fly
run "bench orbit lowcut=110"   --bench orbit --lowcut 110
run "bench fly   lowcut=110"   --bench fly --lowcut 110
run "static exterior"
run "static interior dense"    --cam 0.5 0.5 0.35 0.0 0.15
run "static MIP worst-case"    --mode 1
