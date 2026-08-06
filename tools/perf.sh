#!/usr/bin/env sh
# Automated perf suite: scripted camera paths (orbit / zoom / fly) over the
# real volume, plus static worst-case views, at 1080p uncapped. Prints one
# "profile avg" line per scenario; paste results into docs/measured.md.
#   tools/perf.sh [binary] [volume.u8] [n]
set -eu

BIN="${1:-./build/release/render3d}"
VOL="${2:-volume.u8}"
N="${3:-1024}"
COMMON="--size 1920 1080 --no-vsync --tf 1"
# A discrete GPU that idles low (RTX 4060: 210 MHz idle, ~2055 MHz settled)
# spends the first frames of a cold process ramping, and `profile avg` averages
# those in. At 300 frames that dominated the mean — the same scenario measured
# 1.90 / 4.70 / 5.98 ms across three runs. A throwaway warmup plus a longer
# measured run brings the spread under 1%. Do not shorten these without
# re-checking repeatability on the target GPU.
WARM_FRAMES="${WARM_FRAMES:-400}"
FRAMES="${FRAMES:-1200}"

run() {
  desc="$1"; shift
  timeout 120 "$BIN" "$VOL" "$N" "$N" "$N" $COMMON --frames "$WARM_FRAMES" "$@" >/dev/null 2>&1
  line=$(timeout 180 "$BIN" "$VOL" "$N" "$N" "$N" $COMMON --frames "$FRAMES" "$@" 2>/dev/null \
         | grep 'profile avg' || echo 'FAILED')
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
run "slab zsweep (1-tile)"     --slab 32 --bench zsweep
if [ -f slab3072.u8 ]; then
  timeout 120 "$BIN" slab3072.u8 3072 3072 96 $COMMON --frames "$WARM_FRAMES" \
    --slab 32 --bench zsweep >/dev/null 2>&1
  line=$(timeout 180 "$BIN" slab3072.u8 3072 3072 96 $COMMON --frames "$FRAMES" \
         --slab 32 --bench zsweep 2>/dev/null | grep 'profile avg' || echo 'FAILED')
  printf '%-28s %s\n' "slab zsweep (2x2 3072^2)" "$line"
fi
