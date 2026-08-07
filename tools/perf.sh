#!/usr/bin/env sh
# Automated perf suite: scripted camera paths (orbit / zoom / fly) over the
# real volume, plus static worst-case views, at 1080p uncapped. Each scenario
# stays in one process for warmup + measurement and writes percentile JSON.
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
RESULT_DIR="${RESULT_DIR:-bench-results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RESULT_DIR"

run() {
  desc="$1"; slug="$2"; shift 2
  line=$(timeout 300 "$BIN" "$VOL" "$N" "$N" "$N" $COMMON --warmup "$WARM_FRAMES" \
         --frames "$FRAMES" --bench-name "$desc" --bench-json "$RESULT_DIR/$slug.json" "$@" 2>/dev/null \
         | grep 'profile avg' || echo 'FAILED')
  printf '%-28s %s\n' "$desc" "$line"
}

echo "== render3d perf suite ($(date -I), $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo '?')) =="
run "bench orbit"            orbit          --bench orbit
run "bench zoom"             zoom           --bench zoom
run "bench fly (interior)"   fly            --bench fly
run "bench orbit lowcut=110" orbit-lowcut   --bench orbit --lowcut 110
run "bench fly lowcut=110"   fly-lowcut     --bench fly --lowcut 110
run "static exterior"        exterior
run "static interior dense"  interior       --cam 0.5 0.5 0.35 0.0 0.15
run "static MIP worst-case"  mip            --mode 1
run "slab zsweep (1-tile)"   slab-1tile     --slab 32 --bench zsweep
if [ -f slab3072.u8 ]; then
  line=$(timeout 300 "$BIN" slab3072.u8 3072 3072 96 $COMMON --warmup "$WARM_FRAMES" \
         --frames "$FRAMES" --slab 32 --bench zsweep --bench-name "slab zsweep (3072^2)" \
         --bench-json "$RESULT_DIR/slab-3072.json" 2>/dev/null | grep 'profile avg' || echo 'FAILED')
  printf '%-28s %s\n' "slab zsweep (3072^2 real)" "$line"
fi
printf 'JSON results: %s\n' "$RESULT_DIR"
