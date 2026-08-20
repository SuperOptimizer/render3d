#!/bin/sh
# Automated tracer profiling: trace every seed in a seeds.json (exported
# from the GUI's tracer panel) headlessly under perf, then print the
# hotspot summary. No GUI needed.
#
#   tools/tracebench.sh <pred-root> <seeds.json> [out-dir]
#
# Notes:
#  - run from the data directory (the one the GUI runs from)
#  - uses build/prof when present (symbols + frame pointers), else
#    build/release (line-level attribution degraded)
#  - WSL2 has no hardware PMU: samples on cpu-clock
set -e
[ $# -ge 2 ] || { echo "usage: $0 <pred-root> <seeds.json> [out-dir]" >&2; exit 2; }
ROOT=$1
SEEDS=$2
OUT=${3:-traced-seeds}
R3D=$(cd "$(dirname "$0")/.." && pwd)
BIN=$R3D/build/prof/tracecli
[ -x "$BIN" ] || BIN=$R3D/build/release/tracecli
[ -x "$BIN" ] || { echo "tracebench: build tracecli first (cmake --build --preset release)" >&2; exit 1; }
UMB=""
[ -f "$ROOT/../umbilicus.json" ] && UMB="--umbilicus $ROOT/../umbilicus.json"
for U in p343-lod/umbilicus.json paris4-lod/umbilicus.json; do
  [ -z "$UMB" ] && [ -f "$U" ] && UMB="--umbilicus $U"
done
DATA=$OUT/perf.data
mkdir -p "$OUT"
echo "tracebench: $BIN $ROOT --seeds $SEEDS $UMB -> $OUT"
perf record -e cpu-clock -F 499 -g -o "$DATA" -- \
  "$BIN" "$ROOT" --seeds "$SEEDS" $UMB --out "$OUT" 2>&1 | tee "$OUT/trace.log"
echo
echo "=== hotspots (self time) ==="
perf report -i "$DATA" --stdio --no-children -s symbol -g none --percent-limit 1.0 2>/dev/null |
  sed -n '/Overhead/,$p' | head -25
echo
echo "full call graphs: perf report -i $DATA"
