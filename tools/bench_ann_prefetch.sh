#!/usr/bin/env sh
# Compare decoded-plane look-ahead depths using the repeatable six-jump
# umbilicus navigation stress test.
#   tools/bench_ann_prefetch.sh [binary] [annotation.json] [frames]
set -eu

BIN="${1:-./build/native/render3d}"
ANNOTATION="${2:-PHerc1218-umbilicus.json}"
FRAMES="${3:-600}"
RESULT_DIR="${RESULT_DIR:-${TMPDIR:-/tmp}/render3d-ann-prefetch-$(date +%Y%m%d-%H%M%S)}"
DEPTHS="${DEPTHS:-0 1 3 5}"
Z_MARGINS="${Z_MARGINS:-0 16 32}"
RUN_CPU="${RUN_CPU:-1}"
RUN_Z="${RUN_Z:-1}"
unset R3D_VSLAB_NOPC
mkdir -p "$RESULT_DIR"

if [ "$RUN_CPU" = 1 ]; then
  printf '%s\n' 'annotation-jump CPU look-ahead (GPU z margin disabled)'
  printf '%-10s %-20s %s\n' 'ahead' 'pending cell-frames' 'decoded cache'
  for depth in $DEPTHS; do
    log="$RESULT_DIR/ahead-$depth.log"
    timeout 90 "$BIN" --umbilicus "$ANNOTATION" --tf 1 --bench annscroll \
      --frames "$FRAMES" --ann-prefetch "$depth" --ann-z-prefetch 0 >"$log" 2>&1
    pending=$(awk '/vslab bench: pending cell-frames/ { value=$5 } END { print value+0 }' "$log")
    cache=$(awk '/vslab decoded cache:/ { sub(/^vslab decoded cache: /, ""); value=$0 } END { print value }' "$log")
    printf '%-10s %-20s %s\n' "$depth" "$pending" "$cache"
  done
fi

if [ "$RUN_Z" = 1 ]; then
  printf '%s\n' 'fine-scroll GPU z margin (CPU jump cache disabled)'
  printf '%-10s %-20s\n' 'margin' 'visible pending cell-frames'
  for margin in $Z_MARGINS; do
    log="$RESULT_DIR/z-margin-$margin.log"
    timeout 90 "$BIN" --umbilicus "$ANNOTATION" --tf 1 --bench annwheel \
      --frames "$FRAMES" --ann-prefetch 0 --ann-z-prefetch "$margin" >"$log" 2>&1
    pending=$(awk '/vslab bench: pending cell-frames/ { value=$5 } END { print value+0 }' "$log")
    printf '%-10s %-20s\n' "$margin" "$pending"
  done
fi
printf 'logs: %s\n' "$RESULT_DIR"
