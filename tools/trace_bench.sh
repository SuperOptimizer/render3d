#!/usr/bin/env bash
# Tracer quality benchmark: pinned-seed reference runs over the PHerc0343
# surface-prediction tree, one JSON row per run with the QC metrics from
# meta.json + the finish log. Diffs against a baseline directory.
#
#   tools/trace_bench.sh [--gens N] [--full] [--baseline]
#
# --baseline  record the current build's rows as the new baseline
# --full      also run the 60-gen row (slow) in addition to the 16-gen matrix
#
# Runs tracecli (pure CPU: no window, no GPU) so rows are comparable across
# machines. Compare medians of 3 runs per seed until candidate ordering
# lands (documented same-build nondeterminism).
set -euo pipefail

DATA="${R3D_DATA:-$HOME/r3d-data}"
PRED="$DATA/p343-surf-lod"
DONOR="$DATA/p343-seg1"
BENCH="$DATA/bench/trace"
BASE="$BENCH/BASELINE"
BIN="${TRACECLI:-$HOME/r3d-build/release/tracecli}"
GENS=16
FULL=0
RECORD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gens) GENS="$2"; shift ;;
    --full) FULL=1 ;;
    --baseline) RECORD=1 ;;
  esac
  shift
done

# pinned seeds: volume-center sheet + two offsets + the donor segment center
# (the last runs fused against p343-seg1 for the donor-agreement metrics)
SEEDS=(
  "center      4297 4297 8999 raw"
  "offx        4600 4297 8999 raw"
  "offy        4297 4600 8999 raw"
  "fused       4301 4301 9010 fuse"
)

mkdir -p "$BENCH"
OUT="$BENCH/run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

row() { # name dir log wall
  local name="$1" dir="$2" log="$3" wall="$4"
  local qc meta
  qc=$(grep -E 'tracer: QC .*area' "$log" | tail -1 || true)
  meta="$dir/meta.json"
  local nset area fill hole folds kinks twist slant dmean dcov
  nset=$(grep -oE 'with [0-9]+ point' "$log" | tail -1 | grep -oE '[0-9]+' || echo 0)
  if [ -f "$meta" ]; then
    area=$(grep -oE '"area_vx2": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    fill=$(grep -oE '"fill": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    hole=$(grep -oE '"hole": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    folds=$(grep -oE '"folds": [0-9]+' "$meta" | grep -oE '[0-9]+$' || echo 0)
    kinks=$(grep -oE '"kinks": [0-9]+' "$meta" | grep -oE '[0-9]+$' || echo 0)
    twist=$(grep -oE '"twist": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    slant=$(grep -oE '"slant_p95": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    wrapf=$(grep -oE '"wrap_frac": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    dmean=$(grep -oE '"mean": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
    dcov=$(grep -oE '"coverage": [0-9.]+' "$meta" | grep -oE '[0-9.]+$' || echo 0)
  else
    area=0; fill=0; hole=0; folds=0; kinks=0; twist=0; slant=0; wrapf=0; dmean=0; dcov=0
  fi
  printf '{"name":"%s","nset":%s,"area_vx2":%s,"fill":%s,"hole":%s,"folds":%s,"kinks":%s,"twist":%s,"slant_p95":%s,"donor_mean":%s,"donor_cov":%s,"wall_s":%s}\n' \
    "$name" "$nset" "$area" "$fill" "$hole" "$folds" "$kinks" "$twist" "$slant" "$dmean" "$dcov" "$wall"
}

for rep in 1 2 3; do
  for spec in "${SEEDS[@]}"; do
    read -r name sx sy sz mode <<<"$spec"
    dir="$OUT/$name-r$rep"
    mkdir -p "$dir"
    log="$dir/trace.log"
    args=("$PRED" --seed "$sx" "$sy" "$sz" --gens "$GENS" --level 1 --out "$dir")
    [ -f "$DATA/p343-lod/umbilicus.json" ] && args+=(--umbilicus "$DATA/p343-lod/umbilicus.json")
    [ "$mode" = fuse ] && args+=(--fuse "$DONOR")
    t0=$(date +%s.%N)
    "$BIN" "${args[@]}" >"$log" 2>&1 || echo "  $name-r$rep FAILED rc=$?"
    t1=$(date +%s.%N)
    wall=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')
    row "$name-r$rep" "$dir" "$log" "$wall" | tee -a "$OUT/rows.jsonl"
  done
done

if [ "$FULL" = 1 ]; then
  dir="$OUT/center-full"
  mkdir -p "$dir"
  log="$dir/trace.log"
  t0=$(date +%s.%N)
  "$BIN" "$PRED" --seed 4297 4297 8999 --gens 60 --level 1 --out "$dir" >"$log" 2>&1 || true
  t1=$(date +%s.%N)
  wall=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')
  row "center-full" "$dir" "$log" "$wall" | tee -a "$OUT/rows.jsonl"
fi

echo "rows: $OUT/rows.jsonl"
if [ "$RECORD" = 1 ]; then
  mkdir -p "$BASE"
  cp "$OUT/rows.jsonl" "$BASE/rows.jsonl"
  echo "baseline recorded"
elif [ -f "$BASE/rows.jsonl" ]; then
  echo "--- baseline ---"
  cat "$BASE/rows.jsonl"
  echo "--- current ----"
  cat "$OUT/rows.jsonl"
fi
