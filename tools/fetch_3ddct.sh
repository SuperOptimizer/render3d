#!/usr/bin/env sh
# Vendor the pinned dct3d codec (SuperOptimizer/3ddct, MIT) into tools/3ddct/
# (git-ignored). Only dct3d.{c,h} are used. Bump the pin deliberately.
set -eu

PIN="8e9e31d9ba3ddf5c0901e8443ab751c50b4f0e65"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools/3ddct"
STAMP="$DEST/.version"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$PIN" ]; then
  echo "3ddct $PIN already vendored"
  exit 0
fi

rm -rf "$DEST"
git clone -q https://github.com/SuperOptimizer/3ddct "$DEST"
git -C "$DEST" checkout -q "$PIN"
rm -rf "$DEST/.git"
printf '%s' "$PIN" > "$STAMP"
echo "vendored 3ddct @ $PIN"
