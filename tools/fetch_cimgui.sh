#!/usr/bin/env sh
# Vendor the pinned cimgui (Dear ImGui C bindings) + bundled imgui into
# tools/cimgui/ (git-ignored). Bump CIMGUI_TAG deliberately; record in
# docs/measured.md.
set -eu

CIMGUI_TAG="1.92.9"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools/cimgui"
STAMP="$DEST/.version"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$CIMGUI_TAG" ]; then
  echo "cimgui $CIMGUI_TAG already vendored at $DEST"
  exit 0
fi

rm -rf "$DEST"
git clone --depth 1 --branch "$CIMGUI_TAG" --recurse-submodules --shallow-submodules \
  https://github.com/cimgui/cimgui "$DEST"
rm -rf "$DEST/.git" "$DEST/imgui/.git"
printf '%s' "$CIMGUI_TAG" > "$STAMP"
echo "vendored cimgui $CIMGUI_TAG into $DEST"
