#!/usr/bin/env sh
# Vendor the pinned Slang compiler release into tools/slang/ (git-ignored).
# Bump SLANG_VERSION deliberately; record toolchain bumps in docs/measured.md.
set -eu

SLANG_VERSION="2026.14.1"
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64) SLANG_ARCH="aarch64" ;;
  x86_64) SLANG_ARCH="x86_64" ;;
  *) echo "fetch_slang: unsupported arch $ARCH" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools/slang"
STAMP="$DEST/.version"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$SLANG_VERSION-$SLANG_ARCH" ]; then
  echo "slangc $SLANG_VERSION already vendored at $DEST"
  exit 0
fi

NAME="slang-$SLANG_VERSION-linux-$SLANG_ARCH.tar.gz"
URL="https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/$NAME"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching $URL"
curl -fsSL -o "$TMP/$NAME" "$URL"
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$TMP/$NAME" -C "$DEST"
printf '%s' "$SLANG_VERSION-$SLANG_ARCH" > "$STAMP"
"$DEST/bin/slangc" -v
echo "vendored slangc into $DEST/bin"
