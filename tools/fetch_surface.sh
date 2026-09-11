#!/bin/sh
# Fetch one published segment surface from the Vesuvius Challenge open-data
# bucket and convert it to render3d's default surface type (.sfc).
#
#   tools/fetch_surface.sh <scroll> <segment-id> [volume-suffix] [out-dir] [surfconv]
#   tools/fetch_surface.sh <scroll> --list            segment ids of a scroll
#   tools/fetch_surface.sh <scroll> <segment-id> --list   mesh variants
#
# e.g. tools/fetch_surface.sh PHercParis4 20230702185753 20260411134726-2.4um
# downloads .../segments/20230702185753/mesh/20230702185753-on-20260411134726-2.4um.tifxyz
# into <out-dir>/ (default cache/od/segments), encodes <name>.sfc beside it
# with surfconv (error budget R3D_SFC_ERROR, default 0.1 voxel) and prints the
# .sfc path. Without a volume suffix the first published variant is used.
# The tifxyz planes are kept; `surfconv decode` regenerates them from the
# .sfc at any time, so they can be deleted to bound disk use.
set -eu
B=https://vesuvius-challenge-open-data.s3.amazonaws.com
SCROLL=${1:?scroll (e.g. PHercParis4)}
SEG=${2:?segment id or --list}

list_prefixes() {
  curl -fsS "$B/?list-type=2&delimiter=/&prefix=$1" |
    grep -o "<Prefix>$1[^<]*" | sed "s|<Prefix>$1||;s|/\$||" | grep -v '^$'
}

if [ "$SEG" = "--list" ]; then
  list_prefixes "$SCROLL/segments/"
  exit 0
fi
if [ "${3:-}" = "--list" ]; then
  list_prefixes "$SCROLL/segments/$SEG/mesh/" | grep "\.tifxyz$"
  exit 0
fi
VOL=${3:-}
OUT=${4:-cache/od/segments}
SURFCONV=${5:-./build/macos/surfconv}
[ -x "$SURFCONV" ] || SURFCONV=./build/release/surfconv
[ -x "$SURFCONV" ] || SURFCONV=./build/native/surfconv
[ -x "$SURFCONV" ] || { echo "surfconv binary not found (build it first)" >&2; exit 1; }

if [ -n "$VOL" ]; then
  NAME="$SEG-on-$VOL.tifxyz"
else
  NAME=$(list_prefixes "$SCROLL/segments/$SEG/mesh/" | grep '\.tifxyz$' | head -n 1)
  [ -n "$NAME" ] || { echo "no tifxyz variant published for $SCROLL/$SEG" >&2; exit 1; }
fi
mkdir -p "$OUT/$NAME"
U=$B/$SCROLL/segments/$SEG/mesh/$NAME
echo "fetching $U" >&2
curl -fsS --retry 3 --parallel \
  -o "$OUT/$NAME/x.tif" "$U/x.tif" -o "$OUT/$NAME/y.tif" "$U/y.tif" \
  -o "$OUT/$NAME/z.tif" "$U/z.tif" -o "$OUT/$NAME/meta.json" "$U/meta.json"
"$SURFCONV" resolve "$OUT/$NAME"
