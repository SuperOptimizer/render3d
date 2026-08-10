#!/bin/sh
# Fetch zarr v2 chunks listed by `zarr2c5d --list-missing` into a local mirror.
# A 404 leaves a "<path>.missing" marker so zarr2c5d can tell "absent object
# (= zero-fill chunk)" from "not yet downloaded".  Resumable; parallel.
#
# usage: fetch_chunks.sh <base-url> <mirror-dir> <list-file> [parallel=16]
set -eu
BASE=$1
MIRROR=$2
LIST=$3
PAR=${4:-16}
export BASE MIRROR

xargs -P "$PAR" -n 1 -a "$LIST" sh -c '
  rel=$1
  dst="$MIRROR/$rel"
  [ -s "$dst" ] && exit 0
  [ -e "$dst.missing" ] && exit 0
  mkdir -p "$(dirname "$dst")"
  code=$(curl -sS -w "%{http_code}" -o "$dst.part" "$BASE/$rel") || code=000
  case "$code" in
    200) mv "$dst.part" "$dst" ;;
    404) rm -f "$dst.part"; : > "$dst.missing" ;;
    *)   rm -f "$dst.part"; echo "fetch_chunks: $rel -> HTTP $code" >&2; exit 1 ;;
  esac
' fetch
echo "fetch_chunks: done ($(wc -l < "$LIST") listed)"
