#!/bin/sh
# Fetch every segment of a scroll from the open-data bucket (the variant
# pre-transformed onto the given volume), pack each into the segment store,
# and delete the tifxyz source — disk stays bounded at one segment.
#   tools/fetch_segments.sh <scroll> <volume-suffix> <store-dir> [segpack]
# e.g. tools/fetch_segments.sh PHercParis4 20260411134726-2.4um cache/PHercParis4-segstore
set -u
B=https://vesuvius-challenge-open-data.s3.amazonaws.com
SCROLL=${1:?scroll}
VOL=${2:?volume-suffix}
STORE=${3:?store-dir}
SEGPACK=${4:-./build/native/segpack}
TMP=$STORE/.fetch
mkdir -p "$STORE" "$TMP"
LOG=$TMP/fetch.log

curl -fsS "$B/?list-type=2&delimiter=/&prefix=$SCROLL/segments/" |
  grep -o "<Prefix>$SCROLL/segments/[^<]*" |
  sed "s|<Prefix>$SCROLL/segments/||;s|/\$||" | grep -v '^$' > "$TMP/seglist.txt"
total=$(wc -l < "$TMP/seglist.txt")
echo "$(date +%T) fetching $total segments of $SCROLL onto $VOL" >> "$LOG"

done_n=0
fail_n=0
while read -r seg; do
  # variant dirs are named by the segment's timestamp id, which is the dir
  # name up to the first dash (suffixed dirs like <id>-w046-052_jordi)
  id=${seg%%-*}
  name="$id-on-$VOL.tifxyz"
  if [ -e "$STORE/$name.tfx" ]; then
    done_n=$((done_n + 1))
    continue
  fi
  d=$TMP/$name
  rm -rf "$d"
  mkdir -p "$d"
  u=$B/$SCROLL/segments/$seg/mesh/$name
  if curl -fsS --retry 3 --parallel \
       -o "$d/x.tif" "$u/x.tif" -o "$d/y.tif" "$u/y.tif" \
       -o "$d/z.tif" "$u/z.tif" -o "$d/meta.json" "$u/meta.json"; then
    if "$SEGPACK" "$STORE" -q 2 "$d" >> "$LOG" 2>&1; then
      rm -rf "$d"
      done_n=$((done_n + 1))
      echo "$(date +%T) [$done_n/$total] packed $name" >> "$LOG"
    else
      fail_n=$((fail_n + 1))
      echo "$(date +%T) PACK FAILED $name (source kept)" >> "$LOG"
    fi
  else
    fail_n=$((fail_n + 1))
    rm -rf "$d"
    echo "$(date +%T) FETCH FAILED $seg (no $VOL variant?)" >> "$LOG"
  fi
done < "$TMP/seglist.txt"
echo "$(date +%T) DONE: $done_n packed, $fail_n failed" >> "$LOG"
