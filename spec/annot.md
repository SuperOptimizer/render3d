# TSM faces-annotation packets (v1)

The contract between the TSM exporter (`~/tsm/dev/annot_export.py`) and
render3d's `--annot` mode. A **packet** is one small CT crop with the model's
two-face labels attached; a human corrects those labels by painting a
`correction.u8` next to them, which the trainer reads back as a per-voxel
override.

## Layout

```
<root>/
  packet.json          batch manifest (optional: a lone packet dir also opens)
  p000/
    meta.json
    ct.u8              REQUIRED
    faces_in.u8        1 = in-face voxel
    faces_out.u8       1 = out-face voxel
    ignore.u8          1 = excluded from the loss
    source.u8          0 none / 1 rectoverso / 2 ct / 3 human
    rv_class.u8        0 bg / 1 recto / 2 verso / 3 contact
    pred_in.u8         optional model prediction
    pred_out.u8        optional model prediction
    correction.u8      WRITTEN BY render3d
  p001/
    ...
```

## Volumes

Every `.u8` file is a headerless array of `Z*Y*X` bytes in **C order (z,y,x)** —
x is the fastest-varying axis, exactly `numpy.ascontiguousarray(a.astype(
np.uint8))` for an array indexed `a[z,y,x]`. This matches the rest of render3d
(`spec/volume.md`) and TSM's own zarr order.

Every volume in a packet has the **identical** shape, the one given by
`meta.json`'s `dims_zyx`. A file whose size does not equal `Z*Y*X` is a hard
error, not a warning: a silently mis-shaped label layer would be painted over.
Layer files other than `ct.u8` are optional; a missing file means "this packet
has no such layer" and is not an error.

Typical crops are 256³; 512³ is the practical ceiling (the CT is also uploaded
as one GPU volume, and `maxImageDimension3D` is 2048).

## meta.json

```json
{
  "dims_zyx": [256, 256, 256],
  "origin_zyx": [34176, 4096, 8192],
  "voxel_um": 2.4,
  "layers": ["ct", "faces_in", "faces_out", "ignore", "source", "rv_class"],
  "done": false
}
```

- `dims_zyx` — required, positive, ≤ 4096 per axis.
- `origin_zyx` — packet voxel `(0,0,0)` in the source scroll's coordinates.
  Display only; render3d never resamples. Defaults to zeros if absent.
- `voxel_um` — informational.
- `layers` — advisory list of the layers the exporter wrote. render3d decides
  by file existence, so this may be omitted, but keeping it makes a packet
  self-describing.
- `done` — set to `true` by the "mark packet done" checkbox. render3d rewrites
  **only this key's value token** and preserves every other key and all
  formatting verbatim, so the exporter may carry any extra metadata it likes
  (provenance, model revision, sampling reason) without risk of losing it.

## packet.json

```json
{
  "format": "tsm.annot.v1",
  "voxel_um": 2.4,
  "palette": {
    "correction": ["untouched", "in", "out", "ignore", "erase"],
    "source":     ["none", "rectoverso", "ct", "human"],
    "rv_class":   ["bg", "recto", "verso", "contact"]
  },
  "packets": [
    {"path": "p000", "origin_zyx": [34176, 4096, 8192], "dims_zyx": [256, 256, 256]},
    {"path": "p001", "origin_zyx": [34176, 4096, 8448], "dims_zyx": [256, 256, 256]}
  ]
}
```

Only `packets[].path` is required; it is resolved relative to the directory
holding `packet.json` (an absolute path is taken as-is). `origin_zyx` and
`dims_zyx` are used for the packet list before a packet is opened; `meta.json`
remains the authority once it is. `palette` is documentation — render3d's
class semantics are fixed by this spec, not read from the manifest.

Passing render3d a directory looks for `packet.json` in it and falls back to
treating the directory itself as a single packet (it must contain
`meta.json`). Passing a `packet.json` path directly also works.

## correction.u8

Same shape and order as every other volume. One class per voxel:

| value | meaning |
| --- | --- |
| 0 | untouched — the exporter's labels stand for this voxel |
| 1 | in-face |
| 2 | out-face |
| 3 | ignore |
| 4 | erase — a positive assertion that neither face is here |

`4` and `0` are deliberately distinct. `0` is the absence of an opinion; `4`
says a human looked and there is no face, which is a training signal.

render3d publishes the file atomically — written to
`<packet>/.correction.u8.tmp.<pid>`, fsynced, then `rename(2)`d over
`correction.u8` — after **every** stroke, on packet switch and at exit. A
reader therefore always sees a complete, correctly-sized file, and a killed
session loses at most the stroke in progress. An existing `correction.u8` is
loaded when the packet is opened; values ≥ 5 are clamped to 0.

## Stroke scripts (`--annot-apply`)

The batch/test entry point applies a JSON stroke list to a packet and writes
`correction.u8`, with no window and no GPU:

```json
{"strokes": [
  {"op": "brush", "class": 1, "z": 128, "radius": 3, "depth": 1,
   "points": [[100, 90], [104, 96], [112, 99]]},
  {"op": "line",  "class": 2, "z": 128, "points": [[0, 64], [255, 64]]},
  {"op": "undo"}
]}
```

- `brush` sweeps a capsule of `radius` voxels through the polyline (so a fast
  drag leaves no gaps); `radius: 0` is a single voxel.
- `line` draws a 1-voxel Bresenham polyline and ignores `radius`.
- `depth: k` paints slices `z-k .. z+k` — a cylinder through z, clamped to the
  volume, never wrapped. `depth: 0` (the default) is the current slice only.
- `class: 0` erases back to untouched, which is what the right mouse button
  does interactively.
- `undo` reverts the previous stroke. Undo is per stroke and stores only the
  voxels that actually changed.

Coordinates are packet-local voxels: `x` across, `y` down, `z` the slice.

The packet argument accepts the same three forms as `--annot`; given a
manifest, the script is applied to its **first** packet.
