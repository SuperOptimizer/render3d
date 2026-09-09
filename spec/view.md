# TSM model-view packets and the live prediction server (v1)

The contract between TSM (`~/tsm`, the unified multi-head student and its
teachers) and render3d's `--view` mode. A **view packet** is one CT crop plus
any number of named uint8 **layers** — student heads, teacher outputs, label
stores, human bands — each tagged with a *kind* that tells the viewer how to
decode and draw it. The viewer overlays layers on the CT, compares any two
layers of the same kind, and reads every layer's value under the cursor.

The same layer vocabulary is served live by `tsm serve` (TCP), so the viewer
can ask a remote GPU for the student's prediction of any box on demand.

This spec supersedes nothing: `spec/annot.md` packets stay as they are (they
are a *correction* workflow with a fixed layer set). A view packet is a
superset in spirit — a directory of `.u8` volumes with a `meta.json` — but
the layer list is **data-driven** and no correction is written.

## Layout

```
<root>/
  view.json            batch manifest (optional: a lone packet dir also opens)
  v000/
    meta.json
    ct.u8              REQUIRED
    <layer>.u8         one file per layer listed in meta.json
  v001/
    ...
```

Every `.u8` is a headerless `Z*Y*X` byte array in C order (z,y,x), x
fastest, exactly `numpy.ascontiguousarray(a.astype(np.uint8))` for `a[z,y,x]`
(see `spec/volume.md`). Every volume in a packet has the shape given by
`meta.json`'s `dims_zyx`. A file whose size differs from `Z*Y*X` is a hard
error. A layer listed in `meta.json` whose file is missing is a hard error
(unlike annot packets: a view packet's manifest is the authority, because the
viewer builds its layer list from it before touching any file).

Typical crops are 256³ to 512³. The viewer never resamples.

## meta.json

```json
{
  "format": "tsm.view.v1",
  "dims_zyx": [256, 512, 512],
  "origin_zyx": [34560, 15360, 18688],
  "voxel_um": 2.4,
  "scroll": "PHercParis4",
  "layers": [
    {"file": "ct.u8", "name": "ct", "group": "ct", "kind": "ct"},
    {"file": "student.sdf_in.u8",  "name": "sdf_in",  "group": "student", "kind": "sdf", "clip": 20.0},
    {"file": "student.sdf_out.u8", "name": "sdf_out", "group": "student", "kind": "sdf", "clip": 20.0},
    {"file": "student.valid.u8",   "name": "valid",   "group": "student", "kind": "prob"},
    {"file": "student.ink.u8",     "name": "ink",     "group": "student", "kind": "prob"},
    {"file": "student.sin.u8",     "name": "sin",     "group": "student", "kind": "signed"},
    {"file": "student.cos.u8",     "name": "cos",     "group": "student", "kind": "signed"},
    {"file": "student.density.u8", "name": "density", "group": "student", "kind": "density"},
    {"file": "student.nx.u8",      "name": "nx",      "group": "student", "kind": "signed", "vec": "normal", "axis": 0},
    {"file": "student.ny.u8",      "name": "ny",      "group": "student", "kind": "signed", "vec": "normal", "axis": 1},
    {"file": "student.nz.u8",      "name": "nz",      "group": "student", "kind": "signed", "vec": "normal", "axis": 2},
    {"file": "student.conf.u8",    "name": "conf",    "group": "student", "kind": "prob"},
    {"file": "student.fiber_vt.u8","name": "fiber_vt","group": "student", "kind": "prob"},
    {"file": "student.fiber_hz.u8","name": "fiber_hz","group": "student", "kind": "prob"},
    {"file": "student.thickness.u8","name": "thickness","group": "student","kind": "count"},
    {"file": "teacher.recto.u8",   "name": "recto",   "group": "teacher", "kind": "prob"},
    {"file": "teacher.ink.u8",     "name": "ink",     "group": "teacher", "kind": "prob"},
    {"file": "teacher.fiber_vt.u8","name": "fiber_vt","group": "teacher", "kind": "prob"},
    {"file": "teacher.fiber_hz.u8","name": "fiber_hz","group": "teacher", "kind": "prob"},
    {"file": "label.sdf_in.u8",    "name": "sdf_in",  "group": "label",   "kind": "sdf", "clip": 20.0},
    {"file": "label.sdf_out.u8",   "name": "sdf_out", "group": "label",   "kind": "sdf", "clip": 20.0},
    {"file": "label.faces_valid.u8","name": "faces_valid","group": "label","kind": "class",
       "palette": ["invalid", "valid", "ignore"]},
    {"file": "human.rv_class.u8",  "name": "rv_class","group": "human",   "kind": "class",
       "palette": ["bg", "recto", "verso", "contact"]},
    {"file": "human.hzvt_class.u8","name": "hzvt_class","group": "human", "kind": "class",
       "palette": ["bg", "hz", "vt", "exclude"]}
  ],
  "provenance": {"checkpoint": "/home/forrest/tsm-output/multi_s1_s4/train/latest.pt", "step": 30000}
}
```

- `format` — must be `"tsm.view.v1"`.
- `dims_zyx` — required, positive, ≤ 4096 per axis (≤ 512 recommended).
- `origin_zyx` — packet voxel (0,0,0) in scroll coordinates (level 0).
  Display only. Defaults to zeros.
- `voxel_um`, `scroll` — informational.
- `layers[]` — ordered. Exactly one entry has `kind: "ct"` and it is
  `ct.u8`. `group` + `name` is unique. `file` is relative to the packet dir.
  The viewer shows layers grouped by `group` in manifest order.
- `provenance` — free-form; the viewer shows it verbatim as text.
- Unknown keys anywhere are ignored, never an error.

## Layer kinds and their byte decodings

These are TSM's own encodings (`src/tsm/data.py`, `src/tsm/labels.py`,
`src/tsm/infer.py`). The viewer decodes on the fly; it never rewrites bytes.

| kind | value | decode | default rendering |
| --- | --- | --- | --- |
| `ct` | CT intensity | `u/255` | grayscale base image |
| `sdf` | signed distance, voxels | `(u-128) * clip/127`; `u == 0` means **no data** | zero-crossing band `|d| ≤ tol` (tol default 1.5 vox) in the layer colour; optional signed heatmap (blue negative / red positive) |
| `prob` | probability | `u/255` | tinted alpha over CT, alpha = p × opacity; threshold slider hides `p < t` |
| `signed` | value in [-1,1] | `(u-127.5)/127.5` | diverging heatmap; three `signed` layers sharing a `vec` name are drawn as one RGB normal image (`0.5+0.5·v`) |
| `density` | sheets per voxel | `u/1000` | sequential heatmap, range auto from the slice |
| `count` | integer (thickness, voxels) | `u` | sequential heatmap, 0 = transparent |
| `class` | categorical | `u`, names from `palette` | fixed palette colour per class, class 0 transparent; hover shows the name |

Layer entry keys by kind: `clip` (sdf, required), `palette` (class, optional),
`vec` + `axis` (signed, optional; `axis` 0/1/2 = x/y/z component → R/G/B).

## Viewer behaviour (what `--view` must do)

1. **Slice view** with axis toggle (z, y, x planes), slice slider, zoom/pan,
   over the CT. Same CPU-composited RGBA slice + ImGui texture approach as
   `--annot` (`src/core/annot.c`, `src/annotui.c`).
2. **Layer panel**: every non-ct layer has show / colour / opacity /
   threshold (prob, density, count) / tol (sdf) / signed-heatmap toggle (sdf).
   Groups collapse. "Solo" a layer with a click.
3. **Compare mode**: pick layer A and layer B of the same kind. Draw
   agreement / A-only / B-only in three colours (default green / red / blue).
   For `sdf`, "present" means inside the zero-crossing band; for `prob`, above
   the threshold; for `class`, non-zero (or a chosen class). Show the counts
   and a Dice number for the current slice in the panel.
4. **Hover readout**: decoded value of every visible layer at the cursor,
   plus the scroll-space coordinate.
5. **Packet list** from `view.json` (`packets[].path`, like `packet.json`),
   Prev / Next.
6. **Screenshot** (F12, PNG via `src/core/pngw.c`) of the composited view.
7. **Headless entry** `--view-shot <packet> <out.png> [--axis z --slice 128
   --show group.name,...]` for tests and for TSM's report tooling: composites
   without a window and writes the PNG. Tests use it.
8. **Live mode** (`--view-serve host:port`, optional): the panel gains a
   "predict" button that sends the packet's box (or a box moved by the
   arrow-key nudges) to `tsm serve` and adds/replaces the returned layers in
   the `student` group (or the group the server names). Requests run on a
   worker thread; the UI never blocks. Protocol below.

## Live protocol (`tsm serve`)

Mirrors `tools/surf/surfserver.py`: TCP, little-endian, one request at a time
per connection, framed with 4-byte magics.

```
request : 'TSV1' u32 hdr_len   then hdr_len bytes of JSON
response: 'TSVR' u32 hdr_len   then hdr_len bytes of JSON, then the layer bytes
error   : 'TSVE' u32 msg_len   then msg_len bytes of UTF-8 text
```

Request JSON:

```json
{"origin_zyx": [34560, 15360, 18688], "dims_zyx": [256, 512, 512],
 "want": ["student", "ct"], "tta": "none"}
```

- `origin_zyx` / `dims_zyx` — the box in scroll level-0 voxels. The server
  clamps `dims_zyx` to its configured maximum (default 512³) and rejects boxes
  outside the volume with `TSVE`.
- `want` — groups to return; the server reports which groups it can serve in
  its `hello`. `"ct"` is served from the server's own reader so the viewer can
  open a box it has no packet for.
- `tta` — `none | flip8 | flip8_rot4` (server may refuse).

Response JSON is a `meta.json` (same schema as above, `format`
`"tsm.view.v1"`) whose `layers[]` describe the bytes that follow, in order,
each exactly `Z*Y*X` bytes. The viewer treats it exactly like a packet
directory it read from disk.

On connect the server sends nothing; the viewer sends a `{"hello": true}`
request and gets a `TSVR` with an empty layer list whose JSON carries
`{"groups": ["student", "ct", "teacher"], "max_dims_zyx": [512,512,512],
"scroll": ..., "checkpoint": ...}`.

The server runs on a GPU machine (`forlindesk2` or the Thunder A6000); the
viewer reaches it over an ssh tunnel (`ssh -L 9760:localhost:9760 host`).
The laptop GPU is never used.

## Exporter (`tsm view` / `dev/view_export.py`)

Writes one packet per requested box:

```
uv run python dev/view_export.py configs/paris4_eval_multi.json \
    --out ~/tsm-output/view/multi_p4 \
    --box 34560,15360,18688,256,512,512 [--box ...] \
    [--pred /path/pred.zarr | --checkpoint latest.pt] \
    [--teachers /path/teachers] [--labels /path/fine.zarr] \
    [--rectoverso /path/rectoverso.zarr] [--group-name student]
```

- `--pred` copies channels from an existing `pred.zarr` (fast, no GPU).
  `--checkpoint` runs the student on the box (needs a GPU; refuse on the
  laptop). One of the two is required unless only teacher/label groups are
  wanted.
- `--teachers` adds `teacher.recto`, `teacher.ink`, `teacher.fiber_vt/hz`,
  and from `lasagna.zarr` (9.6 µm) the upsampled `teacher.cos` as `signed`.
- `--labels` adds `label.sdf_in/out`, `label.faces_valid` (class), `label.ink`.
- `--rectoverso` adds `human.rv_class`, `human.hzvt_class` (class).
- Every store is read with its own `origin_zyx`; a box not fully covered by a
  store drops that store's layers with a warning rather than padding.
- Writes `view.json` listing the packets.

Also exposes the same code as `tsm serve` (`src/tsm/serve.py`).
