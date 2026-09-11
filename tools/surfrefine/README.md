# surfrefine — refining published surfaces

Semi-automated refinement of segmentation surfaces (`.sfc` or tifxyz) against
surface-prediction volumes and CT, with human corrections in the viewer, and
export of the refined surfaces as training labels for the prediction models.

## Batch: `surfrefine` and `batch.py`

```sh
build/macos/surfrefine --pred <prediction LOD tree> --in seg.sfc --out cache/refined/seg \
    [--ct-root <CT LOD tree> --ct-cut 128 --ct-reach 6] [--level 1] [--subdivide N] \
    [--no-refine] [--no-ctsnap] [--qc-only] [--cutoff 0.35] [--flag-conf 0.5] \
    [--report qc.json] [--threads N]
```

The surface is loaded into the tracer (`r3d_tracer_load`; a published surface
without `tracer.json` is imported with assumed defaults), its mesh QC is
recorded as the baseline, then: optional `--subdivide` (halve the grid step),
the solve-only refine pass toward the predictions (staged weights, tangential
position memory), the optional CT edge snap, and QC again. The result is a new
tifxyz directory plus `<out>.sfc` and `<out>/refine_qc.json`:

```json
{"before": {"folds": 0, "kinks": 3, "fill": 0.97, "conf_mean": 0.61, "area_vx2": ...},
 "after":  {...},
 "flagged": [{"tile": [ti, tj], "grid": [i0, j0, i1, j1], "center": [x, y, z],
              "trusted": 0.31, "points": 190, "defects": 1}, ...],
 "flagged_count": 4}
```

A flagged tile (16×16 grid cells, the segment-store tile size) has fewer than
half its points at or above `--flag-conf` after refinement, or holds a fold or
sharp kink. `--out` may never be the source or lie inside it.

`batch.py` runs the tool over a corpus with resume-by-skip and one aggregate
report the viewer's review queue reads:

```sh
python3 tools/surfrefine/batch.py --surfrefine build/macos/surfrefine --pred <tree> \
    --out-dir cache/refined --glob 'cache/od/segments/*.sfc' --workers 2
# or straight from the published mirror, uploading refined copies next to it
python3 tools/surfrefine/batch.py --surfrefine build/macos/surfrefine --pred <tree> \
    --out-dir cache/refined --mirror surfcomp --scroll PHercParis4 \
    --upload surfcomp-refined --sftp-config ~/ash2txt
```

## Viewer: editing a surface

In the tracer panel, **edit active surface** loads the displayed segment into
a stopped tracer (`r3d_tracer_import`). Every tracer control then applies:
re-solve, rewind, subdivide, snap to CT edge, spiral fill. Corrections:

- **Shift+drag** a surface vertex in a plane pane to where the sheet should
  pass. Short drags (under 1.5 grid steps) add an anchor at the target and
  re-solve; long drags, or Ctrl held, add the anchor and reopen-and-regrow
  the region around the dragged cell (radius slider). Anchors pull along the
  local sheet normal only, so the parameterisation is not dragged sideways.
- **save version** writes `cache/refined/<name>-vN` (tifxyz + `.sfc`) and opens
  it; the loaded surface is never overwritten (a save inside it is refused).
- **discard edits** drops the tracer and keeps the displayed surface.

The **review** section loads `corpus_report.json`; picking a flagged tile
opens that refined surface centred on the tile, ready for editing.

Headless hooks for tests: `R3D_EDIT_TEST=<frame>` imports the active surface
at that frame; `R3D_DRAG_TEST="x,y,z>x,y,z[!]"` applies a drag ten frames
later (`!` forces the regrow path).

## Label export

- `tifxyz2obj <surface> out.obj [--name N]`: OBJ with `v`/`vt`/`vn`, two
  triangles per valid quad, for villa's `voxelize_objs.py`.
- `surfsamples <surface> out.bin [--spacing 1.0] [--conf-min 0.25]`: surface
  samples `(x, y, z, nx, ny, nz, conf)` at voxel spacing (bilinear subdivision
  of every quad; normal = `(∂P/∂v) × (∂P/∂u)`), header `R3DS` + uint32 count.
- `export_tsm_labels.py --store fine.zarr --out fine-refined.zarr --samples a.bin ...
  [--axis axis.json] [--clip 20] [--conf-min 0.25]`: copy-on-write rewrite of the
  TSM label store's `sdf`/`sdf_valid` channels within a band around the
  samples (sign outward from the scroll axis, as `label_store.md` requires);
  untouched chunks stay byte-identical, in-place writes are refused. Runs in
  the tsm venv (`cd build/tsm-source && uv run python ../../tools/surfrefine/export_tsm_labels.py ...`).

## Tests

`ctest -L quick` covers `surfrefine` (synthetic spiral prediction tree, a
surface perturbed off the wrap: refine must halve the off-sheet distance
without new folds, `.sfc` input, source guard) and `surfrefine_export`;
`surfrefine_gui` (label `gpu`) runs the viewer headless through the import,
drag-to-anchor and reopen-and-regrow paths. `tests/test_export_tsm_labels.py`
runs under `uv run --with pytest` in the tsm checkout.
