# Just-in-time TSM overlays

`tools/tsm/serve.py` runs the Paris4 multi-head model on demand and exposes its
predictions to render3d's existing overlay streamer. On macOS it selects PyTorch
MPS; CUDA and CPU are also supported. Rendering continues through Vulkan/MoltenVK.
The model architecture, preprocessing, window blending and byte encodings come
from the pinned [TSM source](https://github.com/SuperOptimizer/tsm).

## Install and run

Build render3d normally first, including `librender3d_headless.dylib`. From the
render3d repository root, install a Python 3.12 environment:

```sh
uv venv --python 3.12 build/tsm-venv
uv pip install --python build/tsm-venv/bin/python -r tools/tsm/requirements.txt
mkdir -p cache/models/tsm_paris4_rvfw_30k
curl --fail --location --retry 3 \
  https://dl.ash2txt.org/community-uploads/forrest/tsm/checkpoints/tsm_paris4_rvfw_30k/latest.pt \
  -o cache/models/tsm_paris4_rvfw_30k/latest.pt
```

Start the service in one terminal (replace the CT root for another imported
volume). The CT tree needs its `manifest.json` and `source.json`:

```sh
build/tsm-venv/bin/python tools/tsm/serve.py \
  --ct-root cache/volcomp/PHerc1667/20260323082859-1.129um-0.2m-59keV-masked.zarr \
  --checkpoint cache/models/tsm_paris4_rvfw_30k/latest.pt \
  --bundle cache/inference/PHerc1667-tsm
```

After it prints `Listening`, open the GUI in another terminal:

```sh
build/macos/render3d --inference cache/inference/PHerc1667-tsm
```

The bundle selects its bound CT volume automatically. In the **TSM overlays** panel,
choose independent **Blue head** and **Red head** layers, adjust their gains and
choose which views show them. Defaults are inner surface and ink. Drag to pan;
zoom in to request predictions near the model's resolution. Status reports
inference progress and the usable display levels. Switching CT datasets automatically switches inference to the new volume.
The selectors stay visible and are disabled while connecting. The old worker
finishes its active region, discards queued requests and reuses the loaded
model with the new volume’s pitch and axis. Each volume has its own cache
generation; previous-volume predictions are never attached to the new CT.
An explicit `--bricks` argument also selects the inference volume.

## Heads and units

The checkpoint has 14 learned channels. Sixteen layers are exposed: the 13
supervised channels plus inner/outer thin surfaces and thickness; the unused
`spare` channel is omitted. Available fields include ink, inner/outer signed
surface distance, validity, fibre VT/HZ, winding sine/cosine, density, normals,
confidence, thin surfaces and thickness.

Distance bytes decode as `(byte - 128) * 20 / 127` model voxels; zero is missing.
Signed components decode as `(byte - 127.5) / 127.5`, probabilities as `byte / 255`,
and density as `byte / 1000`. Thickness is in model voxels. The GUI tints the
encoded intensities; for a thin surface overlay select `surface_in1` or
`surface_out1`, rather than the signed-distance ramp. Full layer metadata is in
`inference.json`.

## Resolution, scheduling and cache

The service chooses the CT pyramid level closest to the checkpoint's 2.4 µm
training pitch and supplies the actual pitch to the model's scale channel.
PHerc1667 uses L1 at 2.258 µm. Override with `--voxel-um` (level-zero pitch) and
`--level`. Radial inputs use `--axis` in full-resolution umbilicus JSON format,
or automatically use `<ct-root>/umbilicus.json`. On a volume switch, the new
volume supplies its own pitch and umbilicus; the previous CT’s axis is not reused. Without one, the service clearly
reports an approximate volume-center axis. A measured axis is needed to assess
prediction quality; successful execution alone does not establish accuracy.

Each native 128³ output region reads 192³ CT context and blends eight overlapping
128³ model windows. This is windowed inference, not a full-receptive-field exact
whole-volume evaluation. Renderer decoding and its GPU atlas remain on 16³ blocks.
All heads share one inference job and one atomic result file. HTTP requests
return 202 while inference is pending, freeing renderer fetch workers; visible
missing overlays retry after 250 ms. Missing CT input fails the job instead of
silently predicting over zero-filled download failures.

Native and finer display levels reuse these results. One coarser level pools
native tiles, preserving thin surfaces with max pooling and excluding missing
SDF samples from distance averages. More distant overview levels have no overlay;
zoom to the level indicated in the panel. This avoids launching thousands of
fine-resolution predictions for a whole-scroll overview.

One model worker serializes inference. Up to 32 queued regions are prioritized
by recent demand; unrefreshed queued jobs expire after three seconds. The active
region finishes even after navigation changes. `--cache-mb` bounds the shared
prediction cache (default 1024 MiB) **per bundle generation**. Identity includes
the checkpoint hash, CT metadata/path, axis, pitch and inference configuration.
Old generations and render3d's separate per-head compressed download caches
remain on disk until removed. Do not delete caches while they are in use.

## Validation on Apple M4

The downloaded checkpoint completed a real PHerc1667 128³ region in 12.21 s on
MPS, including strict CT input loading and eight model windows. All sixteen
layers returned populated 128³ arrays. Fetching the other heads reused the same
inference result. This is a first-region measurement, not a model-only timing or
an accuracy evaluation.

`tests/test_tsm_runtime.py` checks shared scheduling, cache reuse, failed-input
handling and encodings. Run it with the inference environment:

```sh
build/tsm-venv/bin/python -m pytest -q tests/test_tsm_runtime.py
ctest --test-dir build/macos --output-on-failure
```

The `inference_stream` GPU test serves delayed HTTP 202 responses, then verifies
that both overlay colors appear without navigating or reloading CT. The strict
headless ROI test checks that unavailable CT leaves the caller's output untouched.

The integrated viewer was also exercised for 55 seconds on the real CT volume.
Two center regions completed, the second in 10.53 s; the rendered XY/XZ/YZ views
showed simultaneous blue inner-surface and red ink predictions. The captured
image is `build/tsm-real-gui-final.png`, with the run log in
`build/tsm-real-gui.log`. The Release suite passed 44 tests with one external
shard-fixture skip; the four affected ASan/UBSan checks passed. Four Python
runtime tests passed, including missing-aware SDF pooling and thin-surface
preservation. These measurements used an approximate center axis.

Automatic switching was verified with the real PHerc1667 → PHercParis3 browser
swap path. Inference changed from L1 at 2.258 µm to L0 at 2.400 µm, created a
separate generation and rendered surface and ink overlays on Paris3. See
`build/tsm-volume-swap.png` and `build/tsm-volume-swap.log`. The service retains
one model and one active CT engine; rapid switches coalesce while an old job
drains. The local `requested-ct.txt` file carries the viewer’s selection, and
`ct-manifest.txt` is published only after the new head manifests are ready.
