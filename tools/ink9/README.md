# 9µm ink detection pipeline

Trace a segment, render its 21-layer surface volume, run the scrollprize
hybrid_3d2d model on it, overlay the prediction:

```sh
# 1. trace (spiral prior on) and render the layered surface volume
tracecli cache/<scroll>-surf-lod --seed X Y Z --gens N --out seg/ --umbilicus u.json
rendseg cache/<scroll>-lod seg/ seg-layers.raw --level 0 --up <step> --layers 21

# 2. wrap as the zarr koine_machines expects, run inference (villa @
#    merge-ink-pipelines, ink-detection/: uv sync — needs CFLAGS=-std=gnu11
#    on aarch64 for numcodecs), checkpoints from HF scrollprize/ink_9um
uv run python tools/ink9/raw2zarr.py seg-layers.raw seg-9um.zarr
uv run python -m koine_machines.inference.infer seg-9um.zarr step-075000.pth pred.tif \
    --overlap 0.5 --blend-mode hann --batch-size 8 --num-workers 8 --no-compile

# 3. overlay (applies the (p-0.25)/0.5 display rescale; base image is
#    layer 10 of the stack, so registration is pixel-exact)
uv run python tools/ink9/inkoverlay.py seg-layers.raw pred.tif overlay.png
```

Model notes: 17 of the 21 layers are consumed (centered); per-patch
robust-MAD normalization; the confident no-ink output sits at 0.25, not 0;
seeds 42/43 are independent runs — ensemble by averaging the raw tifs.
z-window offset is the main sensitivity — try --layer-start/--layer-end.
