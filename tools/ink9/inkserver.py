#!/usr/bin/env python3
"""Persistent ink-detection inference server for render3d's live overlay.

Loads one or more koine_machines hybrid_3d2d checkpoints once and serves
predictions over TCP so the viewer can request them per visible region
without paying model-load latency every time.  Run from villa's
ink-detection env:

    uv run python tools/ink9/inkserver.py step-075000-seed42.pth \
        [step-075000-seed43.pth ...] [--port 9743]

Protocol (little-endian):
  request : 'INK1' u32 layers, h, w          then layers*h*w u8
            'INK2' u32 layers, h, w, flags   then layers*h*w u8
            (Z,H,W layer-major, layer stack centered on the surface —
            rendseg --layers order)
  response: 'INKR' u32 h, w                  then h*w f32 probabilities

flags is a TTA/ensemble bitmask (INK1 == flags 0 == fast single pass):
  bit0  flips        identity + h/v/hv flips, inverted before averaging
  bit1  depth shift  the model's layer window slid +-1 within the stack
  bit2  intensity    x0.9 / x1.1 gain variants
  bit3  ensemble     average over every loaded checkpoint, not just the
                     first (pass several .pth files to enable)
A zero-filled region returns zeros without touching the GPU. One request
is served at a time (the models are not reentrant); clients queue.
"""
import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from koine_machines.inference.infer import (
    center_crop_layer_indices,
    compute_importance_map_2d,
    configure_model,
    inference_depth_from_config,
    logits_to_probabilities,
    prepare_model_for_inference,
    resolve_patch_stride,
)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def predict(model, device, amp_dtype, preprocessing, stack, patch, stride,
            weight_map, batch_size):
    """stack: (Z,H,W) u8 with Z == model input depth -> (H,W) f32 probs."""
    z, h, w = stack.shape
    oh, ow = h, w
    if h < patch or w < patch:  # pad small regions up to one full patch
        pad = np.zeros((z, max(h, patch), max(w, patch)), dtype=stack.dtype)
        pad[:, :h, :w] = stack
        stack, (h, w) = pad, pad.shape[1:]
    prob = np.zeros((h, w), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if ys[-1] + patch < h:
        ys.append(h - patch)
    if xs[-1] + patch < w:
        xs.append(w - patch)
    coords = [(y, x) for y in ys for x in xs]
    autocast = (
        torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.inference_mode(), autocast:
        for i in range(0, len(coords), batch_size):
            batch, metas = [], []
            for (y, x) in coords[i:i + batch_size]:
                tile = stack[:, y:y + patch, x:x + patch]
                if not tile.any():
                    continue  # keep raw-uint8 occupancy semantics
                if preprocessing == "tifxyz_robust":
                    from vesuvius.image_proc.intensity.normalization import (
                        normalize_robust,
                    )

                    ft = normalize_robust(tile)
                else:
                    ft = np.ascontiguousarray(tile, dtype=np.float32) / 255.0
                batch.append(torch.from_numpy(ft).unsqueeze(0))
                metas.append((y, x))
            if not batch:
                continue
            images = torch.stack(batch).to(device, non_blocking=True)
            logits = model(images)
            probs_t, _ = logits_to_probabilities(
                logits, image_hw=(patch, patch), logged_resize=True
            )
            probs = probs_t.float().cpu().numpy()[:, 0]
            for bi, (y, x) in enumerate(metas):
                prob[y:y + patch, x:x + patch] += probs[bi] * weight_map
                wsum[y:y + patch, x:x + patch] += weight_map
    out = np.zeros((h, w), dtype=np.float32)
    np.divide(prob, wsum, out=out, where=wsum > 0)
    return np.ascontiguousarray(out[:oh, :ow])


def tta_variants(stack, flags, layers, depth):
    """Deterministic augmentation set: list of (variant stack, prob inverse).

    The client always ships `layers` (21) layers while the model consumes
    `depth`, so the depth shift is a real slide of the crop window inside
    the sampled slab, not zero padding.
    """
    base_idx = center_crop_layer_indices(np.arange(layers), output_depth=depth)

    def window(off):
        idx = base_idx + off
        if idx[0] < 0 or idx[-1] >= layers:
            idx = np.clip(idx, 0, layers - 1)  # edge-replicate at the rim
        return np.ascontiguousarray(stack[idx])

    s0 = window(0)
    out = [(s0, lambda p: p)]
    if flags & 1:  # in-plane flips
        out.append((np.ascontiguousarray(s0[:, :, ::-1]),
                    lambda p: np.ascontiguousarray(p[:, ::-1])))
        out.append((np.ascontiguousarray(s0[:, ::-1, :]),
                    lambda p: np.ascontiguousarray(p[::-1, :])))
        out.append((np.ascontiguousarray(s0[:, ::-1, ::-1]),
                    lambda p: np.ascontiguousarray(p[::-1, ::-1])))
    if flags & 2:  # depth window slid one layer each way
        out.append((window(-1), lambda p: p))
        out.append((window(+1), lambda p: p))
    if flags & 4:  # intensity gain
        for g in (0.9, 1.1):
            sv = np.clip(s0.astype(np.float32) * g, 0, 255).astype(np.uint8)
            out.append((sv, lambda p: p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--port", type=int, default=9743)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    models = []
    patch = depth = None
    for ck in args.checkpoints:
        model_args = argparse.Namespace(
            checkpoint=ck, model_type="auto", metadata_json=None,
            amp_dtype="auto", gpu_ids=[], compile_model=False,
            compile_mode="default",
        )
        cm = configure_model(model_args)
        cm, device, _ = prepare_model_for_inference(
            args=model_args, configured_model=cm
        )
        cm.model.eval()
        if patch is None:
            patch, depth = cm.roi_size, cm.in_chans
        elif (cm.roi_size, cm.in_chans) != (patch, depth):
            print(f"inkserver: {ck.name} geometry mismatch, skipping",
                  flush=True)
            continue
        models.append((cm, device))
    if not models:
        print("inkserver: no usable checkpoints", flush=True)
        return 1
    stride = resolve_patch_stride(patch_size=patch, overlap=args.overlap,
                                  explicit_stride=None)
    weight_map = compute_importance_map_2d(
        patch_size=(patch, patch), mode="hann", sigma_scale=0.125
    ).numpy()
    names = ", ".join(ck.name for ck in args.checkpoints[: len(models)])
    print(f"inkserver: {names} ready on 127.0.0.1:{args.port} "
          f"(patch {patch} stride {stride} depth {depth}, "
          f"{len(models)} model{'s' if len(models) > 1 else ''})",
          flush=True)

    gpu_lock = threading.Lock()  # one inference at a time; IO stays parallel

    def infer(stack, layers, flags):
        use_models = models if (flags & 8) else models[:1]
        variants = tta_variants(stack, flags, layers, depth)
        acc = None
        n = 0
        for cm, device in use_models:
            for sv, inv in variants:
                p = predict(cm.model, device, cm.amp_dtype, cm.preprocessing,
                            sv, patch, stride, weight_map, args.batch_size)
                p = inv(p)
                acc = p.astype(np.float64) if acc is None else acc + p
                n += 1
        return (acc / n).astype(np.float32)

    def serve(conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                hdr = recv_exact(conn, 16)
                magic, layers, h, w = struct.unpack("<4sIII", hdr)
                flags = 0
                if magic == b"INK2":
                    (flags,) = struct.unpack("<I", recv_exact(conn, 4))
                elif magic != b"INK1":
                    raise ConnectionError(f"bad magic {magic!r}")
                raw = recv_exact(conn, layers * h * w)
                stack = np.frombuffer(raw, dtype=np.uint8).reshape(layers, h, w)
                t0 = time.time()
                if stack.any():
                    with gpu_lock:
                        pred = infer(stack, layers, flags)
                else:
                    pred = np.zeros((h, w), dtype=np.float32)
                tag = f" tta=0x{flags:x}" if flags else ""
                print(f"inkserver: {w}x{h} in {time.time() - t0:.2f}s{tag}",
                      flush=True)
                conn.sendall(struct.pack("<4sII", b"INKR", h, w))
                conn.sendall(np.ascontiguousarray(pred).tobytes())
        except (ConnectionError, OSError) as e:
            print(f"inkserver: client gone ({e})", flush=True)
        finally:
            conn.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(8)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
