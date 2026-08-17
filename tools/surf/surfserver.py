#!/usr/bin/env python3
"""Persistent surface-prediction server: nnU-Net v2 `scrollprize/surface_m7_nnunet`
(the model behind the open-data bucket's `surface-m7-L0` prediction trees) served
over TCP so render3d can predict surfaces on demand for volumes that have no
published predictions.  Run from villa's ink-detection env (nnunetv2 installed):

    uv run python tools/surf/surfserver.py /path/to/surf-m7 [--port 9744]

Protocol (little-endian, one request at a time per connection):
  request : 'SRF1' u32 nz, ny, nx   then nz*ny*nx u8 CT voxels (z-major)
  response: 'SRFR' u32 nz, ny, nx   then nz*ny*nx u8 surface probability * 255
The model's 3d_fullres patch is 192^3 at spacing 1 (native voxels of the ~8-9um
scans); blocks smaller than a patch are padded. Send a margin around the region
you keep (32 voxels is plenty) — nnU-Net's gaussian window blends the interior.
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


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--port", type=int, default=9744)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--step", type=float, default=0.75, help="sliding-window step (fraction)")
    ap.add_argument("--mirror", action="store_true", help="mirroring TTA (8x slower)")
    args = ap.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetPredictor(
        tile_step_size=args.step,
        use_gaussian=True,
        use_mirroring=args.mirror,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    pred.initialize_from_trained_model_folder(
        str(args.model_dir), use_folds=(int(args.fold),), checkpoint_name=args.checkpoint
    )
    patch = pred.configuration_manager.patch_size
    print(f"surfserver: {args.model_dir.name} ready on 127.0.0.1:{args.port} "
          f"(patch {patch}, step {args.step}, mirror {args.mirror}, device {device})",
          flush=True)

    gpu_lock = threading.Lock()

    def predict(block):
        # nnU-Net wants (c, z, y, x) float; its preprocessing applies the plan's
        # CTNormalization (clip 0..212, (x-87.5)/47.7) from raw u8 values
        z, y, x = block.shape
        pz, py, px = [max(0, p - s) for p, s in zip(patch, block.shape)]
        img = block
        if pz or py or px:
            img = np.pad(block, ((0, pz), (0, py), (0, px)), mode="reflect")
        img = img[None].astype(np.float32)
        probs = pred.predict_single_npy_array(
            img, {"spacing": [1.0, 1.0, 1.0]}, None, None, True
        )
        # returns (segmentation, probabilities) when return_probabilities=True
        if isinstance(probs, tuple):
            probs = probs[1]
        surf = probs[1]  # class 1 = surface
        surf = surf[:z, :y, :x]
        return np.clip(surf * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def serve(conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                hdr = recv_exact(conn, 16)
                magic, nz, ny, nx = struct.unpack("<4sIII", hdr)
                if magic != b"SRF1":
                    raise ConnectionError(f"bad magic {magic!r}")
                raw = recv_exact(conn, nz * ny * nx)
                block = np.frombuffer(raw, dtype=np.uint8).reshape(nz, ny, nx)
                t0 = time.time()
                if block.any():
                    with gpu_lock:
                        out = predict(np.ascontiguousarray(block))
                else:
                    out = np.zeros((nz, ny, nx), dtype=np.uint8)
                print(f"surfserver: {nx}x{ny}x{nz} in {time.time() - t0:.2f}s", flush=True)
                conn.sendall(struct.pack("<4sIII", b"SRFR", nz, ny, nx))
                conn.sendall(np.ascontiguousarray(out).tobytes())
        except (ConnectionError, OSError) as e:
            print(f"surfserver: client gone ({e})", flush=True)
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
