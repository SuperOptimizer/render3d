#!/usr/bin/env python3
"""Bootstrap a native compressed Zarr pyramid; finer chunks stream on demand."""
import argparse
import json
import subprocess
import struct
import urllib.error
import urllib.request
from pathlib import Path
from fetch_volcomp import metadata


def atomic_json(path, value):
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def bootstrap(base, root, packer):
    base = base.rstrip('/')
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / 'source.json'
    if (root / 'manifest.json').exists() and source_path.exists():
        old = json.loads(source_path.read_text())
        if old.get('url') == base and old.get('native_volcomp') is True:
            print('Opening cached compressed volume', flush=True)
            return
        raise ValueError('cache belongs to a different source')
    levels = []
    for level in range(8):
        try:
            meta, q = metadata(base, level)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and levels:
                break
            raise
        if levels and meta['shape'] != [(d+1)//2 for d in levels[-1][0]['shape']]:
            raise ValueError('pyramid levels must halve each axis')
        levels.append((meta, q))
    if not levels:
        raise ValueError('no supported compressed levels')
    entries = [{'level': l, 'scale': 1 << l, 'shape': m['shape'],
                'shards': [(d+1023)//1024 for d in m['shape']],
                'volcomp': f'volcomp/L{l}/{{z}}_{{y}}_{{x}}.vcs'}
               for l, (m, _) in enumerate(levels)]
    level = len(levels)-1
    nz, ny, nx = entries[-1]['shards']
    target = root / 'volcomp' / f'L{level}'
    target.mkdir(parents=True, exist_ok=True)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                output = target / f'{z}_{y}_{x}.vcs'
                if output.exists():
                    continue
                tmp = target / f'{z}_{y}_{x}.download'
                url = f'{base}/{level}/c/{z}/{y}/{x}'
                print(f'Loading coarse shard {z}/{y}/{x}', flush=True)
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as f:
                        while data := response.read(1024*1024):
                            f.write(data)
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    # An absent Zarr shard consists entirely of fill values.
                    index = b'\xff' * 8192
                    # Same CRC32C as the native index; compute only for this tiny fill index.
                    crc = 0xffffffff
                    for b in index:
                        crc ^= b
                        for _ in range(8):
                            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
                    tmp.write_bytes(index + struct.pack('<I', crc ^ 0xffffffff))
                subprocess.run([str(packer), 'zarr-shard', str(tmp), str(output),
                                str(level), str(levels[level][1])], check=True)
                tmp.unlink()
    source = {'format': 'render3d.volcomp-source.v1', 'url': base,
              'native_volcomp': True, 'quality': levels[0][1],
              'levels': [{'level': l, 'chunk': 128, 'raw': False} for l in range(len(levels))]}
    atomic_json(source_path, source)
    atomic_json(root / 'manifest.json', {'format': 'render3d.volcomp-lod.v1',
                'shape': entries[0]['shape'], 'shard_shape': [1024]*3,
                'brick_shape': [128]*3, 'levels': entries})
    print('Compressed volume ready; finer blocks stream as needed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('output', type=Path)
    parser.add_argument('--packer', type=Path, required=True)
    args = parser.parse_args()
    bootstrap(args.url, args.output, args.packer.resolve())
