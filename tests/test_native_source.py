"""Offline native bootstrap/range transport and CPU demand-decode conformance."""
import http.server
import json
import socket
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import queue

probe, packer = (str(Path(p).resolve()) for p in sys.argv[1:3])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from bootstrap_volcomp import bootstrap


def crc32c(data):
    c = 0xffffffff
    for byte in data:
        c ^= byte
        for _ in range(8):
            c = (c >> 1) ^ (0x82f63b78 if c & 1 else 0)
    return c ^ 0xffffffff


with tempfile.TemporaryDirectory(prefix='r3d-native-') as tmp:
    root = Path(tmp)
    raw = root/'raw'; raw.write_bytes(bytes([100]) * 128**3)
    vcs = root/'sample.vcs'
    subprocess.run([packer, 'raw', str(raw), '128', '128', '128', str(vcs)], check=True)
    data = vcs.read_bytes()
    off, size = struct.unpack_from('<QI', data, len(data)-48)
    payload = data[off:off+size]
    index = bytearray(b'\xff'*8192)
    struct.pack_into('<QQ', index, 0, 0, len(payload))
    shard = payload + index + struct.pack('<I', crc32c(index))
    mode = 'ok'; ranges = []
    payload_started=threading.Event();release_payload=threading.Event()
    meta = {'zarr_format': 3, 'node_type': 'array', 'data_type': 'uint8',
            'fill_value': 0, 'shape': [128]*3,
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [1024]*3}},
            'chunk_key_encoding': {'name': 'default', 'configuration': {'separator': '/'}},
            'codecs': [{'name': 'sharding_indexed', 'configuration': {
                'chunk_shape': [128]*3, 'index_location': 'end',
                'index_codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}},
                                 {'name': 'crc32c'}],
                'codecs': [{'name': 'volcomp', 'configuration': {'q': 2}}]}}]}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == '/0/zarr.json':
                body = json.dumps(meta).encode(); code = 200; content_range = None
            elif self.path != '/0/c/0/0/0' or mode == 'missing':
                self.send_error(404); return
            elif mode == 'error':
                self.send_error(500); return
            else:
                blob = shard
                if mode == 'drag':
                    ix=bytearray(index)
                    struct.pack_into('<QQ',ix,16,len(payload),len(payload))
                    struct.pack_into('<QQ',ix,8*16,2*len(payload),len(payload))
                    blob=payload*3+ix+struct.pack('<I',crc32c(ix))
                if mode == 'badpayload': blob = shard[:4] + b'\xff' + shard[5:]
                if mode == 'badcrc': blob = shard[:-1] + bytes([shard[-1] ^ 1])
                if mode in ('badbounds', 'air'):
                    ix = bytearray(index)
                    struct.pack_into('<QQ', ix, 0, *( (2**64-1, 2**64-1) if mode == 'air' else (len(shard), 16)))
                    blob = payload + ix + struct.pack('<I', crc32c(ix))
                rg = self.headers.get('Range'); ranges.append(rg)
                content_range = None; code = 200; body = blob
                if rg and mode != 'ignore_range':
                    a, b = rg.removeprefix('bytes=').split('-')
                    start = int(a) if a else len(blob)-int(b)
                    end = int(b) if a else len(blob)-1
                    body = blob[start:end+1];code = 206
                    content_range = f'bytes {start}-{end}/{len(blob)}'
                    if mode == 'wrongrange': content_range = f'bytes 0-{len(body)-1}/{len(blob)}'
            self.send_response(code)
            if content_range: self.send_header('Content-Range', content_range)
            self.send_header('Content-Length', str(len(body)));self.end_headers()
            if mode == 'interrupted' and self.path == '/0/c/0/0/0' and rg != 'bytes=-8196':
                self.wfile.write(body[:len(body)//2]);self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR);self.close_connection=True
            elif (mode == 'stalled' or (mode == 'drag' and rg == f'bytes=0-{len(payload)-1}')) and self.path == '/0/c/0/0/0' and rg != 'bytes=-8196':
                payload_started.set();release_payload.wait(10)
                try: self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError): pass
            else: self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        for mode, expected in [('ok', 1), ('missing', 2), ('air', 2), ('error', 0),
                               ('interrupted', 0), ('badpayload', 0), ('badcrc', 0), ('badbounds', 0), ('ignore_range', 0), ('wrongrange', 0)]:
            dest=root/mode;dest.mkdir()
            result=subprocess.run([probe, url, str(dest)])
            assert result.returncode == expected, (mode, result.returncode)
            path=dest/'bricks/L0/0_0_0.volc'
            assert path.exists() == bool(expected)
            if expected: assert path.read_bytes() == (payload if expected == 1 else b'')
        # Retry an interrupted transfer into the same directory. No partial
        # final file or temporary download may poison subsequent requests.
        mode='ok'; dest=root/'interrupted'
        assert not list(dest.rglob('*.tmp.*'))
        assert subprocess.run([probe,url,str(dest)]).returncode==1
        assert (dest/'bricks/L0/0_0_0.volc').read_bytes()==payload
        dest=root/'damaged';dest.mkdir()
        cached=dest/'bricks/L0/0_0_0.volc';cached.parent.mkdir(parents=True)
        cached.write_bytes(b'VOLCbroken')
        assert subprocess.run([probe,url,str(dest)]).returncode==1
        assert cached.read_bytes()==payload
        mode='ok'; dest=root/'boot'  
        bootstrap(url,dest,Path(packer))
        manifest=dest/'manifest.json'; before=manifest.read_bytes()
        bootstrap(url,dest,Path(packer)); assert manifest.read_bytes()==before
        assert json.loads((dest/'source.json').read_text())['native_volcomp'] is True
        # Remove bootstrap residency to require the ordinary CPU miss path to fetch.
        (dest/'volcomp/L0/0_0_0.vcs').unlink()
        subprocess.run([probe, 'cpu', str(dest)],check=True)
        assert (dest/'bricks/L0/0_0_0.volc').read_bytes()==payload
        cached=dest/'bricks/L0/0_0_0.volc'
        cached.write_bytes(b'VOLCbroken')
        assert subprocess.run([probe,'cpu',str(dest)]).returncode==4
        assert not cached.exists()
        subprocess.run([probe,'cpu',str(dest)],check=True)
        assert cached.read_bytes()==payload
        if len(sys.argv)>3:
            for deblock in ('0','1'):
                subprocess.run([sys.argv[3],url,deblock],check=True)
            mode='stalled'
            p=subprocess.Popen([sys.argv[3],url,'0','cancel'],stdin=subprocess.PIPE)
            try:
                assert payload_started.wait(10), 'renderer never requested payload'
                p.communicate(input=b'\n',timeout=5)
                assert p.returncode==0
            finally:
                release_payload.set()
                if p.poll() is None: p.kill();p.wait()
            mode='drag';payload_started.clear();release_payload.clear();drag_range_start=len(ranges)
            p=subprocess.Popen([sys.argv[3],url,'0','drag'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            lines=queue.Queue()
            def read_lines():
                for line in p.stdout: lines.put(line)
                lines.put(None)
            reader=threading.Thread(target=read_lines,daemon=True);reader.start()
            try:
                assert payload_started.wait(10), 'old view never requested payload'
                p.stdin.write('\n');p.stdin.flush()
                while True:
                    line=lines.get(timeout=5)
                    assert line is not None, 'drag probe exited before sharp detail'
                    print(line,end='')
                    if line.startswith('drag sharp: phase=0'):break
                # The new view is sharp while the previous body remains held.
                release_payload.set();p.stdin.write('\n');p.stdin.flush()
                p.wait(timeout=5);assert p.returncode==0
                reader.join(timeout=2)
                assert f'bytes={2*len(payload)}-{3*len(payload)-1}' not in ranges[drag_range_start:], 'obsolete queued chunk was fetched'
                while not lines.empty():
                    line=lines.get()
                    if line:print(line,end='')
            finally:
                release_payload.set()
                if p.poll() is None:p.kill();p.wait()
                p.stdin.close();p.stdout.close()
            mode='ok'
        assert '-8196'   in ranges or 'bytes=-8196' in ranges
    finally:
        server.shutdown();server.server_close();thread.join()
print('native source: bootstrap, cached reopen, range identity, air/error distinction, CRC/bounds and CPU demand decode passed')
