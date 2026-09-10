import http.server,subprocess,sys,tempfile,threading,time
from pathlib import Path
counts={};first={}
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  now=time.monotonic();first.setdefault(self.path,now)
  counts[self.path]=counts.get(self.path,0)+1
  ready=now-first[self.path]>.6
  body=bytes([200])*128**3 if ready else b''
  self.send_response(200 if ready else 202)
  self.send_header('Content-Length',str(len(body)));self.end_headers()
  try:self.wfile.write(body)
  except (BrokenPipeError,ConnectionResetError):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
try:
 with tempfile.TemporaryDirectory(prefix='r3d-infer-') as root:
  subprocess.run([sys.argv[1],f'http://127.0.0.1:{server.server_port}',root],check=True,timeout=20)
  for head in ('blue','red'):
   path=Path(root)/head/'bricks/L0/0_0_0.volc'
   assert path.exists() and path.stat().st_size>0
   assert counts[f'/{head}/0/0/0/0']>=2
finally:
 print('HTTP requests:',counts,flush=True)
 server.shutdown();server.server_close();t.join()
print('pending inference: both heads recovered from 202 without false-air caching')
