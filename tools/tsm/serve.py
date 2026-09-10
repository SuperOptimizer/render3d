#!/usr/bin/env python3
"""Demand-driven multi-head inference for render3d's existing overlay streaming.

HTTP 202 means queued/running, never air. A single worker predicts all heads;
head switches share an atomic, bounded on-disk cache. HTTP handlers never wait
for inference. Model input is read through render3d's strict native CPU reader.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import numpy as np
from runtime import Model,Volume,checkpoint_hash


def atomic_text(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text);os.replace(tmp,path)


class Engine:
    def __init__(self,model,volume,cache,max_bytes=1<<30):
        self.model,self.volume,self.cache=model,volume,Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.max_bytes=max_bytes
        self.cv=threading.Condition()
        self.pending=OrderedDict();self.active=None;self.closed=False
        self.failed={};self.completed=0;self.hits=0
        self.status='Ready'
        self.worker=threading.Thread(target=self.run,name='tsm-inference',daemon=True)
        self.worker.start()

    def path(self,key):return self.cache/('_'.join(map(str,key))+'.npz')

    def request(self,key,head):
        if len(key)!=3 or any(k<0 or k*128>=size for k,size in zip(key,self.volume.shape)):
            return np.zeros((128,)*3,np.uint8)
        with self.cv:
            path=self.path(key)
            if path.exists():
                try:
                    with np.load(path,allow_pickle=False) as stored:a=stored[head].copy()
                    if a.shape!=(128,)*3 or a.dtype!=np.uint8:raise ValueError('invalid cached shape')
                    os.utime(path,None);self.hits+=1
                    return a
                except (ValueError,KeyError,OSError):path.unlink(missing_ok=True)
            failure=self.failed.get(key)
            if failure and time.monotonic()-failure[0]<5:raise IOError(failure[1])
            if key!=self.active:
                self.pending[key]=time.monotonic();self.pending.move_to_end(key,last=False)
                while len(self.pending)>32:self.pending.popitem(last=True)
                self.cv.notify()
            return None

    def run(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.closed or self.pending)
                if self.closed:return
                key,last=self.pending.popitem(last=False)
                if time.monotonic()-last>3:continue
                self.active=key;self.status=f'Predicting region {key}'
            started=time.monotonic()
            try:
                values,windows=self.model.predict(self.volume,tuple(k*128 for k in key))
                tmp=self.path(key).with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**values)
                with self.cv:
                    os.replace(tmp,self.path(key));self.failed.pop(key,None)
                    self.completed+=1
                    self.status=f'Ready: {self.completed} regions; last {time.monotonic()-started:.2f}s / {windows} windows'
                    files=sorted(self.cache.glob('*.npz'),key=lambda p:p.stat().st_mtime)
                    size=sum(p.stat().st_size for p in files)
                    for p in files:
                        if size<=self.max_bytes:break
                        size-=p.stat().st_size;p.unlink()
            except Exception as exc:
                with self.cv:
                    self.failed[key]=(time.monotonic(),str(exc));self.status=f'Waiting for input: {exc}'
                    while len(self.failed)>64:self.failed.pop(next(iter(self.failed)))
                print(self.status,flush=True)
            finally:
                self.path(key).with_suffix('.tmp').unlink(missing_ok=True)
                with self.cv:self.active=None

    def close(self):
        with self.cv:self.closed=True;self.cv.notify()
        self.worker.join();self.volume.close()


def display_request(engine,head,level,key,pred_level):
    """Raw 128^3 overlay block in the CT level grid, reusing native model tiles.

    Native and finer levels are nearest resampled; one coarser level is box
    pooled. Very coarse overviews have no predicted overlay, rather than
    running the trained model on a different physical voxel pitch.
    """
    if level>pred_level+1:return np.zeros((128,)*3,np.uint8)
    if level<=pred_level:
        scale=1<<(pred_level-level)
        start=tuple(k*128//scale for k in key)
        tile=tuple(p//128 for p in start)
        a=engine.request(tile,head)
        if a is None:return None
        indices=[(np.arange(128)//scale+p%128).astype(int) for p in start]
        return np.ascontiguousarray(a[np.ix_(*indices)])
    out=np.zeros((128,)*3,np.uint8);waiting=False
    for z in range(2):
        for y in range(2):
            for x in range(2):
                a=engine.request(tuple(k*2+d for k,d in zip(key,(z,y,x))),head)
                if a is None:waiting=True;continue
                # Surface masks preserve a thin sheet under reduction.
                v=a.reshape(64,2,64,2,64,2)
                if head.startswith('surface'):
                    reduced=v.max(axis=(1,3,5))
                elif head.startswith('sdf'):
                    count=(v!=0).sum(axis=(1,3,5))
                    reduced=np.rint(v.sum(axis=(1,3,5))/np.maximum(count,1)).astype(np.uint8)
                else:
                    reduced=np.rint(v.mean(axis=(1,3,5))).astype(np.uint8)
                out[z*64:(z+1)*64,y*64:(y+1)*64,x*64:(x+1)*64]=reduced
    return None if waiting else out


def prepare_bundle(root,volume,model,checkpoint,axis,voxel_um,port):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=volume.root/'manifest.json'
    source=volume.root/'source.json'
    identity={'checkpoint_sha256':checkpoint_hash(checkpoint),'model_info':model.info,
              'ct_root':str(volume.root),'manifest':hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'source':hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else '',
              'pred_level':volume.level,'voxel_um':voxel_um,'axis':np.asarray(axis).tolist(),
              'patch':model.spec.patch,'halo':model.halo,'clip':model.clip,'format_version':2,
              'runtime_sha256':checkpoint_hash(Path(__file__).with_name('runtime.py'))}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    heads=[l['name'] for l in model.layers]
    heads.sort(key=lambda c:({'surface_in1':0,'ink':1,'surface_out1':2,'fiber_vt':3,'fiber_hz':4}.get(c,5),c))
    generation=root/fingerprint
    for head in heads:
        dest=generation/head;dest.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(manifest,dest/'manifest.json')
        meta={'format':'render3d.volcomp-source.v1','url':f'http://127.0.0.1:{port}/{fingerprint}/{head}',
              'quality':1,'levels':[{'level':l,'chunk':128,'raw':True} for l in range(len(volume.manifest['levels']))]}
        atomic_text(dest/'source.json',json.dumps(meta,indent=2))
    metadata={**identity,'fingerprint':fingerprint,'device':str(model.device),'layers':model.layers,
              'native_overlay_level':volume.level,'coarsest_overlay_level':volume.level+1}
    atomic_text(root/'inference.json',json.dumps(metadata,indent=2))
    atomic_text(root/'heads.txt',''.join(f'{fingerprint}/{head}\n' for head in heads))
    # Publish the CT binding last: the viewer treats it as the commit record.
    atomic_text(root/'ct-manifest.txt',str(manifest.resolve())+'\n')
    return fingerprint,generation/'predictions',heads


def volume_geometry(ct, voxel_um=None, requested_level=None, axis_path=None):
    ct=Path(ct).resolve()
    man=json.loads((ct/'manifest.json').read_text())
    src=(ct/'source.json').read_text() if (ct/'source.json').exists() else str(ct)
    match=re.search(r'(\d+(?:\.\d+)?)um',src)
    pitch=voxel_um or (float(match[1]) if match else None)
    if pitch is None or not math.isfinite(pitch) or pitch<=0:
        raise ValueError('CT voxel pitch is unknown; provide source metadata containing its um pitch')
    level=requested_level if requested_level is not None else max(0,min(len(man['levels'])-1,round(math.log2(2.4/pitch))))
    if not 0<=level<len(man['levels']):raise ValueError('invalid prediction level')
    scale=man['levels'][level]['scale'];pitch*=scale
    if not axis_path and (ct/'umbilicus.json').is_file():axis_path=str(ct/'umbilicus.json')
    if axis_path:
        from tsm.labels import load_axis
        axis=load_axis(axis_path)/scale;note=f'axis: {axis_path}'
    else:
        shape=man['levels'][level]['shape']
        axis=np.array([[0,shape[1]/2,shape[2]/2],[shape[0]-1,shape[1]/2,shape[2]/2]])
        note='approximate volume-center axis; use a measured umbilicus for this CT'
    return man,level,pitch,axis,note


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ct-root',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--bundle',required=True);ap.add_argument('--library',default='build/macos/librender3d_headless.dylib')
    ap.add_argument('--device',choices=['auto','mps','cuda','cpu'],default='auto')
    ap.add_argument('--voxel-um',type=float,help='CT level-0 pitch; otherwise inferred from source URL')
    ap.add_argument('--level',type=int);ap.add_argument('--axis',help='full-resolution umbilicus JSON; default volume center')
    ap.add_argument('--port',type=int,default=9761);ap.add_argument('--cache-mb',type=int,default=1024)
    ap.add_argument('--patch',type=int,default=128);ap.add_argument('--halo',type=int,default=32)
    a=ap.parse_args()
    ct=Path(a.ct_root).resolve()
    if a.cache_mb<64 or a.halo<0 or a.patch<32 or a.patch%16 or a.patch>128 or a.halo>64:
        ap.error('invalid cache/patch/halo bounds')
    try:man,level,pitch,axis,axis_note=volume_geometry(ct,a.voxel_um,a.level,a.axis)
    except ValueError as exc:ap.error(str(exc))
    print(f'Loading TSM at CT L{level}, {pitch:.4f} um; {axis_note}',flush=True)
    model=Model(a.checkpoint,pitch,axis,a.device,a.patch,a.halo)
    volume=Volume(ct,a.library,level)
    bundle=Path(a.bundle).resolve()
    engine=None
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            nonlocal engine
            if engine is None:
                self.send_error(503,'Connecting the current CT volume');return
            if self.path=='/status':
                with engine.cv:data=json.dumps({'status':engine.status,'queued':len(engine.pending),'completed':engine.completed,'hits':engine.hits}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            parts=self.path.strip('/').split('/')
            if len(parts)!=6 or parts[0]!=fingerprint or parts[1] not in heads:
                # 404 is permanent air to the renderer. An old generation or
                # invalid URL must remain retryable, never poison its cache.
                self.send_error(409,'Inference generation or head mismatch');return
            try:
                level,z,y,x=map(int,parts[2:])
                if not 0<=level<len(man['levels']) or min(z,y,x)<0:raise ValueError('invalid coordinates')
                data=display_request(engine,parts[1],level,(z,y,x),volume.level)
                status=202 if data is None else 200
                body=b'' if data is None else data.tobytes()
                self.send_response(status);self.send_header('Content-Length',str(len(body)))
                if status==202:self.send_header('Retry-After','1')
                self.end_headers();self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            except (ValueError,IOError) as exc:self.send_error(503,str(exc))
            with engine.cv:status=engine.status
            atomic_text(bundle/'status.txt',f'TSM {model.device}: {status}\nL{volume.level}, {pitch:.4f} um; zoom to L{volume.level+1} or finer\n{axis_note}\n')
    # Handlers only schedule work or read a bounded cached tile. HTTPServer is
    # deliberately single threaded; the GPU worker is independent of it.
    server=http.server.HTTPServer(('127.0.0.1',a.port),Handler)
    fingerprint,cache,heads=prepare_bundle(bundle,volume,model,a.checkpoint,axis,pitch,server.server_port)
    engine=Engine(model,volume,cache,a.cache_mb<<20)
    atomic_text(bundle/'status.txt',f'TSM {model.device}: Ready\nL{level}, {pitch:.4f} um; {axis_note}\n')
    print(f'Listening on {server.server_port}; render3d --bricks {ct}/manifest.json --inference {bundle}',flush=True)
    request_path=bundle/'requested-ct.txt'
    last_request=None
    server.timeout=.25
    try:
        while True:
            server.handle_request()
            try:requested=request_path.read_text().strip()
            except FileNotFoundError:continue
            if requested==last_request:continue
            last_request=requested
            if requested==str((ct/'manifest.json').resolve()) and engine is not None:continue
            atomic_text(bundle/'status.txt','TSM: connecting the selected volume...\n')
            # The old worker owns its CT handle and coordinate system until
            # its current window batch finishes; queued old work is discarded.
            if engine is not None:engine.close();engine=None
            # Rapid swaps coalesce while the old prediction drains.
            requested=request_path.read_text().strip();last_request=requested
            if not requested:
                atomic_text(bundle/'ct-manifest.txt','\n')
                atomic_text(bundle/'status.txt','TSM: select a CT volume to start inference\n')
                continue
            try:
                target=Path(requested).resolve()
                if target.name!='manifest.json':raise ValueError('CT must be a pyramid manifest')
                next_ct=target.parent
                next_man,next_level,next_pitch,next_axis,next_note=volume_geometry(next_ct)
                next_volume=Volume(next_ct,a.library,next_level)
                try:
                    model.rebind(next_pitch,next_axis)
                    next_fingerprint,next_cache,next_heads=prepare_bundle(bundle,next_volume,model,a.checkpoint,next_axis,next_pitch,server.server_port)
                except Exception:
                    next_volume.close();raise
                ct,man,level,pitch,axis,axis_note=next_ct,next_man,next_level,next_pitch,next_axis,next_note
                volume=next_volume;fingerprint=next_fingerprint;heads=next_heads
                engine=Engine(model,volume,next_cache,a.cache_mb<<20)
                atomic_text(bundle/'status.txt',f'TSM {model.device}: Ready on {ct.name}\nL{level}, {pitch:.4f} um; {axis_note}\n')
                print(f'Switched inference to {ct}; L{level}, {pitch:.4f} um',flush=True)
            except Exception as exc:
                atomic_text(bundle/'ct-manifest.txt','\n')
                atomic_text(bundle/'status.txt',f'TSM: cannot infer this volume: {exc}\n')
                print(f'Volume switch failed: {exc}',flush=True)
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if engine is not None:engine.close()


if __name__=='__main__':main()
