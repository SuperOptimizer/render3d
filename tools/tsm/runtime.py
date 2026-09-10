"""TSM model and render3d CT adapter. No model architecture is duplicated here."""
from __future__ import annotations
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np


class Volume:
    def __init__(self, root, library, level):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.level = level
        self.shape = tuple(self.manifest['levels'][level]['shape'])
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.handle = C.c_void_p()
        self.lib.r3d_headless_volume_open_v1.argtypes = [C.c_char_p,C.c_uint32,C.c_void_p,C.POINTER(C.c_void_p)]
        self.lib.r3d_headless_volume_close_v1.argtypes = [C.c_void_p]
        self.lib.r3d_headless_volume_read_roi_strict_v1.argtypes = [C.c_void_p,C.c_uint32,*([C.c_int64]*3),*([C.c_uint32]*3),C.c_void_p,C.c_void_p]
        rc = self.lib.r3d_headless_volume_open_v1(str(self.root).encode(),16384,None,C.byref(self.handle))
        if rc: raise RuntimeError(f'CT open failed: {rc}')

    def read(self, origin, shape):
        a = np.empty(shape,np.uint8)
        z,y,x = map(int,origin); nz,ny,nx = map(int,shape)
        rc = self.lib.r3d_headless_volume_read_roi_strict_v1(self.handle,self.level,x,y,z,nx,ny,nz,None,a.ctypes.data)
        if rc: raise IOError(f'CT region unavailable ({rc}): {origin} {shape}')
        return a

    def close(self):
        if self.handle:
            self.lib.r3d_headless_volume_close_v1(self.handle)
            self.handle = C.c_void_p()


def checkpoint_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        while b := f.read(1<<20): h.update(b)
    return h.hexdigest()


def layer_info(name, clip=20):
    if name.startswith('sdf'):
        return {'name':name,'kind':'signed_distance','units':'model voxels','zero':128,'scale':clip/127,'missing':0}
    if name in ('sin','cos','nx','ny','nz') or name.startswith('fiber_d'):
        return {'name':name,'kind':'signed','zero':127.5,'scale':1/127.5}
    if name == 'density': return {'name':name,'kind':'density','units':'wraps/model voxel','zero':0,'scale':0.001}
    if name == 'thickness': return {'name':name,'kind':'distance','units':'model voxels','zero':0,'scale':1}
    return {'name':name,'kind':'probability','zero':0,'scale':1/255}


class Model:
    def __init__(self, checkpoint, voxel_um, axis, device='auto', patch=128, halo=32):
        import torch
        from tsm.infer import load_student, StudentNet, pred_channels, n_head_ch
        from tsm.sliding import WindowSpec
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        self.clip = 20.0
        model,self.info = load_student(str(checkpoint),self.device)
        self.channels = pred_channels(self.info['surface_mode'],self.info['fiber'],self.info['fiber_mode'])
        self.nhead = n_head_ch(self.info['surface_mode'],self.info['fiber'],self.info['fiber_mode'])
        self.net = StudentNet(model,voxel_um,self.clip,surface_mode=self.info['surface_mode'],
            input_radial=self.info['input_radial'],axis=axis,input_axis=self.info['input_axis'],fiber_mode=self.info['fiber_mode']).to(self.device)
        self.spec = WindowSpec(patch=patch,step=patch//2,halo=halo,out_tile=128,batch=1,
            dtype=torch.float32,prefetch=False,channels_last=False)
        self.halo = halo
        self.layers = [layer_info(c,self.clip) for c in self.channels if c != 'spare']

    def rebind(self, voxel_um, axis):
        """Change CT geometry after the previous engine has drained."""
        from tsm.infer import StudentNet
        self.net = StudentNet(self.net.net,voxel_um,self.clip,
            surface_mode=self.info['surface_mode'],input_radial=self.info['input_radial'],
            axis=axis,input_axis=self.info['input_axis'],fiber_mode=self.info['fiber_mode']).to(self.device)

    def predict(self, volume, origin):
        import torch
        from tsm.sliding import predict_box
        from tsm.student import normalize_ct
        from tsm.infer import extract_surface
        from tsm.data import decode_sdf
        h = self.halo
        start = tuple(int(v)-h for v in origin)
        raw = volume.read(start,(128+2*h,)*3)
        with torch.inference_mode():
            result,windows = predict_box(raw,self.net,normalize_ct,self.spec,self.nhead,'none',None,
                self.device,core_offset=(h,)*3,core_shape=(128,)*3,box_origin_zyx=start)
        values = {c:result[i] for i,c in enumerate(self.channels[:self.nhead])}
        valid = values['valid']
        for surface,sdf in [('surface1','sdf'),('surface_in1','sdf_in'),('surface_out1','sdf_out')]:
            if surface in self.channels:
                values[surface] = extract_surface(values[sdf],valid,self.clip).astype(np.uint8)*255
        if 'thickness' in self.channels:
            inside,outside = values['sdf_in'],values['sdf_out']
            d = decode_sdf(inside,self.clip)-decode_sdf(outside,self.clip)
            values['thickness'] = np.where((inside!=0)&(outside!=0),np.clip(np.rint(d),0,255),0).astype(np.uint8)
        # The named checkpoint has a class fibre head; refuse unsupported
        # future derived heads rather than silently substituting zero arrays.
        missing = {l['name'] for l in self.layers}-values.keys()
        if missing: raise ValueError(f'unsupported derived channels: {missing}')
        # Padding is no prediction, even when the network produces an output.
        for name,a in values.items():
            for axis,(o,size) in enumerate(zip(origin,volume.shape)):
                edge = max(0,min(128,size-o))
                if edge<128:
                    sl=[slice(None)]*3;sl[axis]=slice(edge,None);a[tuple(sl)]=0
        return {l['name']:np.ascontiguousarray(values[l['name']]) for l in self.layers},windows
