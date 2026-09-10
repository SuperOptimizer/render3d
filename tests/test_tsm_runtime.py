"""Model-independent scheduling/encoding tests; real checkpoint smoke is separate."""
import sys,time,threading
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/tsm'))
from serve import Engine,display_request
from runtime import layer_info

class Volume:
    shape=(256,256,256)
    def close(self):pass

class Model:
    def __init__(self):self.calls=0;self.started=threading.Event();self.release=threading.Event()
    def predict(self,volume,origin):
        self.calls+=1;self.started.set();assert self.release.wait(5)
        return {'ink':np.full((128,)*3,50,np.uint8),'fiber_vt':np.full((128,)*3,200,np.uint8)},1

def test_heads_share_one_job_and_disk_cache(tmp_path):
    model=Model();engine=Engine(model,Volume(),tmp_path)
    try:
        assert engine.request((0,0,0),'ink') is None
        assert model.started.wait(2)
        assert engine.request((0,0,0),'fiber_vt') is None
        assert not engine.pending
        model.release.set()
        end=time.monotonic()+5
        while engine.request((0,0,0),'ink') is None and time.monotonic()<end:time.sleep(.01)
        assert model.calls==1
        assert np.all(engine.request((0,0,0),'fiber_vt')==200)
        assert np.all(display_request(engine,'ink',0,(0,0,0),1)==50)
    finally:model.release.set();engine.close()
    model=Model();engine=Engine(model,Volume(),tmp_path)
    try:
        assert np.all(engine.request((0,0,0),'fiber_vt')==200)
        assert model.calls==0
    finally:model.release.set();engine.close()

def test_failed_input_never_publishes_prediction(tmp_path):
    class Missing(Model):
        def predict(self,*args):raise IOError('CT missing')
    engine=Engine(Missing(),Volume(),tmp_path)
    try:
        assert engine.request((0,0,0),'ink') is None
        end=time.monotonic()+2
        while not engine.failed and time.monotonic()<end:time.sleep(.01)
        assert engine.failed
        assert not list(tmp_path.glob('*.npz'))
    finally:engine.close()

def test_kind_encodings():
    assert layer_info('sdf_in')['zero']==128
    assert layer_info('sdf_out')['scale']==20/127
    assert layer_info('nx')['zero']==127.5
    assert layer_info('fiber_vt')['scale']==1/255
    assert layer_info('density')['scale']==.001

def test_coarse_distance_ignores_missing_and_surface_survives():
    class Cached:
        def request(self,key,head):
            a=np.zeros((128,)*3,np.uint8)
            a[::2,::2,::2]=128 if head=='sdf_in' else 255
            return a
    engine=Cached()
    assert np.all(display_request(engine,'sdf_in',2,(0,0,0),1)==128)
    assert np.all(display_request(engine,'surface_in1',2,(0,0,0),1)==255)

def test_volume_switch_recomputes_geometry(tmp_path):
    import json
    from serve import volume_geometry
    def make(name,pitch,shape):
        root=tmp_path/name;root.mkdir()
        (root/'manifest.json').write_text(json.dumps({'levels':[
            {'scale':1,'shape':shape},{'scale':2,'shape':[s//2 for s in shape]}]}))
        (root/'source.json').write_text(json.dumps({'url':f'https://example.test/{pitch}um/data'}))
        return root
    a=make('first',1.2,[512,256,128]);b=make('second',2.4,[128,512,256])
    _,level_a,pitch_a,axis_a,_=volume_geometry(a)
    _,level_b,pitch_b,axis_b,_=volume_geometry(b)
    assert level_a==1 and level_b==0
    assert pitch_a==pitch_b==2.4
    assert axis_a[0].tolist()==[0,64,32]
    assert axis_b[0].tolist()==[0,256,128]

def test_closing_old_volume_discards_queued_predictions(tmp_path):
    model=Model();engine=Engine(model,Volume(),tmp_path)
    assert engine.request((0,0,0),'ink') is None
    assert model.started.wait(2)
    assert engine.request((1,0,0),'ink') is None
    closer=threading.Thread(target=engine.close);closer.start()
    end=time.monotonic()+2
    while not engine.closed and time.monotonic()<end:time.sleep(.01)
    assert engine.closed
    model.release.set();closer.join(5)
    assert not closer.is_alive() and model.calls==1
    assert engine.path((0,0,0)).exists()
    assert not engine.path((1,0,0)).exists()
