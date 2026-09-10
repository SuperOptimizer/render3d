/* Queued small-texture updates must match a fresh complete snapshot. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

struct label_state { uint64_t revision, probes, fetches; };
static uint64_t label_revision(void *p) { return ((struct label_state*)p)->revision; }
static uint32_t label_gen(void *p,uint32_t l,uint32_t x,uint32_t y,uint32_t z) {
  (void)l;(void)x;(void)y;(void)z;
  struct label_state *s=p;s->probes++;return (uint32_t)s->revision;
}
static void label_fetch(void *p,uint32_t l,uint32_t x,uint32_t y,uint32_t z,uint8_t *out) {
  (void)l;(void)x;(void)y;(void)z;
  struct label_state *s=p;s->fetches++;memset(out,(int)(s->revision%9u),4096);
}
struct partial_state { uint64_t revision, probes; unsigned mode; };
static uint64_t partial_revision(void *p) { return ((struct partial_state *)p)->revision; }
static uint32_t partial_gen(void *p,uint32_t l,uint32_t x,uint32_t y,uint32_t z) {
  (void)l;(void)x;(void)y;(void)z;
  struct partial_state *s=p;s->probes++;return s->mode==2 ? 0 : (uint32_t)s->revision;
}
static bool partial_fetch(void *p,uint32_t l,uint32_t x,uint32_t y,uint32_t z,uint8_t *out) {
  (void)l;(void)x;(void)y;(void)z;
  memset(out,7,4096);return ((struct partial_state *)p)->mode==1;
}
static void capture_label(r3d_renderer *r,uint8_t *image) {
  r3d_frame_params p={.cam_origin={0.5f,0.5f,-1},.cam_right={0.2f,0,0},
    .cam_up={0,0.2f,0},.cam_forward={0,0,1},.vol_r0={1,0,0},.vol_r1={0,1,0},
    .vol_r2={0,0,1},.viewport={64,64},.view_flags=R3D_VIEW_ORTHO,
    .mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_z0=64,.slab_depth=2,
    .overlay_flags=8};
  r3d_bricks_params(r,&p);assert(r3d_frame(r,&p,NULL)==0);
  uint32_t w,h;assert(r3d_read_frame(r,image,&w,&h)==0&&w==64&&h==64);
}
static void capture(r3d_renderer *r, uint8_t image[64*64*4]) {
  r3d_frame_params p={.cam_origin={32,32,0},.cam_right={32,0,0},.cam_up={0,32,0},
    .cam_forward={0,0,1},.viewport={63,64},.view_org=1,.view_flags=R3D_VIEW_SURF,
    .mode=R3D_MODE_FLAT,.density=0.5f,.step_voxels=1,.slab_depth=8,
    .overlay_flags=33,.overlay_gain=0.7f};
  r3d_bricks_params(r,&p);
  r3d_surfvol_params(r,&p);
  for (int i=0;i<8;i++) assert(r3d_frame(r,&p,NULL)==0);
  uint32_t w=0,h=0;
  assert(r3d_read_frame(r,image,&w,&h)==0 && w==64 && h==64);
}
static void set_all(r3d_renderer *r, const uint8_t tf[256][4], const float *pred,
                    const uint8_t *mask, unsigned n) {
  assert(r3d_set_transfer(r,tf)==0);
  assert(r3d_surfvol_inkpred(r,pred,n,n,0,0,(float)n)==0);
  assert(r3d_surfmask(r,mask,n,n)==0);
}
int main(void) {
  assert(SDL_Init(SDL_INIT_VIDEO));
  r3d_config cfg={.headless=true,.headless_w=64,.headless_h=64,.spv_dir=R3D_SPV_DIR};
  r3d_renderer *r=NULL;
  if (r3d_create(NULL,&cfg,&r)) { SDL_Quit(); return 77; }
  char root[]="/tmp/r3d_updates_XXXXXX",path[256];
  assert(mkdtemp(root));
  uint32_t dim[3]={128,128,128};
  assert(st_make_tree(root,dim,1,0)==0);
  snprintf(path,sizeof path,"%s/manifest.json",root);
  assert(r3d_bricks_begin(r,path,8,0)==0);
  float coords[16]={32,32,64,1,96,32,64,1,32,96,64,1,96,96,64,1};
  float normals[16]={0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1};
  assert(r3d_surf_begin(r,2,2,coords,normals)==0);
  assert(r3d_surfvol_begin(r,64,64,16,8,1.0f/64,1.0f/64)==0);
  r3d_surfvol_window(r,0,0,1,0);
  uint8_t tf[256][4],altered[256][4],mask[32*32],scratch_mask[32*32];
  float pred[32*32],scratch_pred[32*32];
  for(unsigned i=0;i<256;i++) {
    tf[i][0]=(uint8_t)i;tf[i][1]=35;tf[i][2]=130;tf[i][3]=255;
    altered[i][0]=30;altered[i][1]=(uint8_t)i;altered[i][2]=60;altered[i][3]=255;
  }
  for(unsigned i=0;i<32*32;i++) {mask[i]=(uint8_t)((i/32+i)%3);pred[i]=(float)(i%32)/64;}
  uint8_t reference[64*64*4],got[64*64*4];
  struct label_state labels={.revision=1};
  r3d_label_src label_source={.gen=label_gen,.fetch=label_fetch,.user=&labels,.revision=label_revision};
  assert(r3d_bricks_labels(r,&label_source)==0);
  r3d_bricks_settle(r);
  uint64_t probes=labels.probes;
  assert(probes==512u && labels.fetches==512u);
  for(unsigned i=0;i<100;i++)r3d_bricks_labels_sync(r,0);
  assert(labels.probes==probes);
  labels.revision++;
  r3d_bricks_settle(r);
  assert(labels.probes==probes*2 && labels.fetches==1024u);
  /* Incomplete asynchronous reads may write nonzero bytes. They must remain
   * visually empty, retry without revision changes, then stop polling after
   * a transform revision moves them outside the source. */
  struct partial_state partial={.revision=1};
  r3d_label_src partial_source={.gen=partial_gen,.fetch=label_fetch,.user=&partial,
    .revision=partial_revision,.fetch_complete=partial_fetch};
  assert(r3d_bricks_labels(r,&partial_source)==0);
  r3d_bricks_settle(r);capture_label(r,reference);
  partial.mode=1;r3d_bricks_settle(r);capture_label(r,got);
  assert(memcmp(reference,got,sizeof got));
  partial.mode=0;partial.revision++;r3d_bricks_settle(r);capture_label(r,got);
  assert(!memcmp(reference,got,sizeof got));
  partial.mode=2;partial.revision++;r3d_bricks_settle(r);capture_label(r,got);
  assert(!memcmp(reference,got,sizeof got));
  probes=partial.probes;
  for(unsigned i=0;i<100;i++)r3d_bricks_labels_sync(r,0);
  assert(partial.probes==probes);
  set_all(r,tf,pred,mask,32);
  capture(r,reference);
  /* Change one resource at a time: its pixels must visibly change, then
   * restoring the exact snapshot must restore every byte (including cache). */
  assert(r3d_set_transfer(r,altered)==0);capture(r,got);assert(memcmp(reference,got,sizeof got));
  assert(r3d_set_transfer(r,tf)==0);capture(r,got);assert(!memcmp(reference,got,sizeof got));
  memcpy(scratch_mask,mask,sizeof mask);scratch_mask[16*32+16]=2;
  scratch_mask[4*32+7]=1;assert(r3d_surfmask(r,scratch_mask,32,32)==0);
  capture(r,got);assert(memcmp(reference,got,sizeof got));
  assert(r3d_surfmask(r,mask,32,32)==0);capture(r,got);assert(!memcmp(reference,got,sizeof got));
  for(unsigned i=0;i<32*32;i++)scratch_pred[i]=1.0f;
  assert(r3d_surfvol_inkpred(r,scratch_pred,32,32,0,0,32)==0);
  capture(r,got);assert(memcmp(reference,got,sizeof got));
  assert(r3d_surfvol_inkpred(r,pred,32,32,0,0,32)==0);
  capture(r,got);assert(!memcmp(reference,got,sizeof got));
  /* Prediction resize must not destroy the independent painted mask. */
  assert(r3d_surfvol_inkpred(r,scratch_pred,16,16,0,0,16)==0);
  assert(r3d_surfvol_inkpred(r,pred,32,32,0,0,32)==0);
  capture(r,got);assert(!memcmp(reference,got,sizeof got));
  /* Coalesce several updates before recording, resize queued targets, and
   * clear a target before its pending update is submitted. */
  for(unsigned k=0;k<3;k++) {
    set_all(r,altered,scratch_pred,scratch_mask,16);
    set_all(r,altered,scratch_pred,scratch_mask,32);
    r3d_surfmask_clear(r);
    set_all(r,tf,pred,mask,32);
    capture(r,got);assert(!memcmp(reference,got,sizeof got));
  }
  assert(r3d_surfvol_inkpred(r,scratch_pred,32,32,0,0,32)==0);
  r3d_surfvol_end(r); /* pending prediction must not refer to destroyed image */
  r3d_destroy(r);
  st_rm_tree(root,1);
  SDL_Quit();
  puts("gpu_updates: deferred TF/prediction/mask snapshots, dirty rects, resize and cancellation match");
  return 0;
}
