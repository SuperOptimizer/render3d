/* Selective cache invalidation: distinct CT slabs and a flattened surface. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static uint32_t frame(r3d_renderer *r,r3d_frame_params views[3]) {
  r3d_frame_stats st;assert(r3d_frame_views(r,views,3,&st)==0);return st.panes_drawn;
}
static void snapshot(r3d_renderer *r,uint8_t *rgba) {
  uint32_t w,h;assert(r3d_read_frame(r,rgba,&w,&h)==0&&w==64&&h==64);
}
int main(void) {
  assert(SDL_Init(SDL_INIT_VIDEO));
  r3d_config cfg={.headless=true,.headless_w=64,.headless_h=64,.spv_dir=R3D_SPV_DIR};
  r3d_renderer *r=NULL;if(r3d_create(NULL,&cfg,&r)){SDL_Quit();return 77;}
  char root[]="/tmp/r3d_views_XXXXXX",path[256];assert(mkdtemp(root));
  uint32_t dim[3]={256,256,256};assert(st_make_tree(root,dim,2,0)==0);
  snprintf(path,sizeof path,"%s/manifest.json",root);assert(r3d_bricks_begin(r,path,10,0)==0);
  float coords[16]={64,64,36,1,80,64,36,1,64,80,36,1,80,80,36,1};
  float normals[16]={0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1};
  assert(r3d_surf_begin(r,2,2,coords,normals)==0);
  assert(r3d_surfvol_begin(r,16,16,4,2,1.0f/16,1.0f/16)==0);r3d_surfvol_window(r,0,0,1,0);
  uint8_t mask[16*16]={0};assert(r3d_surfmask(r,mask,16,16)==0);
  r3d_frame_params views[3]={0};
  views[0]=(r3d_frame_params){.cam_origin={8,8,0},.cam_right={8,0,0},.cam_up={0,8,0},
    .viewport={32,32},.view_flags=R3D_VIEW_SURF,.mode=R3D_MODE_MIP,.density=1,
    .slab_depth=2,.overlay_flags=32};
  r3d_bricks_params(r,&views[0]);r3d_surfvol_params(r,&views[0]);
  for(unsigned i=1;i<3;i++) {
    views[i]=(r3d_frame_params){.cam_origin={72.0f/256,72.0f/256,-1},
      .cam_right={8.0f/256,0,0},.cam_up={0,8.0f/256,0},.cam_forward={0,0,1},
      .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},
      .viewport={32,32},.view_org=32u|((i-1u)*32u<<16),.view_flags=R3D_VIEW_ORTHO,
      .mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_z0=i==1?36.0f:196.0f,.slab_depth=2};
    r3d_bricks_params(r,&views[i]);
  }
  assert(frame(r,views)==3);assert(frame(r,views)==0);
  mask[8*16+8]=2;assert(r3d_surfmask(r,mask,16,16)==0);
  assert(frame(r,views)==1);assert(frame(r,views)==0);
  float point[3]={72.0f/256,72.0f/256,36.0f/256};
  assert(r3d_bricks_stream_begin(r));r3d_bricks_stream_point(r,point,0,0);
  r3d_bricks_stream_submit(r,256);r3d_bricks_flush(r);
  assert(frame(r,views)==2); /* near slab + surface, not distant slab */
  assert(frame(r,views)==0);
  uint8_t cached[64*64*4],redrawn[64*64*4];snapshot(r,cached);
  r3d_set_quality(r,R3D_QUALITY_FULL);assert(frame(r,views)==3);
  snapshot(r,redrawn);assert(!memcmp(cached,redrawn,sizeof cached));
  point[2]=196.0f/256;
  assert(r3d_bricks_stream_begin(r));r3d_bricks_stream_point(r,point,0,0);
  r3d_bricks_stream_submit(r,256);r3d_bricks_flush(r);
  assert(frame(r,views)==1); /* far slab only; segment bounds reject its bake */
  snapshot(r,cached);r3d_set_quality(r,R3D_QUALITY_FULL);assert(frame(r,views)==3);
  snapshot(r,redrawn);assert(!memcmp(cached,redrawn,sizeof cached));
  uint8_t tf[256][4];memset(tf,180,sizeof tf);assert(r3d_set_transfer(r,tf)==0);
  assert(frame(r,views)==3);
  r3d_destroy(r);st_rm_tree(root,2);SDL_Quit();
  puts("viewcache: only dependent panes redraw; cached pixels equal forced full redraws");return 0;
}
