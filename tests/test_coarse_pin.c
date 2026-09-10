/* Fine uploads must preserve coarse fallback without touching its LRU stamps. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  assert(SDL_Init(SDL_INIT_VIDEO));
  r3d_config cfg={.headless=true,.headless_w=64,.headless_h=64,.spv_dir=R3D_SPV_DIR};
  r3d_renderer *r=NULL;
  if(r3d_create(NULL,&cfg,&r)!=0){SDL_Quit();return 77;}
  char root[]="/tmp/r3d_coarse_pin_XXXXXX",path[256];assert(mkdtemp(root));
  uint32_t dims[3]={256,256,256};st_const_value=150;
  assert(st_make_tree(root,dims,2,0)==0);
  snprintf(path,sizeof path,"%s/manifest.json",root);
  assert(r3d_bricks_begin(r,path,8,0)==0);
  for(unsigned i=0;i<8;i++) {
    assert(r3d_bricks_stream_begin(r));
    float lo[3]={0.5f,0,(float)i/8},hi[3]={1,1,(float)(i+1)/8};
    r3d_bricks_stream_box(r,lo,hi,1.0f/256,0);
    r3d_bricks_stream_submit(r,256);
    r3d_bricks_settle(r);
  }
  r3d_bricks_stats stats;r3d_bricks_get_stats(r,&stats);
  assert(stats.failures==0 && stats.decoded>stats.hot_cap && stats.hot<=stats.hot_cap);
  /* This far-away patch never requested L0 and needs the first coarse slot. */
  float coords[16]={8,8,8,1, 24,8,8,1, 8,24,8,1, 24,24,8,1};
  float normals[16]={0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1};
  assert(r3d_surf_begin(r,2,2,coords,normals)==0);
  assert(r3d_surfvol_begin(r,16,16,4,2,1.0f/16,1.0f/16)==0);
  r3d_surfvol_window(r,0,0,1,0);
  r3d_frame_params fp={.cam_origin={8,8,0},.cam_right={8,0,0},.cam_up={0,8,0},
    .viewport={64,64},.view_flags=R3D_VIEW_SURF,.mode=R3D_MODE_MIP,.density=1,.slab_depth=2};
  r3d_bricks_params(r,&fp);r3d_surfvol_params(r,&fp);
  assert(r3d_frame(r,&fp,NULL)==0);
  uint8_t pixels[64*64*4];uint32_t w,h;
  assert(r3d_read_frame(r,pixels,&w,&h)==0);
  assert(w==64 && h==64 && pixels[(32u*64u+32u)*4u]>=145);
  r3d_destroy(r);st_rm_tree(root,2);SDL_Quit();
  puts("coarse fallback survives fine streaming beyond atlas capacity");return 0;
}
