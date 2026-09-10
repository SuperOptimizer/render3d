/* Fine known-air pages override a bright coarse fallback, even after bounded
 * negative-cache eviction. No CPU decode or physical atlas slot is needed. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static unsigned pixel(r3d_renderer *r) {
  r3d_frame_params p={.cam_origin={2,2,0},.cam_right={2,0,0},.cam_up={0,2,0},
    .viewport={32,32},.view_flags=R3D_VIEW_SURF,.mode=R3D_MODE_MIP,.density=1,.slab_depth=2};
  r3d_bricks_params(r,&p);r3d_surfvol_params(r,&p);
  for(unsigned i=0;i<4;i++)assert(r3d_frame(r,&p,NULL)==0);
  uint8_t rgba[32*32*4];uint32_t w,h;assert(r3d_read_frame(r,rgba,&w,&h)==0);
  assert(w==32&&h==32);return rgba[(16*32+16)*4];
}
static void request(r3d_renderer *r,unsigned x,unsigned y,unsigned z) {
  float p[3]={(float)(x*16u+8u)/256,(float)(y*16u+8u)/256,(float)(z*16u+8u)/256};
  assert(r3d_bricks_stream_begin(r));r3d_bricks_stream_point(r,p,0,0);
  r3d_bricks_stream_submit(r,256);r3d_bricks_flush(r);
}
int main(void){
  assert(SDL_Init(SDL_INIT_VIDEO));
  r3d_config cfg={.headless=true,.headless_w=32,.headless_h=32,.spv_dir=R3D_SPV_DIR};
  r3d_renderer *r=NULL;if(r3d_create(NULL,&cfg,&r)){SDL_Quit();return 77;}
  char root[]="/tmp/r3d_air_XXXXXX",path[256];assert(mkdtemp(root));
  uint32_t dim[3]={256,256,256};st_const_value=150;assert(st_make_tree(root,dim,2,0)==0);
  snprintf(path,sizeof path,"%s/volcomp/L0/0_0_0.vcs",root);
  volcomp_shard_writer *w=volcomp_shard_create(path,1024,128,0,1);assert(w);
  for(unsigned z=0;z<2;z++)for(unsigned y=0;y<2;y++)for(unsigned x=0;x<2;x++)
    assert(volcomp_shard_put_zero(w,z*64+y*8+x)==0);
  assert(volcomp_shard_close(w)==0);
  snprintf(path,sizeof path,"%s/manifest.json",root);assert(r3d_bricks_begin(r,path,8,0)==0);
  float coords[16]={18,18,20,1,22,18,20,1,18,22,20,1,22,22,20,1};
  float normals[16]={0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1};
  assert(r3d_surf_begin(r,2,2,coords,normals)==0);
  assert(r3d_surfvol_begin(r,4,4,4,2,0.25f,0.25f)==0);r3d_surfvol_window(r,0,0,1,0);
  assert(pixel(r)>=140); /* no fine mapping yet: visible coarse fallback */
  request(r,1,1,1);assert(pixel(r)<=2); /* real fine air, not unknown */
  for(unsigned z=0;z<16;z++)for(unsigned y=0;y<16;y++)for(unsigned x=0;x<16;x++)request(r,x,y,z);
  r3d_bricks_stats st;r3d_bricks_get_stats(r,&st);
  assert(st.hot==512&&st.decoded==0&&st.page_entries<=st.page_capacity/2);
  request(r,1,1,1); /* CPU negative entry must be forgotten with GPU marker */
  assert(pixel(r)<=2);
  r3d_destroy(r);st_rm_tree(root,2);SDL_Quit();
  puts("sparse_zero: fine air overrides coarse data through bounded negative-page churn");return 0;
}
