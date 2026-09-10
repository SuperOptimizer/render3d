/* A fallback cached in one desired block must not hide its finer neighbor. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static unsigned center(r3d_renderer *r) {
  uint8_t px[64*64*4];uint32_t w,h;
  assert(r3d_read_frame(r,px,&w,&h)==0 && w==64 && h==64);
  return px[(32u*64u+32u)*4u];
}
static void test_detail_priority(const r3d_config *cfg) {
  r3d_renderer *r=NULL;assert(!r3d_create(NULL,cfg,&r));
  char root[]="/tmp/r3d_detail_XXXXXX",path[256];assert(mkdtemp(root));
  uint32_t dims[3]={256,256,256};st_const_value=40;assert(!st_make_tree(root,dims,3,0));
  uint32_t chunks[3]={2,2,2};st_const_value=200;assert(!st_write_level(root,0,chunks,true));
  snprintf(path,sizeof path,"%s/manifest.json",root);assert(!r3d_bricks_begin(r,path,8,0));
  float point[3]={8.0f/256,8.0f/256,8.0f/256};
  assert(r3d_bricks_stream_begin(r));
  /* A single-block budget must deliver visible fine detail before spending
   * that budget on the intermediate L1 parent. L0 is 200; L1/L2 are 40. */
  r3d_bricks_stream_point(r,point,0,0);
  r3d_bricks_stream_submit(r,1);r3d_bricks_flush(r);
  r3d_bricks_stats st;r3d_bricks_get_stats(r,&st);assert(st.hot==65 && !st.failures);
  r3d_frame_params p={.cam_origin={8.0f/256,8.0f/256,0},.cam_forward={0,0,1},
    .cam_right={2.0f/256,0,0},.cam_up={0,2.0f/256,0},.viewport={64,64},
    .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},.view_flags=R3D_VIEW_ORTHO,
    .mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_depth=16};
  r3d_bricks_params(r,&p);assert(!r3d_frame(r,&p,NULL));assert(center(r)>=195);
  /* Requesting L0 again should leave only the intermediate parent to decode,
   * because the sole budgeted upload above was already the desired block. */
  assert(r3d_bricks_stream_begin(r));r3d_bricks_stream_point(r,point,0,0);
  r3d_bricks_stream_submit(r,1);r3d_bricks_flush(r);
  r3d_bricks_get_stats(r,&st);assert(st.hot==66 && !st.failures);
  r3d_destroy(r);st_rm_tree(root,3);
}
int main(int argc, char **argv) {
  setenv("R3D_DEBLOCK","0",1);
  assert(SDL_Init(SDL_INIT_VIDEO));
  r3d_config cfg={.headless=true,.headless_w=64,.headless_h=64,.spv_dir=R3D_SPV_DIR};
  if(argc>1)cfg.spv_dir=argv[1];
  bool coarse_air=getenv("R3D_TEST_COARSE_AIR")!=NULL;
  r3d_renderer *r=NULL;
  if(r3d_create(NULL,&cfg,&r)!=0){SDL_Quit();return 77;}
  char root[]="/tmp/r3d_lod_cache_XXXXXX",path[256];assert(mkdtemp(root));
  uint32_t dims[3]={128,128,128};st_const_value=coarse_air?0:40;
  assert(st_make_tree(root,dims,2,0)==0);
  uint8_t *raw=malloc(128u*128u*128u),*enc=NULL;size_t enc_n=0;assert(raw);
  memset(raw,200,128u*128u*128u);volcomp_brick_params bp=volcomp_brick_defaults(1);
  assert(volcomp_brick_encode(&bp,raw,128,&enc,&enc_n)==0);free(raw);
  snprintf(path,sizeof path,"%s/volcomp/L0/0_0_0.vcs",root);
  volcomp_shard_writer *writer=volcomp_shard_create(path,1024,128,0,1);assert(writer);
  assert(volcomp_shard_put(writer,0,enc,enc_n)==0 && volcomp_shard_close(writer)==0);free(enc);
  snprintf(path,sizeof path,"%s/manifest.json",root);assert(r3d_bricks_begin(r,path,8,0)==0);
  assert(r3d_bricks_stream_begin(r));
  float point[3]={8.0f/128,8.0f/128,24.0f/128};
  r3d_bricks_stream_point(r,point,0,0);r3d_bricks_stream_submit(r,32);r3d_bricks_settle(r);
  r3d_bricks_stats st;r3d_bricks_get_stats(r,&st);assert(st.hot==(coarse_air?1u:65u) && st.failures==0);
  r3d_frame_params p={.cam_origin={8.0f/128,8.0f/128,0},.cam_forward={0,0,1},
    .cam_right={2.0f/128,0,0},.cam_up={0,2.0f/128,0},.viewport={64,64},
    .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},.view_flags=R3D_VIEW_ORTHO,.mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_depth=32};
  r3d_bricks_params(r,&p);assert(r3d_frame(r,&p,NULL)==0);unsigned ray=center(r);
  p.skip_gate=80.0f/255; p.threshold=80.0f/255;
  assert(r3d_frame(r,&p,NULL)==0);unsigned skipped=center(r);
  float coords[16]={8,8,0,1, 12,8,0,1, 8,12,0,1, 12,12,0,1};
  float normals[16]={0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1};
  assert(r3d_surf_begin(r,2,2,coords,normals)==0);
  assert(r3d_surfvol_begin(r,16,16,32,0,1.0f/16,1.0f/16)==0);
  r3d_surfvol_window(r,0,0,1,0);
  r3d_surfvol_visible(r,0,0,0,16,16,32);
  p=(r3d_frame_params){.cam_origin={8,8,0},.cam_right={8,0,0},.cam_up={0,8,0},
    .viewport={64,64},.view_flags=R3D_VIEW_SURF,.mode=R3D_MODE_MIP,.density=1,.step_voxels=1,
    .slab_z0=16,.slab_depth=32};
  r3d_bricks_params(r,&p);r3d_surfvol_params(r,&p);
  assert(r3d_frame(r,&p,NULL)==0);unsigned surf=center(r);
  printf("fine block after coarse fallback: ray=%u skipped=%u surface=%u (expected 200)\n",ray,skipped,surf);
  r3d_destroy(r);st_rm_tree(root,2);test_detail_priority(&cfg);SDL_Quit();
  assert(ray>=195 && skipped>=195 && surf>=195);return 0;
}
