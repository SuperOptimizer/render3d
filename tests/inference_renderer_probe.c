/* HTTP 202 inference must retry promptly and independently fill both heads. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static void source(const char *root,const char *url) {
  char path[512];snprintf(path,sizeof path,"%s/source.json",root);
  FILE *f=fopen(path,"w");assert(f);
  fprintf(f,"{\"format\": \"render3d.volcomp-source.v1\", \"url\": \"%s\", \"quality\": 1, \"levels\": [{\"level\": 0, \"chunk\": 128, \"raw\": true}, {\"level\": 1, \"chunk\": 128, \"raw\": true}]}",url);
  fclose(f);
}
static void pixel(r3d_renderer *r,uint32_t flags,uint8_t out[3]) {
  r3d_frame_params p={.cam_origin={8.0f/256,8.0f/256,0},.cam_forward={0,0,1},
    .cam_right={2.0f/256,0,0},.cam_up={0,2.0f/256,0},.viewport={32,32},
    .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},.view_flags=R3D_VIEW_ORTHO,
    .mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_depth=16,
    .overlay_flags=flags,.overlay_gain=1,.ink3d_gain=1};
  r3d_bricks_params(r,&p);assert(!r3d_frame(r,&p,NULL));
  uint8_t px[32*32*4];uint32_t w,h;assert(!r3d_read_frame(r,px,&w,&h));
  memcpy(out,px+(16*32+16)*4,3);
}
int main(int argc,char **argv) {
  assert(argc==3);setenv("R3D_DEBLOCK","0",1);setenv("R3D_FETCHERS","2",1);
  assert(SDL_Init(SDL_INIT_VIDEO));
  char ct[512],blue[512],red[512],path[1024],url[1024];
  snprintf(ct,sizeof ct,"%s/ct",argv[2]);snprintf(blue,sizeof blue,"%s/blue",argv[2]);snprintf(red,sizeof red,"%s/red",argv[2]);
  uint32_t dims[3]={256,256,256};st_const_value=100;
  assert(!st_make_tree(ct,dims,2,0));assert(!st_make_tree(blue,dims,2,2));assert(!st_make_tree(red,dims,2,2));
  for(unsigned i=0;i<2;i++)for(unsigned l=0;l<2;l++) {
    snprintf(path,sizeof path,"%s/volcomp/L%u/0_0_0.vcs",i?red:blue,l);assert(!unlink(path));
  }
  /* Pretty-printed manifest shapes are valid too (as produced by imports). */
  snprintf(path,sizeof path,"%s/manifest.json",blue);
  FILE *mf=fopen(path,"w");assert(mf);
  fputs("{\"format\": \"render3d.volcomp-lod.v1\", \"shape\": [\n256, 256, 256\n], \"levels\": ["
        "{\"level\": 0, \"scale\": 1, \"shape\": [256,256,256], \"shards\": [1,1,1]},"
        "{\"level\": 1, \"scale\": 2, \"shape\": [128,128,128], \"shards\": [1,1,1]}]}",mf);
  fclose(mf);
  snprintf(url,sizeof url,"%s/ct",argv[1]);source(ct,url);
  snprintf(url,sizeof url,"%s/blue",argv[1]);source(blue,url);
  snprintf(url,sizeof url,"%s/red",argv[1]);source(red,url);
  r3d_config cfg={.headless=true,.headless_w=32,.headless_h=32,.spv_dir=R3D_SPV_DIR};
  r3d_renderer *r=NULL;assert(!r3d_create(NULL,&cfg,&r));
  snprintf(path,sizeof path,"%s/manifest.json",ct);assert(!r3d_bricks_begin(r,path,16,0));
  assert(!r3d_bricks_overlay(r,blue));assert(!r3d_bricks_ink3d(r,red));
  float point[3]={8.0f/256,8.0f/256,8.0f/256};uint8_t b[3]={0},a[3]={0};
  uint64_t start=SDL_GetTicks();
  while(SDL_GetTicks()-start<6000) {
    if(r3d_bricks_stream_begin(r)) { r3d_bricks_stream_point(r,point,0,0);r3d_bricks_stream_submit(r,8); }
    r3d_bricks_flush(r);pixel(r,1u|256u,b);pixel(r,4u,a);
    if(b[2]>b[0]+30 && a[0]>a[2]+30)break;
    SDL_Delay(10);
  }
  printf("inference ready: blue=%u,%u,%u red=%u,%u,%u in %llu ms\n",b[0],b[1],b[2],a[0],a[1],a[2],(unsigned long long)(SDL_GetTicks()-start));
  assert(b[2]>b[0]+30 && a[0]>a[2]+30);
  assert(!r3d_bricks_overlay_switch(r,red));
  assert(!r3d_bricks_ink3d_switch(r,blue));
  start=SDL_GetTicks();
  do {
    if(r3d_bricks_stream_begin(r)) { r3d_bricks_stream_point(r,point,0,0);r3d_bricks_stream_submit(r,8); }
    r3d_bricks_flush(r);pixel(r,1u|256u,b);pixel(r,4u,a);
    if(b[2]>b[0]+30 && a[0]>a[2]+30)break;
    SDL_Delay(10);
  } while(SDL_GetTicks()-start<3000);
  assert(b[2]>b[0]+30 && a[0]>a[2]+30);
  r3d_destroy(r);SDL_Quit();return 0;
}
