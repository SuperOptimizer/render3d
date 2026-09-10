/* A damaged native download must heal through renderer demand streaming. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static unsigned view_center(r3d_renderer *r,float x) {
  r3d_frame_params p={.cam_origin={x,8.0f/256,0},.cam_forward={0,0,1},
    .cam_right={2.0f/256,0,0},.cam_up={0,2.0f/256,0},.viewport={32,32},
    .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},.view_flags=R3D_VIEW_ORTHO,
    .mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_depth=16};
  r3d_bricks_params(r,&p);assert(!r3d_frame(r,&p,NULL));
  uint8_t pixels[32*32*4];uint32_t w,h;assert(!r3d_read_frame(r,pixels,&w,&h));
  return pixels[(16*32+16)*4];
}
int main(int argc,char **argv) {
  assert(argc==3 || argc==4);setenv("R3D_DEBLOCK",argv[2],1);assert(SDL_Init(SDL_INIT_VIDEO));
  bool drag=argc==4 && !strcmp(argv[3],"drag");
  if(drag)setenv("R3D_FETCHERS","1",1);
  char root[]="/tmp/r3d_nativegpu_XXXXXX",path[256],manifest[256];assert(mkdtemp(root));
  uint32_t dims[3]={256,256,256};st_const_value=40;assert(st_make_tree(root,dims,2,1)==0);
  snprintf(path,sizeof path,"%s/volcomp/L0/0_0_0.vcs",root);assert(unlink(path)==0);
  snprintf(path,sizeof path,"%s/source.json",root);FILE *f=fopen(path,"w");assert(f);
  fprintf(f,"{\"format\": \"render3d.volcomp-source.v1\", \"url\": \"%s\", \"quality\": 2, \"native_volcomp\": true, \"levels\": [{\"chunk\": 128}, {\"chunk\": 128}]}",argv[1]);fclose(f);
  snprintf(path,sizeof path,"%s/bricks",root);assert(mkdir(path,0755)==0);
  snprintf(path,sizeof path,"%s/bricks/L0",root);assert(mkdir(path,0755)==0);
  snprintf(path,sizeof path,"%s/bricks/L0/0_0_0.volc",root);f=fopen(path,"w");assert(f);fputs("VOLCbroken",f);fclose(f);
  r3d_config cfg={.headless=true,.headless_w=32,.headless_h=32,.spv_dir=R3D_SPV_DIR};
  r3d_renderer *r=NULL;assert(r3d_create(NULL,&cfg,&r)==0);
  snprintf(manifest,sizeof manifest,"%s/manifest.json",root);assert(r3d_bricks_begin(r,manifest,16,0)==0);
  r3d_bricks_stats st={0};float point[3]={8.0f/256,8.0f/256,8.0f/256};
  if(argc==4) {
    assert(r3d_bricks_stream_begin(r));r3d_bricks_stream_point(r,point,0,0);
    r3d_bricks_stream_submit(r,32);
    /* Parent releases stdin only after its server starts the stalled body. */
    assert(getchar()=='\n');
    if(drag) {
      /* Queue another old-view chunk behind the confirmed stalled transfer.
       * The next collect must discard it before the worker can start it. */
      float obsolete[3]={8.0f/256,136.0f/256,8.0f/256};
      assert(r3d_bricks_stream_begin(r));
      r3d_bricks_stream_point(r,point,0,0);r3d_bricks_stream_point(r,obsolete,0,0);
      r3d_bricks_stream_submit(r,1);
      /* One fetcher is occupied by the old view's stalled body. New detail
       * must arrive before the server releases that body, then returning to
       * the old view must work without transport-failure backoff. */
      for(unsigned phase=0;phase<2;phase++) {
        point[0]=(phase?8.0f:136.0f)/256;
        uint64_t start=SDL_GetTicks(); unsigned pixel=0;
        do {
          if(r3d_bricks_stream_begin(r)) {
            r3d_bricks_stream_point(r,point,0,0);r3d_bricks_stream_submit(r,1);
          }
          r3d_bricks_settle(r);pixel=view_center(r,point[0]);
          if(pixel>=95)break;
          SDL_Delay(5);
        } while(SDL_GetTicks()-start<2500);
        uint64_t elapsed=SDL_GetTicks()-start;
        assert(pixel>=95 && elapsed<2000);
        printf("drag sharp: phase=%u ms=%llu pixel=%u\n",phase,(unsigned long long)elapsed,pixel);fflush(stdout);
        if(!phase)assert(getchar()=='\n');
      }
      r3d_bricks_get_stats(r,&st);assert(st.failures==0);
      r3d_destroy(r);unlink(path);
      snprintf(path,sizeof path,"%s/bricks/L0/0_0_1.volc",root);unlink(path);
      goto cleanup;
    }
    uint64_t start=SDL_GetTicks();r3d_destroy(r);
    uint64_t elapsed=SDL_GetTicks()-start;
    printf("stalled download shutdown: %llu ms\n",(unsigned long long)elapsed);
    assert(elapsed<2000 && access(path,F_OK)!=0);
    goto cleanup;
  }
  for(unsigned i=0;i<1000;i++) {
    if(r3d_bricks_stream_begin(r)) {r3d_bricks_stream_point(r,point,0,0);r3d_bricks_stream_submit(r,32);}
    r3d_bricks_settle(r);r3d_bricks_get_stats(r,&st);
    if(st.hot>512)break;
    SDL_Delay(5);
  }
  printf("native repair: deblock=%s hot=%u failures=%llu\n",argv[2],st.hot,(unsigned long long)st.failures);
  assert(st.hot==513 && st.failures==0);
  r3d_frame_params p={.cam_origin={8.0f/256,8.0f/256,0},.cam_forward={0,0,1},
    .cam_right={2.0f/256,0,0},.cam_up={0,2.0f/256,0},.viewport={32,32},
    .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},.view_flags=R3D_VIEW_ORTHO,
    .mode=R3D_MODE_MIP,.density=1,.step_voxels=1,.slab_depth=16};
  r3d_bricks_params(r,&p);assert(r3d_frame(r,&p,NULL)==0);
  uint8_t pixels[32*32*4];uint32_t w,h;assert(r3d_read_frame(r,pixels,&w,&h)==0);
  assert(pixels[(16*32+16)*4]>=95);
  r3d_destroy(r);unlink(path);
cleanup:
  snprintf(path,sizeof path,"%s/source.json",root);unlink(path);
  snprintf(path,sizeof path,"%s/bricks/L0",root);rmdir(path);
  snprintf(path,sizeof path,"%s/bricks",root);rmdir(path);
  st_rm_tree(root,2);SDL_Quit();return 0;
}
