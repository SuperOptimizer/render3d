/* Perspective shading must match uncached lookup as the footprint grows. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#include <dirent.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static void render(const char *spv,const char *manifest,uint8_t out[2][64*64*4]) {
  r3d_config cfg={.headless=true,.headless_w=64,.headless_h=64,.spv_dir=spv};
  r3d_renderer *r=NULL;assert(r3d_create(NULL,&cfg,&r)==0);
  assert(r3d_bricks_begin(r,manifest,8,0)==0);
  r3d_frame_params p={.cam_origin={.5f,.5f,-.1f},.cam_forward={0,0,1},
    .cam_right={.7f,0,0},.cam_up={0,.7f,0},.viewport={64,64},
    .vol_r0={1,0,0},.vol_r1={0,1,0},.vol_r2={0,0,1},
    .mode=R3D_MODE_FULL,.density=.08f,.step_voxels=.5f,.lod_bias=.3f};
  r3d_bricks_params(r,&p);
  for(unsigned q=0;q<2;q++) {
    r3d_set_quality(r,q);
    assert(r3d_frame(r,&p,NULL)==0);uint32_t w,h;
    assert(r3d_read_frame(r,out[q],&w,&h)==0 && w==64 && h==64);
  }
  r3d_destroy(r);
}
int main(int argc,char **argv) {
  setenv("R3D_DEBLOCK","0",1);unsetenv("R3D_NO_LOD_PIPELINE");
  assert(SDL_Init(SDL_INIT_VIDEO));
  char root[]="/tmp/r3d_gradient_XXXXXX",ref[256],path[512],target[1024],manifest[256];
  assert(mkdtemp(root));uint32_t dims[3]={128,128,128};
  assert(st_make_tree(root,dims,2,0)==0);
  /* The coarsest fallback contains a smooth nonlinear field, so changing
   * gradient radius changes shading even inside a single resident cell. */
  uint8_t *raw=malloc(128u*128u*128u),*enc=NULL;size_t n;assert(raw);
  for(unsigned z=0;z<128;z++)for(unsigned y=0;y<128;y++)for(unsigned x=0;x<128;x++)
    raw[(z*128+y)*128+x]=(uint8_t)(100+50*sin(x*.35)*cos(y*.27)*sin(z*.31));
  volcomp_brick_params bp=volcomp_brick_defaults(1);
  assert(volcomp_brick_encode(&bp,raw,128,&enc,&n)==0);free(raw);
  snprintf(path,sizeof path,"%s/volcomp/L1/0_0_0.vcs",root);
  volcomp_shard_writer *wr=volcomp_shard_create(path,1024,128,1,1);assert(wr);
  assert(volcomp_shard_put(wr,0,enc,n)==0 && volcomp_shard_close(wr)==0);free(enc);
  snprintf(ref,sizeof ref,"%s/reference",root);assert(mkdir(ref,0755)==0);
  DIR *d=opendir(R3D_SPV_DIR);assert(d);struct dirent *e;
  while((e=readdir(d))) {
    if(!strstr(e->d_name,".spv"))continue;
    const char *name=e->d_name;
    if(!strcmp(name,"raycast_lod.spv"))name="raycast_lod_uncached.spv";
    if(!strcmp(name,"raycast_fast_lod.spv"))name="raycast_fast_lod_uncached.spv";
    snprintf(target,sizeof target,"%s/%s",R3D_SPV_DIR,name);
    snprintf(path,sizeof path,"%s/%s",ref,e->d_name);assert(symlink(target,path)==0);
  }
  closedir(d);snprintf(manifest,sizeof manifest,"%s/manifest.json",root);
  uint8_t a[2][64*64*4],b[2][64*64*4];
  render(argc>1?argv[1]:R3D_SPV_DIR,manifest,a);render(ref,manifest,b);
  for(unsigned q=0;q<2;q++) {
    unsigned max=0;size_t sum=0,signal=0;
    for(size_t i=0;i<sizeof a[q];i++)if(i%4!=3) {
      unsigned diff=(unsigned)abs((int)a[q][i]-(int)b[q][i]);
      if(diff>max)max=diff;sum+=diff;signal+=b[q][i];
    }
    printf("gradient quality %u: max=%u total=%zu signal=%zu\n",q,max,sum,signal);
    assert(signal>1000 && max<=1 && sum<100);
  }
  d=opendir(ref);assert(d);while((e=readdir(d)))if(strstr(e->d_name,".spv")) {
    snprintf(path,sizeof path,"%s/%s",ref,e->d_name);unlink(path);
  }
  closedir(d);rmdir(ref);st_rm_tree(root,2);SDL_Quit();return 0;
}
