/* Include the implementation to inspect private cache counters without
 * adding testing-only entry points to the stable shared-library ABI. */
#include "../src/headless/headless.c"
#include "synthtree.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
typedef struct progress_state { uint64_t done; unsigned calls; bool cancel; } progress_state;
static void roi_progress(void *user,const char *phase,uint64_t done,uint64_t total) {
  progress_state *s=user;
  assert(strcmp(phase,"volume-read-roi")==0 && done<=total && done>s->done);
  s->done=done; s->calls++;
}
static int roi_cancel(void *user) {
  progress_state *s=user;
  return s->cancel && s->calls>0;
}
int main(void) {
  char root[]="/tmp/r3d_roi_XXXXXX"; assert(mkdtemp(root));
  uint32_t dim[3]={256,128,128};
  assert(st_make_tree(root,dim,1,0)==0);
  r3d_headless_volume *v=NULL;
  assert(r3d_headless_volume_open_v1(root,8,NULL,&v)==0);
  size_t n=256u*128u*32u;
  uint8_t *got=malloc(n),*expected=malloc(n); assert(got && expected);
  progress_state progress={0};
  r3d_headless_callbacks cb={&progress,roi_cancel,roi_progress};
  assert(r3d_headless_volume_read_roi_v1(v,0,0,0,0,256,128,32,&cb,got)==0);
  assert(progress.calls==2 && progress.done==32);
  r3d_cpuvol_cache_stats stats;
  r3d_cpuvol_get_cache_stats(&v->core,&stats);
  assert(stats.decoded_blocks==256); /* once/block despite eight-slot cache */
  r3d_cpuvol reference; assert(r3d_cpuvol_open(&reference,root,512)==0);
  r3d_cpuvol_read_block(&reference,0,0,0,0,256,128,32,expected);
  assert(memcmp(got,expected,n)==0);
  /* Negative/unaligned origin and both source/decode block boundaries. */
  progress=(progress_state){0};
  assert(r3d_headless_volume_read_roi_v1(v,0,-1,7,-1,256,128,32,&cb,got)==0);
  assert(progress.calls==3);
  r3d_cpuvol_read_block(&reference,0,-1,7,-1,256,128,32,expected);
  assert(memcmp(got,expected,n)==0);
  progress=(progress_state){.cancel=true}; memset(got,0xa7,n);
  assert(r3d_headless_volume_read_roi_v1(v,0,0,0,0,256,128,32,&cb,got)==R3D_HEADLESS_E_CANCELLED);
  for(size_t i=0;i<n;i++) assert(got[i]==0xa7);
  assert(r3d_headless_volume_read_roi_v1(v,0,INT64_MAX,0,0,256,128,32,NULL,got)==R3D_HEADLESS_E_INVALID_ARGUMENT);
  r3d_cpuvol_close(&reference); r3d_headless_volume_close_v1(v);
  /* Strict model inputs distinguish an unavailable source chunk from air,
   * preserving the caller output on failure. The legacy API still pads. */
  char source[512];snprintf(source,sizeof source,"%s/volcomp/L0/0_0_0.vcs",root);
  assert(!unlink(source));assert(!r3d_headless_volume_open_v1(root,8,NULL,&v));
  memset(got,0xa7,n);
  assert(r3d_headless_volume_read_roi_strict_v1(v,0,0,0,0,256,128,32,NULL,got)==R3D_HEADLESS_E_IO);
  for(size_t i=0;i<n;i++)assert(got[i]==0xa7);
  assert(r3d_headless_volume_read_roi_strict_v1(v,0,256,0,0,256,128,32,NULL,got)==0);
  for(size_t i=0;i<n;i++)assert(got[i]==0);
  r3d_headless_volume_close_v1(v);
  free(got);free(expected);st_rm_tree(root,1);
  puts("ROI: once per block, exact unaligned bytes and transactional cancellation OK");
  return 0;
}
