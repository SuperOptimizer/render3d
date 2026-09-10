/* Destroy/reopen volumes immediately after handing work to the decode pool. */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  setenv("R3D_DEBLOCK","1",1);assert(SDL_Init(SDL_INIT_VIDEO));
  char root[]="/tmp/r3d_teardown_XXXXXX",manifest[256];assert(mkdtemp(root));
  uint32_t dims[3]={256,256,256};assert(st_make_tree(root,dims,2,0)==0);
  snprintf(manifest,sizeof manifest,"%s/manifest.json",root);
  const unsigned counts[8]={31,33,63,65,71,73,127,129};
  for(unsigned pass=0;pass<16;pass++) {
    unsigned count=counts[pass%8];
    r3d_config cfg={.headless=true,.headless_w=32,.headless_h=32,.spv_dir=R3D_SPV_DIR};
    r3d_renderer *r=NULL;assert(r3d_create(NULL,&cfg,&r)==0);
    assert(r3d_bricks_begin(r,manifest,16,0)==0);
    assert(r3d_bricks_stream_begin(r));
    for(unsigned i=0;i<count;i++) {
      float p[3]={(float)(i%16*16+8)/256,(float)(i/16*16+8)/256,.25f};
      r3d_bricks_stream_point(r,p,0,0);
    }
    r3d_bricks_stream_submit(r,count);
    r3d_bricks_stats st;r3d_bricks_get_stats(r,&st);assert(st.inflight>0);
    if(pass<8) {
      /* Non-divisible work groups must publish every requested block. */
      r3d_bricks_settle(r);r3d_bricks_get_stats(r,&st);
      assert(st.hot==512u+count && st.failures==0);
    }
    /* The second eight passes deliberately omit settle: shutdown must join
     * before freeing sources, staging, and the shared raw block cache. */
    r3d_destroy(r);
  }
  st_rm_tree(root,2);SDL_Quit();puts("active decode teardown/reopen passed");return 0;
}
