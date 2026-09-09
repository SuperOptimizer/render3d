/* The GPU cache counts and uploads 16^3 blocks, independently of disk chunks.
 */
#include "render/render.h"
#include "synthtree.h"
#include <SDL3/SDL.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  assert(SDL_Init(SDL_INIT_VIDEO));
  r3d_config cfg = {.headless = true,
                    .headless_w = 64,
                    .headless_h = 64,
                    .spv_dir = R3D_SPV_DIR};
  r3d_renderer *r = NULL;
  if (r3d_create(NULL, &cfg, &r) != 0) {
    SDL_Quit();
    return 77;
  }
  char root[] = "/tmp/r3d_block_atlas_XXXXXX";
  assert(mkdtemp(root));
  uint32_t dim[3] = {256, 128, 128};
  assert(st_make_tree(root, dim, 2, 0) == 0);
  char manifest[256];
  snprintf(manifest, sizeof manifest, "%s/manifest.json", root);
  assert(r3d_bricks_begin(r, manifest, 8, 1) == 0);
  (void)r3d_bricks_stream_begin(r);
  r3d_bricks_stats st;
  r3d_bricks_get_stats(r, &st);
  assert(st.nb == 16u * 8u * 8u + 8u * 4u * 4u);
  assert(st.hot_cap == 8u * 8u * 8u);
  assert(st.hot ==
         8u * 4u * 4u); /* only the coarsest level is decoded at startup */
  assert(st.warm_bricks ==
         1); /* all 128 resident blocks share one compressed chunk */
  printf("GPU cache: %u resident 16^3 blocks (%u KiB), %u compressed chunk, %u "
         "slots\n",
         st.hot, st.hot * 4, st.warm_bricks, st.hot_cap);
  r3d_destroy(r);
  st_rm_tree(root, 2);
  SDL_Quit();
  return 0;
}
