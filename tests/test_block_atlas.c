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
  /* A window can be released and reopened without destroying its surface. */
  float coords[16] = {0,0,0,1, 16,0,0,1, 0,16,0,1, 16,16,0,1};
  float normals[16] = {0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1};
  assert(r3d_surf_begin(r, 2, 2, coords, normals) == 0);
  for (int i = 0; i < 2; i++) {
    assert(r3d_surfvol_begin(r, 32, 32, 8, 4, 1, 1) == 0);
    r3d_frame_params p = {0};
    r3d_surfvol_params(r, &p);
    assert(p.slab_nx == 32 && p.slab_wz == 8);
    r3d_surfvol_end(r);
    r3d_surfvol_params(r, &p);
    assert(p.slab_nx == 0 && p.slab_wz == 0);
  }
  r3d_destroy(r);
  assert(r3d_create(NULL, &cfg, &r) == 0);
  snprintf(manifest, sizeof manifest, "%s/volcomp/L0/0_0_0.vcs", root);
  unsetenv("R3D_BRICKS_EAGER");
  assert(r3d_bricks_begin(r, manifest, 8, 1) == 0);
  r3d_bricks_get_stats(r, &st);
  assert(st.hot == 0 && st.decoded == 0); /* no eager standalone decode */
  float eye[3] = {0.5f, 0.5f, -1}, fwd[3] = {0, 0, 1};
  for (uint32_t z = 0; z < 8; z++) {
    r3d_bricks_stream(r, eye, fwd, 2, 0.001f, z*16, 1, 0, 128);
    r3d_bricks_settle(r);
    r3d_bricks_get_stats(r, &st);
    assert(st.failures == 0 && st.decoded == (z+1u)*128u);
    assert(st.hot <= st.hot_cap);
    assert(st.slot_probes <= (uint64_t)(z+1u)*st.hot_cap);
  }
  /* Later slices evict 128 slots in one job: exercise the full batch-sized
   * eviction array, not just its obsolete 32-entry prefix. */
  assert(st.hot == 512);
  r3d_destroy(r);
  /* A large sparse manifest allocates request storage for actual candidates
   * and warm metadata per128^3 chunk, not per16^3 virtual page. */
  assert(r3d_create(NULL, &cfg, &r) == 0);
  snprintf(manifest, sizeof manifest, "%s/sparse.json", root);
  FILE *f = fopen(manifest,"w"); assert(f);
  fputs("{\"format\": \"render3d.volcomp-lod.v1\",\"shape\":[42209,22122,20276],"
        "\"brick_shape\":[128,128,128],\"shard_shape\":[1024,1024,1024],\"levels\":[",f);
  for(unsigned li=0;li<8;li++) {
    unsigned scale=1u<<li,dz=(42209u+scale-1u)/scale,dy=(22122u+scale-1u)/scale,dx=(20276u+scale-1u)/scale;
    unsigned nz=(dz+1023u)/1024u,ny=(dy+1023u)/1024u,nx=(dx+1023u)/1024u;
    fprintf(f,"%s{\"level\":%u,\"scale\":%u,\"shape\":[%u,%u,%u],"
              "\"shards\":[%u,%u,%u],\"volcomp\":\"missing/L%u/{z}_{y}_{x}.vcs\"}",
              li?",":"",li,scale,dz,dy,dx,nz,ny,nx,li);
  }
  fputs("]}",f);assert(fclose(f)==0);
  assert(r3d_bricks_begin(r,manifest,8,0)==0);
  r3d_bricks_get_stats(r,&st);
  assert(st.nb > 5289000000ull && st.candidate_capacity==1024u);
  assert(st.chunk_entries < st.nb/400u);
  assert(st.metadata_bytes < 16u*1024u*1024u);
  assert(st.page_capacity >= 4u * st.hot_cap &&
         st.page_capacity < 8u * st.hot_cap);
  assert(st.warm_cap==0 && st.warm_bytes==0);
  printf("sparse metadata: %llu virtual pages, %u chunk entries, %llu bytes\n",
         (unsigned long long)st.nb,st.chunk_entries,(unsigned long long)st.metadata_bytes);
  /* A far-away native block proves shader lookup uses the complete virtual
   * ID, not an atlas-sized/dense index or a truncated coordinate key. */
  char farpath[256];
  snprintf(farpath,sizeof farpath,"%s/volcomp/L0/41_21_19.vcs",root);
  uint8_t *raw=malloc(128u*128u*128u),*encoded=NULL;size_t encoded_n=0;assert(raw);
  memset(raw,200,128u*128u*128u);
  volcomp_brick_params bp=volcomp_brick_defaults(1);
  assert(volcomp_brick_encode(&bp,raw,128,&encoded,&encoded_n)==0);free(raw);
  volcomp_shard_writer *writer=volcomp_shard_create(farpath,1024,128,0,1);assert(writer);
  assert(volcomp_shard_put(writer,1u*64u+4u*8u+5u,encoded,encoded_n)==0);
  assert(volcomp_shard_close(writer)==0);free(encoded);
  float point[3]={20164.0f/42209.0f,22052.0f/42209.0f,42116.0f/42209.0f};
  assert(r3d_bricks_stream_begin(r));
  r3d_bricks_stream_point(r,point,0,0);
  r3d_bricks_stream_submit(r,256);r3d_bricks_settle(r);
  r3d_bricks_get_stats(r,&st);assert(st.hot==1 && st.page_entries==1 && st.failures==0);
  float fc[16]={20161,22049,42116,1,20174,22049,42116,1,20161,22062,42116,1,20174,22062,42116,1};
  assert(r3d_surf_begin(r,2,2,fc,normals)==0);
  assert(r3d_surfvol_begin(r,16,16,4,2,1.0f/16,1.0f/16)==0);
  r3d_surfvol_window(r,0,0,1,0);
  r3d_frame_params fp={.cam_origin={8,8,0},.cam_right={8,0,0},.cam_up={0,8,0},
    .viewport={64,64},.view_flags=R3D_VIEW_SURF,.mode=R3D_MODE_MIP,.density=1,.slab_depth=2};
  r3d_bricks_params(r,&fp);r3d_surfvol_params(r,&fp);
  assert(r3d_frame(r,&fp,NULL)==0);
  uint8_t pixels[64*64*4];uint32_t pw,ph;
  assert(r3d_read_frame(r,pixels,&pw,&ph)==0);
  assert(pw==64 && ph==64 && pixels[(32u*64u+32u)*4u]>=190u);
  r3d_destroy(r);
  /* Reject an overflowing product before any volume-sized allocation. */
  f=fopen(manifest,"w");assert(f);
  fputs("{\"format\": \"render3d.volcomp-lod.v1\",\"shape\":[4294967295,4294967295,4294967295],"
        "\"levels\":[{\"level\":0,\"scale\":1,\"shape\":[4294967295,4294967295,4294967295],"
        "\"shards\":[4194304,4194304,4194304]}]}",f);
  assert(fclose(f)==0);assert(r3d_create(NULL,&cfg,&r)==0);
  assert(r3d_bricks_begin(r,manifest,8,0)!=0);r3d_destroy(r);
  assert(unlink(manifest)==0);assert(unlink(farpath)==0);
  st_rm_tree(root, 2);
  SDL_Quit();
  return 0;
}
