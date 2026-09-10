/* Exercise actual converter entry point against sparse valid source data. */
#define main lodpack_entry
#include "../tools/lodpack/main.c"
#undef main
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  char root[] = "/tmp/r3d_lodpack_XXXXXX";
  assert(mkdtemp(root));
  char source[512], output[512], path[1024];
  snprintf(source, sizeof source, "%s/source", root);
  snprintf(output, sizeof output, "%s/out", root);
  assert(mkdir(source,0755)==0);
  snprintf(path,sizeof path,"%s/0_0_0.shard",source);
  FILE *f=fopen(path,"wb"); assert(f);
  uint8_t chunk[4096], enc[DCT3D_MAX_BYTES];
  memset(chunk, 128, sizeof chunk);
  size_t en=dct3d_encode_u8(chunk,1.0f,0.0f,0.0f,enc); assert(en);
  assert(fwrite(enc,1,en,f)==en);
  uint8_t *index=malloc(ZARR_INDEX_BYTES+4u); assert(index);
  memset(index,0xff,ZARR_INDEX_BYTES);
  put_u64le(index,0); put_u64le(index+8,en);
  put_u32le(index+ZARR_INDEX_BYTES,volcomp_crc32c(index,ZARR_INDEX_BYTES));
  assert(fwrite(index,1,ZARR_INDEX_BYTES+4u,f)==ZARR_INDEX_BYTES+4u);
  assert(fclose(f)==0); free(index);
  char *args[]={"lodpack",source,output,"1024","1024","1024","1",
                "--threads","2","--volcomp-quality","8"};
  assert(lodpack_entry(11,args)==0);
  for (uint32_t l=0;l<2;l++) {
    snprintf(path,sizeof path,"%s/volcomp/L%u/0_0_0.vcs",output,l);
    volcomp_shard_reader r; assert(volcomp_shard_open(path,&r)==0);
    assert(r.foot.q == (l ? 4.0f : 8.0f) && r.foot.lod_level == l);
    size_t n; const uint8_t *b=volcomp_shard_brick(&r,0,&n); assert(b);
    uint8_t out[4096]; assert(r3d_decode_block(b,n,0,0,0,out)==0);
    assert(abs((int)out[0]-128)<=8);
    assert(volcomp_shard_brick_is_zero(&r,511));
    volcomp_shard_close_reader(&r);
  }
  snprintf(path,sizeof path,"%s/zarr/L1/c/0/0/0",output);
  r3d_shard sh; assert(r3d_shard_open_path(path,&sh)==0);
  assert(r3d_shard_chunk_decode(&sh,0,0,0,chunk)==0);
  uint8_t reference[4096]={0}, reconstructed[4096];
  for (uint32_t z=0;z<8;z++) for(uint32_t y=0;y<8;y++)
    memset(reference+(z*16u+y)*16u,128,8);
  en=dct3d_encode_u8(reference,4.0f,0.0f,8.0f,enc);
  assert(dct3d_decode_u8(enc,en,reconstructed));
  assert(memcmp(chunk,reconstructed,4096)==0);
  r3d_shard_close(&sh);
  /* Resume must retain the same completed outputs. */
  assert(lodpack_entry(11,args)==0);
  char cmd[1200]; snprintf(cmd,sizeof cmd,"rm -rf '%s'",root); assert(system(cmd)==0);
  puts("lodpack: per-level quantizer, bounded writer and resume OK");
  return 0;
}
