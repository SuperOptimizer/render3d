#define main volcomppack_entry
#include "../tools/volcomppack/main.c"
#undef main
#include "dct3d.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  char root[]="/tmp/r3d_pack_XXXXXX";assert(mkdtemp(root));
  char src[512],out[512]; snprintf(src,sizeof src,"%s/source.shard",root);
  snprintf(out,sizeof out,"%s/output.vcs",root);
  FILE *f=fopen(src,"wb");assert(f);
  uint8_t chunk[4096],encoded[DCT3D_MAX_BYTES];memset(chunk,128,sizeof chunk);
  size_t n=dct3d_encode_u8(chunk,1,0,0,encoded);assert(n);
  assert(fwrite(encoded,1,n,f)==n);
  uint8_t *index=malloc(R3D_SHARD_INDEX_BYTES);assert(index);
  memset(index,0xff,R3D_SHARD_INDEX_BYTES);
  r3c_put64(index,0);r3c_put64(index+8,n);
  r3c_put32(index+R3D_SHARD_INDEX_BYTES-4,volcomp_crc32c(index,R3D_SHARD_INDEX_BYTES-4));
  assert(fwrite(index,1,R3D_SHARD_INDEX_BYTES,f)==R3D_SHARD_INDEX_BYTES);
  assert(fclose(f)==0);free(index);
  r3d_shard sh;assert(r3d_shard_open_path(src,&sh)==0);
  assert(pack_volume(NULL,&sh,1024,out,2,2)==0);
  volcomp_shard_reader sr;assert(volcomp_shard_open(out,&sr)==0);
  const uint8_t *payload=volcomp_shard_brick(&sr,0,&n);assert(payload);
  uint8_t *reference=calloc(1,128u*128u*128u),*encoded_ref=NULL;assert(reference);
  for(uint32_t z=0;z<16;z++)for(uint32_t y=0;y<16;y++)
    memset(reference+(z*128u+y)*128u,128,16);
  volcomp_brick_params p=volcomp_brick_defaults(2);size_t nr;
  assert(volcomp_brick_encode(&p,reference,128,&encoded_ref,&nr)==0);
  assert(nr==n && memcmp(payload,encoded_ref,n)==0);
  for(uint32_t b=1;b<512;b++)assert(volcomp_shard_brick_is_zero(&sr,b));
  volcomp_shard_close_reader(&sr);
  /* Invalid encode must preserve previously published output. */
  assert(pack_volume(reference,NULL,128,out,0,2)!=0);
  assert(volcomp_shard_open(out,&sr)==0 && sr.foot.shard_dim==1024);
  volcomp_shard_close_reader(&sr);r3d_shard_close(&sh);
  free(reference);free(encoded_ref);unlink(src);unlink(out);rmdir(root);
  puts("packtools: bounded band conversion exact payload and atomic failure OK");
  return 0;
}
