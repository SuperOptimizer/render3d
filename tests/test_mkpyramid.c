#define NX 1024ull
#define NY 1024ull
#define NZ 1024ull
#define main mkpyramid_entry
#include "../tools/mkpyramid/main.c"
#undef main
#include "bytes.h"
#include "shard.h"
#include "dct3d.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
int main(void) {
  char root[]="/tmp/r3d_pyramid_XXXXXX";assert(mkdtemp(root));
  char src[512],out[512],path[1024];snprintf(src,sizeof src,"%s/source",root);
  snprintf(out,sizeof out,"%s/out",root);assert(mkdir(src,0755)==0);
  snprintf(path,sizeof path,"%s/0_0_0.shard",src);FILE *f=fopen(path,"wb");assert(f);
  uint8_t chunk[4096],enc[DCT3D_MAX_BYTES];memset(chunk,128,sizeof chunk);
  size_t n=dct3d_encode_u8(chunk,1,0,0,enc);assert(n && fwrite(enc,1,n,f)==n);
  uint8_t *idx=malloc(R3D_SHARD_INDEX_BYTES);assert(idx);memset(idx,0xff,R3D_SHARD_INDEX_BYTES);
  for(uint32_t c=7;c<=8;c++) {
    size_t at=((c*64u+c)*64u+c)*16u;r3c_put64(idx+at,0);r3c_put64(idx+at+8,n);
  }
  r3c_put32(idx+R3D_SHARD_INDEX_BYTES-4,volcomp_crc32c(idx,R3D_SHARD_INDEX_BYTES-4));
  assert(fwrite(idx,1,R3D_SHARD_INDEX_BYTES,f)==R3D_SHARD_INDEX_BYTES);
  assert(fclose(f)==0);free(idx);
  char *args[]={"mkpyramid",src,"0",out};assert(mkpyramid_entry(4,args)==0);
  for(uint32_t l=2;l<=5;l++) {
    snprintf(path,sizeof path,"%s/L%u.u8",out,l);f=fopen(path,"rb");assert(f);
    uint32_t dim=1024u>>l;
    for(uint32_t z=0;z<dim;z++)for(uint32_t y=0;y<dim;y++)for(uint32_t x=0;x<dim;x++) {
      int expected=0;
      for(uint32_t c=7;c<=8;c++) {
        uint32_t lo=(c*16u)>>l,hi=((c+1u)*16u-1u)>>l;
        if(x>=lo&&x<=hi&&y>=lo&&y<=hi&&z>=lo&&z<=hi)expected= l==5?16:128;
      }
      assert(fgetc(f)==expected);
    }
    assert(fclose(f)==0);
  }
  assert(mkpyramid_entry(4,args)==0); /* completed shard resume */
  char cmd[1200];snprintf(cmd,sizeof cmd,"rm -rf '%s'",root);assert(system(cmd)==0);
  puts("mkpyramid: exact multilevel rounding across tile boundary and resume OK");return 0;
}
