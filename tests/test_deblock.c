#include "core/deblock.h"
#include <volcomp.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
enum { NX=160, NY=48, NZ=48 };
struct source { uint8_t *raw; bool missing; unsigned reads; };
static bool read_block(void *ctx,uint32_t x,uint32_t y,uint32_t z,
                       uint32_t nx,uint32_t ny,uint32_t nz,uint8_t *out) {
  struct source *s=ctx;
  s->reads++;
  if(s->missing && x==128)return false;
  assert(x+nx<=NX && y+ny<=NY && z+nz<=NZ);
  for(unsigned k=0;k<nz;k++)for(unsigned j=0;j<ny;j++)
    memcpy(out+(k*ny+j)*nx,s->raw+((z+k)*NY+y+j)*NX+x,nx);
  return true;
}
int main(void) {
  size_t n=NX*NY*NZ;
  uint8_t *raw=malloc(n), *reference=malloc(n);
  assert(raw && reference);
  uint32_t dims[3]={NX,NY,NZ};
  const float qs[]={1,2,8,32,255};
  for(unsigned pattern=0;pattern<3;pattern++) {
    for(unsigned z=0;z<NZ;z++)for(unsigned y=0;y<NY;y++)for(unsigned x=0;x<NX;x++)
      raw[(z*NY+y)*NX+x]=(uint8_t)(pattern==0 ? 50+(x/16)*3+(y/16)*5+(z/16)*7 :
        pattern==1 ? ((x/16+y/16+z/16)&1)*200 : (x*17+y*71+z*139)%256);
    for(unsigned qi=0;qi<sizeof qs/sizeof *qs;qi++) {
      memcpy(reference,raw,n);
      volcomp_deblock(reference,NZ,NY,NX,qs[qi]);
      struct source source={raw,false,0};
      for(uint32_t z=0;z<NZ;z+=16)for(uint32_t y=0;y<NY;y+=16)for(uint32_t x=0;x<NX;x+=16) {
        uint32_t origin[3]={x,y,z};uint8_t block[4096];
        assert(read_block(&source,x,y,z,16,16,16,block));
        unsigned before=source.reads;
        assert(r3d_deblock16(block,qs[qi],origin,dims,read_block,&source));
        assert(source.reads-before<=26);
        for(unsigned k=0;k<16;k++)for(unsigned j=0;j<16;j++)
          assert(memcmp(block+(k*16+j)*16,reference+((z+k)*NY+y+j)*NX+x,16)==0);
      }
      if(pattern==0 && qi==2) {
        uint8_t block[4096]; uint32_t origin[3]={112,16,16};
        assert(read_block(&source,112,16,16,16,16,16,block));
        source.missing=true;
        assert(!r3d_deblock16(block,qs[qi],origin,dims,read_block,&source));
        source.missing=false;
        assert(read_block(&source,112,16,16,16,16,16,block));
        assert(r3d_deblock16(block,qs[qi],origin,dims,read_block,&source));
        for(unsigned k=0;k<16;k++)for(unsigned j=0;j<16;j++)
          assert(memcmp(block+(k*16+j)*16,reference+((16+k)*NY+16+j)*NX+112,16)==0);
      }
    }
  }
  /* Partial edge blocks retain the same outer-face behavior as a packed
   * assembled volume, while callers still decode/cache complete 16^3 blocks. */
  uint32_t partial[3]={153,47,35};
  for(uint32_t z=0;z<partial[2];z++)for(uint32_t y=0;y<partial[1];y++)
    memcpy(reference+(z*partial[1]+y)*partial[0],raw+(z*NY+y)*NX,partial[0]);
  volcomp_deblock(reference,partial[2],partial[1],partial[0],8);
  struct source edge_source={raw,false,0};
  for(uint32_t z=0;z<partial[2];z+=16)for(uint32_t y=0;y<partial[1];y+=16)for(uint32_t x=0;x<partial[0];x+=16) {
    uint32_t origin[3]={x,y,z};uint8_t block[4096];
    assert(read_block(&edge_source,x,y,z,16,16,16,block));
    assert(r3d_deblock16(block,8,origin,partial,read_block,&edge_source));
    for(uint32_t k=0;k<16 && z+k<partial[2];k++)for(uint32_t j=0;j<16 && y+j<partial[1];j++)
      for(uint32_t i=0;i<16 && x+i<partial[0];i++)
        assert(block[(k*16+j)*16+i]==reference[((z+k)*partial[1]+y+j)*partial[0]+x+i]);
  }
  free(raw);free(reference);
  puts("deblock: upstream parity at block/chunk seams, edges, corners; late neighbors");
  return 0;
}
