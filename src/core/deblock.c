#include "deblock.h"
#include <stddef.h>
#include <string.h>

bool r3d_deblock16_batch(uint8_t block[4096], float q, const uint32_t origin[3],
                   const uint32_t dims[3], r3d_deblock_read_batch read, void *ctx) {
  uint8_t halo[8000] = {0}, valid[8000] = {0}, neighbors[3904];
  r3d_block_region regions[26];
  struct tile { int lo[3], extent[3], delta[3]; uint32_t start[3]; int region; } tiles[27];
  uint32_t nt=0,nr=0; size_t used=0;
  bool complete = true;
  for (int bz = -1; bz <= 1; bz++)
    for (int by = -1; by <= 1; by++)
      for (int bx = -1; bx <= 1; bx++) {
        struct tile t={.delta={bx,by,bz},.region=-1};
        bool outside=false;
        for(int a=0;a<3;a++) {
          int64_t p=(int64_t)origin[a]+t.delta[a]*16;
          outside|=p<0 || p>=dims[a];
          t.start[a]=(uint32_t)p;
          t.lo[a]=t.delta[a]<0?14:0;
          t.extent[a]=t.delta[a]?2:16;
        }
        if(outside)continue;
        if(bx || by || bz) {
          t.region=(int)nr;
          regions[nr++]=(r3d_block_region){
            .x=t.start[0]+(uint32_t)t.lo[0],.y=t.start[1]+(uint32_t)t.lo[1],.z=t.start[2]+(uint32_t)t.lo[2],
            .nx=(uint32_t)t.extent[0],.ny=(uint32_t)t.extent[1],.nz=(uint32_t)t.extent[2],.out=neighbors+used};
          used+=(size_t)t.extent[0]*(size_t)t.extent[1]*(size_t)t.extent[2];
        }
        tiles[nt++]=t;
      }
  if(nr)read(ctx,regions,nr);
  for(uint32_t i=0;i<nt;i++) {
    const struct tile *t=&tiles[i];
    const int *lo=t->lo,*extent=t->extent,*delta=t->delta;
    const uint8_t *src=block;
    if(t->region>=0) {
      const r3d_block_region *region=&regions[t->region];
      if(!region->available) { complete=false; continue; }
      src=region->out;
    }
    uint32_t valid_extent[3];
    for(int a=0;a<3;a++) {
      uint64_t begin=(uint64_t)t->start[a]+(uint32_t)lo[a];
      uint64_t available=begin<dims[a] ? dims[a]-begin : 0;
      valid_extent[a]=available<(uint32_t)extent[a] ? (uint32_t)available : (uint32_t)extent[a];
    }
    for(int z=0;z<extent[2];z++)for(int y=0;y<extent[1];y++) {
      size_t h=(size_t)(((z+lo[2]+delta[2]*16+2)*20+y+lo[1]+delta[1]*16+2)*20+lo[0]+delta[0]*16+2);
      memcpy(halo+h,src+(z*extent[1]+y)*extent[0],(size_t)extent[0]);
      if((uint32_t)z<valid_extent[2] && (uint32_t)y<valid_extent[1])memset(valid+h,1,valid_extent[0]);
    }
  }
  float cf = 0.8f*q + 1.0f;
  int c = (int)(cf < 1 ? 1 : cf > 24 ? 24 : cf);
  const int stride[3] = {1,20,400};
  /* Same gated four-tap kernel and X/Y/Z order as volcomp_deblock. Filtering
   * the halo as well preserves upstream's result at block edges/corners. */
  for (int axis = 0; axis < 3; axis++) {
    /* After an axis is filtered, its outer halo is dead: later passes
     * never step in that axis. Keep only the central range in completed
     * axes, while preserving the halo needed by the remaining passes. */
    int ulo=axis==2?2:0, uhi=axis==2?18:20;
    int vlo=axis>=1?2:0, vhi=axis>=1?18:20;
    int s=stride[axis], su=stride[(axis+1)%3], sv=stride[(axis+2)%3];
    for (int face = 2; face <= 18; face += 16)
      for (int v = vlo; v < vhi; v++)
        for (int u = ulo; u < uhi; u++) {
          int i=face*s+u*su+v*sv;
          if (!valid[i-2*s] || !valid[i-s] || !valid[i] || !valid[i+s]) continue;
          int d=(int)halo[i]-halo[i-s], ad=d<0?-d:d;
          int dp=(int)halo[i-2*s]-halo[i-s], dq=(int)halo[i+s]-halo[i];
          if (!ad || ad>=4*c || (dp<0?-dp:dp)>=c || (dq<0?-dq:dq)>=c) continue;
          int change=(3*ad+4)>>3;
          if(d<0)change=-change;
          if(face==2)halo[i]=(uint8_t)(halo[i]-change);
          else halo[i-s]=(uint8_t)(halo[i-s]+change);
        }
  }
  for (int z=0;z<16;z++)for(int y=0;y<16;y++)
    memcpy(block+(z*16+y)*16,halo+((z+2)*20+y+2)*20+2,16);
  return complete;
}

struct scalar_reader { r3d_deblock_read read; void *ctx; };
static void read_scalar(void *ctx,r3d_block_region *regions,uint32_t n) {
  struct scalar_reader *s=ctx;
  for(uint32_t i=0;i<n;i++) {
    r3d_block_region *q=&regions[i];
    q->available=s->read(s->ctx,q->x,q->y,q->z,q->nx,q->ny,q->nz,q->out);
  }
}
bool r3d_deblock16(uint8_t block[4096],float q,const uint32_t origin[3],
                   const uint32_t dims[3],r3d_deblock_read read,void *ctx) {
  struct scalar_reader s={read,ctx};
  return r3d_deblock16_batch(block,q,origin,dims,read_scalar,&s);
}
