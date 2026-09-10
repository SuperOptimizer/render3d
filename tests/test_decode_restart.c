/* Restart checkpoints must preserve public decoder output/error behavior,
 * including buffer reuse, payload mutation, reverse reads, and TLS isolation. */
#include "brick.h"
#include <volcomp.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static void compare(const uint8_t *p,size_t n,uint32_t b) {
  uint8_t got[4096],expected[4096];
  int a=r3d_decode_block(p,n,b/64,b/8%8,b%8,got);
  volcomp_status c=volcomp_decode_block(p,n,b/64,b/8%8,b%8,expected,sizeof expected);
  assert((a==0)==(c==VOLCOMP_OK));
  if(a==0)assert(!memcmp(got,expected,sizeof got));
}
struct job { const uint8_t *p; size_t n; unsigned seed; };
static void *reader(void *ptr) {
  struct job *j=ptr;uint32_t seed=j->seed;
  for(unsigned i=0;i<128;i++) {
    seed=seed*1664525u+1013904223u;compare(j->p,j->n,seed%512);
  }
  return NULL;
}
int main(void) {
  uint8_t *raw=malloc(VOLCOMP_CHUNK_VOXELS);assert(raw);
  for(unsigned pattern=0;pattern<3;pattern++) {
    uint32_t seed=7;
    for(unsigned i=0;i<VOLCOMP_CHUNK_VOXELS;i++) {
      seed=seed*1664525u+1013904223u;
      raw[i]=pattern==0?100:pattern==1?(uint8_t)((i/128+i/16384+i%128)%256):(uint8_t)(seed>>24);
    }
    volcomp_brick_params bp=volcomp_brick_defaults(pattern==2?1:8);
    uint8_t *enc=NULL;size_t n=0;assert(volcomp_brick_encode(&bp,raw,128,&enc,&n)==0);
    vf_parsed parsed;assert(vf_parse(enc,n,&parsed)==VOLCOMP_OK);
    size_t payload=(size_t)(parsed.payload-enc);
    for(unsigned b=0;b<512;b++)compare(enc,n,b);
    for(unsigned b=64;b-->0;)compare(enc,n,b);
    struct job jobs[4];pthread_t threads[4];
    for(unsigned i=0;i<4;i++) {jobs[i]=(struct job){enc,n,19+i};assert(!pthread_create(&threads[i],NULL,reader,&jobs[i]));}
    for(unsigned i=0;i<4;i++)assert(!pthread_join(threads[i],NULL));
    uint8_t *copy=malloc(n);assert(copy);
    for(unsigned i=0;i<48;i++) {
      memcpy(copy,enc,n);
      /* Warm checkpoints from exactly this allocation before modifying it. */
      compare(copy,n,15);compare(copy,n,7);
      /* Change the very substream whose restart state was warmed. Header
       * identity alone cannot make those saved bitreader positions safe. */
      size_t byte=payload+(size_t)(i+1)*7919%parsed.off[1];
      copy[byte]^=(uint8_t)(1u<<(i%8));
      compare(copy,n,15);compare(copy,n,7);
      compare(enc,n,15);compare(enc,n,7);
      compare(enc,n-(i+1),15);
    }
    if(pattern==0) {
      /* Bounds-valid but overlong token data exercises the uncached path.
       * Match the public block API's behavior even for noncanonical input;
       * never copy an oversized substream into the bounded restart buffer. */
      size_t extra=65536, split=payload+parsed.tok_n[0];
      uint8_t *large=malloc(n+extra);assert(large);
      memcpy(large,enc,split);memset(large+split,0,extra);
      memcpy(large+split+extra,enc+split,n-split);
      uint32_t token_n=parsed.tok_n[0]+(uint32_t)extra;
      for(unsigned j=0;j<4;j++)large[payload-VF_DIR_BYTES+j]=(uint8_t)(token_n>>(8*j));
      assert(r3d_validate_brick(large,n+extra)==0);
      for(unsigned b=0;b<32;b++)compare(large,n+extra,b);
      free(large);
    }
    free(copy);free(enc);printf("restart parity: pattern %u passed (first substream %u bytes)\n",pattern,parsed.off[1]);
  }
  free(raw);return 0;
}
