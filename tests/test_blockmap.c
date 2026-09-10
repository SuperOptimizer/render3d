#include "vk/blockmap.h"
#include "vk/presence.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
static uint32_t rng=19;
static uint32_t next(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
int main(void){
  struct br_map m={0};assert(br_map_resize(&m,256)==0);
  uint32_t keys[64],n=0;
  for(uint32_t k=0;n<64;k++)if((br_hash(k)&255u)==254u)keys[n++]=k;
  for(uint32_t i=0;i<n;i++)assert(br_map_insert(&m,keys[i],i)!=UINT32_MAX);
  for(uint32_t i=0;i<n;i+=2)br_map_delete(&m,keys[i],NULL,NULL);
  for(uint32_t i=0;i<n;i++){
    uint32_t e=br_map_find(&m,keys[i]);
    assert(i%2 ? e!=UINT32_MAX && m.entries[e].value==i : e==UINT32_MAX);
  }
  for(uint32_t i=0;i<n;i+=2)assert(br_map_insert(&m,keys[i],i+100)!=UINT32_MAX);
  for(uint32_t i=0;i<n;i++)br_map_delete(&m,keys[i],NULL,NULL);
  assert(m.count==0);
  assert(br_map_resize(&m,1024)==0);
  uint32_t expected[4096];memset(expected,0xff,sizeof expected);
  for(unsigned t=0;t<30000;t++){
    uint32_t k=next()%4096;
    if((next()&1u) && m.count<500){
      uint32_t v=next()&0xffffu;assert(br_map_insert(&m,k,v)!=UINT32_MAX);expected[k]=v;
    }else{br_map_delete(&m,k,NULL,NULL);expected[k]=UINT32_MAX;}
    if(t%500==0)for(uint32_t j=0;j<4096;j++){
      uint32_t e=br_map_find(&m,j);
      assert(expected[j]==UINT32_MAX ? e==UINT32_MAX : e!=UINT32_MAX&&m.entries[e].value==expected[j]);
    }
  }
  assert(br_map_resize(&m,2048)==0);
  for(uint32_t j=0;j<4096;j++){
    uint32_t e=br_map_find(&m,j);
    assert(expected[j]==UINT32_MAX ? e==UINT32_MAX : e!=UINT32_MAX&&m.entries[e].value==expected[j]);
  }
  br_map_clear(&m);
  uint64_t wide[]={0xffffffffull,0x1ffffffffull,0xf123456789abcdefull};
  for(unsigned i=0;i<3;i++)assert(br_map_insert(&m,wide[i],i+7)!=UINT32_MAX);
  br_map_delete(&m,wide[1],NULL,NULL);
  assert(br_map_find(&m,wide[1])==UINT32_MAX && br_map_find(&m,UINT64_MAX)==UINT32_MAX);
  assert(m.entries[br_map_find(&m,wide[0])].value==7 && m.entries[br_map_find(&m,wide[2])].value==9);
  struct ni_presence *presence=ni_presence_new();assert(presence);
  for(uint64_t i=0;i<300000;i++)ni_presence_set(presence,i+(1ull<<40),(uint32_t)(i%2)+1);
  assert(presence->map.capacity<=262144 && presence->map.count<=131072);
  assert(ni_presence_get(presence,299999+(1ull<<40))==2);
  ni_presence_clear(presence);assert(ni_presence_get(presence,299999+(1ull<<40))==0);
  ni_presence_free(presence);
  free(m.entries);puts("blockmap: collisions, wraparound deletion, churn and growth preserve all keys");
  return 0;
}
