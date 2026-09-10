#include "vk/vkctx.h"
#include "vk/vkres.h"
#include "vk/blockmap.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
int main(void) {
  r3d_vkctx c;if(r3d_vkctx_create(&c,NULL,0,false))return 77;
  VkCommandPool pool;
  VkCommandPoolCreateInfo ci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=c.qfam};
  assert(vkCreateCommandPool(c.dev,&ci,NULL,&pool)==VK_SUCCESS);
  VkDescriptorType types[3]={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  char path[1024];snprintf(path,sizeof path,"%s/pageprobe.spv",R3D_SPV_DIR);
  r3d_vkcomp comp;assert(r3d_vkcomp_create(&c,path,types,3,4,&comp)==0);
  r3d_vkbuf pages,keys,results;
  assert(r3d_vkbuf_create_host(&c,(64u+256u*4u)*4u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,&pages)==0);
  assert(r3d_vkbuf_create_host(&c,128u*8u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,&keys)==0);
  assert(r3d_vkbuf_create_host(&c,128u*4u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,&results)==0);
  r3d_vkcomp_bind_buffer(&c,&comp,0,pages.buf,0,VK_WHOLE_SIZE);
  r3d_vkcomp_bind_buffer(&c,&comp,1,keys.buf,0,VK_WHOLE_SIZE);
  r3d_vkcomp_bind_buffer(&c,&comp,2,results.buf,0,VK_WHOLE_SIZE);
  struct br_map map={0};assert(br_map_resize(&map,256)==0);
  uint64_t *queries=keys.mapped;
  uint32_t n=0;
  for(uint32_t k=0;n<64;k++)if((br_hash(k)&255u)==254u)queries[n++]=k;
  queries[64]=0x1fffffeffull;queries[65]=0xffffffffull;
  queries[66]=queries[0]|(1ull<<32);
  for(uint32_t i=67;i<128;i++)queries[i]=0x7ffffffffull+(((uint64_t)i*7919u*99991u+i*29u)*99989u+i*13u);
  for(uint32_t pass=0;pass<4;pass++) {
    if(pass==0)for(uint32_t i=0;i<128;i++)assert(br_map_insert(&map,queries[i],i==64?0xfffffffeu:i+1000u)!=UINT32_MAX);
    if(pass==1)for(uint32_t i=0;i<64;i+=2)br_map_delete(&map,queries[i],NULL,NULL);
    if(pass==2)for(uint32_t i=0;i<64;i+=2)assert(br_map_insert(&map,queries[i],i+2000u)!=UINT32_MAX);
    if(pass==3){br_map_delete(&map,queries[64],NULL,NULL);assert(br_map_insert(&map,queries[100],123u)!=UINT32_MAX);}
    uint32_t *p=pages.mapped;memset(p,0,(64u+256u*4u)*4u);p[60]=255;p[0]=0xffffffffu;p[3]=7u;
    memcpy(p+64,map.entries,256u*sizeof(struct br_pair));
    VkCommandBuffer cmd=r3d_vk_oneshot_begin(&c,pool);assert(cmd);
    uint32_t count=128;r3d_vkcomp_dispatch(cmd,&comp,&count,4,2,1,1);
    VkMemoryBarrier2 mb={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,.srcAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.dstAccessMask=VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.memoryBarrierCount=1,.pMemoryBarriers=&mb};
    vkCmdPipelineBarrier2(cmd,&dep);assert(r3d_vk_oneshot_end(&c,pool,cmd)==0);
    const uint32_t *got=results.mapped;
    for(uint32_t i=0;i<count;i++) {
      uint32_t e=br_map_find(&map,queries[i]);uint32_t want=e==UINT32_MAX?UINT32_MAX:map.entries[e].value;
      assert(got[i]==want);
    }
  }
  free(map.entries);r3d_vkcomp_destroy(&c,&comp);
  r3d_vkbuf_destroy(&c,&pages);r3d_vkbuf_destroy(&c,&keys);r3d_vkbuf_destroy(&c,&results);
  vkDestroyCommandPool(c.dev,pool,NULL);r3d_vkctx_destroy(&c);
  puts("pagehash_gpu: production lookup matches CPU for collisions, deletion, reuse, high IDs and known air");
  return 0;
}
