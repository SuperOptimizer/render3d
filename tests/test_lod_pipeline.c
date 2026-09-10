/* Specialized pyramid shaders preserve full and fast rendering. */
#include "synthtree.h"
#include <sys/wait.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static int run(const char *bin,const char *manifest,bool specialized,const char *quality,const char *shot,const char *log) {
  pid_t pid=fork();assert(pid>=0);
  if(!pid) {
    assert(freopen(log,"w",stdout));assert(dup2(fileno(stdout),STDERR_FILENO)>=0);
    setenv("R3D_DEBLOCK","1",1);
    setenv("R3D_NO_PANE_CACHE","1",1);
    if(specialized)unsetenv("R3D_NO_LOD_PIPELINE");else setenv("R3D_NO_LOD_PIPELINE","1",1);
    setenv("R3D_MV_FIT","128",1);
    execl(bin,bin,"--bricks",manifest,"--headless","--size","256","256","--pool","8",
      "--frames","320","--quality",quality,"--shot",shot,(char*)NULL);_exit(127);
  }
  int status;assert(waitpid(pid,&status,0)==pid);return WIFEXITED(status)?WEXITSTATUS(status):1;
}
static unsigned char *readfile(const char *path,size_t *size) {
  FILE *f=fopen(path,"rb");assert(f);assert(!fseek(f,0,SEEK_END));long n=ftell(f);assert(n>=0);
  rewind(f);unsigned char *p=malloc((size_t)n+1);assert(p);
  assert(fread(p,1,(size_t)n,f)==(size_t)n);fclose(f);p[n]=0;*size=(size_t)n;return p;
}
int main(int argc,char **argv) {
  if(argc!=2)return 77;
  char root[]="/tmp/r3d_lodpipe_XXXXXX",manifest[256],path[256],shot[4][256],log[4][256];
  assert(mkdtemp(root));uint32_t dim[3]={128,128,128};
  assert(st_make_tree(root,dim,1,0)==0);
  uint8_t *raw=malloc(128u*128u*128u),*enc=NULL;size_t enc_n=0;assert(raw);
  for(unsigned z=0;z<128;z++)for(unsigned y=0;y<128;y++)for(unsigned x=0;x<128;x++)
    raw[(z*128+y)*128+x]=(uint8_t)(60+4*(x/16)+4*(y/16)+4*(z/16));
  volcomp_brick_params params=volcomp_brick_defaults(8);
  assert(volcomp_brick_encode(&params,raw,128,&enc,&enc_n)==0);
  snprintf(path,sizeof path,"%s/volcomp/L0/0_0_0.vcs",root);
  volcomp_shard_writer *w=volcomp_shard_create(path,1024,128,0,8);assert(w);
  assert(volcomp_shard_put(w,0,enc,enc_n)==0);assert(volcomp_shard_close(w)==0);
  free(raw);free(enc);
  snprintf(manifest,sizeof manifest,"%s/manifest.json",root);
  for(unsigned i=0;i<4;i++) {
    snprintf(shot[i],sizeof shot[i],"%s/shot%u.ppm",root,i);
    snprintf(log[i],sizeof log[i],"%s/log%u.txt",root,i);
    assert(run(argv[1],manifest,(i&1u)!=0,i<2?"full":"fast",shot[i],log[i])==0);
  }
  for(unsigned q=0;q<2;q++) {
    size_t n,m;unsigned char *a=readfile(shot[q*2],&n),*b=readfile(shot[q*2+1],&m);
    assert(n==m);size_t total=0;unsigned largest=0;
    for(size_t i=0;i<n;i++) {
      unsigned d=(unsigned)abs((int)a[i]-(int)b[i]);total+=d;if(d>largest)largest=d;
    }
    printf("quality %u: largest channel difference %u, mean %.6f\n",q,largest,(double)total/(double)n);
    assert(largest<=2 && (double)total/(double)n<0.01);
    free(a);free(b);
  }
  for(unsigned i=0;i<4;i++){unlink(shot[i]);unlink(log[i]);}
  snprintf(path,sizeof path,"%s/seed-deblock.raw",root);unlink(path);
  st_rm_tree(root,1);
  puts("pyramid pipelines: full/fast shader parity");return 0;
}
