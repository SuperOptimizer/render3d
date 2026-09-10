/* Display deblocking changes seam pixels and survives seed-cache reload. */
#include "synthtree.h"
#include <sys/wait.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
static int run(const char *bin,const char *manifest,const char *next,const char *shot,const char *log) {
  pid_t pid=fork();assert(pid>=0);
  if(!pid) {
    assert(freopen(log,"w",stdout));assert(dup2(fileno(stdout),STDERR_FILENO)>=0);
    setenv("R3D_DEBLOCK",next,1);
    setenv("R3D_MV_FIT","128",1);
    execl(bin,bin,"--bricks",manifest,"--headless","--size","256","256","--pool","8",
      "--frames","320","--shot",shot,(char*)NULL);_exit(127);
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
  char root[]="/tmp/r3d_deblock_XXXXXX",manifest[256],path[256],shot[3][256],log[3][256];
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
  for(unsigned i=0;i<3;i++) {
    snprintf(shot[i],sizeof shot[i],"%s/shot%u.ppm",root,i);
    snprintf(log[i],sizeof log[i],"%s/log%u.txt",root,i);
    assert(run(argv[1],manifest,i?"1":"0",shot[i],log[i])==0);
  }
  size_t n,m,k;unsigned char *a=readfile(shot[0],&n),*b=readfile(shot[1],&m),*c=readfile(shot[2],&k);
  assert(n==m && n==k && memcmp(a,b,n)!=0 && memcmp(b,c,n)==0);
  free(a);free(b);free(c);
  a=readfile(log[2],&n);assert(strstr((char*)a,"from seed-deblock.raw"));free(a);
  for(unsigned i=0;i<3;i++){unlink(shot[i]);unlink(log[i]);}
  snprintf(path,sizeof path,"%s/seed-deblock.raw",root);unlink(path);
  st_rm_tree(root,1);
  puts("deblock render: seam pixels change, filtered seed reload is identical");return 0;
}
