/* A failed browser switch must leave the active renderer and volume intact. */
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
    if(next)setenv("R3D_SWAP_TEST",next,1);else unsetenv("R3D_SWAP_TEST");
    setenv("R3D_MV_FIT","128",1);
    execl(bin,bin,"--bricks",manifest,"--headless","--size","128","128","--pool","8",
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
  char root[]="/tmp/r3d_swap_XXXXXX",manifest[256],bad[256],shot[4][256],log[4][256];
  assert(mkdtemp(root));uint32_t dim[3]={128,128,128};st_const_value=150;
  assert(st_make_tree(root,dim,1,0)==0);
  snprintf(manifest,sizeof manifest,"%s/manifest.json",root);
  snprintf(bad,sizeof bad,"%s/invalid.json",root);
  FILE *f=fopen(bad,"w");assert(f);fputs("{invalid metadata}",f);fclose(f);
  char second[256],second_manifest[320];
  snprintf(second,sizeof second,"%s/second",root);st_const_value=220;
  assert(st_make_tree(second,dim,1,0)==0);
  snprintf(second_manifest,sizeof second_manifest,"%s/manifest.json",second);
  for(unsigned i=0;i<4;i++) {
    snprintf(shot[i],sizeof shot[i],"%s/shot%u.ppm",root,i);
    snprintf(log[i],sizeof log[i],"%s/log%u.txt",root,i);
    assert(run(argv[1],i==2?second_manifest:manifest,i==1?bad:i==3?second_manifest:NULL,shot[i],log[i])==0);
  }
  size_t n,m;unsigned char *a=readfile(shot[0],&n),*b=readfile(shot[1],&m);
  assert(n==m && !memcmp(a,b,n));free(a);free(b);
  a=readfile(log[1],&n);assert(strstr((char*)a,"current volume is still open"));free(a);
  a=readfile(shot[2],&n);b=readfile(shot[3],&m);
  assert(n==m && !memcmp(a,b,n));free(a);free(b);
  a=readfile(shot[0],&n);b=readfile(shot[3],&m);
  assert(n==m && memcmp(a,b,n));free(a);free(b);
  st_rm_tree(second,1);
  for(unsigned i=0;i<4;i++){unlink(shot[i]);unlink(log[i]);}
  unlink(bad);st_rm_tree(root,1);
  puts("dataset switch: rejected metadata keeps current volume and exact rendered image");return 0;
}
