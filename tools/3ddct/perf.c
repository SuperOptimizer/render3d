// Throughput benchmark for dct3d. Loads a 128^3 chunk (512 blocks of 16^3),
// times encode and decode separately over many iterations, reports MB/s (of
// raw voxel data), blocks/s, and ns per block.
#include "dct3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CN 128
#define CN3 (CN*CN*CN)
#define BLK 16
#define B3 (BLK*BLK*BLK)
#define NB (CN/BLK)
#define NBLOCKS (NB*NB*NB)      // 512 blocks per chunk

static unsigned char chunk[CN3];
static unsigned char blocks[NBLOCKS][B3];        // pre-gathered blocks
static unsigned char blobs[NBLOCKS][DCT3D_MAX_BYTES];
static size_t bloblen[NBLOCKS];

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

static void gather(const unsigned char *c,int bz,int by,int bx,unsigned char *dst){
    for(int z=0;z<BLK;++z)for(int y=0;y<BLK;++y)for(int x=0;x<BLK;++x)
        dst[(z*BLK+y)*BLK+x]=c[(((size_t)(bz*BLK+z))*CN+(by*BLK+y))*CN+(bx*BLK+x)];
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: perf file.raw\n"); return 2; }
    FILE*f=fopen(argv[1],"rb");
    if(!f||fread(chunk,1,CN3,f)!=CN3){ fprintf(stderr,"read fail\n"); return 1; } fclose(f);
    int bi=0; for(int bz=0;bz<NB;++bz)for(int by=0;by<NB;++by)for(int bx=0;bx<NB;++bx) gather(chunk,bz,by,bx,blocks[bi++]);

    float Q[]={1,4,16,64}; int NQ=sizeof Q/sizeof Q[0];
    const char*base=strrchr(argv[1],'/'); base=base?base+1:argv[1];
    printf("perf on %s  (%d blocks/iter, %d voxels/block)\n\n", base, NBLOCKS, B3);
    printf("  %-6s %-8s  %10s  %12s  %10s\n","q","dir","MB/s","blocks/s","ns/block");
    printf("  --------------------------------------------------------------\n");

    volatile unsigned long sink=0;
    for(int qi=0;qi<NQ;++qi){
        float q=Q[qi];
        // warm + fill blobs
        for(int i=0;i<NBLOCKS;++i) bloblen[i]=dct3d_encode_u8(blocks[i],q,0,0,blobs[i]);

        // pick iteration count so each phase runs ~0.5s+
        int iters=200;
        // ---- encode ----
        double t0=now_s();
        for(int it=0;it<iters;++it)
            for(int i=0;i<NBLOCKS;++i){ size_t n=dct3d_encode_u8(blocks[i],q,0,0,blobs[i]); sink+=n; }
        double te=now_s()-t0;
        // ---- decode ----
        unsigned char back[B3];
        t0=now_s();
        for(int it=0;it<iters;++it)
            for(int i=0;i<NBLOCKS;++i){ dct3d_decode_u8(blobs[i],bloblen[i],back); sink+=back[0]; }
        double td=now_s()-t0;

        double totblk=(double)iters*NBLOCKS;
        double totbytes=totblk*B3;                 // raw voxel bytes processed
        printf("  q%-5g encode   %10.1f  %12.0f  %10.1f\n", q, totbytes/te/1e6, totblk/te, te/totblk*1e9);
        printf("  %-6s decode   %10.1f  %12.0f  %10.1f\n", "", totbytes/td/1e6, totblk/td, td/totblk*1e9);
    }
    fprintf(stderr,"sink=%lu\n",sink);
    return 0;
}
