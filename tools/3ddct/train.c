// Derive static context priors: encode all chunks at all qualities, dump the
// observed P(bit==0) per (class,slot). Build with -DDCT3D_TRAIN.
#include "dct3d.h"
#include <stdio.h>
#include <stdlib.h>

extern long g_tr_n[3][32], g_tr_z[3][32];

#define CN 128
#define CN3 (CN*CN*CN)
static unsigned char chunk[CN3];

int main(int argc,char**argv){
    float Q[]={1,2,4,8,16,32,64}; int NQ=7;
    for(int fi=1;fi<argc;++fi){
        FILE*f=fopen(argv[fi],"rb"); if(!f||fread(chunk,1,CN3,f)!=CN3){fclose(f);continue;} fclose(f);
        for(int bz=0;bz<8;++bz)for(int by=0;by<8;++by)for(int bx=0;bx<8;++bx){
            unsigned char blk[4096],out[DCT3D_MAX_BYTES];
            for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x)
                blk[(z*16+y)*16+x]=chunk[(((size_t)(bz*16+z))*CN+(by*16+y))*CN+(bx*16+x)];
            for(int qi=0;qi<NQ;++qi) dct3d_encode_u8(blk,Q[qi],0,0,out);
        }
    }
    const char*names[3]={"PRI_SIG","PRI_MAG","PRI_EOB"};
    int lens[3]={32,12,14};
    for(int c=0;c<3;++c){
        printf("static const uint16_t %s[] = {\n  ",names[c]);
        for(int s=0;s<lens[c];++s){
            long n=g_tr_n[c][s];
            int p0 = n>50 ? (int)(4096.0*g_tr_z[c][s]/n + 0.5) : 2048;
            if(p0<1)p0=1; if(p0>4095)p0=4095;
            printf("%d,",p0);
        }
        printf("\n};\n");
    }
    return 0;
}
