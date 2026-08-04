// Benchmark dct3d on real 128^3 scroll chunks (u8). Tiles each 128^3 chunk into
// 512 blocks of 16^3, encodes/decodes each, reports ratio + a full error suite:
// PSNR, MAE, SSIM, and 90/95/99/max percentile absolute error.
//
// Sweep: quality in {1,2,4,8,16,32,64}, each alone and with tau = 2*quality.
#include "dct3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CN 128
#define CN3 (CN*CN*CN)
#define BLK 16
#define NB (CN/BLK)

static unsigned char chunk[CN3];
static unsigned char recon[CN3];   // full reconstructed chunk (for SSIM windows)

static void gather(const unsigned char *c, int bz,int by,int bx, unsigned char *dst){
    for(int z=0;z<BLK;++z)for(int y=0;y<BLK;++y)for(int x=0;x<BLK;++x)
        dst[(z*BLK+y)*BLK+x] = c[(((size_t)(bz*BLK+z))*CN + (by*BLK+y))*CN + (bx*BLK+x)];
}
static void scatter(unsigned char *c, int bz,int by,int bx, const unsigned char *src){
    for(int z=0;z<BLK;++z)for(int y=0;y<BLK;++y)for(int x=0;x<BLK;++x)
        c[(((size_t)(bz*BLK+z))*CN + (by*BLK+y))*CN + (bx*BLK+x)] = src[(z*BLK+y)*BLK+x];
}

// Global 3D SSIM over the whole chunk, single window = mean/var/cov of the
// entire volume (standard global-SSIM; C1,C2 per the u8 dynamic range).
static double ssim_global(const unsigned char *a, const unsigned char *b, int n){
    double ma=0,mb=0; for(int i=0;i<n;++i){ ma+=a[i]; mb+=b[i]; } ma/=n; mb/=n;
    double va=0,vb=0,cov=0;
    for(int i=0;i<n;++i){ double da=a[i]-ma, db=b[i]-mb; va+=da*da; vb+=db*db; cov+=da*db; }
    va/=(n-1); vb/=(n-1); cov/=(n-1);
    double C1=(0.01*255)*(0.01*255), C2=(0.03*255)*(0.03*255);
    return ((2*ma*mb+C1)*(2*cov+C2)) / ((ma*ma+mb*mb+C1)*(va+vb+C2));
}

static int cmp_u32(const void*x,const void*y){ uint32_t a=*(const uint32_t*)x,b=*(const uint32_t*)y; return a<b?-1:a>b; }

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: bench file.raw ...\n"); return 2; }
    float Q[] = {1,2,4,8,16,32,64};
    int NQ = sizeof Q/sizeof Q[0];
    static uint32_t abserr[CN3];   // per-voxel |error| for percentiles

    for(int fi=1; fi<argc; ++fi){
        FILE *f=fopen(argv[fi],"rb");
        if(!f){ fprintf(stderr,"open %s failed\n",argv[fi]); continue; }
        if(fread(chunk,1,CN3,f)!=CN3){ fprintf(stderr,"short read\n"); fclose(f); continue; }
        fclose(f);

        double sum=0; long nz=0,mn=255,mx=0;
        for(int i=0;i<CN3;++i){ int v=chunk[i]; sum+=v; nz+=v!=0; if(v<mn)mn=v; if(v>mx)mx=v; }
        const char *base=strrchr(argv[fi],'/'); base=base?base+1:argv[fi];
        printf("\n================ %s ================\n", base);
        printf("mean=%.1f  min=%ld max=%ld  material=%.1f%%\n\n",
               sum/CN3, mn, mx, 100.0*nz/CN3);
        printf("  %-14s %7s  %6s  %7s  %6s  %6s   %s\n",
               "setting","ratio","b/vox","PSNR","MAE","SSIM","err p90/p95/p99/max");
        printf("  ------------------------------------------------------------------------------\n");

        for(int qi=0; qi<NQ; ++qi){
          for(int pass=0; pass<2; ++pass){          // 0 = lossy only, 1 = tau=2q
            float q=Q[qi], tau = pass? 2.0f*q : 0.0f;
            size_t total=0; unsigned char blk[BLK*BLK*BLK], back[BLK*BLK*BLK], out[DCT3D_MAX_BYTES];
            for(int bz=0;bz<NB;++bz)for(int by=0;by<NB;++by)for(int bx=0;bx<NB;++bx){
                gather(chunk,bz,by,bx,blk);
                size_t n=dct3d_encode_u8(blk,q,0.0f,tau,out);
                total+=n;
                if(!dct3d_decode_u8(out,n,back)){ fprintf(stderr,"decode fail\n"); return 1; }
                scatter(recon,bz,by,bx,back);
            }
            // error suite over the whole chunk
            double se=0,ae=0;
            for(int i=0;i<CN3;++i){ int e=(int)chunk[i]-recon[i]; int a=e<0?-e:e; abserr[i]=(uint32_t)a; se+=(double)e*e; ae+=a; }
            double mse=se/CN3, psnr=mse>0?10*log10(255.0*255.0/mse):999.0, mae=ae/CN3;
            double ssim=ssim_global(chunk,recon,CN3);
            qsort(abserr,CN3,sizeof(uint32_t),cmp_u32);
            uint32_t p90=abserr[(int)(0.90*CN3)], p95=abserr[(int)(0.95*CN3)],
                     p99=abserr[(int)(0.99*CN3)], pmax=abserr[CN3-1];
            double ratio=(double)CN3/total, bvox=(double)total/CN3;
            char name[32];
            if(pass) snprintf(name,sizeof name,"q%-2g tau=%g",q,tau);
            else     snprintf(name,sizeof name,"q%-2g",q);
            printf("  %-14s %6.1fx  %6.3f  %6.2f  %6.3f  %6.4f   %u / %u / %u / %u\n",
                   name, ratio, bvox, psnr, mae, ssim, p90,p95,p99,pmax);
          }
        }
    }
    return 0;
}
