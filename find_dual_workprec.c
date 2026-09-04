#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 1000
#define T1 1119
#define T2 856
#define GENPREC 4096
static uint64_t key(double x){union{double d;uint64_t u;}v={x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}
static uint64_t rng=0x9e3779b97f4a7c15ULL;static uint64_t nextu(){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;return rng*2685821657736338717ULL;}static double uni(){return(nextu()>>11)*(1.0/9007199254740992.0);}
int main(){int W[]={64,80,96,112,128,144,160,192,224,256,288,320,352,384,448,512};int nw=sizeof(W)/sizeof(W[0]);arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one;arb_init(one);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);arb_poly_zeta_series(z,s,one,0,T1+1,GENPREC);
 arb_t *c=malloc(sizeof(arb_struct)*T1),*tc=malloc(sizeof(arb_struct)*T2);arb_t p3,t;arb_init(p3);arb_init(t);arb_set_ui(p3,9);for(int m=0;m<T1;m++){arb_init(c[m]);arb_poly_get_coeff_arb(c[m],z,m+1);if(m<T2){arb_init(tc[m]);arb_inv(t,p3,GENPREC);arb_add(tc[m],c[m],t,GENPREC);}arb_mul_ui(p3,p3,3,GENPREC);}
 double A1[N],A2[N],R1[N],R2[N];arb_t x,r;arb_init(x);arb_init(r);for(int i=0;i<N;i++){A1[i]=-2.9+5.8*uni();if(fabs(A1[i])<1e-12)A1[i]=.125;double q=2+98*uni();A2[i]=q+2;arb_set_d(x,A1[i]-2);arb_zeta(r,x,GENPREC);R1[i]=arf_get_d(arb_midref(r),ARF_RND_NEAR);arb_set_d(x,A2[i]-2);arb_zeta(r,x,GENPREC);R2[i]=arf_get_d(arb_midref(r),ARF_RND_NEAR);}
 for(int wi=0;wi<nw;wi++){slong wp=W[wi];int b1=0,b2=0;uint64_t mx1=0,mx2=0;arb_t a,p,y,tmp,third;arb_init(a);arb_init(p);arb_init(y);arb_init(tmp);arb_init(third);arb_one(third);arb_div_ui(third,third,3,wp);
  for(int i=0;i<N;i++){arb_set_d(a,A1[i]);arb_set(p,c[T1-1]);for(int m=T1-2;m>=0;m--){arb_mul(p,p,a,wp);arb_add(p,p,c[m],wp);}arb_mul(y,p,a,wp);double d=arf_get_d(arb_midref(y),ARF_RND_NEAR);uint64_t e=ulp(d,R1[i]);if(e>mx1)mx1=e;b1+=e>1;
   arb_set_d(a,A2[i]);arb_set(p,tc[T2-1]);for(int m=T2-2;m>=0;m--){arb_mul(p,p,a,wp);arb_add(p,p,tc[m],wp);}arb_mul(y,p,a,wp);arb_sub_ui(tmp,a,3,wp);arb_inv(tmp,tmp,wp);arb_add(y,y,tmp,wp);arb_add(y,y,third,wp);d=arf_get_d(arb_midref(y),ARF_RND_NEAR);e=ulp(d,R2[i]);if(e>mx2)mx2=e;b2+=e>1;}
  printf("WORKBITS=%d CASE1_GT1=%d MAX=%llu CASE2_GT1=%d MAX=%llu\n",W[wi],b1,(unsigned long long)mx1,b2,(unsigned long long)mx2);fflush(stdout);arb_clear(third);arb_clear(tmp);arb_clear(y);arb_clear(p);arb_clear(a);if(!b1&&!b2)break;}
 arb_clear(r);arb_clear(x);arb_clear(t);arb_clear(p3);for(int m=0;m<T1;m++){arb_clear(c[m]);if(m<T2)arb_clear(tc[m]);}free(c);free(tc);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();return 0;}
