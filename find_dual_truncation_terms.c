#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define MAXT 1400
#define N 1000
#define PREC 8192
static uint64_t key(double x){union{double d;uint64_t u;}v={x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}
static uint64_t ulp(double a,double b){uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}
static uint64_t rng=0x9e3779b97f4a7c15ULL;static uint64_t nextu(void){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;return rng*2685821657736338717ULL;}static double uni(void){return(nextu()>>11)*(1.0/9007199254740992.0);}
int main(){
 arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,c,tc,p3,tmp,third;arb_init(one);arb_init(c);arb_init(tc);arb_init(p3);arb_init(tmp);arb_init(third);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);fprintf(stderr,"Generating coefficients...\n");arb_poly_zeta_series(z,s,one,0,MAXT+1,PREC);arb_one(third);arb_div_ui(third,third,3,PREC);
 arb_t *a1=malloc(sizeof(arb_struct)*N),*a2=malloc(sizeof(arb_struct)*N),*p1=malloc(sizeof(arb_struct)*N),*p2=malloc(sizeof(arb_struct)*N),*u1=malloc(sizeof(arb_struct)*N),*u2=malloc(sizeof(arb_struct)*N),*r1=malloc(sizeof(arb_struct)*N),*r2=malloc(sizeof(arb_struct)*N);double rd1[N],rd2[N];
 for(int i=0;i<N;i++){arb_init(a1[i]);arb_init(a2[i]);arb_init(p1[i]);arb_init(p2[i]);arb_init(u1[i]);arb_init(u2[i]);arb_init(r1[i]);arb_init(r2[i]);double aa=-2.9+5.8*uni();if(fabs(aa)<1e-12)aa=.125;double x=2+98*uni(),ao=x+2;arb_set_d(a1[i],aa);arb_set_d(a2[i],ao);arb_set(p1[i],a1[i]);arb_set(p2[i],a2[i]);arb_zero(u1[i]);arb_set(u2[i],third);arb_sub_ui(tmp,a2[i],3,PREC);arb_inv(tmp,tmp,PREC);arb_add(u2[i],u2[i],tmp,PREC);arb_sub_ui(tmp,a1[i],2,PREC);arb_zeta(r1[i],tmp,PREC);rd1[i]=arf_get_d(arb_midref(r1[i]),ARF_RND_NEAR);arb_sub_ui(tmp,a2[i],2,PREC);arb_zeta(r2[i],tmp,PREC);rd2[i]=arf_get_d(arb_midref(r2[i]),ARF_RND_NEAR);}
 int best1=0,best2=0;arb_set_ui(p3,9);
 for(int m=0;m<MAXT;m++){
   arb_poly_get_coeff_arb(c,z,m+1);arb_inv(tmp,p3,PREC);arb_add(tc,c,tmp,PREC);
   int bad1=0,bad2=0;uint64_t mx1=0,mx2=0;
   for(int i=0;i<N;i++){
     arb_mul(tmp,c,p1[i],PREC);arb_add(u1[i],u1[i],tmp,PREC);
     arb_mul(tmp,tc,p2[i],PREC);arb_add(u2[i],u2[i],tmp,PREC);
     double d1=arf_get_d(arb_midref(u1[i]),ARF_RND_NEAR),d2=arf_get_d(arb_midref(u2[i]),ARF_RND_NEAR);uint64_t e1=ulp(d1,rd1[i]),e2=ulp(d2,rd2[i]);if(e1>mx1)mx1=e1;if(e2>mx2)mx2=e2;bad1+=e1>1;bad2+=e2>1;
     arb_mul(p1[i],p1[i],a1[i],PREC);arb_mul(p2[i],p2[i],a2[i],PREC);
   }
   int T=m+1;if(!best1&&!bad1){best1=T;printf("CASE1_TRUNC_MIN=%d MAX_ULP=%llu\n",T,(unsigned long long)mx1);fflush(stdout);}if(!best2&&!bad2){best2=T;printf("CASE2_TRUNC_MIN=%d MAX_ULP=%llu\n",T,(unsigned long long)mx2);fflush(stdout);}if(best1&&best2)break;arb_mul_ui(p3,p3,3,PREC);
 }
 if(!best1)printf("CASE1_NO_TRUNC_PASS_UP_TO=%d\n",MAXT);if(!best2)printf("CASE2_NO_TRUNC_PASS_UP_TO=%d\n",MAXT);
 for(int i=0;i<N;i++){arb_clear(a1[i]);arb_clear(a2[i]);arb_clear(p1[i]);arb_clear(p2[i]);arb_clear(u1[i]);arb_clear(u2[i]);arb_clear(r1[i]);arb_clear(r2[i]);}free(a1);free(a2);free(p1);free(p2);free(u1);free(u2);free(r1);free(r2);arb_clear(third);arb_clear(tmp);arb_clear(p3);arb_clear(tc);arb_clear(c);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();return 0;}
