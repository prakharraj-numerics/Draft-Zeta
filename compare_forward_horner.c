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
#define P 4096
static uint64_t key(double x){union{double d;uint64_t u;}v={x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}
static uint64_t rng=0x9e3779b97f4a7c15ULL;static uint64_t nextu(){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;return rng*2685821657736338717ULL;}static double uni(){return(nextu()>>11)*(1.0/9007199254740992.0);}
int main(){arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one;arb_init(one);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);arb_poly_zeta_series(z,s,one,0,T1+1,P);arb_t *c=malloc(sizeof(arb_struct)*T1),*tc=malloc(sizeof(arb_struct)*T2);arb_t p3,t;arb_init(p3);arb_init(t);arb_set_ui(p3,9);for(int m=0;m<T1;m++){arb_init(c[m]);arb_poly_get_coeff_arb(c[m],z,m+1);if(m<T2){arb_init(tc[m]);arb_inv(t,p3,P);arb_add(tc[m],c[m],t,P);}arb_mul_ui(p3,p3,3,P);}int badF1=0,badH1=0,badFH1=0,badF2=0,badH2=0,badFH2=0;uint64_t mF1=0,mH1=0,mFH1=0,mF2=0,mH2=0,mFH2=0;arb_t a,ref,ss,pow,sum,h,tmp,third;arb_init(a);arb_init(ref);arb_init(ss);arb_init(pow);arb_init(sum);arb_init(h);arb_init(tmp);arb_init(third);arb_one(third);arb_div_ui(third,third,3,P);
for(int i=0;i<N;i++){double aa=-2.9+5.8*uni();if(fabs(aa)<1e-12)aa=.125;double q=2+98*uni(),ao=q+2;arb_set_d(a,aa);arb_set(pow,a);arb_zero(sum);for(int m=0;m<T1;m++){arb_mul(tmp,c[m],pow,P);arb_add(sum,sum,tmp,P);arb_mul(pow,pow,a,P);}double F=arf_get_d(arb_midref(sum),ARF_RND_NEAR);arb_set(h,c[T1-1]);for(int m=T1-2;m>=0;m--){arb_mul(h,h,a,P);arb_add(h,h,c[m],P);}arb_mul(h,h,a,P);double H=arf_get_d(arb_midref(h),ARF_RND_NEAR);arb_sub_ui(ss,a,2,P);arb_zeta(ref,ss,P);double R=arf_get_d(arb_midref(ref),ARF_RND_NEAR);uint64_t e=ulp(F,R);if(e>mF1)mF1=e;badF1+=e>1;e=ulp(H,R);if(e>mH1)mH1=e;badH1+=e>1;e=ulp(F,H);if(e>mFH1)mFH1=e;badFH1+=e>0;
 arb_set_d(a,ao);arb_set(pow,a);arb_set(sum,third);arb_sub_ui(tmp,a,3,P);arb_inv(tmp,tmp,P);arb_add(sum,sum,tmp,P);for(int m=0;m<T2;m++){arb_mul(tmp,tc[m],pow,P);arb_add(sum,sum,tmp,P);arb_mul(pow,pow,a,P);}F=arf_get_d(arb_midref(sum),ARF_RND_NEAR);arb_set(h,tc[T2-1]);for(int m=T2-2;m>=0;m--){arb_mul(h,h,a,P);arb_add(h,h,tc[m],P);}arb_mul(h,h,a,P);arb_sub_ui(tmp,a,3,P);arb_inv(tmp,tmp,P);arb_add(h,h,tmp,P);arb_add(h,h,third,P);H=arf_get_d(arb_midref(h),ARF_RND_NEAR);arb_sub_ui(ss,a,2,P);arb_zeta(ref,ss,P);R=arf_get_d(arb_midref(ref),ARF_RND_NEAR);e=ulp(F,R);if(e>mF2)mF2=e;badF2+=e>1;e=ulp(H,R);if(e>mH2)mH2=e;badH2+=e>1;e=ulp(F,H);if(e>mFH2)mFH2=e;badFH2+=e>0;}
printf("CASE1_FORWARD GT1=%d MAX=%llu HORNER GT1=%d MAX=%llu F_NE_H=%d FH_MAX=%llu\n",badF1,(unsigned long long)mF1,badH1,(unsigned long long)mH1,badFH1,(unsigned long long)mFH1);printf("CASE2_FORWARD GT1=%d MAX=%llu HORNER GT1=%d MAX=%llu F_NE_H=%d FH_MAX=%llu\n",badF2,(unsigned long long)mF2,badH2,(unsigned long long)mH2,badFH2,(unsigned long long)mFH2);
arb_clear(third);arb_clear(tmp);arb_clear(h);arb_clear(sum);arb_clear(pow);arb_clear(ss);arb_clear(ref);arb_clear(a);arb_clear(t);arb_clear(p3);for(int m=0;m<T1;m++){arb_clear(c[m]);if(m<T2)arb_clear(tc[m]);}free(c);free(tc);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();}
