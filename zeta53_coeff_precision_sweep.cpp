#include <mpfr.h>
#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int WORK_BITS=384, GEN_BITS=4096, REF_BITS=2048, MAXT=900, N=5000;
static inline int terms(double a){double t;if(a<3.0)t=10.0+10.5*a-2.14*a*a;else if(a<=10.0)t=11.0+5.0*a;else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a;else t=-37.0+6.85*a+0.0194*a*a;return std::max(6,(int)std::ceil(t)+1);}
static uint64_t key(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;auto x=key(a),y=key(b);return x>y?x-y:y-x;}
int main(){
 std::vector<double>A(N),R(N);std::mt19937_64 g1(0xC0EFF1C1E17ULL),g2(0x384B17CACEULL);std::uniform_real_distribution<double>di(1e-12,3.0),doo(3.0,102.0);for(int i=0;i<N/2;i++){double x=di(g1);if(x>=3)x=std::nextafter(3.0,0.0);A[i]=x;}for(int i=N/2;i<N;i++){double x=doo(g2);if(x<=3)x=std::nextafter(3.0,INFINITY);A[i]=x;}
 // Replace a few values by conditioning/stress probes without changing N.
 const double p[]={2.9,2.99,2.9999,std::nextafter(3.0,0.0),std::nextafter(3.0,INFINITY),3.0001,4.0,10.0,20.0,40.0,50.0,75.0,81.764554025359956,100.0,102.0};for(size_t i=0;i<sizeof(p)/sizeof(p[0]);i++)A[i]=p[i];
 arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,tc,p3,inv,aa,ss,rr;arb_init(one);arb_init(tc);arb_init(p3);arb_init(inv);arb_init(aa);arb_init(ss);arb_init(rr);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);std::fprintf(stderr,"Generating corrected coefficients at %d bits...\n",GEN_BITS);arb_poly_zeta_series(z,s,one,0,MAXT+1,GEN_BITS);
 std::vector<arf_struct> FULL(MAXT);arb_set_ui(p3,9);for(int m=0;m<MAXT;m++){arf_init(FULL.data()+m);arb_poly_get_coeff_arb(tc,z,m+1);arb_inv(inv,p3,GEN_BITS);arb_add(tc,tc,inv,GEN_BITS);arf_set_round(FULL.data()+m,arb_midref(tc),GEN_BITS,ARF_RND_NEAR);arb_mul_ui(p3,p3,3,GEN_BITS);}
 for(int i=0;i<N;i++){arb_set_d(aa,A[i]);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(rr,ss,REF_BITS);R[i]=arf_get_d(arb_midref(rr),ARF_RND_NEAR);}
 const int CB[]={384,416,448,480,512,576,640,704,768,896,1024,1280,1536,2048,3072,4096};
 for(int cb:CB){std::vector<arf_struct>C(MAXT);for(int m=0;m<MAXT;m++){arf_init(C.data()+m);arf_set_round(C.data()+m,FULL.data()+m,cb,ARF_RND_NEAR);}arf_t a,pv,base,third;arf_init(a);arf_init(pv);arf_init(base);arf_init(third);arf_set_ui(third,1);arf_div_ui(third,third,3,WORK_BITS,ARF_RND_NEAR);uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;for(int i=0;i<N;i++){arf_set_d(a,A[i]);int T=terms(A[i]);arf_set(pv,C.data()+T-1);for(int m=T-2;m>=0;m--)arf_fma(pv,pv,a,C.data()+m,WORK_BITS,ARF_RND_NEAR);arf_mul(pv,pv,a,WORK_BITS,ARF_RND_NEAR);arf_sub_ui(base,a,3,WORK_BITS,ARF_RND_NEAR);arf_ui_div(base,1,base,WORK_BITS,ARF_RND_NEAR);arf_add(pv,pv,base,WORK_BITS,ARF_RND_NEAR);arf_add(pv,pv,third,WORK_BITS,ARF_RND_NEAR);double y=arf_get_d(pv,ARF_RND_NEAR);uint64_t e=ulp(y,R[i]);if(e==UINT64_MAX){bad++;continue;}if(e>mx){mx=e;w=i;}gt1+=e>1;gt2+=e>2;}std::printf("COEFF_BITS=%d WORK_BITS=%d MAX_ULP=%llu GT1=%d/%d GT2=%d/%d NONFINITE=%d WORST_A=%.17g T=%d\n",cb,WORK_BITS,(unsigned long long)mx,gt1,N,gt2,N,bad,A[w],terms(A[w]));arf_clear(third);arf_clear(base);arf_clear(pv);arf_clear(a);for(int m=0;m<MAXT;m++)arf_clear(C.data()+m);}
 for(int m=0;m<MAXT;m++)arf_clear(FULL.data()+m);arb_clear(rr);arb_clear(ss);arb_clear(aa);arb_clear(inv);arb_clear(p3);arb_clear(tc);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();return 0;
}
