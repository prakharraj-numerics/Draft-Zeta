#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

static int candidate_terms(double a){
    double t;
    if(a<3.0) t=10.0+10.5*a-2.14*a*a;
    else if(a<=10.0) t=11.0+5.0*a;
    else if(a<=40.0) t=11.0+4.54*a+0.0485*a*a;
    else t=-37.0+6.85*a+0.0194*a*a;
    return std::max(6,(int)std::ceil(t)+2);
}
static uint64_t key(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}

int main(){
 constexpr int N=5000, COEFF_BITS=8192, REF_BITS=1024, MAXT=900; const int W[]={384,448,512,640};
 std::vector<double>A(N),R(N);std::mt19937_64 g1(0xa54ff53a5f1d36f1ULL),g2(0x510e527fade682d1ULL);std::uniform_real_distribution<double>di(1e-12,3.0),doo(3.0,102.0);
 for(int i=0;i<2500;i++){double x=di(g1);if(x>=3.0)x=std::nextafter(3.0,0.0);A[i]=x;}for(int i=2500;i<N;i++){double x=doo(g2);if(x<=3.0)x=std::nextafter(3.0,INFINITY);A[i]=x;}
 int mn=9999,mx=0;long long sm=0;for(double a:A){int t=candidate_terms(a);mn=std::min(mn,t);mx=std::max(mx,t);sm+=t;}if(mx>MAXT){std::printf("RULE_OVERFLOW %d\n",mx);return 2;}
 arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,p3,tmp,aa,ss,ref,third;arb_init(one);arb_init(p3);arb_init(tmp);arb_init(aa);arb_init(ss);arb_init(ref);arb_init(third);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);std::fprintf(stderr,"Generating coefficients...\n");arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);std::vector<arb_struct>TC(MAXT);arb_set_ui(p3,9);for(int m=0;m<MAXT;m++){arb_init(TC.data()+m);arb_poly_get_coeff_arb(TC.data()+m,z,m+1);arb_inv(tmp,p3,COEFF_BITS);arb_add(TC.data()+m,TC.data()+m,tmp,COEFF_BITS);arb_mul_ui(p3,p3,3,COEFF_BITS);}
 for(int i=0;i<N;i++){arb_set_d(aa,A[i]);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);R[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);}
 std::printf("N_RANDOM=%d INSIDE=2500 OUTSIDE=2500 GUARD=2 MIN_T=%d MAX_T=%d MEAN_T=%.6f\n",N,mn,mx,(double)sm/N);
 for(int wb:W){uint64_t maxu=0;int gt1=0,gt2=0,bad=0,worst=0;arb_t p,y,base;arb_init(p);arb_init(y);arb_init(base);arb_one(third);arb_div_ui(third,third,3,wb);for(int i=0;i<N;i++){double ad=A[i];int T=candidate_terms(ad);arb_set_d(aa,ad);arb_set(p,TC.data()+T-1);for(int m=T-2;m>=0;m--){arb_mul(p,p,aa,wb);arb_add(p,p,TC.data()+m,wb);}arb_mul(y,p,aa,wb);arb_sub_ui(tmp,aa,3,wb);arb_inv(base,tmp,wb);arb_add(y,y,base,wb);arb_add(y,y,third,wb);double got=arf_get_d(arb_midref(y),ARF_RND_NEAR);uint64_t e=ulp(got,R[i]);if(e==UINT64_MAX){bad++;continue;}if(e>maxu){maxu=e;worst=i;}gt1+=e>1;gt2+=e>2;}std::printf("WORKBITS=%d MAX_ULP=%llu GT1=%d/%d GT2=%d/%d NONFINITE=%d WORST_A=%.17g WORST_T=%d\n",wb,(unsigned long long)maxu,gt1,N,gt2,N,bad,A[worst],candidate_terms(A[worst]));arb_clear(base);arb_clear(y);arb_clear(p);}
 arb_clear(third);for(int m=0;m<MAXT;m++)arb_clear(TC.data()+m);arb_clear(ref);arb_clear(ss);arb_clear(aa);arb_clear(tmp);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();return 0;
}
