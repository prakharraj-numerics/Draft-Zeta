#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
static int base_terms(double a){double t;if(a<3.0)t=10.0+10.5*a-2.14*a*a;else if(a<=10.0)t=11.0+5.0*a;else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a;else t=-37.0+6.85*a+0.0194*a*a;return std::max(6,(int)std::ceil(t));}
static uint64_t key(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}
int main(){constexpr int N=5000,MAXT=900,CB=8192,RB=1024,WB=384;std::vector<double>A(N),R(N);std::mt19937_64 g1(0x1f83d9abfb41bd6bULL),g2(0x5be0cd19137e2179ULL);std::uniform_real_distribution<double>di(1e-12,3.0),doo(3.0,102.0);for(int i=0;i<2500;i++){double x=di(g1);if(x>=3)x=std::nextafter(3.0,0.0);A[i]=x;}for(int i=2500;i<N;i++){double x=doo(g2);if(x<=3)x=std::nextafter(3.0,INFINITY);A[i]=x;}
 arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,p3,tmp,aa,ss,ref,third;arb_init(one);arb_init(p3);arb_init(tmp);arb_init(aa);arb_init(ss);arb_init(ref);arb_init(third);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);std::fprintf(stderr,"Generating coefficients...\n");arb_poly_zeta_series(z,s,one,0,MAXT+1,CB);std::vector<arb_struct>TC(MAXT);arb_set_ui(p3,9);for(int m=0;m<MAXT;m++){arb_init(TC.data()+m);arb_poly_get_coeff_arb(TC.data()+m,z,m+1);arb_inv(tmp,p3,CB);arb_add(TC.data()+m,TC.data()+m,tmp,CB);arb_mul_ui(p3,p3,3,CB);}for(int i=0;i<N;i++){arb_set_d(aa,A[i]);arb_sub_ui(ss,aa,2,RB);arb_zeta(ref,ss,RB);R[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);}arb_one(third);arb_div_ui(third,third,3,WB);
 for(int guard=0;guard<=2;guard++){uint64_t mx=0;int gt1=0,w=0;long long sum=0;int mint=9999,maxt=0;arb_t p,y,base;arb_init(p);arb_init(y);arb_init(base);for(int i=0;i<N;i++){int T=base_terms(A[i])+guard;sum+=T;mint=std::min(mint,T);maxt=std::max(maxt,T);arb_set_d(aa,A[i]);arb_set(p,TC.data()+T-1);for(int m=T-2;m>=0;m--){arb_mul(p,p,aa,WB);arb_add(p,p,TC.data()+m,WB);}arb_mul(y,p,aa,WB);arb_sub_ui(tmp,aa,3,WB);arb_inv(base,tmp,WB);arb_add(y,y,base,WB);arb_add(y,y,third,WB);double got=arf_get_d(arb_midref(y),ARF_RND_NEAR);uint64_t e=ulp(got,R[i]);if(e>mx){mx=e;w=i;}gt1+=e>1;}std::printf("GUARD=%d GT1=%d/%d MAX_ULP=%llu MEAN_T=%.6f MIN_T=%d MAX_T=%d WORST_A=%.17g WORST_T=%d\n",guard,gt1,N,(unsigned long long)mx,(double)sum/N,mint,maxt,A[w],base_terms(A[w])+guard);arb_clear(base);arb_clear(y);arb_clear(p);}
 arb_clear(third);for(int m=0;m<MAXT;m++)arb_clear(TC.data()+m);arb_clear(ref);arb_clear(ss);arb_clear(aa);arb_clear(tmp);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();}
