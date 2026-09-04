#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

/*
 * Research candidate for positive a != 3.
 * Mathematical spine:
 *   zeta(a-2) = 1/(a-3) + 1/3 + a * sum_{m>=0} tc_m a^m,
 *   tc_m = c_m + 1/3^(m+2).
 *
 * Term count is selected ONCE from a before Horner. There is no runtime
 * convergence test and no adaptive early stop inside the polynomial.
 *
 * The rule below is a no-log upper envelope fitted to dense high-precision
 * minimum-term maps, plus two fixed guard terms. Production can later replace
 * the arithmetic here by a tiny LUT if that wins on throughput.
 */
static int candidate_terms(double a)
{
    double t;
    if (a < 3.0)
        t = 10.0 + 10.5*a - 2.14*a*a;
    else if (a <= 10.0)
        t = 11.0 + 5.0*a;
    else if (a <= 40.0)
        t = 11.0 + 4.54*a + 0.0485*a*a;
    else
        t = -37.0 + 6.85*a + 0.0194*a*a;
    int n = (int)std::ceil(t) + 2;   // fixed guard, not self-adjustment
    return std::max(n, 6);
}

extern "C" int draft_zeta53_candidate_terms(double a)
{
    if (!(a > 0.0) || a == 3.0 || a > 102.0) return 0;
    return candidate_terms(a);
}

static uint64_t key(double x){ union{double d;uint64_t u;}v{x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulp(double a,double b){ if(std::isnan(a)||std::isnan(b)) return UINT64_MAX; uint64_t x=key(a),y=key(b); return x>y?x-y:y-x; }

int main()
{
    constexpr int N=5000;
    constexpr int REF_BITS=8192;
    constexpr int MAXT=900;
    const int workbits[] = {384,448,512,640};

    std::vector<double> A(N), R(N);
    std::mt19937_64 gi(0x43414e4449444154ULL), go(0x5a45544135303030ULL);
    std::uniform_real_distribution<double> ui(1.0e-9, 3.0);
    std::uniform_real_distribution<double> uo(3.0, 102.0);
    for(int i=0;i<2500;i++){ double x=ui(gi); if(x>=3.0)x=std::nextafter(3.0,0.0); A[i]=x; }
    for(int i=2500;i<N;i++){ double x=uo(go); if(x<=3.0)x=std::nextafter(3.0,INFINITY); A[i]=x; }

    // Stress points are part of the same 5000, not extra cherry-picked tests.
    const double sin[] = {1e-9,1e-6,0.01,0.1,1.0,2.0,2.5,2.9,2.99,2.9999,std::nextafter(3.0,0.0)};
    const double sout[] = {std::nextafter(3.0,INFINITY),3.000000001,3.001,3.01,3.1,4.0,5.2,10.0,20.0,40.0,50.0,75.0,100.0,102.0};
    for(size_t i=0;i<sizeof(sin)/sizeof(sin[0]);i++) A[i]=sin[i];
    for(size_t i=0;i<sizeof(sout)/sizeof(sout[0]);i++) A[2500+(int)i]=sout[i];

    int maxT=0,minT=100000; long long sumT=0;
    for(double a:A){ int T=candidate_terms(a); maxT=std::max(maxT,T);minT=std::min(minT,T);sumT+=T; }
    if(maxT>MAXT){ std::fprintf(stderr,"MAXT too small: rule asks %d\n",maxT); return 2; }

    arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);
    arb_t one,p3,tmp,aa,ss,ref,third;arb_init(one);arb_init(p3);arb_init(tmp);arb_init(aa);arb_init(ss);arb_init(ref);arb_init(third);
    arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);
    std::fprintf(stderr,"Generating %d pole-removed coefficients at %d bits...\n",MAXT,REF_BITS);
    arb_poly_zeta_series(z,s,one,0,MAXT+1,REF_BITS);
    std::vector<arb_struct> TC(MAXT);arb_set_ui(p3,9);
    for(int m=0;m<MAXT;m++){arb_init(TC.data()+m);arb_poly_get_coeff_arb(TC.data()+m,z,m+1);arb_inv(tmp,p3,REF_BITS);arb_add(TC.data()+m,TC.data()+m,tmp,REF_BITS);arb_mul_ui(p3,p3,3,REF_BITS);}

    // Independent high-precision reference values, with exact binary64 a loaded
    // before subtracting the exact integer 2.
    for(int i=0;i<N;i++){arb_set_d(aa,A[i]);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);R[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);}

    std::printf("N=%d DOMAIN=(0,102], a!=3\n",N);
    std::printf("TERM_RULE_GUARD=2 MIN_TERMS=%d MAX_TERMS=%d MEAN_TERMS=%.6f\n",minT,maxT,(double)sumT/N);

    for(int wb:workbits){
        uint64_t mx=0;int gt1=0,gt2=0,bad=0,worst=0;
        arb_t p,y,base;arb_init(p);arb_init(y);arb_init(base);arb_one(third);arb_div_ui(third,third,3,wb);
        for(int i=0;i<N;i++){
            const double ad=A[i];const int T=candidate_terms(ad);arb_set_d(aa,ad);arb_set(p,TC.data()+T-1);
            for(int m=T-2;m>=0;m--){arb_mul(p,p,aa,wb);arb_add(p,p,TC.data()+m,wb);}arb_mul(y,p,aa,wb);
            arb_sub_ui(tmp,aa,3,wb);arb_inv(base,tmp,wb);arb_add(y,y,base,wb);arb_add(y,y,third,wb);
            const double got=arf_get_d(arb_midref(y),ARF_RND_NEAR);const uint64_t e=ulp(got,R[i]);
            if(e==UINT64_MAX){bad++;continue;}if(e>mx){mx=e;worst=i;}gt1+=e>1;gt2+=e>2;
        }
        std::printf("WORKBITS=%d MAX_ULP=%llu GT1=%d/%d GT2=%d/%d NONFINITE=%d WORST_A=%.17g TERMS=%d\n",wb,(unsigned long long)mx,gt1,N,gt2,N,bad,A[worst],candidate_terms(A[worst]));
        arb_clear(base);arb_clear(y);arb_clear(p);
    }

    arb_clear(third);for(int m=0;m<MAXT;m++)arb_clear(TC.data()+m);arb_clear(ref);arb_clear(ss);arb_clear(aa);arb_clear(tmp);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();
    return 0;
}
