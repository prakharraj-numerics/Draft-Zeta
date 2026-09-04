#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <mpfr.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

/*
 * Research-only productionization lab for the validated positive-a zeta spine.
 * Math is intentionally unchanged:
 *   zeta(a-2) = 1/(a-3) + 1/3 + a * sum_{m=0}^{T(a)-1} tc_m a^m
 *   tc_m = c_m + 1/3^(m+2)
 *
 * This file compares:
 *   - MPFR 384-bit FMA Horner
 *   - FLINT ARF 384-bit FMA Horner
 *   - per-input Horner vs stage-synchronous batch Horner
 *   - scalar scheduler vs AVX-512 scheduler
 *   - exact scheduler vs tiny L1 LUT scheduler
 *
 * Coefficients are generated once at startup at high precision and cached in
 * backend-native 384-bit form. Startup/coefficient generation is excluded from
 * timed hot-path measurements.
 */

static constexpr int WORK_BITS = 384;
static constexpr int COEFF_BITS = 2048;
static constexpr int REF_BITS = 1024;
static constexpr int MAXT = 900;
static constexpr int LUT_SCALE = 16; // 1/16-wide bins; ~3.2 KiB uint16 table
static constexpr int LUT_N = 102 * LUT_SCALE + 1;

static inline int candidate_terms_scalar(double a)
{
    double t;
    if (a < 3.0) t = 10.0 + 10.5*a - 2.14*a*a;
    else if (a <= 10.0) t = 11.0 + 5.0*a;
    else if (a <= 40.0) t = 11.0 + 4.54*a + 0.0485*a*a;
    else t = -37.0 + 6.85*a + 0.0194*a*a;
    int n = (int)std::ceil(t) + 1;
    return std::max(n, 6);
}

__attribute__((target("avx512f")))
static void candidate_terms_avx512(size_t n, const double* a, int* out)
{
    size_t i = 0;
    const __m512d v3 = _mm512_set1_pd(3.0), v10 = _mm512_set1_pd(10.0), v40 = _mm512_set1_pd(40.0);
    const __m512d c10 = _mm512_set1_pd(10.0), c105 = _mm512_set1_pd(10.5), c214 = _mm512_set1_pd(2.14);
    const __m512d c11 = _mm512_set1_pd(11.0), c5 = _mm512_set1_pd(5.0);
    const __m512d c454 = _mm512_set1_pd(4.54), c0485 = _mm512_set1_pd(0.0485);
    const __m512d cm37 = _mm512_set1_pd(-37.0), c685 = _mm512_set1_pd(6.85), c0194 = _mm512_set1_pd(0.0194);
    for (; i + 8 <= n; i += 8) {
        const __m512d x = _mm512_loadu_pd(a + i);
        const __m512d x2 = _mm512_mul_pd(x, x);
        __m512d f0 = _mm512_fnmadd_pd(c214, x2, _mm512_fmadd_pd(c105, x, c10));
        __m512d f1 = _mm512_fmadd_pd(c5, x, c11);
        __m512d f2 = _mm512_fmadd_pd(c0485, x2, _mm512_fmadd_pd(c454, x, c11));
        __m512d f3 = _mm512_fmadd_pd(c0194, x2, _mm512_fmadd_pd(c685, x, cm37));
        const __mmask8 m0 = _mm512_cmp_pd_mask(x, v3, _CMP_LT_OQ);
        const __mmask8 m1 = _mm512_cmp_pd_mask(x, v10, _CMP_LE_OQ);
        const __mmask8 m2 = _mm512_cmp_pd_mask(x, v40, _CMP_LE_OQ);
        __m512d f = f3;
        f = _mm512_mask_mov_pd(f, m2, f2);
        f = _mm512_mask_mov_pd(f, m1, f1);
        f = _mm512_mask_mov_pd(f, m0, f0);
        f = _mm512_roundscale_pd(f, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        __m256i q = _mm512_cvttpd_epi32(f);
        q = _mm256_add_epi32(q, _mm256_set1_epi32(1));
        q = _mm256_max_epi32(q, _mm256_set1_epi32(6));
        _mm256_storeu_si256((__m256i*)(out + i), q);
    }
    for (; i < n; ++i) out[i] = candidate_terms_scalar(a[i]);
}

static uint16_t TERM_LUT[LUT_N];
static void init_term_lut()
{
    for (int b = 0; b < LUT_N; ++b) {
        double lo = (double)b / LUT_SCALE;
        double hi = std::min(102.0, (double)(b + 1) / LUT_SCALE);
        if (lo <= 0.0) lo = std::nextafter(0.0, 1.0);
        int t = std::max(candidate_terms_scalar(lo), candidate_terms_scalar(hi));
        // The inside quadratic is concave and peaks at 10.5/(2*2.14).
        const double vertex = 10.5 / 4.28;
        if (lo < vertex && vertex < hi) t = std::max(t, candidate_terms_scalar(vertex));
        TERM_LUT[b] = (uint16_t)t;
    }
}
static inline int candidate_terms_lut(double a)
{
    int b = (int)(a * LUT_SCALE);
    if (b < 0) b = 0; if (b >= LUT_N) b = LUT_N - 1;
    return TERM_LUT[b];
}

static uint64_t ordkey(double x){ union{double d;uint64_t u;}v{x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulpdist(double a,double b){ if(std::isnan(a)||std::isnan(b)) return UINT64_MAX; auto x=ordkey(a),y=ordkey(b); return x>y?x-y:y-x; }

struct CoeffCache {
    std::vector<arf_struct> ca;
    std::vector<__mpfr_struct> cm;
    arf_struct third_a;
    __mpfr_struct third_m;
    CoeffCache(): ca(MAXT), cm(MAXT) {
        arb_poly_t s,z; arb_poly_init(s); arb_poly_init(z);
        arb_t one, tc, p3, inv; arb_init(one);arb_init(tc);arb_init(p3);arb_init(inv);
        arb_one(one); arb_poly_set_coeff_si(s,0,-2); arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"[setup] generating %d corrected coefficients at %d bits...\n",MAXT,COEFF_BITS);
        arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);
        arb_set_ui(p3,9);
        for(int m=0;m<MAXT;m++){
            arf_init(ca.data()+m); mpfr_init2(cm.data()+m,WORK_BITS);
            arb_poly_get_coeff_arb(tc,z,m+1); arb_inv(inv,p3,COEFF_BITS); arb_add(tc,tc,inv,COEFF_BITS);
            arf_set_round(ca.data()+m,arb_midref(tc),WORK_BITS,ARF_RND_NEAR);
            arf_get_mpfr(cm.data()+m,ca.data()+m,MPFR_RNDN);
            arb_mul_ui(p3,p3,3,COEFF_BITS);
        }
        arf_init(&third_a); arf_set_ui(&third_a,1); arf_div_ui(&third_a,&third_a,3,WORK_BITS,ARF_RND_NEAR);
        mpfr_init2(&third_m,WORK_BITS); mpfr_set_ui(&third_m,1,MPFR_RNDN); mpfr_div_ui(&third_m,&third_m,3,MPFR_RNDN);
        arb_clear(inv);arb_clear(p3);arb_clear(tc);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);
    }
    ~CoeffCache(){ for(int i=0;i<MAXT;i++){arf_clear(ca.data()+i);mpfr_clear(cm.data()+i);} arf_clear(&third_a);mpfr_clear(&third_m); }
};

struct PreparedBatch {
    size_t n;
    std::vector<double> ad;
    std::vector<int> T;
    std::vector<int> order; // exact-T descending
    std::vector<int> count; // count[T]
    std::vector<arf_struct> aa;
    std::vector<__mpfr_struct> am;
    PreparedBatch(const std::vector<double>& x, bool use_lut=false):n(x.size()),ad(x),T(n),order(n),count(MAXT+1,0),aa(n),am(n){
        if (!use_lut && __builtin_cpu_supports("avx512f")) candidate_terms_avx512(n,ad.data(),T.data());
        else for(size_t i=0;i<n;i++)T[i]=use_lut?candidate_terms_lut(ad[i]):candidate_terms_scalar(ad[i]);
        for(size_t i=0;i<n;i++){ if(T[i]<=0||T[i]>MAXT){std::fprintf(stderr,"bad T=%d a=%.17g\n",T[i],ad[i]);std::abort();} count[T[i]]++; order[i]=(int)i; arf_init(aa.data()+i);arf_set_d(aa.data()+i,ad[i]);mpfr_init2(am.data()+i,64);mpfr_set_d(am.data()+i,ad[i],MPFR_RNDN); }
        std::stable_sort(order.begin(),order.end(),[&](int i,int j){return T[i]>T[j];});
    }
    ~PreparedBatch(){for(size_t i=0;i<n;i++){arf_clear(aa.data()+i);mpfr_clear(am.data()+i);}}
};

static void eval_mpfr_scalar(const PreparedBatch& b,const CoeffCache& c,std::vector<double>& out)
{
    out.resize(b.n); mpfr_t p,base;mpfr_init2(p,WORK_BITS);mpfr_init2(base,WORK_BITS);
    for(size_t i=0;i<b.n;i++){
        int T=b.T[i]; mpfr_set(p,c.cm.data()+T-1,MPFR_RNDN);
        for(int m=T-2;m>=0;m--) mpfr_fma(p,p,b.am.data()+i,c.cm.data()+m,MPFR_RNDN);
        mpfr_mul(p,p,b.am.data()+i,MPFR_RNDN);
        mpfr_sub_ui(base,b.am.data()+i,3,MPFR_RNDN); mpfr_ui_div(base,1,base,MPFR_RNDN);
        mpfr_add(p,p,base,MPFR_RNDN); mpfr_add(p,p,&c.third_m,MPFR_RNDN);
        out[i]=mpfr_get_d(p,MPFR_RNDN);
    }
    mpfr_clear(base);mpfr_clear(p);
}

static void eval_arf_scalar(const PreparedBatch& b,const CoeffCache& c,std::vector<double>& out)
{
    out.resize(b.n); arf_t p,base;arf_init(p);arf_init(base);
    for(size_t i=0;i<b.n;i++){
        int T=b.T[i]; arf_set(p,c.ca.data()+T-1);
        for(int m=T-2;m>=0;m--) arf_fma(p,p,b.aa.data()+i,c.ca.data()+m,WORK_BITS,ARF_RND_NEAR);
        arf_mul(p,p,b.aa.data()+i,WORK_BITS,ARF_RND_NEAR);
        arf_sub_ui(base,b.aa.data()+i,3,WORK_BITS,ARF_RND_NEAR); arf_inv(base,base,WORK_BITS,ARF_RND_NEAR);
        arf_add(p,p,base,WORK_BITS,ARF_RND_NEAR); arf_add(p,p,&c.third_a,WORK_BITS,ARF_RND_NEAR);
        out[i]=arf_get_d(p,ARF_RND_NEAR);
    }
    arf_clear(base);arf_clear(p);
}

static void eval_mpfr_stage(const PreparedBatch& b,const CoeffCache& c,std::vector<double>& out)
{
    out.resize(b.n); if(!b.n)return;
    std::vector<__mpfr_struct> p(b.n); for(size_t k=0;k<b.n;k++)mpfr_init2(p.data()+k,WORK_BITS);
    size_t active=0,old=0; int maxT=0;for(int t=MAXT;t>=1;t--)if(b.count[t]){maxT=t;break;}
    for(int m=maxT-1;m>=0;m--){
        old=active; active += b.count[m+1];
        for(size_t k=0;k<old;k++){int i=b.order[k];mpfr_fma(p.data()+k,p.data()+k,b.am.data()+i,c.cm.data()+m,MPFR_RNDN);}
        for(size_t k=old;k<active;k++)mpfr_set(p.data()+k,c.cm.data()+m,MPFR_RNDN);
    }
    mpfr_t base;mpfr_init2(base,WORK_BITS);
    for(size_t k=0;k<b.n;k++){int i=b.order[k];mpfr_mul(p.data()+k,p.data()+k,b.am.data()+i,MPFR_RNDN);mpfr_sub_ui(base,b.am.data()+i,3,MPFR_RNDN);mpfr_ui_div(base,1,base,MPFR_RNDN);mpfr_add(p.data()+k,p.data()+k,base,MPFR_RNDN);mpfr_add(p.data()+k,p.data()+k,&c.third_m,MPFR_RNDN);out[i]=mpfr_get_d(p.data()+k,MPFR_RNDN);}
    mpfr_clear(base);for(size_t k=0;k<b.n;k++)mpfr_clear(p.data()+k);
}

static void eval_arf_stage(const PreparedBatch& b,const CoeffCache& c,std::vector<double>& out)
{
    out.resize(b.n); if(!b.n)return;
    std::vector<arf_struct> p(b.n);for(size_t k=0;k<b.n;k++)arf_init(p.data()+k);
    size_t active=0,old=0;int maxT=0;for(int t=MAXT;t>=1;t--)if(b.count[t]){maxT=t;break;}
    for(int m=maxT-1;m>=0;m--){
        old=active;active+=b.count[m+1];
        for(size_t k=0;k<old;k++){int i=b.order[k];arf_fma(p.data()+k,p.data()+k,b.aa.data()+i,c.ca.data()+m,WORK_BITS,ARF_RND_NEAR);}
        for(size_t k=old;k<active;k++)arf_set(p.data()+k,c.ca.data()+m);
    }
    arf_t base;arf_init(base);
    for(size_t k=0;k<b.n;k++){int i=b.order[k];arf_mul(p.data()+k,p.data()+k,b.aa.data()+i,WORK_BITS,ARF_RND_NEAR);arf_sub_ui(base,b.aa.data()+i,3,WORK_BITS,ARF_RND_NEAR);arf_inv(base,base,WORK_BITS,ARF_RND_NEAR);arf_add(p.data()+k,p.data()+k,base,WORK_BITS,ARF_RND_NEAR);arf_add(p.data()+k,p.data()+k,&c.third_a,WORK_BITS,ARF_RND_NEAR);out[i]=arf_get_d(p.data()+k,ARF_RND_NEAR);}
    arf_clear(base);for(size_t k=0;k<b.n;k++)arf_clear(p.data()+k);
}

static std::vector<double> make_inputs(size_t n,uint64_t seed)
{
    std::vector<double>a(n);std::mt19937_64 g(seed);std::uniform_real_distribution<double> coin(0,1),di(1e-12,3.0),doo(3.0,102.0);
    for(size_t i=0;i<n;i++){double x=(coin(g)<0.5)?di(g):doo(g);if(x==3.0)x=std::nextafter(3.0,INFINITY);a[i]=x;}
    return a;
}

static std::vector<double> references(const std::vector<double>& a)
{
    std::vector<double> r(a.size());arb_t aa,s,z;arb_init(aa);arb_init(s);arb_init(z);
    for(size_t i=0;i<a.size();i++){arb_set_d(aa,a[i]);arb_sub_ui(s,aa,2,REF_BITS);arb_zeta(z,s,REF_BITS);r[i]=arf_get_d(arb_midref(z),ARF_RND_NEAR);}arb_clear(z);arb_clear(s);arb_clear(aa);return r;
}

static void accuracy_report(const char*name,const std::vector<double>&y,const std::vector<double>&r,const std::vector<double>&a,const PreparedBatch&b)
{
    uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;for(size_t i=0;i<y.size();i++){auto e=ulpdist(y[i],r[i]);if(e==UINT64_MAX){bad++;continue;}if(e>mx){mx=e;w=(int)i;}gt1+=e>1;gt2+=e>2;}std::printf("ACC %-12s MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu NONFINITE=%d WORST_A=%.17g T=%d\n",name,(unsigned long long)mx,gt1,y.size(),gt2,y.size(),bad,a[w],b.T[w]);
}

template<class F> static double bench_ns(F&& fn,size_t n,int reps)
{
    fn();auto best=1e300;for(int r=0;r<reps;r++){auto t0=std::chrono::steady_clock::now();fn();auto t1=std::chrono::steady_clock::now();double ns=std::chrono::duration<double,std::nano>(t1-t0).count()/n;if(ns<best)best=ns;}return best;
}

int main()
{
    init_term_lut();CoeffCache cache;
    // Scheduler parity and cost.
    {
        auto a=make_inputs(100000,0x5343484544554c45ULL);std::vector<int>s(a.size()),v(a.size()),l(a.size());
        for(size_t i=0;i<a.size();i++)s[i]=candidate_terms_scalar(a[i]);
        if(__builtin_cpu_supports("avx512f"))candidate_terms_avx512(a.size(),a.data(),v.data());else v=s;
        int mismatch=0,ltextra=0,maxextra=0;for(size_t i=0;i<a.size();i++){l[i]=candidate_terms_lut(a[i]);mismatch+=v[i]!=s[i];int e=l[i]-s[i];ltextra+=e;maxextra=std::max(maxextra,e);}std::printf("SCHED AVX512=%d MISMATCH=%d LUT_MEAN_EXTRA=%.6f LUT_MAX_EXTRA=%d\n",__builtin_cpu_supports("avx512f")?1:0,mismatch,(double)ltextra/a.size(),maxextra);
        auto bs=bench_ns([&](){volatile long long z=0;for(double x:a)z+=candidate_terms_scalar(x);},a.size(),7);
        auto bv=bench_ns([&](){candidate_terms_avx512(a.size(),a.data(),v.data());},a.size(),7);
        auto bl=bench_ns([&](){volatile long long z=0;for(double x:a)z+=candidate_terms_lut(x);},a.size(),7);
        std::printf("SCHED_NS_PER_ELEM scalar=%.3f avx512=%.3f lut16=%.3f\n",bs,bv,bl);
    }

    // Strict 5000 random accuracy for every backend/layout.
    auto A=make_inputs(5000,0x4f50544c41423530ULL);auto R=references(A);PreparedBatch B(A,false);std::vector<double> y;
    eval_mpfr_scalar(B,cache,y);accuracy_report("mpfr_scalar",y,R,A,B);
    eval_mpfr_stage(B,cache,y);accuracy_report("mpfr_stage",y,R,A,B);
    eval_arf_scalar(B,cache,y);accuracy_report("arf_scalar",y,R,A,B);
    eval_arf_stage(B,cache,y);accuracy_report("arf_stage",y,R,A,B);
    PreparedBatch BL(A,true);eval_arf_stage(BL,cache,y);accuracy_report("arf_stage_lut",y,R,A,BL);

    const size_t sizes[]={100,400,1000,3000,6000,15000};
    for(size_t n:sizes){
        auto x=make_inputs(n,0x4241544348000000ULL+n);PreparedBatch b(x,false);PreparedBatch bl(x,true);std::vector<double> o;
        int reps=n<=400?7:n<=3000?5:3;
        double mps=bench_ns([&](){eval_mpfr_scalar(b,cache,o);},n,reps);
        double mpb=bench_ns([&](){eval_mpfr_stage(b,cache,o);},n,reps);
        double ars=bench_ns([&](){eval_arf_scalar(b,cache,o);},n,reps);
        double arb=bench_ns([&](){eval_arf_stage(b,cache,o);},n,reps);
        double arl=bench_ns([&](){eval_arf_stage(bl,cache,o);},n,reps);
        long long tsum=std::accumulate(b.T.begin(),b.T.end(),0LL);
        std::printf("BENCH N=%zu MEAN_T=%.3f MPFR_SCALAR=%.3f MPFR_STAGE=%.3f ARF_SCALAR=%.3f ARF_STAGE=%.3f ARF_STAGE_LUT=%.3f ns/elem\n",n,(double)tsum/n,mps,mpb,ars,arb,arl);
    }
    flint_cleanup();return 0;
}
