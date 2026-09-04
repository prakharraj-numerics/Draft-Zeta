#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#pragma STDC FP_CONTRACT OFF

// Exact-recenter candidate for the current formula only:
//   zeta(x) = 1/(x-1) + G(a), a=x+2
//   G(a)    = 1/3 + a * sum_{m>=0} src[m] a^m.
//
// Offline/setup, for each cached center A, the SAME finite source polynomial is
// transformed by the exact substitution a=A+h using Horner polynomial shifting.
// Runtime then evaluates only the short local polynomial in |h|<=1/4.
// No functional equation, Hasse series, or replacement zeta identity is used.

static constexpr int GEN_BITS=8192;
static constexpr int LOCAL_BITS=640;
static constexpr int REF_BITS=2048;
static constexpr int MAXT=900;
static constexpr int DMAX=64;
static constexpr int NBINS=198;          // [3,102] in width-0.5 bins
static constexpr double BINW=0.5;
static constexpr double A0=3.25;         // centers 3.25,3.75,...,101.75

static uint64_t ordkey(double x){
    uint64_t u=std::bit_cast<uint64_t>(x);
    return (u>>63)?~u:(u|0x8000000000000000ULL);
}
static uint64_t ulpdist(double a,double b){
    if(std::isnan(a)||std::isnan(b)) return UINT64_MAX;
    uint64_t x=ordkey(a),y=ordkey(b); return x>y?x-y:y-x;
}

struct SourceTables {
    std::vector<arf_struct> src;
    arf_t third;
    arb_t aa,ref;

    SourceTables():src(MAXT){
        arf_init(third); arb_init(aa); arb_init(ref);
        arb_poly_t s,z; arb_poly_init(s); arb_poly_init(z);
        arb_t one,p3,t,inv; arb_init(one);arb_init(p3);arb_init(t);arb_init(inv);
        arb_one(one);
        arb_poly_set_coeff_si(s,0,-2);
        arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"[setup] manufacturing current corrected source coefficients at %d bits\n",GEN_BITS);
        arb_poly_zeta_series(z,s,one,0,MAXT+1,GEN_BITS);
        arb_set_ui(p3,9);
        for(int m=0;m<MAXT;m++){
            arf_init(src.data()+m);
            arb_poly_get_coeff_arb(t,z,m+1);
            arb_inv(inv,p3,GEN_BITS);
            arb_add(t,t,inv,GEN_BITS);
            arf_set_round(src.data()+m,arb_midref(t),GEN_BITS,ARF_RND_NEAR);
            arb_mul_ui(p3,p3,3,GEN_BITS);
        }
        arf_set_ui(third,1);
        arf_div_ui(third,third,3,GEN_BITS,ARF_RND_NEAR);
        arb_clear(inv);arb_clear(t);arb_clear(p3);arb_clear(one);
        arb_poly_clear(z);arb_poly_clear(s);
    }
    ~SourceTables(){
        for(auto &v:src) arf_clear(&v);
        arf_clear(third); arb_clear(ref); arb_clear(aa);
    }
    double reference_x(double x){
        arb_set_d(aa,x); arb_zeta(ref,aa,REF_BITS);
        return arf_get_d(arb_midref(ref),ARF_RND_NEAR);
    }
};

struct LocalTables {
    alignas(64) double hi[DMAX+1][NBINS];
    alignas(64) double lo[DMAX+1][NBINS];

    explicit LocalTables(const SourceTables&s){
        std::fprintf(stderr,"[setup] exact Horner-shift to %d half-unit centers, retaining degrees 0..%d at %d bits\n",
                     NBINS,DMAX,LOCAL_BITS);
        std::array<arf_struct,DMAX+1> d;
        for(auto &v:d) arf_init(&v);
        arf_t A,q,resid; arf_init(A);arf_init(q);arf_init(resid);

        for(int b=0;b<NBINS;b++){
            for(auto &v:d) arf_zero(&v);
            const double center=A0+BINW*b; // exact dyadic
            arf_set_d(A,center);

            // Let g_0=1/3 and g_m=src[m-1] for m>=1.
            // Descending Horner with a=A+h:
            //     Q <- Q*(A+h) + g_m.
            // Keeping only coefficients h^0..h^DMAX is exact for those
            // coefficients of the degree-MAXT source polynomial.
            for(int m=MAXT;m>=0;m--){
                for(int k=DMAX;k>=1;k--)
                    arf_fma(&d[k],&d[k],A,&d[k-1],LOCAL_BITS,ARF_RND_NEAR);
                const arf_struct *gm=(m==0)?s.third:(s.src.data()+m-1);
                arf_fma(&d[0],&d[0],A,gm,LOCAL_BITS,ARF_RND_NEAR);
            }

            for(int k=0;k<=DMAX;k++){
                const double h=arf_get_d(&d[k],ARF_RND_NEAR);
                hi[k][b]=h;
                arf_set_d(q,h);
                arf_sub(resid,&d[k],q,LOCAL_BITS,ARF_RND_NEAR);
                lo[k][b]=arf_get_d(resid,ARF_RND_NEAR);
            }
            if((b%32)==0 || b==NBINS-1)
                std::fprintf(stderr,"[shift] center %d/%d A=%.2f\n",b+1,NBINS,center);
        }
        arf_clear(resid);arf_clear(q);arf_clear(A);
        for(auto &v:d) arf_clear(&v);
    }
};

__attribute__((always_inline,target("avx512f,avx512dq,fma")))
static inline void two_sum_v(__m512d a,__m512d b,__m512d&s,__m512d&e){
    s=_mm512_add_pd(a,b);
    __m512d bb=_mm512_sub_pd(s,a);
    e=_mm512_add_pd(_mm512_sub_pd(a,_mm512_sub_pd(s,bb)),_mm512_sub_pd(b,bb));
}

__attribute__((always_inline,target("avx512f,avx512dq,fma")))
static inline void dd_horner_step(__m512d&ph,__m512d&pl,__m512d h,__m512d ch,__m512d cl){
    const __m512d mh=_mm512_mul_pd(ph,h);
    __m512d me=_mm512_fmadd_pd(ph,h,_mm512_sub_pd(_mm512_setzero_pd(),mh));
    me=_mm512_fmadd_pd(pl,h,me);
    __m512d s,e; two_sum_v(mh,ch,s,e);
    e=_mm512_add_pd(e,_mm512_add_pd(me,cl));
    two_sum_v(s,e,ph,pl);
}

static inline int bin_scalar(double x){
    int b=(int)((x-1.0)*2.0);
    if(b<0)b=0; if(b>=NBINS)b=NBINS-1; return b;
}

__attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
static void local_gather_f64(const LocalTables&lt,const double*x,size_t n,double*out,int degree){
    const __m512d one=_mm512_set1_pd(1.0), two=_mm512_set1_pd(2.0);
    const __m512d c125=_mm512_set1_pd(1.25), half=_mm512_set1_pd(0.5);
    const __m256i z=_mm256_setzero_si256(), vmax=_mm256_set1_epi32(NBINS-1);
    for(size_t i=0;i<n;i+=8){
        const int rem=(int)std::min<size_t>(8,n-i);
        const __mmask8 vm=(__mmask8)((1u<<rem)-1u);
        const __m512d vx=_mm512_maskz_loadu_pd(vm,x+i);
        __m512d t=_mm512_mul_pd(_mm512_sub_pd(vx,one),two);
        __m256i bi=_mm512_cvttpd_epi32(t);
        bi=_mm256_max_epi32(z,_mm256_min_epi32(bi,vmax));
        __m512d bc=_mm512_cvtepi32_pd(bi);
        __m512d xc=_mm512_fmadd_pd(bc,half,c125); // A-2 = 1.25 + b/2
        __m512d h=_mm512_sub_pd(vx,xc);            // exact by Sterbenz in each local bin
        __m512d p=_mm512_i32gather_pd(bi,lt.hi[degree],8);
        for(int k=degree-1;k>=0;k--){
            __m512d c=_mm512_i32gather_pd(bi,lt.hi[k],8);
            p=_mm512_fmadd_pd(p,h,c);
        }
        __m512d r=_mm512_div_pd(one,_mm512_sub_pd(vx,one));
        __m512d y=_mm512_add_pd(r,p);
        _mm512_mask_storeu_pd(out+i,vm,y);
    }
}

__attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
static void local_gather_dd(const LocalTables&lt,const double*x,size_t n,double*out,int degree){
    const __m512d one=_mm512_set1_pd(1.0), two=_mm512_set1_pd(2.0);
    const __m512d c125=_mm512_set1_pd(1.25), half=_mm512_set1_pd(0.5), zero=_mm512_setzero_pd();
    const __m256i zi=_mm256_setzero_si256(), vmax=_mm256_set1_epi32(NBINS-1);
    for(size_t i=0;i<n;i+=8){
        const int rem=(int)std::min<size_t>(8,n-i);
        const __mmask8 vm=(__mmask8)((1u<<rem)-1u);
        const __m512d vx=_mm512_maskz_loadu_pd(vm,x+i);
        __m512d t=_mm512_mul_pd(_mm512_sub_pd(vx,one),two);
        __m256i bi=_mm512_cvttpd_epi32(t);
        bi=_mm256_max_epi32(zi,_mm256_min_epi32(bi,vmax));
        __m512d bc=_mm512_cvtepi32_pd(bi);
        __m512d xc=_mm512_fmadd_pd(bc,half,c125);
        __m512d h=_mm512_sub_pd(vx,xc);
        __m512d ph=_mm512_i32gather_pd(bi,lt.hi[degree],8);
        __m512d pl=_mm512_i32gather_pd(bi,lt.lo[degree],8);
        for(int k=degree-1;k>=0;k--){
            __m512d ch=_mm512_i32gather_pd(bi,lt.hi[k],8);
            __m512d cl=_mm512_i32gather_pd(bi,lt.lo[k],8);
            dd_horner_step(ph,pl,h,ch,cl);
        }
        const __m512d den=_mm512_sub_pd(vx,one);
        const __m512d rh=_mm512_div_pd(one,den);
        const __m512d re=_mm512_fmadd_pd(_mm512_sub_pd(zero,rh),den,one);
        const __m512d rl=_mm512_mul_pd(re,rh); // one Newton residual correction
        __m512d sh,se; two_sum_v(rh,ph,sh,se);
        se=_mm512_add_pd(se,_mm512_add_pd(rl,pl));
        __m512d yh,yl; two_sum_v(sh,se,yh,yl);
        __m512d y=_mm512_add_pd(yh,yl);
        _mm512_mask_storeu_pd(out+i,vm,y);
    }
}

struct GroupScratch {
    std::vector<double> sx,sy;
    std::vector<size_t> sidx;
    std::array<size_t,NBINS+1> beg{};
    void ensure(size_t n){sx.resize(n);sy.resize(n);sidx.resize(n);}
};

__attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
static void local_grouped_dd(const LocalTables&lt,const double*x,size_t n,double*out,int degree,GroupScratch&s){
    s.ensure(n);
    std::array<size_t,NBINS> cnt{};
    for(size_t i=0;i<n;i++) cnt[bin_scalar(x[i])]++;
    s.beg[0]=0; for(int b=0;b<NBINS;b++)s.beg[b+1]=s.beg[b]+cnt[b];
    std::array<size_t,NBINS> pos{}; for(int b=0;b<NBINS;b++)pos[b]=s.beg[b];
    for(size_t i=0;i<n;i++){int b=bin_scalar(x[i]);size_t p=pos[b]++;s.sx[p]=x[i];s.sidx[p]=i;}

    const __m512d one=_mm512_set1_pd(1.0),zero=_mm512_setzero_pd();
    for(int b=0;b<NBINS;b++){
        const size_t lo0=s.beg[b],hi0=s.beg[b+1]; if(lo0==hi0)continue;
        const __m512d xc=_mm512_set1_pd(1.25+0.5*b);
        for(size_t p0=lo0;p0<hi0;p0+=8){
            int rem=(int)std::min<size_t>(8,hi0-p0);__mmask8 vm=(__mmask8)((1u<<rem)-1u);
            __m512d vx=_mm512_maskz_loadu_pd(vm,s.sx.data()+p0);
            __m512d h=_mm512_sub_pd(vx,xc);
            __m512d ph=_mm512_set1_pd(lt.hi[degree][b]);
            __m512d pl=_mm512_set1_pd(lt.lo[degree][b]);
            for(int k=degree-1;k>=0;k--)
                dd_horner_step(ph,pl,h,_mm512_set1_pd(lt.hi[k][b]),_mm512_set1_pd(lt.lo[k][b]));
            __m512d den=_mm512_sub_pd(vx,one);
            __m512d rh=_mm512_div_pd(one,den);
            __m512d re=_mm512_fmadd_pd(_mm512_sub_pd(zero,rh),den,one);
            __m512d rl=_mm512_mul_pd(re,rh);
            __m512d sh,se;two_sum_v(rh,ph,sh,se);se=_mm512_add_pd(se,_mm512_add_pd(rl,pl));
            __m512d yh,yl;two_sum_v(sh,se,yh,yl);
            _mm512_mask_storeu_pd(s.sy.data()+p0,vm,_mm512_add_pd(yh,yl));
        }
    }
    for(size_t p=0;p<n;p++)out[s.sidx[p]]=s.sy[p];
}

static std::vector<double> make_inputs(size_t n){
    std::mt19937_64 rng(0x91e10da5c79e7b1dULL);
    std::uniform_real_distribution<double> ud(1.0,100.0);
    std::vector<double>x(n);
    for(size_t i=0;i<n;i++){
        double q=ud(rng);
        if(q<=1.0)q=std::nextafter(1.0,INFINITY);
        if(q>=100.0)q=std::nextafter(100.0,0.0);
        x[i]=q;
    }
    const double hard[]={std::nextafter(1.0,INFINITY),1.000000000001,1.000001,1.01,1.1,1.25,1.5,1.75,
        2.0,2.9,3.0,4.0,8.0,10.0,20.0,38.0,50.0,75.0,79.764554025359956,98.0,99.0,
        std::nextafter(100.0,0.0)};
    for(size_t i=0;i<std::size(hard)&&i<n;i++)x[i]=hard[i];
    return x;
}

struct Acc {uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;};
static Acc assess(const std::vector<double>&x,const std::vector<double>&y,const std::vector<double>&ref){
    Acc a;
    for(size_t i=0;i<x.size();i++){
        uint64_t u=ulpdist(y[i],ref[i]);
        if(u==UINT64_MAX){a.bad++;continue;}
        if(u>a.mx){a.mx=u;a.w=(int)i;}
        a.gt1+=u>1;a.gt2+=u>2;
    }
    return a;
}

template<class F>
static double median_ns(size_t n,int reps,F&&f){
    std::vector<double>v;v.reserve(9);
    for(int trial=0;trial<9;trial++){
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<reps;r++)f();
        auto t1=std::chrono::steady_clock::now();
        v.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/(double)(reps*n));
    }
    std::sort(v.begin(),v.end());return v[v.size()/2];
}

template<class F>
static double mean_ns(size_t n,int reps,F&&f){
    auto t0=std::chrono::steady_clock::now();
    for(int r=0;r<reps;r++)f();
    auto t1=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::nano>(t1-t0).count()/(double)(reps*n);
}

int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||
       !__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma")){
        std::fprintf(stderr,"AVX512F/DQ/VL+FMA required\n");return 2;
    }
    SourceTables src;
    LocalTables lt(src);
    std::printf("LOCAL_TABLE_BYTES=%zu CENTER_SPACING=%.2f WORST_ABS_H=%.2f\n",sizeof(lt),BINW,BINW/2);

    const size_t NACC=5000;
    auto x=make_inputs(NACC);
    std::vector<double>ref(NACC),y(NACC);
    std::fprintf(stderr,"[accuracy] building %zu high-precision references once\n",NACC);
    for(size_t i=0;i<NACC;i++)ref[i]=src.reference_x(x[i]);

    const int degrees[]={12,16,20,24,28,32,36,40,48,56,64};
    int best_f64=-1,best_dd=-1;
    for(int d:degrees){
        local_gather_f64(lt,x.data(),NACC,y.data(),d);
        Acc a=assess(x,y,ref);
        std::printf("ACC_F64 D=%d MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu BAD=%d WORST_X=%.17g OUR=%.17g REF=%.17g\n",
            d,(unsigned long long)a.mx,a.gt1,NACC,a.gt2,NACC,a.bad,x[a.w],y[a.w],ref[a.w]);
        if(best_f64<0 && a.gt1==0 && a.bad==0)best_f64=d;

        local_gather_dd(lt,x.data(),NACC,y.data(),d);
        a=assess(x,y,ref);
        std::printf("ACC_DD D=%d MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu BAD=%d WORST_X=%.17g OUR=%.17g REF=%.17g\n",
            d,(unsigned long long)a.mx,a.gt1,NACC,a.gt2,NACC,a.bad,x[a.w],y[a.w],ref[a.w]);
        if(best_dd<0 && a.gt1==0 && a.bad==0)best_dd=d;
    }
    std::printf("BEST_F64_DEGREE=%d BEST_DD_DEGREE=%d\n",best_f64,best_dd);
    if(best_dd<0){std::puts("LOCAL_CENTER_ACCURACY_GATE=FAIL");flint_cleanup();return 3;}
    std::puts("LOCAL_CENTER_ACCURACY_GATE=PASS");

    // Verify grouped and gather DD are numerically equivalent enough on the full gate set.
    std::vector<double>yg(NACC);GroupScratch gs;
    local_grouped_dd(lt,x.data(),NACC,yg.data(),best_dd,gs);
    Acc ag=assess(x,yg,ref);
    std::printf("ACC_GROUPED_DD D=%d MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu BAD=%d\n",
        best_dd,(unsigned long long)ag.mx,ag.gt1,NACC,ag.gt2,NACC,ag.bad);
    if(ag.gt1||ag.bad){std::puts("GROUPED_ACCURACY_GATE=FAIL");flint_cleanup();return 4;}
    std::puts("GROUPED_ACCURACY_GATE=PASS");

    volatile double sink=0.0;
    const size_t sizes[]={100,500,1000,3000};
    std::vector<double>out(NACC);
    GroupScratch bgs;
    for(size_t n:sizes){
        int reps=std::max(20,(int)(240000/n));
        local_gather_dd(lt,x.data(),n,out.data(),best_dd);
        local_grouped_dd(lt,x.data(),n,out.data(),best_dd,bgs);
        if(best_f64>0)local_gather_f64(lt,x.data(),n,out.data(),best_f64);

        const double gmean=mean_ns(n,reps,[&](){local_gather_dd(lt,x.data(),n,out.data(),best_dd);sink+=out[0];});
        const double gmed=median_ns(n,std::max(5,reps/3),[&](){local_gather_dd(lt,x.data(),n,out.data(),best_dd);sink+=out[0];});
        const double smean=mean_ns(n,reps,[&](){local_grouped_dd(lt,x.data(),n,out.data(),best_dd,bgs);sink+=out[0];});
        const double smed=median_ns(n,std::max(5,reps/3),[&](){local_grouped_dd(lt,x.data(),n,out.data(),best_dd,bgs);sink+=out[0];});
        double fmean=-1.0,fmed=-1.0;
        if(best_f64>0){
            fmean=mean_ns(n,reps,[&](){local_gather_f64(lt,x.data(),n,out.data(),best_f64);sink+=out[0];});
            fmed=median_ns(n,std::max(5,reps/3),[&](){local_gather_f64(lt,x.data(),n,out.data(),best_f64);sink+=out[0];});
        }
        std::printf("BATCH=%zu DD_DEG=%d GATHER_DD_MEAN_NS=%.6f GATHER_DD_MEDIAN_NS=%.6f GROUP_DD_MEAN_NS=%.6f GROUP_DD_MEDIAN_NS=%.6f F64_DEG=%d GATHER_F64_MEAN_NS=%.6f GATHER_F64_MEDIAN_NS=%.6f\n",
            n,best_dd,gmean,gmed,smean,smed,best_f64,fmean,fmed);
    }
    std::printf("SINK=%.17g\n",(double)sink);
    flint_cleanup();return 0;
}
