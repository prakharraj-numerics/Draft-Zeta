#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static constexpr int WORK_BITS = 368;
static constexpr int GEN_BITS  = 8192;   // offline/startup manufacture only
static constexpr int REF_BITS  = 2048;
static constexpr int MAXT      = 900;

static inline int terms_scalar(double a)
{
    double t;
    if (a < 3.0)       t = 10.0 + 10.5*a - 2.14*a*a;
    else if (a <= 10.) t = 11.0 + 5.0*a;
    else if (a <= 40.) t = 11.0 + 4.54*a + 0.0485*a*a;
    else               t = -37.0 + 6.85*a + 0.0194*a*a;
    return std::max(6,(int)std::ceil(t)+1);
}

__attribute__((target("avx512f,avx2,fma")))
static void terms_avx512(size_t n,const double *a,int *out)
{
    size_t i=0;
    const __m512d v3=_mm512_set1_pd(3.0),v10=_mm512_set1_pd(10.0),v40=_mm512_set1_pd(40.0);
    const __m512d c10=_mm512_set1_pd(10.0),c105=_mm512_set1_pd(10.5),c214=_mm512_set1_pd(2.14);
    const __m512d c11=_mm512_set1_pd(11.0),c5=_mm512_set1_pd(5.0),c454=_mm512_set1_pd(4.54),c0485=_mm512_set1_pd(.0485);
    const __m512d cm37=_mm512_set1_pd(-37.0),c685=_mm512_set1_pd(6.85),c0194=_mm512_set1_pd(.0194);
    for(;i+8<=n;i+=8){
        __m512d x=_mm512_loadu_pd(a+i),x2=_mm512_mul_pd(x,x);
        __m512d f0=_mm512_fnmadd_pd(c214,x2,_mm512_fmadd_pd(c105,x,c10));
        __m512d f1=_mm512_fmadd_pd(c5,x,c11);
        __m512d f2=_mm512_fmadd_pd(c0485,x2,_mm512_fmadd_pd(c454,x,c11));
        __m512d f3=_mm512_fmadd_pd(c0194,x2,_mm512_fmadd_pd(c685,x,cm37));
        __mmask8 m0=_mm512_cmp_pd_mask(x,v3,_CMP_LT_OQ),m1=_mm512_cmp_pd_mask(x,v10,_CMP_LE_OQ),m2=_mm512_cmp_pd_mask(x,v40,_CMP_LE_OQ);
        __m512d f=f3; f=_mm512_mask_mov_pd(f,m2,f2); f=_mm512_mask_mov_pd(f,m1,f1); f=_mm512_mask_mov_pd(f,m0,f0);
        f=_mm512_roundscale_pd(f,_MM_FROUND_TO_POS_INF|_MM_FROUND_NO_EXC);
        __m256i q=_mm512_cvttpd_epi32(f); q=_mm256_add_epi32(q,_mm256_set1_epi32(1)); q=_mm256_max_epi32(q,_mm256_set1_epi32(6));
        _mm256_storeu_si256((__m256i*)(out+i),q);
    }
    for(;i<n;i++) out[i]=terms_scalar(a[i]);
}

static uint64_t ordkey(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}
static uint64_t ulpdist(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;auto x=ordkey(a),y=ordkey(b);return x>y?x-y:y-x;}

struct CoeffCache {
    std::vector<arf_struct> c;
    arf_struct third;
    CoeffCache():c(MAXT){
        arb_poly_t s,z; arb_poly_init(s);arb_poly_init(z);
        arb_t one,tc,p3,inv; arb_init(one);arb_init(tc);arb_init(p3);arb_init(inv);
        arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"[setup] manufacture %d corrected coefficients at %d bits, cache at %d bits...\n",MAXT,GEN_BITS,WORK_BITS);
        arb_poly_zeta_series(z,s,one,0,MAXT+1,GEN_BITS);
        arb_set_ui(p3,9);
        for(int m=0;m<MAXT;m++){
            arf_init(c.data()+m);
            arb_poly_get_coeff_arb(tc,z,m+1);arb_inv(inv,p3,GEN_BITS);arb_add(tc,tc,inv,GEN_BITS);
            arf_set_round(c.data()+m,arb_midref(tc),WORK_BITS,ARF_RND_NEAR);
            arb_mul_ui(p3,p3,3,GEN_BITS);
        }
        arf_init(&third);arf_set_ui(&third,1);arf_div_ui(&third,&third,3,WORK_BITS,ARF_RND_NEAR);
        arb_clear(inv);arb_clear(p3);arb_clear(tc);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);
    }
    ~CoeffCache(){for(int m=0;m<MAXT;m++)arf_clear(c.data()+m);arf_clear(&third);}
};

struct Batch {
    size_t n; std::vector<double>a; std::vector<int>T,order,count; std::vector<arf_struct> aa;
    explicit Batch(const std::vector<double>&x):n(x.size()),a(x),T(n),order(n),count(MAXT+1,0),aa(n){
        if(__builtin_cpu_supports("avx512f")) terms_avx512(n,a.data(),T.data()); else for(size_t i=0;i<n;i++)T[i]=terms_scalar(a[i]);
        for(size_t i=0;i<n;i++){if(T[i]<1||T[i]>MAXT)std::abort();count[T[i]]++;order[i]=(int)i;arf_init(aa.data()+i);arf_set_d(aa.data()+i,a[i]);}
        std::stable_sort(order.begin(),order.end(),[&](int i,int j){return T[i]>T[j];});
    }
    ~Batch(){for(size_t i=0;i<n;i++)arf_clear(aa.data()+i);}
};

static void eval_scalar(const Batch&b,const CoeffCache&cc,std::vector<double>&out)
{
    out.resize(b.n);arf_t p,base;arf_init(p);arf_init(base);
    for(size_t i=0;i<b.n;i++){
        int T=b.T[i];arf_set(p,cc.c.data()+T-1);
        for(int m=T-2;m>=0;m--)arf_fma(p,p,b.aa.data()+i,cc.c.data()+m,WORK_BITS,ARF_RND_NEAR);
        arf_mul(p,p,b.aa.data()+i,WORK_BITS,ARF_RND_NEAR);arf_sub_ui(base,b.aa.data()+i,3,WORK_BITS,ARF_RND_NEAR);arf_ui_div(base,1,base,WORK_BITS,ARF_RND_NEAR);
        arf_add(p,p,base,WORK_BITS,ARF_RND_NEAR);arf_add(p,p,&cc.third,WORK_BITS,ARF_RND_NEAR);out[i]=arf_get_d(p,ARF_RND_NEAR);
    }
    arf_clear(base);arf_clear(p);
}

static void eval_stage_batch(const Batch&b,const CoeffCache&cc,std::vector<double>&out)
{
    out.resize(b.n);if(!b.n)return;std::vector<arf_struct>p(b.n);for(size_t k=0;k<b.n;k++)arf_init(p.data()+k);
    size_t active=0,old=0;int maxT=0;for(int t=MAXT;t>=1;t--)if(b.count[t]){maxT=t;break;}
    for(int m=maxT-1;m>=0;m--){old=active;active+=b.count[m+1];for(size_t k=0;k<old;k++){int i=b.order[k];arf_fma(p.data()+k,p.data()+k,b.aa.data()+i,cc.c.data()+m,WORK_BITS,ARF_RND_NEAR);}for(size_t k=old;k<active;k++)arf_set(p.data()+k,cc.c.data()+m);}
    arf_t base;arf_init(base);for(size_t k=0;k<b.n;k++){int i=b.order[k];arf_mul(p.data()+k,p.data()+k,b.aa.data()+i,WORK_BITS,ARF_RND_NEAR);arf_sub_ui(base,b.aa.data()+i,3,WORK_BITS,ARF_RND_NEAR);arf_ui_div(base,1,base,WORK_BITS,ARF_RND_NEAR);arf_add(p.data()+k,p.data()+k,base,WORK_BITS,ARF_RND_NEAR);arf_add(p.data()+k,p.data()+k,&cc.third,WORK_BITS,ARF_RND_NEAR);out[i]=arf_get_d(p.data()+k,ARF_RND_NEAR);}arf_clear(base);for(size_t k=0;k<b.n;k++)arf_clear(p.data()+k);
}

static std::vector<double> make_inputs(size_t n,uint64_t seed){std::vector<double>a(n);std::mt19937_64 g(seed);std::uniform_real_distribution<double>coin(0,1),di(1e-12,3.0),doo(3.0,102.0);for(size_t i=0;i<n;i++){double x=coin(g)<.5?di(g):doo(g);if(x==3.0)x=std::nextafter(3.0,INFINITY);a[i]=x;}return a;}
static std::vector<double> refs(const std::vector<double>&a){std::vector<double>r(a.size());arb_t aa,s,z;arb_init(aa);arb_init(s);arb_init(z);for(size_t i=0;i<a.size();i++){arb_set_d(aa,a[i]);arb_sub_ui(s,aa,2,REF_BITS);arb_zeta(z,s,REF_BITS);r[i]=arf_get_d(arb_midref(z),ARF_RND_NEAR);}arb_clear(z);arb_clear(s);arb_clear(aa);return r;}
static void report(const char*name,const std::vector<double>&y,const std::vector<double>&r,const Batch&b){uint64_t mx=0;int gt1=0,gt2=0,w=0;for(size_t i=0;i<y.size();i++){uint64_t e=ulpdist(y[i],r[i]);if(e>mx){mx=e;w=(int)i;}gt1+=e>1;gt2+=e>2;}std::printf("ACC %s MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu WORST_A=%.17g T=%d\n",name,(unsigned long long)mx,gt1,y.size(),gt2,y.size(),b.a[w],b.T[w]);}
template<class F>static double bench(F&&f,size_t n,int reps){f();double best=1e300;for(int r=0;r<reps;r++){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();best=std::min(best,std::chrono::duration<double,std::nano>(b-a).count()/n);}return best;}

int main(){CoeffCache cc;auto A=make_inputs(5000,0x368B17F17ULL);const double hard[]={2.99,2.9999,std::nextafter(3.0,0.0),std::nextafter(3.0,INFINITY),3.0001,10,40,75,81.764554025359956,100,101,102};for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)A[i]=hard[i];auto R=refs(A);Batch B(A);std::vector<double>y1,y2;eval_scalar(B,cc,y1);eval_stage_batch(B,cc,y2);report("arf_scalar",y1,R,B);report("arf_stage",y2,R,B);int diff=0;for(size_t i=0;i<A.size();i++)diff+=y1[i]!=y2[i];std::printf("SCALAR_STAGE_DIFFER=%d/%zu\n",diff,A.size());for(size_t n:{100UL,400UL,1000UL,3000UL,6000UL,15000UL}){auto x=make_inputs(n,0x900DF00DULL+n);Batch b(x);std::vector<double>o;double s=bench([&](){eval_scalar(b,cc,o);},n,3),st=bench([&](){eval_stage_batch(b,cc,o);},n,3);long long sum=0;for(int t:b.T)sum+=t;std::printf("BENCH N=%zu MEAN_T=%.3f SCALAR=%.3f STAGE=%.3f ns/elem\n",n,(double)sum/n,s,st);}flint_cleanup();}
