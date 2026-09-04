#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#pragma STDC FP_CONTRACT OFF

static constexpr int GEN_BITS=8192, REF_BITS=2048, MAXT=900, MAXK=9;

static inline int terms_scalar(double a){
    double t;
    if(a<3.0)t=10.0+10.5*a-2.14*a*a;
    else if(a<=10.0)t=11.0+5.0*a;
    else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a;
    else t=-37.0+6.85*a+0.0194*a*a;
    return std::max(6,(int)std::ceil(t)+1);
}

__attribute__((target("avx512f,avx512dq,avx2,fma")))
static void terms_avx512(size_t n,const double*a,int*out){
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
    for(;i<n;i++)out[i]=terms_scalar(a[i]);
}

static uint64_t ordkey(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}
static uint64_t ulpdist(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;auto x=ordkey(a),y=ordkey(b);return x>y?x-y:y-x;}

struct CoeffExp {
    alignas(64) double c[MAXT][MAXK];
    alignas(64) double third[MAXK];
    CoeffExp(){
        for(int m=0;m<MAXT;m++)for(int j=0;j<MAXK;j++)c[m][j]=0.0;
        for(int j=0;j<MAXK;j++)third[j]=0.0;
        arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);
        arb_t one,tc,p3,inv,tmp,dv;arb_init(one);arb_init(tc);arb_init(p3);arb_init(inv);arb_init(tmp);arb_init(dv);
        arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"[setup] manufacture corrected coefficients at %d bits -> %d-component binary64 expansions...\n",GEN_BITS,MAXK);
        arb_poly_zeta_series(z,s,one,0,MAXT+1,GEN_BITS);arb_set_ui(p3,9);
        for(int m=0;m<MAXT;m++){
            arb_poly_get_coeff_arb(tc,z,m+1);arb_inv(inv,p3,GEN_BITS);arb_add(tc,tc,inv,GEN_BITS);arb_set(tmp,tc);
            for(int j=0;j<MAXK;j++){
                double q=arf_get_d(arb_midref(tmp),ARF_RND_NEAR);c[m][j]=q;arb_set_d(dv,q);arb_sub(tmp,tmp,dv,GEN_BITS);
            }
            arb_mul_ui(p3,p3,3,GEN_BITS);
        }
        arb_one(tmp);arb_div_ui(tmp,tmp,3,GEN_BITS);
        for(int j=0;j<MAXK;j++){double q=arf_get_d(arb_midref(tmp),ARF_RND_NEAR);third[j]=q;arb_set_d(dv,q);arb_sub(tmp,tmp,dv,GEN_BITS);}
        arb_clear(dv);arb_clear(tmp);arb_clear(inv);arb_clear(p3);arb_clear(tc);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);
    }
};

struct Prepared {
    size_t n; std::vector<double>a,sa; std::vector<int>T,idx,count,off;
    Prepared(const std::vector<double>&x):n(x.size()),a(x),sa(n),T(n),idx(n),count(MAXT+1,0),off(MAXT+2,0){
        if(__builtin_cpu_supports("avx512f"))terms_avx512(n,a.data(),T.data());else for(size_t i=0;i<n;i++)T[i]=terms_scalar(a[i]);
        for(size_t i=0;i<n;i++){if(T[i]<1||T[i]>MAXT)std::abort();count[T[i]]++;}
        for(int t=0;t<=MAXT;t++)off[t+1]=off[t]+count[t];
        std::vector<int>cur=off;
        for(size_t i=0;i<n;i++){int p=cur[T[i]]++;idx[p]=(int)i;sa[p]=a[i];}
    }
};

__attribute__((always_inline,target("avx512f,avx512dq,fma"))) static inline void v_two_sum(__m512d a,__m512d b,__m512d&s,__m512d&e){
    s=_mm512_add_pd(a,b);__m512d t=_mm512_sub_pd(s,a);e=_mm512_add_pd(_mm512_sub_pd(a,_mm512_sub_pd(s,t)),_mm512_sub_pd(b,t));
}

template<int K,int SW>
__attribute__((always_inline,target("avx512f,avx512dq,fma"))) static inline void v_mac(__m512d p[K],__m512d a,const __m512d c[K]){
    __m512d w[K+2],carry0=_mm512_setzero_pd(),carry1=_mm512_setzero_pd();
    for(int i=0;i<K;i++){
        __m512d ph=_mm512_mul_pd(p[i],a),pl=_mm512_fmadd_pd(p[i],a,_mm512_sub_pd(_mm512_setzero_pd(),ph));
        __m512d s0,e0,s1,e1;v_two_sum(ph,c[i],s0,e0);v_two_sum(s0,carry0,s1,e1);w[i]=s1;
        __m512d t0,t1,u0,u1,v0,v1,q0,q1,r0,r1;
        v_two_sum(pl,e0,t0,t1);v_two_sum(t0,e1,u0,u1);v_two_sum(u0,carry1,v0,v1);carry0=v0;
        v_two_sum(t1,u1,q0,q1);v_two_sum(q0,v1,r0,r1);carry1=_mm512_add_pd(r0,_mm512_add_pd(q1,r1));
    }
    w[K]=carry0;w[K+1]=carry1;
    for(int pass=0;pass<SW;pass++)for(int i=K;i>=0;i--){__m512d s,e;v_two_sum(w[i],w[i+1],s,e);w[i]=s;w[i+1]=e;}
    for(int i=K-1;i>=0;i--){__m512d s,e;v_two_sum(w[i],w[i+1],s,e);w[i]=s;w[i+1]=e;}
    for(int i=0;i<K;i++)p[i]=w[i];
}

template<int K,int SW>
__attribute__((target("avx512f,avx512dq,fma"))) static __m512d eval_vec(__m512d a,int T,const CoeffExp&cc){
    __m512d p[K],cv[K];
    for(int j=0;j<K;j++)p[j]=_mm512_set1_pd(cc.c[T-1][j]);
    for(int m=T-2;m>=0;m--){for(int j=0;j<K;j++)cv[j]=_mm512_set1_pd(cc.c[m][j]);v_mac<K,SW>(p,a,cv);}
    for(int j=0;j<K;j++)cv[j]=_mm512_setzero_pd();v_mac<K,SW>(p,a,cv);
    // Add a four-component reciprocal expansion 1/(a-3).
    __m512d d=_mm512_sub_pd(a,_mm512_set1_pd(3.0));
    __m512d r0=_mm512_div_pd(_mm512_set1_pd(1.0),d);
    __m512d e0=_mm512_fmadd_pd(_mm512_sub_pd(_mm512_setzero_pd(),r0),d,_mm512_set1_pd(1.0));
    __m512d r1=_mm512_div_pd(e0,d);
    __m512d e1=_mm512_fmadd_pd(_mm512_sub_pd(_mm512_setzero_pd(),r1),d,e0);
    __m512d r2=_mm512_div_pd(e1,d);
    __m512d e2=_mm512_fmadd_pd(_mm512_sub_pd(_mm512_setzero_pd(),r2),d,e1);
    __m512d r3=_mm512_div_pd(e2,d);
    for(int j=0;j<K;j++)cv[j]=_mm512_setzero_pd();cv[0]=r0;if constexpr(K>1)cv[1]=r1;if constexpr(K>2)cv[2]=r2;if constexpr(K>3)cv[3]=r3;v_mac<K,SW>(p,_mm512_set1_pd(1.0),cv);
    for(int j=0;j<K;j++)cv[j]=_mm512_set1_pd(cc.third[j]);v_mac<K,SW>(p,_mm512_set1_pd(1.0),cv);
    __m512d y=_mm512_setzero_pd();for(int j=K-1;j>=0;j--)y=_mm512_add_pd(y,p[j]);return y;
}

template<int K,int SW>
__attribute__((target("avx512f,avx512dq,fma"))) static void eval_batch(const Prepared&b,const CoeffExp&cc,std::vector<double>&out){
    std::vector<double>sy(b.n);out.resize(b.n);
    for(int t=1;t<=MAXT;t++){
        int lo=b.off[t],hi=b.off[t+1];
        for(int p=lo;p<hi;p+=8){int rem=std::min(8,hi-p);__mmask8 mask=(__mmask8)((1u<<rem)-1u);__m512d x=_mm512_maskz_loadu_pd(mask,b.sa.data()+p);__m512d y=eval_vec<K,SW>(x,t,cc);_mm512_mask_storeu_pd(sy.data()+p,mask,y);}
    }
    for(size_t p=0;p<b.n;p++)out[b.idx[p]]=sy[p];
}

static std::vector<double>make_inputs(size_t n,uint64_t seed){std::vector<double>a(n);std::mt19937_64 g(seed);std::uniform_real_distribution<double>coin(0,1),di(1e-12,3.0),doo(3.0,102.0);for(size_t i=0;i<n;i++){double x=coin(g)<.5?di(g):doo(g);if(x==3.0)x=std::nextafter(3.0,INFINITY);a[i]=x;}return a;}
static std::vector<double>refs(const std::vector<double>&a){std::vector<double>r(a.size());arb_t aa,s,z;arb_init(aa);arb_init(s);arb_init(z);for(size_t i=0;i<a.size();i++){arb_set_d(aa,a[i]);arb_sub_ui(s,aa,2,REF_BITS);arb_zeta(z,s,REF_BITS);r[i]=arf_get_d(arb_midref(z),ARF_RND_NEAR);}arb_clear(z);arb_clear(s);arb_clear(aa);return r;}
static bool report(const char*name,const std::vector<double>&y,const std::vector<double>&r,const std::vector<double>&a,const Prepared&b){uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;for(size_t i=0;i<y.size();i++){auto e=ulpdist(y[i],r[i]);if(e==UINT64_MAX){bad++;continue;}if(e>mx){mx=e;w=(int)i;}gt1+=e>1;gt2+=e>2;}std::printf("ACC %-8s MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu NONFINITE=%d WORST_A=%.17g T=%d\n",name,(unsigned long long)mx,gt1,y.size(),gt2,y.size(),bad,a[w],b.T[w]);return gt1==0&&bad==0;}
template<class F>static double bench(F&&f,size_t n,int reps){f();double best=1e300;for(int r=0;r<reps;r++){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();best=std::min(best,std::chrono::duration<double,std::nano>(b-a).count()/n);}return best;}

template<int K,int SW>static void bench_variant(const char*name,const CoeffExp&cc,bool dobench){auto A=make_inputs(5000,0xA5120000ULL+K*31+SW);const double hard[]={1e-12,1.0,2.9,2.99,2.9999,std::nextafter(3.0,0.0),std::nextafter(3.0,INFINITY),3.0001,4,10,20,40,50,75,81.764554025359956,100,101,102};for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)A[i]=hard[i];auto R=refs(A);Prepared B(A);std::vector<double>y;eval_batch<K,SW>(B,cc,y);bool pass=report(name,y,R,A,B);if(!pass||!dobench)return;for(size_t n:{100UL,400UL,1000UL,3000UL,6000UL,15000UL}){auto x=make_inputs(n,0xBADC0DEULL+n);Prepared b(x);std::vector<double>o;double ns=bench([&](){eval_batch<K,SW>(b,cc,o);},n,4);long long sum=0;for(int t:b.T)sum+=t;std::printf("BENCH %s N=%zu MEAN_T=%.3f AVX512=%.3f ns/elem\n",name,n,(double)sum/n,ns);}}

int main(){if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("fma")){std::fprintf(stderr,"AVX512F+FMA required\n");return 2;}CoeffExp cc;bench_variant<7,3>("K7S3",cc,false);bench_variant<8,2>("K8S2",cc,false);bench_variant<8,3>("K8S3",cc,true);bench_variant<8,4>("K8S4",cc,true);bench_variant<9,3>("K9S3",cc,true);flint_cleanup();return 0;}
