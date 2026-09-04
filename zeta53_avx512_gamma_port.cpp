#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <flint/fmpz.h>
#include <boost/math/special_functions/zeta.hpp>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#pragma STDC FP_CONTRACT OFF

static constexpr int GEN_BITS=8192, REF_BITS=2048, MAXT=900, MAXP=400, KMAX=8;
static constexpr int BINS=1632;
static constexpr double STEP=0.0625;

static uint64_t ordkey(double x){ uint64_t u=std::bit_cast<uint64_t>(x); return (u>>63)?~u:(u|0x8000000000000000ULL); }
static uint64_t ulpdist(double a,double b){ if(std::isnan(a)||std::isnan(b)) return UINT64_MAX; auto x=ordkey(a),y=ordkey(b); return x>y?x-y:y-x; }

static inline int terms_a(double a){
    double t;
    if(a<3.0)t=10.0+10.5*a-2.14*a*a;
    else if(a<=10.0)t=11.0+5.0*a;
    else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a;
    else t=-37.0+6.85*a+0.0194*a*a;
    return std::max(6,(int)std::ceil(t)+1);
}

struct Tables {
    std::vector<arf_struct> src;
    alignas(64) double cm[MAXT][KMAX];
    alignas(64) int ce[MAXT];
    alignas(64) double third[KMAX];
    alignas(64) int plut[BINS];
    arb_t aa,ss,ref;
    arf_t av,p,base,third_arf,coef;

    Tables():src(MAXT){
        for(int m=0;m<MAXT;m++)for(int j=0;j<KMAX;j++)cm[m][j]=0.0;
        for(int j=0;j<KMAX;j++)third[j]=0.0;
        for(int b=0;b<BINS;b++)plut[b]=53;
        arb_init(aa);arb_init(ss);arb_init(ref);arf_init(av);arf_init(p);arf_init(base);arf_init(third_arf);arf_init(coef);
        arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);
        arb_t one,p3,t,inv;arb_init(one);arb_init(p3);arb_init(t);arb_init(inv);
        arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"[setup] 8192-bit corrected coefficient manufacture + stage scaling\n");
        arb_poly_zeta_series(z,s,one,0,MAXT+1,GEN_BITS);arb_set_ui(p3,9);
        arf_t mant,resid,dv;arf_init(mant);arf_init(resid);arf_init(dv);fmpz_t ee;fmpz_init(ee);
        for(int m=0;m<MAXT;m++){
            arf_init(src.data()+m);
            arb_poly_get_coeff_arb(t,z,m+1);arb_inv(inv,p3,GEN_BITS);arb_add(t,t,inv,GEN_BITS);
            arf_set_round(src.data()+m,arb_midref(t),GEN_BITS,ARF_RND_NEAR);
            arf_frexp(mant,ee,src.data()+m);ce[m]=(int)fmpz_get_si(ee);arf_set(resid,mant);
            for(int j=0;j<KMAX;j++){
                double q=arf_get_d(resid,ARF_RND_NEAR);cm[m][j]=q;arf_set_d(dv,q);arf_sub(resid,resid,dv,GEN_BITS,ARF_RND_NEAR);
            }
            arb_mul_ui(p3,p3,3,GEN_BITS);
        }
        arf_set_ui(third_arf,1);arf_div_ui(third_arf,third_arf,3,GEN_BITS,ARF_RND_NEAR);arf_set(resid,third_arf);
        for(int j=0;j<KMAX;j++){ double q=arf_get_d(resid,ARF_RND_NEAR);third[j]=q;arf_set_d(dv,q);arf_sub(resid,resid,dv,GEN_BITS,ARF_RND_NEAR); }
        fmpz_clear(ee);arf_clear(dv);arf_clear(resid);arf_clear(mant);
        arb_clear(inv);arb_clear(t);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);
        build_precision_lut();
    }
    ~Tables(){
        for(auto &x:src)arf_clear(&x);
        arf_clear(coef);arf_clear(third_arf);arf_clear(base);arf_clear(p);arf_clear(av);
        arb_clear(ref);arb_clear(ss);arb_clear(aa);
    }
    double reference_a(double a){ arb_set_d(aa,a);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR); }
    double reference_x(double x){ arb_set_d(aa,x);arb_zeta(ref,aa,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR); }
    double eval_source(double a,int bits){
        int T=terms_a(a);arf_set_d(av,a);arf_set_round(p,src.data()+T-1,bits,ARF_RND_NEAR);
        for(int m=T-2;m>=0;m--){arf_set_round(coef,src.data()+m,bits,ARF_RND_NEAR);arf_fma(p,p,av,coef,bits,ARF_RND_NEAR);}
        arf_mul(p,p,av,bits,ARF_RND_NEAR);arf_sub_ui(base,av,3,bits,ARF_RND_NEAR);arf_ui_div(base,1,base,bits,ARF_RND_NEAR);
        arf_add(p,p,base,bits,ARF_RND_NEAR);arf_add(p,p,third_arf,bits,ARF_RND_NEAR);return arf_get_d(p,ARF_RND_NEAR);
    }
    int pmin(double a,double r){
        int lo=53,hi=MAXP;if(ulpdist(eval_source(a,hi),r)>1)return MAXP;
        while(lo<hi){int mid=(lo+hi)/2;if(ulpdist(eval_source(a,mid),r)<=1)hi=mid;else lo=mid+1;}
        for(int q=std::max(53,lo-6);q<lo;q++)if(ulpdist(eval_source(a,q),r)<=1)return q;return lo;
    }
    void build_precision_lut(){
        const double f[]={0.015625,0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,0.9375,0.984375};
        std::fprintf(stderr,"[setup] deriving 1/16 per-input precision LUT for a in [3,102]\n");
        for(int b=48;b<BINS;b++){
            double l=b*STEP,r=std::min(102.0,(b+1)*STEP);int mx=53;
            for(double q:f){double a=l+(r-l)*q;if(a==3.0)a=std::nextafter(3.0,INFINITY);mx=std::max(mx,pmin(a,reference_a(a)));}
            plut[b]=std::min(MAXP,mx+3);
            if((b&255)==0)std::fprintf(stderr,"[lut] %d/%d P=%d\n",b,BINS,plut[b]);
        }
    }
};

__attribute__((target("avx512f,avx512dq,avx512vl,avx2,fma")))
static void schedule_x(size_t n,const double*x,const Tables&tb,int*T,int*P){
    size_t i=0;
    const __m512d v2=_mm512_set1_pd(2.0),v3=_mm512_set1_pd(3.0),v10=_mm512_set1_pd(10.0),v40=_mm512_set1_pd(40.0),v16=_mm512_set1_pd(16.0);
    const __m512d c10=_mm512_set1_pd(10.0),c105=_mm512_set1_pd(10.5),c214=_mm512_set1_pd(2.14);
    const __m512d c11=_mm512_set1_pd(11.0),c5=_mm512_set1_pd(5.0),c454=_mm512_set1_pd(4.54),c0485=_mm512_set1_pd(.0485);
    const __m512d cm37=_mm512_set1_pd(-37.0),c685=_mm512_set1_pd(6.85),c0194=_mm512_set1_pd(.0194);
    for(;i<n;i+=8){
        int rem=(int)std::min<size_t>(8,n-i);__mmask8 vm=(__mmask8)((1u<<rem)-1u);
        __m512d vx=_mm512_maskz_loadu_pd(vm,x+i),a=_mm512_add_pd(vx,v2),a2=_mm512_mul_pd(a,a);
        __m512d f0=_mm512_fnmadd_pd(c214,a2,_mm512_fmadd_pd(c105,a,c10));
        __m512d f1=_mm512_fmadd_pd(c5,a,c11);
        __m512d f2=_mm512_fmadd_pd(c0485,a2,_mm512_fmadd_pd(c454,a,c11));
        __m512d f3=_mm512_fmadd_pd(c0194,a2,_mm512_fmadd_pd(c685,a,cm37));
        __mmask8 m0=_mm512_cmp_pd_mask(a,v3,_CMP_LT_OQ),m1=_mm512_cmp_pd_mask(a,v10,_CMP_LE_OQ),m2=_mm512_cmp_pd_mask(a,v40,_CMP_LE_OQ);
        __m512d f=f3;f=_mm512_mask_mov_pd(f,m2,f2);f=_mm512_mask_mov_pd(f,m1,f1);f=_mm512_mask_mov_pd(f,m0,f0);
        f=_mm512_roundscale_pd(f,_MM_FROUND_TO_POS_INF|_MM_FROUND_NO_EXC);
        __m256i ti=_mm512_cvttpd_epi32(f);ti=_mm256_add_epi32(ti,_mm256_set1_epi32(1));ti=_mm256_max_epi32(ti,_mm256_set1_epi32(6));
        _mm256_mask_storeu_epi32(T+i,vm,ti);
        __m512d bx=_mm512_mul_pd(vx,v16);__m256i bi=_mm512_cvttpd_epi32(bx);bi=_mm256_add_epi32(bi,_mm256_set1_epi32(32));
        __m256i pi=_mm256_i32gather_epi32(tb.plut,bi,4);pi=_mm256_min_epi32(pi,_mm256_set1_epi32(MAXP));
        _mm256_mask_storeu_epi32(P+i,vm,pi);
    }
}

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

template<int K>
__attribute__((always_inline,target("avx512f,avx512dq,avx512vl,fma"))) static inline void v_round_p(__m512d p[K],__m256i pv,__mmask8 mask){
    if constexpr(K==1){(void)p;(void)pv;(void)mask;return;}
    __m512d ae=_mm512_abs_pd(p[0]);
    ae=_mm512_max_pd(ae,_mm512_set1_pd(0x1p-1022));
    __m512d e=_mm512_getexp_pd(ae),pd=_mm512_cvtepi32_pd(pv);
    __m512d qe=_mm512_add_pd(_mm512_sub_pd(e,pd),_mm512_set1_pd(1.0));
    __m512d q=_mm512_scalef_pd(_mm512_set1_pd(1.0),qe),iq=_mm512_scalef_pd(_mm512_set1_pd(1.0),_mm512_sub_pd(_mm512_setzero_pd(),qe));
    __m512d z=_mm512_mul_pd(p[K-1],iq);z=_mm512_roundscale_pd(z,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);z=_mm512_mul_pd(z,q);
    p[K-1]=_mm512_mask_mov_pd(p[K-1],mask,z);
}

template<int K,int SW>
__attribute__((always_inline,target("avx512f,avx512dq,fma"))) static inline void v_mac_split(__m512d p[K],__m512d hi,__m512d lo,const __m512d c[K]){
    __m512d old[K],r[K],tmp[K];for(int j=0;j<K;j++){old[j]=p[j];r[j]=p[j];}
    v_mac<K,SW>(r,hi,c);
    for(int j=0;j<K;j++)tmp[j]=old[j];
    v_mac<K,SW>(tmp,lo,r);
    for(int j=0;j<K;j++)p[j]=tmp[j];
}

template<int K>
__attribute__((always_inline,target("avx512f,avx512dq,fma"))) static inline void reciprocal_exp(__m512d x,__m256i pv,__mmask8 valid,__m512d r[K]){
    __m512d dhi,dlo;v_two_sum(x,_mm512_set1_pd(-1.0),dhi,dlo);
    r[0]=_mm512_div_pd(_mm512_set1_pd(1.0),dhi);
    __m512d err=_mm512_fmadd_pd(_mm512_sub_pd(_mm512_setzero_pd(),r[0]),dhi,_mm512_set1_pd(1.0));
    err=_mm512_fnmadd_pd(r[0],dlo,err);
    for(int j=1;j<K;j++){
        r[j]=_mm512_div_pd(err,dhi);
        err=_mm512_fmadd_pd(_mm512_sub_pd(_mm512_setzero_pd(),r[j]),dhi,err);
        err=_mm512_fnmadd_pd(r[j],dlo,err);
    }
    v_round_p<K>(r,pv,valid);
}

struct Scratch {
    std::vector<double> sx,sy;std::vector<int> P,T,idx,sP,sT,sidx,count,start,cur,kbeg,kend;
    void ensure(size_t n){sx.resize(n);sy.resize(n);P.resize(n);T.resize(n);idx.resize(n);sP.resize(n);sT.resize(n);sidx.resize(n);count.assign((KMAX+1)*(MAXT+1),0);start.resize((KMAX+1)*(MAXT+1));cur.resize((KMAX+1)*(MAXT+1));kbeg.assign(KMAX+2,0);kend.assign(KMAX+2,0);}
};

static inline int k_for_p(int p){return std::clamp((p+52)/53,1,KMAX);}

static void prepare(const double*x,size_t n,const Tables&tb,Scratch&s){
    s.ensure(n);schedule_x(n,x,tb,s.T.data(),s.P.data());
    const int W=MAXT+1;
    for(size_t i=0;i<n;i++){int k=k_for_p(s.P[i]),t=s.T[i];s.count[k*W+t]++;}
    int pos=0;
    for(int k=1;k<=KMAX;k++){
        s.kbeg[k]=pos;
        for(int t=MAXT;t>=1;t--){int key=k*W+t;s.start[key]=pos;pos+=s.count[key];}
        s.kend[k]=pos;
    }
    s.cur=s.start;
    for(size_t i=0;i<n;i++){
        int k=k_for_p(s.P[i]),t=s.T[i],key=k*W+t,p=s.cur[key]++;
        s.sx[p]=x[i];s.sP[p]=s.P[i];s.sT[p]=t;s.sidx[p]=(int)i;
    }
}

template<int K,int SW>
__attribute__((target("avx512f,avx512dq,avx512vl,avx2,fma"))) static void eval_group(const Tables&tb,Scratch&s,int lo,int hi){
    const __m512d two=_mm512_set1_pd(2.0),one=_mm512_set1_pd(1.0);
    for(int b=lo;b<hi;b+=32){
        int nr=std::min(32,hi-b),maxT=s.sT[b];
        __m512d x[4],ahi[4],alo[4],p[4][K];__m256i pv[4],tv[4];__mmask8 valid[4];
        for(int c=0;c<4;c++){
            int off=b+8*c,rem=std::clamp(nr-8*c,0,8);valid[c]=(rem==8)?(__mmask8)0xff:(__mmask8)((1u<<rem)-1u);
            x[c]=_mm512_maskz_loadu_pd(valid[c],s.sx.data()+off);pv[c]=_mm256_maskz_loadu_epi32(valid[c],s.sP.data()+off);tv[c]=_mm256_maskz_loadu_epi32(valid[c],s.sT.data()+off);
            v_two_sum(x[c],two,ahi[c],alo[c]);for(int j=0;j<K;j++)p[c][j]=_mm512_setzero_pd();
        }
        for(int m=maxT-1;m>=0;m--){
            __m512d cv[K];for(int j=0;j<K;j++)cv[j]=_mm512_set1_pd(tb.cm[m][j]);
            double sc=1.0;if(m<maxT-1){int de=tb.ce[m+1]-tb.ce[m];sc=std::ldexp(1.0,de);}
            for(int c=0;c<4;c++){
                __mmask8 startm=valid[c]&_mm256_cmp_epi32_mask(tv[c],_mm256_set1_epi32(m+1),_MM_CMPINT_EQ);
                __mmask8 recm=valid[c]&_mm256_cmp_epi32_mask(tv[c],_mm256_set1_epi32(m+1),_MM_CMPINT_GT);
                if(startm){for(int j=0;j<K;j++)p[c][j]=_mm512_mask_mov_pd(p[c][j],startm,cv[j]);}
                if(recm){
                    __m512d cand[K];for(int j=0;j<K;j++)cand[j]=p[c][j];
                    __m512d zc[K];for(int j=0;j<K;j++)zc[j]=cv[j];
                    v_mac_split<K,SW>(cand,_mm512_mul_pd(ahi[c],_mm512_set1_pd(sc)),_mm512_mul_pd(alo[c],_mm512_set1_pd(sc)),zc);
                    for(int j=0;j<K;j++)p[c][j]=_mm512_mask_mov_pd(p[c][j],recm,cand[j]);
                }
                __mmask8 active=startm|recm;if(active)v_round_p<K>(p[c],pv[c],active);
            }
        }
        double sc0=std::ldexp(1.0,tb.ce[0]);
        for(int c=0;c<4;c++){
            __m512d zc[K];for(int j=0;j<K;j++)zc[j]=_mm512_setzero_pd();
            v_mac_split<K,SW>(p[c],_mm512_mul_pd(ahi[c],_mm512_set1_pd(sc0)),_mm512_mul_pd(alo[c],_mm512_set1_pd(sc0)),zc);v_round_p<K>(p[c],pv[c],valid[c]);
            __m512d rr[K];reciprocal_exp<K>(x[c],pv[c],valid[c],rr);v_mac<K,SW>(p[c],one,rr);v_round_p<K>(p[c],pv[c],valid[c]);
            for(int j=0;j<K;j++)zc[j]=_mm512_set1_pd(tb.third[j]);v_mac<K,SW>(p[c],one,zc);v_round_p<K>(p[c],pv[c],valid[c]);
            __m512d y=p[c][K-1];for(int j=K-2;j>=0;j--)y=_mm512_add_pd(p[c][j],y);
            _mm512_mask_storeu_pd(s.sy.data()+b+8*c,valid[c],y);
        }
    }
}

static void ours_batch(const Tables&tb,const double*x,size_t n,double*out,Scratch&s){
    prepare(x,n,tb,s);
    if(s.kbeg[1]<s.kend[1])eval_group<1,2>(tb,s,s.kbeg[1],s.kend[1]);
    if(s.kbeg[2]<s.kend[2])eval_group<2,2>(tb,s,s.kbeg[2],s.kend[2]);
    if(s.kbeg[3]<s.kend[3])eval_group<3,2>(tb,s,s.kbeg[3],s.kend[3]);
    if(s.kbeg[4]<s.kend[4])eval_group<4,2>(tb,s,s.kbeg[4],s.kend[4]);
    if(s.kbeg[5]<s.kend[5])eval_group<5,2>(tb,s,s.kbeg[5],s.kend[5]);
    if(s.kbeg[6]<s.kend[6])eval_group<6,2>(tb,s,s.kbeg[6],s.kend[6]);
    if(s.kbeg[7]<s.kend[7])eval_group<7,2>(tb,s,s.kbeg[7],s.kend[7]);
    if(s.kbeg[8]<s.kend[8])eval_group<8,2>(tb,s,s.kbeg[8],s.kend[8]);
    for(size_t p=0;p<n;p++)out[s.sidx[p]]=s.sy[p];
}

static std::vector<double> make_inputs(size_t n){
    std::mt19937_64 rng(0x5A17A20260904ULL);std::uniform_int_distribution<uint64_t> qdist(2,1000000),kdist(1,99);std::vector<double>v;v.reserve(n);
    while(v.size()<n){uint64_t q=qdist(rng);std::uniform_int_distribution<uint64_t>rdist(1,q-1);uint64_t k=kdist(rng),r=rdist(rng),p=k*q+r;double x=(double)p/(double)q;if(x>1.0&&x<100.0&&std::floor(x)!=x)v.push_back(x);}return v;
}
static void intel_batch(const double*x,size_t n,double*out){for(size_t i=0;i<n;i++)out[i]=boost::math::zeta(x[i]);}

template<class F>static double median_ns(size_t n,int reps,F&&fn){constexpr int S=9;std::array<double,S>a{};for(int s=0;s<S;s++){auto t0=std::chrono::steady_clock::now();for(int r=0;r<reps;r++)fn();auto t1=std::chrono::steady_clock::now();a[s]=std::chrono::duration<double,std::nano>(t1-t0).count()/(double)(reps*n);}std::sort(a.begin(),a.end());return a[S/2];}

int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma")){std::fprintf(stderr,"AVX512F/DQ/VL+FMA required\n");return 2;}
    Tables tb;const size_t N=5000;auto x=make_inputs(N);
    const double hard[]={std::nextafter(1.0,INFINITY),1.000000000001,1.01,1.1,2.9,3.0,4.0,8.0,10.0,20.0,38.0,50.0,75.0,79.764554025359956,98.0,99.0,std::nextafter(100.0,0.0)};for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)x[i]=hard[i];
    Scratch sc;std::vector<double>ours(N),intel(N),ref(N);ours_batch(tb,x.data(),N,ours.data(),sc);
    uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;long long sump=0,sumt=0;int minp=999,maxp=0;
    for(size_t i=0;i<N;i++){ref[i]=tb.reference_x(x[i]);auto u=ulpdist(ours[i],ref[i]);if(u==UINT64_MAX){bad++;continue;}if(u>mx){mx=u;w=(int)i;}gt1+=u>1;gt2+=u>2;double ad=x[i]+2.0;int pp=tb.plut[std::clamp((int)std::floor(x[i]*16.0)+32,0,BINS-1)];int tt=terms_a(ad);sump+=pp;sumt+=tt;minp=std::min(minp,pp);maxp=std::max(maxp,pp);}
    std::printf("ACC MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu NONFINITE=%d WORST_X=%.17g OUR=%.17g REF=%.17g MEAN_P=%.3f MIN_P=%d MAX_P=%d MEAN_T=%.3f\n",(unsigned long long)mx,gt1,N,gt2,N,bad,x[w],ours[w],ref[w],(double)sump/N,minp,maxp,(double)sumt/N);
    if(gt1||bad){std::puts("ACCURACY_GATE=FAIL");flint_cleanup();return 3;}std::puts("ACCURACY_GATE=PASS");
    uint64_t h=1469598103934665603ULL;for(double q:x){h^=std::bit_cast<uint64_t>(q);h*=1099511628211ULL;}std::printf("INPUT_HASH=%016llx COUNT=%zu DOMAIN_X=(1,100) SAME_INPUTS=1\n",(unsigned long long)h,x.size());
    volatile double sink=0.0;for(size_t n:{100UL,400UL,800UL,2000UL,5000UL}){
        ours_batch(tb,x.data(),n,ours.data(),sc);intel_batch(x.data(),n,intel.data());for(int wup=0;wup<3;wup++){ours_batch(tb,x.data(),n,ours.data(),sc);intel_batch(x.data(),n,intel.data());sink+=ours[wup%n]+intel[wup%n];}
        int reps=std::max(1,(int)(30000/n));double on=median_ns(n,reps,[&](){ours_batch(tb,x.data(),n,ours.data(),sc);sink+=ours[0];});double in=median_ns(n,reps,[&](){intel_batch(x.data(),n,intel.data());sink+=intel[0];});
        std::printf("BATCH=%zu REPS=%d OURS_NS_PER_EL=%.6f INTEL_XEON_NS_PER_EL=%.6f INTEL_OVER_OURS=%.6f\n",n,reps,on,in,in/on);
    }
    std::printf("SINK=%.17g\n",(double)sink);flint_cleanup();return 0;
}
