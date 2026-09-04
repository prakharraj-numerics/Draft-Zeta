#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int MAXT=900, COEFF_BITS=8192, REF_BITS=2048, MAXP=400;
static constexpr double STEP=0.0625;
static constexpr int BINS=1632;

static int terms(double a){
    double t;
    if(a<3.0)t=10.0+10.5*a-2.14*a*a;
    else if(a<=10.0)t=11.0+5.0*a;
    else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a;
    else t=-37.0+6.85*a+0.0194*a*a;
    return std::max(6,(int)std::ceil(t)+1);
}
static int bin_of(double a){int b=(int)std::floor(a/STEP);return std::clamp(b,0,BINS-1);}
static uint64_t okey(double x){uint64_t u=std::bit_cast<uint64_t>(x);return (u>>63)?~u:(u|0x8000000000000000ULL);}
static uint64_t ulp(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=okey(a),y=okey(b);return x>y?x-y:y-x;}

struct Input{uint64_t p,q; double x;};
static std::vector<Input> make_inputs(size_t n){
    std::mt19937_64 rng(0x5A17A20260904ULL);
    std::uniform_int_distribution<uint64_t> qdist(2,1000000), kdist(1,99);
    std::vector<Input> v;v.reserve(n);
    while(v.size()<n){
        uint64_t q=qdist(rng); std::uniform_int_distribution<uint64_t> rdist(1,q-1);
        uint64_t k=kdist(rng), r=rdist(rng), p=k*q+r;
        double x=(double)p/(double)q;
        if(x>1.0 && x<100.0 && std::floor(x)!=x) v.push_back({p,q,x});
    }
    return v;
}

struct Runtime{
    std::vector<arf_struct> src;
    std::array<uint16_t,BINS> lut{};
    std::array<std::vector<arf_struct>,MAXP+1> cc;
    std::array<arf_struct,MAXP+1> third;
    std::array<unsigned char,MAXP+1> used{};
    arb_t aa,ss,ref;
    arf_t av,p,base,coef,tmp;

    Runtime():src(MAXT){
        arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);
        arb_t one,p3,t;arb_init(one);arb_init(p3);arb_init(t);arb_init(aa);arb_init(ss);arb_init(ref);
        arf_init(av);arf_init(p);arf_init(base);arf_init(coef);arf_init(tmp);
        arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"[setup] generating corrected coefficients at %d bits\n",COEFF_BITS);
        arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);arb_set_ui(p3,9);
        for(int m=0;m<MAXT;m++){
            arf_init(src.data()+m);arb_poly_get_coeff_arb(t,z,m+1);
            arb_t inv;arb_init(inv);arb_inv(inv,p3,COEFF_BITS);arb_add(t,t,inv,COEFF_BITS);arb_clear(inv);
            arf_set_round(src.data()+m,arb_midref(t),COEFF_BITS,ARF_RND_NEAR);arb_mul_ui(p3,p3,3,COEFF_BITS);
        }
        for(int q=0;q<=MAXP;q++) arf_init(&third[q]);
        build_lut(); build_cache();
        arb_clear(t);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);
    }
    ~Runtime(){
        for(int q=0;q<=MAXP;q++){for(auto &x:cc[q])arf_clear(&x);arf_clear(&third[q]);}
        for(auto &x:src)arf_clear(&x);arf_clear(tmp);arf_clear(coef);arf_clear(base);arf_clear(p);arf_clear(av);
        arb_clear(ref);arb_clear(ss);arb_clear(aa);
    }
    double reference_x(double x){arb_set_d(aa,x);arb_zeta(ref,aa,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR);}
    double reference_a(double a){arb_set_d(aa,a);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR);}
    double eval_source(double a,int bits){
        int T=terms(a);arf_set_d(av,a);arf_set_round(p,src.data()+T-1,bits,ARF_RND_NEAR);
        for(int m=T-2;m>=0;m--){arf_set_round(coef,src.data()+m,bits,ARF_RND_NEAR);arf_fma(p,p,av,coef,bits,ARF_RND_NEAR);}
        arf_mul(p,p,av,bits,ARF_RND_NEAR);arf_sub_ui(base,av,3,bits,ARF_RND_NEAR);arf_ui_div(base,1,base,bits,ARF_RND_NEAR);
        arf_add(p,p,base,bits,ARF_RND_NEAR);arf_set_ui(tmp,1);arf_div_ui(tmp,tmp,3,bits,ARF_RND_NEAR);arf_add(p,p,tmp,bits,ARF_RND_NEAR);
        return arf_get_d(p,ARF_RND_NEAR);
    }
    int pmin(double a,double r){int lo=53,hi=MAXP;if(ulp(eval_source(a,hi),r)>1)return MAXP;while(lo<hi){int mid=(lo+hi)/2;if(ulp(eval_source(a,mid),r)<=1)hi=mid;else lo=mid+1;}for(int q=std::max(53,lo-6);q<lo;q++)if(ulp(eval_source(a,q),r)<=1)return q;return lo;}
    void build_lut(){
        const double f[]={0.015625,0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,0.9375,0.984375};
        for(int b=48;b<BINS;b++){
            double l=b*STEP,r=std::min(102.0,(b+1)*STEP);int mx=53;
            for(double q:f){double a=l+(r-l)*q;if(a==3.0)a=std::nextafter(3.0,INFINITY);double rr=reference_a(a);mx=std::max(mx,pmin(a,rr));}
            lut[b]=(uint16_t)mx;
        }
        for(int b=0;b<48;b++)lut[b]=53;
    }
    int precision(double a)const{return std::min(MAXP,(int)lut[bin_of(a)]+3);}
    void build_cache(){
        for(int b=48;b<BINS;b++)used[std::min(MAXP,(int)lut[b]+3)]=1;
        for(int q=53;q<=MAXP;q++)if(used[q]){
            cc[q].resize(MAXT);for(int m=0;m<MAXT;m++){arf_init(cc[q].data()+m);arf_set_round(cc[q].data()+m,src.data()+m,q,ARF_RND_NEAR);}
            arf_set_ui(&third[q],1);arf_div_ui(&third[q],&third[q],3,q,ARF_RND_NEAR);
        }
    }
    double eval_cached_x(double x,int bits,int T){
        const auto &c=cc[bits];arf_set_d(av,x);arf_add_ui(av,av,2,std::max(bits,64),ARF_RND_NEAR);arf_set(p,c.data()+T-1);
        for(int m=T-2;m>=0;m--)arf_fma(p,p,av,c.data()+m,bits,ARF_RND_NEAR);
        arf_mul(p,p,av,bits,ARF_RND_NEAR);arf_sub_ui(base,av,3,bits,ARF_RND_NEAR);arf_ui_div(base,1,base,bits,ARF_RND_NEAR);
        arf_add(p,p,base,bits,ARF_RND_NEAR);arf_add(p,p,&third[bits],bits,ARF_RND_NEAR);return arf_get_d(p,ARF_RND_NEAR);
    }
};

struct BatchScratch{
    std::vector<double>a;std::vector<int>P,T;std::array<std::vector<int>,MAXP+1>buckets;
    void ensure(size_t n){a.resize(n);P.resize(n);T.resize(n);}
};
static void ours_batch(Runtime&r,const std::vector<Input>&in,size_t n,std::vector<double>&out,BatchScratch&s){
    s.ensure(n);for(auto &b:s.buckets)b.clear();
    for(size_t i=0;i<n;i++){double a=in[i].x+2.0;int p=r.precision(a),t=terms(a);s.a[i]=a;s.P[i]=p;s.T[i]=t;s.buckets[p].push_back((int)i);}
    for(int p=53;p<=MAXP;p++)for(int idx:s.buckets[p])out[idx]=r.eval_cached_x(in[idx].x,p,s.T[idx]);
}
static void intel_batch(const std::vector<Input>&in,size_t n,std::vector<double>&out){for(size_t i=0;i<n;i++)out[i]=boost::math::zeta(in[i].x);}

template<class F>static double median_ns(size_t n,int reps,F&&fn,volatile double &sink){
    constexpr int S=11;std::vector<double>v;v.reserve(S);
    for(int s=0;s<S;s++){auto t0=std::chrono::steady_clock::now();for(int r=0;r<reps;r++){fn();sink+=0.0;}auto t1=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/(double)(reps*n));}
    std::sort(v.begin(),v.end());return v[S/2];
}
int main(){
    const size_t MAXN=5000;auto in=make_inputs(MAXN);Runtime rt;BatchScratch bs;std::vector<double>o1(MAXN),o2(MAXN);
    uint64_t h=1469598103934665603ULL;for(auto &z:in){h^=std::bit_cast<uint64_t>(z.x);h*=1099511628211ULL;}std::printf("INPUT_HASH=%016llx COUNT=%zu DOMAIN_X=(1,100) SAME_INPUTS=1\n",(unsigned long long)h,in.size());
    uint64_t mx=0;int gt1=0;for(size_t i=0;i<MAXN;i++){double ad=in[i].x+2.0;double y=rt.eval_cached_x(in[i].x,rt.precision(ad),terms(ad));double ref=rt.reference_x(in[i].x);uint64_t u=ulp(y,ref);mx=std::max(mx,u);gt1+=u>1;}std::printf("OURS_SANITY_MAX_ULP=%llu GT1=%d/%zu\n",(unsigned long long)mx,gt1,MAXN);
    volatile double sink=0.0;const size_t sizes[]={100,400,800,2000,5000};
    for(size_t n:sizes){
        ours_batch(rt,in,n,o1,bs);intel_batch(in,n,o2);for(int w=0;w<3;w++){ours_batch(rt,in,n,o1,bs);intel_batch(in,n,o2);sink+=o1[w%n]+o2[w%n];}
        int reps=std::max(1,(int)(30000/n));
        double ours=median_ns(n,reps,[&](){ours_batch(rt,in,n,o1,bs);sink+=o1[0];},sink);
        double intel=median_ns(n,reps,[&](){intel_batch(in,n,o2);sink+=o2[0];},sink);
        std::printf("BATCH=%zu REPS=%d OURS_NS_PER_EL=%.6f INTEL_XEON_NS_PER_EL=%.6f INTEL_OVER_OURS=%.6f\n",n,reps,ours,intel,intel/ours);
    }
    std::printf("SINK=%.17g\n",(double)sink);flint_cleanup();return 0;
}
