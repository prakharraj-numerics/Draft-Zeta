#include <flint/arb.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void zeta53_local9_batch(const double*, std::size_t, double*);

static uint64_t ordkey(double x){
    uint64_t u=std::bit_cast<uint64_t>(x);
    return (u>>63)?~u:(u|0x8000000000000000ULL);
}
static uint64_t ulpdist(double a,double b){
    if(std::isnan(a)||std::isnan(b)) return UINT64_MAX;
    uint64_t x=ordkey(a),y=ordkey(b); return x>y?x-y:y-x;
}
static std::vector<double> stress_inputs(){
    constexpr int NBINS=198;
    std::vector<double>x; x.reserve(30000);
    auto add=[&](double v){if(v>1.0 && v<100.0 && std::isfinite(v))x.push_back(v);};
    for(int j=0;j<=NBINS;j++){
        double e=1.0+0.5*j;
        if(e>1.0&&e<100.0){add(std::nextafter(e,-INFINITY));add(e);add(std::nextafter(e,INFINITY));}
    }
    static const double f[]={1.0/1024,1.0/256,1.0/64,1.0/16,1.0/8,1.0/4,3.0/8,1.0/2,5.0/8,3.0/4,7.0/8,15.0/16,63.0/64,255.0/256,1023.0/1024};
    for(int b=0;b<NBINS;b++){
        double lo=1.0+0.5*b,c=lo+0.25;
        add(c);add(std::nextafter(c,-INFINITY));add(std::nextafter(c,INFINITY));
        for(double q:f)add(lo+0.5*q);
    }
    for(int k=1;k<=52;k++){
        double q=1.0+std::ldexp(1.0,-k);
        add(q);add(std::nextafter(q,-INFINITY));add(std::nextafter(q,INFINITY));
    }
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::uniform_real_distribution<double>ud(1.0,100.0);
    while(x.size()<30000){double q=ud(rng);if(q<=1.0)q=std::nextafter(1.0,INFINITY);if(q>=100.0)q=std::nextafter(100.0,0.0);x.push_back(q);}
    return x;
}
static double ref_zeta(double x,arb_t a,arb_t z){
    arb_set_d(a,x);arb_zeta(z,a,2048);return arf_get_d(arb_midref(z),ARF_RND_NEAR);
}
template<class F> static double mean_ns(size_t n,int reps,F&&f){
    auto t0=std::chrono::steady_clock::now();for(int r=0;r<reps;r++)f();auto t1=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::nano>(t1-t0).count()/(double)(reps*n);
}
template<class F> static double median_ns(size_t n,int reps,F&&f){
    std::vector<double>v;v.reserve(9);for(int t=0;t<9;t++){auto a=std::chrono::steady_clock::now();for(int r=0;r<reps;r++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/(double)(reps*n));}std::sort(v.begin(),v.end());return v[4];
}
int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma")){std::puts("AVX512_REQUIRED");return 2;}
    auto x=stress_inputs();const size_t n=x.size();std::vector<double>y(n),ref(n);
    arb_t a,z;arb_init(a);arb_init(z);
    std::fprintf(stderr,"[static-check] building %zu 2048-bit Arb references\n",n);
    for(size_t i=0;i<n;i++)ref[i]=ref_zeta(x[i],a,z);
    zeta53_local9_batch(x.data(),n,y.data());
    uint64_t mx=0;size_t wi=0;int gt1=0,gt2=0,bad=0;
    for(size_t i=0;i<n;i++){
        uint64_t u=ulpdist(y[i],ref[i]);if(u==UINT64_MAX){bad++;continue;}if(u>mx){mx=u;wi=i;}gt1+=u>1;gt2+=u>2;
    }
    std::printf("STATIC_ACC MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu BAD=%d WORST_X=%.17g OUR=%.17g REF=%.17g\n",(unsigned long long)mx,gt1,n,gt2,n,bad,x[wi],y[wi],ref[wi]);
    if(gt1||bad){std::puts("STATIC_ACCURACY_GATE=FAIL");return 3;}
    std::puts("STATIC_ACCURACY_GATE=PASS");

    volatile double sink=0.0;std::vector<double>out(n);
    const size_t sizes[]={100,500,1000,3000};
    for(size_t bn:sizes){
        int reps=std::max(20,(int)(1000000/bn));
        zeta53_local9_batch(x.data(),bn,out.data());
        double mean=mean_ns(bn,reps,[&](){zeta53_local9_batch(x.data(),bn,out.data());sink+=out[0];});
        double med=median_ns(bn,std::max(10,reps/3),[&](){zeta53_local9_batch(x.data(),bn,out.data());sink+=out[0];});
        std::printf("STATIC_SPEED BATCH=%zu REPS=%d MEAN_NS_PER_INPUT=%.6f MEDIAN_NS_PER_INPUT=%.6f\n",bn,reps,mean,med);
    }
    std::printf("SINK=%.17g TABLE_BYTES=%zu DEGREE=9 CENTERS=198\n",(double)sink,(size_t)(10*198*sizeof(double)));
    arb_clear(z);arb_clear(a);flint_cleanup();return 0;
}
