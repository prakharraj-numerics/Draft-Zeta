#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <pthread.h>
#include <sched.h>

#include "zeta53_custom_2core.hpp"

extern "C" void zeta53_local9_batch(const double*, std::size_t, double*);
extern "C" void zeta53_batch(const double*, std::size_t, double*);

static void pin_current(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}

static std::vector<double> make_uniform_inputs(std::size_t n) {
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::uniform_real_distribution<double> ud(1.0, 100.0);
    std::vector<double> x; x.reserve(n);
    while (x.size() < n) {
        double v = ud(rng);
        if (v <= 1.0) v = std::nextafter(1.0, INFINITY);
        if (v >= 100.0) v = std::nextafter(100.0, 0.0);
        x.push_back(v);
    }
    return x;
}

struct Stat { double mean, median; };

template<class F>
static Stat bench(F&& f, const double* x, std::size_t n, double* out, int reps, volatile double& sink) {
    for (int w=0; w<16; ++w) { f(x,n,out); sink += out[(std::size_t)w % n]; }
    std::vector<double> s; s.reserve(11);
    for (int t=0;t<11;++t) {
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<reps;++r){ f(x,n,out); sink += out[(std::size_t)(r+t)%n]; }
        auto t1=std::chrono::steady_clock::now();
        s.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)reps*(double)n));
    }
    double mean=0; for(double v:s) mean+=v; mean/=s.size();
    std::sort(s.begin(),s.end()); return {mean,s[s.size()/2]};
}

int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"usage: %s verify|raw|custom2|production|xeon\n",argv[0]);return 2;}
    std::string mode=argv[1];
    if(mode!="verify"&&mode!="raw"&&mode!="custom2"&&mode!="production"&&mode!="xeon") return 2;
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl") || !__builtin_cpu_supports("fma")) return 3;
    pin_current(0);
    constexpr std::size_t NMAX=50000;
    auto x=make_uniform_inputs(NMAX);
    std::vector<double> a(NMAX),b(NMAX);
    const std::size_t sizes[]={100,499,500,1200,3000,4000,10000,25000,50000};
    volatile double sink=0.0;
    double xmin=*std::min_element(x.begin(),x.end()), xmax=*std::max_element(x.begin(),x.end());
    double sum=0;for(double v:x)sum+=v;
    std::printf("INPUT_DIST=UNIFORM RANGE=(1,100) N=%zu MIN=%.17g MAX=%.17g MEAN=%.9f\n",x.size(),xmin,xmax,sum/x.size());

    if(mode=="verify"){
        std::size_t total=0;
        for(std::size_t n:sizes){
            zeta53_local9_batch(x.data(),n,a.data()); zeta53_batch(x.data(),n,b.data());
            std::size_t diff=0; for(std::size_t i=0;i<n;++i) diff += std::memcmp(&a[i],&b[i],sizeof(double))!=0;
            total+=diff; std::printf("VERIFY_PRODUCTION BATCH=%zu ROUTE=%s BITDIFF=%zu\n",n,n<500?"raw":"custom2",diff);
        }
        std::printf("VERIFY_TOTAL_BITDIFF=%zu\n",total); return total?4:0;
    }

    std::printf("MODE=%s\n",mode.c_str());
    std::printf("XEON_BASELINE=boost::math::zeta<double>\n");
    if(mode=="custom2"){
        Zeta53CustomPermanent2Core custom(zeta53_local9_batch);
        for(std::size_t n:sizes){int reps=std::max(4,(int)(4000000/n)); Stat q=bench([&](const double*in,std::size_t m,double*out){custom.run(in,m,out);},x.data(),n,a.data(),reps,sink); std::printf("RESULT MODE=custom2 BATCH=%zu REPS=%d MEAN_NS=%.6f MEDIAN_NS=%.6f\n",n,reps,q.mean,q.median);}
    } else if(mode=="production") {
        for(std::size_t n:sizes){int reps=std::max(4,(int)(4000000/n)); Stat q=bench([](const double*in,std::size_t m,double*out){zeta53_batch(in,m,out);},x.data(),n,a.data(),reps,sink); std::printf("RESULT MODE=production ROUTE=%s BATCH=%zu REPS=%d MEAN_NS=%.6f MEDIAN_NS=%.6f\n",n<500?"raw":"custom2",n,reps,q.mean,q.median);}
    } else if(mode=="raw") {
        for(std::size_t n:sizes){int reps=std::max(4,(int)(4000000/n)); Stat q=bench([](const double*in,std::size_t m,double*out){zeta53_local9_batch(in,m,out);},x.data(),n,a.data(),reps,sink); std::printf("RESULT MODE=raw BATCH=%zu REPS=%d MEAN_NS=%.6f MEDIAN_NS=%.6f\n",n,reps,q.mean,q.median);}
    } else {
        for(std::size_t n:sizes){int reps=std::max(4,(int)(4000000/n)); Stat q=bench([](const double*in,std::size_t m,double*out){for(std::size_t i=0;i<m;++i)out[i]=boost::math::zeta(in[i]);},x.data(),n,a.data(),reps,sink); std::printf("RESULT MODE=xeon BATCH=%zu REPS=%d MEAN_NS=%.6f MEDIAN_NS=%.6f\n",n,reps,q.mean,q.median);}
    }
    std::printf("SINK=%.17g\n",(double)sink); return 0;
}
