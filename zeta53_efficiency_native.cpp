#define _GNU_SOURCE
#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <sys/resource.h>
#include <time.h>
#include <vector>
#include <pthread.h>
#include <sched.h>

extern "C" void zeta53_batch(const double*, std::size_t, double*);

static volatile double g_sink = 0.0;

static void pin_current(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
static double clock_ns(clockid_t id){timespec t{};clock_gettime(id,&t);return (double)t.tv_sec*1e9+(double)t.tv_nsec;}
static double median7(double v[7]){std::sort(v,v+7);return v[3];}

static std::vector<double> make_uniform_inputs(std::size_t n){
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::uniform_real_distribution<double> ud(1.0,100.0);
    std::vector<double> x; x.reserve(n);
    while(x.size()<n){double v=ud(rng);if(v<=1.0)v=std::nextafter(1.0,INFINITY);if(v>=100.0)v=std::nextafter(100.0,0.0);x.push_back(v);}return x;
}

static int run(const std::string& mode,std::size_t n){
    pin_current(0);
    auto x=make_uniform_inputs(n);
    std::vector<double> out(n);
    auto once=[&](){
        if(mode=="prod") zeta53_batch(x.data(),n,out.data());
        else for(std::size_t i=0;i<n;++i) out[i]=boost::math::zeta(x[i]);
    };
    for(int w=0;w<8;++w){once();g_sink+=out[(std::size_t)w%n];}
    const std::size_t target=4000000;
    std::size_t reps=target/n; if(reps<4)reps=4; if(reps>40000)reps=40000;
    double wall[7],cpu[7];
    for(int t=0;t<7;++t){
        double w0=clock_ns(CLOCK_MONOTONIC_RAW), c0=clock_ns(CLOCK_PROCESS_CPUTIME_ID);
        for(std::size_t r=0;r<reps;++r){once();g_sink+=out[(r+(std::size_t)t)%n];}
        double c1=clock_ns(CLOCK_PROCESS_CPUTIME_ID), w1=clock_ns(CLOCK_MONOTONIC_RAW);
        double d=(double)reps*(double)n;
        wall[t]=(w1-w0)/d; cpu[t]=(c1-c0)/d;
    }
    rusage ru{}; getrusage(RUSAGE_SELF,&ru);
    double mw=median7(wall), mc=median7(cpu);
    std::printf("ZETA_EFF_NATIVE mode=%s n=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_kib=%ld helper_expected=%d sink=%.17g\n",
                mode.c_str(),n,reps,mw,mc,mw?mc/mw:0.0,ru.ru_maxrss,(mode=="prod"&&n>=500)?1:0,(double)g_sink);
    return 0;
}

int main(int argc,char**argv){
    if(argc!=3)return 2;
    std::string mode=argv[1]; if(mode!="prod"&&mode!="intel")return 2;
    std::size_t n=(std::size_t)std::strtoull(argv[2],nullptr,10); if(!n)return 2;
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma"))return 3;
    return run(mode,n);
}
