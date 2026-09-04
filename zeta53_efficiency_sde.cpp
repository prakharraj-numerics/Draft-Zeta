#define _GNU_SOURCE
#include <boost/math/special_functions/zeta.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <pthread.h>
#include <sched.h>

extern "C" void zeta53_batch(const double*, std::size_t, double*);

static volatile double g_sink=0.0;
extern "C" __attribute__((noinline)) void zeta53_profile_start(void){asm volatile("":::"memory");}
extern "C" __attribute__((noinline)) void zeta53_profile_stop(void){asm volatile("":::"memory");}

static void pin_current(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
static std::vector<double> make_uniform_inputs(std::size_t n){
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::uniform_real_distribution<double> ud(1.0,100.0);
    std::vector<double>x;x.reserve(n);
    while(x.size()<n){double v=ud(rng);if(v<=1.0)v=std::nextafter(1.0,INFINITY);if(v>=100.0)v=std::nextafter(100.0,0.0);x.push_back(v);}return x;
}

int main(int argc,char**argv){
    if(argc!=3)return 2;
    std::string mode=argv[1]; if(mode!="prod"&&mode!="intel"&&mode!="noop")return 2;
    std::size_t n=(std::size_t)std::strtoull(argv[2],nullptr,10);if(!n)return 2;
    pin_current(0);
    auto x=make_uniform_inputs(n); std::vector<double> out(n);
    auto once=[&](){
        if(mode=="prod") zeta53_batch(x.data(),n,out.data());
        else if(mode=="intel") for(std::size_t i=0;i<n;++i)out[i]=boost::math::zeta(x[i]);
        else asm volatile(""::"r"(x.data()),"r"(out.data()),"r"(n):"memory");
    };
    if(mode!="noop")once();
    zeta53_profile_start(); once(); zeta53_profile_stop();
    if(mode!="noop")g_sink+=out[(n*5u/13u)%n];
    std::printf("ZETA_EFF_SDE mode=%s n=%zu helper_expected=%d sink=%.17g\n",mode.c_str(),n,(mode=="prod"&&n>=500)?1:0,(double)g_sink);
    return 0;
}
