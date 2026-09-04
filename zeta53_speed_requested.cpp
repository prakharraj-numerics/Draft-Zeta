#define main zeta53_v2_embedded_main
#include "zeta53_avx512_batch_v2.cpp"
#undef main

static std::pair<double,double> mean_median_ns(size_t n,int reps,auto&& fn){
    constexpr int S=11;
    std::array<double,S> a{};
    fn(); fn();
    for(int s=0;s<S;s++){
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<reps;r++) fn();
        auto t1=std::chrono::steady_clock::now();
        a[s]=std::chrono::duration<double,std::nano>(t1-t0).count()/((double)reps*(double)n);
    }
    double mean=0.0; for(double v:a) mean+=v; mean/=S;
    std::sort(a.begin(),a.end());
    return {mean,a[S/2]};
}

int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||
       !__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma")) return 2;

    Tables tb;
    const size_t N=3000;
    auto x=make_inputs(N);
    const double hard[]={1.000000000001,1.01,1.1,2.9,3.0,4.0,8.0,10.0,20.0,38.0,50.0,75.0,79.764554025359956,98.0,99.0,std::nextafter(100.0,0.0)};
    for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++) x[i]=hard[i];

    std::vector<double> out(N);
    Scratch sc;
    const bool ok=accuracy_gate("DENSE_NR_REQUESTED",tb,x,out,[&](){
        ours_batch_dense<0,false>(tb,x.data(),N,out.data(),sc);
    });
    if(!ok){ std::puts("ACCURACY_GATE=FAIL"); flint_cleanup(); return 3; }
    std::puts("ACCURACY_GATE=PASS");

    volatile double sink=0.0;
    const size_t sizes[]={100,500,1000,3000};
    for(size_t n:sizes){
        int reps=std::max(8,(int)(120000/n));
        auto [mean_ns,median_ns_v]=mean_median_ns(n,reps,[&](){
            ours_batch_dense<0,false>(tb,x.data(),n,out.data(),sc);
            sink += out[0];
        });
        std::printf("BATCH=%zu REPS=%d MEAN_NS_PER_INPUT=%.6f MEDIAN_NS_PER_INPUT=%.6f MEAN_US_PER_INPUT=%.9f\n",
                    n,reps,mean_ns,median_ns_v,mean_ns/1000.0);
    }
    std::printf("SINK=%.17g\n",(double)sink);
    flint_cleanup();
    return 0;
}
