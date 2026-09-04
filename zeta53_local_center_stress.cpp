#define main zeta53_local_center_embedded_main
#include "zeta53_local_center.cpp"
#undef main

static std::vector<double> make_stress_inputs(){
    std::vector<double> x;
    x.reserve(30000);
    auto add=[&](double v){
        if(v>1.0 && v<100.0 && std::isfinite(v)) x.push_back(v);
    };

    // Every local-bin boundary and center, with adjacent representable doubles.
    // This explicitly stresses |h| ~= 1/4, where local Taylor truncation is worst.
    for(int j=0;j<=NBINS;j++){
        double e=1.0 + 0.5*j;
        if(e>1.0 && e<100.0){
            add(std::nextafter(e,-INFINITY));
            add(e);
            add(std::nextafter(e, INFINITY));
        }
    }
    for(int b=0;b<NBINS;b++){
        double lo=1.0+0.5*b, hi=lo+0.5, c=lo+0.25;
        add(c);
        add(std::nextafter(c,-INFINITY));
        add(std::nextafter(c, INFINITY));
        // Fixed interior fractions, symmetric about center.
        static const double f[]={1.0/1024,1.0/256,1.0/64,1.0/16,1.0/8,1.0/4,3.0/8,1.0/2,5.0/8,3.0/4,7.0/8,15.0/16,63.0/64,255.0/256,1023.0/1024};
        for(double q:f) add(lo + 0.5*q);
    }

    // Pole-neighbourhood ladder, where final addition/reciprocal rounding is most delicate.
    for(int k=1;k<=52;k++){
        double d=std::ldexp(1.0,-k);
        add(1.0+d);
        add(std::nextafter(1.0+d,-INFINITY));
        add(std::nextafter(1.0+d, INFINITY));
    }

    // Deterministic random coverage across all bins.
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::uniform_real_distribution<double> ud(1.0,100.0);
    while(x.size()<30000){
        double q=ud(rng);
        if(q<=1.0) q=std::nextafter(1.0,INFINITY);
        if(q>=100.0) q=std::nextafter(100.0,0.0);
        x.push_back(q);
    }
    return x;
}

int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||
       !__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma")) return 2;

    SourceTables src;
    LocalTables lt(src);
    auto x=make_stress_inputs();
    const size_t n=x.size();
    std::vector<double> ref(n), y(n);
    std::fprintf(stderr,"[stress] building %zu Arb references: every bin edge/center + pole ladder + deterministic random\n",n);
    for(size_t i=0;i<n;i++) ref[i]=src.reference_x(x[i]);

    const int degrees[]={4,5,6,7,8,9,10,11,12};
    int best_f64=-1,best_dd=-1;
    for(int d:degrees){
        local_gather_f64(lt,x.data(),n,y.data(),d);
        Acc af=assess(x,y,ref);
        std::printf("STRESS_F64 D=%d MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu BAD=%d WORST_X=%.17g OUR=%.17g REF=%.17g\n",
            d,(unsigned long long)af.mx,af.gt1,n,af.gt2,n,af.bad,x[af.w],y[af.w],ref[af.w]);
        if(best_f64<0 && af.gt1==0 && af.bad==0) best_f64=d;

        local_gather_dd(lt,x.data(),n,y.data(),d);
        Acc ad=assess(x,y,ref);
        std::printf("STRESS_DD D=%d MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu BAD=%d WORST_X=%.17g OUR=%.17g REF=%.17g\n",
            d,(unsigned long long)ad.mx,ad.gt1,n,ad.gt2,n,ad.bad,x[ad.w],y[ad.w],ref[ad.w]);
        if(best_dd<0 && ad.gt1==0 && ad.bad==0) best_dd=d;
    }
    std::printf("STRESS_BEST_F64_DEGREE=%d STRESS_BEST_DD_DEGREE=%d CASES=%zu\n",best_f64,best_dd,n);
    if(best_f64<0 || best_dd<0){ std::puts("STRESS_GATE=FAIL"); flint_cleanup(); return 3; }
    std::puts("STRESS_GATE=PASS");

    // Re-benchmark the minimum stress-passing degrees on the requested batch sizes.
    volatile double sink=0.0;
    const size_t sizes[]={100,500,1000,3000};
    std::vector<double> out(n);
    for(size_t bn:sizes){
        int reps=std::max(20,(int)(480000/bn));
        double fm=mean_ns(bn,reps,[&](){local_gather_f64(lt,x.data(),bn,out.data(),best_f64);sink+=out[0];});
        double fmed=median_ns(bn,std::max(5,reps/3),[&](){local_gather_f64(lt,x.data(),bn,out.data(),best_f64);sink+=out[0];});
        double dm=mean_ns(bn,reps,[&](){local_gather_dd(lt,x.data(),bn,out.data(),best_dd);sink+=out[0];});
        double dmed=median_ns(bn,std::max(5,reps/3),[&](){local_gather_dd(lt,x.data(),bn,out.data(),best_dd);sink+=out[0];});
        std::printf("STRESS_SPEED BATCH=%zu F64_DEG=%d F64_MEAN_NS=%.6f F64_MEDIAN_NS=%.6f DD_DEG=%d DD_MEAN_NS=%.6f DD_MEDIAN_NS=%.6f\n",
            bn,best_f64,fm,fmed,best_dd,dm,dmed);
    }
    std::printf("SINK=%.17g\n",(double)sink);
    flint_cleanup();
    return 0;
}
