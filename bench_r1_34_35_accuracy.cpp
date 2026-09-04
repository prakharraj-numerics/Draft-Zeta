#include <boost/math/special_functions/zeta.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void draft_zeta53_batch(size_t,const double*,double*);
extern "C" int draft_zeta53_batch_uses_avx512(void);

using mp = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>>;

static uint64_t ulp(double a,double b){
    if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
    uint64_t x=std::bit_cast<uint64_t>(a), y=std::bit_cast<uint64_t>(b);
    return x>y?x-y:y-x;
}

int main(){
    constexpr size_t N=1000;
    std::vector<double> s(N), a(N), y(N);
    std::mt19937_64 g(0x5231343335ULL);
    std::uniform_int_distribution<uint64_t> den(100003,999983), base(1,99), frac(1,99999);

    for(size_t i=0;i<N;i++){
        uint64_t d=den(g), b=base(g), f=1+(frac(g)%(d-1));
        uint64_t num=b*d+f;
        double v=(double)num/(double)d;
        if(v<=1.0 || v>=100.0 || v==std::floor(v)){--i; continue;}
        s[i]=v;
        a[i]=v+2.0;
    }

    draft_zeta53_batch(N,a.data(),y.data());

    uint64_t mx=0, gt1=0, gt2=0, gt4=0, bad=0;
    size_t worst=0;
    for(size_t i=0;i<N;i++){
        mp smp=s[i];
        double ref=(double)boost::math::zeta(smp);
        uint64_t u=ulp(y[i],ref);
        if(u==UINT64_MAX){bad++; continue;}
        if(u>mx){mx=u; worst=i;}
        gt1+=u>1; gt2+=u>2; gt4+=u>4;
    }

    std::printf("AVX512=%d\n",draft_zeta53_batch_uses_avx512());
    std::printf("N=%zu\n",N);
    std::printf("MAX_ULP=%llu\n",(unsigned long long)mx);
    std::printf("GT1=%llu/%zu\n",(unsigned long long)gt1,N);
    std::printf("GT2=%llu/%zu\n",(unsigned long long)gt2,N);
    std::printf("GT4=%llu/%zu\n",(unsigned long long)gt4,N);
    std::printf("NONFINITE=%llu/%zu\n",(unsigned long long)bad,N);
    std::printf("WORST_S=%.17g\n",s[worst]);
    std::printf("WORST_A=%.17g\n",a[worst]);
    std::printf("WORST_GOT=%.17g\n",y[worst]);
    mp wmp=s[worst];
    std::printf("WORST_REF=%.17g\n",(double)boost::math::zeta(wmp));
    return 0;
}
