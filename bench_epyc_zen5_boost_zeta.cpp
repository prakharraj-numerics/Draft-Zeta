#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

struct Input { uint64_t p, q; double x; };

static std::vector<Input> make_inputs(size_t n) {
    std::mt19937_64 rng(0x5A17A20260904ULL);
    std::uniform_int_distribution<uint64_t> qdist(2, 1000000), kdist(1, 99);
    std::vector<Input> v; v.reserve(n);
    while (v.size() < n) {
        uint64_t q = qdist(rng);
        std::uniform_int_distribution<uint64_t> rdist(1, q - 1);
        uint64_t k = kdist(rng), r = rdist(rng), p = k*q + r;
        double x = (double)p / (double)q;
        if (x > 1.0 && x < 100.0 && std::floor(x) != x) v.push_back({p,q,x});
    }
    return v;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void boost_batch(const std::vector<Input>& in, size_t n, double* out) {
    for (size_t i=0;i<n;i++) out[i] = boost::math::zeta(in[i].x);
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(out) : "memory");
#endif
}

template<class F>
static double median_ns(size_t n, int reps, F&& fn) {
    constexpr int S=11;
    std::vector<double> v; v.reserve(S);
    for (int s=0;s<S;s++) {
        auto t0=std::chrono::steady_clock::now();
        for (int r=0;r<reps;r++) fn();
        auto t1=std::chrono::steady_clock::now();
        v.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/(double)(reps*n));
    }
    std::sort(v.begin(),v.end());
    return v[S/2];
}

int main() {
    constexpr size_t MAXN=50000;
    const size_t sizes[]={100,500,1200,4000,10000,30000,50000};
    auto in=make_inputs(MAXN);
    std::vector<double> out(MAXN);

    uint64_t h=1469598103934665603ULL;
    for (auto &z:in) { h ^= std::bit_cast<uint64_t>(z.x); h *= 1099511628211ULL; }
    std::printf("INPUT_HASH=%016llx COUNT=%zu DOMAIN_X=(1,100) INPUT_KIND=NONINTEGER_RATIONAL_FRACTIONS\n",
                (unsigned long long)h,in.size());
    std::printf("IMPLEMENTATION=boost::math::zeta<double> PLATFORM=AMD_EPYC_ZEN5 ANTI_DCE=NOINLINE+MEMORY_BARRIER\n");

    volatile double checksum=0.0;
    for (size_t n:sizes) {
        auto batch=[&](){ boost_batch(in,n,out.data()); checksum += out[n/2]; };
        for (int w=0;w<5;w++) batch();
        int reps=std::max(1,(int)((2500000ULL+n-1)/n));
        double ns=median_ns(n,reps,batch);
        std::printf("BATCH=%zu REPS=%d EPYC_ZEN5_BOOST_ZETA_NS_PER_INPUT=%.6f\n",n,reps,ns);
    }
    std::printf("CHECKSUM=%.17g\n",(double)checksum);
    return 0;
}
