#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void crypto_zeta53_batch(const double*, std::size_t, double*);

static std::vector<double> make_base_inputs() {
    std::mt19937_64 rng(0x5A17A20260908ULL);
    std::uniform_real_distribution<double> dist(1.000001, 99.999999);
    std::vector<double> v;
    v.reserve(1000);
    while (v.size() < 1000) {
        double x = dist(rng);
        if (std::floor(x) != x) v.push_back(x);
    }
    return v;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

int main() {
    const auto base = make_base_inputs();
    const std::size_t batches[] = {100,400,1000,5000,20000,40000,70000,200000,500000,1000000};
    volatile double sink = 0.0;
    std::printf("MODE=SINGLE_THREAD_PINNED\n");
    std::printf("CRYPTO=raw_avx512_local9_no_helpers\n");
    std::printf("INTEL_BASELINE=boost::math::zeta<double>_icpx\n");

    for (std::size_t n : batches) {
        std::vector<double> in(n), out_crypto(n), out_intel(n);
        for (std::size_t i=0;i<n;++i) in[i]=base[i%base.size()];

        const std::size_t warm_reps = std::max<std::size_t>(1, 200000 / n);
        for (std::size_t r=0;r<warm_reps;++r) {
            crypto_zeta53_batch(in.data(), n, out_crypto.data());
            for (std::size_t i=0;i<n;++i) out_intel[i]=boost::math::zeta(in[i]);
            sink += out_crypto[r%n] + out_intel[r%n];
        }

        constexpr int samples=11;
        const std::size_t reps=std::max<std::size_t>(1, 3000000 / n);
        std::vector<double> vc, vi;
        vc.reserve(samples); vi.reserve(samples);

        for (int s=0;s<samples;++s) {
            if ((s & 1)==0) {
                auto t0=std::chrono::steady_clock::now();
                for (std::size_t r=0;r<reps;++r) { crypto_zeta53_batch(in.data(),n,out_crypto.data()); sink += out_crypto[(r+s)%n]; }
                auto t1=std::chrono::steady_clock::now();
                for (std::size_t r=0;r<reps;++r) { for(std::size_t i=0;i<n;++i) out_intel[i]=boost::math::zeta(in[i]); sink += out_intel[(r+s)%n]; }
                auto t2=std::chrono::steady_clock::now();
                vc.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/double(reps*n));
                vi.push_back(std::chrono::duration<double,std::nano>(t2-t1).count()/double(reps*n));
            } else {
                auto t0=std::chrono::steady_clock::now();
                for (std::size_t r=0;r<reps;++r) { for(std::size_t i=0;i<n;++i) out_intel[i]=boost::math::zeta(in[i]); sink += out_intel[(r+s)%n]; }
                auto t1=std::chrono::steady_clock::now();
                for (std::size_t r=0;r<reps;++r) { crypto_zeta53_batch(in.data(),n,out_crypto.data()); sink += out_crypto[(r+s)%n]; }
                auto t2=std::chrono::steady_clock::now();
                vi.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/double(reps*n));
                vc.push_back(std::chrono::duration<double,std::nano>(t2-t1).count()/double(reps*n));
            }
        }
        const double mc=median(vc), mi=median(vi);
        std::printf("BATCH=%zu CRYPTO_NS=%.6f INTEL_NS=%.6f SPEEDUP=%.6f\n", n, mc, mi, mi/mc);
    }
    std::printf("SINK=%.17g\n", (double)sink);
    return 0;
}
