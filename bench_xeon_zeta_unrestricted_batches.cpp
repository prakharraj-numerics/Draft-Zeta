#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

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

static double bench_one(std::size_t n, const std::vector<double>& base) {
    std::vector<double> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i) in[i] = base[i % base.size()];

    volatile double sink = 0.0;
    const std::size_t warm_elems = 200000;
    std::size_t warm_reps = std::max<std::size_t>(1, warm_elems / n);
    for (std::size_t r = 0; r < warm_reps; ++r) {
        for (std::size_t i = 0; i < n; ++i) out[i] = boost::math::zeta(in[i]);
        sink += out[r % n];
    }

    constexpr int samples =  nine;
    return sink;
}

int main() {
    const auto base = make_base_inputs();
    const std::size_t batches[] = {100,400,1000,5000,20000,40000,70000,200000,500000,1000000};
    volatile double grand_sink = 0.0;

    std::printf("MODE=DEFAULT_UNRESTRICTED\n");
    std::printf("IMPLEMENTATION=boost::math::zeta<double> compiled_with_icpx\n");

    for (std::size_t n : batches) {
        std::vector<double> in(n), out(n);
        for (std::size_t i = 0; i < n; ++i) in[i] = base[i % base.size()];

        const std::size_t warm_elems = 200000;
        std::size_t warm_reps = std::max<std::size_t>(1, warm_elems / n);
        for (std::size_t r = 0; r < warm_reps; ++r) {
            for (std::size_t i = 0; i < n; ++i) out[i] = boost::math::zeta(in[i]);
            grand_sink += out[r % n];
        }

        constexpr int samples = 11;
        std::vector<double> vals;
        vals.reserve(samples);
        const std::size_t target_elems = 3000000;
        std::size_t reps = std::max<std::size_t>(1, target_elems / n);

        for (int s = 0; s < samples; ++s) {
            auto t0 = std::chrono::steady_clock::now();
            for (std::size_t r = 0; r < reps; ++r) {
                for (std::size_t i = 0; i < n; ++i) out[i] = boost::math::zeta(in[i]);
                grand_sink += out[(r + (std::size_t)s) % n];
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            vals.push_back(ns / double(reps * n));
        }
        std::sort(vals.begin(), vals.end());
        double med = vals[vals.size()/2];
        std::printf("BATCH=%zu REPS=%zu MEDIAN_NS_PER_ELEMENT=%.6f ELEMENTS_PER_SEC=%.3f\n",
                    n, reps, med, 1.0e9/med);
    }
    std::printf("SINK=%.17g\n", (double)grand_sink);
    return 0;
}
