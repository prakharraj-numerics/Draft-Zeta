#include <boost/math/special_functions/zeta.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using boost::multiprecision::cpp_dec_float_100;

struct Input {
    uint64_t p;
    uint64_t q;
    double x;
};

static uint64_t ordered_bits(double x) {
    uint64_t u = std::bit_cast<uint64_t>(x);
    return (u >> 63) ? ~u : (u | (1ULL << 63));
}

static uint64_t ulp_distance(double a, double b) {
    uint64_t ua = ordered_bits(a), ub = ordered_bits(b);
    return ua > ub ? ua - ub : ub - ua;
}

static std::vector<Input> make_inputs() {
    std::mt19937_64 rng(0x5A17A20260904ULL);
    std::uniform_int_distribution<uint64_t> qdist(2, 1000000);
    std::uniform_int_distribution<uint64_t> kdist(1, 99);
    std::vector<Input> v;
    v.reserve(100);
    while (v.size() < 100) {
        uint64_t q = qdist(rng);
        std::uniform_int_distribution<uint64_t> rdist(1, q - 1);
        uint64_t k = kdist(rng);
        uint64_t r = rdist(rng);
        uint64_t p = k * q + r;
        double x = static_cast<double>(p) / static_cast<double>(q);
        if (!(x > 1.0 && x < 100.0)) continue;
        if (std::floor(x) == x) continue;
        v.push_back({p, q, x});
    }
    return v;
}

int main() {
    auto in = make_inputs();
    std::vector<double> out(100);

    std::printf("ZETA_INPUT_COUNT=%zu\n", in.size());
    for (size_t i = 0; i < in.size(); ++i)
        std::printf("INPUT[%03zu]=%llu/%llu x=%.17g\n", i,
                    (unsigned long long)in[i].p,
                    (unsigned long long)in[i].q, in[i].x);

    uint64_t max_ulp = 0;
    size_t gt1 = 0, gt2 = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        double y = boost::math::zeta(in[i].x);
        cpp_dec_float_100 xx = cpp_dec_float_100(in[i].p) / cpp_dec_float_100(in[i].q);
        cpp_dec_float_100 rr = boost::math::zeta(xx);
        double ref = static_cast<double>(rr);
        uint64_t u = ulp_distance(y, ref);
        max_ulp = std::max(max_ulp, u);
        gt1 += (u > 1);
        gt2 += (u > 2);
    }
    std::printf("ACCURACY_MAX_ULP=%llu\n", (unsigned long long)max_ulp);
    std::printf("ACCURACY_GT1_ULP=%zu/100\n", gt1);
    std::printf("ACCURACY_GT2_ULP=%zu/100\n", gt2);

    volatile double sink = 0.0;
    for (int w = 0; w < 2000; ++w) {
        for (size_t i = 0; i < in.size(); ++i) out[i] = boost::math::zeta(in[i].x);
        sink += out[(unsigned)w % out.size()];
    }

    constexpr int samples = 51;
    constexpr int batches_per_sample = 1000;
    std::vector<double> ns_per_element;
    ns_per_element.reserve(samples);

    for (int s = 0; s < samples; ++s) {
        auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < batches_per_sample; ++b) {
            for (size_t i = 0; i < in.size(); ++i)
                out[i] = boost::math::zeta(in[i].x);
            sink += out[(unsigned)(b + s) % out.size()];
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        ns_per_element.push_back(ns / (batches_per_sample * in.size()));
    }

    std::sort(ns_per_element.begin(), ns_per_element.end());
    double median = ns_per_element[samples / 2];
    std::printf("BATCH_SIZE=100\n");
    std::printf("SAMPLES=%d\n", samples);
    std::printf("BATCHES_PER_SAMPLE=%d\n", batches_per_sample);
    std::printf("MEDIAN_NS_PER_ELEMENT=%.6f\n", median);
    std::printf("MEDIAN_ELEMENTS_PER_SEC=%.3f\n", 1.0e9 / median);
    std::printf("SINK=%.17g\n", (double)sink);
    return 0;
}
