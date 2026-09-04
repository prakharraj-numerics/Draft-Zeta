#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void zeta53_local9_batch(const double*, std::size_t, double*);

static std::vector<double> make_uniform_inputs(std::size_t n) {
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::uniform_real_distribution<double> ud(1.0, 100.0);
    std::vector<double> x;
    x.reserve(n);
    while (x.size() < n) {
        double v = ud(rng);
        if (v <= 1.0) v = std::nextafter(1.0, INFINITY);
        if (v >= 100.0) v = std::nextafter(100.0, 0.0);
        x.push_back(v);
    }
    return x;
}

struct Stat { double mean, median; };

static Stat bench_ours(const double* x, std::size_t n, double* out, int reps, volatile double& sink) {
    for (int w = 0; w < 16; ++w) {
        zeta53_local9_batch(x, n, out);
        sink += out[(std::size_t)w % n];
    }
    std::vector<double> sample;
    sample.reserve(11);
    for (int t = 0; t < 11; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            zeta53_local9_batch(x, n, out);
            sink += out[(std::size_t)(r + t) % n];
        }
        auto t1 = std::chrono::steady_clock::now();
        sample.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                         ((double)reps * (double)n));
    }
    double mean = 0.0;
    for (double v : sample) mean += v;
    mean /= sample.size();
    std::sort(sample.begin(), sample.end());
    return {mean, sample[sample.size()/2]};
}

static Stat bench_xeon(const double* x, std::size_t n, double* out, int reps, volatile double& sink) {
    for (int w = 0; w < 16; ++w) {
        for (std::size_t i = 0; i < n; ++i) out[i] = boost::math::zeta(x[i]);
        sink += out[(std::size_t)w % n];
    }
    std::vector<double> sample;
    sample.reserve(11);
    for (int t = 0; t < 11; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            for (std::size_t i = 0; i < n; ++i) out[i] = boost::math::zeta(x[i]);
            sink += out[(std::size_t)(r + t) % n];
        }
        auto t1 = std::chrono::steady_clock::now();
        sample.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                         ((double)reps * (double)n));
    }
    double mean = 0.0;
    for (double v : sample) mean += v;
    mean /= sample.size();
    std::sort(sample.begin(), sample.end());
    return {mean, sample[sample.size()/2]};
}

int main() {
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512dq") ||
        !__builtin_cpu_supports("avx512vl") || !__builtin_cpu_supports("fma")) {
        std::puts("AVX512_REQUIRED");
        return 2;
    }

    constexpr std::size_t NMAX = 50000;
    auto x = make_uniform_inputs(NMAX);
    std::vector<double> out(NMAX);
    volatile double sink = 0.0;

    double xmin = x[0], xmax = x[0], xsum = 0.0;
    for (double v : x) { xmin = std::min(xmin, v); xmax = std::max(xmax, v); xsum += v; }
    std::printf("INPUT_DIST=UNIFORM RANGE=(1,100) N=%zu MIN=%.17g MAX=%.17g MEAN=%.9f\n",
                x.size(), xmin, xmax, xsum / x.size());
    std::puts("XEON_BASELINE=boost::math::zeta<double>");

    const std::size_t sizes[] = {100, 500, 1200, 3000, 4000, 10000, 25000, 50000};
    for (std::size_t n : sizes) {
        const int reps = std::max(4, (int)(4000000 / n));
        Stat ours = bench_ours(x.data(), n, out.data(), reps, sink);
        Stat xeon = bench_xeon(x.data(), n, out.data(), reps, sink);
        std::printf("COMPARE BATCH=%zu REPS=%d SAMPLES=11 OUR_MEAN_NS=%.6f OUR_MEDIAN_NS=%.6f XEON_MEAN_NS=%.6f XEON_MEDIAN_NS=%.6f SPEEDUP_MEAN=%.6f SPEEDUP_MEDIAN=%.6f\n",
                    n, reps, ours.mean, ours.median, xeon.mean, xeon.median,
                    xeon.mean / ours.mean, xeon.median / ours.median);
    }
    std::printf("SINK=%.17g\n", (double)sink);
    return 0;
}
