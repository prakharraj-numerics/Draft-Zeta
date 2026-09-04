#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void zeta53_local9_batch(const double*, std::size_t, double*);

static std::vector<double> make_gamma_inputs(std::size_t n) {
    // Deterministic Gamma(shape=2, scale=20), shifted by +1 and rejected above 100.
    // This gives a genuine gamma-distributed workload while respecting 1 < x < 100.
    std::mt19937_64 rng(0x6a09e667f3bcc909ULL);
    std::gamma_distribution<double> gamma(2.0, 20.0);
    std::vector<double> x;
    x.reserve(n);
    while (x.size() < n) {
        const double v = 1.0 + gamma(rng);
        if (v > 1.0 && v < 100.0 && std::isfinite(v)) x.push_back(v);
    }
    return x;
}

int main() {
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512dq") ||
        !__builtin_cpu_supports("avx512vl") || !__builtin_cpu_supports("fma")) {
        std::puts("AVX512_REQUIRED");
        return 2;
    }

    constexpr std::size_t NMAX = 50000;
    auto x = make_gamma_inputs(NMAX);
    std::vector<double> out(NMAX);
    volatile double sink = 0.0;

    double xmin = x[0], xmax = x[0], xsum = 0.0;
    for (double v : x) { xmin = std::min(xmin, v); xmax = std::max(xmax, v); xsum += v; }
    std::printf("INPUT_DIST=GAMMA SHAPE=2 SCALE=20 SHIFT=1 RANGE=(1,100) N=%zu MIN=%.17g MAX=%.17g MEAN=%.9f\n",
                x.size(), xmin, xmax, xsum / x.size());

    const std::size_t sizes[] = {100, 500, 1200, 3000, 4000, 10000, 25000, 50000};
    for (std::size_t n : sizes) {
        // Warm cache/kernel before measurements.
        for (int w = 0; w < 16; ++w) {
            zeta53_local9_batch(x.data(), n, out.data());
            sink += out[(std::size_t)w % n];
        }

        // 11 independent timing samples, ~4 million evaluated inputs per sample.
        // Report arithmetic mean of sample ns/input and the median sample.
        const int reps = std::max(4, (int)(4000000 / n));
        std::vector<double> sample;
        sample.reserve(11);
        for (int t = 0; t < 11; ++t) {
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                zeta53_local9_batch(x.data(), n, out.data());
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
        const double median = sample[sample.size()/2];
        const double elems_per_sec = 1.0e9 / mean;
        std::printf("GAMMA_SPEED BATCH=%zu REPS=%d SAMPLES=11 MEAN_NS_PER_INPUT=%.6f MEDIAN_NS_PER_INPUT=%.6f EVALS_PER_SEC=%.3f\n",
                    n, reps, mean, median, elems_per_sec);
    }
    std::printf("SINK=%.17g\n", (double)sink);
    return 0;
}
