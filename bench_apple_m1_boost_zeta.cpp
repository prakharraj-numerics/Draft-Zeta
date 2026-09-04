#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

struct Input {
    uint64_t p, q;
    double x;
};

static std::vector<Input> make_inputs(size_t n) {
    // Same deterministic rational-fraction generator/seed used by the Xeon harness.
    std::mt19937_64 rng(0x5A17A20260904ULL);
    std::uniform_int_distribution<uint64_t> qdist(2, 1000000);
    std::uniform_int_distribution<uint64_t> kdist(1, 99);
    std::vector<Input> v;
    v.reserve(n);
    while (v.size() < n) {
        uint64_t q = qdist(rng);
        std::uniform_int_distribution<uint64_t> rdist(1, q - 1);
        uint64_t k = kdist(rng);
        uint64_t r = rdist(rng);
        uint64_t p = k * q + r;
        double x = static_cast<double>(p) / static_cast<double>(q);
        if (x > 1.0 && x < 100.0 && std::floor(x) != x)
            v.push_back({p, q, x});
    }
    return v;
}

__attribute__((noinline))
static void boost_zeta_batch(const Input* in, size_t n, double* out) {
    for (size_t i = 0; i < n; ++i)
        out[i] = boost::math::zeta(in[i].x);
}

static inline void benchmark_barrier(const void* p) {
    // Make every output store observable to the optimizer without adding an
    // O(n) checksum to the timed region.
    asm volatile("" : : "r"(p) : "memory");
}

template <class F>
static double median_ns_per_input(size_t n, int reps, F&& fn) {
    constexpr int SAMPLES = 11;
    std::vector<double> samples;
    samples.reserve(SAMPLES);
    for (int s = 0; s < SAMPLES; ++s) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            fn();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count()
                          / static_cast<double>(reps * n));
    }
    std::sort(samples.begin(), samples.end());
    return samples[SAMPLES / 2];
}

int main() {
    constexpr size_t MAXN = 50000;
    const size_t sizes[] = {100, 500, 1200, 4000, 10000, 30000, 50000};

    auto in = make_inputs(MAXN);
    std::vector<double> out(MAXN);

    uint64_t h = 1469598103934665603ULL;
    for (const auto& z : in) {
        h ^= std::bit_cast<uint64_t>(z.x);
        h *= 1099511628211ULL;
    }
    std::printf("INPUT_HASH=%016llx COUNT=%zu DOMAIN_X=(1,100) INPUT_KIND=NONINTEGER_RATIONAL_FRACTIONS\n",
                static_cast<unsigned long long>(h), in.size());
    std::printf("IMPLEMENTATION=boost::math::zeta<double> PLATFORM=APPLE_SILICON_NATIVE ANTI_DCE=NOINLINE+MEMORY_BARRIER\n");

    double checksum = 0.0;
    for (size_t n : sizes) {
        auto batch = [&]() {
            boost_zeta_batch(in.data(), n, out.data());
            benchmark_barrier(out.data());
        };

        for (int w = 0; w < 5; ++w)
            batch();

        // Keep each sample large enough to amortize timer noise while avoiding
        // excessive CI time at the largest batches.
        int reps = std::max(1, static_cast<int>((2500000ULL + n - 1) / n));
        double ns = median_ns_per_input(n, reps, batch);

        // Read values after timing so the benchmark also has a concrete sanity checksum.
        checksum += out[0] + out[n / 2] + out[n - 1];
        std::printf("BATCH=%zu REPS=%d APPLE_M1_BOOST_ZETA_NS_PER_INPUT=%.6f\n",
                    n, reps, ns);
    }

    std::printf("CHECKSUM=%.17g\n", checksum);
    return 0;
}
