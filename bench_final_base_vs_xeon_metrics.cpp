#include <boost/math/special_functions/zeta.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

extern "C" void crypto_zeta53_base_batch(const double*, std::size_t, double*);

static std::vector<double> make_seed_inputs() {
    std::mt19937_64 rng(0x5A17A20260908ULL);
    std::uniform_real_distribution<double> dist(1.000001, 99.999999);
    std::vector<double> v;
    v.reserve(1000);
    while (v.size() < 1000) {
        const double x = dist(rng);
        if (std::floor(x) != x) v.push_back(x);
    }
    return v;
}

static inline void boost_batch(const double* x, std::size_t n, double* out) {
    for (std::size_t i = 0; i < n; ++i) out[i] = boost::math::zeta(x[i]);
}

using Fn = void(*)(const double*,std::size_t,double*);

static double median(std::vector<double>& x) {
    std::sort(x.begin(), x.end());
    return x[x.size()/2];
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s base|boost N speed|counter\n", argv[0]);
        return 2;
    }
    const std::string which = argv[1];
    const std::size_t n = std::strtoull(argv[2], nullptr, 10);
    const std::string mode = argv[3];
    Fn fn = nullptr;
    if (which == "base") fn = crypto_zeta53_base_batch;
    else if (which == "boost") fn = boost_batch;
    else return 3;

    auto seed = make_seed_inputs();
    std::vector<double> x(n), y(n);
    for (std::size_t i=0;i<n;++i) x[i]=seed[i%seed.size()];

    volatile double sink = 0.0;
    const std::size_t warm = std::max<std::size_t>(1, 200000/n);
    for (std::size_t r=0;r<warm;++r) { fn(x.data(),n,y.data()); sink += y[r%n]; }

    if (mode == "speed") {
        const std::size_t reps = std::max<std::size_t>(1, 3000000/n);
        std::vector<double> samples;
        samples.reserve(11);
        for (int s=0;s<11;++s) {
            auto t0=std::chrono::steady_clock::now();
            for(std::size_t r=0;r<reps;++r){fn(x.data(),n,y.data()); sink += y[(r+s)%n];}
            auto t1=std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/double(reps*n));
        }
        std::printf("SPEED impl=%s n=%zu ns_per_el=%.9f evals=%zu\n",which.c_str(),n,median(samples),reps*n);
    } else if (mode == "counter") {
        // Long, fixed-work interval for perf stat.  >=10M evaluations stabilizes counters.
        const std::size_t reps = std::max<std::size_t>(1, 10000000/n);
        auto t0=std::chrono::steady_clock::now();
        for(std::size_t r=0;r<reps;++r){fn(x.data(),n,y.data()); sink += y[r%n];}
        auto t1=std::chrono::steady_clock::now();
        const std::size_t evals=reps*n;
        const double ns=std::chrono::duration<double,std::nano>(t1-t0).count()/double(evals);
        std::printf("COUNTER impl=%s n=%zu ns_per_el=%.9f evals=%zu sink=%.17g\n",which.c_str(),n,ns,evals,(double)sink);
    } else return 4;
    if (sink==123.0) std::puts("");
    return 0;
}
