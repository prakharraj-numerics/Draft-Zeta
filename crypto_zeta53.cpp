#include <cstddef>

extern "C" void zeta53_local9_batch(const double*, std::size_t, double*);

// "crypto": copy of the production zeta route with all helper schedulers removed.
// Production mathematics/kernel is unchanged; every batch goes directly through
// the raw AVX-512 local9 kernel. No custom2, Highway, TBB, or other helper backend.
extern "C" void crypto_zeta53_batch(const double* x, std::size_t n, double* out) {
    zeta53_local9_batch(x, n, out);
}
