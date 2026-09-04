#include "zeta53_custom_2core.hpp"
#include <cstddef>

extern "C" void zeta53_local9_batch(const double*, std::size_t, double*);

// Production routing frozen from Xeon 6973P-C benchmark evidence:
//   n < 500  -> raw degree-9 AVX-512 kernel; custom2 is not constructed
//   n >= 500 -> lazy permanent custom2 2-core scheduler
//
// The local-center zeta mathematics is unchanged.
extern "C" void zeta53_batch(const double* x, std::size_t n, double* out) {
    if (n < 500) {
        zeta53_local9_batch(x, n, out);
        return;
    }

    // Function-local static means the helper thread does not exist until the
    // first batch of at least 500 elements reaches this path.
    static Zeta53CustomPermanent2Core custom(zeta53_local9_batch);
    custom.run(x, n, out);
}
