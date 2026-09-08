// Apple-M working copy of the frozen Intel-specific Crypto/Zeta BASE.
// This file is intentionally a snapshot marker only for the Apple porting branch of work.
// The Intel production BASE remains crypto_zeta53_base.cpp and is untouched.
//
// Intel BASE formula/spine to preserve during future Apple optimization:
//   zeta(x) = 1/(x-1) + degree-9 local recentered polynomial.
//
// No Apple implementation is installed here yet; the first Apple run benchmarks
// only Apple's own C++ standard-library Riemann zeta path on Apple M4+ hardware.
