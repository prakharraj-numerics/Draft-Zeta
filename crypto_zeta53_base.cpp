#include "zeta53_local9_table.h"
#include <immintrin.h>
#include <algorithm>
#include <cstddef>

#pragma STDC FP_CONTRACT OFF

// Frozen BASE: best single-thread Crypto optimization from run 34226333457.
// Formula/spine is unchanged:
//   zeta(x) = 1/(x-1) + degree-9 local recentered polynomial.
// Optimization only:
//   - retain existing coefficient-major table/layout
//   - evaluate 1/(x-1) with AVX-512 rcp14 + two Newton refinements
// Accuracy gate on the freezing run: max 1 ULP, 0 cases >1 ULP on 9,600 MPFR-256 references.

namespace {
using namespace zeta53_local9_table;

static inline __m512d reciprocal_nr2(__m512d d) {
    const __m512d two = _mm512_set1_pd(2.0);
    __m512d r = _mm512_rcp14_pd(d);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(d, r, two));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(d, r, two));
    return r;
}
}

extern "C" __attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
void crypto_zeta53_base_batch(const double *x, std::size_t n, double *out) {
    using namespace zeta53_local9_table;
    const __m512d one  = _mm512_set1_pd(1.0);
    const __m512d two  = _mm512_set1_pd(2.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d c125 = _mm512_set1_pd(1.25);
    const __m256i izero = _mm256_setzero_si256();
    const __m256i imax  = _mm256_set1_epi32(NBINS - 1);

    for (std::size_t i = 0; i < n; i += 8) {
        const int rem = (int)std::min<std::size_t>(8, n - i);
        const __mmask8 m = (__mmask8)((1u << rem) - 1u);
        const __m512d vx = _mm512_maskz_loadu_pd(m, x + i);

        const __m512d bt = _mm512_mul_pd(_mm512_sub_pd(vx, one), two);
        __m256i bin = _mm512_cvttpd_epi32(bt);
        bin = _mm256_max_epi32(izero, _mm256_min_epi32(bin, imax));

        const __m512d dbin = _mm512_cvtepi32_pd(bin);
        const __m512d center = _mm512_fmadd_pd(dbin, half, c125);
        const __m512d h = _mm512_sub_pd(vx, center);

        __m512d p = _mm512_i32gather_pd(bin, C[9], 8);
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[8],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[7],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[6],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[5],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[4],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[3],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[2],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[1],8));
        p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[0],8));

        const __m512d pole = reciprocal_nr2(_mm512_sub_pd(vx, one));
        _mm512_mask_storeu_pd(out + i, m, _mm512_add_pd(pole, p));
    }
}
