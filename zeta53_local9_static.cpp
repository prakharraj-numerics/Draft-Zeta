#include "zeta53_local9_table.h"
#include <immintrin.h>
#include <algorithm>
#include <cstddef>

#pragma STDC FP_CONTRACT OFF

// Self-contained runtime kernel for x in (1,100].
// Same formula as the incumbent, exactly recentered offline onto half-unit bins:
//   zeta(x) = 1/(x-1) + sum_{k=0}^9 d_k(A) (x-(A-2))^k.
// The table d_k(A) is generated only from the incumbent corrected coefficients.
// Runtime has no FLINT/Arb dependency, no high-precision setup, no term scheduler.

extern "C" __attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
void zeta53_local9_batch(const double *x, std::size_t n, double *out) {
    using namespace zeta53_local9_table;
    const __m512d one  = _mm512_set1_pd(1.0);
    const __m512d two  = _mm512_set1_pd(2.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d c125 = _mm512_set1_pd(1.25);
    const __m256i izero = _mm256_setzero_si256();
    const __m256i imax  = _mm256_set1_epi32(NBINS - 1);

    for (std::size_t i = 0; i < n; i += 8) {
        const int rem = static_cast<int>(std::min<std::size_t>(8, n - i));
        const __mmask8 valid = static_cast<__mmask8>((1u << rem) - 1u);
        const __m512d vx = _mm512_maskz_loadu_pd(valid, x + i);

        // bin = floor((x-1)/0.5), clamped to [0,197].
        const __m512d bt = _mm512_mul_pd(_mm512_sub_pd(vx, one), two);
        __m256i bin = _mm512_cvttpd_epi32(bt);
        bin = _mm256_max_epi32(izero, _mm256_min_epi32(bin, imax));

        // x-center = x-(1.25+0.5*bin), hence |h| <= 0.25 on the covered domain.
        const __m512d dbin = _mm512_cvtepi32_pd(bin);
        const __m512d center = _mm512_fmadd_pd(dbin, half, c125);
        const __m512d h = _mm512_sub_pd(vx, center);

        __m512d p = _mm512_i32gather_pd(bin, C[9], 8);
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[8], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[7], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[6], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[5], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[4], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[3], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[2], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[1], 8));
        p = _mm512_fmadd_pd(p, h, _mm512_i32gather_pd(bin, C[0], 8));

        const __m512d pole = _mm512_div_pd(one, _mm512_sub_pd(vx, one));
        _mm512_mask_storeu_pd(out + i, valid, _mm512_add_pd(pole, p));
    }
}
