#include "zeta53_local9_table.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>

#pragma STDC FP_CONTRACT OFF

// Gamma-derived optimization attack on the exact Crypto spine.
// Mathematics is unchanged:
//   zeta(x) = 1/(x-1) + sum_{k=0}^9 d_k(A) (x-(A-2))^k.
// Only implementation changes are tested here: reciprocal evaluation,
// coefficient layout/indexing, and multi-vector ILP.

namespace {
using namespace zeta53_local9_table;

inline constexpr auto CELL = [] {
    std::array<std::array<double, DEGREE + 1>, NBINS> a{};
    for (int b = 0; b < NBINS; ++b)
        for (int k = 0; k <= DEGREE; ++k)
            a[b][k] = C[k][b];
    return a;
}();

static inline __m512d reciprocal_nr2(__m512d d) {
    // Same reciprocal 1/d as the production pole, evaluated with AVX-512
    // rcp14 followed by two Newton refinements: r <- r*(2-d*r).
    const __m512d two = _mm512_set1_pd(2.0);
    __m512d r = _mm512_rcp14_pd(d);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(d, r, two));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(d, r, two));
    return r;
}

static inline void prep(__m512d vx, __m256i &bin, __m512d &h) {
    const __m512d one  = _mm512_set1_pd(1.0);
    const __m512d two  = _mm512_set1_pd(2.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d c125 = _mm512_set1_pd(1.25);
    const __m256i izero = _mm256_setzero_si256();
    const __m256i imax  = _mm256_set1_epi32(NBINS - 1);
    const __m512d bt = _mm512_mul_pd(_mm512_sub_pd(vx, one), two);
    bin = _mm512_cvttpd_epi32(bt);
    bin = _mm256_max_epi32(izero, _mm256_min_epi32(bin, imax));
    const __m512d dbin = _mm512_cvtepi32_pd(bin);
    const __m512d center = _mm512_fmadd_pd(dbin, half, c125);
    h = _mm512_sub_pd(vx, center);
}

static inline __m512d horner_coeffmajor(__m256i bin, __m512d h) {
    __m512d p = _mm512_i32gather_pd(bin, C[9], 8);
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[8],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[7],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[6],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[5],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[4],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[3],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[2],8));
    p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[1],8));
    return _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(bin,C[0],8));
}

static inline __m512d horner_cellmajor(__m256i bin, __m512d h) {
    const __m256i stride = _mm256_set1_epi32(DEGREE + 1);
    __m256i ix = _mm256_add_epi32(_mm256_mullo_epi32(bin, stride), _mm256_set1_epi32(9));
    const double *base = &CELL[0][0];
    __m512d p = _mm512_i32gather_pd(ix, base, 8);
#define STEP(K) do { ix = _mm256_add_epi32(_mm256_mullo_epi32(bin,stride),_mm256_set1_epi32(K)); p = _mm512_fmadd_pd(p,h,_mm512_i32gather_pd(ix,base,8)); } while(0)
    STEP(8); STEP(7); STEP(6); STEP(5); STEP(4); STEP(3); STEP(2); STEP(1); STEP(0);
#undef STEP
    return p;
}
}

extern "C" __attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
void crypto_attack_coeff_rcp2(const double *x, std::size_t n, double *out) {
    const __m512d one = _mm512_set1_pd(1.0);
    for (std::size_t i=0;i<n;i+=8) {
        int rem=(int)std::min<std::size_t>(8,n-i); __mmask8 m=(__mmask8)((1u<<rem)-1u);
        __m512d vx=_mm512_maskz_loadu_pd(m,x+i),h; __m256i bin; prep(vx,bin,h);
        __m512d p=horner_coeffmajor(bin,h);
        __m512d pole=reciprocal_nr2(_mm512_sub_pd(vx,one));
        _mm512_mask_storeu_pd(out+i,m,_mm512_add_pd(pole,p));
    }
}

extern "C" __attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
void crypto_attack_cell_div(const double *x, std::size_t n, double *out) {
    const __m512d one = _mm512_set1_pd(1.0);
    for (std::size_t i=0;i<n;i+=8) {
        int rem=(int)std::min<std::size_t>(8,n-i); __mmask8 m=(__mmask8)((1u<<rem)-1u);
        __m512d vx=_mm512_maskz_loadu_pd(m,x+i),h; __m256i bin; prep(vx,bin,h);
        __m512d p=horner_cellmajor(bin,h);
        __m512d pole=_mm512_div_pd(one,_mm512_sub_pd(vx,one));
        _mm512_mask_storeu_pd(out+i,m,_mm512_add_pd(pole,p));
    }
}

extern "C" __attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
void crypto_attack_cell_rcp2(const double *x, std::size_t n, double *out) {
    const __m512d one = _mm512_set1_pd(1.0);
    for (std::size_t i=0;i<n;i+=8) {
        int rem=(int)std::min<std::size_t>(8,n-i); __mmask8 m=(__mmask8)((1u<<rem)-1u);
        __m512d vx=_mm512_maskz_loadu_pd(m,x+i),h; __m256i bin; prep(vx,bin,h);
        __m512d p=horner_cellmajor(bin,h);
        __m512d pole=reciprocal_nr2(_mm512_sub_pd(vx,one));
        _mm512_mask_storeu_pd(out+i,m,_mm512_add_pd(pole,p));
    }
}

extern "C" __attribute__((noinline,target("avx512f,avx512dq,avx512vl,avx2,fma")))
void crypto_attack_cell_rcp2_u2(const double *x, std::size_t n, double *out) {
    const __m512d one = _mm512_set1_pd(1.0);
    std::size_t i=0;
    for (; i+16<=n; i+=16) {
        __m512d x0=_mm512_loadu_pd(x+i), x1=_mm512_loadu_pd(x+i+8), h0,h1; __m256i b0,b1;
        prep(x0,b0,h0); prep(x1,b1,h1);
        // Two independent vectors expose ILP around gather/FMA and reciprocal chains.
        __m512d p0=horner_cellmajor(b0,h0);
        __m512d p1=horner_cellmajor(b1,h1);
        __m512d q0=reciprocal_nr2(_mm512_sub_pd(x0,one));
        __m512d q1=reciprocal_nr2(_mm512_sub_pd(x1,one));
        _mm512_storeu_pd(out+i,_mm512_add_pd(p0,q0));
        _mm512_storeu_pd(out+i+8,_mm512_add_pd(p1,q1));
    }
    if(i<n) crypto_attack_cell_rcp2(x+i,n-i,out+i);
}
