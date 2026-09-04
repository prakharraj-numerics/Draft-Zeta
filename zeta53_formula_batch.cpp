#include <immintrin.h>
#include <stddef.h>
#include <math.h>
#include <mutex>

/*
 * Formula-faithful AVX-512 research kernel.
 *
 * We intentionally keep the user's mathematical spine:
 *
 *   zeta(x) = pi^2/6
 *             - sum_{m>=1} (x-2)^m/m!
 *               sum_{r>=1} (log r)^m / r^x .
 *
 * IMPORTANT: do NOT algebraically resum the m-series to r^(x-2)-1 and do
 * not cancel pi^2/6 against sum r^-2.  That shortcut merely reconstructs
 * the defining Dirichlet series and is forbidden for this project.
 *
 * The finite research kernel below only changes evaluation order: for each
 * cached integer r it forms r^-x once, then accumulates the individual
 * inner moments S_m(x).  The outer m-series is still formed explicitly.
 * No exponential-series collapse is used.
 *
 * Convergence of the literal formula requires x > 3/2.  Inputs at or below
 * 3/2 are therefore rejected instead of silently switching formulas.
 *
 * This file is now structure-correct, not yet the final 1-ULP production
 * kernel.  The remaining accuracy work is a faithful treatment of the
 * r->infinity tail of every S_m; simply truncating r is not sufficient.
 * custom2 remains deliberately unused.
 */

static const double ZETA2_HI = 0x1.a51a6625307d3p+0;
static const double ZETA2_LO = 0x1.1873d8912200cp-55;

/* Research limits only.  They are intentionally centralized so the later
 * rigorous tail/term-count analysis can replace them without changing the
 * machinery. */
static constexpr int ZETA53_MMAX = 192;
static constexpr int ZETA53_RMAX = 2048;

/* log(integer) is invariant across every input and every batch.  Cache it
 * once; no log() occurs in the hot formula evaluation after initialization.
 * The final production table will be generated offline as hi+lo constants. */
static double LOG_R[ZETA53_RMAX + 1];
static std::once_flag LOG_R_ONCE;

static void init_log_cache(void)
{
    LOG_R[0] = 0.0;
    LOG_R[1] = 0.0;                 /* r=1 contributes zero for m>=1 */
    for (int r = 2; r <= ZETA53_RMAX; ++r)
        LOG_R[r] = log((double)r);
}

#define ZAVX __attribute__((target("avx512f,fma")))
#define ZAI  ZAVX __attribute__((always_inline)) static inline

ZAI __m512d vexp_hi(__m512d x)
{
    return _mm512_exp_pd(x);
}

/* Compute one 8-lane block while preserving the original m/r structure.
 * Computationally we traverse r first only to reuse r^-x.  What is stored
 * in moment[m-1] is exactly the finite inner sum
 *      sum_r (log r)^m / r^x,
 * and only after those moments exist do we form the outer m-series. */
ZAVX static __m512d zeta_formula_block(__m512d x)
{
    __m512d moment[ZETA53_MMAX];
    for (int m = 0; m < ZETA53_MMAX; ++m)
        moment[m] = _mm512_setzero_pd();

    const __m512d zero = _mm512_setzero_pd();

    for (int r = 2; r <= ZETA53_RMAX; ++r) {
        const double lr = LOG_R[r];
        const __m512d vlr = _mm512_set1_pd(lr);
        const __m512d negxlogr = _mm512_mul_pd(_mm512_sub_pd(zero, x), vlr);
        const __m512d rx = vexp_hi(negxlogr);      /* r^-x, once per r */

        __m512d lp = vlr;                         /* (log r)^1 */
        for (int m = 0; m < ZETA53_MMAX; ++m) {
            moment[m] = _mm512_fmadd_pd(rx, lp, moment[m]);
            lp = _mm512_mul_pd(lp, vlr);
        }
    }

    const __m512d h = _mm512_sub_pd(x, _mm512_set1_pd(2.0));
    __m512d weight = h;                           /* (x-2)^1 / 1! */
    __m512d outer = _mm512_setzero_pd();
    __m512d comp  = _mm512_setzero_pd();

    for (int m = 1; m <= ZETA53_MMAX; ++m) {
        const __m512d term = _mm512_mul_pd(weight, moment[m - 1]);

        /* Compensated accumulation of the explicit outer m-series. */
        const __m512d y = _mm512_sub_pd(term, comp);
        const __m512d t = _mm512_add_pd(outer, y);
        comp = _mm512_sub_pd(_mm512_sub_pd(t, outer), y);
        outer = t;

        if (m != ZETA53_MMAX)
            weight = _mm512_mul_pd(weight,
                                   _mm512_mul_pd(h,
                                       _mm512_set1_pd(1.0 / (double)(m + 1))));
    }

    __m512d z = _mm512_sub_pd(_mm512_set1_pd(ZETA2_HI), outer);
    z = _mm512_add_pd(z, _mm512_set1_pd(ZETA2_LO));
    return z;
}

ZAVX static void zeta_avx512(size_t n, const double *x, double *out)
{
    size_t i = 0;
    const __m512d lim = _mm512_set1_pd(1.5);

    for (; i + 8 <= n; i += 8) {
        const __m512d xv = _mm512_loadu_pd(x + i);
        const __mmask8 valid = _mm512_cmp_pd_mask(xv, lim, _CMP_GT_OQ);
        __m512d z = zeta_formula_block(xv);
        z = _mm512_mask_mov_pd(_mm512_set1_pd(NAN), valid, z);
        _mm512_storeu_pd(out + i, z);
    }

    if (i < n) {
        const __mmask8 live = (__mmask8)((1u << (n - i)) - 1u);
        const __m512d xv = _mm512_maskz_loadu_pd(live, x + i);
        const __mmask8 domain = _mm512_cmp_pd_mask(xv, lim, _CMP_GT_OQ);
        const __mmask8 valid = (__mmask8)(live & domain);
        __m512d z = zeta_formula_block(xv);
        z = _mm512_mask_mov_pd(_mm512_set1_pd(NAN), valid, z);
        _mm512_mask_storeu_pd(out + i, live, z);
    }
}

extern "C" void draft_zeta53_batch(size_t n, const double *x, double *out)
{
    if (!n || !x || !out)
        return;

    std::call_once(LOG_R_ONCE, init_log_cache);

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        zeta_avx512(n, x, out);
        return;
    }

    /* No hidden alternate zeta formula on the scalar path. */
    for (size_t i = 0; i < n; ++i)
        out[i] = NAN;
}

extern "C" int draft_zeta53_batch_uses_avx512(void)
{
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
}
