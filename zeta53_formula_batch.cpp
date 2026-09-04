#include <immintrin.h>
#include <stddef.h>
#include <math.h>
#include <stdint.h>

/* Resummed form of the user's identity:
   zeta(x)=zeta(2)-sum_r (r^-2-r^-x), with the infinite tail evaluated by
   Euler-Maclaurin.  Runtime constants use double-double-style hi/lo splits
   generated offline at >34 significant decimal digits.  No custom2. */

static const double ZETA2_HI = 0x1.a51a6625307d3p+0;
static const double ZETA2_LO = 0x1.1873d8912200cp-55;
static const double TAIL2_HI = 0x1.082aa228320e4p-4; /* sum_{n>=16} n^-2 */
static const double TAIL2_LO = 0x1.38bb2f7f787d1p-58;
static const double LN16_HI  = 0x1.62e42fefa39efp+1;
static const double LN16_LO  = 0x1.abc9e3b39803fp-54;

static const double LOG_HI[14] = {
0x1.62e42fefa39efp-1,0x1.193ea7aad030bp+0,0x1.62e42fefa39efp+0,
0x1.9c041f7ed8d33p+0,0x1.cab0bfa2a2002p+0,0x1.f2272ae325a57p+0,
0x1.0a2b23f3bab73p+1,0x1.193ea7aad030bp+1,0x1.26bb1bbb55516p+1,
0x1.32ee3b77f374cp+1,0x1.3e116bcd39e7dp+1,0x1.485042b318c51p+1,
0x1.51cca16d7bba7p+1,0x1.5aa16394d481fp+1};
static const double LOG_LO[14] = {
0x1.abc9e3b39803fp-56,-0x1.a256f99caabebp-54,0x1.abc9e3b39803fp-55,
0x1.abf7dde94581dp-54,0x1.9136fea076849p-55,0x1.51bda525b3c98p-54,
0x1.a06bb56359018p-53,-0x1.a256f99caabebp-53,-0x1.f48ad494ea3e9p-53,
-0x1.210e8d00cd605p-53,-0x1.98e40f85bd797p-55,-0x1.798231075c028p-59,
0x1.de580f094ce54p-53,0x1.341c89935864ap-59};
static const double INVN2[14] = {
1.0/4.0,1.0/9.0,1.0/16.0,1.0/25.0,1.0/36.0,1.0/49.0,1.0/64.0,
1.0/81.0,1.0/100.0,1.0/121.0,1.0/144.0,1.0/169.0,1.0/196.0,1.0/225.0};

/* B_2k/(2k)! for k=1..6. */
static const double BC[6] = {
 1.0/12.0, -1.0/720.0, 1.0/30240.0, -1.0/1209600.0,
 1.0/47900160.0, -691.0/1307674368000.0};

__attribute__((target("avx512f,fma"),always_inline)) static inline __m512d vexp(__m512d x) {
    return _mm512_exp_pd(x);
}

__attribute__((target("avx512f,fma"),always_inline)) static inline __m512d pow_from_split_log(__m512d s, double hi, double lo) {
    __m512d a = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), s), _mm512_set1_pd(hi));
    a = _mm512_fmadd_pd(_mm512_sub_pd(_mm512_setzero_pd(), s), _mm512_set1_pd(lo), a);
    return vexp(a);
}

__attribute__((target("avx512f,fma"),always_inline)) static inline __m512d em_tail16(__m512d s) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5);
    __m512d em1 = _mm512_sub_pd(s, one);
    __m512d a1 = _mm512_mul_pd(_mm512_sub_pd(one, s), _mm512_set1_pd(LN16_HI));
    a1 = _mm512_fmadd_pd(_mm512_sub_pd(one, s), _mm512_set1_pd(LN16_LO), a1);
    __m512d n1s = vexp(a1);
    __m512d ns = pow_from_split_log(s, LN16_HI, LN16_LO);
    __m512d out = _mm512_add_pd(_mm512_div_pd(n1s, em1), _mm512_mul_pd(half, ns));

    __m512d poch = s;
    __m512d npow = _mm512_mul_pd(ns, _mm512_set1_pd(1.0/16.0)); /* 16^(-s-1) */
    const __m512d inv256 = _mm512_set1_pd(1.0/256.0);
    for (int k=0;k<6;k++) {
        if (k>0) {
            double a = (double)(2*k-1), b = (double)(2*k);
            poch = _mm512_mul_pd(poch, _mm512_add_pd(s,_mm512_set1_pd(a)));
            poch = _mm512_mul_pd(poch, _mm512_add_pd(s,_mm512_set1_pd(b)));
            npow = _mm512_mul_pd(npow, inv256);
        }
        out = _mm512_fmadd_pd(_mm512_set1_pd(BC[k]), _mm512_mul_pd(poch,npow), out);
    }
    return out;
}

__attribute__((target("avx512f,fma"))) static void zeta_avx512(size_t n, const double *x, double *out) {
    size_t i=0;
    for (; i+8<=n; i+=8) {
        __m512d s = _mm512_loadu_pd(x+i);
        __m512d d = _mm512_setzero_pd();
        __m512d c = _mm512_setzero_pd();
        for (int j=0;j<14;j++) {
            __m512d px = pow_from_split_log(s, LOG_HI[j], LOG_LO[j]);
            __m512d term = _mm512_sub_pd(_mm512_set1_pd(INVN2[j]), px);
            __m512d y = _mm512_sub_pd(term,c);
            __m512d t = _mm512_add_pd(d,y);
            c = _mm512_sub_pd(_mm512_sub_pd(t,d),y);
            d = t;
        }
        __m512d tailx = em_tail16(s);
        __m512d dtail = _mm512_sub_pd(_mm512_set1_pd(TAIL2_HI), tailx);
        dtail = _mm512_add_pd(dtail, _mm512_set1_pd(TAIL2_LO));
        __m512d z = _mm512_sub_pd(_mm512_set1_pd(ZETA2_HI), d);
        z = _mm512_sub_pd(z, dtail);
        z = _mm512_add_pd(z, _mm512_set1_pd(ZETA2_LO));
        _mm512_storeu_pd(out+i,z);
    }
    for (; i<n; ++i) {
        /* tiny scalar tail: batches in production are expected to be SIMD-sized;
           correctness fallback uses Boost in benchmark wrapper for now. */
        out[i] = NAN;
    }
}

extern "C" void draft_zeta53_batch(size_t n, const double *x, double *out) {
    if (!n || !x || !out) return;
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        zeta_avx512(n,x,out); return;
    }
    for (size_t i=0;i<n;i++) out[i]=NAN;
}

extern "C" int draft_zeta53_batch_uses_avx512(void) {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
}
