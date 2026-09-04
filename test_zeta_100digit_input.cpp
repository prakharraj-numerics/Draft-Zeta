#include <flint/arb.h>
#include <flint/fmpq.h>
#include <cstdlib>
#include <cstdio>

#define main zeta_bench_original_main
#include "bench_zeta_perinput_vs_xeon.cpp"
#undef main

int main() {
    const char *xstr = "5.9419857130432093003992359016041690900861028398146838103685586021483884727797660108966186794963460823";
    const char *ratio = "59419857130432093003992359016041690900861028398146838103685586021483884727797660108966186794963460823/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

    double x = std::strtod(xstr, nullptr);
    Runtime rt;
    double a = x + 2.0;
    int bits = rt.precision(a);
    int T = terms(a);
    double ours = rt.eval_cached_x(x, bits, T);
    double ref_rounded_input = rt.reference_x(x);

    fmpq_t q;
    fmpq_init(q);
    if (fmpq_set_str(q, ratio, 10) != 0) {
        std::fprintf(stderr, "failed to parse exact rational\n");
        return 2;
    }
    fmpq_canonicalise(q);

    arb_t xe, ze;
    arb_init(xe); arb_init(ze);
    arb_set_fmpq(xe, q, 4096);
    arb_zeta(ze, xe, 4096);
    double ref_exact_input = arf_get_d(arb_midref(ze), ARF_RND_NEAR);

    std::printf("X_EXACT=%s\n", xstr);
    std::printf("X_BINARY64=%.17g\n", x);
    std::printf("WORK_BITS=%d TERMS=%d\n", bits, T);
    std::printf("OURS=%.17g\n", ours);
    std::printf("REF_ZETA_BINARY64_INPUT=%.17g\n", ref_rounded_input);
    std::printf("REF_ZETA_EXACT_100DIGIT_INPUT=%.17g\n", ref_exact_input);
    std::printf("ULP_OURS_VS_BINARY64_REF=%llu\n", (unsigned long long)ulp(ours, ref_rounded_input));
    std::printf("ULP_OURS_VS_EXACT_INPUT_REF=%llu\n", (unsigned long long)ulp(ours, ref_exact_input));
    std::printf("ULP_BINARY64_REF_VS_EXACT_INPUT_REF=%llu\n", (unsigned long long)ulp(ref_rounded_input, ref_exact_input));
    std::printf("EXACT_ZETA_80DIGITS=");
    arb_printn(ze, 80, ARB_STR_NO_RADIUS);
    std::printf("\n");

    arb_clear(ze); arb_clear(xe); fmpq_clear(q);
    flint_cleanup();
    return 0;
}
