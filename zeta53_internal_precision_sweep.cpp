#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int COEFF_BITS = 8192;   // offline manufacture only
static constexpr int REF_BITS   = 2048;   // independent accuracy oracle
static constexpr int MAXT       = 900;
static constexpr int N          = 5000;

static inline int candidate_terms(double a)
{
    double t;
    if (a < 3.0)       t = 10.0 + 10.5*a - 2.14*a*a;
    else if (a <= 10.) t = 11.0 + 5.0*a;
    else if (a <= 40.) t = 11.0 + 4.54*a + 0.0485*a*a;
    else               t = -37.0 + 6.85*a + 0.0194*a*a;
    return std::max(6, (int)std::ceil(t) + 1);
}

static uint64_t ordkey(double x)
{
    union { double d; uint64_t u; } v{x};
    return (v.u >> 63) ? ~v.u : (v.u | 0x8000000000000000ULL);
}
static uint64_t ulpdist(double a, double b)
{
    if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
    uint64_t x = ordkey(a), y = ordkey(b);
    return x > y ? x-y : y-x;
}

int main()
{
    std::vector<double> A(N), R(N);
    std::mt19937_64 gi(0x5052454353574545ULL), go(0x5a45544135334249ULL);
    std::uniform_real_distribution<double> ui(1.0e-12, 3.0);
    std::uniform_real_distribution<double> uo(3.0, 102.0);
    for (int i=0;i<N/2;i++) {
        double x=ui(gi); if (x>=3.0) x=std::nextafter(3.0,0.0); A[i]=x;
    }
    for (int i=N/2;i<N;i++) {
        double x=uo(go); if (x<=3.0) x=std::nextafter(3.0,INFINITY); A[i]=x;
    }

    // Keep the count exactly 5000 while forcing difficult boundary/large-a cases in.
    const double stress_in[] = {1e-12,1e-9,1e-6,0.01,0.1,1.0,2.0,2.5,2.9,2.99,2.9999,std::nextafter(3.0,0.0)};
    const double stress_out[] = {std::nextafter(3.0,INFINITY),3.000000001,3.001,3.01,3.1,4.0,5.2,10.0,20.0,40.0,50.0,75.0,81.764554025359956,100.0,101.0,102.0};
    for (size_t i=0;i<sizeof(stress_in)/sizeof(stress_in[0]);i++) A[i]=stress_in[i];
    for (size_t i=0;i<sizeof(stress_out)/sizeof(stress_out[0]);i++) A[N/2+(int)i]=stress_out[i];

    // Generate pole-removed coefficients once at very high precision.
    arb_poly_t s,z; arb_poly_init(s); arb_poly_init(z);
    arb_t one,p3,tmp,aa,ss,ref; arb_init(one);arb_init(p3);arb_init(tmp);arb_init(aa);arb_init(ss);arb_init(ref);
    arb_one(one); arb_poly_set_coeff_si(s,0,-2); arb_poly_set_coeff_si(s,1,1);
    std::fprintf(stderr,"[setup] generating %d pole-removed coefficients at %d bits...\n",MAXT,COEFF_BITS);
    arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);
    std::vector<arf_struct> exact_coeff(MAXT);
    arb_set_ui(p3,9);
    for (int m=0;m<MAXT;m++) {
        arf_init(exact_coeff.data()+m);
        arb_poly_get_coeff_arb(tmp,z,m+1);
        arb_t inv; arb_init(inv); arb_inv(inv,p3,COEFF_BITS); arb_add(tmp,tmp,inv,COEFF_BITS); arb_clear(inv);
        arf_set_round(exact_coeff.data()+m,arb_midref(tmp),COEFF_BITS,ARF_RND_NEAR);
        arb_mul_ui(p3,p3,3,COEFF_BITS);
    }

    // Independent references, loading the binary64 input exactly before subtracting 2.
    for (int i=0;i<N;i++) {
        arb_set_d(aa,A[i]); arb_sub_ui(ss,aa,2,REF_BITS); arb_zeta(ref,ss,REF_BITS);
        R[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);
    }

    std::printf("N=%d COEFF_BITS=%d REF_BITS=%d DOMAIN=(0,102],a!=3\n",N,COEFF_BITS,REF_BITS);
    int global_min_pass = 0;
    std::vector<int> gt1_by_bits(385,-1);
    std::vector<unsigned long long> maxulp_by_bits(385,0);

    // Exhaust every integer working precision from 53 through 384 bits.
    for (int bits=53; bits<=384; ++bits) {
        std::vector<arf_struct> c(MAXT), av(N);
        for (int m=0;m<MAXT;m++) { arf_init(c.data()+m); arf_set_round(c.data()+m,exact_coeff.data()+m,bits,ARF_RND_NEAR); }
        for (int i=0;i<N;i++) { arf_init(av.data()+i); arf_set_d(av.data()+i,A[i]); }
        arf_t p,base,third; arf_init(p);arf_init(base);arf_init(third);
        arf_set_ui(third,1); arf_div_ui(third,third,3,bits,ARF_RND_NEAR);

        uint64_t mx=0; int gt1=0,gt2=0,bad=0,worst=0;
        for (int i=0;i<N;i++) {
            const int T=candidate_terms(A[i]);
            arf_set(p,c.data()+T-1);
            for (int m=T-2;m>=0;m--) arf_fma(p,p,av.data()+i,c.data()+m,bits,ARF_RND_NEAR);
            arf_mul(p,p,av.data()+i,bits,ARF_RND_NEAR);
            arf_sub_ui(base,av.data()+i,3,bits,ARF_RND_NEAR);
            arf_ui_div(base,1,base,bits,ARF_RND_NEAR);
            arf_add(p,p,base,bits,ARF_RND_NEAR);
            arf_add(p,p,third,bits,ARF_RND_NEAR);
            double y=arf_get_d(p,ARF_RND_NEAR);
            uint64_t e=ulpdist(y,R[i]);
            if (e==UINT64_MAX) { bad++; continue; }
            if (e>mx) { mx=e; worst=i; }
            gt1 += e>1; gt2 += e>2;
        }
        gt1_by_bits[bits]=gt1; maxulp_by_bits[bits]=(unsigned long long)mx;
        if (gt1==0 && bad==0 && global_min_pass==0) global_min_pass=bits;
        if (bits%8==0 || bits==53 || gt1==0 || (global_min_pass && bits<=global_min_pass+4))
            std::printf("BITS=%d MAX_ULP=%llu GT1=%d/%d GT2=%d/%d NONFINITE=%d WORST_A=%.17g T=%d\n",
                bits,(unsigned long long)mx,gt1,N,gt2,N,bad,A[worst],candidate_terms(A[worst]));

        arf_clear(third);arf_clear(base);arf_clear(p);
        for (int i=0;i<N;i++) arf_clear(av.data()+i);
        for (int m=0;m<MAXT;m++) arf_clear(c.data()+m);
    }

    std::printf("MIN_PASS_BITS=%d\n",global_min_pass);
    if (global_min_pass) {
        int lo=std::max(53,global_min_pass-5), hi=std::min(384,global_min_pass+5);
        for (int b=lo;b<=hi;b++) std::printf("BOUNDARY bits=%d gt1=%d maxulp=%llu\n",b,gt1_by_bits[b],maxulp_by_bits[b]);
    }

    for (int m=0;m<MAXT;m++) arf_clear(exact_coeff.data()+m);
    arb_clear(ref);arb_clear(ss);arb_clear(aa);arb_clear(tmp);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();
    return 0;
}
