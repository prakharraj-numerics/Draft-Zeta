#include <immintrin.h>
#include <stddef.h>
#include <math.h>

/* Accuracy-only extension of the R1(a) spine through c36.
   R1(a)=a*sum_{m>=0} c_m a^m.  c35 and c36 were regenerated directly
   from the supplied q/r/j coefficient definition at high precision. */
#ifndef R1_TERMS
#define R1_TERMS 37
#endif

static const double C_HI[37] = {
 -0x1.f2de15d1e2a9fp-6, -0x1.0d5e0b74b825bp-5,
 -0x1.c6736c6ebde3ep-7, -0x1.0cc026a9cf23ap-8,
 -0x1.607b5c5e443c3p-10,-0x1.e2af042de36aap-12,
 -0x1.3fbfa0d06f8ddp-13,-0x1.a9cdb927bb87cp-15,
 -0x1.1c33aa684f942p-16,-0x1.7ad3a3fc6c38fp-18,
 -0x1.f9196d5c950d0p-20,-0x1.50bdd4fb58d1bp-21,
 -0x1.c0fc27f147a37p-23,-0x1.2b52d94f9c893p-24,
 -0x1.8f19268f54a8bp-26,-0x1.0a10c2521ae10p-27,
 -0x1.62c103b9b7cbfp-29,-0x1.d9015a3904123p-31,
 -0x1.3b563c235f313p-32,-0x1.a472fadba28b7p-34,
 -0x1.184ca73cd64c8p-35,-0x1.75bb89a67bd08p-37,
 -0x1.f24f623350a9dp-39,-0x1.4c34ec2235583p-40,
 -0x1.baf13ad84740fp-42,-0x1.274b7c902f7dep-43,
 -0x1.89b9fb6ae9fd4p-45,-0x1.067bfcf1f1539p-46,
 -0x1.5dfaa697ec6f7p-48,-0x1.d2a388ca90949p-50,
 -0x1.3717b0870b0dbp-51,-0x1.9eca40b40ebcfp-53,
 -0x1.1486d5cd5f28ap-54,-0x1.70b3c7bc7ee0dp-56,
 -0x1.eb9a5fa5fe812p-58,
 -0x1.47bc3fc3ff00cp-59,
 -0x1.b4faffaffeabap-61
};

static const double C_LO[37] = {
  0x1.5aeebd0ffc3abp-62,-0x1.5107aca33deedp-61,
 -0x1.307ba3fc3117fp-61,-0x1.62117c1516b0ep-64,
  0x1.d1837aa902a43p-64, 0x1.dd8e6f71cf2cdp-67,
  0x1.8e13e08bdbda8p-68, 0x1.f2d4f20e1736ep-69,
 -0x1.cb5c0fcc2f515p-73,-0x1.ab69eaecf9c3fp-73,
 -0x1.001bf9582e290p-74, 0x1.d761f1568ec4ap-75,
  0x1.47f627a8a295ap-79,-0x1.e2fce52ceadadp-78,
  0x1.7160d14f43e60p-82, 0x1.cb0fb61da9c0fp-81,
  0x1.cae3d9ce42c3fp-84, 0x1.b155f94249c0ep-86,
  0x1.92dfdae333c22p-90, 0x1.fe940c2cd935ap-92,
  0x1.5d5b8570d499dp-89, 0x1.a25a268ba49bep-91,
 -0x1.34c8f8c9b192dp-93, 0x1.41a3af1296536p-94,
  0x1.6cfd4a3c81f63p-96,-0x1.7150ed9e513d8p-97,
 -0x1.f952d7d841e33p-99,-0x1.68469d8055deep-101,
  0x1.a5e0befe74078p-102,0x1.371f0f73f88f1p-105,
 -0x1.d408e5b4a2dfep-106,0x1.d87326657f523p-111,
  0x1.3946e5fb07a60p-112,-0x1.3b32e0cb1a392p-110,
  0x1.0665a55fd5868p-112,
  0x1.5ddcf53c9777bp-114,
 -0x1.c16cb04e9ec26p-115
};

#define ZAVX __attribute__((target("avx512f,fma")))
#define ZAI ZAVX __attribute__((always_inline)) static inline
typedef struct { __m512d hi,lo; } dd8;
ZAI dd8 renorm8(__m512d a,__m512d b){dd8 z;z.hi=_mm512_add_pd(a,b);z.lo=_mm512_sub_pd(b,_mm512_sub_pd(z.hi,a));return z;}
ZAI dd8 mul_d8(dd8 x,__m512d y){const __m512d p=_mm512_mul_pd(x.hi,y);__m512d e=_mm512_fmsub_pd(x.hi,y,p);e=_mm512_fmadd_pd(x.lo,y,e);return renorm8(p,e);}
ZAI dd8 add_const8(dd8 x,double chi,double clo){const __m512d c=_mm512_set1_pd(chi);const __m512d s=_mm512_add_pd(x.hi,c);const __m512d bb=_mm512_sub_pd(s,x.hi);const __m512d e1=_mm512_add_pd(_mm512_sub_pd(x.hi,_mm512_sub_pd(s,bb)),_mm512_sub_pd(c,bb));const __m512d e=_mm512_add_pd(_mm512_add_pd(x.lo,e1),_mm512_set1_pd(clo));return renorm8(s,e);}
ZAI dd8 step8(dd8 p,__m512d a,double hi,double lo){return add_const8(mul_d8(p,a),hi,lo);}
ZAI dd8 poly8(__m512d a){dd8 p={_mm512_set1_pd(C_HI[R1_TERMS-1]),_mm512_set1_pd(C_LO[R1_TERMS-1])};for(int k=R1_TERMS-2;k>=0;--k)p=step8(p,a,C_HI[k],C_LO[k]);return mul_d8(p,a);}
ZAVX static void batch(size_t n,const double*a,double*out){size_t i=0;for(;i+8<=n;i+=8){__m512d av=_mm512_loadu_pd(a+i);dd8 z=poly8(av);_mm512_storeu_pd(out+i,_mm512_add_pd(z.hi,z.lo));}if(i<n){__mmask8 m=(__mmask8)((1u<<(n-i))-1u);__m512d av=_mm512_maskz_loadu_pd(m,a+i);dd8 z=poly8(av);_mm512_mask_storeu_pd(out+i,m,_mm512_add_pd(z.hi,z.lo));}}
extern "C" void draft_zeta53_batch(size_t n,const double*a,double*out){if(!n||!a||!out)return;if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma")){batch(n,a,out);return;}for(size_t i=0;i<n;++i)out[i]=NAN;}
extern "C" int draft_zeta53_batch_uses_avx512(void){return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma");}
