#include <immintrin.h>
#include <stddef.h>
#include <math.h>

/* Exact algebraic resummation of the supplied formula:
   zeta(x)=zeta(2)-sum_{r>=1}(r^-2-r^-x).
   The r>=16 difference tail is evaluated with Euler-Maclaurin.
   Constants/logs are offline >34-decimal-digit values stored as hi+lo binary64.
   custom2 is deliberately not used. */
static const double ZETA2_HI=0x1.a51a6625307d3p+0, ZETA2_LO=0x1.1873d8912200cp-55;
static const double TAIL2_HI=0x1.082aa228320e4p-4, TAIL2_LO=0x1.38bb2f7f787d1p-58;
static const double LN16_HI=0x1.62e42fefa39efp+1, LN16_LO=0x1.abc9e3b39803fp-54;
static const double LOG_HI[14]={0x1.62e42fefa39efp-1,0x1.193ea7aad030bp+0,0x1.62e42fefa39efp+0,0x1.9c041f7ed8d33p+0,0x1.cab0bfa2a2002p+0,0x1.f2272ae325a57p+0,0x1.0a2b23f3bab73p+1,0x1.193ea7aad030bp+1,0x1.26bb1bbb55516p+1,0x1.32ee3b77f374cp+1,0x1.3e116bcd39e7dp+1,0x1.485042b318c51p+1,0x1.51cca16d7bba7p+1,0x1.5aa16394d481fp+1};
static const double LOG_LO[14]={0x1.abc9e3b39803fp-56,-0x1.a256f99caabebp-54,0x1.abc9e3b39803fp-55,0x1.abf7dde94581dp-54,0x1.9136fea076849p-55,0x1.51bda525b3c98p-54,0x1.a06bb56359018p-53,-0x1.a256f99caabebp-53,-0x1.f48ad494ea3e9p-53,-0x1.210e8d00cd605p-53,-0x1.98e40f85bd797p-55,-0x1.798231075c028p-59,0x1.de580f094ce54p-53,0x1.341c89935864ap-59};
static const double INVN2[14]={1./4,1./9,1./16,1./25,1./36,1./49,1./64,1./81,1./100,1./121,1./144,1./169,1./196,1./225};
static const double BC[6]={1./12,-1./720,1./30240,-1./1209600,1./47900160,-691./1307674368000.};
#define ZAVX __attribute__((target("avx512f,fma")))
#define ZAI ZAVX __attribute__((always_inline)) static inline
ZAI __m512d ve(__m512d x){return _mm512_exp_pd(x);}
ZAI __m512d psl(__m512d s,double hi,double lo){__m512d ns=_mm512_sub_pd(_mm512_setzero_pd(),s);__m512d a=_mm512_mul_pd(ns,_mm512_set1_pd(hi));a=_mm512_fmadd_pd(ns,_mm512_set1_pd(lo),a);return ve(a);}
ZAI __m512d tail16(__m512d s){const __m512d one=_mm512_set1_pd(1.), half=_mm512_set1_pd(.5);__m512d sm1=_mm512_sub_pd(s,one), oms=_mm512_sub_pd(one,s);__m512d a=_mm512_mul_pd(oms,_mm512_set1_pd(LN16_HI));a=_mm512_fmadd_pd(oms,_mm512_set1_pd(LN16_LO),a);__m512d n1s=ve(a),ns=psl(s,LN16_HI,LN16_LO);__m512d o=_mm512_add_pd(_mm512_div_pd(n1s,sm1),_mm512_mul_pd(half,ns));__m512d po=s,np=_mm512_mul_pd(ns,_mm512_set1_pd(1./16)),i256=_mm512_set1_pd(1./256);for(int k=0;k<6;k++){if(k){po=_mm512_mul_pd(po,_mm512_add_pd(s,_mm512_set1_pd(2.*k-1)));po=_mm512_mul_pd(po,_mm512_add_pd(s,_mm512_set1_pd(2.*k)));np=_mm512_mul_pd(np,i256);}o=_mm512_fmadd_pd(_mm512_set1_pd(BC[k]),_mm512_mul_pd(po,np),o);}return o;}
ZAI __m512d core(__m512d s){__m512d d=_mm512_setzero_pd(),c=d;for(int j=0;j<14;j++){__m512d px=psl(s,LOG_HI[j],LOG_LO[j]);__m512d term=_mm512_sub_pd(_mm512_set1_pd(INVN2[j]),px),y=_mm512_sub_pd(term,c),t=_mm512_add_pd(d,y);c=_mm512_sub_pd(_mm512_sub_pd(t,d),y);d=t;}__m512d dt=_mm512_sub_pd(_mm512_set1_pd(TAIL2_HI),tail16(s));dt=_mm512_add_pd(dt,_mm512_set1_pd(TAIL2_LO));__m512d z=_mm512_sub_pd(_mm512_set1_pd(ZETA2_HI),d);z=_mm512_sub_pd(z,dt);return _mm512_add_pd(z,_mm512_set1_pd(ZETA2_LO));}
ZAVX static void zeta_avx512(size_t n,const double*x,double*out){size_t i=0;for(;i+8<=n;i+=8)_mm512_storeu_pd(out+i,core(_mm512_loadu_pd(x+i)));if(i<n){__mmask8 m=(__mmask8)((1u<<(n-i))-1u);__m512d s=_mm512_maskz_loadu_pd(m,x+i);_mm512_mask_storeu_pd(out+i,m,core(s));}}
extern "C" void draft_zeta53_batch(size_t n,const double*x,double*out){if(!n||!x||!out)return;if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma")){zeta_avx512(n,x,out);return;}for(size_t i=0;i<n;i++)out[i]=NAN;}
extern "C" int draft_zeta53_batch_uses_avx512(void){return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma");}
