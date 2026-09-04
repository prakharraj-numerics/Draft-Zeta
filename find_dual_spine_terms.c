#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXT 1400
#define N 1000
#define PREC 8192

static double chi[MAXT], clo[MAXT], thi[MAXT], tlo[MAXT];

typedef struct { double hi, lo; } dd;
static inline dd renorm(double a,double b){ dd z; z.hi=a+b; z.lo=b-(z.hi-a); return z; }
static inline dd mul_d(dd x,double y){ double p=x.hi*y; double e=fma(x.hi,y,-p)+x.lo*y; return renorm(p,e); }
static inline dd add_c(dd x,double hi,double lo){ double s=x.hi+hi; double bb=s-x.hi; double e1=(x.hi-(s-bb))+(hi-bb); return renorm(s, x.lo+e1+lo); }
static dd eval_dd(const double *hi,const double *lo,int T,double a){ dd p={hi[T-1],lo[T-1]}; for(int k=T-2;k>=0;k--) p=add_c(mul_d(p,a),hi[k],lo[k]); return mul_d(p,a); }
static double eval(const double *hi,const double *lo,int T,double a){ dd p=eval_dd(hi,lo,T,a); return p.hi+p.lo; }
static double add_scalar_dd(dd x,double y){ return add_c(x,y,0.0).hi + add_c(x,y,0.0).lo; }
static double eval_case2(int T,double a){
  dd p=eval_dd(thi,tlo,T,a);
  p=add_c(p,1.0/(a-3.0),0.0);
  p=add_c(p,1.0/3.0,0.0);
  return p.hi+p.lo;
}
static uint64_t key(double x){ union{double d; uint64_t u;}v={x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulp(double a,double b){ uint64_t x=key(a),y=key(b); return x>y?x-y:y-x; }
static double arb_to_hi_lo(const arb_t x,double *lo){ double h=arf_get_d(arb_midref(x),ARF_RND_NEAR); arb_t t,hb; arb_init(t); arb_init(hb); arb_set(t,x); arb_set_d(hb,h); arb_sub(t,t,hb,PREC); *lo=arf_get_d(arb_midref(t),ARF_RND_NEAR); arb_clear(hb); arb_clear(t); return h; }
static uint64_t rng=0x9e3779b97f4a7c15ULL; static uint64_t nextu(void){ rng^=rng>>12; rng^=rng<<25; rng^=rng>>27; return rng*2685821657736338717ULL; }
static double uni(void){ return (nextu()>>11)*(1.0/9007199254740992.0); }

int main(void){
  arb_poly_t s,z; arb_t one,c,t,pow3,ss,aa,ref; arb_poly_init(s); arb_poly_init(z); arb_init(one); arb_init(c); arb_init(t); arb_init(pow3); arb_init(ss); arb_init(aa); arb_init(ref);
  arb_one(one); arb_poly_set_coeff_si(s,0,-2); arb_poly_set_coeff_si(s,1,1);
  fprintf(stderr,"Generating %d zeta Taylor coefficients at %d bits...\n",MAXT,PREC);
  arb_poly_zeta_series(z,s,one,0,MAXT+1,PREC);
  arb_set_ui(pow3,9);
  for(int m=0;m<MAXT;m++){
    arb_poly_get_coeff_arb(c,z,m+1);
    chi[m]=arb_to_hi_lo(c,&clo[m]);
    arb_inv(t,pow3,PREC); arb_add(t,t,c,PREC);
    thi[m]=arb_to_hi_lo(t,&tlo[m]);
    arb_mul_ui(pow3,pow3,3,PREC);
  }
  double ain[N], aout[N], rin[N], rout[N];
  for(int i=0;i<N;i++){
    ain[i]=-2.9+5.8*uni(); if(fabs(ain[i])<1e-12) ain[i]=0.125;
    double x=2.0+98.0*uni(); aout[i]=x+2.0;

    /* Reference must be zeta(a-2) for the exact binary64 a.  Do NOT round
       a-2 in binary64 before converting to Arb. */
    arb_set_d(aa,ain[i]); arb_sub_ui(ss,aa,2,PREC); arb_zeta(ref,ss,PREC); rin[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);
    arb_set_d(aa,aout[i]); arb_sub_ui(ss,aa,2,PREC); arb_zeta(ref,ss,PREC); rout[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);
  }
  int best1=0,best2=0;
  for(int T=1;T<=MAXT;T++){
    uint64_t mx=0; int bad=0; for(int i=0;i<N;i++){ uint64_t u=ulp(eval(chi,clo,T,ain[i]),rin[i]); if(u>mx)mx=u; if(u>1)bad++; }
    if(!bad){best1=T; printf("CASE1_MIN_TERMS=%d CASE1_MAX_ULP=%llu\n",T,(unsigned long long)mx); break;}
  }
  for(int T=1;T<=MAXT;T++){
    uint64_t mx=0; int bad=0; for(int i=0;i<N;i++){ double y=eval_case2(T,aout[i]); uint64_t u=ulp(y,rout[i]); if(u>mx)mx=u; if(u>1)bad++; }
    if(!bad){best2=T; printf("CASE2_MIN_TERMS=%d CASE2_MAX_ULP=%llu\n",T,(unsigned long long)mx); break;}
  }
  if(!best1) printf("CASE1_NO_PASS_UP_TO=%d\n",MAXT); if(!best2) printf("CASE2_NO_PASS_UP_TO=%d\n",MAXT);
  if(best1){ uint64_t mx=0; int gt1=0; for(int i=0;i<N;i++){uint64_t u=ulp(eval(chi,clo,best1,ain[i]),rin[i]); if(u>mx)mx=u; gt1+=u>1;} printf("CASE1_VERIFY N=%d TERMS=%d MAX_ULP=%llu GT1=%d\n",N,best1,(unsigned long long)mx,gt1); }
  if(best2){ uint64_t mx=0; int gt1=0; for(int i=0;i<N;i++){double y=eval_case2(best2,aout[i]);uint64_t u=ulp(y,rout[i]); if(u>mx)mx=u; gt1+=u>1;} printf("CASE2_VERIFY N=%d TERMS=%d MAX_ULP=%llu GT1=%d\n",N,best2,(unsigned long long)mx,gt1); }
  arb_clear(ref); arb_clear(aa); arb_clear(ss); arb_clear(pow3); arb_clear(t); arb_clear(c); arb_clear(one); arb_poly_clear(z); arb_poly_clear(s); flint_cleanup(); return 0;
}
