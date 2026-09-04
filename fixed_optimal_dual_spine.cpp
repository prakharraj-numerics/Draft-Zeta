#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define N 1000
#define MAXT 1400
#define PREC 8192

typedef struct { double hi, lo; } dd;
static inline dd renorm(double a,double b){ dd z; z.hi=a+b; z.lo=b-(z.hi-a); return z; }
static inline dd mul_d(dd x,double y){ double p=x.hi*y; double e=std::fma(x.hi,y,-p)+x.lo*y; return renorm(p,e); }
static inline dd add_c(dd x,double hi,double lo){ double s=x.hi+hi; double bb=s-x.hi; double e1=(x.hi-(s-bb))+(hi-bb); return renorm(s,x.lo+e1+lo); }
static double eval_dd(const std::vector<double>& hi,const std::vector<double>& lo,int T,double a){ dd p{hi[T-1],lo[T-1]}; for(int k=T-2;k>=0;--k) p=add_c(mul_d(p,a),hi[k],lo[k]); p=mul_d(p,a); return p.hi+p.lo; }
static uint64_t key(double x){ union{double d; uint64_t u;}v{x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulp(double a,double b){ if(std::isnan(a)||std::isnan(b)) return UINT64_MAX; uint64_t x=key(a),y=key(b); return x>y?x-y:y-x; }
static double arb_hi_lo(const arb_t x,double *lo){ double h=arf_get_d(arb_midref(x),ARF_RND_NEAR); arb_t t,hb; arb_init(t); arb_init(hb); arb_set(t,x); arb_set_d(hb,h); arb_sub(t,t,hb,PREC); *lo=arf_get_d(arb_midref(t),ARF_RND_NEAR); arb_clear(hb); arb_clear(t); return h; }

int main(){
  std::vector<double> ain(N), aout(N), rin(N), rout(N);
  std::mt19937_64 gin(0x494e5349444533ULL), gout(0x5231343335ULL);
  std::uniform_real_distribution<double> ud(-2.9,2.9);
  ain[0]=-2.9; ain[1]=2.9;
  for(int i=2;i<N;i++){ double a=ud(gin); if(std::fabs(a)<1e-14) a=0.125; ain[i]=a; }
  std::uniform_int_distribution<uint64_t> den(100003,999983), base(1,99), frac(1,99999);
  for(int i=0;i<N;){ uint64_t d=den(gout), b=base(gout), f=1+(frac(gout)%(d-1)); uint64_t num=b*d+f; double s=(double)num/(double)d; if(s<=1.0||s>=100.0||s==std::floor(s)) continue; aout[i]=s+2.0; ++i; }

  arb_poly_t s,z; arb_poly_init(s); arb_poly_init(z);
  arb_t one,c,tc,pow3,tmp,aa,ss,ref; arb_init(one);arb_init(c);arb_init(tc);arb_init(pow3);arb_init(tmp);arb_init(aa);arb_init(ss);arb_init(ref);
  arb_one(one); arb_poly_set_coeff_si(s,0,-2); arb_poly_set_coeff_si(s,1,1);
  std::fprintf(stderr,"Generating %d coefficients at %d bits...\n",MAXT,PREC);
  arb_poly_zeta_series(z,s,one,0,MAXT+1,PREC);

  std::vector<double> chi(MAXT),clo(MAXT),thi(MAXT),tlo(MAXT);
  std::vector<arb_t*> dummy;
  arb_set_ui(pow3,9);
  std::vector<arb_struct> C(MAXT), TC(MAXT);
  for(int m=0;m<MAXT;m++){ arb_init(C.data()+m); arb_init(TC.data()+m); arb_poly_get_coeff_arb(C.data()+m,z,m+1); arb_inv(tmp,pow3,PREC); arb_add(TC.data()+m,C.data()+m,tmp,PREC); chi[m]=arb_hi_lo(C.data()+m,&clo[m]); thi[m]=arb_hi_lo(TC.data()+m,&tlo[m]); arb_mul_ui(pow3,pow3,3,PREC); }

  std::vector<arb_struct> sum1(N),pow1(N),sum2(N),pow2(N),base2(N);
  for(int i=0;i<N;i++){ arb_init(sum1.data()+i);arb_init(pow1.data()+i);arb_init(sum2.data()+i);arb_init(pow2.data()+i);arb_init(base2.data()+i);
    arb_zero(sum1.data()+i); arb_zero(sum2.data()+i);
    arb_set_d(pow1.data()+i,ain[i]); arb_set_d(pow2.data()+i,aout[i]);
    arb_set_d(aa,ain[i]); arb_sub_ui(ss,aa,2,PREC); arb_zeta(ref,ss,PREC); rin[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);
    arb_set_d(aa,aout[i]); arb_sub_ui(ss,aa,2,PREC); arb_zeta(ref,ss,PREC); rout[i]=arf_get_d(arb_midref(ref),ARF_RND_NEAR);
    arb_set_d(aa,aout[i]); arb_sub_ui(tmp,aa,3,PREC); arb_inv(base2.data()+i,tmp,PREC); arb_set_ui(tmp,1); arb_div_ui(tmp,tmp,3,PREC); arb_add(base2.data()+i,base2.data()+i,tmp,PREC);
  }

  int best1=0,best2=0; uint64_t prevmx1=0,prevmx2=0; int prevbad1=0,prevbad2=0; uint64_t bestmx1=0,bestmx2=0;
  for(int m=0;m<MAXT;m++){
    for(int i=0;i<N;i++){
      arb_mul(tmp,C.data()+m,pow1.data()+i,PREC); arb_add(sum1.data()+i,sum1.data()+i,tmp,PREC); arb_mul(pow1.data()+i,pow1.data()+i,aa,PREC); /* reset correctly below */
    }
    /* pow1 multiplication above cannot reuse shared aa; repair powers exactly from current a. */
    for(int i=0;i<N;i++){ arb_set_d(aa,ain[i]); if(m==0) arb_mul(pow1.data()+i,aa,aa,PREC); else arb_mul(pow1.data()+i,pow1.data()+i,aa,PREC); }
    for(int i=0;i<N;i++){
      arb_mul(tmp,TC.data()+m,pow2.data()+i,PREC); arb_add(sum2.data()+i,sum2.data()+i,tmp,PREC);
      arb_set_d(aa,aout[i]); arb_mul(pow2.data()+i,pow2.data()+i,aa,PREC);
    }
    int T=m+1;
    if(!best1){ uint64_t mx=0; int bad=0; for(int i=0;i<N;i++){ double y=arf_get_d(arb_midref(sum1.data()+i),ARF_RND_NEAR); uint64_t u=ulp(y,rin[i]); if(u>mx)mx=u; bad+=(u>1); } if(!bad){best1=T;bestmx1=mx;} else {prevmx1=mx;prevbad1=bad;} }
    if(!best2){ uint64_t mx=0; int bad=0; for(int i=0;i<N;i++){ arb_add(tmp,base2.data()+i,sum2.data()+i,PREC); double y=arf_get_d(arb_midref(tmp),ARF_RND_NEAR); uint64_t u=ulp(y,rout[i]); if(u>mx)mx=u; bad+=(u>1); } if(!bad){best2=T;bestmx2=mx;} else {prevmx2=mx;prevbad2=bad;} }
    if(best1&&best2) break;
  }

  std::printf("CASE1_DOMAIN=[-2.9,2.9], N=%d\n",N);
  if(best1) std::printf("CASE1_MATH_MIN=%d MAX_ULP=%llu PREV_T=%d PREV_GT1=%d PREV_MAX_ULP=%llu\n",best1,(unsigned long long)bestmx1,best1-1,prevbad1,(unsigned long long)prevmx1); else std::printf("CASE1_MATH_NO_PASS_UP_TO=%d\n",MAXT);
  std::printf("CASE2_DOMAIN=s in (1,100), a=s+2 in (3,102), N=%d\n",N);
  if(best2) std::printf("CASE2_MATH_MIN=%d MAX_ULP=%llu PREV_T=%d PREV_GT1=%d PREV_MAX_ULP=%llu\n",best2,(unsigned long long)bestmx2,best2-1,prevbad2,(unsigned long long)prevmx2); else std::printf("CASE2_MATH_NO_PASS_UP_TO=%d\n",MAXT);

  if(best1){ uint64_t mx=0; int gt1=0,w=0; for(int i=0;i<N;i++){ double y=eval_dd(chi,clo,best1,ain[i]); uint64_t u=ulp(y,rin[i]); if(u>mx){mx=u;w=i;} gt1+=(u>1); } std::printf("CASE1_FIXED_DD TERMS=%d MAX_ULP=%llu GT1=%d/%d WORST_A=%.17g GOT=%.17g REF=%.17g\n",best1,(unsigned long long)mx,gt1,N,ain[w],eval_dd(chi,clo,best1,ain[w]),rin[w]); }
  if(best2){ uint64_t mx=0; int gt1=0,w=0; for(int i=0;i<N;i++){ double y=1.0/(aout[i]-3.0)+1.0/3.0+eval_dd(thi,tlo,best2,aout[i]); uint64_t u=ulp(y,rout[i]); if(u>mx){mx=u;w=i;} gt1+=(u>1); } std::printf("CASE2_FIXED_DD TERMS=%d MAX_ULP=%llu GT1=%d/%d WORST_A=%.17g GOT=%.17g REF=%.17g\n",best2,(unsigned long long)mx,gt1,N,aout[w],1.0/(aout[w]-3.0)+1.0/3.0+eval_dd(thi,tlo,best2,aout[w]),rout[w]); }

  for(int m=0;m<MAXT;m++){arb_clear(C.data()+m);arb_clear(TC.data()+m);} for(int i=0;i<N;i++){arb_clear(sum1.data()+i);arb_clear(pow1.data()+i);arb_clear(sum2.data()+i);arb_clear(pow2.data()+i);arb_clear(base2.data()+i);} arb_clear(ref);arb_clear(ss);arb_clear(aa);arb_clear(tmp);arb_clear(pow3);arb_clear(tc);arb_clear(c);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();
  return 0;
}
