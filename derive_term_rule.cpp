#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define MAXT 1400
#define PREC 8192

static uint64_t key(double x){ union{double d;uint64_t u;}v{x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulp(double a,double b){ uint64_t x=key(a),y=key(b); return x>y?x-y:y-x; }

int main(){
  arb_poly_t s,z; arb_poly_init(s); arb_poly_init(z);
  arb_t one,c,tc,p3,tmp,third,aa,ss,ref; arb_init(one);arb_init(c);arb_init(tc);arb_init(p3);arb_init(tmp);arb_init(third);arb_init(aa);arb_init(ss);arb_init(ref);
  arb_one(one); arb_poly_set_coeff_si(s,0,-2); arb_poly_set_coeff_si(s,1,1);
  std::fprintf(stderr,"Generating %d coefficients...\n",MAXT);
  arb_poly_zeta_series(z,s,one,0,MAXT+1,PREC);
  std::vector<arb_struct> C(MAXT),TC(MAXT); arb_set_ui(p3,9);
  for(int m=0;m<MAXT;m++){ arb_init(C.data()+m);arb_init(TC.data()+m);arb_poly_get_coeff_arb(C.data()+m,z,m+1);arb_inv(tmp,p3,PREC);arb_add(TC.data()+m,C.data()+m,tmp,PREC);arb_mul_ui(p3,p3,3,PREC); }
  arb_one(third); arb_div_ui(third,third,3,PREC);

  auto findT=[&](double a,bool outside){
    arb_t p,sum,base; arb_init(p);arb_init(sum);arb_init(base); arb_set_d(aa,a); arb_set(p,aa); arb_zero(sum); arb_sub_ui(ss,aa,2,PREC); arb_zeta(ref,ss,PREC); double rd=arf_get_d(arb_midref(ref),ARF_RND_NEAR);
    if(outside){ arb_sub_ui(tmp,aa,3,PREC);arb_inv(base,tmp,PREC);arb_add(base,base,third,PREC); } else arb_zero(base);
    int hit=0;
    for(int m=0;m<MAXT;m++){
      arb_mul(tmp,outside?TC.data()+m:C.data()+m,p,PREC);arb_add(sum,sum,tmp,PREC);arb_add(tmp,sum,base,PREC);double y=arf_get_d(arb_midref(tmp),ARF_RND_NEAR);
      if(ulp(y,rd)<=1){ hit=m+1; break; }
      arb_mul(p,p,aa,PREC);
    }
    arb_clear(base);arb_clear(sum);arb_clear(p);return hit;
  };

  std::puts("# INSIDE a T");
  for(int i=1;i<=240;i++){
    double a=2.99*(double)i/240.0; int T=findT(a,false); std::printf("I %.17g %d\n",a,T);
  }
  const double near3[]={2.9,2.95,2.97,2.98,2.99,2.995,2.997,2.999,2.9995,2.9999};
  for(double a:near3) std::printf("I %.17g %d\n",a,findT(a,false));

  std::puts("# OUTSIDE a T");
  for(int i=0;i<=396;i++){
    double a=3.01 + (102.0-3.01)*(double)i/396.0; int T=findT(a,true); std::printf("O %.17g %d\n",a,T);
  }
  const double near3o[]={3.0001,3.0005,3.001,3.003,3.005,3.01,3.05,3.1,3.5,4,5.2,10,20,50,75,102};
  for(double a:near3o) std::printf("O %.17g %d\n",a,findT(a,true));

  for(int m=0;m<MAXT;m++){arb_clear(C.data()+m);arb_clear(TC.data()+m);} arb_clear(ref);arb_clear(ss);arb_clear(aa);arb_clear(third);arb_clear(tmp);arb_clear(p3);arb_clear(tc);arb_clear(c);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();
}