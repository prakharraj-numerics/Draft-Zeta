#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#define MAXT 1400
#define PREC 8192
static uint64_t key(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);} static uint64_t ulp(double a,double b){uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}
int main(){
 arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,c,tc,p3,tmp,third,a,ss,ref,p1,p2,u1,u2,base;arb_init(one);arb_init(c);arb_init(tc);arb_init(p3);arb_init(tmp);arb_init(third);arb_init(a);arb_init(ss);arb_init(ref);arb_init(p1);arb_init(p2);arb_init(u1);arb_init(u2);arb_init(base);
 arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);std::fprintf(stderr,"Generating coefficients...\n");arb_poly_zeta_series(z,s,one,0,MAXT+1,PREC);std::vector<arb_struct>C(MAXT),TC(MAXT);arb_set_ui(p3,9);for(int m=0;m<MAXT;m++){arb_init(C.data()+m);arb_init(TC.data()+m);arb_poly_get_coeff_arb(C.data()+m,z,m+1);arb_inv(tmp,p3,PREC);arb_add(TC.data()+m,C.data()+m,tmp,PREC);arb_mul_ui(p3,p3,3,PREC);}arb_one(third);arb_div_ui(third,third,3,PREC);
 for(int ix=1;ix<=300;ix++){double ad=2.999*(double)ix/300.0;arb_set_d(a,ad);arb_sub_ui(ss,a,2,PREC);arb_zeta(ref,ss,PREC);double rd=arf_get_d(arb_midref(ref),ARF_RND_NEAR);arb_set(p1,a);arb_set(p2,a);arb_zero(u1);arb_sub_ui(tmp,a,3,PREC);arb_inv(base,tmp,PREC);arb_add(base,base,third,PREC);arb_set(u2,base);int t1=0,t2=0;for(int m=0;m<MAXT;m++){arb_mul(tmp,C.data()+m,p1,PREC);arb_add(u1,u1,tmp,PREC);arb_mul(tmp,TC.data()+m,p2,PREC);arb_add(u2,u2,tmp,PREC);if(!t1&&ulp(arf_get_d(arb_midref(u1),ARF_RND_NEAR),rd)<=1)t1=m+1;if(!t2&&ulp(arf_get_d(arb_midref(u2),ARF_RND_NEAR),rd)<=1)t2=m+1;if(t1&&t2)break;arb_mul(p1,p1,a,PREC);arb_mul(p2,p2,a,PREC);}std::printf("X %.17g %d %d\n",ad,t1,t2);
 }
 for(int m=0;m<MAXT;m++){arb_clear(C.data()+m);arb_clear(TC.data()+m);}arb_clear(base);arb_clear(u2);arb_clear(u1);arb_clear(p2);arb_clear(p1);arb_clear(ref);arb_clear(ss);arb_clear(a);arb_clear(third);arb_clear(tmp);arb_clear(p3);arb_clear(tc);arb_clear(c);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();}
