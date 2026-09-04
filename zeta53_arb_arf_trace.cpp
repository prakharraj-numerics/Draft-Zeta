#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <cstdio>
#include <cmath>
#include <vector>
static int terms(double a){double t;if(a<3)t=10+10.5*a-2.14*a*a;else if(a<=10)t=11+5*a;else if(a<=40)t=11+4.54*a+.0485*a*a;else t=-37+6.85*a+.0194*a*a;return std::max(6,(int)ceil(t)+1);}
int main(){const slong GEN=8192,WB=384,MAXT=900;const double ad=100.10072209666959;const int T=terms(ad);arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,p3,tmp,aa,ss,ref,third,pb,yb,baseb;for(auto*x:{&one,&p3,&tmp,&aa,&ss,&ref,&third,&pb,&yb,&baseb})arb_init(*x);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);arb_poly_zeta_series(z,s,one,0,MAXT+1,GEN);std::vector<arb_struct>C(MAXT);std::vector<arf_struct>F(MAXT);arb_set_ui(p3,9);for(int m=0;m<MAXT;m++){arb_init(C.data()+m);arf_init(F.data()+m);arb_poly_get_coeff_arb(C.data()+m,z,m+1);arb_inv(tmp,p3,GEN);arb_add(C.data()+m,C.data()+m,tmp,GEN);arf_set(F.data()+m,arb_midref(C.data()+m));arb_mul_ui(p3,p3,3,GEN);}arb_set_d(aa,ad);arb_sub_ui(ss,aa,2,GEN);arb_zeta(ref,ss,GEN);printf("a=%.17g T=%d REF=",ad,T);arf_printd(arb_midref(ref),25);printf("\n");arb_set(pb,C.data()+T-1);arf_t fa,fp,ft,fb,fthird;arf_init(fa);arf_init(fp);arf_init(ft);arf_init(fb);arf_init(fthird);arf_set_d(fa,ad);arf_set(fp,F.data()+T-1);int mism=0;for(int m=T-2;m>=0;m--){arb_mul(pb,pb,aa,WB);arb_add(pb,pb,C.data()+m,WB);arf_mul(fp,fp,fa,WB,ARF_RND_NEAR);arf_add(fp,fp,F.data()+m,WB,ARF_RND_NEAR);if(!arf_equal(arb_midref(pb),fp)&&mism<10){printf("MISMATCH m=%d arb=",m);arf_printd(arb_midref(pb),20);printf(" arf=");arf_printd(fp,20);printf("\n");mism++;}}
arb_mul(yb,pb,aa,WB);arb_sub_ui(tmp,aa,3,WB);arb_inv(baseb,tmp,WB);arb_one(third);arb_div_ui(third,third,3,WB);arb_add(yb,yb,baseb,WB);arb_add(yb,yb,third,WB);
arf_mul(fp,fp,fa,WB,ARF_RND_NEAR);arf_sub_ui(fb,fa,3,WB,ARF_RND_NEAR);arf_ui_div(fb,1,fb,WB,ARF_RND_NEAR);arf_set_ui(fthird,1);arf_div_ui(fthird,fthird,3,WB,ARF_RND_NEAR);arf_add(fp,fp,fb,WB,ARF_RND_NEAR);arf_add(fp,fp,fthird,WB,ARF_RND_NEAR);
printf("FINAL_ARB=");arf_printd(arb_midref(yb),30);printf("\nFINAL_ARF=");arf_printd(fp,30);printf("\nFINAL_EQUAL=%d MISMATCHES_SHOWN=%d\n",arf_equal(arb_midref(yb),fp),mism);
// Also compare base reciprocal directly.
printf("BASE_ARB=");arf_printd(arb_midref(baseb),30);printf("\nBASE_ARF=");arf_printd(fb,30);printf("\nBASE_EQUAL=%d\n",arf_equal(arb_midref(baseb),fb));
arf_clear(fthird);arf_clear(fb);arf_clear(ft);arf_clear(fp);arf_clear(fa);for(int m=0;m<MAXT;m++){arf_clear(F.data()+m);arb_clear(C.data()+m);}for(auto*x:{&baseb,&yb,&pb,&third,&ref,&ss,&aa,&tmp,&p3,&one})arb_clear(*x);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();}
