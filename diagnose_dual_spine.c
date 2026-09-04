#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#define M 300
#define PREC 4096
static double ch[M],cl[M],th[M],tl[M];
typedef struct{double hi,lo;}dd;
static dd rn(double a,double b){dd z={a+b,0};z.lo=b-(z.hi-a);return z;}
static dd mul(dd x,double y){double p=x.hi*y,e=fma(x.hi,y,-p)+x.lo*y;return rn(p,e);}
static dd add(dd x,double h,double l){double s=x.hi+h,bb=s-x.hi,e1=(x.hi-(s-bb))+(h-bb);return rn(s,x.lo+e1+l);}
static double ev(const double*h,const double*l,int n,double a){dd p={h[n-1],l[n-1]};for(int k=n-2;k>=0;k--)p=add(mul(p,a),h[k],l[k]);p=mul(p,a);return p.hi+p.lo;}
static double split(const arb_t x,double *lo){double h=arf_get_d(arb_midref(x),ARF_RND_NEAR);arb_t q,b;arb_init(q);arb_init(b);arb_set(q,x);arb_set_d(b,h);arb_sub(q,q,b,PREC);*lo=arf_get_d(arb_midref(q),ARF_RND_NEAR);arb_clear(q);arb_clear(b);return h;}
int main(){arb_poly_t s,z;arb_t one,c,t,p3,ref,ss;arb_poly_init(s);arb_poly_init(z);arb_init(one);arb_init(c);arb_init(t);arb_init(p3);arb_init(ref);arb_init(ss);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);arb_poly_zeta_series(z,s,one,0,M+1,PREC);arb_set_ui(p3,9);for(int m=0;m<M;m++){arb_poly_get_coeff_arb(c,z,m+1);ch[m]=split(c,&cl[m]);arb_inv(t,p3,PREC);arb_add(t,t,c,PREC);th[m]=split(t,&tl[m]);arb_mul_ui(p3,p3,3,PREC);}for(int m=0;m<6;m++)printf("c%d=%.17g lo=%+.17g sum=%.25g\n",m,ch[m],cl[m],ch[m]+cl[m]);
 double aa[]={1.0,2.0,2.9,4.0,5.2,10.0,50.0,102.0};int nn[]={20,35,50,100,200,300};
 for(int i=0;i<8;i++){double a=aa[i];arb_set_d(ss,a-2);arb_zeta(ref,ss,PREC);double r=arf_get_d(arb_midref(ref),ARF_RND_NEAR);printf("A=%.17g REF=%.17g\n",a,r);for(int k=0;k<6;k++){int n=nn[k];double y=fabs(a)<3?ev(ch,cl,n,a):(1.0/(a-3.0)+1.0/3.0+ev(th,tl,n,a));printf(" T=%d GOT=%.17g ERR=%.17g\n",n,y,y-r);} }
 arb_clear(ss);arb_clear(ref);arb_clear(p3);arb_clear(t);arb_clear(c);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);flint_cleanup();}
