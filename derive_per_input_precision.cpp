#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int terms(double a){double t;if(a<3.0)t=10.0+10.5*a-2.14*a*a;else if(a<=10.0)t=11.0+5.0*a;else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a;else t=-37.0+6.85*a+0.0194*a*a;return std::max(6,(int)std::ceil(t)+1);}
static uint64_t key(double x){union{double d;uint64_t u;}v{x};return(v.u>>63)?~v.u:(v.u|0x8000000000000000ULL);}
static uint64_t ulp(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=key(a),y=key(b);return x>y?x-y:y-x;}

struct Env{
 static constexpr int MAXT=900,COEFF_BITS=8192,REF_BITS=2048,MAXP=400;
 std::vector<arf_struct> c;arb_t aa,ss,ref;arf_t av,p,base,third,coef;
 Env():c(MAXT){arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,p3,tmp;arb_init(one);arb_init(p3);arb_init(tmp);arb_init(aa);arb_init(ss);arb_init(ref);arf_init(av);arf_init(p);arf_init(base);arf_init(third);arf_init(coef);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);std::fprintf(stderr,"[setup] 8192-bit coefficient manufacture\n");arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);arb_set_ui(p3,9);for(int m=0;m<MAXT;m++){arf_init(c.data()+m);arb_poly_get_coeff_arb(tmp,z,m+1);arb_t inv;arb_init(inv);arb_inv(inv,p3,COEFF_BITS);arb_add(tmp,tmp,inv,COEFF_BITS);arb_clear(inv);arf_set_round(c.data()+m,arb_midref(tmp),COEFF_BITS,ARF_RND_NEAR);arb_mul_ui(p3,p3,3,COEFF_BITS);}arb_clear(tmp);arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);}
 ~Env(){for(auto &x:c)arf_clear(&x);arf_clear(coef);arf_clear(third);arf_clear(base);arf_clear(p);arf_clear(av);arb_clear(ref);arb_clear(ss);arb_clear(aa);}
 double reference(double a){arb_set_d(aa,a);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR);}
 double eval(double a,int bits){int T=terms(a);arf_set_d(av,a);arf_set_round(p,c.data()+T-1,bits,ARF_RND_NEAR);for(int m=T-2;m>=0;m--){arf_set_round(coef,c.data()+m,bits,ARF_RND_NEAR);arf_fma(p,p,av,coef,bits,ARF_RND_NEAR);}arf_mul(p,p,av,bits,ARF_RND_NEAR);arf_sub_ui(base,av,3,bits,ARF_RND_NEAR);arf_ui_div(base,1,base,bits,ARF_RND_NEAR);arf_add(p,p,base,bits,ARF_RND_NEAR);arf_set_ui(third,1);arf_div_ui(third,third,3,bits,ARF_RND_NEAR);arf_add(p,p,third,bits,ARF_RND_NEAR);return arf_get_d(p,ARF_RND_NEAR);}
 int pmin(double a,double r){int lo=53,hi=MAXP;if(ulp(eval(a,hi),r)>1)return MAXP+1;while(lo<hi){int mid=(lo+hi)/2;if(ulp(eval(a,mid),r)<=1)hi=mid;else lo=mid+1;}int first=lo;for(int q=std::max(53,lo-6);q<lo;q++)if(ulp(eval(a,q),r)<=1){first=q;break;}return first;}
};

static constexpr double STEP=0.0625;static constexpr int BINS=1632;
static int bin_of(double a){int b=(int)std::floor(a/STEP);if(b<0)b=0;if(b>=BINS)b=BINS-1;return b;}
static std::vector<uint16_t> build_lut(Env&e){std::vector<uint16_t>L(BINS,53);const double f[]={0.015625,0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,0.9375,0.984375};for(int b=1;b<BINS;b++){double l=b*STEP,r=std::min(102.0,(b+1)*STEP);int mx=53;for(double q:f){double a=l+(r-l)*q;if(a==3.0)a=std::nextafter(3.0,q<0.5?0.0:INFINITY);double rr=e.reference(a);mx=std::max(mx,e.pmin(a,rr));}L[b]=(uint16_t)mx;if((b&127)==0)std::fprintf(stderr,"[lut] %d/%d baseP=%u\n",b,BINS,(unsigned)L[b]);}return L;}
static int schedule_p(double a,const std::vector<uint16_t>&L,int zeroC,int guard){if(a<STEP){int p=(int)std::ceil((double)zeroC-std::log2(a));return std::clamp(p,53,400);}return std::min(400,(int)L[bin_of(a)]+guard);}
static void validate(Env&e,const std::vector<uint16_t>&L,int zeroC,int guard,const std::vector<double>&A,const std::vector<double>&R,const std::vector<int>&PM){uint64_t mxu=0;int gt1=0,worst=0,maxOver=-999,minOver=999,minP=999,maxP=0;long long sumP=0,sumM=0,sumO=0;for(size_t i=0;i<A.size();i++){int ps=schedule_p(A[i],L,zeroC,guard);uint64_t u=ulp(e.eval(A[i],ps),R[i]);if(u>mxu){mxu=u;worst=(int)i;}gt1+=u>1;int ov=ps-PM[i];sumP+=ps;sumM+=PM[i];sumO+=ov;maxOver=std::max(maxOver,ov);minOver=std::min(minOver,ov);minP=std::min(minP,ps);maxP=std::max(maxP,ps);}std::printf("SCHED zeroC=%d lut_guard=%d GT1=%d/%zu MAX_ULP=%llu PMIN_MEAN=%.4f P_MEAN=%.4f MEAN_OVER=%.4f MIN_OVER=%d MAX_OVER=%d MIN_P=%d MAX_P=%d WORST_A=%.17g PMIN=%d PSCHED=%d\n",zeroC,guard,gt1,A.size(),(unsigned long long)mxu,(double)sumM/A.size(),(double)sumP/A.size(),(double)sumO/A.size(),minOver,maxOver,minP,maxP,A[worst],PM[worst],schedule_p(A[worst],L,zeroC,guard));}

int main(){Env e;constexpr int N=5000;std::vector<double>A(N),R(N);std::vector<int>PM(N);std::mt19937_64 g1(0x504d494e5f494e31ULL),g2(0x504d494e5f4f5554ULL);std::uniform_real_distribution<double>u1(1e-12,3.0),u2(3.0,102.0);for(int i=0;i<N/2;i++){double a=u1(g1);if(a>=3)a=std::nextafter(3.0,0.0);A[i]=a;}for(int i=N/2;i<N;i++){double a=u2(g2);if(a<=3)a=std::nextafter(3.0,INFINITY);A[i]=a;}const double hard[]={1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.03,0.06,0.1,1,2,2.9,2.99,2.9999,std::nextafter(3.0,0.0),std::nextafter(3.0,INFINITY),3.0001,3.01,4,10,20,40,50,75,81.764554025359956,100,101,102};for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)A[i]=hard[i];int mn=999,mx=0;long long sm=0;for(int i=0;i<N;i++){R[i]=e.reference(A[i]);PM[i]=e.pmin(A[i],R[i]);mn=std::min(mn,PM[i]);mx=std::max(mx,PM[i]);sm+=PM[i];if(i<(int)(sizeof(hard)/sizeof(hard[0])))std::printf("PROBE a=%.17g T=%d PMIN=%d\n",A[i],terms(A[i]),PM[i]);if((i&511)==0)std::fprintf(stderr,"[oracle] %d/%d\n",i,N);}std::printf("ORACLE N=%d PMIN_MIN=%d PMIN_MEAN=%.4f PMIN_MAX=%d\n",N,mn,(double)sm/N,mx);auto L=build_lut(e);for(int zc=55;zc<=57;zc++)for(int g=0;g<=3;g++)validate(e,L,zc,g,A,R,PM);FILE*f=std::fopen("precision_lut_base_1_16.txt","w");if(f){std::fprintf(f,"// base precision LUT for a>=1/16; bin width=1/16\n");for(int b=0;b<BINS;b++)std::fprintf(f,"%u%c",(unsigned)L[b],((b+1)%32)?' ':'\n');std::fclose(f);}flint_cleanup();return 0;}
