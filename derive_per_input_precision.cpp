#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int terms(double a){ double t; if(a<3.0)t=10.0+10.5*a-2.14*a*a; else if(a<=10.0)t=11.0+5.0*a; else if(a<=40.0)t=11.0+4.54*a+0.0485*a*a; else t=-37.0+6.85*a+0.0194*a*a; return std::max(6,(int)std::ceil(t)+1); }
static uint64_t key(double x){ union{double d;uint64_t u;}v{x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulp(double a,double b){ if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=key(a),y=key(b);return x>y?x-y:y-x; }

struct Env{
 static constexpr int MAXT=900,COEFF_BITS=8192,REF_BITS=4096;
 std::vector<arb_struct>tc;arb_t aa,tmp,third,p,y,base,ref,ss;
 Env():tc(MAXT){arb_poly_t s,z;arb_poly_init(s);arb_poly_init(z);arb_t one,p3;arb_init(one);arb_init(p3);arb_one(one);arb_poly_set_coeff_si(s,0,-2);arb_poly_set_coeff_si(s,1,1);std::fprintf(stderr,"coefficients: %d bits\n",COEFF_BITS);arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);arb_set_ui(p3,9);arb_init(aa);arb_init(tmp);arb_init(third);arb_init(p);arb_init(y);arb_init(base);arb_init(ref);arb_init(ss);for(int m=0;m<MAXT;m++){arb_init(tc.data()+m);arb_poly_get_coeff_arb(tc.data()+m,z,m+1);arb_inv(tmp,p3,COEFF_BITS);arb_add(tc.data()+m,tc.data()+m,tmp,COEFF_BITS);arb_mul_ui(p3,p3,3,COEFF_BITS);}arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);}
 ~Env(){for(auto &x:tc)arb_clear(&x);arb_clear(ss);arb_clear(ref);arb_clear(base);arb_clear(y);arb_clear(p);arb_clear(third);arb_clear(tmp);arb_clear(aa);}
 double reference(double a){arb_set_d(aa,a);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR);}
 double eval(double a,int prec){int T=terms(a);arb_set_d(aa,a);arb_set(p,tc.data()+T-1);for(int m=T-2;m>=0;m--){arb_mul(p,p,aa,prec);arb_add(p,p,tc.data()+m,prec);}arb_mul(y,p,aa,prec);arb_sub_ui(tmp,aa,3,prec);arb_inv(base,tmp,prec);arb_add(y,y,base,prec);arb_one(third);arb_div_ui(third,third,3,prec);arb_add(y,y,third,prec);return arf_get_d(arb_midref(y),ARF_RND_NEAR);}
 int pmin(double a,double r){int lo=53,hi=400;if(ulp(eval(a,hi),r)>1)return 401;while(lo<hi){int mid=(lo+hi)/2;if(ulp(eval(a,mid),r)<=1)hi=mid;else lo=mid+1;}int first=lo;for(int q=std::max(53,lo-5);q<lo;q++)if(ulp(eval(a,q),r)<=1){first=q;break;}return first;}
};

static constexpr double STEP=0.0625; // 1/16
static constexpr int BINS=1632;
static int bin_of(double a){int b=(int)std::floor(a/STEP);if(b<0)b=0;if(b>=BINS)b=BINS-1;return b;}

static std::vector<uint16_t> build_base_lut(Env&e){std::vector<uint16_t>L(BINS,53);const double f[]={0.03125,0.125,0.25,0.5,0.75,0.875,0.96875};for(int b=0;b<BINS;b++){double l=b*STEP,r=std::min(102.0,(b+1)*STEP);int mx=53;for(double q:f){double a=l+(r-l)*q;if(!(a>0))a=std::nextafter(0.0,1.0);if(a==3.0)a=std::nextafter(3.0,q<0.5?0.0:INFINITY);double rr=e.reference(a);mx=std::max(mx,e.pmin(a,rr));}L[b]=(uint16_t)mx;if((b&127)==0)std::fprintf(stderr,"lut %d/%d p=%u\n",b,BINS,(unsigned)L[b]);}return L;}

static void validate(Env&e,const std::vector<uint16_t>&L,int guard,const std::vector<double>&A,const std::vector<double>&R,const std::vector<int>&PM){uint64_t mxu=0;int gt1=0,worst=0,maxOver=-999,minP=999,maxP=0;long long sumP=0,sumM=0,sumO=0;for(size_t i=0;i<A.size();i++){int p=std::min(400,(int)L[bin_of(A[i])]+guard);uint64_t u=ulp(e.eval(A[i],p),R[i]);if(u>mxu){mxu=u;worst=(int)i;}gt1+=u>1;int ov=p-PM[i];sumP+=p;sumM+=PM[i];sumO+=ov;maxOver=std::max(maxOver,ov);minP=std::min(minP,p);maxP=std::max(maxP,p);}std::printf("LUT step=1/16 bins=%d bytes=%d guard=%d GT1=%d/%zu MAX_ULP=%llu PMIN_MEAN=%.4f P_MEAN=%.4f MEAN_OVER=%.4f MAX_OVER=%d MIN_P=%d MAX_P=%d WORST_A=%.17g PMIN=%d PSCHED=%d\n",BINS,BINS*2,guard,gt1,A.size(),(unsigned long long)mxu,(double)sumM/A.size(),(double)sumP/A.size(),(double)sumO/A.size(),maxOver,minP,maxP,A[worst],PM[worst],std::min(400,(int)L[bin_of(A[worst])]+guard));}

int main(){Env e;constexpr int N=5000;std::vector<double>A(N),R(N);std::vector<int>PM(N);std::mt19937_64 g1(0x504d494e5f494e31ULL),g2(0x504d494e5f4f5554ULL);std::uniform_real_distribution<double>u1(1e-12,3.0),u2(3.0,102.0);for(int i=0;i<N/2;i++){double a=u1(g1);if(a>=3)a=std::nextafter(3.0,0.0);A[i]=a;}for(int i=N/2;i<N;i++){double a=u2(g2);if(a<=3)a=std::nextafter(3.0,INFINITY);A[i]=a;}const double hard[]={1e-12,1e-9,1e-6,0.01,0.1,1,2,2.9,2.99,2.9999,std::nextafter(3.0,0.0),std::nextafter(3.0,INFINITY),3.0001,3.01,4,10,20,40,50,75,100,102};for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)A[i]=hard[i];int mn=999,mx=0;long long sm=0;for(int i=0;i<N;i++){R[i]=e.reference(A[i]);PM[i]=e.pmin(A[i],R[i]);mn=std::min(mn,PM[i]);mx=std::max(mx,PM[i]);sm+=PM[i];if((i&255)==0)std::fprintf(stderr,"oracle %d/%d a=%.9g pmin=%d T=%d\n",i,N,A[i],PM[i],terms(A[i]));}std::printf("ORACLE N=%d PMIN_MIN=%d PMIN_MEAN=%.4f PMIN_MAX=%d\n",N,mn,(double)sm/N,mx);auto L=build_base_lut(e);for(int g=0;g<=4;g++)validate(e,L,g,A,R,PM);FILE*f=std::fopen("precision_lut_base_1_16.txt","w");if(f){std::fprintf(f,"// base per-input precision LUT, step=1/16, no guard, bins=%d\n",BINS);for(int b=0;b<BINS;b++)std::fprintf(f,"%u%c",(unsigned)L[b],((b+1)%32)?' ':'\n');std::fclose(f);}flint_cleanup();return 0;}
