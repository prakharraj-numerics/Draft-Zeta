#include <flint/arb.h>
#include <flint/arb_poly.h>
#include <flint/arf.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int terms(double a)
{
    double t;
    if (a < 3.0) t = 10.0 + 10.5*a - 2.14*a*a;
    else if (a <= 10.0) t = 11.0 + 5.0*a;
    else if (a <= 40.0) t = 11.0 + 4.54*a + 0.0485*a*a;
    else t = -37.0 + 6.85*a + 0.0194*a*a;
    return std::max(6, (int)std::ceil(t) + 1);
}

static uint64_t key(double x){ union{double d;uint64_t u;}v{x}; return (v.u>>63)?~v.u:(v.u|0x8000000000000000ULL); }
static uint64_t ulp(double a,double b){ if(std::isnan(a)||std::isnan(b)) return UINT64_MAX; uint64_t x=key(a),y=key(b); return x>y?x-y:y-x; }

struct Env {
    static constexpr int MAXT=900;
    static constexpr int COEFF_BITS=8192;
    static constexpr int REF_BITS=4096;
    std::vector<arb_struct> tc;
    arb_t aa,tmp,third,p,y,base,ref,ss;
    Env(): tc(MAXT) {
        arb_poly_t s,z; arb_poly_init(s); arb_poly_init(z);
        arb_t one,p3; arb_init(one); arb_init(p3);
        arb_one(one); arb_poly_set_coeff_si(s,0,-2); arb_poly_set_coeff_si(s,1,1);
        std::fprintf(stderr,"Generating %d corrected coefficients at %d bits...\n",MAXT,COEFF_BITS);
        arb_poly_zeta_series(z,s,one,0,MAXT+1,COEFF_BITS);
        arb_set_ui(p3,9); arb_init(aa);arb_init(tmp);arb_init(third);arb_init(p);arb_init(y);arb_init(base);arb_init(ref);arb_init(ss);
        for(int m=0;m<MAXT;m++){
            arb_init(tc.data()+m);
            arb_poly_get_coeff_arb(tc.data()+m,z,m+1);
            arb_inv(tmp,p3,COEFF_BITS);
            arb_add(tc.data()+m,tc.data()+m,tmp,COEFF_BITS);
            arb_mul_ui(p3,p3,3,COEFF_BITS);
        }
        arb_clear(p3);arb_clear(one);arb_poly_clear(z);arb_poly_clear(s);
    }
    ~Env(){ for(auto &x:tc) arb_clear(&x); arb_clear(ss);arb_clear(ref);arb_clear(base);arb_clear(y);arb_clear(p);arb_clear(third);arb_clear(tmp);arb_clear(aa); }

    double reference(double a){ arb_set_d(aa,a);arb_sub_ui(ss,aa,2,REF_BITS);arb_zeta(ref,ss,REF_BITS);return arf_get_d(arb_midref(ref),ARF_RND_NEAR); }

    double eval(double a,int prec){
        int T=terms(a); arb_set_d(aa,a); arb_set(p,tc.data()+T-1);
        for(int m=T-2;m>=0;m--){ arb_mul(p,p,aa,prec); arb_add(p,p,tc.data()+m,prec); }
        arb_mul(y,p,aa,prec); arb_sub_ui(tmp,aa,3,prec); arb_inv(base,tmp,prec); arb_add(y,y,base,prec);
        arb_one(third);arb_div_ui(third,third,3,prec);arb_add(y,y,third,prec);
        return arf_get_d(arb_midref(y),ARF_RND_NEAR);
    }

    int pmin(double a,double r){
        int lo=53,hi=400;
        if(ulp(eval(a,hi),r)>1) return 401;
        while(lo<hi){ int mid=(lo+hi)/2; if(ulp(eval(a,mid),r)<=1) hi=mid; else lo=mid+1; }
        int start=std::max(53,lo-4), end=std::min(400,lo+4), first=lo;
        for(int p0=start;p0<=end;p0++) if(ulp(eval(a,p0),r)<=1){ first=p0; break; }
        return first;
    }
};

struct TableResult {
    int bins=0;
    double step=0;
    int guard=0;
    std::vector<uint16_t> p;
};

static int bin_of(double a,double step,int bins){ int b=(int)std::floor(a/step); if(b<0)b=0; if(b>=bins)b=bins-1; return b; }

static TableResult build_table(Env &e,double step,int guard)
{
    const int bins=(int)std::ceil(102.0/step);
    TableResult tr; tr.bins=bins; tr.step=step; tr.guard=guard; tr.p.assign(bins,53);
    const double f[] = {0.03125,0.125,0.25,0.375,0.5,0.625,0.75,0.875,0.96875};
    for(int b=0;b<bins;b++){
        double left=b*step, right=std::min(102.0,(b+1)*step);
        int mx=53;
        for(double q:f){
            double a=left+(right-left)*q;
            if(!(a>0.0)) a=std::nextafter(0.0,1.0);
            if(a==3.0) a=std::nextafter(3.0, (q<0.5?0.0:INFINITY));
            double r=e.reference(a); int pm=e.pmin(a,r); mx=std::max(mx,pm);
        }
        tr.p[b]=(uint16_t)std::min(400,mx+guard);
        if((b&127)==0) std::fprintf(stderr,"table step %.8g bin %d/%d p=%u\n",step,b,bins,(unsigned)tr.p[b]);
    }
    return tr;
}

static void validate(Env &e,const TableResult &tr,const std::vector<double>& A,const std::vector<double>& R,const std::vector<int>& PM)
{
    uint64_t maxu=0; int gt1=0,worst=0; long long sumP=0,sumMin=0,sumOver=0; int maxOver=-999,minSched=999,maxSched=0;
    for(size_t i=0;i<A.size();i++){
        int b=bin_of(A[i],tr.step,tr.bins); int p=tr.p[b]; double got=e.eval(A[i],p); uint64_t u=ulp(got,R[i]);
        if(u>maxu){maxu=u;worst=(int)i;} if(u>1)gt1++;
        int over=p-PM[i]; sumP+=p;sumMin+=PM[i];sumOver+=over;maxOver=std::max(maxOver,over);minSched=std::min(minSched,p);maxSched=std::max(maxSched,p);
    }
    std::printf("TABLE step=%.8g bins=%d bytes=%d guard=%d GT1=%d/%zu MAX_ULP=%llu AVG_P=%.4f AVG_PMIN=%.4f AVG_OVER=%.4f MAX_OVER=%d MIN_P=%d MAX_P=%d WORST_A=%.17g PMIN=%d PSCHED=%u\n",
        tr.step,tr.bins,tr.bins*2,tr.guard,gt1,A.size(),(unsigned long long)maxu,(double)sumP/A.size(),(double)sumMin/A.size(),(double)sumOver/A.size(),maxOver,minSched,maxSched,A[worst],PM[worst],(unsigned)tr.p[bin_of(A[worst],tr.step,tr.bins)]);
}

int main()
{
    Env e;
    constexpr int N=5000;
    std::vector<double>A(N),R(N);std::vector<int>PM(N);
    std::mt19937_64 g1(0x504d494e5f494e31ULL),g2(0x504d494e5f4f5554ULL);
    std::uniform_real_distribution<double>u1(std::nextafter(0.0,1.0),3.0),u2(3.0,102.0);
    for(int i=0;i<N/2;i++){double a=u1(g1);if(a>=3)a=std::nextafter(3.0,0.0);A[i]=a;}
    for(int i=N/2;i<N;i++){double a=u2(g2);if(a<=3)a=std::nextafter(3.0,INFINITY);A[i]=a;}
    const double hard[]={1e-12,1e-9,1e-6,0.01,0.1,1,2,2.9,2.99,2.9999,std::nextafter(3.0,0.0),std::nextafter(3.0,INFINITY),3.0001,3.01,4,10,20,40,50,75,100,102};
    for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++) A[i]=hard[i];
    int globalMin=999,globalMax=0;long long sum=0;
    for(int i=0;i<N;i++){R[i]=e.reference(A[i]);PM[i]=e.pmin(A[i],R[i]);globalMin=std::min(globalMin,PM[i]);globalMax=std::max(globalMax,PM[i]);sum+=PM[i];if((i&255)==0)std::fprintf(stderr,"oracle %d/%d a=%.9g pmin=%d T=%d\n",i,N,A[i],PM[i],terms(A[i]));}
    std::printf("ORACLE N=%d PMIN_MIN=%d PMIN_MEAN=%.4f PMIN_MAX=%d\n",N,globalMin,(double)sum/N,globalMax);

    const double steps[]={0.125,0.0625,0.03125};
    for(double st:steps){
        for(int guard=0;guard<=2;guard++){
            TableResult tr=build_table(e,st,guard);
            validate(e,tr,A,R,PM);
            if(st==0.0625 && guard==1){
                FILE*f=std::fopen("precision_lut_1_16.txt","w");
                if(f){std::fprintf(f,"// uint16_t precision LUT, step=1/16, guard=1, bins=%d\n",tr.bins);for(int b=0;b<tr.bins;b++)std::fprintf(f,"%u%c",(unsigned)tr.p[b],((b+1)%32)?' ':'\n');std::fclose(f);}
            }
        }
    }
    flint_cleanup();
    return 0;
}
