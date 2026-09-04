// Isolated term-count reduction experiment for the current 53-bit zeta machinery.
// Does not modify production.  It uses the same manufactured coefficients and
// per-input precision LUT, but evaluates with candidate shortened T values using
// ARF at the scheduled precision so we can isolate whether the current T(x) law
// has large safety margin for the <=1 ULP binary64 target.

#define main zeta53_gamma_port_reference_main
#include "zeta53_avx512_gamma_port.cpp"
#undef main

#include <map>
#include <string>

static double eval_source_T(const Tables& tb, double a, int bits, int T){
    arf_t av,p,base,coef;
    arf_init(av); arf_init(p); arf_init(base); arf_init(coef);
    arf_set_d(av,a);
    arf_set_round(p,tb.src.data()+T-1,bits,ARF_RND_NEAR);
    for(int m=T-2;m>=0;m--){
        arf_set_round(coef,tb.src.data()+m,bits,ARF_RND_NEAR);
        arf_fma(p,p,av,coef,bits,ARF_RND_NEAR);
    }
    arf_mul(p,p,av,bits,ARF_RND_NEAR);
    arf_sub_ui(base,av,3,bits,ARF_RND_NEAR);
    arf_ui_div(base,1,base,bits,ARF_RND_NEAR);
    arf_add(p,p,base,bits,ARF_RND_NEAR);
    arf_add(p,p,tb.third_arf,bits,ARF_RND_NEAR);
    double y=arf_get_d(p,ARF_RND_NEAR);
    arf_clear(coef); arf_clear(base); arf_clear(p); arf_clear(av);
    return y;
}

static int scheduled_p(const Tables& tb,double x){
    int b=(int)std::floor((x+2.0)*16.0);
    b=std::clamp(b,0,BINS-1);
    return tb.plut[b];
}

struct Stat {
    uint64_t maxulp=0;
    long long gt1=0;
    long long n=0;
    long long tsum=0;
    int tmin=9999,tmax=0;
    double worstx=0.0;
    void add(double x,int t,uint64_t u){
        n++; tsum+=t; tmin=std::min(tmin,t); tmax=std::max(tmax,t);
        if(u>maxulp){maxulp=u;worstx=x;}
        if(u>1)gt1++;
    }
};

static std::vector<double> make_sweep_inputs(){
    std::vector<double> x;
    // Dense deterministic 1/32 grid across (1,100), offset to avoid integers.
    for(int k=0;k<99*32;k++) x.push_back(1.0+(k+0.37)/32.0);
    // Random exact-rational-ish binary64 inputs.
    std::mt19937_64 rng(0x53A66E55ULL);
    std::uniform_real_distribution<double> ud(1.0,100.0);
    for(int i=0;i<2500;i++){
        double q=ud(rng); if(q<=1.0)q=std::nextafter(1.0,INFINITY); if(q>=100.0)q=std::nextafter(100.0,0.0); x.push_back(q);
    }
    const double hard[]={
        std::nextafter(1.0,INFINITY),1.000000000001,1.0000001,1.001,1.01,1.1,1.5,1.999999999999,
        2.9,3.0,4.0,5.0,8.0,10.0,20.0,38.0,40.0,50.0,60.0,75.0,80.0,90.0,98.0,99.0,
        std::nextafter(100.0,0.0)
    };
    x.insert(x.end(),std::begin(hard),std::end(hard));
    return x;
}

static int band(double x){
    if(x<3) return 0;
    if(x<10) return 1;
    if(x<20) return 2;
    if(x<40) return 3;
    if(x<60) return 4;
    if(x<80) return 5;
    return 6;
}
static const char* bandname(int b){
    static const char* n[]={"1-3","3-10","10-20","20-40","40-60","60-80","80-100"};
    return n[b];
}

int main(){
    Tables tb;
    auto xs=make_sweep_inputs();
    std::vector<double> refs(xs.size());
    for(size_t i=0;i<xs.size();i++) refs[i]=tb.reference_x(xs[i]);

    struct Cand { const char* name; int kind; int val; };
    // kind 0: T-delta; kind 1: ceil(T*percent/100)
    const Cand cands[]={
        {"BASE",0,0},{"M1",0,1},{"M2",0,2},{"M4",0,4},{"M8",0,8},{"M12",0,12},{"M16",0,16},
        {"M24",0,24},{"M32",0,32},{"M48",0,48},{"M64",0,64},{"M96",0,96},{"M128",0,128},
        {"P95",1,95},{"P90",1,90},{"P85",1,85},{"P80",1,80},{"P75",1,75},{"P70",1,70},{"P60",1,60},{"P50",1,50}
    };

    for(const auto& c:cands){
        Stat all,bands[7];
        for(size_t i=0;i<xs.size();i++){
            double x=xs[i],a=x+2.0; int t0=terms_a(a),t=t0;
            if(c.kind==0) t=std::max(6,t0-c.val);
            else t=std::max(6,(int)std::ceil((double)t0*c.val/100.0));
            int p=scheduled_p(tb,x);
            double y=eval_source_T(tb,a,p,t);
            uint64_t u=ulpdist(y,refs[i]);
            all.add(x,t,u); bands[band(x)].add(x,t,u);
        }
        std::printf("CAND %-5s N=%lld FAIL=%lld MAXULP=%llu WORST_X=%.17g MEAN_T=%.3f T=[%d,%d]\n",
            c.name,all.n,all.gt1,(unsigned long long)all.maxulp,all.worstx,(double)all.tsum/all.n,all.tmin,all.tmax);
        for(int b=0;b<7;b++) std::printf("  BAND %-6s FAIL=%lld/%lld MAXULP=%llu WORST_X=%.17g MEAN_T=%.3f\n",
            bandname(b),bands[b].gt1,bands[b].n,(unsigned long long)bands[b].maxulp,bands[b].worstx,
            bands[b].n?(double)bands[b].tsum/bands[b].n:0.0);
    }
    return 0;
}
