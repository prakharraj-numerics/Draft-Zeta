#define main zeta_gamma_port_reference_main
#include "zeta53_avx512_gamma_port.cpp"
#undef main

template<int SW>
static void ours_batch_fast(const Tables&tb,const double*x,size_t n,double*out,Scratch&s){
    prepare(x,n,tb,s);
    if(s.kbeg[1]<s.kend[1])eval_group<1,SW>(tb,s,s.kbeg[1],s.kend[1]);
    if(s.kbeg[2]<s.kend[2])eval_group<2,SW>(tb,s,s.kbeg[2],s.kend[2]);
    if(s.kbeg[3]<s.kend[3])eval_group<3,SW>(tb,s,s.kbeg[3],s.kend[3]);
    if(s.kbeg[4]<s.kend[4])eval_group<4,SW>(tb,s,s.kbeg[4],s.kend[4]);
    if(s.kbeg[5]<s.kend[5])eval_group<5,SW>(tb,s,s.kbeg[5],s.kend[5]);
    if(s.kbeg[6]<s.kend[6])eval_group<6,SW>(tb,s,s.kbeg[6],s.kend[6]);
    if(s.kbeg[7]<s.kend[7])eval_group<7,SW>(tb,s,s.kbeg[7],s.kend[7]);
    if(s.kbeg[8]<s.kend[8])eval_group<8,SW>(tb,s,s.kbeg[8],s.kend[8]);
    for(size_t p=0;p<n;p++)out[s.sidx[p]]=s.sy[p];
}

template<int SW>
static bool accuracy_fast(const char*name,const Tables&tb,const std::vector<double>&x,std::vector<double>&out,Scratch&s){
    ours_batch_fast<SW>(tb,x.data(),x.size(),out.data(),s);uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;
    for(size_t i=0;i<x.size();i++){double r=const_cast<Tables&>(tb).reference_x(x[i]);uint64_t u=ulpdist(out[i],r);if(u==UINT64_MAX){bad++;continue;}if(u>mx){mx=u;w=(int)i;}gt1+=u>1;gt2+=u>2;}
    std::printf("ACC_%s MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu NONFINITE=%d WORST_X=%.17g\n",name,(unsigned long long)mx,gt1,x.size(),gt2,x.size(),bad,x[w]);return gt1==0&&bad==0;
}

int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma"))return 2;
    Tables tb;const size_t N=5000;auto x=make_inputs(N);const double hard[]={std::nextafter(1.0,INFINITY),1.000000000001,1.01,1.1,2.9,3.0,4.0,8.0,10.0,20.0,38.0,50.0,75.0,79.764554025359956,98.0,99.0,std::nextafter(100.0,0.0)};for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)x[i]=hard[i];
    Scratch s0,s1;std::vector<double>o0(N),o1(N),intel(N);bool p0=accuracy_fast<0>("SW0",tb,x,o0,s0),p1=accuracy_fast<1>("SW1",tb,x,o1,s1);if(!p0&&!p1){std::puts("NO_FAST_VARIANT_PASS");flint_cleanup();return 3;}
    uint64_t h=1469598103934665603ULL;for(double q:x){h^=std::bit_cast<uint64_t>(q);h*=1099511628211ULL;}std::printf("INPUT_HASH=%016llx SAME_INPUTS=1\n",(unsigned long long)h);
    volatile double sink=0.0;for(size_t n:{100UL,400UL,800UL,2000UL,5000UL}){
        int reps=std::max(1,(int)(30000/n));double n0=-1,n1=-1;if(p0){ours_batch_fast<0>(tb,x.data(),n,o0.data(),s0);n0=median_ns(n,reps,[&](){ours_batch_fast<0>(tb,x.data(),n,o0.data(),s0);sink+=o0[0];});}if(p1){ours_batch_fast<1>(tb,x.data(),n,o1.data(),s1);n1=median_ns(n,reps,[&](){ours_batch_fast<1>(tb,x.data(),n,o1.data(),s1);sink+=o1[0];});}intel_batch(x.data(),n,intel.data());double in=median_ns(n,reps,[&](){intel_batch(x.data(),n,intel.data());sink+=intel[0];});std::printf("BATCH=%zu SW0_NS=%.6f SW1_NS=%.6f INTEL_NS=%.6f\n",n,n0,n1,in);
    }std::printf("SINK=%.17g\n",(double)sink);flint_cleanup();return 0;
}
