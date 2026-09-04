#include <boost/math/special_functions/zeta.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
extern "C" void draft_zeta53_batch(size_t,const double*,double*);
extern "C" int draft_zeta53_batch_uses_avx512(void);
using mp=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>>;
static uint64_t ulp(double a,double b){uint64_t x=std::bit_cast<uint64_t>(a),y=std::bit_cast<uint64_t>(b);return x>y?x-y:y-x;}
int main(){constexpr size_t N=100;std::vector<double>x(N),y(N);std::mt19937_64 g(0x5a455441ULL);std::uniform_int_distribution<uint64_t> den(100003,999983), q(2,99), frac(1,99999);for(size_t i=0;i<N;i++){uint64_t d=den(g);uint64_t base=q(g);uint64_t f=1+(frac(g)%(d-1));uint64_t num=base*d+f;double v=(double)num/(double)d;if(v<=1.0||v>=100.0||v==floor(v)){--i;continue;}x[i]=v;}
draft_zeta53_batch(N,x.data(),y.data());uint64_t mx=0,gt1=0,gt2=0;for(size_t i=0;i<N;i++){mp s=x[i];double ref=(double)boost::math::zeta(s);uint64_t u=ulp(y[i],ref);mx=std::max(mx,u);gt1+=u>1;gt2+=u>2;}printf("AVX512=%d\nMAX_ULP=%llu\nGT1=%llu/100\nGT2=%llu/100\n",draft_zeta53_batch_uses_avx512(),(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)gt2);
for(int w=0;w<500;w++)draft_zeta53_batch(N,x.data(),y.data());std::vector<double> t;volatile double sink=0;for(int k=0;k<51;k++){auto a=std::chrono::steady_clock::now();for(int r=0;r<20000;r++){draft_zeta53_batch(N,x.data(),y.data());sink+=y[(size_t)r%N];}auto b=std::chrono::steady_clock::now();double ns=std::chrono::duration<double,std::nano>(b-a).count()/(20000.0*N);t.push_back(ns);}std::sort(t.begin(),t.end());printf("MEDIAN_NS_PER_ELEMENT=%.6f\nSINK=%.17g\n",t[25],(double)sink);return gt2?2:0;}
