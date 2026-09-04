// Spill-aware, dense-batch AVX-512 candidate for the current zeta53 machinery.
// Keeps the incumbent formula, coefficient manufacture, P/T scheduler and final
// precision rounding.  The hot Horner loop is reorganized to reduce register
// pressure and masked per-lane bookkeeping.

#define main zeta_gamma_port_reference_main
#include "zeta53_avx512_gamma_port.cpp"
#undef main

template<int K>
static constexpr int VEC_UNROLL = (K <= 2 ? 4 : (K <= 4 ? 2 : 1));

template<int K,int SW,bool ROUND_HORNER>
__attribute__((target("avx512f,avx512dq,avx512vl,avx2,fma")))
static void eval_group_dense(const Tables&tb,Scratch&s,int lo,int hi){
    constexpr int V = VEC_UNROLL<K>;
    constexpr int BLK = 8*V;
    const __m512d two=_mm512_set1_pd(2.0), one=_mm512_set1_pd(1.0);

    for(int b=lo;b<hi;b+=BLK){
        const int nr=std::min(BLK,hi-b);
        const int maxT=s.sT[b];
        const int minT=s.sT[b+nr-1]; // prepare() sorts T descending inside each K group

        __m512d x[V],ahi[V],alo[V],p[V][K];
        __m256i pv[V],tv[V];
        __mmask8 valid[V];

        for(int c=0;c<V;c++){
            const int off=b+8*c;
            const int rem=std::clamp(nr-8*c,0,8);
            valid[c]=(rem==8)?(__mmask8)0xff:(__mmask8)((1u<<rem)-1u);
            x[c]=_mm512_maskz_loadu_pd(valid[c],s.sx.data()+off);
            pv[c]=_mm256_maskz_loadu_epi32(valid[c],s.sP.data()+off);
            tv[c]=_mm256_maskz_loadu_epi32(valid[c],s.sT.data()+off);
            v_two_sum(x[c],two,ahi[c],alo[c]);
            for(int j=0;j<K;j++)p[c][j]=_mm512_setzero_pd();
        }

        // Only the short ramp where lanes begin at different T values needs masks.
        for(int m=maxT-1;m>=minT-1;m--){
            __m512d cv[K];
            for(int j=0;j<K;j++)cv[j]=_mm512_set1_pd(tb.cm[m][j]);
            double sc=1.0;
            if(m<maxT-1){const int de=tb.ce[m+1]-tb.ce[m];sc=std::ldexp(1.0,de);}
            const __m512d vsc=_mm512_set1_pd(sc);

            for(int c=0;c<V;c++){
                if(!valid[c])continue;
                const __mmask8 startm=valid[c]&_mm256_cmp_epi32_mask(tv[c],_mm256_set1_epi32(m+1),_MM_CMPINT_EQ);
                const __mmask8 recm=valid[c]&_mm256_cmp_epi32_mask(tv[c],_mm256_set1_epi32(m+1),_MM_CMPINT_GT);

                if(startm){
                    for(int j=0;j<K;j++)p[c][j]=_mm512_mask_mov_pd(p[c][j],startm,cv[j]);
                }
                if(recm){
                    __m512d cand[K],zc[K];
                    for(int j=0;j<K;j++){cand[j]=p[c][j];zc[j]=cv[j];}
                    v_mac_split<K,SW>(cand,
                        _mm512_mul_pd(ahi[c],vsc),
                        _mm512_mul_pd(alo[c],vsc),zc);
                    for(int j=0;j<K;j++)p[c][j]=_mm512_mask_mov_pd(p[c][j],recm,cand[j]);
                }
                if constexpr(ROUND_HORNER){
                    const __mmask8 active=startm|recm;
                    if(active)v_round_p<K>(p[c],pv[c],active);
                }
            }
        }

        // From here every valid lane is active: no T compares, no candidate copies,
        // no masked p[] moves.  This is the dominant part of the polynomial.
        for(int m=minT-2;m>=0;m--){
            __m512d cv[K];
            for(int j=0;j<K;j++)cv[j]=_mm512_set1_pd(tb.cm[m][j]);
            const int de=tb.ce[m+1]-tb.ce[m];
            const __m512d vsc=_mm512_set1_pd(std::ldexp(1.0,de));

            for(int c=0;c<V;c++){
                if(!valid[c])continue;
                __m512d zc[K];for(int j=0;j<K;j++)zc[j]=cv[j];
                v_mac_split<K,SW>(p[c],
                    _mm512_mul_pd(ahi[c],vsc),
                    _mm512_mul_pd(alo[c],vsc),zc);
                if constexpr(ROUND_HORNER)v_round_p<K>(p[c],pv[c],valid[c]);
            }
        }

        // Keep incumbent P-rounding around the final multiply, reciprocal and 1/3.
        const __m512d sc0=_mm512_set1_pd(std::ldexp(1.0,tb.ce[0]));
        for(int c=0;c<V;c++){
            if(!valid[c])continue;
            __m512d zc[K];for(int j=0;j<K;j++)zc[j]=_mm512_setzero_pd();
            v_mac_split<K,SW>(p[c],
                _mm512_mul_pd(ahi[c],sc0),
                _mm512_mul_pd(alo[c],sc0),zc);
            v_round_p<K>(p[c],pv[c],valid[c]);

            __m512d rr[K];reciprocal_exp<K>(x[c],pv[c],valid[c],rr);
            v_mac<K,SW>(p[c],one,rr);v_round_p<K>(p[c],pv[c],valid[c]);

            for(int j=0;j<K;j++)zc[j]=_mm512_set1_pd(tb.third[j]);
            v_mac<K,SW>(p[c],one,zc);v_round_p<K>(p[c],pv[c],valid[c]);

            __m512d y=p[c][K-1];
            for(int j=K-2;j>=0;j--)y=_mm512_add_pd(p[c][j],y);
            _mm512_mask_storeu_pd(s.sy.data()+b+8*c,valid[c],y);
        }
    }
}

template<int K,int SW,bool ROUND_HORNER>
static inline void eval_k_dense(const Tables&tb,Scratch&s){
    if(s.kbeg[K]<s.kend[K])eval_group_dense<K,SW,ROUND_HORNER>(tb,s,s.kbeg[K],s.kend[K]);
}

template<int SW,bool ROUND_HORNER>
static void ours_batch_dense(const Tables&tb,const double*x,size_t n,double*out,Scratch&s){
    prepare(x,n,tb,s);
    eval_k_dense<1,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<2,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<3,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<4,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<5,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<6,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<7,SW,ROUND_HORNER>(tb,s);
    eval_k_dense<8,SW,ROUND_HORNER>(tb,s);
    for(size_t p=0;p<n;p++)out[s.sidx[p]]=s.sy[p];
}

template<int SW>
static void ours_batch_current_fast(const Tables&tb,const double*x,size_t n,double*out,Scratch&s){
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

template<class F>
static bool accuracy_gate(const char*name,Tables&tb,const std::vector<double>&x,std::vector<double>&out,F&&fn){
    fn();
    uint64_t mx=0;int gt1=0,gt2=0,bad=0,w=0;
    for(size_t i=0;i<x.size();i++){
        const double ref=tb.reference_x(x[i]);
        const uint64_t u=ulpdist(out[i],ref);
        if(u==UINT64_MAX){bad++;continue;}
        if(u>mx){mx=u;w=(int)i;}
        gt1+=u>1;gt2+=u>2;
    }
    std::printf("ACC_%s MAX_ULP=%llu GT1=%d/%zu GT2=%d/%zu NONFINITE=%d WORST_X=%.17g OUR=%.17g REF=%.17g\n",
        name,(unsigned long long)mx,gt1,x.size(),gt2,x.size(),bad,x[w],out[w],tb.reference_x(x[w]));
    return gt1==0&&bad==0;
}

int main(){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512dq")||
       !__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("fma")){
        std::fprintf(stderr,"AVX512F/DQ/VL+FMA required\n");return 2;
    }

    Tables tb;
    const size_t N=5000;
    auto x=make_inputs(N);
    const double hard[]={std::nextafter(1.0,INFINITY),1.000000000001,1.01,1.1,2.9,3.0,4.0,8.0,10.0,20.0,38.0,50.0,75.0,79.764554025359956,98.0,99.0,std::nextafter(100.0,0.0)};
    for(size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++)x[i]=hard[i];

    std::vector<double>cur(N),strict(N),nr(N),intel(N);
    Scratch sc_cur,sc_strict,sc_nr;

    const bool ok_strict=accuracy_gate("DENSE_STRICT",tb,x,strict,[&](){ours_batch_dense<0,true>(tb,x.data(),N,strict.data(),sc_strict);});
    const bool ok_nr=accuracy_gate("DENSE_NR",tb,x,nr,[&](){ours_batch_dense<0,false>(tb,x.data(),N,nr.data(),sc_nr);});
    if(!ok_strict){std::puts("STRICT_ACCURACY_GATE=FAIL");flint_cleanup();return 3;}
    std::puts("STRICT_ACCURACY_GATE=PASS");
    std::printf("NR_ACCURACY_GATE=%s\n",ok_nr?"PASS":"FAIL");

    uint64_t h=1469598103934665603ULL;
    for(double q:x){h^=std::bit_cast<uint64_t>(q);h*=1099511628211ULL;}
    std::printf("INPUT_HASH=%016llx SAME_INPUTS=1\n",(unsigned long long)h);

    volatile double sink=0.0;
    const size_t sizes[]={100,400,800,2000,5000};
    for(size_t n:sizes){
        int reps=std::max(1,(int)(24000/n));
        ours_batch_current_fast<0>(tb,x.data(),n,cur.data(),sc_cur);
        ours_batch_dense<0,true>(tb,x.data(),n,strict.data(),sc_strict);
        if(ok_nr)ours_batch_dense<0,false>(tb,x.data(),n,nr.data(),sc_nr);
        intel_batch(x.data(),n,intel.data());

        const double tcur=median_ns(n,reps,[&](){ours_batch_current_fast<0>(tb,x.data(),n,cur.data(),sc_cur);sink+=cur[0];});
        const double tstrict=median_ns(n,reps,[&](){ours_batch_dense<0,true>(tb,x.data(),n,strict.data(),sc_strict);sink+=strict[0];});
        double tnr=0.0;
        if(ok_nr)tnr=median_ns(n,reps,[&](){ours_batch_dense<0,false>(tb,x.data(),n,nr.data(),sc_nr);sink+=nr[0];});
        const double tintel=median_ns(n,reps,[&](){intel_batch(x.data(),n,intel.data());sink+=intel[0];});

        std::printf("BATCH=%zu CURRENT_SW0_NS=%.6f DENSE_STRICT_NS=%.6f STRICT_SPEEDUP=%.6f DENSE_NR_NS=%.6f NR_SPEEDUP=%.6f INTEL_NS=%.6f\n",
            n,tcur,tstrict,tcur/tstrict,ok_nr?tnr:-1.0,ok_nr?tcur/tnr:0.0,tintel);
    }
    std::printf("SINK=%.17g\n",(double)sink);
    flint_cleanup();return 0;
}
