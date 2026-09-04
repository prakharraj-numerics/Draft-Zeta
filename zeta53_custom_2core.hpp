#pragma once

/* ZETA53 custom2 candidate: permanent 2-core scheduler.

   Port of the validated custom2 execution layer used by the SINE53 project.
   Scheduling only: this does not alter zeta mathematics, local-center tables,
   polynomial evaluation, pole evaluation, or output semantics.

   Execution model on the Intel Xeon 6973P-C runner:
     - caller thread pinned to CPU0
     - one permanent helper pinned to CPU2
     - work split on a 32-double boundary
     - release/acquire generation handoff and completion counter
     - no queue, work stealing, task allocation, or per-call thread creation

   Fewer than two complete 32-value blocks execute on the caller only.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

class Zeta53CustomPermanent2Core {
public:
    using fn_t = void (*)(const double *, std::size_t, double *);

    explicit Zeta53CustomPermanent2Core(fn_t fn)
        : generation_(0), completed_(0), stop_(false),
          in_(nullptr), n2_(0), out_(nullptr), fn_(fn) {
        pin_current_thread(0);
        helper_ = std::thread([this] { helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Zeta53CustomPermanent2Core() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Zeta53CustomPermanent2Core(const Zeta53CustomPermanent2Core&) = delete;
    Zeta53CustomPermanent2Core& operator=(const Zeta53CustomPermanent2Core&) = delete;

    void run(const double *in, std::size_t n, double *out) {
        if (!n) return;
        const std::size_t full32 = n / 32;
        if (full32 < 2) {
            fn_(in, n, out);
            return;
        }

        const std::size_t split_blocks = full32 / 2;
        const std::size_t split = split_blocks * 32;
        if (split == 0 || split >= n) {
            fn_(in, n, out);
            return;
        }

        in_ = in + split;
        n2_ = n - split;
        out_ = out + split;

        const std::uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        fn_(in, split, out);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

private:
    static void pin_current_thread(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    void helper_loop() {
        pin_current_thread(2);
        helper_ready_.store(true, std::memory_order_release);
        std::uint64_t seen = generation_.load(std::memory_order_relaxed);
        for (;;) {
            std::uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) _mm_pause();
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;

            const double *in = in_;
            const std::size_t n = n2_;
            double *out = out_;
            fn_(in, n, out);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<std::uint64_t> generation_;
    alignas(64) std::atomic<std::uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;

    const double *in_;
    std::size_t n2_;
    double *out_;
    fn_t fn_;
};
