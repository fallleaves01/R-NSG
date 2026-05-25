#pragma once

#include <PCH.hpp>
#include <omp.h>

namespace TDFANN::Utils {

inline int& thread_count_override() {
    static int value = 0;
    return value;
}

inline void set_thread_count_override(int threads) {
    thread_count_override() = threads;
    if (threads > 0) {
        omp_set_num_threads(std::max(1, threads));
    }
}

inline int configured_thread_count(int default_cap = 64) {
    const int omp_threads = std::max(1, omp_get_max_threads());
    const int override_threads = thread_count_override();
    if (override_threads > 0) {
        return std::max(1, std::min(override_threads, omp_threads));
    }
    const char* raw = std::getenv("TDFANN_THREAD_CAP");
    if (raw == nullptr || *raw == '\0') {
        raw = std::getenv("RNSG_THREAD_CAP");
    }
    if (raw == nullptr || *raw == '\0') {
        return std::max(1, std::min(default_cap, omp_threads));
    }

    char* end = nullptr;
    const long cap = std::strtol(raw, &end, 10);
    if (end == raw) {
        return std::max(1, std::min(default_cap, omp_threads));
    }
    if (cap <= 0) {
        return omp_threads;
    }
    return std::max(1, std::min(static_cast<int>(cap), omp_threads));
}

}  // namespace TDFANN::Utils
