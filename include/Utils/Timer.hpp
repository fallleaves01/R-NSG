#pragma once
#include <PCH.hpp>

namespace TDFANN {

class Timer {
   public:
    static size_t start(const std::string& s) {
        start_time[s] = std::chrono::high_resolution_clock::now();
        return during_time[s];
    }
    static size_t end(const std::string& s) {
        return during_time[s] +=
               std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now() - start_time[s])
                   .count();
    }
    static size_t read(const std::string& s) { return during_time[s]; }
    static size_t add(const std::string& s, size_t val) {
        return during_time[s] += val;
    }

   private:
    static std::map<std::string, std::chrono::high_resolution_clock::time_point>
        start_time;
    static std::map<std::string, long long> during_time;
};
inline std::map<std::string, std::chrono::high_resolution_clock::time_point>
    Timer::start_time;
inline std::map<std::string, long long> Timer::during_time;

}  // namespace TDFANN