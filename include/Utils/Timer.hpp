#pragma once
#include <PCH.hpp>

namespace TDFANN {

class Timer {
   public:
    static void start(const std::string& s) {
        start_time[s] = std::chrono::high_resolution_clock::now();
    }
    static int64_t end(const std::string& s) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now() - start_time[s])
            .count();
    }

   private:
    static std::map<std::string, std::chrono::high_resolution_clock::time_point>
        start_time;
};
inline std::map<std::string, std::chrono::high_resolution_clock::time_point>
    Timer::start_time;

}  // namespace TDFANN