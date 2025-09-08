#pragma once
#include <PCH.hpp>

namespace TDFANN {

template <typename T>
class Recorder {
   public:
    static T read(const std::string& s) { return rec[s]; }
    static T add(const std::string& s, T val) { return rec[s] += val; }
    static T set(const std::string& s, T val) { return rec[s] = val; }

   private:
    static std::map<std::string, T> rec;
};

template <typename T>
inline std::map<std::string, T> Recorder<T>::rec;

}  // namespace TDFANN