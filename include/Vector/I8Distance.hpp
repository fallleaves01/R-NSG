#pragma once

#include <cstdint>

// GCC 11 in the current environment lacks complete C++ support for SimSIMD's
// native f16/bf16 and AVX512-FP16 paths. RNSG only needs signed-int8 kernels.
#ifndef SIMSIMD_NATIVE_F16
#define SIMSIMD_NATIVE_F16 0
#endif
#ifndef SIMSIMD_NATIVE_BF16
#define SIMSIMD_NATIVE_BF16 0
#endif
#ifndef SIMSIMD_TARGET_SAPPHIRE
#define SIMSIMD_TARGET_SAPPHIRE 0
#endif

#include <simsimd/simsimd.h>

namespace TDFANN::Vector::I8Distance {

inline float dot(const std::int8_t* a, const std::int8_t* b, unsigned dim) {
    simsimd_distance_t result = 0;
    simsimd_dot_i8(reinterpret_cast<const simsimd_i8_t*>(a),
                   reinterpret_cast<const simsimd_i8_t*>(b),
                   static_cast<simsimd_size_t>(dim), &result);
    return static_cast<float>(result);
}

inline float l2sq(const std::int8_t* a, const std::int8_t* b, unsigned dim) {
    simsimd_distance_t result = 0;
    simsimd_l2sq_i8(reinterpret_cast<const simsimd_i8_t*>(a),
                    reinterpret_cast<const simsimd_i8_t*>(b),
                    static_cast<simsimd_size_t>(dim), &result);
    return static_cast<float>(result);
}

}  // namespace TDFANN::Vector::I8Distance
