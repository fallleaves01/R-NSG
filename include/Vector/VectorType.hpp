#pragma once

#include <PCH.hpp>

namespace TDFANN {

template <typename T>
struct VectorTypeImpl {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "VectorType can only be used with float or double types");
    using type = std::conditional_t<std::is_same_v<T, float>,
                                    typename Eigen::VectorXf,
                                    typename Eigen::VectorXd>;
};

template <typename T>
using VectorType = typename VectorTypeImpl<T>::type;

}  // namespace TDFANN