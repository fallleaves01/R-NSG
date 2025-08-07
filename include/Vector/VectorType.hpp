#pragma once

#include <PCH.hpp>
#include <Vector/Concepts.hpp>

namespace TDFANN {

namespace Vector {

template <typename T>
struct VectorTypeImpl {
    static_assert(IsFloat<T>,
                  "VectorType can only be used with float or double types");
    using type = std::conditional_t<std::is_same_v<T, float>,
                                    typename Eigen::VectorXf,
                                    typename Eigen::VectorXd>;
};

template <typename T>
using VectorType = typename VectorTypeImpl<T>::type;

}  // namespace Vector

}  // namespace TDFANN