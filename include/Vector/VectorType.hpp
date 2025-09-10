#pragma once

#include <PCH.hpp>
#include <Vector/Concepts.hpp>

namespace TDFANN {

namespace Vector {

template <typename T>
struct VectorListTypeImpl {
    static_assert(IsFloat<T>,
                  "VectorType can only be used with float or double types");
    using type = std::conditional_t<std::is_same_v<T, float>,
                                    typename Eigen::MatrixXf,
                                    typename Eigen::MatrixXd>;
};

template <typename T>
using VectorListType = typename VectorListTypeImpl<T>::type;

template <typename T>
using VectorType = decltype(std::declval<const VectorListType<T>>().row(std::declval<size_t>()));

}  // namespace Vector

}  // namespace TDFANN