#pragma once

#include <PCH.hpp>

namespace TDFANN {

namespace Vector {

template <typename T>
concept VectorLike = requires(const T& v, unsigned index) {
    { v[index] } -> std::convertible_to<double>;

    { v.size() } -> std::same_as<size_t>;

    { v.begin() } -> std::input_iterator;
    { v.end() } -> std::input_iterator;
};

template <typename T>
concept indexable = requires(T container, unsigned index) {
    { container[index] } -> std::convertible_to<double>;
};

template <typename T>
concept IsFloat = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
struct VectorListTypeImpl {
    static_assert(IsFloat<T>,
                  "VectorType can only be used with float or double types");
    using type = std::conditional_t<std::is_same_v<T, float>,
                                    typename Eigen::MatrixXf,
                                    typename Eigen::MatrixXd>;
    using v_type = std::conditional_t<std::is_same_v<T, float>,
                                     typename Eigen::VectorXf,
                                     typename Eigen::VectorXd>;
};

template <typename T>
using VectorListType = typename VectorListTypeImpl<T>::type;

template <typename T>
using VectorType = typename VectorListTypeImpl<T>::v_type;

template<typename Op, typename T>
concept DotProductWithVectorType = requires(const Op& op, const VectorType<T>& vec) {
    { vec.dot(op) } -> std::same_as<T>;
};

}  // namespace VectorLib

}  // namespace TDFANN