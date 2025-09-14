#pragma once

#include <PCH.hpp>

namespace TDFANN {

namespace Vector {

/// @concept vector_like
/// @brief 定义向量类类型的要求
template <typename T>
concept VectorLike = requires(const T& v, size_t index) {
    // 必须支持 operator[] 访问
    { v[index] } -> std::convertible_to<double>;

    // 必须提供 size() 方法
    { v.size() } -> std::same_as<size_t>;

    // 必须有迭代器支持（可选但推荐）
    { v.begin() } -> std::input_iterator;
    { v.end() } -> std::input_iterator;
};

/// @concept indexable
/// @brief 定义支持索引操作的类型
template <typename T>
concept indexable = requires(T container, size_t index) {
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
};

template <typename T>
using VectorListType = typename VectorListTypeImpl<T>::type;

template <typename T>
using VectorType = decltype(std::declval<const VectorListType<T>>().col(std::declval<size_t>()));

template<typename Op, typename T>
concept DotProductWithVectorType = requires(const Op& op, const VectorType<T>& vec) {
    { vec.dot(op) } -> std::same_as<T>;
};

}  // namespace VectorLib

}  // namespace TDFANN