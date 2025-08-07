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

}  // namespace VectorLib

}  // namespace TDFANN