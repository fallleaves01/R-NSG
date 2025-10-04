#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace Graph {

template <typename T>
concept IndexList = std::ranges::range<T> &&
                    std::convertible_to<std::ranges::range_value_t<T>, unsigned>;

// 检查是否有get_neighbours方法
template <typename T>
concept GraphLike = requires(const T& obj, unsigned id) {
    { obj.get_neighbours_id(id) } -> IndexList;
};

}  // namespace Graph

}  // namespace TDFANN