#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace Graph {

// 检查是否有get_neighbours方法
template <typename T>
concept GraphLike = requires(const T& obj, size_t id) { obj.get_neighbours_id(id); };

}  // namespace Graph

}  // namespace TDFANN