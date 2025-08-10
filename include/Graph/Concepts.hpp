#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace Graph {

// 检查是否有get_neighbours方法
template <typename T>
concept GraphLike = requires(const T& obj) { obj.get_neighbours(); };

}  // namespace Graph

}  // namespace TDFANN