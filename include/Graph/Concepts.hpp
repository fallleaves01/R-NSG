#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace Graph {

// 检查是否有get_neighbours方法
template <typename T>
concept GraphLike = requires(const T& obj) { obj.get_neighbours(); };

template <typename T>
concept CandidateGet = requires(const T& obj, size_t index) {
    { obj.get_candidate(index) } -> std::convertible_to<std::vector<size_t>>;
};

}  // namespace Graph

}  // namespace TDFANN