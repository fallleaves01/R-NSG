#pragma once

#include <PCH.hpp>

#include <Graph/Concepts.hpp>
#include <Vector/Concepts.hpp>

namespace TDFANN {

template <typename T>
concept IndexOrList =
    std::convertible_to<T, size_t> || Graph::IndexList<std::decay_t<T>>;

template <typename Op, typename T>
concept IndexOrVector =
    std::convertible_to<Op, size_t> || Vector::DotProductWithVectorType<Op, T>;

}  // namespace TDFANN