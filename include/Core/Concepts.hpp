#pragma once

#include <PCH.hpp>

#include <Graph/Concepts.hpp>
#include <Vector/Concepts.hpp>
#include "Graph/GraphIndex.hpp"

namespace TDFANN {

template <typename T>
concept IndexOrList =
    std::convertible_to<T, size_t> || Graph::IndexList<std::decay_t<T>>;

template <typename Op, typename T>
concept IndexOrVector =
    std::convertible_to<Op, size_t> || Vector::DotProductWithVectorType<Op, T>;

template <typename Op>
concept IsTDFG = std::is_same_v<Op, Graph::TDGraphIndexBase> ||
                 std::is_same_v<Op, Graph::TDGraphIndexBase::TDGraphIndex>;

}  // namespace TDFANN
