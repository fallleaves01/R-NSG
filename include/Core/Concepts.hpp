#pragma once

#include <PCH.hpp>
#include <Vector/Concepts.hpp>

namespace TDFANN {

template <typename T>
concept IndexOrList =
    std::convertible_to<T, size_t> ||
    (std::ranges::range<T> &&
     std::convertible_to<std::ranges::range_value_t<T>, size_t>);

template <typename Op, typename T>
concept IndexOrVector = std::convertible_to<Op, size_t> ||
                        Vector::DotProductWithVectorType<Op, T>;

}