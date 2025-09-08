#pragma once

#include <PCH.hpp>

namespace TDFANN {
namespace Utils {

template <typename U>
U collect(std::ranges::range auto&& r) {
    return U(r.begin(), r.end());
}

template <typename T>
    requires std::ranges::range<T>
auto to_vector(T&& r) {
    return collect<std::vector<std::ranges::range_value_t<T>>>(r);
}

}  // namespace Utils
}  // namespace TDFANN