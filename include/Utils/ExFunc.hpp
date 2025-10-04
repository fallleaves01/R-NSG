#pragma once

#include <PCH.hpp>

#define GET(member)                                                         \
    [](auto&& x)                                                            \
        noexcept(noexcept((std::forward<decltype(x)>(x).member)))           \
        -> decltype(auto) {                                                 \
            return std::forward<decltype(x)>(x).member;                     \
        }

namespace TDFANN {
namespace Utils {

template <typename U>
U collect(std::ranges::range auto&& r) {
    return U(r.begin(), r.end());
}

template <typename T>
    requires std::ranges::range<T>
auto to_vector(T&& r) {
    std::vector<std::ranges::range_value_t<T>> res;
    std::ranges::copy(r, std::back_inserter(res));
    return res;
}

template <typename T>
std::vector<T> sorted_vec(std::vector<T> vec) {
    std::ranges::sort(vec);
    return vec;
}

template <typename T>
std::pair<std::vector<size_t>, std::vector<size_t>> order_of_label(
    const std::vector<T>& label) {
    std::vector<size_t> index(label.size());
    std::iota(index.begin(), index.end(), 0);
    std::ranges::sort(index, [&](size_t i, size_t j) {
        return std::pair{label[i], i} < std::pair{label[j], j};
    });
    std::vector<size_t> pos(label.size());
    for (size_t i = 0; i < index.size(); i++) {
        pos[index[i]] = i;
    }
    return {index, pos};
}

}  // namespace Utils
}  // namespace TDFANN