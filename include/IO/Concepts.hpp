#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace IO {

template <typename T>
concept HasSaveFSMethod = requires(const T& obj, std::ostream& os) {
    { obj.save(os) } -> std::convertible_to<bool>;
};

template <typename T>
concept HasLoadFSMethod = requires(T& obj, std::istream& is) {
    { obj.load(is) } -> std::convertible_to<bool>;
};

template <typename T>
concept TriviallySerializable = std::is_trivially_copyable_v<T> &&
                                !HasSaveFSMethod<T> && !HasLoadFSMethod<T>;

template <typename T>
concept StandardContainer = requires(T& container) {
    typename T::value_type;
    typename T::size_type;
    typename T::iterator;
    container.begin();
    container.end();
    container.size();
};

}  // namespace IO

}  // namespace TDFANN