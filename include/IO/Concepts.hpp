#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace IO {

// 检查是否有save方法
template <typename T>
concept HasSaveFSMethod =
    requires(const T& obj, std::ostream& os) { obj.save(os); };

// 检查是否有load方法
template <typename T>
concept HasLoadFSMethod = requires(T& obj, std::istream& is) { obj.load(is); };

// 检查是否是平凡类型（可直接二进制读写）
template <typename T>
concept TriviallySerializable = std::is_trivially_copyable_v<T> &&
                                !HasSaveFSMethod<T> && !HasLoadFSMethod<T>;

// 检查是否是标准容器
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