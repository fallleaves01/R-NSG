#pragma once
#include <IO/Concepts.hpp>
#include <PCH.hpp>

namespace TDFANN {

namespace IO {

template <typename T>
bool save(std::ostream& fout, const T& obj);

template <typename T>
bool load(std::istream& fin, T& obj);

//>===========================================================<

// implementation of IO functions

template <typename T>
bool save(std::ostream& fout, const T& obj) {
    if (!fout) {
        return false;
    }
    if constexpr (HasSaveFSMethod<T>) {
        return obj.save(fout);
    } else if constexpr (TriviallySerializable<T>) {
        fout.write(reinterpret_cast<const char*>(&obj), sizeof(T));
        return fout.good();
    } else if constexpr (StandardContainer<T>) {
        size_t size = obj.size();
        if (!save(fout, size)) {
            return false;
        }
        for (const auto& item : obj) {
            if (!save(fout, item)) {
                return false;
            }
        }
        return true;
    } else {
        static_assert(false, "Type T is not serializable");
    }
}

template <typename T>
bool load(std::istream& fin, T& obj) {
    if (!fin) {
        return false;
    }
    if constexpr (HasLoadFSMethod<T>) {
        return obj.load(fin);
    } else if constexpr (TriviallySerializable<T>) {
        fin.read(reinterpret_cast<char*>(&obj), sizeof(T));
        return fin.good();
    } else if constexpr (StandardContainer<T>) {
        size_t size;
        if (!load(fin, size)) {
            return false;
        }
        std::vector<typename T::value_type> temp(size);
        for (size_t i = 0; i < size; ++i) {
            if (!load(fin, temp[i])) {
                return false;
            }
        }
        obj = T(temp.begin(), temp.end());
        return true;
    } else {
        static_assert(false, "Type T is not deserializable");
    }
}

template <typename T>
std::optional<T> load(std::istream& fin) {
    T obj;
    if (!load(fin, obj)) {
        return std::nullopt;
    }
    return obj;
}

}  // namespace IO

}  // namespace TDFANN