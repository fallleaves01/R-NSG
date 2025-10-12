#pragma once
#include <PCH.hpp>

#include <IO/Concepts.hpp>
#include <nlohmann/json.hpp>

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
        unsigned size = obj.size();
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
        unsigned size;
        if (!load(fin, size)) {
            return false;
        }
        std::vector<typename T::value_type> temp(size);
        for (unsigned i = 0; i < size; ++i) {
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

inline std::pair<unsigned, unsigned> get_fvecs_size(std::ifstream& fin) {
    unsigned dimension = 0;
    fin.read((char*)&dimension, sizeof(unsigned));
    fin.seekg(0, std::ios::end);
    std::ios::pos_type ss = fin.tellg();
    size_t file_size = ss;
    unsigned n = file_size / (dimension + 1) / sizeof(float);
    fin.seekg(0, std::ios::beg);
    spdlog::info("Vector dimension: {}, size: {}", dimension, n);
    return {n, dimension};
}

template <typename T>
void read_fvecs(std::ifstream& fin, unsigned dimension, T* data) {
    unsigned tmp, idx = 0;
    while (fin.read(reinterpret_cast<char*>(&tmp), sizeof(unsigned))) {
        if (!fin.read(reinterpret_cast<char*>(data + idx * dimension),
                      dimension * sizeof(T))) {
            spdlog::error("Failed to read vector data from file");
            throw std::runtime_error("Failed to read vector data from file");
        }
        idx++;
        if (tmp != dimension) {
            spdlog::error("Inconsistent vector dimensions");
            throw std::runtime_error("Inconsistent vector dimensions in file");
        }
    }
}

inline std::vector<unsigned> load_json_to_vec(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.good()) {
        spdlog::error("Failed to open json file: {}", filename);
        throw std::runtime_error("Failed to open json file");
    }
    nlohmann::json j;
    fin >> j;
    if (!j.is_array()) {
        spdlog::error("JSON file does not contain an array: {}", filename);
        throw std::runtime_error("JSON file does not contain an array");
    }
    std::vector<unsigned> result;
    result.reserve(j.size());
    for (const auto& item : j) {
        if (!item.is_number_unsigned()) {
            spdlog::error("JSON array contains non-unsigned integer values");
            throw std::runtime_error(
                "JSON array contains non-unsigned integer values");
        }
        result.push_back(item.get<unsigned>());
    }
    return result;
}
}  // namespace IO

}  // namespace TDFANN