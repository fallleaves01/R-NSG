#pragma once
#include <Vector/VectorType.hpp>

namespace TDFANN {

namespace Vector {

template <typename T>
class VectorList {
   public:
    using value_type = T;

    // file operations
    VectorList() = default;
    VectorList(const std::string& filename);
    void save(const std::string& filename) const;
    void load(const std::string& filename);

    // calculation operations
    void init_sqrs();
    template <typename Op>
    T dist2(size_t source, const Op& goal);
    template <typename Op>
    T dist(size_t source, const Op& goal);

    // struct operations
    const VectorType<T>& operator[](size_t index) const {
        return vectors[index];
    }
    VectorType<T>& operator[](size_t index) { return vectors[index]; }
    size_t size() const { return vectors.size(); }
    size_t dim() const { return dimension; }

   private:
    std::vector<VectorType<T>> vectors;
    std::vector<T> sqrs;     // 用于存储平方和
    unsigned dimension = 0;  // 向量维度
};

}  // namespace Vector

}  // namespace TDFANN

//>===========================================================<

// Implementation of VectorList methods
namespace TDFANN {

namespace Vector {
// file operations
template <typename T>
VectorList<T>::VectorList(const std::string& filename) {
    load(filename);
}

template <typename T>
void VectorList<T>::save(const std::string& filename) const {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout.is_open()) {
        spdlog::error("Failed to open vector file for writing: {}", filename);
        throw std::runtime_error("Failed to open vector file for writing");
    }
    spdlog::info("Saving vector list to file: {}", filename);
    for (const auto& f : vectors) {
        unsigned n = f.size();
        if (!fout.write(reinterpret_cast<const char*>(&n), sizeof(unsigned))) {
            spdlog::error("Failed to write vector size to file");
            throw std::runtime_error("Failed to write vector size to file");
        }
        if (!fout.write(reinterpret_cast<const char*>(f.data()),
                        n * sizeof(T))) {
            spdlog::error("Failed to write vector data to file");
            throw std::runtime_error("Failed to write vector data to file");
        }
    }
}

template <typename T>
void VectorList<T>::load(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        spdlog::error("Failed to open vector file: {}", filename);
        throw std::runtime_error("Failed to open vector file");
    }
    spdlog::info("Loading vector list from file: {}", filename);
    vectors.clear();
    dimension = 0;  // 重置维度
    unsigned n;
    while (fin.read(reinterpret_cast<char*>(&n), sizeof(unsigned))) {
        VectorType<T> vec(n);
        if (!fin.read(reinterpret_cast<char*>(vec.data()), n * sizeof(T))) {
            spdlog::error("Failed to read vector data from file");
            throw std::runtime_error("Failed to read vector data from file");
        }
        vectors.push_back(vec);
        if (dimension == 0) {
            dimension = n;  // 设置向量维度
        } else if (dimension != n) {
            spdlog::error("Inconsistent vector dimensions in file: {}",
                          filename);
            throw std::runtime_error("Inconsistent vector dimensions in file");
        }
    }
    init_sqrs();
}

// calculation operations
template <typename T>
void VectorList<T>::init_sqrs() {
    sqrs.resize(vectors.size());
    for (size_t i = 0; i < vectors.size(); ++i) {
        sqrs[i] = vectors[i].squaredNorm();
    }
}

template <typename T>
template <typename Op>
T VectorList<T>::dist2(size_t source, const Op& goal) {
    static_assert(std::is_convertible_v<Op, size_t> ||
                      std::is_convertible_v<Op, VectorType<T>>,
                  "Op must be convertible to size_t or a vector-like type");

    if constexpr (std::is_convertible_v<Op, size_t>) {
        return sqrs[source] + sqrs[goal] -
               2 * vectors[source].dot(vectors[goal]);
    } else {
        return (vectors[source] - goal).squaredNorm();
    }
}

template <typename T>
template <typename Op>
T VectorList<T>::dist(size_t source, const Op& goal) {
    static_assert(std::is_convertible_v<Op, size_t> ||
                      std::is_convertible_v<Op, VectorType<T>>,
                  "Op must be convertible to size_t or a vector-like type");

    return dist2(source, goal);
}

}  // namespace Vector

}  // namespace TDFANN