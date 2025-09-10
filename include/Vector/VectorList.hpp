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
    T dist2(size_t source, const Op& goal) const;
    template <typename Op>
    T dist(size_t source, const Op& goal) const;
    template <typename Op, std::ranges::range R_op>
    std::vector<T> dist_all(const Op& source, const R_op& goal) const;

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
T VectorList<T>::dist2(size_t source, const Op& goal) const {
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
T VectorList<T>::dist(size_t source, const Op& goal) const {
    static_assert(std::is_convertible_v<Op, size_t> ||
                      std::is_convertible_v<Op, VectorType<T>>,
                  "Op must be convertible to size_t or a vector-like type");

    return dist2(source, goal);
}

template <typename T>
template <typename Op, std::ranges::range R_op>
std::vector<T> VectorList<T>::dist_all(const Op& source,
                                       const R_op& goal) const {
    using Item = decltype(*std::ranges::begin(goal));
    static_assert(std::is_convertible_v<Op, size_t> ||
                      std::is_convertible_v<Op, VectorType<T>>,
                  "Op must be convertible to size_t or a vector-like type");

    static_assert(std::is_convertible_v<Item, size_t> ||
                      std::is_convertible_v<Item, VectorType<T>>,
                  "Item must be convertible to size_t or a vector-like type");

    std::vector<T> result;
    result.reserve(goal.size());
    if constexpr (std::is_convertible_v<Op, size_t> ||
                  std::is_convertible_v<Item, VectorType<T>>) {
        for (const auto& g : goal) {
            result.push_back(dist(source, g));
        }
    } else {
        T now_2 = source.squaredNorm();
        for (const auto& g : goal) {
            result.push_back(now_2 + sqrs[g] - 2 * source.dot(vectors[g]));
        }
    }
    return result;
}

}  // namespace Vector

}  // namespace TDFANN