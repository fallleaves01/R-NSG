#pragma once
#include <Vector/Concepts.hpp>

namespace TDFANN {

namespace Vector {

template <typename T>
class VectorList {
   public:
    using value_type = T;

    // file operations
    VectorList() = default;
    VectorList(const std::string& filename);
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
    auto operator[](size_t index) const { return vectors.col(index); }
    size_t size() const { return vectors.cols(); }
    size_t dim() const { return dimension; }

   private:
    Eigen::MatrixXf vectors;
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
void VectorList<T>::load(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        spdlog::error("Failed to open vector file: {}", filename);
        throw std::runtime_error("Failed to open vector file");
    }
    spdlog::info("Loading vector list from file: {}", filename);

    fin.read((char*)&dimension, sizeof(unsigned));
    fin.seekg(0, std::ios::end);
    std::ios::pos_type ss = fin.tellg();
    size_t file_size = ss;
    unsigned n = file_size / (dimension + 1) / sizeof(T);
    fin.seekg(0, std::ios::beg);
    vectors.resize(dimension, n);
    unsigned tmp, idx = 0;
    spdlog::info("Vector dimension: {}, size: {}", dimension, n);

    T *data = vectors.data();
    while (fin.read(reinterpret_cast<char*>(&tmp), sizeof(unsigned))) {
        // spdlog::info("Reading vector {}/{} with dim = {}", idx + 1, n, tmp);
        if (!fin.read(reinterpret_cast<char*>(data + idx * dimension),
                      dimension * sizeof(T))) {
            spdlog::error("Failed to read vector data from file");
            throw std::runtime_error("Failed to read vector data from file");
        }
        idx++;
        if (tmp != dimension) {
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
    sqrs.resize(size());
    for (size_t i = 0; i < size(); ++i) {
        sqrs[i] = vectors.col(i).squaredNorm();
    }
}

template <typename T>
template <typename Op>
T VectorList<T>::dist2(size_t source, const Op& goal) const {
    static_assert(std::is_convertible_v<Op, size_t> ||
                      DotProductWithVectorType<Op, T>,
                  "Op must be convertible to size_t or a vector-like type");

    if constexpr (std::is_convertible_v<Op, size_t>) {
        return sqrs[source] + sqrs[goal] -
               2 * vectors.col(source).dot(vectors.col(goal));
    } else {
        return (vectors.col(source) - goal).squaredNorm();
    }
}

template <typename T>
template <typename Op>
T VectorList<T>::dist(size_t source, const Op& goal) const {
    static_assert(std::is_convertible_v<Op, size_t> ||
                      DotProductWithVectorType<Op, T>,
                  "Op must be convertible to size_t or a vector-like type");

    return dist2(source, goal);
}

template <typename T>
template <typename Op, std::ranges::range R_op>
std::vector<T> VectorList<T>::dist_all(const Op& source,
                                       const R_op& goal) const {
    using Item = decltype(*std::ranges::begin(goal));
    constexpr int O_id =
        std::is_convertible_v<Op, size_t>
            ? 1
            : (std::is_convertible_v<Op, VectorType<T>> ? -1 : 0);
    constexpr int I_id =
        std::is_convertible_v<Item, size_t>
            ? 1
            : (std::is_convertible_v<Item, VectorType<T>> ? -1 : 0);
    static_assert(O_id != 0,
                  "Op must be convertible to size_t or a vector-like type");
    static_assert(I_id != 0,
                  "Item must be convertible to size_t or a vector-like type");
    static_assert(O_id != -1 || I_id != -1,
                  "At least one of Op or Item must be an index type");

    std::vector<T> result;
    result.reserve(goal.size());
    if constexpr (O_id == 1 || I_id == -1) {
        for (const auto& g : goal) {
            result.push_back(dist(source, g));
        }
    } else {
        T now_2 = source.squaredNorm();
        for (const auto& g : goal) {
            result.push_back(now_2 + sqrs[g] - 2 * source.dot(vectors.col(g)));
        }
    }
    return result;
}

}  // namespace Vector

}  // namespace TDFANN