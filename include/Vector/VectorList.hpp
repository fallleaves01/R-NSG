#pragma once
#include <PCH.hpp>

#include <IO/TypeIO.hpp>
#include <Vector/Concepts.hpp>
#include <nlohmann/json.hpp>

namespace TDFANN {

namespace Vector {

template <typename T>
class VectorList {
   public:
    using value_type = T;

    // file operations
    VectorList() = default;
    VectorList(const std::string& dataset);
    void load(const std::string& dataset);

    // calculation operations
    void init_sqrs();
    template <typename Op>
    T dist2(size_t source, const Op& goal) const;
    template <typename Op>
    T dist(size_t source, const Op& goal) const;
    template <typename Op, std::ranges::range R_op>
    std::vector<T> dist_all(const Op& source, const R_op& goal) const;
    Vector::VectorType<T> mean() const;
    template <typename Op>
    T sqr_sub_2dot(size_t, const Op&) const;
    void reorder(const std::vector<size_t>& order);

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
    auto [n, dim] = IO::get_fvecs_size(fin);
    dimension = dim;
    vectors.resize(dimension, n);
    IO::read_fvecs(fin, dimension, vectors.data());
    init_sqrs();
}

template <typename T>
void VectorList<T>::reorder(const std::vector<size_t>& order) {
    assert(order.size() == size());
    Eigen::MatrixXf tmp = vectors;
    auto tmp_s = sqrs;
    for (size_t i = 0; i < order.size(); i++) {
        vectors.col(i) = tmp.col(order[i]);
        sqrs[i] = tmp_s[order[i]];
    }
    spdlog::info("reorder done");
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
    static_assert(
        std::is_convertible_v<Op, size_t> || DotProductWithVectorType<Op, T>,
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
    static_assert(
        std::is_convertible_v<Op, size_t> || DotProductWithVectorType<Op, T>,
        "Op must be convertible to size_t or a vector-like type");

    return dist2(source, goal);
}

template <typename T>
template <typename Op, std::ranges::range R_op>
std::vector<T> VectorList<T>::dist_all(const Op& source,
                                       const R_op& goal) const {
    using Item = decltype(*std::ranges::begin(goal));
    constexpr int O_id = std::is_convertible_v<Op, size_t>
                             ? 1
                             : (DotProductWithVectorType<Op, T> ? -1 : 0);
    constexpr int I_id = std::is_convertible_v<Item, size_t>
                             ? 1
                             : (DotProductWithVectorType<Item, T> ? -1 : 0);
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

template <typename T>
Vector::VectorType<T> VectorList<T>::mean() const {
    Vector::VectorType<T> result(dimension);
    result.setZero();
    for (size_t i = 0; i < size(); i++) {
        result += vectors.col(i);
    }
    result /= size();
    return result;
}

template <typename T>
template <typename Op>
T VectorList<T>::sqr_sub_2dot(size_t idx, const Op& vec) const {
    static_assert(DotProductWithVectorType<Op, T>,
                  "Op must support dot product with VectorType<T>");
    return sqrs[idx] - 2 * vectors.col(idx).dot(vec);
}

}  // namespace Vector

}  // namespace TDFANN