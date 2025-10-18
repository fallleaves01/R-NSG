#pragma once
#include <PCH.hpp>

#include <IO/TypeIO.hpp>
#include <Vector/Concepts.hpp>
#include <nlohmann/json.hpp>
#include <type_traits>

namespace TDFANN {

namespace Vector {

template <typename T>
class VectorList {
   public:
    using value_type = T;

    VectorList() = default;
    VectorList(const std::string& dataset);
    void load(const std::string& dataset);

    void init_sqrs();
    template <typename Op>
    T dist2(unsigned source, const Op& goal) const;
    template <typename Op>
    T dist(unsigned source, const Op& goal) const;
    template <typename Op, std::ranges::range R_op>
    std::vector<T> dist_all(const Op& source, const R_op& goal) const;
    template <typename Op>
    void dist_all_into(const Op& source,
                       std::vector<std::pair<T, unsigned>>& id) const;
    Vector::VectorType<T> mean() const;
    template <typename Op>
    T sqr_sub_2dot(unsigned, const Op&) const;
    void reorder(const std::vector<unsigned>& order);

    auto operator[](unsigned index) const { return vectors.col(index); }
    unsigned size() const { return vectors.cols(); }
    unsigned dim() const { return dimension; }
    const T* data() const { return vectors.data(); }

   private:
    Eigen::MatrixXf vectors;
    std::vector<T> sqrs;
    unsigned dimension = 0;
};

}  // namespace Vector

}  // namespace TDFANN

//>===========================================================<

// Implementation of VectorList methods
namespace TDFANN {

namespace Vector {

const unsigned S = 128;

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
    auto ext_pos = filename.find_last_of('.');
    if (ext_pos == std::string::npos) {
        spdlog::error("File extension not found: {}", filename);
        throw std::runtime_error("File extension not found");
    }
    auto ext = filename.substr(ext_pos);
    if (ext == ".fvecs") {
        auto [n, dim] = IO::get_fvecs_size(fin);
        dimension = dim;
        vectors.resize(dimension, n);
        IO::read_fvecs(fin, dimension, vectors.data());
    } else if (ext == ".u8bin") {
        auto [n, dim] = IO::get_u8bin_size(fin);
        dimension = dim;
        vectors.resize(dimension, n);
        IO::read_u8bin(fin, dimension, vectors.data());
    } else {
        spdlog::error("Unsupported file extension: {}", ext);
        throw std::runtime_error("Unsupported file extension");
    }
    init_sqrs();
}

template <typename T>
void VectorList<T>::reorder(const std::vector<unsigned>& order) {
    assert(order.size() == size());
    Eigen::MatrixXf tmp = vectors;
    auto tmp_s = sqrs;
    for (unsigned i = 0; i < order.size(); i++) {
        vectors.col(i) = tmp.col(order[i]);
        sqrs[i] = tmp_s[order[i]];
    }
    spdlog::info("reorder done");
}

template <typename T>
void VectorList<T>::init_sqrs() {
    spdlog::info("Initializing squared norms...");
    sqrs.resize(size());
    for (unsigned i = 0; i < size(); ++i) {
        sqrs[i] = vectors.col(i).squaredNorm();
    }
    spdlog::info("Squared norms initialized.");
}

template <typename T>
template <typename Op>
T VectorList<T>::dist2(unsigned source, const Op& goal) const {
    static_assert(
        std::is_convertible_v<Op, unsigned> || DotProductWithVectorType<Op, T>,
        "Op must be convertible to unsigned or a vector-like type");

    if constexpr (std::is_convertible_v<Op, unsigned>) {
        return sqrs[source] + sqrs[goal] -
               2 * vectors.col(source).dot(vectors.col(goal));
    } else {
        return (vectors.col(source) - goal).squaredNorm();
    }
}

template <typename T>
template <typename Op>
T VectorList<T>::dist(unsigned source, const Op& goal) const {
    static_assert(
        std::is_convertible_v<Op, unsigned> || DotProductWithVectorType<Op, T>,
        "Op must be convertible to unsigned or a vector-like type");

    return dist2(source, goal);
}

template <typename T>
template <typename Op>
void VectorList<T>::dist_all_into(
    const Op& source,
    std::vector<std::pair<T, unsigned>>& p) const {
    constexpr int O_id = std::is_convertible_v<Op, unsigned>
                             ? 1
                             : (DotProductWithVectorType<Op, T> ? -1 : 0);
    if constexpr (O_id == 1) {
        for (auto& [dis, idx] : p) {
            dis = sqrs[idx] - 2 * vectors.col(idx).dot(vectors.col(source));
        }
    } else {
        for (size_t i = 0; i < p.size(); i++) {
            if (i + 4 < p.size()) {
                __builtin_prefetch(vectors.col(p[i + 2].second).data());
            }
            auto& [dis, idx] = p[i];
            dis = sqrs[idx] - 2 * source.dot(vectors.col(idx));
        }
    }
}

template <typename T>
template <typename Op, std::ranges::range R_op>
std::vector<T> VectorList<T>::dist_all(const Op& source,
                                       const R_op& goal) const {
    using Item = decltype(*std::ranges::begin(goal));
    constexpr int O_id = std::is_convertible_v<Op, unsigned>
                             ? 1
                             : (DotProductWithVectorType<Op, T> ? -1 : 0);
    constexpr int I_id = std::is_convertible_v<Item, unsigned>
                             ? 1
                             : (DotProductWithVectorType<Item, T> ? -1 : 0);
    static_assert(O_id != 0,
                  "Op must be convertible to unsigned or a vector-like type");
    static_assert(I_id != 0,
                  "Item must be convertible to unsigned or a vector-like type");
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
    for (unsigned i = 0; i < size(); i++) {
        result += vectors.col(i);
    }
    result /= size();
    return result;
}

template <typename T>
template <typename Op>
T VectorList<T>::sqr_sub_2dot(unsigned idx, const Op& vec) const {
    static_assert(DotProductWithVectorType<Op, T>,
                  "Op must support dot product with VectorType<T>");
    return sqrs[idx] - 2 * vectors.col(idx).dot(vec);
}

}  // namespace Vector

}  // namespace TDFANN