#pragma once
#include <PCH.hpp>

#include <IO/TypeIO.hpp>
#include <Vector/Concepts.hpp>
#include <Vector/I8Distance.hpp>
#include <nlohmann/json.hpp>
#include <type_traits>

namespace TDFANN {

namespace Vector {

template <typename T>
class VectorList {
   public:
    using value_type = T;

    class VectorRef {
       public:
        using value_type = T;

        VectorRef(const VectorList* owner, unsigned index)
            : owner(owner), index(index) {}

        T operator[](size_t dim) const {
            return owner->value_at(index, static_cast<unsigned>(dim));
        }

        size_t size() const { return owner->dim(); }

        T squaredNorm() const {
            T sum = 0;
            for (unsigned d = 0; d < owner->dim(); ++d) {
                const T v = (*this)[d];
                sum += v * v;
            }
            return sum;
        }

        template <typename Op>
        T dot(const Op& other) const {
            T sum = 0;
            for (unsigned d = 0; d < owner->dim(); ++d) {
                sum += (*this)[d] * static_cast<T>(other[d]);
            }
            return sum;
        }

        const T* float_data() const {
            return owner->storage == Storage::Float32
                       ? owner->float_ptr(index)
                       : nullptr;
        }

        const std::int8_t* int8_data() const {
            return owner->storage == Storage::Int8
                       ? owner->int8_ptr(index)
                       : nullptr;
        }

       private:
        const VectorList* owner = nullptr;
        unsigned index = 0;
    };

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

    VectorRef operator[](unsigned index) const { return VectorRef(this, index); }
    unsigned size() const { return vector_count; }
    unsigned dim() const { return dimension; }
    bool is_i8bin() const { return storage == Storage::Int8; }
    std::vector<float> to_float_data() const;
    const T* data() const {
        if (storage != Storage::Float32) {
            throw std::runtime_error(
                "Raw i8bin VectorList has no contiguous float data()");
        }
        return vectors.data();
    }

   private:
    enum class Storage { Float32, Int8 };

    T value_at(unsigned index, unsigned dim) const {
        if (storage == Storage::Int8) {
            return static_cast<T>(
                i8_vectors[static_cast<size_t>(index) * dimension + dim]);
        }
        return static_cast<T>(vectors(dim, index));
    }

    const T* float_ptr(unsigned index) const {
        return vectors.data() + static_cast<size_t>(index) * dimension;
    }

    const std::int8_t* int8_ptr(unsigned index) const {
        return i8_vectors.data() + static_cast<size_t>(index) * dimension;
    }

    static T dot_i8_ptr(const std::int8_t* a,
                        const std::int8_t* b,
                        unsigned dim) {
        return static_cast<T>(I8Distance::dot(a, b, dim));
    }

    static T l2sq_i8_ptr(const std::int8_t* a,
                         const std::int8_t* b,
                         unsigned dim) {
        return static_cast<T>(I8Distance::l2sq(a, b, dim));
    }

    T dot_index_index(unsigned lhs, unsigned rhs) const {
        if (storage == Storage::Float32) {
            return static_cast<T>(vectors.col(lhs).dot(vectors.col(rhs)));
        }
        return dot_i8_ptr(int8_ptr(lhs), int8_ptr(rhs), dimension);
    }

    template <typename Op>
    T dot_vector_index(const Op& vec, unsigned index) const {
        if constexpr (requires(const Op& op) { op.float_data(); }) {
            if (storage == Storage::Float32) {
                const T* src = vec.float_data();
                if (src != nullptr) {
                    Eigen::Map<const Eigen::VectorXf> src_map(src, dimension);
                    return static_cast<T>(src_map.dot(vectors.col(index)));
                }
            }
        }
        if constexpr (requires(const Op& op) { op.int8_data(); }) {
            if (storage == Storage::Int8) {
                const std::int8_t* src = vec.int8_data();
                if (src != nullptr) {
                    return dot_i8_ptr(src, int8_ptr(index), dimension);
                }
            }
        }
        if constexpr (DotProductWithVectorType<Op, T>) {
            if (storage == Storage::Float32) {
                return static_cast<T>(vec.dot(vectors.col(index)));
            }
        }
        T sum = 0;
        for (unsigned d = 0; d < dimension; ++d) {
            sum += static_cast<T>(vec[d]) * value_at(index, d);
        }
        return sum;
    }

    template <typename Op>
    T squared_norm_of(const Op& vec) const {
        if constexpr (requires { vec.squaredNorm(); }) {
            return static_cast<T>(vec.squaredNorm());
        } else {
            T sum = 0;
            for (unsigned d = 0; d < dimension; ++d) {
                const T v = static_cast<T>(vec[d]);
                sum += v * v;
            }
            return sum;
        }
    }

    Eigen::MatrixXf vectors;
    std::vector<std::int8_t> i8_vectors;
    std::vector<T> sqrs;
    unsigned dimension = 0;
    unsigned vector_count = 0;
    Storage storage = Storage::Float32;
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
        storage = Storage::Float32;
        dimension = dim;
        vector_count = n;
        i8_vectors.clear();
        vectors.resize(dimension, n);
        IO::read_fvecs(fin, dimension, vectors.data());
    } else if (ext == ".u8bin") {
        auto [n, dim] = IO::get_u8bin_size(fin);
        storage = Storage::Float32;
        dimension = dim;
        vector_count = n;
        i8_vectors.clear();
        vectors.resize(dimension, n);
        IO::read_u8bin(fin, dimension, vectors.data());
    } else if (ext == ".i8bin") {
        auto [n, dim] = IO::get_i8bin_size(fin);
        storage = Storage::Int8;
        dimension = dim;
        vector_count = n;
        vectors.resize(0, 0);
        i8_vectors.resize(static_cast<size_t>(n) * dim);
        IO::read_i8bin_raw(fin, dimension, i8_vectors.data());
    } else {
        spdlog::error("Unsupported file extension: {}", ext);
        throw std::runtime_error("Unsupported file extension");
    }
    init_sqrs();
}

template <typename T>
void VectorList<T>::reorder(const std::vector<unsigned>& order) {
    assert(order.size() == size());
    auto tmp_s = sqrs;
    if (storage == Storage::Int8) {
        auto tmp = i8_vectors;
        for (unsigned i = 0; i < order.size(); i++) {
            const auto dst = static_cast<size_t>(i) * dimension;
            const auto src = static_cast<size_t>(order[i]) * dimension;
            std::copy_n(tmp.begin() + static_cast<std::ptrdiff_t>(src),
                        dimension,
                        i8_vectors.begin() + static_cast<std::ptrdiff_t>(dst));
            sqrs[i] = tmp_s[order[i]];
        }
    } else {
        Eigen::MatrixXf tmp = vectors;
        for (unsigned i = 0; i < order.size(); i++) {
            vectors.col(i) = tmp.col(order[i]);
            sqrs[i] = tmp_s[order[i]];
        }
    }
    spdlog::info("reorder done");
}

template <typename T>
void VectorList<T>::init_sqrs() {
    spdlog::info("Initializing squared norms...");
    sqrs.resize(size());
    for (unsigned i = 0; i < size(); ++i) {
        if (storage == Storage::Int8) {
            T sum = 0;
            const auto off = static_cast<size_t>(i) * dimension;
            for (unsigned d = 0; d < dimension; ++d) {
                const T v = static_cast<T>(i8_vectors[off + d]);
                sum += v * v;
            }
            sqrs[i] = sum;
        } else {
            sqrs[i] = vectors.col(i).squaredNorm();
        }
    }
    spdlog::info("Squared norms initialized.");
}

template <typename T>
std::vector<float> VectorList<T>::to_float_data() const {
    std::vector<float> out(static_cast<size_t>(size()) * dimension);
    if (storage == Storage::Float32) {
        const auto* src = vectors.data();
        std::copy(src, src + out.size(), out.begin());
        return out;
    }
    for (unsigned i = 0; i < size(); ++i) {
        const auto off = static_cast<size_t>(i) * dimension;
        for (unsigned d = 0; d < dimension; ++d) {
            out[off + d] = static_cast<float>(i8_vectors[off + d]);
        }
    }
    return out;
}

template <typename T>
template <typename Op>
T VectorList<T>::dist2(unsigned source, const Op& goal) const {
    static_assert(std::is_convertible_v<Op, unsigned> ||
                      DotProductWithVectorType<Op, T> ||
                      VectorIndexable<Op>,
                  "Op must be convertible to unsigned or a vector-like type");

    if constexpr (std::is_convertible_v<Op, unsigned>) {
        const unsigned dst = static_cast<unsigned>(goal);
        return sqrs[source] + sqrs[dst] -
               2 * dot_index_index(source, static_cast<unsigned>(goal));
    } else {
        if constexpr (requires(const Op& op) { op.int8_data(); }) {
            if (storage == Storage::Int8) {
                const std::int8_t* dst = goal.int8_data();
                if (dst != nullptr) {
                    return l2sq_i8_ptr(int8_ptr(source), dst, dimension);
                }
            }
        }
        T sum = 0;
        for (unsigned d = 0; d < dimension; ++d) {
            const T diff = value_at(source, d) - static_cast<T>(goal[d]);
            sum += diff * diff;
        }
        return sum;
    }
}

template <typename T>
template <typename Op>
T VectorList<T>::dist(unsigned source, const Op& goal) const {
    static_assert(std::is_convertible_v<Op, unsigned> ||
                      DotProductWithVectorType<Op, T> ||
                      VectorIndexable<Op>,
                  "Op must be convertible to unsigned or a vector-like type");

    return dist2(source, goal);
}

template <typename T>
template <typename Op>
void VectorList<T>::dist_all_into(
    const Op& source,
    std::vector<std::pair<T, unsigned>>& p) const {
    constexpr bool O_id = std::is_convertible_v<Op, unsigned>;
    static_assert(O_id || DotProductWithVectorType<Op, T> ||
                      VectorIndexable<Op>,
                  "Op must be convertible to unsigned or a vector-like type");
    if constexpr (O_id) {
        const unsigned src = static_cast<unsigned>(source);
        if (storage == Storage::Int8) {
            const std::int8_t* src_ptr = int8_ptr(src);
            auto* items = p.data();
            const size_t count = p.size();
            constexpr size_t prefetch_distance = 8;
            for (size_t i = 0; i < count; ++i) {
                if (i + prefetch_distance < count) {
                    __builtin_prefetch(
                        int8_ptr(items[i + prefetch_distance].second), 0, 1);
                }
                const unsigned idx = items[i].second;
                items[i].first =
                    sqrs[idx] - 2 * dot_i8_ptr(int8_ptr(idx), src_ptr, dimension);
            }
            return;
        }
        for (auto& [dis, idx] : p) {
            dis = sqrs[idx] - 2 * dot_index_index(idx, src);
        }
    } else {
        if constexpr (requires(const Op& op) { op.float_data(); }) {
            if (storage == Storage::Float32) {
                const T* src = source.float_data();
                if (src != nullptr) {
                    Eigen::Map<const Eigen::VectorXf> src_map(src, dimension);
                    for (auto& [dis, idx] : p) {
                        dis = sqrs[idx] -
                              2 * static_cast<T>(
                                      src_map.dot(vectors.col(idx)));
                    }
                    return;
                }
            }
        }
        if constexpr (requires(const Op& op) { op.int8_data(); }) {
            if (storage == Storage::Int8) {
                const std::int8_t* src = source.int8_data();
                if (src != nullptr) {
                    auto* items = p.data();
                    const size_t count = p.size();
                    constexpr size_t prefetch_distance = 8;
                    for (size_t i = 0; i < count; ++i) {
                        if (i + prefetch_distance < count) {
                            __builtin_prefetch(
                                int8_ptr(items[i + prefetch_distance].second), 0, 1);
                        }
                        const unsigned idx = items[i].second;
                        items[i].first =
                            sqrs[idx] - 2 * dot_i8_ptr(src, int8_ptr(idx), dimension);
                    }
                    return;
                }
            }
        }
        for (auto& [dis, idx] : p) {
            dis = sqrs[idx] - 2 * dot_vector_index(source, idx);
        }
    }
}

template <typename T>
template <typename Op, std::ranges::range R_op>
std::vector<T> VectorList<T>::dist_all(const Op& source,
                                       const R_op& goal) const {
    using Item = std::remove_cvref_t<decltype(*std::ranges::begin(goal))>;
    constexpr bool O_id = std::is_convertible_v<Op, unsigned>;
    constexpr bool I_id = std::is_convertible_v<Item, unsigned>;
    constexpr bool O_vec =
        DotProductWithVectorType<Op, T> || VectorIndexable<Op>;
    constexpr bool I_vec =
        DotProductWithVectorType<Item, T> || VectorIndexable<Item>;
    static_assert(O_id || O_vec,
                  "Op must be convertible to unsigned or a vector-like type");
    static_assert(I_id || I_vec,
                  "Item must be convertible to unsigned or a vector-like type");
    static_assert(O_id || I_id,
                  "At least one of Op or Item must be an index type");

    std::vector<T> result;
    if constexpr (std::ranges::sized_range<R_op>) {
        result.reserve(std::ranges::size(goal));
    }
    if constexpr (O_id || I_vec) {
        for (const auto& g : goal) {
            result.push_back(dist(source, g));
        }
    } else {
        const T source_sqr = squared_norm_of(source);
        for (const auto& g : goal) {
            const unsigned idx = static_cast<unsigned>(g);
            result.push_back(source_sqr + sqrs[idx] -
                             2 * dot_vector_index(source, idx));
        }
    }
    return result;
}

template <typename T>
Vector::VectorType<T> VectorList<T>::mean() const {
    Vector::VectorType<T> result(dimension);
    result.setZero();
    if (storage == Storage::Int8) {
        for (unsigned i = 0; i < size(); i++) {
            const auto off = static_cast<size_t>(i) * dimension;
            for (unsigned d = 0; d < dimension; ++d) {
                result[d] += static_cast<T>(i8_vectors[off + d]);
            }
        }
    } else {
        for (unsigned i = 0; i < size(); i++) {
            result += vectors.col(i);
        }
    }
    result /= size();
    return result;
}

template <typename T>
template <typename Op>
T VectorList<T>::sqr_sub_2dot(unsigned idx, const Op& vec) const {
    static_assert(DotProductWithVectorType<Op, T> || VectorIndexable<Op>,
                  "Op must support dot product with VectorType<T>");
    return sqrs[idx] - 2 * dot_vector_index(vec, idx);
}

}  // namespace Vector

}  // namespace TDFANN
