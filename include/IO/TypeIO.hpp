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
        std::cerr << "Type T is not serializable" << std::endl;
        exit(-1);
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
        std::cerr << "Type T is not deserializable" << std::endl;
        exit(-1);
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
    fin.read(reinterpret_cast<char*>(&dimension), sizeof(unsigned));
    fin.seekg(0, std::ios::end);
    const auto file_size = static_cast<size_t>(fin.tellg());
    unsigned n = file_size / (dimension + 1) / sizeof(float);
    fin.seekg(0, std::ios::beg);
    spdlog::info("Vector dimension: {}, size: {}", dimension, n);
    return {n, dimension};
}

inline std::pair<unsigned, unsigned> get_u8bin_size(std::ifstream& fin) {
    unsigned dimension = 0, n = 0;
    fin.read(reinterpret_cast<char*>(&n), sizeof(unsigned));
    fin.read(reinterpret_cast<char*>(&dimension), sizeof(unsigned));
    fin.seekg(0, std::ios::beg);
    spdlog::info("Vector dimension: {}, size: {}", dimension, n);
    return {n, dimension};
}

inline std::pair<unsigned, unsigned> get_i8bin_size(std::ifstream& fin) {
    std::int32_t n = 0, dimension = 0;
    fin.read(reinterpret_cast<char*>(&n), sizeof(std::int32_t));
    fin.read(reinterpret_cast<char*>(&dimension), sizeof(std::int32_t));
    fin.seekg(0, std::ios::beg);
    if (n <= 0 || dimension <= 0) {
        spdlog::error("Invalid i8bin header: n={}, dim={}", n, dimension);
        throw std::runtime_error("Invalid i8bin header");
    }
    spdlog::info("Vector dimension: {}, size: {}", dimension, n);
    return {static_cast<unsigned>(n), static_cast<unsigned>(dimension)};
}

template <typename T>
void read_fvecs(std::ifstream& fin, unsigned dimension, T* data) {
    size_t tmp = 0, idx = 0;
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

template <typename T>
void read_u8bin(std::ifstream& fin, unsigned dimension, T* data) {
    unsigned n = 0, tmp = 0;
    fin.read(reinterpret_cast<char*>(&n), sizeof(unsigned));
    fin.read(reinterpret_cast<char*>(&tmp), sizeof(unsigned));
    spdlog::info("Reading {} vectors of dimension {}", n, tmp);
    if (tmp != dimension) {
        spdlog::error("Inconsistent vector dimensions");
        throw std::runtime_error("Inconsistent vector dimensions in file");
    }
    auto buffer = std::make_unique<std::uint8_t[]>(dimension);
    for (size_t i = 0; i < n; i++) {
        if (!fin.read(reinterpret_cast<char*>(buffer.get()),
                      dimension * sizeof(std::uint8_t))) {
            spdlog::error("Failed to read vector data from file");
            throw std::runtime_error("Failed to read vector data from file");
        }
        for (size_t j = 0; j < dimension; j++) {
            data[i * dimension + j] = static_cast<T>(buffer[j]);
        }
    }
    spdlog::info("Finished reading u8bin file");
}

inline void read_i8bin_raw(std::ifstream& fin, unsigned dimension,
                           std::int8_t* data) {
    std::int32_t n = 0, tmp = 0;
    fin.read(reinterpret_cast<char*>(&n), sizeof(std::int32_t));
    fin.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
    spdlog::info("Reading {} signed-int8 vectors of dimension {}", n, tmp);
    if (n <= 0 || tmp <= 0 || static_cast<unsigned>(tmp) != dimension) {
        spdlog::error("Inconsistent i8bin dimensions");
        throw std::runtime_error("Inconsistent vector dimensions in file");
    }
    const auto bytes = static_cast<std::streamsize>(
        static_cast<size_t>(n) * static_cast<size_t>(dimension) *
        sizeof(std::int8_t));
    if (!fin.read(reinterpret_cast<char*>(data), bytes)) {
        spdlog::error("Failed to read vector data from i8bin file");
        throw std::runtime_error("Failed to read vector data from i8bin file");
    }
    spdlog::info("Finished reading i8bin file");
}

struct FloatVectorData {
    std::vector<float> data;
    unsigned n = 0;
    unsigned dimension = 0;
};

inline FloatVectorData load_i8bin_as_float_data(
    const std::string& filename, size_t chunk_vectors = 1'000'000) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        spdlog::error("Failed to open i8bin vector file: {}", filename);
        throw std::runtime_error("Failed to open i8bin vector file");
    }

    auto [n, dimension] = get_i8bin_size(fin);
    FloatVectorData out;
    out.n = n;
    out.dimension = dimension;
    out.data.resize(static_cast<size_t>(n) * dimension);

    std::int32_t header_n = 0, header_dim = 0;
    fin.read(reinterpret_cast<char*>(&header_n), sizeof(std::int32_t));
    fin.read(reinterpret_cast<char*>(&header_dim), sizeof(std::int32_t));
    if (header_n <= 0 || header_dim <= 0 ||
        static_cast<unsigned>(header_n) != n ||
        static_cast<unsigned>(header_dim) != dimension) {
        spdlog::error("Inconsistent i8bin header while loading {}", filename);
        throw std::runtime_error("Inconsistent i8bin header");
    }

    chunk_vectors = std::max<size_t>(1, chunk_vectors);
    std::vector<std::int8_t> buffer(
        std::min<size_t>(chunk_vectors, n) * dimension);
    size_t done = 0;
    const size_t progress_step = std::max<size_t>(1, n / 20);
    size_t next_progress = progress_step;
    spdlog::info(
        "Converting i8bin to transient float buffer for FAISS: n={}, dim={}, "
        "float_bytes={}",
        n, dimension, out.data.size() * sizeof(float));
    while (done < n) {
        const size_t rows = std::min(chunk_vectors, n - done);
        const size_t values = rows * dimension;
        if (buffer.size() < values) {
            buffer.resize(values);
        }
        if (!fin.read(reinterpret_cast<char*>(buffer.data()),
                      static_cast<std::streamsize>(values))) {
            spdlog::error("Failed to read i8bin payload from {}", filename);
            throw std::runtime_error("Failed to read i8bin payload");
        }
        float* dst = out.data.data() + done * dimension;
        for (size_t i = 0; i < values; ++i) {
            dst[i] = static_cast<float>(buffer[i]);
        }
        done += rows;
        if (done >= next_progress || done == n) {
            spdlog::info("i8bin->float progress: {}/{} ({:.2f}%)", done, n,
                         100.0 * static_cast<double>(done) /
                             static_cast<double>(n));
            while (next_progress <= done) {
                next_progress += progress_step;
            }
        }
    }
    return out;
}

template <typename T>
void read_i8bin(std::ifstream& fin, unsigned dimension, T* data) {
    std::int32_t n = 0, tmp = 0;
    fin.read(reinterpret_cast<char*>(&n), sizeof(std::int32_t));
    fin.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
    spdlog::info("Reading {} signed-int8 vectors of dimension {}", n, tmp);
    if (n <= 0 || tmp <= 0 || static_cast<unsigned>(tmp) != dimension) {
        spdlog::error("Inconsistent i8bin dimensions");
        throw std::runtime_error("Inconsistent vector dimensions in file");
    }
    auto buffer = std::make_unique<std::int8_t[]>(dimension);
    for (std::int32_t i = 0; i < n; i++) {
        if (!fin.read(reinterpret_cast<char*>(buffer.get()),
                      dimension * sizeof(std::int8_t))) {
            spdlog::error("Failed to read vector data from file");
            throw std::runtime_error("Failed to read vector data from file");
        }
        for (size_t j = 0; j < dimension; j++) {
            data[static_cast<size_t>(i) * dimension + j] =
                static_cast<T>(buffer[j]);
        }
    }
    spdlog::info("Finished reading i8bin file");
}

template <typename T = unsigned>
inline std::vector<T> load_json_to_vec(const std::string& filename) {
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
    std::vector<T> result;
    result.reserve(j.size());
    for (const auto& item : j) {
        if (!item.is_number_unsigned()) {
            spdlog::error("JSON array contains non-unsigned integer values");
            throw std::runtime_error(
                "JSON array contains non-unsigned integer values");
        }
        result.push_back(item.get<T>());
    }
    return result;
}
}  // namespace IO

}  // namespace TDFANN
