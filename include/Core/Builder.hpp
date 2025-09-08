#pragma once
#include <PCH.hpp>

#include <faiss/IndexNNDescent.h>
#include <Core/Searcher.hpp>
#include <Graph/GraphIndex.hpp>
#include <Utils/Recorder.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN {

template <typename T>
class Builder {
   public:
    Builder(const Vector::VectorList<T>& data) : vector_list(data) {};
    Graph::GraphIndex<std::monostate> nn_descent(size_t k, bool verbose = false) const;
    Graph::TDGraphIndexBase build(Graph::GraphLike auto&& knng, size_t d) const;

   private:
    std::vector<std::pair<T, size_t>> prune(
        const std::vector<std::pair<T, size_t>>&) const;
    const Vector::VectorList<T>& vector_list;  // 向量列表
};

//>===========================================================<

// Implement of builder functions

template <typename T>
Graph::GraphIndex<std::monostate> Builder<T>::nn_descent(size_t k, bool verbose) const {
    Graph::GraphIndex<std::monostate> graph(vector_list.size());
    std::vector<T> data(vector_list.size() * vector_list.dim());
    for (size_t i = 0; i < vector_list.size(); i++) {
        std::ranges::copy(vector_list[i], data.begin() + i * vector_list.dim());
    }
    spdlog::info("start KNN train with size {}, dim {}", vector_list.size(),
                 vector_list.dim());
    faiss::IndexNNDescentFlat index(vector_list.dim(), k);
    index.verbose = verbose;
    index.add(vector_list.size(), data.data());
    for (size_t i = 0; i < vector_list.size(); i++) {
        graph.add_neighbours(
            i, index.nndescent.final_graph | std::views::drop(i * k) |
                   std::views::take(k) |
                   std::views::transform([&](size_t id) { return id; }));
    }
    spdlog::info("KNN train finished.");
    return graph;
}

template <typename T>
Graph::TDGraphIndexBase Builder<T>::build(Graph::GraphLike auto&& knng,
                                          size_t d) const {
    spdlog::info("Building TDF Graph Index, index size {}...",
                 vector_list.size());
    Graph::TDGraphIndexBase g(vector_list.size());
    size_t n = vector_list.size(), total_degree = 0;
    const size_t step = (vector_list.size() + 99) / 100;
    for (size_t i = 0; i < vector_list.size(); i++) {
        bool output_tag = (i + 1) % step == 0 || i == vector_list.size();

        std::vector<std::pair<T, size_t>> c_left, c_right;
        for (const auto& neighbour : knng.get_neighbours_id(i)) {
            if (neighbour < i) {
                c_left.push_back({vector_list.dist(i, neighbour), neighbour});
            } else {
                c_right.push_back({vector_list.dist(i, neighbour), neighbour});
            }
        }
        for (size_t j = i - std::min(i, d); j < i; j++) {
            c_left.push_back({vector_list.dist(i, j), j});
        }
        for (size_t j = i + 1; j < std::min(i + d, n); j++) {
            c_right.push_back({vector_list.dist(i, j), j});
        }

        std::ranges::sort(
            c_left, [&](auto&& x, auto&& y) { return x.second > y.second; });
        std::ranges::sort(
            c_right, [&](auto&& x, auto&& y) { return x.second < y.second; });

        c_left.erase(std::begin(std::ranges::unique(c_left)), c_left.end());
        c_right.erase(std::begin(std::ranges::unique(c_right)), c_right.end());

        size_t candidate_size = 0;
        if (output_tag) {
            Timer::start("prune");
            candidate_size = c_left.size() + c_right.size();
        }

        c_left = prune(c_left);
        c_right = prune(c_right);

        total_degree += c_left.size() + c_right.size();

        if (output_tag) {
            auto t = Timer::end("prune");
            spdlog::info(
                "Build progress: {}/{} ({:.2f}%), prune time cost {}, "
                "candidate size {} -> {}",
                i + 1, vector_list.size(), (i + 1) * 100.0 / vector_list.size(),
                t, candidate_size, c_left.size() + c_right.size());
        }

        g.add_neighbours(
            i, std::array{c_left, c_right} | std::views::join |
                   std::views::transform([&](auto& x) { return x.second; }));
    }
    spdlog::info("average degree {:.2f}",
                 total_degree * 1.0 / vector_list.size());
    spdlog::info("Build finished.");
    return g;
}

template <typename T>
std::vector<std::pair<T, size_t>> Builder<T>::prune(
    const std::vector<std::pair<T, size_t>>& candidates) const {
    std::vector<std::pair<T, size_t>> result;
    auto check_valid = [&](const std::pair<T, size_t>& now) {
        auto [d_now, i_now] = now;
        for (auto [d_lst, i_lst] : result) {
            if (d_now > d_lst && d_now > vector_list.dist(i_now, i_lst)) {
                return false;
            }
        }
        return true;
    };
    for (size_t i = 0; i < candidates.size(); i++) {
        if (check_valid(candidates[i])) {
            result.push_back(candidates[i]);
        }
    }
    return result;
}

}  // namespace TDFANN