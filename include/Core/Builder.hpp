#pragma once
#include <PCH.hpp>

#include <faiss/IndexNNDescent.h>
#include <Core/Searcher.hpp>
#include <Graph/GraphIndex.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN {

template <typename T>
class Builder {
   public:
    Builder(const Vector::VectorList<T>& data) : vector_list(data) {};
    Graph::GraphIndex<std::monostate> nn_descent(size_t k) const;
    Graph::TDGraphIndexBase build(size_t k) const;

   private:
    std::vector<std::pair<T, size_t>> prune(
        const std::vector<std::pair<T, size_t>>&) const;
    const Vector::VectorList<T>& vector_list;  // 向量列表
};

//>===========================================================<

// Implement of builder functions

template <typename T>
Graph::GraphIndex<std::monostate> Builder<T>::nn_descent(size_t k) const {
    std::ifstream fin("knng.in");
    if (fin.good()) {
        Graph::GraphIndex<std::monostate> graph(0);
        graph.load(fin);
        return graph;
    }
    fin.close();
    Graph::GraphIndex<std::monostate> graph(vector_list.size());
    std::vector<T> data(vector_list.size() * vector_list.dim());
    for (size_t i = 0; i < vector_list.size(); i++) {
        std::ranges::copy(vector_list[i], data.begin() + i * vector_list.dim());
    }
    spdlog::info("start KNN train with size {}, dim {}", vector_list.size(),
                 vector_list.dim());
    faiss::IndexNNDescentFlat index(vector_list.dim(), k);
    index.verbose = true;
    index.add(vector_list.size(), data.data());
    spdlog::info("KNN final trained = {}, graph size = {}", index.is_trained,
                 index.nndescent.graph.size());
    for (size_t i = 0; i < vector_list.size(); i++) {
        graph.add_neighbours(
            i, index.nndescent.final_graph | std::views::drop(i * k) |
                   std::views::take(k) | std::views::transform([&](size_t id) {
                       return Graph::GraphIndex<std::monostate>::Node{id, {}};
                   }));
    }
    std::ofstream fout("knng.in");
    graph.save(fout);
    return graph;
}

template <typename T>
Graph::TDGraphIndexBase Builder<T>::build(size_t k) const {
    spdlog::info("Building TDF Graph Index, index size {}...",
                 vector_list.size());
    Graph::GraphIndex<size_t> left(vector_list.size()),
        right(vector_list.size());
    auto knng = nn_descent(k);
    auto build = [&](Graph::GraphIndex<size_t>& g, auto&& id, auto&& cmp) {
        spdlog::info("start building one side");

        Searcher searcher(vector_list, knng);
        std::vector<std::vector<std::pair<T, size_t>>> candidates(
            vector_list.size());
        size_t lst = -1;
        const size_t block = std::max(size_t(1), vector_list.size() / 100);
        Timer::start("build");
        for (auto i : id) {
            if (lst != size_t(-1)) {
                candidates[i] =
                    searcher.beam_search(i, vector_list.size(), i - 1, 200);
                for (auto [dis, id] : candidates[i]) {
                    candidates[id].push_back({dis, i});
                }
                if (i % block == 0) {
                    Timer::end("build");
                    spdlog::info("building rate {:.2f}%",
                                 i * 100.0 / vector_list.size());
                    spdlog::info("searching time rate {:.2f}%",
                                 Timer::read("beam_search") * 100.0f /
                                     Timer::read("build"));
                    Timer::start("build");
                }
            }
            lst = i;
        }
        size_t total_size = 0;
        for (size_t i = 0; i < vector_list.size(); i++) {
            std::ranges::sort(candidates[i], cmp);
            total_size += candidates[i].size();
            candidates[i] = prune(candidates[i]);
            auto r = candidates[i] | std::views::transform([](const auto& x) {
                         return Graph::GraphIndex<size_t>::Node{x.second,
                                                                size_t(-1)};
                     });
            g.add_neighbours(i, r);
            if (i % block == 0) {
                spdlog::info("pruning rate {:.2f}%",
                             i * 100.0 / vector_list.size());
            }
        }
        spdlog::debug("candidate total size {}", total_size);
    };
    build(left, std::views::iota(0u, vector_list.size()),
          [](auto a, auto b) { return a.second > b.second; });
    build(right, std::views::iota(0u, vector_list.size()) | std::views::reverse,
          [](auto a, auto b) { return a.second < b.second; });
    spdlog::info("Build finished.");
    return Graph::TDGraphIndexBase(std::move(left), std::move(right));
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