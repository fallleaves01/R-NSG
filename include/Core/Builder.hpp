#pragma once
#include <Core/Searcher.hpp>
#include <Graph/GraphIndex.hpp>
#include <PCH.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN {

template <typename T>
class Builder {
   public:
    Builder(const Vector::VectorList<T>& data) : vector_list(data) {};
    Graph::TDGraphIndexBase build();

   private:
    std::vector<std::pair<T, size_t>> prune(
        const std::vector<std::pair<T, size_t>>&);
    const Vector::VectorList<T>& vector_list;  // 向量列表
};

//>===========================================================<

// Implement of builder functions

template <typename T>
Graph::TDGraphIndexBase Builder<T>::build() {
    spdlog::info("Building TDF Graph Index, index size {}...",
                 vector_list.size());
    Graph::TagGraphIndex left(vector_list.size()), right(vector_list.size());
    auto build = [&](Graph::TagGraphIndex& g, auto&& id, auto&& cmp) {
        Searcher searcher(vector_list, g);
        std::vector<std::vector<std::pair<T, size_t>>> candidates(
            vector_list.size());
        size_t lst = -1;
        for (auto i : id) {
            if (lst != size_t(-1)) {
                candidates[i] =
                    searcher.beam_search(i, vector_list.size(), lst, 50);
                for (auto [dis, id] : candidates[i]) {
                    candidates[id].push_back({dis, i});
                }
            }
            lst = i;
        }
        size_t total_size = 0;
        for (size_t i = 0; i < vector_list.size(); i++) {
            std::ranges::sort(candidates[i], cmp);
            total_size += candidates[i].size();
            candidates[i] = prune(candidates[i]);
            auto r =
                candidates[i] | std::views::transform([](const auto& x) {
                    return Graph::TagGraphIndex::Node{x.second, size_t(-1)};
                });
            g.add_neighbours(i, r);
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
    const std::vector<std::pair<T, size_t>>& candidates) {
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