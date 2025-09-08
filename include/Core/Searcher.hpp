#pragma once
#include <PCH.hpp>

#include <Graph/Concepts.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>
#include <Vector/VectorType.hpp>

namespace TDFANN {

template <typename T, Graph::GraphLike G>
class Searcher {
   public:
    Searcher(const Vector::VectorList<T>& data, const G& graph);

    /**
     * @brief Perform a linear search for the top k results.
     *
     * @return top k results as {distance, index}
     */
    template <typename GoalId>
    std::vector<std::pair<T, size_t>> linear_search(const GoalId& goal,
                                                    size_t k);

    /**
     * @brief return top k nearest neighbours
     *
     * @param goal node id
     * @param k number of neighbours
     * @param start_node start routing node on the graph
     * @param beam_size searching beam size
     * @return return vector{{distance, index}}
     */
    template <typename GoalId>
    std::vector<std::pair<T, size_t>> beam_search(const GoalId& goal,
                                                  size_t k,
                                                  size_t start_node,
                                                  size_t beam_size);

   private:
    const Vector::VectorList<T>& dataset;
    const G& graph;
};

// implementation of Searcher methods

template <typename T, Graph::GraphLike G>
Searcher<T, G>::Searcher(const Vector::VectorList<T>& data, const G& graph)
    : dataset(data), graph(graph) {}

template <typename T, Graph::GraphLike G>
template <typename GoalId>
std::vector<std::pair<T, size_t>> Searcher<T, G>::linear_search(
    const GoalId& goal,
    size_t k) {
    static_assert(std::is_convertible_v<GoalId, size_t> ||
                      std::is_convertible_v<GoalId, Vector::VectorType<T>>,
                  "GoalId must be convertible to size_t or a vector-like type");

    std::priority_queue<std::pair<T, size_t>> heap;
    for (size_t i = 0; i < dataset.size(); i++) {
        heap.push({dataset.dist(i, goal), i});
        if (heap.size() > k) {
            heap.pop();
        }
    }
    std::vector<std::pair<T, size_t>> result;
    result.reserve(k);
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    return result;
}

template <typename T, Graph::GraphLike G>
template <typename GoalId>
std::vector<std::pair<T, size_t>> Searcher<T, G>::beam_search(
    const GoalId& goal,
    size_t k,
    size_t start_node,
    size_t beam_size) {
    static_assert(std::is_convertible_v<GoalId, size_t> ||
                      std::is_convertible_v<GoalId, Vector::VectorType<T>>,
                  "GoalId must be convertible to size_t or a vector-like type");

    // spdlog::debug("start beam search");

    std::vector<std::tuple<T, size_t, bool>> candidates;
    candidates.push_back({dataset.dist(start_node, goal), start_node, false});

    // std::set<size_t> visited;
    // visited.insert(start_node);
    for (size_t uid = 0; uid < beam_size; uid++) {
        if (std::get<2>(candidates[uid])) {
            continue;  // 已处理过，跳过
        }
        size_t current_node = std::get<1>(candidates[uid]);
        std::get<2>(candidates[uid]) = true;  // 标记为已处理

        // 获取当前节点的邻居
        auto neighbours = graph.get_neighbours_id(current_node);
        for (const auto& neighbour : neighbours) {
            // if (visited.count(neighbour)) {
            //     continue;  // 已访问过，跳过
            // }
            // visited.insert(neighbour);
            T dist = dataset.dist(neighbour, goal);
            auto now = std::tuple{dist, neighbour, false};
            auto it = std::ranges::partition_point(
                candidates, [&](const auto& x) { return x < now; });
            if (it != candidates.end() && std::get<1>(*it) != neighbour) {
                if (candidates.size() == beam_size) {
                    candidates.pop_back();
                }
                uid = std::min(uid, it - candidates.begin() - size_t(1));
                candidates.insert(it, now);
            } else if (candidates.size() < beam_size) {
                candidates.push_back(now);
            }
        }
    }

    auto r =
        candidates | std::views::take(k) |
        std::views::transform([](const auto& c) {
            return std::pair{std::get<0>(c), std::get<1>(c)};  // 提取节点索引
        });

    // spdlog::debug("beam search ends");

    // spdlog::debug("beam search total candidates {}, time cost {}",
    // total_candidates, Timer::read("beam_search"));
    return std::vector(r.begin(), r.end());
}

}  // namespace TDFANN