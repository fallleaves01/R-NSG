#pragma once
#include <Graph/Concepts.hpp>
#include <PCH.hpp>
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
    for (int i = 0; i < dataset.size(); i++) {
        heap.push({dataset.dist(i, goal), i});
        if (heap.size() > k) {
            heap.pop();
        }
    }
    std::vector<size_t> result;
    result.reserve(k);
    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }
    return result;
}

template <typename T, Graph::GraphLike G>
template <typename GoalId>
std::vector<std::pair<T, size_t>> Searcher<T, G>::beam_search(const GoalId& goal,
                                                size_t k,
                                                size_t start_node,
                                                size_t beam_size) {
    static_assert(std::is_convertible_v<GoalId, size_t> ||
                      std::is_convertible_v<GoalId, Vector::VectorType<T>>,
                  "GoalId must be convertible to size_t or a vector-like type");

    std::vector<std::tuple<T, size_t, bool>> candidates;
    candidates.push_back({dataset.dist(start_node, goal), start_node, false});

    auto check_unused = [](const auto& c) {
        return !std::get<2>(c);  // 检查是否未处理
    };

    for (auto it = std::ranges::find_if(candidates, check_unused);
         it != candidates.end();
         it = std::ranges::find_if(it, candidates.end(), check_unused)) {
        size_t current_node = std::get<1>(*it);
        std::get<2>(*it) = true;  // 标记为已处理

        // 获取当前节点的邻居
        auto neighbours = graph.get_neighbours_id(current_node);
        for (const auto& neighbour : neighbours) {
            T dist = dataset.dist(neighbour, goal);
            candidates.push_back({dist, neighbour, false});
        }

        // 保持候选项数量不超过 beam_size
        if (candidates.size() > beam_size) {
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) {
                          return std::get<0>(a) < std::get<0>(b);
                      });
            candidates.resize(beam_size);
        }
    }

    auto r = candidates | std::views::take(k) |
             std::views::transform([](const auto& c) {
                 return std::pair{std::get<0>(c), std::get<1>(c)};  // 提取节点索引
             });
    return std::vector(r.begin(), r.end());
}

}  // namespace TDFANN