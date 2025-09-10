#pragma once
#include <PCH.hpp>

#include <Graph/Concepts.hpp>
#include <Utils/ExFunc.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>
#include <Vector/VectorType.hpp>
#include "Utils/Recorder.hpp"

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

    struct Node {
        T dis;
        bool visited;
        size_t id;
        bool operator<(const Node& other) {
            return std::pair{dis, id} < std::pair{other.dis, other.id};
        }
    };

    std::vector<Node> candidates;
    candidates.reserve(beam_size + 1);
    candidates.push_back({dataset.dist(start_node, goal), false, start_node});

    for (size_t uid = 0; uid < beam_size; uid++) {
        if (candidates[uid].visited) {
            continue;  // 已处理过，跳过
        }
        size_t current_node = candidates[uid].id;
        candidates[uid].visited = true;  // 标记为已处理

        // 获取当前节点的邻居
        for (const auto& neighbour : graph.get_neighbours_id(current_node)) {
            T dist = dataset.dist(neighbour, goal);
            auto now = Node{dist, false, neighbour};
            auto it = std::lower_bound(candidates.begin(), candidates.end(), now);
            // if (it != candidates.end() && it->id == neighbour) {
            //     Recorder<size_t>::add("repeat_node", 1);
            // } else {
            //     Recorder<size_t>::add("new_node", 1);
            // }
            if (it != candidates.end() && it->id != neighbour) {
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

    return Utils::to_vector(candidates |
                            std::views::take(std::min(k, candidates.size())) |
                            std::views::transform([](const auto& c) {
                                return std::pair{c.dis, c.id};  // 提取节点索引
                            }));
}

}  // namespace TDFANN