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

    template <typename GoalId>
    std::vector<size_t> linear_search(const GoalId& goal, size_t k);

    template <typename GoalId>
    std::vector<size_t> beam_search(const GoalId& goal,
                                    size_t k,
                                    size_t start_node,
                                    size_t beam_size);

   private:
    const Vector::VectorList<T>& dataset;
    const G& graph;
};

}  // namespace TDFANN

// implementation of Searcher methods
namespace TDFANN {

template <typename T, Graph::GraphLike G>
Searcher<T, G>::Searcher(const Vector::VectorList<T>& data, const G& graph)
    : dataset(data), graph(graph) {}

template <typename T, Graph::GraphLike G>
template <typename GoalId>
std::vector<size_t> Searcher<T, G>::linear_search(const GoalId& goal,
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
std::vector<size_t> Searcher<T, G>::beam_search(const GoalId& goal,
                                                size_t k,
                                                size_t start_node,
                                                size_t beam_size) {
    static_assert(std::is_convertible_v<GoalId, size_t> ||
                      std::is_convertible_v<GoalId, Vector::VectorType<T>>,
                  "GoalId must be convertible to size_t or a vector-like type");

    std::vector<std::tuple<T, size_t, bool>> candidates;
    candidates.push_back({dataset.dist(start_node, goal), start_node, false});
}

}  // namespace TDFANN