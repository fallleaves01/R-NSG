#pragma once
#include <PCH.hpp>

#include <parallel_hashmap/phmap.h>
#include <Core/Concepts.hpp>
#include <Graph/Concepts.hpp>
#include <Utils/ExFunc.hpp>
#include <Utils/Recorder.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

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
    std::vector<std::pair<T, unsigned>> linear_search(const GoalId& goal,
                                                      unsigned k);

    /**
     * @brief return top k nearest neighbours
     *
     * @param goal node id
     * @param k number of neighbours
     * @param start_node start routing node on the graph
     * @param beam_size searching beam size
     * @return return vector{{distance, index}}
     */
    template <typename GoalId, IndexOrList StartNode>
    std::vector<std::pair<T, unsigned>> beam_search(
        const GoalId& goal,
        unsigned k,
        StartNode start_node,
        unsigned beam_size,
        unsigned trunc_size,
        std::vector<std::pair<T, unsigned>>* candidates_ptr = nullptr);

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
std::vector<std::pair<T, unsigned>> Searcher<T, G>::linear_search(
    const GoalId& goal,
    unsigned k) {
    static_assert(
        IndexOrVector<GoalId, T>,
        "GoalId must be convertible to unsigned or a vector-like type");

    std::priority_queue<std::pair<T, unsigned>> heap;
    for (unsigned i = 0; i < dataset.size(); i++) {
        heap.push({dataset.dist(i, goal), i});
        if (heap.size() > k) {
            heap.pop();
        }
    }
    std::vector<std::pair<T, unsigned>> result;
    result.reserve(k);
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    return result;
}

template <typename T, Graph::GraphLike G>
template <typename GoalId, IndexOrList StartNode>
std::vector<std::pair<T, unsigned>> Searcher<T, G>::beam_search(
    const GoalId& goal,
    unsigned k,
    StartNode start_node,
    unsigned beam_size,
    unsigned trunc_size,
    std::vector<std::pair<T, unsigned>>* candidates_ptr) {
    static_assert(
        IndexOrVector<GoalId, T>,
        "GoalId must be convertible to unsigned or a vector-like type");

    phmap::flat_hash_map<unsigned, T> vis_dis;

    unsigned offset = dataset.size();
    std::vector<std::pair<T, unsigned>> candidates(beam_size), neighbours;
    candidates.clear(), candidates.reserve(beam_size);
    neighbours.clear(), neighbours.reserve(beam_size * 2);
    if constexpr (std::convertible_to<StartNode, unsigned>) {
        candidates.push_back({0, unsigned(start_node)});
    } else {
        for (auto it : start_node) {
            candidates.push_back({0, unsigned(it)});
        }
    }
    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    for (auto& [dis, id] : candidates) {
        vis_dis[id] = dis;
        id += offset;
    }
    candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});

    unsigned total = 0;
    for (int uid = 0; uid < (int)beam_size; uid++) {
        if (candidates[uid].second < offset) {
            continue;  // 已处理过，跳过
        }
        candidates[uid].second -= offset;  // 标记为已处理
        unsigned current_node = candidates[uid].second;

        // 获取当前节点的邻居
        neighbours.clear();
        std::ranges::copy(graph.get_neighbours(current_node) |
                              std::views::filter([&](auto&& x) {
                                  return !vis_dis.contains(x.to);
                              }) |
                              std::views::take(trunc_size) |
                              std::views::transform([&](auto&& x) {
                                  return std::pair{T(0), x.to};
                              }),
                          std::back_inserter(neighbours));
        dataset.dist_all_into_trunc(goal, neighbours, candidates.back().first);
        total += neighbours.size();
        for (const auto& [dist, nto] : neighbours) {
            if (dist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < dist; });
                uid = std::min(uid, (int)(it - candidates.begin() - 1));
                candidates.insert(it, {dist, nto + offset});
                vis_dis.insert({nto, dist});
            }
        }
    }

    Recorder<unsigned>::add("total_visited", total);

    if (candidates_ptr != nullptr) {
        auto& c = *candidates_ptr;
        c.reserve(c.size() + vis_dis.size());
        for (auto [i, d] : vis_dis) {
            c.push_back({d, i});
        }
    }

    return Utils::to_vector(candidates | std::views::take(std::min(
                                             k, (unsigned)candidates.size())));
}

}  // namespace TDFANN
