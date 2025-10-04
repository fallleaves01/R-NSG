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
    template <typename GoalId, IndexOrList StartNode>
    std::vector<std::pair<T, size_t>> beam_search(
        const GoalId& goal,
        size_t k,
        StartNode start_node,
        size_t beam_size,
        std::vector<std::pair<T, size_t>>* candidates_ptr = nullptr);

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
    static_assert(IndexOrVector<GoalId, T>,
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
template <typename GoalId, IndexOrList StartNode>
std::vector<std::pair<T, size_t>> Searcher<T, G>::beam_search(
    const GoalId& goal,
    size_t k,
    StartNode start_node,
    size_t beam_size,
    std::vector<std::pair<T, size_t>>* candidates_ptr) {
    static_assert(IndexOrVector<GoalId, T>,
                  "GoalId must be convertible to size_t or a vector-like type");

    phmap::flat_hash_map<size_t, T> vis_dis;

    struct Node {
        T dis;
        bool visited;
        size_t id;
    };

    std::vector<Node> candidates(beam_size, {T(1e100), true, size_t(-1)});
    if constexpr (std::convertible_to<StartNode, size_t>) {
        candidates[0] = {dataset.dist(start_node, goal), false,
                         size_t(start_node)};
    } else {
        size_t id = 0;
        for (auto it : start_node) {
            if (id >= beam_size) {
                candidates.push_back(
                    {dataset.dist(it, goal), false, size_t(it)});
            } else {
                candidates[id++] = {dataset.dist(it, goal), false, size_t(it)};
            }
        }
        std::ranges::sort(candidates, [](const auto& a, const auto& b) {
            return a.dis < b.dis;
        });
        if (candidates.size() > beam_size) {
            candidates.resize(beam_size);
        }
    }

    size_t total = 0;
    for (size_t uid = 0; uid < beam_size; uid++) {
        if (candidates[uid].visited) [[unlikely]] {
            continue;  // 已处理过，跳过
        }
        size_t current_node = candidates[uid].id;
        candidates[uid].visited = true;  // 标记为已处理

        // 获取当前节点的邻居
        auto neighbours = Utils::to_vector(graph.get_neighbours(current_node) |
                                           std::views::filter([&](auto x) {
                                               return !vis_dis.contains(x.to);
                                           }));
        std::vector<T> dists = dataset.dist_all(
            goal,
            neighbours | std::views::transform(GET(to)));
        size_t dist_id = 0;
        total += dists.size();
        // phmap::flat_hash_set<size_t> prunned;
        for (const auto& neighbour : neighbours) {
            T dist = dists[dist_id++];
            // T dist = dataset.sqr_sub_2dot(neighbour, goal);
            if (dist < candidates.back().dis) [[unlikely]] {
                // if constexpr (IsTDFG<G>) {
                //     prunned.insert(neighbour.to);
                //     if (prunned.contains(neighbour.data.banned_id)) {
                //         --total;
                //         continue;
                //     }
                // }
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.dis < dist; });
                uid = std::min(uid, it - candidates.begin() - size_t(1));
                candidates.insert(it, {dist, false, neighbour.to});
                vis_dis.insert({neighbour.to, dist});
            }
        }
    }

    Recorder<size_t>::add("total_visited", total);

    if (candidates_ptr != nullptr) {
        auto& c = *candidates_ptr;
        c.reserve(c.size() + vis_dis.size());
        for (auto [i, d] : vis_dis) {
            c.push_back({d, i});
        }
    }

    return Utils::to_vector(candidates |
                            std::views::take(std::min(k, candidates.size())) |
                            std::views::transform([](const auto& c) {
                                return std::pair{c.dis, c.id};  // 提取节点索引
                            }));
}

}  // namespace TDFANN
