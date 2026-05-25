#pragma once

#include <PCH.hpp>

#include <parallel_hashmap/phmap.h>
#include <Core/Concepts.hpp>
#include <Graph/Concepts.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN::RNSG {

template <typename T>
struct BeamScratch {
    phmap::flat_hash_map<unsigned, T> vis_dis;
    std::vector<std::pair<T, unsigned>> candidates;
    std::vector<std::pair<T, unsigned>> neighbours;

    size_t candidate_reserved = 0;
    size_t neighbour_reserved = 0;
    size_t vis_reserved = 0;

    BeamScratch() = default;

    BeamScratch(size_t dataset_size, unsigned beam_size, unsigned trunc_size) {
        ensure(dataset_size, beam_size, trunc_size);
    }

    void ensure(size_t dataset_size, unsigned beam_size, unsigned trunc_size) {
        (void)dataset_size;
        const size_t cand_need = static_cast<size_t>(beam_size) + 8;
        if (cand_need > candidate_reserved) {
            candidates.reserve(cand_need);
            candidate_reserved = cand_need;
        }

        const size_t neigh_need =
            std::max<size_t>(static_cast<size_t>(trunc_size) + 8,
                             static_cast<size_t>(beam_size) * 2 + 8);
        if (neigh_need > neighbour_reserved) {
            neighbours.reserve(neigh_need);
            neighbour_reserved = neigh_need;
        }

        const size_t vis_need =
            std::max<size_t>(1024, static_cast<size_t>(beam_size) *
                                       (static_cast<size_t>(trunc_size) + 2));
        if (vis_need > vis_reserved) {
            vis_dis.reserve(vis_need);
            vis_reserved = vis_need;
        }
    }

    void next_query() {
        vis_dis.clear();
        candidates.clear();
        neighbours.clear();
    }
};

template <typename T, Graph::GraphLike G>
class Searcher {
   public:
    Searcher(const Vector::VectorList<T>& data, const G& graph)
        : dataset(data), graph(graph) {}

    template <typename GoalId>
    std::vector<std::pair<T, unsigned>> linear_search(const GoalId& goal,
                                                      unsigned k) {
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

    template <typename GoalId, IndexOrList StartNode>
    std::vector<std::pair<T, unsigned>> beam_search(
        const GoalId& goal,
        unsigned k,
        StartNode start_node,
        unsigned beam_size,
        unsigned trunc_size,
        BeamScratch<T>& scratch,
        std::vector<std::pair<T, unsigned>>* candidates_ptr = nullptr) {
        static_assert(
            IndexOrVector<GoalId, T>,
            "GoalId must be convertible to unsigned or a vector-like type");

        if (beam_size == 0) {
            return {};
        }

        const unsigned offset = dataset.size();
        scratch.ensure(dataset.size(), beam_size, trunc_size);
        scratch.next_query();

        auto& vis_dis = scratch.vis_dis;
        auto& candidates = scratch.candidates;
        auto& neighbours = scratch.neighbours;

        if constexpr (std::convertible_to<StartNode, unsigned>) {
            candidates.push_back({T(0), static_cast<unsigned>(start_node)});
        } else {
            for (auto id : start_node) {
                candidates.push_back({T(0), static_cast<unsigned>(id)});
            }
        }

        if (candidates.empty()) {
            return {};
        }

        dataset.dist_all_into(goal, candidates);
        std::ranges::sort(candidates);
        for (auto& [dis, id] : candidates) {
            vis_dis[id] = dis;
            id += offset;
        }

        if (candidates.size() < beam_size) {
            candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});
        } else if (candidates.size() > beam_size) {
            candidates.resize(beam_size);
        }

        for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
            if (candidates[uid].second < offset) {
                continue;
            }
            candidates[uid].second -= offset;
            const unsigned current_node = candidates[uid].second;

            if (trunc_size == 0) {
                continue;
            }

            neighbours.clear();
            for (const auto& x : graph.get_neighbours(current_node)) {
                if (vis_dis.contains(x.to)) {
                    continue;
                }
                neighbours.push_back({T(0), x.to});
                if (neighbours.size() >= trunc_size) {
                    break;
                }
            }

            if (neighbours.empty()) {
                continue;
            }

            dataset.dist_all_into(goal, neighbours);
            for (const auto& [dist, nto] : neighbours) {
                if (dist < candidates.back().first) {
                    candidates.pop_back();
                    auto it = std::partition_point(
                        candidates.begin(), candidates.end(),
                        [&](const auto& a) { return a.first < dist; });
                    const int pos = static_cast<int>(it - candidates.begin());
                    uid = std::min(uid, pos - 1);
                    candidates.insert(it, {dist, nto + offset});
                    vis_dis.insert({nto, dist});
                }
            }
        }

        if (candidates_ptr != nullptr) {
            auto& c = *candidates_ptr;
            c.reserve(c.size() + vis_dis.size());
            for (auto [i, d] : vis_dis) {
                c.push_back({d, i});
            }
        }

        const size_t out_size = std::min<size_t>(k, candidates.size());
        std::vector<std::pair<T, unsigned>> out;
        out.reserve(out_size);
        for (size_t i = 0; i < out_size; ++i) {
            auto [dist, id] = candidates[i];
            if (id >= offset) {
                id -= offset;
            }
            out.push_back({dist, id});
        }
        return out;
    }

   private:
    const Vector::VectorList<T>& dataset;
    const G& graph;
};

}  // namespace TDFANN::RNSG
