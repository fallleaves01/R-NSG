#pragma once

#include <PCH.hpp>

#include <Core/Concepts.hpp>
#include <Graph/Concepts.hpp>
#include <Vector/VectorList.hpp>
#include <deque>
#include <unordered_set>

namespace TDFANN::RNSG {

template <typename T>
struct BeamScratch {
    std::vector<std::uint32_t> visit_stamp;
    std::uint32_t visit_epoch = 1;
    std::vector<std::pair<unsigned, T>> visited_nodes;
    std::vector<std::pair<T, unsigned>> candidates;
    std::vector<std::pair<T, unsigned>> neighbours;
    std::vector<unsigned> raw_nodes;
    std::vector<std::pair<unsigned, unsigned>> indexed_nodes;

    size_t visit_stamp_size = 0;
    size_t visited_reserved = 0;
    size_t candidate_reserved = 0;
    size_t neighbour_reserved = 0;
    size_t raw_reserved = 0;
    size_t indexed_reserved = 0;

    BeamScratch() = default;

    BeamScratch(size_t dataset_size, unsigned beam_size, unsigned trunc_size) {
        ensure(dataset_size, beam_size, trunc_size);
    }

    void ensure(size_t dataset_size, unsigned beam_size, unsigned trunc_size) {
        if (dataset_size > visit_stamp_size) {
            visit_stamp.resize(dataset_size, 0);
            visit_stamp_size = dataset_size;
        }

        const size_t visited_need =
            std::max<size_t>(1024, static_cast<size_t>(beam_size) *
                                       (static_cast<size_t>(trunc_size) + 2));
        if (visited_need > visited_reserved) {
            visited_nodes.reserve(visited_need);
            visited_reserved = visited_need;
        }

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

        const size_t raw_need =
            std::max<size_t>(static_cast<size_t>(trunc_size) * 4 + 8,
                             static_cast<size_t>(trunc_size) + 8);
        if (raw_need > raw_reserved) {
            raw_nodes.reserve(raw_need);
            raw_reserved = raw_need;
        }
        if (raw_need > indexed_reserved) {
            indexed_nodes.reserve(raw_need);
            indexed_reserved = raw_need;
        }
    }

    void next_query() {
        visit_epoch++;
        if (visit_epoch == 0) {
            std::fill(visit_stamp.begin(), visit_stamp.end(), 0);
            visit_epoch = 1;
        }
        visited_nodes.clear();
        candidates.clear();
        neighbours.clear();
        raw_nodes.clear();
        indexed_nodes.clear();
    }

    bool is_visited(unsigned id) const {
        return id < visit_stamp.size() && visit_stamp[id] == visit_epoch;
    }

    void mark_visited(unsigned id, T dist) {
        if (id >= visit_stamp.size() || visit_stamp[id] == visit_epoch) {
            return;
        }
        visit_stamp[id] = visit_epoch;
        visited_nodes.push_back({id, dist});
    }
};

/// Per-round diagnostic snapshot for signal analysis (zero overhead when
/// diag_ptr == nullptr).
struct BeamSearchDiag {
    /// One entry per expansion round.
    struct RoundSnapshot {
        unsigned round = 0;             // 0-based expansion index
        unsigned expanded_node = 0;     // which node was expanded
        unsigned insertions = 0;        // insertions this round
        unsigned cumulative_insertions = 0;
        double best_dist = 1e100;       // best dist in beam after round
        double best_dist_ever = 1e100;  // best dist ever seen
        double best_dist_delta = 0.0;   // improvement this round (0=none)
        unsigned beam_label_span = 0;   // max - min node id in beam
        unsigned beam_size_effective = 0;  // non-placeholder entries
        unsigned stall_rounds = 0;
        // Label-gap distribution of inserted edges
        double inserted_gap_min = 0.0;
        double inserted_gap_max = 0.0;
        double inserted_gap_mean = 0.0;
        // How many of the current beam entries are "bridge" (high gap) vs
        // "core" (low gap)
        unsigned beam_bridge_count = 0;
        unsigned beam_core_count = 0;

        // --- yield trace fields ---
        unsigned raw_count = 0;         // raw_nodes scanned before select
        unsigned selected_count = 0;    // neighbours after select_from_raw
        unsigned range_filtered = 0;    // neighbours filtered by range
        unsigned dist_eval_count = 0;   // actual distance computations
        unsigned improve_count = 0;     // insertions that improved best_dist
    };

    std::vector<RoundSnapshot> rounds;
    bool enabled = false;

    void clear() { rounds.clear(); }
};

/// Edge provenance categories for attribution analysis.
enum EdgeProv : uint8_t {
    kProvDirect = 0,        // direct fill (no select_from_raw)
    kProvPrefix = 1,        // prefix positions (front_keep)
    kProvCore = 2,          // core selection (reciprocal+position sorted)
    kProvBridge = 3,        // bridge selection (gap sorted)
    kProvDeepBridge = 4,    // deep bridge floor (deep position + gap sorted)
    kProvSuffixPromote = 5, // suffix promote (uniform deep sampling)
    kProvRescue = 6,        // rescue candidates
    kProvOther = 7          // fill remaining by original order
};

/// Record of a single GT node hit during beam search, with edge provenance.
struct GtHitProv {
    unsigned gt_node_id = 0;       // the GT node that was hit
    unsigned expansion_round = 0;  // 0-based expansion index
    uint8_t edge_category = 7;     // EdgeProv value
    unsigned edge_raw_position = 0;// position in raw_nodes (UINT_MAX for direct)
    unsigned edge_gap = 0;         // |gt_node_id - current_node|
    bool is_post_arrival = false;  // true if after first GT hit
    unsigned current_node = 0;     // node being expanded when hit occurred
};

/// Per-evaluated-neighbor record for P10-1 provenance analysis.
struct EvalProvRecord {
    unsigned query_id = 0;
    unsigned expansion_round = 0;
    unsigned current_node = 0;
    unsigned neighbor_node = 0;
    uint8_t edge_category = 7;       // EdgeProv value
    unsigned selected_rank = 0;      // position in neighbours vector
    bool prior_insertion = false;    // did an earlier neighbor insert?
    bool was_inserted = false;       // did this neighbor enter the beam?
    bool improved_best = false;      // did this neighbor beat best_dist_ever?
    bool is_post_arrival = false;
    float dist = 0.0f;
};

/// Optional counters for query-level logging / analysis (zero overhead when
/// unused).
struct BeamSearchStats {
    std::uint64_t distance_computations = 0;
    std::uint64_t expanded_nodes = 0;
    std::uint64_t raw_neighbors_scanned = 0;
    std::uint64_t range_filtered_out_neighbors = 0;
    std::uint64_t in_range_neighbors_evaluated = 0;
    std::uint64_t beam_insertions = 0;
    std::uint64_t beam_rewinds = 0;
    bool early_stop_triggered = false;
    std::uint64_t visited_nodes_count = 0;

    // --- trigger telemetry ---
    bool fallback_triggered = false;  // whether fallback/rescue was triggered
    unsigned fallback_trigger_step =
        0;  // 0-based expansion index when triggered
    unsigned fallback_trigger_stall_rounds =
        0;  // stall_rounds value when triggered
    std::uint64_t post_trigger_dco =
        0;  // distance_computations accumulated after trigger
    unsigned fallback_reason_code =
        0;  // 0=not triggered, 1=stall_threshold+warmup

    // --- rescue slots telemetry ---
    unsigned rescue_candidates_added =
        0;  // rescue candidates injected (extra budget)
    unsigned rescue_candidates_evaluated =
        0;  // rescue candidates that were distance-computed
    unsigned rescue_candidates_inserted =
        0;  // rescue candidates that entered the beam

    // --- composite trigger snapshots (captured at trigger time) ---
    unsigned fallback_trigger_recent_insertions =
        0;  // rolling-window insertion sum at trigger time
    unsigned fallback_trigger_flat_rounds =
        0;  // best_dist flat rounds at trigger time

    // --- post-arrival DCO breakdown ---
    unsigned first_gt_hit_expansion =
        0;  // expansion index when first GT node was hit (0-based)
    std::uint64_t dco_at_first_gt_hit =
        0;  // cumulative DCO at first GT hit

    // --- select_from_raw branch tracking ---
    unsigned branch_direct = 0;      // |raw| <= local_budget (no selection needed)
    unsigned branch_selected = 0;    // |raw| > local_budget, select_from_raw called
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
    std::vector<std::pair<T, unsigned>> beam_search_range_fast(
        const GoalId& goal,
        unsigned k,
        StartNode start_node,
        unsigned beam_size,
        unsigned trunc_size,
        unsigned range_l,
        unsigned range_r,
        BeamScratch<T>& scratch,
        unsigned nav_degree = 16,
        unsigned nav_scan_factor = 4,
        unsigned nav_stall_rounds = 8,
        unsigned nav_front_keep = 8,
        unsigned nav_tail_degree = 0,
        unsigned nav_early_stop_rounds = 0,
        bool use_sorted_range_idx = false) {
        static_assert(
            IndexOrVector<GoalId, T>,
            "GoalId must be convertible to unsigned or a vector-like type");

        if (beam_size == 0 || range_l > range_r) {
            return {};
        }

        const unsigned offset = dataset.size();
        scratch.ensure(dataset.size(), beam_size, trunc_size);
        scratch.next_query();

        auto& candidates = scratch.candidates;
        auto& neighbours = scratch.neighbours;
        auto& raw_nodes = scratch.raw_nodes;
        auto& indexed_nodes = scratch.indexed_nodes;

        if constexpr (std::convertible_to<StartNode, unsigned>) {
            const unsigned id = static_cast<unsigned>(start_node);
            if (id >= range_l && id <= range_r) {
                candidates.push_back({T(0), id});
            }
        } else {
            for (auto id : start_node) {
                const unsigned uid = static_cast<unsigned>(id);
                if (uid >= range_l && uid <= range_r) {
                    candidates.push_back({T(0), uid});
                }
            }
        }

        if (candidates.empty()) {
            return {};
        }

        dataset.dist_all_into(goal, candidates);
        std::ranges::sort(candidates);
        for (auto& [dis, id] : candidates) {
            scratch.mark_visited(id, dis);
            id += offset;
        }

        if (candidates.size() < beam_size) {
            candidates.resize(beam_size,
                              {T(1e100), candidates[0].second - offset});
        } else if (candidates.size() > beam_size) {
            candidates.resize(beam_size);
        }

        const unsigned effective_nav_degree =
            (nav_degree == 0 ? trunc_size : std::min(nav_degree, trunc_size));
        const unsigned effective_scan_factor = std::max(1u, nav_scan_factor);
        const unsigned effective_stall_rounds = std::max(1u, nav_stall_rounds);
        const unsigned effective_front_keep = std::max(1u, nav_front_keep);
        const unsigned effective_tail_degree =
            (nav_tail_degree == 0 ? 0 : std::min(nav_tail_degree, trunc_size));
        const unsigned effective_early_stop_rounds = nav_early_stop_rounds;

        unsigned stall_rounds = 0;

        for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
            if (candidates[uid].second < offset) {
                continue;
            }
            candidates[uid].second -= offset;
            const unsigned current_node = candidates[uid].second;

            if (trunc_size == 0) {
                continue;
            }

            unsigned local_budget = trunc_size;
            if (effective_nav_degree > 0 && effective_nav_degree < trunc_size) {
                local_budget = effective_nav_degree;
                if (stall_rounds >= effective_stall_rounds) {
                    local_budget = trunc_size;
                }
            }
            if (effective_tail_degree > 0 &&
                uid >= static_cast<int>(beam_size / 2)) {
                local_budget = std::min(local_budget, effective_tail_degree);
            }
            if (local_budget == 0) {
                continue;
            }

            unsigned scan_limit = trunc_size;
            if (local_budget < trunc_size) {
                const unsigned nav_scan_limit = std::min(
                    trunc_size, std::max(local_budget,
                                         local_budget * effective_scan_factor));
                scan_limit = std::max(scan_limit, nav_scan_limit);
            }

            raw_nodes.clear();
            neighbours.clear();

            const bool use_indexed_range =
                use_sorted_range_idx && graph.has_sorted_neighbor_index();
            auto append_range_neighbors =
                [&](std::vector<std::pair<T, unsigned>>& dst,
                    unsigned limit) {
                    if (use_indexed_range) {
                        graph.get_neighbours_in_range_indexed_into(
                            current_node, range_l, range_r, indexed_nodes);
                        for (const auto& [nid, original_pos] : indexed_nodes) {
                            (void)original_pos;
                            if (scratch.is_visited(nid)) {
                                continue;
                            }
                            dst.push_back({T(0), nid});
                            if (dst.size() >= limit) {
                                break;
                            }
                        }
                        return;
                    }
                    for (const auto& x : graph.get_neighbours(current_node)) {
                        const unsigned nid = x.to;
                        if (nid < range_l || nid > range_r ||
                            scratch.is_visited(nid)) {
                            continue;
                        }
                        dst.push_back({T(0), nid});
                        if (dst.size() >= limit) {
                            break;
                        }
                    }
                };
            auto append_range_raw = [&](unsigned limit) {
                if (use_indexed_range) {
                    graph.get_neighbours_in_range_indexed_into(
                        current_node, range_l, range_r, indexed_nodes);
                    for (const auto& [nid, original_pos] : indexed_nodes) {
                        (void)original_pos;
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        raw_nodes.push_back(nid);
                        if (raw_nodes.size() >= limit) {
                            break;
                        }
                    }
                    return;
                }
                for (const auto& x : graph.get_neighbours(current_node)) {
                    const unsigned nid = x.to;
                    if (nid < range_l || nid > range_r ||
                        scratch.is_visited(nid)) {
                        continue;
                    }
                    raw_nodes.push_back(nid);
                    if (raw_nodes.size() >= limit) {
                        break;
                    }
                }
            };

            if (scan_limit <= local_budget) {
                append_range_neighbors(neighbours, local_budget);
            } else {
                append_range_raw(scan_limit);

                if (raw_nodes.size() <= local_budget) {
                    for (auto nid : raw_nodes) {
                        neighbours.push_back({T(0), nid});
                    }
                } else {
                    const size_t raw_n = raw_nodes.size();
                    const size_t need =
                        std::min<size_t>(local_budget, raw_n);
                    const size_t prefix_keep = std::min<size_t>(
                        std::min<size_t>(effective_front_keep, need), raw_n);
                    for (size_t i = 0; i < prefix_keep; ++i) {
                        neighbours.push_back({T(0), raw_nodes[i]});
                    }
                    const size_t remain = need - neighbours.size();
                    if (remain > 0) {
                        const size_t tail_n = raw_n - prefix_keep;
                        size_t last_idx = std::numeric_limits<size_t>::max();
                        for (size_t i = 0; i < remain; ++i) {
                            const size_t rel =
                                (remain == 1)
                                    ? 0
                                    : (i * (tail_n - 1)) / (remain - 1);
                            const size_t idx = prefix_keep + rel;
                            if (idx == last_idx) {
                                continue;
                            }
                            last_idx = idx;
                            neighbours.push_back({T(0), raw_nodes[idx]});
                        }
                    }
                }
            }

            if (neighbours.empty()) {
                continue;
            }

            dataset.dist_all_into(goal, neighbours);
            bool improved = false;
            for (const auto& [dist, nto] : neighbours) {
                scratch.mark_visited(nto, dist);
                if (dist < candidates.back().first) {
                    candidates.pop_back();
                    auto it = std::partition_point(
                        candidates.begin(), candidates.end(),
                        [&](const auto& a) { return a.first < dist; });
                    const int pos =
                        static_cast<int>(it - candidates.begin());
                    uid = std::min(uid, pos - 1);
                    candidates.insert(it, {dist, nto + offset});
                    if (dist < candidates.front().first) {
                        improved = true;
                    }
                }
            }

            if (improved) {
                stall_rounds = 0;
            } else {
                stall_rounds++;
                if (effective_early_stop_rounds > 0 &&
                    stall_rounds >= effective_early_stop_rounds &&
                    uid >= static_cast<int>(beam_size / 3)) {
                    break;
                }
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

    template <typename GoalId, IndexOrList StartNode>
    std::vector<std::pair<T, unsigned>> beam_search(
        const GoalId& goal,
        unsigned k,
        StartNode start_node,
        unsigned beam_size,
        unsigned trunc_size,
        BeamScratch<T>& scratch,
        unsigned nav_degree = 16,
        unsigned nav_scan_factor = 4,
        unsigned nav_stall_rounds = 8,
        unsigned nav_front_keep = 8,
        unsigned nav_tail_degree = 0,
        unsigned nav_early_stop_rounds = 0,
        unsigned pick_scan_factor = 1,
        unsigned pick_front_keep = 0,
        unsigned edge_pick_policy = 0,
        unsigned edge_pick_recip_depth = 32,
        double edge_pick_core_ratio = 0.6,
        unsigned fallback_stall_rounds = 0,
        unsigned fallback_pick_policy = 0,
        double fallback_core_ratio = 0.40,
        unsigned fallback_pick_front_keep = 0,
        unsigned fallback_pick_scan_factor = 1,
        bool fallback_release_nav = false,
        std::vector<std::pair<T, unsigned>>* candidates_ptr = nullptr,
        BeamSearchStats* beam_stats = nullptr,
        unsigned rescue_slot_count = 0,
        unsigned rescue_pick_policy = 0,
        unsigned warmup_min = 16,
        unsigned trigger_recent_window = 0,
        unsigned trigger_flat_threshold = 0,
        unsigned bridge_quota_floor = 0) {
        static_assert(
            IndexOrVector<GoalId, T>,
            "GoalId must be convertible to unsigned or a vector-like type");

        if (beam_size == 0) {
            return {};
        }

        const unsigned offset = dataset.size();
        scratch.ensure(dataset.size(), beam_size, trunc_size);
        scratch.next_query();
        if (beam_stats != nullptr) {
            *beam_stats = BeamSearchStats{};
        }

        auto& visited_nodes = scratch.visited_nodes;
        auto& candidates = scratch.candidates;
        auto& neighbours = scratch.neighbours;
        auto& raw_nodes = scratch.raw_nodes;

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
        if (beam_stats != nullptr) {
            beam_stats->distance_computations +=
                static_cast<std::uint64_t>(candidates.size());
        }
        std::ranges::sort(candidates);
        for (auto& [dis, id] : candidates) {
            scratch.mark_visited(id, dis);
            id += offset;
        }

        if (candidates.size() < beam_size) {
            candidates.resize(beam_size,
                              {T(1e100), candidates[0].second - offset});
        } else if (candidates.size() > beam_size) {
            candidates.resize(beam_size);
        }

        const unsigned effective_nav_degree =
            (nav_degree == 0 ? trunc_size : std::min(nav_degree, trunc_size));
        const unsigned effective_scan_factor = std::max(1u, nav_scan_factor);
        const unsigned effective_stall_rounds = std::max(1u, nav_stall_rounds);
        const unsigned effective_front_keep = std::max(1u, nav_front_keep);
        const unsigned effective_tail_degree =
            (nav_tail_degree == 0 ? 0 : std::min(nav_tail_degree, trunc_size));
        const unsigned effective_early_stop_rounds = nav_early_stop_rounds;
        const unsigned effective_pick_scan_factor =
            std::max(1u, pick_scan_factor);
        const unsigned effective_pick_front_keep =
            (pick_front_keep == 0 ? effective_front_keep : pick_front_keep);
        unsigned stall_rounds = 0;

        unsigned eff_fb_pick_policy = edge_pick_policy;
        double eff_fb_core_ratio = edge_pick_core_ratio;
        unsigned eff_fb_pick_front_keep = effective_pick_front_keep;
        unsigned eff_fb_pick_scan_factor = effective_pick_scan_factor;
        bool local_fb_triggered = false;
        unsigned rescue_remaining = 0;
        size_t rescue_offset = 0;
        const unsigned fb_stall = fallback_stall_rounds;
        const unsigned fb_policy = fallback_pick_policy;
        const double fb_core = fallback_core_ratio;
        const unsigned fb_front_keep = (fallback_pick_front_keep == 0)
                                           ? effective_pick_front_keep
                                           : fallback_pick_front_keep;
        const unsigned fb_scan_factor = (fallback_pick_scan_factor <= 1)
                                            ? effective_pick_scan_factor
                                            : fallback_pick_scan_factor;

        // Composite trigger runtime state (local variables, not in stats)
        std::deque<unsigned> insertion_hist;
        unsigned insertion_hist_sum = 0;
        double best_dist_ever = 1e100;
        unsigned best_dist_flat_rounds = 0;
        unsigned diag_cumulative_insertions = 0;
        unsigned expansion_round = 0;

        // Per-round yield trace locals
        unsigned round_raw_count = 0;
        unsigned round_selected_count = 0;
        unsigned round_range_filtered = 0;
        unsigned round_improve_count = 0;
        // GT hit tracking (not used in beam_search, only beam_search_range)
        bool gt_hit_recorded = true;  // always true = skip
        const std::unordered_set<unsigned>* gt_set_ptr = nullptr;
        (void)gt_hit_recorded; (void)gt_set_ptr;

        for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
            if (candidates[uid].second < offset) {
                continue;
            }
            candidates[uid].second -= offset;
            const unsigned current_node = candidates[uid].second;

            if (trunc_size == 0) {
                continue;
            }

            unsigned local_budget = trunc_size;
            if (effective_nav_degree > 0 && effective_nav_degree < trunc_size) {
                local_budget = effective_nav_degree;
                if (stall_rounds >= effective_stall_rounds) {
                    local_budget = trunc_size;
                }
            }
            if (effective_tail_degree > 0 &&
                uid >= static_cast<int>(beam_size / 2)) {
                local_budget = std::min(local_budget, effective_tail_degree);
            }
            if (local_budget == 0) {
                continue;
            }

            if (beam_stats != nullptr) {
                beam_stats->expanded_nodes++;
            }
            expansion_round++;

            // --- Composite trigger condition check ---
            bool should_trigger = false;

            if (!local_fb_triggered && fb_stall > 0 &&
                stall_rounds >= fb_stall &&
                (!beam_stats || beam_stats->expanded_nodes >= warmup_min)) {

                bool no_recent_insertions = true;
                bool best_dist_flat = true;

                if (trigger_recent_window > 0) {
                    if (insertion_hist.size() >= trigger_recent_window) {
                        no_recent_insertions = (insertion_hist_sum == 0);
                    } else {
                        no_recent_insertions = false;
                    }
                }

                if (trigger_flat_threshold > 0) {
                    best_dist_flat =
                        (best_dist_flat_rounds >= trigger_flat_threshold);
                }

                if (trigger_recent_window == 0 &&
                    trigger_flat_threshold == 0) {
                    should_trigger = true;
                } else {
                    should_trigger =
                        no_recent_insertions && best_dist_flat;
                }

                if (should_trigger) {
                    if (rescue_slot_count > 0) {
                        rescue_remaining = rescue_slot_count;
                    } else {
                        eff_fb_pick_policy = fb_policy;
                        eff_fb_core_ratio = fb_core;
                        eff_fb_pick_front_keep = fb_front_keep;
                        eff_fb_pick_scan_factor = fb_scan_factor;
                        if (fallback_release_nav) {
                            local_budget = trunc_size;
                        }
                    }
                    local_fb_triggered = true;
                    if (beam_stats != nullptr) {
                        beam_stats->fallback_triggered = true;
                        beam_stats->fallback_trigger_step =
                            beam_stats->expanded_nodes;
                        beam_stats->fallback_trigger_stall_rounds =
                            stall_rounds;
                        beam_stats->fallback_trigger_recent_insertions =
                            insertion_hist_sum;
                        beam_stats->fallback_trigger_flat_rounds =
                            best_dist_flat_rounds;

                        if (trigger_recent_window > 0 ||
                            trigger_flat_threshold > 0) {
                            beam_stats->fallback_reason_code = 2;
                        } else {
                            beam_stats->fallback_reason_code = 1;
                        }
                    }
                }
            }

            unsigned scan_limit = trunc_size;
            if (eff_fb_pick_scan_factor > 1) {
                const auto scaled_scan =
                    static_cast<std::uint64_t>(trunc_size) *
                    static_cast<std::uint64_t>(eff_fb_pick_scan_factor);
                if (scaled_scan > static_cast<std::uint64_t>(
                                      std::numeric_limits<unsigned>::max())) {
                    scan_limit = std::numeric_limits<unsigned>::max();
                } else {
                    scan_limit = static_cast<unsigned>(scaled_scan);
                }
            }
            if (local_budget < trunc_size) {
                const unsigned nav_scan_limit = std::min(
                    trunc_size, std::max(local_budget,
                                         local_budget * effective_scan_factor));
                scan_limit = std::max(scan_limit, nav_scan_limit);
            }

            raw_nodes.clear();
            neighbours.clear();
            round_raw_count = 0;
            round_selected_count = 0;
            round_range_filtered = 0;
            round_improve_count = 0;
            if (scan_limit <= local_budget) {
                // Direct path: no selection needed
                for (const auto& x : graph.get_neighbours(current_node)) {
                    if (beam_stats != nullptr) {
                        beam_stats->raw_neighbors_scanned++;
                    }
                    if (scratch.is_visited(x.to)) {
                        continue;
                    }
                    neighbours.push_back({T(0), x.to});
                    if (neighbours.size() >= local_budget) {
                        break;
                    }
                }
                round_raw_count = static_cast<unsigned>(neighbours.size());
                round_selected_count = static_cast<unsigned>(neighbours.size());
                if (beam_stats != nullptr) {
                    beam_stats->branch_direct++;
                }
            } else {
                for (const auto& x : graph.get_neighbours(current_node)) {
                    if (beam_stats != nullptr) {
                        beam_stats->raw_neighbors_scanned++;
                    }
                    if (scratch.is_visited(x.to)) {
                        continue;
                    }
                    raw_nodes.push_back(x.to);
                    if (raw_nodes.size() >= scan_limit) {
                        break;
                    }
                }

                round_raw_count = static_cast<unsigned>(raw_nodes.size());

                if (raw_nodes.size() <= local_budget) {
                    for (auto nid : raw_nodes) {
                        neighbours.push_back({T(0), nid});
                    }
                    round_selected_count = static_cast<unsigned>(neighbours.size());
                } else {
                    select_from_raw(current_node, raw_nodes, local_budget,
                                    eff_fb_pick_front_keep, eff_fb_pick_policy,
                                    edge_pick_recip_depth, eff_fb_core_ratio,
                                    bridge_quota_floor, 0,  // suffix_promote not in beam_search
                                    0, 32,  // deep_bridge_floor not in beam_search
                                    neighbours);
                    round_selected_count = static_cast<unsigned>(neighbours.size());
                    if (beam_stats != nullptr) {
                        beam_stats->branch_selected++;
                    }
                }
            }

            rescue_offset = 0;
            size_t rescue_added = 0;
            if (rescue_remaining > 0 && !raw_nodes.empty()) {
                rescue_offset = neighbours.size();
                const unsigned eff_fkeep = eff_fb_pick_front_keep;

                if (rescue_pick_policy == 1) {
                    // Gap-based rescue: select neighbors with largest label gap
                    std::vector<std::pair<unsigned, int>> label_gaps;
                    label_gaps.reserve(raw_nodes.size());
                    for (size_t ri = 0; ri < raw_nodes.size(); ++ri) {
                        if (ri < eff_fkeep) {
                            continue;
                        }
                        unsigned nid = raw_nodes[ri];
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        int gap = static_cast<int>(std::abs(
                            static_cast<long long>(nid) -
                            static_cast<long long>(current_node)));
                        label_gaps.push_back({nid, gap});
                    }
                    std::ranges::sort(label_gaps,
                        [](const auto& a, const auto& b) {
                            return a.second > b.second;
                        });
                    for (size_t i = 0;
                         i < label_gaps.size() && rescue_added < rescue_remaining;
                         ++i) {
                        neighbours.push_back({T(0), label_gaps[i].first});
                        ++rescue_added;
                    }
                } else {
                    for (size_t ri = 0;
                         ri < raw_nodes.size() && rescue_added < rescue_remaining;
                         ++ri) {
                        if (ri < eff_fkeep) {
                            continue;
                        }
                        unsigned nid = raw_nodes[ri];
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        neighbours.push_back({T(0), nid});
                        ++rescue_added;
                    }
                }
                if (beam_stats != nullptr && rescue_added > 0) {
                    beam_stats->rescue_candidates_added += rescue_added;
                }
                rescue_remaining = 0;
            }

            // Record yield trace: raw and selected counts
            round_raw_count = static_cast<unsigned>(raw_nodes.size());
            round_selected_count = static_cast<unsigned>(neighbours.size());

            if (neighbours.empty()) {
                continue;
            }

            if (beam_stats != nullptr) {
                beam_stats->in_range_neighbors_evaluated +=
                    static_cast<std::uint64_t>(neighbours.size());
                beam_stats->distance_computations +=
                    static_cast<std::uint64_t>(neighbours.size());
                if (beam_stats->fallback_triggered) {
                    beam_stats->post_trigger_dco +=
                        static_cast<std::uint64_t>(neighbours.size());
                }
            }
            dataset.dist_all_into(goal, neighbours);
            bool improved = false;
            unsigned insertions_this_round = 0;
            for (size_t ni = 0; ni < neighbours.size(); ++ni) {
                const auto& [dist, nto] = neighbours[ni];
                scratch.mark_visited(nto, dist);
                const bool is_rescue =
                    (rescue_added > 0 && ni >= rescue_offset);
                if (is_rescue && beam_stats != nullptr) {
                    beam_stats->rescue_candidates_evaluated++;
                }
                if (dist < candidates.back().first) {
                    if (beam_stats != nullptr) {
                        beam_stats->beam_insertions++;
                        if (is_rescue) {
                            beam_stats->rescue_candidates_inserted++;
                        }
                    }
                    insertions_this_round++;
                    diag_cumulative_insertions++;
                    candidates.pop_back();
                    auto it = std::partition_point(
                        candidates.begin(), candidates.end(),
                        [&](const auto& a) { return a.first < dist; });
                    const int pos = static_cast<int>(it - candidates.begin());
                    if (beam_stats != nullptr && pos > 0 && pos - 1 < uid) {
                        beam_stats->beam_rewinds++;
                    }
                    uid = std::min(uid, pos - 1);
                    candidates.insert(it, {dist, nto + offset});
                    if (dist < candidates.front().first) {
                        improved = true;
                        round_improve_count++;
                    }
                }
            }

            // Yield trace: improve_count accumulated per-insertion above

            // GT hit tracking (only when gt_set_ptr is provided)
            if (!gt_hit_recorded && gt_set_ptr != nullptr && beam_stats != nullptr) {
                for (size_t ni = 0; ni < neighbours.size(); ++ni) {
                    const auto& [dist, nto] = neighbours[ni];
                    if (gt_set_ptr->count(nto) > 0) {
                        gt_hit_recorded = true;
                        beam_stats->first_gt_hit_expansion =
                            static_cast<unsigned>(beam_stats->expanded_nodes);
                        beam_stats->dco_at_first_gt_hit =
                            beam_stats->distance_computations;
                        break;
                    }
                }
            }

            // Update composite trigger runtime state
            insertion_hist.push_back(insertions_this_round);
            insertion_hist_sum += insertions_this_round;
            if (insertion_hist.size() > trigger_recent_window) {
                insertion_hist_sum -= insertion_hist.front();
                insertion_hist.pop_front();
            }

            double current_best = candidates.empty()
                                      ? 1e100
                                      : static_cast<double>(candidates[0].first);
            if (current_best < best_dist_ever - 1e-9) {
                best_dist_ever = current_best;
                best_dist_flat_rounds = 0;
            } else {
                best_dist_flat_rounds++;
            }

            if (improved) {
                stall_rounds = 0;
            } else {
                stall_rounds++;
                if (effective_early_stop_rounds > 0 &&
                    stall_rounds >= effective_early_stop_rounds &&
                    uid >= static_cast<int>(beam_size / 3)) {
                    if (beam_stats != nullptr) {
                        beam_stats->early_stop_triggered = true;
                    }
                    break;
                }
            }
        }

        if (candidates_ptr != nullptr) {
            auto& c = *candidates_ptr;
            c.reserve(c.size() + visited_nodes.size());
            for (const auto& [id, dis] : visited_nodes) {
                c.push_back({dis, id});
            }
        }

        if (beam_stats != nullptr) {
            beam_stats->visited_nodes_count =
                static_cast<std::uint64_t>(visited_nodes.size());
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

    template <typename GoalId, IndexOrList StartNode>
    std::vector<std::pair<T, unsigned>> beam_search_range(
        const GoalId& goal,
        unsigned k,
        StartNode start_node,
        unsigned beam_size,
        unsigned trunc_size,
        unsigned range_l,
        unsigned range_r,
        BeamScratch<T>& scratch,
        unsigned nav_degree = 16,
        unsigned nav_scan_factor = 4,
        unsigned nav_stall_rounds = 8,
        unsigned nav_front_keep = 8,
        unsigned nav_tail_degree = 0,
        unsigned nav_early_stop_rounds = 0,
        unsigned pick_scan_factor = 1,
        unsigned pick_front_keep = 0,
        unsigned edge_pick_policy = 0,
        unsigned edge_pick_recip_depth = 32,
        double edge_pick_core_ratio = 0.6,
        unsigned fallback_stall_rounds = 0,
        unsigned fallback_pick_policy = 0,
        double fallback_core_ratio = 0.40,
        unsigned fallback_pick_front_keep = 0,
        unsigned fallback_pick_scan_factor = 1,
        bool fallback_release_nav = false,
        std::vector<std::pair<T, unsigned>>* candidates_ptr = nullptr,
        BeamSearchStats* beam_stats = nullptr,
        unsigned rescue_slot_count = 0,
        unsigned rescue_pick_policy = 0,
        unsigned warmup_min = 16,
        unsigned trigger_recent_window = 0,
        unsigned trigger_flat_threshold = 0,
        unsigned trigger_span_round = 0,
        double trigger_span_norm_threshold = 0.0,
        BeamSearchDiag* diag_ptr = nullptr,
        unsigned bridge_quota_floor = 0,
        const std::unordered_set<unsigned>* gt_set_ptr = nullptr,
        unsigned suffix_promote = 0,
        unsigned deep_bridge_floor = 0,
        unsigned deep_bridge_pos_threshold = 32,
        std::vector<GtHitProv>* gt_provenance = nullptr,
        unsigned post_arrival_eval_cap = 0,
        unsigned post_arrival_core_cap = 0,
        unsigned post_arrival_bridge_cap = 0,
        unsigned post_arrival_prefix_cap = 0,
        std::vector<EvalProvRecord>* eval_prov_out = nullptr,
        // P10-2: Adaptive early-stop parameters
        unsigned pa_adaptive_cap = 0,
        unsigned pa_adaptive_window = 1,
        unsigned pa_stall_threshold = 0,
        unsigned pa_stall_decay = 2,
        unsigned pa_stall_floor = 4,
        unsigned pa_stable_threshold = 0,
        unsigned pa_stable_cap = 0,
        // P10-3: Rank-aware tail pruning
        unsigned pa_rank_head_keep = 0,
        unsigned pa_rank_stale_window = 3,
        // P11-2: Head-gated tail unlock
        unsigned pa_head_gate_keep = 0,
        unsigned pa_head_gate_tail_budget = 0,
        // P12B: Range-aware access via dual-index
        bool use_sorted_range_idx = false) {
        static_assert(
            IndexOrVector<GoalId, T>,
            "GoalId must be convertible to unsigned or a vector-like type");

        if (beam_size == 0 || range_l > range_r) {
            return {};
        }

        const unsigned offset = dataset.size();
        bool gt_hit_recorded = false;
        scratch.ensure(dataset.size(), beam_size, trunc_size);
        scratch.next_query();
        if (beam_stats != nullptr) {
            *beam_stats = BeamSearchStats{};
        }
        if (diag_ptr != nullptr) {
            diag_ptr->clear();
            diag_ptr->enabled = true;
        }

        auto& visited_nodes = scratch.visited_nodes;
        auto& candidates = scratch.candidates;
        auto& neighbours = scratch.neighbours;
        auto& raw_nodes = scratch.raw_nodes;

        if constexpr (std::convertible_to<StartNode, unsigned>) {
            candidates.push_back({T(0), static_cast<unsigned>(start_node)});
        } else {
            for (auto id : start_node) {
                if (id < range_l || id > range_r) {
                    continue;
                }
                candidates.push_back({T(0), static_cast<unsigned>(id)});
            }
        }

        if (candidates.empty()) {
            return {};
        }

        dataset.dist_all_into(goal, candidates);
        if (beam_stats != nullptr) {
            beam_stats->distance_computations +=
                static_cast<std::uint64_t>(candidates.size());
        }
        std::ranges::sort(candidates);
        for (auto& [dis, id] : candidates) {
            scratch.mark_visited(id, dis);
            id += offset;
        }

        if (candidates.size() < beam_size) {
            candidates.resize(beam_size,
                              {T(1e100), candidates[0].second - offset});
        } else if (candidates.size() > beam_size) {
            candidates.resize(beam_size);
        }

        const unsigned effective_nav_degree =
            (nav_degree == 0 ? trunc_size : std::min(nav_degree, trunc_size));
        const unsigned effective_scan_factor = std::max(1u, nav_scan_factor);
        const unsigned effective_stall_rounds = std::max(1u, nav_stall_rounds);
        const unsigned effective_front_keep = std::max(1u, nav_front_keep);
        const unsigned effective_tail_degree =
            (nav_tail_degree == 0 ? 0 : std::min(nav_tail_degree, trunc_size));
        const unsigned effective_early_stop_rounds = nav_early_stop_rounds;
        const unsigned effective_pick_scan_factor =
            std::max(1u, pick_scan_factor);
        const unsigned effective_pick_front_keep =
            (pick_front_keep == 0 ? effective_front_keep : pick_front_keep);
        unsigned stall_rounds = 0;

        unsigned eff_fb_pick_policy = edge_pick_policy;
        double eff_fb_core_ratio = edge_pick_core_ratio;
        unsigned eff_fb_pick_front_keep = effective_pick_front_keep;
        unsigned eff_fb_pick_scan_factor = effective_pick_scan_factor;
        bool local_fb_triggered = false;
        unsigned rescue_remaining = 0;
        size_t rescue_offset = 0;
        const unsigned fb_stall = fallback_stall_rounds;
        const unsigned fb_policy = fallback_pick_policy;
        const double fb_core = fallback_core_ratio;
        const unsigned fb_front_keep = (fallback_pick_front_keep == 0)
                                           ? effective_pick_front_keep
                                           : fallback_pick_front_keep;
        const unsigned fb_scan_factor = (fallback_pick_scan_factor <= 1)
                                            ? effective_pick_scan_factor
                                            : fallback_pick_scan_factor;

        // Composite trigger runtime state (local variables, not in stats)
        std::deque<unsigned> insertion_hist;
        unsigned insertion_hist_sum = 0;
        double best_dist_ever = 1e100;
        unsigned best_dist_flat_rounds = 0;
        unsigned diag_cumulative_insertions = 0;
        unsigned expansion_round = 0;

        // P10-2/3: Adaptive early-stop state (post-arrival only)
        unsigned pa_stall = 0;
        std::deque<unsigned> pa_insertion_hist;

        // Per-round yield trace locals
        unsigned round_raw_count = 0;
        unsigned round_selected_count = 0;
        unsigned round_range_filtered = 0;
        unsigned round_improve_count = 0;

        // Per-round provenance tracking (parallel to neighbours)
        std::vector<uint8_t> neighbour_prov;

        for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
            if (candidates[uid].second < offset) {
                continue;
            }
            candidates[uid].second -= offset;
            const unsigned current_node = candidates[uid].second;

            if (trunc_size == 0) {
                continue;
            }

            unsigned local_budget = trunc_size;
            if (effective_nav_degree > 0 && effective_nav_degree < trunc_size) {
                local_budget = effective_nav_degree;
                if (stall_rounds >= effective_stall_rounds) {
                    local_budget = trunc_size;
                }
            }
            if (effective_tail_degree > 0 &&
                uid >= static_cast<int>(beam_size / 2)) {
                local_budget = std::min(local_budget, effective_tail_degree);
            }
            if (local_budget == 0) {
                continue;
            }

            if (beam_stats != nullptr) {
                beam_stats->expanded_nodes++;
            }
            expansion_round++;

            // --- Composite trigger condition check ---
            bool should_trigger = false;

            if (!local_fb_triggered && fb_stall > 0 &&
                stall_rounds >= fb_stall &&
                (!beam_stats || beam_stats->expanded_nodes >= warmup_min)) {

                bool no_recent_insertions = true;
                bool best_dist_flat = true;

                if (trigger_recent_window > 0) {
                    if (insertion_hist.size() >= trigger_recent_window) {
                        no_recent_insertions = (insertion_hist_sum == 0);
                    } else {
                        no_recent_insertions = false;
                    }
                }

                if (trigger_flat_threshold > 0) {
                    best_dist_flat =
                        (best_dist_flat_rounds >= trigger_flat_threshold);
                }

                if (trigger_recent_window == 0 &&
                    trigger_flat_threshold == 0) {
                    should_trigger = true;
                } else {
                    should_trigger =
                        no_recent_insertions && best_dist_flat;
                }

                if (should_trigger) {
                    if (rescue_slot_count > 0) {
                        rescue_remaining = rescue_slot_count;
                    } else {
                        eff_fb_pick_policy = fb_policy;
                        eff_fb_core_ratio = fb_core;
                        eff_fb_pick_front_keep = fb_front_keep;
                        eff_fb_pick_scan_factor = fb_scan_factor;
                        if (fallback_release_nav) {
                            local_budget = trunc_size;
                        }
                    }
                    local_fb_triggered = true;
                    if (beam_stats != nullptr) {
                        beam_stats->fallback_triggered = true;
                        beam_stats->fallback_trigger_step =
                            beam_stats->expanded_nodes;
                        beam_stats->fallback_trigger_stall_rounds =
                            stall_rounds;
                        beam_stats->fallback_trigger_recent_insertions =
                            insertion_hist_sum;
                        beam_stats->fallback_trigger_flat_rounds =
                            best_dist_flat_rounds;

                        if (trigger_recent_window > 0 ||
                            trigger_flat_threshold > 0) {
                            beam_stats->fallback_reason_code = 2;
                        } else {
                            beam_stats->fallback_reason_code = 1;
                        }
                    }
                }
            }

            // --- Span-based trigger check (beam_search_range only) ---
            if (!local_fb_triggered && trigger_span_round > 0 &&
                trigger_span_norm_threshold > 0.0 &&
                expansion_round == trigger_span_round) {
                unsigned s_min = std::numeric_limits<unsigned>::max();
                unsigned s_max = 0;
                unsigned s_eff = 0;
                for (const auto& [d, nid_raw] : candidates) {
                    unsigned nid = (nid_raw >= offset) ? nid_raw - offset
                                                       : nid_raw;
                    if (d < 1e99) {
                        s_eff++;
                        s_min = std::min(s_min, nid);
                        s_max = std::max(s_max, nid);
                    }
                }
                unsigned beam_span = (s_eff > 0) ? s_max - s_min : 0;
                const unsigned rw = range_r - range_l + 1;
                double span_norm = (rw > 0)
                                       ? static_cast<double>(beam_span) /
                                             static_cast<double>(rw)
                                       : 1.0;

                if (span_norm < trigger_span_norm_threshold) {
                    if (rescue_slot_count > 0) {
                        rescue_remaining = rescue_slot_count;
                    }
                    local_fb_triggered = true;
                    if (beam_stats != nullptr) {
                        beam_stats->fallback_triggered = true;
                        beam_stats->fallback_trigger_step =
                            beam_stats->expanded_nodes;
                        beam_stats->fallback_trigger_stall_rounds =
                            stall_rounds;
                        // Store span_norm*10000 in flat_rounds field for
                        // diagnostics (reuse existing CSV column)
                        beam_stats->fallback_trigger_flat_rounds =
                            static_cast<unsigned>(span_norm * 10000);
                        beam_stats->fallback_reason_code = 3;
                    }
                }
            }

            unsigned scan_limit = trunc_size;
            if (eff_fb_pick_scan_factor > 1) {
                const auto scaled_scan =
                    static_cast<std::uint64_t>(trunc_size) *
                    static_cast<std::uint64_t>(eff_fb_pick_scan_factor);
                if (scaled_scan > static_cast<std::uint64_t>(
                                      std::numeric_limits<unsigned>::max())) {
                    scan_limit = std::numeric_limits<unsigned>::max();
                } else {
                    scan_limit = static_cast<unsigned>(scaled_scan);
                }
            }
            if (local_budget < trunc_size) {
                const unsigned nav_scan_limit = std::min(
                    trunc_size, std::max(local_budget,
                                         local_budget * effective_scan_factor));
                scan_limit = std::max(scan_limit, nav_scan_limit);
            }

            raw_nodes.clear();
            neighbours.clear();
            neighbour_prov.clear();
            round_raw_count = 0;
            round_selected_count = 0;
            round_range_filtered = 0;
            round_improve_count = 0;

            // P12B: Range-aware access path via dual-index
            const bool use_range_idx =
                (use_sorted_range_idx && graph.has_sorted_neighbor_index());

            if (use_range_idx) {
                // Binary search for in-range neighbors, skip out-of-range
                // entirely. Returns (neighbor_id, original_position) sorted
                // by original position → same relative order as linear scan.
                auto in_range = graph.get_neighbours_in_range_indexed(
                    current_node, range_l, range_r);

                if (scan_limit <= local_budget) {
                    for (const auto& [nid, orig_pos] : in_range) {
                        if (beam_stats != nullptr) {
                            beam_stats->raw_neighbors_scanned++;
                        }
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        neighbours.push_back({T(0), nid});
                        neighbour_prov.push_back(kProvDirect);
                        if (neighbours.size() >= local_budget) {
                            break;
                        }
                    }
                } else {
                    for (const auto& [nid, orig_pos] : in_range) {
                        if (beam_stats != nullptr) {
                            beam_stats->raw_neighbors_scanned++;
                        }
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        raw_nodes.push_back(nid);
                        if (raw_nodes.size() >= scan_limit) {
                            break;
                        }
                    }

                    if (raw_nodes.size() <= local_budget) {
                        for (auto nid : raw_nodes) {
                            neighbours.push_back({T(0), nid});
                            neighbour_prov.push_back(kProvDirect);
                        }
                    } else {
                        select_from_raw(current_node, raw_nodes, local_budget,
                                        eff_fb_pick_front_keep,
                                        eff_fb_pick_policy,
                                        edge_pick_recip_depth,
                                        eff_fb_core_ratio, bridge_quota_floor,
                                        suffix_promote, deep_bridge_floor,
                                        deep_bridge_pos_threshold,
                                        neighbours, &neighbour_prov);
                    }
                }
            } else if (scan_limit <= local_budget) {
                for (const auto& x : graph.get_neighbours(current_node)) {
                    if (beam_stats != nullptr) {
                        beam_stats->raw_neighbors_scanned++;
                    }
                    if (x.to < range_l || x.to > range_r) {
                        if (beam_stats != nullptr) {
                            beam_stats->range_filtered_out_neighbors++;
                        }
                        continue;
                    }
                    if (scratch.is_visited(x.to)) {
                        continue;
                    }
                    neighbours.push_back({T(0), x.to});
                    neighbour_prov.push_back(kProvDirect);
                    if (neighbours.size() >= local_budget) {
                        break;
                    }
                }
            } else {
                for (const auto& x : graph.get_neighbours(current_node)) {
                    if (beam_stats != nullptr) {
                        beam_stats->raw_neighbors_scanned++;
                    }
                    if (x.to < range_l || x.to > range_r) {
                        if (beam_stats != nullptr) {
                            beam_stats->range_filtered_out_neighbors++;
                        }
                        continue;
                    }
                    if (scratch.is_visited(x.to)) {
                        continue;
                    }
                    raw_nodes.push_back(x.to);
                    if (raw_nodes.size() >= scan_limit) {
                        break;
                    }
                }

                if (raw_nodes.size() <= local_budget) {
                    for (auto nid : raw_nodes) {
                        neighbours.push_back({T(0), nid});
                        neighbour_prov.push_back(kProvDirect);
                    }
                } else {
                    select_from_raw(current_node, raw_nodes, local_budget,
                                    eff_fb_pick_front_keep, eff_fb_pick_policy,
                                    edge_pick_recip_depth, eff_fb_core_ratio,
                                    bridge_quota_floor, suffix_promote,
                                    deep_bridge_floor, deep_bridge_pos_threshold,
                                    neighbours, &neighbour_prov);
                }
            }

            rescue_offset = 0;
            size_t rescue_added = 0;
            if (rescue_remaining > 0 && !raw_nodes.empty()) {
                rescue_offset = neighbours.size();
                const unsigned eff_fkeep = eff_fb_pick_front_keep;

                if (rescue_pick_policy == 1) {
                    std::vector<std::pair<unsigned, int>> label_gaps;
                    label_gaps.reserve(raw_nodes.size());
                    for (size_t ri = 0; ri < raw_nodes.size(); ++ri) {
                        if (ri < eff_fkeep) {
                            continue;
                        }
                        unsigned nid = raw_nodes[ri];
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        int gap = static_cast<int>(std::abs(
                            static_cast<long long>(nid) -
                            static_cast<long long>(current_node)));
                        label_gaps.push_back({nid, gap});
                    }
                    std::ranges::sort(label_gaps,
                        [](const auto& a, const auto& b) {
                            return a.second > b.second;
                        });
                    for (size_t i = 0;
                         i < label_gaps.size() && rescue_added < rescue_remaining;
                         ++i) {
                        neighbours.push_back({T(0), label_gaps[i].first});
                        neighbour_prov.push_back(kProvRescue);
                        ++rescue_added;
                    }
                } else {
                    for (size_t ri = 0;
                         ri < raw_nodes.size() && rescue_added < rescue_remaining;
                         ++ri) {
                        if (ri < eff_fkeep) {
                            continue;
                        }
                        unsigned nid = raw_nodes[ri];
                        if (scratch.is_visited(nid)) {
                            continue;
                        }
                        neighbours.push_back({T(0), nid});
                        neighbour_prov.push_back(kProvRescue);
                        ++rescue_added;
                    }
                }
                if (beam_stats != nullptr && rescue_added > 0) {
                    beam_stats->rescue_candidates_added += rescue_added;
                }
                rescue_remaining = 0;
            }

            // Record yield trace: raw and selected counts
            round_raw_count = static_cast<unsigned>(raw_nodes.size());
            round_selected_count = static_cast<unsigned>(neighbours.size());

            // Post-arrival eval cap: truncate neighbours before distance eval
            // to reduce DCO in post-arrival phase
            if (gt_hit_recorded) {
                // P10-2/3: Adaptive early-stop and rank-aware tail pruning
                // Priority: ES-3 (stable) > ES-2 (stall decay) > ES-1 (adaptive)
                // > P10-3 (rank-aware) > existing per-lane/uniform cap
                unsigned pa_effective_cap = static_cast<unsigned>(neighbours.size());

                // ES-3: best_dist stable → aggressive cap
                if (pa_stable_cap > 0 && best_dist_flat_rounds >= pa_stable_threshold) {
                    pa_effective_cap = std::min(pa_effective_cap, pa_stable_cap);
                }

                // ES-2: consecutive empty rounds → progressive decay
                if (pa_stall_threshold > 0 && pa_stall >= pa_stall_threshold) {
                    unsigned excess = pa_stall - pa_stall_threshold;
                    unsigned decayed = static_cast<unsigned>(
                        neighbours.size() / std::pow(
                            static_cast<double>(pa_stall_decay),
                            static_cast<double>(excess)));
                    decayed = std::max(decayed, pa_stall_floor);
                    pa_effective_cap = std::min(pa_effective_cap, decayed);
                }

                // ES-1: after N empty rounds → fixed cap
                if (pa_adaptive_cap > 0 && pa_stall >= pa_adaptive_window) {
                    pa_effective_cap = std::min(pa_effective_cap, pa_adaptive_cap);
                }

                // P10-3: Rank-aware tail pruning
                if (pa_rank_head_keep > 0
                    && neighbours.size() > pa_rank_head_keep) {
                    // Check if recent post-arrival rounds had insertions
                    bool pa_recent_insertion = false;
                    unsigned pa_win = std::min(
                        static_cast<unsigned>(pa_insertion_hist.size()),
                        pa_rank_stale_window);
                    for (size_t hi = pa_insertion_hist.size() - pa_win;
                         hi < pa_insertion_hist.size(); ++hi) {
                        if (pa_insertion_hist[hi] > 0) {
                            pa_recent_insertion = true;
                            break;
                        }
                    }
                    if (!pa_recent_insertion) {
                        pa_effective_cap = std::min(
                            pa_effective_cap, pa_rank_head_keep);
                    }
                }

                // Apply the effective cap from adaptive mechanisms
                if (pa_effective_cap < neighbours.size()) {
                    if (neighbour_prov.size() > pa_effective_cap) {
                        neighbour_prov.resize(pa_effective_cap);
                    }
                    neighbours.resize(pa_effective_cap);
                    round_selected_count = pa_effective_cap;
                }

                bool any_lane_cap = (post_arrival_core_cap > 0
                    || post_arrival_bridge_cap > 0
                    || post_arrival_prefix_cap > 0);
                if (any_lane_cap && neighbour_prov.size() == neighbours.size()) {
                    // Per-lane cap: count per type, keep up to cap per type
                    unsigned core_count = 0, bridge_count = 0, prefix_count = 0;
                    size_t write = 0;
                    for (size_t i = 0; i < neighbours.size(); ++i) {
                        uint8_t prov = neighbour_prov[i];
                        bool keep = true;
                        if (prov == static_cast<uint8_t>(kProvCore)) {
                            if (post_arrival_core_cap > 0
                                && core_count >= post_arrival_core_cap) {
                                keep = false;
                            } else {
                                ++core_count;
                            }
                        } else if (prov == static_cast<uint8_t>(kProvBridge)
                            || prov == static_cast<uint8_t>(kProvDeepBridge)) {
                            if (post_arrival_bridge_cap > 0
                                && bridge_count >= post_arrival_bridge_cap) {
                                keep = false;
                            } else {
                                ++bridge_count;
                            }
                        } else if (prov == static_cast<uint8_t>(kProvPrefix)) {
                            if (post_arrival_prefix_cap > 0
                                && prefix_count >= post_arrival_prefix_cap) {
                                keep = false;
                            } else {
                                ++prefix_count;
                            }
                        }
                        if (keep) {
                            if (write != i) {
                                neighbours[write] = std::move(neighbours[i]);
                                neighbour_prov[write] = neighbour_prov[i];
                            }
                            ++write;
                        }
                    }
                    neighbours.resize(write);
                    neighbour_prov.resize(write);
                    round_selected_count = static_cast<unsigned>(write);
                } else if (post_arrival_eval_cap > 0
                    && neighbours.size() > post_arrival_eval_cap) {
                    // Uniform cap fallback
                    if (neighbour_prov.size() > post_arrival_eval_cap) {
                        neighbour_prov.resize(post_arrival_eval_cap);
                    }
                    neighbours.resize(post_arrival_eval_cap);
                    round_selected_count = post_arrival_eval_cap;
                }
            }

            if (neighbours.empty()) {
                continue;
            }

            bool improved = false;
            unsigned insertions_this_round = 0;

            // P11-2: Head-gated tail unlock (post-arrival only)
            const bool do_head_gate = (gt_hit_recorded
                && pa_head_gate_keep > 0
                && neighbours.size() > static_cast<size_t>(pa_head_gate_keep));

            if (do_head_gate) {
                // === Two-phase evaluation: head first, tail conditionally ===
                size_t hk = static_cast<size_t>(pa_head_gate_keep);

                // Helper: process a batch of evaluated neighbors
                auto process_batch = [&](
                    std::vector<std::pair<T, unsigned>>& batch,
                    const std::vector<uint8_t>& batch_prov,
                    size_t rank_off, bool& bat_improved) {
                    for (size_t ni = 0; ni < batch.size(); ++ni) {
                        const auto& [dist, nto] = batch[ni];
                        scratch.mark_visited(nto, dist);
                        const bool is_rescue =
                            (rescue_added > 0 && (rank_off + ni) >= rescue_offset);
                        if (is_rescue && beam_stats != nullptr) {
                            beam_stats->rescue_candidates_evaluated++;
                        }
                        const bool ni_prior = (insertions_this_round > 0);
                        bool ni_ins = false, ni_imp = false;
                        if (dist < candidates.back().first) {
                            ni_ins = true;
                            if (beam_stats != nullptr) {
                                beam_stats->beam_insertions++;
                                if (is_rescue) {
                                    beam_stats->rescue_candidates_inserted++;
                                }
                            }
                            insertions_this_round++;
                            diag_cumulative_insertions++;
                            candidates.pop_back();
                            auto it = std::partition_point(
                                candidates.begin(), candidates.end(),
                                [&](const auto& a) { return a.first < dist; });
                            const int pos =
                                static_cast<int>(it - candidates.begin());
                            if (beam_stats != nullptr && pos > 0 &&
                                pos - 1 < uid) {
                                beam_stats->beam_rewinds++;
                            }
                            uid = std::min(uid, pos - 1);
                            candidates.insert(it, {dist, nto + offset});
                            if (dist < candidates.front().first) {
                                bat_improved = true;
                                ni_imp = true;
                                round_improve_count++;
                            }
                        }
                        if (eval_prov_out != nullptr) {
                            EvalProvRecord rec;
                            rec.expansion_round = expansion_round;
                            rec.current_node = current_node;
                            rec.neighbor_node = nto;
                            rec.edge_category =
                                (ni < batch_prov.size())
                                    ? batch_prov[ni]
                                    : static_cast<uint8_t>(kProvOther);
                            rec.selected_rank =
                                static_cast<unsigned>(rank_off + ni);
                            rec.prior_insertion = ni_prior;
                            rec.was_inserted = ni_ins;
                            rec.improved_best = ni_imp;
                            rec.is_post_arrival = true;  // always post-arrival
                            rec.dist = dist;
                            eval_prov_out->push_back(rec);
                        }
                        // GT hit tracking
                        if (gt_set_ptr != nullptr &&
                            gt_set_ptr->count(nto) > 0) {
                            if (!gt_hit_recorded && beam_stats != nullptr) {
                                gt_hit_recorded = true;
                                beam_stats->first_gt_hit_expansion =
                                    beam_stats->expanded_nodes;
                                beam_stats->dco_at_first_gt_hit =
                                    beam_stats->distance_computations;
                            }
                            if (gt_provenance != nullptr) {
                                GtHitProv grec;
                                grec.gt_node_id = nto;
                                grec.expansion_round =
                                    beam_stats
                                        ? beam_stats->expanded_nodes
                                        : expansion_round;
                                grec.edge_category = static_cast<uint8_t>(
                                    (ni < batch_prov.size())
                                        ? batch_prov[ni]
                                        : static_cast<uint8_t>(kProvOther));
                                grec.edge_raw_position =
                                    std::numeric_limits<unsigned>::max();
                                grec.edge_gap = static_cast<unsigned>(
                                    std::abs(static_cast<long long>(nto) -
                                             static_cast<long long>(
                                                 current_node)));
                                grec.is_post_arrival = gt_hit_recorded;
                                grec.current_node = current_node;
                                gt_provenance->push_back(grec);
                            }
                            if (!gt_hit_recorded) {
                                gt_hit_recorded = true;
                            }
                        }
                    }
                };

                // --- Phase 1: Evaluate head ---
                std::vector<std::pair<T, unsigned>> head_batch(
                    neighbours.begin(), neighbours.begin() + hk);
                std::vector<uint8_t> head_prov(
                    neighbour_prov.begin(), neighbour_prov.begin() + hk);

                if (beam_stats != nullptr) {
                    beam_stats->in_range_neighbors_evaluated += hk;
                    beam_stats->distance_computations += hk;
                    if (beam_stats->fallback_triggered) {
                        beam_stats->post_trigger_dco += hk;
                    }
                }
                dataset.dist_all_into(goal, head_batch);
                process_batch(head_batch, head_prov, 0, improved);

                // --- Phase 2: Conditional tail evaluation ---
                if (insertions_this_round > 0) {
                    // Head had insertion → full tail
                    size_t tail_sz = neighbours.size() - hk;
                    std::vector<std::pair<T, unsigned>> tail_batch(
                        neighbours.begin() + hk, neighbours.end());
                    std::vector<uint8_t> tail_prov(
                        neighbour_prov.begin() + hk, neighbour_prov.end());

                    if (beam_stats != nullptr) {
                        beam_stats->in_range_neighbors_evaluated += tail_sz;
                        beam_stats->distance_computations += tail_sz;
                        if (beam_stats->fallback_triggered) {
                            beam_stats->post_trigger_dco += tail_sz;
                        }
                    }
                    dataset.dist_all_into(goal, tail_batch);
                    bool tail_improved = false;
                    process_batch(tail_batch, tail_prov, hk, tail_improved);
                    if (tail_improved) improved = true;
                } else if (pa_head_gate_tail_budget > 0) {
                    // No head insertion → limited tail
                    size_t tb = std::min(
                        static_cast<size_t>(pa_head_gate_tail_budget),
                        neighbours.size() - hk);
                    std::vector<std::pair<T, unsigned>> tail_batch(
                        neighbours.begin() + hk,
                        neighbours.begin() + hk + tb);
                    std::vector<uint8_t> tail_prov(
                        neighbour_prov.begin() + hk,
                        neighbour_prov.begin() + hk + tb);

                    if (beam_stats != nullptr) {
                        beam_stats->in_range_neighbors_evaluated += tb;
                        beam_stats->distance_computations += tb;
                        if (beam_stats->fallback_triggered) {
                            beam_stats->post_trigger_dco += tb;
                        }
                    }
                    dataset.dist_all_into(goal, tail_batch);
                    bool tail_improved = false;
                    process_batch(tail_batch, tail_prov, hk, tail_improved);
                    if (tail_improved) improved = true;
                }
                // else: no tail evaluation (DCO saved!)

            } else {
                // === Normal path: evaluate all neighbors at once ===
                if (beam_stats != nullptr) {
                    beam_stats->in_range_neighbors_evaluated +=
                        static_cast<std::uint64_t>(neighbours.size());
                    beam_stats->distance_computations +=
                        static_cast<std::uint64_t>(neighbours.size());
                    if (beam_stats->fallback_triggered) {
                        beam_stats->post_trigger_dco +=
                            static_cast<std::uint64_t>(neighbours.size());
                    }
                }
                dataset.dist_all_into(goal, neighbours);
                for (size_t ni = 0; ni < neighbours.size(); ++ni) {
                    const auto& [dist, nto] = neighbours[ni];
                    scratch.mark_visited(nto, dist);
                    const bool is_rescue =
                        (rescue_added > 0 && ni >= rescue_offset);
                    if (is_rescue && beam_stats != nullptr) {
                        beam_stats->rescue_candidates_evaluated++;
                    }
                    const bool ni_prior_insertion =
                        (insertions_this_round > 0);
                    bool ni_inserted = false;
                    bool ni_improved = false;
                    if (dist < candidates.back().first) {
                        ni_inserted = true;
                        if (beam_stats != nullptr) {
                            beam_stats->beam_insertions++;
                            if (is_rescue) {
                                beam_stats->rescue_candidates_inserted++;
                            }
                        }
                        insertions_this_round++;
                        diag_cumulative_insertions++;
                        candidates.pop_back();
                        auto it = std::partition_point(
                            candidates.begin(), candidates.end(),
                            [&](const auto& a) { return a.first < dist; });
                        const int pos =
                            static_cast<int>(it - candidates.begin());
                        if (beam_stats != nullptr && pos > 0 &&
                            pos - 1 < uid) {
                            beam_stats->beam_rewinds++;
                        }
                        uid = std::min(uid, pos - 1);
                        candidates.insert(it, {dist, nto + offset});
                        if (dist < candidates.front().first) {
                            improved = true;
                            ni_improved = true;
                            round_improve_count++;
                        }
                    }
                    if (eval_prov_out != nullptr) {
                        EvalProvRecord rec;
                        rec.expansion_round = expansion_round;
                        rec.current_node = current_node;
                        rec.neighbor_node = nto;
                        rec.edge_category = (ni < neighbour_prov.size())
                            ? neighbour_prov[ni] : kProvOther;
                        rec.selected_rank = static_cast<unsigned>(ni);
                        rec.prior_insertion = ni_prior_insertion;
                        rec.was_inserted = ni_inserted;
                        rec.improved_best = ni_improved;
                        rec.is_post_arrival = gt_hit_recorded;
                        rec.dist = dist;
                        eval_prov_out->push_back(rec);
                    }
                }

                // GT hit tracking (only when gt_set_ptr is provided)
                if (gt_set_ptr != nullptr) {
                    for (size_t ni = 0; ni < neighbours.size(); ++ni) {
                        const auto& [dist, nto] = neighbours[ni];
                        if (gt_set_ptr->count(nto) > 0) {
                            if (!gt_hit_recorded && beam_stats != nullptr) {
                                gt_hit_recorded = true;
                                beam_stats->first_gt_hit_expansion =
                                    static_cast<unsigned>(
                                        beam_stats->expanded_nodes);
                                beam_stats->dco_at_first_gt_hit =
                                    beam_stats->distance_computations;
                            }
                            if (gt_provenance != nullptr) {
                                GtHitProv rec;
                                rec.gt_node_id = nto;
                                rec.expansion_round =
                                    static_cast<unsigned>(
                                        beam_stats != nullptr
                                            ? beam_stats->expanded_nodes
                                            : expansion_round);
                                rec.edge_category =
                                    static_cast<uint8_t>(
                                        (ni < neighbour_prov.size())
                                            ? neighbour_prov[ni]
                                            : kProvOther);
                                rec.edge_raw_position =
                                    std::numeric_limits<unsigned>::max();
                                rec.edge_gap = static_cast<unsigned>(std::abs(
                                    static_cast<long long>(nto) -
                                    static_cast<long long>(current_node)));
                                rec.is_post_arrival = gt_hit_recorded;
                                rec.current_node = current_node;
                                gt_provenance->push_back(rec);
                            }
                            if (!gt_hit_recorded) {
                                gt_hit_recorded = true;
                            }
                        }
                    }
                }
            }

            // Update composite trigger runtime state
            insertion_hist.push_back(insertions_this_round);
            insertion_hist_sum += insertions_this_round;
            if (insertion_hist.size() > trigger_recent_window) {
                insertion_hist_sum -= insertion_hist.front();
                insertion_hist.pop_front();
            }

            // P10-2/3: Update post-arrival stall tracking
            if (gt_hit_recorded) {
                if (insertions_this_round > 0) {
                    pa_stall = 0;
                } else {
                    pa_stall++;
                }
                pa_insertion_hist.push_back(insertions_this_round);
                constexpr unsigned kPaHistMax = 32;
                if (pa_insertion_hist.size() > kPaHistMax) {
                    pa_insertion_hist.pop_front();
                }
            }

            double current_best = candidates.empty()
                                      ? 1e100
                                      : static_cast<double>(candidates[0].first);
            if (current_best < best_dist_ever - 1e-9) {
                best_dist_ever = current_best;
                best_dist_flat_rounds = 0;
            } else {
                best_dist_flat_rounds++;
            }

            if (improved) {
                stall_rounds = 0;
            } else {
                stall_rounds++;
                if (effective_early_stop_rounds > 0 &&
                    stall_rounds >= effective_early_stop_rounds &&
                    uid >= static_cast<int>(beam_size / 3)) {
                    if (beam_stats != nullptr) {
                        beam_stats->early_stop_triggered = true;
                    }
                    break;
                }
            }

            // --- Diagnostic snapshot (beam_search_range only) ---
            if (diag_ptr != nullptr && diag_ptr->enabled) {
                BeamSearchDiag::RoundSnapshot snap;
                snap.round = static_cast<unsigned>(diag_ptr->rounds.size());
                snap.expanded_node = current_node;
                snap.insertions = insertions_this_round;
                snap.cumulative_insertions = diag_cumulative_insertions;
                snap.best_dist = candidates.empty()
                                     ? 1e100
                                     : static_cast<double>(
                                           candidates[0].first);
                snap.best_dist_ever = best_dist_ever;
                snap.stall_rounds = stall_rounds;

                // Beam label span & effective size
                unsigned min_id = std::numeric_limits<unsigned>::max();
                unsigned max_id = 0;
                unsigned eff_size = 0;
                for (const auto& [d, nid_raw] : candidates) {
                    unsigned nid = (nid_raw >= offset) ? nid_raw - offset
                                                       : nid_raw;
                    if (d < 1e99) {
                        eff_size++;
                        min_id = std::min(min_id, nid);
                        max_id = std::max(max_id, nid);
                    }
                }
                snap.beam_size_effective = eff_size;
                snap.beam_label_span = (eff_size > 0) ? max_id - min_id : 0;

                // Beam bridge/core split
                unsigned bridge_count = 0;
                unsigned core_count = 0;
                if (eff_size > 2) {
                    std::vector<unsigned> beam_ids;
                    beam_ids.reserve(eff_size);
                    for (const auto& [d, nid_raw] : candidates) {
                        unsigned nid = (nid_raw >= offset) ? nid_raw - offset
                                                           : nid_raw;
                        if (d < 1e99) {
                            beam_ids.push_back(nid);
                        }
                    }
                    std::vector<unsigned> gaps;
                    gaps.reserve(beam_ids.size() - 1);
                    for (size_t gi = 1; gi < beam_ids.size(); ++gi) {
                        gaps.push_back(
                            beam_ids[gi] > beam_ids[gi - 1]
                                ? beam_ids[gi] - beam_ids[gi - 1]
                                : beam_ids[gi - 1] - beam_ids[gi]);
                    }
                    if (!gaps.empty()) {
                        std::ranges::sort(gaps);
                        unsigned median_gap = gaps[gaps.size() / 2];
                        for (auto g : gaps) {
                            if (g > median_gap) {
                                bridge_count++;
                            } else {
                                core_count++;
                            }
                        }
                    }
                }
                snap.beam_bridge_count = bridge_count;
                snap.beam_core_count = core_count;

                // Yield trace
                snap.raw_count = round_raw_count;
                snap.selected_count = round_selected_count;
                snap.range_filtered = round_range_filtered;
                snap.dist_eval_count = round_selected_count;
                snap.improve_count = round_improve_count;

                diag_ptr->rounds.push_back(snap);
            }
        }

        if (candidates_ptr != nullptr) {
            auto& c = *candidates_ptr;
            c.reserve(c.size() + visited_nodes.size());
            for (const auto& [id, dis] : visited_nodes) {
                c.push_back({dis, id});
            }
        }

        if (beam_stats != nullptr) {
            beam_stats->visited_nodes_count =
                static_cast<std::uint64_t>(visited_nodes.size());
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
    static unsigned normalize_edge_pick_policy(unsigned p) {
        return (p <= 3) ? p : 0;
    }

    static void append_even_sample(const std::vector<unsigned>& src,
                                   size_t begin,
                                   size_t end,
                                   size_t want,
                                   std::vector<unsigned>& out,
                                   const std::vector<char>& used) {
        if (want == 0 || begin >= end) {
            return;
        }
        const size_t n = end - begin;
        if (want >= n) {
            for (size_t i = begin; i < end; ++i) {
                if (!used[i]) {
                    out.push_back(src[i]);
                }
            }
            return;
        }
        size_t last_idx = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < want; ++i) {
            const size_t rel = (want == 1) ? 0 : (i * (n - 1)) / (want - 1);
            const size_t idx = begin + rel;
            if (idx == last_idx || used[idx]) {
                continue;
            }
            last_idx = idx;
            out.push_back(src[idx]);
        }
    }

    bool is_reciprocal(unsigned current_node,
                       unsigned nid,
                       unsigned depth) const {
        unsigned seen = 0;
        for (const auto& x : graph.get_neighbours(nid)) {
            if (x.to == current_node) {
                return true;
            }
            ++seen;
            if (seen >= depth) {
                break;
            }
        }
        return false;
    }

    void select_from_raw(
        unsigned current_node,
        const std::vector<unsigned>& raw_nodes,
        unsigned local_budget,
        unsigned front_keep,
        unsigned policy_in,
        unsigned recip_depth,
        double core_ratio,
        unsigned bridge_quota_floor,
        unsigned suffix_promote,
        unsigned deep_bridge_floor,
        unsigned deep_bridge_pos_threshold,
        std::vector<std::pair<T, unsigned>>& neighbours,
        std::vector<uint8_t>* prov_out = nullptr) const {
        const unsigned policy = normalize_edge_pick_policy(policy_in);
        const size_t raw_n = raw_nodes.size();
        const size_t need = std::min<size_t>(local_budget, raw_n);
        if (need == 0) {
            return;
        }

        const size_t prefix_keep =
            std::min<size_t>(std::min<size_t>(front_keep, need), raw_n);

        // Policy 0: prefix + uniform tail (baseline behavior).
        if (policy == 0) {
            for (size_t i = 0; i < prefix_keep; ++i) {
                neighbours.push_back({T(0), raw_nodes[i]});
                if (prov_out) prov_out->push_back(kProvPrefix);
            }
            const size_t remain = need - neighbours.size();
            if (remain > 0) {
                const size_t tail_n = raw_n - prefix_keep;
                size_t last_idx = std::numeric_limits<size_t>::max();
                for (size_t i = 0; i < remain; ++i) {
                    const size_t rel =
                        (remain == 1) ? 0 : (i * (tail_n - 1)) / (remain - 1);
                    const size_t idx = prefix_keep + rel;
                    if (idx == last_idx) {
                        continue;
                    }
                    last_idx = idx;
                    neighbours.push_back({T(0), raw_nodes[idx]});
                    if (prov_out) prov_out->push_back(kProvOther);
                }
            }
            return;
        }

        // reusable buffers
        scratch_ids.clear();
        scratch_ids.reserve(need);
        scratch_used.assign(raw_n, 0);
        if (prov_out != nullptr) {
            scratch_prov.clear();
            scratch_prov.reserve(need);
        }

        auto add_by_index = [&](size_t idx, uint8_t prov = kProvOther) {
            if (idx >= raw_n || scratch_used[idx]) {
                return false;
            }
            scratch_used[idx] = 1;
            scratch_ids.push_back(raw_nodes[idx]);
            if (prov_out != nullptr) {
                scratch_prov.push_back(prov);
            }
            return true;
        };

        // keep a tiny adjacency prefix for stability
        for (size_t i = 0; i < prefix_keep; ++i) {
            add_by_index(i, kProvPrefix);
        }

        // Suffix promote: evenly sample from deep suffix positions
        // Takes suffix_promote edges from positions [prefix_keep*2, raw_n)
        // to bring deeply buried edges into the selection pool
        if (suffix_promote > 0 && scratch_ids.size() < need) {
            const size_t deep_begin = std::min(prefix_keep * 2, raw_n);
            const size_t deep_end = raw_n;
            if (deep_end > deep_begin) {
                const size_t want = std::min(
                    static_cast<size_t>(suffix_promote),
                    need - scratch_ids.size());
                size_t last_idx = std::numeric_limits<size_t>::max();
                const size_t n = deep_end - deep_begin;
                for (size_t i = 0; i < want; ++i) {
                    const size_t rel = (want == 1) ? 0 : (i * (n - 1)) / (want - 1);
                    const size_t idx = deep_begin + rel;
                    if (idx != last_idx) {
                        add_by_index(idx, kProvSuffixPromote);
                        last_idx = idx;
                    }
                }
            }
        }

        if (scratch_ids.size() >= need) {
            for (auto nid : scratch_ids) {
                neighbours.push_back({T(0), nid});
            }
            return;
        }

        if (policy == 1) {
            // side_balance: keep both directions + tail diversity.
            std::vector<size_t> left, right;
            left.reserve(raw_n);
            right.reserve(raw_n);
            for (size_t i = prefix_keep; i < raw_n; ++i) {
                if (raw_nodes[i] < current_node) {
                    left.push_back(i);
                } else {
                    right.push_back(i);
                }
            }
            size_t remain = need - scratch_ids.size();
            size_t lq = std::min(left.size(), remain / 2);
            size_t rq = std::min(right.size(), remain - lq);
            if (lq + rq < remain) {
                const size_t extra = remain - lq - rq;
                const size_t l_left = left.size() - lq;
                const size_t r_left = right.size() - rq;
                if (l_left >= r_left) {
                    lq += std::min(extra, l_left);
                    rq += (extra - std::min(extra, l_left));
                } else {
                    rq += std::min(extra, r_left);
                    lq += (extra - std::min(extra, r_left));
                }
            }
            auto pick_side = [&](const std::vector<size_t>& side,
                                 size_t quota) {
                if (quota == 0 || side.empty()) {
                    return;
                }
                const size_t near_keep = std::min<size_t>(
                    quota, std::max<size_t>(1, (quota * 3) / 5));
                size_t used = 0;
                for (size_t i = 0; i < near_keep && i < side.size(); ++i) {
                    if (add_by_index(side[i])) {
                        used++;
                    }
                }
                const size_t remain_q = quota - used;
                if (remain_q > 0 && side.size() > near_keep) {
                    const size_t n = side.size() - near_keep;
                    size_t last = std::numeric_limits<size_t>::max();
                    for (size_t i = 0; i < remain_q; ++i) {
                        const size_t rel = (remain_q == 1)
                                               ? 0
                                               : (i * (n - 1)) / (remain_q - 1);
                        const size_t idx = side[near_keep + rel];
                        if (idx == last) {
                            continue;
                        }
                        last = idx;
                        add_by_index(idx);
                    }
                }
            };
            pick_side(left, lq);
            pick_side(right, rq);
        } else {
            // Policies 2/3 need reciprocal flags and gap scores.
            std::vector<size_t> order(raw_n - prefix_keep);
            for (size_t i = prefix_keep; i < raw_n; ++i) {
                order[i - prefix_keep] = i;
            }
            std::vector<unsigned> gap(raw_n, 0);
            std::vector<char> recip(raw_n, 0);
            for (size_t i = prefix_keep; i < raw_n; ++i) {
                gap[i] = static_cast<unsigned>(
                    std::abs(static_cast<long long>(raw_nodes[i]) -
                             static_cast<long long>(current_node)));
                recip[i] =
                    is_reciprocal(current_node, raw_nodes[i], recip_depth) ? 1
                                                                           : 0;
            }

            if (policy == 2) {
                // reciprocal-first + bridge-gap
                std::ranges::sort(order, [&](size_t a, size_t b) {
                    if (recip[a] != recip[b]) {
                        return recip[a] > recip[b];
                    }
                    if (gap[a] != gap[b]) {
                        return gap[a] > gap[b];
                    }
                    return a < b;
                });
                for (auto idx : order) {
                    if (scratch_ids.size() >= need) {
                        break;
                    }
                    add_by_index(idx);
                }
            } else {
                // policy 3: corebridge
                const double cr = std::clamp(core_ratio, 0.1, 0.95);
                size_t remain = need - scratch_ids.size();

                // Reserve deep_bridge_floor slots upfront
                size_t dbf_reserved = 0;
                if (deep_bridge_floor > 0 && remain > deep_bridge_floor) {
                    dbf_reserved = std::min(
                        static_cast<size_t>(deep_bridge_floor), remain);
                    remain -= dbf_reserved;
                }

                size_t core_quota = static_cast<size_t>(
                    std::llround(static_cast<double>(remain) * cr));
                core_quota = std::min(remain, std::max<size_t>(1, core_quota));
                size_t bridge_quota = remain - core_quota;

                // Enforce bridge_quota_floor: shrink core if needed
                if (bridge_quota_floor > 0 && bridge_quota < bridge_quota_floor) {
                    if (remain >= bridge_quota_floor) {
                        bridge_quota = bridge_quota_floor;
                        core_quota = remain - bridge_quota;
                    }
                }

                auto core_order = order;
                std::ranges::sort(core_order, [&](size_t a, size_t b) {
                    if (recip[a] != recip[b]) {
                        return recip[a] > recip[b];
                    }
                    if (a != b) {
                        return a < b;  // keep near-prefix as core
                    }
                    return false;
                });
                for (auto idx : core_order) {
                    if (core_quota == 0) {
                        break;
                    }
                    if (add_by_index(idx, kProvCore)) {
                        core_quota--;
                    }
                }

                auto bridge_order = order;
                std::ranges::sort(bridge_order, [&](size_t a, size_t b) {
                    if (gap[a] != gap[b]) {
                        return gap[a] > gap[b];
                    }
                    if (recip[a] != recip[b]) {
                        return recip[a] > recip[b];
                    }
                    return a < b;
                });
                for (auto idx : bridge_order) {
                    if (bridge_quota == 0) {
                        break;
                    }
                    if (add_by_index(idx, kProvBridge)) {
                        bridge_quota--;
                    }
                }

                // Deep bridge floor: fill reserved slots with highest-gap
                // edges from deep positions (>deep_bridge_pos_threshold).
                if (dbf_reserved > 0) {
                    std::vector<size_t> deep_candidates;
                    for (size_t i = 0; i < order.size(); ++i) {
                        const size_t raw_idx = order[i];
                        if (raw_idx >= deep_bridge_pos_threshold
                            && !scratch_used[raw_idx]) {
                            deep_candidates.push_back(raw_idx);
                        }
                    }
                    std::ranges::sort(deep_candidates,
                        [&](size_t a, size_t b) {
                            return gap[a] > gap[b];
                        });
                    size_t dbf_remaining = dbf_reserved;
                    for (auto idx : deep_candidates) {
                        if (dbf_remaining == 0) break;
                        if (add_by_index(idx, kProvDeepBridge)) {
                            dbf_remaining--;
                        }
                    }
                }
            }
        }

        // fill remaining by original order
        for (size_t i = prefix_keep; i < raw_n && scratch_ids.size() < need;
             ++i) {
            add_by_index(i);
        }

        for (auto nid : scratch_ids) {
            neighbours.push_back({T(0), nid});
        }
        if (prov_out != nullptr) {
            for (auto tag : scratch_prov) {
                prov_out->push_back(tag);
            }
        }
    }

    const Vector::VectorList<T>& dataset;
    const G& graph;
    mutable std::vector<unsigned> scratch_ids;
    mutable std::vector<char> scratch_used;
    mutable std::vector<uint8_t> scratch_prov;  // provenance tags for scratch_ids
};

}  // namespace TDFANN::RNSG
