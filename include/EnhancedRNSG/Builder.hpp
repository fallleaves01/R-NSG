#pragma once

#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <RNSG/Builder.hpp>
#include <Utils/Threading.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>
#include <chrono>
#include <nlohmann/json.hpp>
#include <omp.h>

namespace TDFANN::EnhancedRNSG {

struct BuildOptions : public RNSG::BuildOptions {
    bool enable_centroid_seed_search = true;
    bool enable_reverse_refine = true;
    std::string reverse_refine_mode = "incoming";
    std::string seed_collect_mode = "beam";
    std::string seed_collect_policy = "discovered";
    std::string monotone_seed_policy = "far";
    unsigned monotone_seed_limit = 6;
    unsigned seed_collect_keep = 0;
    unsigned seed_collect_max_expand = 32;
    unsigned seed_batch_size = 4;
    unsigned seed_search_beam_size = 256;
    unsigned seed_search_knng_cap = 64;
    unsigned knng_degree_cap = 128;
    std::string candidate_merge_mode = "legacy";
    double core_ratio = 0.6;
    unsigned reverse_incoming_quota = 0;
    std::string reverse_incoming_policy = "dist";
    unsigned bridge_witness_reserve = 0;
    unsigned support_reserve = 0;
    std::string support_reserve_policy = "support";
    unsigned tail_reserve = 0;
    std::string role_select_policy = "off";
    unsigned role_pool_extra = 0;
    bool role_support_append = true;
    unsigned role_local_warmup = 8;
    unsigned role_mid_gap_min = 128;
    unsigned role_far_gap_min = 2048;
    double role_mid_ratio = 0.22;
    double role_far_ratio = 0.08;
    unsigned range_window_cap = 0;
    std::string build_profile_json;
};

template <typename T>
class Builder {
   public:
    explicit Builder(Vector::VectorList<T>& data) : vector_list(data) {}

    Graph::GraphIndex<std::monostate> nn_descent(unsigned k,
                                                 bool verbose = false) const {
        RNSG::Builder<T> base(vector_list);
        return base.nn_descent(k, verbose);
    }

    Graph::TDGraphIndexBase build(Graph::GraphLike auto&& knng,
                                  unsigned range_step, unsigned ef_max,
                                  const std::vector<std::uint64_t>& label,
                                  const BuildOptions& options = {});

   private:
    Vector::VectorList<T>& vector_list;
};

namespace detail {

struct alignas(64) ThreadBuildProfile {
    std::uint64_t first_knng_ns = 0;
    std::uint64_t first_window_ns = 0;
    std::uint64_t first_seed_ns = 0;
    std::uint64_t first_prune_ns = 0;
    std::uint64_t first_reorder_ns = 0;
    std::uint64_t reverse_collect_ns = 0;
    std::uint64_t reverse_prune_ns = 0;
    std::uint64_t reverse_reorder_ns = 0;

    std::uint64_t first_knng_candidates = 0;
    std::uint64_t first_window_attempted = 0;
    std::uint64_t first_window_inserted = 0;
    std::uint64_t first_bridge_candidates = 0;
    std::uint64_t first_pruned_candidates = 0;
    std::uint64_t reverse_core_candidates = 0;
    std::uint64_t reverse_window_attempted = 0;
    std::uint64_t reverse_window_inserted = 0;
    std::uint64_t reverse_bridge_candidates = 0;
    std::uint64_t reverse_pruned_candidates = 0;

    std::uint64_t seed_batches = 0;
    std::uint64_t seed_seeds = 0;
    std::uint64_t seed_expanded = 0;
    std::uint64_t seed_collected = 0;
    std::uint64_t seed_neighbor_scans = 0;
    std::uint64_t first_nodes = 0;
    std::uint64_t reverse_nodes = 0;
};

inline std::uint64_t ns_since(
    const std::chrono::steady_clock::time_point& start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

inline double ns_to_ms(std::uint64_t ns) { return ns / 1e6; }
inline double ns_to_us(std::uint64_t ns) { return ns / 1e3; }

template <typename T>
std::vector<int> build_prev_smaller(const std::vector<T>& vals) {
    std::vector<int> prev(vals.size(), -1);
    std::vector<unsigned> st;
    st.reserve(vals.size());
    for (unsigned i = 0; i < vals.size(); ++i) {
        while (!st.empty() && vals[st.back()] >= vals[i]) {
            st.pop_back();
        }
        prev[i] = st.empty() ? -1 : static_cast<int>(st.back());
        st.push_back(i);
    }
    return prev;
}

template <typename T>
std::vector<int> build_next_smaller(const std::vector<T>& vals) {
    std::vector<int> next(vals.size(), -1);
    std::vector<unsigned> st;
    st.reserve(vals.size());
    for (int i = static_cast<int>(vals.size()) - 1; i >= 0; --i) {
        while (!st.empty() && vals[st.back()] >= vals[static_cast<unsigned>(i)]) {
            st.pop_back();
        }
        next[static_cast<unsigned>(i)] =
            st.empty() ? -1 : static_cast<int>(st.back());
        st.push_back(static_cast<unsigned>(i));
    }
    return next;
}

template <typename KNNG>
std::vector<std::vector<unsigned>> build_sorted_knng(
    KNNG&& knng, const std::vector<unsigned>& index,
    const std::vector<unsigned>& pos, unsigned cap) {
    const unsigned n = static_cast<unsigned>(index.size());
    std::vector<std::vector<unsigned>> out(n);
#pragma omp parallel for schedule(dynamic)
    for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
        const unsigned i = static_cast<unsigned>(ii);
        auto& row = out[i];
        row.reserve(cap == 0 ? 64 : cap);
        phmap::flat_hash_set<unsigned> seen;
        seen.reserve(128);
        for (auto raw : knng.get_neighbours_id(index[i])) {
            const unsigned mapped = pos[raw];
            if (mapped == i) {
                continue;
            }
            if (!seen.insert(mapped).second) {
                continue;
            }
            row.push_back(mapped);
            if (cap > 0 && row.size() >= cap) {
                break;
            }
        }
    }
    return out;
}

template <typename T>
std::vector<unsigned> collect_from_seed(
    const Vector::VectorList<T>& dataset,
    const std::vector<std::vector<unsigned>>& knng_adj, unsigned target,
    unsigned seed, unsigned max_keep, unsigned max_expand) {
    if (seed == target || seed >= dataset.size() || max_keep == 0 ||
        max_expand == 0) {
        return {};
    }

    using QItem = std::pair<T, unsigned>;
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;
    phmap::flat_hash_set<unsigned> seen;
    seen.reserve(max_keep * 8 + 16);
    seen.insert(seed);
    pq.push({dataset.dist(target, seed), seed});

    std::vector<unsigned> collected;
    collected.reserve(max_keep);
    unsigned expanded = 0;
    while (!pq.empty() && expanded < max_expand &&
           collected.size() < max_keep) {
        auto [dist_u, u] = pq.top();
        (void)dist_u;
        pq.pop();
        collected.push_back(u);
        expanded++;
        for (auto nb : knng_adj[u]) {
            if (nb == target) {
                continue;
            }
            if (!seen.insert(nb).second) {
                continue;
            }
            pq.push({dataset.dist(target, nb), nb});
        }
    }
    return collected;
}

template <typename T>
struct SeedSearchResult {
    std::vector<unsigned> nodes;
    unsigned expanded = 0;
    unsigned accepted = 0;
};

enum class SeedCollectPolicy : unsigned char {
    Expanded,
    Discovered,
    Evaluated,
};

inline SeedCollectPolicy parse_seed_collect_policy(std::string_view policy) {
    if (policy == "expanded") {
        return SeedCollectPolicy::Expanded;
    }
    if (policy == "evaluated") {
        return SeedCollectPolicy::Evaluated;
    }
    return SeedCollectPolicy::Discovered;
}

template <typename T>
SeedSearchResult<T> collect_from_seed_batch_pq(
    const Vector::VectorList<T>& dataset,
    const std::vector<std::vector<unsigned>>& knng_adj, unsigned target,
    const std::vector<unsigned>& seeds, unsigned max_keep,
    unsigned max_expand, unsigned knng_degree_cap = 0,
    SeedCollectPolicy collect_policy = SeedCollectPolicy::Discovered,
    std::uint64_t* neighbor_scans = nullptr) {
    SeedSearchResult<T> result;
    if (seeds.empty()) {
        return {};
    }
    const bool unlimited_expand = max_expand == 0;

    using QItem = std::pair<T, unsigned>;
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;
    phmap::flat_hash_set<unsigned> seen;
    const size_t reserve_hint =
        max_keep > 0 ? max_keep : (max_expand > 0 ? max_expand : 1024u);
    seen.reserve(reserve_hint * 8 + seeds.size() * 4 + 16);

    auto collect_node = [&](unsigned v) {
        if (v == target) {
            return;
        }
        if (max_keep > 0 && result.nodes.size() >= max_keep) {
            return;
        }
        result.nodes.push_back(v);
    };

    for (auto seed : seeds) {
        if (seed == target || seed >= dataset.size()) {
            continue;
        }
        if (!seen.insert(seed).second) {
            continue;
        }
        pq.push({dataset.dist(target, seed), seed});
        result.accepted++;
        if (collect_policy == SeedCollectPolicy::Discovered ||
            collect_policy == SeedCollectPolicy::Evaluated) {
            collect_node(seed);
        }
    }
    if (pq.empty()) {
        return result;
    }

    unsigned expanded = 0;
    while (!pq.empty() && (unlimited_expand || expanded < max_expand)) {
        auto [dist_u, u] = pq.top();
        (void)dist_u;
        pq.pop();
        if (collect_policy == SeedCollectPolicy::Expanded) {
            collect_node(u);
        }
        expanded++;
        result.expanded = expanded;
        const auto& row = knng_adj[u];
        const size_t degree =
            (knng_degree_cap == 0)
                ? row.size()
                : std::min(row.size(), static_cast<size_t>(knng_degree_cap));
        if (neighbor_scans != nullptr) {
            *neighbor_scans += degree;
        }
        for (size_t idx = 0; idx < degree; ++idx) {
            auto nb = row[idx];
            if (nb == target) {
                continue;
            }
            if (!seen.insert(nb).second) {
                continue;
            }
            const auto d = dataset.dist(target, nb);
            if (collect_policy == SeedCollectPolicy::Evaluated) {
                collect_node(nb);
            }
            pq.push({d, nb});
            result.accepted++;
            if (collect_policy == SeedCollectPolicy::Discovered) {
                collect_node(nb);
            }
        }
    }
    return result;
}

template <typename T>
struct SeedCollectScratch {
    std::vector<std::uint32_t> visit_stamp;
    std::uint32_t visit_epoch = 1;
    std::vector<std::pair<T, unsigned>> candidates;
    std::vector<std::pair<T, unsigned>> neighbours;
    std::vector<unsigned> raw_nodes;

    size_t visit_stamp_size = 0;
    size_t candidates_reserved = 0;
    size_t neighbours_reserved = 0;
    size_t raw_reserved = 0;

    void ensure(size_t dataset_size, size_t candidate_need,
                size_t neighbour_need) {
        if (dataset_size > visit_stamp_size) {
            visit_stamp.resize(dataset_size, 0);
            visit_stamp_size = dataset_size;
        }
        if (candidate_need > candidates_reserved) {
            candidates.reserve(candidate_need);
            candidates_reserved = candidate_need;
        }
        if (neighbour_need > neighbours_reserved) {
            neighbours.reserve(neighbour_need);
            neighbours_reserved = neighbour_need;
        }
        if (neighbour_need > raw_reserved) {
            raw_nodes.reserve(neighbour_need);
            raw_reserved = neighbour_need;
        }
    }

    void next_round() {
        visit_epoch++;
        if (visit_epoch == 0) {
            std::fill(visit_stamp.begin(), visit_stamp.end(), 0);
            visit_epoch = 1;
        }
        candidates.clear();
        neighbours.clear();
        raw_nodes.clear();
    }

    bool try_visit(unsigned id) {
        if (id >= visit_stamp.size() || visit_stamp[id] == visit_epoch) {
            return false;
        }
        visit_stamp[id] = visit_epoch;
        return true;
    }
};

template <typename T>
SeedSearchResult<T> collect_from_seed_batch_beam(
    const Vector::VectorList<T>& dataset,
    const std::vector<std::vector<unsigned>>& knng_adj, unsigned target,
    const std::vector<unsigned>& seeds, unsigned max_keep,
    unsigned max_expand, unsigned beam_size, unsigned knng_degree_cap = 0,
    SeedCollectPolicy collect_policy = SeedCollectPolicy::Discovered,
    std::uint64_t* neighbor_scans = nullptr) {
    SeedSearchResult<T> result;
    if (seeds.empty()) {
        return {};
    }
    const bool unlimited_expand = max_expand == 0;
    const unsigned effective_beam =
        beam_size == 0
            ? std::max(1u, max_expand == 0 ? 1024u : max_expand)
            : beam_size;
    const unsigned offset = static_cast<unsigned>(dataset.size());

    thread_local SeedCollectScratch<T> scratch;
    const size_t neighbour_need = std::max<size_t>(
        16, knng_degree_cap == 0 ? 128 : static_cast<size_t>(knng_degree_cap));
    const size_t beam_reserve =
        std::max<size_t>(seeds.size() + 8,
                         static_cast<size_t>(effective_beam) + seeds.size() + 1);
    const size_t candidate_need =
        std::max<size_t>(beam_reserve, max_keep > 0 ? max_keep : 0u);
    scratch.ensure(dataset.size(), candidate_need, neighbour_need);
    scratch.next_round();

    auto& candidates = scratch.candidates;
    auto& neighbours = scratch.neighbours;
    auto& raw_nodes = scratch.raw_nodes;

    auto collect_node = [&](unsigned v) {
        if (v == target) {
            return;
        }
        if (max_keep > 0 && result.nodes.size() >= max_keep) {
            return;
        }
        result.nodes.push_back(v);
    };

    for (auto seed : seeds) {
        if (seed == target || seed >= dataset.size()) {
            continue;
        }
        if (!scratch.try_visit(seed)) {
            continue;
        }
        neighbours.push_back({T(0), seed});
    }
    if (neighbours.empty()) {
        return result;
    }

    dataset.dist_all_into(target, neighbours);
    std::ranges::sort(neighbours);
    for (const auto& item : neighbours) {
        candidates.push_back(item);
        result.accepted++;
        if (collect_policy == SeedCollectPolicy::Discovered ||
            collect_policy == SeedCollectPolicy::Evaluated) {
            collect_node(item.second);
        }
    }
    if (candidates.empty()) {
        return result;
    }
    for (auto& [dist, id] : candidates) {
        (void)dist;
        id += offset;
    }
    if (candidates.size() < effective_beam) {
        candidates.resize(effective_beam,
                          {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > effective_beam) {
        candidates.resize(effective_beam);
    }

    unsigned expanded = 0;

    for (int uid = 0; uid < static_cast<int>(effective_beam); ++uid) {
        if (!unlimited_expand && expanded >= max_expand) {
            break;
        }
        if (candidates[static_cast<size_t>(uid)].second < offset) {
            continue;
        }
        candidates[static_cast<size_t>(uid)].second -= offset;
        const unsigned u = candidates[uid].second;
        if (collect_policy == SeedCollectPolicy::Expanded) {
            collect_node(u);
        }
        expanded++;
        result.expanded = expanded;

        const auto& row = knng_adj[u];
        const size_t degree =
            (knng_degree_cap == 0)
                ? row.size()
                : std::min(row.size(), static_cast<size_t>(knng_degree_cap));
        if (neighbor_scans != nullptr) {
            *neighbor_scans += degree;
        }

        raw_nodes.clear();
        for (size_t idx = 0; idx < degree; ++idx) {
            const auto nb = row[idx];
            if (nb == target) {
                continue;
            }
            if (!scratch.try_visit(nb)) {
                continue;
            }
            raw_nodes.push_back(nb);
        }
        if (raw_nodes.empty()) {
            continue;
        }

        neighbours.clear();
        for (auto nb : raw_nodes) {
            neighbours.push_back({T(0), nb});
        }
        dataset.dist_all_into(target, neighbours);
        if (collect_policy == SeedCollectPolicy::Evaluated) {
            for (const auto& item : neighbours) {
                collect_node(item.second);
            }
        }
        std::ranges::sort(neighbours);
        for (const auto& item : neighbours) {
            const auto& [dist, nto] = item;
            if (dist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < dist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {dist, nto + offset});
                result.accepted++;
                if (collect_policy == SeedCollectPolicy::Discovered) {
                    collect_node(nto);
                }
            }
        }
    }
    return result;
}

template <typename T>
std::vector<std::pair<T, unsigned>> sort_unique_pairs(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& ids);

template <typename T>
std::vector<unsigned> prune_candidates(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& ids, unsigned ef_max,
    bool side_split_pruning, bool use_mrng_pruning = false) {
    if (use_mrng_pruning || !side_split_pruning) {
        auto all = sort_unique_pairs(dataset, src, ids);
        auto final_edges = RNSG::prune(dataset, all, ef_max);
        if (final_edges.size() > ef_max) {
            final_edges.resize(ef_max);
        }

        std::vector<unsigned> out;
        out.reserve(final_edges.size());
        for (const auto& [d, v] : final_edges) {
            (void)d;
            out.push_back(v);
        }
        return out;
    }

    std::vector<unsigned> left_ids, right_ids;
    left_ids.reserve(ids.size());
    right_ids.reserve(ids.size());
    for (auto v : ids) {
        if (v == src) {
            continue;
        }
        if (v < src) {
            left_ids.push_back(v);
        } else if (v > src) {
            right_ids.push_back(v);
        }
    }

    // Match the original RNSG construction semantics:
    // keep left/right candidates in label-order proximity before applying
    // triangle pruning, so the induced subgraph inheritance is preserved.
    std::ranges::sort(left_ids, std::greater<unsigned>{});
    std::ranges::sort(right_ids);
    left_ids.erase(std::ranges::unique(left_ids).begin(), left_ids.end());
    right_ids.erase(std::ranges::unique(right_ids).begin(), right_ids.end());

    std::vector<std::pair<T, unsigned>> c_left, c_right;
    c_left.reserve(left_ids.size());
    c_right.reserve(right_ids.size());
    for (auto v : left_ids) {
        c_left.push_back({0, v});
    }
    for (auto v : right_ids) {
        c_right.push_back({0, v});
    }
    dataset.dist_all_into(src, c_left);
    dataset.dist_all_into(src, c_right);
    const T src_sqr = dataset[src].squaredNorm();
    for (auto& [d, v] : c_left) {
        (void)v;
        d += src_sqr;
    }
    for (auto& [d, v] : c_right) {
        (void)v;
        d += src_sqr;
    }

    std::vector<std::pair<T, unsigned>> final_edges;
    c_left = RNSG::prune(dataset, c_left, (ef_max + 1) / 2);
    c_right = RNSG::prune(dataset, c_right, (ef_max + 1) / 2);
    if (c_left.size() > ef_max / 2) {
        c_left.resize(ef_max / 2);
    }
    c_left.insert(c_left.end(), c_right.begin(),
                  c_right.begin() +
                      std::min(c_right.size(),
                               static_cast<size_t>(ef_max) - c_left.size()));
    std::ranges::sort(c_left, [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    final_edges = std::move(c_left);

    std::vector<unsigned> out;
    out.reserve(final_edges.size());
    for (const auto& [d, v] : final_edges) {
        (void)d;
        out.push_back(v);
    }
    return out;
}

template <typename T>
std::vector<std::pair<T, unsigned>> sort_unique_pairs(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& ids) {
    std::vector<std::pair<T, unsigned>> out;
    out.reserve(ids.size());
    phmap::flat_hash_set<unsigned> seen;
    seen.reserve(ids.size() * 2 + 8);
    for (auto v : ids) {
        if (v == src) {
            continue;
        }
        if (!seen.insert(v).second) {
            continue;
        }
        out.push_back({0, v});
    }
    dataset.dist_all_into(src, out);
    const T src_sqr = dataset[src].squaredNorm();
    for (auto& [d, v] : out) {
        (void)v;
        d += src_sqr;
    }
    std::ranges::sort(out, [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    return out;
}

inline std::string normalize_role_select_policy(std::string p) {
    for (auto& c : p) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (p == "off" || p == "roles") {
        return p;
    }
    return "off";
}

template <typename T>
struct RoleCand {
    T dist;
    unsigned v;
    unsigned gap;
    unsigned support;
};

template <typename T, typename Compare>
void take_role_balanced(const std::vector<RoleCand<T>>& left,
                        const std::vector<RoleCand<T>>& right,
                        size_t quota,
                        Compare&& cmp,
                        std::vector<unsigned>& out,
                        phmap::flat_hash_set<unsigned>& chosen) {
    if (quota == 0 || (left.empty() && right.empty())) {
        return;
    }
    std::vector<RoleCand<T>> l = left;
    std::vector<RoleCand<T>> r = right;
    std::ranges::sort(l, cmp);
    std::ranges::sort(r, cmp);

    size_t li = 0, ri = 0;
    int last_side = -1;
    while (out.size() < quota && (li < l.size() || ri < r.size())) {
        while (li < l.size() && chosen.contains(l[li].v)) {
            ++li;
        }
        while (ri < r.size() && chosen.contains(r[ri].v)) {
            ++ri;
        }
        if (li >= l.size() && ri >= r.size()) {
            break;
        }
        if (li >= l.size()) {
            chosen.insert(r[ri].v);
            out.push_back(r[ri].v);
            ++ri;
            last_side = 1;
            continue;
        }
        if (ri >= r.size()) {
            chosen.insert(l[li].v);
            out.push_back(l[li].v);
            ++li;
            last_side = 0;
            continue;
        }
        const bool take_left = (last_side == 1 || last_side == -1);
        if (take_left) {
            chosen.insert(l[li].v);
            out.push_back(l[li].v);
            ++li;
            last_side = 0;
        } else {
            chosen.insert(r[ri].v);
            out.push_back(r[ri].v);
            ++ri;
            last_side = 1;
        }
    }
}

template <typename T>
void append_role_pool_support(std::vector<unsigned>& ids,
                              const Vector::VectorList<T>& dataset,
                              unsigned src,
                              const phmap::flat_hash_map<unsigned, unsigned>& support,
                              unsigned reserve_budget,
                              unsigned min_gap) {
    if (reserve_budget == 0 || support.empty()) {
        return;
    }
    phmap::flat_hash_set<unsigned> chosen;
    chosen.reserve(ids.size() * 2 + reserve_budget * 2 + 8);
    for (auto v : ids) {
        chosen.insert(v);
    }

    std::vector<RoleCand<T>> left, right;
    left.reserve(support.size());
    right.reserve(support.size());
    for (const auto& [v, s] : support) {
        if (v == src || chosen.contains(v)) {
            continue;
        }
        const unsigned gap = static_cast<unsigned>(std::abs(
            static_cast<long long>(v) - static_cast<long long>(src)));
        if (gap < min_gap) {
            continue;
        }
        RoleCand<T> cand{dataset.dist(src, v), v, gap, s};
        if (v < src) {
            left.push_back(cand);
        } else {
            right.push_back(cand);
        }
    }
    auto cmp = [](const auto& a, const auto& b) {
        if (a.support != b.support) {
            return a.support > b.support;
        }
        if (a.gap != b.gap) {
            return a.gap > b.gap;
        }
        if (a.dist != b.dist) {
            return a.dist < b.dist;
        }
        return a.v < b.v;
    };
    const auto base = ids.size();
    take_role_balanced(left, right, base + reserve_budget, cmp, ids, chosen);
}

template <typename T>
std::vector<unsigned> select_role_quota_candidates(
    const Vector::VectorList<T>& dataset,
    unsigned src,
    const std::vector<unsigned>& ids,
    unsigned target_budget,
    const BuildOptions& options,
    const phmap::flat_hash_map<unsigned, unsigned>* support = nullptr) {
    if (target_budget == 0 || ids.size() <= target_budget) {
        return ids;
    }

    const unsigned mid_gap_min = std::max(1u, options.role_mid_gap_min);
    const unsigned far_gap_min =
        std::max(mid_gap_min + 1, options.role_far_gap_min);
    auto pairs = sort_unique_pairs(dataset, src, ids);
    const size_t target = std::min<size_t>(target_budget, pairs.size());
    const size_t warmup = std::min<size_t>(options.role_local_warmup, target);

    std::vector<unsigned> out;
    out.reserve(target);
    phmap::flat_hash_set<unsigned> chosen;
    chosen.reserve(target * 4 + 8);
    for (size_t i = 0; i < warmup; ++i) {
        out.push_back(pairs[i].second);
        chosen.insert(pairs[i].second);
    }

    std::vector<RoleCand<T>> local_left, local_right, mid_left, mid_right,
        far_left, far_right;
    local_left.reserve(pairs.size());
    local_right.reserve(pairs.size());
    mid_left.reserve(pairs.size());
    mid_right.reserve(pairs.size());
    far_left.reserve(pairs.size());
    far_right.reserve(pairs.size());

    for (size_t i = warmup; i < pairs.size(); ++i) {
        const auto& [dist, v] = pairs[i];
        const unsigned gap = static_cast<unsigned>(std::abs(
            static_cast<long long>(v) - static_cast<long long>(src)));
        const unsigned sup =
            (support != nullptr && support->contains(v)) ? support->at(v) : 0u;
        RoleCand<T> cand{dist, v, gap, sup};
        if (gap >= far_gap_min) {
            (v < src ? far_left : far_right).push_back(cand);
        } else if (gap >= mid_gap_min) {
            (v < src ? mid_left : mid_right).push_back(cand);
        } else {
            (v < src ? local_left : local_right).push_back(cand);
        }
    }

    const size_t remain = target - out.size();
    const size_t far_target = std::min<size_t>(
        far_left.size() + far_right.size(),
        std::llround(static_cast<double>(remain) * options.role_far_ratio));
    const size_t mid_target = std::min<size_t>(
        mid_left.size() + mid_right.size(),
        std::llround(static_cast<double>(remain) * options.role_mid_ratio));

    auto cmp_local = [](const auto& a, const auto& b) {
        if (a.dist != b.dist) {
            return a.dist < b.dist;
        }
        return a.v < b.v;
    };
    auto cmp_mid = [](const auto& a, const auto& b) {
        if (a.support != b.support) {
            return a.support > b.support;
        }
        if (a.dist != b.dist) {
            return a.dist < b.dist;
        }
        if (a.gap != b.gap) {
            return a.gap > b.gap;
        }
        return a.v < b.v;
    };
    auto cmp_far = [](const auto& a, const auto& b) {
        if (a.support != b.support) {
            return a.support > b.support;
        }
        if (a.gap != b.gap) {
            return a.gap > b.gap;
        }
        if (a.dist != b.dist) {
            return a.dist < b.dist;
        }
        return a.v < b.v;
    };

    const size_t out_after_warmup = out.size();
    take_role_balanced(local_left, local_right,
                       out_after_warmup + (remain - mid_target - far_target),
                       cmp_local, out, chosen);
    take_role_balanced(mid_left, mid_right, out.size() + mid_target, cmp_mid,
                       out, chosen);
    take_role_balanced(far_left, far_right, out.size() + far_target, cmp_far,
                       out, chosen);

    if (out.size() < target) {
        take_role_balanced(local_left, local_right, target, cmp_local, out,
                           chosen);
    }
    if (out.size() < target) {
        take_role_balanced(mid_left, mid_right, target, cmp_mid, out, chosen);
    }
    if (out.size() < target) {
        take_role_balanced(far_left, far_right, target, cmp_far, out, chosen);
    }
    return out;
}

template <typename T>
std::vector<std::pair<T, unsigned>> select_bucket_quota(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& core_ids, const std::vector<unsigned>& bridge_ids,
    unsigned budget, unsigned core_budget) {
    if (budget == 0) {
        return {};
    }

    auto core = sort_unique_pairs(dataset, src, core_ids);
    auto bridge = sort_unique_pairs(dataset, src, bridge_ids);
    std::vector<std::pair<T, unsigned>> selected;
    selected.reserve(budget);
    phmap::flat_hash_set<unsigned> used;
    used.reserve(budget * 4 + 8);

    auto take_from = [&](const std::vector<std::pair<T, unsigned>>& pool,
                         unsigned limit) {
        for (const auto& cand : pool) {
            if (selected.size() >= limit) {
                break;
            }
            if (used.contains(cand.second)) {
                continue;
            }
            if (!RNSG::check_valid(dataset, cand, selected)) {
                continue;
            }
            selected.push_back(cand);
            used.insert(cand.second);
        }
    };

    take_from(core, std::min(core_budget, budget));
    take_from(bridge, budget);
    take_from(core, budget);
    take_from(bridge, budget);
    return selected;
}

template <typename T>
std::vector<unsigned> select_quota_candidates(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& core_ids, const std::vector<unsigned>& bridge_ids,
    unsigned ef_max, bool side_split_pruning, double core_ratio) {
    core_ratio = std::clamp(core_ratio, 0.0, 1.0);
    if (ef_max == 0) {
        return {};
    }

    std::vector<std::pair<T, unsigned>> selected;
    selected.reserve(ef_max);
    phmap::flat_hash_set<unsigned> used;
    used.reserve(ef_max * 4 + 8);

    auto append_pairs = [&](const std::vector<std::pair<T, unsigned>>& pool) {
        for (const auto& cand : pool) {
            if (selected.size() >= ef_max) {
                break;
            }
            if (used.contains(cand.second)) {
                continue;
            }
            if (!RNSG::check_valid(dataset, cand, selected)) {
                continue;
            }
            selected.push_back(cand);
            used.insert(cand.second);
        }
    };

    auto fill_leftover = [&](const std::vector<unsigned>& ids) {
        auto pool = sort_unique_pairs(dataset, src, ids);
        append_pairs(pool);
    };

    if (side_split_pruning) {
        std::vector<unsigned> core_left, core_right, bridge_left, bridge_right;
        core_left.reserve(core_ids.size());
        core_right.reserve(core_ids.size());
        bridge_left.reserve(bridge_ids.size());
        bridge_right.reserve(bridge_ids.size());
        for (auto v : core_ids) {
            if (v < src) {
                core_left.push_back(v);
            } else if (v > src) {
                core_right.push_back(v);
            }
        }
        for (auto v : bridge_ids) {
            if (v < src) {
                bridge_left.push_back(v);
            } else if (v > src) {
                bridge_right.push_back(v);
            }
        }

        const unsigned left_budget = ef_max / 2;
        const unsigned right_budget = ef_max - left_budget;
        const unsigned left_core_budget = static_cast<unsigned>(
            std::llround(static_cast<double>(left_budget) * core_ratio));
        const unsigned right_core_budget = static_cast<unsigned>(
            std::llround(static_cast<double>(right_budget) * core_ratio));

        append_pairs(select_bucket_quota(dataset, src, core_left, bridge_left,
                                         left_budget, left_core_budget));
        append_pairs(select_bucket_quota(dataset, src, core_right, bridge_right,
                                         right_budget, right_core_budget));

        if (selected.size() < ef_max) {
            std::vector<unsigned> leftovers;
            leftovers.reserve(core_ids.size() + bridge_ids.size());
            leftovers.insert(leftovers.end(), core_ids.begin(), core_ids.end());
            leftovers.insert(leftovers.end(), bridge_ids.begin(), bridge_ids.end());
            fill_leftover(leftovers);
        }
    } else {
        const unsigned core_budget = static_cast<unsigned>(
            std::llround(static_cast<double>(ef_max) * core_ratio));
        append_pairs(select_bucket_quota(dataset, src, core_ids, bridge_ids,
                                         ef_max, core_budget));
        if (selected.size() < ef_max) {
            std::vector<unsigned> leftovers;
            leftovers.reserve(core_ids.size() + bridge_ids.size());
            leftovers.insert(leftovers.end(), core_ids.begin(), core_ids.end());
            leftovers.insert(leftovers.end(), bridge_ids.begin(), bridge_ids.end());
            fill_leftover(leftovers);
        }
    }

    std::vector<unsigned> out;
    out.reserve(selected.size());
    for (const auto& [d, v] : selected) {
        (void)d;
        out.push_back(v);
    }
    return out;
}

template <typename T>
void reorder_final(std::vector<unsigned>& ids, const Vector::VectorList<T>& dataset,
                   unsigned src, const BuildOptions& options) {
    std::vector<std::pair<T, unsigned>> edges;
    edges.reserve(ids.size());
    for (auto v : ids) {
        edges.push_back({0, v});
    }
    dataset.dist_all_into(src, edges);
    std::ranges::sort(edges, [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    RNSG::reorder_prefix_edges(edges, src,
                               static_cast<const RNSG::BuildOptions&>(options));
    ids.clear();
    ids.reserve(edges.size());
    for (const auto& [d, v] : edges) {
        (void)d;
        ids.push_back(v);
    }
}

template <typename T>
std::vector<unsigned> select_balanced_quota_by_dist(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& ids, unsigned quota) {
    if (quota == 0 || ids.size() <= quota) {
        return ids;
    }

    auto pairs = sort_unique_pairs(dataset, src, ids);
    std::vector<unsigned> left, right;
    left.reserve(pairs.size());
    right.reserve(pairs.size());
    for (const auto& [d, v] : pairs) {
        (void)d;
        if (v < src) {
            left.push_back(v);
        } else if (v > src) {
            right.push_back(v);
        }
    }

    std::vector<unsigned> out;
    out.reserve(quota);
    size_t li = 0, ri = 0;
    int last_side = -1;
    while (out.size() < quota && (li < left.size() || ri < right.size())) {
        if (li >= left.size()) {
            out.push_back(right[ri++]);
            last_side = 1;
            continue;
        }
        if (ri >= right.size()) {
            out.push_back(left[li++]);
            last_side = 0;
            continue;
        }
        const bool take_left = (last_side == 1 || last_side == -1);
        if (take_left) {
            out.push_back(left[li++]);
            last_side = 0;
        } else {
            out.push_back(right[ri++]);
            last_side = 1;
        }
    }
    return out;
}

template <typename T>
std::vector<unsigned> select_gap_priority_quota_by_dist(
    const Vector::VectorList<T>& dataset, unsigned src,
    const std::vector<unsigned>& ids, unsigned quota, unsigned bridge_gap_min) {
    if (quota == 0 || ids.size() <= quota) {
        return ids;
    }

    auto pairs = sort_unique_pairs(dataset, src, ids);
    std::vector<unsigned> local_left, local_right, bridge_left, bridge_right;
    local_left.reserve(pairs.size());
    local_right.reserve(pairs.size());
    bridge_left.reserve(pairs.size());
    bridge_right.reserve(pairs.size());

    for (const auto& [d, v] : pairs) {
        (void)d;
        const unsigned gap = static_cast<unsigned>(std::abs(
            static_cast<long long>(v) - static_cast<long long>(src)));
        const bool is_bridge = gap >= bridge_gap_min;
        if (v < src) {
            (is_bridge ? bridge_left : local_left).push_back(v);
        } else if (v > src) {
            (is_bridge ? bridge_right : local_right).push_back(v);
        }
    }

    const unsigned bridge_target = std::min<unsigned>(
        quota, std::max(2u, static_cast<unsigned>(std::llround(quota * 0.5))));
    std::vector<unsigned> out;
    out.reserve(quota);
    size_t lli = 0, lri = 0, bli = 0, bri = 0;
    unsigned bridge_used = 0;
    int last_side = -1;

    auto emit_from = [&](std::vector<unsigned>& left, size_t& li,
                         std::vector<unsigned>& right, size_t& ri) -> bool {
        if (li >= left.size() && ri >= right.size()) {
            return false;
        }
        if (li >= left.size()) {
            out.push_back(right[ri++]);
            last_side = 1;
            return true;
        }
        if (ri >= right.size()) {
            out.push_back(left[li++]);
            last_side = 0;
            return true;
        }
        const bool take_left = (last_side == 1 || last_side == -1);
        if (take_left) {
            out.push_back(left[li++]);
            last_side = 0;
        } else {
            out.push_back(right[ri++]);
            last_side = 1;
        }
        return true;
    };

    while (out.size() < quota) {
        const bool want_bridge =
            bridge_used < bridge_target &&
            (out.size() * bridge_target < (bridge_used + 1) * quota);
        bool emitted = false;
        if (want_bridge) {
            emitted = emit_from(bridge_left, bli, bridge_right, bri);
            bridge_used += emitted ? 1 : 0;
        }
        if (!emitted) {
            emitted = emit_from(local_left, lli, local_right, lri);
        }
        if (!emitted) {
            emitted = emit_from(bridge_left, bli, bridge_right, bri);
            bridge_used += emitted ? 1 : 0;
        }
        if (!emitted) {
            break;
        }
    }
    return out;
}

template <typename T>
void append_bridge_witness(std::vector<unsigned>& ids,
                           const Vector::VectorList<T>& dataset,
                           unsigned src,
                           const std::vector<unsigned>& bridge_candidates,
                           unsigned reserve_budget) {
    if (reserve_budget == 0 || bridge_candidates.empty()) {
        return;
    }

    phmap::flat_hash_set<unsigned> chosen;
    chosen.reserve(ids.size() * 2 + reserve_budget * 2 + 8);
    for (auto v : ids) {
        chosen.insert(v);
    }

    auto jump_level = [src](unsigned v) -> unsigned {
        const unsigned gap = static_cast<unsigned>(
            std::abs(static_cast<long long>(v) - static_cast<long long>(src)));
        unsigned lv = 0;
        while ((lv + 1) < 31U && (1U << (lv + 1)) <= gap) {
            ++lv;
        }
        return lv;
    };

    auto pool = sort_unique_pairs(dataset, src, bridge_candidates);
    constexpr unsigned kMaxLevel = 30;
    const size_t bin_count = static_cast<size_t>((kMaxLevel + 1) * 2);
    std::vector<std::vector<unsigned>> bins(bin_count);
    for (const auto& [d, v] : pool) {
        (void)d;
        if (v == src || chosen.contains(v)) {
            continue;
        }
        unsigned lv = std::min<unsigned>(jump_level(v), kMaxLevel);
        const unsigned side = (v < src) ? 0U : 1U;
        bins[static_cast<size_t>(side * (kMaxLevel + 1) + lv)].push_back(v);
    }
    for (auto& bin : bins) {
        std::ranges::reverse(bin);
    }

    unsigned added = 0;
    while (added < reserve_budget) {
        bool progressed = false;
        for (int lv = static_cast<int>(kMaxLevel); lv >= 0; --lv) {
            for (unsigned side = 0; side < 2; ++side) {
                auto& bin = bins[static_cast<size_t>(side * (kMaxLevel + 1) + lv)];
                while (!bin.empty() && chosen.contains(bin.back())) {
                    bin.pop_back();
                }
                if (bin.empty()) {
                    continue;
                }
                const unsigned v = bin.back();
                bin.pop_back();
                if (!chosen.insert(v).second) {
                    continue;
                }
                ids.push_back(v);
                ++added;
                progressed = true;
                if (added >= reserve_budget) {
                    break;
                }
            }
            if (added >= reserve_budget) {
                break;
            }
        }
        if (!progressed) {
            break;
        }
    }
}

template <typename T>
void append_support_reserve(
    std::vector<unsigned>& ids,
    const Vector::VectorList<T>& dataset,
    unsigned src,
    const phmap::flat_hash_map<unsigned, unsigned>& bridge_support,
    unsigned reserve_budget) {
    if (reserve_budget == 0 || bridge_support.empty()) {
        return;
    }

    phmap::flat_hash_set<unsigned> chosen;
    chosen.reserve(ids.size() * 2 + reserve_budget * 2 + 8);
    for (auto v : ids) {
        chosen.insert(v);
    }

    std::vector<std::tuple<unsigned, T, unsigned>> scored;
    scored.reserve(bridge_support.size());
    for (const auto& [v, s] : bridge_support) {
        if (v == src || chosen.contains(v)) {
            continue;
        }
        scored.push_back({s, dataset.dist(src, v), v});
    }
    std::ranges::sort(scored, [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) > std::get<0>(b);
        }
        if (std::get<1>(a) != std::get<1>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        return std::get<2>(a) < std::get<2>(b);
    });

    std::vector<unsigned> left, right;
    left.reserve(scored.size());
    right.reserve(scored.size());
    for (const auto& [s, d, v] : scored) {
        (void)s;
        (void)d;
        if (v < src) {
            left.push_back(v);
        } else {
            right.push_back(v);
        }
    }

    size_t li = 0, ri = 0;
    int last_side = -1;
    while (reserve_budget > 0 && (li < left.size() || ri < right.size())) {
        if (li >= left.size()) {
            ids.push_back(right[ri++]);
            last_side = 1;
            --reserve_budget;
            continue;
        }
        if (ri >= right.size()) {
            ids.push_back(left[li++]);
            last_side = 0;
            --reserve_budget;
            continue;
        }
        const bool take_left = (last_side == 1 || last_side == -1);
        if (take_left) {
            ids.push_back(left[li++]);
            last_side = 0;
        } else {
            ids.push_back(right[ri++]);
            last_side = 1;
        }
        --reserve_budget;
    }
}

template <typename T>
void append_support_reserve_gap(
    std::vector<unsigned>& ids,
    const Vector::VectorList<T>& dataset,
    unsigned src,
    const phmap::flat_hash_map<unsigned, unsigned>& bridge_support,
    unsigned reserve_budget,
    unsigned bridge_gap_min) {
    if (reserve_budget == 0 || bridge_support.empty()) {
        return;
    }

    phmap::flat_hash_set<unsigned> chosen;
    chosen.reserve(ids.size() * 2 + reserve_budget * 2 + 8);
    for (auto v : ids) {
        chosen.insert(v);
    }

    auto jump_level = [src](unsigned v) -> unsigned {
        const unsigned gap = static_cast<unsigned>(
            std::abs(static_cast<long long>(v) - static_cast<long long>(src)));
        unsigned lv = 0;
        while ((lv + 1) < 31U && (1U << (lv + 1)) <= gap) {
            ++lv;
        }
        return lv;
    };

    using Item = std::tuple<unsigned, unsigned, T, unsigned>;
    std::vector<Item> pool;
    pool.reserve(bridge_support.size());
    for (const auto& [v, s] : bridge_support) {
        if (v == src || chosen.contains(v)) {
            continue;
        }
        const unsigned gap = static_cast<unsigned>(std::abs(
            static_cast<long long>(v) - static_cast<long long>(src)));
        if (gap < bridge_gap_min) {
            continue;
        }
        pool.push_back({jump_level(v), s, dataset.dist(src, v), v});
    }
    std::ranges::sort(pool, [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) > std::get<0>(b);
        }
        if (std::get<1>(a) != std::get<1>(b)) {
            return std::get<1>(a) > std::get<1>(b);
        }
        if (std::get<2>(a) != std::get<2>(b)) {
            return std::get<2>(a) < std::get<2>(b);
        }
        return std::get<3>(a) < std::get<3>(b);
    });

    std::vector<unsigned> left, right;
    left.reserve(pool.size());
    right.reserve(pool.size());
    for (const auto& [lv, s, d, v] : pool) {
        (void)lv;
        (void)s;
        (void)d;
        if (v < src) {
            left.push_back(v);
        } else {
            right.push_back(v);
        }
    }

    size_t li = 0, ri = 0;
    int last_side = -1;
    while (reserve_budget > 0 && (li < left.size() || ri < right.size())) {
        if (li >= left.size()) {
            ids.push_back(right[ri++]);
            last_side = 1;
            --reserve_budget;
            continue;
        }
        if (ri >= right.size()) {
            ids.push_back(left[li++]);
            last_side = 0;
            --reserve_budget;
            continue;
        }
        const bool take_left = (last_side == 1 || last_side == -1);
        if (take_left) {
            ids.push_back(left[li++]);
            last_side = 0;
        } else {
            ids.push_back(right[ri++]);
            last_side = 1;
        }
        --reserve_budget;
    }
}

template <typename T>
void append_tail_reserve(std::vector<unsigned>& ids,
                         const Vector::VectorList<T>& dataset,
                         unsigned src,
                         const std::vector<unsigned>& core_candidates,
                         const std::vector<unsigned>& bridge_candidates,
                         unsigned reserve_budget) {
    if (reserve_budget == 0) {
        return;
    }

    auto jump_level = [src](unsigned v) -> unsigned {
        const unsigned gap = static_cast<unsigned>(
            std::abs(static_cast<long long>(v) - static_cast<long long>(src)));
        unsigned lv = 0;
        while ((lv + 1) < 31U && (1U << (lv + 1)) <= gap) {
            ++lv;
        }
        return lv;
    };

    phmap::flat_hash_set<unsigned> chosen;
    chosen.reserve(ids.size() * 2 + reserve_budget * 2 + 8);
    for (auto v : ids) {
        chosen.insert(v);
    }

    auto gather = [&](const std::vector<unsigned>& srcs) {
        std::vector<std::pair<T, unsigned>> pool;
        pool.reserve(srcs.size());
        for (auto v : srcs) {
            if (v == src || chosen.contains(v)) {
                continue;
            }
            chosen.insert(v);
            pool.push_back({dataset.dist(src, v), v});
        }
        return pool;
    };

    auto core_pool = gather(core_candidates);
    auto bridge_pool = gather(bridge_candidates);
    auto by_dist = [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    };
    std::ranges::sort(core_pool, by_dist);
    std::ranges::sort(bridge_pool, by_dist);

    constexpr unsigned kMaxLevel = 30;
    const size_t bin_count = static_cast<size_t>((kMaxLevel + 1) * 2);
    std::vector<std::vector<unsigned>> bins(bin_count);
    auto push_bins = [&](const std::vector<std::pair<T, unsigned>>& pool) {
        for (const auto& [d, v] : pool) {
            (void)d;
            unsigned lv = std::min<unsigned>(jump_level(v), kMaxLevel);
            const unsigned side = (v < src) ? 0U : 1U;
            size_t b = static_cast<size_t>(side * (kMaxLevel + 1) + lv);
            bins[b].push_back(v);
        }
    };
    push_bins(core_pool);
    push_bins(bridge_pool);
    for (auto& bin : bins) {
        std::ranges::reverse(bin);
    }

    unsigned added = 0;
    phmap::flat_hash_set<unsigned> appended;
    appended.reserve(reserve_budget * 2 + 8);

    while (added < reserve_budget) {
        bool progressed = false;
        for (int lv = static_cast<int>(kMaxLevel); lv >= 0; --lv) {
            for (unsigned side = 0; side < 2; ++side) {
                size_t b = static_cast<size_t>(side * (kMaxLevel + 1) + lv);
                auto& bin = bins[b];
                while (!bin.empty() && appended.contains(bin.back())) {
                    bin.pop_back();
                }
                if (bin.empty()) {
                    continue;
                }
                unsigned v = bin.back();
                bin.pop_back();
                if (!appended.insert(v).second) {
                    continue;
                }
                ids.push_back(v);
                ++added;
                progressed = true;
                if (added >= reserve_budget) {
                    break;
                }
            }
            if (added >= reserve_budget) {
                break;
            }
        }
        if (!progressed) {
            break;
        }
    }
}

}  // namespace detail

template <typename T>
Graph::TDGraphIndexBase Builder<T>::build(
    Graph::GraphLike auto&& knng, unsigned range_step, unsigned ef_max,
    const std::vector<std::uint64_t>& label, const BuildOptions& options) {
    const int thread_count = Utils::configured_thread_count();
    omp_set_num_threads(thread_count);
    const bool profile_enabled = !options.build_profile_json.empty();
    std::vector<detail::ThreadBuildProfile> profiles(
        static_cast<size_t>(thread_count));
    spdlog::info(
        "Building EnhancedRNSG Index, size={}, range_step={}, ef_max={}, range_window_cap={}",
        vector_list.size(), range_step, ef_max, options.range_window_cap);
    const std::string seed_limit_desc =
        options.monotone_seed_limit == 0
            ? "all"
            : std::to_string(options.monotone_seed_limit);
    spdlog::info(
        "Enhanced options: centroid_seed={}, reverse_refine={}, seed_limit={}, "
        "seed_policy={}, reverse_mode={}, seed_mode={}, collect_policy={}, seed_keep={}, seed_expand={}, seed_batch={}, seed_beam={}, seed_knng_cap={}, knng_cap={}, mrng_pruning={}, merge_mode={}, "
        "core_ratio={:.3f}, incoming_quota={}({}), bridge_witness={}, support_reserve={}({}), tail_reserve={}, "
        "role_select={} extra={} support_append={} warmup={} mid_gap={} far_gap={} mid_ratio={:.3f} far_ratio={:.3f}, prefix_policy={}",
        options.enable_centroid_seed_search, options.enable_reverse_refine,
        seed_limit_desc, options.monotone_seed_policy, options.reverse_refine_mode,
        options.seed_collect_mode,
        options.seed_collect_policy,
        options.seed_collect_keep,
        options.seed_collect_max_expand, options.seed_batch_size,
        options.seed_search_beam_size,
        options.seed_search_knng_cap,
        options.knng_degree_cap,
        options.use_mrng_pruning,
        options.candidate_merge_mode, options.core_ratio,
        options.reverse_incoming_quota,
        options.reverse_incoming_policy,
        options.bridge_witness_reserve,
        options.support_reserve,
        options.support_reserve_policy,
        options.tail_reserve,
        detail::normalize_role_select_policy(options.role_select_policy),
        options.role_pool_extra,
        options.role_support_append,
        options.role_local_warmup,
        options.role_mid_gap_min,
        options.role_far_gap_min,
        options.role_mid_ratio,
        options.role_far_ratio,
        RNSG::normalize_prefix_policy(options.prefix_policy));

    Timer::start("enhanced_build_time");

    std::uint64_t center_ns = 0;
    std::uint64_t label_order_ns = 0;
    std::uint64_t reorder_ns = 0;
    std::uint64_t centroid_pass_ns = 0;
    std::uint64_t parent_chain_ns = 0;
    std::uint64_t knng_remap_ns = 0;
    std::uint64_t incoming_build_ns = 0;
    std::uint64_t header_init_ns = 0;
    std::uint64_t finalize_graph_ns = 0;

    auto phase_tick = std::chrono::steady_clock::now();
    auto center = vector_list.mean();
    center_ns = detail::ns_since(phase_tick);

    phase_tick = std::chrono::steady_clock::now();
    std::vector<unsigned> index, pos;
    std::tie(index, pos) = Utils::order_of_label(label);
    auto sorted_label = Utils::sorted_vec(label);
    label_order_ns = detail::ns_since(phase_tick);
    auto& dataset = vector_list;
    const bool enable_role_select =
        detail::normalize_role_select_policy(options.role_select_policy) ==
        "roles";

    phase_tick = std::chrono::steady_clock::now();
    dataset.reorder(index);
    reorder_ns = detail::ns_since(phase_tick);

    const unsigned n = dataset.size();
    std::vector<T> centroid_dist(n);
    phase_tick = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
        const unsigned i = static_cast<unsigned>(ii);
        centroid_dist[i] = dataset.dist(i, center);
    }
    centroid_pass_ns = detail::ns_since(phase_tick);

    phase_tick = std::chrono::steady_clock::now();
    auto left_parent = detail::build_prev_smaller(centroid_dist);
    auto right_parent = detail::build_next_smaller(centroid_dist);
    parent_chain_ns = detail::ns_since(phase_tick);

    phase_tick = std::chrono::steady_clock::now();
    auto sorted_knng = detail::build_sorted_knng(
        knng, index, pos, options.knng_degree_cap);
    knng_remap_ns = detail::ns_since(phase_tick);

    auto add_local_window = [&](unsigned src, std::vector<unsigned>& dst) {
        if (!options.enable_range_augmentation || range_step == 0) {
            return;
        }
        const unsigned left_n = std::min(src, range_step);
        const unsigned right_n =
            (src + 1 < n) ? std::min(range_step, n - src - 1) : 0u;
        const unsigned full_n = left_n + right_n;
        if (options.range_window_cap == 0 ||
            full_n <= options.range_window_cap) {
            for (unsigned j = src - left_n; j < src; ++j) {
                dst.push_back(j);
            }
            for (unsigned j = src + 1; j <= src + right_n; ++j) {
                dst.push_back(j);
            }
            return;
        }

        unsigned left_quota =
            std::min(left_n, (options.range_window_cap + 1) / 2);
        unsigned right_quota =
            std::min(right_n, options.range_window_cap - left_quota);
        const unsigned spare = options.range_window_cap - left_quota - right_quota;
        if (spare > 0 && left_quota < left_n) {
            const unsigned add = std::min(spare, left_n - left_quota);
            left_quota += add;
        }
        if (left_quota + right_quota < options.range_window_cap &&
            right_quota < right_n) {
            const unsigned add =
                std::min(options.range_window_cap - left_quota - right_quota,
                         right_n - right_quota);
            right_quota += add;
        }

        auto emit_sampled = [&](unsigned begin, unsigned count,
                                unsigned quota) {
            if (quota == 0 || count == 0) {
                return;
            }
            if (quota >= count) {
                for (unsigned k = 0; k < count; ++k) {
                    dst.push_back(begin + k);
                }
                return;
            }
            if (quota == 1) {
                dst.push_back(begin + count / 2);
                return;
            }
            for (unsigned k = 0; k < quota; ++k) {
                const auto off = static_cast<unsigned>(
                    static_cast<unsigned long long>(k) * (count - 1) /
                    (quota - 1));
                dst.push_back(begin + off);
            }
        };

        emit_sampled(src - left_n, left_n, left_quota);
        emit_sampled(src + 1, right_n, right_quota);
    };

    auto capped_window_count = [&](unsigned full_count) -> unsigned {
        if (options.range_window_cap == 0) {
            return full_count;
        }
        return std::min(full_count, options.range_window_cap);
    };

    auto local_window_count = [&](unsigned src) -> std::uint64_t {
        if (!options.enable_range_augmentation || range_step == 0) {
            return 0;
        }
        const unsigned left_n = std::min(src, range_step);
        const unsigned right_n =
            (src + 1 < n) ? std::min(range_step, n - src - 1) : 0u;
        return capped_window_count(left_n + right_n);
    };

    auto collect_seed_chain = [&](unsigned src, bool left_side,
                                  phmap::flat_hash_set<unsigned>& seen,
                                  std::vector<unsigned>& dst,
                                  detail::ThreadBuildProfile& prof,
                                  phmap::flat_hash_map<unsigned, unsigned>* support) {
        if (!options.enable_centroid_seed_search) {
            return;
        }
        int cur = -1;
        if (left_side) {
            cur = (src == 0) ? -1 : static_cast<int>(src - 1);
        } else {
            cur = (src + 1 >= n) ? -1 : static_cast<int>(src + 1);
        }
        const unsigned batch_size = std::max(1u, options.seed_batch_size);
        std::vector<unsigned> seed_batch;
        seed_batch.reserve(batch_size);
        const bool full_chain = options.monotone_seed_limit == 0;
        const auto seed_collect_policy =
            detail::parse_seed_collect_policy(options.seed_collect_policy);

        auto process_seed_batch = [&](const std::vector<unsigned>& batch) {
            if (batch.empty()) {
                return;
            }
            prof.seed_batches += 1;
            prof.seed_seeds += batch.size();
            const unsigned scaled_keep =
                options.seed_collect_keep == 0
                    ? 0u
                    : static_cast<unsigned>(
                          std::min<size_t>(
                              std::numeric_limits<unsigned>::max(),
                              static_cast<size_t>(options.seed_collect_keep) *
                                  batch.size()));
            const unsigned scaled_expand = static_cast<unsigned>(
                std::min<size_t>(
                    std::numeric_limits<unsigned>::max(),
                    options.seed_collect_max_expand == 0
                        ? size_t{0}
                        : static_cast<size_t>(
                              options.seed_collect_max_expand) *
                              batch.size()));
            auto result = options.seed_collect_mode == "pq"
                              ? detail::collect_from_seed_batch_pq(
                                    dataset, sorted_knng, src, batch,
                                    scaled_keep, scaled_expand,
                                    options.seed_search_knng_cap,
                                    seed_collect_policy,
                                    &prof.seed_neighbor_scans)
                              : detail::collect_from_seed_batch_beam(
                                    dataset, sorted_knng, src, batch,
                                    scaled_keep, scaled_expand,
                                    options.seed_search_beam_size,
                                    options.seed_search_knng_cap,
                                    seed_collect_policy,
                                    &prof.seed_neighbor_scans);
            prof.seed_expanded += result.expanded;
            prof.seed_collected += result.nodes.size();
            for (auto v : result.nodes) {
                if (v == src) {
                    continue;
                }
                if (support != nullptr) {
                    (*support)[v] += 1;
                }
                    if (seen.insert(v).second) {
                        dst.push_back(v);
                    }
                }
            };

        auto append_seed = [&](unsigned seed) {
            seed_batch.push_back(seed);
            if (support != nullptr && seed != src) {
                (*support)[seed] += 1;
            }
            if (seen.insert(seed).second) {
                dst.push_back(seed);
            }
            if (seed_batch.size() >= batch_size) {
                process_seed_batch(seed_batch);
                seed_batch.clear();
            }
        };

        if (options.monotone_seed_policy == "far") {
            std::vector<unsigned> chain;
            chain.reserve(full_chain ? 64u : options.monotone_seed_limit);
            while (cur >= 0) {
                const unsigned seed = static_cast<unsigned>(cur);
                chain.push_back(seed);
                cur = left_side ? left_parent[seed] : right_parent[seed];
            }
            size_t first = 0;
            if (!full_chain && chain.size() > options.monotone_seed_limit) {
                first = chain.size() - options.monotone_seed_limit;
            }
            for (size_t idx = chain.size(); idx > first; --idx) {
                append_seed(chain[idx - 1]);
            }
        } else {
            unsigned used = 0;
            while (cur >= 0 &&
                   (full_chain || used < options.monotone_seed_limit)) {
                const unsigned seed = static_cast<unsigned>(cur);
                append_seed(seed);
                cur = left_side ? left_parent[seed] : right_parent[seed];
                used++;
            }
        }
        process_seed_batch(seed_batch);
    };

    std::vector<std::vector<unsigned>> first_adj(n);
    const unsigned step = std::max(1u, (n + 99) / 100);
    std::atomic<unsigned> build_step = 0;
#pragma omp parallel for schedule(dynamic)
    for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
        const unsigned i = static_cast<unsigned>(ii);
        auto& prof = profiles[static_cast<size_t>(omp_get_thread_num())];
        prof.first_nodes += 1;
        const unsigned build_now = build_step.fetch_add(1) + 1;
        const bool output_tag = (build_now % step == 0) || (build_now == n);
        if (output_tag) {
            Timer::start("enhanced_collect");
        }

        phmap::flat_hash_set<unsigned> seen;
        const auto window_budget = local_window_count(i);
        const unsigned seed_reserve_hint =
            options.monotone_seed_limit == 0 ? 32u : options.monotone_seed_limit;
        const unsigned seed_effective_beam_hint =
            options.seed_search_beam_size == 0
                ? (options.seed_collect_max_expand == 0 ? 1024u
                                                        : options.seed_collect_max_expand)
                : options.seed_search_beam_size;
        const unsigned seed_expand_hint =
            options.seed_collect_max_expand == 0 ? seed_effective_beam_hint
                                                 : options.seed_collect_max_expand;
        const unsigned seed_row_cap_hint =
            options.seed_search_knng_cap == 0
                ? (options.knng_degree_cap == 0
                       ? static_cast<unsigned>(std::min<size_t>(
                             sorted_knng[i].size(),
                             std::numeric_limits<unsigned>::max()))
                       : options.knng_degree_cap)
                : options.seed_search_knng_cap;
        const unsigned seed_collect_reserve =
            options.seed_collect_keep > 0
                ? options.seed_collect_keep
                : (options.seed_collect_policy == "evaluated"
                       ? static_cast<unsigned>(std::min<size_t>(
                             std::numeric_limits<unsigned>::max(),
                             static_cast<size_t>(seed_expand_hint) *
                                     std::max(1u, seed_row_cap_hint) +
                                 seed_reserve_hint))
                       : (options.seed_collect_policy == "discovered"
                              ? seed_effective_beam_hint
                              : seed_expand_hint));
        const auto bridge_budget = static_cast<size_t>(
            2u * seed_reserve_hint +
            2u * seed_reserve_hint * seed_collect_reserve);
        seen.reserve(sorted_knng[i].size() + bridge_budget + 64);
        std::vector<unsigned> core_candidates;
        std::vector<unsigned> bridge_candidates;
        phmap::flat_hash_map<unsigned, unsigned> bridge_support;
        core_candidates.reserve(sorted_knng[i].size() + window_budget + 32);
        bridge_candidates.reserve(bridge_budget + 32);
        bridge_support.reserve(bridge_budget + 32);

        auto t_stage = std::chrono::steady_clock::now();
        for (auto v : sorted_knng[i]) {
            if (v == i) {
                continue;
            }
            if (seen.insert(v).second) {
                core_candidates.push_back(v);
            }
        }
        prof.first_knng_ns += detail::ns_since(t_stage);
        prof.first_knng_candidates += core_candidates.size();

        const auto core_before_window = core_candidates.size();
        t_stage = std::chrono::steady_clock::now();
        add_local_window(i, core_candidates);
        prof.first_window_ns += detail::ns_since(t_stage);
        prof.first_window_attempted += window_budget;
        prof.first_window_inserted +=
            core_candidates.size() - core_before_window;

        const auto bridge_before_seed = bridge_candidates.size();
        t_stage = std::chrono::steady_clock::now();
        collect_seed_chain(i, true, seen, bridge_candidates, prof,
                           &bridge_support);
        collect_seed_chain(i, false, seen, bridge_candidates, prof,
                           &bridge_support);
        prof.first_seed_ns += detail::ns_since(t_stage);
        prof.first_bridge_candidates +=
            bridge_candidates.size() - bridge_before_seed;

        std::vector<unsigned> pruned;
        t_stage = std::chrono::steady_clock::now();
        const unsigned prune_budget =
            enable_role_select
                ? static_cast<unsigned>(std::min<size_t>(
                      std::numeric_limits<unsigned>::max(),
                      static_cast<size_t>(ef_max) + options.role_pool_extra))
                : ef_max;
        if (options.candidate_merge_mode == "quota" &&
            !options.use_mrng_pruning) {
            pruned = detail::select_quota_candidates(
                dataset, i, core_candidates, bridge_candidates, prune_budget,
                options.enable_side_split_pruning, options.core_ratio);
        } else {
            std::vector<unsigned> merged = core_candidates;
            merged.insert(merged.end(), bridge_candidates.begin(),
                          bridge_candidates.end());
            pruned = detail::prune_candidates(dataset, i, merged, prune_budget,
                                              options.enable_side_split_pruning,
                                              options.use_mrng_pruning);
        }
        prof.first_prune_ns += detail::ns_since(t_stage);
        prof.first_pruned_candidates += pruned.size();

        t_stage = std::chrono::steady_clock::now();
        if (enable_role_select) {
            auto role_pool = pruned;
            if (options.role_support_append && options.role_pool_extra > 0) {
                detail::append_role_pool_support(
                    role_pool, dataset, i, bridge_support,
                    options.role_pool_extra,
                    std::max(1u, options.role_mid_gap_min));
            }
            pruned = detail::select_role_quota_candidates(
                dataset, i, role_pool, ef_max, options, &bridge_support);
            detail::reorder_final(pruned, dataset, i, options);
        } else {
            if (options.support_reserve_policy == "bridgegap") {
                detail::append_support_reserve_gap(
                    pruned, dataset, i, bridge_support, options.support_reserve,
                    std::max(1u, options.prefix_jump_min_gap));
            } else {
                detail::append_support_reserve(
                    pruned, dataset, i, bridge_support, options.support_reserve);
            }
            detail::append_bridge_witness(pruned, dataset, i, bridge_candidates,
                                          options.bridge_witness_reserve);
            detail::reorder_final(pruned, dataset, i, options);
            detail::append_tail_reserve(pruned, dataset, i, core_candidates,
                                        bridge_candidates, options.tail_reserve);
        }
        prof.first_reorder_ns += detail::ns_since(t_stage);
        first_adj[i] = std::move(pruned);

        if (output_tag) {
            const auto t = Timer::end("enhanced_collect");
            spdlog::info(
                "Enhanced build progress: {}/{} ({:.2f}%), core {} + bridge {} -> {} in {} ns",
                i + 1, n, (i + 1) * 100.0 / n, core_candidates.size(),
                bridge_candidates.size(), first_adj[i].size(), t);
        }
    }

    std::vector<std::vector<unsigned>> final_adj = first_adj;
    if (options.enable_reverse_refine) {
        std::vector<std::vector<unsigned>> incoming(n);
        phase_tick = std::chrono::steady_clock::now();
        for (unsigned u = 0; u < n; ++u) {
            for (auto v : first_adj[u]) {
                incoming[v].push_back(u);
            }
        }
        incoming_build_ns = detail::ns_since(phase_tick);
        std::atomic<unsigned> reverse_step = 0;
#pragma omp parallel for schedule(dynamic)
        for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
            const unsigned i = static_cast<unsigned>(ii);
            auto& prof = profiles[static_cast<size_t>(omp_get_thread_num())];
            prof.reverse_nodes += 1;
            const unsigned reverse_now = reverse_step.fetch_add(1) + 1;
            const bool output_tag = (reverse_now % step == 0) || (reverse_now == n);
            if (output_tag) {
                Timer::start("enhanced_reverse_collect");
            }
            phmap::flat_hash_set<unsigned> seen;
            const bool full_reverse_refine =
                options.reverse_refine_mode == "full";
            const auto window_budget =
                full_reverse_refine ? local_window_count(i) : 0u;
            seen.reserve(first_adj[i].size() + incoming[i].size() +
                         (full_reverse_refine ? sorted_knng[i].size() : 0u) +
                         64);
            std::vector<unsigned> core_candidates;
            std::vector<unsigned> bridge_candidates;
            phmap::flat_hash_map<unsigned, unsigned> reverse_support;
            core_candidates.reserve(first_adj[i].size() +
                                    (full_reverse_refine ? sorted_knng[i].size() + window_budget : 0u) +
                                    32);
            bridge_candidates.reserve(incoming[i].size() + 32);
            reverse_support.reserve(incoming[i].size() + 32);

            auto append_all = [&](const std::vector<unsigned>& srcs,
                                  std::vector<unsigned>& dst) {
                for (auto v : srcs) {
                    if (v == i) {
                        continue;
                    }
                if (seen.insert(v).second) {
                    dst.push_back(v);
                }
            }
        };

            auto t_stage = std::chrono::steady_clock::now();
            append_all(first_adj[i], core_candidates);
            size_t core_before_window = core_candidates.size();
            if (full_reverse_refine) {
                // "full" keeps the original heavy second pass: reopen local
                // KNN/window candidates and let reverse edges compete again.
                append_all(sorted_knng[i], core_candidates);
                core_before_window = core_candidates.size();
                add_local_window(i, core_candidates);
            }
            // "incoming" is the lightweight mode: preserve the first-pass
            // prefix/core set, and only add reverse incoming neighbours as new
            // competitors instead of rebuilding the whole local candidate pool.
            if (options.reverse_incoming_quota > 0) {
                auto incoming_selected =
                    options.reverse_incoming_policy == "bridgegap"
                        ? detail::select_gap_priority_quota_by_dist(
                              dataset, i, incoming[i],
                              options.reverse_incoming_quota,
                              std::max(1u, options.prefix_jump_min_gap))
                        : detail::select_balanced_quota_by_dist(
                              dataset, i, incoming[i],
                              options.reverse_incoming_quota);
                for (auto v : incoming_selected) {
                    reverse_support[v] += 1;
                }
                append_all(incoming_selected, bridge_candidates);
            } else {
                for (auto v : incoming[i]) {
                    reverse_support[v] += 1;
                }
                append_all(incoming[i], bridge_candidates);
            }
            prof.reverse_collect_ns += detail::ns_since(t_stage);
            prof.reverse_core_candidates += core_candidates.size();
            prof.reverse_window_attempted += window_budget;
            prof.reverse_window_inserted +=
                core_candidates.size() - core_before_window;
            prof.reverse_bridge_candidates += bridge_candidates.size();

            std::vector<unsigned> pruned;
            t_stage = std::chrono::steady_clock::now();
            const unsigned prune_budget =
                enable_role_select
                    ? static_cast<unsigned>(std::min<size_t>(
                          std::numeric_limits<unsigned>::max(),
                          static_cast<size_t>(ef_max) + options.role_pool_extra))
                    : ef_max;
            if (options.candidate_merge_mode == "quota" &&
                !options.use_mrng_pruning) {
                pruned = detail::select_quota_candidates(
                    dataset, i, core_candidates, bridge_candidates, prune_budget,
                    options.enable_side_split_pruning, options.core_ratio);
            } else {
                std::vector<unsigned> merged = core_candidates;
                merged.insert(merged.end(), bridge_candidates.begin(),
                              bridge_candidates.end());
                pruned = detail::prune_candidates(
                    dataset, i, merged, prune_budget,
                    options.enable_side_split_pruning,
                    options.use_mrng_pruning);
            }
            prof.reverse_prune_ns += detail::ns_since(t_stage);
            prof.reverse_pruned_candidates += pruned.size();
            t_stage = std::chrono::steady_clock::now();
            if (enable_role_select) {
                auto role_pool = pruned;
                if (options.role_support_append && options.role_pool_extra > 0) {
                    detail::append_role_pool_support(
                        role_pool, dataset, i, reverse_support,
                        options.role_pool_extra,
                        std::max(1u, options.role_mid_gap_min));
                }
                pruned = detail::select_role_quota_candidates(
                    dataset, i, role_pool, ef_max, options, &reverse_support);
                detail::reorder_final(pruned, dataset, i, options);
            } else {
                detail::append_bridge_witness(
                    pruned, dataset, i, bridge_candidates,
                    options.bridge_witness_reserve);
                detail::reorder_final(pruned, dataset, i, options);
                detail::append_tail_reserve(pruned, dataset, i, core_candidates,
                                            bridge_candidates,
                                            options.tail_reserve);
            }
            prof.reverse_reorder_ns += detail::ns_since(t_stage);
            final_adj[i] = std::move(pruned);
            if (output_tag) {
                const auto t = Timer::end("enhanced_reverse_collect");
                spdlog::info(
                    "Enhanced reverse progress: {}/{} ({:.2f}%), core {} + bridge {} -> {} in {} ns",
                    i + 1, n, (i + 1) * 100.0 / n, core_candidates.size(),
                    bridge_candidates.size(), final_adj[i].size(), t);
            }
        }
    }

    Graph::TDGraphIndexBase g(n);
    RNSG::Builder<T> base(vector_list);
    phase_tick = std::chrono::steady_clock::now();
    base.init_header(g, center, sorted_label, std::views::iota(0u, n), dataset);
    header_init_ns = detail::ns_since(phase_tick);

    std::uint64_t total_degree = 0;
    phase_tick = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < n; ++i) {
        total_degree += final_adj[i].size();
        g.add_neighbours(i, final_adj[i] | std::views::transform([](unsigned x) {
                             return Graph::to_node(x);
                         }));
    }
    finalize_graph_ns = detail::ns_since(phase_tick);

    spdlog::info("Enhanced average degree {:.2f}",
                 total_degree * 1.0 / std::max(1u, n));
    if (profile_enabled) {
        detail::ThreadBuildProfile agg;
        for (const auto& prof : profiles) {
            agg.first_knng_ns += prof.first_knng_ns;
            agg.first_window_ns += prof.first_window_ns;
            agg.first_seed_ns += prof.first_seed_ns;
            agg.first_prune_ns += prof.first_prune_ns;
            agg.first_reorder_ns += prof.first_reorder_ns;
            agg.reverse_collect_ns += prof.reverse_collect_ns;
            agg.reverse_prune_ns += prof.reverse_prune_ns;
            agg.reverse_reorder_ns += prof.reverse_reorder_ns;
            agg.first_knng_candidates += prof.first_knng_candidates;
            agg.first_window_attempted += prof.first_window_attempted;
            agg.first_window_inserted += prof.first_window_inserted;
            agg.first_bridge_candidates += prof.first_bridge_candidates;
            agg.first_pruned_candidates += prof.first_pruned_candidates;
            agg.reverse_core_candidates += prof.reverse_core_candidates;
            agg.reverse_window_attempted += prof.reverse_window_attempted;
            agg.reverse_window_inserted += prof.reverse_window_inserted;
            agg.reverse_bridge_candidates += prof.reverse_bridge_candidates;
            agg.reverse_pruned_candidates += prof.reverse_pruned_candidates;
            agg.seed_batches += prof.seed_batches;
            agg.seed_seeds += prof.seed_seeds;
            agg.seed_expanded += prof.seed_expanded;
            agg.seed_collected += prof.seed_collected;
            agg.seed_neighbor_scans += prof.seed_neighbor_scans;
            agg.first_nodes += prof.first_nodes;
            agg.reverse_nodes += prof.reverse_nodes;
        }

        const double first_nodes =
            std::max<std::uint64_t>(1, agg.first_nodes);
        const double reverse_nodes =
            std::max<std::uint64_t>(1, agg.reverse_nodes);
        spdlog::info(
            "Enhanced build profile setup(ms): center={:.2f}, label_order={:.2f}, "
            "reorder={:.2f}, centroid_pass={:.2f}, parent_chain={:.2f}, "
            "knng_remap={:.2f}, incoming_build={:.2f}, header_init={:.2f}, "
            "graph_finalize={:.2f}",
            detail::ns_to_ms(center_ns), detail::ns_to_ms(label_order_ns),
            detail::ns_to_ms(reorder_ns), detail::ns_to_ms(centroid_pass_ns),
            detail::ns_to_ms(parent_chain_ns), detail::ns_to_ms(knng_remap_ns),
            detail::ns_to_ms(incoming_build_ns), detail::ns_to_ms(header_init_ns),
            detail::ns_to_ms(finalize_graph_ns));
        spdlog::info(
            "Enhanced build profile first-pass avg(us/node): knng={:.2f}, "
            "window={:.2f}, seed={:.2f}, prune={:.2f}, reorder={:.2f}",
            detail::ns_to_us(agg.first_knng_ns) / first_nodes,
            detail::ns_to_us(agg.first_window_ns) / first_nodes,
            detail::ns_to_us(agg.first_seed_ns) / first_nodes,
            detail::ns_to_us(agg.first_prune_ns) / first_nodes,
            detail::ns_to_us(agg.first_reorder_ns) / first_nodes);
        spdlog::info(
            "Enhanced build profile first-pass candidates(avg/node): knng={:.2f}, "
            "window_attempt={:.2f}, window_kept={:.2f}, bridge={:.2f}, "
            "pruned={:.2f}, seed_batches={:.2f}, seed_size={:.2f}, "
            "seed_expanded={:.2f}, seed_collected={:.2f}, seed_neighbor_scans={:.2f}",
            agg.first_knng_candidates / first_nodes,
            agg.first_window_attempted / first_nodes,
            agg.first_window_inserted / first_nodes,
            agg.first_bridge_candidates / first_nodes,
            agg.first_pruned_candidates / first_nodes,
            agg.seed_batches / first_nodes,
            agg.seed_seeds * 1.0 /
                std::max<std::uint64_t>(1, agg.seed_batches),
            agg.seed_expanded * 1.0 /
                std::max<std::uint64_t>(1, agg.seed_batches),
            agg.seed_collected * 1.0 /
                std::max<std::uint64_t>(1, agg.seed_batches),
            agg.seed_neighbor_scans * 1.0 /
                std::max<std::uint64_t>(1, agg.seed_batches));
        spdlog::info(
            "Enhanced build profile reverse-pass avg(us/node): collect={:.2f}, "
            "prune={:.2f}, reorder={:.2f}",
            detail::ns_to_us(agg.reverse_collect_ns) / reverse_nodes,
            detail::ns_to_us(agg.reverse_prune_ns) / reverse_nodes,
            detail::ns_to_us(agg.reverse_reorder_ns) / reverse_nodes);
        spdlog::info(
            "Enhanced build profile reverse-pass candidates(avg/node): core={:.2f}, "
            "window_attempt={:.2f}, window_kept={:.2f}, bridge={:.2f}, "
            "pruned={:.2f}",
            agg.reverse_core_candidates / reverse_nodes,
            agg.reverse_window_attempted / reverse_nodes,
            agg.reverse_window_inserted / reverse_nodes,
            agg.reverse_bridge_candidates / reverse_nodes,
            agg.reverse_pruned_candidates / reverse_nodes);

        nlohmann::json j;
        j["size"] = n;
        j["range_step"] = range_step;
        j["ef_max"] = ef_max;
        j["range_window_cap"] = options.range_window_cap;
        j["thread_count"] = thread_count;
        j["enable_range_augmentation"] = options.enable_range_augmentation;
        j["enable_side_split_pruning"] = options.enable_side_split_pruning;
        j["use_mrng_pruning"] = options.use_mrng_pruning;
        j["knng_degree_cap"] = options.knng_degree_cap;
        j["candidate_merge_mode"] = options.candidate_merge_mode;
        j["core_ratio"] = options.core_ratio;
        j["prefix_policy"] = RNSG::normalize_prefix_policy(options.prefix_policy);
        j["monotone_seed_policy"] = options.monotone_seed_policy;
        j["monotone_seed_limit"] = options.monotone_seed_limit;
        j["monotone_seed_full_chain"] = (options.monotone_seed_limit == 0);
        j["seed_collect_keep"] = options.seed_collect_keep;
        j["seed_collect_max_expand"] = options.seed_collect_max_expand;
        j["seed_batch_size"] = options.seed_batch_size;
        j["seed_search_beam_size"] = options.seed_search_beam_size;
        j["seed_search_knng_cap"] = options.seed_search_knng_cap;
        j["reverse_refine_mode"] = options.reverse_refine_mode;
        j["seed_collect_mode"] = options.seed_collect_mode;
        j["seed_collect_policy"] = options.seed_collect_policy;
        j["reverse_incoming_quota"] = options.reverse_incoming_quota;
        j["reverse_incoming_policy"] = options.reverse_incoming_policy;
        j["bridge_witness_reserve"] = options.bridge_witness_reserve;
        j["support_reserve"] = options.support_reserve;
        j["support_reserve_policy"] = options.support_reserve_policy;
        j["role_select_policy"] =
            detail::normalize_role_select_policy(options.role_select_policy);
        j["role_pool_extra"] = options.role_pool_extra;
        j["role_support_append"] = options.role_support_append;
        j["role_local_warmup"] = options.role_local_warmup;
        j["role_mid_gap_min"] = options.role_mid_gap_min;
        j["role_far_gap_min"] = options.role_far_gap_min;
        j["role_mid_ratio"] = options.role_mid_ratio;
        j["role_far_ratio"] = options.role_far_ratio;
        j["average_degree"] = total_degree * 1.0 / std::max(1u, n);
        j["setup_ms"] = {
            {"center", detail::ns_to_ms(center_ns)},
            {"label_order", detail::ns_to_ms(label_order_ns)},
            {"reorder", detail::ns_to_ms(reorder_ns)},
            {"centroid_pass", detail::ns_to_ms(centroid_pass_ns)},
            {"parent_chain", detail::ns_to_ms(parent_chain_ns)},
            {"knng_remap", detail::ns_to_ms(knng_remap_ns)},
            {"incoming_build", detail::ns_to_ms(incoming_build_ns)},
            {"header_init", detail::ns_to_ms(header_init_ns)},
            {"graph_finalize", detail::ns_to_ms(finalize_graph_ns)}};
        j["first_pass_avg_us_per_node"] = {
            {"knng", detail::ns_to_us(agg.first_knng_ns) / first_nodes},
            {"window", detail::ns_to_us(agg.first_window_ns) / first_nodes},
            {"seed", detail::ns_to_us(agg.first_seed_ns) / first_nodes},
            {"prune", detail::ns_to_us(agg.first_prune_ns) / first_nodes},
            {"reorder", detail::ns_to_us(agg.first_reorder_ns) / first_nodes}};
        j["first_pass_avg_candidates_per_node"] = {
            {"knng", agg.first_knng_candidates / first_nodes},
            {"window_attempted", agg.first_window_attempted / first_nodes},
            {"window_inserted", agg.first_window_inserted / first_nodes},
            {"bridge", agg.first_bridge_candidates / first_nodes},
            {"pruned", agg.first_pruned_candidates / first_nodes},
            {"seed_batches", agg.seed_batches / first_nodes},
            {"seed_batch_size_mean",
             agg.seed_seeds * 1.0 /
                 std::max<std::uint64_t>(1, agg.seed_batches)},
            {"seed_expanded_mean",
             agg.seed_expanded * 1.0 /
                 std::max<std::uint64_t>(1, agg.seed_batches)},
            {"seed_collected_mean",
             agg.seed_collected * 1.0 /
                 std::max<std::uint64_t>(1, agg.seed_batches)},
            {"seed_neighbor_scans_mean",
             agg.seed_neighbor_scans * 1.0 /
                 std::max<std::uint64_t>(1, agg.seed_batches)}};
        j["reverse_pass_avg_us_per_node"] = {
            {"collect", detail::ns_to_us(agg.reverse_collect_ns) / reverse_nodes},
            {"prune", detail::ns_to_us(agg.reverse_prune_ns) / reverse_nodes},
            {"reorder", detail::ns_to_us(agg.reverse_reorder_ns) / reverse_nodes}};
        j["reverse_pass_avg_candidates_per_node"] = {
            {"core", agg.reverse_core_candidates / reverse_nodes},
            {"window_attempted", agg.reverse_window_attempted / reverse_nodes},
            {"window_inserted", agg.reverse_window_inserted / reverse_nodes},
            {"bridge", agg.reverse_bridge_candidates / reverse_nodes},
            {"pruned", agg.reverse_pruned_candidates / reverse_nodes}};
        std::ofstream jout(options.build_profile_json);
        if (jout.good()) {
            jout << j.dump(2);
        } else {
            spdlog::warn("Failed to write build profile json to {}",
                         options.build_profile_json);
        }
    }
    spdlog::info("Enhanced build finished in {} s",
                 Timer::end("enhanced_build_time") / 1e9);
    return g;
}

}  // namespace TDFANN::EnhancedRNSG
