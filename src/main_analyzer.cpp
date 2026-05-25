#include <PCH.hpp>

#include <Core/Concepts.hpp>
#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/ExFunc.hpp>
#include <Vector/VectorList.hpp>

#include <parallel_hashmap/phmap.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace {

using Clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

struct SummaryStats {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
};

SummaryStats summarize(std::vector<double> values) {
    SummaryStats s;
    if (values.empty()) {
        return s;
    }
    std::sort(values.begin(), values.end());
    s.min = values.front();
    s.max = values.back();
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    auto pct = [&](double p) {
        if (values.size() == 1) {
            return values[0];
        }
        double pos = p * static_cast<double>(values.size() - 1);
        auto idx = static_cast<size_t>(pos);
        auto idx2 = std::min(idx + 1, values.size() - 1);
        double frac = pos - static_cast<double>(idx);
        return values[idx] * (1.0 - frac) + values[idx2] * frac;
    };

    s.p50 = pct(0.50);
    s.p90 = pct(0.90);
    s.p99 = pct(0.99);
    return s;
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double my = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double num = 0.0, dx2 = 0.0, dy2 = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if (dx2 <= 0.0 || dy2 <= 0.0) {
        return 0.0;
    }
    return num / std::sqrt(dx2 * dy2);
}

std::vector<unsigned> sample_ids(unsigned n, unsigned sample, unsigned seed) {
    std::vector<unsigned> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    if (sample >= n) {
        return ids;
    }
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(sample);
    return ids;
}

nlohmann::json stats_to_json(const SummaryStats& s) {
    return {
        {"min", s.min},
        {"max", s.max},
        {"mean", s.mean},
        {"p50", s.p50},
        {"p90", s.p90},
        {"p99", s.p99},
    };
}

struct QueryTrace {
    unsigned query_id = 0;
    std::uint64_t range_size = 0;
    std::uint64_t header_size = 0;
    std::uint64_t start_size = 0;
    std::uint64_t expansions = 0;
    std::uint64_t row_entries_scanned = 0;
    std::uint64_t neighbours_filtered = 0;
    std::uint64_t neighbours_unvisited = 0;
    std::uint64_t raw_pool = 0;
    std::uint64_t neighbours_used = 0;
    std::uint64_t distance_evals = 0;
    std::uint64_t candidate_inserts = 0;
    std::uint64_t early_stop_expansions = 0;
    std::uint64_t visited_nodes = 0;
    double query_ns = 0.0;
};

struct ExactTraceOptions {
    unsigned nav_degree = 16;
    unsigned nav_scan_factor = 4;
    unsigned nav_stall_rounds = 8;
    unsigned nav_front_keep = 8;
    unsigned nav_tail_degree = 0;
    unsigned nav_early_stop_rounds = 0;
    unsigned pick_scan_factor = 1;
    unsigned pick_front_keep = 0;
    unsigned edge_pick_policy = 0;
    unsigned edge_pick_recip_depth = 32;
    double edge_pick_core_ratio = 0.6;
};

struct BenefitCounts {
    std::uint64_t evals = 0;
    std::uint64_t inserts = 0;
    std::uint64_t bests = 0;
};

struct QueryGainTrace {
    unsigned query_id = 0;
    std::uint64_t range_size = 0;
    std::uint64_t header_size = 0;
    std::uint64_t start_size = 0;
    std::uint64_t expansions = 0;
    std::uint64_t raw_pool = 0;
    std::uint64_t neighbours_used = 0;
    std::uint64_t distance_evals = 0;
    std::uint64_t candidate_inserts = 0;
    std::uint64_t best_improvements = 0;
    BenefitCounts prefix;
    BenefitCounts tail;
    BenefitCounts left;
    BenefitCounts right;
    double query_ns = 0.0;
};

inline unsigned normalize_edge_pick_policy(unsigned p) {
    return (p <= 3) ? p : 0;
}

inline void record_benefit(BenefitCounts& c, bool inserted, bool best) {
    c.evals++;
    c.inserts += inserted ? 1 : 0;
    c.bests += best ? 1 : 0;
}

inline nlohmann::json benefit_to_json(const BenefitCounts& c) {
    const double evals = static_cast<double>(c.evals);
    return {
        {"evals", c.evals},
        {"inserts", c.inserts},
        {"bests", c.bests},
        {"insert_rate", c.evals > 0 ? static_cast<double>(c.inserts) / evals
                                    : 0.0},
        {"best_rate", c.evals > 0 ? static_cast<double>(c.bests) / evals
                                  : 0.0},
        {"best_given_insert_rate",
         c.inserts > 0 ? static_cast<double>(c.bests) /
                             static_cast<double>(c.inserts)
                       : 0.0},
    };
}

inline unsigned gap_bucket(unsigned gap) {
    unsigned b = 0;
    while (gap > 1) {
        gap >>= 1;
        ++b;
    }
    return b;
}

inline void append_even_sample(const std::vector<unsigned>& src,
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

template <typename GoalId, TDFANN::Graph::GraphLike G>
QueryTrace trace_beam(const TDFANN::Vector::VectorList<float>& dataset,
                      const G& graph,
                      const GoalId& goal,
                      unsigned k,
                      const std::vector<unsigned>& start_nodes,
                      unsigned beam_size,
                      unsigned trunc_size) {
    QueryTrace t;

    phmap::flat_hash_map<unsigned, float> vis_dis;
    unsigned offset = dataset.size();

    std::vector<std::pair<float, unsigned>> candidates;
    candidates.reserve(std::max<unsigned>(beam_size, start_nodes.size()));
    for (unsigned node : start_nodes) {
        candidates.push_back({0.0f, node});
    }
    if (candidates.empty()) {
        return t;
    }

    dataset.dist_all_into(goal, candidates);
    t.distance_evals += candidates.size();
    std::ranges::sort(candidates);

    for (auto& [dis, id] : candidates) {
        vis_dis[id] = dis;
        id += offset;
    }

    if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }
    candidates.resize(beam_size, {1e100f, candidates[0].second - offset});

    std::vector<std::pair<float, unsigned>> neighbours;
    neighbours.reserve(std::max<unsigned>(1, beam_size * 2));

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        unsigned current_node = candidates[uid].second;
        t.expansions++;

        neighbours.clear();
        std::uint64_t filtered = 0;
        std::uint64_t unvisited = 0;

        for (const auto& x : graph.get_neighbours(current_node)) {
            filtered++;
            if (vis_dis.contains(x.to)) {
                continue;
            }
            unvisited++;
            if (neighbours.size() < trunc_size) {
                neighbours.push_back({0.0f, x.to});
            }
        }

        t.neighbours_filtered += filtered;
        t.neighbours_unvisited += unvisited;
        t.neighbours_used += neighbours.size();

        dataset.dist_all_into(goal, neighbours);
        t.distance_evals += neighbours.size();

        for (const auto& [dist, nto] : neighbours) {
            if (dist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < dist; });
                int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {dist, nto + offset});
                vis_dis.insert({nto, dist});
                t.candidate_inserts++;
            }
        }
    }

    t.visited_nodes = vis_dis.size();
    (void)k;
    return t;
}

template <typename G>
bool is_reciprocal(const G& graph,
                   unsigned current_node,
                   unsigned nid,
                   unsigned depth) {
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

template <typename G>
void select_from_raw_exact(const G& graph,
                           unsigned current_node,
                           const std::vector<unsigned>& raw_nodes,
                           unsigned local_budget,
                           unsigned front_keep,
                           unsigned policy_in,
                           unsigned recip_depth,
                           double core_ratio,
                           std::vector<std::pair<float, unsigned>>& neighbours) {
    const unsigned policy = normalize_edge_pick_policy(policy_in);
    const size_t raw_n = raw_nodes.size();
    const size_t need = std::min<size_t>(local_budget, raw_n);
    if (need == 0) {
        return;
    }

    const size_t prefix_keep =
        std::min<size_t>(std::min<size_t>(front_keep, need), raw_n);

    std::vector<unsigned> picked;
    picked.reserve(need);
    std::vector<char> used(raw_n, 0);
    for (size_t i = 0; i < prefix_keep; ++i) {
        picked.push_back(raw_nodes[i]);
        used[i] = 1;
    }

    if (policy == 0) {
        const size_t remain = need - picked.size();
        if (remain > 0) {
            append_even_sample(raw_nodes, prefix_keep, raw_n, remain, picked,
                               used);
        }
        for (auto nid : picked) {
            neighbours.push_back({0.0f, nid});
        }
        return;
    }

    if (policy == 1) {
        std::vector<unsigned> left, right;
        left.reserve(raw_n);
        right.reserve(raw_n);
        for (size_t i = prefix_keep; i < raw_n; ++i) {
            if (raw_nodes[i] < current_node) {
                left.push_back(raw_nodes[i]);
            } else {
                right.push_back(raw_nodes[i]);
            }
        }
        const size_t remain = need - picked.size();
        for (size_t i = 0; i < remain; ++i) {
            if (i < left.size()) {
                picked.push_back(left[i]);
            } else if ((i - left.size()) < right.size()) {
                picked.push_back(right[i - left.size()]);
            }
        }
    } else if (policy == 2) {
        for (size_t i = prefix_keep; i < raw_n && picked.size() < need; ++i) {
            if (is_reciprocal(graph, current_node, raw_nodes[i], recip_depth)) {
                picked.push_back(raw_nodes[i]);
                used[i] = 1;
            }
        }
        if (picked.size() < need) {
            append_even_sample(raw_nodes, prefix_keep, raw_n, need - picked.size(),
                               picked, used);
        }
    } else {
        std::vector<unsigned> reciprocal;
        std::vector<unsigned> others;
        reciprocal.reserve(raw_n);
        others.reserve(raw_n);
        for (size_t i = prefix_keep; i < raw_n; ++i) {
            if (is_reciprocal(graph, current_node, raw_nodes[i], recip_depth)) {
                reciprocal.push_back(raw_nodes[i]);
            } else {
                others.push_back(raw_nodes[i]);
            }
        }
        const size_t remain = need - picked.size();
        const size_t core_need = std::min(
            remain, static_cast<size_t>(std::llround(remain * core_ratio)));
        for (size_t i = 0; i < core_need && i < reciprocal.size(); ++i) {
            picked.push_back(reciprocal[i]);
        }
        size_t oi = 0;
        while (picked.size() < need && oi < others.size()) {
            picked.push_back(others[oi++]);
        }
        size_t ri = core_need;
        while (picked.size() < need && ri < reciprocal.size()) {
            picked.push_back(reciprocal[ri++]);
        }
    }

    if (picked.size() < need && policy != 0) {
        phmap::flat_hash_set<unsigned> picked_ids;
        picked_ids.reserve(picked.size() * 2 + 8);
        for (auto nid : picked) {
            picked_ids.insert(nid);
        }
        for (size_t i = prefix_keep; i < raw_n && picked.size() < need; ++i) {
            if (!picked_ids.contains(raw_nodes[i])) {
                picked.push_back(raw_nodes[i]);
                used[i] = 1;
            }
        }
    }

    for (auto nid : picked) {
        neighbours.push_back({0.0f, nid});
    }
}

template <typename GoalId>
QueryTrace trace_beam_exact(const TDFANN::Vector::VectorList<float>& dataset,
                            const TDFANN::Graph::TDGraphIndexBase& index,
                            const std::vector<std::uint64_t>& sorted_label,
                            std::uint64_t ql,
                            std::uint64_t qr,
                            const GoalId& goal,
                            unsigned beam_size,
                            unsigned trunc_size,
                            const ExactTraceOptions& opt) {
    QueryTrace t;
    if (beam_size == 0 || trunc_size == 0) {
        return t;
    }

    const auto range_l =
        static_cast<unsigned>(std::ranges::lower_bound(sorted_label, ql) -
                              sorted_label.begin());
    const auto range_r_ex =
        static_cast<unsigned>(std::ranges::upper_bound(sorted_label, qr) -
                              sorted_label.begin());
    if (range_l >= range_r_ex) {
        return t;
    }
    const unsigned range_r = range_r_ex - 1;

    auto g_sub = index(sorted_label, ql, qr);
    auto header = TDFANN::Utils::to_vector(g_sub.get_header());
    std::vector<unsigned> start_nodes;
    if (!header.empty()) {
        start_nodes = header;
    } else {
        start_nodes.push_back(range_l);
    }

    phmap::flat_hash_map<unsigned, float> vis_dis;
    const unsigned offset = dataset.size();

    std::vector<std::pair<float, unsigned>> candidates;
    candidates.reserve(std::max<unsigned>(beam_size, start_nodes.size()));
    for (auto id : start_nodes) {
        candidates.push_back({0.0f, id});
    }
    if (candidates.empty()) {
        return t;
    }

    dataset.dist_all_into(goal, candidates);
    t.distance_evals += candidates.size();
    std::ranges::sort(candidates);
    for (auto& [dis, id] : candidates) {
        vis_dis[id] = dis;
        id += offset;
    }

    if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }
    candidates.resize(beam_size, {1e100f, candidates[0].second - offset});

    std::vector<std::pair<float, unsigned>> neighbours;
    neighbours.reserve(std::max<unsigned>(1, beam_size * 2));
    std::vector<unsigned> raw_nodes;
    raw_nodes.reserve(std::max<unsigned>(trunc_size * 4 + 8, trunc_size + 8));

    const unsigned effective_nav_degree =
        (opt.nav_degree == 0 ? trunc_size : std::min(opt.nav_degree, trunc_size));
    const unsigned effective_scan_factor = std::max(1u, opt.nav_scan_factor);
    const unsigned effective_stall_rounds = std::max(1u, opt.nav_stall_rounds);
    const unsigned effective_front_keep = std::max(1u, opt.nav_front_keep);
    const unsigned effective_tail_degree =
        (opt.nav_tail_degree == 0 ? 0 : std::min(opt.nav_tail_degree, trunc_size));
    const unsigned effective_early_stop_rounds = opt.nav_early_stop_rounds;
    const unsigned effective_pick_scan_factor = std::max(1u, opt.pick_scan_factor);
    const unsigned effective_pick_front_keep =
        (opt.pick_front_keep == 0 ? effective_front_keep : opt.pick_front_keep);

    unsigned stall_rounds = 0;

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        t.expansions++;

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
        if (effective_pick_scan_factor > 1) {
            const auto scaled_scan =
                static_cast<std::uint64_t>(trunc_size) *
                static_cast<std::uint64_t>(effective_pick_scan_factor);
            scan_limit = (scaled_scan >
                          static_cast<std::uint64_t>(
                              std::numeric_limits<unsigned>::max()))
                             ? std::numeric_limits<unsigned>::max()
                             : static_cast<unsigned>(scaled_scan);
        }
        if (local_budget < trunc_size) {
            const unsigned nav_scan_limit = std::min(
                trunc_size,
                std::max(local_budget, local_budget * effective_scan_factor));
            scan_limit = std::max(scan_limit, nav_scan_limit);
        }

        neighbours.clear();
        raw_nodes.clear();
        std::uint64_t scanned = 0;
        std::uint64_t filtered = 0;
        std::uint64_t unvisited = 0;

        for (const auto& x : index.get_neighbours(current_node)) {
            scanned++;
            if (x.to < range_l || x.to > range_r) {
                continue;
            }
            filtered++;
            if (vis_dis.contains(x.to)) {
                continue;
            }
            unvisited++;
            raw_nodes.push_back(x.to);
            if (raw_nodes.size() >= scan_limit) {
                t.early_stop_expansions++;
                break;
            }
        }

        t.row_entries_scanned += scanned;
        t.neighbours_filtered += filtered;
        t.neighbours_unvisited += unvisited;
        t.raw_pool += raw_nodes.size();

        if (raw_nodes.empty()) {
            continue;
        }
        if (raw_nodes.size() <= local_budget) {
            for (auto nid : raw_nodes) {
                neighbours.push_back({0.0f, nid});
            }
        } else {
            select_from_raw_exact(index, current_node, raw_nodes, local_budget,
                                  effective_pick_front_keep,
                                  opt.edge_pick_policy,
                                  opt.edge_pick_recip_depth,
                                  opt.edge_pick_core_ratio, neighbours);
        }

        t.neighbours_used += neighbours.size();
        dataset.dist_all_into(goal, neighbours);
        t.distance_evals += neighbours.size();

        bool improved = false;
        for (const auto& [dist, nto] : neighbours) {
            if (!vis_dis.contains(nto)) {
                vis_dis.insert({nto, dist});
            }
            if (dist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < dist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {dist, nto + offset});
                t.candidate_inserts++;
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

    t.range_size = static_cast<std::uint64_t>(range_r_ex - range_l);
    t.header_size = static_cast<std::uint64_t>(header.size());
    t.start_size = static_cast<std::uint64_t>(start_nodes.size());
    t.visited_nodes = vis_dis.size();
    return t;
}

template <typename GoalId>
QueryGainTrace trace_beam_subgraph_gain(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& index,
    const std::vector<std::uint64_t>& sorted_label,
    std::uint64_t ql,
    std::uint64_t qr,
    const GoalId& goal,
    unsigned beam_size,
    unsigned trunc_size,
    const ExactTraceOptions& opt,
    std::vector<BenefitCounts>& rank_stats,
    std::vector<BenefitCounts>& gap_stats) {
    QueryGainTrace t;
    if (beam_size == 0 || trunc_size == 0) {
        return t;
    }

    const auto range_l =
        static_cast<unsigned>(std::ranges::lower_bound(sorted_label, ql) -
                              sorted_label.begin());
    const auto range_r_ex =
        static_cast<unsigned>(std::ranges::upper_bound(sorted_label, qr) -
                              sorted_label.begin());
    if (range_l >= range_r_ex) {
        return t;
    }

    auto g_sub = index(sorted_label, ql, qr);
    auto header = TDFANN::Utils::to_vector(g_sub.get_header());
    std::vector<unsigned> start_nodes;
    if (!header.empty()) {
        start_nodes = header;
    } else {
        start_nodes.push_back(range_l);
    }

    phmap::flat_hash_map<unsigned, float> vis_dis;
    const unsigned offset = dataset.size();

    std::vector<std::pair<float, unsigned>> candidates;
    candidates.reserve(std::max<unsigned>(beam_size, start_nodes.size()));
    for (auto id : start_nodes) {
        candidates.push_back({0.0f, id});
    }
    if (candidates.empty()) {
        return t;
    }

    dataset.dist_all_into(goal, candidates);
    t.distance_evals += candidates.size();
    std::ranges::sort(candidates);
    for (auto& [dis, id] : candidates) {
        vis_dis[id] = dis;
        id += offset;
    }

    if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }
    candidates.resize(beam_size, {1e100f, candidates[0].second - offset});

    std::vector<std::pair<float, unsigned>> neighbours;
    neighbours.reserve(std::max<unsigned>(1, beam_size * 2));
    std::vector<unsigned> raw_nodes;
    raw_nodes.reserve(std::max<unsigned>(trunc_size * 4 + 8, trunc_size + 8));

    const unsigned effective_nav_degree =
        (opt.nav_degree == 0 ? trunc_size : std::min(opt.nav_degree, trunc_size));
    const unsigned effective_scan_factor = std::max(1u, opt.nav_scan_factor);
    const unsigned effective_stall_rounds = std::max(1u, opt.nav_stall_rounds);
    const unsigned effective_front_keep = std::max(1u, opt.nav_front_keep);
    const unsigned effective_tail_degree =
        (opt.nav_tail_degree == 0 ? 0 : std::min(opt.nav_tail_degree, trunc_size));
    const unsigned effective_early_stop_rounds = opt.nav_early_stop_rounds;
    const unsigned effective_pick_scan_factor = std::max(1u, opt.pick_scan_factor);
    const unsigned effective_pick_front_keep =
        (opt.pick_front_keep == 0 ? effective_front_keep : opt.pick_front_keep);

    unsigned stall_rounds = 0;

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        t.expansions++;

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
        if (effective_pick_scan_factor > 1) {
            const auto scaled_scan =
                static_cast<std::uint64_t>(trunc_size) *
                static_cast<std::uint64_t>(effective_pick_scan_factor);
            scan_limit = (scaled_scan >
                          static_cast<std::uint64_t>(
                              std::numeric_limits<unsigned>::max()))
                             ? std::numeric_limits<unsigned>::max()
                             : static_cast<unsigned>(scaled_scan);
        }
        if (local_budget < trunc_size) {
            const unsigned nav_scan_limit = std::min(
                trunc_size,
                std::max(local_budget, local_budget * effective_scan_factor));
            scan_limit = std::max(scan_limit, nav_scan_limit);
        }

        neighbours.clear();
        raw_nodes.clear();
        std::uint64_t unvisited = 0;

        for (const auto& x : g_sub.get_neighbours(current_node)) {
            if (vis_dis.contains(x.to)) {
                continue;
            }
            unvisited++;
            raw_nodes.push_back(x.to);
            if (raw_nodes.size() >= scan_limit) {
                break;
            }
        }

        t.raw_pool += raw_nodes.size();
        if (raw_nodes.empty()) {
            continue;
        }
        if (raw_nodes.size() <= local_budget) {
            for (auto nid : raw_nodes) {
                neighbours.push_back({0.0f, nid});
            }
        } else {
            select_from_raw_exact(g_sub, current_node, raw_nodes, local_budget,
                                  effective_pick_front_keep,
                                  opt.edge_pick_policy,
                                  opt.edge_pick_recip_depth,
                                  opt.edge_pick_core_ratio, neighbours);
        }

        if (neighbours.empty()) {
            continue;
        }

        const size_t prefix_keep =
            std::min<size_t>(effective_pick_front_keep, neighbours.size());

        dataset.dist_all_into(goal, neighbours);
        t.distance_evals += neighbours.size();
        t.neighbours_used += neighbours.size();

        bool improved = false;
        for (size_t ridx = 0; ridx < neighbours.size(); ++ridx) {
            const auto [dist, nto] = neighbours[ridx];
            const float best_before = candidates.front().first;
            const float tail_before = candidates.back().first;
            const bool best = dist < best_before;
            const bool inserted = dist < tail_before;

            if (!vis_dis.contains(nto)) {
                vis_dis.insert({nto, dist});
            }
            if (inserted) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < dist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {dist, nto + offset});
                t.candidate_inserts++;
                if (best) {
                    t.best_improvements++;
                    improved = true;
                }
            }

            if (ridx < rank_stats.size()) {
                record_benefit(rank_stats[ridx], inserted, best);
            }
            const auto gap = static_cast<unsigned>(
                std::abs(static_cast<long long>(nto) -
                         static_cast<long long>(current_node)));
            const auto gb = gap_bucket(gap);
            if (gb >= gap_stats.size()) {
                gap_stats.resize(gb + 1);
            }
            record_benefit(gap_stats[gb], inserted, best);

            if (ridx < prefix_keep) {
                record_benefit(t.prefix, inserted, best);
            } else {
                record_benefit(t.tail, inserted, best);
            }
            if (nto < current_node) {
                record_benefit(t.left, inserted, best);
            } else {
                record_benefit(t.right, inserted, best);
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

    t.range_size = static_cast<std::uint64_t>(range_r_ex - range_l);
    t.header_size = static_cast<std::uint64_t>(header.size());
    t.start_size = static_cast<std::uint64_t>(start_nodes.size());
    return t;
}

int run_querygain_subgraph(const std::string& dataset_file,
                           const std::string& index_file,
                           const std::string& query_file,
                           const std::string& label_file,
                           const std::string& qrange_file,
                           unsigned qnumber,
                           unsigned beam_size,
                           unsigned trunc_size,
                           unsigned sample_queries,
                           unsigned seed,
                           const ExactTraceOptions& options,
                           const std::string& out_json,
                           const std::string& out_per_query_json) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<QueryGainTrace> traces;
    traces.reserve(ids.size());
    std::vector<BenefitCounts> rank_stats(trunc_size);
    std::vector<BenefitCounts> gap_stats;
    std::uint64_t skipped_empty = 0;

    for (unsigned qid : ids) {
        auto ql = qrange[qid * 2];
        auto qr = qrange[qid * 2 + 1];
        auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();
        if (range_l >= range_r) {
            skipped_empty++;
            continue;
        }

        auto t0 = Clock::now();
        auto tr = trace_beam_subgraph_gain(
            dataset, index, sorted_label, ql, qr, query_list[qid], beam_size,
            trunc_size, options, rank_stats, gap_stats);
        auto t1 = Clock::now();
        tr.query_id = qid;
        tr.query_ns = static_cast<double>(
            std::chrono::duration_cast<ns>(t1 - t0).count());
        traces.push_back(tr);
    }

    auto collect_metric = [&](auto fn) {
        std::vector<double> v;
        v.reserve(traces.size());
        for (const auto& t : traces) {
            v.push_back(fn(t));
        }
        return v;
    };

    auto ns_vec =
        collect_metric([](const QueryGainTrace& t) { return t.query_ns; });
    auto range_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.range_size);
    });
    auto header_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.header_size);
    });
    auto expand_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.expansions);
    });
    auto raw_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.raw_pool);
    });
    auto used_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.neighbours_used);
    });
    auto dist_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.distance_evals);
    });
    auto insert_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.candidate_inserts);
    });
    auto best_vec = collect_metric([](const QueryGainTrace& t) {
        return static_cast<double>(t.best_improvements);
    });

    std::vector<double> raw_per_exp;
    std::vector<double> used_per_exp;
    std::vector<double> ns_per_eval;
    std::vector<double> insert_rate;
    std::vector<double> best_rate;
    std::vector<double> best_given_insert;
    raw_per_exp.reserve(traces.size());
    used_per_exp.reserve(traces.size());
    ns_per_eval.reserve(traces.size());
    insert_rate.reserve(traces.size());
    best_rate.reserve(traces.size());
    best_given_insert.reserve(traces.size());
    for (const auto& t : traces) {
        raw_per_exp.push_back(
            t.expansions > 0
                ? static_cast<double>(t.raw_pool) /
                      static_cast<double>(t.expansions)
                : 0.0);
        used_per_exp.push_back(
            t.expansions > 0
                ? static_cast<double>(t.neighbours_used) /
                      static_cast<double>(t.expansions)
                : 0.0);
        ns_per_eval.push_back(
            t.distance_evals > 0
                ? t.query_ns / static_cast<double>(t.distance_evals)
                : 0.0);
        insert_rate.push_back(
            t.distance_evals > 0
                ? static_cast<double>(t.candidate_inserts) /
                      static_cast<double>(t.distance_evals)
                : 0.0);
        best_rate.push_back(
            t.distance_evals > 0
                ? static_cast<double>(t.best_improvements) /
                      static_cast<double>(t.distance_evals)
                : 0.0);
        best_given_insert.push_back(
            t.candidate_inserts > 0
                ? static_cast<double>(t.best_improvements) /
                      static_cast<double>(t.candidate_inserts)
                : 0.0);
    }

    nlohmann::json rank_json = nlohmann::json::array();
    for (size_t i = 0; i < rank_stats.size(); ++i) {
        auto j = benefit_to_json(rank_stats[i]);
        j["rank"] = i;
        rank_json.push_back(std::move(j));
    }

    nlohmann::json gap_json = nlohmann::json::array();
    for (size_t i = 0; i < gap_stats.size(); ++i) {
        auto j = benefit_to_json(gap_stats[i]);
        const std::uint64_t lo = (i == 0 ? 0ull : (1ull << i));
        const std::uint64_t hi = ((1ull << (i + 1)) - 1ull);
        j["bucket"] = i;
        j["gap_lo"] = lo;
        j["gap_hi"] = hi;
        gap_json.push_back(std::move(j));
    }

    nlohmann::json summary;
    summary["sampled"] = traces.size();
    summary["skipped_empty"] = skipped_empty;
    summary["query_ns"] = stats_to_json(summarize(ns_vec));
    summary["range_size"] = stats_to_json(summarize(range_vec));
    summary["header_size"] = stats_to_json(summarize(header_vec));
    summary["expansions"] = stats_to_json(summarize(expand_vec));
    summary["raw_pool"] = stats_to_json(summarize(raw_vec));
    summary["neigh_used"] = stats_to_json(summarize(used_vec));
    summary["distance_evals"] = stats_to_json(summarize(dist_vec));
    summary["candidate_inserts"] = stats_to_json(summarize(insert_vec));
    summary["best_improvements"] = stats_to_json(summarize(best_vec));
    summary["raw_per_expansion"] = stats_to_json(summarize(raw_per_exp));
    summary["used_per_expansion"] = stats_to_json(summarize(used_per_exp));
    summary["ns_per_distance_eval"] = stats_to_json(summarize(ns_per_eval));
    summary["insert_rate"] = stats_to_json(summarize(insert_rate));
    summary["best_rate"] = stats_to_json(summarize(best_rate));
    summary["best_given_insert_rate"] =
        stats_to_json(summarize(best_given_insert));
    summary["corr_query_ns_range_size"] = correlation(ns_vec, range_vec);
    summary["corr_query_ns_distance_evals"] = correlation(ns_vec, dist_vec);
    summary["corr_query_ns_expansions"] = correlation(ns_vec, expand_vec);
    summary["corr_query_ns_raw_pool"] = correlation(ns_vec, raw_vec);
    summary["corr_query_ns_neigh_used"] = correlation(ns_vec, used_vec);
    summary["prefix"] = benefit_to_json(std::accumulate(
        traces.begin(), traces.end(), BenefitCounts{},
        [](BenefitCounts acc, const QueryGainTrace& t) {
            acc.evals += t.prefix.evals;
            acc.inserts += t.prefix.inserts;
            acc.bests += t.prefix.bests;
            return acc;
        }));
    summary["tail"] = benefit_to_json(std::accumulate(
        traces.begin(), traces.end(), BenefitCounts{},
        [](BenefitCounts acc, const QueryGainTrace& t) {
            acc.evals += t.tail.evals;
            acc.inserts += t.tail.inserts;
            acc.bests += t.tail.bests;
            return acc;
        }));
    summary["left"] = benefit_to_json(std::accumulate(
        traces.begin(), traces.end(), BenefitCounts{},
        [](BenefitCounts acc, const QueryGainTrace& t) {
            acc.evals += t.left.evals;
            acc.inserts += t.left.inserts;
            acc.bests += t.left.bests;
            return acc;
        }));
    summary["right"] = benefit_to_json(std::accumulate(
        traces.begin(), traces.end(), BenefitCounts{},
        [](BenefitCounts acc, const QueryGainTrace& t) {
            acc.evals += t.right.evals;
            acc.inserts += t.right.inserts;
            acc.bests += t.right.bests;
            return acc;
        }));
    summary["rank_benefit"] = std::move(rank_json);
    summary["gap_benefit"] = std::move(gap_json);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_per_query_json.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& t : traces) {
            arr.push_back({
                {"query_id", t.query_id},
                {"query_ns", t.query_ns},
                {"range_size", t.range_size},
                {"header_size", t.header_size},
                {"start_size", t.start_size},
                {"expansions", t.expansions},
                {"raw_pool", t.raw_pool},
                {"neighbours_used", t.neighbours_used},
                {"distance_evals", t.distance_evals},
                {"candidate_inserts", t.candidate_inserts},
                {"best_improvements", t.best_improvements},
                {"prefix", benefit_to_json(t.prefix)},
                {"tail", benefit_to_json(t.tail)},
                {"left", benefit_to_json(t.left)},
                {"right", benefit_to_json(t.right)},
            });
        }
        std::ofstream fout(out_per_query_json);
        fout << arr.dump(2);
    }

    return 0;
}

int run_querytrace(const std::string& dataset_file,
                   const std::string& index_file,
                   const std::string& query_file,
                   const std::string& label_file,
                   const std::string& qrange_file,
                   unsigned qnumber,
                   unsigned beam_size,
                   unsigned trunc_size,
                   unsigned sample_queries,
                   unsigned seed,
                   const std::string& out_json,
                   const std::string& out_per_query_json) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);

    std::vector<QueryTrace> traces;
    traces.reserve(ids.size());
    std::uint64_t skipped_empty = 0;

    for (unsigned qid : ids) {
        auto ql = qrange[qid * 2];
        auto qr = qrange[qid * 2 + 1];
        auto range_l = std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        auto range_r = std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();
        if (range_l >= range_r) {
            skipped_empty++;
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> start_nodes;
        if (!header.empty()) {
            start_nodes = header;
        } else {
            start_nodes.push_back(static_cast<unsigned>(range_l));
        }

        auto t0 = Clock::now();
        auto tr = trace_beam(dataset, g_sub, query_list[qid], qnumber,
                             start_nodes, beam_size, trunc_size);
        auto t1 = Clock::now();

        tr.query_id = qid;
        tr.range_size = static_cast<std::uint64_t>(range_r - range_l);
        tr.header_size = static_cast<std::uint64_t>(header.size());
        tr.start_size = static_cast<std::uint64_t>(start_nodes.size());
        tr.query_ns = static_cast<double>(
            std::chrono::duration_cast<ns>(t1 - t0).count());
        traces.push_back(tr);
    }

    auto collect_metric = [&](auto fn) {
        std::vector<double> v;
        v.reserve(traces.size());
        for (const auto& t : traces) {
            v.push_back(fn(t));
        }
        return v;
    };

    auto ns_vec = collect_metric([](const QueryTrace& t) { return t.query_ns; });
    auto range_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.range_size); });
    auto header_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.header_size); });
    auto visited_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.visited_nodes); });
    auto dist_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.distance_evals); });
    auto expand_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.expansions); });
    auto neigh_used_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.neighbours_used); });
    auto neigh_unvisited_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.neighbours_unvisited); });
    auto neigh_filtered_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.neighbours_filtered); });

    std::vector<double> ns_per_dist;
    ns_per_dist.reserve(traces.size());
    for (const auto& t : traces) {
        ns_per_dist.push_back(t.distance_evals > 0 ? t.query_ns / static_cast<double>(t.distance_evals) : 0.0);
    }

    auto print_stats = [&](const std::string& name, const std::vector<double>& v) {
        auto s = summarize(v);
        spdlog::info("{}: mean={:.4f}, p50={:.4f}, p90={:.4f}, p99={:.4f}, min={:.4f}, max={:.4f}",
                     name, s.mean, s.p50, s.p90, s.p99, s.min, s.max);
    };

    spdlog::info("[querytrace] sampled={} skipped_empty={}", traces.size(), skipped_empty);
    print_stats("query_ns", ns_vec);
    print_stats("range_size", range_vec);
    print_stats("header_size", header_vec);
    print_stats("visited_nodes", visited_vec);
    print_stats("distance_evals", dist_vec);
    print_stats("expansions", expand_vec);
    print_stats("neigh_filtered", neigh_filtered_vec);
    print_stats("neigh_unvisited", neigh_unvisited_vec);
    print_stats("neigh_used", neigh_used_vec);
    print_stats("ns_per_distance_eval", ns_per_dist);

    if (!ns_vec.empty()) {
        double mean_ns = summarize(ns_vec).mean;
        spdlog::info("sample_qps={:.4f}", mean_ns > 0.0 ? 1e9 / mean_ns : 0.0);
    }

    spdlog::info("corr(query_ns, range_size)={:.4f}", correlation(ns_vec, range_vec));
    spdlog::info("corr(query_ns, distance_evals)={:.4f}", correlation(ns_vec, dist_vec));
    spdlog::info("corr(query_ns, visited_nodes)={:.4f}", correlation(ns_vec, visited_vec));
    spdlog::info("corr(query_ns, neigh_used)={:.4f}", correlation(ns_vec, neigh_used_vec));

    nlohmann::json summary;
    summary["sampled"] = traces.size();
    summary["skipped_empty"] = skipped_empty;
    summary["query_ns"] = stats_to_json(summarize(ns_vec));
    summary["range_size"] = stats_to_json(summarize(range_vec));
    summary["header_size"] = stats_to_json(summarize(header_vec));
    summary["visited_nodes"] = stats_to_json(summarize(visited_vec));
    summary["distance_evals"] = stats_to_json(summarize(dist_vec));
    summary["expansions"] = stats_to_json(summarize(expand_vec));
    summary["neigh_filtered"] = stats_to_json(summarize(neigh_filtered_vec));
    summary["neigh_unvisited"] = stats_to_json(summarize(neigh_unvisited_vec));
    summary["neigh_used"] = stats_to_json(summarize(neigh_used_vec));
    summary["ns_per_distance_eval"] = stats_to_json(summarize(ns_per_dist));
    summary["corr_query_ns_range_size"] = correlation(ns_vec, range_vec);
    summary["corr_query_ns_distance_evals"] = correlation(ns_vec, dist_vec);
    summary["corr_query_ns_visited_nodes"] = correlation(ns_vec, visited_vec);
    summary["corr_query_ns_neigh_used"] = correlation(ns_vec, neigh_used_vec);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_per_query_json.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& t : traces) {
            arr.push_back({
                {"query_id", t.query_id},
                {"query_ns", t.query_ns},
                {"range_size", t.range_size},
                {"header_size", t.header_size},
                {"start_size", t.start_size},
                {"expansions", t.expansions},
                {"neighbours_filtered", t.neighbours_filtered},
                {"neighbours_unvisited", t.neighbours_unvisited},
                {"neighbours_used", t.neighbours_used},
                {"distance_evals", t.distance_evals},
                {"candidate_inserts", t.candidate_inserts},
                {"visited_nodes", t.visited_nodes},
            });
        }
        std::ofstream fout(out_per_query_json);
        fout << arr.dump(2);
    }

    return 0;
}

int run_querytrace_exact(const std::string& dataset_file,
                         const std::string& index_file,
                         const std::string& query_file,
                         const std::string& label_file,
                         const std::string& qrange_file,
                         unsigned qnumber,
                         unsigned beam_size,
                         unsigned trunc_size,
                         unsigned sample_queries,
                         unsigned seed,
                         const ExactTraceOptions& options,
                         const std::string& out_json,
                         const std::string& out_per_query_json) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<QueryTrace> traces;
    traces.reserve(ids.size());
    std::uint64_t skipped_empty = 0;

    for (unsigned qid : ids) {
        auto ql = qrange[qid * 2];
        auto qr = qrange[qid * 2 + 1];
        auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();
        if (range_l >= range_r) {
            skipped_empty++;
            continue;
        }

        auto t0 = Clock::now();
        auto tr = trace_beam_exact(dataset, index, sorted_label, ql, qr,
                                   query_list[qid], beam_size, trunc_size,
                                   options);
        auto t1 = Clock::now();
        tr.query_id = qid;
        tr.query_ns = static_cast<double>(
            std::chrono::duration_cast<ns>(t1 - t0).count());
        traces.push_back(tr);
    }

    auto collect_metric = [&](auto fn) {
        std::vector<double> v;
        v.reserve(traces.size());
        for (const auto& t : traces) {
            v.push_back(fn(t));
        }
        return v;
    };

    auto ns_vec = collect_metric([](const QueryTrace& t) { return t.query_ns; });
    auto range_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.range_size); });
    auto header_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.header_size); });
    auto visited_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.visited_nodes); });
    auto dist_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.distance_evals); });
    auto expand_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.expansions); });
    auto scanned_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.row_entries_scanned); });
    auto filtered_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.neighbours_filtered); });
    auto unvisited_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.neighbours_unvisited); });
    auto raw_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.raw_pool); });
    auto used_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.neighbours_used); });
    auto inserted_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.candidate_inserts); });
    auto early_vec = collect_metric([](const QueryTrace& t) { return static_cast<double>(t.early_stop_expansions); });

    std::vector<double> ns_per_eval;
    std::vector<double> scanned_per_exp;
    std::vector<double> used_per_exp;
    std::vector<double> raw_per_exp;
    std::vector<double> insert_rate;
    ns_per_eval.reserve(traces.size());
    scanned_per_exp.reserve(traces.size());
    used_per_exp.reserve(traces.size());
    raw_per_exp.reserve(traces.size());
    insert_rate.reserve(traces.size());
    for (const auto& t : traces) {
        ns_per_eval.push_back(t.distance_evals > 0
                                  ? t.query_ns / static_cast<double>(t.distance_evals)
                                  : 0.0);
        scanned_per_exp.push_back(
            t.expansions > 0
                ? static_cast<double>(t.row_entries_scanned) /
                      static_cast<double>(t.expansions)
                : 0.0);
        used_per_exp.push_back(
            t.expansions > 0
                ? static_cast<double>(t.neighbours_used) /
                      static_cast<double>(t.expansions)
                : 0.0);
        raw_per_exp.push_back(
            t.expansions > 0
                ? static_cast<double>(t.raw_pool) /
                      static_cast<double>(t.expansions)
                : 0.0);
        insert_rate.push_back(
            t.neighbours_used > 0
                ? static_cast<double>(t.candidate_inserts) /
                      static_cast<double>(t.neighbours_used)
                : 0.0);
    }

    nlohmann::json summary;
    summary["sampled"] = traces.size();
    summary["skipped_empty"] = skipped_empty;
    summary["query_ns"] = stats_to_json(summarize(ns_vec));
    summary["range_size"] = stats_to_json(summarize(range_vec));
    summary["header_size"] = stats_to_json(summarize(header_vec));
    summary["visited_nodes"] = stats_to_json(summarize(visited_vec));
    summary["distance_evals"] = stats_to_json(summarize(dist_vec));
    summary["expansions"] = stats_to_json(summarize(expand_vec));
    summary["row_entries_scanned"] = stats_to_json(summarize(scanned_vec));
    summary["neigh_filtered"] = stats_to_json(summarize(filtered_vec));
    summary["neigh_unvisited"] = stats_to_json(summarize(unvisited_vec));
    summary["raw_pool"] = stats_to_json(summarize(raw_vec));
    summary["neigh_used"] = stats_to_json(summarize(used_vec));
    summary["candidate_inserts"] = stats_to_json(summarize(inserted_vec));
    summary["early_stop_expansions"] = stats_to_json(summarize(early_vec));
    summary["ns_per_distance_eval"] = stats_to_json(summarize(ns_per_eval));
    summary["scanned_per_expansion"] = stats_to_json(summarize(scanned_per_exp));
    summary["raw_per_expansion"] = stats_to_json(summarize(raw_per_exp));
    summary["used_per_expansion"] = stats_to_json(summarize(used_per_exp));
    summary["insert_rate"] = stats_to_json(summarize(insert_rate));
    summary["corr_query_ns_range_size"] = correlation(ns_vec, range_vec);
    summary["corr_query_ns_distance_evals"] = correlation(ns_vec, dist_vec);
    summary["corr_query_ns_visited_nodes"] = correlation(ns_vec, visited_vec);
    summary["corr_query_ns_neigh_used"] = correlation(ns_vec, used_vec);
    summary["corr_query_ns_row_entries_scanned"] =
        correlation(ns_vec, scanned_vec);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_per_query_json.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& t : traces) {
            arr.push_back({
                {"query_id", t.query_id},
                {"query_ns", t.query_ns},
                {"range_size", t.range_size},
                {"header_size", t.header_size},
                {"start_size", t.start_size},
                {"expansions", t.expansions},
                {"row_entries_scanned", t.row_entries_scanned},
                {"neighbours_filtered", t.neighbours_filtered},
                {"neighbours_unvisited", t.neighbours_unvisited},
                {"raw_pool", t.raw_pool},
                {"neighbours_used", t.neighbours_used},
                {"distance_evals", t.distance_evals},
                {"candidate_inserts", t.candidate_inserts},
                {"early_stop_expansions", t.early_stop_expansions},
                {"visited_nodes", t.visited_nodes},
            });
        }
        std::ofstream fout(out_per_query_json);
        fout << arr.dump(2);
    }

    return 0;
}

int run_graphstats(const std::string& index_file,
                  const std::string& label_file,
                  const std::string& qrange_file,
                  unsigned sample_queries,
                  unsigned sample_nodes_per_query,
                  unsigned seed,
                  const std::string& out_json) {
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto sorted_label = TDFANN::Utils::sorted_vec(label);

    std::vector<double> deg_all;
    deg_all.reserve(label.size());
    for (unsigned i = 0; i < label.size(); ++i) {
        deg_all.push_back(static_cast<double>(index.get_neighbours(i).size()));
    }

    std::vector<double> range_sizes;
    std::vector<double> header_sizes;
    std::vector<double> sampled_full_deg;
    std::vector<double> sampled_inrange_deg;
    std::vector<double> keep_ratios;
    std::uint64_t skipped_empty = 0;

    if (!qrange_file.empty()) {
        auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
        if (qrange.size() % 2 != 0) {
            throw std::runtime_error("qrange size must be even");
        }
        unsigned qn = qrange.size() / 2;
        auto qids = sample_ids(qn, sample_queries, seed);
        std::mt19937 rng(seed + 17);

        for (unsigned qid : qids) {
            auto ql = qrange[qid * 2];
            auto qr = qrange[qid * 2 + 1];
            auto range_l = std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
            auto range_r = std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();
            if (range_l >= range_r) {
                skipped_empty++;
                continue;
            }

            auto g_sub = index(sorted_label, ql, qr);
            auto header = TDFANN::Utils::to_vector(g_sub.get_header());

            unsigned rsize = static_cast<unsigned>(range_r - range_l);
            range_sizes.push_back(static_cast<double>(rsize));
            header_sizes.push_back(static_cast<double>(header.size()));

            unsigned samples = std::min(sample_nodes_per_query, rsize);
            std::uniform_int_distribution<unsigned> dis(
                static_cast<unsigned>(range_l), static_cast<unsigned>(range_r - 1));
            for (unsigned i = 0; i < samples; ++i) {
                unsigned node = dis(rng);
                const auto& nbrs = index.get_neighbours(node);
                unsigned full_deg = nbrs.size();
                unsigned in_deg = 0;
                for (const auto& x : nbrs) {
                    if (x.to >= static_cast<unsigned>(range_l) &&
                        x.to < static_cast<unsigned>(range_r)) {
                        in_deg++;
                    }
                }
                sampled_full_deg.push_back(static_cast<double>(full_deg));
                sampled_inrange_deg.push_back(static_cast<double>(in_deg));
                if (full_deg > 0) {
                    keep_ratios.push_back(static_cast<double>(in_deg) /
                                          static_cast<double>(full_deg));
                }
            }
        }
    }

    auto print_stats = [&](const std::string& name, const std::vector<double>& v) {
        auto s = summarize(v);
        spdlog::info("{}: mean={:.4f}, p50={:.4f}, p90={:.4f}, p99={:.4f}, min={:.4f}, max={:.4f}",
                     name, s.mean, s.p50, s.p90, s.p99, s.min, s.max);
    };

    spdlog::info("[graphstats] nodes={}", label.size());
    print_stats("global_degree", deg_all);
    if (!range_sizes.empty()) {
        spdlog::info("[graphstats] sampled_ranges={} skipped_empty={}",
                     range_sizes.size(), skipped_empty);
        print_stats("range_size", range_sizes);
        print_stats("header_size", header_sizes);
        print_stats("sampled_full_degree", sampled_full_deg);
        print_stats("sampled_inrange_degree", sampled_inrange_deg);
        print_stats("sampled_keep_ratio", keep_ratios);
    }

    nlohmann::json j;
    j["nodes"] = label.size();
    j["global_degree"] = stats_to_json(summarize(deg_all));
    j["sampled_ranges"] = range_sizes.size();
    j["skipped_empty"] = skipped_empty;
    j["range_size"] = stats_to_json(summarize(range_sizes));
    j["header_size"] = stats_to_json(summarize(header_sizes));
    j["sampled_full_degree"] = stats_to_json(summarize(sampled_full_deg));
    j["sampled_inrange_degree"] = stats_to_json(summarize(sampled_inrange_deg));
    j["sampled_keep_ratio"] = stats_to_json(summarize(keep_ratios));

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << j.dump(2);
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"RNSG query/graph analyzer"};

    auto qt = app.add_subcommand("querytrace", "Trace query search process metrics");
    std::string qt_dataset, qt_index, qt_query, qt_label, qt_qrange;
    std::string qt_out, qt_per_query;
    unsigned qt_k = 10, qt_beam = 120, qt_trunc = 50;
    unsigned qt_samples = 200, qt_seed = 20260322;
    qt->add_option("-d,--dataset_file", qt_dataset)->required();
    qt->add_option("-i,--index_file", qt_index)->required();
    qt->add_option("-q,--query_file", qt_query)->required();
    qt->add_option("-l,--label_file", qt_label)->required();
    qt->add_option("-Q,--qrange_file", qt_qrange)->required();
    qt->add_option("-n,--qnumber", qt_k)->required();
    qt->add_option("-s,--beam_size", qt_beam)->required();
    qt->add_option("-t,--trunc_size", qt_trunc)->required();
    qt->add_option("--sample_queries", qt_samples);
    qt->add_option("--seed", qt_seed);
    qt->add_option("--out_json", qt_out);
    qt->add_option("--out_per_query_json", qt_per_query);

    auto qte = app.add_subcommand(
        "querytrace_exact",
        "Trace exact beam_search_range metrics with nav/pick settings");
    std::string qte_dataset, qte_index, qte_query, qte_label, qte_qrange;
    std::string qte_out, qte_per_query, qte_edge_pick_policy = "prefix";
    unsigned qte_k = 10, qte_beam = 120, qte_trunc = 50;
    unsigned qte_samples = 200, qte_seed = 20260329;
    unsigned qte_nav_degree = 16, qte_nav_scan_factor = 4,
             qte_nav_stall_rounds = 8, qte_nav_front_keep = 8,
             qte_nav_tail_degree = 0, qte_nav_early_stop_rounds = 0,
             qte_pick_scan_factor = 1, qte_pick_front_keep = 0,
             qte_edge_pick_recip_depth = 32;
    double qte_edge_pick_core_ratio = 0.6;
    qte->add_option("-d,--dataset_file", qte_dataset)->required();
    qte->add_option("-i,--index_file", qte_index)->required();
    qte->add_option("-q,--query_file", qte_query)->required();
    qte->add_option("-l,--label_file", qte_label)->required();
    qte->add_option("-Q,--qrange_file", qte_qrange)->required();
    qte->add_option("-n,--qnumber", qte_k)->required();
    qte->add_option("-s,--beam_size", qte_beam)->required();
    qte->add_option("-t,--trunc_size", qte_trunc)->required();
    qte->add_option("--sample_queries", qte_samples);
    qte->add_option("--seed", qte_seed);
    qte->add_option("--nav_degree", qte_nav_degree);
    qte->add_option("--nav_scan_factor", qte_nav_scan_factor);
    qte->add_option("--nav_stall_rounds", qte_nav_stall_rounds);
    qte->add_option("--nav_front_keep", qte_nav_front_keep);
    qte->add_option("--nav_tail_degree", qte_nav_tail_degree);
    qte->add_option("--nav_early_stop_rounds", qte_nav_early_stop_rounds);
    qte->add_option("--pick_scan_factor", qte_pick_scan_factor);
    qte->add_option("--pick_front_keep", qte_pick_front_keep);
    qte->add_option("--edge_pick_policy", qte_edge_pick_policy)
        ->check(CLI::IsMember({"prefix", "side", "reciprocal", "corebridge"}));
    qte->add_option("--edge_pick_recip_depth", qte_edge_pick_recip_depth);
    qte->add_option("--edge_pick_core_ratio", qte_edge_pick_core_ratio);
    qte->add_option("--out_json", qte_out);
    qte->add_option("--out_per_query_json", qte_per_query);

    auto qgs = app.add_subcommand(
        "querygain_subgraph",
        "Trace default subgraph-path distance-eval gain metrics");
    std::string qgs_dataset, qgs_index, qgs_query, qgs_label, qgs_qrange;
    std::string qgs_out, qgs_per_query, qgs_edge_pick_policy = "prefix";
    unsigned qgs_k = 10, qgs_beam = 120, qgs_trunc = 50;
    unsigned qgs_samples = 200, qgs_seed = 20260401;
    unsigned qgs_nav_degree = 16, qgs_nav_scan_factor = 4,
             qgs_nav_stall_rounds = 8, qgs_nav_front_keep = 8,
             qgs_nav_tail_degree = 0, qgs_nav_early_stop_rounds = 0,
             qgs_pick_scan_factor = 1, qgs_pick_front_keep = 0,
             qgs_edge_pick_recip_depth = 32;
    double qgs_edge_pick_core_ratio = 0.6;
    qgs->add_option("-d,--dataset_file", qgs_dataset)->required();
    qgs->add_option("-i,--index_file", qgs_index)->required();
    qgs->add_option("-q,--query_file", qgs_query)->required();
    qgs->add_option("-l,--label_file", qgs_label)->required();
    qgs->add_option("-Q,--qrange_file", qgs_qrange)->required();
    qgs->add_option("-n,--qnumber", qgs_k)->required();
    qgs->add_option("-s,--beam_size", qgs_beam)->required();
    qgs->add_option("-t,--trunc_size", qgs_trunc)->required();
    qgs->add_option("--sample_queries", qgs_samples);
    qgs->add_option("--seed", qgs_seed);
    qgs->add_option("--nav_degree", qgs_nav_degree);
    qgs->add_option("--nav_scan_factor", qgs_nav_scan_factor);
    qgs->add_option("--nav_stall_rounds", qgs_nav_stall_rounds);
    qgs->add_option("--nav_front_keep", qgs_nav_front_keep);
    qgs->add_option("--nav_tail_degree", qgs_nav_tail_degree);
    qgs->add_option("--nav_early_stop_rounds", qgs_nav_early_stop_rounds);
    qgs->add_option("--pick_scan_factor", qgs_pick_scan_factor);
    qgs->add_option("--pick_front_keep", qgs_pick_front_keep);
    qgs->add_option("--edge_pick_policy", qgs_edge_pick_policy)
        ->check(CLI::IsMember({"prefix", "side", "reciprocal", "corebridge"}));
    qgs->add_option("--edge_pick_recip_depth", qgs_edge_pick_recip_depth);
    qgs->add_option("--edge_pick_core_ratio", qgs_edge_pick_core_ratio);
    qgs->add_option("--out_json", qgs_out);
    qgs->add_option("--out_per_query_json", qgs_per_query);

    auto gs = app.add_subcommand("graphstats", "Analyze graph degree/filter access metrics");
    std::string gs_index, gs_label, gs_qrange, gs_out;
    unsigned gs_samples = 200, gs_nodes = 256, gs_seed = 20260322;
    gs->add_option("-i,--index_file", gs_index)->required();
    gs->add_option("-l,--label_file", gs_label)->required();
    gs->add_option("-Q,--qrange_file", gs_qrange);
    gs->add_option("--sample_queries", gs_samples);
    gs->add_option("--sample_nodes_per_query", gs_nodes);
    gs->add_option("--seed", gs_seed);
    gs->add_option("--out_json", gs_out);

    app.require_subcommand(1);
    CLI11_PARSE(app, argc, argv);

    try {
        if (*qt) {
            return run_querytrace(qt_dataset, qt_index, qt_query, qt_label, qt_qrange,
                                  qt_k, qt_beam, qt_trunc, qt_samples, qt_seed,
                                  qt_out, qt_per_query);
        }
        if (*qte) {
            const unsigned edge_pick_policy =
                qte_edge_pick_policy == "side"
                    ? 1u
                    : (qte_edge_pick_policy == "reciprocal"
                           ? 2u
                           : (qte_edge_pick_policy == "corebridge" ? 3u
                                                                      : 0u));
            ExactTraceOptions options;
            options.nav_degree = qte_nav_degree;
            options.nav_scan_factor = qte_nav_scan_factor;
            options.nav_stall_rounds = qte_nav_stall_rounds;
            options.nav_front_keep = qte_nav_front_keep;
            options.nav_tail_degree = qte_nav_tail_degree;
            options.nav_early_stop_rounds = qte_nav_early_stop_rounds;
            options.pick_scan_factor = qte_pick_scan_factor;
            options.pick_front_keep = qte_pick_front_keep;
            options.edge_pick_policy = edge_pick_policy;
            options.edge_pick_recip_depth = qte_edge_pick_recip_depth;
            options.edge_pick_core_ratio = qte_edge_pick_core_ratio;
            return run_querytrace_exact(
                qte_dataset, qte_index, qte_query, qte_label, qte_qrange,
                qte_k, qte_beam, qte_trunc, qte_samples, qte_seed, options,
                qte_out, qte_per_query);
        }
        if (*qgs) {
            const unsigned edge_pick_policy =
                qgs_edge_pick_policy == "side"
                    ? 1u
                    : (qgs_edge_pick_policy == "reciprocal"
                           ? 2u
                           : (qgs_edge_pick_policy == "corebridge" ? 3u
                                                                      : 0u));
            ExactTraceOptions options;
            options.nav_degree = qgs_nav_degree;
            options.nav_scan_factor = qgs_nav_scan_factor;
            options.nav_stall_rounds = qgs_nav_stall_rounds;
            options.nav_front_keep = qgs_nav_front_keep;
            options.nav_tail_degree = qgs_nav_tail_degree;
            options.nav_early_stop_rounds = qgs_nav_early_stop_rounds;
            options.pick_scan_factor = qgs_pick_scan_factor;
            options.pick_front_keep = qgs_pick_front_keep;
            options.edge_pick_policy = edge_pick_policy;
            options.edge_pick_recip_depth = qgs_edge_pick_recip_depth;
            options.edge_pick_core_ratio = qgs_edge_pick_core_ratio;
            return run_querygain_subgraph(
                qgs_dataset, qgs_index, qgs_query, qgs_label, qgs_qrange,
                qgs_k, qgs_beam, qgs_trunc, qgs_samples, qgs_seed, options,
                qgs_out, qgs_per_query);
        }
        if (*gs) {
            return run_graphstats(gs_index, gs_label, gs_qrange,
                                  gs_samples, gs_nodes, gs_seed, gs_out);
        }
    } catch (const std::exception& e) {
        spdlog::error("analyzer failed: {}", e.what());
        return 1;
    }

    return 0;
}
