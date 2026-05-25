#include <PCH.hpp>

#include <Core/Concepts.hpp>
#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/ExFunc.hpp>
#include <Vector/VectorList.hpp>

#include <RNSG/Searcher.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>

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
        const double pos = p * static_cast<double>(values.size() - 1);
        const auto idx = static_cast<size_t>(pos);
        const auto idx2 = std::min(idx + 1, values.size() - 1);
        const double frac = pos - static_cast<double>(idx);
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
    const double mx = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    const double my = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double num = 0.0;
    double dx2 = 0.0;
    double dy2 = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
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
        {"min", s.min}, {"max", s.max}, {"mean", s.mean},
        {"p50", s.p50}, {"p90", s.p90}, {"p99", s.p99},
    };
}

struct ExpansionProbe {
    unsigned step = 0;
    unsigned current_node = 0;
    std::uint64_t unvisited_deg = 0;
    std::uint64_t scanned_deg = 0;
    std::uint64_t selected_deg = 0;
    double scan_coverage = 0.0;
    double oracle_best_rank_in_adj = 0.0;
    double selected_rank_pct_mean_global = 0.0;
    double selected_gap_mean_ratio = 0.0;
    double oracle_gap_mean_ratio = 0.0;
    double selected_span_ratio = 0.0;
    double oracle_span_ratio = 0.0;
    double selected_far_ratio = 0.0;
    double oracle_far_ratio = 0.0;
};

struct RunDiag {
    std::vector<std::pair<float, unsigned>> result;
    std::vector<ExpansionProbe> expansion_rows;

    std::uint64_t expansions = 0;
    std::uint64_t neighbours_total = 0;
    std::uint64_t neighbours_unvisited = 0;
    std::uint64_t neighbours_selected = 0;
    std::uint64_t neighbours_selected_extra = 0;
    std::uint64_t neighbours_selected_switch = 0;
    std::uint64_t neighbours_selected_dom = 0;
    std::uint64_t distance_evals = 0;
    // Additional pairwise distance calls (e.g., greedy pruning), not used in
    // the legacy truncation flow but kept for new experiments.
    std::uint64_t distance_evals_pair = 0;
    // Additional auxiliary distance calls that are not query-to-node evals,
    // such as source-to-neighbour ordering for dynamic pruning experiments.
    std::uint64_t distance_evals_aux = 0;
    // Cheap metadata predicate checks, e.g. dominance-tag presence tests.
    std::uint64_t dominance_checks = 0;
    std::uint64_t visited_nodes = 0;

    std::uint64_t oracle_best_missed = 0;
    std::uint64_t miss_not_scanned = 0;
    std::uint64_t miss_scanned_not_selected = 0;
    std::uint64_t topk_hits = 0;
    std::uint64_t topk_total = 0;
    std::uint64_t better_unselected_than_worst = 0;
    std::uint64_t oracle_rank_samples = 0;
    std::uint64_t oracle_rank_missed_samples = 0;
    double oracle_rank_sum = 0.0;
    double oracle_rank_missed_sum = 0.0;

    double query_ns = 0.0;
};

struct DiagConfig {
    unsigned beam_size = 120;
    unsigned trunc_size = 50;
    unsigned nav_degree = 0;
    unsigned nav_scan_factor = 4;
    unsigned nav_stall_rounds = 8;
    unsigned nav_front_keep = 8;
    unsigned nav_tail_degree = 0;
    unsigned nav_early_stop_rounds = 0;
    double bridge_gap_ratio = 0.125;
};

struct DomTagRow;
struct DomTagCache;

struct LaneQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall_trunc = 0.0;
    double recall_lane = 0.0;
    double dist_eval_trunc = 0.0;
    double dist_eval_lane = 0.0;
    double kept_degree_trunc = 0.0;
    double kept_degree_lane = 0.0;
    double switch_added_lane = 0.0;
    double dom_added_lane = 0.0;
    double dominance_checks_lane = 0.0;
};

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_diag(const TDFANN::Vector::VectorList<float>& dataset,
                        const G& graph,
                        const GoalId& goal,
                        unsigned k,
                        const std::vector<unsigned>& start_nodes,
                        const DiagConfig& cfg,
                        unsigned range_l = 0,
                        unsigned range_r_exclusive = 0);

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_greedy_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned greedy_degree);

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_strict_rng_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned rng_cap = 0);

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_two_stage(
    const TDFANN::Vector::VectorList<float>& dataset,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_route,
    unsigned trunc_route,
    unsigned route_expansions,
    unsigned beam_refine,
    unsigned trunc_refine);

template <typename GoalId>
RunDiag beam_trace_domtag_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& base_graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned range_l,
    unsigned range_r,
    unsigned active_cap,
    DomTagCache& cache);

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_mixdom_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& base_graph,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned local_offset,
    unsigned local_keep,
    unsigned dom_keep,
    unsigned range_l,
    unsigned range_r,
    DomTagCache& cache);

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_lane_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& base_graph,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned local_keep,
    unsigned switch_skip,
    unsigned switch_keep,
    unsigned dom_keep,
    unsigned range_l,
    unsigned range_r,
    double switch_gap_lo_ratio,
    double switch_gap_hi_ratio,
    DomTagCache& cache);

double topk_overlap(const std::vector<std::pair<float, unsigned>>& a,
                    const std::vector<std::pair<float, unsigned>>& b,
                    unsigned k) {
    if (k == 0) {
        return 0.0;
    }
    const unsigned ka = std::min<unsigned>(k, a.size());
    const unsigned kb = std::min<unsigned>(k, b.size());
    unsigned hit = 0;
    for (unsigned i = 0; i < ka; ++i) {
        for (unsigned j = 0; j < kb; ++j) {
            if (a[i].second == b[j].second) {
                hit++;
                break;
            }
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

double topk_recall_vs_gt(const std::vector<std::pair<float, unsigned>>& res,
                         const std::vector<unsigned>& gt_ids,
                         unsigned k) {
    if (k == 0) {
        return 0.0;
    }
    const unsigned kr = std::min<unsigned>(k, res.size());
    unsigned hit = 0;
    for (unsigned i = 0; i < kr; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            if (res[i].second == gt_ids[j]) {
                hit++;
                break;
            }
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

struct GreedyQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall_trunc = 0.0;
    double recall_greedy = 0.0;
    double dist_eval_trunc = 0.0;
    double dist_eval_greedy = 0.0;
    double dist_eval_greedy_pair = 0.0;
    double full_degree_greedy = 0.0;
    double kept_degree_trunc = 0.0;
    double kept_degree_greedy = 0.0;
};

struct RngQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall_trunc = 0.0;
    double recall_rng = 0.0;
    double dist_eval_trunc = 0.0;
    double dist_eval_rng_goal = 0.0;
    double dist_eval_rng_aux = 0.0;
    double dist_eval_rng_pair = 0.0;
    double full_degree_rng = 0.0;
    double kept_degree_trunc = 0.0;
    double kept_degree_rng = 0.0;
};

struct TwoStageQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall_base = 0.0;
    double recall_stage = 0.0;
    double dist_eval_base = 0.0;
    double dist_eval_stage = 0.0;
    double kept_degree_base = 0.0;
    double kept_degree_stage = 0.0;
};

struct DomTagQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall_trunc = 0.0;
    double recall_tag = 0.0;
    double dist_eval_trunc = 0.0;
    double dist_eval_tag = 0.0;
    double kept_degree_trunc = 0.0;
    double kept_degree_tag = 0.0;
    double full_degree_tag = 0.0;
    double dominance_checks_tag = 0.0;
};

struct MixDomQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall_trunc = 0.0;
    double recall_mix = 0.0;
    double dist_eval_trunc = 0.0;
    double dist_eval_mix = 0.0;
    double kept_degree_trunc = 0.0;
    double kept_degree_mix = 0.0;
    double dom_added_mix = 0.0;
    double dominance_checks_mix = 0.0;
};

struct DomTagRow {
    std::vector<unsigned> nodes_sorted;
    std::vector<std::vector<unsigned>> dominators;
    std::uint64_t tag_arcs = 0;
};

struct DomTagCache {
    const TDFANN::Vector::VectorList<float>& dataset;
    const TDFANN::Graph::TDGraphIndexBase& base_graph;
    unsigned dominator_cap = 0;

    std::unordered_map<unsigned, DomTagRow> rows;
    std::uint64_t rows_built = 0;
    std::uint64_t edges_indexed = 0;
    std::uint64_t tag_arcs_total = 0;

    DomTagCache(const TDFANN::Vector::VectorList<float>& dataset_,
                const TDFANN::Graph::TDGraphIndexBase& base_graph_,
                unsigned dominator_cap_)
        : dataset(dataset_),
          base_graph(base_graph_),
          dominator_cap(dominator_cap_) {}

    const DomTagRow& get(unsigned node) {
        auto it = rows.find(node);
        if (it != rows.end()) {
            return it->second;
        }

        std::vector<std::pair<float, unsigned>> ordered;
        for (const auto& x : base_graph.get_neighbours(node)) {
            ordered.push_back({0.0f, x.to});
        }
        if (!ordered.empty()) {
            dataset.dist_all_into(node, ordered);
            std::ranges::sort(ordered);
        }

        DomTagRow row;
        row.nodes_sorted.reserve(ordered.size());
        row.dominators.resize(ordered.size());
        for (size_t i = 0; i < ordered.size(); ++i) {
            row.nodes_sorted.push_back(ordered[i].second);
        }

        for (size_t i = 0; i < ordered.size(); ++i) {
            const float d_now = ordered[i].first;
            auto& doms = row.dominators[i];
            for (size_t j = 0; j < i; ++j) {
                const float d_prev = ordered[j].first;
                if (!(d_prev < d_now)) {
                    continue;
                }
                const float d_ij = dataset.dist(ordered[i].second, ordered[j].second);
                if (d_ij < d_now) {
                    doms.push_back(static_cast<unsigned>(j));
                    row.tag_arcs++;
                    if (dominator_cap > 0 && doms.size() >= dominator_cap) {
                        break;
                    }
                }
            }
        }

        rows_built++;
        edges_indexed += ordered.size();
        tag_arcs_total += row.tag_arcs;
        auto [ins, _] = rows.emplace(node, std::move(row));
        return ins->second;
    }
};

int run_greedy_check(const std::string& dataset_file,
                     const std::string& index_file,
                     const std::string& query_file,
                     const std::string& label_file,
                     const std::string& qrange_file,
                     const std::string& gt_file,
                     unsigned qnumber,
                     unsigned beam_size,
                     unsigned trunc_size,
                     unsigned greedy_degree,
                     unsigned sample_queries,
                     unsigned seed,
                     const std::string& out_json,
                     const std::string& out_md) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    std::vector<unsigned> gt;
    const bool has_gt = !gt_file.empty();
    if (has_gt) {
        gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
        if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
            throw std::runtime_error("groundtruth size mismatch");
        }
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    if (has_gt) {
        for (auto& id : gt) {
            if (id >= pos.size()) {
                throw std::runtime_error("groundtruth id out of range");
            }
            id = pos[id];
        }
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<GreedyQueryReport> reports;
    reports.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();

        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_trunc{};
        cfg_trunc.beam_size = beam_size;
        cfg_trunc.trunc_size = trunc_size;

        auto trunc_run = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_trunc,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto greedy_run =
            beam_trace_greedy_prune(dataset, g_sub, query_list[ui], qnumber,
                                    starts, beam_size, greedy_degree);

        GreedyQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.dist_eval_trunc = static_cast<double>(trunc_run.distance_evals);
        r.dist_eval_greedy = static_cast<double>(greedy_run.distance_evals);
        r.dist_eval_greedy_pair =
            static_cast<double>(greedy_run.distance_evals_pair);
        r.full_degree_greedy =
            greedy_run.expansions == 0
                ? 0.0
                : static_cast<double>(greedy_run.neighbours_total) /
                      static_cast<double>(greedy_run.expansions);
        r.kept_degree_trunc =
            trunc_run.expansions == 0
                ? 0.0
                : static_cast<double>(trunc_run.neighbours_selected) /
                      static_cast<double>(trunc_run.expansions);
        r.kept_degree_greedy =
            greedy_run.expansions == 0
                ? 0.0
                : static_cast<double>(greedy_run.neighbours_selected) /
                      static_cast<double>(greedy_run.expansions);

        if (has_gt) {
            std::vector<unsigned> gt_ids(qnumber);
            for (unsigned j = 0; j < qnumber; ++j) {
                gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
            }
            r.recall_trunc =
                topk_recall_vs_gt(trunc_run.result, gt_ids, qnumber);
            r.recall_greedy =
                topk_recall_vs_gt(greedy_run.result, gt_ids, qnumber);
        }

        reports.push_back(r);
        if (((t + 1) % 32 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[greedy] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_trunc_vec =
        collect([](const GreedyQueryReport& r) { return r.recall_trunc; });
    auto recall_greedy_vec =
        collect([](const GreedyQueryReport& r) { return r.recall_greedy; });
    auto dist_trunc_vec =
        collect([](const GreedyQueryReport& r) { return r.dist_eval_trunc; });
    auto dist_greedy_vec =
        collect([](const GreedyQueryReport& r) { return r.dist_eval_greedy; });
    auto dist_greedy_pair_vec = collect(
        [](const GreedyQueryReport& r) { return r.dist_eval_greedy_pair; });
    auto full_degree_greedy_vec = collect(
        [](const GreedyQueryReport& r) { return r.full_degree_greedy; });
    auto kept_degree_trunc_vec =
        collect([](const GreedyQueryReport& r) { return r.kept_degree_trunc; });
    auto kept_degree_greedy_vec = collect(
        [](const GreedyQueryReport& r) { return r.kept_degree_greedy; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["has_groundtruth"] = has_gt;
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_size", trunc_size},
        {"greedy_degree", greedy_degree},
    };
    summary["recall_trunc"] = stats_to_json(summarize(recall_trunc_vec));
    summary["recall_greedy"] = stats_to_json(summarize(recall_greedy_vec));
    summary["dist_eval_trunc"] = stats_to_json(summarize(dist_trunc_vec));
    summary["dist_eval_greedy"] = stats_to_json(summarize(dist_greedy_vec));
    summary["dist_eval_greedy_pair"] =
        stats_to_json(summarize(dist_greedy_pair_vec));
    summary["full_degree_greedy"] =
        stats_to_json(summarize(full_degree_greedy_vec));
    summary["kept_degree_trunc"] =
        stats_to_json(summarize(kept_degree_trunc_vec));
    summary["kept_degree_greedy"] =
        stats_to_json(summarize(kept_degree_greedy_vec));

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Greedy-Prune vs Trunc");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- beam_size: `" + std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_size (baseline): `" +
                           std::to_string(trunc_size) + "`");
        lines.emplace_back("- greedy_degree: `" +
                           std::to_string(greedy_degree) + "`");
        lines.emplace_back("");
        if (has_gt) {
            lines.emplace_back("## Recall");
            lines.emplace_back(
                "- recall_trunc.mean: `" +
                fmt(summary["recall_trunc"]["mean"].get<double>()) + "`");
            lines.emplace_back(
                "- recall_greedy.mean: `" +
                fmt(summary["recall_greedy"]["mean"].get<double>()) + "`");
        }
        lines.emplace_back("## Distance Evaluations");
        lines.emplace_back(
            "- dist_eval_trunc.mean: `" +
            fmt(summary["dist_eval_trunc"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- dist_eval_greedy.mean (goal): `" +
            fmt(summary["dist_eval_greedy"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- dist_eval_greedy_pair.mean (pairwise): `" +
            fmt(summary["dist_eval_greedy_pair"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- dist_eval_greedy_total.mean: `" +
            fmt(summary["dist_eval_greedy"]["mean"].get<double>() +
                summary["dist_eval_greedy_pair"]["mean"].get<double>()) +
            "`");
        lines.emplace_back("## Effective Degrees");
        lines.emplace_back(
            "- full_degree_greedy.mean: `" +
            fmt(summary["full_degree_greedy"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- kept_degree_trunc.mean: `" +
            fmt(summary["kept_degree_trunc"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- kept_degree_greedy.mean: `" +
            fmt(summary["kept_degree_greedy"]["mean"].get<double>()) + "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[greedy] sampled_queries={}", reports.size());
    if (has_gt) {
        spdlog::info("[greedy] recall_trunc.mean={:.6f}",
                     summary["recall_trunc"]["mean"].get<double>());
        spdlog::info("[greedy] recall_greedy.mean={:.6f}",
                     summary["recall_greedy"]["mean"].get<double>());
    }
    spdlog::info("[greedy] dist_eval_trunc.mean={:.6f}",
                 summary["dist_eval_trunc"]["mean"].get<double>());
    spdlog::info("[greedy] dist_eval_greedy.mean={:.6f}",
                 summary["dist_eval_greedy"]["mean"].get<double>());
    spdlog::info("[greedy] dist_eval_greedy_pair.mean={:.6f}",
                 summary["dist_eval_greedy_pair"]["mean"].get<double>());

    return 0;
}

int run_rng_check(const std::string& dataset_file,
                  const std::string& index_file,
                  const std::string& query_file,
                  const std::string& label_file,
                  const std::string& qrange_file,
                  const std::string& gt_file,
                  unsigned qnumber,
                  unsigned beam_size,
                  unsigned trunc_size,
                  unsigned sample_queries,
                  unsigned seed,
                  unsigned rng_cap,
                  const std::string& out_json,
                  const std::string& out_md) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
    if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
        throw std::runtime_error("groundtruth size mismatch");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    for (auto& id : gt) {
        if (id >= pos.size()) {
            throw std::runtime_error("groundtruth id out of range");
        }
        id = pos[id];
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<RngQueryReport> reports;
    reports.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();

        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_trunc{};
        cfg_trunc.beam_size = beam_size;
        cfg_trunc.trunc_size = trunc_size;

        auto trunc_run = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_trunc,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto rng_run = beam_trace_strict_rng_prune(
            dataset, g_sub, query_list[ui], qnumber, starts, beam_size, rng_cap);

        RngQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.dist_eval_trunc = static_cast<double>(trunc_run.distance_evals);
        r.dist_eval_rng_goal = static_cast<double>(rng_run.distance_evals);
        r.dist_eval_rng_aux = static_cast<double>(rng_run.distance_evals_aux);
        r.dist_eval_rng_pair = static_cast<double>(rng_run.distance_evals_pair);
        r.full_degree_rng =
            rng_run.expansions == 0
                ? 0.0
                : static_cast<double>(rng_run.neighbours_total) /
                      static_cast<double>(rng_run.expansions);
        r.kept_degree_trunc =
            trunc_run.expansions == 0
                ? 0.0
                : static_cast<double>(trunc_run.neighbours_selected) /
                      static_cast<double>(trunc_run.expansions);
        r.kept_degree_rng =
            rng_run.expansions == 0
                ? 0.0
                : static_cast<double>(rng_run.neighbours_selected) /
                      static_cast<double>(rng_run.expansions);

        std::vector<unsigned> gt_ids(qnumber);
        for (unsigned j = 0; j < qnumber; ++j) {
            gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
        }
        r.recall_trunc = topk_recall_vs_gt(trunc_run.result, gt_ids, qnumber);
        r.recall_rng = topk_recall_vs_gt(rng_run.result, gt_ids, qnumber);

        reports.push_back(r);
        if (((t + 1) % 32 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[rng] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_trunc_vec =
        collect([](const RngQueryReport& r) { return r.recall_trunc; });
    auto recall_rng_vec =
        collect([](const RngQueryReport& r) { return r.recall_rng; });
    auto dist_trunc_vec =
        collect([](const RngQueryReport& r) { return r.dist_eval_trunc; });
    auto dist_rng_goal_vec =
        collect([](const RngQueryReport& r) { return r.dist_eval_rng_goal; });
    auto dist_rng_aux_vec =
        collect([](const RngQueryReport& r) { return r.dist_eval_rng_aux; });
    auto dist_rng_pair_vec =
        collect([](const RngQueryReport& r) { return r.dist_eval_rng_pair; });
    auto full_degree_rng_vec =
        collect([](const RngQueryReport& r) { return r.full_degree_rng; });
    auto kept_degree_trunc_vec =
        collect([](const RngQueryReport& r) { return r.kept_degree_trunc; });
    auto kept_degree_rng_vec =
        collect([](const RngQueryReport& r) { return r.kept_degree_rng; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_size", trunc_size},
        {"rng_cap", rng_cap},
    };
    summary["recall_trunc"] = stats_to_json(summarize(recall_trunc_vec));
    summary["recall_rng"] = stats_to_json(summarize(recall_rng_vec));
    summary["dist_eval_trunc"] = stats_to_json(summarize(dist_trunc_vec));
    summary["dist_eval_rng_goal"] = stats_to_json(summarize(dist_rng_goal_vec));
    summary["dist_eval_rng_aux"] = stats_to_json(summarize(dist_rng_aux_vec));
    summary["dist_eval_rng_pair"] = stats_to_json(summarize(dist_rng_pair_vec));
    if (!dist_trunc_vec.empty()) {
        std::vector<double> goal_ratio_vec;
        goal_ratio_vec.reserve(reports.size());
        for (const auto& r : reports) {
            if (r.dist_eval_trunc <= 0.0) {
                goal_ratio_vec.push_back(0.0);
            } else {
                goal_ratio_vec.push_back(r.dist_eval_rng_goal / r.dist_eval_trunc);
            }
        }
        summary["dist_eval_rng_goal_over_trunc"] =
            stats_to_json(summarize(goal_ratio_vec));
    }
    summary["full_degree_rng"] = stats_to_json(summarize(full_degree_rng_vec));
    summary["kept_degree_trunc"] =
        stats_to_json(summarize(kept_degree_trunc_vec));
    summary["kept_degree_rng"] =
        stats_to_json(summarize(kept_degree_rng_vec));

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Strict-RNG Query Check");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- beam_size: `" + std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_size (baseline): `" +
                           std::to_string(trunc_size) + "`");
        lines.emplace_back("- rng_cap: `" + std::to_string(rng_cap) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Recall");
        lines.emplace_back("- recall_trunc.mean: `" +
                           fmt(summary["recall_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- recall_rng.mean: `" +
                           fmt(summary["recall_rng"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Distance Evaluations");
        lines.emplace_back("- dist_eval_trunc.mean: `" +
                           fmt(summary["dist_eval_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back(
            "- dist_eval_rng_goal.mean: `" +
            fmt(summary["dist_eval_rng_goal"]["mean"].get<double>()) +
            "`  (idealized query-time count: assumes build-stage metadata can "
            "materialize the strict-RNG survivors without paying aux/pair "
            "costs at query time)");
        if (summary.contains("dist_eval_rng_goal_over_trunc")) {
            lines.emplace_back(
                "- dist_eval_rng_goal_over_trunc.mean: `" +
                fmt(summary["dist_eval_rng_goal_over_trunc"]["mean"].get<double>()) +
                "`");
        }
        lines.emplace_back(
            "### Auxiliary Build-Time-Or-Oracle Costs");
        lines.emplace_back("- dist_eval_rng_aux.mean: `" +
                           fmt(summary["dist_eval_rng_aux"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dist_eval_rng_pair.mean: `" +
                           fmt(summary["dist_eval_rng_pair"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Effective Degrees");
        lines.emplace_back("- full_degree_rng.mean: `" +
                           fmt(summary["full_degree_rng"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- kept_degree_trunc.mean: `" +
                           fmt(summary["kept_degree_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- kept_degree_rng.mean: `" +
                           fmt(summary["kept_degree_rng"]["mean"].get<double>()) +
                           "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[rng] sampled_queries={}", reports.size());
    spdlog::info("[rng] recall_trunc.mean={:.6f}",
                 summary["recall_trunc"]["mean"].get<double>());
    spdlog::info("[rng] recall_rng.mean={:.6f}",
                 summary["recall_rng"]["mean"].get<double>());
    spdlog::info("[rng] dist_eval_trunc.mean={:.6f}",
                 summary["dist_eval_trunc"]["mean"].get<double>());
    spdlog::info("[rng] dist_eval_rng_goal.mean={:.6f}",
                 summary["dist_eval_rng_goal"]["mean"].get<double>());
    spdlog::info("[rng] dist_eval_rng_aux.mean={:.6f}",
                 summary["dist_eval_rng_aux"]["mean"].get<double>());
    spdlog::info("[rng] dist_eval_rng_pair.mean={:.6f}",
                 summary["dist_eval_rng_pair"]["mean"].get<double>());

    return 0;
}

int run_domtag_check(const std::string& dataset_file,
                     const std::string& index_file,
                     const std::string& query_file,
                     const std::string& label_file,
                     const std::string& qrange_file,
                     const std::string& gt_file,
                     unsigned qnumber,
                     unsigned beam_size,
                     unsigned trunc_size,
                     unsigned active_cap,
                     unsigned dominator_cap,
                     unsigned sample_queries,
                     unsigned seed,
                     const std::string& out_json,
                     const std::string& out_md) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
    if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
        throw std::runtime_error("groundtruth size mismatch");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    for (auto& id : gt) {
        if (id >= pos.size()) {
            throw std::runtime_error("groundtruth id out of range");
        }
        id = pos[id];
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    DomTagCache tag_cache(dataset, index, dominator_cap);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<DomTagQueryReport> reports;
    reports.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();

        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_trunc{};
        cfg_trunc.beam_size = beam_size;
        cfg_trunc.trunc_size = trunc_size;

        auto trunc_run = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_trunc,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto tag_run = beam_trace_domtag_prune(
            dataset, index, query_list[ui], qnumber, starts, beam_size,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r),
            active_cap, tag_cache);

        DomTagQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.dist_eval_trunc = static_cast<double>(trunc_run.distance_evals);
        r.dist_eval_tag = static_cast<double>(tag_run.distance_evals);
        r.full_degree_tag =
            tag_run.expansions == 0
                ? 0.0
                : static_cast<double>(tag_run.neighbours_total) /
                      static_cast<double>(tag_run.expansions);
        r.kept_degree_trunc =
            trunc_run.expansions == 0
                ? 0.0
                : static_cast<double>(trunc_run.neighbours_selected) /
                      static_cast<double>(trunc_run.expansions);
        r.kept_degree_tag =
            tag_run.expansions == 0
                ? 0.0
                : static_cast<double>(tag_run.neighbours_selected) /
                      static_cast<double>(tag_run.expansions);
        r.dominance_checks_tag =
            tag_run.expansions == 0
                ? 0.0
                : static_cast<double>(tag_run.dominance_checks) /
                      static_cast<double>(tag_run.expansions);

        std::vector<unsigned> gt_ids(qnumber);
        for (unsigned j = 0; j < qnumber; ++j) {
            gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
        }
        r.recall_trunc = topk_recall_vs_gt(trunc_run.result, gt_ids, qnumber);
        r.recall_tag = topk_recall_vs_gt(tag_run.result, gt_ids, qnumber);

        reports.push_back(r);
        if (((t + 1) % 32 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[domtag] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_trunc_vec =
        collect([](const DomTagQueryReport& r) { return r.recall_trunc; });
    auto recall_tag_vec =
        collect([](const DomTagQueryReport& r) { return r.recall_tag; });
    auto dist_trunc_vec =
        collect([](const DomTagQueryReport& r) { return r.dist_eval_trunc; });
    auto dist_tag_vec =
        collect([](const DomTagQueryReport& r) { return r.dist_eval_tag; });
    auto full_degree_tag_vec =
        collect([](const DomTagQueryReport& r) { return r.full_degree_tag; });
    auto kept_degree_trunc_vec =
        collect([](const DomTagQueryReport& r) { return r.kept_degree_trunc; });
    auto kept_degree_tag_vec =
        collect([](const DomTagQueryReport& r) { return r.kept_degree_tag; });
    auto dominance_checks_vec =
        collect([](const DomTagQueryReport& r) { return r.dominance_checks_tag; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_size", trunc_size},
        {"active_cap", active_cap},
        {"dominator_cap", dominator_cap},
    };
    summary["recall_trunc"] = stats_to_json(summarize(recall_trunc_vec));
    summary["recall_tag"] = stats_to_json(summarize(recall_tag_vec));
    summary["dist_eval_trunc"] = stats_to_json(summarize(dist_trunc_vec));
    summary["dist_eval_tag"] = stats_to_json(summarize(dist_tag_vec));
    if (!dist_trunc_vec.empty()) {
        std::vector<double> ratio_vec;
        ratio_vec.reserve(reports.size());
        for (const auto& r : reports) {
            if (r.dist_eval_trunc <= 0.0) {
                ratio_vec.push_back(0.0);
            } else {
                ratio_vec.push_back(r.dist_eval_tag / r.dist_eval_trunc);
            }
        }
        summary["dist_eval_tag_over_trunc"] =
            stats_to_json(summarize(ratio_vec));
    }
    summary["full_degree_tag"] = stats_to_json(summarize(full_degree_tag_vec));
    summary["kept_degree_trunc"] =
        stats_to_json(summarize(kept_degree_trunc_vec));
    summary["kept_degree_tag"] = stats_to_json(summarize(kept_degree_tag_vec));
    summary["dominance_checks_tag"] =
        stats_to_json(summarize(dominance_checks_vec));
    summary["tag_rows_built"] = tag_cache.rows_built;
    summary["tag_edges_indexed"] = tag_cache.edges_indexed;
    summary["tag_arcs_total"] = tag_cache.tag_arcs_total;
    summary["tag_arcs_per_edge"] =
        tag_cache.edges_indexed == 0
            ? 0.0
            : static_cast<double>(tag_cache.tag_arcs_total) /
                  static_cast<double>(tag_cache.edges_indexed);
    summary["tag_edges_per_row"] =
        tag_cache.rows_built == 0
            ? 0.0
            : static_cast<double>(tag_cache.edges_indexed) /
                  static_cast<double>(tag_cache.rows_built);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Dominance-Tag Query Check");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- beam_size: `" + std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_size (baseline): `" +
                           std::to_string(trunc_size) + "`");
        lines.emplace_back("- active_cap (tag): `" +
                           std::to_string(active_cap) + "`");
        lines.emplace_back("- dominator_cap per edge: `" +
                           std::to_string(dominator_cap) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Recall");
        lines.emplace_back("- recall_trunc.mean: `" +
                           fmt(summary["recall_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- recall_tag.mean: `" +
                           fmt(summary["recall_tag"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Query Distance Evaluations");
        lines.emplace_back("- dist_eval_trunc.mean: `" +
                           fmt(summary["dist_eval_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dist_eval_tag.mean: `" +
                           fmt(summary["dist_eval_tag"]["mean"].get<double>()) +
                           "`");
        if (summary.contains("dist_eval_tag_over_trunc")) {
            lines.emplace_back(
                "- dist_eval_tag_over_trunc.mean: `" +
                fmt(summary["dist_eval_tag_over_trunc"]["mean"].get<double>()) +
                "`");
        }
        lines.emplace_back("## Degrees / Checks");
        lines.emplace_back("- full_degree_tag.mean: `" +
                           fmt(summary["full_degree_tag"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- kept_degree_trunc.mean: `" +
                           fmt(summary["kept_degree_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- kept_degree_tag.mean: `" +
                           fmt(summary["kept_degree_tag"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dominance_checks_tag.mean: `" +
                           fmt(summary["dominance_checks_tag"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Tag Build Footprint (Touched Rows)");
        lines.emplace_back("- tag_rows_built: `" +
                           std::to_string(summary["tag_rows_built"].get<std::uint64_t>()) +
                           "`");
        lines.emplace_back("- tag_edges_indexed: `" +
                           std::to_string(summary["tag_edges_indexed"].get<std::uint64_t>()) +
                           "`");
        lines.emplace_back("- tag_arcs_total: `" +
                           std::to_string(summary["tag_arcs_total"].get<std::uint64_t>()) +
                           "`");
        lines.emplace_back("- tag_arcs_per_edge: `" +
                           fmt(summary["tag_arcs_per_edge"].get<double>()) + "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[domtag] sampled_queries={}", reports.size());
    spdlog::info("[domtag] recall_trunc.mean={:.6f}",
                 summary["recall_trunc"]["mean"].get<double>());
    spdlog::info("[domtag] recall_tag.mean={:.6f}",
                 summary["recall_tag"]["mean"].get<double>());
    spdlog::info("[domtag] dist_eval_trunc.mean={:.6f}",
                 summary["dist_eval_trunc"]["mean"].get<double>());
    spdlog::info("[domtag] dist_eval_tag.mean={:.6f}",
                 summary["dist_eval_tag"]["mean"].get<double>());
    spdlog::info("[domtag] tag_arcs_per_edge={:.6f}",
                 summary["tag_arcs_per_edge"].get<double>());

    return 0;
}

int run_mixdom_check(const std::string& dataset_file,
                     const std::string& index_file,
                     const std::string& query_file,
                     const std::string& label_file,
                     const std::string& qrange_file,
                     const std::string& gt_file,
                     unsigned qnumber,
                     unsigned beam_size,
                     unsigned trunc_size,
                     unsigned local_offset,
                     unsigned local_keep,
                     unsigned dom_keep,
                     unsigned dominator_cap,
                     unsigned sample_queries,
                     unsigned seed,
                     const std::string& out_json,
                     const std::string& out_md) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
    if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
        throw std::runtime_error("groundtruth size mismatch");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    for (auto& id : gt) {
        if (id >= pos.size()) {
            throw std::runtime_error("groundtruth id out of range");
        }
        id = pos[id];
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    DomTagCache tag_cache(dataset, index, dominator_cap);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<MixDomQueryReport> reports;
    reports.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();
        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_trunc{};
        cfg_trunc.beam_size = beam_size;
        cfg_trunc.trunc_size = trunc_size;
        auto trunc_run = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_trunc,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto mix_run = beam_trace_mixdom_prune(
            dataset, index, g_sub, query_list[ui], qnumber, starts, beam_size,
            local_offset, local_keep, dom_keep, static_cast<unsigned>(range_l),
            static_cast<unsigned>(range_r), tag_cache);

        MixDomQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.dist_eval_trunc = static_cast<double>(trunc_run.distance_evals);
        r.dist_eval_mix = static_cast<double>(mix_run.distance_evals);
        r.kept_degree_trunc =
            trunc_run.expansions == 0
                ? 0.0
                : static_cast<double>(trunc_run.neighbours_selected) /
                      static_cast<double>(trunc_run.expansions);
        r.kept_degree_mix =
            mix_run.expansions == 0
                ? 0.0
                : static_cast<double>(mix_run.neighbours_selected) /
                      static_cast<double>(mix_run.expansions);
        r.dom_added_mix =
            mix_run.expansions == 0
                ? 0.0
                : static_cast<double>(mix_run.neighbours_selected_extra) /
                      static_cast<double>(mix_run.expansions);
        r.dominance_checks_mix =
            mix_run.expansions == 0
                ? 0.0
                : static_cast<double>(mix_run.dominance_checks) /
                      static_cast<double>(mix_run.expansions);

        std::vector<unsigned> gt_ids(qnumber);
        for (unsigned j = 0; j < qnumber; ++j) {
            gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
        }
        r.recall_trunc = topk_recall_vs_gt(trunc_run.result, gt_ids, qnumber);
        r.recall_mix = topk_recall_vs_gt(mix_run.result, gt_ids, qnumber);
        reports.push_back(r);
        if (((t + 1) % 32 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[mixdom] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_trunc_vec =
        collect([](const MixDomQueryReport& r) { return r.recall_trunc; });
    auto recall_mix_vec =
        collect([](const MixDomQueryReport& r) { return r.recall_mix; });
    auto dist_trunc_vec =
        collect([](const MixDomQueryReport& r) { return r.dist_eval_trunc; });
    auto dist_mix_vec =
        collect([](const MixDomQueryReport& r) { return r.dist_eval_mix; });
    auto kept_degree_trunc_vec =
        collect([](const MixDomQueryReport& r) { return r.kept_degree_trunc; });
    auto kept_degree_mix_vec =
        collect([](const MixDomQueryReport& r) { return r.kept_degree_mix; });
    auto dom_added_vec =
        collect([](const MixDomQueryReport& r) { return r.dom_added_mix; });
    auto dominance_checks_vec =
        collect([](const MixDomQueryReport& r) { return r.dominance_checks_mix; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_size", trunc_size},
        {"local_offset", local_offset},
        {"local_keep", local_keep},
        {"dom_keep", dom_keep},
        {"dominator_cap", dominator_cap},
    };
    summary["recall_trunc"] = stats_to_json(summarize(recall_trunc_vec));
    summary["recall_mix"] = stats_to_json(summarize(recall_mix_vec));
    summary["dist_eval_trunc"] = stats_to_json(summarize(dist_trunc_vec));
    summary["dist_eval_mix"] = stats_to_json(summarize(dist_mix_vec));
    if (!dist_trunc_vec.empty()) {
        std::vector<double> ratio_vec;
        ratio_vec.reserve(reports.size());
        for (const auto& r : reports) {
            ratio_vec.push_back(r.dist_eval_trunc <= 0.0
                                    ? 0.0
                                    : (r.dist_eval_mix / r.dist_eval_trunc));
        }
        summary["dist_eval_mix_over_trunc"] =
            stats_to_json(summarize(ratio_vec));
    }
    summary["kept_degree_trunc"] =
        stats_to_json(summarize(kept_degree_trunc_vec));
    summary["kept_degree_mix"] = stats_to_json(summarize(kept_degree_mix_vec));
    summary["dom_added_mix"] = stats_to_json(summarize(dom_added_vec));
    summary["dominance_checks_mix"] =
        stats_to_json(summarize(dominance_checks_vec));
    summary["tag_rows_built"] = tag_cache.rows_built;
    summary["tag_edges_indexed"] = tag_cache.edges_indexed;
    summary["tag_arcs_total"] = tag_cache.tag_arcs_total;
    summary["tag_arcs_per_edge"] =
        tag_cache.edges_indexed == 0
            ? 0.0
            : static_cast<double>(tag_cache.tag_arcs_total) /
                  static_cast<double>(tag_cache.edges_indexed);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Mixed Baseline+DomTag Check");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- beam_size: `" + std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_size (baseline): `" +
                           std::to_string(trunc_size) + "`");
        lines.emplace_back("- local_keep: `" + std::to_string(local_keep) + "`");
        lines.emplace_back("- dom_keep: `" + std::to_string(dom_keep) + "`");
        lines.emplace_back("- dominator_cap: `" +
                           std::to_string(dominator_cap) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Recall");
        lines.emplace_back("- recall_trunc.mean: `" +
                           fmt(summary["recall_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- recall_mix.mean: `" +
                           fmt(summary["recall_mix"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Query Distance Evaluations");
        lines.emplace_back("- dist_eval_trunc.mean: `" +
                           fmt(summary["dist_eval_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dist_eval_mix.mean: `" +
                           fmt(summary["dist_eval_mix"]["mean"].get<double>()) +
                           "`");
        if (summary.contains("dist_eval_mix_over_trunc")) {
            lines.emplace_back("- dist_eval_mix_over_trunc.mean: `" +
                               fmt(summary["dist_eval_mix_over_trunc"]["mean"].get<double>()) +
                               "`");
        }
        lines.emplace_back("## Mix Composition");
        lines.emplace_back("- kept_degree_mix.mean: `" +
                           fmt(summary["kept_degree_mix"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dom_added_mix.mean: `" +
                           fmt(summary["dom_added_mix"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dominance_checks_mix.mean: `" +
                           fmt(summary["dominance_checks_mix"]["mean"].get<double>()) +
                           "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[mixdom] sampled_queries={}", reports.size());
    spdlog::info("[mixdom] recall_trunc.mean={:.6f}",
                 summary["recall_trunc"]["mean"].get<double>());
    spdlog::info("[mixdom] recall_mix.mean={:.6f}",
                 summary["recall_mix"]["mean"].get<double>());
    spdlog::info("[mixdom] dist_eval_trunc.mean={:.6f}",
                 summary["dist_eval_trunc"]["mean"].get<double>());
    spdlog::info("[mixdom] dist_eval_mix.mean={:.6f}",
                 summary["dist_eval_mix"]["mean"].get<double>());

    return 0;
}

int run_lane_check(const std::string& dataset_file,
                   const std::string& index_file,
                   const std::string& query_file,
                   const std::string& label_file,
                   const std::string& qrange_file,
                   const std::string& gt_file,
                   unsigned qnumber,
                   unsigned beam_size,
                   unsigned trunc_size,
                   unsigned local_keep,
                   unsigned switch_skip,
                   unsigned switch_keep,
                   unsigned dom_keep,
                   double switch_gap_lo_ratio,
                   double switch_gap_hi_ratio,
                   unsigned dominator_cap,
                   unsigned sample_queries,
                   unsigned seed,
                   const std::string& out_json,
                   const std::string& out_md) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
    if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
        throw std::runtime_error("groundtruth size mismatch");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    for (auto& id : gt) {
        if (id >= pos.size()) {
            throw std::runtime_error("groundtruth id out of range");
        }
        id = pos[id];
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    DomTagCache tag_cache(dataset, index, dominator_cap);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<LaneQueryReport> reports;
    reports.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();
        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_trunc{};
        cfg_trunc.beam_size = beam_size;
        cfg_trunc.trunc_size = trunc_size;
        auto trunc_run = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_trunc,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto lane_run = beam_trace_lane_prune(
            dataset, index, g_sub, query_list[ui], qnumber, starts, beam_size,
            local_keep, switch_skip, switch_keep, dom_keep,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r),
            switch_gap_lo_ratio, switch_gap_hi_ratio, tag_cache);

        LaneQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.dist_eval_trunc = static_cast<double>(trunc_run.distance_evals);
        r.dist_eval_lane = static_cast<double>(lane_run.distance_evals);
        r.kept_degree_trunc =
            trunc_run.expansions == 0
                ? 0.0
                : static_cast<double>(trunc_run.neighbours_selected) /
                      static_cast<double>(trunc_run.expansions);
        r.kept_degree_lane =
            lane_run.expansions == 0
                ? 0.0
                : static_cast<double>(lane_run.neighbours_selected) /
                      static_cast<double>(lane_run.expansions);
        r.switch_added_lane =
            lane_run.expansions == 0
                ? 0.0
                : static_cast<double>(lane_run.neighbours_selected_switch) /
                      static_cast<double>(lane_run.expansions);
        r.dom_added_lane =
            lane_run.expansions == 0
                ? 0.0
                : static_cast<double>(lane_run.neighbours_selected_dom) /
                      static_cast<double>(lane_run.expansions);
        r.dominance_checks_lane =
            lane_run.expansions == 0
                ? 0.0
                : static_cast<double>(lane_run.dominance_checks) /
                      static_cast<double>(lane_run.expansions);

        std::vector<unsigned> gt_ids(qnumber);
        for (unsigned j = 0; j < qnumber; ++j) {
            gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
        }
        r.recall_trunc = topk_recall_vs_gt(trunc_run.result, gt_ids, qnumber);
        r.recall_lane = topk_recall_vs_gt(lane_run.result, gt_ids, qnumber);
        reports.push_back(r);
        if (((t + 1) % 32 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[lane] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_trunc_vec =
        collect([](const LaneQueryReport& r) { return r.recall_trunc; });
    auto recall_lane_vec =
        collect([](const LaneQueryReport& r) { return r.recall_lane; });
    auto dist_trunc_vec =
        collect([](const LaneQueryReport& r) { return r.dist_eval_trunc; });
    auto dist_lane_vec =
        collect([](const LaneQueryReport& r) { return r.dist_eval_lane; });
    auto kept_degree_trunc_vec =
        collect([](const LaneQueryReport& r) { return r.kept_degree_trunc; });
    auto kept_degree_lane_vec =
        collect([](const LaneQueryReport& r) { return r.kept_degree_lane; });
    auto switch_added_vec =
        collect([](const LaneQueryReport& r) { return r.switch_added_lane; });
    auto dom_added_vec =
        collect([](const LaneQueryReport& r) { return r.dom_added_lane; });
    auto dominance_checks_vec = collect(
        [](const LaneQueryReport& r) { return r.dominance_checks_lane; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_size", trunc_size},
        {"local_keep", local_keep},
        {"switch_skip", switch_skip},
        {"switch_keep", switch_keep},
        {"dom_keep", dom_keep},
        {"switch_gap_lo_ratio", switch_gap_lo_ratio},
        {"switch_gap_hi_ratio", switch_gap_hi_ratio},
        {"dominator_cap", dominator_cap},
    };
    summary["recall_trunc"] = stats_to_json(summarize(recall_trunc_vec));
    summary["recall_lane"] = stats_to_json(summarize(recall_lane_vec));
    summary["dist_eval_trunc"] = stats_to_json(summarize(dist_trunc_vec));
    summary["dist_eval_lane"] = stats_to_json(summarize(dist_lane_vec));
    if (!dist_trunc_vec.empty()) {
        std::vector<double> ratio_vec;
        ratio_vec.reserve(reports.size());
        for (const auto& r : reports) {
            ratio_vec.push_back(r.dist_eval_trunc <= 0.0
                                    ? 0.0
                                    : r.dist_eval_lane / r.dist_eval_trunc);
        }
        summary["dist_eval_lane_over_trunc"] =
            stats_to_json(summarize(ratio_vec));
    }
    summary["kept_degree_trunc"] =
        stats_to_json(summarize(kept_degree_trunc_vec));
    summary["kept_degree_lane"] =
        stats_to_json(summarize(kept_degree_lane_vec));
    summary["switch_added_lane"] =
        stats_to_json(summarize(switch_added_vec));
    summary["dom_added_lane"] = stats_to_json(summarize(dom_added_vec));
    summary["dominance_checks_lane"] =
        stats_to_json(summarize(dominance_checks_vec));

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) { return fmt::format("{:.6f}", v); };
        std::vector<std::string> lines;
        lines.emplace_back("# lanecheck Summary");
        lines.emplace_back("");
        lines.emplace_back("## Config");
        lines.emplace_back("- beam_size: `" +
                           std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_size: `" +
                           std::to_string(trunc_size) + "`");
        lines.emplace_back("- local_keep: `" +
                           std::to_string(local_keep) + "`");
        lines.emplace_back("- switch_skip: `" +
                           std::to_string(switch_skip) + "`");
        lines.emplace_back("- switch_keep: `" +
                           std::to_string(switch_keep) + "`");
        lines.emplace_back("- dom_keep: `" +
                           std::to_string(dom_keep) + "`");
        lines.emplace_back("- switch_gap_lo_ratio: `" +
                           fmt(switch_gap_lo_ratio) + "`");
        lines.emplace_back("- switch_gap_hi_ratio: `" +
                           fmt(switch_gap_hi_ratio) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Recall");
        lines.emplace_back("- recall_trunc.mean: `" +
                           fmt(summary["recall_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- recall_lane.mean: `" +
                           fmt(summary["recall_lane"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Distance Evals");
        lines.emplace_back("- dist_eval_trunc.mean: `" +
                           fmt(summary["dist_eval_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dist_eval_lane.mean: `" +
                           fmt(summary["dist_eval_lane"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dist_eval_lane_over_trunc.mean: `" +
                           fmt(summary["dist_eval_lane_over_trunc"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Lane Counts");
        lines.emplace_back("- kept_degree_lane.mean: `" +
                           fmt(summary["kept_degree_lane"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- switch_added_lane.mean: `" +
                           fmt(summary["switch_added_lane"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dom_added_lane.mean: `" +
                           fmt(summary["dom_added_lane"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dominance_checks_lane.mean: `" +
                           fmt(summary["dominance_checks_lane"]["mean"].get<double>()) +
                           "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[lane] sampled_queries={}", reports.size());
    spdlog::info("[lane] recall_trunc.mean={:.6f}",
                 summary["recall_trunc"]["mean"].get<double>());
    spdlog::info("[lane] recall_lane.mean={:.6f}",
                 summary["recall_lane"]["mean"].get<double>());
    spdlog::info("[lane] dist_eval_trunc.mean={:.6f}",
                 summary["dist_eval_trunc"]["mean"].get<double>());
    spdlog::info("[lane] dist_eval_lane.mean={:.6f}",
                 summary["dist_eval_lane"]["mean"].get<double>());

    return 0;
}

int run_two_stage_check(const std::string& dataset_file,
                        const std::string& index_file,
                        const std::string& query_file,
                        const std::string& label_file,
                        const std::string& qrange_file,
                        const std::string& gt_file,
                        unsigned qnumber,
                        unsigned beam_route,
                        unsigned trunc_route,
                        unsigned route_expansions,
                        unsigned beam_refine,
                        unsigned trunc_refine,
                        unsigned sample_queries,
                        unsigned seed,
                        const std::string& out_json,
                        const std::string& out_md) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
    if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
        throw std::runtime_error("groundtruth size mismatch");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    for (auto& id : gt) {
        if (id >= pos.size()) {
            throw std::runtime_error("groundtruth id out of range");
        }
        id = pos[id];
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<TwoStageQueryReport> reports;
    reports.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();

        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_base{};
        cfg_base.beam_size = beam_refine;
        cfg_base.trunc_size = trunc_refine;
        auto base_run = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_base,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto stage_run = beam_trace_two_stage(
            dataset, g_sub, query_list[ui], qnumber, starts, beam_route,
            trunc_route, route_expansions, beam_refine, trunc_refine);

        TwoStageQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.dist_eval_base = static_cast<double>(base_run.distance_evals);
        r.dist_eval_stage = static_cast<double>(stage_run.distance_evals);
        r.kept_degree_base =
            base_run.expansions == 0
                ? 0.0
                : static_cast<double>(base_run.neighbours_selected) /
                      static_cast<double>(base_run.expansions);
        r.kept_degree_stage =
            stage_run.expansions == 0
                ? 0.0
                : static_cast<double>(stage_run.neighbours_selected) /
                      static_cast<double>(stage_run.expansions);

        std::vector<unsigned> gt_ids(qnumber);
        for (unsigned j = 0; j < qnumber; ++j) {
            gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
        }
        r.recall_base = topk_recall_vs_gt(base_run.result, gt_ids, qnumber);
        r.recall_stage = topk_recall_vs_gt(stage_run.result, gt_ids, qnumber);

        reports.push_back(r);
        if (((t + 1) % 32 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[twostage] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_base_vec =
        collect([](const TwoStageQueryReport& r) { return r.recall_base; });
    auto recall_stage_vec =
        collect([](const TwoStageQueryReport& r) { return r.recall_stage; });
    auto dist_base_vec =
        collect([](const TwoStageQueryReport& r) { return r.dist_eval_base; });
    auto dist_stage_vec =
        collect([](const TwoStageQueryReport& r) { return r.dist_eval_stage; });
    auto kept_base_vec =
        collect([](const TwoStageQueryReport& r) { return r.kept_degree_base; });
    auto kept_stage_vec = collect(
        [](const TwoStageQueryReport& r) { return r.kept_degree_stage; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["config"] = {
        {"beam_route", beam_route},
        {"trunc_route", trunc_route},
        {"route_expansions", route_expansions},
        {"beam_refine", beam_refine},
        {"trunc_refine", trunc_refine},
    };
    summary["recall_base"] = stats_to_json(summarize(recall_base_vec));
    summary["recall_stage"] = stats_to_json(summarize(recall_stage_vec));
    summary["dist_eval_base"] = stats_to_json(summarize(dist_base_vec));
    summary["dist_eval_stage"] = stats_to_json(summarize(dist_stage_vec));
    summary["kept_degree_base"] = stats_to_json(summarize(kept_base_vec));
    summary["kept_degree_stage"] = stats_to_json(summarize(kept_stage_vec));

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Two-Stage Query Check");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- beam_route: `" + std::to_string(beam_route) +
                           "`");
        lines.emplace_back("- trunc_route: `" + std::to_string(trunc_route) +
                           "`");
        lines.emplace_back("- route_expansions: `" +
                           std::to_string(route_expansions) + "`");
        lines.emplace_back("- beam_refine: `" + std::to_string(beam_refine) +
                           "`");
        lines.emplace_back("- trunc_refine: `" +
                           std::to_string(trunc_refine) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Recall");
        lines.emplace_back("- recall_base.mean: `" +
                           fmt(summary["recall_base"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- recall_stage.mean: `" +
                           fmt(summary["recall_stage"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Distance Evaluations");
        lines.emplace_back("- dist_eval_base.mean: `" +
                           fmt(summary["dist_eval_base"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- dist_eval_stage.mean: `" +
                           fmt(summary["dist_eval_stage"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("## Effective Degrees");
        lines.emplace_back("- kept_degree_base.mean: `" +
                           fmt(summary["kept_degree_base"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- kept_degree_stage.mean: `" +
                           fmt(summary["kept_degree_stage"]["mean"].get<double>()) +
                           "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[twostage] sampled_queries={}", reports.size());
    spdlog::info("[twostage] recall_base.mean={:.6f}",
                 summary["recall_base"]["mean"].get<double>());
    spdlog::info("[twostage] recall_stage.mean={:.6f}",
                 summary["recall_stage"]["mean"].get<double>());
    spdlog::info("[twostage] dist_eval_base.mean={:.6f}",
                 summary["dist_eval_base"]["mean"].get<double>());
    spdlog::info("[twostage] dist_eval_stage.mean={:.6f}",
                 summary["dist_eval_stage"]["mean"].get<double>());

    return 0;
}

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_diag(const TDFANN::Vector::VectorList<float>& dataset,
                        const G& graph,
                        const GoalId& goal,
                        unsigned k,
                        const std::vector<unsigned>& start_nodes,
                        const DiagConfig& cfg,
                        unsigned range_l,
                        unsigned range_r_exclusive) {
    RunDiag out;
    if (cfg.beam_size == 0) {
        return out;
    }

    using T = float;
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), cfg.beam_size,
                                         cfg.trunc_size);
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    auto& neighbours = scratch.neighbours;
    auto& raw_nodes = scratch.raw_nodes;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < cfg.beam_size) {
        candidates.resize(cfg.beam_size,
                          {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > cfg.beam_size) {
        candidates.resize(cfg.beam_size);
    }

    const unsigned effective_nav_degree =
        (cfg.nav_degree == 0 ? cfg.trunc_size
                             : std::min(cfg.nav_degree, cfg.trunc_size));
    const unsigned effective_scan_factor = std::max(1u, cfg.nav_scan_factor);
    const unsigned effective_stall_rounds = std::max(1u, cfg.nav_stall_rounds);
    const unsigned effective_front_keep = std::max(1u, cfg.nav_front_keep);
    const unsigned effective_tail_degree =
        (cfg.nav_tail_degree == 0
             ? 0
             : std::min(cfg.nav_tail_degree, cfg.trunc_size));
    const unsigned effective_early_stop_rounds = cfg.nav_early_stop_rounds;
    unsigned stall_rounds = 0;

    std::vector<unsigned> all_unvisited;
    std::vector<unsigned> selected_ids;
    std::vector<std::pair<T, unsigned>> all_pairs;
    std::unordered_set<unsigned> selected_set;
    all_unvisited.reserve(256);
    selected_ids.reserve(256);
    all_pairs.reserve(256);
    selected_set.reserve(256);

    const double effective_range_width =
        (range_r_exclusive > range_l)
            ? static_cast<double>(range_r_exclusive - range_l)
            : 1.0;
    const double bridge_gap_threshold =
        std::max(1.0, cfg.bridge_gap_ratio * effective_range_width);

    auto mean_rank_pct = [](const std::vector<unsigned>& ids,
                            const std::vector<std::pair<T, unsigned>>& ranked) {
        if (ids.empty() || ranked.empty()) {
            return 0.0;
        }
        std::unordered_map<unsigned, unsigned> rank_map;
        rank_map.reserve(ids.size() * 2 + 1);
        for (size_t i = 0; i < ranked.size(); ++i) {
            rank_map.emplace(ranked[i].second, static_cast<unsigned>(i) + 1);
        }
        double sum = 0.0;
        unsigned hit = 0;
        for (auto nid : ids) {
            auto it = rank_map.find(nid);
            if (it == rank_map.end()) {
                continue;
            }
            sum += static_cast<double>(it->second) /
                   static_cast<double>(ranked.size());
            hit++;
        }
        return hit == 0 ? 0.0 : (sum / static_cast<double>(hit));
    };

    auto gap_mean_ratio = [&](unsigned src, const std::vector<unsigned>& ids) {
        if (ids.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (auto nid : ids) {
            const auto gap = static_cast<double>(
                std::abs(static_cast<long long>(nid) -
                         static_cast<long long>(src)));
            sum += gap / effective_range_width;
        }
        return sum / static_cast<double>(ids.size());
    };

    auto span_ratio = [&](const std::vector<unsigned>& ids) {
        if (ids.empty()) {
            return 0.0;
        }
        auto [mn, mx] = std::minmax_element(ids.begin(), ids.end());
        return (static_cast<double>(*mx) - static_cast<double>(*mn) + 1.0) /
               effective_range_width;
    };

    auto far_ratio = [&](unsigned src, const std::vector<unsigned>& ids) {
        if (ids.empty()) {
            return 0.0;
        }
        unsigned cnt = 0;
        for (auto nid : ids) {
            const auto gap = static_cast<double>(
                std::abs(static_cast<long long>(nid) -
                         static_cast<long long>(src)));
            if (gap >= bridge_gap_threshold) {
                cnt++;
            }
        }
        return static_cast<double>(cnt) / static_cast<double>(ids.size());
    };

    const auto t0 = Clock::now();
    for (int uid = 0; uid < static_cast<int>(cfg.beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        out.expansions++;

        if (cfg.trunc_size == 0) {
            continue;
        }

        unsigned local_budget = cfg.trunc_size;
        if (effective_nav_degree > 0 && effective_nav_degree < cfg.trunc_size) {
            local_budget = effective_nav_degree;
            if (stall_rounds >= effective_stall_rounds) {
                local_budget = cfg.trunc_size;
            }
        }
        if (effective_tail_degree > 0 &&
            uid >= static_cast<int>(cfg.beam_size / 2)) {
            local_budget = std::min(local_budget, effective_tail_degree);
        }
        if (local_budget == 0) {
            continue;
        }

        unsigned scan_limit = cfg.trunc_size;
        if (local_budget < cfg.trunc_size) {
            scan_limit = std::min(
                cfg.trunc_size,
                std::max(local_budget, local_budget * effective_scan_factor));
        }

        all_unvisited.clear();
        for (const auto& x : graph.get_neighbours(current_node)) {
            out.neighbours_total++;
            if (scratch.is_visited(x.to)) {
                continue;
            }
            all_unvisited.push_back(x.to);
        }
        out.neighbours_unvisited += all_unvisited.size();
        if (all_unvisited.empty()) {
            out.expansion_rows.push_back(ExpansionProbe{
                .step = static_cast<unsigned>(out.expansions),
                .current_node = current_node,
            });
            continue;
        }

        const unsigned scanned_cap = std::min<unsigned>(
            scan_limit, static_cast<unsigned>(all_unvisited.size()));

        selected_ids.clear();
        if (local_budget >= cfg.trunc_size) {
            const size_t take =
                std::min<size_t>(cfg.trunc_size, all_unvisited.size());
            for (size_t i = 0; i < take; ++i) {
                selected_ids.push_back(all_unvisited[i]);
            }
        } else {
            raw_nodes.clear();
            for (size_t i = 0; i < all_unvisited.size(); ++i) {
                raw_nodes.push_back(all_unvisited[i]);
                if (raw_nodes.size() >= scan_limit) {
                    break;
                }
            }

            if (raw_nodes.size() <= local_budget) {
                for (auto nid : raw_nodes) {
                    selected_ids.push_back(nid);
                }
            } else {
                const size_t need = local_budget;
                const size_t raw_n = raw_nodes.size();
                const size_t prefix_keep = std::min<size_t>(
                    std::min<size_t>(effective_front_keep, need), raw_n);
                for (size_t i = 0; i < prefix_keep; ++i) {
                    selected_ids.push_back(raw_nodes[i]);
                }
                const size_t remain_need = need - selected_ids.size();
                if (remain_need > 0 && raw_n > prefix_keep) {
                    const size_t tail_n = raw_n - prefix_keep;
                    size_t last_idx = std::numeric_limits<size_t>::max();
                    for (size_t i = 0; i < remain_need; ++i) {
                        const size_t rel =
                            (remain_need == 1)
                                ? 0
                                : (i * (tail_n - 1)) / (remain_need - 1);
                        const size_t idx = prefix_keep + rel;
                        if (idx == last_idx) {
                            continue;
                        }
                        last_idx = idx;
                        selected_ids.push_back(raw_nodes[idx]);
                    }
                }
            }
        }

        if (selected_ids.empty()) {
            continue;
        }

        out.neighbours_selected += selected_ids.size();
        neighbours.clear();
        for (auto nid : selected_ids) {
            neighbours.push_back({T(0), nid});
        }
        dataset.dist_all_into(goal, neighbours);
        out.distance_evals += neighbours.size();

        bool improved = false;
        T worst_selected = T(0);
        bool worst_init = false;
        selected_set.clear();
        selected_set.reserve(selected_ids.size() * 2 + 1);
        for (const auto& [dist, nid] : neighbours) {
            selected_set.insert(nid);
            if (!worst_init || dist > worst_selected) {
                worst_selected = dist;
                worst_init = true;
            }
            scratch.mark_visited(nid, dist);
            if (dist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < dist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {dist, nid + offset});
                if (dist < candidates.front().first) {
                    improved = true;
                }
            }
        }

        // Oracle diagnostic: among all unvisited neighbours, how many better
        // nodes were skipped by current pre-distance trunc policy?
        if (!all_unvisited.empty()) {
            all_pairs.clear();
            all_pairs.reserve(all_unvisited.size());
            for (auto nid : all_unvisited) {
                all_pairs.push_back({T(0), nid});
            }
            dataset.dist_all_into(goal, all_pairs);
            std::ranges::sort(all_pairs, [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

            const size_t topk =
                std::min<size_t>(selected_ids.size(), all_pairs.size());
            out.topk_total += topk;
            unsigned hit = 0;
            for (size_t i = 0; i < topk; ++i) {
                if (selected_set.contains(all_pairs[i].second)) {
                    hit++;
                }
            }
            out.topk_hits += hit;

            unsigned oracle_rank_probe = 0;
            if (!all_pairs.empty()) {
                const unsigned oracle_best = all_pairs.front().second;
                for (size_t i = 0; i < all_unvisited.size(); ++i) {
                    if (all_unvisited[i] == oracle_best) {
                        oracle_rank_probe = static_cast<unsigned>(i) + 1;
                        break;
                    }
                }
                if (oracle_rank_probe > 0) {
                    out.oracle_rank_samples++;
                    out.oracle_rank_sum +=
                        static_cast<double>(oracle_rank_probe);
                }

                if (!selected_set.contains(oracle_best)) {
                    out.oracle_best_missed++;
                    if (oracle_rank_probe > 0) {
                        out.oracle_rank_missed_samples++;
                        out.oracle_rank_missed_sum +=
                            static_cast<double>(oracle_rank_probe);
                    }
                    if (oracle_rank_probe == 0 ||
                        oracle_rank_probe > scanned_cap) {
                        out.miss_not_scanned++;
                    } else {
                        out.miss_scanned_not_selected++;
                    }
                }
            }

            unsigned better_unselected = 0;
            if (worst_init) {
                for (const auto& [dist, nid] : all_pairs) {
                    if (selected_set.contains(nid)) {
                        continue;
                    }
                    if (dist < worst_selected) {
                        better_unselected++;
                    }
                }
            }
            out.better_unselected_than_worst += better_unselected;

            std::vector<unsigned> oracle_ids;
            oracle_ids.reserve(topk);
            for (size_t i = 0; i < topk; ++i) {
                oracle_ids.push_back(all_pairs[i].second);
            }

            out.expansion_rows.push_back(ExpansionProbe{
                .step = static_cast<unsigned>(out.expansions),
                .current_node = current_node,
                .unvisited_deg = static_cast<std::uint64_t>(all_unvisited.size()),
                .scanned_deg = static_cast<std::uint64_t>(scanned_cap),
                .selected_deg = static_cast<std::uint64_t>(selected_ids.size()),
                .scan_coverage =
                    static_cast<double>(scanned_cap) /
                    static_cast<double>(all_unvisited.size()),
                .oracle_best_rank_in_adj =
                    static_cast<double>(oracle_rank_probe),
                .selected_rank_pct_mean_global =
                    mean_rank_pct(selected_ids, all_pairs),
                .selected_gap_mean_ratio =
                    gap_mean_ratio(current_node, selected_ids),
                .oracle_gap_mean_ratio =
                    gap_mean_ratio(current_node, oracle_ids),
                .selected_span_ratio = span_ratio(selected_ids),
                .oracle_span_ratio = span_ratio(oracle_ids),
                .selected_far_ratio = far_ratio(current_node, selected_ids),
                .oracle_far_ratio = far_ratio(current_node, oracle_ids),
            });
        }

        if (improved) {
            stall_rounds = 0;
        } else {
            stall_rounds++;
            if (effective_early_stop_rounds > 0 &&
                stall_rounds >= effective_early_stop_rounds &&
                uid >= static_cast<int>(cfg.beam_size / 3)) {
                break;
            }
        }
    }
    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

static std::vector<unsigned> select_switch_lane_from_row(
    unsigned src, const std::vector<unsigned>& row, size_t start_rank,
    unsigned keep, unsigned gap_lo, unsigned gap_hi,
    const std::unordered_set<unsigned>& picked) {
    std::vector<unsigned> left;
    std::vector<unsigned> right;
    left.reserve(keep);
    right.reserve(keep);

    const size_t begin = std::min(start_rank, row.size());
    for (size_t i = begin; i < row.size(); ++i) {
        const unsigned nid = row[i];
        if (picked.contains(nid)) {
            continue;
        }
        const unsigned gap = static_cast<unsigned>(
            std::abs(static_cast<int>(nid) - static_cast<int>(src)));
        if (gap < gap_lo || gap > gap_hi) {
            continue;
        }
        if (nid < src) {
            left.push_back(nid);
        } else if (nid > src) {
            right.push_back(nid);
        }
    }

    std::vector<unsigned> out;
    out.reserve(keep);
    size_t li = 0, ri = 0;
    int last_side = -1;
    while (out.size() < keep && (li < left.size() || ri < right.size())) {
        const bool take_left =
            (li < left.size()) && (ri >= right.size() || last_side == 1 || last_side == -1);
        if (take_left) {
            out.push_back(left[li++]);
            last_side = 0;
        } else if (ri < right.size()) {
            out.push_back(right[ri++]);
            last_side = 1;
        } else {
            break;
        }
    }
    return out;
}

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_lane_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& base_graph,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned local_keep,
    unsigned switch_skip,
    unsigned switch_keep,
    unsigned dom_keep,
    unsigned range_l,
    unsigned range_r,
    double switch_gap_lo_ratio,
    double switch_gap_hi_ratio,
    DomTagCache& cache) {
    RunDiag out;
    (void)base_graph;
    if (beam_size == 0) {
        return out;
    }

    using T = float;
    const unsigned total_budget =
        std::max(1u, local_keep + switch_keep + dom_keep);
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), beam_size, total_budget);
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < beam_size) {
        candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }

    const auto t0 = Clock::now();
    std::vector<std::pair<T, unsigned>> neighbours;
    std::vector<unsigned> local_raw;
    std::vector<unsigned char> present;
    std::unordered_set<unsigned> picked;
    neighbours.reserve(256);
    local_raw.reserve(256);
    picked.reserve(256);

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        out.expansions++;

        neighbours.clear();
        local_raw.clear();
        picked.clear();

        for (const auto& x : graph.get_neighbours(current_node)) {
            out.neighbours_total++;
            local_raw.push_back(x.to);
        }

        const size_t local_take = std::min<size_t>(local_keep, local_raw.size());
        for (size_t i = 0; i < local_take; ++i) {
            if (!picked.insert(local_raw[i]).second) {
                continue;
            }
            neighbours.push_back({T(0), local_raw[i]});
        }

        const unsigned range_width = std::max(1u, range_r - range_l);
        const unsigned gap_lo = std::max(
            1u, static_cast<unsigned>(std::ceil(switch_gap_lo_ratio * range_width)));
        const unsigned gap_hi = std::max(
            gap_lo + 1,
            static_cast<unsigned>(std::ceil(switch_gap_hi_ratio * range_width)));
        const auto switch_nodes = select_switch_lane_from_row(
            current_node, local_raw, switch_skip, switch_keep, gap_lo, gap_hi,
            picked);
        for (unsigned nid : switch_nodes) {
            if (!picked.insert(nid).second) {
                continue;
            }
            neighbours.push_back({T(0), nid});
            out.neighbours_selected_switch++;
        }

        const auto& row = cache.get(current_node);
        if (!row.nodes_sorted.empty() && dom_keep > 0) {
            present.assign(row.nodes_sorted.size(), 0);
            for (size_t i = 0; i < row.nodes_sorted.size(); ++i) {
                const unsigned nid = row.nodes_sorted[i];
                if (nid >= range_l && nid < range_r) {
                    present[i] = 1;
                }
            }

            unsigned dom_added = 0;
            for (size_t i = 0; i < row.nodes_sorted.size(); ++i) {
                if (!present[i]) {
                    continue;
                }
                const unsigned nid = row.nodes_sorted[i];
                if (picked.contains(nid)) {
                    continue;
                }
                bool dominated = false;
                for (unsigned j : row.dominators[i]) {
                    out.dominance_checks++;
                    if (present[j]) {
                        dominated = true;
                        break;
                    }
                }
                if (dominated) {
                    continue;
                }
                neighbours.push_back({T(0), nid});
                picked.insert(nid);
                dom_added++;
                out.neighbours_selected_dom++;
                if (dom_added >= dom_keep) {
                    break;
                }
            }
        }

        out.neighbours_selected += neighbours.size();
        auto it = std::ranges::remove_if(neighbours, [&](const auto& p) {
            return scratch.is_visited(p.second);
        });
        neighbours.erase(it.begin(), it.end());
        out.neighbours_unvisited += neighbours.size();
        if (neighbours.empty()) {
            continue;
        }

        dataset.dist_all_into(goal, neighbours);
        out.distance_evals += neighbours.size();

        for (const auto& [qdist, nid] : neighbours) {
            scratch.mark_visited(nid, qdist);
        }
        for (const auto& [qdist, nid] : neighbours) {
            if (qdist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < qdist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {qdist, nid + offset});
            }
        }
    }

    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

template <typename T>
std::vector<std::pair<T, unsigned>> greedy_prune_neighbors(
    const TDFANN::Vector::VectorList<T>& dataset,
    const std::vector<std::pair<T, unsigned>>& ordered,
    unsigned max_degree,
    std::uint64_t& pair_dist_evals) {
    // ordered: sorted by distance to source (asc), already includes that dist.
    // Keep the NSG-style heuristic core, but backfill nearest rejected
    // neighbours so the final degree stays close to the requested cap.
    std::vector<std::pair<T, unsigned>> kept;
    if (max_degree == 0 || ordered.empty()) {
        return kept;
    }
    kept.reserve(std::min<size_t>(max_degree, ordered.size()));
    std::vector<std::pair<T, unsigned>> rejected;
    rejected.reserve(std::min<size_t>(ordered.size(), max_degree * 4));

    for (const auto& cand : ordered) {
        const T d_now = cand.first;
        const unsigned i_now = cand.second;
        bool ok = true;
        for (const auto& sel : kept) {
            const T d_lst = sel.first;
            const unsigned i_lst = sel.second;
            if (i_lst == i_now) {
                ok = false;
                break;
            }
            pair_dist_evals++;
            const T d_ij = dataset.dist(i_now, i_lst);
            if (d_now > d_lst && d_now > d_ij) {
                ok = false;
                break;
            }
        }
        if (ok) {
            kept.push_back(cand);
            if (kept.size() >= max_degree) {
                break;
            }
        } else {
            rejected.push_back(cand);
        }
    }

    if (kept.size() < max_degree) {
        for (const auto& cand : rejected) {
            bool dup = false;
            for (const auto& sel : kept) {
                if (sel.second == cand.second) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            kept.push_back(cand);
            if (kept.size() >= max_degree) {
                break;
            }
        }
    }
    return kept;
}

template <typename T>
std::vector<std::pair<T, unsigned>> strict_rng_prune_neighbors(
    const TDFANN::Vector::VectorList<T>& dataset,
    const std::vector<std::pair<T, unsigned>>& ordered,
    std::uint64_t& pair_dist_evals) {
    std::vector<std::pair<T, unsigned>> kept;
    if (ordered.empty()) {
        return kept;
    }
    kept.reserve(ordered.size());

    for (size_t i = 0; i < ordered.size(); ++i) {
        const T d_now = ordered[i].first;
        const unsigned i_now = ordered[i].second;
        bool ok = true;
        for (size_t j = 0; j < i; ++j) {
            const T d_prev = ordered[j].first;
            if (!(d_prev < d_now)) {
                continue;
            }
            pair_dist_evals++;
            const T d_ij = dataset.dist(i_now, ordered[j].second);
            if (d_ij < d_now) {
                ok = false;
                break;
            }
        }
        if (ok) {
            kept.push_back(ordered[i]);
        }
    }
    return kept;
}

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_greedy_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned greedy_degree) {
    RunDiag out;
    if (beam_size == 0) {
        return out;
    }

    using T = float;
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), beam_size,
                                         greedy_degree);
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < beam_size) {
        candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }

    const auto t0 = Clock::now();
    std::vector<std::pair<T, unsigned>> neighbours;
    neighbours.reserve(256);

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        out.expansions++;

        neighbours.clear();
        for (const auto& x : graph.get_neighbours(current_node)) {
            neighbours.push_back({T(0), x.to});
        }
        dataset.dist_all_into(current_node, neighbours);
        std::ranges::sort(neighbours);
        neighbours = greedy_prune_neighbors(dataset, neighbours, neighbours.size(),
                                            out.distance_evals_pair);

        out.neighbours_selected += neighbours.size();
        auto it = std::ranges::remove_if(neighbours, [&](const auto& p) {
            return scratch.is_visited(p.second);
        });
        neighbours.erase(it.begin(), it.end());
        if (neighbours.size() > greedy_degree) {
            neighbours.resize(greedy_degree);
        }
        out.neighbours_unvisited += neighbours.size();
        dataset.dist_all_into(goal, neighbours);
        out.distance_evals += neighbours.size();

        // neighbours.clear();
        // for (const auto& x : graph.get_neighbours(current_node)) {
        //     out.neighbours_total++;
        //     if (scratch.is_visited(x.to)) {
        //         continue;
        //     }
        //     out.neighbours_unvisited++;
        //     neighbours.push_back({T(0), x.to});
        // }
        // if (neighbours.empty()) {
        //     continue;
        // }

        // // The query-time checker should first expose all in-range
        // neighbours,
        // // evaluate them against the current query, and only then apply the
        // // HNSW-like diversity heuristic.
        // dataset.dist_all_into(goal, neighbours);
        // out.distance_evals += neighbours.size();
        // std::ranges::sort(neighbours, [](const auto& a, const auto& b) {
        //     return a.first < b.first;
        // });

        // auto selected = greedy_prune_neighbors(dataset, neighbours,
        // greedy_degree,
        //                                        out.distance_evals_pair);
        // out.neighbours_selected += selected.size();
        // if (selected.empty()) {
        //     for (const auto& [qdist, nid] : neighbours) {
        //         scratch.mark_visited(nid, qdist);
        //     }
        //     continue;
        // }

        for (const auto& [qdist, nid] : neighbours) {
            scratch.mark_visited(nid, qdist);
        }
        for (const auto& [qdist, nid] : neighbours) {
            if (qdist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < qdist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {qdist, nid + offset});
            }
        }
    }
    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_strict_rng_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned rng_cap) {
    RunDiag out;
    if (beam_size == 0) {
        return out;
    }

    using T = float;
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), beam_size, beam_size);
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < beam_size) {
        candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }

    const auto t0 = Clock::now();
    std::vector<std::pair<T, unsigned>> neighbours;
    neighbours.reserve(256);

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        out.expansions++;

        neighbours.clear();
        for (const auto& x : graph.get_neighbours(current_node)) {
            out.neighbours_total++;
            neighbours.push_back({T(0), x.to});
        }
        if (neighbours.empty()) {
            continue;
        }

        dataset.dist_all_into(current_node, neighbours);
        out.distance_evals_aux += neighbours.size();
        std::ranges::sort(neighbours);
        neighbours = strict_rng_prune_neighbors(dataset, neighbours,
                                                out.distance_evals_pair);
        if (rng_cap > 0 && neighbours.size() > rng_cap) {
            neighbours.resize(rng_cap);
        }

        out.neighbours_selected += neighbours.size();
        auto it = std::ranges::remove_if(neighbours, [&](const auto& p) {
            return scratch.is_visited(p.second);
        });
        neighbours.erase(it.begin(), it.end());
        out.neighbours_unvisited += neighbours.size();
        if (neighbours.empty()) {
            continue;
        }

        dataset.dist_all_into(goal, neighbours);
        out.distance_evals += neighbours.size();

        for (const auto& [qdist, nid] : neighbours) {
            scratch.mark_visited(nid, qdist);
        }
        for (const auto& [qdist, nid] : neighbours) {
            if (qdist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < qdist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {qdist, nid + offset});
            }
        }
    }

    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

template <typename GoalId>
RunDiag beam_trace_domtag_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& base_graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned range_l,
    unsigned range_r,
    unsigned active_cap,
    DomTagCache& cache) {
    RunDiag out;
    (void)base_graph;
    if (beam_size == 0) {
        return out;
    }

    using T = float;
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), beam_size, beam_size);
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < beam_size) {
        candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }

    const auto t0 = Clock::now();
    std::vector<std::pair<T, unsigned>> neighbours;
    std::vector<unsigned char> present;
    neighbours.reserve(256);

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        out.expansions++;

        const auto& row = cache.get(current_node);
        if (row.nodes_sorted.empty()) {
            continue;
        }

        present.assign(row.nodes_sorted.size(), 0);
        for (size_t i = 0; i < row.nodes_sorted.size(); ++i) {
            const unsigned nid = row.nodes_sorted[i];
            if (nid >= range_l && nid < range_r) {
                present[i] = 1;
                out.neighbours_total++;
            }
        }

        neighbours.clear();
        for (size_t i = 0; i < row.nodes_sorted.size(); ++i) {
            if (!present[i]) {
                continue;
            }
            bool dominated = false;
            for (unsigned j : row.dominators[i]) {
                out.dominance_checks++;
                if (present[j]) {
                    dominated = true;
                    break;
                }
            }
            if (dominated) {
                continue;
            }
            neighbours.push_back({T(0), row.nodes_sorted[i]});
            if (active_cap > 0 && neighbours.size() >= active_cap) {
                break;
            }
        }

        out.neighbours_selected += neighbours.size();
        auto it = std::ranges::remove_if(neighbours, [&](const auto& p) {
            return scratch.is_visited(p.second);
        });
        neighbours.erase(it.begin(), it.end());
        out.neighbours_unvisited += neighbours.size();
        if (neighbours.empty()) {
            continue;
        }

        dataset.dist_all_into(goal, neighbours);
        out.distance_evals += neighbours.size();

        for (const auto& [qdist, nid] : neighbours) {
            scratch.mark_visited(nid, qdist);
        }
        for (const auto& [qdist, nid] : neighbours) {
            if (qdist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < qdist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {qdist, nid + offset});
            }
        }
    }

    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_mixdom_prune(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDFANN::Graph::TDGraphIndexBase& base_graph,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_size,
    unsigned local_offset,
    unsigned local_keep,
    unsigned dom_keep,
    unsigned range_l,
    unsigned range_r,
    DomTagCache& cache) {
    RunDiag out;
    (void)base_graph;
    if (beam_size == 0) {
        return out;
    }

    using T = float;
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), beam_size,
                                         std::max(1u, local_keep + dom_keep));
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < beam_size) {
        candidates.resize(beam_size, {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > beam_size) {
        candidates.resize(beam_size);
    }

    const auto t0 = Clock::now();
    std::vector<std::pair<T, unsigned>> neighbours;
    std::vector<unsigned> local_raw;
    std::vector<unsigned char> present;
    std::unordered_set<unsigned> picked;
    neighbours.reserve(256);
    local_raw.reserve(256);
    picked.reserve(256);

    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (candidates[uid].second < offset) {
            continue;
        }
        candidates[uid].second -= offset;
        const unsigned current_node = candidates[uid].second;
        out.expansions++;

        neighbours.clear();
        local_raw.clear();
        picked.clear();

        for (const auto& x : graph.get_neighbours(current_node)) {
            out.neighbours_total++;
            local_raw.push_back(x.to);
        }

        const size_t local_begin =
            std::min<size_t>(local_offset, local_raw.size());
        const size_t local_end = std::min<size_t>(
            local_begin + static_cast<size_t>(local_keep), local_raw.size());
        for (size_t i = local_begin; i < local_end; ++i) {
            if (!picked.insert(local_raw[i]).second) {
                continue;
            }
            neighbours.push_back({T(0), local_raw[i]});
        }

        const auto& row = cache.get(current_node);
        if (!row.nodes_sorted.empty() && dom_keep > 0) {
            present.assign(row.nodes_sorted.size(), 0);
            for (size_t i = 0; i < row.nodes_sorted.size(); ++i) {
                const unsigned nid = row.nodes_sorted[i];
                if (nid >= range_l && nid < range_r) {
                    present[i] = 1;
                }
            }

            unsigned dom_added = 0;
            for (size_t i = 0; i < row.nodes_sorted.size(); ++i) {
                if (!present[i]) {
                    continue;
                }
                const unsigned nid = row.nodes_sorted[i];
                if (picked.contains(nid)) {
                    continue;
                }
                bool dominated = false;
                for (unsigned j : row.dominators[i]) {
                    out.dominance_checks++;
                    if (present[j]) {
                        dominated = true;
                        break;
                    }
                }
                if (dominated) {
                    continue;
                }
                neighbours.push_back({T(0), nid});
                picked.insert(nid);
                dom_added++;
                out.neighbours_selected_extra++;
                if (dom_added >= dom_keep) {
                    break;
                }
            }
        }

        out.neighbours_selected += neighbours.size();
        auto it = std::ranges::remove_if(neighbours, [&](const auto& p) {
            return scratch.is_visited(p.second);
        });
        neighbours.erase(it.begin(), it.end());
        out.neighbours_unvisited += neighbours.size();
        if (neighbours.empty()) {
            continue;
        }

        dataset.dist_all_into(goal, neighbours);
        out.distance_evals += neighbours.size();

        for (const auto& [qdist, nid] : neighbours) {
            scratch.mark_visited(nid, qdist);
        }
        for (const auto& [qdist, nid] : neighbours) {
            if (qdist < candidates.back().first) {
                candidates.pop_back();
                auto it = std::partition_point(
                    candidates.begin(), candidates.end(),
                    [&](const auto& a) { return a.first < qdist; });
                const int pos = static_cast<int>(it - candidates.begin());
                uid = std::min(uid, pos - 1);
                candidates.insert(it, {qdist, nid + offset});
            }
        }
    }

    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

template <typename GoalId, TDFANN::Graph::GraphLike G>
RunDiag beam_trace_two_stage(
    const TDFANN::Vector::VectorList<float>& dataset,
    const G& graph,
    const GoalId& goal,
    unsigned k,
    const std::vector<unsigned>& start_nodes,
    unsigned beam_route,
    unsigned trunc_route,
    unsigned route_expansions,
    unsigned beam_refine,
    unsigned trunc_refine) {
    RunDiag out;
    if (beam_refine == 0 || trunc_refine == 0) {
        return out;
    }
    beam_route = std::min(beam_route, beam_refine);
    if (beam_route == 0) {
        beam_route = std::min(beam_refine, 1u);
    }

    using T = float;
    TDFANN::RNSG::BeamScratch<T> scratch(dataset.size(), beam_refine,
                                         std::max(trunc_route, trunc_refine));
    scratch.next_query();

    auto& visited_nodes = scratch.visited_nodes;
    auto& candidates = scratch.candidates;
    const unsigned offset = dataset.size();

    for (auto id : start_nodes) {
        candidates.push_back({T(0), id});
    }
    if (candidates.empty()) {
        return out;
    }

    dataset.dist_all_into(goal, candidates);
    std::ranges::sort(candidates);
    out.distance_evals += candidates.size();
    for (auto& [dis, id] : candidates) {
        scratch.mark_visited(id, dis);
        id += offset;
    }

    if (candidates.size() < beam_route) {
        candidates.resize(beam_route, {T(1e100), candidates[0].second - offset});
    } else if (candidates.size() > beam_route) {
        candidates.resize(beam_route);
    }

    auto run_phase = [&](unsigned active_beam, unsigned trunc_size,
                         std::optional<unsigned> max_expansions) {
        std::vector<std::pair<T, unsigned>> neighbours;
        neighbours.reserve(256);
        unsigned phase_exp = 0;
        for (int uid = 0; uid < static_cast<int>(active_beam); ++uid) {
            if (max_expansions && phase_exp >= *max_expansions) {
                break;
            }
            if (candidates[uid].second < offset) {
                continue;
            }
            candidates[uid].second -= offset;
            const unsigned current_node = candidates[uid].second;
            out.expansions++;
            phase_exp++;

            neighbours.clear();
            for (const auto& x : graph.get_neighbours(current_node)) {
                out.neighbours_total++;
                if (scratch.is_visited(x.to)) {
                    continue;
                }
                out.neighbours_unvisited++;
                neighbours.push_back({T(0), x.to});
                if (neighbours.size() >= trunc_size) {
                    break;
                }
            }
            out.neighbours_selected += neighbours.size();
            if (neighbours.empty()) {
                continue;
            }

            dataset.dist_all_into(goal, neighbours);
            out.distance_evals += neighbours.size();

            for (const auto& [qdist, nid] : neighbours) {
                scratch.mark_visited(nid, qdist);
            }
            for (const auto& [qdist, nid] : neighbours) {
                if (qdist >= candidates[active_beam - 1].first) {
                    continue;
                }
                auto it = std::partition_point(
                    candidates.begin(), candidates.begin() + active_beam,
                    [&](const auto& a) { return a.first < qdist; });
                const size_t pos = static_cast<size_t>(it - candidates.begin());
                candidates.insert(it, {qdist, nid + offset});
                candidates.erase(candidates.begin() + active_beam);
                uid = std::min(uid, static_cast<int>(pos) - 1);
            }
        }
    };

    const auto t0 = Clock::now();
    run_phase(beam_route, trunc_route,
              route_expansions > 0 ? std::optional<unsigned>(route_expansions)
                                   : std::nullopt);

    if (candidates.size() < beam_refine) {
        candidates.resize(beam_refine, {T(1e100), 0});
    } else if (candidates.size() > beam_refine) {
        candidates.resize(beam_refine);
    }
    run_phase(beam_refine, trunc_refine, std::nullopt);

    const auto t1 = Clock::now();
    out.query_ns =
        static_cast<double>(std::chrono::duration_cast<ns>(t1 - t0).count());
    out.visited_nodes = visited_nodes.size();

    const size_t out_size = std::min<size_t>(k, candidates.size());
    out.result.reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        auto [dist, id] = candidates[i];
        if (id >= offset) {
            id -= offset;
        }
        out.result.push_back({dist, id});
    }
    return out;
}

struct QueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double overlap_ref_test = 0.0;
    double recall_ref = 0.0;
    double recall_test = 0.0;
    double recall_drop = 0.0;

    double miss_rate_ref = 0.0;
    double miss_rate_test = 0.0;
    double miss_not_scanned_rate_ref = 0.0;
    double miss_not_scanned_rate_test = 0.0;
    double miss_scanned_not_selected_rate_ref = 0.0;
    double miss_scanned_not_selected_rate_test = 0.0;
    double better_unselected_per_exp_ref = 0.0;
    double better_unselected_per_exp_test = 0.0;
    double topk_hit_ratio_ref = 0.0;
    double topk_hit_ratio_test = 0.0;
    double oracle_best_rank_ref = 0.0;
    double oracle_best_rank_test = 0.0;
    double oracle_best_rank_missed_ref = 0.0;
    double oracle_best_rank_missed_test = 0.0;

    double distance_per_exp_ref = 0.0;
    double distance_per_exp_test = 0.0;
    double query_ns_ref = 0.0;
    double query_ns_test = 0.0;
};

struct SingleQueryReport {
    unsigned query_i = 0;
    unsigned mapped_query_id = 0;
    std::uint64_t range_size = 0;

    double recall = 0.0;
    double miss_rate = 0.0;
    double miss_not_scanned_rate = 0.0;
    double miss_scanned_not_selected_rate = 0.0;
    double better_unselected_per_exp = 0.0;
    double topk_hit_ratio = 0.0;
    double oracle_best_rank = 0.0;
    double oracle_best_rank_missed = 0.0;
    double distance_per_exp = 0.0;
    double query_ns = 0.0;
    double scan_coverage = 0.0;
    double selected_rank_pct_mean_global = 0.0;
    double selected_gap_mean_ratio = 0.0;
    double oracle_gap_mean_ratio = 0.0;
    double selected_span_ratio = 0.0;
    double oracle_span_ratio = 0.0;
    double selected_far_ratio = 0.0;
    double oracle_far_ratio = 0.0;
};

int run_collect_test(const std::string& dataset_file,
                     const std::string& index_file,
                     const std::string& query_file,
                     const std::string& label_file,
                     const std::string& qrange_file,
                     const std::string& gt_file,
                     unsigned qnumber,
                     unsigned beam_size,
                     unsigned trunc_size,
                     unsigned sample_queries,
                     unsigned seed,
                     const DiagConfig& nav_cfg,
                     double recall_threshold,
                     const std::string& out_json,
                     const std::string& out_md,
                     const std::string& out_per_query_json,
                     const std::string& out_per_expansion_json) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    auto gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
    if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
        throw std::runtime_error("groundtruth size mismatch");
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    for (auto& id : gt) {
        if (id >= pos.size()) {
            throw std::runtime_error("groundtruth id out of range");
        }
        id = pos[id];
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<SingleQueryReport> reports;
    reports.reserve(ids.size());
    nlohmann::json expansion_arr = nlohmann::json::array();

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();

        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg = nav_cfg;
        cfg.beam_size = beam_size;
        cfg.trunc_size = trunc_size;
        auto run = beam_trace_diag(dataset, g_sub, query_list[ui], qnumber,
                                   starts, cfg,
                                   static_cast<unsigned>(range_l),
                                   static_cast<unsigned>(range_r));

        SingleQueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);

        std::vector<unsigned> gt_ids(qnumber);
        for (unsigned j = 0; j < qnumber; ++j) {
            gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
        }
        r.recall = topk_recall_vs_gt(run.result, gt_ids, qnumber);

        r.miss_rate = run.expansions == 0
                          ? 0.0
                          : static_cast<double>(run.oracle_best_missed) /
                                static_cast<double>(run.expansions);
        r.miss_not_scanned_rate =
            run.expansions == 0
                ? 0.0
                : static_cast<double>(run.miss_not_scanned) /
                      static_cast<double>(run.expansions);
        r.miss_scanned_not_selected_rate =
            run.expansions == 0
                ? 0.0
                : static_cast<double>(run.miss_scanned_not_selected) /
                      static_cast<double>(run.expansions);
        r.better_unselected_per_exp =
            run.expansions == 0
                ? 0.0
                : static_cast<double>(run.better_unselected_than_worst) /
                      static_cast<double>(run.expansions);
        r.topk_hit_ratio = run.topk_total == 0
                               ? 0.0
                               : static_cast<double>(run.topk_hits) /
                                     static_cast<double>(run.topk_total);
        r.oracle_best_rank =
            run.oracle_rank_samples == 0
                ? 0.0
                : run.oracle_rank_sum /
                      static_cast<double>(run.oracle_rank_samples);
        r.oracle_best_rank_missed =
            run.oracle_rank_missed_samples == 0
                ? 0.0
                : run.oracle_rank_missed_sum /
                      static_cast<double>(run.oracle_rank_missed_samples);
        r.distance_per_exp =
            run.expansions == 0
                ? 0.0
                : static_cast<double>(run.distance_evals) /
                      static_cast<double>(run.expansions);
        r.query_ns = run.query_ns;

        auto avg_exp = [&](auto fn) {
            if (run.expansion_rows.empty()) {
                return 0.0;
            }
            double sum = 0.0;
            for (const auto& e : run.expansion_rows) {
                sum += fn(e);
            }
            return sum / static_cast<double>(run.expansion_rows.size());
        };
        r.scan_coverage =
            avg_exp([](const ExpansionProbe& e) { return e.scan_coverage; });
        r.selected_rank_pct_mean_global = avg_exp(
            [](const ExpansionProbe& e) {
                return e.selected_rank_pct_mean_global;
            });
        r.selected_gap_mean_ratio = avg_exp(
            [](const ExpansionProbe& e) { return e.selected_gap_mean_ratio; });
        r.oracle_gap_mean_ratio = avg_exp(
            [](const ExpansionProbe& e) { return e.oracle_gap_mean_ratio; });
        r.selected_span_ratio =
            avg_exp([](const ExpansionProbe& e) { return e.selected_span_ratio; });
        r.oracle_span_ratio =
            avg_exp([](const ExpansionProbe& e) { return e.oracle_span_ratio; });
        r.selected_far_ratio =
            avg_exp([](const ExpansionProbe& e) { return e.selected_far_ratio; });
        r.oracle_far_ratio =
            avg_exp([](const ExpansionProbe& e) { return e.oracle_far_ratio; });

        if (!out_per_expansion_json.empty()) {
            for (const auto& e : run.expansion_rows) {
                expansion_arr.push_back({
                    {"query_i", r.query_i},
                    {"mapped_query_id", r.mapped_query_id},
                    {"range_size", r.range_size},
                    {"step", e.step},
                    {"current_node", e.current_node},
                    {"unvisited_deg", e.unvisited_deg},
                    {"scanned_deg", e.scanned_deg},
                    {"selected_deg", e.selected_deg},
                    {"scan_coverage", e.scan_coverage},
                    {"oracle_best_rank_in_adj", e.oracle_best_rank_in_adj},
                    {"selected_rank_pct_mean_global",
                     e.selected_rank_pct_mean_global},
                    {"selected_gap_mean_ratio", e.selected_gap_mean_ratio},
                    {"oracle_gap_mean_ratio", e.oracle_gap_mean_ratio},
                    {"selected_span_ratio", e.selected_span_ratio},
                    {"oracle_span_ratio", e.oracle_span_ratio},
                    {"selected_far_ratio", e.selected_far_ratio},
                    {"oracle_far_ratio", e.oracle_far_ratio},
                });
            }
        }

        reports.push_back(r);
        if (((t + 1) % 64 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[collect] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto recall_vec = collect([](const SingleQueryReport& r) { return r.recall; });
    auto miss_vec = collect([](const SingleQueryReport& r) { return r.miss_rate; });
    auto miss_not_scanned_vec =
        collect([](const SingleQueryReport& r) { return r.miss_not_scanned_rate; });
    auto miss_scanned_not_selected_vec = collect(
        [](const SingleQueryReport& r) { return r.miss_scanned_not_selected_rate; });
    auto better_vec =
        collect([](const SingleQueryReport& r) { return r.better_unselected_per_exp; });
    auto hit_vec = collect([](const SingleQueryReport& r) { return r.topk_hit_ratio; });
    auto rank_vec = collect([](const SingleQueryReport& r) { return r.oracle_best_rank; });
    auto rank_miss_vec =
        collect([](const SingleQueryReport& r) { return r.oracle_best_rank_missed; });
    auto dpe_vec =
        collect([](const SingleQueryReport& r) { return r.distance_per_exp; });
    auto qns_vec = collect([](const SingleQueryReport& r) { return r.query_ns; });
    auto scan_cov_vec =
        collect([](const SingleQueryReport& r) { return r.scan_coverage; });
    auto sel_rank_pct_vec = collect([](const SingleQueryReport& r) {
        return r.selected_rank_pct_mean_global;
    });
    auto sel_gap_vec = collect([](const SingleQueryReport& r) {
        return r.selected_gap_mean_ratio;
    });
    auto oracle_gap_vec = collect([](const SingleQueryReport& r) {
        return r.oracle_gap_mean_ratio;
    });
    auto sel_span_vec = collect([](const SingleQueryReport& r) {
        return r.selected_span_ratio;
    });
    auto oracle_span_vec = collect([](const SingleQueryReport& r) {
        return r.oracle_span_ratio;
    });
    auto sel_far_vec = collect([](const SingleQueryReport& r) {
        return r.selected_far_ratio;
    });
    auto oracle_far_vec = collect([](const SingleQueryReport& r) {
        return r.oracle_far_ratio;
    });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_size", trunc_size},
        {"nav_degree", nav_cfg.nav_degree},
        {"nav_scan_factor", nav_cfg.nav_scan_factor},
        {"nav_stall_rounds", nav_cfg.nav_stall_rounds},
        {"nav_front_keep", nav_cfg.nav_front_keep},
        {"nav_tail_degree", nav_cfg.nav_tail_degree},
        {"nav_early_stop_rounds", nav_cfg.nav_early_stop_rounds},
        {"bridge_gap_ratio", nav_cfg.bridge_gap_ratio},
        {"recall_threshold", recall_threshold},
    };
    summary["recall"] = stats_to_json(summarize(recall_vec));
    summary["miss_rate"] = stats_to_json(summarize(miss_vec));
    summary["miss_not_scanned_rate"] =
        stats_to_json(summarize(miss_not_scanned_vec));
    summary["miss_scanned_not_selected_rate"] =
        stats_to_json(summarize(miss_scanned_not_selected_vec));
    summary["better_unselected_per_exp"] =
        stats_to_json(summarize(better_vec));
    summary["topk_hit_ratio"] = stats_to_json(summarize(hit_vec));
    summary["oracle_best_rank"] = stats_to_json(summarize(rank_vec));
    summary["oracle_best_rank_missed"] =
        stats_to_json(summarize(rank_miss_vec));
    summary["distance_per_exp"] = stats_to_json(summarize(dpe_vec));
    summary["query_ns"] = stats_to_json(summarize(qns_vec));
    summary["scan_coverage"] = stats_to_json(summarize(scan_cov_vec));
    summary["selected_rank_pct_mean_global"] =
        stats_to_json(summarize(sel_rank_pct_vec));
    summary["selected_gap_mean_ratio"] =
        stats_to_json(summarize(sel_gap_vec));
    summary["oracle_gap_mean_ratio"] =
        stats_to_json(summarize(oracle_gap_vec));
    summary["selected_span_ratio"] =
        stats_to_json(summarize(sel_span_vec));
    summary["oracle_span_ratio"] =
        stats_to_json(summarize(oracle_span_vec));
    summary["selected_far_ratio"] =
        stats_to_json(summarize(sel_far_vec));
    summary["oracle_far_ratio"] =
        stats_to_json(summarize(oracle_far_vec));

    const double recall_mean = summary["recall"]["mean"].get<double>();
    summary["pass_threshold"] = (recall_mean >= recall_threshold);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_per_query_json.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : reports) {
            arr.push_back({
                {"query_i", r.query_i},
                {"mapped_query_id", r.mapped_query_id},
                {"range_size", r.range_size},
                {"recall", r.recall},
                {"miss_rate", r.miss_rate},
                {"miss_not_scanned_rate", r.miss_not_scanned_rate},
                {"miss_scanned_not_selected_rate",
                 r.miss_scanned_not_selected_rate},
                {"better_unselected_per_exp", r.better_unselected_per_exp},
                {"topk_hit_ratio", r.topk_hit_ratio},
                {"oracle_best_rank", r.oracle_best_rank},
                {"oracle_best_rank_missed", r.oracle_best_rank_missed},
                {"distance_per_exp", r.distance_per_exp},
                {"query_ns", r.query_ns},
                {"scan_coverage", r.scan_coverage},
                {"selected_rank_pct_mean_global",
                 r.selected_rank_pct_mean_global},
                {"selected_gap_mean_ratio", r.selected_gap_mean_ratio},
                {"oracle_gap_mean_ratio", r.oracle_gap_mean_ratio},
                {"selected_span_ratio", r.selected_span_ratio},
                {"oracle_span_ratio", r.oracle_span_ratio},
                {"selected_far_ratio", r.selected_far_ratio},
                {"oracle_far_ratio", r.oracle_far_ratio},
            });
        }
        std::ofstream fout(out_per_query_json);
        fout << arr.dump(2);
    }

    if (!out_per_expansion_json.empty()) {
        std::ofstream fout(out_per_expansion_json);
        fout << expansion_arr.dump(2);
    }

    if (!out_md.empty()) {
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Collect+Test (Single Config)");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- beam_size: `" + std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_size: `" + std::to_string(trunc_size) + "`");
        lines.emplace_back("- recall_threshold: `" + fmt(recall_threshold) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Core Metrics");
        lines.emplace_back("- recall.mean: `" +
                           fmt(summary["recall"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back("- pass_threshold: `" +
                           std::string(summary["pass_threshold"].get<bool>()
                                           ? "true"
                                           : "false") +
                           "`");
        lines.emplace_back("- miss_rate.mean: `" +
                           fmt(summary["miss_rate"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back(
            "- miss_not_scanned_rate.mean: `" +
            fmt(summary["miss_not_scanned_rate"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- miss_scanned_not_selected_rate.mean: `" +
            fmt(summary["miss_scanned_not_selected_rate"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- better_unselected_per_exp.mean: `" +
            fmt(summary["better_unselected_per_exp"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- topk_hit_ratio.mean: `" +
            fmt(summary["topk_hit_ratio"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_best_rank.mean: `" +
            fmt(summary["oracle_best_rank"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- distance_per_exp.mean: `" +
            fmt(summary["distance_per_exp"]["mean"].get<double>()) + "`");
        lines.emplace_back("");
        lines.emplace_back("## Prefix Quality");
        lines.emplace_back(
            "- scan_coverage.mean: `" +
            fmt(summary["scan_coverage"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- selected_rank_pct_mean_global.mean: `" +
            fmt(summary["selected_rank_pct_mean_global"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- selected_gap_mean_ratio.mean: `" +
            fmt(summary["selected_gap_mean_ratio"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_gap_mean_ratio.mean: `" +
            fmt(summary["oracle_gap_mean_ratio"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- selected_span_ratio.mean: `" +
            fmt(summary["selected_span_ratio"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_span_ratio.mean: `" +
            fmt(summary["oracle_span_ratio"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- selected_far_ratio.mean: `" +
            fmt(summary["selected_far_ratio"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_far_ratio.mean: `" +
            fmt(summary["oracle_far_ratio"]["mean"].get<double>()) + "`");
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[collect] sampled_queries={}", reports.size());
    spdlog::info("[collect] recall.mean={:.6f}", recall_mean);
    spdlog::info("[collect] pass_threshold={} (threshold={:.6f})",
                 recall_mean >= recall_threshold, recall_threshold);
    spdlog::info("[collect] miss_rate.mean={:.6f}",
                 summary["miss_rate"]["mean"].get<double>());
    spdlog::info("[collect] topk_hit_ratio.mean={:.6f}",
                 summary["topk_hit_ratio"]["mean"].get<double>());
    spdlog::info("[collect] selected_rank_pct_mean_global.mean={:.6f}",
                 summary["selected_rank_pct_mean_global"]["mean"].get<double>());
    spdlog::info("[collect] selected_span_ratio.mean={:.6f}, oracle_span_ratio.mean={:.6f}",
                 summary["selected_span_ratio"]["mean"].get<double>(),
                 summary["oracle_span_ratio"]["mean"].get<double>());

    if (recall_mean < recall_threshold) {
        spdlog::error(
            "[collect] recall.mean {:.6f} is below threshold {:.6f}",
            recall_mean, recall_threshold);
        return 2;
    }
    return 0;
}

int run_trunc_diag(const std::string& dataset_file,
                   const std::string& index_file,
                   const std::string& query_file,
                   const std::string& label_file,
                   const std::string& qrange_file,
                   const std::string& gt_file,
                   unsigned qnumber,
                   unsigned beam_size,
                   unsigned trunc_ref,
                   unsigned trunc_test,
                   unsigned sample_queries,
                   unsigned seed,
                   const DiagConfig& nav_cfg,
                   const std::string& out_json,
                   const std::string& out_md,
                   const std::string& out_per_query_json) {
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    TDFANN::Graph::TDGraphIndexBase index(index_file);
    TDFANN::Vector::VectorList<float> query_list(query_file);

    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto qrange = TDFANN::IO::load_json_to_vec<std::uint64_t>(qrange_file);
    if (qrange.size() != static_cast<size_t>(query_list.size()) * 2) {
        throw std::runtime_error("qrange size mismatch with query count");
    }

    std::vector<unsigned> gt;
    const bool has_gt = !gt_file.empty();
    if (has_gt) {
        gt = TDFANN::IO::load_json_to_vec<unsigned>(gt_file);
        if (gt.size() != static_cast<size_t>(query_list.size()) * qnumber) {
            throw std::runtime_error("groundtruth size mismatch");
        }
    }

    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    if (has_gt) {
        for (auto& id : gt) {
            if (id >= pos.size()) {
                throw std::runtime_error("groundtruth id out of range");
            }
            id = pos[id];
        }
    }
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    dataset.reorder(ord);

    auto ids = sample_ids(query_list.size(), sample_queries, seed);
    std::vector<QueryReport> reports;
    reports.reserve(ids.size());

    std::vector<double> v_overlap, v_recall_drop, v_miss_test, v_better_test;
    v_overlap.reserve(ids.size());
    v_recall_drop.reserve(ids.size());
    v_miss_test.reserve(ids.size());
    v_better_test.reserve(ids.size());

    for (size_t t = 0; t < ids.size(); ++t) {
        const unsigned qi = ids[t];
        const unsigned ui = qi;

        const auto ql = qrange[ui * 2];
        const auto qr = qrange[ui * 2 + 1];
        const auto range_l =
            std::ranges::lower_bound(sorted_label, ql) - sorted_label.begin();
        const auto range_r =
            std::ranges::upper_bound(sorted_label, qr) - sorted_label.begin();

        if (range_l >= range_r) {
            continue;
        }

        auto g_sub = index(sorted_label, ql, qr);
        auto header = TDFANN::Utils::to_vector(g_sub.get_header());
        std::vector<unsigned> starts;
        if (!header.empty()) {
            starts = header;
        } else {
            starts.push_back(static_cast<unsigned>(range_l));
        }

        DiagConfig cfg_ref = nav_cfg;
        cfg_ref.beam_size = beam_size;
        cfg_ref.trunc_size = trunc_ref;
        DiagConfig cfg_test = nav_cfg;
        cfg_test.beam_size = beam_size;
        cfg_test.trunc_size = trunc_test;

        auto ref = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_ref,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));
        auto test = beam_trace_diag(
            dataset, g_sub, query_list[ui], qnumber, starts, cfg_test,
            static_cast<unsigned>(range_l), static_cast<unsigned>(range_r));

        QueryReport r;
        r.query_i = qi;
        r.mapped_query_id = ui;
        r.range_size = static_cast<std::uint64_t>(range_r - range_l);
        r.overlap_ref_test = topk_overlap(ref.result, test.result, qnumber);

        if (has_gt) {
            std::vector<unsigned> gt_ids(qnumber);
            for (unsigned j = 0; j < qnumber; ++j) {
                gt_ids[j] = gt[static_cast<size_t>(ui) * qnumber + j];
            }
            r.recall_ref = topk_recall_vs_gt(ref.result, gt_ids, qnumber);
            r.recall_test = topk_recall_vs_gt(test.result, gt_ids, qnumber);
            r.recall_drop = r.recall_ref - r.recall_test;
        }

        r.miss_rate_ref = ref.expansions == 0
                              ? 0.0
                              : static_cast<double>(ref.oracle_best_missed) /
                                    static_cast<double>(ref.expansions);
        r.miss_rate_test = test.expansions == 0
                               ? 0.0
                               : static_cast<double>(test.oracle_best_missed) /
                                     static_cast<double>(test.expansions);
        r.miss_not_scanned_rate_ref =
            ref.expansions == 0 ? 0.0
                                : static_cast<double>(ref.miss_not_scanned) /
                                      static_cast<double>(ref.expansions);
        r.miss_not_scanned_rate_test =
            test.expansions == 0 ? 0.0
                                 : static_cast<double>(test.miss_not_scanned) /
                                       static_cast<double>(test.expansions);
        r.miss_scanned_not_selected_rate_ref =
            ref.expansions == 0
                ? 0.0
                : static_cast<double>(ref.miss_scanned_not_selected) /
                      static_cast<double>(ref.expansions);
        r.miss_scanned_not_selected_rate_test =
            test.expansions == 0
                ? 0.0
                : static_cast<double>(test.miss_scanned_not_selected) /
                      static_cast<double>(test.expansions);
        r.better_unselected_per_exp_ref =
            ref.expansions == 0
                ? 0.0
                : static_cast<double>(ref.better_unselected_than_worst) /
                      static_cast<double>(ref.expansions);
        r.better_unselected_per_exp_test =
            test.expansions == 0
                ? 0.0
                : static_cast<double>(test.better_unselected_than_worst) /
                      static_cast<double>(test.expansions);
        r.topk_hit_ratio_ref = ref.topk_total == 0
                                   ? 0.0
                                   : static_cast<double>(ref.topk_hits) /
                                         static_cast<double>(ref.topk_total);
        r.topk_hit_ratio_test = test.topk_total == 0
                                    ? 0.0
                                    : static_cast<double>(test.topk_hits) /
                                          static_cast<double>(test.topk_total);
        r.oracle_best_rank_ref =
            ref.oracle_rank_samples == 0
                ? 0.0
                : ref.oracle_rank_sum /
                      static_cast<double>(ref.oracle_rank_samples);
        r.oracle_best_rank_test =
            test.oracle_rank_samples == 0
                ? 0.0
                : test.oracle_rank_sum /
                      static_cast<double>(test.oracle_rank_samples);
        r.oracle_best_rank_missed_ref =
            ref.oracle_rank_missed_samples == 0
                ? 0.0
                : ref.oracle_rank_missed_sum /
                      static_cast<double>(ref.oracle_rank_missed_samples);
        r.oracle_best_rank_missed_test =
            test.oracle_rank_missed_samples == 0
                ? 0.0
                : test.oracle_rank_missed_sum /
                      static_cast<double>(test.oracle_rank_missed_samples);
        r.distance_per_exp_ref = ref.expansions == 0
                                     ? 0.0
                                     : static_cast<double>(ref.distance_evals) /
                                           static_cast<double>(ref.expansions);
        r.distance_per_exp_test =
            test.expansions == 0 ? 0.0
                                 : static_cast<double>(test.distance_evals) /
                                       static_cast<double>(test.expansions);
        r.query_ns_ref = ref.query_ns;
        r.query_ns_test = test.query_ns;

        reports.push_back(r);
        v_overlap.push_back(r.overlap_ref_test);
        v_recall_drop.push_back(r.recall_drop);
        v_miss_test.push_back(r.miss_rate_test);
        v_better_test.push_back(r.better_unselected_per_exp_test);

        if (((t + 1) % 64 == 0) || (t + 1 == ids.size())) {
            spdlog::info("[diag] processed {}/{}", t + 1, ids.size());
        }
    }

    auto collect = [&](auto fn) {
        std::vector<double> out;
        out.reserve(reports.size());
        for (const auto& r : reports) {
            out.push_back(fn(r));
        }
        return out;
    };

    auto overlap_vec =
        collect([](const QueryReport& r) { return r.overlap_ref_test; });
    auto miss_ref_vec =
        collect([](const QueryReport& r) { return r.miss_rate_ref; });
    auto miss_test_vec =
        collect([](const QueryReport& r) { return r.miss_rate_test; });
    auto miss_not_scanned_ref_vec = collect(
        [](const QueryReport& r) { return r.miss_not_scanned_rate_ref; });
    auto miss_not_scanned_test_vec = collect(
        [](const QueryReport& r) { return r.miss_not_scanned_rate_test; });
    auto miss_scanned_not_selected_ref_vec = collect([](const QueryReport& r) {
        return r.miss_scanned_not_selected_rate_ref;
    });
    auto miss_scanned_not_selected_test_vec = collect([](const QueryReport& r) {
        return r.miss_scanned_not_selected_rate_test;
    });
    auto better_ref_vec = collect(
        [](const QueryReport& r) { return r.better_unselected_per_exp_ref; });
    auto better_test_vec = collect(
        [](const QueryReport& r) { return r.better_unselected_per_exp_test; });
    auto hit_ref_vec =
        collect([](const QueryReport& r) { return r.topk_hit_ratio_ref; });
    auto hit_test_vec =
        collect([](const QueryReport& r) { return r.topk_hit_ratio_test; });
    auto oracle_rank_ref_vec =
        collect([](const QueryReport& r) { return r.oracle_best_rank_ref; });
    auto oracle_rank_test_vec =
        collect([](const QueryReport& r) { return r.oracle_best_rank_test; });
    auto oracle_rank_missed_ref_vec = collect(
        [](const QueryReport& r) { return r.oracle_best_rank_missed_ref; });
    auto oracle_rank_missed_test_vec = collect(
        [](const QueryReport& r) { return r.oracle_best_rank_missed_test; });
    auto dpe_ref_vec =
        collect([](const QueryReport& r) { return r.distance_per_exp_ref; });
    auto dpe_test_vec =
        collect([](const QueryReport& r) { return r.distance_per_exp_test; });
    auto qns_ref_vec =
        collect([](const QueryReport& r) { return r.query_ns_ref; });
    auto qns_test_vec =
        collect([](const QueryReport& r) { return r.query_ns_test; });
    auto recall_ref_vec =
        collect([](const QueryReport& r) { return r.recall_ref; });
    auto recall_test_vec =
        collect([](const QueryReport& r) { return r.recall_test; });
    auto recall_drop_vec =
        collect([](const QueryReport& r) { return r.recall_drop; });

    nlohmann::json summary;
    summary["sampled_queries"] = reports.size();
    summary["has_groundtruth"] = has_gt;
    summary["config"] = {
        {"beam_size", beam_size},
        {"trunc_ref", trunc_ref},
        {"trunc_test", trunc_test},
        {"nav_degree", nav_cfg.nav_degree},
        {"nav_scan_factor", nav_cfg.nav_scan_factor},
        {"nav_stall_rounds", nav_cfg.nav_stall_rounds},
        {"nav_front_keep", nav_cfg.nav_front_keep},
        {"nav_tail_degree", nav_cfg.nav_tail_degree},
        {"nav_early_stop_rounds", nav_cfg.nav_early_stop_rounds},
    };
    summary["overlap_ref_test"] = stats_to_json(summarize(overlap_vec));
    summary["miss_rate_ref"] = stats_to_json(summarize(miss_ref_vec));
    summary["miss_rate_test"] = stats_to_json(summarize(miss_test_vec));
    summary["miss_not_scanned_rate_ref"] =
        stats_to_json(summarize(miss_not_scanned_ref_vec));
    summary["miss_not_scanned_rate_test"] =
        stats_to_json(summarize(miss_not_scanned_test_vec));
    summary["miss_scanned_not_selected_rate_ref"] =
        stats_to_json(summarize(miss_scanned_not_selected_ref_vec));
    summary["miss_scanned_not_selected_rate_test"] =
        stats_to_json(summarize(miss_scanned_not_selected_test_vec));
    summary["better_unselected_per_exp_ref"] =
        stats_to_json(summarize(better_ref_vec));
    summary["better_unselected_per_exp_test"] =
        stats_to_json(summarize(better_test_vec));
    summary["topk_hit_ratio_ref"] = stats_to_json(summarize(hit_ref_vec));
    summary["topk_hit_ratio_test"] = stats_to_json(summarize(hit_test_vec));
    summary["oracle_best_rank_ref"] =
        stats_to_json(summarize(oracle_rank_ref_vec));
    summary["oracle_best_rank_test"] =
        stats_to_json(summarize(oracle_rank_test_vec));
    summary["oracle_best_rank_missed_ref"] =
        stats_to_json(summarize(oracle_rank_missed_ref_vec));
    summary["oracle_best_rank_missed_test"] =
        stats_to_json(summarize(oracle_rank_missed_test_vec));
    summary["distance_per_exp_ref"] = stats_to_json(summarize(dpe_ref_vec));
    summary["distance_per_exp_test"] = stats_to_json(summarize(dpe_test_vec));
    summary["query_ns_ref"] = stats_to_json(summarize(qns_ref_vec));
    summary["query_ns_test"] = stats_to_json(summarize(qns_test_vec));
    summary["recall_ref"] = stats_to_json(summarize(recall_ref_vec));
    summary["recall_test"] = stats_to_json(summarize(recall_test_vec));
    summary["recall_drop"] = stats_to_json(summarize(recall_drop_vec));
    summary["corr_overlap_vs_miss_test"] =
        correlation(overlap_vec, miss_test_vec);
    summary["corr_recall_drop_vs_miss_test"] =
        correlation(recall_drop_vec, miss_test_vec);
    summary["corr_recall_drop_vs_miss_not_scanned_test"] =
        correlation(recall_drop_vec, miss_not_scanned_test_vec);
    summary["corr_recall_drop_vs_miss_scanned_not_selected_test"] =
        correlation(recall_drop_vec, miss_scanned_not_selected_test_vec);
    summary["corr_recall_drop_vs_better_test"] =
        correlation(recall_drop_vec, better_test_vec);

    if (!out_json.empty()) {
        std::ofstream fout(out_json);
        fout << summary.dump(2);
    }

    if (!out_per_query_json.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : reports) {
            arr.push_back({
                {"query_i", r.query_i},
                {"mapped_query_id", r.mapped_query_id},
                {"range_size", r.range_size},
                {"overlap_ref_test", r.overlap_ref_test},
                {"recall_ref", r.recall_ref},
                {"recall_test", r.recall_test},
                {"recall_drop", r.recall_drop},
                {"miss_rate_ref", r.miss_rate_ref},
                {"miss_rate_test", r.miss_rate_test},
                {"miss_not_scanned_rate_ref", r.miss_not_scanned_rate_ref},
                {"miss_not_scanned_rate_test", r.miss_not_scanned_rate_test},
                {"miss_scanned_not_selected_rate_ref",
                 r.miss_scanned_not_selected_rate_ref},
                {"miss_scanned_not_selected_rate_test",
                 r.miss_scanned_not_selected_rate_test},
                {"better_unselected_per_exp_ref",
                 r.better_unselected_per_exp_ref},
                {"better_unselected_per_exp_test",
                 r.better_unselected_per_exp_test},
                {"topk_hit_ratio_ref", r.topk_hit_ratio_ref},
                {"topk_hit_ratio_test", r.topk_hit_ratio_test},
                {"oracle_best_rank_ref", r.oracle_best_rank_ref},
                {"oracle_best_rank_test", r.oracle_best_rank_test},
                {"oracle_best_rank_missed_ref", r.oracle_best_rank_missed_ref},
                {"oracle_best_rank_missed_test",
                 r.oracle_best_rank_missed_test},
                {"distance_per_exp_ref", r.distance_per_exp_ref},
                {"distance_per_exp_test", r.distance_per_exp_test},
                {"query_ns_ref", r.query_ns_ref},
                {"query_ns_test", r.query_ns_test},
            });
        }
        std::ofstream fout(out_per_query_json);
        fout << arr.dump(2);
    }

    if (!out_md.empty()) {
        std::vector<std::string> lines;
        lines.emplace_back("# RNSG Truncation Diagnosis");
        lines.emplace_back("");
        lines.emplace_back("- sampled_queries: `" +
                           std::to_string(reports.size()) + "`");
        lines.emplace_back("- has_groundtruth: `" +
                           std::string(has_gt ? "true" : "false") + "`");
        lines.emplace_back("- beam_size: `" + std::to_string(beam_size) + "`");
        lines.emplace_back("- trunc_ref: `" + std::to_string(trunc_ref) + "`");
        lines.emplace_back("- trunc_test: `" + std::to_string(trunc_test) +
                           "`");
        lines.emplace_back("");
        auto fmt = [](double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        };
        lines.emplace_back("## Core Metrics");
        lines.emplace_back(
            "- overlap_ref_test.mean: `" +
            fmt(summary["overlap_ref_test"]["mean"].get<double>()) + "`");
        lines.emplace_back("- miss_rate_ref.mean: `" +
                           fmt(summary["miss_rate_ref"]["mean"].get<double>()) +
                           "`");
        lines.emplace_back(
            "- miss_rate_test.mean: `" +
            fmt(summary["miss_rate_test"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- miss_not_scanned_rate_ref.mean: `" +
            fmt(summary["miss_not_scanned_rate_ref"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- miss_not_scanned_rate_test.mean: `" +
            fmt(summary["miss_not_scanned_rate_test"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- miss_scanned_not_selected_rate_ref.mean: `" +
            fmt(summary["miss_scanned_not_selected_rate_ref"]["mean"]
                    .get<double>()) +
            "`");
        lines.emplace_back(
            "- miss_scanned_not_selected_rate_test.mean: `" +
            fmt(summary["miss_scanned_not_selected_rate_test"]["mean"]
                    .get<double>()) +
            "`");
        lines.emplace_back("- better_unselected_per_exp_ref.mean: `" +
                           fmt(summary["better_unselected_per_exp_ref"]["mean"]
                                   .get<double>()) +
                           "`");
        lines.emplace_back("- better_unselected_per_exp_test.mean: `" +
                           fmt(summary["better_unselected_per_exp_test"]["mean"]
                                   .get<double>()) +
                           "`");
        lines.emplace_back(
            "- topk_hit_ratio_ref.mean: `" +
            fmt(summary["topk_hit_ratio_ref"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- topk_hit_ratio_test.mean: `" +
            fmt(summary["topk_hit_ratio_test"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_best_rank_ref.mean: `" +
            fmt(summary["oracle_best_rank_ref"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_best_rank_test.mean: `" +
            fmt(summary["oracle_best_rank_test"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- oracle_best_rank_missed_ref.mean: `" +
            fmt(summary["oracle_best_rank_missed_ref"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- oracle_best_rank_missed_test.mean: `" +
            fmt(summary["oracle_best_rank_missed_test"]["mean"].get<double>()) +
            "`");
        lines.emplace_back(
            "- distance_per_exp_ref.mean: `" +
            fmt(summary["distance_per_exp_ref"]["mean"].get<double>()) + "`");
        lines.emplace_back(
            "- distance_per_exp_test.mean: `" +
            fmt(summary["distance_per_exp_test"]["mean"].get<double>()) + "`");
        if (has_gt) {
            lines.emplace_back(
                "- recall_ref.mean: `" +
                fmt(summary["recall_ref"]["mean"].get<double>()) + "`");
            lines.emplace_back(
                "- recall_test.mean: `" +
                fmt(summary["recall_test"]["mean"].get<double>()) + "`");
            lines.emplace_back(
                "- recall_drop.mean: `" +
                fmt(summary["recall_drop"]["mean"].get<double>()) + "`");
            lines.emplace_back(
                "- corr(recall_drop, miss_rate_test): `" +
                fmt(summary["corr_recall_drop_vs_miss_test"].get<double>()) +
                "`");
            lines.emplace_back(
                "- corr(recall_drop, miss_not_scanned_rate_test): `" +
                fmt(summary["corr_recall_drop_vs_miss_not_scanned_test"]
                        .get<double>()) +
                "`");
            lines.emplace_back(
                "- corr(recall_drop, miss_scanned_not_selected_rate_test): `" +
                fmt(summary
                        ["corr_recall_drop_vs_miss_scanned_not_selected_test"]
                            .get<double>()) +
                "`");
            lines.emplace_back(
                "- corr(recall_drop, better_unselected_per_exp_test): `" +
                fmt(summary["corr_recall_drop_vs_better_test"].get<double>()) +
                "`");
        }
        std::ofstream fout(out_md);
        for (const auto& line : lines) {
            fout << line << "\n";
        }
    }

    spdlog::info("[diag] sampled_queries={}", reports.size());
    spdlog::info("[diag] overlap_ref_test.mean={:.6f}",
                 summary["overlap_ref_test"]["mean"].get<double>());
    spdlog::info("[diag] miss_rate_ref.mean={:.6f}",
                 summary["miss_rate_ref"]["mean"].get<double>());
    spdlog::info("[diag] miss_rate_test.mean={:.6f}",
                 summary["miss_rate_test"]["mean"].get<double>());
    spdlog::info(
        "[diag] better_unselected_per_exp_test.mean={:.6f}",
        summary["better_unselected_per_exp_test"]["mean"].get<double>());
    if (has_gt) {
        spdlog::info("[diag] recall_ref.mean={:.6f}",
                     summary["recall_ref"]["mean"].get<double>());
        spdlog::info("[diag] recall_test.mean={:.6f}",
                     summary["recall_test"]["mean"].get<double>());
        spdlog::info("[diag] recall_drop.mean={:.6f}",
                     summary["recall_drop"]["mean"].get<double>());
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"RNSG truncation diagnosis analyzer"};

    auto diag = app.add_subcommand("truncdiag",
                                   "Compare high-trunc reference vs low-trunc "
                                   "test and diagnose recall drop reasons");

    auto greedy = app.add_subcommand(
        "greedycheck",
        "Run baseline truncation vs full-neighbour greedy-prune (HNSW style) "
        "and compare recall/dist evals");

    auto rng = app.add_subcommand(
        "rngcheck",
        "Run baseline truncation vs strict-RNG-on-full-inrange-neighbours "
        "and compare recall/dist evals");

    auto twostage = app.add_subcommand(
        "twostagecheck",
        "Run baseline single-stage query vs route-then-refine search and "
        "compare recall/dist evals");

    auto domtag = app.add_subcommand(
        "domtagcheck",
        "Run baseline truncation vs build-time-dominance-tag synthesis and "
        "compare recall/dist evals");

    auto mixdom = app.add_subcommand(
        "mixdomcheck",
        "Run baseline truncation vs mixed baseline-prefix + dominance-tag "
        "survivors and compare recall/dist evals");

    auto lane = app.add_subcommand(
        "lanecheck",
        "Run baseline truncation vs local/switch/dom lane synthesis and "
        "compare recall/dist evals");

    auto collect = app.add_subcommand(
        "collecttest",
        "Run single-config real query collection and enforce recall threshold");

    std::string dataset_file, index_file, query_file, label_file, qrange_file,
        gt_file;
    unsigned qnumber = 10;
    unsigned beam_size = 120;
    unsigned trunc_ref = 50;
    unsigned trunc_test = 30;
    unsigned greedy_trunc = 30;
    unsigned greedy_degree = 20;
    unsigned collect_trunc = 20;
    unsigned rng_cap = 0;
    unsigned domtag_active_cap = 0;
    unsigned dominator_cap = 0;
    unsigned mix_local_offset = 0;
    unsigned mix_local_keep = 20;
    unsigned mix_dom_keep = 10;
    unsigned lane_local_keep = 4;
    unsigned lane_switch_skip = 4;
    unsigned lane_switch_keep = 8;
    unsigned lane_dom_keep = 38;
    double lane_switch_gap_lo_ratio = 0.04;
    double lane_switch_gap_hi_ratio = 0.25;
    unsigned beam_route = 48;
    unsigned trunc_route = 12;
    unsigned route_expansions = 12;
    unsigned beam_refine = 120;
    unsigned trunc_refine = 30;
    unsigned sample_queries = 200;
    unsigned seed = 20260324;
    double recall_threshold = 0.90;
    double bridge_gap_ratio = 0.125;
    unsigned nav_degree = 0;
    unsigned nav_scan_factor = 4;
    unsigned nav_stall_rounds = 8;
    unsigned nav_front_keep = 8;
    unsigned nav_tail_degree = 0;
    unsigned nav_early_stop_rounds = 0;
    std::string out_json, out_md, out_per_query_json;
    std::string greedy_out_json, greedy_out_md;
    std::string rng_out_json, rng_out_md;
    std::string domtag_out_json, domtag_out_md;
    std::string mixdom_out_json, mixdom_out_md;
    std::string lane_out_json, lane_out_md;
    std::string twostage_out_json, twostage_out_md;
    std::string collect_out_json, collect_out_md, collect_out_per_query_json,
        collect_out_per_expansion_json;

    diag->add_option("-d,--dataset_file", dataset_file)->required();
    diag->add_option("-i,--index_file", index_file)->required();
    diag->add_option("-q,--query_file", query_file)->required();
    diag->add_option("-l,--label_file", label_file)->required();
    diag->add_option("-Q,--qrange_file", qrange_file)->required();
    diag->add_option("-g,--groundtruth_file", gt_file,
                     "Optional groundtruth file for true recall delta");
    diag->add_option("-n,--qnumber", qnumber)->required();
    diag->add_option("-s,--beam_size", beam_size)->required();
    diag->add_option("--trunc_ref", trunc_ref)->required();
    diag->add_option("--trunc_test", trunc_test)->required();
    diag->add_option("--sample_queries", sample_queries);
    diag->add_option("--seed", seed);
    diag->add_option("--nav_degree", nav_degree);
    diag->add_option("--nav_scan_factor", nav_scan_factor);
    diag->add_option("--nav_stall_rounds", nav_stall_rounds);
    diag->add_option("--nav_front_keep", nav_front_keep);
    diag->add_option("--nav_tail_degree", nav_tail_degree);
    diag->add_option("--nav_early_stop_rounds", nav_early_stop_rounds);
    diag->add_option("--out_json", out_json);
    diag->add_option("--out_md", out_md);
    diag->add_option("--out_per_query_json", out_per_query_json);

    greedy->add_option("-d,--dataset_file", dataset_file)->required();
    greedy->add_option("-i,--index_file", index_file)->required();
    greedy->add_option("-q,--query_file", query_file)->required();
    greedy->add_option("-l,--label_file", label_file)->required();
    greedy->add_option("-Q,--qrange_file", qrange_file)->required();
    greedy
        ->add_option("-g,--groundtruth_file", gt_file,
                     "Groundtruth file for recall computation")
        ->required();
    greedy->add_option("-n,--qnumber", qnumber)->required();
    greedy->add_option("-s,--beam_size", beam_size)->required();
    greedy
        ->add_option("--trunc_size", greedy_trunc,
                     "Baseline truncation size for comparison")
        ->required();
    greedy->add_option("--greedy_degree", greedy_degree,
                       "Greedy-prune target degree (HNSW-style)");
    greedy->add_option("--sample_queries", sample_queries);
    greedy->add_option("--seed", seed);
    greedy->add_option("--out_json", greedy_out_json);
    greedy->add_option("--out_md", greedy_out_md);

    rng->add_option("-d,--dataset_file", dataset_file)->required();
    rng->add_option("-i,--index_file", index_file)->required();
    rng->add_option("-q,--query_file", query_file)->required();
    rng->add_option("-l,--label_file", label_file)->required();
    rng->add_option("-Q,--qrange_file", qrange_file)->required();
    rng->add_option("-g,--groundtruth_file", gt_file)->required();
    rng->add_option("-n,--qnumber", qnumber)->required();
    rng->add_option("-s,--beam_size", beam_size)->required();
    rng->add_option("--trunc_size", greedy_trunc,
                    "Baseline truncation size for comparison")
        ->required();
    rng->add_option("--rng_cap", rng_cap,
                    "Optional cap after strict RNG pruning (0 = keep all "
                    "strict-RNG survivors)");
    rng->add_option("--sample_queries", sample_queries);
    rng->add_option("--seed", seed);
    rng->add_option("--out_json", rng_out_json);
    rng->add_option("--out_md", rng_out_md);

    domtag->add_option("-d,--dataset_file", dataset_file)->required();
    domtag->add_option("-i,--index_file", index_file)->required();
    domtag->add_option("-q,--query_file", query_file)->required();
    domtag->add_option("-l,--label_file", label_file)->required();
    domtag->add_option("-Q,--qrange_file", qrange_file)->required();
    domtag->add_option("-g,--groundtruth_file", gt_file)->required();
    domtag->add_option("-n,--qnumber", qnumber)->required();
    domtag->add_option("-s,--beam_size", beam_size)->required();
    domtag->add_option("--trunc_size", greedy_trunc,
                       "Baseline truncation size for comparison")
        ->required();
    domtag->add_option("--active_cap", domtag_active_cap,
                       "Cap after dominance-tag synthesis (0 = keep all tag survivors)");
    domtag->add_option("--dominator_cap", dominator_cap,
                       "Max stored dominators per edge (0 = keep all)");
    domtag->add_option("--sample_queries", sample_queries);
    domtag->add_option("--seed", seed);
    domtag->add_option("--out_json", domtag_out_json);
    domtag->add_option("--out_md", domtag_out_md);

    mixdom->add_option("-d,--dataset_file", dataset_file)->required();
    mixdom->add_option("-i,--index_file", index_file)->required();
    mixdom->add_option("-q,--query_file", query_file)->required();
    mixdom->add_option("-l,--label_file", label_file)->required();
    mixdom->add_option("-Q,--qrange_file", qrange_file)->required();
    mixdom->add_option("-g,--groundtruth_file", gt_file)->required();
    mixdom->add_option("-n,--qnumber", qnumber)->required();
    mixdom->add_option("-s,--beam_size", beam_size)->required();
    mixdom->add_option("--trunc_size", greedy_trunc,
                       "Baseline truncation size for comparison")
        ->required();
    mixdom->add_option("--local_offset", mix_local_offset,
                       "Starting rank in the filtered baseline row used for "
                       "the local band");
    mixdom->add_option("--local_keep", mix_local_keep)->required();
    mixdom->add_option("--dom_keep", mix_dom_keep)->required();
    mixdom->add_option("--dominator_cap", dominator_cap,
                       "Max stored dominators per edge (0 = keep all)");
    mixdom->add_option("--sample_queries", sample_queries);
    mixdom->add_option("--seed", seed);
    mixdom->add_option("--out_json", mixdom_out_json);
    mixdom->add_option("--out_md", mixdom_out_md);

    lane->add_option("-d,--dataset_file", dataset_file)->required();
    lane->add_option("-i,--index_file", index_file)->required();
    lane->add_option("-q,--query_file", query_file)->required();
    lane->add_option("-l,--label_file", label_file)->required();
    lane->add_option("-Q,--qrange_file", qrange_file)->required();
    lane->add_option("-g,--groundtruth_file", gt_file)->required();
    lane->add_option("-n,--qnumber", qnumber)->required();
    lane->add_option("-s,--beam_size", beam_size)->required();
    lane->add_option("--trunc_size", greedy_trunc,
                     "Baseline truncation size for comparison")
        ->required();
    lane->add_option("--local_keep", lane_local_keep)->required();
    lane->add_option("--switch_skip", lane_switch_skip)->required();
    lane->add_option("--switch_keep", lane_switch_keep)->required();
    lane->add_option("--dom_keep", lane_dom_keep)->required();
    lane->add_option("--switch_gap_lo_ratio", lane_switch_gap_lo_ratio);
    lane->add_option("--switch_gap_hi_ratio", lane_switch_gap_hi_ratio);
    lane->add_option("--dominator_cap", dominator_cap,
                     "Max stored dominators per edge (0 = keep all)");
    lane->add_option("--sample_queries", sample_queries);
    lane->add_option("--seed", seed);
    lane->add_option("--out_json", lane_out_json);
    lane->add_option("--out_md", lane_out_md);

    twostage->add_option("-d,--dataset_file", dataset_file)->required();
    twostage->add_option("-i,--index_file", index_file)->required();
    twostage->add_option("-q,--query_file", query_file)->required();
    twostage->add_option("-l,--label_file", label_file)->required();
    twostage->add_option("-Q,--qrange_file", qrange_file)->required();
    twostage->add_option("-g,--groundtruth_file", gt_file)->required();
    twostage->add_option("-n,--qnumber", qnumber)->required();
    twostage->add_option("--beam_route", beam_route)->required();
    twostage->add_option("--trunc_route", trunc_route)->required();
    twostage->add_option("--route_expansions", route_expansions)->required();
    twostage->add_option("--beam_refine", beam_refine)->required();
    twostage->add_option("--trunc_refine", trunc_refine)->required();
    twostage->add_option("--sample_queries", sample_queries);
    twostage->add_option("--seed", seed);
    twostage->add_option("--out_json", twostage_out_json);
    twostage->add_option("--out_md", twostage_out_md);

    collect->add_option("-d,--dataset_file", dataset_file)->required();
    collect->add_option("-i,--index_file", index_file)->required();
    collect->add_option("-q,--query_file", query_file)->required();
    collect->add_option("-l,--label_file", label_file)->required();
    collect->add_option("-Q,--qrange_file", qrange_file)->required();
    collect
        ->add_option("-g,--groundtruth_file", gt_file,
                     "Groundtruth file for recall computation")
        ->required();
    collect->add_option("-n,--qnumber", qnumber)->required();
    collect->add_option("-s,--beam_size", beam_size)->required();
    collect->add_option("--trunc_size", collect_trunc)->required();
    collect->add_option("--sample_queries", sample_queries);
    collect->add_option("--seed", seed);
    collect->add_option("--nav_degree", nav_degree);
    collect->add_option("--nav_scan_factor", nav_scan_factor);
    collect->add_option("--nav_stall_rounds", nav_stall_rounds);
    collect->add_option("--nav_front_keep", nav_front_keep);
    collect->add_option("--nav_tail_degree", nav_tail_degree);
    collect->add_option("--nav_early_stop_rounds", nav_early_stop_rounds);
    collect->add_option("--bridge_gap_ratio", bridge_gap_ratio);
    collect->add_option("--recall_threshold", recall_threshold);
    collect->add_option("--out_json", collect_out_json);
    collect->add_option("--out_md", collect_out_md);
    collect->add_option("--out_per_query_json", collect_out_per_query_json);
    collect->add_option("--out_per_expansion_json",
                        collect_out_per_expansion_json);

    app.require_subcommand(1);
    CLI11_PARSE(app, argc, argv);

    try {
        if (*diag) {
            DiagConfig cfg;
            cfg.beam_size = beam_size;
            cfg.nav_degree = nav_degree;
            cfg.nav_scan_factor = nav_scan_factor;
            cfg.nav_stall_rounds = nav_stall_rounds;
            cfg.nav_front_keep = nav_front_keep;
            cfg.nav_tail_degree = nav_tail_degree;
            cfg.nav_early_stop_rounds = nav_early_stop_rounds;
            return run_trunc_diag(dataset_file, index_file, query_file,
                                  label_file, qrange_file, gt_file, qnumber,
                                  beam_size, trunc_ref, trunc_test,
                                  sample_queries, seed, cfg, out_json, out_md,
                                  out_per_query_json);
        }
        if (*greedy) {
            return run_greedy_check(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_size, greedy_trunc, greedy_degree,
                sample_queries, seed, greedy_out_json, greedy_out_md);
        }
        if (*rng) {
            return run_rng_check(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_size, greedy_trunc, sample_queries,
                seed, rng_cap, rng_out_json, rng_out_md);
        }
        if (*domtag) {
            return run_domtag_check(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_size, greedy_trunc, domtag_active_cap,
                dominator_cap, sample_queries, seed, domtag_out_json,
                domtag_out_md);
        }
        if (*mixdom) {
            return run_mixdom_check(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_size, greedy_trunc, mix_local_offset,
                mix_local_keep,
                mix_dom_keep, dominator_cap, sample_queries, seed,
                mixdom_out_json, mixdom_out_md);
        }
        if (*lane) {
            return run_lane_check(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_size, greedy_trunc, lane_local_keep,
                lane_switch_skip, lane_switch_keep, lane_dom_keep,
                lane_switch_gap_lo_ratio, lane_switch_gap_hi_ratio,
                dominator_cap, sample_queries, seed, lane_out_json,
                lane_out_md);
        }
        if (*twostage) {
            return run_two_stage_check(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_route, trunc_route, route_expansions,
                beam_refine, trunc_refine, sample_queries, seed,
                twostage_out_json, twostage_out_md);
        }
        if (*collect) {
            DiagConfig cfg;
            cfg.beam_size = beam_size;
            cfg.nav_degree = nav_degree;
            cfg.nav_scan_factor = nav_scan_factor;
            cfg.nav_stall_rounds = nav_stall_rounds;
            cfg.nav_front_keep = nav_front_keep;
            cfg.nav_tail_degree = nav_tail_degree;
            cfg.nav_early_stop_rounds = nav_early_stop_rounds;
            cfg.bridge_gap_ratio = bridge_gap_ratio;
            return run_collect_test(
                dataset_file, index_file, query_file, label_file, qrange_file,
                gt_file, qnumber, beam_size, collect_trunc, sample_queries,
                seed, cfg, recall_threshold, collect_out_json, collect_out_md,
                collect_out_per_query_json, collect_out_per_expansion_json);
        }
    } catch (const std::exception& e) {
        spdlog::error("rnsg_diag failed: {}", e.what());
        return 1;
    }

    return 0;
}
