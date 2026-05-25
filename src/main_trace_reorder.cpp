#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/InitFunc.hpp>
#include <Utils/Threading.hpp>
#include <Vector/VectorList.hpp>
#include <omp.h>
#include <unordered_set>

namespace {

using TDFANN::Graph::TDGraphIndexBase;
using TDFANN::Graph::to_node;

struct TraceStats {
    std::uint64_t queries = 0;
    std::uint64_t reached = 0;
    std::uint64_t expansions = 0;
    std::uint64_t distance_computations = 0;
    std::uint64_t evals = 0;
    std::uint64_t inserts = 0;
    std::uint64_t topk_inserts = 0;
    std::uint64_t improves = 0;
    std::uint64_t beam_retentions = 0;
    std::uint64_t topk_retentions = 0;
};

struct SearchScratch {
    std::vector<std::uint32_t> visit_stamp;
    std::uint32_t epoch = 1;

    struct Candidate {
        float dist = 0.0f;
        unsigned id = 0;
        std::uint64_t edge_id = 0;
        bool has_edge = false;
    };
    std::vector<Candidate> candidates;

    struct RawEdge {
        unsigned nid = 0;
        unsigned pos = 0;
    };
    struct SelectedEdge {
        unsigned nid = 0;
        unsigned pos = 0;
        float dist = 0.0f;
    };
    std::vector<RawEdge> raw;
    std::vector<SelectedEdge> selected;

    SearchScratch(unsigned n, unsigned beam, unsigned edge_reserve) {
        visit_stamp.resize(n, 0);
        candidates.reserve(beam + edge_reserve + 32);
        raw.reserve(edge_reserve + 32);
        selected.reserve(edge_reserve + 32);
    }

    void next_query() {
        ++epoch;
        if (epoch == 0) {
            std::fill(visit_stamp.begin(), visit_stamp.end(), 0);
            epoch = 1;
        }
        candidates.clear();
        raw.clear();
        selected.clear();
    }

    bool visited(unsigned id) const { return visit_stamp[id] == epoch; }

    bool mark(unsigned id) {
        if (visited(id)) {
            return false;
        }
        visit_stamp[id] = epoch;
        return true;
    }
};

std::vector<unsigned> make_label_stratified_sample(unsigned n,
                                                   unsigned sample_count) {
    if (n == 0 || sample_count == 0) {
        return {};
    }
    if (sample_count >= n) {
        std::vector<unsigned> out(n);
        std::iota(out.begin(), out.end(), 0u);
        return out;
    }
    std::vector<unsigned> out;
    out.reserve(sample_count);
    std::unordered_set<unsigned> seen;
    seen.reserve(sample_count * 2 + 8);
    for (unsigned i = 0; i < sample_count; ++i) {
        const unsigned id =
            (sample_count == 1)
                ? n / 2
                : static_cast<unsigned>(
                      (static_cast<unsigned long long>(i) * (n - 1)) /
                      (sample_count - 1));
        if (seen.insert(id).second) {
            out.push_back(id);
        }
    }
    return out;
}

void add_score(std::atomic<std::uint32_t>* scores, std::uint64_t edge_id,
               unsigned weight) {
    if (weight == 0) {
        return;
    }
    scores[edge_id].fetch_add(weight, std::memory_order_relaxed);
}

bool by_candidate_dist(const SearchScratch::Candidate& a,
                       const SearchScratch::Candidate& b) {
    return a.dist < b.dist;
}

template <typename T>
void run_fast_trace_query(
    const TDFANN::Vector::VectorList<T>& dataset,
    const TDGraphIndexBase& graph,
    const std::vector<std::uint64_t>& edge_offsets,
    std::atomic<std::uint32_t>* scores,
    std::atomic<std::uint32_t>* expanded_counts,
    const std::vector<unsigned>& start_nodes,
    unsigned target,
    unsigned beam_size,
    unsigned teacher_scan_size,
    unsigned teacher_eval_size,
    unsigned nav_degree,
    unsigned nav_scan_factor,
    unsigned nav_stall_rounds,
    unsigned nav_front_keep,
    bool stop_at_target,
    unsigned eval_weight,
    unsigned insert_weight,
    unsigned topk_insert_weight,
    unsigned improve_weight,
    unsigned beam_retention_weight,
    unsigned topk_retention_weight,
    unsigned trace_topk,
    bool teacher_sort_by_dist,
    SearchScratch& scratch,
    TraceStats& stats) {
    if (beam_size == 0 || start_nodes.empty()) {
        return;
    }

    scratch.next_query();
    const unsigned n = dataset.size();
    const unsigned offset = n;
    for (auto s : start_nodes) {
        if (s >= n || !scratch.mark(s)) {
            continue;
        }
        scratch.candidates.push_back({dataset.dist2(s, target), s, 0, false});
        stats.distance_computations++;
    }
    if (scratch.candidates.empty()) {
        return;
    }
    std::ranges::sort(scratch.candidates, by_candidate_dist);
    for (auto& cand : scratch.candidates) {
        cand.id += offset;
    }
    if (scratch.candidates.size() < beam_size) {
        scratch.candidates.resize(beam_size,
                                  {T(1e100), scratch.candidates[0].id - offset, 0, false});
    } else if (scratch.candidates.size() > beam_size) {
        scratch.candidates.resize(beam_size);
    }

    const unsigned effective_eval_cap = teacher_eval_size;
    const unsigned effective_nav_degree =
        (effective_eval_cap == 0)
            ? 0
            : (nav_degree == 0 ? effective_eval_cap
                               : std::min(nav_degree, effective_eval_cap));
    const unsigned effective_scan_factor = std::max(1u, nav_scan_factor);
    const unsigned effective_stall_rounds = std::max(1u, nav_stall_rounds);
    const unsigned effective_front_keep = std::max(1u, nav_front_keep);

    bool reached = false;
    unsigned stall_rounds = 0;
    for (int uid = 0; uid < static_cast<int>(beam_size); ++uid) {
        if (scratch.candidates[static_cast<size_t>(uid)].id < offset) {
            continue;
        }
        scratch.candidates[static_cast<size_t>(uid)].id -= offset;
        const unsigned current = scratch.candidates[static_cast<size_t>(uid)].id;
        if (current == target) {
            reached = true;
            if (stop_at_target) {
                break;
            }
        }
        expanded_counts[current].fetch_add(1, std::memory_order_relaxed);
        stats.expansions++;

        unsigned local_budget = effective_eval_cap;
        if (local_budget > 0 && effective_nav_degree > 0 &&
            effective_nav_degree < local_budget) {
            local_budget = effective_nav_degree;
            if (stall_rounds >= effective_stall_rounds) {
                local_budget = effective_eval_cap;
            }
        }

        unsigned scan_limit = teacher_scan_size;
        if (local_budget > 0 && scan_limit > 0 && local_budget < scan_limit) {
            const unsigned nav_scan_limit =
                std::min(scan_limit,
                         std::max(local_budget, local_budget * effective_scan_factor));
            scan_limit = std::max(scan_limit, nav_scan_limit);
        }

        scratch.raw.clear();
        scratch.selected.clear();
        const auto& row = graph.get_neighbours(current);
        for (unsigned pos = 0; pos < row.size(); ++pos) {
            const unsigned nid = row[pos].to;
            if (scratch.visited(nid)) {
                continue;
            }
            scratch.raw.push_back({nid, pos});
            if (scan_limit > 0 && scratch.raw.size() >= scan_limit) {
                break;
            }
        }
        if (scratch.raw.empty()) {
            stall_rounds++;
            continue;
        }

        if (local_budget == 0 || scratch.raw.size() <= local_budget) {
            for (const auto& e : scratch.raw) {
                scratch.selected.push_back({e.nid, e.pos, T(0)});
            }
        } else {
            const size_t raw_n = scratch.raw.size();
            const size_t need = std::min<size_t>(local_budget, raw_n);
            const size_t prefix_keep =
                std::min<size_t>(std::min<size_t>(effective_front_keep, need), raw_n);
            for (size_t i = 0; i < prefix_keep; ++i) {
                const auto& e = scratch.raw[i];
                scratch.selected.push_back({e.nid, e.pos, T(0)});
            }
            const size_t remain = need - scratch.selected.size();
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
                    const auto& e = scratch.raw[idx];
                    scratch.selected.push_back({e.nid, e.pos, T(0)});
                }
            }
        }

        bool improved = false;
        for (auto& e : scratch.selected) {
            const std::uint64_t edge_id = edge_offsets[current] + e.pos;
            add_score(scores, edge_id, eval_weight);
            stats.evals++;
            e.dist = dataset.dist2(e.nid, target);
            stats.distance_computations++;
            scratch.mark(e.nid);
        }
        if (teacher_sort_by_dist) {
            std::ranges::stable_sort(scratch.selected, {}, &SearchScratch::SelectedEdge::dist);
        }
        for (auto& e : scratch.selected) {
            const std::uint64_t edge_id = edge_offsets[current] + e.pos;
            const float old_best = scratch.candidates.front().dist;
            const size_t topk_idx =
                std::min<size_t>(std::max(1u, trace_topk), scratch.candidates.size()) - 1;
            const float old_topk = scratch.candidates[topk_idx].dist;
            if (e.dist < scratch.candidates.back().dist) {
                add_score(scores, edge_id, insert_weight);
                stats.inserts++;
                if (e.dist < old_topk) {
                    add_score(scores, edge_id, topk_insert_weight);
                    stats.topk_inserts++;
                }
                scratch.candidates.pop_back();
                auto it = std::partition_point(
                    scratch.candidates.begin(), scratch.candidates.end(),
                    [&](const auto& a) { return a.dist < e.dist; });
                const int pos = static_cast<int>(it - scratch.candidates.begin());
                uid = std::min(uid, pos - 1);
                scratch.candidates.insert(it, {e.dist, e.nid + offset, edge_id, true});
                if (e.dist < old_best) {
                    add_score(scores, edge_id, improve_weight);
                    stats.improves++;
                    improved = true;
                }
            }
            if (e.nid == target) {
                reached = true;
                if (stop_at_target) {
                    break;
                }
            }
        }
        if (reached && stop_at_target) {
            break;
        }
        if (improved) {
            stall_rounds = 0;
        } else {
            stall_rounds++;
        }
    }
    const size_t final_topk =
        std::min<size_t>(std::max(1u, trace_topk), scratch.candidates.size());
    for (size_t i = 0; i < scratch.candidates.size(); ++i) {
        const auto& cand = scratch.candidates[i];
        if (!cand.has_edge || cand.dist >= T(1e99)) {
            continue;
        }
        add_score(scores, cand.edge_id, beam_retention_weight);
        stats.beam_retentions++;
        if (i < final_topk) {
            add_score(scores, cand.edge_id, topk_retention_weight);
            stats.topk_retentions++;
        }
    }
    stats.queries++;
    if (reached) {
        stats.reached++;
    }
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"rnsg_trace_reorder: fast self-trace graph neighbor reorder"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string output_index;
    std::string report_json;
    unsigned sample_count = 100000;
    unsigned beam_size = 40;
    unsigned trunc_size = 16;
    unsigned teacher_scan_size = 0;
    unsigned teacher_eval_size = 0;
    unsigned nav_degree = 8;
    unsigned nav_scan_factor = 2;
    unsigned nav_stall_rounds = 8;
    unsigned nav_front_keep = 4;
    unsigned warmup = 8;
    unsigned degree_cap = 0;
    unsigned eval_weight = 0;
    unsigned insert_weight = 1;
    unsigned topk_insert_weight = 4;
    unsigned improve_weight = 8;
    unsigned beam_retention_weight = 1;
    unsigned topk_retention_weight = 4;
    unsigned trace_topk = 10;
    bool teacher_sort_by_dist = true;
    bool stop_at_target = true;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-o,--output_index", output_index, "Output graph index")->required();
    app.add_option("--report_json", report_json, "Optional JSON report");
    app.add_option("--sample_count", sample_count,
                   "Number of label-stratified base points used as pseudo queries");
    app.add_option("--beam_size", beam_size, "Fast teacher beam size");
    app.add_option("--trunc_size", trunc_size,
                   "Deprecated compatibility field; teacher caps are controlled by "
                   "--teacher_scan_size/--teacher_eval_size");
    app.add_option("--teacher_scan_size", teacher_scan_size,
                   "Max unvisited neighbors scanned per expanded node (0 = all)");
    app.add_option("--teacher_eval_size", teacher_eval_size,
                   "Max scanned neighbors distance-evaluated per expanded node (0 = all scanned)");
    app.add_option("--nav_degree", nav_degree,
                   "Fast teacher local navigation degree (0 = trunc_size)");
    app.add_option("--nav_scan_factor", nav_scan_factor, "Navigation scan factor");
    app.add_option("--nav_stall_rounds", nav_stall_rounds,
                   "Rounds before releasing nav_degree to trunc_size");
    app.add_option("--nav_front_keep", nav_front_keep,
                   "Prefix kept before sparse tail sampling");
    app.add_option("--warmup", warmup,
                   "Per-node prefix kept in original order during reorder");
    app.add_option("--degree_cap", degree_cap,
                   "Optional max output degree per node (0 = keep full degree)");
    app.add_option("--eval_weight", eval_weight, "Score added for evaluated edge");
    app.add_option("--insert_weight", insert_weight, "Score added for inserted edge");
    app.add_option("--topk_insert_weight", topk_insert_weight,
                   "Score added when an edge inserts into the teacher top-k frontier");
    app.add_option("--improve_weight", improve_weight,
                   "Score added for best-improving edge");
    app.add_option("--beam_retention_weight", beam_retention_weight,
                   "Final score added for the incoming edge of a beam-retained node");
    app.add_option("--topk_retention_weight", topk_retention_weight,
                   "Final score added for the incoming edge of a top-k-retained node");
    app.add_option("--trace_topk", trace_topk,
                   "Teacher top-k frontier used for top-k insert/retention scoring");
    app.add_flag("--teacher_sort_by_dist,!--no_teacher_sort_by_dist",
                 teacher_sort_by_dist,
                 "Sort teacher-evaluated neighbors by distance before beam update");
    app.add_flag("--stop_at_target,!--no_stop_at_target", stop_at_target,
                 "Stop each pseudo search once the target base point is reached");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_trace_reorder");

    if (input_index == output_index) {
        throw std::runtime_error("input_index and output_index must differ");
    }

    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);
    auto sorted_label = TDFANN::Utils::sorted_vec(label);

    TDGraphIndexBase in_graph(input_index);
    const unsigned n = static_cast<unsigned>(in_graph.size());
    if (dataset.size() != n) {
        throw std::runtime_error("dataset size and graph size differ");
    }

    std::uint64_t total_edges = 0;
    std::vector<std::uint64_t> edge_offsets(static_cast<size_t>(n) + 1);
    for (unsigned i = 0; i < n; ++i) {
        edge_offsets[i] = total_edges;
        total_edges += in_graph.get_neighbours(i).size();
    }
    edge_offsets[n] = total_edges;
    spdlog::info("Graph loaded: nodes={}, edges={}, avg_degree={:.2f}", n,
                 total_edges, static_cast<double>(total_edges) / std::max(1u, n));

    auto scores = std::make_unique<std::atomic<std::uint32_t>[]>(total_edges);
    for (std::uint64_t i = 0; i < total_edges; ++i) {
        scores[i].store(0, std::memory_order_relaxed);
    }
    auto expanded_counts = std::make_unique<std::atomic<std::uint32_t>[]>(n);
    for (unsigned i = 0; i < n; ++i) {
        expanded_counts[i].store(0, std::memory_order_relaxed);
    }

    std::vector<unsigned> start_nodes;
    if (!sorted_label.empty()) {
        const unsigned header_id =
            in_graph.get_header_index_for_right_bound(sorted_label.back());
        auto header = in_graph.get_header(header_id);
        start_nodes.assign(header.begin(), header.end());
    }
    if (start_nodes.empty()) {
        start_nodes.push_back(0);
    }
    std::ranges::sort(start_nodes);
    start_nodes.erase(std::ranges::unique(start_nodes).begin(), start_nodes.end());
    spdlog::info("Using {} header start nodes", start_nodes.size());

    auto samples = make_label_stratified_sample(n, sample_count);
    spdlog::info("Trace teacher: samples={}, beam={}, scan={}, eval={}, nav_degree={}, sort_by_dist={}",
                 samples.size(), beam_size,
                                 teacher_scan_size == 0 ? std::string("all") : std::to_string(teacher_scan_size),
                                 teacher_eval_size == 0 ? std::string("all") : std::to_string(teacher_eval_size),
                                 nav_degree, teacher_sort_by_dist);

    const int thread_count = TDFANN::Utils::configured_thread_count();
    omp_set_num_threads(thread_count);
    std::vector<TraceStats> per_thread(static_cast<size_t>(thread_count));
    std::atomic<unsigned> processed = 0;
    const unsigned progress_step =
        std::max(1u, static_cast<unsigned>(samples.size() / 20 + 1));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const unsigned edge_reserve =
            std::max({teacher_scan_size, teacher_eval_size, trunc_size, 256u});
        SearchScratch scratch(n, beam_size, edge_reserve);
#pragma omp for schedule(dynamic, 64)
        for (int64_t si = 0; si < static_cast<int64_t>(samples.size()); ++si) {
            const unsigned target = samples[static_cast<size_t>(si)];
            run_fast_trace_query(dataset, in_graph, edge_offsets, scores.get(),
                                 expanded_counts.get(), start_nodes, target,
                                 beam_size, teacher_scan_size, teacher_eval_size,
                                 nav_degree,
                                 nav_scan_factor, nav_stall_rounds,
                                 nav_front_keep, stop_at_target, eval_weight,
                                 insert_weight, topk_insert_weight, improve_weight,
                                 beam_retention_weight, topk_retention_weight,
                                 trace_topk, teacher_sort_by_dist, scratch,
                                 per_thread[static_cast<size_t>(tid)]);
            const unsigned done = processed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (tid == 0 && (done % progress_step == 0 || done == samples.size())) {
                spdlog::info("Trace progress: {}/{}", done, samples.size());
            }
        }
    }

    TraceStats stats;
    for (const auto& s : per_thread) {
        stats.queries += s.queries;
        stats.reached += s.reached;
        stats.expansions += s.expansions;
        stats.distance_computations += s.distance_computations;
        stats.evals += s.evals;
        stats.inserts += s.inserts;
        stats.topk_inserts += s.topk_inserts;
        stats.improves += s.improves;
        stats.beam_retentions += s.beam_retentions;
        stats.topk_retentions += s.topk_retentions;
    }
    spdlog::info("Trace done: queries={}, reached={}, expansions={}, dco={}, evals={}, inserts={}, topk_inserts={}, improves={}, beam_retained={}, topk_retained={}",
                 stats.queries, stats.reached, stats.expansions,
                 stats.distance_computations, stats.evals, stats.inserts,
                 stats.topk_inserts, stats.improves, stats.beam_retentions,
                 stats.topk_retentions);

    TDGraphIndexBase out_graph(n);
    out_graph.copy_headers_from(in_graph);

    std::atomic<std::uint64_t> signal_edges = 0;
    std::atomic<std::uint64_t> signal_nodes = 0;
    std::atomic<std::uint64_t> expanded_nodes = 0;
    std::atomic<std::uint64_t> total_out_edges = 0;

    using Node = TDFANN::Graph::GraphIndex<std::monostate>::Node;
    std::vector<std::vector<Node>> reordered(n);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
        const unsigned u = static_cast<unsigned>(ii);
        const auto& row = in_graph.get_neighbours(u);
        const size_t deg = row.size();
        if (expanded_counts[u].load(std::memory_order_relaxed) > 0) {
            expanded_nodes.fetch_add(1, std::memory_order_relaxed);
        }
        bool node_has_signal = false;
        for (size_t p = 0; p < deg; ++p) {
            if (scores[edge_offsets[u] + p].load(std::memory_order_relaxed) > 0) {
                signal_edges.fetch_add(1, std::memory_order_relaxed);
                node_has_signal = true;
            }
        }
        if (node_has_signal) {
            signal_nodes.fetch_add(1, std::memory_order_relaxed);
        }

        std::vector<unsigned> order(deg);
        std::iota(order.begin(), order.end(), 0u);
        const size_t keep_prefix = std::min<size_t>(warmup, deg);
        if (deg > keep_prefix) {
            std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(keep_prefix),
                             order.end(), [&](unsigned a, unsigned b) {
                const auto sa = scores[edge_offsets[u] + a].load(std::memory_order_relaxed);
                const auto sb = scores[edge_offsets[u] + b].load(std::memory_order_relaxed);
                if ((sa > 0) != (sb > 0)) {
                    return sa > 0;
                }
                if (sa != sb) {
                    return sa > sb;
                }
                return a < b;
            });
        }

        const size_t out_deg =
            degree_cap == 0 ? deg : std::min<size_t>(deg, degree_cap);
        auto& out = reordered[u];
        out.reserve(out_deg);
        for (size_t i = 0; i < out_deg; ++i) {
            out.push_back(to_node(row[order[i]].to));
        }
        total_out_edges.fetch_add(out.size(), std::memory_order_relaxed);
    }

    for (unsigned i = 0; i < n; ++i) {
        out_graph.add_neighbours(i, reordered[i]);
    }

    std::ofstream fout(output_index);
    if (!fout.good() || !out_graph.save(fout)) {
        throw std::runtime_error("Failed to save reordered graph to " + output_index);
    }
    spdlog::info("Saved trace-reordered graph to {}", output_index);

    if (!report_json.empty()) {
        nlohmann::json js = {
            {"dataset_file", dataset_file},
            {"label_file", label_file},
            {"input_index", input_index},
            {"output_index", output_index},
            {"node_count", n},
            {"input_edges", total_edges},
            {"output_edges", total_out_edges.load()},
            {"sample_count_requested", sample_count},
            {"sample_count_actual", samples.size()},
            {"beam_size", beam_size},
            {"trunc_size", trunc_size},
            {"teacher_scan_size", teacher_scan_size},
            {"teacher_eval_size", teacher_eval_size},
            {"nav_degree", nav_degree},
            {"nav_scan_factor", nav_scan_factor},
            {"nav_stall_rounds", nav_stall_rounds},
            {"nav_front_keep", nav_front_keep},
            {"warmup", warmup},
            {"degree_cap", degree_cap},
            {"stop_at_target", stop_at_target},
            {"eval_weight", eval_weight},
            {"insert_weight", insert_weight},
            {"topk_insert_weight", topk_insert_weight},
            {"improve_weight", improve_weight},
            {"beam_retention_weight", beam_retention_weight},
            {"topk_retention_weight", topk_retention_weight},
            {"trace_topk", trace_topk},
            {"teacher_sort_by_dist", teacher_sort_by_dist},
            {"queries", stats.queries},
            {"targets_reached", stats.reached},
            {"trace_expansions", stats.expansions},
            {"trace_distance_computations", stats.distance_computations},
            {"trace_evals", stats.evals},
            {"trace_inserts", stats.inserts},
            {"trace_topk_inserts", stats.topk_inserts},
            {"trace_improves", stats.improves},
            {"trace_beam_retentions", stats.beam_retentions},
            {"trace_topk_retentions", stats.topk_retentions},
            {"expanded_node_coverage",
             static_cast<double>(expanded_nodes.load()) / std::max(1u, n)},
            {"signal_node_coverage",
             static_cast<double>(signal_nodes.load()) / std::max(1u, n)},
            {"signal_edge_coverage",
             static_cast<double>(signal_edges.load()) /
                 static_cast<double>(std::max<std::uint64_t>(1, total_edges))},
            {"header_start_nodes", start_nodes.size()},
        };
        std::ofstream jout(report_json);
        jout << js.dump(2) << '\n';
    }
    return 0;
}
