#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <RNSG/Searcher.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <unordered_set>

using namespace TDFANN;

namespace {

using GraphIndex = Graph::GraphIndex<std::monostate>;
using Node = GraphIndex::Node;

struct Label2D {
    std::uint64_t x = 0;
    std::uint64_t y = 0;
};

struct Range2D {
    std::uint64_t lx = 0;
    std::uint64_t rx = 0;
    std::uint64_t ly = 0;
    std::uint64_t ry = 0;
};

struct QueryConfig {
    std::string dataset_file;
    std::string query_file;
    std::string label_file;
    std::string qrange_file;
    std::string graph_file;
    std::string entry_index_file;
    std::string groundtruth_file;
    std::string result_file;
    std::string report_json;
    unsigned topk = 10;
    unsigned beam_size = 64;
    unsigned trunc_size = 32;
    unsigned entry_candidates = 8;
    unsigned entry_scan_limit = 4096;
    unsigned nav_degree = 16;
    unsigned nav_scan_factor = 4;
    unsigned nav_stall_rounds = 8;
    unsigned nav_front_keep = 8;
    unsigned nav_tail_degree = 0;
    unsigned nav_early_stop_rounds = 0;
    unsigned pick_scan_factor = 1;
    unsigned pick_front_keep = 0;
    std::string edge_pick_policy = "prefix";
    unsigned edge_pick_recip_depth = 32;
    double edge_pick_core_ratio = 0.6;
    unsigned fallback_stall_rounds = 0;
    std::string fallback_pick_policy = "prefix";
    double fallback_core_ratio = 0.40;
    unsigned fallback_pick_front_keep = 0;
    unsigned fallback_pick_scan_factor = 1;
    bool fallback_release_nav = false;
    unsigned rescue_slot_count = 0;
    unsigned warmup_min = 16;
};

double now_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

std::string default_entry_index_path(const std::string& graph_file) {
    return graph_file + ".entry2d";
}

bool in_range(const Label2D& label, const Range2D& range) {
    return label.x >= range.lx && label.x <= range.rx &&
           label.y >= range.ly && label.y <= range.ry;
}

std::vector<Label2D> load_labels_2d(const std::string& filename,
                                    size_t expected_n = 0) {
    std::ifstream fin(filename);
    if (!fin.good()) {
        throw std::runtime_error("Failed to open 2D label file: " + filename);
    }
    nlohmann::json j;
    fin >> j;
    if (!j.is_array()) {
        throw std::runtime_error("2D label file must be a JSON array");
    }

    std::vector<Label2D> labels;
    if (!j.empty() && j.front().is_array()) {
        labels.reserve(j.size());
        for (const auto& item : j) {
            if (!item.is_array() || item.size() != 2) {
                throw std::runtime_error(
                    "2D labels must be [[x,y], ...] or flat [x,y,...]");
            }
            labels.push_back({item[0].get<std::uint64_t>(),
                              item[1].get<std::uint64_t>()});
        }
    } else {
        if (j.size() % 2 != 0) {
            throw std::runtime_error(
                "Flat 2D label file must contain an even number of values");
        }
        labels.reserve(j.size() / 2);
        for (size_t i = 0; i < j.size(); i += 2) {
            labels.push_back({j[i].get<std::uint64_t>(),
                              j[i + 1].get<std::uint64_t>()});
        }
    }
    if (expected_n != 0 && labels.size() != expected_n) {
        throw std::runtime_error("2D label count does not match dataset size");
    }
    return labels;
}

std::vector<Range2D> load_ranges_2d(const std::string& filename,
                                    size_t expected_q = 0) {
    std::ifstream fin(filename);
    if (!fin.good()) {
        throw std::runtime_error("Failed to open 2D qrange file: " + filename);
    }
    nlohmann::json j;
    fin >> j;
    if (!j.is_array()) {
        throw std::runtime_error("2D qrange file must be a JSON array");
    }

    std::vector<Range2D> ranges;
    if (!j.empty() && j.front().is_array()) {
        ranges.reserve(j.size());
        for (const auto& item : j) {
            if (!item.is_array() || item.size() != 4) {
                throw std::runtime_error(
                    "2D qrange nested format must be [[lx,rx,ly,ry], ...]");
            }
            Range2D r{item[0].get<std::uint64_t>(),
                      item[1].get<std::uint64_t>(),
                      item[2].get<std::uint64_t>(),
                      item[3].get<std::uint64_t>()};
            if (r.lx > r.rx) std::swap(r.lx, r.rx);
            if (r.ly > r.ry) std::swap(r.ly, r.ry);
            ranges.push_back(r);
        }
    } else {
        if (j.size() % 4 != 0) {
            throw std::runtime_error(
                "2D qrange flat format must contain 4 values per query");
        }
        ranges.reserve(j.size() / 4);
        for (size_t i = 0; i < j.size(); i += 4) {
            Range2D r{j[i].get<std::uint64_t>(),
                      j[i + 1].get<std::uint64_t>(),
                      j[i + 2].get<std::uint64_t>(),
                      j[i + 3].get<std::uint64_t>()};
            if (r.lx > r.rx) std::swap(r.lx, r.rx);
            if (r.ly > r.ry) std::swap(r.ly, r.ry);
            ranges.push_back(r);
        }
    }
    if (expected_q != 0 && ranges.size() < expected_q) {
        throw std::runtime_error("2D qrange count is smaller than query size");
    }
    return ranges;
}

struct EntryGridIndex {
    static constexpr std::uint32_t kMagic = 0x32454752u;

    std::uint32_t grid_size = 0;
    std::uint64_t min_x = 0;
    std::uint64_t max_x = 0;
    std::uint64_t min_y = 0;
    std::uint64_t max_y = 0;
    std::vector<std::vector<unsigned>> cells;

    unsigned coord(std::uint64_t value, std::uint64_t lo,
                   std::uint64_t hi) const {
        if (grid_size <= 1 || hi <= lo) {
            return 0;
        }
        if (value <= lo) {
            return 0;
        }
        if (value >= hi) {
            return grid_size - 1;
        }
        const long double numerator =
            static_cast<long double>(value - lo) * grid_size;
        const long double denominator =
            static_cast<long double>(hi - lo + 1);
        auto c = static_cast<unsigned>(numerator / denominator);
        return std::min<unsigned>(c, grid_size - 1);
    }

    size_t cell_id(unsigned gx, unsigned gy) const {
        return static_cast<size_t>(gx) * grid_size + gy;
    }

    std::vector<unsigned> candidates(const std::vector<Label2D>& labels,
                                     const Range2D& range,
                                     unsigned limit) const {
        std::vector<unsigned> out;
        if (grid_size == 0 || cells.empty()) {
            return out;
        }
        const unsigned gx0 = coord(range.lx, min_x, max_x);
        const unsigned gx1 = coord(range.rx, min_x, max_x);
        const unsigned gy0 = coord(range.ly, min_y, max_y);
        const unsigned gy1 = coord(range.ry, min_y, max_y);
        const size_t cell_count =
            static_cast<size_t>(gx1 - gx0 + 1) * (gy1 - gy0 + 1);
        const unsigned per_cell =
            limit == 0
                ? std::numeric_limits<unsigned>::max()
                : std::max<unsigned>(
                      1, static_cast<unsigned>(std::ceil(
                             static_cast<double>(limit) /
                             std::max<size_t>(cell_count, 1))));
        out.reserve(limit == 0 ? 1024 : limit);
        for (unsigned gx = gx0; gx <= gx1; ++gx) {
            for (unsigned gy = gy0; gy <= gy1; ++gy) {
                const auto& cell = cells[cell_id(gx, gy)];
                if (cell.empty()) {
                    continue;
                }
                unsigned accepted = 0;
                const size_t step =
                    per_cell == std::numeric_limits<unsigned>::max()
                        ? 1
                        : std::max<size_t>(1, cell.size() / per_cell);
                for (size_t p = 0; p < cell.size(); p += step) {
                    const unsigned id = cell[p];
                    if (!in_range(labels[id], range)) {
                        continue;
                    }
                    out.push_back(id);
                    ++accepted;
                    if (limit != 0 && out.size() >= limit) {
                        return out;
                    }
                    if (accepted >= per_cell) {
                        break;
                    }
                }
            }
        }
        return out;
    }

    bool load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin.good()) {
            return false;
        }
        std::uint32_t magic = 0;
        if (!IO::load(fin, magic) || magic != kMagic) {
            return false;
        }
        return IO::load(fin, grid_size) && IO::load(fin, min_x) &&
               IO::load(fin, max_x) && IO::load(fin, min_y) &&
               IO::load(fin, max_y) && IO::load(fin, cells);
    }
};

class FilteredGraph2D {
   public:
    FilteredGraph2D(const GraphIndex& base_graph,
                    const std::vector<Label2D>& labels,
                    const Range2D& range)
        : base_(base_graph), labels_(labels), range_(range) {}

    auto get_neighbours(unsigned node) const {
        return base_.get_neighbours(node) |
               std::views::filter([this](const Node& x) {
                   return x.to < labels_.size() &&
                          in_range(labels_[x.to], range_);
               });
    }

    auto get_neighbours_id(unsigned node) const {
        return get_neighbours(node) | std::views::transform(GET(to));
    }

   private:
    const GraphIndex& base_;
    const std::vector<Label2D>& labels_;
    Range2D range_;
};

unsigned parse_policy_code(const std::string& policy) {
    if (policy == "side") {
        return 1;
    }
    if (policy == "reciprocal") {
        return 2;
    }
    if (policy == "corebridge") {
        return 3;
    }
    return 0;
}

unsigned max_graph_degree(const GraphIndex& graph) {
    unsigned out = 0;
    for (unsigned i = 0; i < graph.size(); ++i) {
        out = std::max<unsigned>(
            out, static_cast<unsigned>(graph.get_neighbours(i).size()));
    }
    return out;
}

template <typename QueryVec>
std::vector<unsigned> select_start_nodes(
    const Vector::VectorList<float>& dataset,
    const QueryVec& query,
    const std::vector<Label2D>& labels,
    const Range2D& range,
    const EntryGridIndex* entry_index,
    unsigned entry_candidates,
    unsigned entry_scan_limit,
    std::uint64_t& entry_select_dco) {
    std::vector<unsigned> ids;
    if (entry_index != nullptr) {
        ids = entry_index->candidates(labels, range, entry_scan_limit);
    } else {
        const unsigned limit = entry_scan_limit == 0
                                   ? std::numeric_limits<unsigned>::max()
                                   : entry_scan_limit;
        for (unsigned j = 0; j < labels.size(); ++j) {
            if (!in_range(labels[j], range)) {
                continue;
            }
            ids.push_back(j);
            if (ids.size() >= limit) {
                break;
            }
        }
    }
    if (ids.empty()) {
        return ids;
    }

    if (entry_candidates != 0 && ids.size() > entry_candidates) {
        std::vector<std::pair<float, unsigned>> ranked;
        ranked.reserve(ids.size());
        for (auto id : ids) {
            ranked.push_back({dataset.dist2(id, query), id});
        }
        entry_select_dco += ranked.size();
        std::partial_sort(ranked.begin(), ranked.begin() + entry_candidates,
                          ranked.end());
        ranked.resize(entry_candidates);
        ids.clear();
        ids.reserve(ranked.size());
        for (const auto& [dist, id] : ranked) {
            (void)dist;
            ids.push_back(id);
        }
    }
    return ids;
}

double recall_at_k(const std::vector<unsigned>& result,
                   const std::vector<unsigned>& gt,
                   unsigned topk) {
    if (gt.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    unsigned hits = 0;
    for (unsigned q = 0; q < gt.size() / topk; ++q) {
        std::unordered_set<unsigned> truth;
        truth.reserve(topk * 2);
        for (unsigned j = 0; j < topk; ++j) {
            const unsigned id = gt[q * topk + j];
            if (id != std::numeric_limits<unsigned>::max()) {
                truth.insert(id);
            }
        }
        for (unsigned j = 0; j < topk; ++j) {
            const unsigned id = result[q * topk + j];
            if (truth.contains(id)) {
                ++hits;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(gt.size());
}

int run_query(const QueryConfig& cfg) {
    Vector::VectorList<float> dataset(cfg.dataset_file);
    Vector::VectorList<float> queries(cfg.query_file);
    const auto labels = load_labels_2d(cfg.label_file, dataset.size());
    const auto ranges = load_ranges_2d(cfg.qrange_file, queries.size());
    GraphIndex graph(cfg.graph_file);
    if (graph.size() != dataset.size()) {
        throw std::runtime_error("graph node count does not match dataset");
    }
    const unsigned effective_trunc_size =
        cfg.trunc_size == 0 ? std::max(1u, max_graph_degree(graph))
                            : cfg.trunc_size;
    if (cfg.trunc_size == 0) {
        spdlog::info("Mapped trunc_size=0 to effective_trunc_size={}",
                     effective_trunc_size);
    }

    EntryGridIndex entry_index;
    const std::string entry_path =
        cfg.entry_index_file.empty() ? default_entry_index_path(cfg.graph_file)
                                     : cfg.entry_index_file;
    const bool has_entry_index =
        !entry_path.empty() && std::filesystem::exists(entry_path) &&
        entry_index.load(entry_path);
    if (has_entry_index) {
        spdlog::info("Loaded 2D entry index: {}", entry_path);
    } else {
        spdlog::warn("2D entry index not found/invalid; falling back to scan");
    }

    std::vector<unsigned> gt;
    if (!cfg.groundtruth_file.empty()) {
        gt = IO::load_json_to_vec<unsigned>(cfg.groundtruth_file);
        if (gt.size() < static_cast<size_t>(queries.size()) * cfg.topk) {
            throw std::runtime_error("groundtruth length is too small");
        }
    }

    const unsigned edge_policy_code = parse_policy_code(cfg.edge_pick_policy);
    const unsigned fb_policy_code = parse_policy_code(cfg.fallback_pick_policy);
    RNSG::BeamScratch<float> scratch(dataset.size(), cfg.beam_size,
                                     effective_trunc_size);

    std::vector<unsigned> flat;
    flat.reserve(static_cast<size_t>(queries.size()) * cfg.topk);
    std::uint64_t total_entry_select_dco = 0;
    std::uint64_t total_search_dco = 0;
    std::uint64_t total_expanded = 0;
    std::uint64_t total_raw_scanned = 0;
    std::uint64_t total_evaluated = 0;
    std::uint64_t total_visited = 0;
    std::uint64_t total_empty_seed = 0;
    std::uint64_t total_start_nodes = 0;

    const double t0 = now_seconds();
    for (unsigned qi = 0; qi < queries.size(); ++qi) {
        std::uint64_t entry_select_dco = 0;
        auto start_nodes = select_start_nodes(
            dataset, queries[qi], labels, ranges[qi],
            has_entry_index ? &entry_index : nullptr, cfg.entry_candidates,
            cfg.entry_scan_limit, entry_select_dco);
        total_entry_select_dco += entry_select_dco;
        total_start_nodes += start_nodes.size();
        if (start_nodes.empty()) {
            ++total_empty_seed;
            flat.insert(flat.end(), cfg.topk,
                        std::numeric_limits<unsigned>::max());
            continue;
        }

        FilteredGraph2D view(graph, labels, ranges[qi]);
        RNSG::Searcher<float, FilteredGraph2D> searcher(dataset, view);
        RNSG::BeamSearchStats stats{};
        auto result = searcher.beam_search(
            queries[qi], cfg.topk, start_nodes, cfg.beam_size,
            effective_trunc_size,
            scratch, cfg.nav_degree, cfg.nav_scan_factor,
            cfg.nav_stall_rounds, cfg.nav_front_keep, cfg.nav_tail_degree,
            cfg.nav_early_stop_rounds, cfg.pick_scan_factor,
            cfg.pick_front_keep, edge_policy_code, cfg.edge_pick_recip_depth,
            cfg.edge_pick_core_ratio, cfg.fallback_stall_rounds,
            fb_policy_code, cfg.fallback_core_ratio,
            cfg.fallback_pick_front_keep, cfg.fallback_pick_scan_factor,
            cfg.fallback_release_nav, nullptr, &stats,
            cfg.rescue_slot_count, 0, cfg.warmup_min);

        total_search_dco += stats.distance_computations;
        total_expanded += stats.expanded_nodes;
        total_raw_scanned += stats.raw_neighbors_scanned;
        total_evaluated += stats.in_range_neighbors_evaluated;
        total_visited += stats.visited_nodes_count;

        for (unsigned j = 0; j < cfg.topk; ++j) {
            if (j < result.size()) {
                flat.push_back(result[j].second);
            } else {
                flat.push_back(std::numeric_limits<unsigned>::max());
            }
        }
    }
    const double query_seconds = now_seconds() - t0;
    const double qps = queries.size() / std::max(query_seconds, 1e-12);
    const double recall = gt.empty() ? std::numeric_limits<double>::quiet_NaN()
                                     : recall_at_k(flat, gt, cfg.topk);

    if (!cfg.result_file.empty()) {
        nlohmann::json out = flat;
        std::ofstream fout(cfg.result_file);
        fout << out.dump() << "\n";
    }

    const double qn = static_cast<double>(queries.size());
    nlohmann::json report;
    report["target"] = "rnsg-2d-fast";
    report["dataset_file"] = cfg.dataset_file;
    report["query_file"] = cfg.query_file;
    report["label_file"] = cfg.label_file;
    report["qrange_file"] = cfg.qrange_file;
    report["graph_file"] = cfg.graph_file;
    report["entry_index_file"] = has_entry_index ? entry_path : "";
    report["entry_index_used"] = has_entry_index;
    report["groundtruth_file"] = cfg.groundtruth_file;
    report["topk"] = cfg.topk;
    report["beam_size"] = cfg.beam_size;
    report["trunc_size"] = cfg.trunc_size;
    report["effective_trunc_size"] = effective_trunc_size;
    report["entry_candidates"] = cfg.entry_candidates;
    report["entry_scan_limit"] = cfg.entry_scan_limit;
    report["nav_degree"] = cfg.nav_degree;
    report["nav_scan_factor"] = cfg.nav_scan_factor;
    report["nav_stall_rounds"] = cfg.nav_stall_rounds;
    report["nav_front_keep"] = cfg.nav_front_keep;
    report["nav_tail_degree"] = cfg.nav_tail_degree;
    report["nav_early_stop_rounds"] = cfg.nav_early_stop_rounds;
    report["pick_scan_factor"] = cfg.pick_scan_factor;
    report["pick_front_keep"] = cfg.pick_front_keep;
    report["edge_pick_policy"] = cfg.edge_pick_policy;
    report["fallback_stall_rounds"] = cfg.fallback_stall_rounds;
    report["fallback_pick_policy"] = cfg.fallback_pick_policy;
    report["fallback_release_nav"] = cfg.fallback_release_nav;
    report["rescue_slot_count"] = cfg.rescue_slot_count;
    report["warmup_min"] = cfg.warmup_min;
    report["query_count"] = queries.size();
    report["query_seconds"] = query_seconds;
    report["qps"] = qps;
    report["recall"] = recall;
    report["avg_dco"] =
        static_cast<double>(total_entry_select_dco + total_search_dco) / qn;
    report["avg_entry_select_dco"] =
        static_cast<double>(total_entry_select_dco) / qn;
    report["avg_search_dco"] = static_cast<double>(total_search_dco) / qn;
    report["avg_expanded"] = static_cast<double>(total_expanded) / qn;
    report["avg_raw_neighbors_scanned"] =
        static_cast<double>(total_raw_scanned) / qn;
    report["avg_in_range_neighbors_evaluated"] =
        static_cast<double>(total_evaluated) / qn;
    report["avg_visited_nodes"] = static_cast<double>(total_visited) / qn;
    report["avg_start_nodes"] = static_cast<double>(total_start_nodes) / qn;
    report["empty_seed_queries"] = total_empty_seed;

    if (!cfg.report_json.empty()) {
        std::ofstream jout(cfg.report_json);
        jout << report.dump(2) << "\n";
    }
    spdlog::info(
        "2D fast query done: recall={}, qps={:.2f}, avg_dco={:.2f}, "
        "search_dco={:.2f}, entry_dco={:.2f}, expanded={:.2f}",
        std::isnan(recall) ? -1.0 : recall, qps,
        report["avg_dco"].get<double>(),
        report["avg_search_dco"].get<double>(),
        report["avg_entry_select_dco"].get<double>(),
        report["avg_expanded"].get<double>());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"rnsg-2d-fast"};
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    QueryConfig query_cfg;
    auto* query = app.add_subcommand(
        "query",
        "Query a 2D-label RNSG graph through a filtered graph view and the "
        "shared RNSG Searcher");
    query->add_option("-d,--dataset_file", query_cfg.dataset_file)->required();
    query->add_option("-q,--query_file", query_cfg.query_file)->required();
    query->add_option("-l,--label_file", query_cfg.label_file)->required();
    query->add_option("-Q,--qrange_file", query_cfg.qrange_file)->required();
    query->add_option("-i,--graph_file", query_cfg.graph_file)->required();
    query->add_option("--entry_index_file", query_cfg.entry_index_file);
    query->add_option("-g,--groundtruth_file", query_cfg.groundtruth_file);
    query->add_option("-o,--result_file", query_cfg.result_file);
    query->add_option("--report_json", query_cfg.report_json);
    query->add_option("-K,--topk", query_cfg.topk);
    query->add_option("-b,--beam_size", query_cfg.beam_size);
    query->add_option("-T,--trunc_size", query_cfg.trunc_size);
    query->add_option("--entry_candidates", query_cfg.entry_candidates);
    query->add_option("--entry_scan_limit", query_cfg.entry_scan_limit);
    query->add_option("--nav_degree", query_cfg.nav_degree);
    query->add_option("--nav_scan_factor", query_cfg.nav_scan_factor);
    query->add_option("--nav_stall_rounds", query_cfg.nav_stall_rounds);
    query->add_option("--nav_front_keep", query_cfg.nav_front_keep);
    query->add_option("--nav_tail_degree", query_cfg.nav_tail_degree);
    query->add_option("--nav_early_stop_rounds",
                      query_cfg.nav_early_stop_rounds);
    query->add_option("--pick_scan_factor", query_cfg.pick_scan_factor);
    query->add_option("--pick_front_keep", query_cfg.pick_front_keep);
    query->add_option("--edge_pick_policy", query_cfg.edge_pick_policy)
        ->check(CLI::IsMember({"prefix", "side", "reciprocal",
                               "corebridge"}));
    query->add_option("--edge_pick_recip_depth",
                      query_cfg.edge_pick_recip_depth);
    query->add_option("--edge_pick_core_ratio",
                      query_cfg.edge_pick_core_ratio);
    query->add_option("--fallback_stall_rounds",
                      query_cfg.fallback_stall_rounds);
    query->add_option("--fallback_pick_policy",
                      query_cfg.fallback_pick_policy)
        ->check(CLI::IsMember({"prefix", "side", "reciprocal",
                               "corebridge"}));
    query->add_option("--fallback_core_ratio", query_cfg.fallback_core_ratio);
    query->add_option("--fallback_pick_front_keep",
                      query_cfg.fallback_pick_front_keep);
    query->add_option("--fallback_pick_scan_factor",
                      query_cfg.fallback_pick_scan_factor);
    query->add_flag("--fallback_release_nav",
                    query_cfg.fallback_release_nav);
    query->add_option("--rescue_slot_count", query_cfg.rescue_slot_count);
    query->add_option("--warmup_min", query_cfg.warmup_min);

    CLI11_PARSE(app, argc, argv);
    Utils::setup_logger(verbose, "rnsg-2d-fast");

    try {
        if (query->parsed()) {
            return run_query(query_cfg);
        }
        spdlog::error("No subcommand specified");
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 2;
    }
}
