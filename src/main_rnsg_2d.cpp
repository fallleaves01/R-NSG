#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

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

struct BuildConfig {
    std::string dataset_file;
    std::string label_file;
    std::string knng_file;
    std::string graph_file;
    std::string entry_index_file;
    std::string report_json;
    unsigned max_degree = 32;
    unsigned range_window = 64;
    unsigned exact_knn_k = 32;
    unsigned exact_knn_limit = 20000;
    unsigned entry_grid_size = 128;
    unsigned threads = 0;
    bool reverse_refine = true;
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
};

struct GTConfig {
    std::string dataset_file;
    std::string query_file;
    std::string label_file;
    std::string qrange_file;
    std::string output_file;
    unsigned topk = 10;
};

double now_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

bool in_range(const Label2D& label, const Range2D& range);

std::string default_entry_index_path(const std::string& graph_file) {
    return graph_file + ".entry2d";
}

struct EntryGridIndex {
    static constexpr std::uint32_t kMagic = 0x32454752u;  // RGE2

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

    void build(const std::vector<Label2D>& labels, unsigned requested_grid) {
        grid_size = std::max(1u, requested_grid);
        cells.clear();
        cells.resize(static_cast<size_t>(grid_size) * grid_size);
        if (labels.empty()) {
            return;
        }
        min_x = max_x = labels.front().x;
        min_y = max_y = labels.front().y;
        for (const auto& label : labels) {
            min_x = std::min(min_x, label.x);
            max_x = std::max(max_x, label.x);
            min_y = std::min(min_y, label.y);
            max_y = std::max(max_y, label.y);
        }
        for (unsigned i = 0; i < labels.size(); ++i) {
            const unsigned gx = coord(labels[i].x, min_x, max_x);
            const unsigned gy = coord(labels[i].y, min_y, max_y);
            cells[cell_id(gx, gy)].push_back(i);
        }
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
            limit == 0 ? std::numeric_limits<unsigned>::max()
                       : std::max<unsigned>(
                             1, static_cast<unsigned>(
                                    std::ceil(static_cast<double>(limit) /
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

    bool save(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);
        if (!fout.good()) {
            return false;
        }
        return IO::save(fout, kMagic) && IO::save(fout, grid_size) &&
               IO::save(fout, min_x) && IO::save(fout, max_x) &&
               IO::save(fout, min_y) && IO::save(fout, max_y) &&
               IO::save(fout, cells);
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
                    "2D label nested format must be [[x,y], ...]");
            }
            labels.push_back({item[0].get<std::uint64_t>(),
                              item[1].get<std::uint64_t>()});
        }
    } else {
        if (j.size() % 2 != 0) {
            throw std::runtime_error(
                "2D label flat format must contain an even number of values");
        }
        labels.reserve(j.size() / 2);
        for (size_t i = 0; i < j.size(); i += 2) {
            labels.push_back(
                {j[i].get<std::uint64_t>(), j[i + 1].get<std::uint64_t>()});
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

bool in_range(const Label2D& label, const Range2D& range) {
    return label.x >= range.lx && label.x <= range.rx && label.y >= range.ly &&
           label.y <= range.ry;
}

bool box_between(const Label2D& a, const Label2D& b, const Label2D& c) {
    const auto min_x = std::min(a.x, c.x);
    const auto max_x = std::max(a.x, c.x);
    const auto min_y = std::min(a.y, c.y);
    const auto max_y = std::max(a.y, c.y);
    return b.x >= min_x && b.x <= max_x && b.y >= min_y && b.y <= max_y;
}

std::vector<unsigned> order_by_dim(const std::vector<Label2D>& labels,
                                   unsigned dim) {
    std::vector<unsigned> order(labels.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](unsigned lhs, unsigned rhs) {
        const auto lv = dim == 0 ? labels[lhs].x : labels[lhs].y;
        const auto rv = dim == 0 ? labels[rhs].x : labels[rhs].y;
        return std::pair{lv, lhs} < std::pair{rv, rhs};
    });
    return order;
}

std::vector<unsigned> inverse_order(const std::vector<unsigned>& order) {
    std::vector<unsigned> pos(order.size());
    for (unsigned i = 0; i < order.size(); ++i) {
        pos[order[i]] = i;
    }
    return pos;
}

void sort_unique(std::vector<unsigned>& values) {
    std::ranges::sort(values);
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

template <typename T>
std::vector<unsigned> exact_knn_candidates(
    const Vector::VectorList<T>& dataset,
    unsigned src,
    unsigned k) {
    std::vector<std::pair<T, unsigned>> dists;
    dists.reserve(dataset.size() > 0 ? dataset.size() - 1 : 0);
    for (unsigned j = 0; j < dataset.size(); ++j) {
        if (j == src) continue;
        dists.push_back({dataset.dist2(src, j), j});
    }
    if (k < dists.size()) {
        std::nth_element(dists.begin(), dists.begin() + k, dists.end());
        dists.resize(k);
    }
    std::ranges::sort(dists);
    std::vector<unsigned> out;
    out.reserve(dists.size());
    for (auto [_, id] : dists) {
        out.push_back(id);
    }
    return out;
}

template <typename T>
std::vector<unsigned> prune_2d(
    const Vector::VectorList<T>& dataset,
    const std::vector<Label2D>& labels,
    unsigned src,
    std::vector<unsigned> candidates,
    unsigned max_degree) {
    sort_unique(candidates);
    candidates.erase(std::remove(candidates.begin(), candidates.end(), src),
                     candidates.end());

    std::vector<std::pair<T, unsigned>> ordered;
    ordered.reserve(candidates.size());
    for (unsigned c : candidates) {
        ordered.push_back({dataset.dist2(src, c), c});
    }
    std::ranges::sort(ordered);

    std::vector<std::pair<unsigned, T>> selected;
    selected.reserve(max_degree);
    for (auto [dist_ac, cand] : ordered) {
        bool dominated = false;
        for (const auto& [witness, dist_ab] : selected) {
            if (!box_between(labels[src], labels[witness], labels[cand])) {
                continue;
            }
            const T dist_bc = dataset.dist2(witness, cand);
            if (dist_ac > dist_ab && dist_ac > dist_bc) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            selected.push_back({cand, dist_ac});
            if (selected.size() >= max_degree) {
                break;
            }
        }
    }

    std::vector<unsigned> out;
    out.reserve(selected.size());
    for (auto [id, _] : selected) {
        out.push_back(id);
    }
    return out;
}

GraphIndex make_graph_from_adj(const std::vector<std::vector<unsigned>>& adj) {
    GraphIndex graph(static_cast<unsigned>(adj.size()));
    for (unsigned i = 0; i < adj.size(); ++i) {
        std::vector<Node> nodes;
        nodes.reserve(adj[i].size());
        for (unsigned to : adj[i]) {
            nodes.push_back(Node{to});
        }
        graph.set_neighbours(i, std::move(nodes));
    }
    return graph;
}

std::vector<std::vector<unsigned>> load_knng_adj(const std::string& path,
                                                 size_t expected_n) {
    GraphIndex knng(path);
    if (knng.size() != expected_n) {
        throw std::runtime_error("KNNG node count does not match dataset size");
    }
    std::vector<std::vector<unsigned>> adj(expected_n);
    for (unsigned i = 0; i < expected_n; ++i) {
        for (auto n : knng.get_neighbours(i)) {
            adj[i].push_back(n.to);
        }
    }
    return adj;
}

int run_build(const BuildConfig& cfg) {
    if (cfg.graph_file.empty()) {
        throw std::runtime_error("build requires --graph_file");
    }
#ifdef _OPENMP
    if (cfg.threads > 0) {
        omp_set_num_threads(static_cast<int>(cfg.threads));
    }
#endif

    const double t0 = now_seconds();
    Vector::VectorList<float> dataset(cfg.dataset_file);
    const auto labels = load_labels_2d(cfg.label_file, dataset.size());
    const auto order_x = order_by_dim(labels, 0);
    const auto order_y = order_by_dim(labels, 1);
    const auto pos_x = inverse_order(order_x);
    const auto pos_y = inverse_order(order_y);
    const unsigned n = dataset.size();

    std::vector<std::vector<unsigned>> knng_adj;
    if (!cfg.knng_file.empty()) {
        knng_adj = load_knng_adj(cfg.knng_file, n);
        spdlog::info("Loaded KNNG candidates from {}", cfg.knng_file);
    } else {
        if (n > cfg.exact_knn_limit) {
            throw std::runtime_error(
                "No --knng_file supplied and dataset exceeds --exact_knn_limit");
        }
        knng_adj.resize(n);
        spdlog::info("Building exact KNN candidates: n={}, k={}", n,
                     cfg.exact_knn_k);
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            knng_adj[static_cast<unsigned>(i)] =
                exact_knn_candidates(dataset, static_cast<unsigned>(i),
                                     cfg.exact_knn_k);
        }
    }

    std::vector<std::vector<unsigned>> adj(n);
    const auto add_window = [&](std::vector<unsigned>& cand,
                                const std::vector<unsigned>& order,
                                const std::vector<unsigned>& pos,
                                unsigned src) {
        const int p = static_cast<int>(pos[src]);
        const int nn = static_cast<int>(order.size());
        for (unsigned off = 1; off <= cfg.range_window; ++off) {
            const int left = p - static_cast<int>(off);
            const int right = p + static_cast<int>(off);
            if (left >= 0) cand.push_back(order[static_cast<size_t>(left)]);
            if (right < nn) cand.push_back(order[static_cast<size_t>(right)]);
        }
    };

    spdlog::info("Pruning 2D RNSG graph: n={}, max_degree={}, range_window={}",
                 n, cfg.max_degree, cfg.range_window);
#pragma omp parallel for schedule(dynamic, 64)
    for (int si = 0; si < static_cast<int>(n); ++si) {
        const unsigned src = static_cast<unsigned>(si);
        std::vector<unsigned> cand;
        cand.reserve(knng_adj[src].size() + cfg.range_window * 4);
        cand.insert(cand.end(), knng_adj[src].begin(), knng_adj[src].end());
        add_window(cand, order_x, pos_x, src);
        add_window(cand, order_y, pos_y, src);
        adj[src] = prune_2d(dataset, labels, src, std::move(cand),
                            cfg.max_degree);
    }

    if (cfg.reverse_refine) {
        spdlog::info("Applying 2D reverse-refine pass");
        std::vector<std::vector<unsigned>> augmented = adj;
        for (unsigned src = 0; src < n; ++src) {
            for (unsigned dst : adj[src]) {
                augmented[dst].push_back(src);
            }
        }
#pragma omp parallel for schedule(dynamic, 64)
        for (int si = 0; si < static_cast<int>(n); ++si) {
            const unsigned src = static_cast<unsigned>(si);
            adj[src] = prune_2d(dataset, labels, src, std::move(augmented[src]),
                                cfg.max_degree);
        }
    }

    std::uint64_t edge_count = 0;
    unsigned max_out_degree = 0;
    for (const auto& row : adj) {
        edge_count += row.size();
        max_out_degree =
            std::max<unsigned>(max_out_degree, static_cast<unsigned>(row.size()));
    }

    auto graph = make_graph_from_adj(adj);
    std::ofstream fout(cfg.graph_file, std::ios::binary);
    if (!graph.save(fout)) {
        throw std::runtime_error("Failed to save graph: " + cfg.graph_file);
    }
    fout.close();

    const std::string entry_path =
        cfg.entry_index_file.empty() ? default_entry_index_path(cfg.graph_file)
                                     : cfg.entry_index_file;
    EntryGridIndex entry_index;
    entry_index.build(labels, cfg.entry_grid_size);
    if (!entry_index.save(entry_path)) {
        throw std::runtime_error("Failed to save entry index: " + entry_path);
    }

    const double build_seconds = now_seconds() - t0;
    const auto graph_bytes =
        std::filesystem::exists(cfg.graph_file)
            ? std::filesystem::file_size(cfg.graph_file)
            : 0;
    const auto entry_index_bytes =
        std::filesystem::exists(entry_path)
            ? std::filesystem::file_size(entry_path)
            : 0;

    nlohmann::json report;
    report["target"] = "rnsg-2d";
    report["dataset_file"] = cfg.dataset_file;
    report["label_file"] = cfg.label_file;
    report["knng_file"] = cfg.knng_file;
    report["graph_file"] = cfg.graph_file;
    report["entry_index_file"] = entry_path;
    report["n"] = n;
    report["dim"] = dataset.dim();
    report["max_degree"] = cfg.max_degree;
    report["range_window"] = cfg.range_window;
    report["entry_grid_size"] = cfg.entry_grid_size;
    report["exact_knn_k"] = cfg.exact_knn_k;
    report["reverse_refine"] = cfg.reverse_refine;
    report["edge_count"] = edge_count;
    report["avg_degree"] = n == 0 ? 0.0 : static_cast<double>(edge_count) / n;
    report["max_out_degree"] = max_out_degree;
    report["build_seconds"] = build_seconds;
    report["index_bytes"] = graph_bytes;
    report["entry_index_bytes"] = entry_index_bytes;

    if (!cfg.report_json.empty()) {
        std::ofstream jout(cfg.report_json);
        jout << report.dump(2) << "\n";
    }
    spdlog::info("2D build done: avg_degree={:.2f}, seconds={:.3f}",
                 report["avg_degree"].get<double>(), build_seconds);
    return 0;
}

template <typename T>
std::vector<unsigned> brute_force_topk(const Vector::VectorList<T>& dataset,
                                       const auto& query,
                                       const std::vector<Label2D>& labels,
                                       const Range2D& range,
                                       unsigned topk) {
    std::vector<std::pair<T, unsigned>> dists;
    for (unsigned j = 0; j < dataset.size(); ++j) {
        if (in_range(labels[j], range)) {
            dists.push_back({dataset.dist2(j, query), j});
        }
    }
    if (topk < dists.size()) {
        std::nth_element(dists.begin(), dists.begin() + topk, dists.end());
        dists.resize(topk);
    }
    std::ranges::sort(dists);
    std::vector<unsigned> out(topk, std::numeric_limits<unsigned>::max());
    for (unsigned i = 0; i < std::min<unsigned>(topk, dists.size()); ++i) {
        out[i] = dists[i].second;
    }
    return out;
}

int run_groundtruth(const GTConfig& cfg) {
    Vector::VectorList<float> dataset(cfg.dataset_file);
    Vector::VectorList<float> queries(cfg.query_file);
    const auto labels = load_labels_2d(cfg.label_file, dataset.size());
    const auto ranges = load_ranges_2d(cfg.qrange_file, queries.size());

    std::vector<unsigned> flat;
    flat.reserve(static_cast<size_t>(queries.size()) * cfg.topk);
    const double t0 = now_seconds();
    for (unsigned qi = 0; qi < queries.size(); ++qi) {
        auto ans = brute_force_topk(dataset, queries[qi], labels, ranges[qi],
                                    cfg.topk);
        flat.insert(flat.end(), ans.begin(), ans.end());
    }
    nlohmann::json out = flat;
    std::ofstream fout(cfg.output_file);
    fout << out.dump() << "\n";
    spdlog::info("2D groundtruth done: queries={}, topk={}, seconds={:.3f}",
                 queries.size(), cfg.topk, now_seconds() - t0);
    return 0;
}

template <typename T>
std::vector<std::pair<T, unsigned>> select_entries(
    const Vector::VectorList<T>& dataset,
    const auto& query,
    const std::vector<Label2D>& labels,
    const Range2D& range,
    const EntryGridIndex* entry_index,
    unsigned entry_candidates,
    unsigned entry_scan_limit) {
    std::vector<std::pair<T, unsigned>> entries;
    if (entry_index != nullptr) {
        auto ids = entry_index->candidates(labels, range, entry_scan_limit);
        for (unsigned id : ids) {
            entries.push_back({dataset.dist2(id, query), id});
        }
    } else {
        unsigned accepted = 0;
        for (unsigned j = 0; j < dataset.size(); ++j) {
            if (!in_range(labels[j], range)) {
                continue;
            }
            entries.push_back({dataset.dist2(j, query), j});
            ++accepted;
            if (entry_scan_limit != 0 && accepted >= entry_scan_limit) {
                break;
            }
        }
    }
    if (entry_candidates < entries.size()) {
        std::nth_element(entries.begin(), entries.begin() + entry_candidates,
                         entries.end());
        entries.resize(entry_candidates);
    }
    std::ranges::sort(entries);
    return entries;
}

template <typename T>
void insert_sorted_limited(std::vector<std::pair<T, unsigned>>& values,
                           std::pair<T, unsigned> item,
                           size_t limit) {
    auto it = std::lower_bound(values.begin(), values.end(), item);
    values.insert(it, item);
    if (values.size() > limit) {
        values.pop_back();
    }
}

template <typename T>
std::vector<unsigned> search_2d(const Vector::VectorList<T>& dataset,
                                const GraphIndex& graph,
                                const std::vector<Label2D>& labels,
                                const EntryGridIndex* entry_index,
                                const auto& query,
                                const Range2D& range,
                                unsigned topk,
                                unsigned beam_size,
                                unsigned trunc_size,
                                unsigned entry_candidates,
                                unsigned entry_scan_limit,
                                std::uint64_t& dco,
                                unsigned& expanded) {
    std::vector<std::uint8_t> visited(dataset.size(), 0);
    std::vector<std::pair<T, unsigned>> beam;
    std::vector<std::pair<T, unsigned>> seen;
    beam.reserve(beam_size + entry_candidates + 8);
    seen.reserve(beam_size * std::max<unsigned>(trunc_size, 1) + 8);

    auto entries = select_entries(dataset, query, labels, range, entry_index,
                                  entry_candidates, entry_scan_limit);
    for (auto e : entries) {
        visited[e.second] = 1;
        insert_sorted_limited(beam, e, std::max<unsigned>(beam_size, topk));
        seen.push_back(e);
        ++dco;
    }
    if (beam.empty()) {
        return std::vector<unsigned>(topk, std::numeric_limits<unsigned>::max());
    }

    size_t cursor = 0;
    expanded = 0;
    while (cursor < beam.size() && expanded < beam_size) {
        const unsigned cur = beam[cursor++].second;
        ++expanded;
        unsigned accepted = 0;
        for (auto node : graph.get_neighbours(cur)) {
            const unsigned nb = node.to;
            if (nb >= dataset.size() || visited[nb]) {
                continue;
            }
            visited[nb] = 1;
            if (!in_range(labels[nb], range)) {
                continue;
            }
            if (trunc_size != 0 && accepted >= trunc_size) {
                continue;
            }
            ++accepted;
            const T dist = dataset.dist2(nb, query);
            ++dco;
            insert_sorted_limited(beam, {dist, nb},
                                  std::max<unsigned>(beam_size, topk));
            seen.push_back({dist, nb});
        }
    }

    std::ranges::sort(seen);
    seen.erase(std::unique(seen.begin(), seen.end(),
                           [](const auto& a, const auto& b) {
                               return a.second == b.second;
                           }),
               seen.end());
    std::ranges::sort(seen);
    std::vector<unsigned> out(topk, std::numeric_limits<unsigned>::max());
    for (unsigned i = 0; i < std::min<unsigned>(topk, seen.size()); ++i) {
        out[i] = seen[i].second;
    }
    return out;
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

    std::vector<unsigned> flat;
    flat.reserve(static_cast<size_t>(queries.size()) * cfg.topk);
    std::uint64_t total_dco = 0;
    std::uint64_t total_expanded = 0;
    const double t0 = now_seconds();
    for (unsigned qi = 0; qi < queries.size(); ++qi) {
        std::uint64_t dco = 0;
        unsigned expanded = 0;
        auto ans = search_2d(dataset, graph, labels,
                             has_entry_index ? &entry_index : nullptr,
                             queries[qi], ranges[qi], cfg.topk,
                             cfg.beam_size, cfg.trunc_size,
                             cfg.entry_candidates, cfg.entry_scan_limit, dco,
                             expanded);
        total_dco += dco;
        total_expanded += expanded;
        flat.insert(flat.end(), ans.begin(), ans.end());
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

    nlohmann::json report;
    report["target"] = "rnsg-2d";
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
    report["entry_candidates"] = cfg.entry_candidates;
    report["entry_scan_limit"] = cfg.entry_scan_limit;
    report["query_count"] = queries.size();
    report["query_seconds"] = query_seconds;
    report["qps"] = qps;
    report["recall"] = recall;
    report["avg_dco"] = static_cast<double>(total_dco) / queries.size();
    report["avg_expanded"] =
        static_cast<double>(total_expanded) / queries.size();

    if (!cfg.report_json.empty()) {
        std::ofstream jout(cfg.report_json);
        jout << report.dump(2) << "\n";
    }
    spdlog::info(
        "2D query done: recall={}, qps={:.2f}, avg_dco={:.2f}, seconds={:.3f}",
        std::isnan(recall) ? -1.0 : recall, qps,
        report["avg_dco"].get<double>(), query_seconds);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"rnsg-2d"};
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    BuildConfig build_cfg;
    bool no_reverse_refine = false;
    auto* build = app.add_subcommand("build", "Build a 2D-label RNSG graph");
    build->add_option("-d,--dataset_file", build_cfg.dataset_file)
        ->required();
    build->add_option("-l,--label_file", build_cfg.label_file)->required();
    build->add_option("-k,--knng_file", build_cfg.knng_file);
    build->add_option("-o,--graph_file", build_cfg.graph_file)->required();
    build->add_option("--entry_index_file", build_cfg.entry_index_file);
    build->add_option("--report_json", build_cfg.report_json);
    build->add_option("-M,--max_degree", build_cfg.max_degree);
    build->add_option("--range_window", build_cfg.range_window);
    build->add_option("--exact_knn_k", build_cfg.exact_knn_k);
    build->add_option("--exact_knn_limit", build_cfg.exact_knn_limit);
    build->add_option("--entry_grid_size", build_cfg.entry_grid_size);
    build->add_option("-t,--threads", build_cfg.threads);
    build->add_flag("--no_reverse_refine", no_reverse_refine);

    GTConfig gt_cfg;
    auto* gt =
        app.add_subcommand("groundtruth", "Generate brute-force 2D GT");
    gt->add_option("-d,--dataset_file", gt_cfg.dataset_file)->required();
    gt->add_option("-q,--query_file", gt_cfg.query_file)->required();
    gt->add_option("-l,--label_file", gt_cfg.label_file)->required();
    gt->add_option("-Q,--qrange_file", gt_cfg.qrange_file)->required();
    gt->add_option("-o,--output_file", gt_cfg.output_file)->required();
    gt->add_option("-K,--topk", gt_cfg.topk);

    QueryConfig query_cfg;
    auto* query = app.add_subcommand("query", "Query a 2D-label RNSG graph");
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

    CLI11_PARSE(app, argc, argv);
    Utils::setup_logger(verbose, "rnsg-2d");

    try {
        if (build->parsed()) {
            build_cfg.reverse_refine = !no_reverse_refine;
            return run_build(build_cfg);
        }
        if (gt->parsed()) {
            return run_groundtruth(gt_cfg);
        }
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
