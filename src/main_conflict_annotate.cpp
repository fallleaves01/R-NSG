#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/ExFunc.hpp>
#include <Utils/InitFunc.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

using TDFANN::Graph::TDGraphIndexBase;

// Binary format:
// Header:
//   [uint32_t magic = 0xC0NF1C70]
//   [uint32_t version = 1]
//   [uint32_t N]
//   [uint8_t  mode]  0=pairwise, 1=exact, 2=full
// Per node (sequential):
//   [uint32_t deg]
//   Per edge i = 0..deg-1 (in original adjacency order):
//     [float    dist_to_u]
//     [uint16_t conflict_count]
//     [uint16_t conflict_index[conflict_count]]  // indices in original adjacency order

static constexpr uint32_t CONFLICT_MAGIC = 0xC0FE1C70;
static constexpr uint32_t CONFLICT_VERSION = 1;

int main(int argc, char** argv) {
    CLI::App app{"rnsg_conflict_annotate: Explicit conflict list annotation"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string output_anno;
    std::string mode_str = "full";
    std::string report_json;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-o,--output_anno", output_anno,
                   "Output conflict annotation file (binary)")
        ->required();
    app.add_option("-m,--mode", mode_str, "Annotation mode: pairwise | exact | full")
        ->check(CLI::IsMember({"pairwise", "exact", "full"}))
        ->default_val("full");
    app.add_option("--report_json", report_json, "Optional JSON report");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_conflict_annotate");

    const bool is_exact = (mode_str == "exact");
    const bool is_full = (mode_str == "full");
    const uint8_t mode_byte = is_full ? 2 : (is_exact ? 1 : 0);

    // Load dataset and labels
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);

    // Load graph
    TDGraphIndexBase in_graph(input_index);
    const unsigned N = static_cast<unsigned>(in_graph.size());
    spdlog::info("Graph: {} nodes, avg degree {:.1f}", N,
                 static_cast<double>([&]() {
                     size_t total = 0;
                     for (unsigned i = 0; i < N; ++i)
                         total += in_graph.get_neighbours(i).size();
                     return total;
                 }()) /
                     N);
    spdlog::info("Mode: {}", mode_str);

    // Open output file
    std::ofstream fout(output_anno, std::ios::binary);
    if (!fout.good()) {
        throw std::runtime_error("Cannot open output file: " + output_anno);
    }

    // Write header
    uint32_t magic = CONFLICT_MAGIC;
    uint32_t version = CONFLICT_VERSION;
    uint32_t n_nodes = N;
    fout.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    fout.write(reinterpret_cast<const char*>(&version), sizeof(version));
    fout.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));
    fout.write(reinterpret_cast<const char*>(&mode_byte), sizeof(mode_byte));

    // Statistics
    std::atomic<size_t> total_edges{0};
    std::atomic<size_t> edges_with_conflicts{0};
    std::atomic<size_t> edges_no_conflicts{0};
    std::atomic<size_t> total_conflict_entries{0};
    std::atomic<size_t> max_conflict_len{0};
    std::atomic<size_t> nodes_processed{0};

    // Histogram buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, 32+
    std::atomic<size_t> hist_buckets[7] = {};

    // Per-node serialized data buffers — filled in parallel, written
    // sequentially afterwards.  This fixes the critical bug where
    // #pragma omp critical wrote node data in thread-completion order,
    // not in node-ID order, causing the reader to misattribute data.
    std::vector<std::vector<uint8_t>> node_buffers(N);

    spdlog::info("Computing conflict annotations (mode={})...", mode_str);

#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t ii = 0; ii < static_cast<int64_t>(N); ++ii) {
        const unsigned u = static_cast<unsigned>(ii);
        const auto& neighbors = in_graph.get_neighbours(u);
        const size_t deg = neighbors.size();

        if (deg == 0) {
#pragma omp critical
            {
                uint32_t d = 0;
                fout.write(reinterpret_cast<const char*>(&d), sizeof(d));
            }
            continue;
        }

        // Step 1: Compute distances from u to all neighbors
        struct EdgeInfo {
            float dist_to_u;
            unsigned nb_id;    // neighbor node ID
            unsigned orig_idx; // position in adjacency list
        };
        std::vector<EdgeInfo> edges(deg);
        for (size_t i = 0; i < deg; ++i) {
            edges[i] = {dataset.dist(u, neighbors[i].to),
                        neighbors[i].to, static_cast<unsigned>(i)};
        }

        // Step 2: Sort by (distance, orig_idx) ascending — must match
        // runtime's std::sort on pair<float,unsigned> which uses
        // lexicographic (dist, orig_idx) ordering.
        auto dist_order = edges;
        std::ranges::sort(dist_order, [](const EdgeInfo& a, const EdgeInfo& b) {
            return std::tie(a.dist_to_u, a.orig_idx) <
                   std::tie(b.dist_to_u, b.orig_idx);
        });

        // Step 3: Compute conflicts per edge
        // conflicts_by_orig[orig_idx] = list of orig_idx values that prune
        // this edge. Storing orig_idx (not dist_order position) so runtime
        // can match by original adjacency index regardless of range filtering.
        std::vector<std::vector<uint16_t>> conflicts_by_orig(deg);
        std::vector<unsigned> accepted;  // dist_order positions accepted
        std::vector<uint16_t> accepted_orig;  // corresponding orig_idx values

        for (size_t di = 0; di < deg; ++di) {
            const float d_ux = dist_order[di].dist_to_u;
            const unsigned x_id = dist_order[di].nb_id;
            const unsigned orig_idx = dist_order[di].orig_idx;
            std::vector<uint16_t>& conf = conflicts_by_orig[orig_idx];

            if (is_full) {
                // Full conflict closure: ALL preceding edges in distance
                // order that geometrically satisfy the HNSW pruning
                // condition, regardless of whether they are in the accepted
                // set. This is the key difference from exact/pairwise modes
                // which only check accepted edges.
                // Pruning condition (from Builder.hpp check_valid):
                //   d_ux > d_uy AND d_ux > dist(x, y)
                for (size_t dj = 0; dj < di; ++dj) {
                    if (dist_order[dj].dist_to_u >= d_ux) continue;
                    float d_yx = dataset.dist(dist_order[dj].nb_id, x_id);
                    if (d_yx < d_ux) {
                        conf.push_back(
                            static_cast<uint16_t>(dist_order[dj].orig_idx));
                    }
                }
            } else if (is_exact) {
                // Exact mode: find first accepted edge that prunes this
                for (size_t ai = 0; ai < accepted.size(); ++ai) {
                    unsigned aj = accepted[ai];
                    if (dist_order[aj].dist_to_u >= d_ux) continue;
                    float d_yx = dataset.dist(dist_order[aj].nb_id, x_id);
                    if (d_yx < d_ux) {
                        // Store the pruner's orig_idx, not dist_order position
                        conf.push_back(accepted_orig[ai]);
                        break;  // exact: one pruner suffices
                    }
                }
            } else {
                // Pairwise mode: all accepted edges that geometrically prune
                for (size_t ai = 0; ai < accepted.size(); ++ai) {
                    unsigned aj = accepted[ai];
                    if (dist_order[aj].dist_to_u >= d_ux) continue;
                    float d_yx = dataset.dist(dist_order[aj].nb_id, x_id);
                    if (d_yx < d_ux) {
                        conf.push_back(accepted_orig[ai]);
                    }
                }
            }

            if (!is_full && conf.empty()) {
                accepted.push_back(static_cast<unsigned>(di));
                accepted_orig.push_back(static_cast<uint16_t>(orig_idx));
            }

            // Update statistics
            size_t clen = conf.size();
            total_conflict_entries.fetch_add(clen);
            if (clen > 0) {
                edges_with_conflicts.fetch_add(1);
            } else {
                edges_no_conflicts.fetch_add(1);
            }
            size_t prev_max = max_conflict_len.load();
            while (clen > prev_max &&
                   !max_conflict_len.compare_exchange_weak(prev_max, clen))
                ;

            // Histogram bucket
            size_t bucket;
            if (clen == 0)
                bucket = 0;
            else if (clen == 1)
                bucket = 1;
            else if (clen <= 3)
                bucket = 2;
            else if (clen <= 7)
                bucket = 3;
            else if (clen <= 15)
                bucket = 4;
            else if (clen <= 31)
                bucket = 5;
            else
                bucket = 6;
            hist_buckets[bucket].fetch_add(1);

            total_edges.fetch_add(1);
        }

        // Serialize per-node data into buffer (in original adjacency order)
        // This replaces the old #pragma omp critical file write which
        // wrote data in thread-completion order, not node-ID order.
        {
            std::vector<uint8_t>& buf = node_buffers[u];
            // Pre-compute buffer size
            size_t buf_size = sizeof(uint32_t);  // deg
            for (size_t i = 0; i < deg; ++i) {
                buf_size += sizeof(float);  // dist_to_u
                buf_size += sizeof(uint16_t);  // conflict_count
                buf_size += conflicts_by_orig[i].size() * sizeof(uint16_t);
            }
            buf.resize(buf_size);
            size_t off = 0;

            uint32_t d = static_cast<uint32_t>(deg);
            std::memcpy(buf.data() + off, &d, sizeof(d));
            off += sizeof(d);
            for (size_t i = 0; i < deg; ++i) {
                // dist_to_u
                std::memcpy(buf.data() + off, &edges[i].dist_to_u,
                           sizeof(float));
                off += sizeof(float);
                // conflict count + indices
                const auto& conf = conflicts_by_orig[i];
                uint16_t cnt = static_cast<uint16_t>(conf.size());
                std::memcpy(buf.data() + off, &cnt, sizeof(cnt));
                off += sizeof(cnt);
                if (cnt > 0) {
                    std::memcpy(buf.data() + off, conf.data(),
                               cnt * sizeof(uint16_t));
                    off += cnt * sizeof(uint16_t);
                }
            }
        }

        size_t done = ++nodes_processed;
        if (done % 100000 == 0) {
            spdlog::info("Processed {}/{} nodes...", done, N);
        }
    }

    // Write all node data sequentially in node-ID order
    spdlog::info("Writing annotation data in node order...");
    for (unsigned ni = 0; ni < N; ++ni) {
        fout.write(reinterpret_cast<const char*>(node_buffers[ni].data()),
                   node_buffers[ni].size());
    }

    spdlog::info("Annotation complete.");
    spdlog::info("  Total edges: {}", total_edges.load());
    spdlog::info("  Edges with conflicts: {} ({:.1f}%)",
                 edges_with_conflicts.load(),
                 100.0 * edges_with_conflicts.load() / total_edges.load());
    spdlog::info("  Edges no conflicts (always survive): {} ({:.1f}%)",
                 edges_no_conflicts.load(),
                 100.0 * edges_no_conflicts.load() / total_edges.load());
    spdlog::info("  Total conflict entries: {}",
                 total_conflict_entries.load());
    spdlog::info(
        "  Avg conflicts per edge: {:.2f}",
        static_cast<double>(total_conflict_entries.load()) / total_edges.load());
    spdlog::info(
        "  Avg conflicts per pruned edge: {:.2f}",
        edges_with_conflicts.load() > 0
            ? static_cast<double>(total_conflict_entries.load()) /
                  edges_with_conflicts.load()
            : 0.0);
    spdlog::info("  Max conflict list length: {}", max_conflict_len.load());
    spdlog::info("  Histogram: 0={}, 1={}, 2-3={}, 4-7={}, 8-15={}, "
                 "16-31={}, 32+={}",
                 hist_buckets[0].load(), hist_buckets[1].load(),
                 hist_buckets[2].load(), hist_buckets[3].load(),
                 hist_buckets[4].load(), hist_buckets[5].load(),
                 hist_buckets[6].load());

    fout.close();
    spdlog::info("Saved conflict annotations to {}", output_anno);

    // Optional report
    if (!report_json.empty()) {
        nlohmann::json js = {
            {"mode", mode_str},
            {"node_count", N},
            {"total_edges", total_edges.load()},
            {"edges_with_conflicts", edges_with_conflicts.load()},
            {"edges_with_conflicts_pct",
             100.0 * edges_with_conflicts.load() / total_edges.load()},
            {"edges_no_conflicts", edges_no_conflicts.load()},
            {"edges_no_conflicts_pct",
             100.0 * edges_no_conflicts.load() / total_edges.load()},
            {"total_conflict_entries", total_conflict_entries.load()},
            {"avg_conflicts_per_edge",
             static_cast<double>(total_conflict_entries.load()) /
                 total_edges.load()},
            {"avg_conflicts_per_pruned_edge",
             edges_with_conflicts.load() > 0
                 ? static_cast<double>(total_conflict_entries.load()) /
                       edges_with_conflicts.load()
                 : 0.0},
            {"max_conflict_list_length", max_conflict_len.load()},
            {"histogram",
             {{"0", hist_buckets[0].load()},
              {"1", hist_buckets[1].load()},
              {"2-3", hist_buckets[2].load()},
              {"4-7", hist_buckets[3].load()},
              {"8-15", hist_buckets[4].load()},
              {"16-31", hist_buckets[5].load()},
              {"32+", hist_buckets[6].load()}}},
        };
        std::ofstream jout(report_json);
        jout << js.dump(2);
        spdlog::info("Saved report to {}", report_json);
    }

    return 0;
}
