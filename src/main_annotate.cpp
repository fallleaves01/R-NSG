#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/ExFunc.hpp>
#include <Utils/InitFunc.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

using TDFANN::Graph::TDGraphIndexBase;
using TDFANN::Graph::to_node;

int main(int argc, char** argv) {
    CLI::App app{"rnsg_annotate: Left/right independent survival annotation"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string output_anno;  // annotation file (binary)
    std::string report_json;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-o,--output_anno", output_anno,
                   "Output annotation file (binary: per edge, 2x uint64_t L_drop/R_drop)")
        ->required();
    app.add_option("--report_json", report_json, "Optional JSON report");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_annotate");

    // Load dataset and labels
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);
    auto sorted_label = TDFANN::Utils::sorted_vec(label);
    const uint64_t MAX_LABEL = sorted_label.empty() ? 99999 : sorted_label.back();

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

    // Annotation format: for each node, per edge: (L_drop, R_drop)
    // L_drop = 0 means no left pruner (always survives left expansion)
    // R_drop = MAX_LABEL means no right pruner

    // Statistics
    std::atomic<size_t> total_edges{0};
    std::atomic<size_t> always_survive{0};
    std::atomic<size_t> left_only_pruned{0};
    std::atomic<size_t> right_only_pruned{0};
    std::atomic<size_t> both_sides_pruned{0};
    std::atomic<double> core_size_sum{0.0};
    std::atomic<size_t> nodes_processed{0};

    // Open output annotation file
    // Format: [uint32_t N] [per node: uint32_t deg, then deg pairs of (uint64_t L_drop, uint64_t R_drop)]
    std::ofstream fout(output_anno, std::ios::binary);
    if (!fout.good()) {
        throw std::runtime_error("Cannot open output annotation file: " + output_anno);
    }
    uint32_t n_nodes = N;
    fout.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));

    spdlog::info("Computing left/right independent survival annotations...");

#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t ii = 0; ii < static_cast<int64_t>(N); ++ii) {
        const unsigned u = static_cast<unsigned>(ii);
        const auto& neighbors = in_graph.get_neighbours(u);
        const size_t deg = neighbors.size();

        // Per-node annotation buffer
        std::vector<std::pair<uint64_t, uint64_t>> anno(deg, {0, MAX_LABEL});

        if (deg <= 2) {
            // Too few neighbors, all always survive
            // Write to file under critical section
            #pragma omp critical
            {
                uint32_t d = static_cast<uint32_t>(deg);
                fout.write(reinterpret_cast<const char*>(&d), sizeof(d));
                for (size_t i = 0; i < deg; ++i) {
                    fout.write(reinterpret_cast<const char*>(&anno[i]), 2 * sizeof(uint64_t));
                }
            }
            total_edges.fetch_add(deg);
            always_survive.fetch_add(deg);
            core_size_sum += static_cast<double>(deg);
            continue;
        }
        total_edges.fetch_add(deg);

        const uint64_t mid = sorted_label[u];

        // Step 1: Compute distances from u to all neighbors
        std::vector<std::pair<float, unsigned>> edges;
        edges.reserve(deg);
        for (const auto& nb : neighbors) {
            edges.push_back({dataset.dist(u, nb.to), nb.to});
        }

        // Step 2: Split into left, right, center
        // left: label < mid, sorted by |label - mid| ascending (closest to mid first)
        // right: label > mid, sorted by |label - mid| ascending
        // center: label == mid
        struct CandInfo {
            uint64_t label_gap;  // |label - mid|
            uint64_t label;
            float dist_to_u;
            unsigned edge_idx;  // index into edges/anno
        };

        std::vector<CandInfo> left_cands, right_cands;
        std::vector<unsigned> center_indices;

        for (size_t i = 0; i < deg; ++i) {
            uint64_t nb_label = sorted_label[edges[i].second];
            if (nb_label < mid) {
                left_cands.push_back({mid - nb_label, nb_label, edges[i].first,
                                      static_cast<unsigned>(i)});
            } else if (nb_label > mid) {
                right_cands.push_back({nb_label - mid, nb_label, edges[i].first,
                                       static_cast<unsigned>(i)});
            } else {
                center_indices.push_back(static_cast<unsigned>(i));
            }
        }

        // Sort by label gap ascending (closest to mid first)
        std::ranges::sort(left_cands, {}, &CandInfo::label_gap);
        std::ranges::sort(right_cands, {}, &CandInfo::label_gap);

        // Step 3: For each edge x, compute L_drop and R_drop independently
        for (size_t xi = 0; xi < deg; ++xi) {
            const unsigned x_idx = static_cast<unsigned>(xi);
            const float d_ux = edges[xi].first;
            const unsigned x_id = edges[xi].second;

            // === LEFT EXPANSION TEACHER ===
            // Find the first left candidate (closest to mid in label) that can prune x
            // HNSW rule: dist(u,y) < dist(u,x) AND dist(y,x) < dist(u,x)
            for (const auto& lc : left_cands) {
                if (lc.dist_to_u >= d_ux) continue;  // y must be closer to u than x
                float d_yx = dataset.dist(edges[lc.edge_idx].second, x_id);
                if (d_yx < d_ux) {
                    // y can prune x! Record L_drop = y's label (the current left boundary)
                    anno[x_idx].first = lc.label;
                    break;
                }
            }

            // === RIGHT EXPANSION TEACHER ===
            // Find the first right candidate (closest to mid in label) that can prune x
            for (const auto& rc : right_cands) {
                if (rc.dist_to_u >= d_ux) continue;
                float d_yx = dataset.dist(edges[rc.edge_idx].second, x_id);
                if (d_yx < d_ux) {
                    anno[x_idx].second = rc.label;
                    break;
                }
            }

            // Statistics
            bool l_survives = (anno[x_idx].first == 0);
            bool r_survives = (anno[x_idx].second == MAX_LABEL);
            if (l_survives && r_survives) {
                always_survive++;
            } else if (!l_survives && r_survives) {
                left_only_pruned++;
            } else if (l_survives && !r_survives) {
                right_only_pruned++;
            } else {
                both_sides_pruned++;
            }
        }

        // Count core size for this node
        size_t core = 0;
        for (size_t i = 0; i < deg; ++i) {
            if (anno[i].first == 0 && anno[i].second == MAX_LABEL) core++;
        }
        core_size_sum += static_cast<double>(core);

        // Write annotation to file
        #pragma omp critical
        {
            uint32_t d = static_cast<uint32_t>(deg);
            fout.write(reinterpret_cast<const char*>(&d), sizeof(d));
            for (size_t i = 0; i < deg; ++i) {
                fout.write(reinterpret_cast<const char*>(&anno[i]), 2 * sizeof(uint64_t));
            }
        }

        size_t done = ++nodes_processed;
        if (done % 100000 == 0) {
            spdlog::info("Processed {}/{} nodes...", done, N);
        }
    }

    spdlog::info("Annotation complete.");
    spdlog::info("  Total edges: {}", total_edges.load());
    spdlog::info("  Always survive (core): {} ({:.1f}%)", always_survive.load(),
                 100.0 * always_survive.load() / total_edges.load());
    spdlog::info("  Left-only pruned: {} ({:.1f}%)", left_only_pruned.load(),
                 100.0 * left_only_pruned.load() / total_edges.load());
    spdlog::info("  Right-only pruned: {} ({:.1f}%)", right_only_pruned.load(),
                 100.0 * right_only_pruned.load() / total_edges.load());
    spdlog::info("  Both-sides pruned: {} ({:.1f}%)", both_sides_pruned.load(),
                 100.0 * both_sides_pruned.load() / total_edges.load());
    spdlog::info("  Any-side pruned: {} ({:.1f}%)",
                 total_edges.load() - always_survive.load(),
                 100.0 * (total_edges.load() - always_survive.load()) / total_edges.load());
    spdlog::info("  Avg core size per node: {:.1f}", core_size_sum.load() / N);

    fout.close();
    spdlog::info("Saved annotations to {}", output_anno);

    // Optional report
    if (!report_json.empty()) {
        nlohmann::json js = {
            {"node_count", N},
            {"total_edges", total_edges.load()},
            {"always_survive", always_survive.load()},
            {"always_survive_pct", 100.0 * always_survive.load() / total_edges.load()},
            {"left_only_pruned", left_only_pruned.load()},
            {"right_only_pruned", right_only_pruned.load()},
            {"both_sides_pruned", both_sides_pruned.load()},
            {"any_side_pruned", total_edges.load() - always_survive.load()},
            {"any_side_pruned_pct",
             100.0 * (total_edges.load() - always_survive.load()) / total_edges.load()},
            {"avg_core_size", core_size_sum.load() / N},
        };
        std::ofstream jout(report_json);
        jout << js.dump(2);
        spdlog::info("Saved report to {}", report_json);
    }

    return 0;
}
