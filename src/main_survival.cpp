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
    CLI::App app{"rnsg_reorder: Multi-strategy graph neighbor reordering"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string output_index;
    std::string report_json;
    std::string sort_strategy = "hybrid";
    unsigned skeleton_size = 8;
    unsigned max_degree = 0;
    double alpha = 0.5;  // hybrid: weight for distance rank (1-alpha = label rank)
    double relax_factor = 1.0;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-o,--output_index", output_index, "Output graph index")->required();
    app.add_option("--report_json", report_json, "Optional JSON report for statistics");
    app.add_option("--sort_strategy", sort_strategy,
                   "Sorting strategy: dist|labelgap|hybrid|survival|hnsw_dist (default: hybrid)")
        ->check([](const std::string& s) -> std::string {
            if (s != "dist" && s != "labelgap" && s != "hybrid" && s != "survival" && s != "hnsw_dist")
                return "Must be one of: dist, labelgap, hybrid, survival, hnsw_dist";
            return {};
        });
    app.add_option("--skeleton_size", skeleton_size,
                   "Number of prefix edges to always keep as skeleton (default: 8)");
    app.add_option("--max_degree", max_degree,
                   "Maximum degree per node (0 = no limit, default: 0)");
    app.add_option("--alpha", alpha,
                   "Hybrid weight for distance rank (0=labelgap only, 1=dist only, "
                   "default: 0.5)");
    app.add_option("--relax_factor", relax_factor,
                   "Relaxation for survival pruning: dist(y,x) < factor * dist(u,x) "
                   "(default: 1.0, only for survival strategy)");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_reorder");

    if (input_index == output_index) {
        throw std::runtime_error("input_index and output_index must differ");
    }

    spdlog::info("Strategy: {}, alpha={:.2f}, skeleton={}, max_deg={}, relax={:.1f}",
                 sort_strategy, alpha, skeleton_size, max_degree, relax_factor);

    // Load dataset and labels
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);
    auto sorted_label = TDFANN::Utils::sorted_vec(label);

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

    std::vector<std::vector<TDFANN::Graph::GraphIndex<std::monostate>::Node>> reordered(
        N);

    // Statistics
    std::atomic<size_t> total_edges_orig{0};
    std::atomic<size_t> total_edges_out{0};
    std::atomic<size_t> nodes_processed{0};

    spdlog::info("Reordering graph neighbors...");

#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t ii = 0; ii < static_cast<int64_t>(N); ++ii) {
        const unsigned u = static_cast<unsigned>(ii);
        const auto& neighbors = in_graph.get_neighbours(u);
        const size_t deg = neighbors.size();

        if (deg <= 2) {
            reordered[u].reserve(deg);
            for (const auto& nb : neighbors) {
                reordered[u].push_back(nb);
            }
            total_edges_orig.fetch_add(deg);
            continue;
        }
        total_edges_orig.fetch_add(deg);

        // Step 1: Compute distances from u to all neighbors
        // NOTE: dist_all_into returns sqrs[idx] - 2*<idx,source>, MISSING sqrs[source].
        // This breaks absolute distance comparisons. Use dataset.dist() instead.
        std::vector<std::pair<float, unsigned>> edges;
        edges.reserve(deg);
        for (const auto& nb : neighbors) {
            edges.push_back({dataset.dist(u, nb.to), nb.to});
        }

        // Step 2: Compute label gaps
        const uint64_t u_label = sorted_label[u];
        std::vector<uint64_t> label_gaps(deg);
        for (size_t i = 0; i < deg; ++i) {
            uint64_t nb_label = sorted_label[edges[i].second];
            label_gaps[i] = nb_label > u_label ? nb_label - u_label : u_label - nb_label;
        }

        // Step 3: Compute sorting key based on strategy
        // Sort key: lower = higher priority (keep first)
        std::vector<double> sort_key(deg);

        if (sort_strategy == "dist") {
            // Sort by distance ascending (NSG default order)
            for (size_t i = 0; i < deg; ++i) {
                sort_key[i] = static_cast<double>(edges[i].first);
            }
        } else if (sort_strategy == "labelgap") {
            // Sort by label gap ascending (local neighbors first)
            for (size_t i = 0; i < deg; ++i) {
                sort_key[i] = static_cast<double>(label_gaps[i]);
            }
        } else if (sort_strategy == "hybrid") {
            // Rank-fused: combine distance rank and label gap rank
            // Compute distance ranks
            std::vector<unsigned> dist_order(deg);
            std::iota(dist_order.begin(), dist_order.end(), 0);
            std::ranges::stable_sort(dist_order, [&](unsigned a, unsigned b) {
                return edges[a].first < edges[b].first;
            });
            std::vector<unsigned> dist_rank(deg);
            for (size_t i = 0; i < deg; ++i) {
                dist_rank[dist_order[i]] = static_cast<unsigned>(i);
            }

            // Compute label gap ranks
            std::vector<unsigned> gap_order(deg);
            std::iota(gap_order.begin(), gap_order.end(), 0);
            std::ranges::stable_sort(gap_order, [&](unsigned a, unsigned b) {
                return label_gaps[a] < label_gaps[b];
            });
            std::vector<unsigned> gap_rank(deg);
            for (size_t i = 0; i < deg; ++i) {
                gap_rank[gap_order[i]] = static_cast<unsigned>(i);
            }

            // Fused rank
            for (size_t i = 0; i < deg; ++i) {
                sort_key[i] = alpha * static_cast<double>(dist_rank[i]) +
                              (1.0 - alpha) * static_cast<double>(gap_rank[i]);
            }
        } else if (sort_strategy == "survival") {
            // Original HNSW survival annotation with relaxed pruning
            const uint64_t MAX_LABEL =
                sorted_label.empty()
                    ? 99999
                    : *std::max_element(sorted_label.begin(), sorted_label.end());

            std::vector<uint64_t> l_drop(deg, 0);
            std::vector<uint64_t> r_drop(deg, MAX_LABEL);
            std::vector<unsigned> active_set;

            // Sort by label gap for activation order
            std::vector<unsigned> act_order(deg);
            std::iota(act_order.begin(), act_order.end(), 0);
            std::ranges::sort(act_order, [&](unsigned a, unsigned b) {
                return label_gaps[a] < label_gaps[b];
            });

            for (unsigned gi = 0; gi < deg; ++gi) {
                unsigned ei = act_order[gi];
                unsigned x_id = edges[ei].second;
                float dist_u_x = edges[ei].first;
                uint64_t x_label = sorted_label[x_id];

                bool pruned = false;
                for (unsigned ai : active_set) {
                    float dist_u_y = edges[ai].first;
                    if (dist_u_y >= dist_u_x) continue;

                    unsigned y_id = edges[ai].second;
                    float dist_y_x = dataset.dist(y_id, x_id);

                    if (dist_y_x < static_cast<float>(relax_factor) * dist_u_x) {
                        uint64_t y_label = sorted_label[y_id];
                        if (y_label < x_label) {
                            l_drop[ei] = std::max(l_drop[ei], y_label);
                        } else {
                            r_drop[ei] = std::min(r_drop[ei], y_label);
                        }
                        pruned = true;
                        break;
                    }
                }

                if (!pruned) {
                    active_set.push_back(ei);
                    std::erase_if(active_set, [&](unsigned ai2) {
                        if (ai2 == ei) return false;
                        float dist_u_y2 = edges[ai2].first;
                        if (dist_u_x >= dist_u_y2) return false;

                        unsigned y2_id = edges[ai2].second;
                        float dist_x_y2 = dataset.dist(x_id, y2_id);

                        if (dist_x_y2 < static_cast<float>(relax_factor) * dist_u_y2) {
                            uint64_t y2_label = sorted_label[y2_id];
                            if (x_label < y2_label) {
                                l_drop[ai2] = std::max(l_drop[ai2], x_label);
                            } else {
                                r_drop[ai2] = std::min(r_drop[ai2], x_label);
                            }
                            return true;
                        }
                        return false;
                    });
                }
            }

            // Sort by survival span descending (longer span = keep first)
            for (size_t i = 0; i < deg; ++i) {
                uint64_t span = r_drop[i] - l_drop[i];
                // Negate so that larger span sorts first (lower key = higher priority)
                sort_key[i] = -static_cast<double>(span);
            }
        } else if (sort_strategy == "hnsw_dist") {
            // HNSW pruning in DISTANCE order (ascending).
            // Identifies core edges (survive pruning) vs redundant edges.
            // Core edges first (keep distance order), then redundant (keep distance order).
            // For redundant edges, compute survival span via label-gap-based annotation.

            // Step 3a: Distance-order HNSW pruning
            std::vector<unsigned> dist_order(deg);
            std::iota(dist_order.begin(), dist_order.end(), 0);
            std::ranges::stable_sort(dist_order, [&](unsigned a, unsigned b) {
                return edges[a].first < edges[b].first;
            });

            std::vector<bool> is_core(deg, false);
            std::vector<std::pair<float, unsigned>> accepted;  // (dist, edge_idx)

            for (unsigned ei : dist_order) {
                float d_ux = edges[ei].first;
                unsigned x_id = edges[ei].second;
                bool pruned = false;

                for (auto& [d_uy, ai] : accepted) {
                    if (d_uy >= d_ux) continue;
                    float d_yx = dataset.dist(edges[ai].second, x_id);
                    if (d_yx < static_cast<float>(relax_factor) * d_ux) {
                        pruned = true;
                        break;
                    }
                }

                if (!pruned) {
                    is_core[ei] = true;
                    accepted.push_back({d_ux, ei});
                }
            }

            // Step 3b: Sort key — core edges first by distance, then redundant by distance
            // Use a two-tier sort key: (is_core ? 0 : 1, distance)
            for (size_t i = 0; i < deg; ++i) {
                sort_key[i] = (is_core[i] ? 0.0 : 1e15) + static_cast<double>(edges[i].first);
            }
        }

        // Step 4: Build sorted order — skeleton first, then by sort_key ascending
        std::vector<unsigned> order(deg);
        std::iota(order.begin(), order.end(), 0);
        std::ranges::stable_sort(order, [&](unsigned a, unsigned b) {
            return sort_key[a] < sort_key[b];
        });

        // Mark skeleton (first skeleton_size in ORIGINAL order)
        // Skeleton edges keep their original positions for search warmup
        std::vector<bool> is_skeleton(deg, false);
        for (size_t i = 0; i < std::min<size_t>(skeleton_size, deg); ++i) {
            is_skeleton[i] = true;
        }

        // Build output: skeleton first (original order), then rest sorted by key
        size_t out_size = deg;
        if (max_degree > 0 && out_size > max_degree) {
            out_size = max_degree;
        }
        reordered[u].reserve(out_size);

        // Add skeleton edges in original order
        for (size_t i = 0; i < deg && reordered[u].size() < out_size; ++i) {
            if (is_skeleton[i]) {
                reordered[u].push_back(to_node(edges[i].second));
            }
        }
        // Add remaining edges in sorted order (skipping skeleton)
        for (unsigned idx : order) {
            if (reordered[u].size() >= out_size) break;
            if (!is_skeleton[idx]) {
                reordered[u].push_back(to_node(edges[idx].second));
            }
        }

        total_edges_out += reordered[u].size();

        size_t done = ++nodes_processed;
        if (done % 100000 == 0) {
            spdlog::info("Processed {}/{} nodes...", done, N);
        }
    }

    spdlog::info("Reordering complete.");
    spdlog::info("  Original edges: {}", total_edges_orig.load());
    spdlog::info("  Output edges: {} ({:.1f}% reduction)", total_edges_out.load(),
                 100.0 * (1.0 - static_cast<double>(total_edges_out.load()) /
                                    total_edges_orig.load()));

    // Build output graph
    TDGraphIndexBase out_graph(N);
    out_graph.copy_headers_from(in_graph);
    for (unsigned i = 0; i < N; ++i) {
        out_graph.add_neighbours(i, reordered[i]);
    }

    // Save
    std::ofstream fout(output_index);
    if (!fout.good() || !out_graph.save(fout)) {
        throw std::runtime_error("Failed to save reordered graph to " + output_index);
    }
    spdlog::info("Saved reordered graph to {}", output_index);

    // Optional report
    if (!report_json.empty()) {
        nlohmann::json js = {
            {"node_count", N},
            {"total_edges_orig", total_edges_orig.load()},
            {"total_edges_out", total_edges_out.load()},
            {"edge_reduction_pct",
             100.0 * (1.0 - static_cast<double>(total_edges_out.load()) /
                                total_edges_orig.load())},
            {"sort_strategy", sort_strategy},
            {"alpha", alpha},
            {"skeleton_size", skeleton_size},
            {"max_degree", max_degree},
            {"relax_factor", relax_factor},
        };
        std::ofstream jout(report_json);
        jout << js.dump(2);
        spdlog::info("Saved report to {}", report_json);
    }

    return 0;
}
