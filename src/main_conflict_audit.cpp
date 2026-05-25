#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/ExFunc.hpp>
#include <Utils/InitFunc.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

using TDFANN::Graph::TDGraphIndexBase;

// Audit tool: verify that ConflictGatedGraph::get_neighbours() produces
// the same result as a brute-force offline accepted-set replay.
//
// For each sampled (node, range) pair:
//   1. Runtime path: replicate ConflictGatedGraph::get_neighbours logic
//      using stored conflict annotations
//   2. Brute-force path: filter edges in range, compute distances,
//      sort by (dist, orig_idx), do accepted-set walk checking geometric
//      pruning condition on the fly
//   3. Compare: both must produce identical accepted sets

static constexpr uint32_t CONFLICT_MAGIC = 0xC0FE1C70;

struct EdgeAnno {
    float dist_to_u;
    std::vector<uint16_t> conflicts;  // orig_idx values
};

// Replicate runtime ConflictGatedGraph::get_neighbours logic
std::vector<unsigned> runtime_get_neighbours(
    const TDGraphIndexBase& graph,
    const std::vector<EdgeAnno>& node_anno,
    unsigned node, unsigned range_l, unsigned range_r) {
    const auto& nbs = graph.get_neighbours(node);

    // Step 1: Collect active edges in range with distances
    std::vector<std::pair<float, unsigned>> active;
    for (size_t i = 0; i < nbs.size(); ++i) {
        if (nbs[i].to >= range_l && nbs[i].to <= range_r) {
            float d = (i < node_anno.size()) ? node_anno[i].dist_to_u : 0.0f;
            active.push_back({d, static_cast<unsigned>(i)});
        }
    }

    // Step 2: Sort by (dist, orig_idx) — matches runtime std::sort on
    // pair<float,unsigned>
    std::sort(active.begin(), active.end());

    // Step 3: Accepted-set walk using stored conflicts
    std::vector<uint8_t> orig_accepted(nbs.size(), 0);
    std::vector<unsigned> result;

    for (size_t di = 0; di < active.size(); ++di) {
        unsigned orig_idx = active[di].second;
        bool pruned = false;

        if (orig_idx < node_anno.size()) {
            for (uint16_t cj : node_anno[orig_idx].conflicts) {
                if (cj < orig_accepted.size() && orig_accepted[cj]) {
                    pruned = true;
                    break;
                }
            }
        }

        if (!pruned) {
            orig_accepted[orig_idx] = 1;
            result.push_back(nbs[orig_idx].to);
        }
    }
    return result;
}

// Brute-force offline replay: compute everything from scratch
std::vector<unsigned> brute_force_replay(
    const TDFANN::Vector::VectorList<float>& dataset,
    const TDGraphIndexBase& graph,
    unsigned node, unsigned range_l, unsigned range_r) {
    const auto& nbs = graph.get_neighbours(node);
    const unsigned u = node;

    // Step 1: Collect active edges in range
    struct EdgeInfo {
        float dist_to_u;
        unsigned nb_id;
        unsigned orig_idx;
    };
    std::vector<EdgeInfo> active;
    for (size_t i = 0; i < nbs.size(); ++i) {
        if (nbs[i].to >= range_l && nbs[i].to <= range_r) {
            float d = dataset.dist(u, nbs[i].to);
            active.push_back({d, nbs[i].to, static_cast<unsigned>(i)});
        }
    }

    // Step 2: Sort by (dist_to_u, orig_idx) — same key as runtime
    std::sort(active.begin(), active.end(), [](const EdgeInfo& a, const EdgeInfo& b) {
        return std::tie(a.dist_to_u, a.orig_idx) < std::tie(b.dist_to_u, b.orig_idx);
    });

    // Step 3: Accepted-set walk with geometric pruning condition
    // HNSW pruning: edge (u,x) with dist d_ux is pruned by accepted edge
    // (u,y) with dist d_uy if d_ux > d_uy AND d_ux > dist(x,y)
    std::vector<uint8_t> accepted(active.size(), 0);
    std::vector<unsigned> result;

    for (size_t di = 0; di < active.size(); ++di) {
        const float d_ux = active[di].dist_to_u;
        const unsigned x_id = active[di].nb_id;

        bool pruned = false;
        for (size_t dj = 0; dj < di; ++dj) {
            if (!accepted[dj]) continue;
            if (active[dj].dist_to_u >= d_ux) continue;
            float d_yx = dataset.dist(active[dj].nb_id, x_id);
            if (d_yx < d_ux) {
                pruned = true;
                break;
            }
        }

        if (!pruned) {
            accepted[di] = 1;
            result.push_back(active[di].nb_id);
        }
    }
    return result;
}

int main(int argc, char** argv) {
    CLI::App app{"rnsg_conflict_audit: Verify runtime vs brute-force consistency"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string anno_file;
    int num_samples = 500;
    int seed = 42;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-a,--anno", anno_file, "Conflict annotation file")->required();
    app.add_option("-n,--num_samples", num_samples, "Number of (node,range) samples")
        ->default_val("500");
    app.add_option("--seed", seed, "Random seed")->default_val("42");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_conflict_audit");

    // Load dataset and labels
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);

    // Load graph
    TDGraphIndexBase graph(input_index);
    const unsigned N = static_cast<unsigned>(graph.size());
    spdlog::info("Graph: {} nodes", N);

    // Load annotation
    spdlog::info("Loading conflict annotations from {}...", anno_file);
    std::ifstream fin(anno_file, std::ios::binary);
    uint32_t magic, version, anno_n;
    uint8_t anno_mode;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    fin.read(reinterpret_cast<char*>(&version), sizeof(version));
    fin.read(reinterpret_cast<char*>(&anno_n), sizeof(anno_n));
    fin.read(reinterpret_cast<char*>(&anno_mode), sizeof(anno_mode));
    if (magic != CONFLICT_MAGIC) {
        spdlog::error("Invalid magic: {:08X}", magic);
        return 1;
    }
    spdlog::info("  Mode: {}", anno_mode == 2 ? "full" : (anno_mode == 1 ? "exact" : "pairwise"));

    std::vector<std::vector<EdgeAnno>> anno(anno_n);
    for (uint32_t ni = 0; ni < anno_n; ++ni) {
        uint32_t deg;
        fin.read(reinterpret_cast<char*>(&deg), sizeof(deg));
        anno[ni].resize(deg);
        for (uint32_t ei = 0; ei < deg; ++ei) {
            fin.read(reinterpret_cast<char*>(&anno[ni][ei].dist_to_u), sizeof(float));
            uint16_t cnt;
            fin.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
            anno[ni][ei].conflicts.resize(cnt);
            if (cnt > 0) {
                fin.read(reinterpret_cast<char*>(anno[ni][ei].conflicts.data()),
                         cnt * sizeof(uint16_t));
            }
        }
    }
    spdlog::info("Loaded annotations for {} nodes", anno_n);

    // Load a SECOND dataset WITHOUT reordering to check if stored
    // distances match original-order distances
    spdlog::info("Loading UNREORDERED dataset for distance verification...");
    TDFANN::Vector::VectorList<float> dataset_raw(dataset_file);

    // Quick sanity check: compare stored distances with both reordered
    // and unreordered dataset for first 10 nodes
    int reorder_match = 0, raw_match = 0;
    for (unsigned ni = 0; ni < std::min(anno_n, 10u); ++ni) {
        const auto& nbs_check = graph.get_neighbours(ni);
        for (size_t ei = 0; ei < std::min(nbs_check.size(), (size_t)5); ++ei) {
            if (ei >= anno[ni].size()) break;
            float stored = anno[ni][ei].dist_to_u;
            float d_reordered = dataset.dist(ni, nbs_check[ei].to);
            float d_raw = dataset_raw.dist(ni, nbs_check[ei].to);
            bool match_reorder = (stored == d_reordered);
            bool match_raw = (stored == d_raw);
            if (match_reorder) reorder_match++;
            if (match_raw) raw_match++;
            spdlog::info("  node={} edge={}: stored={:.6f} reordered={:.6f} raw={:.6f} "
                         "match_reorder={} match_raw={}",
                         ni, ei, stored, d_reordered, d_raw,
                         match_reorder, match_raw);
        }
    }
    spdlog::info("Distance verification: reordered matches={}, raw matches={}",
                 reorder_match, raw_match);

    // Sample random (node, range) pairs and compare
    std::mt19937 rng(seed);
    std::uniform_int_distribution<unsigned> node_dist(0, N - 1);
    // For range, pick two label values and use min/max
    const unsigned max_label = static_cast<unsigned>(label.size());

    int total_tests = 0;
    int total_matches = 0;
    int total_mismatches = 0;
    int total_nodes_with_active_edges = 0;

    spdlog::info("Sampling {} (node, range) pairs...", num_samples);

    for (int sample = 0; sample < num_samples; ++sample) {
        unsigned node = node_dist(rng);

        // Random range: pick two label positions, use [min, max]
        unsigned l1 = node_dist(rng) % max_label;
        unsigned l2 = node_dist(rng) % max_label;
        unsigned range_l = std::min(l1, l2);
        unsigned range_r = std::max(l1, l2);

        // Make sure range is not trivially empty
        if (range_r - range_l < 10) {
            range_r = std::min(range_l + 100, max_label - 1);
        }

        const auto& nbs = graph.get_neighbours(node);

        // Count active edges
        int active_count = 0;
        for (size_t i = 0; i < nbs.size(); ++i) {
            if (nbs[i].to >= range_l && nbs[i].to <= range_r) {
                active_count++;
            }
        }

        if (active_count == 0) continue;
        total_nodes_with_active_edges++;

        // Runtime path
        auto runtime_result = runtime_get_neighbours(graph, anno[node], node,
                                                       range_l, range_r);

        // Brute-force path
        auto bf_result = brute_force_replay(dataset, graph, node,
                                             range_l, range_r);

        total_tests++;

        // Compare: sort both results for comparison (order might differ)
        std::sort(runtime_result.begin(), runtime_result.end());
        std::sort(bf_result.begin(), bf_result.end());

        if (runtime_result == bf_result) {
            total_matches++;
        } else {
            total_mismatches++;
            if (verbose || total_mismatches <= 20) {
                spdlog::warn("MISMATCH at node={}, range=[{},{}]: "
                             "runtime {} edges, brute-force {} edges",
                             node, range_l, range_r,
                             runtime_result.size(), bf_result.size());

                // Show details of mismatch
                std::vector<unsigned> rt_only;
                std::set_difference(runtime_result.begin(), runtime_result.end(),
                                    bf_result.begin(), bf_result.end(),
                                    std::back_inserter(rt_only));
                std::vector<unsigned> bf_only;
                std::set_difference(bf_result.begin(), bf_result.end(),
                                    runtime_result.begin(), runtime_result.end(),
                                    std::back_inserter(bf_only));

                if (!rt_only.empty()) {
                    std::string rt_str;
                    for (size_t k = 0; k < std::min(rt_only.size(), (size_t)10); ++k) {
                        if (k > 0) rt_str += ", ";
                        rt_str += std::to_string(rt_only[k]);
                    }
                    spdlog::warn("  In runtime only ({}): {}",
                                 rt_only.size(), rt_str);
                }
                if (!bf_only.empty()) {
                    std::string bf_str;
                    for (size_t k = 0; k < std::min(bf_only.size(), (size_t)10); ++k) {
                        if (k > 0) bf_str += ", ";
                        bf_str += std::to_string(bf_only[k]);
                    }
                    spdlog::warn("  In brute-force only ({}): {}",
                                 bf_only.size(), bf_str);
                }
            }

            // Detailed trace for first 3 mismatches
            if (total_mismatches <= 3) {
                spdlog::info("=== DETAILED TRACE for node {} range [{},{}] ===",
                             node, range_l, range_r);

                // First: check if stored distances match fresh distances
                bool dist_match = true;
                for (size_t i = 0; i < nbs.size(); ++i) {
                    if (nbs[i].to < range_l || nbs[i].to > range_r) continue;
                    float stored = (i < anno[node].size()) ? anno[node][i].dist_to_u : -999.0f;
                    float fresh = dataset.dist(node, nbs[i].to);
                    if (stored != fresh) {
                        spdlog::warn("  DIST MISMATCH at orig_idx={}: stored={:.8f} fresh={:.8f} diff={:.2e}",
                                     i, stored, fresh, fresh - stored);
                        dist_match = false;
                    }
                }
                if (dist_match) {
                    spdlog::info("  All stored distances match fresh distances.");
                }

                // Now trace both algorithms side by side
                // Collect active edges with full info
                struct FullEdge {
                    float stored_dist;
                    float fresh_dist;
                    unsigned nb_id;
                    unsigned orig_idx;
                };
                std::vector<FullEdge> active;
                for (size_t i = 0; i < nbs.size(); ++i) {
                    if (nbs[i].to >= range_l && nbs[i].to <= range_r) {
                        float sd = (i < anno[node].size()) ? anno[node][i].dist_to_u : 0.0f;
                        float fd = dataset.dist(node, nbs[i].to);
                        active.push_back({sd, fd, nbs[i].to, static_cast<unsigned>(i)});
                    }
                }
                // Sort by (fresh_dist, orig_idx) - same for both
                std::sort(active.begin(), active.end(), [](const FullEdge& a, const FullEdge& b) {
                    return std::tie(a.fresh_dist, a.orig_idx) < std::tie(b.fresh_dist, b.orig_idx);
                });

                spdlog::info("  Active edges ({} total) in distance order:", active.size());
                for (size_t di = 0; di < active.size(); ++di) {
                    spdlog::info("    di={}, orig_idx={}, nb={}, stored_dist={:.8f}, fresh_dist={:.8f}",
                                 di, active[di].orig_idx, active[di].nb_id,
                                 active[di].stored_dist, active[di].fresh_dist);
                }

                // Trace brute-force step by step
                spdlog::info("  --- Brute-force trace ---");
                std::vector<uint8_t> bf_accepted(active.size(), 0);
                for (size_t di = 0; di < active.size(); ++di) {
                    const float d_ux = active[di].fresh_dist;
                    const unsigned x_id = active[di].nb_id;
                    bool pruned = false;
                    unsigned pruner_di = UINT_MAX;
                    for (size_t dj = 0; dj < di; ++dj) {
                        if (!bf_accepted[dj]) continue;
                        if (active[dj].fresh_dist >= d_ux) continue;
                        float d_yx = dataset.dist(active[dj].nb_id, x_id);
                        if (d_yx < d_ux) {
                            pruned = true;
                            pruner_di = static_cast<unsigned>(dj);
                            break;
                        }
                    }
                    bf_accepted[di] = pruned ? 0 : 1;
                    if (di < 20 || pruned) {
                        spdlog::info("    di={} orig={} nb={} dist={:.8f} {}{}",
                                     di, active[di].orig_idx, x_id, d_ux,
                                     pruned ? "PRUNED by di=" : "ACCEPTED",
                                     pruned ? std::to_string(pruner_di) : "");
                    }
                }

                // Trace runtime step by step
                spdlog::info("  --- Runtime trace ---");
                std::vector<uint8_t> rt_accepted(nbs.size(), 0);
                // Use same distance order as brute-force
                for (size_t di = 0; di < active.size(); ++di) {
                    unsigned orig_idx = active[di].orig_idx;
                    bool pruned = false;
                    unsigned pruner_cj = UINT16_MAX;

                    if (orig_idx < anno[node].size()) {
                        for (uint16_t cj : anno[node][orig_idx].conflicts) {
                            if (cj < rt_accepted.size() && rt_accepted[cj]) {
                                pruned = true;
                                pruner_cj = cj;
                                break;
                            }
                        }
                    }

                    rt_accepted[orig_idx] = pruned ? 0 : 1;
                    if (di < 20 || pruned) {
                        // Show conflict list
                        std::string conf_str;
                        if (orig_idx < anno[node].size()) {
                            const auto& conf = anno[node][orig_idx].conflicts;
                            for (size_t k = 0; k < std::min(conf.size(), (size_t)10); ++k) {
                                if (k > 0) conf_str += ", ";
                                conf_str += std::to_string(conf[k]);
                                // Show if this conflict is in range and accepted
                                if (conf[k] < nbs.size()) {
                                    bool in_range = (nbs[conf[k]].to >= range_l && nbs[conf[k]].to <= range_r);
                                    conf_str += in_range ? "[R" : "[O";
                                    conf_str += rt_accepted[conf[k]] ? "A]" : "R]";
                                }
                            }
                            if (conf.size() > 10) conf_str += "...";
                        }
                        spdlog::info("    di={} orig={} nb={} {} conf=[{}]{}",
                                     di, orig_idx, active[di].nb_id,
                                     pruned ? "PRUNED" : "ACCEPTED",
                                     conf_str,
                                     pruned ? (" by cj=" + std::to_string(pruner_cj)).c_str() : "");
                    }
                }
            }
        }
    }

    spdlog::info("=== AUDIT SUMMARY ===");
    spdlog::info("Total samples with active edges: {}", total_nodes_with_active_edges);
    spdlog::info("Total comparisons: {}", total_tests);
    spdlog::info("Matches: {}", total_matches);
    spdlog::info("Mismatches: {}", total_mismatches);
    if (total_tests > 0) {
        spdlog::info("Match rate: {:.2f}%",
                     100.0 * total_matches / total_tests);
    }

    if (total_mismatches > 0) {
        spdlog::error("AUDIT FAILED: {} mismatches found!", total_mismatches);
        return 1;
    } else {
        spdlog::info("AUDIT PASSED: All {} comparisons match.", total_tests);
        return 0;
    }
}
