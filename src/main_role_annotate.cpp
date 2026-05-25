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
//   [uint32_t magic = 0x801E0000]  // "ROLE"
//   [uint32_t version = 1]
//   [uint32_t N]
//   [uint32_t num_roles = 4]
// Per node (sequential, node 0..N-1):
//   [uint32_t deg]
//   Per edge i = 0..deg-1 (in original adjacency order):
//     [uint8_t  role]       // 0=reserve, 1=useful, 2=bridge, 3=skeleton
//     [float    label_gap]  // |rank(u) - rank(v)| / N

static constexpr uint32_t ROLE_MAGIC = 0x801E0000;
static constexpr uint32_t ROLE_VERSION = 1;
static constexpr uint32_t NUM_ROLES = 4;

// Role codes
static constexpr uint8_t ROLE_RESERVE = 0;
static constexpr uint8_t ROLE_USEFUL = 1;
static constexpr uint8_t ROLE_BRIDGE = 2;
static constexpr uint8_t ROLE_SKELETON = 3;

int main(int argc, char** argv) {
    CLI::App app{"rnsg_role_annotate: Edge role annotation for P16-R"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string output_anno;
    std::string skeleton_mode_str = "reciprocal";
    unsigned skeleton_prefix_k = 8;
    double bridge_gap_ratio = 0.3;
    double useful_gap_ratio = 0.1;
    std::string report_json;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-o,--output_anno", output_anno,
                   "Output role annotation file (binary)")
        ->required();
    app.add_option("--skeleton_mode", skeleton_mode_str,
                   "Skeleton mode: reciprocal | prefix")
        ->check(CLI::IsMember({"reciprocal", "prefix"}))
        ->default_val("reciprocal");
    app.add_option("--skeleton_prefix_k", skeleton_prefix_k,
                   "For prefix mode: first K edges are skeleton")
        ->default_val(8);
    app.add_option("--bridge_gap_ratio", bridge_gap_ratio,
                   "Label gap ratio threshold for bridge edges (> this)")
        ->default_val(0.3);
    app.add_option("--useful_gap_ratio", useful_gap_ratio,
                   "Label gap ratio threshold for useful edges (<= this)")
        ->default_val(0.1);
    app.add_option("--report_json", report_json, "Optional JSON report");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_role_annotate");

    const bool skeleton_reciprocal = (skeleton_mode_str == "reciprocal");

    // Load dataset and labels
    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);

    // Load graph
    TDGraphIndexBase in_graph(input_index);
    const unsigned N = static_cast<unsigned>(in_graph.size());
    const float N_f = static_cast<float>(N);

    spdlog::info("Graph: {} nodes, avg degree {:.1f}", N,
                 static_cast<double>([&]() {
                     size_t total = 0;
                     for (unsigned i = 0; i < N; ++i)
                         total += in_graph.get_neighbours(i).size();
                     return total;
                 }()) /
                     N);
    spdlog::info("Skeleton mode: {}", skeleton_mode_str);
    spdlog::info("Bridge gap ratio: {:.2f}, Useful gap ratio: {:.2f}",
                 bridge_gap_ratio, useful_gap_ratio);

    // Open output file
    std::ofstream fout(output_anno, std::ios::binary);
    if (!fout.good()) {
        throw std::runtime_error("Cannot open output file: " + output_anno);
    }

    // Write header
    uint32_t magic = ROLE_MAGIC;
    uint32_t version = ROLE_VERSION;
    uint32_t n_nodes = N;
    uint32_t num_roles = NUM_ROLES;
    fout.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    fout.write(reinterpret_cast<const char*>(&version), sizeof(version));
    fout.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));
    fout.write(reinterpret_cast<const char*>(&num_roles), sizeof(num_roles));

    // Statistics
    std::atomic<size_t> total_edges{0};
    std::atomic<size_t> count_skeleton{0};
    std::atomic<size_t> count_useful{0};
    std::atomic<size_t> count_bridge{0};
    std::atomic<size_t> count_reserve{0};
    std::atomic<size_t> nodes_processed{0};

    // For reciprocal check, build reverse adjacency sets
    // (which nodes have u as a neighbor)
    spdlog::info("Building reverse adjacency for reciprocal check...");
    std::vector<std::vector<uint32_t>> reverse_adj(N);
    for (unsigned u = 0; u < N; ++u) {
        const auto& nbs = in_graph.get_neighbours(u);
        for (const auto& nb : nbs) {
            if (nb.to < N) {
                reverse_adj[nb.to].push_back(u);
            }
        }
    }
    // Sort reverse adjacency for binary search
    for (unsigned v = 0; v < N; ++v) {
        std::sort(reverse_adj[v].begin(), reverse_adj[v].end());
    }
    spdlog::info("Reverse adjacency built.");

    // Per-node serialized data buffers
    std::vector<std::vector<uint8_t>> node_buffers(N);

    spdlog::info("Computing role annotations...");

#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t ii = 0; ii < static_cast<int64_t>(N); ++ii) {
        const unsigned u = static_cast<unsigned>(ii);
        const auto& neighbors = in_graph.get_neighbours(u);
        const size_t deg = neighbors.size();

        if (deg == 0) {
            std::vector<uint8_t>& buf = node_buffers[u];
            buf.resize(sizeof(uint32_t));
            uint32_t d = 0;
            std::memcpy(buf.data(), &d, sizeof(d));
            continue;
        }

        // For reciprocal check: get the set of nodes that have u as neighbor
        const auto& u_reverse = reverse_adj[u];

        // Assign role per edge
        struct RoleAnno {
            uint8_t role;
            float label_gap;
        };
        std::vector<RoleAnno> edge_roles(deg);

        for (size_t i = 0; i < deg; ++i) {
            unsigned v = neighbors[i].to;
            float gap = std::abs(static_cast<float>(u) - static_cast<float>(v)) / N_f;
            edge_roles[i].label_gap = gap;

            if (skeleton_reciprocal) {
                // Check if v has u as a neighbor (reciprocal edge)
                bool is_reciprocal = std::binary_search(
                    u_reverse.begin(), u_reverse.end(), v);

                if (is_reciprocal) {
                    edge_roles[i].role = ROLE_SKELETON;
                    count_skeleton.fetch_add(1);
                } else if (gap > bridge_gap_ratio) {
                    edge_roles[i].role = ROLE_BRIDGE;
                    count_bridge.fetch_add(1);
                } else if (gap <= useful_gap_ratio) {
                    edge_roles[i].role = ROLE_USEFUL;
                    count_useful.fetch_add(1);
                } else {
                    edge_roles[i].role = ROLE_RESERVE;
                    count_reserve.fetch_add(1);
                }
            } else {
                // Prefix mode: first K edges are skeleton
                if (i < skeleton_prefix_k) {
                    edge_roles[i].role = ROLE_SKELETON;
                    count_skeleton.fetch_add(1);
                } else if (gap > bridge_gap_ratio) {
                    edge_roles[i].role = ROLE_BRIDGE;
                    count_bridge.fetch_add(1);
                } else if (gap <= useful_gap_ratio) {
                    edge_roles[i].role = ROLE_USEFUL;
                    count_useful.fetch_add(1);
                } else {
                    edge_roles[i].role = ROLE_RESERVE;
                    count_reserve.fetch_add(1);
                }
            }

            total_edges.fetch_add(1);
        }

        // Serialize per-node data into buffer
        {
            std::vector<uint8_t>& buf = node_buffers[u];
            size_t buf_size = sizeof(uint32_t);  // deg
            buf_size += deg * (sizeof(uint8_t) + sizeof(float));  // role + label_gap per edge
            buf.resize(buf_size);
            size_t off = 0;

            uint32_t d = static_cast<uint32_t>(deg);
            std::memcpy(buf.data() + off, &d, sizeof(d));
            off += sizeof(d);

            for (size_t i = 0; i < deg; ++i) {
                std::memcpy(buf.data() + off, &edge_roles[i].role,
                           sizeof(uint8_t));
                off += sizeof(uint8_t);
                std::memcpy(buf.data() + off, &edge_roles[i].label_gap,
                           sizeof(float));
                off += sizeof(float);
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
    spdlog::info("  Skeleton (reciprocal/prefix): {} ({:.1f}%)",
                 count_skeleton.load(),
                 100.0 * count_skeleton.load() / total_edges.load());
    spdlog::info("  Useful (local, gap <= {:.2f}): {} ({:.1f}%)",
                 useful_gap_ratio, count_useful.load(),
                 100.0 * count_useful.load() / total_edges.load());
    spdlog::info("  Bridge (cross-basin, gap > {:.2f}): {} ({:.1f}%)",
                 bridge_gap_ratio, count_bridge.load(),
                 100.0 * count_bridge.load() / total_edges.load());
    spdlog::info("  Reserve (everything else): {} ({:.1f}%)",
                 count_reserve.load(),
                 100.0 * count_reserve.load() / total_edges.load());

    fout.close();
    spdlog::info("Saved role annotations to {}", output_anno);

    // Optional report
    if (!report_json.empty()) {
        nlohmann::json js = {
            {"skeleton_mode", skeleton_mode_str},
            {"bridge_gap_ratio", bridge_gap_ratio},
            {"useful_gap_ratio", useful_gap_ratio},
            {"node_count", N},
            {"total_edges", total_edges.load()},
            {"count_skeleton", count_skeleton.load()},
            {"count_skeleton_pct",
             100.0 * count_skeleton.load() / total_edges.load()},
            {"count_useful", count_useful.load()},
            {"count_useful_pct",
             100.0 * count_useful.load() / total_edges.load()},
            {"count_bridge", count_bridge.load()},
            {"count_bridge_pct",
             100.0 * count_bridge.load() / total_edges.load()},
            {"count_reserve", count_reserve.load()},
            {"count_reserve_pct",
             100.0 * count_reserve.load() / total_edges.load()},
        };
        std::ofstream jout(report_json);
        jout << js.dump(2);
        spdlog::info("Saved report to {}", report_json);
    }

    return 0;
}
