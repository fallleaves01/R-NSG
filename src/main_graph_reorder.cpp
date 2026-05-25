#include <PCH.hpp>

#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <RNSG/Builder.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>

namespace {

using TDFANN::Graph::TDGraphIndexBase;

struct PrefixStats {
    double mean_gap = 0.0;
    double left_ratio = 0.0;
    double side_switch_ratio = 0.0;
    double unique_gap_levels = 0.0;
};

PrefixStats compute_prefix_stats(const std::vector<unsigned>& ids, unsigned src,
                                 unsigned k) {
    PrefixStats s;
    const size_t take = std::min<size_t>(k, ids.size());
    if (take == 0) {
        return s;
    }
    unsigned left_cnt = 0;
    unsigned switches = 0;
    int prev_side = -1;
    double gap_sum = 0.0;
    phmap::flat_hash_set<unsigned> gap_levels;
    gap_levels.reserve(take * 2 + 8);
    for (size_t i = 0; i < take; ++i) {
        const unsigned v = ids[i];
        const int side = (v < src) ? 0 : 1;
        if (side == 0) {
            left_cnt++;
        }
        if (prev_side != -1 && prev_side != side) {
            switches++;
        }
        prev_side = side;
        const auto gap = static_cast<unsigned>(
            std::abs(static_cast<long long>(v) - static_cast<long long>(src)));
        gap_sum += gap;
        unsigned lv = 0;
        while ((lv + 1) < 31U && (1U << (lv + 1)) <= gap) {
            ++lv;
        }
        gap_levels.insert(lv);
    }
    s.mean_gap = gap_sum / static_cast<double>(take);
    s.left_ratio = static_cast<double>(left_cnt) / static_cast<double>(take);
    s.side_switch_ratio =
        (take <= 1) ? 0.0 : static_cast<double>(switches) / (take - 1);
    s.unique_gap_levels =
        static_cast<double>(gap_levels.size()) / static_cast<double>(take);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"rnsg_graph_reorder"};
    std::string dataset_file;
    std::string label_file;
    std::string input_index;
    std::string output_index;
    std::string report_json;
    std::string prefix_policy = "dist";
    double prefix_mix_ratio = 0.0;
    unsigned prefix_warmup = 8;
    unsigned prefix_jump_min_gap = 0;
    double prefix_score_alpha = 0.0;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-d,--dataset_file", dataset_file, "Dataset file")->required();
    app.add_option("-l,--label_file", label_file, "Label file")->required();
    app.add_option("-i,--input_index", input_index, "Input graph index")->required();
    app.add_option("-o,--output_index", output_index, "Output graph index")->required();
    app.add_option("--report_json", report_json,
                   "Optional JSON report for prefix statistics");
    app.add_option("--prefix_policy", prefix_policy,
                   "Prefix ordering policy: dist|mix|score|scoreg|cover|balance|labelg|tierbal|balmix|bridgefirst|thinbridge|switchband")
        ->check(CLI::IsMember(
            {"dist", "mix", "score", "scoreg", "cover", "balance", "labelg", "tierbal", "balmix", "bridgefirst", "thinbridge", "switchband"}));
    app.add_option("--prefix_mix_ratio", prefix_mix_ratio,
                   "mix/cover/balance policy ratio");
    app.add_option("--prefix_warmup", prefix_warmup,
                   "Warmup count kept before gated reorder");
    app.add_option("--prefix_jump_min_gap", prefix_jump_min_gap,
                   "Minimum gap for jump edges in mix/cover");
    app.add_option("--prefix_score_alpha", prefix_score_alpha,
                   "score/scoreg label-gap bonus coefficient");
    CLI11_PARSE(app, argc, argv);
    TDFANN::Utils::setup_logger(verbose, "rnsg_graph_reorder");

    if (input_index == output_index) {
        throw std::runtime_error("input_index and output_index must differ");
    }

    TDFANN::Vector::VectorList<float> dataset(dataset_file);
    auto label = TDFANN::IO::load_json_to_vec<std::uint64_t>(label_file);
    auto [ord, pos] = TDFANN::Utils::order_of_label(label);
    (void)pos;
    dataset.reorder(ord);

    TDGraphIndexBase in_graph(input_index);
    TDGraphIndexBase out_graph(static_cast<unsigned>(in_graph.size()));
    out_graph.copy_headers_from(in_graph);

    TDFANN::RNSG::BuildOptions options;
    options.prefix_policy = prefix_policy;
    options.prefix_mix_ratio = prefix_mix_ratio;
    options.prefix_warmup = prefix_warmup;
    options.prefix_jump_min_gap = prefix_jump_min_gap;
    options.prefix_score_alpha = prefix_score_alpha;

    std::vector<std::vector<TDFANN::Graph::GraphIndex<std::monostate>::Node>>
        reordered(in_graph.size());
    std::atomic<double> gap20_sum = 0.0, left20_sum = 0.0, switch20_sum = 0.0,
                        levels20_sum = 0.0;
    std::atomic<double> gap40_sum = 0.0, left40_sum = 0.0, switch40_sum = 0.0,
                        levels40_sum = 0.0;

#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t ii = 0; ii < static_cast<int64_t>(in_graph.size()); ++ii) {
        const unsigned i = static_cast<unsigned>(ii);
        std::vector<std::pair<float, unsigned>> edges;
        const auto& row = in_graph.get_neighbours(i);
        edges.reserve(row.size());
        for (const auto& node : row) {
            edges.push_back({0.0f, node.to});
        }
        dataset.dist_all_into(i, edges);
        std::ranges::sort(edges, [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });
        TDFANN::RNSG::reorder_prefix_edges(edges, i, options);

        std::vector<unsigned> ids;
        ids.reserve(edges.size());
        auto& out = reordered[i];
        out.reserve(edges.size());
        for (const auto& [d, v] : edges) {
            (void)d;
            ids.push_back(v);
            out.push_back(TDFANN::Graph::to_node(v));
        }

        const auto s20 = compute_prefix_stats(ids, i, 20);
        const auto s40 = compute_prefix_stats(ids, i, 40);
        gap20_sum += s20.mean_gap;
        left20_sum += s20.left_ratio;
        switch20_sum += s20.side_switch_ratio;
        levels20_sum += s20.unique_gap_levels;
        gap40_sum += s40.mean_gap;
        left40_sum += s40.left_ratio;
        switch40_sum += s40.side_switch_ratio;
        levels40_sum += s40.unique_gap_levels;
    }

    for (unsigned i = 0; i < in_graph.size(); ++i) {
        out_graph.add_neighbours(i, reordered[i]);
    }

    std::ofstream fout(output_index);
    if (!fout.good() || !out_graph.save(fout)) {
        throw std::runtime_error("Failed to save reordered graph to " +
                                 output_index);
    }

    spdlog::info("Saved reordered graph to {}", output_index);

    if (!report_json.empty()) {
        const double n = static_cast<double>(in_graph.size());
        nlohmann::json js = {
            {"prefix_policy", prefix_policy},
            {"prefix_mix_ratio", prefix_mix_ratio},
            {"prefix_warmup", prefix_warmup},
            {"prefix_jump_min_gap", prefix_jump_min_gap},
            {"prefix_score_alpha", prefix_score_alpha},
            {"node_count", in_graph.size()},
            {"prefix20",
             {{"mean_gap", gap20_sum.load() / n},
              {"left_ratio", left20_sum.load() / n},
              {"side_switch_ratio", switch20_sum.load() / n},
              {"unique_gap_levels", levels20_sum.load() / n}}},
            {"prefix40",
             {{"mean_gap", gap40_sum.load() / n},
              {"left_ratio", left40_sum.load() / n},
              {"side_switch_ratio", switch40_sum.load() / n},
              {"unique_gap_levels", levels40_sum.load() / n}}},
        };
        std::ofstream jout(report_json);
        jout << js.dump(2);
    }
    return 0;
}
