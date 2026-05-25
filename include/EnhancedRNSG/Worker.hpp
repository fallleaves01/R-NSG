#pragma once

#include <PCH.hpp>

#include <omp.h>
#include <parallel_hashmap/phmap.h>
#include <EnhancedRNSG/Builder.hpp>
#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <RNSG/QueryLogCsv.hpp>
#include <RNSG/Searcher.hpp>
#include <Utils/InitFunc.hpp>
#include <Utils/Threading.hpp>
#include <Vector/VectorList.hpp>
#include <chrono>

namespace TDFANN::EnhancedRNSG {

class Worker {
   public:
    explicit Worker(CLI::App& app) {
        app.add_flag("--verbose", verbose, "Enable verbose logging");
        app.require_subcommand(1);
    }

    bool verbosed() const { return verbose; }

    auto init_knng(CLI::App& app) {
        auto knng_cmd = app.add_subcommand("knng", "Build the KNN Graph");
        knng_cmd
            ->add_option("-d,--dataset_file", dataset_file,
                         "Path to the dataset file")
            ->required();
        knng_cmd->add_option("-k", k, "K value for KNNG")->required();
        knng_cmd
            ->add_option("-g,--graph_file", knng_file,
                         "Path to the graph file to save knng")
            ->required();
        add_threads_option(knng_cmd);
        return knng_cmd;
    }

    auto init_build(CLI::App& app) {
        auto build_cmd =
            app.add_subcommand("build", "Build the TDF Graph Index");
        build_cmd
            ->add_option("-s,--range_step", range_step,
                         "Range step for building the index")
            ->required();
        build_cmd->add_option(
            "--range_window_cap", range_window_cap,
            "Sample at most this many local range-window candidates per node "
            "(0 keeps the full +-range_step window)");
        build_cmd
            ->add_option("-d,--dataset_file", dataset_file,
                         "Path to the dataset file")
            ->required();
        build_cmd
            ->add_option("-k,--knng_file", knng_file, "Path to the KNNG file")
            ->required();
        build_cmd
            ->add_option("-l,--label_file", label_file,
                         "Path to the label file")
            ->required();
        build_cmd
            ->add_option("-i,--index_file", index_file,
                         "Path to save the index")
            ->required();
        build_cmd
            ->add_option("-m,--ef_max", ef_max,
                         "Max out edges while construction")
            ->required();
        build_cmd->add_flag(
            "--disable_range_augmentation", disable_range_augmentation,
            "Disable range-window augmentation in candidate generation");
        build_cmd->add_flag("--disable_side_split_pruning",
                            disable_side_split_pruning,
                            "Disable left/right split pruning and prune one "
                            "merged candidate set");
        build_cmd->add_flag(
            "--use_mrng_pruning", use_mrng_pruning,
            "Use vanilla MRNG pruning: sort all candidates by vector distance "
            "and prune one merged candidate set");
        build_cmd
            ->add_option("--prefix_policy", prefix_policy,
                         "Prefix ordering policy: "
                         "dist|mix|score|scoreg|cover|balance|labelg|tierbal|"
                         "balmix|bridgefirst|thinbridge|switchband")
            ->check(CLI::IsMember({"dist", "mix", "score", "scoreg", "cover",
                                   "balance", "labelg", "tierbal", "balmix",
                                   "bridgefirst", "thinbridge", "switchband"}));
        build_cmd->add_option(
            "--prefix_mix_ratio", prefix_mix_ratio,
            "mix/cover policy: ratio of long-jump edges injected into early "
            "prefix");
        build_cmd->add_option(
            "--prefix_warmup", prefix_warmup,
            "mix/cover policy: keep this many nearest edges before jump "
            "injection");
        build_cmd->add_option(
            "--prefix_jump_min_gap", prefix_jump_min_gap,
            "mix/cover policy: minimum label-order gap for long-jump "
            "candidates");
        build_cmd->add_option(
            "--prefix_score_alpha", prefix_score_alpha,
            "score/scoreg policy: label-gap bonus coefficient");
        build_cmd->add_flag(
            "--disable_centroid_seed_search", disable_centroid_seed_search,
            "Disable centroid-monotone seed expansion over the KNN graph");
        build_cmd->add_flag(
            "--disable_reverse_refine", disable_reverse_refine,
            "Disable reverse-edge refine pass after first-stage build");
        build_cmd
            ->add_option(
                "--reverse_refine_mode", reverse_refine_mode,
                "Reverse refine mode: full(reopen knng+window) or "
                "incoming(only first-pass edges plus reverse candidates)")
            ->check(CLI::IsMember({"full", "incoming"}));
        build_cmd
            ->add_option("--seed_collect_mode", seed_collect_mode,
                         "Centroid-seed search collector: pq|beam")
            ->check(CLI::IsMember({"pq", "beam"}));
        build_cmd
            ->add_option("--seed_collect_policy", seed_collect_policy,
                         "Which seed-search nodes become bridge candidates: "
                         "expanded|discovered|evaluated")
            ->check(CLI::IsMember({"expanded", "discovered", "evaluated"}));
        build_cmd
            ->add_option("--monotone_seed_policy", monotone_seed_policy,
                         "Which centroid-chain seeds to use: near|far")
            ->check(CLI::IsMember({"near", "far"}));
        build_cmd->add_option(
            "--monotone_seed_limit", monotone_seed_limit,
            "Max centroid-monotone seeds collected on each side; 0 follows "
            "the full chain");
        build_cmd->add_option(
            "--seed_collect_keep", seed_collect_keep,
            "Cap collected seed-search bridge candidates per seed batch "
            "(0 keeps all candidates selected by --seed_collect_policy)");
        build_cmd->add_option(
            "--seed_collect_max_expand", seed_collect_max_expand,
            "Max beam-search expansions from each centroid seed batch "
            "(0 runs until the beam is exhausted)");
        build_cmd->add_option(
            "--seed_batch_size", seed_batch_size,
            "Batch this many consecutive centroid seeds into one shared "
            "frontier expansion");
        build_cmd->add_option(
            "--seed_search_beam_size", seed_search_beam_size,
            "Beam size for centroid-seed graph search "
            "(0 uses --seed_collect_max_expand, or 1024 when that is also 0)");
        build_cmd->add_option(
            "--seed_search_knng_cap", seed_search_knng_cap,
            "Cap KNN degree only for centroid-seed search expansion "
            "(0 keeps the full KNN row for seed search)");
        build_cmd->add_option(
            "--knng_degree_cap", knng_degree_cap,
            "Optional cap on KNNG degree used during centroid-seed search "
            "(0 keeps all loaded KNNG neighbours)");
        build_cmd
            ->add_option(
                "--candidate_merge_mode", candidate_merge_mode,
                "How to merge core and bridge candidates: legacy|quota")
            ->check(CLI::IsMember({"legacy", "quota"}));
        build_cmd->add_option(
            "--core_ratio", core_ratio,
            "Fraction of final out-edges reserved for direct/local core "
            "candidates before bridge candidates compete");
        build_cmd->add_option("--reverse_incoming_quota",
                              reverse_incoming_quota,
                              "Cap reverse incoming candidates per node before "
                              "reverse refine prune (0 keeps all)");
        build_cmd
            ->add_option("--reverse_incoming_policy", reverse_incoming_policy,
                         "How to choose reverse incoming candidates under "
                         "quota: dist|bridgegap")
            ->check(CLI::IsMember({"dist", "bridgegap"}));
        build_cmd->add_option(
            "--bridge_witness_reserve", bridge_witness_reserve,
            "Append this many bridge witness edges after prune before reorder");
        build_cmd->add_option("--support_reserve", support_reserve,
                              "Append this many high-support bridge edges "
                              "after prune before reorder");
        build_cmd
            ->add_option(
                "--support_reserve_policy", support_reserve_policy,
                "How to choose support reserve edges: support|bridgegap")
            ->check(CLI::IsMember({"support", "bridgegap"}));
        build_cmd->add_option(
            "--tail_reserve", tail_reserve,
            "Append this many range-reserve edges after the main prefix to "
            "improve high-trunc/high-recall connectivity");
        build_cmd
            ->add_option(
                "--role_select_policy", role_select_policy,
                "Build-stage role selector after relaxed prune: off|roles")
            ->check(CLI::IsMember({"off", "roles"}));
        build_cmd->add_option("--role_pool_extra", role_pool_extra,
                              "Extra candidate budget retained before "
                              "build-stage role selection");
        build_cmd->add_option(
            "--role_support_append", role_support_append,
            "Whether build-stage role selection may late-append high-support "
            "candidates into the role pool (1/0)");
        build_cmd->add_option(
            "--role_local_warmup", role_local_warmup,
            "Always keep this many nearest edges before role-based selection");
        build_cmd->add_option(
            "--role_mid_gap_min", role_mid_gap_min,
            "Minimum label gap for mid-gap candidates in role selection");
        build_cmd->add_option(
            "--role_far_gap_min", role_far_gap_min,
            "Minimum label gap for far-bridge candidates in role selection");
        build_cmd->add_option(
            "--role_mid_ratio", role_mid_ratio,
            "Target ratio of mid-gap edges in build-stage role selection");
        build_cmd->add_option(
            "--role_far_ratio", role_far_ratio,
            "Target ratio of far-bridge edges in build-stage role selection");
        build_cmd->add_option(
            "--profile_build_json", profile_build_json,
            "Write detailed build-phase profiling data to this JSON file");
        build_cmd->add_flag(
            "--allow_missing_knng_build", allow_missing_knng_build,
            "If --knng_file is missing, build a temporary in-memory KNNG");
        add_threads_option(build_cmd);
        return build_cmd;
    }

    auto init_query(CLI::App& app) {
        auto query_cmd = app.add_subcommand(
            "query", "Query nearest neighbors by TDF Graph Index");
        query_cmd->add_flag("-b,--brute", brute,
                            "Use brute-force linear search instead of index");
        query_cmd
            ->add_option("-d,--dataset_file", dataset_file,
                         "Path to the vector file")
            ->required();
        query_cmd
            ->add_option("-i,--index_file", index_file,
                         "Path to the index file")
            ->required();
        query_cmd
            ->add_option("-q,--query_file", query_file,
                         "Path to the query file")
            ->required();
        query_cmd
            ->add_option("-l,--label_file", label_file,
                         "Path to the label file")
            ->required();
        query_cmd
            ->add_option("-Q,--qrange_file", qrange_file,
                         "Path to the query range file")
            ->required();
        query_cmd
            ->add_option("-n,--qnumber", qnumber, "Number of nearest neighbors")
            ->required();
        query_cmd->add_option("-s,--beam_size", beam_size, "beam size")
            ->required();
        query_cmd
            ->add_option("-r,--result_file", result_file,
                         "Path to the result file")
            ->required();
        query_cmd->add_option("-g,--groundtruth_file", groundtruth_file,
                              "Path to the groundtruth file");
        query_cmd->add_option("-t,--trunc_size", trunc_size, "trunc size")
            ->required();
        query_cmd->add_option(
            "--entry_mode", entry_mode,
            "Compatibility option accepted for older review scripts; "
            "enhanced-RNSG uses --seed_policy instead");
        query_cmd->add_option(
            "--entry_seed", entry_seed,
            "Compatibility option accepted for older review scripts");
        query_cmd->add_option("--nav_degree", nav_degree,
                              "Navigation fanout before fallback (0 disables)");
        query_cmd->add_option("--nav_scan_factor", nav_scan_factor,
                              "Scan factor for nav neighbor selection");
        query_cmd->add_option(
            "--nav_stall_rounds", nav_stall_rounds,
            "Fallback to full trunc after this many non-improving expansions");
        query_cmd->add_option(
            "--nav_front_keep", nav_front_keep,
            "Keep this many adjacency-prefix neighbours before tail sampling");
        query_cmd->add_option(
            "--nav_tail_degree", nav_tail_degree,
            "Tail fanout cap for later beam positions (0 disables)");
        query_cmd->add_option(
            "--nav_early_stop_rounds", nav_early_stop_rounds,
            "Early stop after this many non-improving expansions (0 disables)");
        query_cmd->add_option(
            "--nav_width_split", nav_width_split,
            "Range width split for adaptive nav policy (0 disables)");
        query_cmd->add_option(
            "--nav_degree_wide", nav_degree_wide,
            "Use this nav_degree for width >= nav_width_split (0 disables)");
        query_cmd->add_option(
            "--pick_scan_factor", pick_scan_factor,
            "Scan this multiple of trunc_size before selecting neighbours "
            "(1 keeps legacy first-trunc behavior)");
        query_cmd->add_option(
            "--pick_front_keep", pick_front_keep,
            "When pick_scan_factor>1, keep this many adjacency-prefix "
            "neighbours before tail sampling (0 follows nav_front_keep)");
        query_cmd
            ->add_option(
                "--edge_pick_policy", edge_pick_policy,
                "Edge retention policy when selecting from scanned candidates: "
                "prefix|side|reciprocal|corebridge")
            ->check(
                CLI::IsMember({"prefix", "side", "reciprocal", "corebridge"}));
        query_cmd->add_option("--edge_pick_recip_depth", edge_pick_recip_depth,
                              "Max depth for reciprocal-edge check");
        query_cmd->add_option(
            "--edge_pick_core_ratio", edge_pick_core_ratio,
            "corebridge policy: ratio of core edges among retained edges");
        query_cmd->add_option(
            "--fallback_stall_rounds", fallback_stall_rounds,
            "Dynamic fallback: stall rounds threshold to trigger (0=disabled)");
        query_cmd
            ->add_option("--fallback_pick_policy", fallback_pick_policy,
                         "Dynamic fallback: pick policy on fallback "
                         "(prefix/side/reciprocal/corebridge)")
            ->check(
                CLI::IsMember({"prefix", "side", "reciprocal", "corebridge"}));
        query_cmd->add_option("--fallback_core_ratio", fallback_core_ratio,
                              "Dynamic fallback: core ratio on fallback");
        query_cmd->add_option(
            "--fallback_pick_front_keep", fallback_pick_front_keep,
            "Dynamic fallback: front_keep on fallback (0=use default)");
        query_cmd->add_option(
            "--fallback_pick_scan_factor", fallback_pick_scan_factor,
            "Dynamic fallback: scan factor on fallback (1=use default)");
        query_cmd->add_option(
            "--fallback_release_nav", fallback_release_nav,
            "Dynamic fallback: release nav degree limit on fallback");
        query_cmd->add_option("--rescue_slot_count", rescue_slot_count,
                              "Rescue slots: number of extra candidates to "
                              "inject on stall (0=disabled)");
        query_cmd->add_option("--warmup_min", warmup_min,
                              "Rescue slots: minimum expansions before rescue "
                              "can trigger (default 16)");
        query_cmd
            ->add_option(
                "--range_scan_mode", range_scan_mode,
                "Range scan mode: subgraph(filtered view) or direct(explicit "
                "range check on base graph)")
            ->check(CLI::IsMember({"subgraph", "direct"}));
        query_cmd->add_option(
            "--query_log_csv", query_log_csv,
            "Append one CSV row per query with search metrics");
        query_cmd->add_option("--log_dataset", log_dataset,
                              "dataset column in query log (metadata)");
        query_cmd->add_option("--log_method", log_method,
                              "method column in query log (metadata)");
        query_cmd->add_option("--log_bucket", log_bucket,
                              "bucket column in query log");
        query_cmd->add_option("--log_target_recall", log_target_recall,
                              "target_recall metadata; use negative to omit");
        query_cmd->add_flag("--query_log_append", query_log_append,
                            "Append to query_log_csv instead of truncating");
        query_cmd
            ->add_option("--seed_policy", seed_policy,
                         "Seed policy: header | oracle_query_seed | "
                         "cheap_query_seed | spread")
            ->check(CLI::IsMember(
                {"header", "oracle_query_seed", "cheap_query_seed", "spread"}));
        query_cmd->add_option("--oracle_seed_count", oracle_seed_count,
                              "Top-m oracle seeds for --seed_policy "
                              "oracle_query_seed (default: 4)");
        query_cmd->add_option("--cheap_query_sample_count",
                              cheap_query_sample_count,
                              "Sampled candidates for --seed_policy "
                              "cheap_query_seed (default: 64)");
        query_cmd->add_option(
            "--cheap_query_seed_count", cheap_query_seed_count,
            "Top-m seeds from cheap query samples (default: 4)");
        query_cmd->add_option(
            "--spread_anchor_count", spread_anchor_count,
            "Number of spread anchors for --seed_policy spread (default: 4)");
        query_cmd->add_flag(
            "--spread_include_header", spread_include_header,
            "Include current header seeds alongside spread anchors");
        query_cmd->add_flag(
            "--query_log_compute_seed_qrank", query_log_compute_seed_qrank,
            "Compute seed query-distance rank/ratio in query log "
            "(expensive: O(range_width) per query)");
        add_threads_option(query_cmd);
        return query_cmd;
    }

    auto init_groundtruth(CLI::App& app) {
        auto gt_cmd = app.add_subcommand(
            "groundtruth",
            "Query nearest neighbors by brute-force for groundtruth");
        gt_cmd
            ->add_option("-d,--dataset_file", dataset_file,
                         "Path to the vector file")
            ->required();
        gt_cmd
            ->add_option("-q,--query_file", query_file,
                         "Path to the query file")
            ->required();
        gt_cmd
            ->add_option("-l,--label_file", label_file,
                         "Path to the label file")
            ->required();
        gt_cmd
            ->add_option("-Q,--qrange_file", qrange_file,
                         "Path to the query range file")
            ->required();
        gt_cmd
            ->add_option("-n,--qnumber", qnumber, "Number of nearest neighbors")
            ->required();
        gt_cmd
            ->add_option("-r,--result_file", result_file,
                         "Path to the result file")
            ->required();
        add_threads_option(gt_cmd);
        return gt_cmd;
    }

    int knng() {
        apply_thread_limit();
        spdlog::info("Building KNN Graph...");
        if (std::filesystem::path(dataset_file).extension() == ".i8bin") {
            auto float_data = IO::load_i8bin_as_float_data(dataset_file);
            auto knng = RNSG::nn_descent_float_data(
                float_data.data.data(), float_data.n, float_data.dimension, k,
                verbose);
            std::ofstream fout(knng_file);
            if (!fout.good() || !knng.save(fout)) {
                spdlog::error("Failed to save graph to {}", knng_file);
                return 1;
            }
            return 0;
        }
        Vector::VectorList<float> vector_list(dataset_file);
        auto builder = Builder(vector_list);
        auto knng = builder.nn_descent(k, verbose);
        std::ofstream fout(knng_file);
        if (!fout.good() || !knng.save(fout)) {
            spdlog::error("Failed to save graph to {}", knng_file);
            return 1;
        }
        return 0;
    }

    int build() {
        apply_thread_limit();
        spdlog::info("Building EnhancedRNSG Index...");
        Vector::VectorList<float> vector_list(dataset_file);
        auto builder = Builder(vector_list);
        std::unique_ptr<Graph::GraphIndex<std::monostate>> knng_ptr;
        try {
            knng_ptr =
                std::make_unique<Graph::GraphIndex<std::monostate>>(knng_file);
        } catch (std::exception& e) {
            if (!allow_missing_knng_build) {
                spdlog::error(
                    "Failed to load KNNG from {}: {}. Build it with the knng "
                    "subcommand first, or pass --allow_missing_knng_build for "
                    "the old in-memory fallback.",
                    knng_file, e.what());
                return 2;
            }
            spdlog::warn(
                "Failed to load KNNG from {}, building new one. Error: {}, "
                "building in-memory",
                knng_file, e.what());
            knng_ptr = std::make_unique<Graph::GraphIndex<std::monostate>>(
                builder.nn_descent(50, verbose));
        }
        auto label = IO::load_json_to_vec<std::uint64_t>(label_file);
        BuildOptions options;
        options.enable_range_augmentation = !disable_range_augmentation;
        options.enable_side_split_pruning = !disable_side_split_pruning;
        options.use_mrng_pruning = use_mrng_pruning;
        options.prefix_policy = prefix_policy;
        options.prefix_mix_ratio = prefix_mix_ratio;
        options.prefix_warmup = prefix_warmup;
        options.prefix_jump_min_gap = prefix_jump_min_gap;
        options.prefix_score_alpha = prefix_score_alpha;
        options.enable_centroid_seed_search = !disable_centroid_seed_search;
        options.enable_reverse_refine = !disable_reverse_refine;
        options.reverse_refine_mode = reverse_refine_mode;
        options.seed_collect_mode = seed_collect_mode;
        options.seed_collect_policy = seed_collect_policy;
        options.monotone_seed_policy = monotone_seed_policy;
        options.monotone_seed_limit = monotone_seed_limit;
        options.seed_collect_keep = seed_collect_keep;
        options.seed_collect_max_expand = seed_collect_max_expand;
        options.seed_batch_size = seed_batch_size;
        options.seed_search_beam_size = seed_search_beam_size;
        options.seed_search_knng_cap = seed_search_knng_cap;
        options.knng_degree_cap = knng_degree_cap;
        options.candidate_merge_mode = candidate_merge_mode;
        options.core_ratio = core_ratio;
        options.reverse_incoming_quota = reverse_incoming_quota;
        options.reverse_incoming_policy = reverse_incoming_policy;
        options.bridge_witness_reserve = bridge_witness_reserve;
        options.support_reserve = support_reserve;
        options.support_reserve_policy = support_reserve_policy;
        options.tail_reserve = tail_reserve;
        options.role_select_policy = role_select_policy;
        options.role_pool_extra = role_pool_extra;
        options.role_support_append = (role_support_append != 0);
        options.role_local_warmup = role_local_warmup;
        options.role_mid_gap_min = role_mid_gap_min;
        options.role_far_gap_min = role_far_gap_min;
        options.role_mid_ratio = role_mid_ratio;
        options.role_far_ratio = role_far_ratio;
        options.range_window_cap = range_window_cap;
        options.build_profile_json = profile_build_json;
        auto g = builder.build(*knng_ptr, range_step, ef_max, label, options);
        std::ofstream fout(index_file);
        if (!fout.good() || !g.save(fout)) {
            spdlog::error("Failed to save index to {}", index_file);
            return 1;
        }
        return 0;
    }

    int query() {
        apply_thread_limit();
        spdlog::info("Querying Nearest Neighbors...");
        Vector::VectorList<float> vector_list(dataset_file);
        Graph::TDGraphIndexBase index(index_file);
        Vector::VectorList<float> query_list(query_file);
        std::ofstream fout(result_file);
        if (!fout.good()) {
            spdlog::error("Failed to open result file {}", result_file);
            return 1;
        }
        auto label = IO::load_json_to_vec<std::uint64_t>(label_file);
        auto qrange = IO::load_json_to_vec<std::uint64_t>(qrange_file);
        std::vector<unsigned> ans((size_t)query_list.size() * qnumber);

        auto [ord, pos] = Utils::order_of_label(label);
        auto& dataset = vector_list;
        auto sorted_label = Utils::sorted_vec(label);
        const int thread_count = runtime_thread_count();

        std::vector<unsigned> gt_remapped;
        if (!groundtruth_file.empty()) {
            gt_remapped = IO::load_json_to_vec(groundtruth_file);
            for (auto& i : gt_remapped) {
                i = pos[i];
            }
        }

        const bool do_query_log = !query_log_csv.empty();
        std::ofstream qlog_out;
        if (do_query_log) {
            const bool need_header = RNSG::QueryLogCsv::file_needs_header(
                query_log_csv, query_log_append);
            qlog_out.open(query_log_csv,
                          query_log_append ? std::ios::app : std::ios::out);
            if (!qlog_out.good()) {
                spdlog::error("Failed to open query log {}", query_log_csv);
                return 1;
            }
            if (need_header) {
                RNSG::QueryLogCsv::write_header(qlog_out);
            }
        }

        if (brute) {
            Timer::start("Query");
            for (size_t i = 0; i < query_list.size(); i++) {
                auto ui = static_cast<unsigned>(i);
                auto ql = qrange[ui * 2], qr = qrange[ui * 2 + 1];
                auto t0 = std::chrono::high_resolution_clock::now();
                std::priority_queue<std::pair<float, unsigned>> hp;
                auto l = ql, r = qr;
                std::uint64_t dist_comp = 0;
                for (unsigned j = 0; j < dataset.size(); j++) {
                    if (label[j] >= l && label[j] <= r) {
                        dist_comp++;
                        auto now =
                            std::pair{dataset.dist2(j, query_list[ui]), j};
                        if (hp.size() < qnumber) {
                            hp.push(now);
                        } else if (now < hp.top()) {
                            hp.pop();
                            hp.push(now);
                        }
                    }
                }
                for (size_t j = 0; j < qnumber; j++) {
                    ans[ui * qnumber + j] = pos[hp.top().second];
                    hp.pop();
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                if (do_query_log) {
                    auto range_l_it =
                        std::ranges::lower_bound(sorted_label, ql) -
                        sorted_label.begin();
                    auto range_r_it =
                        std::ranges::upper_bound(sorted_label, qr) -
                        sorted_label.begin();
                    const unsigned rw =
                        range_r_it > range_l_it
                            ? static_cast<unsigned>(range_r_it - range_l_it)
                            : 0;
                    const double sel =
                        dataset.size() > 0
                            ? static_cast<double>(rw) /
                                  static_cast<double>(dataset.size())
                            : 0.0;
                    RNSG::QueryLogCsv::Row row;
                    row.dataset = log_dataset;
                    row.method =
                        log_method.empty() ? std::string("brute") : log_method;
                    row.query_id = ui;
                    row.range_l = ql;
                    row.range_r = qr;
                    row.range_width = rw;
                    row.selectivity = sel;
                    row.k = qnumber;
                    if (log_target_recall >= 0.0) {
                        row.target_recall = log_target_recall;
                    }
                    row.latency_us =
                        std::chrono::duration<double, std::micro>(t1 - t0)
                            .count();
                    if (!gt_remapped.empty()) {
                        phmap::flat_hash_map<float, unsigned> answer_cnt;
                        for (unsigned j = 0; j < qnumber; ++j) {
                            answer_cnt[dataset.dist2(ans[ui * qnumber + j],
                                                     query_list[ui])]++;
                        }
                        unsigned lc = 0;
                        for (unsigned j = 0; j < qnumber; ++j) {
                            const float now = dataset.dist2(
                                gt_remapped[ui * qnumber + j], query_list[ui]);
                            if (answer_cnt[now] > 0) {
                                lc++;
                                answer_cnt[now]--;
                            }
                        }
                        row.recall_at_k = static_cast<double>(lc) /
                                          static_cast<double>(qnumber);
                        const unsigned nn = gt_remapped[ui * qnumber];
                        row.true_nn_id = static_cast<long long>(ord[nn]);
                        row.true_nn_dist = static_cast<double>(
                            dataset.dist2(nn, query_list[ui]));
                        const float tdist = dataset.dist2(nn, query_list[ui]);
                        row.found_true_nn = 0;
                        for (unsigned j = 0; j < qnumber; ++j) {
                            if (dataset.dist2(ans[ui * qnumber + j],
                                              query_list[ui]) == tdist) {
                                row.found_true_nn = 1;
                                break;
                            }
                        }
                    }
                    row.distance_computations = dist_comp;
                    row.visited_nodes = dist_comp;
                    row.range_scan_mode = "brute";
                    row.log_bucket = log_bucket;
                    RNSG::QueryLogCsv::write_row(qlog_out, row);
                }
                if (((i + 1) % 1024 == 0) || (i + 1 == query_list.size())) {
                    spdlog::info("Processed {}/{} queries", i + 1,
                                 query_list.size());
                }
            }
        } else {
            dataset.reorder(ord);
            RNSG::BeamScratch<float> scratch(dataset.size(), beam_size,
                                             trunc_size);
            RNSG::Searcher<float, Graph::TDGraphIndexBase> direct_searcher(
                dataset, index);
            const unsigned pick_policy_code =
                edge_pick_policy == "side"
                    ? 1u
                    : (edge_pick_policy == "reciprocal"
                           ? 2u
                           : (edge_pick_policy == "corebridge" ? 3u : 0u));
            const unsigned fb_policy_code =
                fallback_pick_policy == "side"
                    ? 1u
                    : (fallback_pick_policy == "reciprocal"
                           ? 2u
                           : (fallback_pick_policy == "corebridge" ? 3u : 0u));
            Timer::start("Query");
            RNSG::BeamSearchStats beam_stats{};
            for (size_t i = 0; i < query_list.size(); i++) {
                auto ui = static_cast<unsigned>(i);
                auto ql = qrange[ui * 2], qr = qrange[ui * 2 + 1];
                auto range_l = std::ranges::lower_bound(sorted_label, ql) -
                               sorted_label.begin();
                auto range_r = std::ranges::upper_bound(sorted_label, qr) -
                               sorted_label.begin();
                if (range_l >= range_r) {
                    continue;
                }
                const unsigned range_width =
                    static_cast<unsigned>(range_r - range_l);
                unsigned q_nav_degree = nav_degree;
                unsigned q_nav_tail_degree = nav_tail_degree;
                unsigned q_nav_early_stop_rounds = nav_early_stop_rounds;
                if (nav_width_split > 0 && nav_degree_wide > 0 &&
                    range_width >= nav_width_split) {
                    q_nav_degree = nav_degree_wide;
                    q_nav_tail_degree = 0;
                    q_nav_early_stop_rounds = 0;
                }

                auto g_sub = index(sorted_label, ql, qr);
                auto header = Utils::to_vector(g_sub.get_header());

                // --- Seed policy: build start_nodes ---
                std::vector<unsigned> start_nodes;
                double seed_search_us = -1.0;
                const unsigned range_lu = static_cast<unsigned>(range_l);
                const unsigned range_ru = static_cast<unsigned>(range_r - 1);

                if (seed_policy == "oracle_query_seed") {
                    // Diagnostic only: find top-m closest to query in range
                    auto ts0 = std::chrono::high_resolution_clock::now();
                    const unsigned m = std::min(oracle_seed_count, range_width);
                    std::vector<std::pair<float, unsigned>> dists;
                    dists.reserve(range_width);
                    for (unsigned j = range_lu;
                         j < static_cast<unsigned>(range_r); ++j) {
                        dists.push_back({dataset.dist2(j, query_list[ui]), j});
                    }
                    if (dists.size() > m) {
                        std::partial_sort(dists.begin(), dists.begin() + m,
                                          dists.end());
                        dists.resize(m);
                    }
                    start_nodes.clear();
                    for (auto& [d, id] : dists)
                        start_nodes.push_back(id);
                    auto ts1 = std::chrono::high_resolution_clock::now();
                    seed_search_us =
                        std::chrono::duration<double, std::micro>(ts1 - ts0)
                            .count();
                } else if (seed_policy == "cheap_query_seed") {
                    // Uniformly sample, then take top-m closest to query
                    auto ts0 = std::chrono::high_resolution_clock::now();
                    const unsigned sample_n =
                        std::min(cheap_query_sample_count, range_width);
                    const unsigned m =
                        std::min(cheap_query_seed_count, sample_n);
                    std::vector<unsigned> samples;
                    samples.reserve(sample_n);
                    if (sample_n >= range_width) {
                        for (unsigned j = range_lu;
                             j < static_cast<unsigned>(range_r); ++j) {
                            samples.push_back(j);
                        }
                    } else {
                        for (unsigned si = 0; si < sample_n; ++si) {
                            unsigned off = static_cast<unsigned>(
                                static_cast<unsigned long long>(si) *
                                (range_width - 1) / std::max(1u, sample_n - 1));
                            samples.push_back(range_lu + off);
                        }
                    }
                    std::vector<std::pair<float, unsigned>> dists;
                    dists.reserve(samples.size());
                    for (auto j : samples) {
                        dists.push_back({dataset.dist2(j, query_list[ui]), j});
                    }
                    if (dists.size() > m) {
                        std::partial_sort(dists.begin(), dists.begin() + m,
                                          dists.end());
                        dists.resize(m);
                    }
                    start_nodes.clear();
                    for (auto& [d, id] : dists)
                        start_nodes.push_back(id);
                    auto ts1 = std::chrono::high_resolution_clock::now();
                    seed_search_us =
                        std::chrono::duration<double, std::micro>(ts1 - ts0)
                            .count();
                } else if (seed_policy == "spread") {
                    // Uniformly spaced anchors across the label range
                    const unsigned anc_n = std::max(
                        1u, std::min(spread_anchor_count, range_width));
                    phmap::flat_hash_set<unsigned> seen;
                    start_nodes.clear();
                    for (unsigned ai = 0; ai < anc_n; ++ai) {
                        unsigned off;
                        if (anc_n == 1) {
                            off = range_width / 2;
                        } else {
                            off = static_cast<unsigned>(
                                static_cast<unsigned long long>(ai) *
                                (range_width - 1) / (anc_n - 1));
                        }
                        unsigned p = range_lu + off;
                        if (seen.insert(p).second)
                            start_nodes.push_back(p);
                    }
                    if (spread_include_header) {
                        for (auto h : header) {
                            if (seen.insert(h).second)
                                start_nodes.push_back(h);
                        }
                    }
                } else {
                    // header (default)
                    start_nodes = header;
                }

                // Fallback: empty seeds -> use range_l
                if (start_nodes.empty()) {
                    start_nodes.push_back(range_lu);
                }

                std::vector<std::pair<float, unsigned>> result;
                RNSG::BeamSearchStats* stats_ptr =
                    do_query_log ? &beam_stats : nullptr;
                auto t0 = std::chrono::high_resolution_clock::now();
                if (range_scan_mode == "direct") {
                    result = direct_searcher.beam_search_range(
                        query_list[ui], qnumber, start_nodes, beam_size,
                        trunc_size, range_lu, range_ru, scratch, q_nav_degree,
                        nav_scan_factor, nav_stall_rounds, nav_front_keep,
                        q_nav_tail_degree, q_nav_early_stop_rounds,
                        pick_scan_factor, pick_front_keep, pick_policy_code,
                        edge_pick_recip_depth, edge_pick_core_ratio,
                        fallback_stall_rounds, fb_policy_code,
                        fallback_core_ratio, fallback_pick_front_keep,
                        fallback_pick_scan_factor, fallback_release_nav,
                        nullptr, stats_ptr, rescue_slot_count, warmup_min);
                } else {
                    RNSG::Searcher<float, decltype(g_sub)> searcher(dataset,
                                                                    g_sub);
                    result = searcher.beam_search(
                        query_list[ui], qnumber, start_nodes, beam_size,
                        trunc_size, scratch, q_nav_degree, nav_scan_factor,
                        nav_stall_rounds, nav_front_keep, q_nav_tail_degree,
                        q_nav_early_stop_rounds, pick_scan_factor,
                        pick_front_keep, pick_policy_code,
                        edge_pick_recip_depth, edge_pick_core_ratio,
                        fallback_stall_rounds, fb_policy_code,
                        fallback_core_ratio, fallback_pick_front_keep,
                        fallback_pick_scan_factor, fallback_release_nav,
                        nullptr, stats_ptr, rescue_slot_count, warmup_min);
                }
                auto t1 = std::chrono::high_resolution_clock::now();

                std::ranges::copy(result | std::views::transform(GET(second)),
                                  ans.begin() + ui * qnumber);

                if (do_query_log) {
                    const double sel =
                        dataset.size() > 0
                            ? static_cast<double>(range_width) /
                                  static_cast<double>(dataset.size())
                            : 0.0;
                    std::string seed_ids;
                    for (size_t si = 0; si < start_nodes.size(); ++si) {
                        if (si > 0) {
                            seed_ids += ';';
                        }
                        seed_ids += std::to_string(start_nodes[si]);
                    }
                    unsigned best_seed = start_nodes[0];
                    float best_d = dataset.dist2(best_seed, query_list[ui]);
                    for (unsigned h : start_nodes) {
                        const float d = dataset.dist2(h, query_list[ui]);
                        if (d < best_d) {
                            best_d = d;
                            best_seed = h;
                        }
                    }
                    const long long seed_rank =
                        static_cast<long long>(best_seed) -
                        static_cast<long long>(range_l);

                    RNSG::QueryLogCsv::Row row;
                    row.dataset = log_dataset;
                    row.method = log_method.empty()
                                     ? std::string("EnhancedRNSG")
                                     : log_method;
                    row.query_id = ui;
                    row.range_l = ql;
                    row.range_r = qr;
                    row.range_width = range_width;
                    row.selectivity = sel;
                    row.k = qnumber;
                    if (log_target_recall >= 0.0) {
                        row.target_recall = log_target_recall;
                    }
                    row.latency_us =
                        std::chrono::duration<double, std::micro>(t1 - t0)
                            .count();
                    row.seed_search_us = seed_search_us;

                    // Seed query-distance metrics (expensive)
                    if (query_log_compute_seed_qrank) {
                        row.best_seed_query_dist = static_cast<double>(best_d);
                        long long better_count = 0;
                        for (unsigned j = range_lu;
                             j < static_cast<unsigned>(range_r); ++j) {
                            if (dataset.dist2(j, query_list[ui]) < best_d) {
                                better_count++;
                            }
                        }
                        row.best_seed_query_rank = better_count;
                        row.best_seed_query_rank_ratio =
                            (range_width > 0)
                                ? static_cast<double>(better_count) /
                                      range_width
                                : -1.0;
                    }
                    if (!gt_remapped.empty()) {
                        phmap::flat_hash_map<float, unsigned> answer_cnt;
                        for (unsigned j = 0; j < qnumber; ++j) {
                            answer_cnt[dataset.dist2(ans[ui * qnumber + j],
                                                     query_list[ui])]++;
                        }
                        unsigned lc = 0;
                        for (unsigned j = 0; j < qnumber; ++j) {
                            const float now = dataset.dist2(
                                gt_remapped[ui * qnumber + j], query_list[ui]);
                            if (answer_cnt[now] > 0) {
                                lc++;
                                answer_cnt[now]--;
                            }
                        }
                        row.recall_at_k = static_cast<double>(lc) /
                                          static_cast<double>(qnumber);
                        const unsigned nn = gt_remapped[ui * qnumber];
                        row.true_nn_id = static_cast<long long>(ord[nn]);
                        const float tdist = dataset.dist2(nn, query_list[ui]);
                        row.true_nn_dist = static_cast<double>(tdist);
                        row.found_true_nn = 0;
                        for (unsigned j = 0; j < qnumber; ++j) {
                            if (dataset.dist2(ans[ui * qnumber + j],
                                              query_list[ui]) == tdist) {
                                row.found_true_nn = 1;
                                break;
                            }
                        }
                        row.first_hit_hop = -1;
                        for (size_t h = 0; h < scratch.visited_nodes.size();
                             ++h) {
                            if (scratch.visited_nodes[h].first == nn) {
                                row.first_hit_hop = static_cast<long long>(h);
                                break;
                            }
                        }
                    }
                    row.distance_computations =
                        beam_stats.distance_computations;
                    row.visited_nodes = beam_stats.visited_nodes_count;
                    row.expanded_nodes = beam_stats.expanded_nodes;
                    row.raw_neighbors_scanned =
                        beam_stats.raw_neighbors_scanned;
                    row.range_filtered_out_neighbors =
                        beam_stats.range_filtered_out_neighbors;
                    row.in_range_neighbors_evaluated =
                        beam_stats.in_range_neighbors_evaluated;
                    row.beam_insertions = beam_stats.beam_insertions;
                    row.beam_rewinds = beam_stats.beam_rewinds;
                    row.early_stop_triggered = beam_stats.early_stop_triggered;
                    row.fallback_triggered = beam_stats.fallback_triggered;
                    row.fallback_trigger_step =
                        beam_stats.fallback_trigger_step;
                    row.fallback_trigger_stall_rounds =
                        beam_stats.fallback_trigger_stall_rounds;
                    row.post_trigger_dco = beam_stats.post_trigger_dco;
                    row.fallback_reason_code = beam_stats.fallback_reason_code;
                    row.rescue_candidates_added =
                        beam_stats.rescue_candidates_added;
                    row.rescue_candidates_evaluated =
                        beam_stats.rescue_candidates_evaluated;
                    row.rescue_candidates_inserted =
                        beam_stats.rescue_candidates_inserted;
                    row.seed_ids = std::move(seed_ids);
                    row.best_seed_rank_in_range = seed_rank;
                    row.range_scan_mode = range_scan_mode;
                    row.log_bucket = log_bucket;
                    RNSG::QueryLogCsv::write_row(qlog_out, row);
                }

                if (((i + 1) % 1024 == 0) || (i + 1 == query_list.size())) {
                    spdlog::info("Processed {}/{} queries", i + 1,
                                 query_list.size());
                }
            }
        }

        auto time = Timer::end("Query");
        spdlog::info("Average query time: {:.4f} ns",
                     (double)time / query_list.size());
        spdlog::info("QPS: {:.4f}", query_list.size() * 1e9 / time);

        if (!groundtruth_file.empty()) {
            unsigned correct = 0;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic) \
    reduction(+ : correct)
            for (int64_t i = 0; i < static_cast<int64_t>(query_list.size());
                 i++) {
                auto ui = static_cast<unsigned>(i);
                phmap::flat_hash_map<float, unsigned> answer_cnt;
                for (size_t j = ui * qnumber; j < (ui + 1) * qnumber; j++) {
                    answer_cnt[vector_list.dist2(ans[j], query_list[ui])]++;
                }
                unsigned local_correct = 0;
                for (size_t j = ui * qnumber; j < (ui + 1) * qnumber; j++) {
                    auto now =
                        vector_list.dist2(gt_remapped[j], query_list[ui]);
                    if (answer_cnt[now] > 0) {
                        local_correct++;
                        answer_cnt[now]--;
                    }
                }
                correct += local_correct;
            }
            spdlog::info(
                "Recall: {:.4f}",
                (double)correct / ((size_t)qnumber * query_list.size()));
        }

        for (auto& i : ans) {
            i = ord[i];
        }
        fout << "[";
        for (unsigned i = 0; i < ans.size(); i++) {
            fout << ans[i];
            if (i != ans.size() - 1) {
                fout << ",";
            }
        }
        fout << "]";
        return 0;
    }

    int gen_groundtruth() {
        apply_thread_limit();
        Vector::VectorList<float> dataset(dataset_file);
        Vector::VectorList<float> query_list(query_file);
        std::ofstream fout(result_file);
        auto label = IO::load_json_to_vec<std::uint64_t>(label_file);
        auto qrange = IO::load_json_to_vec<std::uint64_t>(qrange_file);
        std::vector<unsigned> ans((size_t)query_list.size() * qnumber);
        if (!fout.good()) {
            spdlog::error("Failed to open result file {}", result_file);
            return 1;
        }

        Timer::start("Query");
        const int thread_count = runtime_thread_count();
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
        for (int64_t i = 0; i < static_cast<int64_t>(query_list.size()); i++) {
            auto ui = static_cast<unsigned>(i);
            std::priority_queue<std::pair<float, unsigned>> hp;
            auto l = qrange[ui * 2], r = qrange[ui * 2 + 1];
            for (unsigned j = 0; j < dataset.size(); j++) {
                if (label[j] >= l && label[j] <= r) {
                    auto now = std::pair{dataset.dist2(j, query_list[ui]), j};
                    if (hp.size() < qnumber) {
                        hp.push(now);
                    } else if (now < hp.top()) {
                        hp.pop();
                        hp.push(now);
                    }
                }
            }
            for (size_t j = 0; j < qnumber; j++) {
                ans[ui * qnumber + j] = hp.top().second;
                hp.pop();
            }
            if ((ui & 1023u) == 0) {
                spdlog::info("Processed {}/{} queries", ui, query_list.size());
            }
        }

        auto time = Timer::end("Query");
        spdlog::info("Average query time: {:.4f} ns",
                     (double)time / query_list.size());
        spdlog::info("QPS: {:.4f}", query_list.size() * 1e9 / time);

        fout << "[";
        for (unsigned i = 0; i < ans.size(); i++) {
            fout << ans[i];
            if (i != ans.size() - 1) {
                fout << ",";
            }
        }
        fout << "]";
        return 0;
    }

   private:
    void add_threads_option(CLI::App* cmd) {
        cmd->add_option("--threads", threads,
                        "Maximum worker threads for this subcommand "
                        "(0 uses TDFANN_THREAD_CAP/RNSG_THREAD_CAP/default)");
    }

    void apply_thread_limit() const {
        Utils::set_thread_count_override(threads);
    }

    static int runtime_thread_count() {
        return Utils::configured_thread_count();
    }

    bool verbose = false;
    bool brute = false;
    int threads = 0;
    std::string dataset_file;
    std::string index_file;
    std::string query_file;
    std::string result_file;
    std::string groundtruth_file;
    std::string knng_file;
    std::string qrange_file;
    std::string label_file;
    unsigned k = 0;
    unsigned beam_size = 0;
    unsigned range_step = 0;
    unsigned qnumber = 0;
    unsigned trunc_size = 0;
    unsigned ef_max = 0;
    bool disable_range_augmentation = false;
    bool disable_side_split_pruning = false;
    bool use_mrng_pruning = false;
    bool disable_centroid_seed_search = false;
    bool disable_reverse_refine = false;
    bool allow_missing_knng_build = false;
    std::string reverse_refine_mode = "incoming";
    std::string seed_collect_mode = "beam";
    std::string seed_collect_policy = "discovered";
    std::string monotone_seed_policy = "far";
    std::string prefix_policy = "dist";
    double prefix_mix_ratio = 0.0;
    unsigned prefix_warmup = 8;
    unsigned prefix_jump_min_gap = 0;
    double prefix_score_alpha = 0.0;
    unsigned monotone_seed_limit = 6;
    unsigned seed_collect_keep = 0;
    unsigned seed_collect_max_expand = 32;
    unsigned seed_batch_size = 4;
    unsigned seed_search_beam_size = 256;
    unsigned seed_search_knng_cap = 64;
    unsigned knng_degree_cap = 128;
    std::string candidate_merge_mode = "legacy";
    double core_ratio = 0.6;
    unsigned reverse_incoming_quota = 0;
    std::string reverse_incoming_policy = "dist";
    unsigned bridge_witness_reserve = 0;
    unsigned support_reserve = 0;
    std::string support_reserve_policy = "support";
    unsigned tail_reserve = 0;
    std::string role_select_policy = "off";
    unsigned role_pool_extra = 0;
    unsigned role_support_append = 1;
    unsigned role_local_warmup = 8;
    unsigned role_mid_gap_min = 128;
    unsigned role_far_gap_min = 2048;
    double role_mid_ratio = 0.22;
    double role_far_ratio = 0.08;
    unsigned range_window_cap = 0;
    std::string profile_build_json;
    std::string entry_mode = "header";
    unsigned entry_seed = 42;
    unsigned nav_degree = 16;
    unsigned nav_scan_factor = 4;
    unsigned nav_stall_rounds = 8;
    unsigned nav_front_keep = 8;
    unsigned nav_tail_degree = 0;
    unsigned nav_early_stop_rounds = 0;
    unsigned nav_width_split = 0;
    unsigned nav_degree_wide = 0;
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
    std::string range_scan_mode = "subgraph";
    std::string query_log_csv;
    std::string log_dataset;
    std::string log_method;
    std::string log_bucket;
    double log_target_recall = -1.0;
    bool query_log_append = false;
    std::string seed_policy = "header";
    unsigned oracle_seed_count = 4;
    unsigned cheap_query_sample_count = 64;
    unsigned cheap_query_seed_count = 4;
    unsigned spread_anchor_count = 4;
    bool spread_include_header = false;
    bool query_log_compute_seed_qrank = false;
};

}  // namespace TDFANN::EnhancedRNSG
