#pragma once

#include <PCH.hpp>

#include <omp.h>
#include <parallel_hashmap/phmap.h>
#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <RNSG/Builder.hpp>
#include <RNSG/QueryLogCsv.hpp>
#include <RNSG/Searcher.hpp>
#include <Utils/InitFunc.hpp>
#include <Utils/Threading.hpp>
#include <Vector/VectorList.hpp>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace TDFANN::RNSG {

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
        build_cmd
            ->add_option("--prefix_policy", prefix_policy,
                         "Prefix ordering policy: "
                         "dist|mix|score|scoreg|cover|balance|labelg|tierbal|"
                         "balmix|switchband")
            ->check(CLI::IsMember({"dist", "mix", "score", "scoreg", "cover",
                                   "balance", "labelg", "tierbal", "balmix",
                                   "switchband"}));
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
            "Compatibility option accepted for review scripts; use "
            "--seed_policy for the RNSG query implementation");
        query_cmd->add_option(
            "--entry_seed", entry_seed,
            "Compatibility option accepted for review scripts");
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
            "--bridge_quota_floor", bridge_quota_floor,
            "corebridge policy: minimum bridge quota regardless of prefix (0=disabled)");
        query_cmd->add_option(
            "--suffix_promote", suffix_promote,
            "Promote N evenly-spaced edges from deep suffix into selection (0=disabled)");
        query_cmd->add_option(
            "--deep_bridge_floor", deep_bridge_floor,
            "Reserve N slots for highest-gap edges from positions > deep_bridge_pos_threshold (0=disabled)");
        query_cmd->add_option(
            "--deep_bridge_pos_threshold", deep_bridge_pos_threshold,
            "Minimum adjacency position to be considered 'deep' for deep_bridge_floor (default 32)");
        query_cmd->add_option(
            "--provenance_csv", provenance_csv,
            "Output file for per-GT-hit provenance data (empty=disabled)");
        query_cmd->add_option(
            "--post_arrival_eval_cap", post_arrival_eval_cap,
            "Max neighbors to evaluate in post-arrival expansions (0=unlimited)");
        query_cmd->add_option(
            "--post_arrival_core_cap", post_arrival_core_cap,
            "Max core-type neighbors in post-arrival (0=unlimited)");
        query_cmd->add_option(
            "--post_arrival_bridge_cap", post_arrival_bridge_cap,
            "Max bridge/deep_bridge neighbors in post-arrival (0=unlimited)");
        query_cmd->add_option(
            "--post_arrival_prefix_cap", post_arrival_prefix_cap,
            "Max prefix-type neighbors in post-arrival (0=unlimited)");
        query_cmd->add_option(
            "--eval_prov_csv", eval_prov_csv,
            "Output file for per-evaluated-neighbor provenance (P10-1, empty=disabled)");
        // P10-2: Adaptive early-stop
        query_cmd->add_option(
            "--pa_adaptive_cap", pa_adaptive_cap,
            "Post-arrival: reduce eval to this cap after N empty rounds (0=disabled)");
        query_cmd->add_option(
            "--pa_adaptive_window", pa_adaptive_window,
            "Post-arrival: number of empty rounds before adaptive cap (default 1)");
        query_cmd->add_option(
            "--pa_stall_threshold", pa_stall_threshold,
            "Post-arrival: start progressive decay after N empty rounds (0=disabled)");
        query_cmd->add_option(
            "--pa_stall_decay", pa_stall_decay,
            "Post-arrival: decay factor per additional empty round (default 2)");
        query_cmd->add_option(
            "--pa_stall_floor", pa_stall_floor,
            "Post-arrival: minimum eval count after decay (default 4)");
        query_cmd->add_option(
            "--pa_stable_threshold", pa_stable_threshold,
            "Post-arrival: apply aggressive cap when best_dist stable for N rounds (0=disabled)");
        query_cmd->add_option(
            "--pa_stable_cap", pa_stable_cap,
            "Post-arrival: aggressive eval cap when beam is stable (0=disabled)");
        // P10-3: Rank-aware tail pruning
        query_cmd->add_option(
            "--pa_rank_head_keep", pa_rank_head_keep,
            "Post-arrival: always keep first K neighbors by rank (0=disabled)");
        query_cmd->add_option(
            "--pa_rank_stale_window", pa_rank_stale_window,
            "Post-arrival: drop tail if no insertions in last N rounds (default 3)");
        // P11-2: Head-gated tail unlock
        query_cmd->add_option(
            "--pa_head_gate_keep", pa_head_gate_keep,
            "Post-arrival: evaluate first K neighbors, tail only if head inserts (0=disabled)");
        query_cmd->add_option(
            "--pa_head_gate_tail_budget", pa_head_gate_tail_budget,
            "Post-arrival: evaluate this many tail neighbors when head has no insertion (0=skip tail entirely)");
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
        query_cmd->add_option("--rescue_pick_policy", rescue_pick_policy,
                              "Rescue slots: 0=prefix (select from front), "
                              "1=label_gap_bridge (select by label gap)");
        query_cmd->add_option("--warmup_min", warmup_min,
                              "Rescue slots: minimum expansions before rescue "
                              "can trigger (default 16)");
        query_cmd->add_option(
            "--trigger_recent_window", trigger_recent_window,
            "Trigger: how many recent rounds to track insertions "
            "(0=disable composite, default 0)");
        query_cmd->add_option(
            "--trigger_flat_threshold", trigger_flat_threshold,
            "Trigger: best_dist flat rounds threshold "
            "(0=disable composite, default 0)");
        query_cmd->add_option(
            "--trigger_span_round", trigger_span_round,
            "Span trigger: check at this expansion count (0=disable, default 0)");
        query_cmd->add_option(
            "--trigger_span_norm", trigger_span_norm_threshold,
            "Span trigger: trigger if span_norm < threshold (0=disable, "
            "default 0)");
        query_cmd->add_option(
            "--diag_csv", diag_csv,
            "Write per-round diagnostic snapshots to this CSV "
            "(empty=disabled)");
        query_cmd
            ->add_option(
                "--range_scan_mode", range_scan_mode,
                "Range scan mode: subgraph(filtered view) or direct(explicit "
                "range check on base graph)")
            ->check(CLI::IsMember({"subgraph", "direct"}));
        query_cmd->add_option(
            "--survival_anno", survival_anno_file,
            "Path to survival annotation file (binary, from rnsg_annotate)");
        query_cmd->add_option(
            "--survival_skeleton", survival_skeleton,
            "Number of prefix edges to always keep as skeleton (default: 8)");
        query_cmd->add_option(
            "--conflict_anno", conflict_anno_file,
            "Path to conflict annotation file (binary, from rnsg_conflict_annotate)");
        query_cmd->add_option(
            "--role_anno", role_anno_file,
            "Path to role annotation file (binary, from rnsg_role_annotate)");
        query_cmd->add_option(
            "--skeleton_quota", skeleton_quota,
            "Max skeleton edges per expansion in role-gated mode (default: 6)");
        query_cmd->add_option(
            "--useful_quota", useful_quota,
            "Max useful edges per expansion in role-gated mode (default: 16)");
        query_cmd->add_option(
            "--bridge_quota", bridge_quota,
            "Max bridge edges per expansion in role-gated mode (default: 2)");
        query_cmd->add_option(
            "--reserve_quota", reserve_quota,
            "Max reserve edges per expansion in role-gated mode (default: 2)");
        query_cmd->add_option(
            "--role_sort_mode", role_sort_mode_str,
            "Role sort mode: gap_asc|original|floor_fill (default: gap_asc)")
            ->check(CLI::IsMember({"gap_asc", "original", "floor_fill"}))
            ->default_val("gap_asc");
        query_cmd->add_option(
            "--skeleton_floor", skeleton_floor,
            "Min skeleton edges to keep in floor_fill mode (default: 6)");
        query_cmd->add_option(
            "--bridge_floor", bridge_floor,
            "Min bridge edges to keep in floor_fill mode (default: 2)");
        query_cmd->add_option(
            "--edge_limit", edge_limit,
            "Total edge budget per expansion in floor_fill mode (default: 26)");
        query_cmd->add_flag(
            "--sort_neighbors", sort_neighbors,
            "Sort each node's neighbors by ID for binary search range access");
        query_cmd->add_flag(
            "--build_sorted_idx", build_sorted_idx,
            "Build dual-index sorted neighbor index (preserves original order)");
        query_cmd->add_flag(
            "--use_sorted_range_idx", use_sorted_range_idx,
            "Use dual-index for range-aware neighbor access (requires "
            "--build_sorted_idx)");
        query_cmd->add_flag(
            "--fast_query", fast_query,
            "Use the lightweight range-query fast path when the selected "
            "query options are compatible");
        query_cmd->add_flag(
            "--disable_fast_query", disable_fast_query,
            "Force the full diagnostic-capable query path");
        query_cmd
            ->add_option(
                "--seed_policy", seed_policy,
                "Seed policy: header | anchors | hybrid | oracle_query_seed | "
                "cheap_query_seed | spread")
            ->check(CLI::IsMember({"header", "anchors", "hybrid",
                                   "oracle_query_seed", "cheap_query_seed",
                                   "spread"}));
        query_cmd->add_option(
            "--seed_anchor_count", seed_anchor_count,
            "How many range anchors to use when seed_policy includes anchors");
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
        query_cmd->add_option(
            "--query_log_csv", query_log_csv,
            "Append one CSV row per query with search metrics (see "
            "RNSG/QueryLogCsv.hpp)");
        query_cmd->add_option("--log_dataset", log_dataset,
                              "dataset column in query log (metadata)");
        query_cmd->add_option("--log_method", log_method,
                              "method column in query log (metadata)");
        query_cmd->add_option("--log_bucket", log_bucket,
                              "bucket column in query log (e.g. width bucket)");
        query_cmd->add_option(
            "--log_target_recall", log_target_recall,
            "target_recall metadata for this run; use negative to omit");
        query_cmd->add_flag("--query_log_append", query_log_append,
                            "Append to query_log_csv instead of truncating");
        add_threads_option(query_cmd);
        return query_cmd;
    }

    auto init_query_batch(CLI::App& app) {
        auto batch_cmd = app.add_subcommand(
            "query_batch",
            "Run multiple lightweight rnsg query settings after one data/index load");
        batch_cmd
            ->add_option("-d,--dataset_file", dataset_file,
                         "Path to the vector file")
            ->required();
        batch_cmd
            ->add_option("-i,--index_file", index_file,
                         "Path to the index file")
            ->required();
        batch_cmd
            ->add_option("-q,--query_file", query_file,
                         "Path to the query file")
            ->required();
        batch_cmd
            ->add_option("-l,--label_file", label_file,
                         "Path to the label file")
            ->required();
        batch_cmd
            ->add_option("--batch_params_file", batch_params_file,
                         "CSV with run_id,topk,beam_size,trunc_size,qrange_file,"
                         "groundtruth_file,result_file")
            ->required();
        batch_cmd
            ->add_option("--batch_csv", batch_csv_file,
                         "Output CSV for one row per query setting")
            ->required();
        batch_cmd->add_option(
            "--batch_result_dir", batch_result_dir,
            "Default directory for result JSON files when result_file is empty");
        batch_cmd->add_flag("--batch_append", batch_append,
                            "Append to batch_csv instead of overwriting");
        batch_cmd->add_option(
            "--entry_seed", entry_seed,
            "Compatibility option accepted for review scripts");
        batch_cmd->add_option("--nav_degree", nav_degree,
                              "Navigation fanout before fallback (0 disables)");
        batch_cmd->add_option("--nav_scan_factor", nav_scan_factor,
                              "Scan factor for nav neighbor selection");
        batch_cmd->add_option(
            "--nav_stall_rounds", nav_stall_rounds,
            "Fallback to full trunc after this many non-improving expansions");
        batch_cmd->add_option(
            "--nav_front_keep", nav_front_keep,
            "Keep this many adjacency-prefix neighbours before tail sampling");
        batch_cmd->add_option(
            "--nav_tail_degree", nav_tail_degree,
            "Tail fanout cap for later beam positions (0 disables)");
        batch_cmd->add_option(
            "--nav_early_stop_rounds", nav_early_stop_rounds,
            "Early stop after this many non-improving expansions (0 disables)");
        batch_cmd
            ->add_option(
                "--seed_policy", seed_policy,
                "Seed policy: header | anchors | hybrid | oracle_query_seed | "
                "cheap_query_seed | spread")
            ->check(CLI::IsMember({"header", "anchors", "hybrid",
                                   "oracle_query_seed", "cheap_query_seed",
                                   "spread"}));
        batch_cmd->add_option(
            "--seed_anchor_count", seed_anchor_count,
            "How many range anchors to use when seed_policy includes anchors");
        batch_cmd->add_option("--oracle_seed_count", oracle_seed_count,
                              "Top-m oracle seeds for --seed_policy "
                              "oracle_query_seed (default: 4)");
        batch_cmd->add_option("--cheap_query_sample_count",
                              cheap_query_sample_count,
                              "Sampled candidates for --seed_policy "
                              "cheap_query_seed (default: 64)");
        batch_cmd->add_option(
            "--cheap_query_seed_count", cheap_query_seed_count,
            "Top-m seeds from cheap query samples (default: 4)");
        batch_cmd->add_option(
            "--spread_anchor_count", spread_anchor_count,
            "Number of spread anchors for --seed_policy spread (default: 4)");
        batch_cmd->add_flag(
            "--spread_include_header", spread_include_header,
            "Include current header seeds alongside spread anchors");
        batch_cmd->add_flag(
            "--build_sorted_idx", build_sorted_idx,
            "Build dual-index sorted neighbor index before batch queries");
        batch_cmd->add_flag(
            "--use_sorted_range_idx", use_sorted_range_idx,
            "Use dual-index range-aware neighbor access in batch fast path");
        add_threads_option(batch_cmd);
        return batch_cmd;
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
        spdlog::info("Building TDF Graph Index...");
        Vector::VectorList<float> vector_list(dataset_file);
        auto builder = Builder(vector_list);
        std::unique_ptr<Graph::GraphIndex<std::monostate>> knng_ptr;
        try {
            knng_ptr =
                std::make_unique<Graph::GraphIndex<std::monostate>>(knng_file);
        } catch (std::exception& e) {
            spdlog::warn(
                "Failed to load KNNG from {}, building new one. Error: {}, "
                "building",
                knng_file, e.what());
            knng_ptr = std::make_unique<Graph::GraphIndex<std::monostate>>(
                builder.nn_descent(50, verbose));
            std::ofstream fout(knng_file);
            if (!fout.good() || !knng_ptr->save(fout)) {
                spdlog::error("Failed to save graph to {}", knng_file);
                return 1;
            }
        }
        auto label = IO::load_json_to_vec<std::uint64_t>(label_file);
        auto g = builder.build(
            *knng_ptr, range_step, ef_max, label,
            BuildOptions{!disable_range_augmentation,
                         !disable_side_split_pruning, false, prefix_policy,
                         prefix_mix_ratio, prefix_warmup, prefix_jump_min_gap,
                         prefix_score_alpha});
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
        if (sort_neighbors) {
            index.sort_neighbors_by_id();
            spdlog::info("Neighbors sorted by ID for binary search range access");
        }
        if (build_sorted_idx) {
            index.build_sorted_neighbor_index();
            spdlog::info("Built sorted neighbor dual-index for range access");
        }
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
            const bool need_header =
                QueryLogCsv::file_needs_header(query_log_csv, query_log_append);
            qlog_out.open(query_log_csv,
                          query_log_append ? std::ios::app : std::ios::out);
            if (!qlog_out.good()) {
                spdlog::error("Failed to open query log {}", query_log_csv);
                return 1;
            }
            if (need_header) {
                QueryLogCsv::write_header(qlog_out);
            }
        }

        const bool do_provenance = !provenance_csv.empty();
        std::ofstream prov_out;
        if (do_provenance) {
            prov_out.open(provenance_csv, std::ios::out);
            if (!prov_out.good()) {
                spdlog::error("Failed to open provenance csv {}", provenance_csv);
                return 1;
            }
            prov_out << "query_id,gt_node_id,expansion_round,edge_category,"
                        "edge_raw_position,edge_gap,is_post_arrival,current_node\n";
        }

        std::ofstream eval_prov_out;
        if (!eval_prov_csv.empty()) {
            eval_prov_out.open(eval_prov_csv, std::ios::out);
            if (!eval_prov_out.good()) {
                spdlog::error("Failed to open eval_prov csv {}", eval_prov_csv);
                return 1;
            }
            eval_prov_out << "query_id,expansion_round,current_node,neighbor_node,"
                             "edge_category,selected_rank,prior_insertion,"
                             "was_inserted,improved_best,is_post_arrival,dist,"
                             "in_final_beam,in_final_topk,is_gt_node\n";
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
                    QueryLogCsv::Row row;
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
                    QueryLogCsv::write_row(qlog_out, row);
                }
                if (((i + 1) % 1024 == 0) || (i + 1 == query_list.size())) {
                    spdlog::info("Processed {}/{} queries", i + 1,
                                 query_list.size());
                }
            }
        } else {
            dataset.reorder(ord);
            BeamScratch<float> scratch(dataset.size(), beam_size, trunc_size);
            Searcher<float, Graph::TDGraphIndexBase> direct_searcher(dataset,
                                                                     index);

            // --- Survival annotation loading ---
            std::vector<std::vector<std::pair<uint64_t, uint64_t>>> survival_anno;
            const uint64_t SURVIVAL_MAX_LABEL =
                sorted_label.empty() ? 0 : sorted_label.back();

            // Survival Gated Graph Wrapper
            struct SurvivalGatedGraph {
                const Graph::TDGraphIndexBase& base;
                const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& anno;
                unsigned skeleton;
                uint64_t max_label;
                // Per-query state (mutable, updated via set_query)
                uint64_t ql_val = 0, qr_val = 0;
                unsigned range_l_id = 0, range_r_id = 0;

                void set_query(uint64_t ql, uint64_t qr, unsigned rl,
                               unsigned rr) {
                    ql_val = ql;
                    qr_val = qr;
                    range_l_id = rl;
                    range_r_id = rr;
                }

                auto get_neighbours(unsigned node) const {
                    const auto& nbs = base.get_neighbours(node);
                    std::vector<Graph::GraphIndex<std::monostate>::Node> filtered;
                    unsigned edge_idx = 0;
                    for (const auto& nb : nbs) {
                        // Skeleton: always keep first N edges (if in range)
                        if (edge_idx < skeleton) {
                            if (nb.to >= range_l_id &&
                                nb.to <= range_r_id) {
                                filtered.push_back(nb);
                            }
                            edge_idx++;
                            continue;
                        }
                        // Range filter
                        if (nb.to < range_l_id || nb.to > range_r_id) {
                            edge_idx++;
                            continue;
                        }
                        // Survival gating
                        if (node < anno.size() &&
                            edge_idx < anno[node].size()) {
                            auto [l_drop, r_drop] = anno[node][edge_idx];
                            if (l_drop > 0 && ql_val <= l_drop) {
                                edge_idx++;
                                continue;
                            }
                            if (r_drop < max_label && qr_val >= r_drop) {
                                edge_idx++;
                                continue;
                            }
                        }
                        filtered.push_back(nb);
                        edge_idx++;
                    }
                    return filtered;
                }

                auto get_neighbours_id(unsigned node) const {
                    auto nbs = get_neighbours(node);
                    std::vector<unsigned> ids;
                    ids.reserve(nbs.size());
                    for (const auto& x : nbs) {
                        ids.push_back(x.to);
                    }
                    return ids;
                }

                bool has_sorted_neighbor_index() const { return false; }

                auto get_neighbours_in_range_indexed(
                    unsigned node, unsigned l_id,
                    unsigned r_id) const {
                    (void)node;
                    (void)l_id;
                    (void)r_id;
                    return std::vector<std::pair<unsigned, unsigned>>{};
                }
            };

            // Conflict Gated Graph Wrapper
            struct ConflictGatedGraph {
                const Graph::TDGraphIndexBase& base;
                struct EdgeAnno {
                    float dist_to_u;
                    std::vector<uint16_t> conflicts;
                };
                const std::vector<std::vector<EdgeAnno>>& anno;
                unsigned range_l_id = 0, range_r_id = 0;

                void set_query_range(unsigned rl, unsigned rr) {
                    range_l_id = rl;
                    range_r_id = rr;
                }

                auto get_neighbours(unsigned node) const {
                    const auto& nbs = base.get_neighbours(node);
                    const auto& node_anno =
                        (node < anno.size()) ? anno[node] : get_empty_anno();

                    // Step 1: Collect active edges in range with distances
                    // active[i] = {dist_to_u, orig_idx_in_adjacency}
                    std::vector<std::pair<float, unsigned>> active;
                    for (size_t i = 0; i < nbs.size(); ++i) {
                        if (nbs[i].to >= range_l_id &&
                            nbs[i].to <= range_r_id) {
                            float d = (i < node_anno.size())
                                          ? node_anno[i].dist_to_u
                                          : 0.0f;
                            active.push_back({d, static_cast<unsigned>(i)});
                        }
                    }

                    // Step 2: Sort by distance (faithful HNSW order)
                    std::sort(active.begin(), active.end());

                    // Step 3: Build reverse map: orig_idx -> accepted?
                    // Conflict indices are in original adjacency space
                    std::vector<uint8_t> orig_accepted(nbs.size(), 0);
                    std::vector<Graph::GraphIndex<std::monostate>::Node>
                        result;

                    for (size_t di = 0; di < active.size(); ++di) {
                        unsigned orig_idx = active[di].second;
                        bool pruned = false;

                        if (orig_idx < node_anno.size()) {
                            for (uint16_t cj :
                                 node_anno[orig_idx].conflicts) {
                                // cj is an orig_idx of the pruner
                                if (cj < orig_accepted.size() &&
                                    orig_accepted[cj]) {
                                    pruned = true;
                                    break;
                                }
                            }
                        }

                        if (!pruned) {
                            orig_accepted[orig_idx] = 1;
                            result.push_back(nbs[orig_idx]);
                        }
                    }
                    return result;
                }

                auto get_neighbours_id(unsigned node) const {
                    auto nbs = get_neighbours(node);
                    std::vector<unsigned> ids;
                    ids.reserve(nbs.size());
                    for (const auto& x : nbs) ids.push_back(x.to);
                    return ids;
                }

                bool has_sorted_neighbor_index() const { return false; }
                auto get_neighbours_in_range_indexed(
                    unsigned, unsigned, unsigned) const {
                    return std::vector<std::pair<unsigned, unsigned>>{};
                }

               private:
                static const std::vector<EdgeAnno>& get_empty_anno() {
                    static const std::vector<EdgeAnno> empty;
                    return empty;
                }
            };

            // --- Role-gated graph (P16-R Phase 0.5) ---
            // sort_mode: 0=gap_asc (Phase 0), 1=original, 2=floor_fill
            struct RoleGatedGraph {
                const Graph::TDGraphIndexBase& base;
                struct EdgeRoleAnno {
                    uint8_t role;      // 0=reserve, 1=useful, 2=bridge, 3=skeleton
                    float label_gap;   // |rank(u) - rank(v)| / N
                };
                const std::vector<std::vector<EdgeRoleAnno>>& anno;
                unsigned range_l_id = 0, range_r_id = 0;
                unsigned skeleton_quota = 6;
                unsigned useful_quota = 16;
                unsigned bridge_quota = 2;
                unsigned reserve_quota = 2;
                // Phase 0.5 additions
                unsigned sort_mode = 0;  // 0=gap_asc, 1=original, 2=floor_fill
                unsigned skeleton_floor = 6;
                unsigned bridge_floor = 2;
                unsigned edge_limit = 26;

                // Quota utilization diagnostics
                mutable uint64_t stat_expansions = 0;
                mutable uint64_t stat_total_inrange = 0;
                mutable uint64_t stat_skel_avail = 0, stat_skel_used = 0;
                mutable uint64_t stat_useful_avail = 0, stat_useful_used = 0;
                mutable uint64_t stat_bridge_avail = 0, stat_bridge_used = 0;
                mutable uint64_t stat_reserve_avail = 0, stat_reserve_used = 0;
                mutable uint64_t stat_total_returned = 0;

                void set_query_range(unsigned rl, unsigned rr) {
                    range_l_id = rl;
                    range_r_id = rr;
                }

                void print_stats() const {
                    if (stat_expansions == 0) return;
                    auto avg = [](uint64_t v, uint64_t n) -> double {
                        return n > 0 ? static_cast<double>(v) / n : 0.0;
                    };
                    spdlog::info("RoleGatedGraph stats (mode={}):",
                                 sort_mode == 0 ? "gap_asc" :
                                 sort_mode == 1 ? "original" : "floor_fill");
                    spdlog::info("  Expansions: {}", stat_expansions);
                    spdlog::info("  Avg in-range edges: {:.1f}",
                                 avg(stat_total_inrange, stat_expansions));
                    spdlog::info("  Skeleton: avg avail {:.1f}, used {:.1f}",
                                 avg(stat_skel_avail, stat_expansions),
                                 avg(stat_skel_used, stat_expansions));
                    spdlog::info("  Useful:   avg avail {:.1f}, used {:.1f}",
                                 avg(stat_useful_avail, stat_expansions),
                                 avg(stat_useful_used, stat_expansions));
                    spdlog::info("  Bridge:   avg avail {:.1f}, used {:.1f}",
                                 avg(stat_bridge_avail, stat_expansions),
                                 avg(stat_bridge_used, stat_expansions));
                    spdlog::info("  Reserve:  avg avail {:.1f}, used {:.1f}",
                                 avg(stat_reserve_avail, stat_expansions),
                                 avg(stat_reserve_used, stat_expansions));
                    spdlog::info("  Avg returned: {:.1f}",
                                 avg(stat_total_returned, stat_expansions));
                }

                auto get_neighbours(unsigned node) const {
                    const auto& nbs = base.get_neighbours(node);
                    const auto& node_anno =
                        (node < anno.size()) ? anno[node] : get_empty_anno();

                    // Collect in-range edges grouped by role.
                    // Each entry: (orig_idx, label_gap) — idx first for
                    // natural ordering when sort_mode != gap_asc.
                    struct Entry {
                        unsigned idx;
                        float gap;
                    };
                    std::vector<Entry> skel, useful, brid, resv;

                    for (size_t i = 0; i < nbs.size(); ++i) {
                        if (nbs[i].to >= range_l_id &&
                            nbs[i].to <= range_r_id) {
                            uint8_t role = (i < node_anno.size())
                                               ? node_anno[i].role
                                               : 0;
                            float gap = (i < node_anno.size())
                                            ? node_anno[i].label_gap
                                            : 1.0f;
                            Entry e{static_cast<unsigned>(i), gap};
                            switch (role) {
                                case 3: skel.push_back(e); break;
                                case 1: useful.push_back(e); break;
                                case 2: brid.push_back(e); break;
                                default: resv.push_back(e); break;
                            }
                        }
                    }

                    // Update stats
                    stat_expansions++;
                    stat_total_inrange += skel.size() + useful.size()
                                         + brid.size() + resv.size();
                    stat_skel_avail += skel.size();
                    stat_useful_avail += useful.size();
                    stat_bridge_avail += brid.size();
                    stat_reserve_avail += resv.size();

                    std::vector<unsigned> selected;

                    if (sort_mode == 2) {
                        // === floor_fill mode ===
                        // 1. Take skeleton_floor from skeleton (original order)
                        // 2. Take bridge_floor from bridge (original order)
                        // 3. Collect ALL remaining edges into pool
                        // 4. Fill up to edge_limit by original index order

                        // skeleton entries are already in original idx order
                        unsigned skel_take = std::min(
                            static_cast<unsigned>(skel.size()),
                            skeleton_floor);
                        for (unsigned si = 0; si < skel_take; ++si)
                            selected.push_back(skel[si].idx);

                        unsigned brid_take = std::min(
                            static_cast<unsigned>(brid.size()),
                            bridge_floor);
                        for (unsigned bi = 0; bi < brid_take; ++bi)
                            selected.push_back(brid[bi].idx);

                        // Pool remaining: all useful + reserve + unused skel/brid
                        std::vector<Entry> pool;
                        for (unsigned si = skel_take; si < skel.size(); ++si)
                            pool.push_back(skel[si]);
                        for (auto& e : useful) pool.push_back(e);
                        for (unsigned bi = brid_take; bi < brid.size(); ++bi)
                            pool.push_back(brid[bi]);
                        for (auto& e : resv) pool.push_back(e);

                        // Sort pool by original index (utility order)
                        std::sort(pool.begin(), pool.end(),
                                  [](const Entry& a, const Entry& b) {
                                      return a.idx < b.idx;
                                  });

                        // Fill remaining budget
                        unsigned remaining = (edge_limit > selected.size())
                            ? edge_limit - static_cast<unsigned>(selected.size())
                            : 0;
                        for (unsigned pi = 0;
                             pi < pool.size() && pi < remaining; ++pi)
                            selected.push_back(pool[pi].idx);

                        // Sort final selection by original index
                        std::sort(selected.begin(), selected.end());

                        stat_skel_used += skel_take;
                        stat_bridge_used += brid_take;
                        // Count useful/reserve from pool
                        for (auto& e : pool) {
                            if (std::binary_search(selected.begin(),
                                                   selected.end(), e.idx)) {
                                uint8_t role = (e.idx < node_anno.size())
                                    ? node_anno[e.idx].role : 0;
                                if (role == 1) stat_useful_used++;
                                else if (role == 0) stat_reserve_used++;
                                else if (role == 3)
                                    stat_skel_used++;  // unused skel in pool
                                else if (role == 2)
                                    stat_bridge_used++;  // unused brid in pool
                            }
                        }
                    } else {
                        // === gap_asc (mode 0) or original (mode 1) ===
                        auto apply_quota = [this](auto& vec,
                                                  unsigned quota) {
                            if (sort_mode == 0) {
                                // Sort by label_gap ascending
                                std::sort(vec.begin(), vec.end(),
                                    [](const Entry& a, const Entry& b) {
                                        return a.gap < b.gap;
                                    });
                            }
                            // mode 1: already in original index order
                            if (vec.size() > quota) vec.resize(quota);
                        };
                        apply_quota(skel, skeleton_quota);
                        apply_quota(useful, useful_quota);
                        apply_quota(brid, bridge_quota);
                        apply_quota(resv, reserve_quota);

                        // Collect selected indices
                        for (auto& e : skel) selected.push_back(e.idx);
                        for (auto& e : useful) selected.push_back(e.idx);
                        for (auto& e : brid) selected.push_back(e.idx);
                        for (auto& e : resv) selected.push_back(e.idx);
                        std::sort(selected.begin(), selected.end());

                        stat_skel_used += skel.size();
                        stat_useful_used += useful.size();
                        stat_bridge_used += brid.size();
                        stat_reserve_used += resv.size();
                    }

                    stat_total_returned += selected.size();

                    // Build result vector
                    std::vector<Graph::GraphIndex<std::monostate>::Node>
                        result;
                    result.reserve(selected.size());
                    for (unsigned idx : selected) result.push_back(nbs[idx]);
                    return result;
                }

                auto get_neighbours_id(unsigned node) const {
                    const auto& nbs = get_neighbours(node);
                    std::vector<unsigned> ids;
                    ids.reserve(nbs.size());
                    for (const auto& x : nbs) ids.push_back(x.to);
                    return ids;
                }

                bool has_sorted_neighbor_index() const { return false; }

                auto get_neighbours_in_range_indexed(
                    unsigned, unsigned, unsigned) const {
                    return std::vector<std::pair<unsigned, unsigned>>{};
                }

               private:
                static const std::vector<EdgeRoleAnno>& get_empty_anno() {
                    static const std::vector<EdgeRoleAnno> empty;
                    return empty;
                }
            };

            // Load survival annotation if specified
            std::unique_ptr<SurvivalGatedGraph> sgg_ptr;
            std::unique_ptr<Searcher<float, SurvivalGatedGraph>> sgg_searcher_ptr;

            if (!survival_anno_file.empty()) {
                spdlog::info("Loading survival annotations from {}...",
                             survival_anno_file);
                std::ifstream fin(survival_anno_file, std::ios::binary);
                if (!fin.good()) {
                    spdlog::error("Cannot open survival annotation file: {}",
                                  survival_anno_file);
                    return 1;
                }
                uint32_t anno_n;
                fin.read(reinterpret_cast<char*>(&anno_n), sizeof(anno_n));
                if (anno_n != static_cast<uint32_t>(dataset.size())) {
                    spdlog::error("Annotation node count {} != dataset size {}",
                                  anno_n, dataset.size());
                    return 1;
                }
                survival_anno.resize(anno_n);
                for (uint32_t ni = 0; ni < anno_n; ++ni) {
                    uint32_t deg;
                    fin.read(reinterpret_cast<char*>(&deg), sizeof(deg));
                    survival_anno[ni].resize(deg);
                    fin.read(reinterpret_cast<char*>(survival_anno[ni].data()),
                             deg * 2 * sizeof(uint64_t));
                }
                spdlog::info("Loaded survival annotations for {} nodes",
                             anno_n);

                sgg_ptr = std::make_unique<SurvivalGatedGraph>(
                    SurvivalGatedGraph{index, survival_anno,
                                       survival_skeleton,
                                       SURVIVAL_MAX_LABEL});
                sgg_searcher_ptr =
                    std::make_unique<Searcher<float, SurvivalGatedGraph>>(
                        dataset, *sgg_ptr);
            }

            // --- Conflict annotation loading ---
            std::vector<std::vector<ConflictGatedGraph::EdgeAnno>>
                conflict_anno;
            std::unique_ptr<ConflictGatedGraph> cgg_ptr;
            std::unique_ptr<Searcher<float, ConflictGatedGraph>>
                cgg_searcher_ptr;

            if (!conflict_anno_file.empty()) {
                spdlog::info("Loading conflict annotations from {}...",
                             conflict_anno_file);
                std::ifstream fin(conflict_anno_file, std::ios::binary);
                if (!fin.good()) {
                    spdlog::error(
                        "Cannot open conflict annotation file: {}",
                        conflict_anno_file);
                    return 1;
                }
                // Read header
                uint32_t magic, version, anno_n;
                uint8_t anno_mode;
                fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                fin.read(reinterpret_cast<char*>(&version),
                         sizeof(version));
                fin.read(reinterpret_cast<char*>(&anno_n), sizeof(anno_n));
                fin.read(reinterpret_cast<char*>(&anno_mode),
                         sizeof(anno_mode));
                if (magic != 0xC0FE1C70) {
                    spdlog::error(
                        "Invalid conflict annotation magic: {:08X} "
                        "(expected C0FE1C70)",
                        magic);
                    return 1;
                }
                if (version != 1) {
                    spdlog::error(
                        "Unsupported conflict annotation version: {}",
                        version);
                    return 1;
                }
                if (anno_n != static_cast<uint32_t>(dataset.size())) {
                    spdlog::error(
                        "Conflict annotation node count {} != dataset "
                        "size {}",
                        anno_n, dataset.size());
                    return 1;
                }
                spdlog::info("  Mode: {}",
                             anno_mode == 2 ? "full"
                                            : (anno_mode == 1 ? "exact"
                                                              : "pairwise"));

                conflict_anno.resize(anno_n);
                for (uint32_t ni = 0; ni < anno_n; ++ni) {
                    uint32_t deg;
                    fin.read(reinterpret_cast<char*>(&deg), sizeof(deg));
                    conflict_anno[ni].resize(deg);
                    for (uint32_t ei = 0; ei < deg; ++ei) {
                        fin.read(
                            reinterpret_cast<char*>(
                                &conflict_anno[ni][ei].dist_to_u),
                            sizeof(float));
                        uint16_t cnt;
                        fin.read(reinterpret_cast<char*>(&cnt),
                                 sizeof(cnt));
                        conflict_anno[ni][ei].conflicts.resize(cnt);
                        if (cnt > 0) {
                            fin.read(reinterpret_cast<char*>(
                                         conflict_anno[ni][ei]
                                             .conflicts.data()),
                                     cnt * sizeof(uint16_t));
                        }
                    }
                }
                spdlog::info("Loaded conflict annotations for {} nodes",
                             anno_n);

                cgg_ptr = std::make_unique<ConflictGatedGraph>(
                    ConflictGatedGraph{index, conflict_anno});
                cgg_searcher_ptr =
                    std::make_unique<
                        Searcher<float, ConflictGatedGraph>>(dataset,
                                                             *cgg_ptr);
            }

            // --- Role annotation loading (P16-R) ---
            std::vector<std::vector<RoleGatedGraph::EdgeRoleAnno>> role_anno;
            std::unique_ptr<RoleGatedGraph> rgg_ptr;
            std::unique_ptr<Searcher<float, RoleGatedGraph>> rgg_searcher_ptr;

            if (!role_anno_file.empty()) {
                spdlog::info("Loading role annotations from {}...",
                             role_anno_file);
                std::ifstream fin(role_anno_file, std::ios::binary);
                if (!fin.good()) {
                    spdlog::error(
                        "Cannot open role annotation file: {}",
                        role_anno_file);
                    return 1;
                }
                // Read header
                uint32_t r_magic, r_version, r_n, r_num_roles;
                fin.read(reinterpret_cast<char*>(&r_magic),
                         sizeof(r_magic));
                fin.read(reinterpret_cast<char*>(&r_version),
                         sizeof(r_version));
                fin.read(reinterpret_cast<char*>(&r_n), sizeof(r_n));
                fin.read(reinterpret_cast<char*>(&r_num_roles),
                         sizeof(r_num_roles));
                if (r_magic != 0x801E0000) {
                    spdlog::error(
                        "Invalid role annotation magic: {:08X} "
                        "(expected 801E0000)",
                        r_magic);
                    return 1;
                }
                if (r_version != 1) {
                    spdlog::error(
                        "Unsupported role annotation version: {}",
                        r_version);
                    return 1;
                }
                if (r_n != static_cast<uint32_t>(dataset.size())) {
                    spdlog::error(
                        "Role annotation node count {} != dataset "
                        "size {}",
                        r_n, dataset.size());
                    return 1;
                }

                role_anno.resize(r_n);
                for (uint32_t ni = 0; ni < r_n; ++ni) {
                    uint32_t deg;
                    fin.read(reinterpret_cast<char*>(&deg), sizeof(deg));
                    role_anno[ni].resize(deg);
                    for (uint32_t ei = 0; ei < deg; ++ei) {
                        fin.read(reinterpret_cast<char*>(
                                     &role_anno[ni][ei].role),
                                 sizeof(uint8_t));
                        fin.read(reinterpret_cast<char*>(
                                     &role_anno[ni][ei].label_gap),
                                 sizeof(float));
                    }
                }
                spdlog::info("Loaded role annotations for {} nodes", r_n);

                const unsigned sort_mode_code =
                    role_sort_mode_str == "gap_asc" ? 0u :
                    (role_sort_mode_str == "original" ? 1u : 2u);
                spdlog::info("Role sort mode: {} (code={})",
                             role_sort_mode_str, sort_mode_code);

                rgg_ptr = std::make_unique<RoleGatedGraph>(
                    RoleGatedGraph{index, role_anno, 0, 0,
                                   skeleton_quota, useful_quota,
                                   bridge_quota, reserve_quota,
                                   sort_mode_code,
                                   skeleton_floor, bridge_floor,
                                   edge_limit});
                rgg_searcher_ptr =
                    std::make_unique<Searcher<float, RoleGatedGraph>>(
                        dataset, *rgg_ptr);
            }

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
            const bool can_use_fast_query =
                fast_query && !disable_fast_query && !do_query_log &&
                provenance_csv.empty() && eval_prov_csv.empty() &&
                diag_csv.empty() && survival_anno_file.empty() &&
                conflict_anno_file.empty() && role_anno_file.empty() &&
                !sort_neighbors && !build_sorted_idx && !use_sorted_range_idx &&
                pick_scan_factor == 1 && pick_front_keep == 0 &&
                edge_pick_policy == "prefix" && bridge_quota_floor == 0 &&
                suffix_promote == 0 && deep_bridge_floor == 0 &&
                post_arrival_eval_cap == 0 && post_arrival_core_cap == 0 &&
                post_arrival_bridge_cap == 0 && post_arrival_prefix_cap == 0 &&
                pa_adaptive_cap == 0 && pa_stall_threshold == 0 &&
                pa_stable_threshold == 0 && pa_rank_head_keep == 0 &&
                pa_head_gate_keep == 0 && fallback_stall_rounds == 0 &&
                rescue_slot_count == 0 && trigger_recent_window == 0 &&
                trigger_flat_threshold == 0 && trigger_span_round == 0 &&
                trigger_span_norm_threshold == 0.0;
            if (can_use_fast_query) {
                spdlog::info(
                    "Using lightweight RNSG range-query fast path "
                    "(direct range scan, no diagnostics)");
            } else if (fast_query && !disable_fast_query) {
                spdlog::info(
                    "Lightweight query fast path disabled by non-compatible "
                    "query options");
            }
            Timer::start("Query");
            BeamSearchStats beam_stats{};
            BeamSearchDiag diag_data;
            BeamSearchDiag* diag_ptr = diag_csv.empty() ? nullptr : &diag_data;
            std::ofstream diag_out;
            if (diag_ptr) {
                diag_out.open(diag_csv, std::ios::out);
                if (diag_out.good()) {
                    diag_out << "query_id,round,expanded_node,insertions,"
                             << "cumulative_insertions,best_dist,best_dist_ever,"
                             << "stall_rounds,beam_label_span,"
                             << "beam_size_effective,beam_bridge_count,"
                             << "beam_core_count,"
                             << "raw_count,selected_count,range_filtered,"
                             << "dist_eval_count,improve_count\n";
                } else {
                    diag_ptr = nullptr;
                }
            }
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
                std::vector<unsigned> start_nodes;
                double seed_search_us_val = -1.0;
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
                    seed_search_us_val =
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
                    seed_search_us_val =
                        std::chrono::duration<double, std::micro>(ts1 - ts0)
                            .count();
                } else if (seed_policy == "spread") {
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
                } else if (seed_policy == "header") {
                    start_nodes = header;
                } else {
                    // anchors / hybrid (existing logic)
                    const unsigned width = range_ru - range_lu + 1;
                    const unsigned anc_n = std::max(1u, seed_anchor_count);
                    start_nodes.reserve(
                        anc_n + (seed_policy == "hybrid" ? header.size() : 0));
                    if (anc_n == 1) {
                        start_nodes.push_back(range_lu + width / 2);
                    } else {
                        for (unsigned ai = 0; ai < anc_n; ++ai) {
                            const unsigned off = static_cast<unsigned>(
                                (static_cast<unsigned long long>(ai) *
                                 (width - 1)) /
                                (anc_n - 1));
                            start_nodes.push_back(range_lu + off);
                        }
                    }
                    if (seed_policy == "hybrid") {
                        start_nodes.insert(start_nodes.end(), header.begin(),
                                           header.end());
                    }
                    std::sort(start_nodes.begin(), start_nodes.end());
                    start_nodes.erase(
                        std::unique(start_nodes.begin(), start_nodes.end()),
                        start_nodes.end());
                }
                if (can_use_fast_query) {
                    if (start_nodes.empty()) {
                        start_nodes.push_back(range_lu);
                    }
                    auto result = direct_searcher.beam_search_range_fast(
                        query_list[ui], qnumber, start_nodes, beam_size,
                        trunc_size, range_lu, range_ru, scratch, q_nav_degree,
                        nav_scan_factor, nav_stall_rounds, nav_front_keep,
                        q_nav_tail_degree, q_nav_early_stop_rounds);
                    std::ranges::copy(result | std::views::transform(GET(second)),
                                      ans.begin() + ui * qnumber);
                    if (((i + 1) % 1024 == 0) ||
                        (i + 1 == query_list.size())) {
                        spdlog::info("Processed {}/{} queries", i + 1,
                                     query_list.size());
                    }
                    continue;
                }
                std::vector<std::pair<float, unsigned>> result;
                BeamSearchStats* stats_ptr =
                    do_query_log ? &beam_stats : nullptr;
                // Per-query provenance tracking (declared outside branch for scope)
                std::vector<GtHitProv> query_prov;
                std::vector<GtHitProv>* prov_ptr = nullptr;
                // Per-query eval provenance tracking (P10-1, declared outside for scope)
                std::vector<EvalProvRecord> query_eval_prov;
                std::vector<EvalProvRecord>* eval_prov_ptr = nullptr;
                if (!eval_prov_csv.empty()) {
                    eval_prov_ptr = &query_eval_prov;
                }
                // P11-1: Build per-query GT set in outer scope for retained analysis
                std::unordered_set<unsigned> query_gt_set;
                const std::unordered_set<unsigned>* gt_set_arg = nullptr;
                if (!gt_remapped.empty()) {
                    for (size_t gi = ui * qnumber;
                         gi < (ui + 1) * qnumber && gi < gt_remapped.size();
                         ++gi) {
                        query_gt_set.insert(
                            static_cast<unsigned>(gt_remapped[gi]));
                    }
                    gt_set_arg = &query_gt_set;
                }
                auto t0 = std::chrono::high_resolution_clock::now();
                if (range_scan_mode == "direct") {
                    const unsigned range_lu = static_cast<unsigned>(range_l);
                    const unsigned range_ru =
                        static_cast<unsigned>(range_r - 1);
                    if (start_nodes.empty()) {
                        start_nodes.push_back(range_lu);
                    }
                    if (!provenance_csv.empty() && gt_set_arg != nullptr) {
                        prov_ptr = &query_prov;
                    }
                    if (rgg_searcher_ptr) {
                        // Role-gated search (P16-R)
                        rgg_ptr->set_query_range(range_lu, range_ru);
                        result = rgg_searcher_ptr->beam_search_range(
                            query_list[ui], qnumber, start_nodes, beam_size,
                            trunc_size, range_lu, range_ru, scratch,
                            q_nav_degree, nav_scan_factor, nav_stall_rounds,
                            nav_front_keep, q_nav_tail_degree,
                            q_nav_early_stop_rounds, pick_scan_factor,
                            pick_front_keep, pick_policy_code,
                            edge_pick_recip_depth, edge_pick_core_ratio,
                            fallback_stall_rounds, fb_policy_code,
                            fallback_core_ratio, fallback_pick_front_keep,
                            fallback_pick_scan_factor, fallback_release_nav,
                            nullptr, stats_ptr, rescue_slot_count,
                            rescue_pick_policy, warmup_min,
                            trigger_recent_window, trigger_flat_threshold,
                            trigger_span_round, trigger_span_norm_threshold,
                            diag_ptr, bridge_quota_floor, gt_set_arg,
                            suffix_promote, deep_bridge_floor,
                            deep_bridge_pos_threshold, prov_ptr,
                            post_arrival_eval_cap, post_arrival_core_cap,
                            post_arrival_bridge_cap, post_arrival_prefix_cap,
                            eval_prov_ptr, pa_adaptive_cap, pa_adaptive_window,
                            pa_stall_threshold, pa_stall_decay, pa_stall_floor,
                            pa_stable_threshold, pa_stable_cap,
                            pa_rank_head_keep, pa_rank_stale_window,
                            pa_head_gate_keep, pa_head_gate_tail_budget,
                            false);
                    } else if (cgg_searcher_ptr) {
                        // Conflict-gated search
                        cgg_ptr->set_query_range(range_lu, range_ru);
                        result = cgg_searcher_ptr->beam_search_range(
                            query_list[ui], qnumber, start_nodes, beam_size,
                            trunc_size, range_lu, range_ru, scratch,
                            q_nav_degree, nav_scan_factor, nav_stall_rounds,
                            nav_front_keep, q_nav_tail_degree,
                            q_nav_early_stop_rounds, pick_scan_factor,
                            pick_front_keep, pick_policy_code,
                            edge_pick_recip_depth, edge_pick_core_ratio,
                            fallback_stall_rounds, fb_policy_code,
                            fallback_core_ratio, fallback_pick_front_keep,
                            fallback_pick_scan_factor, fallback_release_nav,
                            nullptr, stats_ptr, rescue_slot_count,
                            rescue_pick_policy, warmup_min,
                            trigger_recent_window, trigger_flat_threshold,
                            trigger_span_round, trigger_span_norm_threshold,
                            diag_ptr, bridge_quota_floor, gt_set_arg,
                            suffix_promote, deep_bridge_floor,
                            deep_bridge_pos_threshold, prov_ptr,
                            post_arrival_eval_cap, post_arrival_core_cap,
                            post_arrival_bridge_cap, post_arrival_prefix_cap,
                            eval_prov_ptr, pa_adaptive_cap, pa_adaptive_window,
                            pa_stall_threshold, pa_stall_decay, pa_stall_floor,
                            pa_stable_threshold, pa_stable_cap,
                            pa_rank_head_keep, pa_rank_stale_window,
                            pa_head_gate_keep, pa_head_gate_tail_budget,
                            false /* no sorted idx with conflict gating */);
                    } else if (sgg_searcher_ptr) {
                        // Survival-gated search
                        sgg_ptr->set_query(ql, qr, range_lu, range_ru);
                        result = sgg_searcher_ptr->beam_search_range(
                            query_list[ui], qnumber, start_nodes, beam_size,
                            trunc_size, range_lu, range_ru, scratch,
                            q_nav_degree, nav_scan_factor, nav_stall_rounds,
                            nav_front_keep, q_nav_tail_degree,
                            q_nav_early_stop_rounds, pick_scan_factor,
                            pick_front_keep, pick_policy_code,
                            edge_pick_recip_depth, edge_pick_core_ratio,
                            fallback_stall_rounds, fb_policy_code,
                            fallback_core_ratio, fallback_pick_front_keep,
                            fallback_pick_scan_factor, fallback_release_nav,
                            nullptr, stats_ptr, rescue_slot_count,
                            rescue_pick_policy, warmup_min,
                            trigger_recent_window, trigger_flat_threshold,
                            trigger_span_round, trigger_span_norm_threshold,
                            diag_ptr, bridge_quota_floor, gt_set_arg,
                            suffix_promote, deep_bridge_floor,
                            deep_bridge_pos_threshold, prov_ptr,
                            post_arrival_eval_cap, post_arrival_core_cap,
                            post_arrival_bridge_cap, post_arrival_prefix_cap,
                            eval_prov_ptr, pa_adaptive_cap, pa_adaptive_window,
                            pa_stall_threshold, pa_stall_decay, pa_stall_floor,
                            pa_stable_threshold, pa_stable_cap,
                            pa_rank_head_keep, pa_rank_stale_window,
                            pa_head_gate_keep, pa_head_gate_tail_budget,
                            false /* no sorted idx with survival gating */);
                    } else {
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
                        nullptr, stats_ptr, rescue_slot_count,
                        rescue_pick_policy, warmup_min,
                        trigger_recent_window, trigger_flat_threshold,
                        trigger_span_round, trigger_span_norm_threshold,
                        diag_ptr, bridge_quota_floor, gt_set_arg,
                        suffix_promote, deep_bridge_floor,
                        deep_bridge_pos_threshold, prov_ptr,
                        post_arrival_eval_cap,
                        post_arrival_core_cap, post_arrival_bridge_cap,
                        post_arrival_prefix_cap, eval_prov_ptr,
                        pa_adaptive_cap, pa_adaptive_window,
                        pa_stall_threshold, pa_stall_decay, pa_stall_floor,
                        pa_stable_threshold, pa_stable_cap,
                        pa_rank_head_keep, pa_rank_stale_window,
                        pa_head_gate_keep, pa_head_gate_tail_budget,
                        use_sorted_range_idx);
                    }
                } else {
                    Searcher<float, decltype(g_sub)> searcher(dataset, g_sub);
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
                        nullptr, stats_ptr, rescue_slot_count,
                        rescue_pick_policy, warmup_min,
                        trigger_recent_window, trigger_flat_threshold,
                        bridge_quota_floor);
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
                    unsigned best_seed = start_nodes.empty()
                                             ? static_cast<unsigned>(range_l)
                                             : start_nodes[0];
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

                    QueryLogCsv::Row row;
                    row.dataset = log_dataset;
                    row.method =
                        log_method.empty() ? std::string("RNSG") : log_method;
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
                    row.seed_search_us = seed_search_us_val;

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
                    row.fallback_trigger_recent_insertions =
                        beam_stats.fallback_trigger_recent_insertions;
                    row.fallback_trigger_flat_rounds =
                        beam_stats.fallback_trigger_flat_rounds;
                    row.rescue_candidates_added =
                        beam_stats.rescue_candidates_added;
                    row.rescue_candidates_evaluated =
                        beam_stats.rescue_candidates_evaluated;
                    row.rescue_candidates_inserted =
                        beam_stats.rescue_candidates_inserted;

                    // Post-arrival DCO breakdown
                    row.first_gt_hit_expansion =
                        beam_stats.first_gt_hit_expansion;
                    row.dco_at_first_gt_hit =
                        beam_stats.dco_at_first_gt_hit;
                    row.seed_ids = std::move(seed_ids);
                    row.best_seed_rank_in_range = seed_rank;
                    row.range_scan_mode = range_scan_mode;
                    row.log_bucket = log_bucket;
                    QueryLogCsv::write_row(qlog_out, row);
                }

                // Write diagnostic snapshots
                if (diag_ptr && diag_out.good()) {
                    for (const auto& snap : diag_data.rounds) {
                        diag_out << ui << ',' << snap.round << ','
                                 << snap.expanded_node << ','
                                 << snap.insertions << ','
                                 << snap.cumulative_insertions << ','
                                 << std::setprecision(17) << snap.best_dist
                                 << ',' << snap.best_dist_ever << ','
                                 << snap.stall_rounds << ','
                                 << snap.beam_label_span << ','
                                 << snap.beam_size_effective << ','
                                 << snap.beam_bridge_count << ','
                                 << snap.beam_core_count << ','
                                 << snap.raw_count << ','
                                 << snap.selected_count << ','
                                 << snap.range_filtered << ','
                                 << snap.dist_eval_count << ','
                                 << snap.improve_count << '\n';
                    }
                }

                // Write provenance data
                if (prov_ptr && !query_prov.empty() && prov_out.good()) {
                    for (const auto& rec : query_prov) {
                        prov_out << ui << ',' << rec.gt_node_id << ','
                                 << rec.expansion_round << ','
                                 << static_cast<unsigned>(rec.edge_category) << ','
                                 << rec.edge_raw_position << ','
                                 << rec.edge_gap << ','
                                 << (rec.is_post_arrival ? 1 : 0) << ','
                                 << rec.current_node << '\n';
                    }
                }

                // Write eval provenance data (P10-1 + P11-1 retained analysis)
                if (eval_prov_ptr && !query_eval_prov.empty()
                    && eval_prov_out.good()) {
                    // P11-1: Compute retained sets from final beam state
                    std::unordered_set<unsigned> final_beam_nodes;
                    for (const auto& [dist, id_raw] : scratch.candidates) {
                        unsigned nid = (id_raw >= dataset.size())
                            ? id_raw - dataset.size() : id_raw;
                        if (dist < static_cast<float>(1e30)) {
                            final_beam_nodes.insert(nid);
                        }
                    }
                    std::unordered_set<unsigned> final_topk_nodes;
                    for (const auto& [dist, nid] : result) {
                        final_topk_nodes.insert(nid);
                    }
                    for (const auto& rec : query_eval_prov) {
                        bool in_beam = final_beam_nodes.count(rec.neighbor_node) > 0;
                        bool in_topk = final_topk_nodes.count(rec.neighbor_node) > 0;
                        bool is_gt = query_gt_set.count(rec.neighbor_node) > 0;
                        eval_prov_out << ui << ',' << rec.expansion_round << ','
                                     << rec.current_node << ','
                                     << rec.neighbor_node << ','
                                     << static_cast<unsigned>(rec.edge_category) << ','
                                     << rec.selected_rank << ','
                                     << (rec.prior_insertion ? 1 : 0) << ','
                                     << (rec.was_inserted ? 1 : 0) << ','
                                     << (rec.improved_best ? 1 : 0) << ','
                                     << (rec.is_post_arrival ? 1 : 0) << ','
                                     << rec.dist << ','
                                     << (in_beam ? 1 : 0) << ','
                                     << (in_topk ? 1 : 0) << ','
                                     << (is_gt ? 1 : 0) << '\n';
                    }
                }

                if (((i + 1) % 1024 == 0) || (i + 1 == query_list.size())) {
                    spdlog::info("Processed {}/{} queries", i + 1,
                                 query_list.size());
                }
            }

            // Print role-gated graph quota utilization stats
            if (rgg_ptr) {
                rgg_ptr->print_stats();
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

    int query_batch() {
        apply_thread_limit();
        spdlog::info("Running batch lightweight RNSG queries...");
        Vector::VectorList<float> vector_list(dataset_file);
        Graph::TDGraphIndexBase index(index_file);
        if (build_sorted_idx || use_sorted_range_idx) {
            index.build_sorted_neighbor_index();
            spdlog::info("Built sorted neighbor dual-index for batch range access");
        }
        Vector::VectorList<float> query_list(query_file);
        auto label = IO::load_json_to_vec<std::uint64_t>(label_file);
        auto [ord, pos] = Utils::order_of_label(label);
        auto sorted_label = Utils::sorted_vec(label);
        auto& dataset = vector_list;
        dataset.reorder(ord);
        Searcher<float, Graph::TDGraphIndexBase> direct_searcher(dataset,
                                                                 index);
        const int thread_count = runtime_thread_count();

        struct BatchRow {
            std::unordered_map<std::string, std::string> value;
        };
        std::vector<BatchRow> rows;
        {
            std::ifstream fin(batch_params_file);
            if (!fin.good()) {
                spdlog::error("Failed to open batch params file {}",
                              batch_params_file);
                return 1;
            }
            std::string line;
            std::vector<std::string> header;
            while (std::getline(fin, line)) {
                line = trim_copy(line);
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                if (header.empty()) {
                    header = split_csv_line(line);
                    continue;
                }
                auto fields = split_csv_line(line);
                fields.resize(header.size());
                BatchRow row;
                for (size_t i = 0; i < header.size(); ++i) {
                    row.value[header[i]] = fields[i];
                }
                rows.push_back(std::move(row));
            }
        }
        if (rows.empty()) {
            spdlog::error("No batch rows in {}", batch_params_file);
            return 1;
        }

        std::unordered_map<std::string, std::vector<std::uint64_t>>
            qrange_cache;
        std::unordered_map<std::string, std::vector<unsigned>> gt_cache;

        std::filesystem::path csv_path(batch_csv_file);
        if (csv_path.has_parent_path()) {
            std::filesystem::create_directories(csv_path.parent_path());
        }
        const bool need_header =
            !batch_append || !std::filesystem::exists(csv_path) ||
            std::filesystem::file_size(csv_path) == 0;
        std::ofstream csv_out(
            batch_csv_file,
            batch_append ? std::ios::app : std::ios::out);
        if (!csv_out.good()) {
            spdlog::error("Failed to open batch output CSV {}",
                          batch_csv_file);
            return 1;
        }
        if (need_header) {
            csv_out
                << "run_id,selectivity,selectivity_label,topk,beam_size,"
                   "trunc_size,nav_degree,nav_scan_factor,nav_stall_rounds,"
                   "nav_front_keep,nav_tail_degree,nav_early_stop_rounds,"
                   "seed_policy,status,recall,qps,average_query_time_ns,"
                   "real_seconds,result_file,qrange_file,groundtruth_file\n";
        }

        auto get_str = [](const BatchRow& row, const std::string& key,
                          const std::string& def = std::string()) {
            auto it = row.value.find(key);
            return it == row.value.end() || it->second.empty() ? def
                                                               : it->second;
        };
        auto get_uint = [&](const BatchRow& row, const std::string& key,
                            unsigned def) {
            const auto s = get_str(row, key);
            return s.empty() ? def : static_cast<unsigned>(std::stoul(s));
        };
        auto get_double_str = [&](const BatchRow& row,
                                  const std::string& key) {
            return get_str(row, key);
        };

        unsigned failed = 0;
        for (size_t ri = 0; ri < rows.size(); ++ri) {
            const auto& row = rows[ri];
            const std::string run_id =
                get_str(row, "run_id", "run" + std::to_string(ri));
            const std::string qrange_path = get_str(row, "qrange_file");
            const std::string gt_path = get_str(row, "groundtruth_file");
            unsigned run_topk = get_uint(row, "topk", qnumber);
            unsigned run_beam = get_uint(row, "beam_size", beam_size);
            unsigned run_trunc = get_uint(row, "trunc_size", trunc_size);
            run_beam = std::max(run_beam, run_topk);
            const unsigned run_nav_degree =
                get_uint(row, "nav_degree", nav_degree);
            const unsigned run_nav_scan_factor =
                get_uint(row, "nav_scan_factor", nav_scan_factor);
            const unsigned run_nav_stall_rounds =
                get_uint(row, "nav_stall_rounds", nav_stall_rounds);
            const unsigned run_nav_front_keep =
                get_uint(row, "nav_front_keep", nav_front_keep);
            const unsigned run_nav_tail_degree =
                get_uint(row, "nav_tail_degree", nav_tail_degree);
            const unsigned run_nav_early_stop_rounds =
                get_uint(row, "nav_early_stop_rounds",
                         nav_early_stop_rounds);
            const std::string run_seed_policy =
                get_str(row, "seed_policy", seed_policy);
            std::string run_result_file = get_str(row, "result_file");
            if (run_result_file.empty() && !batch_result_dir.empty()) {
                run_result_file =
                    (std::filesystem::path(batch_result_dir) /
                     (run_id + ".json"))
                        .string();
            }

            double recall = -1.0;
            double qps = 0.0;
            double avg_ns = 0.0;
            double real_seconds = 0.0;
            std::string status = "completed";
            std::vector<unsigned> ans;

            try {
                if (qrange_path.empty() || run_topk == 0 || run_beam == 0 ||
                    run_trunc == 0) {
                    throw std::runtime_error(
                        "batch row requires qrange_file, topk, beam_size, "
                        "trunc_size");
                }
                auto& qrange = qrange_cache[qrange_path];
                if (qrange.empty()) {
                    qrange =
                        IO::load_json_to_vec<std::uint64_t>(qrange_path);
                }
                if (qrange.size() < query_list.size() * 2) {
                    throw std::runtime_error("qrange length is too small");
                }
                std::vector<unsigned>* gt_ptr = nullptr;
                if (!gt_path.empty()) {
                    auto& gt = gt_cache[gt_path];
                    if (gt.empty()) {
                        gt = IO::load_json_to_vec<unsigned>(gt_path);
                        for (auto& id : gt) {
                            id = pos[id];
                        }
                    }
                    if (gt.size() < query_list.size() * run_topk) {
                        throw std::runtime_error(
                            "groundtruth length is too small");
                    }
                    gt_ptr = &gt;
                }

                ans.assign(static_cast<size_t>(query_list.size()) * run_topk,
                           0u);
                BeamScratch<float> scratch(dataset.size(), run_beam,
                                           run_trunc);
                const auto t0 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < query_list.size(); i++) {
                    auto ui = static_cast<unsigned>(i);
                    auto ql = qrange[ui * 2], qr = qrange[ui * 2 + 1];
                    auto range_l =
                        std::ranges::lower_bound(sorted_label, ql) -
                        sorted_label.begin();
                    auto range_r =
                        std::ranges::upper_bound(sorted_label, qr) -
                        sorted_label.begin();
                    if (range_l >= range_r) {
                        continue;
                    }
                    const unsigned range_width =
                        static_cast<unsigned>(range_r - range_l);
                    const unsigned range_lu = static_cast<unsigned>(range_l);
                    const unsigned range_ru =
                        static_cast<unsigned>(range_r - 1);
                    const unsigned header_id =
                        index.get_header_index_for_right_bound(qr);
                    std::vector<unsigned> header;
                    auto header_span = index.get_header(header_id);
                    header.reserve(header_span.size());
                    for (auto h : header_span) {
                        if (h >= range_lu && h <= range_ru) {
                            header.push_back(h);
                        }
                    }
                    std::vector<unsigned> start_nodes;

                    if (run_seed_policy == "oracle_query_seed") {
                        const unsigned m =
                            std::min(oracle_seed_count, range_width);
                        std::vector<std::pair<float, unsigned>> dists;
                        dists.reserve(range_width);
                        for (unsigned j = range_lu;
                             j < static_cast<unsigned>(range_r); ++j) {
                            dists.push_back(
                                {dataset.dist2(j, query_list[ui]), j});
                        }
                        if (dists.size() > m) {
                            std::partial_sort(dists.begin(),
                                              dists.begin() + m, dists.end());
                            dists.resize(m);
                        }
                        for (auto& [d, id] : dists) {
                            (void)d;
                            start_nodes.push_back(id);
                        }
                    } else if (run_seed_policy == "cheap_query_seed") {
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
                                    (range_width - 1) /
                                    std::max(1u, sample_n - 1));
                                samples.push_back(range_lu + off);
                            }
                        }
                        std::vector<std::pair<float, unsigned>> dists;
                        dists.reserve(samples.size());
                        for (auto j : samples) {
                            dists.push_back(
                                {dataset.dist2(j, query_list[ui]), j});
                        }
                        if (dists.size() > m) {
                            std::partial_sort(dists.begin(),
                                              dists.begin() + m, dists.end());
                            dists.resize(m);
                        }
                        for (auto& [d, id] : dists) {
                            (void)d;
                            start_nodes.push_back(id);
                        }
                    } else if (run_seed_policy == "spread") {
                        const unsigned anc_n = std::max(
                            1u, std::min(spread_anchor_count, range_width));
                        phmap::flat_hash_set<unsigned> seen;
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
                            if (seen.insert(p).second) {
                                start_nodes.push_back(p);
                            }
                        }
                        if (spread_include_header) {
                            for (auto h : header) {
                                if (seen.insert(h).second) {
                                    start_nodes.push_back(h);
                                }
                            }
                        }
                    } else if (run_seed_policy == "header") {
                        start_nodes = header;
                    } else {
                        const unsigned width = range_ru - range_lu + 1;
                        const unsigned anc_n =
                            std::max(1u, seed_anchor_count);
                        start_nodes.reserve(
                            anc_n +
                            (run_seed_policy == "hybrid" ? header.size()
                                                          : 0));
                        if (anc_n == 1) {
                            start_nodes.push_back(range_lu + width / 2);
                        } else {
                            for (unsigned ai = 0; ai < anc_n; ++ai) {
                                const unsigned off = static_cast<unsigned>(
                                    (static_cast<unsigned long long>(ai) *
                                     (width - 1)) /
                                    (anc_n - 1));
                                start_nodes.push_back(range_lu + off);
                            }
                        }
                        if (run_seed_policy == "hybrid") {
                            start_nodes.insert(start_nodes.end(),
                                               header.begin(), header.end());
                        }
                        std::sort(start_nodes.begin(), start_nodes.end());
                        start_nodes.erase(
                            std::unique(start_nodes.begin(),
                                        start_nodes.end()),
                            start_nodes.end());
                    }
                    if (start_nodes.empty()) {
                        start_nodes.push_back(range_lu);
                    }

                    auto result = direct_searcher.beam_search_range_fast(
                        query_list[ui], run_topk, start_nodes, run_beam,
                        run_trunc, range_lu, range_ru, scratch,
                        run_nav_degree, run_nav_scan_factor,
                        run_nav_stall_rounds, run_nav_front_keep,
                        run_nav_tail_degree, run_nav_early_stop_rounds,
                        use_sorted_range_idx);
                    std::ranges::copy(
                        result | std::views::transform(GET(second)),
                        ans.begin() + static_cast<size_t>(ui) * run_topk);
                }
                const auto t1 = std::chrono::high_resolution_clock::now();
                const auto ns =
                    std::chrono::duration<double, std::nano>(t1 - t0).count();
                real_seconds =
                    std::chrono::duration<double>(t1 - t0).count();
                avg_ns = ns / static_cast<double>(query_list.size());
                qps = static_cast<double>(query_list.size()) * 1e9 / ns;

                if (gt_ptr != nullptr) {
                    unsigned correct = 0;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic) \
    reduction(+ : correct)
                    for (int64_t i = 0;
                         i < static_cast<int64_t>(query_list.size()); i++) {
                        auto ui = static_cast<unsigned>(i);
                        phmap::flat_hash_map<float, unsigned> answer_cnt;
                        for (size_t j = static_cast<size_t>(ui) * run_topk;
                             j < static_cast<size_t>(ui + 1) * run_topk;
                             j++) {
                            answer_cnt[dataset.dist2(ans[j],
                                                     query_list[ui])]++;
                        }
                        unsigned local_correct = 0;
                        for (size_t j = static_cast<size_t>(ui) * run_topk;
                             j < static_cast<size_t>(ui + 1) * run_topk;
                             j++) {
                            auto now = dataset.dist2((*gt_ptr)[j],
                                                     query_list[ui]);
                            if (answer_cnt[now] > 0) {
                                local_correct++;
                                answer_cnt[now]--;
                            }
                        }
                        correct += local_correct;
                    }
                    recall = static_cast<double>(correct) /
                             (static_cast<size_t>(run_topk) *
                              query_list.size());
                }
                write_json_array(run_result_file, ans, ord);
            } catch (const std::exception& e) {
                status = std::string("failed:") + e.what();
                failed++;
                spdlog::error("Batch row {} failed: {}", run_id, e.what());
            }

            csv_out << csv_escape(run_id) << ','
                    << csv_escape(get_double_str(row, "selectivity")) << ','
                    << csv_escape(get_str(row, "selectivity_label")) << ','
                    << run_topk << ',' << run_beam << ',' << run_trunc << ','
                    << run_nav_degree << ',' << run_nav_scan_factor << ','
                    << run_nav_stall_rounds << ',' << run_nav_front_keep
                    << ',' << run_nav_tail_degree << ','
                    << run_nav_early_stop_rounds << ','
                    << csv_escape(run_seed_policy) << ','
                    << csv_escape(status) << ',';
            if (recall >= 0.0) {
                csv_out << std::setprecision(10) << recall;
            }
            csv_out << ',' << std::setprecision(10) << qps << ','
                    << std::setprecision(10) << avg_ns << ','
                    << std::setprecision(10) << real_seconds << ','
                    << csv_escape(run_result_file) << ','
                    << csv_escape(qrange_path) << ',' << csv_escape(gt_path)
                    << '\n';
            csv_out.flush();
            spdlog::info(
                "Batch {}/{} {} status={} recall={:.4f} qps={:.2f}",
                ri + 1, rows.size(), run_id, status,
                recall >= 0.0 ? recall : 0.0, qps);
        }
        spdlog::info("Batch query completed: total={}, failed={}",
                     rows.size(), failed);
        return failed == 0 ? 0 : 1;
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
    static std::string trim_copy(std::string s) {
        const auto is_space = [](unsigned char c) {
            return std::isspace(c) != 0;
        };
        while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) {
            s.erase(s.begin());
        }
        while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) {
            s.pop_back();
        }
        return s;
    }

    static std::vector<std::string> split_csv_line(const std::string& line) {
        std::vector<std::string> out;
        std::string cur;
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            const char c = line[i];
            if (c == '"') {
                if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = !in_quotes;
                }
            } else if (c == ',' && !in_quotes) {
                out.push_back(trim_copy(cur));
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        out.push_back(trim_copy(cur));
        return out;
    }

    static std::string csv_escape(const std::string& s) {
        if (s.find_first_of(",\"\n\r") == std::string::npos) {
            return s;
        }
        std::string out = "\"";
        for (char c : s) {
            if (c == '"') {
                out += "\"\"";
            } else {
                out.push_back(c);
            }
        }
        out.push_back('"');
        return out;
    }

    static bool str_to_bool(const std::string& s) {
        const auto v = trim_copy(s);
        return v == "1" || v == "true" || v == "TRUE" || v == "yes" ||
               v == "on";
    }

    static void write_json_array(const std::string& path,
                                 const std::vector<unsigned>& ans,
                                 const std::vector<unsigned>& ord) {
        if (path.empty()) {
            return;
        }
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream fout(path);
        if (!fout.good()) {
            throw std::runtime_error("failed to open result file: " + path);
        }
        fout << "[";
        for (unsigned i = 0; i < ans.size(); i++) {
            fout << ord[ans[i]];
            if (i + 1 != ans.size()) {
                fout << ",";
            }
        }
        fout << "]";
    }

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
    std::string batch_params_file;
    std::string batch_csv_file;
    std::string batch_result_dir;
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
    std::string entry_mode = "header";
    unsigned entry_seed = 42;
    std::string prefix_policy = "dist";
    double prefix_mix_ratio = 0.0;
    unsigned prefix_warmup = 8;
    unsigned prefix_jump_min_gap = 0;
    double prefix_score_alpha = 0.0;
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
    unsigned bridge_quota_floor = 0;
    unsigned suffix_promote = 0;
    unsigned deep_bridge_floor = 0;
    unsigned deep_bridge_pos_threshold = 32;
    std::string provenance_csv;
    std::string eval_prov_csv;
    bool sort_neighbors = false;
    bool build_sorted_idx = false;
    bool use_sorted_range_idx = false;
    bool fast_query = true;
    bool disable_fast_query = false;
    unsigned post_arrival_eval_cap = 0;
    unsigned post_arrival_core_cap = 0;
    unsigned post_arrival_bridge_cap = 0;
    unsigned post_arrival_prefix_cap = 0;
    // P10-2: Adaptive early-stop
    unsigned pa_adaptive_cap = 0;
    unsigned pa_adaptive_window = 1;
    unsigned pa_stall_threshold = 0;
    unsigned pa_stall_decay = 2;
    unsigned pa_stall_floor = 4;
    unsigned pa_stable_threshold = 0;
    unsigned pa_stable_cap = 0;
    // P10-3: Rank-aware tail pruning
    unsigned pa_rank_head_keep = 0;
    unsigned pa_rank_stale_window = 3;
    // P11-2: Head-gated tail unlock
    unsigned pa_head_gate_keep = 0;
    unsigned pa_head_gate_tail_budget = 0;
    unsigned fallback_stall_rounds = 0;
    std::string fallback_pick_policy = "prefix";
    double fallback_core_ratio = 0.40;
    unsigned fallback_pick_front_keep = 0;
    unsigned fallback_pick_scan_factor = 1;
    bool fallback_release_nav = false;
    unsigned rescue_slot_count = 0;
    unsigned rescue_pick_policy = 0;
    unsigned warmup_min = 16;
    unsigned trigger_recent_window = 0;
    unsigned trigger_flat_threshold = 0;
    unsigned trigger_span_round = 0;
    double trigger_span_norm_threshold = 0.0;
    std::string diag_csv;
    std::string range_scan_mode = "subgraph";
    std::string seed_policy = "header";
    unsigned seed_anchor_count = 3;
    unsigned oracle_seed_count = 4;
    unsigned cheap_query_sample_count = 64;
    unsigned cheap_query_seed_count = 4;
    unsigned spread_anchor_count = 4;
    bool spread_include_header = false;
    bool query_log_compute_seed_qrank = false;
    std::string query_log_csv;
    std::string log_dataset;
    std::string log_method;
    std::string log_bucket;
    double log_target_recall = -1.0;
    bool query_log_append = false;
    bool batch_append = false;
    std::string survival_anno_file;
    unsigned survival_skeleton = 8;
    std::string conflict_anno_file;
    std::string role_anno_file;
    unsigned skeleton_quota = 6;
    unsigned useful_quota = 16;
    unsigned bridge_quota = 2;
    unsigned reserve_quota = 2;
    std::string role_sort_mode_str = "gap_asc";
    unsigned skeleton_floor = 6;
    unsigned bridge_floor = 2;
    unsigned edge_limit = 26;
};

}  // namespace TDFANN::RNSG
