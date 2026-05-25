#pragma once

#include <PCH.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

namespace TDFANN::RNSG::QueryLogCsv {

inline std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

inline void write_header(std::ostream& o) {
    o << "dataset,method,query_id,range_l,range_r,range_width,selectivity,k,"
         "target_recall,latency_us,recall_at_k,distance_computations,"
         "visited_nodes,expanded_nodes,raw_neighbors_scanned,"
         "range_filtered_out_neighbors,in_range_neighbors_evaluated,"
         "beam_insertions,beam_rewinds,early_stop_triggered,"
         "fallback_triggered,fallback_trigger_step,fallback_trigger_stall_"
         "rounds,"
         "post_trigger_dco,fallback_reason_code,"
         "fallback_trigger_recent_insertions,fallback_trigger_flat_rounds,"
         "rescue_candidates_added,rescue_candidates_evaluated,rescue_"
         "candidates_inserted,"
         "seed_ids,"
         "best_seed_rank_in_range,true_nn_id,true_nn_dist,found_true_nn,"
         "first_hit_hop,best_seed_query_rank,best_seed_query_rank_ratio,"
         "best_seed_query_dist,seed_search_us,range_scan_mode,log_bucket,"
         "first_gt_hit_expansion,dco_at_first_gt_hit\n";
}

struct Row {
    std::string dataset;
    std::string method;
    unsigned query_id = 0;
    std::uint64_t range_l = 0;
    std::uint64_t range_r = 0;
    unsigned range_width = 0;
    double selectivity = 0.0;
    unsigned k = 0;
    double target_recall = -1.0;
    double latency_us = 0.0;
    double recall_at_k = -1.0;
    std::uint64_t distance_computations = 0;
    std::uint64_t visited_nodes = 0;
    std::uint64_t expanded_nodes = 0;
    std::uint64_t raw_neighbors_scanned = 0;
    std::uint64_t range_filtered_out_neighbors = 0;
    std::uint64_t in_range_neighbors_evaluated = 0;
    std::uint64_t beam_insertions = 0;
    std::uint64_t beam_rewinds = 0;
    bool early_stop_triggered = false;
    bool fallback_triggered = false;
    unsigned fallback_trigger_step = 0;
    unsigned fallback_trigger_stall_rounds = 0;
    std::uint64_t post_trigger_dco = 0;
    unsigned fallback_reason_code = 0;
    unsigned rescue_candidates_added = 0;
    unsigned rescue_candidates_evaluated = 0;
    unsigned rescue_candidates_inserted = 0;

    // --- composite trigger snapshots ---
    unsigned fallback_trigger_recent_insertions = 0;
    unsigned fallback_trigger_flat_rounds = 0;

    std::string seed_ids;
    long long best_seed_rank_in_range = -1;
    long long true_nn_id = -1;
    double true_nn_dist = -1.0;
    int found_true_nn = -1;
    long long first_hit_hop = -1;
    long long best_seed_query_rank = -1;
    double best_seed_query_rank_ratio = -1.0;
    double best_seed_query_dist = -1.0;
    double seed_search_us = -1.0;
    std::string range_scan_mode;
    std::string log_bucket;

    // --- post-arrival DCO breakdown ---
    unsigned first_gt_hit_expansion = 0;
    std::uint64_t dco_at_first_gt_hit = 0;
};

inline void write_row(std::ostream& o, const Row& r) {
    o << csv_escape(r.dataset) << ',' << csv_escape(r.method) << ','
      << r.query_id << ',' << r.range_l << ',' << r.range_r << ','
      << r.range_width << ',' << std::setprecision(17) << r.selectivity << ','
      << r.k << ',';
    if (r.target_recall >= 0.0) {
        o << std::setprecision(10) << r.target_recall;
    }
    o << ',' << std::setprecision(6) << r.latency_us << ',';
    if (r.recall_at_k >= 0.0) {
        o << std::setprecision(10) << r.recall_at_k;
    }
    o << ',' << r.distance_computations << ',' << r.visited_nodes << ','
      << r.expanded_nodes << ',' << r.raw_neighbors_scanned << ','
      << r.range_filtered_out_neighbors << ',' << r.in_range_neighbors_evaluated
      << ',' << r.beam_insertions << ',' << r.beam_rewinds << ','
      << (r.early_stop_triggered ? 1 : 0) << ','
      << (r.fallback_triggered ? 1 : 0) << ',' << r.fallback_trigger_step << ','
      << r.fallback_trigger_stall_rounds << ',' << r.post_trigger_dco << ','
      << r.fallback_reason_code << ',' << r.fallback_trigger_recent_insertions
      << ',' << r.fallback_trigger_flat_rounds << ','
      << r.rescue_candidates_added << ','
      << r.rescue_candidates_evaluated << ',' << r.rescue_candidates_inserted
      << ',' << csv_escape(r.seed_ids) << ',';
    if (r.best_seed_rank_in_range >= 0) {
        o << r.best_seed_rank_in_range;
    }
    o << ',';
    if (r.true_nn_id >= 0) {
        o << r.true_nn_id;
    }
    o << ',';
    if (r.true_nn_dist >= 0.0) {
        o << std::setprecision(10) << r.true_nn_dist;
    }
    o << ',';
    if (r.found_true_nn >= 0) {
        o << r.found_true_nn;
    }
    o << ',';
    if (r.first_hit_hop >= 0) {
        o << r.first_hit_hop;
    }
    o << ',';
    if (r.best_seed_query_rank >= 0) {
        o << r.best_seed_query_rank;
    }
    o << ',';
    if (r.best_seed_query_rank_ratio >= 0.0) {
        o << std::setprecision(10) << r.best_seed_query_rank_ratio;
    }
    o << ',';
    if (r.best_seed_query_dist >= 0.0) {
        o << std::setprecision(10) << r.best_seed_query_dist;
    }
    o << ',';
    if (r.seed_search_us >= 0.0) {
        o << std::setprecision(6) << r.seed_search_us;
    }
    o << ',' << csv_escape(r.range_scan_mode) << ',' << csv_escape(r.log_bucket)
      << ',' << r.first_gt_hit_expansion << ',' << r.dco_at_first_gt_hit
      << '\n';
}

inline bool file_needs_header(const std::string& path, bool append) {
    if (!append) {
        return true;
    }
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return true;
    }
    return std::filesystem::file_size(path, ec) == 0;
}

}  // namespace TDFANN::RNSG::QueryLogCsv
