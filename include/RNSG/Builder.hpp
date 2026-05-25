#pragma once
#include <PCH.hpp>

#include <faiss/IndexNNDescent.h>
#include <omp.h>
#include <atomic>
#include <cmath>
#include <cctype>
#include <numeric>
#include <Core/Searcher.hpp>
#include <Graph/GraphIndex.hpp>
#include <Utils/Recorder.hpp>
#include <Utils/Threading.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN::RNSG {

struct BuildOptions {
    bool enable_range_augmentation = true;
    bool enable_side_split_pruning = true;
    bool use_mrng_pruning = false;
    // Prefix ordering policy for neighbour list in final graph.
    // dist: keep pure source-distance order (baseline)
    // mix: distance prefix + periodic long-jump label edges
    // score: rank-fused distance/label-gap ordering
    // scoreg: score with quality gate (only re-rank a near-distance pool)
    // cover: reserve early prefix for multi-scale long-jump coverage
    // balance: keep a distance-good pool but interleave left/right sides
    //          whenever both sides are similarly near, to stabilize trunc
    //          after range filtering without over-injecting far jumps.
    // labelg: keep a distance-good pool, but prefer smaller label-gap
    //         neighbours inside each side before interleaving them.
    // balmix: like balance, but inside each side apply a mild local label-gap
    //         bias while largely preserving the original distance order.
    // bridgefirst: keep a local warmup, then explicitly reserve early slots
    //              for larger label-gap bridge edges while preserving side
    //              balance and near-distance order inside each bucket.
    // thinbridge: keep the near-distance warmup and an extra local-only block,
    //             then sparsely inject bridge edges into the later prefix to
    //             thin low-yield tail locals without disturbing the earliest
    //             local descent hops too much.
    // switchband: keep the near-distance warmup, then reserve a small band for
    //             moderate-gap side-switch edges before falling back to the
    //             original distance order. This targets the rank-8~15 band
    //             that still contributes improvements on WIT without pushing
    //             very long jumps into the earliest prefix.
    std::string prefix_policy = "dist";
    // Used by prefix_policy=mix/cover.
    double prefix_mix_ratio = 0.0;
    unsigned prefix_warmup = 8;
    unsigned prefix_jump_min_gap = 0;
    // Used by prefix_policy=score.
    double prefix_score_alpha = 0.0;
};

template <typename T>
class Builder {
   public:
    Builder(Vector::VectorList<T>& data) : vector_list(data) {};
    Graph::GraphIndex<std::monostate> nn_descent(unsigned k,
                                                 bool verbose = false) const;

    Graph::TDGraphIndexBase build(Graph::GraphLike auto&& knng,
                                  unsigned range_step, unsigned ef_max,
                                  const std::vector<std::uint64_t>& label,
                                  const BuildOptions& options = {});

    void init_header(Graph::TDGraphIndexBase&,
                     const Vector::VectorType<T>&,
                     const std::vector<std::uint64_t>& label,
                     const std::ranges::range auto&& order,
                     const Vector::VectorList<T>& vector_list) const;

   private:
    Vector::VectorList<T>& vector_list;
};

//>===========================================================<

// Implement of builder functions

inline Graph::GraphIndex<std::monostate> nn_descent_float_data(
    const float* data, unsigned n, unsigned dimension, unsigned k,
    bool verbose = false) {
    const int thread_count = Utils::configured_thread_count();
    omp_set_num_threads(thread_count);
    Timer::start("knng_time");
    Graph::GraphIndex<std::monostate> graph(n);
    spdlog::info("start KNN train with size {}, dim {}", n, dimension);
    faiss::IndexNNDescentFlat index(dimension, k);
    index.nndescent.iter = 15;
    index.verbose = verbose;
    index.add(n, data);
    spdlog::info("KNNG done, size = {}", index.nndescent.final_graph.size());
    size_t step = std::max<size_t>(1, n / 10);
    for (size_t i = 0; i < n; i++) {
        if (i % step == 0) {
            spdlog::info("Processing vector {}/{}, range = [{}, {}]", i, n,
                         i * k, (i + 1) * k);
        }
        graph.add_neighbours(i, index.nndescent.final_graph |
                                    std::views::drop(i * k) |
                                    std::views::take(k));
    }
    spdlog::info("KNN train finished.");
    spdlog::info("KNNG build time: {} s", Timer::end("knng_time") / 1e9);
    return graph;
}

template <typename T>
Graph::GraphIndex<std::monostate> Builder<T>::nn_descent(unsigned k,
                                                         bool verbose) const {
    if (vector_list.is_i8bin()) {
        spdlog::info(
            "KNNG source is i8bin; converting to transient float buffer for "
            "FAISS NNDescent");
        auto float_data = vector_list.to_float_data();
        return nn_descent_float_data(float_data.data(), vector_list.size(),
                                     vector_list.dim(), k, verbose);
    }
    return nn_descent_float_data(vector_list.data(), vector_list.size(),
                                 vector_list.dim(), k, verbose);
}

template <typename T>
bool check_valid(const Vector::VectorList<T>& vector_list,
                 const std::pair<T, unsigned>& now,
                 const std::vector<std::pair<T, unsigned>>& result) {
    auto [d_now, i_now] = now;
    for (auto [d_lst, i_lst] : result) {
        if ((d_now > d_lst && d_now > vector_list.dist(i_now, i_lst)) ||
            i_lst == i_now) {
            return false;
        }
    }
    return true;
}

template <typename T>
std::vector<std::pair<T, unsigned>> prune(
    const Vector::VectorList<T>& vector_list,
    const std::vector<std::pair<T, unsigned>>& candidates,
    unsigned ef_max = 1000,
    std::vector<unsigned>* tag = nullptr) {
    std::vector<std::pair<T, unsigned>> result;
    if (tag == nullptr) {
        for (unsigned i = 0; i < candidates.size(); i++) {
            if (check_valid(vector_list, candidates[i], result)) {
                result.push_back(candidates[i]);
                // if (result.size() >= ef_max) {
                //     break;
                // }
            }
        }
    } else {
        auto& suf = *tag;
        suf.clear();
        for (unsigned i = 0; i < candidates.size(); i++) {
            auto [d_now, i_now] = candidates[i];
            bool append = true;
            for (unsigned j = 0; j < result.size(); j++) {
                auto [d_lst, i_lst] = result[j];
                if (d_now > d_lst || suf[j] == unsigned(-1)) {
                    auto d_ij = vector_list.dist(i_now, i_lst);
                    if (d_now > d_lst && d_now > d_ij) {
                        append = false;
                        break;
                    }
                    if (suf[j] == unsigned(-1) && d_lst > d_now &&
                        d_lst > d_ij) {
                        suf[j] = i_now;
                    }
                }
            }
            if (!append) {
                for (auto& x : suf) {
                    if (x == i_now) {
                        x = unsigned(-1);
                    }
                }
            } else {
                result.push_back(candidates[i]);
                suf.push_back(unsigned(-1));
                if (result.size() >= ef_max) {
                    break;
                }
            }
        }
    }
    return result;
}

inline std::string normalize_prefix_policy(std::string p) {
    for (auto& c : p) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (p == "dist" || p == "mix" || p == "score" || p == "scoreg" ||
        p == "cover" || p == "balance" || p == "labelg" || p == "tierbal" ||
        p == "balmix" || p == "bridgefirst" || p == "thinbridge" ||
        p == "switchband") {
        return p;
    }
    return "dist";
}

template <typename T>
void reorder_prefix_edges(std::vector<std::pair<T, unsigned>>& edges,
                          unsigned src_id,
                          const BuildOptions& options) {
    if (edges.size() <= 2) {
        return;
    }
    const std::string policy = normalize_prefix_policy(options.prefix_policy);
    if (policy == "dist") {
        return;
    }

    if (policy == "labelg") {
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        const double ratio = options.prefix_mix_ratio <= 1e-9
                                 ? 0.5
                                 : std::clamp(options.prefix_mix_ratio, 0.0,
                                              1.0);
        const size_t gated_cnt = std::max<size_t>(
            16, static_cast<size_t>(std::llround(
                    static_cast<double>(n - warmup) * ratio)));
        const size_t pool_end = std::min<size_t>(n, warmup + gated_cnt);
        if (pool_end <= warmup + 1) {
            return;
        }

        std::vector<size_t> left, right;
        left.reserve(pool_end - warmup);
        right.reserve(pool_end - warmup);
        for (size_t i = warmup; i < pool_end; ++i) {
            if (edges[i].second < src_id) {
                left.push_back(i);
            } else {
                right.push_back(i);
            }
        }
        auto by_label_gap = [&](size_t a, size_t b) {
            const auto ga = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[a].second) -
                         static_cast<long long>(src_id)));
            const auto gb = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[b].second) -
                         static_cast<long long>(src_id)));
            if (ga != gb) {
                return ga < gb;
            }
            return a < b;
        };
        std::ranges::sort(left, by_label_gap);
        std::ranges::sort(right, by_label_gap);

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
        }

        size_t li = 0, ri = 0;
        int last_side = -1;
        constexpr size_t kRankTolerance = 4;
        auto emit_left = [&]() {
            out.push_back(edges[left[li++]]);
            last_side = 0;
        };
        auto emit_right = [&]() {
            out.push_back(edges[right[ri++]]);
            last_side = 1;
        };

        while (li < left.size() || ri < right.size()) {
            if (li >= left.size()) {
                emit_right();
                continue;
            }
            if (ri >= right.size()) {
                emit_left();
                continue;
            }
            const size_t left_rank = left[li];
            const size_t right_rank = right[ri];
            int preferred = (left_rank <= right_rank) ? 0 : 1;
            if (last_side == preferred) {
                const size_t pref_rank = (preferred == 0) ? left_rank : right_rank;
                const size_t alt_rank = (preferred == 0) ? right_rank : left_rank;
                if (alt_rank <= pref_rank + kRankTolerance) {
                    preferred = 1 - preferred;
                }
            }
            if (preferred == 0) {
                emit_left();
            } else {
                emit_right();
            }
        }

        for (size_t i = pool_end; i < n; ++i) {
            out.push_back(edges[i]);
        }
        edges.swap(out);
        return;
    }

    if (policy == "tierbal") {
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        const double ratio = options.prefix_mix_ratio <= 1e-9
                                 ? 0.5
                                 : std::clamp(options.prefix_mix_ratio, 0.0,
                                              1.0);
        const size_t gated_cnt = std::max<size_t>(
            16, static_cast<size_t>(std::llround(
                    static_cast<double>(n - warmup) * ratio)));
        const size_t pool_end = std::min<size_t>(n, warmup + gated_cnt);
        if (pool_end <= warmup + 1) {
            return;
        }

        auto jump_level = [&](size_t idx) -> unsigned {
            auto gap = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[idx].second) -
                         static_cast<long long>(src_id)));
            unsigned lv = 0;
            while ((lv + 1) < 31U && (1U << (lv + 1)) <= gap) {
                ++lv;
            }
            return lv;
        };

        std::vector<size_t> left, right;
        left.reserve(pool_end - warmup);
        right.reserve(pool_end - warmup);
        for (size_t i = warmup; i < pool_end; ++i) {
            if (edges[i].second < src_id) {
                left.push_back(i);
            } else {
                right.push_back(i);
            }
        }

        auto tier_reorder = [&](std::vector<size_t>& side) {
            if (side.size() <= 2) {
                return;
            }
            const size_t rep_cap =
                std::max<size_t>(4, std::min<size_t>(
                                        8, static_cast<size_t>(
                                               std::sqrt(static_cast<double>(
                                                             side.size())) +
                                               0.5)));
            phmap::flat_hash_set<unsigned> seen_level;
            seen_level.reserve(rep_cap * 2 + 8);
            std::vector<size_t> primary, secondary;
            primary.reserve(rep_cap);
            secondary.reserve(side.size());
            for (auto idx : side) {
                const auto lv = jump_level(idx);
                if (primary.size() < rep_cap && seen_level.insert(lv).second) {
                    primary.push_back(idx);
                } else {
                    secondary.push_back(idx);
                }
            }
            side.clear();
            side.reserve(primary.size() + secondary.size());
            std::ranges::copy(primary, std::back_inserter(side));
            std::ranges::copy(secondary, std::back_inserter(side));
        };
        tier_reorder(left);
        tier_reorder(right);

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
        }

        size_t li = 0, ri = 0;
        int last_side = -1;
        constexpr size_t kRankTolerance = 4;
        auto emit_left = [&]() {
            out.push_back(edges[left[li++]]);
            last_side = 0;
        };
        auto emit_right = [&]() {
            out.push_back(edges[right[ri++]]);
            last_side = 1;
        };

        while (li < left.size() || ri < right.size()) {
            if (li >= left.size()) {
                emit_right();
                continue;
            }
            if (ri >= right.size()) {
                emit_left();
                continue;
            }
            const size_t left_rank = left[li];
            const size_t right_rank = right[ri];
            int preferred = (left_rank <= right_rank) ? 0 : 1;
            if (last_side == preferred) {
                const size_t pref_rank = (preferred == 0) ? left_rank : right_rank;
                const size_t alt_rank = (preferred == 0) ? right_rank : left_rank;
                if (alt_rank <= pref_rank + kRankTolerance) {
                    preferred = 1 - preferred;
                }
            }
            if (preferred == 0) {
                emit_left();
            } else {
                emit_right();
            }
        }

        for (size_t i = pool_end; i < n; ++i) {
            out.push_back(edges[i]);
        }
        edges.swap(out);
        return;
    }

    if (policy == "balmix") {
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        const double ratio = options.prefix_mix_ratio <= 1e-9
                                 ? 0.5
                                 : std::clamp(options.prefix_mix_ratio, 0.0,
                                              1.0);
        const double alpha =
            options.prefix_score_alpha <= 1e-9 ? 0.35 : options.prefix_score_alpha;
        const size_t gated_cnt = std::max<size_t>(
            16, static_cast<size_t>(std::llround(
                    static_cast<double>(n - warmup) * ratio)));
        const size_t pool_end = std::min<size_t>(n, warmup + gated_cnt);
        if (pool_end <= warmup + 1) {
            return;
        }

        std::vector<size_t> left, right;
        left.reserve(pool_end - warmup);
        right.reserve(pool_end - warmup);
        for (size_t i = warmup; i < pool_end; ++i) {
            if (edges[i].second < src_id) {
                left.push_back(i);
            } else {
                right.push_back(i);
            }
        }

        auto reorder_side = [&](std::vector<size_t>& side) {
            if (side.size() <= 2) {
                return;
            }
            std::vector<size_t> by_gap = side;
            std::ranges::sort(by_gap, [&](size_t a, size_t b) {
                const auto ga = static_cast<unsigned>(
                    std::abs(static_cast<long long>(edges[a].second) -
                             static_cast<long long>(src_id)));
                const auto gb = static_cast<unsigned>(
                    std::abs(static_cast<long long>(edges[b].second) -
                             static_cast<long long>(src_id)));
                if (ga != gb) {
                    return ga < gb;
                }
                return a < b;
            });
            phmap::flat_hash_map<size_t, size_t> gap_rank;
            gap_rank.reserve(by_gap.size() * 2 + 8);
            for (size_t r = 0; r < by_gap.size(); ++r) {
                gap_rank[by_gap[r]] = r;
            }
            std::ranges::sort(side, [&](size_t a, size_t b) {
                const double sa =
                    static_cast<double>(a) + alpha * static_cast<double>(gap_rank[a]);
                const double sb =
                    static_cast<double>(b) + alpha * static_cast<double>(gap_rank[b]);
                if (sa != sb) {
                    return sa < sb;
                }
                return a < b;
            });
        };
        reorder_side(left);
        reorder_side(right);

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
        }

        size_t li = 0, ri = 0;
        int last_side = -1;
        constexpr size_t kRankTolerance = 4;
        auto emit_left = [&]() {
            out.push_back(edges[left[li++]]);
            last_side = 0;
        };
        auto emit_right = [&]() {
            out.push_back(edges[right[ri++]]);
            last_side = 1;
        };

        while (li < left.size() || ri < right.size()) {
            if (li >= left.size()) {
                emit_right();
                continue;
            }
            if (ri >= right.size()) {
                emit_left();
                continue;
            }
            const size_t left_rank = left[li];
            const size_t right_rank = right[ri];
            int preferred = (left_rank <= right_rank) ? 0 : 1;
            if (last_side == preferred) {
                const size_t pref_rank = (preferred == 0) ? left_rank : right_rank;
                const size_t alt_rank = (preferred == 0) ? right_rank : left_rank;
                if (alt_rank <= pref_rank + kRankTolerance) {
                    preferred = 1 - preferred;
                }
            }
            if (preferred == 0) {
                emit_left();
            } else {
                emit_right();
            }
        }

        for (size_t i = pool_end; i < n; ++i) {
            out.push_back(edges[i]);
        }
        edges.swap(out);
        return;
    }

    if (policy == "bridgefirst" || policy == "thinbridge" ||
        policy == "switchband") {
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        const bool thinbridge = (policy == "thinbridge");
        const bool switchband = (policy == "switchband");
        const double ratio = options.prefix_mix_ratio <= 1e-9
                                 ? (switchband ? 0.25
                                               : (thinbridge ? 0.20 : 0.35))
                                 : std::clamp(options.prefix_mix_ratio, 0.0,
                                              1.0);
        const size_t gated_cnt = switchband
                                     ? std::max<size_t>(
                                           24, std::min<size_t>(
                                                   n - warmup,
                                                   std::max<size_t>(
                                                       warmup * 4,
                                                       static_cast<size_t>(
                                                           std::llround(
                                                               static_cast<
                                                                   double>(
                                                                   n - warmup) *
                                                               std::max(
                                                                   0.35,
                                                                   ratio))))))
                                     : std::max<size_t>(
                                           16, static_cast<size_t>(std::llround(
                                                   static_cast<double>(
                                                       n - warmup) *
                                                   std::max(0.5, ratio))));
        const size_t pool_end = std::min<size_t>(n, warmup + gated_cnt);
        if (pool_end <= warmup + 1) {
            return;
        }

        const unsigned bridge_gap_min =
            std::max(1u, options.prefix_jump_min_gap);
        const unsigned switch_gap_lo =
            std::max(1u, options.prefix_jump_min_gap);
        const unsigned switch_gap_hi = std::max(
            switch_gap_lo + 1,
            std::max<unsigned>(switch_gap_lo + 1, options.prefix_jump_min_gap * 8));
        const size_t slots = pool_end - warmup;
        const size_t local_prefix_target =
            thinbridge ? std::min<size_t>(slots, std::max<size_t>(8, warmup))
                       : size_t(0);
        const size_t bridge_target = switchband
                                         ? std::min<size_t>(
                                               slots,
                                               std::max<size_t>(
                                                   4, std::min<size_t>(
                                                          12,
                                                          static_cast<size_t>(
                                                              std::llround(
                                                                  static_cast<
                                                                      double>(
                                                                      slots) *
                                                                  ratio)))))
                                         : std::min<size_t>(
                                               slots - local_prefix_target,
                                               std::max<size_t>(
                                                   thinbridge ? 1 : 2,
                                                   static_cast<size_t>(
                                                       std::llround(
                                                           static_cast<double>(
                                                               slots -
                                                               local_prefix_target) *
                                                           ratio))));

        std::vector<size_t> local_left, local_right, bridge_left, bridge_right;
        local_left.reserve(pool_end - warmup);
        local_right.reserve(pool_end - warmup);
        bridge_left.reserve(pool_end - warmup);
        bridge_right.reserve(pool_end - warmup);

        auto gap_of = [&](size_t idx) -> unsigned {
            return static_cast<unsigned>(std::abs(
                static_cast<long long>(edges[idx].second) -
                static_cast<long long>(src_id)));
        };

        for (size_t i = warmup; i < pool_end; ++i) {
            const auto gap = gap_of(i);
            const bool is_bridge = switchband ? (gap >= switch_gap_lo &&
                                                 gap < switch_gap_hi)
                                              : (gap >= bridge_gap_min);
            if (edges[i].second < src_id) {
                (is_bridge ? bridge_left : local_left).push_back(i);
            } else {
                (is_bridge ? bridge_right : local_right).push_back(i);
            }
        }

        if (switchband) {
            auto switch_cmp = [&](size_t a, size_t b) {
                const auto ga = gap_of(a);
                const auto gb = gap_of(b);
                const auto target_gap = switch_gap_lo;
                const auto da =
                    (ga >= target_gap) ? (ga - target_gap) : (target_gap - ga);
                const auto db =
                    (gb >= target_gap) ? (gb - target_gap) : (target_gap - gb);
                if (da != db) {
                    return da < db;
                }
                if (ga != gb) {
                    return ga < gb;
                }
                return a < b;
            };
            std::ranges::sort(bridge_left, switch_cmp);
            std::ranges::sort(bridge_right, switch_cmp);
        }

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
        }

        size_t lli = 0, lri = 0, bli = 0, bri = 0;
        size_t bridge_used = 0;
        size_t local_prefix_used = 0;
        int last_side = -1;

        auto emit_from = [&](std::vector<size_t>& left, size_t& li,
                             std::vector<size_t>& right, size_t& ri) -> bool {
            if (li >= left.size() && ri >= right.size()) {
                return false;
            }
            if (li >= left.size()) {
                out.push_back(edges[right[ri++]]);
                last_side = 1;
                return true;
            }
            if (ri >= right.size()) {
                out.push_back(edges[left[li++]]);
                last_side = 0;
                return true;
            }
            const size_t left_rank = left[li];
            const size_t right_rank = right[ri];
            int preferred = (left_rank <= right_rank) ? 0 : 1;
            if (last_side == preferred) {
                const size_t pref_rank =
                    (preferred == 0) ? left_rank : right_rank;
                const size_t alt_rank =
                    (preferred == 0) ? right_rank : left_rank;
                if (alt_rank <= pref_rank + 6) {
                    preferred = 1 - preferred;
                }
            }
            if (preferred == 0) {
                out.push_back(edges[left[li++]]);
                last_side = 0;
            } else {
                out.push_back(edges[right[ri++]]);
                last_side = 1;
            }
            return true;
        };

        while (out.size() < pool_end) {
            if (switchband) {
                if (bridge_used < bridge_target) {
                    bool emitted = emit_from(bridge_left, bli, bridge_right, bri);
                    bridge_used += emitted ? 1 : 0;
                    if (emitted) {
                        continue;
                    }
                }
                bool emitted = false;
                if (lli < local_left.size() || lri < local_right.size()) {
                    if (lli >= local_left.size()) {
                        out.push_back(edges[local_right[lri++]]);
                        last_side = 1;
                        emitted = true;
                    } else if (lri >= local_right.size()) {
                        out.push_back(edges[local_left[lli++]]);
                        last_side = 0;
                        emitted = true;
                    } else {
                        const size_t left_rank = local_left[lli];
                        const size_t right_rank = local_right[lri];
                        if (left_rank <= right_rank) {
                            out.push_back(edges[local_left[lli++]]);
                            last_side = 0;
                        } else {
                            out.push_back(edges[local_right[lri++]]);
                            last_side = 1;
                        }
                        emitted = true;
                    }
                }
                if (!emitted) {
                    emitted = emit_from(bridge_left, bli, bridge_right, bri);
                    bridge_used += emitted ? 1 : 0;
                }
                if (!emitted) {
                    break;
                }
                continue;
            }

            const size_t produced = out.size() - warmup;
            if (thinbridge && local_prefix_used < local_prefix_target) {
                bool emitted = emit_from(local_left, lli, local_right, lri);
                if (emitted) {
                    ++local_prefix_used;
                } else {
                    emitted = emit_from(bridge_left, bli, bridge_right, bri);
                    bridge_used += emitted ? 1 : 0;
                }
                if (!emitted) {
                    break;
                }
                continue;
            }

            const size_t tail_pos =
                produced - std::min(produced, local_prefix_target);
            const size_t tail_slots = slots - local_prefix_target;
            const bool want_bridge =
                (bridge_used < bridge_target) && (tail_slots > 0) &&
                (tail_pos * bridge_target < (bridge_used + 1) * tail_slots);
            bool emitted = false;
            if (want_bridge) {
                emitted = emit_from(bridge_left, bli, bridge_right, bri);
                bridge_used += emitted ? 1 : 0;
            }
            if (!emitted) {
                emitted = emit_from(local_left, lli, local_right, lri);
            }
            if (!emitted) {
                emitted = emit_from(bridge_left, bli, bridge_right, bri);
                bridge_used += emitted ? 1 : 0;
            }
            if (!emitted) {
                break;
            }
        }

        for (size_t i = pool_end; i < n; ++i) {
            out.push_back(edges[i]);
        }
        edges.swap(out);
        return;
    }

    if (policy == "balance") {
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        const double ratio = options.prefix_mix_ratio <= 1e-9
                                 ? 0.5
                                 : std::clamp(options.prefix_mix_ratio, 0.0,
                                              1.0);
        const size_t gated_cnt = std::max<size_t>(
            16, static_cast<size_t>(std::llround(
                    static_cast<double>(n - warmup) * ratio)));
        const size_t pool_end = std::min<size_t>(n, warmup + gated_cnt);
        if (pool_end <= warmup + 1) {
            return;
        }

        std::vector<size_t> left, right;
        left.reserve(pool_end - warmup);
        right.reserve(pool_end - warmup);
        for (size_t i = warmup; i < pool_end; ++i) {
            if (edges[i].second < src_id) {
                left.push_back(i);
            } else {
                right.push_back(i);
            }
        }

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
        }

        size_t li = 0, ri = 0;
        int last_side = -1;
        constexpr size_t kRankTolerance = 4;
        auto emit_left = [&]() {
            out.push_back(edges[left[li++]]);
            last_side = 0;
        };
        auto emit_right = [&]() {
            out.push_back(edges[right[ri++]]);
            last_side = 1;
        };

        while (li < left.size() || ri < right.size()) {
            if (li >= left.size()) {
                emit_right();
                continue;
            }
            if (ri >= right.size()) {
                emit_left();
                continue;
            }

            const size_t left_rank = left[li];
            const size_t right_rank = right[ri];
            int preferred = (left_rank <= right_rank) ? 0 : 1;
            if (last_side == preferred) {
                const size_t pref_rank = (preferred == 0) ? left_rank : right_rank;
                const size_t alt_rank = (preferred == 0) ? right_rank : left_rank;
                if (alt_rank <= pref_rank + kRankTolerance) {
                    preferred = 1 - preferred;
                }
            }
            if (preferred == 0) {
                emit_left();
            } else {
                emit_right();
            }
        }

        for (size_t i = pool_end; i < n; ++i) {
            out.push_back(edges[i]);
        }
        edges.swap(out);
        return;
    }

    if (policy == "mix") {
        const double ratio = std::clamp(options.prefix_mix_ratio, 0.0, 1.0);
        if (ratio <= 1e-9) {
            return;
        }
        const size_t n = edges.size();
        std::vector<size_t> by_jump(n);
        std::iota(by_jump.begin(), by_jump.end(), 0);
        std::ranges::sort(by_jump, [&](size_t a, size_t b) {
            const auto ga = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[a].second) -
                         static_cast<long long>(src_id)));
            const auto gb = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[b].second) -
                         static_cast<long long>(src_id)));
            if (ga != gb) {
                return ga > gb;
            }
            return edges[a].first < edges[b].first;
        });

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        std::vector<char> used(n, 0);

        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
            used[i] = 1;
        }

        const double stride_f = (ratio <= 1e-9) ? 1e9 : (1.0 / ratio);
        const size_t stride =
            std::max<size_t>(1, static_cast<size_t>(std::round(stride_f)));
        size_t next_near = warmup;
        size_t next_jump = 0;

        while (out.size() < n) {
            const bool want_jump =
                ((out.size() - warmup) % stride == 0) && (next_jump < n);
            bool inserted = false;

            if (want_jump) {
                while (next_jump < n) {
                    const size_t idx = by_jump[next_jump++];
                    if (used[idx]) {
                        continue;
                    }
                    const auto gap = static_cast<unsigned>(
                        std::abs(static_cast<long long>(edges[idx].second) -
                                 static_cast<long long>(src_id)));
                    if (gap < options.prefix_jump_min_gap) {
                        continue;
                    }
                    out.push_back(edges[idx]);
                    used[idx] = 1;
                    inserted = true;
                    break;
                }
            }

            if (!inserted) {
                while (next_near < n && used[next_near]) {
                    ++next_near;
                }
                if (next_near < n) {
                    out.push_back(edges[next_near]);
                    used[next_near] = 1;
                    ++next_near;
                    inserted = true;
                }
            }

            if (!inserted) {
                for (size_t i = 0; i < n; ++i) {
                    if (!used[i]) {
                        out.push_back(edges[i]);
                        used[i] = 1;
                        break;
                    }
                }
            }
        }
        edges.swap(out);
        return;
    }

    if (policy == "cover") {
        const double ratio = std::clamp(options.prefix_mix_ratio, 0.0, 1.0);
        if (ratio <= 1e-9) {
            return;
        }
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        size_t cover_quota = static_cast<size_t>(
            std::llround(static_cast<double>(n - warmup) * ratio));
        if (cover_quota == 0) {
            return;
        }
        cover_quota = std::min(cover_quota, n - warmup);

        auto jump_level = [](unsigned gap) -> unsigned {
            unsigned lv = 0;
            while ((lv + 1) < 31U && (1U << (lv + 1)) <= gap) {
                ++lv;
            }
            return lv;
        };

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        std::vector<char> used(n, 0);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
            used[i] = 1;
        }

        constexpr unsigned kMaxLevel = 30;
        const size_t bin_count = static_cast<size_t>((kMaxLevel + 1) * 2);
        std::vector<std::vector<size_t>> bins(bin_count);
        unsigned max_level = 0;
        // Quality gate: only inject long-jump coverage from a distance-good pool.
        const size_t pool_end = std::min<size_t>(
            n, warmup + std::max<size_t>(16, cover_quota * 4));
        for (size_t i = warmup; i < pool_end; ++i) {
            const auto gap = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[i].second) -
                         static_cast<long long>(src_id)));
            if (gap < options.prefix_jump_min_gap) {
                continue;
            }
            unsigned lv = jump_level(gap);
            lv = std::min<unsigned>(lv, kMaxLevel);
            const unsigned side = (edges[i].second < src_id) ? 0U : 1U;
            bins[static_cast<size_t>(side * (kMaxLevel + 1) + lv)].push_back(i);
            max_level = std::max(max_level, lv);
        }
        for (auto& bin : bins) {
            std::ranges::sort(bin, [&](size_t a, size_t b) {
                if (edges[a].first != edges[b].first) {
                    return edges[a].first < edges[b].first;
                }
                return a < b;
            });
        }

        std::vector<size_t> bin_ptr(bin_count, 0);
        size_t selected_cover = 0;
        while (selected_cover < cover_quota) {
            bool picked = false;
            for (int lv = static_cast<int>(max_level); lv >= 0; --lv) {
                for (unsigned side = 0; side < 2; ++side) {
                    const size_t b =
                        static_cast<size_t>(side * (kMaxLevel + 1) + lv);
                    auto& ptr = bin_ptr[b];
                    auto& bin = bins[b];
                    while (ptr < bin.size() && used[bin[ptr]]) {
                        ++ptr;
                    }
                    if (ptr >= bin.size()) {
                        continue;
                    }
                    const size_t idx = bin[ptr++];
                    if (used[idx]) {
                        continue;
                    }
                    out.push_back(edges[idx]);
                    used[idx] = 1;
                    ++selected_cover;
                    picked = true;
                    if (selected_cover >= cover_quota) {
                        break;
                    }
                }
                if (selected_cover >= cover_quota) {
                    break;
                }
            }
            if (!picked) {
                break;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            if (!used[i]) {
                out.push_back(edges[i]);
            }
        }
        edges.swap(out);
        return;
    }

    if (policy == "scoreg") {
        // quality-gated score policy:
        // 1) keep nearest warmup untouched
        // 2) re-rank only a distance-good pool by score
        // 3) keep the tail unchanged
        const double alpha = std::max(0.0, options.prefix_score_alpha);
        if (alpha <= 1e-9) {
            return;
        }
        const size_t n = edges.size();
        const size_t warmup = std::min<size_t>(options.prefix_warmup, n);
        const double ratio = options.prefix_mix_ratio <= 1e-9
                                 ? 0.5
                                 : std::clamp(options.prefix_mix_ratio, 0.0,
                                              1.0);
        const size_t gated_cnt = std::max<size_t>(
            16, static_cast<size_t>(std::llround(
                    static_cast<double>(n - warmup) * ratio)));
        const size_t pool_end = std::min<size_t>(n, warmup + gated_cnt);
        if (pool_end <= warmup + 1) {
            return;
        }

        std::vector<size_t> pool;
        pool.reserve(pool_end - warmup);
        for (size_t i = warmup; i < pool_end; ++i) {
            pool.push_back(i);
        }
        std::ranges::sort(pool, [&](size_t a, size_t b) {
            const auto ga = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[a].second) -
                         static_cast<long long>(src_id)));
            const auto gb = static_cast<unsigned>(
                std::abs(static_cast<long long>(edges[b].second) -
                         static_cast<long long>(src_id)));
            if (ga != gb) {
                return ga > gb;
            }
            return a < b;
        });

        std::vector<double> jump_rank(n, 0.0);
        for (size_t r = 0; r < pool.size(); ++r) {
            jump_rank[pool[r]] = static_cast<double>(pool.size() - 1 - r);
        }

        std::vector<size_t> pool_order(pool.size());
        std::iota(pool_order.begin(), pool_order.end(), 0);
        std::ranges::sort(pool_order, [&](size_t ia, size_t ib) {
            const size_t a = pool[ia];
            const size_t b = pool[ib];
            const double sa = static_cast<double>(a) - alpha * jump_rank[a];
            const double sb = static_cast<double>(b) - alpha * jump_rank[b];
            if (sa != sb) {
                return sa < sb;
            }
            return edges[a].first < edges[b].first;
        });

        std::vector<std::pair<T, unsigned>> out;
        out.reserve(n);
        for (size_t i = 0; i < warmup; ++i) {
            out.push_back(edges[i]);
        }
        for (auto idx : pool_order) {
            out.push_back(edges[pool[idx]]);
        }
        for (size_t i = pool_end; i < n; ++i) {
            out.push_back(edges[i]);
        }
        edges.swap(out);
        return;
    }

    // score policy: score = distance_rank - alpha * jump_rank
    // Larger label-gap gets better (smaller) score.
    const double alpha = std::max(0.0, options.prefix_score_alpha);
    if (alpha <= 1e-9) {
        return;
    }
    const size_t n = edges.size();
    std::vector<size_t> by_jump(n);
    std::iota(by_jump.begin(), by_jump.end(), 0);
    std::ranges::sort(by_jump, [&](size_t a, size_t b) {
        const auto ga = static_cast<unsigned>(
            std::abs(static_cast<long long>(edges[a].second) -
                     static_cast<long long>(src_id)));
        const auto gb = static_cast<unsigned>(
            std::abs(static_cast<long long>(edges[b].second) -
                     static_cast<long long>(src_id)));
        if (ga != gb) {
            return ga > gb;
        }
        return a < b;
    });

    std::vector<double> jump_rank(n, 0.0);
    for (size_t r = 0; r < n; ++r) {
        // higher gap should have higher bonus
        jump_rank[by_jump[r]] = static_cast<double>(n - 1 - r);
    }

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](size_t a, size_t b) {
        const double sa =
            static_cast<double>(a) - alpha * jump_rank[a];
        const double sb =
            static_cast<double>(b) - alpha * jump_rank[b];
        if (sa != sb) {
            return sa < sb;
        }
        return edges[a].first < edges[b].first;
    });

    std::vector<std::pair<T, unsigned>> out;
    out.reserve(n);
    for (auto idx : order) {
        out.push_back(edges[idx]);
    }
    edges.swap(out);
}

template <typename T>
void Builder<T>::init_header(Graph::TDGraphIndexBase& g,
                             const Vector::VectorType<T>& center,
                             const std::vector<std::uint64_t>& label,
                             const std::ranges::range auto&& order,
                             const Vector::VectorList<T>& vector_list) const {
    spdlog::info("Init header");
    std::vector<std::pair<T, unsigned>> pre_header;
    std::uint64_t lst_label = std::numeric_limits<std::uint64_t>::max(),
                  header_size = 0, header_cnt = 0;
    for (auto i : order) {
        if (label[i] != lst_label && lst_label != unsigned(-1)) {
            if (pre_header.size() > 20) {
                pre_header.resize(20);
            }
            header_size += pre_header.size(), header_cnt++;
            g.append_header(lst_label,
                            pre_header | std::views::transform(GET(second)));
        }
        auto now = std::pair{vector_list.dist(i, center), i};
        while (!pre_header.empty() && pre_header[0].first >= now.first) {
            pre_header.erase(pre_header.begin());
        }
        pre_header.insert(pre_header.begin(), now);
        lst_label = label[i];
    }
    if (pre_header.size() > 20) {
        pre_header.resize(20);
    }
    header_size += pre_header.size(), header_cnt++;
    g.append_header(lst_label, pre_header | std::views::transform(GET(second)));
    spdlog::info("Init header done, header averange size {}",
                 (double)header_size / header_cnt);
}

template <typename T>
Graph::TDGraphIndexBase Builder<T>::build(
    Graph::GraphLike auto&& knng,
    unsigned range_step, unsigned ef_max,
    const std::vector<std::uint64_t>& label,
    const BuildOptions& options) {
    const int thread_count = std::max(1, std::min(64, omp_get_max_threads()));
    omp_set_num_threads(thread_count);
    spdlog::info("Building TDF Graph Index, index size {}...",
                 vector_list.size());
    spdlog::info(
        "Build options: range_augmentation={}, side_split_pruning={}, "
        "mrng_pruning={}, prefix_policy={}, mix_ratio={:.3f}, warmup={}, "
        "jump_min_gap={}, score_alpha={:.3f}",
        options.enable_range_augmentation, options.enable_side_split_pruning,
        options.use_mrng_pruning, normalize_prefix_policy(options.prefix_policy),
        options.prefix_mix_ratio, options.prefix_warmup,
        options.prefix_jump_min_gap, options.prefix_score_alpha);
    Timer::start("build_time");
    Graph::TDGraphIndexBase g(vector_list.size());
    auto center = vector_list.mean();

    std::vector<unsigned> index, pos;
    std::tie(index, pos) = Utils::order_of_label(label);
    auto sorted_label = Utils::sorted_vec(label);
    auto& dataset = vector_list;
    dataset.reorder(index);
    init_header(g, center, sorted_label, std::views::iota(0ul, label.size()),
                dataset);

    unsigned n = dataset.size();
    const unsigned step = (dataset.size() + 99) / 100;
    std::atomic<unsigned> build_step = 0, total_degree = 0;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (unsigned i = 0; i < dataset.size(); i++) {
        unsigned build_now = build_step.fetch_add(1) + 1;
        bool output_tag = build_now % step == 0 || build_now == dataset.size();

        std::vector<std::pair<T, unsigned>> c_left, c_right;
        auto append_candidate = [&](unsigned neighbour) {
            if (neighbour < i) {
                c_left.push_back({0, neighbour});
            } else if (neighbour > i) {
                c_right.push_back({0, neighbour});
            }
        };

        for (const auto& neighbour :
             knng.get_neighbours_id(index[i]) |
                 std::views::transform([&](unsigned x) { return pos[x]; })) {
            append_candidate(neighbour);
        }
        if (options.enable_range_augmentation) {
            for (unsigned j = i - std::min(i, range_step); j < i; j++) {
                append_candidate(j);
            }
            for (unsigned j = i + 1; j < std::min(i + range_step, n); j++) {
                append_candidate(j);
            }
        }

        std::ranges::sort(c_left, std::greater<std::pair<T, unsigned>>{});
        std::ranges::sort(c_right);
        c_left.erase(std::ranges::unique(c_left).begin(), c_left.end());
        c_right.erase(std::ranges::unique(c_right).begin(), c_right.end());

        auto l_dis =
            dataset.dist_all(i, c_left | std::views::transform(GET(second)));
        auto r_dis =
            dataset.dist_all(i, c_right | std::views::transform(GET(second)));
        for (unsigned j = 0; j < c_left.size(); j++) {
            c_left[j].first = l_dis[j];
        }
        for (unsigned j = 0; j < c_right.size(); j++) {
            c_right[j].first = r_dis[j];
        }

        unsigned candidate_size = c_left.size() + c_right.size();
        if (output_tag) {
            Timer::start("prune");
        }

        std::vector<std::pair<T, unsigned>> final_edges;
        if (options.enable_side_split_pruning && !options.use_mrng_pruning) {
            c_left = prune(dataset, c_left, (ef_max + 1) / 2);
            c_right = prune(dataset, c_right, (ef_max + 1) / 2);

            if (c_left.size() > ef_max / 2) {
                c_left.resize(ef_max / 2);
            }
            c_left.insert(
                c_left.end(), c_right.begin(),
                c_right.begin() +
                    std::min(c_right.size(), (size_t)ef_max - c_left.size()));
            std::ranges::sort(c_left);
            final_edges = std::move(c_left);
        } else {
            std::vector<std::pair<T, unsigned>> c_all;
            c_all.reserve(candidate_size);
            c_all.insert(c_all.end(), c_left.begin(), c_left.end());
            c_all.insert(c_all.end(), c_right.begin(), c_right.end());
            std::ranges::sort(c_all);
            final_edges = prune(dataset, c_all, ef_max);
            if (final_edges.size() > ef_max) {
                final_edges.resize(ef_max);
            }
        }

        // Reorder the final list for prefix quality under low trunc search.
        // This keeps one unified graph while changing only neighbour order.
        reorder_prefix_edges(final_edges, i, options);

        total_degree += final_edges.size();

        if (output_tag) {
            auto t = Timer::end("prune");
            spdlog::info(
                "Build progress: {}/{} ({:.2f}%), prune time cost {}, "
                "candidate size {} -> {}",
                i + 1, dataset.size(), (i + 1) * 100.0 / dataset.size(), t,
                candidate_size, final_edges.size());
        }

        g.add_neighbours(i, std::views::iota(0ul, final_edges.size()) |
                                std::views::transform([&](unsigned x) {
                                    unsigned pid = final_edges[x].second;
                                    return Graph::to_node(pid);
                                }));
    }
    spdlog::info("average degree {:.2f}", total_degree * 1.0 / dataset.size());
    spdlog::info("Build finished.");
    spdlog::info("Index build time: {} s", Timer::end("build_time") / 1e9);
    return g;
}

}  // namespace TDFANN::RNSG
