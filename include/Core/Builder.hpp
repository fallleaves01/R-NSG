#pragma once
#include <PCH.hpp>

#include <faiss/IndexNNDescent.h>
#include <Core/Searcher.hpp>
#include <Graph/GraphIndex.hpp>
#include <Utils/Recorder.hpp>
#include <Utils/Timer.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN {

template <typename T>
class Builder {
   public:
    Builder(const Vector::VectorList<T>& data) : vector_list(data) {};
    Graph::GraphIndex<std::monostate> nn_descent(size_t k,
                                                 bool verbose = false) const;
    Graph::TDGraphIndexBase build(Graph::GraphLike auto&& knng,
                                  size_t range_step,
                                  const std::vector<size_t>& label) const;
    // Graph::TDGraphIndexBase build_routing(Graph::GraphLike auto&& knng,
    //                                       size_t d) const;
    void init_header(Graph::TDGraphIndexBase&,
                     const Vector::VectorType<T>&,
                     const std::vector<size_t>& label,
                     const std::vector<size_t>& order) const;

   private:
    bool check_valid(const std::pair<T, size_t>&,
                     const std::vector<std::pair<T, size_t>>&) const;
    std::vector<std::pair<T, size_t>> prune(
        const std::vector<std::pair<T, size_t>>&,
        std::vector<size_t>* = nullptr) const;
    const Vector::VectorList<T>& vector_list;  // 向量列表
};

//>===========================================================<

// Implement of builder functions

template <typename T>
Graph::GraphIndex<std::monostate> Builder<T>::nn_descent(size_t k,
                                                         bool verbose) const {
    Graph::GraphIndex<std::monostate> graph(vector_list.size());
    std::vector<T> data(vector_list.size() * vector_list.dim());
    for (size_t i = 0; i < vector_list.size(); i++) {
        std::ranges::copy(vector_list[i], data.begin() + i * vector_list.dim());
    }
    spdlog::info("start KNN train with size {}, dim {}", vector_list.size(),
                 vector_list.dim());
    faiss::IndexNNDescentFlat index(vector_list.dim(), k);
    index.verbose = verbose;
    index.add(vector_list.size(), data.data());
    for (size_t i = 0; i < vector_list.size(); i++) {
        graph.add_neighbours(
            i, index.nndescent.final_graph | std::views::drop(i * k) |
                   std::views::take(k) |
                   std::views::transform([&](size_t id) { return id; }));
    }
    spdlog::info("KNN train finished.");
    return graph;
}

template <typename T>
void Builder<T>::init_header(Graph::TDGraphIndexBase& g,
                             const Vector::VectorType<T>& center,
                             const std::vector<size_t>& label,
                             const std::vector<size_t>& order) const {
    spdlog::info("Init header");
    std::vector<std::pair<T, size_t>> pre_header;
    size_t lst_label = size_t(-1), header_size = 0, header_cnt = 0;
    for (auto i : order) {
        pre_header.insert(pre_header.begin(), {vector_list.dist(i, center), i});
        auto dnow = pre_header[0].first;
        for (auto it = std::next(pre_header.begin()); it != pre_header.end();) {
            if (it->first > dnow &&
                it->first > vector_list.dist(it->second, i)) {
                it = pre_header.erase(it);
            } else {
                ++it;
            }
        }
        // g.set_header(i, pre_header | std::views::transform(
        //                                  [](auto& x) { return x.second; }));
        if (label[i] != lst_label) {
            // g.append_header(label[i], pre_header | std::views::transform(
            //                              [](auto& x) { return x.second; }));
            // header_size += pre_header.size(), header_cnt += 1;
            g.append_header(label[i], std::array{i});
        }
        lst_label = label[i];
    }
    spdlog::info("Init header done, header averange size {}", (double)header_size / header_cnt);
}

template <typename T>
Graph::TDGraphIndexBase Builder<T>::build(
    Graph::GraphLike auto&& knng,
    size_t range_step,
    const std::vector<size_t>& label) const {
    spdlog::info("Building TDF Graph Index, index size {}...",
                 vector_list.size());
    Graph::TDGraphIndexBase g(vector_list.size());
    auto center = vector_list.mean();

    std::vector<size_t> index(label.size()), pos(label.size());
    std::iota(index.begin(), index.end(), 0);
    std::ranges::sort(index.begin(), index.end(), [&](size_t i, size_t j) {
        return std::pair{label[i], i} < std::pair{label[j], j};
    });
    for (size_t i = 0; i < index.size(); i++) {
        pos[index[i]] = i;
    }
    init_header(g, center, label, index);

    size_t n = vector_list.size(), total_degree = 0;
    const size_t step = (vector_list.size() + 99) / 100;
    for (size_t i = 0; i < vector_list.size(); i++) {
        bool output_tag = (i + 1) % step == 0 || i == vector_list.size();

        std::vector<std::pair<T, size_t>> c_left, c_right;
        for (const auto& neighbour : knng.get_neighbours_id(i)) {
            if (neighbour < i) {
                c_left.push_back({vector_list.dist(i, neighbour), neighbour});
            } else {
                c_right.push_back({vector_list.dist(i, neighbour), neighbour});
            }
        }
        for (size_t j = pos[i] - std::min(pos[i], range_step); j < pos[i];
             j++) {
            c_left.push_back({vector_list.dist(i, index[j]), index[j]});
        }
        for (size_t j = pos[i] + 1; j < std::min(pos[i] + range_step, n); j++) {
            c_right.push_back({vector_list.dist(i, index[j]), index[j]});
        }

        std::ranges::sort(c_left, [&](auto&& x, auto&& y) {
            return pos[x.second] > pos[y.second];
        });
        std::ranges::sort(c_right, [&](auto&& x, auto&& y) {
            return pos[x.second] < pos[y.second];
        });

        c_left.erase(std::begin(std::ranges::unique(c_left)), c_left.end());
        c_right.erase(std::begin(std::ranges::unique(c_right)), c_right.end());

        size_t candidate_size = 0;
        if (output_tag) {
            Timer::start("prune");
            candidate_size = c_left.size() + c_right.size();
        }

        std::vector<size_t> l_bid, r_bid;
        c_left = prune(c_left, &l_bid);
        c_right = prune(c_right, &r_bid);

        total_degree += c_left.size() + c_right.size();

        if (output_tag) {
            auto t = Timer::end("prune");
            auto sub_prune = [&](auto& c) {
                std::vector<bool> tag(c.size());
                int res = 0;
                for (size_t i = 0; i < c.size(); i++) {
                    for (size_t j = i + 1; j < c.size(); j++) {
                        if (c[i].first > c[j].first &&
                            c[i].first >
                                vector_list.dist(c[i].second, c[j].second)) {
                            tag[i] = true;
                            break;
                        }
                    }
                    res += !tag[i];
                }
                return res;
            };
            spdlog::info(
                "Build progress: {}/{} ({:.2f}%), prune time cost {}, "
                "candidate size {} -> {} -> {}",
                i + 1, vector_list.size(), (i + 1) * 100.0 / vector_list.size(),
                t, candidate_size, c_left.size() + c_right.size(),
                sub_prune(c_left) + sub_prune(c_right));
        }

        g.add_neighbours(i, std::views::iota(0ul, c_left.size()) |
                                std::views::transform([&](size_t x) {
                                    size_t pid = c_left[x].second;
                                    return Graph::to_node(pid, label[pid],
                                                          l_bid[x]);
                                }) | std::views::reverse);
        g.add_neighbours(i, std::views::iota(0ul, c_right.size()) |
                                std::views::transform([&](size_t x) {
                                    size_t pid = c_right[x].second;
                                    return Graph::to_node(pid, label[pid],
                                                          r_bid[x]);
                                }) | std::views::reverse);
    }
    spdlog::info("average degree {:.2f}",
                 total_degree * 1.0 / vector_list.size());
    spdlog::info("Build finished.");
    return g;
}

template <typename T>
bool Builder<T>::check_valid(
    const std::pair<T, size_t>& now,
    const std::vector<std::pair<T, size_t>>& result) const {
    auto [d_now, i_now] = now;
    for (auto [d_lst, i_lst] : result) {
        if ((d_now > d_lst && d_now > vector_list.dist(i_now, i_lst)) ||
            i_lst == i_now) {
            return false;
        }
    }
    return true;
}

// template <typename T>
// Graph::TDGraphIndexBase Builder<T>::build_routing(Graph::GraphLike auto&&
// knng,
//                                                   size_t d) const {
//     spdlog::info("Building TDF Graph Index, index size {}...",
//                  vector_list.size());

//     size_t n = vector_list.size(), total_degree = 0;
//     const size_t step = (n + 99) / 100;

//     Graph::TDGraphIndexBase g(n);
//     auto center = vector_list.mean();
//     init_header(g, center);

//     std::vector<std::vector<std::pair<T, size_t>>> c_left(n), c_right(n);
//     for (size_t i = 0; i < vector_list.size(); i++) {
//         bool output_tag = (i + 1) % step == 0 || i == vector_list.size();

//         auto &c_l = c_left[i], &c_r = c_right[i];
//         for (const auto& neighbour : knng.get_neighbours_id(i)) {
//             if (neighbour < i) {
//                 c_l.push_back({vector_list.dist(i, neighbour), neighbour});
//             } else {
//                 c_r.push_back({vector_list.dist(i, neighbour), neighbour});
//             }
//         }
//         for (size_t j = i - std::min(i, d); j < i; j++) {
//             c_l.push_back({vector_list.dist(i, j), j});
//         }
//         for (size_t j = i + 1; j < std::min(i + d, n); j++) {
//             c_r.push_back({vector_list.dist(i, j), j});
//         }

//         std::ranges::sort(
//             c_l, [&](auto&& x, auto&& y) { return x.second > y.second; });
//         std::ranges::sort(
//             c_r, [&](auto&& x, auto&& y) { return x.second < y.second; });

//         c_l.erase(std::begin(std::ranges::unique(c_l)), c_l.end());
//         c_r.erase(std::begin(std::ranges::unique(c_r)), c_r.end());

//         size_t candidate_size = 0;
//         if (output_tag) {
//             Timer::start("prune");
//             candidate_size = c_l.size() + c_r.size();
//         }

//         c_l = prune(c_l);
//         c_r = prune(c_r);

//         total_degree += c_l.size() + c_r.size();

//         if (output_tag) {
//             auto t = Timer::end("prune");
//             spdlog::info(
//                 "Build progress: {}/{} ({:.2f}%), prune time cost {}, "
//                 "candidate size {} -> {}",
//                 i + 1, vector_list.size(), (i + 1) * 100.0 /
//                 vector_list.size(), t, candidate_size, c_l.size() +
//                 c_r.size());
//         }

//         g.add_neighbours(
//             i, std::array{c_l, c_r} | std::views::join |
//                    std::views::transform([&](auto& x) { return x.second; }));
//     }

//     Searcher searcher(vector_list, g);
//     std::vector<std::vector<std::pair<T, size_t>>> routing_path(n);
//     std::vector<size_t> cnt(n + 1, 0);
//     for (size_t i = 0; i < n; i++) {
//         bool output_tag = (i + 1) % step == 0 || i == n;
//         searcher.beam_search(center, 1, i, 20, &routing_path[i]);
//         for (auto [d, id] : routing_path[i]) {
//             ++cnt[id];
//         }
//         if (output_tag) {
//             spdlog::info(
//                 "Routing progress: {}/{} ({:.2f}%), cand = {}, now routing "
//                 "path = {}",
//                 i + 1, n, (i + 1) * 100.0 / n, routing_path[i].size(),
//                 routing_path[i].size());
//         }
//     }

//     spdlog::info("Start reorganize the routing edges...");
//     for (size_t i = 1; i <= n; i++) {
//         cnt[i] += cnt[i - 1];
//     }
//     std::vector<std::pair<T, size_t>> all_routing_path(cnt[n]);
//     for (size_t i = 0; i < n; i++) {
//         for (auto [d, id] : routing_path[i]) {
//             all_routing_path[--cnt[id]] = {d, i};
//         }
//     }
//     routing_path.clear(), routing_path.shrink_to_fit();

//     spdlog::info("Start prunning the routing edges...");
//     size_t total_add = 0;
//     for (size_t i = 0; i < n; i++) {
//         bool output_tag = (i + 1) % 10 == 0 || i == n;
//         std::vector<size_t> valid;
//         if (cnt[i + 1] - cnt[i] > 10000) {
//             spdlog::warn("Warning: node {} has too many routing edges {}", i,
//                          cnt[i + 1] - cnt[i]);
//         }
//         for (size_t j = cnt[i]; j < cnt[i + 1] && j < cnt[i] + 50000; j++) {
//             auto [d, to] = all_routing_path[j];
//             if (to < i && check_valid({d, i}, c_left[i])) {
//                 c_left[i].push_back({d, to}), valid.push_back(to);
//             } else if (to > i && check_valid({d, i}, c_right[i])) {
//                 c_right[i].push_back({d, to}), valid.push_back(to);
//             }
//         }
//         g.add_neighbours(i, valid);
//         total_add += valid.size();

//         if (output_tag) {
//             static auto last_time =
//             std::chrono::high_resolution_clock::now(); auto now =
//             std::chrono::high_resolution_clock::now(); auto duration =
//                 std::chrono::duration_cast<std::chrono::milliseconds>(now -
//                                                                       last_time)
//                     .count();
//             if (duration > 1000 || ((i + 1) % step == 0 || i + 1 == n)) {
//                 last_time = now;
//                 spdlog::info(
//                     "Prune routing progress: {}/{} ({:.2f}%), cand = {},
//                     total " "add "
//                     "{} edges",
//                     i + 1, n, (i + 1) * 100.0 / n, valid.size(), total_add);
//             }
//         }
//     }

//     spdlog::info("average degree {:.2f}", total_degree * 1.0 / n);
//     spdlog::info("Build finished.");
//     return g;
// }

template <typename T>
std::vector<std::pair<T, size_t>> Builder<T>::prune(
    const std::vector<std::pair<T, size_t>>& candidates,
    std::vector<size_t>* tag) const {
    std::vector<std::pair<T, size_t>> result;
    if (tag == nullptr) {
        for (size_t i = 0; i < candidates.size(); i++) {
            if (check_valid(candidates[i], result)) {
                result.push_back(candidates[i]);
            }
        }
    } else {
        auto& suf = *tag;
        suf.clear();
        for (size_t i = 0; i < candidates.size(); i++) {
            auto [d_now, i_now] = candidates[i];
            bool append = true;
            for (size_t j = 0; j < result.size(); j++) {
                auto [d_lst, i_lst] = result[j];
                if (d_now > d_lst || suf[j] == size_t(-1)) {
                    auto d_ij = vector_list.dist(i_now, i_lst);
                    if (d_now > d_lst && d_now > d_ij) {
                        append = false;
                        break;
                    }
                    if (suf[j] == size_t(-1) && d_lst > d_now && d_lst > d_ij) {
                        suf[j] = i_now;
                    }
                }
            }
            if (!append) {
                for (auto& x : suf) {
                    if (x == i_now) {
                        x = size_t(-1);
                    }
                }
            } else {
                result.push_back(candidates[i]);
                suf.push_back(size_t(-1));
            }
        }
    }
    return result;
}

}  // namespace TDFANN
