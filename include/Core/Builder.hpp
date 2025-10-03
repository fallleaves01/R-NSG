#pragma once
#include <PCH.hpp>

#include <faiss/IndexNNDescent.h>
#include <omp.h>
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
                     const std::vector<size_t>& order,
                     const Vector::VectorList<T>& vector_list) const;

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
                             const std::vector<size_t>& order, 
                             const Vector::VectorList<T> &vector_list) const {
    spdlog::info("Init header");
    std::vector<std::pair<T, size_t>> pre_header;
    size_t lst_label = size_t(-1), header_size = 0, header_cnt = 0;
    for (auto i : order) {
        auto now = std::pair{vector_list.dist(i, center), i};
        while (!pre_header.empty() && pre_header[0].first >= now.first) {
            pre_header.erase(pre_header.begin());
        }
        pre_header.insert(pre_header.begin(), now);
        if (label[i] != lst_label) {
            if (pre_header.size() > 20) {
                pre_header.resize(20);
            }
            header_size += pre_header.size(), header_cnt++;
            g.append_header(
                label[i], pre_header | std::views::transform(
                                           [](auto&& x) { return x.second; }));
        }
        lst_label = label[i];
    }
    spdlog::info("Init header done, header averange size {}",
                 (double)header_size / header_cnt);
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
    init_header(g, center, label, index, vector_list);

    size_t n = vector_list.size();
    const size_t step = (vector_list.size() + 99) / 100;
    std::atomic<size_t> build_step = 0, total_degree = 0;

#pragma omp parallel for num_threads(32) schedule(dynamic)
    for (size_t i = 0; i < vector_list.size(); i++) {
        size_t build_now = build_step.fetch_add(1) + 1;
        bool output_tag =
            build_now % step == 0 || build_now == vector_list.size();

        std::vector<std::pair<T, size_t>> c_left, c_right;
        for (const auto& neighbour : knng.get_neighbours_id(i)) {
            if (pos[neighbour] < pos[i]) {
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

        // std::vector<size_t> l_bid, r_bid;
        c_left = prune(c_left);
        c_right = prune(c_right);

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
                                    return Graph::to_node(pid, label[pid]);
                                }) |
                                std::views::reverse);
        g.add_neighbours(i, std::views::iota(0ul, c_right.size()) |
                                std::views::transform([&](size_t x) {
                                    size_t pid = c_right[x].second;
                                    return Graph::to_node(pid, label[pid]);
                                }) |
                                std::views::reverse);
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
