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
    Builder(Vector::VectorList<T>& data) : vector_list(data) {};
    Graph::GraphIndex<std::monostate> nn_descent(unsigned k,
                                                 bool verbose = false) const;

    Graph::TDGraphIndexBase build(Graph::GraphLike auto&& knng,
                                  unsigned range_step, unsigned ef_max,
                                  const std::vector<std::uint64_t>& label);

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

template <typename T>
Graph::GraphIndex<std::monostate> Builder<T>::nn_descent(unsigned k,
                                                         bool verbose) const {
    omp_set_num_threads(64);
    Timer::start("knng_time");
    Graph::GraphIndex<std::monostate> graph(vector_list.size());
    spdlog::info("start KNN train with size {}, dim {}", vector_list.size(),
                 vector_list.dim());
    faiss::IndexNNDescentFlat index(vector_list.dim(), k);
    index.nndescent.iter = 15;
    index.verbose = verbose;
    index.add(vector_list.size(), vector_list.data());
    spdlog::info("KNNG done, size = {}", index.nndescent.final_graph.size());
    size_t step = vector_list.size() / 10;
    for (size_t i = 0; i < vector_list.size(); i++) {
        if (i % step == 0) {
            spdlog::info("Processing vector {}/{}, range = [{}, {}]", i, vector_list.size(), i * k, (i + 1) * k);
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

template <typename T>
void Builder<T>::init_header(Graph::TDGraphIndexBase& g,
                             const Vector::VectorType<T>& center,
                             const std::vector<std::uint64_t>& label,
                             const std::ranges::range auto&& order,
                             const Vector::VectorList<T>& vector_list) const {
    spdlog::info("Init header");
    std::vector<std::pair<T, unsigned>> pre_header;
    std::uint64_t lst_label = std::numeric_limits<std::uint64_t>::max(), header_size = 0, header_cnt = 0;
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
    const std::vector<std::uint64_t>& label) {
    omp_set_num_threads(64);
    spdlog::info("Building TDF Graph Index, index size {}...",
                 vector_list.size());
    Timer::start("build_time");
    Graph::TDGraphIndexBase g(vector_list.size());
    auto center = vector_list.mean();

    std::vector<unsigned> index, pos;
    std::tie(index, pos) = Utils::order_of_label(label);
    auto sorted_label = Utils::sorted_vec(label);
    auto &dataset = vector_list;
    dataset.reorder(index);
    init_header(g, center, sorted_label, std::views::iota(0ul, label.size()),
                dataset);

    unsigned n = dataset.size();
    const unsigned step = (dataset.size() + 99) / 100;
    std::atomic<unsigned> build_step = 0, total_degree = 0;
#pragma omp parallel for num_threads(64) schedule(dynamic)
    for (unsigned i = 0; i < dataset.size(); i++) {
        unsigned build_now = build_step.fetch_add(1) + 1;
        bool output_tag = build_now % step == 0 || build_now == dataset.size();

        std::vector<std::pair<T, unsigned>> c_left, c_right;
        for (const auto& neighbour :
             knng.get_neighbours_id(index[i]) |
                 std::views::transform([&](unsigned x) { return pos[x]; })) {
            if (neighbour < i) {
                c_left.push_back({0, neighbour});
            } else {
                c_right.push_back({0, neighbour});
            }
        }
        for (unsigned j = i - std::min(i, range_step); j < i; j++) {
            c_left.push_back({0, j});
        }
        for (unsigned j = i + 1; j < std::min(i + range_step, n); j++) {
            c_right.push_back({0, j});
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

        // if (c_left.size() > 300) {
        //     std::nth_element(l_dis.begin(), l_dis.begin() + l_dis.size() / 4, l_dis.end());
        //     T lim = l_dis[l_dis.size() / 4];
        //     auto it = std::remove_if(c_left.begin() + 300, c_left.end(), [&](auto &&x) { return x.first > lim; });
        //     c_left.erase(it, c_left.end());
        // }
        // if (c_right.size() > 300) {
        //     std::nth_element(r_dis.begin(), r_dis.begin() + r_dis.size() / 4, r_dis.end());
        //     T lim = r_dis[r_dis.size() / 4];
        //     auto it = std::remove_if(c_right.begin() + 300, c_right.end(), [&](auto &&x) { return x.first > lim; });
        //     c_right.erase(it, c_right.end());
        // }

        unsigned candidate_size = 0;
        if (output_tag) {
            Timer::start("prune");
            candidate_size = c_left.size() + c_right.size();
        }

        // std::vector<unsigned> l_bid, r_bid;
        c_left = prune(dataset, c_left, (ef_max + 1) / 2);
        c_right = prune(dataset, c_right, (ef_max + 1) / 2);

        total_degree += std::min(c_left.size() + c_right.size(), (size_t)ef_max);

        if (output_tag) {
            auto t = Timer::end("prune");
            spdlog::info(
                "Build progress: {}/{} ({:.2f}%), prune time cost {}, "
                "candidate size {} -> {}",
                i + 1, dataset.size(), (i + 1) * 100.0 / dataset.size(), t,
                candidate_size, c_left.size() + c_right.size());
        }
        if (c_left.size() > ef_max / 2) {
            c_left.resize(ef_max / 2);
        }
        c_left.insert(c_left.end(), c_right.begin(), c_right.begin() + std::min(c_right.size(), (size_t)ef_max - c_left.size()));
        std::ranges::sort(c_left);

        g.add_neighbours(i, std::views::iota(0ul, c_left.size()) |
                                std::views::transform([&](unsigned x) {
                                    unsigned pid = c_left[x].second;
                                    return Graph::to_node(pid);
                                }));
    }
    spdlog::info("average degree {:.2f}", total_degree * 1.0 / dataset.size());
    spdlog::info("Build finished.");
    spdlog::info("Index build time: {} s", Timer::end("build_time") / 1e9);
    return g;
}

}  // namespace TDFANN
