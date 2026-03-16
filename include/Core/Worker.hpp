#pragma once

#include <PCH.hpp>

#include <Core/Builder.hpp>
#include <Graph/GraphIndex.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>
#include "IO/TypeIO.hpp"

namespace TDFANN {

class Worker {
   public:
    Worker(CLI::App& app) {
        app.add_flag("--verbose", verbose, "Enable verbose logging");
        app.require_subcommand(1);
    }

    bool verbosed() { return verbose; }

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
        return query_cmd;
    }

    auto init_groundtruth(CLI::App& app) {
        auto gt_cmd = app.add_subcommand(
            "groundtruth", "Query nearest neighbors by brute-force for groundtruth");
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
        return gt_cmd;
    }

    int knng() {
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
        auto g = builder.build(*knng_ptr, range_step, ef_max, label);
        std::ofstream fout(index_file);
        if (!fout.good() || !g.save(fout)) {
            spdlog::error("Failed to save index to {}", index_file);
            return 1;
        }
        return 0;
    }

    int query() {
        spdlog::info("Querying Nearest Neighbors...");
        Vector::VectorList<float> vector_list(dataset_file);
        Graph::TDGraphIndexBase index(index_file);
        // Searcher searcher(vector_list, index);
        Vector::VectorList<float> query_list(query_file);
        std::ofstream fout(result_file);
        if (!fout.good()) {
            spdlog::error("Failed to open result file {}", result_file);
            return 1;
        }
        auto label = IO::load_json_to_vec<std::uint64_t>(label_file);
        auto qrange = IO::load_json_to_vec<std::uint64_t>(qrange_file);
        std::vector<unsigned> ans((size_t)query_list.size() * qnumber);
        std::vector<unsigned> node_id(label.size());
        std::ranges::sort(node_id, [&](unsigned x, unsigned y) {
            return std::pair{label[x], x} < std::pair{label[y], y};
        });

        // auto a = vector_list[0], b = vector_list[1];
        // const float *a_ptr = a.data(), *b_ptr = b.data();
        // int dim = vector_list.dim();
        // using VecA = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
        // 1>,
        //                         Eigen::Aligned64>;
        // float fulla = VecA(a_ptr, dim).squaredNorm();
        // float fullb = VecA(b_ptr, dim).squaredNorm();
        // std::vector<float> r_a(4), r_b(4);
        // for (int i = 0; i < 4; i++) {
        //     r_a[i] = VecA(a_ptr + i * 32, 32).squaredNorm();
        //     r_b[i] = VecA(b_ptr + i * 32, 32).squaredNorm();
        // }
        // float resb = 0, resa = 0;
        // size_t ta = 0, tb = 0;
        // for (int x = 0; x < 1000000; x++) {
        //     auto start = std::chrono::high_resolution_clock::now();
        //     for (int i = 0; i < 4; i++) {
        //         resb += r_a[i] + r_b[i] -
        //                 VecA(a_ptr + i * 32, 32).dot(VecA(b_ptr + i * 32,
        //                 32));
        //     }
        //     auto midx = std::chrono::high_resolution_clock::now();
        //     tb += std::chrono::duration_cast<std::chrono::nanoseconds>(midx -
        //                                                                start)
        //               .count();
        //     auto mid = std::chrono::high_resolution_clock::now();
        //     resa += fulla + fullb - VecA(a_ptr, dim).dot(VecA(b_ptr, dim));
        //     auto end = std::chrono::high_resolution_clock::now();
        //     ta +=
        //         std::chrono::duration_cast<std::chrono::nanoseconds>(end -
        //         mid)
        //             .count();
        // }
        // spdlog::info("Dot product time: {} ns", ta);
        // spdlog::info("32-dim dot product time: {} ns", tb);
        // spdlog::info("Result: {} {}", resa, resb);
        // exit(0);

        auto [ord, pos] = Utils::order_of_label(label);
        auto& dataset = vector_list;
        auto sorted_label = Utils::sorted_vec(label);
        if (brute) {
            Timer::start("Query");
#pragma omp parallel for num_threads(64) schedule(dynamic)
            for (unsigned i = 0; i < query_list.size(); i++) {
                std::priority_queue<std::pair<float, unsigned>> hp;
                auto l = qrange[i * 2], r = qrange[i * 2 + 1];
                for (unsigned j = 0; j < dataset.size(); j++) {
                    if (label[j] >= l && label[j] <= r) {
                        auto now =
                            std::pair{dataset.dist2(j, query_list[i]), j};
                        if (hp.size() < qnumber) {
                            hp.push(now);
                        } else if (now < hp.top()) {
                            hp.pop();
                            hp.push(now);
                        }
                    }
                }
                for (size_t j = 0; j < qnumber; j++) {
                    ans[i * qnumber + j] = pos[hp.top().second];
                    hp.pop();
                }
                if (i % 1024 == 0) {
                    spdlog::info("Processed {}/{} queries", i,
                                 query_list.size());
                }
            }
        } else {
            dataset.reorder(ord);
            Timer::start("Query");
            for (size_t i = 0; i < query_list.size(); i++) {
                auto g_sub =
                    index(sorted_label, qrange[i * 2], qrange[i * 2 + 1]);
                Searcher searcher(dataset, g_sub);
                auto result = searcher.beam_search(query_list[i], qnumber,
                                                   // it,
                                                   g_sub.get_header(),
                                                   beam_size, trunc_size);
                std::ranges::copy(result | std::views::transform(GET(second)),
                                  ans.begin() + i * qnumber);
                if (i % 1024 == 0) {
                    spdlog::info("Processed {}/{} queries", i,
                                 query_list.size());
                }
            }
        }
        auto time = Timer::end("Query");
        spdlog::info("Average query time: {:.4f} ns",
                     (double)time / query_list.size());
        spdlog::info("QPS: {:.4f}", query_list.size() * 1e9 / time);
        if (!groundtruth_file.empty()) {
            auto gt = IO::load_json_to_vec(groundtruth_file);
            for (auto& i : gt) {
                i = pos[i];
            }
            unsigned correct = 0;
            for (size_t i = 0; i < query_list.size(); i++) {
                phmap::flat_hash_map<float, unsigned> answer_cnt;
                for (size_t j = i * qnumber; j < (i + 1) * qnumber; j++) {
                    answer_cnt[vector_list.dist2(ans[j], query_list[i])]++;
                }
                for (size_t j = i * qnumber; j < (i + 1) * qnumber; j++) {
                    auto now = vector_list.dist2(gt[j], query_list[i]);
                    if (answer_cnt[now] > 0) {
                        correct++;
                        answer_cnt[now]--;
                    }
                }
            }
            spdlog::info("Recall: {:.4f}",
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
#pragma omp parallel for num_threads(64) schedule(dynamic)
        for (unsigned i = 0; i < query_list.size(); i++) {
            std::priority_queue<std::pair<float, unsigned>> hp;
            auto l = qrange[i * 2], r = qrange[i * 2 + 1];
            for (unsigned j = 0; j < dataset.size(); j++) {
                if (label[j] >= l && label[j] <= r) {
                    auto now =
                        std::pair{dataset.dist2(j, query_list[i]), j};
                    if (hp.size() < qnumber) {
                        hp.push(now);
                    } else if (now < hp.top()) {
                        hp.pop();
                        hp.push(now);
                    }
                }
            }
            for (size_t j = 0; j < qnumber; j++) {
                ans[i * qnumber + j] = hp.top().second;
                hp.pop();
            }
            if (i % 1024 == 0) {
                spdlog::info("Processed {}/{} queries", i,
                                query_list.size());
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
    bool verbose, brute = false;
    std::string dataset_file;
    std::string index_file;
    std::string query_file;
    std::string result_file;
    std::string groundtruth_file;
    std::string knng_file;
    std::string qrange_file;
    std::string label_file;
    unsigned k, beam_size, range_step, qnumber, trunc_size, ef_max;
};

}  // namespace TDFANN
