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
        return query_cmd;
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
        auto label = IO::load_json_to_vec(label_file);
        auto g = builder.build(*knng_ptr, range_step, label);
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
        auto label = IO::load_json_to_vec(label_file);
        auto qrange = IO::load_json_to_vec(qrange_file);
        std::vector<size_t> ans(query_list.size() * qnumber);
        std::vector<size_t> node_id(label.size());
        std::ranges::sort(node_id, [&](size_t x, size_t y) {
            return std::pair{label[x], x} < std::pair{label[y], y};
        });

        // auto header = index.get_header(vector_list.size() - 1);
        // header.clear();
        // auto center = vector_list.mean();
        // auto dist_center = vector_list.dist_all(
        //     center, std::views::iota(0u, vector_list.size()));

        // auto mn_dis = dist_center[0];
        // header.push_back(0);
        // for (size_t i = 0; i < query_list.size(); i++) {
        //     if (dist_center[i] < mn_dis) {
        //         mn_dis = dist_center[i];
        //         header.clear();
        //     }
        //     if (dist_center[i] == mn_dis) {
        //         header.push_back(i);
        //     }
        // }
        // for (size_t i = 0; i < header.size(); i++) {
        //     std::cout << header[i] << " ";
        // }
        // std::cout << std::endl;
        auto [ord, pos] = Utils::order_of_label(label);
        auto dataset = vector_list;
        dataset.reorder(ord);
        auto sorted_label = Utils::sorted_vec(label);
        assert(dataset[0] == vector_list[ord[0]]);
        Timer::start("Query");
        if (brute) {
            Searcher searcher(vector_list, index);
            for (size_t i = 0; i < query_list.size(); i++) {
                auto result = searcher.linear_search(query_list[i], qnumber);
                std::ranges::copy(result | std::views::transform(GET(second)),
                                  ans.begin() + i * qnumber);
                if (i % 128 == 0) {
                    spdlog::info("Processed {}/{} queries", i,
                                 query_list.size());
                }
            }
        } else {
            for (size_t i = 0; i < query_list.size(); i++) {
                auto g_sub =
                    index(sorted_label, qrange[i * 2], qrange[i * 2 + 1]);
                Searcher searcher(dataset, g_sub);
                auto result = searcher.beam_search(
                    query_list[i], qnumber, g_sub.get_header(), beam_size);
                std::ranges::copy(result | std::views::transform(GET(second)),
                                  ans.begin() + i * qnumber);
                if (i % 1024 == 0) {
                    spdlog::info("Processed {}/{} queries", i,
                                 query_list.size());
                }
            }
            for (auto& i : ans) {
                i = ord[i];
            }
        }
        auto time = Timer::end("Query");
        spdlog::info(
            "average cmps: {:.4f}",
            Recorder<size_t>::read("total_visited") * 1.0 / query_list.size());
        spdlog::info("Average query time: {:.4f} ns",
                     (double)time / query_list.size());
        if (!groundtruth_file.empty()) {
            auto gt = IO::load_json_to_vec(groundtruth_file);
            size_t correct = 0;
            for (size_t i = 0; i < query_list.size(); i++) {
                phmap::flat_hash_map<float, size_t> answer_cnt;
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
                         (double)correct / (qnumber * query_list.size()));
        }
        IO::save(fout, ans);
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
    size_t k, beam_size, range_step, qnumber;
};

}  // namespace TDFANN
