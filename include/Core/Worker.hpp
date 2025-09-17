#pragma once

#include <PCH.hpp>

#include <Core/Builder.hpp>
#include <Graph/GraphIndex.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>

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
            ->add_option("-v,--vector", vector_file, "Path to the vector file")
            ->required();
        knng_cmd->add_option("-k,--k", k, "K value for KNNG")->required();
        knng_cmd
            ->add_option("-g,--graph", knng_file,
                         "Path to the graph file to save knng")
            ->required();
        return knng_cmd;
    }

    auto init_build(CLI::App& app) {
        auto build_cmd =
            app.add_subcommand("build", "Build the TDF Graph Index");
        build_cmd
            ->add_option("-v,--vector", vector_file, "Path to the vector file")
            ->required();
        build_cmd
            ->add_option("-i,--index", index_file, "Path to save the index")
            ->required();
        build_cmd->add_option("-k,--knng", knng_file, "Path to the KNNG file")
            ->required();
        return build_cmd;
    }

    auto init_query(CLI::App& app) {
        auto query_cmd = app.add_subcommand(
            "query", "Query nearest neighbors by TDF Graph Index");
        query_cmd->add_flag("-l,--linear", brute,
                            "Use brute-force linear search instead of index");
        query_cmd
            ->add_option("-v,--vector", vector_file, "Path to the vector file")
            ->required();
        query_cmd
            ->add_option("-i,--index", index_file, "Path to the index file")
            ->required();
        query_cmd
            ->add_option("-q,--query", query_file, "Path to the query file")
            ->required();
        query_cmd->add_option("-n,--number", k, "Number of nearest neighbors")
            ->required();
        query_cmd->add_option("-b,--beam", beam_size, "beam size")->required();
        query_cmd
            ->add_option("-r,--result", result_file, "Path to the result file")
            ->required();
        query_cmd->add_option("-a,--answer", answer_file,
                              "Path to the answer file");
        return query_cmd;
    }

    int knng() {
        spdlog::info("Building KNN Graph...");
        Vector::VectorList<float> vector_list(vector_file);
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
        Vector::VectorList<float> vector_list(vector_file);
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
        auto g = builder.build(*knng_ptr, 200);
        std::ofstream fout(index_file);
        if (!fout.good() || !g.save(fout)) {
            spdlog::error("Failed to save index to {}", index_file);
            return 1;
        }
        return 0;
    }

    int query() {
        spdlog::info("Querying Nearest Neighbors...");
        Vector::VectorList<float> vector_list(vector_file);
        Graph::TDGraphIndexBase index(index_file);
        Searcher searcher(vector_list, index);
        Vector::VectorList<float> query_list(query_file);
        std::ofstream fout(result_file);
        if (!fout.good()) {
            spdlog::error("Failed to open result file {}", result_file);
            return 1;
        }
        std::vector<size_t> ans(query_list.size() * k);

        auto header = index.get_header(vector_list.size() - 1);
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
        Timer::start("Query");
        if (brute) {
            for (size_t i = 0; i < query_list.size(); i++) {
                auto result = searcher.linear_search(query_list[i], k);
                std::ranges::copy(result | std::views::transform([](auto& p) {
                                      return p.second;
                                  }),
                                  ans.begin() + i * k);
                if (i % 128 == 0) {
                    spdlog::info("Processed {}/{} queries", i,
                                 query_list.size());
                }
            }
        } else {
            for (size_t i = 0; i < query_list.size(); i++) {
                auto result =
                    searcher.beam_search(query_list[i], k, header, beam_size);
                std::ranges::copy(result | std::views::transform([](auto& p) {
                                      return p.second;
                                  }),
                                  ans.begin() + i * k);
                if (i % 1024 == 0) {
                    spdlog::info("Processed {}/{} queries", i,
                                 query_list.size());
                }
            }
        }
        auto time = Timer::end("Query");
        spdlog::info("Average query time: {:.4f} ns",
                     (double)time / query_list.size());
        if (!answer_file.empty()) {
            std::ifstream fin(answer_file);
            if (!fin.good()) {
                spdlog::error("Failed to open answer file {}", answer_file);
                return 1;
            }
            std::vector<size_t> answer(query_list.size() * k);
            IO::load(fin, answer);
            size_t correct = 0;
            for (size_t i = 0; i < query_list.size(); i++) {
                auto answer_r =
                    answer | std::views::drop(i * k) | std::views::take(k);
                for (size_t j = 0; j < k; j++) {
                    if (std::ranges::find(answer_r, ans[i * k + j]) !=
                        answer_r.end()) {
                        correct++;
                    }
                }
            }
            spdlog::info("Recall: {:.4f}",
                         (double)correct / (k * query_list.size()));
        }
        IO::save(fout, ans);
        return 0;
    }

   private:
    bool verbose, brute = false;
    std::string vector_file;
    std::string index_file;
    std::string query_file;
    std::string result_file;
    std::string answer_file;
    std::string knng_file;
    size_t k, beam_size;
};

}  // namespace TDFANN