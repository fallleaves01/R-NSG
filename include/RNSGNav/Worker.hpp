#pragma once

#include <PCH.hpp>

#include <Core/Builder.hpp>
#include <Graph/GraphIndex.hpp>
#include <IO/TypeIO.hpp>
#include <RNSG/Searcher.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>
#include <omp.h>

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

        if (brute) {
            Timer::start("Query");
            for (size_t i = 0; i < query_list.size(); i++) {
                auto ui = static_cast<unsigned>(i);
                std::priority_queue<std::pair<float, unsigned>> hp;
                auto l = qrange[ui * 2], r = qrange[ui * 2 + 1];
                for (unsigned j = 0; j < dataset.size(); j++) {
                    if (label[j] >= l && label[j] <= r) {
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
                if (((i + 1) % 1024 == 0) || (i + 1 == query_list.size())) {
                    spdlog::info("Processed {}/{} queries", i + 1,
                                 query_list.size());
                }
            }
        } else {
            dataset.reorder(ord);
            BeamScratch<float> scratch(dataset.size(), beam_size, trunc_size);
            Timer::start("Query");
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

                auto g_sub = index(sorted_label, ql, qr);
                Searcher<float, decltype(g_sub)> searcher(dataset, g_sub);

                auto header = Utils::to_vector(g_sub.get_header());
                std::vector<std::pair<float, unsigned>> result;
                if (!header.empty()) {
                    result = searcher.beam_search(query_list[ui], qnumber,
                                                  header, beam_size,
                                                  trunc_size, scratch);
                } else {
                    auto fallback = static_cast<unsigned>(range_l);
                    result = searcher.beam_search(query_list[ui], qnumber,
                                                  fallback, beam_size,
                                                  trunc_size, scratch);
                }

                std::ranges::copy(result | std::views::transform(GET(second)),
                                  ans.begin() + ui * qnumber);

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
            auto gt = IO::load_json_to_vec(groundtruth_file);
            for (auto& i : gt) {
                i = pos[i];
            }
            unsigned correct = 0;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic) reduction(+ : correct)
            for (int64_t i = 0; i < static_cast<int64_t>(query_list.size()); i++) {
                auto ui = static_cast<unsigned>(i);
                phmap::flat_hash_map<float, unsigned> answer_cnt;
                for (size_t j = ui * qnumber; j < (ui + 1) * qnumber; j++) {
                    answer_cnt[vector_list.dist2(ans[j], query_list[ui])]++;
                }
                unsigned local_correct = 0;
                for (size_t j = ui * qnumber; j < (ui + 1) * qnumber; j++) {
                    auto now = vector_list.dist2(gt[j], query_list[ui]);
                    if (answer_cnt[now] > 0) {
                        local_correct++;
                        answer_cnt[now]--;
                    }
                }
                correct += local_correct;
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
        const int thread_count = runtime_thread_count();
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
        for (int64_t i = 0; i < static_cast<int64_t>(query_list.size()); i++) {
            auto ui = static_cast<unsigned>(i);
            std::priority_queue<std::pair<float, unsigned>> hp;
            auto l = qrange[ui * 2], r = qrange[ui * 2 + 1];
            for (unsigned j = 0; j < dataset.size(); j++) {
                if (label[j] >= l && label[j] <= r) {
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
                ans[ui * qnumber + j] = hp.top().second;
                hp.pop();
            }
            if ((ui & 1023u) == 0) {
                spdlog::info("Processed {}/{} queries", ui,
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
    static int runtime_thread_count() {
        return std::max(1, std::min(64, omp_get_max_threads()));
    }

    bool verbose = false;
    bool brute = false;
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
};

}  // namespace TDFANN::RNSG
