#include <PCH.hpp>

#include <Core/Builder.hpp>
#include <Graph/GraphIndex.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>
using namespace TDFANN;

int main(int argc, char** argv) {
    CLI::App app{"TDFANN"};

    std::string vector_file;
    std::string index_file;
    std::string query_file;
    std::string result_file;
    std::string knng_file;
    bool verbose = false;
    app.add_flag("--verbose", verbose, "Enable verbose logging");

    auto build_cmd = app.add_subcommand("build", "Build the TDF Graph Index");
    build_cmd->add_option("-v,--vector", vector_file, "Path to the vector file")
        ->required();
    build_cmd
        ->add_option("-i,--index", index_file, "Path to the index file to save")
        ->required();
    build_cmd->add_option("-k,--knng", knng_file, "Path to the KNN graph file")
        ->required();

    auto query_cmd = app.add_subcommand(
        "query", "Query nearest neighbors by TDF Graph Index");
    query_cmd->add_option("-i,--index", index_file, "Path to the index file")
        ->required();
    query_cmd->add_option("-q,--query", "Path to the query file")->required();
    query_cmd->add_option("-r,--result", "Path to the result file")->required();

    app.require_subcommand(1);
    CLI11_PARSE(app, argc, argv);

    Utils::setup_logger(verbose, "TDFANN");

    if (build_cmd->parsed()) {
        spdlog::info("Building TDF Graph Index...");
        Vector::VectorList<float> vector_list(vector_file);
        auto builder = Builder(vector_list);
        std::unique_ptr<Graph::GraphIndex<std::monostate>> knng_ptr;
        try {
            knng_ptr =
                std::make_unique<Graph::GraphIndex<std::monostate>>(knng_file);
        } catch (std::exception& e) {
            spdlog::warn(
                "Failed to load KNNG from {}, building new one. Error: {}, building",
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
    } else if (query_cmd->parsed()) {
        spdlog::info("Querying Nearest Neighbors...");
    }
    return 0;
}
