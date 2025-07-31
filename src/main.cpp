#include <PCH.hpp>
#include <Utils/InitFunc.hpp>
#include "Vector/VectorList.hpp"
using namespace TDFANN;

int main(int argc, char** argv) {

    CLI::App app{"TDFANN"};

    std::string vector_file;
    std::string index_file;
    std::string query_file;
    std::string result_file;
    bool verbose = false;
    app.add_flag("--verbose", verbose, "Enable verbose logging");

    auto build_cmd = app.add_subcommand("build", "Build the TDF Graph Index");
    build_cmd->add_option("-v,--vector", vector_file, "Path to the vector file")
        ->required();
    build_cmd
        ->add_option("-i,--index", index_file, "Path to the index file to save")
        ->required();

    auto query_cmd = app.add_subcommand(
        "query", "Query nearest neighbors by TDF Graph Index");
    query_cmd->add_option("-i,--index", index_file, "Path to the index file")
        ->required();
    query_cmd->add_option("-q,--query", "Path to the query file")->required();
    query_cmd->add_option("-r,--result", "Path to the result file")->required();

    app.require_subcommand(1);
    CLI11_PARSE(app, argc, argv);

    
    Init::setup_logger(verbose);

    if (build_cmd->parsed()) {
        spdlog::info("Building TDF Graph Index...");
        VectorList<float> vector_list(vector_file);
        
    } else if (query_cmd->parsed()) {
        spdlog::info("Querying Nearest Neighbors...");
    }
    return 0;
}
