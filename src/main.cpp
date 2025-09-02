#include <PCH.hpp>
#include <Utils/InitFunc.hpp>
#include <Core/Builder.hpp>
#include <Core/NN_Descent.hpp>
#include <Vector/VectorList.hpp>
#include <chrono>
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

    
    Utils::setup_logger(verbose);

    if (build_cmd->parsed()) {
        spdlog::info("Building TDF Graph Index...");
        Vector::VectorList<float> vector_list(vector_file);
        auto start_time = std::chrono::high_resolution_clock::now();
        auto g = KNNG::nn_descent(vector_list, 5);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Built KNN Graph Index in {} ms", duration.count());
        // Builder builder(vector_list);
        // auto g = builder.build();
        std::ofstream fout(index_file);
        if (!fout.good() || !g.save(fout)) {
            spdlog::error("Failed to save index to {}", index_file);
            return 1;
        }
        int x = 99853;
        Searcher searcher(vector_list, g);
        auto r = searcher.linear_search(x, 5);
        std::cout << "Nearest nodes: ";
        for (auto [d, v] : r) {
            std::cout << v << ' ';
        }
        std::cout << "\nNearest neighbours: ";
        for (auto d : g.get_neighbours_id(x)) {
            std::cout << d << ' ';
        }
        std::cout << '\n';
    } else if (query_cmd->parsed()) {
        spdlog::info("Querying Nearest Neighbors...");
    }
    return 0;
}
