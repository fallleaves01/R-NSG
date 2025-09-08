#include <PCH.hpp>

#include <Core/Builder.hpp>
#include <Utils/InitFunc.hpp>
#include <Vector/VectorList.hpp>
using namespace TDFANN;

int main(int argc, char** argv) {
    CLI::App app{"KNNG Builder"};

    std::string vector_file;
    std::string knng_file;
    size_t k;
    bool verbose = false;

    app.add_flag("--verbose", verbose, "Enable verbose logging");
    app.add_option("-v,--vector", vector_file, "Path to the vector file")
        ->required();
    app.add_option("-k,--k", k, "K value for KNNG")->required();
    app.add_option("-g,--graph", knng_file,
                   "Path to the graph file to save knng")
        ->required();

    CLI11_PARSE(app, argc, argv);

    Utils::setup_logger(verbose, "knng_builder");

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