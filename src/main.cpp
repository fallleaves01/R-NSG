#include <PCH.hpp>

#include <Utils/InitFunc.hpp>
#include <Core/Worker.hpp>
using namespace TDFANN;

int main(int argc, char** argv) {
    CLI::App app{"TDFANN"};

    Worker worker(app);
    auto knng_cmd = worker.init_knng(app);
    auto build_cmd = worker.init_build(app);
    auto query_cmd = worker.init_query(app);
    CLI11_PARSE(app, argc, argv);
    Utils::setup_logger(worker.verbosed(), "TDFANN");

    int r;
    if (knng_cmd->parsed() && (r = worker.knng()) != 0) {
        return r;
    }
    if (build_cmd->parsed() && (r = worker.build()) != 0) {
        return r;
    }
    if (query_cmd->parsed() && (r = worker.query()) != 0) {
        return r;
    }
    return 0;
}
