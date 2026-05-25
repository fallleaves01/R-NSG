#include <PCH.hpp>

#include <EnhancedRNSG/Worker.hpp>
#include <RNSG/Worker.hpp>
#include <Utils/InitFunc.hpp>

using namespace TDFANN;

int main(int argc, char** argv) {
    CLI::App app{"rnsg-new"};

    EnhancedRNSG::Worker worker(app);
    CLI::App query_dummy_app{"rnsg-new-query"};
    RNSG::Worker query_worker(query_dummy_app);
    auto knng_cmd = worker.init_knng(app);
    auto build_cmd = worker.init_build(app);
    auto query_cmd = query_worker.init_query(app);
    auto query_batch_cmd = query_worker.init_query_batch(app);
    auto gt_cmd = worker.init_groundtruth(app);
    CLI11_PARSE(app, argc, argv);
    Utils::setup_logger(worker.verbosed(), "rnsg-new");

    int r;
    if (knng_cmd->parsed() && (r = worker.knng()) != 0) {
        return r;
    }
    if (build_cmd->parsed() && (r = worker.build()) != 0) {
        return r;
    }
    if (query_cmd->parsed() && (r = query_worker.query()) != 0) {
        return r;
    }
    if (query_batch_cmd->parsed() && (r = query_worker.query_batch()) != 0) {
        return r;
    }
    if (gt_cmd->parsed() && (r = worker.gen_groundtruth()) != 0) {
        return r;
    }
    return 0;
}
