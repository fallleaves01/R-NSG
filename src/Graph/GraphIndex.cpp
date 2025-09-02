#include <Graph/GraphIndex.hpp>

namespace TDFANN {

namespace Graph {

TagGraphIndex::TagGraphIndex(size_t node_cnt) : edges(node_cnt) {}

size_t TagGraphIndex::add_node() {
    edges.emplace_back();
    return edges.size() - 1;
}

bool TagGraphIndex::save(std::ofstream& fout) const {
    return IO::save(fout, edges);

}

bool TagGraphIndex::load(std::ifstream& fin) {
    return IO::load(fin, edges);
}

bool TDGraphIndexBase::save(std::ofstream& fout) const {
    return left.save(fout) && right.save(fout);
}

bool TDGraphIndexBase::load(std::ifstream& fin) {
    return left.load(fin) && right.load(fin);
}

}  // namespace Graph

}  // namespace TDFANN