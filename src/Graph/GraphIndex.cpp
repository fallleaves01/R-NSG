#include <Graph/GraphIndex.hpp>

namespace TDFANN {

namespace Graph {

TagGraphIndex::TagGraphIndex(size_t node_cnt) : edges(node_cnt) {}

size_t TagGraphIndex::add_node() {
    edges.emplace_back();
    return edges.size() - 1;
}

void TagGraphIndex::add_neighbours(
    size_t from,
    const std::vector<Node>& to) {
    edges[from].insert(edges[from].end(), to.begin(), to.end());
}

bool TagGraphIndex::save(std::ofstream& fout) const {
    return IO::save(fout, edges);

}

bool TagGraphIndex::load(std::ifstream& fin) {
    return IO::load(fin, edges);
}

}  // namespace Graph

}  // namespace TDFANN