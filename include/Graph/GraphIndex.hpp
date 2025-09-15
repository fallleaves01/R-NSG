#pragma once

#include <PCH.hpp>

#include <IO/TypeIO.hpp>

namespace TDFANN {

namespace Graph {

template <typename Data>
class GraphIndex {
   public:
    struct Node {
        size_t to;
        [[no_unique_address]] Data data;
        Node() : to(-1) {}
        Node(size_t _to) : to(_to) {}
        Node(size_t _to, Data&& _data) : to(_to), data(std::forward(_data)) {}
    };
    GraphIndex(std::string filename) {
        std::ifstream fin(filename);
        if (!fin.good() || !load(fin)) {
            throw std::runtime_error("Failed to load graph from " + filename);
        }
    }
    GraphIndex(size_t node_cnt) : edges(node_cnt) {}
    size_t add_node() {
        edges.emplace_back();
        return edges.size() - 1;
    }
    void add_neighbours(size_t from, std::ranges::range auto&& to) {
        edges[from].insert(edges[from].end(), to.begin(), to.end());
    }
    const std::vector<Node>& get_neighbours(size_t node) const {
        return edges[node];
    }
    IndexList auto get_neighbours_id(size_t node) const {
        return get_neighbours(node) |
               std::views::transform([](auto x) { return x.to; });
    }
    bool save(std::ofstream& fout) const { return IO::save(fout, edges); }
    bool load(std::ifstream& fin) { return IO::load(fin, edges); }

   private:
    std::vector<std::vector<Node>> edges;
};

class TDGraphIndexBase : public GraphIndex<size_t> {
   public:
    TDGraphIndexBase(size_t node_cnt)
        : GraphIndex<size_t>(node_cnt), header(node_cnt) {}
    TDGraphIndexBase(std::string filename) : GraphIndex<size_t>(0) {
        std::ifstream fin(filename);
        if (!fin.good() || !load(fin)) {
            throw std::runtime_error("Failed to load graph from " + filename);
        }
    }

    class TDGraphIndex {
       public:
        TDGraphIndex(const TDGraphIndexBase& _base, size_t _l, size_t _r)
            : base(_base), l(_l), r(_r) {}
        auto get_neighbours(size_t node) const {
            return base.get_neighbours(node) | std::views::filter([&](auto& x) {
                       return x.to >= l && x.to < r;
                   });
        }
        IndexList auto get_neighbours_id(size_t node) const {
            return get_neighbours(node) |
                   std::views::transform([](auto x) { return x.to; });
        }
        auto get_header() const {
            return base.get_header(l) |
                   std::views::filter([&](auto& x) { return x >= l && x < r; });
        }

       private:
        const TDGraphIndexBase& base;
        size_t l, r;
    };
    TDGraphIndex operator()(size_t _l, size_t _r) const {
        return TDGraphIndex(*this, _l, _r);
    }
    bool save(std::ofstream& fout) const {
        return GraphIndex::save(fout) && IO::save(fout, header);
    }
    bool load(std::ifstream& fin) {
        return GraphIndex::load(fin) && IO::load(fin, header);
    }
    const std::vector<size_t>& get_header(size_t node) const {
        return header[node];
    }
    void set_header(size_t node, IndexList auto&& h) {
        header[node] = std::vector<size_t>(h.begin(), h.end());
    }

   private:
    std::vector<std::vector<size_t>> header;
};

}  // namespace Graph

}  // namespace TDFANN
