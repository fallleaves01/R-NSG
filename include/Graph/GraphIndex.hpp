#pragma once

#include <PCH.hpp>

#include <Graph/Concepts.hpp>
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
        Node(size_t _to, Data&& _data) : to(_to), data(std::forward<Data>(_data)) {}
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
        std::ranges::copy(to, std::back_inserter(edges[from]));
        // edges[from].insert(edges[from].end(), to.begin(), to.end());
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

struct TDData {
    size_t label, banned_id;
};

inline GraphIndex<TDData>::Node to_node(size_t to, size_t label, size_t banned_id) {
    return GraphIndex<TDData>::Node{to, TDData{label, banned_id}};
}

class TDGraphIndexBase : public GraphIndex<TDData> {
   public:
    TDGraphIndexBase(size_t node_cnt)
        : GraphIndex<TDData>(node_cnt), header_index(1, 0) {}
    TDGraphIndexBase(std::string filename) : GraphIndex<TDData>(0) {
        std::ifstream fin(filename);
        if (!fin.good() || !load(fin)) {
            throw std::runtime_error("Failed to load graph from " + filename);
        }
    }

    class TDGraphIndex {
       public:
        TDGraphIndex(const TDGraphIndexBase& _base, const std::vector<size_t> &_label, size_t _l, size_t _r)
            : base(_base), label(_label), l_label(_l), r_label(_r) {}
        auto get_neighbours(size_t node) const {
            return base.get_neighbours(node) | std::views::filter([&](auto& x) {
                       return x.data.label >= l_label && x.data.label <= r_label;
                   });
        }
        IndexList auto get_neighbours_id(size_t node) const {
            return get_neighbours(node) |
                   std::views::transform([](auto x) { return x.to; });
        }
        auto get_header() const {
            return base.get_header(r_label) |
                   std::views::filter([&](auto& x) { return label[x] >= l_label && x < r_label; });
        }

       private:
        const TDGraphIndexBase& base;
        const std::vector<size_t> &label;
        size_t l_label, r_label;
    };
    TDGraphIndex operator()(const std::vector<size_t> &_label, size_t _l, size_t _r) const {
        return TDGraphIndex(*this, _label, _l, _r);
    }
    bool save(std::ofstream& fout) const {
        return GraphIndex::save(fout) && IO::save(fout, header_index) &&
               IO::save(fout, header_data);
    }
    bool load(std::ifstream& fin) {
        return GraphIndex::load(fin) && IO::load(fin, header_index) &&
               IO::load(fin, header_data);
    }
    std::span<const size_t> get_header(size_t node) const {
        return std::span<const size_t>(
            header_data.begin() + header_index[node],
            header_index[node + 1] - header_index[node]);
    }
    void append_header(IndexList auto&& h) {
        header_index.push_back(header_data.size() + h.size());
        header_data.insert(header_data.end(), h.begin(), h.end());
    }

   private:
    std::vector<size_t> header_index;
    std::vector<size_t> header_data;
};

}  // namespace Graph

}  // namespace TDFANN
