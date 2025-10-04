#pragma once

#include <PCH.hpp>

#include <Graph/Concepts.hpp>
#include <IO/TypeIO.hpp>
#include <Utils/ExFunc.hpp>

namespace TDFANN {

namespace Graph {

template <typename Data>
class GraphIndex {
   public:
    struct Node {
        unsigned to;
        [[no_unique_address]] Data data;
        Node() : to(-1) {}
        Node(unsigned _to) : to(_to) {}
        Node(unsigned _to, Data&& _data)
            : to(_to), data(std::forward<Data>(_data)) {}
    };
    static_assert(IO::TriviallySerializable<GraphIndex<Data>::Node>);
    GraphIndex(std::string filename) {
        std::ifstream fin(filename);
        if (!fin.good() || !load(fin)) {
            throw std::runtime_error("Failed to load graph from " + filename);
        }
    }
    GraphIndex(unsigned node_cnt) : edges(node_cnt) {}
    unsigned add_node() {
        edges.emplace_back();
        return edges.size() - 1;
    }
    void add_neighbours(unsigned from, std::ranges::range auto&& to) {
        std::ranges::copy(to, std::back_inserter(edges[from]));
        // edges[from].insert(edges[from].end(), to.begin(), to.end());
    }
    const std::vector<Node>& get_neighbours(unsigned node) const {
        return edges[node];
    }
    IndexList auto get_neighbours_id(unsigned node) const {
        return get_neighbours(node) | std::views::transform(GET(to));
    }
    bool save(std::ofstream& fout) const { return IO::save(fout, edges); }
    bool load(std::ifstream& fin) { return IO::load(fin, edges); }

   private:
    std::vector<std::vector<Node>> edges;
};

struct TDData {
    unsigned label;
    // unsigned banned_id;
};

// inline GraphIndex<TDData>::Node to_node(unsigned to,
//                                         unsigned label,
//                                         unsigned banned_id) {
//     return GraphIndex<TDData>::Node{to, TDData{label, banned_id}};
// }
inline GraphIndex<std::monostate>::Node to_node(unsigned to) {
    return GraphIndex<std::monostate>::Node{to};
}

class TDGraphIndexBase : public GraphIndex<std::monostate> {
   public:
    TDGraphIndexBase(unsigned node_cnt)
        : GraphIndex<std::monostate>(node_cnt), header_index(1, 0) {}
    TDGraphIndexBase(std::string filename) : GraphIndex<std::monostate>(0) {
        std::ifstream fin(filename);
        if (!fin.good() || !load(fin)) {
            throw std::runtime_error("Failed to load graph from " + filename);
        }
    }

    class TDGraphIndex {
       public:
        TDGraphIndex(const TDGraphIndexBase& _base,
                     const std::vector<unsigned>& _label,
                     unsigned _l,
                     unsigned _r,
                     unsigned _header_id)
            : base(_base), header_id(_header_id) {
            l_id = std::ranges::lower_bound(_label, _l) - _label.begin();
            r_id = std::ranges::upper_bound(_label, _r) - _label.begin() - 1;
        }
        auto get_neighbours(unsigned node) const {
            return base.get_neighbours(node) | std::views::filter([&](auto& x) {
                       return x.to >= l_id && x.to <= r_id;
                   });
        }
        IndexList auto get_neighbours_id(unsigned node) const {
            return get_neighbours(node) | std::views::transform(GET(to));
        }
        auto get_header() const {
            return base.get_header(header_id) |
                   std::views::filter(
                       [&](auto& x) { return x >= l_id && x <= r_id; });
        }

       private:
        const TDGraphIndexBase& base;
        unsigned l_id, r_id, header_id;
    };
    TDGraphIndex operator()(const std::vector<unsigned>& _label,
                            unsigned _l,
                            unsigned _r) const {
        unsigned hid =
            std::upper_bound(header_label.begin(), header_label.end(), _r) -
            header_label.begin() - 1;
        return TDGraphIndex(*this, _label, _l, _r, hid);
    }
    bool save(std::ofstream& fout) const {
        return GraphIndex::save(fout) && IO::save(fout, header_index) &&
               IO::save(fout, header_data) && IO::save(fout, header_label);
    }
    bool load(std::ifstream& fin) {
        return GraphIndex::load(fin) && IO::load(fin, header_index) &&
               IO::load(fin, header_data) && IO::load(fin, header_label);
    }
    std::span<const unsigned> get_header(unsigned index) const {
        return std::span<const unsigned>(
            header_data.begin() + header_index[index],
            header_index[index + 1] - header_index[index]);
    }
    void append_header(unsigned label, IndexList auto&& h) {
        header_label.push_back(label);
        header_index.push_back(header_data.size() + h.size());
        header_data.insert(header_data.end(), h.begin(), h.end());
    }

   private:
    std::vector<unsigned> header_index, header_data;
    std::vector<unsigned> header_label;
};

}  // namespace Graph

}  // namespace TDFANN
