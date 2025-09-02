#pragma once
#include <IO/TypeIO.hpp>
#include <PCH.hpp>

namespace TDFANN {

namespace Graph {

template <typename Data>
class GraphIndex {
   public:
    struct Node {
        size_t to;
        [[no_unique_address]] Data data;
    };
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
    auto get_neighbours_id(size_t node) const {
        return get_neighbours(node) |
               std::views::transform([](auto x) { return x.to; });
    }
    bool save(std::ofstream& fout) const {
        return IO::save(fout, edges);
    }
    bool load(std::ifstream& fin) {
        return IO::load(fin, edges);
    }

   private:
    std::vector<std::vector<Node>> edges;
};

class TDGraphIndexBase {
   public:
    TDGraphIndexBase(GraphIndex<size_t>&& left, GraphIndex<size_t>&& right)
        : left(std::move(left)), right(std::move(right)) {}
    bool save(std::ofstream& fout) const {
        return left.save(fout) && right.save(fout);
    }
    bool load(std::ifstream& fin) {
        return left.load(fin) && right.load(fin);
    }

    class TDGraphIndex {
       public:
        TDGraphIndex(const TDGraphIndexBase& _base, size_t _l, size_t _r)
            : base(_base), l(_l), r(_r) {}
        auto get_neighbours(size_t node) const {
            auto& id_l = base.left.get_neighbours(node);
            auto& id_r = base.right.get_neighbours(node);
            return std::array{id_l, id_r} | std::views::join |
                   std::views::filter(
                       [this](auto x) { return l <= x.to && x.to <= r; });
        }
        auto get_neighbours_id(size_t node) const {
            return get_neighbours(node) |
                   std::views::transform([](auto x) { return x.to; });
        }

       private:
        const TDGraphIndexBase& base;
        size_t l, r;
    };
    TDGraphIndex operator()(size_t _l, size_t _r) const {
        return TDGraphIndex(*this, _l, _r);
    }

   private:
    GraphIndex<size_t> left, right;
};

}  // namespace Graph

}  // namespace TDFANN
