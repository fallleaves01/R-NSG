#pragma once
#include <IO/TypeIO.hpp>
#include <PCH.hpp>

namespace TDFANN {

namespace Graph {

class TagGraphIndex {
   public:
    struct Node {
        size_t to, banned_id;
    };
    TagGraphIndex(size_t node_cnt);
    size_t add_node();
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
    bool save(std::ofstream& fout) const;
    bool load(std::ifstream& fin);

   private:
    std::vector<std::vector<Node>> edges;
};

class TDGraphIndexBase {
   public:
    TDGraphIndexBase(TagGraphIndex&& left, TagGraphIndex&& right)
        : left(std::move(left)), right(std::move(right)) {}
    bool save(std::ofstream& fout) const;
    bool load(std::ifstream& fin);

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
    TagGraphIndex left, right;
};

}  // namespace Graph

}  // namespace TDFANN
