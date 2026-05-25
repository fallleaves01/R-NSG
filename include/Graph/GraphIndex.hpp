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
    }
    void set_neighbours(unsigned from, std::vector<Node> value) {
        edges[from] = std::move(value);
    }
    const std::vector<Node>& get_neighbours(unsigned node) const {
        return edges[node];
    }
    std::vector<Node>& get_neighbours_mut(unsigned node) { return edges[node]; }
    IndexList auto get_neighbours_id(unsigned node) const {
        return get_neighbours(node) | std::views::transform(GET(to));
    }
    size_t size() const { return edges.size(); }
    bool save(std::ofstream& fout) const { return IO::save(fout, edges); }
    bool load(std::ifstream& fin) { return IO::load(fin, edges); }

   private:
    std::vector<std::vector<Node>> edges;

   protected:
    std::vector<std::vector<Node>>& edges_mut() { return edges; }
};

struct TDData {
    unsigned label;
};
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

    // Sort each node's neighbor list by ID (to enable binary search)
    void sort_neighbors_by_id() {
        for (auto& nb_list : edges_mut()) {
            std::ranges::sort(nb_list, {}, &Node::to);
        }
        neighbors_sorted_ = true;
    }

    bool neighbors_sorted() const { return neighbors_sorted_; }

    // Get in-range neighbors using binary search (requires sorted neighbors)
    // Returns a pair of iterators [begin, end) for neighbors in [l_id, r_id]
    auto get_neighbours_in_range(unsigned node, unsigned l_id,
                                  unsigned r_id) const {
        const auto& nbs = get_neighbours(node);
        auto cmp = [](const Node& a, const Node& b) { return a.to < b.to; };
        Node l_key{l_id}, r_key{r_id};
        auto begin = std::lower_bound(nbs.begin(), nbs.end(), l_key, cmp);
        auto end = std::upper_bound(nbs.begin(), nbs.end(), r_key, cmp);
        return std::make_pair(begin, end);
    }

    class TDGraphIndex {
       public:
        TDGraphIndex(const TDGraphIndexBase& _base,
                     const std::vector<std::uint64_t>& _label,
                     std::uint64_t _l,
                     std::uint64_t _r,
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
    TDGraphIndex operator()(const std::vector<std::uint64_t>& _label,
                            std::uint64_t _l,
                            std::uint64_t _r) const {
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
    unsigned get_header_index_for_right_bound(std::uint64_t r) const {
        auto it = std::upper_bound(header_label.begin(), header_label.end(), r);
        if (it == header_label.begin()) {
            return 0;
        }
        return static_cast<unsigned>(it - header_label.begin() - 1);
    }
    void append_header(std::uint64_t label, IndexList auto&& h) {
        header_label.push_back(label);
        header_index.push_back(header_data.size() + h.size());
        header_data.insert(header_data.end(), h.begin(), h.end());
    }
    void copy_headers_from(const TDGraphIndexBase& other) {
        header_index = other.header_index;
        header_data = other.header_data;
        header_label = other.header_label;
    }

   private:
    std::vector<unsigned> header_index, header_data;
    std::vector<std::uint64_t> header_label;
    bool neighbors_sorted_ = false;

    // P10-4: Dual-index for range-aware access (preserves original order)
    // Each entry: (neighbor_id, original_position_in_adjacency_list)
    std::vector<std::vector<std::pair<unsigned, unsigned>>> sorted_neighbor_idx_;
    bool has_sorted_idx_ = false;

   public:
    /// Build a sorted-by-ID index per node, preserving original positions.
    /// After building, use get_neighbours_in_range_indexed() for binary search.
    void build_sorted_neighbor_index() {
        sorted_neighbor_idx_.clear();
        sorted_neighbor_idx_.resize(edges_mut().size());
        for (size_t node = 0; node < edges_mut().size(); ++node) {
            const auto& nbs = get_neighbours(static_cast<unsigned>(node));
            auto& sorted = sorted_neighbor_idx_[node];
            sorted.reserve(nbs.size());
            for (size_t i = 0; i < nbs.size(); ++i) {
                sorted.push_back({nbs[i].to, static_cast<unsigned>(i)});
            }
            std::ranges::sort(sorted, {},
                &std::pair<unsigned, unsigned>::first);
        }
        has_sorted_idx_ = true;
    }

    bool has_sorted_neighbor_index() const { return has_sorted_idx_; }

    /// Get in-range neighbors using binary search on sorted index.
    /// Returns vector of (neighbor_id, original_position), sorted by original_position.
    std::vector<std::pair<unsigned, unsigned>>
    get_neighbours_in_range_indexed(
        unsigned node, unsigned l_id, unsigned r_id) const {
        std::vector<std::pair<unsigned, unsigned>> result;
        if (node >= sorted_neighbor_idx_.size()) return result;
        const auto& sorted = sorted_neighbor_idx_[node];
        auto begin = std::lower_bound(sorted.begin(), sorted.end(),
            std::make_pair(l_id, 0u),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        auto end = std::upper_bound(sorted.begin(), sorted.end(),
            std::make_pair(r_id, UINT_MAX),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        result.insert(result.end(), begin, end);
        // Sort by original position to restore adjacency order
        std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        return result;
    }

    void get_neighbours_in_range_indexed_into(
        unsigned node, unsigned l_id, unsigned r_id,
        std::vector<std::pair<unsigned, unsigned>>& result) const {
        result.clear();
        if (node >= sorted_neighbor_idx_.size()) {
            return;
        }
        const auto& sorted = sorted_neighbor_idx_[node];
        auto begin = std::lower_bound(
            sorted.begin(), sorted.end(), std::make_pair(l_id, 0u),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        auto end = std::upper_bound(
            sorted.begin(), sorted.end(),
            std::make_pair(r_id, std::numeric_limits<unsigned>::max()),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        result.insert(result.end(), begin, end);
        std::sort(result.begin(), result.end(), [](const auto& a,
                                                   const auto& b) {
            return a.second < b.second;
        });
    }
};

}  // namespace Graph

}  // namespace TDFANN
