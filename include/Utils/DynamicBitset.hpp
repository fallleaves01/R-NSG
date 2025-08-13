#pragma once
#include <PCH.hpp>

namespace TDFANN {

namespace Utils {

class DynamicBitset {
   public:
    using B = uint64_t;
    constexpr size_t size_per_bit() const;

    DynamicBitset(size_t);
    bool at(size_t) const;
    bool operator[](size_t);
    void set(size_t);
    void unset(size_t);
    size_t find_first() const;
    size_t find_next(size_t index) const;

    class iterator {
       public:
        iterator& operator++();
        iterator operator++(int);
        iterator(DynamicBitset&, size_t);
        std::strong_ordering operator<=>(const iterator&) const;

       private:
        DynamicBitset& bitset;
        size_t index;
    };

    iterator begin();
    iterator end();
    bool save(std::ofstream&) const;
    bool load(std::ifstream&);

   private:
    void expand(size_t);
    std::vector<B> bits;
};

}  // namespace Utils

}  // namespace TDFANN