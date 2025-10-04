#pragma once
#include <PCH.hpp>
#include <IO/TypeIO.hpp>

namespace TDFANN {

namespace Utils {

class DynamicBitset {
   public:
    using B = uint64_t;
    constexpr unsigned size_per_bit() const;

    DynamicBitset(unsigned);
    bool at(unsigned) const;
    bool operator[](unsigned);
    void set(unsigned);
    void unset(unsigned);
    unsigned find_first() const;
    unsigned find_next(unsigned index) const;

    class iterator {
       public:
        iterator& operator++();
        iterator operator++(int);
        iterator(DynamicBitset&, unsigned);
        std::strong_ordering operator<=>(const iterator&) const;

       private:
        DynamicBitset& bitset;
        unsigned index;
    };

    iterator begin();
    iterator end();
    bool save(std::ofstream&) const;
    bool load(std::ifstream&);

   private:
    void expand(unsigned);
    std::vector<B> bits;
};

inline DynamicBitset::DynamicBitset(unsigned size)
    : bits((size + size_per_bit() - 1) / (size_per_bit()), 0) {}

constexpr unsigned DynamicBitset::size_per_bit() const {
    return sizeof(B) * 8;
}

inline bool DynamicBitset::at(unsigned index) const {
    if (index < bits.size() * size_per_bit()) {
        unsigned bit_index = index / size_per_bit();
        unsigned bit_offset = index % size_per_bit();
        return (bits[bit_index] >> bit_offset) & 1;
    }
    throw std::out_of_range("Index out of range");
}

inline bool DynamicBitset::operator[](unsigned index) {
    expand(index);
    return (bits[index / size_per_bit()] >> (index % size_per_bit())) & 1;
}

inline void DynamicBitset::set(unsigned index) {
    expand(index);
    bits[index / size_per_bit()] |= (B(1) << (index % size_per_bit()));
}

inline void DynamicBitset::unset(unsigned index) {
    expand(index);
    bits[index / size_per_bit()] &= ~(B(1) << (index % size_per_bit()));
}

inline unsigned DynamicBitset::find_first() const {
    for (unsigned i = 0; i < bits.size(); ++i) {
        if (bits[i] != 0) {
            return i * size_per_bit() + std::countr_zero(bits[i]);
        }
    }
    return unsigned(-1);
}

inline unsigned DynamicBitset::find_next(unsigned index) const {
    ++index;
    if (index < bits.size() * size_per_bit()) {
        B rv = bits[index] >> (index % size_per_bit());
        if (rv) {
            return index + std::countr_zero(rv);
        } else {
            index = (index | (size_per_bit() - 1)) + 1;
        }
        while (index < bits.size() * size_per_bit()) {
            if (bits[index]) {
                return index + std::countr_zero(bits[index]);
            }
            index += size_per_bit();
        }
    }
    return unsigned(-1);
}

inline DynamicBitset::iterator::iterator(DynamicBitset& b, unsigned id) : bitset(b), index(id) {}

inline DynamicBitset::iterator& DynamicBitset::iterator::operator++() {
    index = bitset.find_next(index);
    return *this;
}

inline DynamicBitset::iterator DynamicBitset::iterator::operator++(int) {
    iterator tmp = *this;
    ++(*this);
    return tmp;
}

inline std::strong_ordering DynamicBitset::iterator::operator<=>(const iterator& other) const {
    if (index != other.index) {
        return index <=> other.index;
    }
    return std::strong_ordering::equal;
}

inline DynamicBitset::iterator DynamicBitset::begin() {
    return iterator(*this, find_first());
}

inline DynamicBitset::iterator DynamicBitset::end() {
    return iterator(*this, unsigned(-1));
}

inline bool DynamicBitset::save(std::ofstream &fout) const {
    if (!fout) {
        return false;
    }
    return IO::save(fout, bits);
}

inline bool DynamicBitset::load(std::ifstream &fin) {
    if (!fin) {
        return false;
    }
    return IO::load(fin, bits);
}

inline void DynamicBitset::expand(unsigned index) {
    if (index >= bits.size() * size_per_bit()) {
        bits.resize(index / size_per_bit() + 1, 0);
    }
}


}  // namespace Utils

}  // namespace TDFANN