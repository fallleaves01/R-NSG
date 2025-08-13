#include <Utils/DynamicBitset.hpp>
#include <IO/TypeIO.hpp>

// Implementation of DynamicBitset methods

namespace TDFANN {

namespace Utils {

DynamicBitset::DynamicBitset(size_t size)
    : bits((size + size_per_bit() - 1) / (size_per_bit()), 0) {}

constexpr size_t DynamicBitset::size_per_bit() const {
    return sizeof(B) * 8;
}

bool DynamicBitset::at(size_t index) const {
    if (index < bits.size() * size_per_bit()) {
        size_t bit_index = index / size_per_bit();
        size_t bit_offset = index % size_per_bit();
        return (bits[bit_index] >> bit_offset) & 1;
    }
    throw std::out_of_range("Index out of range");
}

bool DynamicBitset::operator[](size_t index) {
    expand(index);
    return (bits[index / size_per_bit()] >> (index % size_per_bit())) & 1;
}

void DynamicBitset::set(size_t index) {
    expand(index);
    bits[index / size_per_bit()] |= (B(1) << (index % size_per_bit()));
}

void DynamicBitset::unset(size_t index) {
    expand(index);
    bits[index / size_per_bit()] &= ~(B(1) << (index % size_per_bit()));
}

size_t DynamicBitset::find_first() const {
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] != 0) {
            return i * size_per_bit() + std::countr_zero(bits[i]);
        }
    }
    return size_t(-1);
}

size_t DynamicBitset::find_next(size_t index) const {
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
    return size_t(-1);
}

DynamicBitset::iterator::iterator(DynamicBitset& b, size_t id) : bitset(b), index(id) {}

DynamicBitset::iterator& DynamicBitset::iterator::operator++() {
    index = bitset.find_next(index);
    return *this;
}

DynamicBitset::iterator DynamicBitset::iterator::operator++(int) {
    iterator tmp = *this;
    ++(*this);
    return tmp;
}

std::strong_ordering DynamicBitset::iterator::operator<=>(const iterator& other) const {
    if (index != other.index) {
        return index <=> other.index;
    }
    return std::strong_ordering::equal;
}

DynamicBitset::iterator DynamicBitset::begin() {
    return iterator(*this, find_first());
}

DynamicBitset::iterator DynamicBitset::end() {
    return iterator(*this, size_t(-1));
}

bool DynamicBitset::save(std::ofstream &fout) const {
    if (!fout) {
        return false;
    }
    return IO::save(fout, bits);
}

bool DynamicBitset::load(std::ifstream &fin) {
    if (!fin) {
        return false;
    }
    return IO::load(fin, bits);
}

void DynamicBitset::expand(size_t index) {
    if (index >= bits.size() * size_per_bit()) {
        bits.resize(index / size_per_bit() + 1, 0);
    }
}

}  // namespace Utils

}  // namespace TDFANN