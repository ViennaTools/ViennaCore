#pragma once

#include <random>

namespace viennacore {
#ifdef VIENNACORE_RNG_MT19937_64
using RNG = std::mt19937_64;
#elifdef VIENNACORE_RNG_MT19937_32
using RNG = std::mt19937;
#elifdef VIENNACORE_RNG_RANLUX48
using RNG = std::ranlux48;
#elifdef VIENNACORE_RNG_RANLUX24
using RNG = std::ranlux24;
#elifdef VIENNACORE_RNG_MINSTD
using RNG = std::minstd_rand;
#else
using RNG = std::mt19937_64;
#endif

template <size_t N, class ValueType = uint_fast64_t> class RandonNumbers {
  const std::array<ValueType, N> numbers_;
  const ValueType min_, max_;

public:
  RandonNumbers(ValueType max) : max_(max) {}
  RandonNumbers(std::array<ValueType, N> numbers, ValueType min, ValueType max)
      : numbers_(numbers), min_(min), max_(max) {}

  template <typename T> T get(size_t i) const {
    return static_cast<T>(numbers_[i] / max_);
  }

  ValueType operator[](size_t i) const { return numbers_[i]; }
};

// tiny encryption algorithm
template <unsigned int N>
static unsigned int tea(unsigned int v0, unsigned int v1) {
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}
} // namespace viennacore
