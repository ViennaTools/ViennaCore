#pragma once

#include <random>

namespace viennacore {
/// Use mersenne twister 19937 as random number generator.
using RNG = std::mt19937_64;

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
