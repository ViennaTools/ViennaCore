#pragma once

#include "vcPhiloxRNG.hpp"
#include "vcUtil.hpp"

#include <random>

#ifdef VIENNACORE_COMPILE_GPU
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace viennacore {

#ifdef VIENNACORE_COMPILE_GPU
using CudaRNG = curandStatePhilox4_32_10_t;
// Other possible RNG State types:
// typedef curandStateXORWOW_t curtRNGState; // bad
// typedef curandStateMRG32k3a_t curtRNGState // not tested
// typedef curandStateSobol32_t curtRNGState; // not tested
// typedef curandStateScrambledSobol32_t curtRNGState; // not tested
#endif

#if defined(VIENNACORE_RNG_MT19937_64)
using RNG = std::mt19937_64;
#elif defined(VIENNACORE_RNG_MT19937_32)
using RNG = std::mt19937;
#elif defined(VIENNACORE_RNG_RANLUX48)
using RNG = std::ranlux48;
#elif defined(VIENNACORE_RNG_RANLUX24)
using RNG = std::ranlux24;
#elif defined(VIENNACORE_RNG_MINSTD)
using RNG = std::minstd_rand;
#elif defined(VIENNACORE_RNG_PHILOX)
using RNG = viennacore::PhiloxRNG;
#else
using RNG = viennacore::PhiloxRNG;
#endif

template <size_t N, class ValueType = uint32_t> class RandomNumbers {
  std::array<ValueType, N> numbers_;

public:
  explicit RandomNumbers(const std::array<ValueType, N> &numbers)
      : numbers_(numbers) {}

  template <typename T, std::enable_if_t<std::is_floating_point_v<T>,
                                         std::nullptr_t> = std::nullptr_t()>
  T get(size_t i) const {
    return static_cast<T>(numbers_[i]) /
           static_cast<T>(std::numeric_limits<ValueType>::max());
  }

  ValueType operator[](size_t i) const { return numbers_[i]; }
  ValueType &operator[](size_t i) { return numbers_[i]; }
};

// tiny encryption algorithm
template <unsigned int N>
__both__ unsigned int tea(unsigned int v0, unsigned int v1) {
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

#ifdef __CUDACC__
__device__ __inline__ float getNextRand(CudaRNG *state) {
  return curand_uniform(state);
}

__device__ __inline__ float getNormalDistRand(CudaRNG *state) {
  float4 u0 = curand_uniform4(state);
  float r = sqrtf(-2.f * logf(u0.x));
  float theta = 2.f * M_PIf * u0.y;
  return r * sinf(theta);
}
#endif

} // namespace viennacore
