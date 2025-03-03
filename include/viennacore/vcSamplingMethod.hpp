#pragma once

#include "vcRNG.hpp"

#include <array>

namespace viennacore {

template <class NumericType, int D> class BaseSamplingMethod {
public:
  virtual ~BaseSamplingMethod() = default;

  virtual std::array<NumericType, D> sample(RNG &rngState) = 0;
};

} // namespace viennacore