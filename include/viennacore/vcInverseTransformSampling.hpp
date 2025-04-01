#pragma once

#include "vcSamplingMethod.hpp"
#include "vcVectorType.hpp"

#include <cassert>
#include <functional>
#include <vector>

namespace viennacore {

template <class NumericType,
          class SamplingMethod =
              std::piecewise_linear_distribution<NumericType>>
class InverseTransformSampling : public BaseSamplingMethod<NumericType, 1> {
  SamplingMethod dist_;

public:
  InverseTransformSampling(const Vec2D<NumericType> &bounds,
                           const unsigned nBins,
                           std::function<NumericType(NumericType)> pdf)
      : dist_(nBins, bounds[0], bounds[1], pdf) {}

  InverseTransformSampling(const std::vector<NumericType> &xValues,
                           const std::vector<NumericType> &pdfValues)
      : dist_(xValues.begin(), xValues.end(), pdfValues.begin()) {}

  InverseTransformSampling(const BaseSamplingMethod<NumericType, 1> &other)
      : dist_(static_cast<const InverseTransformSampling &>(other).dist_) {}

  std::array<NumericType, 1> sample(RNG &rngState) final {
    return {dist_(rngState)};
  };
};

} // namespace viennacore