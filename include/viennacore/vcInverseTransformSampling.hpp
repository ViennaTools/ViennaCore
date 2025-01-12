#pragma once

#include "vcSamplingMethod.hpp"
#include "vcVectorUtil.hpp"

#include <vector>

namespace viennacore {

template <class NumericType>
class InverseTransformSampling : public BaseSamplingMethod<NumericType, 1> {
  const std::vector<NumericType> cdfValues_;
  const Vec2D<NumericType> bounds_;

public:
  InverseTransformSampling(const std::vector<NumericType> &cdfValues,
                           const Vec2D<NumericType> &bounds)
      : cdfValues_(cdfValues), bounds_(bounds) {}

  InverseTransformSampling(const BaseSamplingMethod<NumericType, 1> &other)
      : cdfValues_(
            static_cast<const InverseTransformSampling &>(other).cdfValues_),
        bounds_(static_cast<const InverseTransformSampling &>(other).bounds_) {}

  std::array<NumericType, 1> sample(RNG &rngState) const override final {
    std::uniform_real_distribution<NumericType> uniform(0, 1);

    NumericType u = uniform(rngState) * cdfValues_.back();
    auto it = std::lower_bound(cdfValues_.begin(), cdfValues_.end(), u);
    size_t idx = std::distance(cdfValues_.begin(), it);

    NumericType binRand = uniform(rngState);
    NumericType x =
        bounds_[0] + idx * (bounds_[1] - bounds_[0]) / cdfValues_.size();
    NumericType xNext =
        bounds_[0] + (idx + 1) * (bounds_[1] - bounds_[0]) / cdfValues_.size();
    NumericType y = binRand * (xNext - x) + x;

    return {y};
  };
};

} // namespace viennacore