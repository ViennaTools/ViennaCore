#pragma once

#include "vcSamplingMethod.hpp"
#include "vcVectorUtil.hpp"

#include <vector>

namespace viennacore {

template <class NumericType>
class AliasSampling : public BaseSamplingMethod<NumericType, 1> {
  // alias table
  const std::vector<NumericType> probabilities_;
  const std::vector<NumericType> alias_;

  const NumericType min_;
  const NumericType size_;

public:
  AliasSampling(const std::vector<NumericType> &probabilities,
                const std::vector<NumericType> &alias, NumericType min,
                NumericType size)
      : probabilities_(probabilities), alias_(alias), min_(min), size_(size) {}

  AliasSampling(const BaseSamplingMethod<NumericType, 1> &other)
      : probabilities_(
            static_cast<const AliasSampling &>(other).probabilities_),
        alias_(static_cast<const AliasSampling &>(other).alias_),
        min_(static_cast<const AliasSampling &>(other).min_),
        size_(static_cast<const AliasSampling &>(other).size_) {}

  std::array<NumericType, 1> sample(RNG &rngState) override final {
    std::uniform_real_distribution<NumericType> uniform(0, 1);

    unsigned i = uniform(rngState) * probabilities_.size();
    NumericType u = uniform(rngState);

    unsigned bin = i;
    if (u > probabilities_[i])
      bin = alias_[i];

    return {min_ + bin * size_ + uniform(rngState) * size_};
  }
};
} // namespace viennacore