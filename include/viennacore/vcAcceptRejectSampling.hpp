#pragma once

#include "vcSamplingMethod.hpp"
#include "vcVectorUtil.hpp"

#include <vector>

namespace viennacore {

template <class NumericType>
class AcceptRejectSampling : public BaseSamplingMethod<NumericType, 2> {
  const std::vector<std::vector<NumericType>> pdfValues_;
  const std::array<Vec2D<NumericType>, 2> bounds_;
  const NumericType maxPdfValue_;

public:
  AcceptRejectSampling(const std::vector<std::vector<NumericType>> &pdfValues,
                       const std::array<Vec2D<NumericType>, 2> &bounds,
                       NumericType maxPdfValue)
      : pdfValues_(pdfValues), bounds_(bounds), maxPdfValue_(maxPdfValue) {}

  AcceptRejectSampling(const BaseSamplingMethod<NumericType, 2> &other)
      : pdfValues_(static_cast<const AcceptRejectSampling &>(other).pdfValues_),
        bounds_(static_cast<const AcceptRejectSampling &>(other).bounds_),
        maxPdfValue_(
            static_cast<const AcceptRejectSampling &>(other).maxPdfValue_) {}

  std::array<NumericType, 2> sample(RNG &rngState) override final {
    std::uniform_real_distribution<NumericType> uniform(0, 1);
    std::uniform_real_distribution<NumericType> uniformX(bounds_[0][0],
                                                         bounds_[0][1]);
    std::uniform_real_distribution<NumericType> uniformY(bounds_[1][0],
                                                         bounds_[1][1]);

    NumericType u, x, y, pdfValue;

    do {
      x = uniformX(rngState);
      y = uniformY(rngState);
      u = uniform(rngState);

      auto [i, j] = findBin(x, y);
      pdfValue = pdfValues_[i][j];
    } while (u * maxPdfValue_ > pdfValue);

    return {x, y};
  }

private:
  std::pair<unsigned, unsigned> findBin(NumericType x, NumericType y) const {
    unsigned i = static_cast<unsigned>((x - bounds_[0][0]) /
                                       (bounds_[0][1] - bounds_[0][0]) *
                                       pdfValues_.size());
    unsigned j = static_cast<unsigned>((y - bounds_[1][0]) /
                                       (bounds_[1][1] - bounds_[1][0]) *
                                       pdfValues_[0].size());
    return {i, j};
  };
};
} // namespace viennacore