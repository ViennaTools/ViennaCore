#pragma once

#include "vcAcceptRejectSampling.hpp"
#include "vcInverseTransformSampling.hpp"
#include "vcLogger.hpp"

#include <memory>

namespace viennacore {

template <class NumericType, int D> class Sampling {
  std::unique_ptr<BaseSamplingMethod<NumericType, D>> algo_;

public:
  Sampling() = default;

  Sampling(const Sampling &other) {
    if constexpr (D == 1) {
      algo_ =
          std::make_unique<InverseTransformSampling<NumericType>>(*other.algo_);
    } else if constexpr (D == 2) {
      algo_ = std::make_unique<AcceptRejectSampling<NumericType>>(*other.algo_);
    }
  }

  Sampling &operator=(const Sampling &other) {
    if (this != &other) {
      if constexpr (D == 1) {
        algo_ = std::make_unique<InverseTransformSampling<NumericType>>(
            *other.algo_);
      } else if constexpr (D == 2) {
        algo_ =
            std::make_unique<AcceptRejectSampling<NumericType>>(*other.algo_);
      }
    }
    return *this;
  }

  void setPDF(const std::function<NumericType(NumericType)> &pdf,
              const Vec2D<NumericType> &bounds, const unsigned nBins) {
    if (algo_)
      algo_.reset();

    std::vector<NumericType> cdfValues(nBins);
    Vec2D<NumericType> minBounds = bounds;
    bool foundMin = false;

    NumericType step = (bounds[1] - bounds[0]) / nBins;

    NumericType x = bounds[0];
    cdfValues[0] = pdf(x);
    for (unsigned i = 1; i < nBins; ++i) {
      auto pdfVal = pdf(x);

      if (!foundMin && pdfVal < 1e-6) {
        minBounds[0] = x;
      } else {
        foundMin = true;
      }

      x += step;
      cdfValues[i] = cdfValues[i - 1] + pdfVal;
    }

    if (!foundMin) {
      Logger::getInstance().addError("PDF is zero everywhere.");
    }

    // look for upper bound
    x = bounds[1];
    for (unsigned i = nBins - 1; i > 0; --i) {
      if (pdf(x) < 1e-6) {
        minBounds[1] = x;
        x -= step;
      } else {
        break;
      }
    }

    Logger::getInstance()
        .addDebug("Min bounds: " + std::to_string(minBounds[0]) + " " +
                  std::to_string(minBounds[1]))
        .print();

    algo_ = std::make_unique<InverseTransformSampling<NumericType>>(cdfValues,
                                                                    minBounds);
  }

  void setPDF(const std::function<NumericType(NumericType, NumericType)> &pdf,
              const std::array<Vec2D<NumericType>, 2> &bounds,
              const std::array<unsigned, D> &nBins) {
    if (algo_)
      algo_.reset();

    std::vector<std::vector<NumericType>> pdfValues(nBins[0]);

    NumericType xStep = (bounds[0][1] - bounds[0][0]) / nBins[0];
    NumericType yStep = (bounds[1][1] - bounds[1][0]) / nBins[1];
    NumericType maxValue = 0;

    NumericType x = bounds[0][0];
    for (unsigned i = 0; i < nBins[0]; ++i) {
      pdfValues[i].resize(nBins[1]);

      NumericType y = bounds[1][0];
      for (unsigned j = 0; j < nBins[1]; j++) {
        pdfValues[i][j] = pdf(x, y);
        if (pdfValues[i][j] > maxValue)
          maxValue = pdfValues[i][j];
        y += yStep;
      }
      x += xStep;
    }

    algo_ = std::make_unique<AcceptRejectSampling<NumericType>>(
        pdfValues, bounds, maxValue);
  }

  auto sample(RNG &rngState) const {
    assert(algo_);
    return algo_->sample(rngState);
  }
};

} // namespace viennacore
