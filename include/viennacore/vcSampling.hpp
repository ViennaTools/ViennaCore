#pragma once

#include "vcAcceptRejectSampling.hpp"
#include "vcInverseTransformSampling.hpp"
#include "vcLogger.hpp"

#include <functional>
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

  void setPDF(const std::vector<NumericType> &pdfValues,
              const std::vector<NumericType> &xValues) {

    Vec2D<NumericType> minBounds = getSupport(pdfValues, xValues);

    Logger::getInstance()
        .addDebug("Univariate PDF support: " + std::to_string(minBounds[0]) +
                  " " + std::to_string(minBounds[1]))
        .print();

    if (algo_)
      algo_.reset();
    algo_ = std::make_unique<InverseTransformSampling<NumericType>>(xValues,
                                                                    pdfValues);
  }

  void setPDF(const std::function<NumericType(NumericType)> &pdf,
              const Vec2D<NumericType> &bounds, const unsigned nBins) {

    std::vector<NumericType> pdfValues(nBins);
    std::vector<NumericType> xValues(nBins);
    const NumericType step = (bounds[1] - bounds[0]) / (nBins - 1);
    for (unsigned i = 0; i < nBins; ++i) {
      xValues[i] = bounds[0] + i * step;
      pdfValues[i] = pdf(xValues[i]);
    }

    setPDF(pdfValues, xValues);
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

  auto sample(RNG &rngState) {
    assert(algo_);
    return algo_->sample(rngState);
  }

private:
  Vec2D<NumericType> getSupport(const std::vector<NumericType> &pdfValues,
                                const std::vector<NumericType> &xValues) {
    assert(pdfValues.size() == xValues.size());
    Vec2D<NumericType> support = {xValues.front(), xValues.back()};
    unsigned nBins = pdfValues.size();

    // look for lower bound
    bool foundMin = false;
    for (unsigned i = 0; i < nBins; ++i) {
      if (pdfValues[i] > 1e-6) {
        foundMin = true;
        break;
      }
      support[0] = xValues[i];
    }

    if (!foundMin) {
      Logger::getInstance().addError("Univariate PDF is zero everywhere.");
    }

    // look for upper bound
    for (unsigned i = nBins - 1; i > 0; --i) {
      if (pdfValues[i] > 1e-6)
        break;
      support[1] = xValues[i];
    }

    return support;
  }
};

} // namespace viennacore
