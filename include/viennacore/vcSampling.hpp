#pragma once

#include "vcAcceptRejectSampling.hpp"
#include "vcAliasSampling.hpp"
#include "vcInverseTransformSampling.hpp"
#include "vcLogger.hpp"

#include <functional>
#include <memory>
#include <queue>

namespace viennacore {

// Class for sampling from a given probability density function (PDF). The PDF
// can be set either by providing a vector of PDF values and corresponding
// x-values or by providing a function that evaluates the PDF at a given point.
// The sampling is done using either the inverse transform method (D = 1) or the
// accept-reject method (D = 2).
template <class NumericType, int D, bool useAlias = false> class Sampling {
  std::unique_ptr<BaseSamplingMethod<NumericType, D>> algo_;

public:
  Sampling() { static_assert(D == 1 || D == 2, "D must be 1 or 2."); }

  Sampling(const Sampling &other) {
    if constexpr (D == 1) {
      if constexpr (useAlias) {
        algo_ = std::make_unique<AliasSampling<NumericType>>(*other.algo_);
      } else {
        algo_ = std::make_unique<InverseTransformSampling<NumericType>>(
            *other.algo_);
      }
    } else if constexpr (D == 2) {
      algo_ = std::make_unique<AcceptRejectSampling<NumericType>>(*other.algo_);
    }
  }

  Sampling &operator=(const Sampling &other) {
    if (algo_)
      algo_.reset();

    if (this != &other) {
      if constexpr (D == 1) {
        if constexpr (useAlias) {
          algo_ = std::make_unique<AliasSampling<NumericType>>(*other.algo_);
        } else {
          algo_ = std::make_unique<InverseTransformSampling<NumericType>>(
              *other.algo_);
        }
      } else if constexpr (D == 2) {
        algo_ =
            std::make_unique<AcceptRejectSampling<NumericType>>(*other.algo_);
      }
    }
    return *this;
  }

  bool hasPDF() const { return algo_ != nullptr; }

  // The sampling method in 1D can be either piecewise linear or piecewise
  // constant.
  template <
      class SamplingMethod = std::piecewise_linear_distribution<NumericType>>
  void setPDF(const std::vector<NumericType> &pdfValues,
              const std::vector<NumericType> &xValues) {
    static_assert(D == 1, "D must be 1 for univariate sampling.");

    auto minBounds = getSupport(pdfValues, xValues);
    auto trimmedPdfValues = std::vector<NumericType>(
        pdfValues.begin() + minBounds[0], pdfValues.begin() + minBounds[1] + 1);
    auto trimmedXValues = std::vector<NumericType>(
        xValues.begin() + minBounds[0], xValues.begin() + minBounds[1] + 1);

    Logger::getInstance()
        .addDebug("Univariate PDF support: " +
                  std::to_string(trimmedXValues.front()) + " " +
                  std::to_string(trimmedXValues.back()))
        .print();

    if (algo_)
      algo_.reset();

    if constexpr (useAlias) {
      // prepare alias table (stable Vose algorithm)

      // normalize PDF
      NumericType sum = 0;
      for (auto &pdf : trimmedPdfValues) {
        sum += pdf;
      }

      for (auto &pdf : trimmedPdfValues) {
        pdf /= sum;
      }

      std::vector<NumericType> probabilities(trimmedPdfValues.size());
      std::vector<NumericType> alias(trimmedPdfValues.size());

      std::queue<unsigned> small;
      std::queue<unsigned> large;

      for (unsigned i = 0; i < trimmedPdfValues.size(); ++i) {
        probabilities[i] = trimmedPdfValues[i] * trimmedPdfValues.size();
        if (probabilities[i] < 1)
          small.push(i);
        else
          large.push(i);
      }

      while (!small.empty() && !large.empty()) {
        unsigned less = small.front();
        small.pop();
        unsigned more = large.front();
        large.pop();

        alias[less] = more;
        probabilities[more] = probabilities[more] + probabilities[less] - 1;
        if (probabilities[more] < 1)
          small.push(more);
        else
          large.push(more);
      }

      while (!large.empty()) {
        unsigned last = large.front();
        large.pop();
        probabilities[last] = 1;
      }

      while (!small.empty()) {
        unsigned last = small.front();
        small.pop();
        probabilities[last] = 1;
      }

      algo_ = std::make_unique<AliasSampling<NumericType>>(
          probabilities, alias, trimmedXValues.front(),
          (trimmedXValues.back() - trimmedXValues.front()) /
              trimmedPdfValues.size());

    } else {
      algo_ = std::make_unique<
          InverseTransformSampling<NumericType, SamplingMethod>>(
          trimmedXValues, trimmedPdfValues);
    }
  }

  template <
      class SamplingMethod = std::piecewise_linear_distribution<NumericType>>
  void setPDF(const std::function<NumericType(NumericType)> &pdf,
              const Vec2D<NumericType> &bounds, const unsigned nBins) {
    static_assert(D == 1, "D must be 1 for univariate sampling.");
    std::vector<NumericType> pdfValues(nBins);
    std::vector<NumericType> xValues(nBins);
    const NumericType step = (bounds[1] - bounds[0]) / (nBins - 1);
    for (unsigned i = 0; i < nBins; ++i) {
      xValues[i] = bounds[0] + i * step;
      pdfValues[i] = pdf(xValues[i]);
    }

    setPDF<SamplingMethod>(pdfValues, xValues);
  }

  void prepareAlias(const std::vector<NumericType> &pdfValues,
                    const std::vector<NumericType> &xValues) {
    static_assert(D == 1, "D must be 1 for univariate sampling.");
    auto minBounds = getSupport(pdfValues, xValues);

    Logger::getInstance()
        .addDebug("Univariate PDF support: " + std::to_string(minBounds[0]) +
                  " " + std::to_string(minBounds[1]))
        .print();

    if (algo_)
      algo_.reset();
    // algo_ = std::make_unique<AliasSampling<NumericType>>(
    // pdfValues, xValues, minBounds[0], minBounds[1] - minBounds[0]);
  }

  // 2D sampling is done using the accept-reject method.
  void setPDF(const std::vector<std::vector<NumericType>> &pdfValues,
              const std::vector<NumericType> &xValues,
              const std::vector<NumericType> &yValues) {
    static_assert(D == 2, "D must be 2 for bivariate sampling.");
    NumericType maxValue = 0;
    auto support = getSupport(pdfValues, {xValues, yValues}, maxValue);

    Logger::getInstance()
        .addDebug("Bivariate PDF support: " + std::to_string(support[0][0]) +
                  " " + std::to_string(support[0][1]) + "; " +
                  std::to_string(support[1][0]) + " " +
                  std::to_string(support[1][1]))
        .print();

    if (algo_)
      algo_.reset();
    algo_ = std::make_unique<AcceptRejectSampling<NumericType>>(
        pdfValues, support, maxValue);
  }

  void setPDF(const std::function<NumericType(NumericType, NumericType)> &pdf,
              const std::array<Vec2D<NumericType>, 2> &bounds,
              const std::array<unsigned, D> &nBins) {
    static_assert(D == 2, "D must be 2 for bivariate sampling.");
    std::vector<std::vector<NumericType>> pdfValues(nBins[0]);
    std::vector<NumericType> xValues(nBins[0]);
    std::vector<NumericType> yValues(nBins[1]);

    NumericType xStep = (bounds[0][1] - bounds[0][0]) / (nBins[0] - 1);
    NumericType yStep = (bounds[1][1] - bounds[1][0]) / (nBins[1] - 1);

    NumericType x = bounds[0][0];
    for (unsigned i = 0; i < nBins[0]; ++i) {
      pdfValues[i].resize(nBins[1]);

      NumericType y = bounds[1][0];
      for (unsigned j = 0; j < nBins[1]; j++) {
        pdfValues[i][j] = pdf(x, y);
        y += yStep;
        yValues[j] = y;
      }
      x += xStep;
      xValues[i] = x;
    }

    setPDF(pdfValues, xValues, yValues);
  }

  auto sample(RNG &rngState) {
    assert(algo_);
    return algo_->sample(rngState);
  }

private:
  Vec2D<unsigned> getSupport(const std::vector<NumericType> &pdfValues,
                             const std::vector<NumericType> &xValues) {
    assert(pdfValues.size() == xValues.size());
    assert(!pdfValues.empty());
    auto support =
        Vec2D<unsigned>(0, static_cast<unsigned>(xValues.size() - 1));
    unsigned nBins = pdfValues.size();

    // look for lower bound
    bool foundMin = false;
    for (unsigned i = 0; i < nBins; ++i) {
      if (pdfValues[i] > 1e-6) {
        foundMin = true;
        break;
      }
      support[0] = i;
    }

    if (!foundMin) {
      Logger::getInstance().addError("Uni-variate PDF is zero everywhere.");
    }

    // look for upper bound
    for (unsigned i = nBins - 1; i > 0; --i) {
      if (pdfValues[i] > 1e-6)
        break;
      support[1] = i;
    }

    return support;
  }

  std::array<Vec2D<NumericType>, 2>
  getSupport(const std::vector<std::vector<NumericType>> &pdfValues,
             const std::array<std::vector<NumericType>, 2> &grid,
             NumericType &maxValue) {
    auto support_x = Vec2D<NumericType>{grid[0].back(), grid[0].front()};
    auto support_y = Vec2D<NumericType>{grid[1].back(), grid[1].front()};

    unsigned nBins_x = grid[0].size();
    unsigned nBins_y = grid[1].size();

    bool first_y = false;
    for (int j = 0; j < nBins_y; ++j) {
      auto y = grid[1][j];

      bool first_x = false;
      for (int i = 0; i < nBins_x; ++i) {
        auto x = grid[0][i];
        auto pdf_eval = pdfValues[i][j];
        if (pdf_eval > maxValue)
          maxValue = pdf_eval;

        if (pdf_eval > 1e-6) {
          if (!first_x) {
            if (x < support_x[0]) {
              support_x[0] = x;
            }
            first_x = true;
          } else {
            if (x > support_x[1]) {
              support_x[1] = x;
            }
          }
        }
      }

      if (first_x) {
        if (!first_y) {
          if (y < support_y[0]) {
            support_y[0] = y;
          }
          first_y = true;
        } else {
          if (y > support_y[1]) {
            support_y[1] = y;
          }
        }
      }
    }

    return {support_x, support_y};
  }
};

} // namespace viennacore
