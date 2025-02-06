#include <vcSampling.hpp>
#include <vcTimer.hpp>

#include <cmath>
#include <fstream>

using namespace viennacore;

int main() {

  Logger::setLogLevel(LogLevel::INFO);

  // univariate sampling
  Sampling<double, 1> s;
  RNG rng(123512);

  const int n = 100000;
  auto pdf = [](double x) { return std::exp(-(x * x)); };
  s.setPDF(pdf, {-5, 5}, 1000);

  Timer timer;
  timer.start();
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    auto sample = s.sample(rng);
    sum += sample[0];
  }
  timer.finish();

  std::cout << "Time to generate " << n << " 1D samples (piecewise constant): "
            << timer.currentDuration * 1e-6 << " ms" << "\n";

  // alias sampling
  Sampling<double, 1, true> s_alias;
  s_alias.setPDF(pdf, {-5, 5}, 1000);

  timer.start();
  sum = 0;
  for (int i = 0; i < n; ++i) {
    auto sample = s_alias.sample(rng);
    sum += sample[0];
  }
  timer.finish();

  std::cout << "Time to generate " << n
            << " 1D samples (alias method): " << timer.currentDuration * 1e-6
            << " ms" << "\n";

  // bivariate sampling
  Sampling<double, 2> s_2;

  auto pdf_2 = [](double x, double y) {
    return std::exp(-(x * x)) * std::exp(-(y * y));
  };
  std::array<Vec2D<double>, 2> bounds = {-5, 5, -5, 5};
  s_2.setPDF(pdf_2, bounds, {1000, 1000});

  timer.start();
  double sum_1 = 0;
  double sum_2 = 0;
  for (int i = 0; i < n; ++i) {
    auto sample = s_2.sample(rng);
    sum_1 += sample[0];
    sum_2 += sample[1];
  }
  timer.finish();

  std::cout << "Time to generate " << n
            << " 2D samples: " << timer.currentDuration * 1e-6 << " ms" << "\n";

  return 0;
}