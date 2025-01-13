#include <vcSampling.hpp>

#include <cmath>
#include <fstream>

using namespace viennacore;

void run1D() {
  // univariate sampling
  Sampling<double, 1> s;

  auto pdf = [](double x) { return std::exp(-(x * x)); };
  s.setPDF(pdf, {-5, 5}, 1000);

  auto copy = s;

  RNG rng(123512);

  std::fstream file;
  file.open("samples_1D.txt", std::ios::out);

  for (int i = 0; i < 10000; ++i) {
    auto sample = copy.sample(rng);
    file << sample[0] << "\n";
  }

  file.close();
}

void run2D() {
  // univariate sampling
  Sampling<double, 2> s;

  auto pdf = [](double x, double y) {
    return std::exp(-(x * x)) * std::exp(-(y * y));
  };
  std::array<Vec2D<double>, 2> bounds = {-5, 5, -5, 5};
  s.setPDF(pdf, bounds, {1000, 1000});

  auto copy = s;

  RNG rng(123512);

  std::fstream file;
  file.open("samples_2D.txt", std::ios::out);

  for (int i = 0; i < 10000; ++i) {
    auto sample = copy.sample(rng);
    file << sample[0] << "," << sample[1] << "\n";
  }

  file.close();
}

int main() {

  Logger::setLogLevel(LogLevel::DEBUG);

  run1D();
  run2D();

  return 0;
}