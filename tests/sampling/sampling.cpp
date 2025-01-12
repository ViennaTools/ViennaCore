#include <vcSampling.hpp>

#include <cmath>
#include <fstream>

using namespace viennacore;

int main() {

  Logger::setLogLevel(LogLevel::DEBUG);

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

  return 0;
}