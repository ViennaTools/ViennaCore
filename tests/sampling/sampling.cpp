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

void runAlias() {
  // univariate sampling
  Sampling<double, 1, true> s;

  auto pdf = [](double x) { return std::exp(-(x * x)); };
  s.setPDF(pdf, {-5, 5}, 1000);

  auto copy = s;

  RNG rng(123512);

  std::fstream file;
  file.open("samples_1D_alias.txt", std::ios::out);

  for (int i = 0; i < 10000; ++i) {
    auto sample = copy.sample(rng);
    file << sample[0] << "\n";
  }

  file.close();
}

void run2D() {
  // bivariate sampling
  Sampling<double, 2> s;

  auto pdf = [](double x, double y) {
    return std::exp(-(x * x)) * std::exp(-(y * y));
  };
  std::array<Vec2D<double>, 2> bounds = {-5, 5, -5, 5};
  s.setPDF(pdf, bounds, {1000, 1000});

  // test copy constructor
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

void runCustom() {
  Sampling<double, 1> s;

  std::ifstream distFile("custom_distribution.txt");
  std::vector<double> xValues;
  std::vector<double> pdfValues;

  if (!distFile.is_open()) {
    std::cerr << "Could not open file\n";
    return;
  }

  double x, pdf;
  while (distFile >> x >> pdf) {
    xValues.push_back(x);
    pdfValues.push_back(pdf);
  }

  std::cout << "Read " << xValues.size() << " values\n";

  s.setPDF(pdfValues, xValues);

  // test copy constructor
  auto copy = s;

  RNG rng(123512);

  std::fstream file;
  file.open("samples_custom.txt", std::ios::out);

  for (int i = 0; i < 10000; ++i) {
    auto sample = copy.sample(rng);
    file << sample[0] << "\n";
  }

  file.close();
}

int main() {

  Logger::setLogLevel(LogLevel::DEBUG);

  run1D();
  run2D();
  runAlias();
  runCustom();

  return 0;
}