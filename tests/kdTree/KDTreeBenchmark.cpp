#include <vcKDTree.hpp>
#include <vcNFKDTree.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Point = viennacore::Vec3D<double>;
using NativeTree = viennacore::KDTree<double, Point>;
using NanoflannTree = viennacore::NFKDTree<double, Point, 3>;
using Clock = std::chrono::steady_clock;
constexpr int runs = 10;

struct Measurement {
  double buildMs;
  double lookupMs;
  std::uint64_t matches = 0;
  std::uint64_t indexSum = 0;
};

struct Minimum {
  double buildMs = std::numeric_limits<double>::infinity();
  double lookupMs = std::numeric_limits<double>::infinity();

  void add(const Measurement &measurement) {
    buildMs = std::min(buildMs, measurement.buildMs);
    lookupMs = std::min(lookupMs, measurement.lookupMs);
  }
};

std::vector<Point> generatePoints(std::size_t count, std::uint32_t seed) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<double> distribution(0, 1);
  std::vector<Point> points(count);
  for (auto &point : points)
    for (auto &coordinate : point)
      coordinate = distribution(engine);
  return points;
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

template <class Tree>
Measurement measure(const std::vector<Point> &points,
                    const std::vector<Point> &queries, double radiusSquared) {
  // Copying input and destroying the tree are excluded from build timing.
  // Each run uses a fresh tree; repeated build() on KDTree is not required.
  Tree tree(points);
  const auto buildStart = Clock::now();
  tree.build();
  const auto buildEnd = Clock::now();

  if constexpr (std::is_same_v<Tree, NanoflannTree>) {
    // Match KDTree's inclusive radius boundary.
    radiusSquared =
        std::nextafter(radiusSquared, std::numeric_limits<double>::infinity());
  }
  Measurement result{milliseconds(buildStart, buildEnd), 0};
  const auto lookupStart = Clock::now();
  for (const auto &query : queries) {
    const auto neighbors = tree.findNearestWithinRadius(query, radiusSquared);
    if (!neighbors)
      throw std::runtime_error("Radius search returned no result object.");
    result.matches += neighbors->size();
    // Consume results to keep the searches observable, regardless of ordering.
    for (const auto &neighbor : *neighbors)
      result.indexSum += neighbor.first;
  }
  result.lookupMs = milliseconds(lookupStart, Clock::now());
  return result;
}

std::size_t parseCount(const std::string &value) {
  if (value.empty() ||
      value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Counts must be positive integers.");
  const auto count = std::stoull(value);
  if (count == 0 || count > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Count is out of range.");
  return static_cast<std::size_t>(count);
}

void printUsage(const char *program) {
  std::cout << "Usage: " << program
            << " [--queries N] [--radius R] [point_count ...]\n"
            << "Defaults: 1000 queries, radius 0.05, point counts "
               "1000 10000 100000 1000000. Always uses 10 runs.\n";
}
} // namespace

int main(int argc, char *argv[]) {
  try {
    // Empty radius results are normal here; logging them would skew timings.
    viennacore::Logger::setLogLevel(viennacore::LogLevel::ERROR);
    std::size_t queryCount = 1000;
    double radius = 0.05;
    std::vector<std::size_t> pointCounts;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--help") {
        printUsage(argv[0]);
        return 0;
      }
      if (argument == "--queries" || argument == "--radius") {
        if (++i == argc)
          throw std::invalid_argument("Missing value for " + argument);
        if (argument == "--queries") {
          queryCount = parseCount(argv[i]);
        } else {
          std::size_t consumed = 0;
          radius = std::stod(argv[i], &consumed);
          if (consumed != std::string(argv[i]).size() || radius <= 0 ||
              !std::isfinite(radius) || !std::isfinite(radius * radius))
            throw std::invalid_argument("Radius must be positive and finite.");
        }
      } else {
        pointCounts.push_back(parseCount(argument));
      }
    }
    if (pointCounts.empty())
      pointCounts = {1000, 10000, 100000, 1000000};

    const auto queries = generatePoints(queryCount, 67890);
    std::cout
        << "# Uniform double-precision 3D points in [0, 1]^3; " << "minimum of "
        << runs << " runs per backend.\n"
        << "# Build excludes input copy; lookup includes result allocation "
           "and consumption for "
        << queryCount << " serial queries.\n"
        << "# Radius: " << radius
        << "; KDTree returns sorted matches, NFKDTree unsorted.\n";
#ifdef _OPENMP
    std::cout << "# Maximum build threads: " << omp_get_max_threads() << '\n';
#else
    std::cout << "# Build threads: 1 (OpenMP disabled).\n";
#endif
    std::cout << "points,queries,matches,KDTree_build_min_ms,"
                 "NFKDTree_build_min_ms,KDTree_lookup_min_ms,"
                 "NFKDTree_lookup_min_ms\n"
              << std::fixed << std::setprecision(6);
    for (const auto count : pointCounts) {
      const auto points = generatePoints(count, 12345);
      Minimum nativeMin, nanoflannMin;
      std::uint64_t matches = 0;
      for (int run = 0; run < runs; ++run) {
        Measurement native, nanoflann;
        // Alternate order to reduce systematic cache and scheduling bias.
        if (run % 2 == 0) {
          native = measure<NativeTree>(points, queries, radius * radius);
          nanoflann = measure<NanoflannTree>(points, queries, radius * radius);
        } else {
          nanoflann = measure<NanoflannTree>(points, queries, radius * radius);
          native = measure<NativeTree>(points, queries, radius * radius);
        }
        if (native.matches != nanoflann.matches ||
            native.indexSum != nanoflann.indexSum)
          throw std::runtime_error("KD-tree radius search results disagree.");
        nativeMin.add(native);
        nanoflannMin.add(nanoflann);
        matches = native.matches;
      }
      std::cout << count << ',' << queryCount << ',' << matches << ','
                << nativeMin.buildMs << ',' << nanoflannMin.buildMs << ','
                << nativeMin.lookupMs << ',' << nanoflannMin.lookupMs
                << std::endl;
    }
  } catch (const std::exception &error) {
    std::cerr << "KDTreeBenchmark: " << error.what() << '\n';
    return 1;
  }
}
