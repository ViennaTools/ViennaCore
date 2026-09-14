#include <vcKDTree.hpp>
#include <vcTestAsserts.hpp>

#include <algorithm>

namespace viennacore {
template <typename NumericType, int D> void RunTest() {
  KDTree<NumericType, Vec3D<NumericType>> tree;

  std::vector<Vec3D<NumericType>> points = {Vec3D<NumericType>{1, 2, 3},
                                            Vec3D<NumericType>{4, 5, 6},
                                            Vec3D<NumericType>{7, 8, 9}};

  tree.setPoints(points);
  tree.build();

  auto nearest = tree.findNearest(Vec3D<NumericType>{0, 0, 0});

  VC_TEST_ASSERT(nearest->first == 0);
  VC_TEST_ASSERT(nearest->second == 14);

  // Compare all search modes with exhaustive squared distances, including
  // scaled coordinates and radii both below and above one.
  using Point = VectorType<NumericType, D>;
  std::vector<Point> samples(40);
  for (std::size_t i = 0; i < samples.size(); ++i)
    for (int axis = 0; axis < D; ++axis)
      samples[i][axis] = NumericType((i * (axis + 3)) % 19) / 4;
  std::vector<NumericType> scaling(D, 1);
  scaling[0] = 2;
  KDTree<NumericType, Point> scaledTree;
  scaledTree.setPoints(samples, scaling);
  scaledTree.build();
  for (int q = 0; q < 6; ++q) {
    Point query{};
    query.fill(NumericType(q) / 4);
    std::vector<std::pair<std::size_t, NumericType>> reference;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      NumericType squared = 0;
      for (int axis = 0; axis < D; ++axis) {
        const auto delta = (samples[i][axis] - query[axis]) * scaling[axis];
        squared += delta * delta;
      }
      reference.emplace_back(i, squared);
    }
    std::sort(reference.begin(), reference.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    const auto closest = scaledTree.findNearest(query);
    VC_TEST_ASSERT(closest && closest->second == reference.front().second);
    for (int k : {1, 7, 50}) {
      const auto actual = scaledTree.findKNearest(query, k);
      VC_TEST_ASSERT(actual && actual->size() ==
                                   std::min(std::size_t(k), samples.size()));
      for (std::size_t i = 0; i < actual->size(); ++i)
        VC_TEST_ASSERT((*actual)[i].second == reference[i].second);
    }
    for (NumericType radiusSquared :
         {NumericType(0), NumericType(0.25), NumericType(4), NumericType(25)}) {
      auto actual = scaledTree.findNearestWithinRadius(query, radiusSquared);
      VC_TEST_ASSERT(actual);
      VC_TEST_ASSERT(std::is_sorted(
          actual->begin(), actual->end(),
          [](const auto &a, const auto &b) { return a.second < b.second; }));
      std::vector<std::pair<std::size_t, NumericType>> expected;
      for (const auto &neighbor : reference)
        if (neighbor.second <= radiusSquared)
          expected.push_back(neighbor);
      std::sort(actual->begin(), actual->end());
      std::sort(expected.begin(), expected.end());
      VC_TEST_ASSERT(*actual == expected);
    }
  }

  // Zero radius includes all duplicates, including across split planes.
  KDTree<NumericType, Point> duplicates(std::vector<Point>(5, Point{}));
  duplicates.build();
  VC_TEST_ASSERT(duplicates.findNearestWithinRadius(Point{}, 0)->size() == 5);
  VC_TEST_ASSERT(!duplicates.findNearestWithinRadius(Point{}, -1));
  VC_TEST_ASSERT(!duplicates.findKNearest(Point{}, 0));
}
} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
