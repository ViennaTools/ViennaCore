#include <vcNeighborSearch.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {
template <typename NumericType, template <class, class> class Tree>
void testBackend() {
  PointCloud<NumericType> cloud;
  cloud.positions = {{0, 0, 0}, {3, 4, 0}, {0, 0, 2}, {0, 0, 8}};
  cloud.normals.resize(cloud.size());
  NeighborSearch<NumericType, Tree> search(cloud);
  cloud.positions.clear(); // Searches use the original snapshot.

  const auto nearest = search.getKNN(0, 10);
  VC_TEST_ASSERT(nearest.size() == 4);
  VC_TEST_ASSERT(nearest[0].first == 0 && nearest[0].second == 0);
  VC_TEST_ASSERT(nearest[1].first == 2 && nearest[1].second == 2);
  VC_TEST_ASSERT(nearest[2].first == 1 && nearest[2].second == 5);
  VC_TEST_ASSERT(nearest[3].first == 3 && nearest[3].second == 8);
  VC_TEST_ASSERT(search.getKNN(0, 2).size() == 2);
  VC_TEST_ASSERT(search.getKNN(0, 0).empty());
  VC_TEST_ASSERT(search.getKNN(0, -1).empty());

  const auto radius = search.getRadius(0, 5);
  VC_TEST_ASSERT(radius.size() == 3);
  VC_TEST_ASSERT(std::equal(radius.begin(), radius.end(), nearest.begin()));
  const auto zeroRadius = search.getRadius(0, 0);
  VC_TEST_ASSERT(zeroRadius.size() == 1 && zeroRadius[0].first == 0);
  VC_TEST_ASSERT(search.getRadius(0, -1).empty());
}
} // namespace viennacore

int main() {
  viennacore::testBackend<float, viennacore::KDTree>();
  viennacore::testBackend<double, viennacore::KDTree>();
  viennacore::testBackend<float, viennacore::NFKDTree>();
  viennacore::testBackend<double, viennacore::NFKDTree>();

  viennacore::PointCloud<double> cloud;
  cloud.positions = {{0, 0, 0}};
  cloud.normals.resize(1);
  viennacore::NeighborSearch<double> defaultSearch(cloud);
  VC_TEST_ASSERT(defaultSearch.getKNN(0, 1).size() == 1);
}
