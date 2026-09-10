#include <vcNFKDTree.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {
template <typename NumericType, int D> void RunTest() {
  using Point = VectorType<NumericType, D>;
  NFKDTree<NumericType, Point, D> tree;
  Point query{};
  VC_TEST_ASSERT(!tree.findNearest(query));
  std::vector<Point> points(4);
  points[0][0] = 3;
  points[1][1] = 2;
  points[2][0] = 1;
  points[3][0] = 5;
  tree.setPoints(points);
  points[2][0] = 100; // The wrapper owns its input.
  VC_TEST_ASSERT(tree.getNumberOfPoints() == 4);
  VC_TEST_ASSERT(!tree.findNearest(query));
  tree.build();
  const auto nearest = tree.findNearest(query);
  VC_TEST_ASSERT(nearest && nearest->first == 2 && nearest->second == 1);
  const auto neighbors = tree.findKNearest(query, 10);
  VC_TEST_ASSERT(neighbors && neighbors->size() == 4);
  VC_TEST_ASSERT((*neighbors)[1].first == 1 && (*neighbors)[1].second == 4);
  VC_TEST_ASSERT((*neighbors)[2].first == 0 && (*neighbors)[2].second == 9);
  VC_TEST_ASSERT((*neighbors)[3].first == 3 && (*neighbors)[3].second == 25);
  VC_TEST_ASSERT(!tree.findKNearest(query, 0));
  VC_TEST_ASSERT(!tree.findKNearest(query, -1));
  // Radius queries take a squared radius, exclude its boundary, and are
  // unsorted.
  auto radius = tree.findNearestWithinRadius(query, 9);
  VC_TEST_ASSERT(radius && radius->size() == 2);
  std::sort(radius->begin(), radius->end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  VC_TEST_ASSERT((*radius)[0].first == 1 && (*radius)[0].second == 4);
  VC_TEST_ASSERT((*radius)[1].first == 2 && (*radius)[1].second == 1);
  VC_TEST_ASSERT(tree.findNearestWithinRadius(query, 0)->empty());
  VC_TEST_ASSERT(
      tree.findNearestWithinRadius(Point{}, NumericType(0.5))->empty());
  VC_TEST_ASSERT(!tree.findNearestWithinRadius(query, -1));

  NFKDTree<NumericType> dynamicTree({{0, 0}, {3, 4}});
  dynamicTree.build();
  VC_TEST_ASSERT(dynamicTree.findKNearest({0, 0}, 2)->back().second == 25);
  VC_TEST_ASSERT(dynamicTree.findNearestWithinRadius({0, 0}, 0)->empty());
  const auto within = dynamicTree.findNearestWithinRadius({0, 0}, 25);
  VC_TEST_ASSERT(within && within->size() == 1);
  VC_TEST_ASSERT(within->front().first == 0 && within->front().second == 0);

  PointCloud<NumericType> cloud;
  cloud.positions = {{3, 4, 0}};
  cloud.normals = {{0, 0, 1}};
  cloud.validate();
  NFKDTree<NumericType> cloudTree(cloud);
  cloud.positions.clear();
  cloudTree.build();
  VC_TEST_ASSERT(cloudTree.findNearest({0, 0, 0})->second == 25);
}
} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
