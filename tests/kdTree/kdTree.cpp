#include <vcKDTree.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {
template <typename NumericType, int D> void RunTest() {
  KDTree<NumericType, Vec3D<NumericType>> tree;

  std::vector<Vec3D<NumericType>> points = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  tree.setPoints(points);
  tree.build();

  auto nearest = tree.findNearest({0, 0, 0});

  VC_TEST_ASSERT(nearest->first == 0);
}
} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }