#include <vcTestAsserts.hpp>
#include <vcVectorType.hpp>

namespace viennacore {
template <typename NumericType, int D> void RunTest() {

  Vec3Dd vecDouble(0.);
  Vec3Di vecInt(0);

  if constexpr (D == 2) {
    Vec2D<NumericType> vec2A{1, 2};

    Vec2D<NumericType> vec2B{3, 4};

    Vec2D<NumericType> vec2C = vec2A + vec2B;
    VC_TEST_ASSERT(vec2C[0] == 4);
    VC_TEST_ASSERT(vec2C[1] == 6);

    vec2A = vec2A * NumericType(2);
    VC_TEST_ASSERT(vec2A[0] == 2);
    VC_TEST_ASSERT(vec2A[1] == 4);

    vec2A = vec2A / NumericType(2);
    VC_TEST_ASSERT(vec2A[0] == 1);
    VC_TEST_ASSERT(vec2A[1] == 2);

    auto dp = DotProduct(vec2A, vec2B);
    VC_TEST_ASSERT(dp == 11);

    auto distance = Distance(vec2A, vec2B);
    VC_TEST_ASSERT_ISCLOSE(distance, 2.8284271247461903, 1e-6);

  } else {
    Vec3D<NumericType> vec3A{1, 2, 3};

    Vec3D<NumericType> vec3B{4, 5, 6};

    Vec3D<NumericType> vec3C = vec3A + vec3B;
    VC_TEST_ASSERT(vec3C[0] == 5);
    VC_TEST_ASSERT(vec3C[1] == 7);
    VC_TEST_ASSERT(vec3C[2] == 9);

    vec3A = vec3A * NumericType(2);
    VC_TEST_ASSERT(vec3A[0] == 2);
    VC_TEST_ASSERT(vec3A[1] == 4);
    VC_TEST_ASSERT(vec3A[2] == 6);

    vec3A = vec3A / NumericType(2);
    VC_TEST_ASSERT(vec3A[0] == 1);
    VC_TEST_ASSERT(vec3A[1] == 2);
    VC_TEST_ASSERT(vec3A[2] == 3);

    auto dp = DotProduct(vec3A, vec3B);
    VC_TEST_ASSERT(dp == 32);

    auto distance = Distance(vec3A, vec3B);
    VC_TEST_ASSERT_ISCLOSE(distance, 5.196152422706632, 1e-6);
  }

  VectorType<NumericType, D> vec3A{1, 2, 3};
  VectorType<NumericType, D> vec3B{4, 5, 6};
  VectorType<NumericType, D> vec3C{7, 8, 9};

  auto cross = CrossProduct(vec3A, vec3B);
  std::cout << cross << std::endl;
}
} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }