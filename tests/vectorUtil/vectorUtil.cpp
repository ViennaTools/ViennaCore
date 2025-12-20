#include <vcTestAsserts.hpp>
#include <vcVectorType.hpp>

namespace viennacore {
template <typename NumericType, int D> void RunTest() {

  Vec3Dd vecDouble;
  Vec3Di vecInt;

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

    auto cross = CrossProduct(vec2A, vec2B);

    Normalize(vec2A);
    VC_TEST_ASSERT(IsNormalized(vec2A));

    const auto cvec2B = vec2B;
    Vec2D<NumericType> nvec2B = Normalize(cvec2B);
    VC_TEST_ASSERT(IsNormalized(nvec2B));
    VC_TEST_ASSERT(vec2B[0] == 3);
    VC_TEST_ASSERT(vec2B[1] == 4);
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

    auto cross = CrossProduct(vec3A, vec3B);
  }

  VectorType<NumericType, D> vecTest;
  std::fill(vecTest.begin(), vecTest.end(), NumericType(1));

  VectorHash<NumericType, D> hash;
  std::cout << "Hash: " << hash(vecTest) << std::endl;

  Vec3D<NumericType> zeroVec{0, 0, 0};
  Normalize(zeroVec);
  VC_TEST_ASSERT(!IsNormalized(zeroVec));
  VC_TEST_ASSERT(zeroVec[0] == 0);
  VC_TEST_ASSERT(zeroVec[1] == 0);
  VC_TEST_ASSERT(zeroVec[2] == 0);
}
} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }