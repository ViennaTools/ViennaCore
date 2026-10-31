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

    // quarter turns, and the orientation of a triangle: whether its third
    // point lies left of the edge from the first to the second
    const Vec2D<NumericType> edge{3, 1};
    const auto left = RotateLeft(edge);
    const auto right = RotateRight(edge);
    VC_TEST_ASSERT(left[0] == -1 && left[1] == 3);
    VC_TEST_ASSERT(right[0] == 1 && right[1] == -3);
    VC_TEST_ASSERT(CrossProduct(edge, left) == 10);
    const Vec2D<NumericType> counterclockwise[3] = {{0, 0}, {1, 0}, {0, 1}};
    const Vec2D<NumericType> clockwise[3] = {{0, 0}, {0, 1}, {1, 0}};
    VC_TEST_ASSERT(Orientation(counterclockwise));
    VC_TEST_ASSERT(!Orientation(clockwise));
    VC_TEST_ASSERT(Volume(Vec2D<NumericType>{2, 3}) == 6);
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

    // the orientation of a tetrahedron: whether its fourth point lies on the
    // side of (b - a) x (c - a)
    const Vec3D<NumericType> positive[4] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const Vec3D<NumericType> negative[4] = {
        {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
    VC_TEST_ASSERT(Orientation(positive));
    VC_TEST_ASSERT(!Orientation(negative));
    const auto normal = ComputeNormal(std::array<Vec3D<NumericType>, 3>{
        positive[0], positive[1], positive[2]});
    VC_TEST_ASSERT(normal[0] == 0 && normal[1] == 0 && normal[2] == 1);
    VC_TEST_ASSERT(Volume(Vec3D<NumericType>{2, 3, 4}) == 24);
    PrintBoundingBox(
        std::array<Vec3D<NumericType>, 2>{positive[0], positive[3]});
  }

  // the element-wise functions, with a = (1, 2[, 3]) and b = (2, 1[, 0])
  VectorType<NumericType, D> a, b;
  for (int i = 0; i < D; ++i) {
    a[i] = NumericType(i + 1);
    b[i] = NumericType(2 - i);
  }
  const auto sum = Sum(a, b, a);
  const auto inverse = Inv(a);
  const auto scaled = ScaleAdd(a, b, NumericType(2));
  const auto smaller = Min(a, b);
  const auto larger = Max(a, b);
  const auto absolute = Abs(inverse);
  for (int i = 0; i < D; ++i) {
    VC_TEST_ASSERT(sum[i] == 2 * a[i] + b[i]);
    VC_TEST_ASSERT(inverse[i] == -a[i]);
    VC_TEST_ASSERT(scaled[i] == b[i] + 2 * a[i]);
    VC_TEST_ASSERT(smaller[i] == std::min(a[i], b[i]));
    VC_TEST_ASSERT(larger[i] == std::max(a[i], b[i]));
    VC_TEST_ASSERT(absolute[i] == a[i]);
  }
  VC_TEST_ASSERT(MinElement(b) == b[D - 1] && MinIndex(b) == D - 1);
  VC_TEST_ASSERT(MaxElement(a) == a[D - 1] && MaxIndex(a) == D - 1);
  VC_TEST_ASSERT(AnyEqualElement(a, a) && !AnyEqualElement(a, b));
  VC_TEST_ASSERT(Norm2(a) == DotProduct(a, a));
  auto lengthTwo = a;
  ScaleToLength(lengthTwo, NumericType(2));
  VC_TEST_ASSERT_ISCLOSE(Norm(lengthTwo), 2, 1e-5);
  std::cout << "Vector: " << a << std::endl;

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