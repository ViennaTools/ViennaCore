#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace viennacore {

template <typename NumericType> using Vec2D = std::array<NumericType, 2>;

template <typename NumericType> using Vec3D = std::array<NumericType, 3>;

/* ------------- Vector operation functions ------------- */

#define _define_operator(op)                                                   \
  /* vec op vec */                                                             \
  template <typename T>                                                        \
  inline Vec2D<T> operator op(const Vec2D<T> &a, const Vec2D<T> &b) {          \
    return Vec2D<T>{a[0] op b[0], a[1] op b[1]};                               \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline Vec3D<T> operator op(const Vec3D<T> &a, const Vec3D<T> &b) {          \
    return Vec3D<T>{a[0] op b[0], a[1] op b[1], a[2] op b[2]};                 \
  }                                                                            \
                                                                               \
  template <typename T, typename OT>                                           \
  inline Vec3D<T> operator op(const Vec3D<T> &a, const std::array<OT, 3> &b) { \
    return Vec3D<T>{a[0] op b[0], a[1] op b[1], a[2] op b[2]};                 \
  }                                                                            \
                                                                               \
  /* vec op scalar */                                                          \
  template <typename T>                                                        \
  inline Vec2D<T> operator op(const Vec2D<T> &a, const T &b) {                 \
    return Vec2D<T>{a[0] op b, a[1] op b};                                     \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline Vec3D<T> operator op(const Vec3D<T> &a, const T &b) {                 \
    return Vec3D<T>{a[0] op b, a[1] op b, a[2] op b};                          \
  }                                                                            \
                                                                               \
  /* scalar op vec */                                                          \
  template <typename T>                                                        \
  inline Vec2D<T> operator op(const T &a, const Vec2D<T> &b) {                 \
    return Vec2D<T>{a op b[0], a op b[1]};                                     \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline Vec3D<T> operator op(const T &a, const Vec3D<T> &b) {                 \
    return Vec3D<T>{a op b[0], a op b[1], a op b[2]};                          \
  }

_define_operator(*);
_define_operator(/);
_define_operator(+);
_define_operator(-);

#undef _define_operator

template <typename NumericType, std::size_t D>
[[nodiscard]] std::array<NumericType, D>
Sum(const std::array<NumericType, D> &pVecA,
    const std::array<NumericType, D> &pVecB,
    const std::array<NumericType, D> &pVecC) {
  std::array<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = pVecA[i] + pVecB[i] + pVecC[i];
  }
  return rr;
}

template <typename NumericType, std::size_t D>
[[nodiscard]] NumericType DotProduct(const std::array<NumericType, D> &pVecA,
                                     const std::array<NumericType, D> &pVecB) {
  NumericType dot = 0;
  for (size_t i = 0; i < D; ++i) {
    dot += pVecA[i] * pVecB[i];
  }
  return dot;
}

template <typename NumericType>
[[nodiscard]] Vec3D<NumericType> CrossProduct(const Vec3D<NumericType> &pVecA,
                                              const Vec3D<NumericType> &pVecB) {
  Vec3D<NumericType> rr;
  rr[0] = pVecA[1] * pVecB[2] - pVecA[2] * pVecB[1];
  rr[1] = pVecA[2] * pVecB[0] - pVecA[0] * pVecB[2];
  rr[2] = pVecA[0] * pVecB[1] - pVecA[1] * pVecB[0];
  return rr;
}

template <typename NumericType, std::size_t D>
[[nodiscard]] NumericType Norm(const std::array<NumericType, D> &vec) {
  NumericType norm = 0;
  std::for_each(vec.begin(), vec.end(),
                [&norm](NumericType entry) { norm += entry * entry; });
  return std::sqrt(norm);
}

template <typename NumericType, std::size_t D>
void Normalize(std::array<NumericType, D> &vec) {
  NumericType norm = 1. / Norm(vec);
  if (norm == 1.)
    return;
  std::for_each(vec.begin(), vec.end(),
                [&norm](NumericType &entry) { entry *= norm; });
}

template <typename NumericType, size_t D>
[[nodiscard]] std::array<NumericType, D>
Normalize(const std::array<NumericType, D> &vec) {
  std::array<NumericType, D> normedVec = vec;
  auto norm = 1. / Norm(normedVec);
  if (norm == 1.)
    return normedVec;
  for (size_t i = 0; i < D; ++i) {
    normedVec[i] = norm * vec[i];
  }
  return normedVec;
}

template <typename NumericType, std::size_t D>
void ScaleToLength(std::array<NumericType, D> &vec, const NumericType length) {
  const auto vecLength = Norm(vec);
  for (size_t i = 0; i < D; i++)
    vec[i] *= length / vecLength;
}

template <typename NumericType, std::size_t D>
[[nodiscard]] std::array<NumericType, D>
Inv(const std::array<NumericType, D> &vec) {
  std::array<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = -vec[i];
  }
  return rr;
}

template <typename NumericType, std::size_t D>
std::array<NumericType, D> ScaleAdd(const std::array<NumericType, D> &mult,
                                    const std::array<NumericType, D> &add,
                                    const NumericType fac) {
  std::array<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = add[i] + mult[i] * fac;
  }
  return rr;
}

template <typename NumericType, size_t D>
[[nodiscard]] NumericType Distance(const std::array<NumericType, D> &pVecA,
                                   const std::array<NumericType, D> &pVecB) {
  return Norm(pVecA - pVecB);
}

template <typename NumericType>
[[nodiscard]] Vec3D<NumericType>
ComputeNormal(const Vec3D<Vec3D<NumericType>> &planeCoords) {
  auto uu = planeCoords[1] - planeCoords[0];
  auto vv = planeCoords[2] - planeCoords[0];
  return CrossProduct(uu, vv);
}

template <typename NumericType, std::size_t D>
bool IsNormalized(const std::array<NumericType, D> &vec) {
  constexpr double eps = 1e-4;
  auto norm = Norm(vec);
  return std::fabs(norm - 1) < eps;
}

/* ------------- Debug convenience functions ------------- */

template <typename T, std::size_t D>
inline std::ostream &operator<<(std::ostream &o, const std::array<T, D> &v) {
  o << "(";
  for (size_t i = 0; i < D; ++i) {
    o << v[i];
    if (i < D - 1)
      o << ", ";
  }
  o << ")";
  return o;
}

template <typename NumericType>
void PrintBoundingBox(const std::array<Vec3D<NumericType>, 2> &bdBox) {
  std::cout << "Bounding box min coords: " << bdBox[0] << std::endl
            << "Bounding box max coords: " << bdBox[1] << std::endl;
}

}; // namespace viennacore