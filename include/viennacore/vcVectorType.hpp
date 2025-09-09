#pragma once

#include "vcUtil.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace viennacore {

template <typename NumericType, size_t D>
using VectorType = std::array<NumericType, D>;

template <typename NumericType> using Vec2D = VectorType<NumericType, 2>;

template <typename NumericType> using Vec3D = VectorType<NumericType, 3>;

#define _define_vec_types(T, t)                                                \
  using Vec2D##t = VectorType<T, 2>;                                           \
  using Vec3D##t = VectorType<T, 3>;

_define_vec_types(int8_t, c);
_define_vec_types(int16_t, s);
_define_vec_types(int32_t, i);
_define_vec_types(int64_t, l);
_define_vec_types(uint8_t, uc);
_define_vec_types(uint16_t, us);
_define_vec_types(uint32_t, ui);
_define_vec_types(uint64_t, ul);
_define_vec_types(float, f);
_define_vec_types(double, d);

#undef _define_vec_types

/* ------------- Vector operation functions ------------- */

#define _define_operator(op)                                                   \
  /* vec op vec */                                                             \
  template <typename T>                                                        \
  inline __both__ Vec2D<T> operator op(const Vec2D<T> &a, const Vec2D<T> &b) { \
    return Vec2D<T>{a[0] op b[0], a[1] op b[1]};                               \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ Vec3D<T> operator op(const Vec3D<T> &a, const Vec3D<T> &b) { \
    return Vec3D<T>{a[0] op b[0], a[1] op b[1], a[2] op b[2]};                 \
  }                                                                            \
                                                                               \
  template <typename T, typename OT>                                           \
  inline __both__ Vec3D<T> operator op(const Vec3D<T> &a,                      \
                                       const std::array<OT, 3> &b) {           \
    return Vec3D<T>{a[0] op b[0], a[1] op b[1], a[2] op b[2]};                 \
  }                                                                            \
                                                                               \
  /* vec op scalar */                                                          \
  template <typename T>                                                        \
  inline __both__ Vec2D<T> operator op(const Vec2D<T> &a, const T & b) {       \
    return Vec2D<T>{a[0] op b, a[1] op b};                                     \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ Vec3D<T> operator op(const Vec3D<T> &a, const T & b) {       \
    return Vec3D<T>{a[0] op b, a[1] op b, a[2] op b};                          \
  }                                                                            \
                                                                               \
  /* scalar op vec */                                                          \
  template <typename T>                                                        \
  inline __both__ Vec2D<T> operator op(const T & a, const Vec2D<T> &b) {       \
    return Vec2D<T>{a op b[0], a op b[1]};                                     \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ Vec3D<T> operator op(const T & a, const Vec3D<T> &b) {       \
    return Vec3D<T>{a op b[0], a op b[1], a op b[2]};                          \
  }

_define_operator(*);
_define_operator(/);
_define_operator(+);
_define_operator(-);

#undef _define_operator

// Base case for single vector
template <typename NumericType, size_t D>
[[nodiscard]] __both__ VectorType<NumericType, D>
Sum(const VectorType<NumericType, D> &vec) noexcept {
  return vec;
}

// Recursive variadic template for summing any number of vectors
template <typename NumericType, size_t D, typename... Args>
[[nodiscard]] __both__ VectorType<NumericType, D>
Sum(const VectorType<NumericType, D> &first,
    const VectorType<NumericType, D> &second, const Args &...args) noexcept {
  VectorType<NumericType, D> result;
  for (size_t i = 0; i < D; ++i) {
    result[i] = first[i] + second[i];
  }

  if constexpr (sizeof...(args) > 0) {
    return Sum(result, args...);
  } else {
    return result;
  }
}

template <typename NumericType, size_t D>
[[nodiscard]] __both__ NumericType
DotProduct(const VectorType<NumericType, D> &a,
           const VectorType<NumericType, D> &b) noexcept {
  NumericType dot = 0;
  for (size_t i = 0; i < D; ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

template <typename NumericType>
[[nodiscard]] __both__ Vec3D<NumericType>
CrossProduct(const Vec3D<NumericType> &a,
             const Vec3D<NumericType> &b) noexcept {
  Vec3D<NumericType> rr;
  rr[0] = a[1] * b[2] - a[2] * b[1];
  rr[1] = a[2] * b[0] - a[0] * b[2];
  rr[2] = a[0] * b[1] - a[1] * b[0];
  return rr;
}

template <class NumericType>
[[nodiscard]] __both__ NumericType CrossProduct(
    const Vec2D<NumericType> &v1, const Vec2D<NumericType> &v2) noexcept {
  return v1[0] * v2[1] - v1[1] * v2[0];
}

template <typename NumericType, size_t D>
[[nodiscard]] __both__ NumericType
Norm2(const VectorType<NumericType, D> &vec) noexcept {
  return DotProduct(vec, vec);
}

template <typename NumericType, size_t D>
[[nodiscard]] __both__ NumericType Norm(const VectorType<NumericType, D> &vec) {
  return sqrt(Norm2(vec));
}

template <typename NumericType, size_t D>
void __both__ Normalize(VectorType<NumericType, D> &vec) {
  NumericType n = Norm2(vec);
  if (n <= std::numeric_limits<NumericType>::min()) {
    for (size_t i = 0; i < D; ++i)
      vec[i] = 0;
    return;
  }

#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<NumericType, float>) {
    n = rsqrtf(n); // fast on GPU for float
  } else {
    n = NumericType(1) / std::sqrt(n); // for double and other types
  }
#else
  // Optional fast path: skip if already ~unit
  if (std::abs(n - NumericType(1)) < NumericType(1e-6))
    return;

  n = NumericType(1) / std::sqrt(n); // CPU
#endif

  for (size_t i = 0; i < D; ++i)
    vec[i] *= n;
}

template <typename NumericType, size_t D>
[[nodiscard]] __both__ VectorType<NumericType, D>
Normalize(const VectorType<NumericType, D> &vec) {
  VectorType<NumericType, D> rr = vec;
  NumericType norm = Norm2(vec);
  if (norm <= std::numeric_limits<NumericType>::min()) {
    for (size_t i = 0; i < D; ++i)
      rr[i] = 0;
    return rr;
  }

#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<NumericType, float>) {
    norm = rsqrtf(norm); // fast on GPU for float
  } else {
    norm = NumericType(1) / std::sqrt(norm); // for double and other types
  }
#else
  if (std::abs(norm - NumericType(1)) < NumericType(1e-6))
    return rr;

  norm = NumericType(1) / std::sqrt(norm); // CPU
#endif

  for (size_t i = 0; i < D; ++i)
    rr[i] *= norm;

  return rr;
}

template <typename NumericType, size_t D>
__both__ void ScaleToLength(VectorType<NumericType, D> &vec,
                            const NumericType length) {
  const auto norm = Norm(vec);
  if (norm <= std::numeric_limits<NumericType>::min()) {
    // Handle zero vector case - can't scale a zero vector
    return;
  }
  const auto scale = length / norm;
  for (size_t i = 0; i < D; i++)
    vec[i] *= scale;
}

template <typename NumericType, size_t D>
[[nodiscard]] __both__ VectorType<NumericType, D>
Inv(const VectorType<NumericType, D> &vec) noexcept {
  VectorType<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = -vec[i];
  }
  return rr;
}

template <typename NumericType, size_t D>
[[nodiscard]] __both__ VectorType<NumericType, D>
ScaleAdd(const VectorType<NumericType, D> &mult,
         const VectorType<NumericType, D> &add,
         const NumericType fac) noexcept {
  VectorType<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = add[i] + mult[i] * fac;
  }
  return rr;
}

template <typename NumericType, size_t D>
__both__ [[nodiscard]] NumericType
Distance(const VectorType<NumericType, D> &a,
         const VectorType<NumericType, D> &b) {
  return Norm(a - b);
}

template <typename NumericType>
[[nodiscard]] __both__ Vec3D<NumericType>
ComputeNormal(const std::array<Vec3D<NumericType>, 3> &planeCoords) noexcept {
  auto uu = planeCoords[1] - planeCoords[0];
  auto vv = planeCoords[2] - planeCoords[0];
  return CrossProduct(uu, vv);
}

template <typename NumericType, size_t D>
__both__ bool IsNormalized(const VectorType<NumericType, D> &vec,
                           const NumericType eps = NumericType(1e-4)) {
  auto norm = Norm(vec);
  return std::abs(norm - NumericType(1)) < eps;
}

template <class T> __both__ Vec2D<T> RotateLeft(const Vec2D<T> &v) noexcept {
  return Vec2D<T>(-v[1], v[0]);
}

template <class T> Vec2D<T> __both__ RotateRight(const Vec2D<T> &v) noexcept {
  return Vec2D<T>(v[1], -v[0]);
}

template <class T, size_t D>
__both__ VectorType<T, D> Min(const VectorType<T, D> &v1,
                              const VectorType<T, D> &v2) noexcept {
  VectorType<T, D> v;
  for (size_t i = 0; i < D; i++)
    v[i] = std::min(v1[i], v2[i]);
  return v;
}

template <class T, size_t D>
__both__ VectorType<T, D> Max(const VectorType<T, D> &v1,
                              const VectorType<T, D> &v2) noexcept {
  VectorType<T, D> v;
  for (size_t i = 0; i < D; i++)
    v[i] = std::max(v1[i], v2[i]);
  return v;
}

template <class T, size_t D>
__both__ T MaxElement(const VectorType<T, D> &v1) noexcept {
  T v = v1[0];
  for (size_t i = 1; i < D; i++)
    v = std::max(v1[i], v);
  return v;
}

template <class T> __both__ T Volume(const VectorType<T, 2> &p) noexcept {
  return p[0] * p[1];
}

template <class T> __both__ T Volume(const VectorType<T, 3> &p) noexcept {
  return p[0] * p[1] * p[2];
}

template <class T, size_t D>
__both__ size_t MinIndex(const VectorType<T, D> &v) noexcept {
  size_t idx = 0;
  for (size_t i = 1; i < D; i++) {
    if (v[i] < v[idx])
      idx = i;
  }
  return idx;
}

template <class T, size_t D>
__both__ size_t MaxIndex(const VectorType<T, D> &v) noexcept {
  size_t idx = 0;
  for (size_t i = 1; i < D; i++) {
    if (v[i] > v[idx])
      idx = i;
  }
  return idx;
}

template <class T> __both__ bool Orientation(const Vec3D<T> *v) noexcept {
  return DotProduct(CrossProduct(v[1] - v[0], v[2] - v[0]), v[3] - v[0]) >= -0.;
}

template <class T> __both__ bool Orientation(const Vec2D<T> *v) noexcept {
  return DotProduct(RotateLeft(v[1] - v[0]), v[2] - v[0]) >= -0.;
}

template <class T, size_t D>
__both__ bool AnyEqualElement(const VectorType<T, D> &v1,
                              const VectorType<T, D> &v2) noexcept {
  for (size_t i = 0; i < D; ++i) {
    if (v1[i] == v2[i])
      return true;
  }
  return false;
}

template <typename T, size_t D> struct VectorHash {
private:
  static size_t hash_combine(size_t lhs, size_t rhs) noexcept {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }

public:
  size_t operator()(const VectorType<T, D> &v) const noexcept {
    /*
      https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
    */
    size_t result = std::hash<T>{}(v[0]);
    for (size_t i = 1; i < D; ++i) {
      result = hash_combine(result, std::hash<T>{}(v[i]));
    }
    return result;
  }
};

/* ------------- Debug convenience functions ------------- */

template <class S, typename T, size_t D>
S &operator<<(S &o, const VectorType<T, D> &v) {
  o << "[" << v[0];
  for (size_t i = 1; i < D; ++i)
    o << ", " << v[i];
  o << "]";
  return o;
}

template <typename NumericType>
void PrintBoundingBox(const std::array<Vec3D<NumericType>, 2> &bdBox) {
  std::cout << "Bounding box min coords: " << bdBox[0] << std::endl
            << "Bounding box max coords: " << bdBox[1] << std::endl;
}

}; // namespace viennacore