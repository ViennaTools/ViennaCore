#pragma once

#include "vcUtil.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace viennacore {

template <class T, int D> class VectorType {
  std::array<T, D> x = {};

public:
  using value_type = T;
  static constexpr int dimension = D;

  VectorType() = default;
  VectorType(const VectorType &v) = default;
  VectorType &operator=(const VectorType &v) = default;
  explicit VectorType(const std::array<T, D> &v) { x = v; }
  template <class T1, int D1> explicit VectorType(const VectorType<T1, D1> &v) {
    const int k = std::min(D, D1);
    for (int i = 0; i < k; ++i)
      x[i] = v[i];
    for (int i = k; i < D; ++i)
      x[i] = T(0);
  }
  template <class T1, int D1> explicit VectorType(const std::array<T1, D1> &v) {
    const int k = std::min(D, D1);
    for (int i = 0; i < k; ++i)
      x[i] = v[i];
    for (int i = k; i < D; ++i)
      x[i] = T(0);
  }
  explicit VectorType(const T v[]) {
    for (int i = 0; i < D; i++)
      x[i] = v[i];
  }
  explicit VectorType(T x0, T x1, T x2) {
    x[0] = x0;
    x[1] = x1;
    if constexpr (D == 3)
      x[2] = x2;
  }
  explicit VectorType(T x0, T x1) {
    x[0] = x0;
    x[1] = x1;
  }
  explicit VectorType(T d) {
    for (int i = 0; i < D; i++)
      x[i] = d;
  }
  template <class X> explicit VectorType(X d) {
    for (int i = 0; i < D; i++)
      x[i] = d[i];
  }

  template <class V> VectorType &operator-=(const V &v) {
    for (int i = 0; i < D; i++)
      x[i] -= v[i];
    return *this;
  }
  template <class V> VectorType &operator+=(const V &v) {
    for (int i = 0; i < D; i++)
      x[i] += v[i];
    return *this;
  }
  VectorType &operator*=(T d) {
    for (int i = 0; i < D; i++)
      x[i] *= d;
    return *this;
  }
  VectorType &operator/=(T d) {
    for (int i = 0; i < D; i++)
      x[i] /= d;
    return *this;
  }
  VectorType operator-() const {
    VectorType v;
    for (int i = 0; i < D; i++)
      v.x[i] = -x[i];
    return v;
  }

  template <class V> bool operator==(const V &v) const {
    for (int i = D - 1; i >= 0; --i)
      if (x[i] != v[i])
        return false;
    return true;
  }
  template <class V> bool operator!=(const V &v) const {
    for (int i = D - 1; i >= 0; --i)
      if (x[i] != v[i])
        return true;
    return false;
  }
  template <class V> bool operator<(const V &v) const {
    for (int i = D - 1; i >= 0; --i) {
      if (x[i] < v[i])
        return true;
      if (x[i] > v[i])
        return false;
    }
    return false;
  }
  template <class V> bool operator<=(const V &v) const { return !(*this > v); }
  template <class V> bool operator>(const V &v) const {
    for (int i = D - 1; i >= 0; --i) {
      if (x[i] > v[i])
        return true;
      if (x[i] < v[i])
        return false;
    }
    return false;
  }
  template <class V> bool operator>=(const V &v) const { return !(*this < v); }

  T &operator[](int i) { return x[i]; }
  const T &operator[](int i) const { return x[i]; }
  T &at(int i) { return x.at(i); }
  const T &at(int i) const { return x.at(i); }

  template <class V> VectorType &operator=(const V &v) {
    for (int i = 0; i < D; i++)
      x[i] = v[i];
    return *this;
  }

  T size() const { return x.size(); }
  T max_element() const {
    return *std::max_element(std::begin(x), std::end(x));
  }
  void sort() { std::sort(x.begin(), x.end()); }
  void reverse_sort() { std::sort(x.begin(), x.end(), std::greater<T>()); }

  T *data() { return x.data(); }
  const T *data() const { return x.data(); }

  auto begin() { return x.begin(); }
  auto end() { return x.end(); }
  auto begin() const { return x.begin(); }
  auto end() const { return x.end(); }

  void swap(VectorType &v) noexcept { x.swap(v.x); }
  void fill(T value) { std::fill(x.begin(), x.end(), value); }

  std::string toString() const {
    std::string str = "[";
    for (int i = 0; i < D - 1; i++)
      str += std::to_string(x[i]) + ", ";
    str += std::to_string(x[D - 1]) + "]";
    return str;
  }

  struct hash {
  private:
    static size_t hash_combine(size_t lhs, size_t rhs) {
      lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
      return lhs;
    }

  public:
    size_t operator()(const VectorType<T, D> &v) const {
      using std::hash;

      /*
        https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
      */
      size_t result = hash<T>()(v[0]);
      result = hash_combine(result, hash<T>()(v[1]));
      if (D == 3) {
        result = hash_combine(result, hash<T>()(v[2]));
      }
      return result;

      // Compute individual hash values for first,
      // second and third and combine them using XOR
      // and bit shifting:

      // size_t result = hash<T>()(v[0]);
      // result ^= hash<T>()(v[1]) << 1;
      // if (D == 3) {
      //   result = (result >> 1) ^ (hash<T>()(v[2]) << 1);
      // }
      // return result;
    }
  };
};

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

template <typename NumericType, int D>
[[nodiscard]] __both__ VectorType<NumericType, D>
Sum(const VectorType<NumericType, D> &pVecA,
    const VectorType<NumericType, D> &pVecB,
    const VectorType<NumericType, D> &pVecC) {
  VectorType<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = pVecA[i] + pVecB[i] + pVecC[i];
  }
  return rr;
}

template <typename NumericType, int D>
[[nodiscard]] __both__ NumericType
DotProduct(const VectorType<NumericType, D> &pVecA,
           const VectorType<NumericType, D> &pVecB) {
  NumericType dot = 0;
  for (size_t i = 0; i < D; ++i) {
    dot += pVecA[i] * pVecB[i];
  }
  return dot;
}

template <typename NumericType>
[[nodiscard]] __both__ Vec3D<NumericType>
CrossProduct(const Vec3D<NumericType> &pVecA, const Vec3D<NumericType> &pVecB) {
  Vec3D<NumericType> rr;
  rr[0] = pVecA[1] * pVecB[2] - pVecA[2] * pVecB[1];
  rr[1] = pVecA[2] * pVecB[0] - pVecA[0] * pVecB[2];
  rr[2] = pVecA[0] * pVecB[1] - pVecA[1] * pVecB[0];
  return rr;
}

template <class NumericType>
[[nodiscard]] __both__ NumericType CrossProduct(const Vec2D<NumericType> &v1,
                                                const Vec2D<NumericType> &v2) {
  return v1[0] * v2[1] - v1[1] * v2[0];
}

template <typename NumericType, int D>
[[nodiscard]] __both__ NumericType Norm(const VectorType<NumericType, D> &vec) {
  NumericType norm = 0;
  for (size_t i = 0; i < D; ++i) {
    norm += vec[i] * vec[i];
  }
  return sqrt(norm);
}

template <typename NumericType, int D>
void __both__ Normalize(VectorType<NumericType, D> &vec) {
  NumericType norm = 1. / Norm(vec);
  if (norm == 1.)
    return;
  for (size_t i = 0; i < D; ++i) {
    vec[i] *= norm;
  }
}

template <typename NumericType, int D>
[[nodiscard]] __both__ VectorType<NumericType, D>
Normalize(const VectorType<NumericType, D> &vec) {
  VectorType<NumericType, D> normedVec = vec;
  NumericType norm = NumericType(1) / Norm(normedVec);
  if (norm == NumericType(1))
    return normedVec;
  for (size_t i = 0; i < D; ++i) {
    normedVec[i] = norm * vec[i];
  }
  return normedVec;
}

template <typename NumericType, int D>
__both__ void ScaleToLength(VectorType<NumericType, D> &vec,
                            const NumericType length) {
  const auto vecLength = Norm(vec);
  for (size_t i = 0; i < D; i++)
    vec[i] *= length / vecLength;
}

template <typename NumericType, int D>
[[nodiscard]] __both__ VectorType<NumericType, D>
Inv(const VectorType<NumericType, D> &vec) {
  VectorType<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = -vec[i];
  }
  return rr;
}

template <typename NumericType, int D>
__both__ VectorType<NumericType, D>
ScaleAdd(const VectorType<NumericType, D> &mult,
         const VectorType<NumericType, D> &add, const NumericType fac) {
  VectorType<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = add[i] + mult[i] * fac;
  }
  return rr;
}

template <typename NumericType, int D>
__both__ [[nodiscard]] NumericType
Distance(const VectorType<NumericType, D> &pVecA,
         const VectorType<NumericType, D> &pVecB) {
  return Norm(pVecA - pVecB);
}

template <typename NumericType>
[[nodiscard]] __both__ Vec3D<NumericType>
ComputeNormal(const Vec3D<Vec3D<NumericType>> &planeCoords) {
  auto uu = planeCoords[1] - planeCoords[0];
  auto vv = planeCoords[2] - planeCoords[0];
  return CrossProduct(uu, vv);
}

template <typename NumericType, int D>
__both__ bool IsNormalized(const VectorType<NumericType, D> &vec) {
  constexpr double eps = 1e-4;
  auto norm = Norm(vec);
  return std::fabs(norm - 1) < eps;
}

template <class T> __both__ Vec2D<T> RotateLeft(const Vec2D<T> &v) {
  return Vec2D<T>(-v[1], v[0]);
}

template <class T> Vec2D<T> __both__ RotateRight(const Vec2D<T> &v) {
  return Vec2D<T>(v[1], -v[0]);
}

template <class T, int D>
__both__ VectorType<T, D> Min(const VectorType<T, D> &v1,
                              const VectorType<T, D> &v2) {
  VectorType<T, D> v;
  for (int i = 0; i < D; i++)
    v[i] = std::min(v1[i], v2[i]);
  return v;
}

template <class T, int D>
__both__ VectorType<T, D> Max(const VectorType<T, D> &v1,
                              const VectorType<T, D> &v2) {
  VectorType<T, D> v;
  for (int i = 0; i < D; i++)
    v[i] = std::max(v1[i], v2[i]);
  return v;
}

template <class T, int D>
__both__ T ElementMax(const VectorType<T, D> &v1, const VectorType<T, D> &v2) {
  VectorType<T, D> v;
  for (int i = 0; i < D; i++)
    v[i] = std::max(v1[i], v2[i]);
  return v;
}

template <class T> __both__ T Volume(const VectorType<T, 2> &p) {
  return p[0] * p[1];
}

template <class T> __both__ T Volume(const VectorType<T, 3> &p) {
  return p[0] * p[1] * p[2];
}

template <int D, class T>
__both__ VectorType<T, D> BitMaskToVector(unsigned int i) {
  VectorType<T, D> tmp(T(0));
  for (unsigned int k = 0; k < D; k++) {
    if (((1 << k) & i) != 0)
      ++tmp[k];
  }
  return tmp;
}

template <class T, int D> __both__ int MinIndex(const VectorType<T, D> &v) {
  int idx = 0;
  for (int i = 1; i < D; i++) {
    if (v[i] < v[idx])
      idx = i;
  }
  return idx;
}

template <class T, int D> __both__ int MaxIndex(const VectorType<T, D> &v) {
  int idx = 0;
  for (int i = 1; i < D; i++) {
    if (v[i] > v[idx])
      idx = i;
  }
  return idx;
}

template <class T> __both__ bool Orientation(const Vec3D<T> *v) {
  return DotProduct(CrossProduct(v[1] - v[0], v[2] - v[0]), v[3] - v[0]) >= -0.;
}

template <class T> __both__ bool Orientation(const Vec2D<T> *v) {
  return DotProduct(RotateLeft(v[1] - v[0]), v[2] - v[0]) >= -0.;
}

template <class T, int D>
__both__ int Compare(const VectorType<T, D> &v1, const VectorType<T, D> &v2) {
  for (int i = D - 1; i >= 0; --i) {
    if (v1[i] > v2[i])
      return 1;
    if (v1[i] < v2[i])
      return -1;
  }
  return 0;
}

template <class T, int D>
__both__ bool AnyEqualElement(const VectorType<T, D> &v1,
                              const VectorType<T, D> &v2) {
  for (int i = 0; i < D - 1; ++i) {
    if (v1[i] == v2[i])
      return true;
  }
  return false;
}

/* ------------- Debug convenience functions ------------- */

template <class S, typename T, int D>
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