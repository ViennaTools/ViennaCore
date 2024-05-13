#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace core {

template <typename NumericType> using Pair = std::array<NumericType, 2>;

template <typename NumericType> using Triple = std::array<NumericType, 3>;

template <typename NumericType> using Quadruple = std::array<NumericType, 4>;

/* ------------- Vector operation functions ------------- */
template <typename NumericType>
[[nodiscard]] Triple<NumericType> Sum(const Triple<NumericType> &pVecA,
                                      const Triple<NumericType> &pVecB) {
  return {pVecA[0] + pVecB[0], pVecA[1] + pVecB[1], pVecA[2] + pVecB[2]};
}

template <typename NumericType>
[[nodiscard]] Triple<NumericType> Sum(const Triple<NumericType> &pVecA,
                                      const Triple<NumericType> &pVecB,
                                      const Triple<NumericType> &pT) {
  return {pVecA[0] + pVecB[0] + pT[0], pVecA[1] + pVecB[1] + pT[1],
          pVecA[2] + pVecB[2] + pT[2]};
}

template <typename NumericType>
[[nodiscard]] Triple<NumericType> Diff(const Triple<NumericType> &pVecA,
                                       const Triple<NumericType> &pVecB) {
  return {pVecA[0] - pVecB[0], pVecA[1] - pVecB[1], pVecA[2] - pVecB[2]};
}

template <typename NumericType>
[[nodiscard]] Pair<NumericType> Diff(const Pair<NumericType> &pVecA,
                                     const Pair<NumericType> &pVecB) {
  return {pVecA[0] - pVecB[0], pVecA[1] - pVecB[1]};
}

template <typename NumericType>
[[nodiscard]] NumericType DotProduct(const Triple<NumericType> &pVecA,
                                     const Triple<NumericType> &pVecB) {
  return pVecA[0] * pVecB[0] + pVecA[1] * pVecB[1] + pVecA[2] * pVecB[2];
}

template <typename NumericType>
[[nodiscard]] Triple<NumericType>
CrossProduct(const Triple<NumericType> &pVecA,
             const Triple<NumericType> &pVecB) {
  Triple<NumericType> rr;
  rr[0] = pVecA[1] * pVecB[2] - pVecA[2] * pVecB[1];
  rr[1] = pVecA[2] * pVecB[0] - pVecA[0] * pVecB[2];
  rr[2] = pVecA[0] * pVecB[1] - pVecA[1] * pVecB[0];
  return rr;
}

template <typename NumericType, size_t D>
[[nodiscard]] NumericType Norm(const std::array<NumericType, D> &vec) {
  NumericType norm = 0;
  std::for_each(vec.begin(), vec.end(),
                [&norm](NumericType entry) { norm += entry * entry; });
  return std::sqrt(norm);
}

template <typename NumericType, size_t D>
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

template <typename T, size_t D>
void ScaleToLength(std::array<T, D> &vec, const T length) {
  const auto vecLength = Norm(vec);
  for (size_t i = 0; i < D; i++)
    vec[i] *= length / vecLength;
}

template <typename NumericType>
[[nodiscard]] Triple<NumericType> Inv(const Triple<NumericType> &vec) {
  return {-vec[0], -vec[1], -vec[2]};
}

template <typename NumericType>
void Scale(const NumericType pF, Triple<NumericType> &pT) {
  pT[0] *= pF;
  pT[1] *= pF;
  pT[2] *= pF;
}

template <typename NumericType>
[[nodiscard]] Triple<NumericType> Scale(const NumericType pF,
                                        const Triple<NumericType> &pT) {
  return {pF * pT[0], pF * pT[1], pF * pT[2]};
}

template <typename NumericType>
Triple<NumericType> ScaleAdd(const Triple<NumericType> &mult,
                             const Triple<NumericType> &add,
                             const NumericType fac) {
  return {add[0] + mult[0] * fac, add[1] + mult[1] * fac,
          add[2] + mult[2] * fac};
}

template <typename NumericType, size_t D>
[[nodiscard]] NumericType Distance(const std::array<NumericType, D> &pVecA,
                                   const std::array<NumericType, D> &pVecB) {
  auto diff = Diff(pVecA, pVecB);
  return Norm(diff);
}

template <typename NumericType>
[[nodiscard]] Triple<NumericType>
ComputeNormal(const Triple<Triple<NumericType>> &planeCoords) {
  auto uu = Diff(planeCoords[1], planeCoords[0]);
  auto vv = Diff(planeCoords[2], planeCoords[0]);
  return CrossProduct(uu, vv);
}

template <typename NumericType>
bool IsNormalized(const Triple<NumericType> &vec) {
  constexpr double eps = 1e-4;
  auto norm = Norm(vec);
  return std::fabs(norm - 1) < eps;
}

/* ------------- Debug convenience functions ------------- */
template <typename NumericType>
void Print(const Triple<NumericType> &vec, bool endl = true) {
  std::cout << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")"
            << (endl ? "\n" : "");
}

template <typename NumericType> void Print(const Pair<NumericType> &vec) {
  std::cout << "(" << vec[0] << ", " << vec[1] << ")" << std::endl;
}

template <typename NumericType>
void PrintBoundingBox(const Pair<Triple<NumericType>> &bdBox) {
  std::cout << "Bounding box min coords: ";
  Print(bdBox[0]);
  std::cout << "Bounding box max coords: ";
  Print(bdBox[1]);
}

}; // namespace core