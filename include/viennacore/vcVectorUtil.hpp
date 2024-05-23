#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace viennacore {

template <typename NumericType> using Pair = std::array<NumericType, 2>;

template <typename NumericType> using Triple = std::array<NumericType, 3>;

template <typename NumericType> using Quadruple = std::array<NumericType, 4>;

/* ------------- Vector operation functions ------------- */
template <typename NumericType, std::size_t D>
[[nodiscard]] std::array<NumericType, D>
Sum(const std::array<NumericType, D> &pVecA,
    const std::array<NumericType, D> &pVecB) {
  std::array<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = pVecA[i] + pVecB[i];
  }
  return rr;
}

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
[[nodiscard]] std::array<NumericType, D>
Diff(const std::array<NumericType, D> &pVecA,
     const std::array<NumericType, D> &pVecB) {
  std::array<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = pVecA[i] - pVecB[i];
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
[[nodiscard]] Triple<NumericType>
CrossProduct(const Triple<NumericType> &pVecA,
             const Triple<NumericType> &pVecB) {
  Triple<NumericType> rr;
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
void Scale(const NumericType pF, std::array<NumericType, D> &pT) {
  for (size_t i = 0; i < D; ++i) {
    pT[i] *= pF;
  }
}

template <typename NumericType, std::size_t D>
[[nodiscard]] std::array<NumericType, D>
Scale(const NumericType pF, const std::array<NumericType, D> &pT) {
  std::array<NumericType, D> rr;
  for (size_t i = 0; i < D; ++i) {
    rr[i] = pF * pT[i];
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

template <typename NumericType, std::size_t D>
bool IsNormalized(const std::array<NumericType, D> &vec) {
  constexpr double eps = 1e-4;
  auto norm = Norm(vec);
  return std::fabs(norm - 1) < eps;
}

/* ------------- Debug convenience functions ------------- */
template <typename NumericType, std::size_t D>
void Print(const std::array<NumericType, D> &vec, bool endl = true) {
  std::cout << "(";
  for (size_t i = 0; i < D; ++i) {
    std::cout << vec[i];
    if (i < D - 1)
      std::cout << ", ";
  }
  std::cout << ")" << (endl ? "\n" : "");
}

template <typename NumericType>
void PrintBoundingBox(const Pair<Triple<NumericType>> &bdBox) {
  std::cout << "Bounding box min coords: ";
  Print(bdBox[0]);
  std::cout << "Bounding box max coords: ";
  Print(bdBox[1]);
}

}; // namespace viennacore