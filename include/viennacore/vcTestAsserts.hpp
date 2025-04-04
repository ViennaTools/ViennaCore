#pragma once

#include "vcVectorType.hpp"

#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#define VC_TEST_ASSERT(condition)                                              \
  {                                                                            \
    if (!(condition)) {                                                        \
      throw std::runtime_error(std::string(__FILE__) + std::string(":") +      \
                               std::to_string(__LINE__) +                      \
                               std::string(" in ") +                           \
                               std::string(__PRETTY_FUNCTION__) +              \
                               std::string(" Condition not fulfilled"));       \
    }                                                                          \
  }

#define VC_TEST_ASSERT_ISCLOSE(first, second, eps)                             \
  {                                                                            \
    if ((std::fabs(first - second) > eps)) {                                   \
      throw std::runtime_error(                                                \
          std::string(__FILE__) + std::string(":") +                           \
          std::to_string(__LINE__) + std::string(" in ") +                     \
          std::string(__PRETTY_FUNCTION__) +                                   \
          std::string(" Numbers not close ") + std::to_string(first) +         \
          std::string(" ") + std::to_string(second));                          \
    }                                                                          \
  }

#define VC_TEST_ASSERT_ISNORMAL(first, second, eps)                            \
  {                                                                            \
    if ((std::fabs(viennacore::DotProduct(first, second)) > eps)) {            \
      throw std::runtime_error(std::string(__FILE__) + std::string(":") +      \
                               std::to_string(__LINE__) +                      \
                               std::string(" in ") +                           \
                               std::string(__PRETTY_FUNCTION__) +              \
                               std::string(" Vectors not normal"));            \
    }                                                                          \
  }

#define VC_RUN_2D_TESTS                                                        \
  viennacore::RunTest<double, 2>();                                            \
  viennacore::RunTest<float, 2>();

#define VC_RUN_3D_TESTS                                                        \
  viennacore::RunTest<double, 3>();                                            \
  viennacore::RunTest<float, 3>();

#define VC_RUN_ALL_TESTS                                                       \
  viennacore::RunTest<double, 2>();                                            \
  viennacore::RunTest<double, 3>();                                            \
  viennacore::RunTest<float, 2>();                                             \
  viennacore::RunTest<float, 3>();
