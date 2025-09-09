#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <iostream>
#include <sstream>

#include "vcLogger.hpp"

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t rc = cuda##call;                                               \
    if (rc != cudaSuccess) {                                                   \
      std::stringstream txt;                                                   \
      cudaError_t err = rc; /*cudaGetLastError();*/                            \
      txt << TM_RED << "CUDA Error " << cudaGetErrorName(err) << " ("          \
          << cudaGetErrorString(err) << ")" << TM_RESET;                       \
      std::cerr << txt.str() << std::endl;                                     \
      exit(2);                                                                 \
    }                                                                          \
  }

#define CUDA_CHECK_NOEXCEPT(call)                                              \
  { cuda##call; }

#define OPTIX_CHECK(call)                                                      \
  {                                                                            \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      fprintf(stderr,                                                          \
              "\033[1;31mOptix call (%s) failed with code %d (%s: line %d): "  \
              "%s\033[0m\n",                                                   \
              #call, res, __FILE__, __LINE__, optixGetErrorString(res));       \
      exit(2);                                                                 \
    }                                                                          \
  }

#define CUDA_SYNC_CHECK()                                                      \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "\033[1;31mCuda Error (%s: line %d): %s\033[0m\n",       \
              __FILE__, __LINE__, cudaGetErrorString(error));                  \
      exit(2);                                                                 \
    }                                                                          \
  }
