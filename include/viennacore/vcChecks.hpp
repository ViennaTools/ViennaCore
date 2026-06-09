#pragma once

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>

#include <iostream>
#include <sstream>

inline const char *getCudaErrorString(CUresult result) {
  switch (result) {
  case CUDA_SUCCESS:
    return "CUDA_SUCCESS";
  case CUDA_ERROR_INVALID_DEVICE:
    return "CUDA_ERROR_INVALID_DEVICE";
  case CUDA_ERROR_INVALID_VALUE:
    return "CUDA_ERROR_INVALID_VALUE";
  case CUDA_ERROR_OUT_OF_MEMORY:
    return "CUDA_ERROR_OUT_OF_MEMORY";
  case CUDA_ERROR_NOT_INITIALIZED:
    return "CUDA_ERROR_NOT_INITIALIZED";
  case CUDA_ERROR_DEINITIALIZED:
    return "CUDA_ERROR_DEINITIALIZED";
  case CUDA_ERROR_NO_DEVICE:
    return "CUDA_ERROR_NO_DEVICE";
  case CUDA_ERROR_INVALID_IMAGE:
    return "CUDA_ERROR_INVALID_IMAGE";
  case CUDA_ERROR_INVALID_CONTEXT:
    return "CUDA_ERROR_INVALID_CONTEXT";
  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
  case CUDA_ERROR_MAP_FAILED:
    return "CUDA_ERROR_MAP_FAILED";
  case CUDA_ERROR_UNMAP_FAILED:
    return "CUDA_ERROR_UNMAP_FAILED";
  case CUDA_ERROR_ARRAY_IS_MAPPED:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";
  case CUDA_ERROR_ALREADY_MAPPED:
    return "CUDA_ERROR_ALREADY_MAPPED";
  case CUDA_ERROR_NO_BINARY_FOR_GPU:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";
  case CUDA_ERROR_ALREADY_ACQUIRED:
    return "CUDA_ERROR_ALREADY_ACQUIRED";
  case CUDA_ERROR_NOT_MAPPED:
    return "CUDA_ERROR_NOT_MAPPED";
  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
  case CUDA_ERROR_ECC_UNCORRECTABLE:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";
  case CUDA_ERROR_UNSUPPORTED_LIMIT:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";
  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
    return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
  case CUDA_ERROR_INVALID_PTX:
    return "CUDA_ERROR_INVALID_PTX";
  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
    return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
  case CUDA_ERROR_NVLINK_UNCORRECTABLE:
    return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
  case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
    return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
  case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
    return "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
  case CUDA_ERROR_NOT_SUPPORTED:
    return "CUDA_ERROR_NOT_SUPPORTED";
  case CUDA_ERROR_UNKNOWN:
    return "CUDA_ERROR_UNKNOWN";
  default:
    return "CUDA_ERROR_UNKNOWN (unrecognized error code)";
  }
}

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    CUresult rc = call;                                                        \
    if (rc != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "\033[1;31mCuda Error %s (%s: line %d)\033[0m\n",        \
              getCudaErrorString(rc), __FILE__, __LINE__);                     \
      exit(2);                                                                 \
    }                                                                          \
  }

#define CUDA_CHECK_NOEXCEPT(call)                                              \
  {                                                                            \
    call;                                                                      \
  }

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

#define OPTIX_CHECK_RESULT(res)                                                \
  {                                                                            \
    if (res != OPTIX_SUCCESS) {                                                \
      fprintf(stderr,                                                          \
              "\033[1;31mOptix failed with code %d (%s: line %d): "            \
              "%s\033[0m\n",                                                   \
              res, __FILE__, __LINE__, optixGetErrorString(res));              \
      exit(2);                                                                 \
    }                                                                          \
  }

#define CUDA_SYNC_CHECK()                                                      \
  {                                                                            \
    CUresult error = cuCtxSynchronize();                                       \
    if (error != CUDA_SUCCESS) {                                               \
      fprintf(stderr, "\033[1;31mCuda Error %s (%s: line %d)\033[0m\n",        \
              getCudaErrorString(error), __FILE__, __LINE__);                  \
      exit(2);                                                                 \
    }                                                                          \
  }
