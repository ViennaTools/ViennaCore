#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <cassert>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

#include "vcChecks.hpp"
#include "vcLogger.hpp"

struct CudaHandle {
  void *handle = nullptr;

  CudaHandle() {
    const char *candidates[] = {
        "libcuda.so.1",
        "libcuda.so",
    };
    for (auto *name : candidates) {
      handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle) {
        VIENNACORE_LOG_DEBUG("Successfully loaded CUDA driver library: " +
                             std::string(name));
        return;
      }
    }
    VIENNACORE_LOG_ERROR(std::string("CUDA driver not found. ") +
                         "Install CUDA driver (libcuda) and ensure it "
                         "is on the loader path.");
  }

  ~CudaHandle() {
    if (handle)
      dlclose(handle);
  }

  void load() {
    cuMemAlloc = load<CUresult (*)(CUdeviceptr *, size_t)>("cuMemAlloc");
    cuMemcpyHtoD =
        load<CUresult (*)(CUdeviceptr, const void *, size_t)>("cuMemcpyHtoD");
    cuMemcpyDtoH =
        load<CUresult (*)(void *, CUdeviceptr, size_t)>("cuMemcpyDtoH");
    cuMemFree = load<CUresult (*)(CUdeviceptr)>("cuMemFree");
  }

  template <class Fn> Fn load(const char *symbol) {
    assert(handle != nullptr);
    dlerror(); // clear
    auto *p = dlsym(handle, symbol);
    const char *e = dlerror();
    if (e)
      VIENNACORE_LOG_ERROR(std::string("Missing CUDA symbol: ") + symbol +
                           " (" + e + ")");
    return reinterpret_cast<Fn>(p);
  }

  template <class... Args> auto call(const char *symbol, Args... args) {
    auto fn = load<CUresult (*)(Args...)>(symbol);
    CUDA_CHECK(fn(args...));
  }

  // CUDA driver API function pointers
  CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = nullptr;
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t) = nullptr;
  CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t) = nullptr;
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
};

#endif // VIENNACORE_COMPILE_GPU