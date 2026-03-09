#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <cassert>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "vcChecks.hpp"
#include "vcLogger.hpp"

namespace viennacore {

struct CudaHandle {
  void *handle = nullptr;

  CudaHandle() {
#ifdef VIENNACORE_FORCE_NOLOAD_CUDA
    return;
#endif

#ifdef _WIN32
    const char *candidates[] = {
        "nvcuda.dll",
        "nvcuda",
    };
    for (auto *name : candidates) {
      HMODULE hModule = LoadLibraryA(name);
      if (hModule) {
        handle = hModule;
        VIENNACORE_LOG_DEBUG("Successfully loaded CUDA driver library: " +
                             std::string(name));
        return;
      }
    }
    VIENNACORE_LOG_DEBUG("CUDA driver not found.");
#else
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
    VIENNACORE_LOG_DEBUG("CUDA driver not found.");
#endif
  }

  ~CudaHandle() {
    if (handle) {
#ifdef _WIN32
      FreeLibrary(static_cast<HMODULE>(handle));
#else
      dlclose(handle);
#endif
    }
  }

  void load() {
    cuMemAlloc = load<CUresult (*)(CUdeviceptr *, size_t)>("cuMemAlloc");
    cuMemcpyHtoD =
        load<CUresult (*)(CUdeviceptr, const void *, size_t)>("cuMemcpyHtoD");
    cuMemcpyDtoH =
        load<CUresult (*)(void *, CUdeviceptr, size_t)>("cuMemcpyDtoH");
    cuMemFree = load<CUresult (*)(CUdeviceptr)>("cuMemFree");
  }

  template <class Fn> Fn load(const char *symbol) const {
    if (!handle) {
      VIENNACORE_LOG_ERROR(std::string("Cannot load CUDA symbol: ") + symbol +
                           " (CUDA driver library not loaded)");
      return nullptr;
    }

#ifdef _WIN32
    auto *p = GetProcAddress(static_cast<HMODULE>(handle), symbol);
    if (!p) {
      VIENNACORE_LOG_ERROR(std::string("Missing CUDA symbol: ") + symbol);
    }
    return reinterpret_cast<Fn>(p);
#else
    auto *p = dlsym(handle, symbol);
    const char *e = dlerror();
    if (e)
      VIENNACORE_LOG_ERROR(std::string("Missing CUDA symbol: ") + symbol +
                           " (" + e + ")");
    return reinterpret_cast<Fn>(p);
#endif
  }

  template <class... Args> auto call(const char *symbol, Args &&...args) const {
    auto fn = load<CUresult (*)(std::remove_reference_t<Args>...)>(symbol);
    CUDA_CHECK(fn(std::forward<Args>(args)...));
  }

  // CUDA driver API function pointers
  CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = nullptr;
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t) = nullptr;
  CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t) = nullptr;
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
};

} // namespace viennacore
#endif // VIENNACORE_COMPILE_GPU