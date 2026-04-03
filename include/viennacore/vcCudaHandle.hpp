#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <cuda.h>

#ifndef VIENNACORE_LINK_CUDA_DRIVER
#include <cudaTypedefs.h>
#endif

#ifdef VIENNACORE_LINK_CUDA_DRIVER
namespace viennacore {
struct CudaHandle {
  bool isLoaded() const { return true; }

  decltype(&::cuInit) cuInit_ = &cuInit;
  decltype(&::cuDeviceGetCount) cuDeviceGetCount_ = &cuDeviceGetCount;
  decltype(&::cuDeviceGet) cuDeviceGet_ = &cuDeviceGet;
  decltype(&::cuDeviceGetName) cuDeviceGetName_ = &cuDeviceGetName;

  // Use explicit versioned typedef here to avoid ambiguity.
  decltype(&cuCtxCreate) cuCtxCreate_ = &cuCtxCreate;

  decltype(&::cuCtxSetCurrent) cuCtxSetCurrent_ = &cuCtxSetCurrent;
  decltype(&::cuCtxGetCurrent) cuCtxGetCurrent_ = &cuCtxGetCurrent;
  decltype(&::cuCtxDestroy) cuCtxDestroy_ = &cuCtxDestroy;
  decltype(&::cuCtxSynchronize) cuCtxSynchronize_ = &cuCtxSynchronize;

  decltype(&::cuStreamCreate) cuStreamCreate_ = &cuStreamCreate;
  decltype(&::cuStreamDestroy) cuStreamDestroy_ = &cuStreamDestroy;
  decltype(&::cuStreamSynchronize) cuStreamSynchronize_ = &cuStreamSynchronize;

  decltype(&::cuModuleLoad) cuModuleLoad_ = &cuModuleLoad;
  decltype(&::cuModuleUnload) cuModuleUnload_ = &cuModuleUnload;
  decltype(&::cuModuleGetFunction) cuModuleGetFunction_ = &cuModuleGetFunction;
  decltype(&::cuMemAlloc) cuMemAlloc_ = &cuMemAlloc;
  decltype(&::cuMemcpyHtoD) cuMemcpyHtoD_ = &cuMemcpyHtoD;
  decltype(&::cuMemcpyDtoH) cuMemcpyDtoH_ = &cuMemcpyDtoH;
  decltype(&::cuMemFree) cuMemFree_ = &cuMemFree;
  decltype(&::cuLaunchKernel) cuLaunchKernel_ = &cuLaunchKernel;

  CUresult createContext(CUcontext *ctx, unsigned int flags, CUdevice dev) {
#if CUDA_VERSION >= 13010
    return cuCtxCreate_(ctx, nullptr, flags, dev);
#else
    return cuCtxCreate_(ctx, flags, dev);
#endif
  }
};
} // namespace viennacore
#else

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "vcLogger.hpp"

namespace viennacore {
struct CudaHandle {
  const int cuda_version;
  void *handle = nullptr;

  // cuGetProcAddress itself is loaded from the driver library once.
  using CuGetProcAddressFn = CUresult(CUDAAPI *)(
      const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
      CUdriverProcAddressQueryResult *symbolStatus);

  CuGetProcAddressFn cuGetProcAddress_ = nullptr;

  decltype(&::cuInit) cuInit_ = nullptr;
  decltype(&::cuDeviceGetCount) cuDeviceGetCount_ = nullptr;
  decltype(&::cuDeviceGet) cuDeviceGet_ = nullptr;
  decltype(&::cuDeviceGetName) cuDeviceGetName_ = nullptr;

  // Use explicit versioned typedef here to avoid ambiguity.
  PFN_cuCtxCreate_v3020 cuCtxCreate_ = nullptr;

  decltype(&::cuCtxSetCurrent) cuCtxSetCurrent_ = nullptr;
  decltype(&::cuCtxGetCurrent) cuCtxGetCurrent_ = nullptr;
  decltype(&::cuCtxDestroy) cuCtxDestroy_ = nullptr;
  decltype(&::cuCtxSynchronize) cuCtxSynchronize_ = nullptr;

  decltype(&::cuStreamCreate) cuStreamCreate_ = nullptr;
  decltype(&::cuStreamDestroy) cuStreamDestroy_ = nullptr;
  decltype(&::cuStreamSynchronize) cuStreamSynchronize_ = nullptr;

  decltype(&::cuModuleLoad) cuModuleLoad_ = nullptr;
  decltype(&::cuModuleUnload) cuModuleUnload_ = nullptr;
  decltype(&::cuModuleGetFunction) cuModuleGetFunction_ = nullptr;
  decltype(&::cuMemAlloc) cuMemAlloc_ = nullptr;
  decltype(&::cuMemcpyHtoD) cuMemcpyHtoD_ = nullptr;
  decltype(&::cuMemcpyDtoH) cuMemcpyDtoH_ = nullptr;
  decltype(&::cuMemFree) cuMemFree_ = nullptr;
  decltype(&::cuLaunchKernel) cuLaunchKernel_ = nullptr;

  CudaHandle(int version = CUDA_VERSION) : cuda_version(version) {
#ifdef VIENNACORE_FORCE_NOLOAD_CUDA
    return;
#endif

#ifdef _WIN32
    const char *candidates[] = {"nvcuda.dll", "nvcuda"};
    for (auto *name : candidates) {
      HMODULE hModule = LoadLibraryA(name);
      if (hModule) {
        handle = hModule;
        VIENNACORE_LOG_DEBUG("Successfully loaded CUDA driver library: " +
                             std::string(name));
        if (!load()) {
          VIENNACORE_LOG_WARNING("Failed to load CUDA driver symbols.");
          FreeLibrary(static_cast<HMODULE>(handle));
          handle = nullptr;
        }
        return;
      }
    }
#else
    // Stick to the canonical soname to avoid path/ABI confusion.
    const char *candidates[] = {"libcuda.so.1"};
    for (auto *name : candidates) {
      handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle) {
        VIENNACORE_LOG_DEBUG("Successfully loaded CUDA driver library: " +
                             std::string(name));
        if (!load()) {
          VIENNACORE_LOG_WARNING("Failed to load CUDA driver symbols.");
          dlclose(handle);
          handle = nullptr;
        }
        return;
      }
    }
#endif

    VIENNACORE_LOG_DEBUG("CUDA driver not found.");
  }

  ~CudaHandle() {
    if (handle) {
#ifdef _WIN32
      FreeLibrary(static_cast<HMODULE>(handle));
#else
      dlclose(handle);
#endif
      handle = nullptr;
    }
  }

  bool isLoaded() const { return handle != nullptr; }

  CUresult createContext(CUcontext *ctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate_(ctx, flags, dev);
  }

private:
  template <class Fn> Fn loadDriverExport(const char *symbol) const {
    if (!handle) {
      VIENNACORE_LOG_ERROR("CUDA driver library not loaded.");
      return nullptr;
    }

#ifdef _WIN32
    auto *p = GetProcAddress(static_cast<HMODULE>(handle), symbol);
    if (!p) {
      VIENNACORE_LOG_ERROR(std::string("Missing CUDA export: ") + symbol);
      return nullptr;
    }
    return reinterpret_cast<Fn>(p);
#else
    dlerror(); // clear stale error
    auto *p = dlsym(handle, symbol);
    const char *e = dlerror();
    if (e != nullptr) {
      VIENNACORE_LOG_ERROR(std::string("Missing CUDA export: ") + symbol +
                           " (" + e + ")");
      return nullptr;
    }
    return reinterpret_cast<Fn>(p);
#endif
  }

  template <class Fn>
  bool loadProc(Fn &fn, const char *symbol, int cudaVersion,
                cuuint64_t flags = CU_GET_PROC_ADDRESS_DEFAULT) {
    if (!cuGetProcAddress_) {
      VIENNACORE_LOG_ERROR("Cannot load CUDA proc address before "
                           "cuGetProcAddress is available.");
      fn = nullptr;
      return false;
    }

    void *p = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
    CUresult result =
        cuGetProcAddress_(symbol, &p, cudaVersion, flags, &status);

    if (result != CUDA_SUCCESS || p == nullptr ||
        status != CU_GET_PROC_ADDRESS_SUCCESS) {
      VIENNACORE_LOG_ERROR("Failed to load CUDA symbol via cuGetProcAddress: " +
                           std::string(symbol));
      fn = nullptr;
      return false;
    }

    fn = reinterpret_cast<Fn>(p);
    return true;
  }

  bool load() {
    // First load cuGetProcAddress from the driver library itself.
    cuGetProcAddress_ =
        loadDriverExport<CuGetProcAddressFn>("cuGetProcAddress");
    if (!cuGetProcAddress_) {
      VIENNACORE_LOG_ERROR("cuGetProcAddress not available in CUDA driver.");
      return false;
    }

    bool ok = true;

    // Use CUDA_VERSION for the normal functions.
    ok &= loadProc(cuInit_, "cuInit", cuda_version);
    ok &= loadProc(cuDeviceGetCount_, "cuDeviceGetCount", cuda_version);
    ok &= loadProc(cuDeviceGet_, "cuDeviceGet", cuda_version);
    ok &= loadProc(cuDeviceGetName_, "cuDeviceGetName", cuda_version);

    // Explicit old ABI for cuCtxCreate.
    ok &= loadProc(cuCtxCreate_, "cuCtxCreate", 3020);

    ok &= loadProc(cuCtxSetCurrent_, "cuCtxSetCurrent", cuda_version);
    ok &= loadProc(cuCtxGetCurrent_, "cuCtxGetCurrent", cuda_version);
    ok &= loadProc(cuCtxDestroy_, "cuCtxDestroy", cuda_version);
    ok &= loadProc(cuCtxSynchronize_, "cuCtxSynchronize", cuda_version);

    ok &= loadProc(cuStreamCreate_, "cuStreamCreate", cuda_version);
    ok &= loadProc(cuStreamDestroy_, "cuStreamDestroy", cuda_version);
    ok &= loadProc(cuStreamSynchronize_, "cuStreamSynchronize", cuda_version);

    ok &= loadProc(cuModuleLoad_, "cuModuleLoad", cuda_version);
    ok &= loadProc(cuModuleUnload_, "cuModuleUnload", cuda_version);
    ok &= loadProc(cuModuleGetFunction_, "cuModuleGetFunction", cuda_version);
    ok &= loadProc(cuMemAlloc_, "cuMemAlloc", cuda_version);
    ok &= loadProc(cuMemcpyHtoD_, "cuMemcpyHtoD", cuda_version);
    ok &= loadProc(cuMemcpyDtoH_, "cuMemcpyDtoH", cuda_version);
    ok &= loadProc(cuMemFree_, "cuMemFree", cuda_version);
    ok &= loadProc(cuLaunchKernel_, "cuLaunchKernel", cuda_version);

    return ok;
  }
};
} // namespace viennacore
#endif

#endif // VIENNACORE_COMPILE_GPU