#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <cuda.h>
#include <cudaTypedefs.h>

#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "vcLogger.hpp"

namespace viennacore {

struct CudaHandle {
  void *handle = nullptr;

  // cuGetProcAddress itself is loaded from the driver library once.
  using CuGetProcAddressFn = CUresult(CUDAAPI *)(
      const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
      CUdriverProcAddressQueryResult *symbolStatus);

  CuGetProcAddressFn cuGetProcAddress_ = nullptr;

  PFN_cuInit cuInit_ = nullptr;
  PFN_cuDeviceGetCount cuDeviceGetCount_ = nullptr;
  PFN_cuDeviceGet cuDeviceGet_ = nullptr;
  PFN_cuDeviceGetName cuDeviceGetName_ = nullptr;

  // Use explicit versioned typedef here to avoid ambiguity.
  PFN_cuCtxCreate_v3020 cuCtxCreate_ = nullptr;

  PFN_cuCtxSetCurrent cuCtxSetCurrent_ = nullptr;
  PFN_cuCtxGetCurrent cuCtxGetCurrent_ = nullptr;
  PFN_cuCtxDestroy cuCtxDestroy_ = nullptr;
  PFN_cuCtxSynchronize cuCtxSynchronize_ = nullptr;
  PFN_cuModuleLoad cuModuleLoad_ = nullptr;
  PFN_cuModuleUnload cuModuleUnload_ = nullptr;
  PFN_cuModuleGetFunction cuModuleGetFunction_ = nullptr;
  PFN_cuMemAlloc cuMemAlloc_ = nullptr;
  PFN_cuMemcpyHtoD cuMemcpyHtoD_ = nullptr;
  PFN_cuMemcpyDtoH cuMemcpyDtoH_ = nullptr;
  PFN_cuMemFree cuMemFree_ = nullptr;
  PFN_cuLaunchKernel cuLaunchKernel_ = nullptr;

  CudaHandle() {
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
        load();
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
        load();
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

private:
  template <class Fn> Fn loadDriverExport(const char *symbol) const {
    if (!handle) {
      VIENNACORE_LOG_ERROR(std::string("Cannot load CUDA export: ") + symbol +
                           " (CUDA driver library not loaded)");
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
    if (!handle) {
      VIENNACORE_LOG_ERROR("CUDA driver library not loaded.");
      return false;
    }

    // First load cuGetProcAddress from the driver library itself.
    cuGetProcAddress_ =
        loadDriverExport<CuGetProcAddressFn>("cuGetProcAddress");
    if (!cuGetProcAddress_) {
      VIENNACORE_LOG_ERROR("cuGetProcAddress not available in CUDA driver.");
      return false;
    }

    bool ok = true;

    // Use CUDA_VERSION for the normal functions.
    ok &= loadProc(cuInit_, "cuInit", CUDA_VERSION);
    ok &= loadProc(cuDeviceGetCount_, "cuDeviceGetCount", CUDA_VERSION);
    ok &= loadProc(cuDeviceGet_, "cuDeviceGet", CUDA_VERSION);
    ok &= loadProc(cuDeviceGetName_, "cuDeviceGetName", CUDA_VERSION);

    // Explicit old ABI for cuCtxCreate.
    ok &= loadProc(cuCtxCreate_, "cuCtxCreate", 3020);

    ok &= loadProc(cuCtxSetCurrent_, "cuCtxSetCurrent", CUDA_VERSION);
    ok &= loadProc(cuCtxGetCurrent_, "cuCtxGetCurrent", CUDA_VERSION);
    ok &= loadProc(cuCtxDestroy_, "cuCtxDestroy", CUDA_VERSION);
    ok &= loadProc(cuCtxSynchronize_, "cuCtxSynchronize", CUDA_VERSION);
    ok &= loadProc(cuModuleLoad_, "cuModuleLoad", CUDA_VERSION);
    ok &= loadProc(cuModuleUnload_, "cuModuleUnload", CUDA_VERSION);
    ok &= loadProc(cuModuleGetFunction_, "cuModuleGetFunction", CUDA_VERSION);
    ok &= loadProc(cuMemAlloc_, "cuMemAlloc", CUDA_VERSION);
    ok &= loadProc(cuMemcpyHtoD_, "cuMemcpyHtoD", CUDA_VERSION);
    ok &= loadProc(cuMemcpyDtoH_, "cuMemcpyDtoH", CUDA_VERSION);
    ok &= loadProc(cuMemFree_, "cuMemFree", CUDA_VERSION);
    ok &= loadProc(cuLaunchKernel_, "cuLaunchKernel", CUDA_VERSION);

    return ok;
  }
};

} // namespace viennacore

#endif // VIENNACORE_COMPILE_GPU