#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <cuda.h>
#include <cudaTypedefs.h>

#ifdef VIENNACORE_LINK_CUDA_DRIVER
namespace viennacore {
struct CudaHandle {
  bool isLoaded() const { return true; }

  PFN_cuInit cuInit_ = &cuInit;
  PFN_cuDeviceGetCount cuDeviceGetCount_ = &cuDeviceGetCount;
  PFN_cuDeviceGet cuDeviceGet_ = &cuDeviceGet;
  PFN_cuDeviceGetName cuDeviceGetName_ = &cuDeviceGetName;

  // Use explicit versioned typedef here to avoid ambiguity.
  PFN_cuCtxCreate_v3020 cuCtxCreate_ = &cuCtxCreate_v2;

  PFN_cuCtxSetCurrent cuCtxSetCurrent_ = &cuCtxSetCurrent;
  PFN_cuCtxGetCurrent cuCtxGetCurrent_ = &cuCtxGetCurrent;
  PFN_cuCtxDestroy cuCtxDestroy_ = &cuCtxDestroy;
  PFN_cuCtxSynchronize cuCtxSynchronize_ = &cuCtxSynchronize;

  PFN_cuStreamCreate cuStreamCreate_ = &cuStreamCreate;
  PFN_cuStreamDestroy cuStreamDestroy_ = &cuStreamDestroy;
  PFN_cuStreamSynchronize cuStreamSynchronize_ = &cuStreamSynchronize;

  PFN_cuModuleLoad cuModuleLoad_ = &cuModuleLoad;
  PFN_cuModuleUnload cuModuleUnload_ = &cuModuleUnload;
  PFN_cuModuleGetFunction cuModuleGetFunction_ = &cuModuleGetFunction;
  PFN_cuMemAlloc cuMemAlloc_ = &cuMemAlloc;
  PFN_cuMemcpyHtoD cuMemcpyHtoD_ = &cuMemcpyHtoD;
  PFN_cuMemcpyDtoH cuMemcpyDtoH_ = &cuMemcpyDtoH;
  PFN_cuMemFree cuMemFree_ = &cuMemFree;
  PFN_cuLaunchKernel cuLaunchKernel_ = &cuLaunchKernel;
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

  PFN_cuStreamCreate cuStreamCreate_ = nullptr;
  PFN_cuStreamDestroy cuStreamDestroy_ = nullptr;
  PFN_cuStreamSynchronize cuStreamSynchronize_ = nullptr;

  PFN_cuModuleLoad cuModuleLoad_ = nullptr;
  PFN_cuModuleUnload cuModuleUnload_ = nullptr;
  PFN_cuModuleGetFunction cuModuleGetFunction_ = nullptr;
  PFN_cuMemAlloc cuMemAlloc_ = nullptr;
  PFN_cuMemcpyHtoD cuMemcpyHtoD_ = nullptr;
  PFN_cuMemcpyDtoH cuMemcpyDtoH_ = nullptr;
  PFN_cuMemFree cuMemFree_ = nullptr;
  PFN_cuLaunchKernel cuLaunchKernel_ = nullptr;

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