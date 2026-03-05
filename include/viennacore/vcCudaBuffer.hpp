#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <vector>

#include "vcChecks.hpp"
#include "vcContext.hpp"

namespace viennacore {

/// simple wrapper for creating, and managing a device-side CUDA buffer
struct CudaBuffer {
  CudaBuffer() {
    assert(!DeviceContextRegistry::getInstance().isEmpty() &&
           "No DeviceContext registered!");
    context = DeviceContext::getContextFromRegistry(0);
  }

  CudaBuffer(std::shared_ptr<DeviceContext> context) : context(context) {
    assert(context != nullptr && "Context cannot be null!");
  }

#ifndef NDEBUG
  ~CudaBuffer() {
    assert((isRef || allocFreeCount == 0) &&
           "CudaBuffer destroyed without freeing allocated memory!");
  }
#endif

  [[nodiscard]] inline CUdeviceptr dPointer() const { return d_ptr; }

  // free allocated memory
  void free() {
    if (d_ptr == 0) {
      assert(sizeInBytes == 0);
      return;
    }
    CUDA_CHECK(context->ch.cuMemFree(d_ptr));
#ifndef NDEBUG
    --allocFreeCount;
#endif
    d_ptr = 0;
    sizeInBytes = 0;
  }

  // allocate to given number of bytes
  void alloc(size_t size) {
    if (d_ptr != 0)
      free();
    sizeInBytes = size;
    CUDA_CHECK(context->ch.cuMemAlloc(&d_ptr, sizeInBytes));
#ifndef NDEBUG
    ++allocFreeCount;
#endif
  }

  template <typename T> void set(size_t count, const T init) {
    assert(d_ptr != 0);
    assert(sizeInBytes == count * sizeof(T));
    // Create host buffer filled with init value and copy to device
    std::vector<T> initBuffer(count, init);
    CUDA_CHECK(context->ch.cuMemcpyHtoD(d_ptr, initBuffer.data(), sizeInBytes));
  }

  template <typename T> void allocInit(size_t size, const T init) {
    alloc(size * sizeof(T));
    set(size, init);
  }

  template <typename T> void upload(const T *t, size_t count) {
    assert(d_ptr != 0);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(context->ch.cuMemcpyHtoD(d_ptr, (void *)t, count * sizeof(T)));
  }

  template <typename T> void allocUpload(const std::vector<T> &vt) {
    alloc(vt.size() * sizeof(T));
    upload((const T *)vt.data(), vt.size());
  }

  template <typename T> void allocUploadSingle(const T &vt) {
    alloc(sizeof(T));
    upload(&vt, 1);
  }

  template <typename T> void download(T *t, size_t count) {
    assert(d_ptr != 0);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(context->ch.cuMemcpyDtoH((void *)t, d_ptr, count * sizeof(T)));
  }

  std::shared_ptr<DeviceContext> context;
  size_t sizeInBytes{0};
  CUdeviceptr d_ptr{0};
#ifndef NDEBUG
  int allocFreeCount = 0;
  bool isRef = false;
#endif
};

} // namespace viennacore

#endif
