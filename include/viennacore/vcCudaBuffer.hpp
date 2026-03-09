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
  CudaBuffer() : context(*DeviceContextRegistry::getInstance().getContext(0)) {}

  CudaBuffer(const CudaBuffer &other) : context(other.context) {
    // Create a new buffer that shares the same device pointer and size, but
    // does not take ownership of the memory (i.e. will not free it).
    d_ptr = other.d_ptr;
    sizeInBytes = other.sizeInBytes;
    isRef = true;
  }

  CudaBuffer &operator=(const CudaBuffer &other) {
    if (this != &other) {
      // Free existing memory if we own it
      if (!isRef && d_ptr != 0) {
        free();
      }
      context = other.context;
      d_ptr = other.d_ptr;
      sizeInBytes = other.sizeInBytes;
      isRef = true; // This buffer is now a reference to the same memory
    }
    return *this;
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
    if (isRef) {
      // This buffer is a reference to memory owned by another buffer, so we
      // should not free it.
      d_ptr = 0;
      sizeInBytes = 0;
      return;
    }

    if (d_ptr == 0) {
      assert(sizeInBytes == 0);
      return;
    }
    CUDA_CHECK(context.ch.cuMemFree_(d_ptr));
#ifndef NDEBUG
    --allocFreeCount;
#endif
    d_ptr = 0;
    sizeInBytes = 0;
  }

  // allocate to given number of bytes
  void alloc(size_t size) {
    if (d_ptr != 0) {
      // Free existing memory (also if this buffer is a reference)
      CUDA_CHECK(context.ch.cuMemFree_(d_ptr));
#ifndef NDEBUG
      --allocFreeCount;
#endif
      d_ptr = 0;
      sizeInBytes = 0;
    }

    sizeInBytes = size;
    CUDA_CHECK(context.ch.cuMemAlloc_(&d_ptr, sizeInBytes));
#ifndef NDEBUG
    ++allocFreeCount;
#endif
  }

  template <typename T> void set(size_t count, const T init) {
    assert(d_ptr != 0);
    assert(sizeInBytes == count * sizeof(T));
    // Create host buffer filled with init value and copy to device
    std::vector<T> initBuffer(count, init);
    CUDA_CHECK(context.ch.cuMemcpyHtoD_(d_ptr, initBuffer.data(), sizeInBytes));
  }

  template <typename T> void allocInit(size_t size, const T init) {
    alloc(size * sizeof(T));
    set(size, init);
  }

  template <typename T> void upload(const T *t, size_t count) {
    assert(d_ptr != 0);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(context.ch.cuMemcpyHtoD_(d_ptr, (void *)t, count * sizeof(T)));
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
    CUDA_CHECK(context.ch.cuMemcpyDtoH_((void *)t, d_ptr, count * sizeof(T)));
  }

  DeviceContext &context;
  size_t sizeInBytes{0};
  CUdeviceptr d_ptr{0};
  bool isRef = false;
#ifndef NDEBUG
  int allocFreeCount = 0;
#endif
};

} // namespace viennacore

#endif
