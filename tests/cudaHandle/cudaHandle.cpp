#include <cuda.h>

#include "vcCudaHandle.hpp"

#include <iostream>

using namespace viennacore;

int main() {
  CudaHandle handle;
  handle.cuInit_(0);

  // Check how many devices are available
  int deviceCount = 0;
  handle.cuDeviceGetCount_(&deviceCount);
  std::cout << "Device count: " << deviceCount << std::endl;

  // Get the first CUDA device
  CUdevice device;
  handle.cuDeviceGet_(&device, 0);

  // Create a new context for the device
  CUcontext context;
  handle.createContext(&context, 0, device);

  std::cout << "Successfully created context: " << context << std::endl;

  // Now allocate memory
  CUdeviceptr d_ptr{0};
  CUresult result = handle.cuMemAlloc_(&d_ptr, 1024);

  if (result != 0) {
    std::cout << "cuMemAlloc failed with error code: " << result << std::endl;
  } else {
    std::cout << "Successfully allocated CUDA memory at device pointer: "
              << d_ptr << std::endl;
  }

  // Free the allocated memory
  if (d_ptr) {
    CUresult freeResult = handle.cuMemFree_(d_ptr);
    if (freeResult != 0) {
      std::cout << "cuMemFree failed with error code: " << freeResult
                << std::endl;
    } else {
      std::cout << "Successfully freed CUDA memory at device pointer: " << d_ptr
                << std::endl;
    }
  }
}