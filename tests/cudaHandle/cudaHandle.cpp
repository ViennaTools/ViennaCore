#include <cuda.h>

#include "vcCudaHandle.hpp"

#include <iostream>

using namespace viennacore;

int main() {
  CudaHandle handle;
  handle.call("cuInit", 0);

  // Check how many devices are available
  int deviceCount = 0;
  handle.call("cuDeviceGetCount", &deviceCount);
  std::cout << "Device count: " << deviceCount << std::endl;

  try {
    auto nonExisting = handle.load<OptixResult (*)(
        CUcontext, const OptixDeviceContextOptions *, OptixDeviceContext *)>(
        "optixDeviceContextCreate");
  } catch (const std::runtime_error &e) {
    // Expected to fail
    std::cout << "Caught expected error: " << e.what() << std::endl;
  }

  // Get the first CUDA device
  CUdevice device;
  handle.call("cuDeviceGet", &device, 0);

  // Create a new context for the device
  CUcontext context;
  handle.call("cuCtxCreate", &context, 0, device);

  std::cout << "Successfully created context: " << context << std::endl;

  handle.load();

  // Now allocate memory
  CUdeviceptr d_ptr{0};
  CUresult result = handle.cuMemAlloc(&d_ptr, 1024);

  if (result != 0) {
    std::cout << "cuMemAlloc failed with error code: " << result << std::endl;
  } else {
    std::cout << "Successfully allocated CUDA memory at device pointer: "
              << d_ptr << std::endl;
  }

  // Free the allocated memory
  if (d_ptr) {
    CUresult freeResult = handle.cuMemFree(d_ptr);
    if (freeResult != 0) {
      std::cout << "cuMemFree failed with error code: " << freeResult
                << std::endl;
    } else {
      std::cout << "Successfully freed CUDA memory at device pointer: " << d_ptr
                << std::endl;
    }
  }
}