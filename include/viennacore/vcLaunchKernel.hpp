#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <string>

#include "vcContext.hpp"

namespace viennacore {

// wrapper for launching kernel from a ptx file
class LaunchKernel {
public:
  static void launch(const std::string &moduleName,
                     const std::string &kernelName, void **kernel_args,
                     DeviceContext &context,
                     unsigned long sharedMemoryInBytes = 0) {

    CUmodule module = context.getModule(moduleName);
    CUfunction function;
    context.ch.cuModuleGetFunction_(&function, module, kernelName.data());
    context.ch.cuLaunchKernel_(function,              // function to call
                               blocks, 1, 1,          /* grid dims */
                               threadsPerBlock, 1, 1, /* block dims */
                               sharedMemoryInBytes *
                                   threadsPerBlock, // shared memory
                               0,                   // stream
                               kernel_args,         // kernel parameters
                               nullptr);
  }

  static void launchSingle(const std::string &moduleName,
                           const std::string &kernelName, void **kernel_args,
                           DeviceContext &context,
                           unsigned long sharedMemoryInBytes = 0) {

    CUmodule module = context.getModule(moduleName);
    CUfunction function;
    context.ch.cuModuleGetFunction_(&function, module, kernelName.data());
    context.ch.cuLaunchKernel_(function,            // function to call
                               1, 1, 1,             /* grid dims */
                               1, 1, 1,             /* block dims */
                               sharedMemoryInBytes, // shared memory
                               0,                   // stream
                               kernel_args,         // kernel parameters
                               nullptr);
  }

  static constexpr int blocks = 512;
  static constexpr int threadsPerBlock = 512;
};

} // namespace viennacore

#endif
