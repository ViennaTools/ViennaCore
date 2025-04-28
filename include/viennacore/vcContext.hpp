#pragma once

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <optix_stubs.h>
#include <string>
#include <vector>

#include "vcChecks.hpp"
#include "vcLogger.hpp"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#define STRINGIFY_HELPER(X) #X
#define STRINGIFY(X) STRINGIFY_HELPER(X)

#ifndef VIENNACORE_KERNELS_PATH_DEFINE
#define VIENNACORE_KERNELS_PATH_DEFINE
#endif

#define VIENNACORE_KERNELS_PATH STRINGIFY(VIENNACORE_KERNELS_PATH_DEFINE)

namespace viennacore {

static void contextLogCallback(unsigned int level, const char *tag,
                               const char *message, void *) {
#ifndef NDEBUG
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
#endif
}

struct Context {
  void create(std::filesystem::path modulePath = VIENNACORE_KERNELS_PATH,
              const int deviceID = 0);
  CUmodule getModule(const std::string &moduleName);
  void addModule(const std::string &moduleName);
  std::string getModulePath() const { return modulePath.string(); }
  void destroy() {
    if (deviceID == -1)
      return;

    for (auto module : modules) {
      cuModuleUnload(module);
    }
    optixDeviceContextDestroy(optix);
    cuCtxDestroy(cuda);
    deviceID = -1;
  }

  std::filesystem::path modulePath;
  std::vector<std::string> moduleNames;
  std::vector<CUmodule> modules;

  CUcontext cuda;
  cudaDeviceProp deviceProps;
  OptixDeviceContext optix;
  int deviceID = -1;
};

CUmodule Context::getModule(const std::string &moduleName) {
  int idx = -1;
  for (int i = 0; i < modules.size(); i++) {
    if (this->moduleNames[i] == moduleName) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    viennacore::Logger::getInstance()
        .addError("Module " + moduleName + " not in context.")
        .print();
  }

  return modules[idx];
}

void Context::addModule(const std::string &moduleName) {
  if (deviceID == -1) {
    viennacore::Logger::getInstance()
        .addError("Context not initialized. Use 'create' to "
                  "initialize context.")
        .print();
  }

  if (std::find(moduleNames.begin(), moduleNames.end(), moduleName) !=
      moduleNames.end()) {
    return;
  }

  CUmodule module;
  CUresult err;
  err = cuModuleLoad(&module, (modulePath / moduleName).c_str());
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance().addModuleError(moduleName, err).print();

  modules.push_back(module);
  moduleNames.push_back(moduleName);
}

void Context::create(std::filesystem::path modulePath, const int deviceID) {

  // create new context
  this->modulePath = modulePath;
  this->deviceID = deviceID;

  // initialize CUDA runtime API (cuda## prefix, cuda_runtime_api.h)
  CUDA_CHECK(Free(0));

  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    viennacore::Logger::getInstance()
        .addError("No CUDA capable devices found!")
        .print();
  }

  cudaSetDevice(deviceID);
  cudaGetDeviceProperties(&deviceProps, deviceID);
  viennacore::Logger::getInstance()
      .addDebug("Running on device: " + std::string(deviceProps.name))
      .print();

  // initialize CUDA driver API (cu## prefix, cuda.h)
  // we need the CUDA driver API to load kernels from PTX files
  CUresult err;
  err = cuInit(0);
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance().addModuleError("cuInit", err).print();

  err = cuCtxGetCurrent(&(cuda));
  if (err != CUDA_SUCCESS) {
    viennacore::Logger::getInstance()
        .addError("Error querying current context: error code " +
                  std::to_string(err))
        .print();
  }

  // add default modules
  viennacore::Logger::getInstance()
      .addDebug("PTX kernels path: " + modulePath.string())
      .print();

  // initialize OptiX context
  OPTIX_CHECK(optixInit());

  optixDeviceContextCreate(cuda, 0, &optix);
  optixDeviceContextSetLogCallback(optix, contextLogCallback, nullptr, 4);
}

} // namespace viennacore