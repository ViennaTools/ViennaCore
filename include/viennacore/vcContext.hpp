#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// this include may only appear in a single source file:
#ifndef VIENNACORE_COMPILE_SHARED_LIB
#include <optix_function_table_definition.h>
#endif
#include <optix_stubs.h>

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vcChecks.hpp"
#include "vcLogger.hpp"

#define STRINGIFY_HELPER(X) #X
#define STRINGIFY(X) STRINGIFY_HELPER(X)

#ifdef VIENNACORE_KERNELS_PATH_OVERRIDE
#define VIENNACORE_KERNELS_PATH STRINGIFY(VIENNACORE_KERNELS_PATH_OVERRIDE)
#else
#ifndef VIENNACORE_KERNELS_PATH_DEFINE
#define VIENNACORE_KERNELS_PATH_DEFINE .
#endif
#define VIENNACORE_KERNELS_PATH STRINGIFY(VIENNACORE_KERNELS_PATH_DEFINE)
#endif

#ifdef VIENNACORE_DYNAMIC_MODULE_PATH
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace viennacore {

/**
 * Global DeviceContext Registry Usage Examples:
 *
 * // Create and register a context for device 0
 * auto context0 = DeviceContext::createContext("/path/to/modules", 0);
 *
 * // Later, query the context from anywhere in your code
 * auto queriedContext = DeviceContext::getContextFromRegistry(0);
 *
 * // Check if a context exists for a device
 * if (DeviceContext::hasContextInRegistry(1)) {
 *   auto context1 = DeviceContext::getContextFromRegistry(1);
 * }
 *
 * // Get all registered device IDs
 * auto deviceIDs = DeviceContext::getRegisteredDeviceIDs();
 *
 * // Create a context without registering it globally
 * auto localContext = DeviceContext::createContext("/path/to/modules", 2,
 * false);
 */

static void contextLogCallback(unsigned int level, const char *tag,
                               const char *message, void *) {
#ifdef VIENNACORE_CUDA_LOG_DEBUG
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
#endif
}

// Forward declaration
struct DeviceContext;

class DeviceContextRegistry {
public:
  static DeviceContextRegistry &getInstance() {
    static DeviceContextRegistry instance;
    return instance;
  }

  // Register a context with the registry
  void registerContext(int deviceID, std::shared_ptr<DeviceContext> context) {
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_[deviceID] = context;
  }

  // Get a context by device ID
  std::shared_ptr<DeviceContext> getContext(int deviceID) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(deviceID);
    if (it != contexts_.end()) {
      return it->second;
    }
    return nullptr;
  }

  // Check if a context exists for a device ID
  bool hasContext(int deviceID) {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.find(deviceID) != contexts_.end();
  }

  // Remove a context from the registry
  void unregisterContext(int deviceID) {
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_.erase(deviceID);
  }

  // Get all registered device IDs
  std::vector<int> getRegisteredDeviceIDs() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> deviceIDs;
    for (const auto &pair : contexts_) {
      deviceIDs.push_back(pair.first);
    }
    return deviceIDs;
  }

  // Clear all contexts from the registry
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_.clear();
  }

  bool isEmpty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.empty();
  }

private:
  DeviceContextRegistry() = default;
  ~DeviceContextRegistry() = default;
  DeviceContextRegistry(const DeviceContextRegistry &) = delete;
  DeviceContextRegistry &operator=(const DeviceContextRegistry &) = delete;

  std::unordered_map<int, std::shared_ptr<DeviceContext>> contexts_;
  std::mutex mutex_;
};

struct DeviceContext {
  // Static factory methods for creating and managing contexts
  static std::shared_ptr<DeviceContext>
  createContext(std::filesystem::path modulePath = VIENNACORE_KERNELS_PATH,
                const int deviceID = 0, bool registerInGlobal = true);

  static std::shared_ptr<DeviceContext> getContextFromRegistry(int deviceID);

  static bool hasContextInRegistry(int deviceID);

  static std::vector<int> getRegisteredDeviceIDs();

  // Instance methods
  void create(std::filesystem::path modulePath = VIENNACORE_KERNELS_PATH,
              const int deviceID = 0);
  CUmodule getModule(const std::string &moduleName);
  void addModule(const std::string &moduleName);
  std::string getModulePath() const { return modulePath.string(); }
  std::string getDeviceName() const { return deviceProps.name; }
  int getDeviceID() const { return deviceID; }

  void destroy() {
    if (deviceID == -1)
      return;

    for (auto module : modules) {
      cuModuleUnload(module);
    }
    optixDeviceContextDestroy(optix);
    cuCtxDestroy(cuda);

    // Unregister from global registry
    DeviceContextRegistry::getInstance().unregisterContext(deviceID);

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

inline CUmodule DeviceContext::getModule(const std::string &moduleName) {
  int idx = -1;
  for (int i = 0; i < modules.size(); i++) {
    if (this->moduleNames[i] == moduleName) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    VIENNACORE_LOG_ERROR("Module " + moduleName + " not in context.");
  }

  return modules[idx];
}

inline void DeviceContext::addModule(const std::string &moduleName) {
  if (deviceID == -1) {
    VIENNACORE_LOG_ERROR(
        "Context not initialized. Use 'create' to initialize context.");
  }

  // Check if module already loaded
  if (std::find(moduleNames.begin(), moduleNames.end(), moduleName) !=
      moduleNames.end()) {
    return;
  }

  CUmodule module;
  CUresult err;
  const std::string p = (modulePath / moduleName).string();
  err = cuModuleLoad(&module, p.c_str());
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance().addModuleError(moduleName, err).print();

  modules.push_back(module);
  moduleNames.push_back(moduleName);
}

inline void DeviceContext::create(std::filesystem::path modulePath,
                                  const int deviceID) {

  // create new context
  this->modulePath = modulePath;
  this->deviceID = deviceID;

  // initialize CUDA **runtime** API (cuda## prefix, cuda_runtime_api.h)
  CUDA_CHECK(Free(0));

  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    VIENNACORE_LOG_ERROR("No CUDA capable devices found!");
  }

  cudaSetDevice(deviceID);
  cudaGetDeviceProperties(&deviceProps, deviceID);
  VIENNACORE_LOG_DEBUG("Registered context for device: " +
                       std::string(deviceProps.name));

  // initialize CUDA **driver** API (cu## prefix, cuda.h)
  // we need the CUDA driver API to load kernels from PTX files
  CUresult err;
  err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    viennacore::Logger::getInstance().addModuleError("cuInit", err).print();
  }

  err = cuCtxGetCurrent(&(cuda));
  if (err != CUDA_SUCCESS) {
    VIENNACORE_LOG_ERROR("Error querying current context: error code " +
                         std::to_string(err));
  }

  // add default modules
  VIENNACORE_LOG_DEBUG("PTX kernels path: " + modulePath.string());
  // no default modules for now

  // initialize OptiX context
  OPTIX_CHECK(optixInit());

  optixDeviceContextCreate(cuda, 0, &optix);
  optixDeviceContextSetLogCallback(optix, contextLogCallback, nullptr, 4);
}

// Static factory method implementations
inline std::shared_ptr<DeviceContext>
DeviceContext::createContext(std::filesystem::path modulePath,
                             const int deviceID, bool registerInGlobal) {

  // Check if context already exists for this device
  if (registerInGlobal &&
      DeviceContextRegistry::getInstance().hasContext(deviceID)) {
    VIENNACORE_LOG_WARNING(
        "Context for device " + std::to_string(deviceID) +
        " already exists in registry. Returning existing context.");
    return DeviceContextRegistry::getInstance().getContext(deviceID);
  }

#ifdef VIENNACORE_DYNAMIC_MODULE_PATH
// If a dynamic modules path is specified (at build or runtime), adjust the
// modulePath to be relative to the current module's location
#ifdef _WIN32
  HMODULE hModule = GetModuleHandle(NULL);
  char path[MAX_PATH];
  GetModuleFileNameA(hModule, path, MAX_PATH);
  std::filesystem::path mPath = std::filesystem::path(path).parent_path();
  modulePath = mPath / modulePath;
#else
  Dl_info info;
  // Get the address of this function to locate our own module
  if (dladdr((void *)&createContext, &info) && info.dli_fname) {
    auto mPath = std::filesystem::path(info.dli_fname).parent_path();
    modulePath = mPath / modulePath;
  }
#endif
#endif

  auto context = std::make_shared<DeviceContext>();
  context->create(modulePath, deviceID);

  if (registerInGlobal) {
    DeviceContextRegistry::getInstance().registerContext(deviceID, context);
    VIENNACORE_LOG_DEBUG("Context for device " + std::to_string(deviceID) +
                         " registered in global registry.");
  }

  return context;
}

inline std::shared_ptr<DeviceContext>
DeviceContext::getContextFromRegistry(int deviceID) {
  return DeviceContextRegistry::getInstance().getContext(deviceID);
}

inline bool DeviceContext::hasContextInRegistry(int deviceID) {
  return DeviceContextRegistry::getInstance().hasContext(deviceID);
}

inline std::vector<int> DeviceContext::getRegisteredDeviceIDs() {
  return DeviceContextRegistry::getInstance().getRegisteredDeviceIDs();
}

} // namespace viennacore