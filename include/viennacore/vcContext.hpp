#pragma once

#include <cuda.h>

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
#include "vcCudaHandle.hpp"
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
                const int deviceID = 0, bool registerInGlobal = true) {
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

  static std::shared_ptr<DeviceContext> getContextFromRegistry(int deviceID) {
    return DeviceContextRegistry::getInstance().getContext(deviceID);
  }

  static bool hasContextInRegistry(int deviceID) {
    return DeviceContextRegistry::getInstance().hasContext(deviceID);
  }

  static std::vector<int> getRegisteredDeviceIDs() {
    return DeviceContextRegistry::getInstance().getRegisteredDeviceIDs();
  }

  // Instance methods
  void create(std::filesystem::path modulePath = VIENNACORE_KERNELS_PATH,
              const int deviceID = 0) {
    if (!ch.isLoaded()) {
      // CUDA driver not available, context cannot be created.
      return;
    }

    // create new context
    this->modulePath = modulePath;
    this->deviceID = deviceID;

    // initialize CUDA driver API (cuda.h)
    ch.cuInit_(0);

    int numDevices;
    ch.cuDeviceGetCount_(&numDevices);
    if (numDevices == 0) {
      VIENNACORE_LOG_ERROR("No CUDA capable devices found!");
    }

    CUdevice device;
    ch.cuDeviceGet_(&device, deviceID);

    // Get device name
    char deviceNameBuffer[256];
    ch.cuDeviceGetName_(deviceNameBuffer, 256, device);
    deviceName = deviceNameBuffer;
    VIENNACORE_LOG_DEBUG("Registered context for device: " + deviceName);

    // Create CUDA device context
    ch.cuCtxCreate_(&cuda, 0, device);

    // Test that context is functional by allocating and freeing a small amount
    // of memory
    CUdeviceptr d_ptr;
    CUDA_CHECK(ch.cuMemAlloc_(&d_ptr, 1));
    assert(d_ptr != 0);
    CUDA_CHECK(ch.cuMemFree_(d_ptr));

    // add default modules
    VIENNACORE_LOG_DEBUG("PTX kernels path: " + modulePath.string());
    // no default modules for now

    // initialize OptiX context
    OPTIX_CHECK(optixInit());

    OPTIX_CHECK(optixDeviceContextCreate(cuda, 0, &optix));
    optixDeviceContextSetLogCallback(optix, contextLogCallback, nullptr, 4);
  }

  CUmodule getModule(const std::string &moduleName) {
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

  void addModule(const std::string &moduleName) {
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
    const std::string p = (modulePath / moduleName).string();
    ch.cuModuleLoad_(&module, p.c_str());

    modules.push_back(module);
    moduleNames.push_back(moduleName);
  }

  std::string getModulePath() const { return modulePath.string(); }
  std::string getDeviceName() const { return deviceName; }
  int getDeviceID() const { return deviceID; }
  bool foundCuda() const { return ch.isLoaded(); }
  void sync() const { ch.cuCtxSynchronize_(); }

  void destroy() {
    if (deviceID == -1)
      return;

    for (auto module : modules) {
      ch.cuModuleUnload_(module);
    }
    optixDeviceContextDestroy(optix);
    ch.cuCtxDestroy_(cuda);

    // Unregister from global registry
    DeviceContextRegistry::getInstance().unregisterContext(deviceID);

    deviceID = -1;
  }

  std::filesystem::path modulePath;
  std::vector<std::string> moduleNames;
  std::vector<CUmodule> modules;

  CudaHandle ch;
  CUcontext cuda;
  std::string deviceName;
  OptixDeviceContext optix;
  int deviceID = -1;
};

} // namespace viennacore