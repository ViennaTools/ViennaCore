#pragma once
#include <dlfcn.h>
#include <stdexcept>
#include <string>

struct CudartHandle {
  void *handle = nullptr;

  CudartHandle() {
    const char *candidates[] = {
        "libcuda.so.1",
        "libcuda.so",
    };
    for (auto *name : candidates) {
      handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle)
        return;
    }
    throw std::runtime_error(std::string("CUDA driver not found. ") +
                             "Install CUDA driver (libcuda) and ensure it "
                             "is on the loader path.");
  }

  ~CudartHandle() {
    if (handle)
      dlclose(handle);
  }

  template <class Fn> Fn load(const char *symbol) {
    dlerror(); // clear
    auto *p = dlsym(handle, symbol);
    const char *e = dlerror();
    if (e)
      throw std::runtime_error(std::string("Missing CUDA symbol: ") + symbol +
                               " (" + e + ")");
    return reinterpret_cast<Fn>(p);
  }
};