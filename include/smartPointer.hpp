#pragma once

#include <memory>

namespace core {

/// std::shared_ptr wrapper for use with ViennaTools libraries.
/// Smart pointers should be created using the function ::New(...).
/// All other interface functions are identical to std::shared_ptr
template <class T> class SmartPointer : public std::shared_ptr<T> {
public:
  // SmartPointer(T& passedObject) :
  // std::shared_ptr<T>(std::make_shared(passedObject)) {}

  // Make visible all constructors of std::shared_ptr
  // including copy constructors
  template <typename... Args>
  SmartPointer(Args &&...args)
      : std::shared_ptr<T>(std::forward<Args>(args)...) {}

  /// Use this function to create new objects when using ViennaLS
  template <typename... TArgs> static SmartPointer New(TArgs &&...targs) {
    return SmartPointer(std::make_shared<T>(std::forward<TArgs>(targs)...));
  }
};

}; // namespace core