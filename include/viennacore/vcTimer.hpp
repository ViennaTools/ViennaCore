#pragma once

#include <chrono>

namespace viennacore {

template <class Clock = std::chrono::high_resolution_clock> struct Timer {
  using TimePoint = typename Clock::time_point;
  using DurationType = typename Clock::duration::rep;

  TimePoint start_;
  DurationType totalDuration{};   // in ns
  DurationType currentDuration{}; // in ns

  void start() { start_ = Clock::now(); }
  void finish() {
    TimePoint end = Clock::now();
    typename Clock::duration dur(end - start_);
    currentDuration = dur.count();
    totalDuration += currentDuration;
  }
  void reset() {
    currentDuration = static_cast<DurationType>(0);
    totalDuration = static_cast<DurationType>(0);
  }
};

}; // namespace viennacore