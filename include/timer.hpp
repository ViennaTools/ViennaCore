#pragma once

#include <chrono>

namespace core {

template <class Clock = std::chrono::high_resolution_clock> struct Timer {
  using TimePoint = typename Clock::time_point;

  TimePoint start_;
  typename Clock::duration::rep totalDuration = 0.; // in ns
  typename Clock::duration::rep currentDuration;    // in ns

  void start() { start_ = Clock::now(); }
  void finish() {
    TimePoint end = Clock::now();
    typename Clock::duration dur(end - start_);
    currentDuration = dur.count();
    totalDuration += currentDuration;
  }
  void reset() {
    currentDuration = 0.;
    totalDuration = 0.;
  }
};

}; // namespace core