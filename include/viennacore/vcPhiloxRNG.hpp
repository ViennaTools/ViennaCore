#pragma once

#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>

namespace viennacore {

// Philox RNG implementation
class PhiloxRNG {
public:
  using result_type = uint32_t;

  explicit PhiloxRNG(uint64_t seed = 0) { set_seed(seed); }

  void set_seed(uint64_t seed) {
    counter = {0, 0, 0, 0};
    key = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    output_index = 4;
  }

  result_type operator()() {
    if (output_index >= 4) {
      generate_block();
      output_index = 0;
    }
    return output[output_index++];
  }

  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }

  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }

private:
  std::array<uint32_t, 4> counter = {};
  std::array<uint32_t, 2> key = {};
  std::array<uint32_t, 4> output = {};
  size_t output_index = 4;

  static constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53;
  static constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
  static constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9;
  static constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85;

  static void mulhilo(uint32_t a, uint32_t b, uint32_t &hi, uint32_t &lo) {
    uint64_t product = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(product >> 32);
    lo = static_cast<uint32_t>(product);
  }

  void generate_block() {
    std::array<uint32_t, 4> ctr = counter;
    std::array<uint32_t, 2> k = key;

    for (int round = 0; round < 10; ++round) {
      uint32_t hi0, lo0, hi1, lo1;
      mulhilo(PHILOX_M4x32_0, ctr[0], hi0, lo0);
      mulhilo(PHILOX_M4x32_1, ctr[2], hi1, lo1);

      ctr = {hi1 ^ ctr[1] ^ k[0], lo1, hi0 ^ ctr[3] ^ k[1], lo0};

      k[0] += PHILOX_W32_0;
      k[1] += PHILOX_W32_1;
    }

    output = ctr;
    increment_counter();
  }

  void increment_counter() {
    for (int i = 0; i < 4; ++i) {
      if (++counter[i] != 0)
        break;
    }
  }
};
} // namespace viennacore