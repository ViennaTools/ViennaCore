#pragma once

#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>

namespace viennacore {

// High-performance Philox4x32 RNG implementation
class PhiloxRNG {
public:
  using result_type = uint32_t;

  // Default constructor uses random device for seeding
  PhiloxRNG() : PhiloxRNG(std::random_device{}()) {}

  explicit PhiloxRNG(uint64_t seed) { set_seed(seed); }

  void set_seed(uint64_t seed) noexcept {
    counter = {0, 0, 0, 0};
    key = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    output_index = 4; // Force regeneration on next call
  }

  result_type operator()() noexcept {
    if (output_index >= 4) {
      generate_block();
      output_index = 0;
    }
    return output[output_index++];
  }

  // Generate multiple values at once for better performance
  void generate(result_type *dest, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = operator()();
    }
  }

  // Skip ahead by n values (useful for parallel RNG streams)
  void discard(unsigned long long n) noexcept {
    // Handle remaining values in current block
    unsigned long long remaining_in_block = 4 - output_index;
    if (n < remaining_in_block) {
      output_index += static_cast<size_t>(n);
      return;
    }

    n -= remaining_in_block;
    output_index = 4; // Mark block as exhausted

    // Skip full blocks
    unsigned long long full_blocks = n / 4;
    advance_counter(full_blocks);

    // Handle remaining values
    if (n % 4 != 0) {
      generate_block();
      output_index = n % 4;
    }
  }

  static constexpr result_type min() noexcept {
    return std::numeric_limits<result_type>::min();
  }

  static constexpr result_type max() noexcept {
    return std::numeric_limits<result_type>::max();
  }

  // Get current state for reproducibility
  struct State {
    std::array<uint32_t, 4> counter;
    std::array<uint32_t, 2> key;
    size_t output_index;
  };

  State get_state() const noexcept { return {counter, key, output_index}; }

  void set_state(const State &state) noexcept {
    counter = state.counter;
    key = state.key;
    output_index = state.output_index;
  }

private:
  std::array<uint32_t, 4> counter = {};
  std::array<uint32_t, 2> key = {};
  std::array<uint32_t, 4> output = {};
  size_t output_index = 4;

  // Philox4x32 constants (validated against reference implementation)
  static constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53;
  static constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
  static constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9;
  static constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85;
  static constexpr int ROUNDS = 10;

  // Optimized multiply-high-low operation
  static constexpr void mulhilo(uint32_t a, uint32_t b, uint32_t &hi,
                                uint32_t &lo) noexcept {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(product >> 32);
    lo = static_cast<uint32_t>(product & 0xFFFFFFFF);
  }

  void generate_block() noexcept {
    std::array<uint32_t, 4> ctr = counter;
    std::array<uint32_t, 2> k = key;

    // Philox rounds for cryptographic strength
    for (int round = 0; round < ROUNDS; ++round) {
      uint32_t hi0, lo0, hi1, lo1;
      mulhilo(PHILOX_M4x32_0, ctr[0], hi0, lo0);
      mulhilo(PHILOX_M4x32_1, ctr[2], hi1, lo1);

      // Philox bijection - use temporary variables for clarity
      const uint32_t new_0 = hi1 ^ ctr[1] ^ k[0];
      const uint32_t new_1 = lo1;
      const uint32_t new_2 = hi0 ^ ctr[3] ^ k[1];
      const uint32_t new_3 = lo0;

      ctr[0] = new_0;
      ctr[1] = new_1;
      ctr[2] = new_2;
      ctr[3] = new_3;

      // Key schedule
      k[0] += PHILOX_W32_0;
      k[1] += PHILOX_W32_1;
    }

    output = ctr;
    increment_counter();
  }

  void increment_counter() noexcept {
    // 128-bit counter increment with carry propagation
    for (size_t i = 0; i < 4; ++i) {
      if (++counter[i] != 0) {
        break; // No carry needed
      }
    }
  }

  // Advance counter by n blocks (for discard implementation)
  void advance_counter(uint64_t n) noexcept {
    // Add n to the 128-bit counter
    uint64_t carry = n;
    for (size_t i = 0; i < 4 && carry > 0; ++i) {
      const uint64_t sum = static_cast<uint64_t>(counter[i]) + carry;
      counter[i] = static_cast<uint32_t>(sum & 0xFFFFFFFF);
      carry = sum >> 32;
    }
  }
};

} // namespace viennacore