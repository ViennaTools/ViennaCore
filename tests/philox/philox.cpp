#include <vcPhiloxRNG.hpp>

#include <chrono>
#include <iostream>
#include <vector>

int main() {
  using namespace viennacore;

  // Test basic functionality
  std::cout << "Testing PhiloxRNG..." << std::endl;

  // Test 1: Basic generation
  PhiloxRNG rng(12345);
  std::cout << "First 10 values: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << rng() << " ";
  }
  std::cout << std::endl;

  // Test 2: State save/restore
  PhiloxRNG rng2(12345);
  auto saved_state = rng2.get_state();
  uint32_t val1 = rng2();
  uint32_t val2 = rng2();

  rng2.set_state(saved_state);
  uint32_t val1_restored = rng2();
  uint32_t val2_restored = rng2();

  std::cout << "State save/restore test: "
            << (val1 == val1_restored && val2 == val2_restored ? "PASSED"
                                                               : "FAILED")
            << std::endl;

  // Test 3: Bulk generation
  PhiloxRNG rng3(54321);
  std::vector<uint32_t> bulk_values(1000);
  rng3.generate(bulk_values.data(), bulk_values.size());
  std::cout << "Bulk generation: Generated " << bulk_values.size() << " values"
            << std::endl;

  // Test 4: Discard functionality
  PhiloxRNG rng4(11111);
  PhiloxRNG rng5(11111);

  // Generate 50 values with rng4
  for (int i = 0; i < 50; ++i) {
    rng4();
  }
  uint32_t val_after_50 = rng4();

  // Skip 50 values with rng5 and get next value
  rng5.discard(50);
  uint32_t val_after_discard = rng5();

  std::cout << "Discard test: "
            << (val_after_50 == val_after_discard ? "PASSED" : "FAILED")
            << std::endl;

  // Test 5: Performance comparison (basic)
  const size_t num_values = 1000000;

  auto start = std::chrono::high_resolution_clock::now();
  PhiloxRNG perf_rng(99999);
  for (size_t i = 0; i < num_values; ++i) {
    volatile uint32_t val = perf_rng();
    (void)val; // Suppress unused variable warning
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Generated " << num_values << " values in " << duration.count()
            << " microseconds" << std::endl;

  std::cout << "All tests completed!" << std::endl;
  return 0;
}
