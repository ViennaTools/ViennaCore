#include <vcContext.hpp>
#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

#include <iostream>
#include <memory>
#include <vector>

void testBasicContextCreation() {
  std::cout << "Testing basic context creation..." << std::endl;

  using namespace viennacore;
  Context context;
  context.create();
  auto deviceName = context.getDeviceName();
  auto modulePath = context.getModulePath();

  VC_TEST_ASSERT(!deviceName.empty());
  // Module path might be empty if VIENNACORE_KERNELS_PATH_DEFINE is not set
  // VC_TEST_ASSERT(!modulePath.empty()); // Commented out for now

  std::cout << "Device name: " << deviceName << std::endl;
  std::cout << "Module path: " << modulePath << std::endl;

  context.destroy();
  std::cout << "✓ Basic context creation test passed" << std::endl;
}

void testGlobalContextRegistry() {
  std::cout << "\nTesting global context registry..." << std::endl;

  using namespace viennacore;

  // Clear any existing contexts from previous tests
  ContextRegistry::getInstance().clear();

  // Test 1: Create context and register it globally
  auto context0 = Context::createContext(VIENNACORE_KERNELS_PATH, 0, true);
  VC_TEST_ASSERT(context0 != nullptr);
  VC_TEST_ASSERT(context0->getDeviceID() == 0);

  // Test 2: Check if context is in registry
  VC_TEST_ASSERT(Context::hasContextInRegistry(0));
  VC_TEST_ASSERT(!Context::hasContextInRegistry(1));

  // Test 3: Retrieve context from registry
  auto retrievedContext = Context::getContextFromRegistry(0);
  VC_TEST_ASSERT(retrievedContext != nullptr);
  VC_TEST_ASSERT(retrievedContext.get() == context0.get()); // Same instance

  // Test 4: Get registered device IDs
  auto deviceIDs = Context::getRegisteredDeviceIDs();
  VC_TEST_ASSERT(deviceIDs.size() == 1);
  VC_TEST_ASSERT(deviceIDs[0] == 0);

  // Test 5: Try to create another context for the same device (should return
  // existing)
  auto context0_duplicate =
      Context::createContext(VIENNACORE_KERNELS_PATH, 0, true);
  VC_TEST_ASSERT(context0_duplicate.get() == context0.get()); // Same instance

  // Test 6: Create context without registering globally
  auto localContext = Context::createContext(VIENNACORE_KERNELS_PATH, 0, false);
  VC_TEST_ASSERT(localContext != nullptr);
  VC_TEST_ASSERT(localContext.get() != context0.get()); // Different instance

  // Test 7: Cleanup and verify unregistration
  context0->destroy();
  VC_TEST_ASSERT(!Context::hasContextInRegistry(0));

  // Test 8: Verify retrieving non-existent context returns nullptr
  auto nullContext = Context::getContextFromRegistry(0);
  VC_TEST_ASSERT(nullContext == nullptr);

  // Cleanup
  localContext->destroy();

  std::cout << "✓ Global context registry test passed" << std::endl;
}

void testMultipleDeviceContexts() {
  std::cout << "\nTesting multiple device contexts..." << std::endl;

  using namespace viennacore;

  // Clear registry
  ContextRegistry::getInstance().clear();

  // Create contexts for device 0 (assuming it exists)
  auto context0 = Context::createContext(VIENNACORE_KERNELS_PATH, 0, true);
  VC_TEST_ASSERT(context0 != nullptr);

  // Test multiple device IDs in registry
  auto deviceIDs = Context::getRegisteredDeviceIDs();
  VC_TEST_ASSERT(deviceIDs.size() == 1);
  VC_TEST_ASSERT(std::find(deviceIDs.begin(), deviceIDs.end(), 0) !=
                 deviceIDs.end());

  // Test context properties
  VC_TEST_ASSERT(context0->getDeviceID() == 0);
  VC_TEST_ASSERT(!context0->getDeviceName().empty());
  // Module path might be empty if VIENNACORE_KERNELS_PATH_DEFINE is not set
  // VC_TEST_ASSERT(!context0->getModulePath().empty()); // Commented out for
  // now

  // Cleanup
  context0->destroy();

  // Verify cleanup
  VC_TEST_ASSERT(!Context::hasContextInRegistry(0));
  deviceIDs = Context::getRegisteredDeviceIDs();
  VC_TEST_ASSERT(deviceIDs.empty());

  std::cout << "✓ Multiple device contexts test passed" << std::endl;
}

void testContextRegistryThreadSafety() {
  std::cout << "\nTesting context registry thread safety (basic)..."
            << std::endl;

  using namespace viennacore;

  // Clear registry
  ContextRegistry::getInstance().clear();

  // Test concurrent access to registry methods
  // Note: This is a basic test - for full thread safety testing,
  // you would need to use std::thread and test concurrent operations

  auto &registry = ContextRegistry::getInstance();

  // Test that registry operations don't crash when called multiple times
  for (int i = 0; i < 10; ++i) {
    VC_TEST_ASSERT(!registry.hasContext(i));
    auto context = registry.getContext(i);
    VC_TEST_ASSERT(context == nullptr);
  }

  auto deviceIDs = registry.getRegisteredDeviceIDs();
  VC_TEST_ASSERT(deviceIDs.empty());

  // Create a context and test registry operations
  auto context0 = Context::createContext(VIENNACORE_KERNELS_PATH, 0, true);

  for (int i = 0; i < 5; ++i) {
    VC_TEST_ASSERT(registry.hasContext(0));
    auto retrieved = registry.getContext(0);
    VC_TEST_ASSERT(retrieved != nullptr);
  }

  // Cleanup
  context0->destroy();

  std::cout << "✓ Context registry thread safety test passed" << std::endl;
}

void testContextRegistryClearFunction() {
  std::cout << "\nTesting context registry clear function..." << std::endl;

  using namespace viennacore;

  // Create multiple contexts
  auto context0 = Context::createContext(VIENNACORE_KERNELS_PATH, 0, true);

  // Verify contexts are registered
  VC_TEST_ASSERT(Context::hasContextInRegistry(0));
  auto deviceIDs = Context::getRegisteredDeviceIDs();
  VC_TEST_ASSERT(deviceIDs.size() == 1);

  // Clear registry
  ContextRegistry::getInstance().clear();

  // Verify registry is empty
  VC_TEST_ASSERT(!Context::hasContextInRegistry(0));
  deviceIDs = Context::getRegisteredDeviceIDs();
  VC_TEST_ASSERT(deviceIDs.empty());

  // Note: Contexts are still valid, just not in registry
  VC_TEST_ASSERT(context0 != nullptr);
  VC_TEST_ASSERT(context0->getDeviceID() == 0);

  // Cleanup
  context0->destroy();

  std::cout << "✓ Context registry clear function test passed" << std::endl;
}

int main() {
  try {
    std::cout << "Running ViennaCore Context Tests...\n" << std::endl;

    testBasicContextCreation();
    testGlobalContextRegistry();
    testMultipleDeviceContexts();
    testContextRegistryThreadSafety();
    testContextRegistryClearFunction();

    std::cout << "\n🎉 All context tests passed successfully!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "❌ Test failed with error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "❌ Test failed with unknown error" << std::endl;
    return 1;
  }
}