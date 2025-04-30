#include <vcContext.hpp>
#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

int main() {
  using namespace viennacore;
  Context context;
  context.create();
  auto deviceName = context.getDeviceName();
  auto modulePath = context.getModulePath();
  std::cout << "Device name: " << deviceName << std::endl;
  std::cout << "Module path: " << modulePath << std::endl;
  context.destroy();
}